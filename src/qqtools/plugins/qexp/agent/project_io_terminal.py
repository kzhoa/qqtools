"""Shared terminal observation and publication for isolated Project I/O workers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Callable, Mapping

from ..config_types import RootConfig
from ..lifecycle import TerminalTransition, commit_terminal_transition_locked
from ..runtime.authority_lock import authority_locks
from ..runtime.paths import attempt_path
from ..runtime.records import AttemptRecord, TaskRecord, utc_now
from ..runtime.store import atomic_replace
from ..runtime.tasks import load_task
from ..runtime.terminal_evidence import load_settled_terminal_attempt


def _has_superseded_ownership(task: TaskRecord, attempt: AttemptRecord) -> bool:
    """Prove historical ownership from the committed operation, never age."""
    receipt = task.attempt_control.get("last_supersession")
    if not isinstance(receipt, dict) or receipt.get("attempt_id") != attempt.attempt_id:
        receipt = attempt.authorization.get("supersession_receipt")
    return (
        isinstance(receipt, dict)
        and receipt.get("task_id") == task.task_id == attempt.task_id
        and receipt.get("attempt_id") == attempt.attempt_id
        and receipt.get("attempt_number") == attempt.attempt_number
        and receipt.get("fencing_token") == attempt.current_fencing_token
        and receipt.get("old_machine") == attempt.machine_name
        and receipt.get("reservation_id") == attempt.reservation_id
        and receipt.get("terminate_old_process") is False
        and receipt.get("duplicate_risk_acknowledged") is True
        and task.attempt_control.get("current_attempt_id") != attempt.attempt_id
        and (task.claim_control.get("active_claim") or {}).get("attempt_id") != attempt.attempt_id
        and task.claim_control.get("fencing_epoch", 0) > attempt.current_fencing_token
    )


def observe_project_io_terminal_state(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Read one exact active or detached Attempt snapshot under authority locks."""

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        try:
            raw = attempt_path(cfg.shared_root, task_id, attempt_number).read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def evidence(
        outcome: str,
        reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reason": reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": None if attempt is None else attempt.reservation_id,
            "process_identity": dict(process_identity),
            "mode": mode,
            "task_phase": None if task is None else task.state.get("projection"),
            "attempt_phase": None if attempt is None else attempt.phase,
            "attempt_result_reason": None if attempt is None else attempt.result.get("reason"),
            "attempt_exit_code": None if attempt is None else attempt.result.get("exit_code"),
            "execution_machine_name": None if attempt is None else attempt.machine_name,
            "reservation_machine_name": None if attempt is None else attempt.machine_name,
            "termination_result": None if attempt is None else attempt.termination.get("result"),
            "cancel_requested": False if task is None else bool(task.control.get("terminate_running")),
            "source_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "authority_granted": False,
            "local_effects": [],
        }

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        if task.depends_on_task_ids != initial.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while Project-I/O locks were acquired.")
        attempt, attempt_digest = read_attempt()
        if attempt is None:
            return evidence("stale", "attempt_missing", task, None, None)
        if (
            attempt.task_id != task_id
            or attempt.attempt_id != attempt_id
            or attempt.attempt_number != attempt_number
            or attempt.current_fencing_token != fencing_token
        ):
            return evidence("stale", "attempt_identity_mismatch", task, attempt, attempt_digest)
        if attempt.machine_name != machine_name:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        if attempt.reservation_id != reservation_id:
            return evidence("stale", "reservation_mismatch", task, attempt, attempt_digest)
        if any(attempt.process.get(field) != process_identity.get(field) for field in process_identity):
            return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest)

        task_phase = task.state.get("projection")
        attempt_phase = attempt.phase
        if _has_superseded_ownership(task, attempt) and attempt_phase in {"starting", "running"}:
            observed = evidence("historical_current", None, task, attempt, attempt_digest)
            observed["cancel_requested"] = False
            return observed
        terminal_pair = (
            task_phase in {"succeeded", "failed", "cancelled"}
            and attempt_phase == task_phase
            and task.state.get("reason") == attempt.result.get("reason")
            and task.control.get("termination_result") == attempt.termination.get("result")
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("current_attempt_id") is None
            and not task.claim_control.get("active_claim")
        )
        if terminal_pair:
            return evidence("already_terminal", None, task, attempt, attempt_digest)

        if attempt_phase in {"succeeded", "failed", "cancelled"}:
            settled_attempt = load_settled_terminal_attempt(
                cfg,
                task_id,
                attempt_id,
                attempt_number=attempt_number,
            )
            if settled_attempt is not None:
                return evidence("settled_terminal", None, task, attempt, attempt_digest)

        if mode == "active":
            if (
                task.attempt_control.get("current_attempt_id") != attempt_id
                or task.attempt_control.get("current_attempt_number") != attempt_number
            ):
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            claim = task.claim_control.get("active_claim") or {}
            if (
                task_phase != "running"
                or claim.get("attempt_id") != attempt_id
                or claim.get("attempt_number") != attempt_number
                or claim.get("fencing_token") != fencing_token
                or claim.get("machine_name") != machine_name
                or claim.get("reservation_id") != reservation_id
            ):
                return evidence("stale", "claim_not_current", task, attempt, attempt_digest)
            if attempt_phase != "running":
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        elif mode == "detached_orphan":
            if task_phase != "blocked" or task.claim_control.get("active_claim"):
                return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)
            if (
                task.attempt_control.get("current_attempt_id") is not None
                or task.attempt_control.get("current_attempt_number") != attempt_number
            ):
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            if task.attempt_control.get("next_attempt_number") != attempt_number + 1:
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            if attempt_phase != "orphaned":
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        else:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        return evidence("current", None, task, attempt, attempt_digest)


def publish_project_io_terminal_transition(
    cfg: RootConfig,
    *,
    request_id: str,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    mode: str,
    phase: str,
    reason: str,
    exit_code: int | None,
    termination_result: str | None,
    expected_task_revision: int,
    expected_attempt_digest: str,
    transition_digest: str,
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Publish one digest-bound terminal transition and its shared lifecycle evidence."""

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        try:
            raw = attempt_path(cfg.shared_root, task_id, attempt_number).read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def evidence(
        outcome: str,
        stale_reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
        result: object | None = None,
    ) -> dict[str, Any]:
        event = getattr(result, "event", None)
        event_value = None if event is None else asdict(event)
        if event_value is not None and event_value["task_name"] is not None:
            # Bound the wire projection, including JSON escaping, while keeping
            # the full user name in shared truth and ordinary lifecycle events.
            name = event_value["task_name"]
            low, high = 0, min(len(name), 16_384)
            while low < high:
                middle = (low + high + 1) // 2
                if len(json.dumps(name[:middle], ensure_ascii=False).encode("utf-8")) <= 16_384:
                    low = middle
                else:
                    high = middle - 1
            event_value["task_name"] = name[:low]
        reservation = getattr(result, "reservation_id", None)
        reservation_machine = getattr(result, "reservation_machine_name", None)
        return {
            "outcome": outcome,
            "reason": stale_reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": reservation
            if result is not None
            else (None if attempt is None else attempt.reservation_id),
            "reservation_machine_name": reservation_machine
            if result is not None
            else (None if attempt is None else attempt.machine_name),
            "process_identity": dict(process_identity),
            "mode": mode,
            "phase": phase,
            "transition_reason": reason,
            "exit_code": exit_code,
            "termination_result": termination_result,
            "source_revisions": {"task": expected_task_revision, "attempt_digest": expected_attempt_digest},
            "committed_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "lifecycle_event": event_value,
            "transition_digest": transition_digest,
            "authority_granted": False,
            "local_effects": [],
        }

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        if task.depends_on_task_ids != initial.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while Project-I/O locks were acquired.")
        attempt, attempt_digest = read_attempt()
        if attempt is None:
            return evidence("stale", "attempt_missing", task, None, None)
        if (
            attempt.task_id != task_id
            or attempt.attempt_id != attempt_id
            or attempt.attempt_number != attempt_number
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != machine_name
        ):
            return evidence("stale", "attempt_identity_mismatch", task, attempt, attempt_digest)
        if attempt.reservation_id != reservation_id:
            return evidence("stale", "reservation_mismatch", task, attempt, attempt_digest)
        if any(attempt.process.get(field) != process_identity.get(field) for field in process_identity):
            return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest)

        task_phase = task.state.get("projection")
        active_claim = task.claim_control.get("active_claim") or {}
        target_attempt_matches = (
            attempt.phase == phase
            and attempt.result.get("reason") == reason
            and attempt.result.get("exit_code") == exit_code
            and attempt.termination.get("result") == termination_result
            and attempt.termination.get("project_io_publication_id") == request_id
        )
        target_task_matches = (
            task_phase == phase
            and task.state.get("reason") == reason
            and task.attempt_control.get("current_attempt_id") is None
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and not active_claim
            and task.control.get("termination_result") == termination_result
            and task.control.get("project_io_publication_id") == request_id
        )
        if _has_superseded_ownership(task, attempt):
            if target_attempt_matches:
                return evidence("historical_already_committed", None, task, attempt, attempt_digest)
            if (
                mode != "active"
                or phase not in {"succeeded", "failed"}
                or termination_result is not None
                or exit_code is None
                or phase != ("succeeded" if exit_code == 0 else "failed")
                or reason != ("completed" if exit_code == 0 else "nonzero_exit")
            ):
                return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)
            if task.meta.get("revision") != expected_task_revision or attempt_digest != expected_attempt_digest:
                return evidence("stale", "source_revision_mismatch", task, attempt, attempt_digest)
            if attempt.phase not in {"starting", "running"}:
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
            attempt.phase = phase
            attempt.result.update(exit_code=exit_code, reason=reason, signal=-exit_code if exit_code < 0 else None)
            attempt.timestamps["finished_at"] = utc_now()
            attempt.termination.update(result=None, project_io_publication_id=request_id)
            attempt.meta["revision"] += 1
            attempt.meta["updated_at"] = utc_now()
            mutation_fence()
            atomic_replace(
                attempt_path(cfg.shared_root, task_id, attempt_number),
                attempt.to_dict(),
                before_replace=lambda _stat: mutation_fence(),
            )
            committed_attempt, committed_digest = read_attempt()
            return evidence("historical_committed", None, task, committed_attempt, committed_digest)
        if target_attempt_matches and target_task_matches:
            replay_transition = TerminalTransition(
                task_id,
                attempt_id,
                attempt_number,
                fencing_token,
                phase,
                reason,
                exit_code,
                frozenset({phase}),
                frozenset({phase}),
                "active" if mode == "active" else "detached",
                termination_result,
                project_io_publication_id=request_id,
            )
            committed = commit_terminal_transition_locked(
                cfg,
                task,
                replay_transition,
                mutation_fence=mutation_fence,
            )
            if committed.outcome == "already_committed":
                _attempt, current_digest = read_attempt()
                task = load_task(cfg, task_id)
                return evidence("already_committed", None, task, _attempt, current_digest, committed)
            return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)
        if (attempt.phase == phase and not target_attempt_matches) or (task_phase == phase and not target_task_matches):
            return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)

        if mode == "active":
            current_identity = (
                task.attempt_control.get("current_attempt_id") == attempt_id
                and task.attempt_control.get("current_attempt_number") == attempt_number
            )
        elif mode == "detached_orphan":
            current_identity = (
                task_phase == "blocked"
                and not active_claim
                and task.attempt_control.get("current_attempt_id") is None
                and task.attempt_control.get("current_attempt_number") == attempt_number
                and task.attempt_control.get("next_attempt_number") == attempt_number + 1
            )
        else:
            current_identity = False
        if not current_identity:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        if mode == "active":
            if (
                task_phase != "running"
                or active_claim.get("attempt_id") != attempt_id
                or active_claim.get("attempt_number") != attempt_number
                or active_claim.get("fencing_token") != fencing_token
                or active_claim.get("machine_name") != machine_name
                or active_claim.get("reservation_id") != reservation_id
            ):
                return evidence("stale", "claim_not_current", task, attempt, attempt_digest)
            allowed_attempt_phases = frozenset({"running"})
        elif mode == "detached_orphan":
            if task_phase != "blocked" or task.claim_control.get("active_claim"):
                return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)
            if task.attempt_control.get("next_attempt_number") != attempt_number + 1:
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            allowed_attempt_phases = frozenset({"orphaned"})
        else:
            return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)

        partial_attempt_matches = target_attempt_matches and (
            task_phase in ({"running"} if mode == "active" else {"blocked"})
        )
        if not partial_attempt_matches and (
            task.meta.get("revision") != expected_task_revision or attempt_digest != expected_attempt_digest
        ):
            return evidence("stale", "source_revision_mismatch", task, attempt, attempt_digest)
        if attempt.phase not in allowed_attempt_phases and not partial_attempt_matches:
            return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        if task_phase not in ({"running"} if mode == "active" else {"blocked"}):
            return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)

        transition = TerminalTransition(
            task_id,
            attempt_id,
            attempt_number,
            fencing_token,
            phase,
            reason,
            exit_code,
            frozenset({"running"} if mode == "active" else {"blocked"}),
            allowed_attempt_phases,
            "active" if mode == "active" else "detached",
            termination_result,
            project_io_publication_id=request_id,
        )
        committed = commit_terminal_transition_locked(
            cfg,
            task,
            transition,
            mutation_fence=mutation_fence,
        )
        if committed.outcome not in {"committed", "already_committed"}:
            stale_reason = {
                "attempt_missing": "attempt_missing",
                "stale_claim": "claim_not_current",
                "active_claim_present": "claim_not_current",
                "invalid_attempt_identity_or_phase": "attempt_identity_mismatch",
                "invalid_attempt_source_phase": "attempt_phase_mismatch",
                "invalid_task_source_phase": "task_phase_mismatch",
            }.get(committed.reason, "terminal_conflict")
            _attempt, current_digest = read_attempt()
            current_task = load_task(cfg, task_id)
            return evidence("stale", stale_reason, current_task, _attempt, current_digest)
        current_task = load_task(cfg, task_id)
        current_attempt, current_digest = read_attempt()
        return evidence(
            committed.outcome,
            None,
            current_task,
            current_attempt,
            current_digest,
            committed,
        )
