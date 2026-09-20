"""Bounded settlement of Group-scoped Task authority changes.

The context manager records an authority change before a caller mutates the
Task, while the settler only acknowledges effects that are already visible in
durable Task and Attempt truth.  Locks are deliberately owned by callers.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from ...config_types import RootConfig
from ..group_namespace import is_group_authority_isolated
from ..operation_store import locate_operation_path
from ..paths import attempt_path
from ..records import AttemptRecord, TaskRecord
from ..store import read_json
from ..tasks import load_task
from .rechecks import GroupRechecks, RecheckTicket

if TYPE_CHECKING:
    from ...lifecycle import TerminalCommitResult


_OWNERS = frozenset(
    {"retry", "claim", "claim_loss", "launch", "availability", "terminal", "recovery", "cleanup", "task_cancel"}
)
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})
_ACTIVE_ATTEMPT_PHASES = frozenset({"claimed", "starting", "running"})
_LAUNCH_ATTEMPT_PHASES = frozenset({"starting", "running"})
_CLAIM_LOSS_ATTEMPT_PHASES = frozenset({"orphaned", "cancelled", "succeeded", "failed"})
_EVENT_KEYS = frozenset(
    {
        "generation",
        "sequence",
        "task_id",
        "submission_operation_id",
        "membership_sequence",
        "owner",
        "evidence",
        "state",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "task_before",
        "attempt_before",
        "task_revision_before",
        "ready_generation_before",
        "attempt_number",
        "details",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_digest(task: TaskRecord) -> str:
    return _sha256(_canonical_json(task.to_dict()))


def _read_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_number: int | None,
) -> tuple[AttemptRecord | None, str | None]:
    """Read one Attempt and hash its exact persisted JSON bytes."""

    if attempt_number is None:
        return None, None
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("attempt number must be a positive integer or null")
    path = attempt_path(cfg.shared_root, task_id, attempt_number)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, None
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Attempt record must contain a JSON object")
    return AttemptRecord.from_dict(value), _sha256(raw)


def _current_attempt_number(task: TaskRecord) -> int | None:
    value = task.attempt_control.get("current_attempt_number")
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("Task current_attempt_number must be a positive integer or null")
    return value


def _before_evidence(
    cfg: RootConfig,
    persisted_task: TaskRecord,
    details: dict[str, Any],
) -> dict[str, Any]:
    attempt_number = _current_attempt_number(persisted_task)
    _attempt, attempt_digest = _read_attempt(cfg, persisted_task.task_id, attempt_number)
    return {
        "task_before": _task_digest(persisted_task),
        "attempt_before": attempt_digest,
        "task_revision_before": persisted_task.meta["revision"],
        "ready_generation_before": persisted_task.ready_generation,
        "attempt_number": attempt_number,
        "details": dict(details),
    }


def _invalidate_after_io_error(journal: GroupRechecks) -> None:
    """Invalidate a partially published ticket, allowing its original error."""

    journal.invalidate()


@contextmanager
def record_task_change(
    cfg: RootConfig,
    task: TaskRecord,
    owner: str,
    *,
    details: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record one Group Task change before the caller applies its authority effect.

    The caller must hold the schema, Group, and Task locks for the entire
    context.  The passed Task is used only for event identity; evidence comes
    from a fresh persisted Task read.
    """

    if task.group_name is None or not is_group_authority_isolated(cfg.shared_root):
        yield
        return

    journal = GroupRechecks(cfg.shared_root, task.group_name)
    if journal.snapshot() is None:
        yield
        return

    persisted_task = load_task(cfg, task.task_id)
    evidence = _before_evidence(cfg, persisted_task, {} if details is None else details)
    try:
        ticket = journal.begin(
            task.task_id,
            task.submission_operation_id,
            task.group_membership_sequence,
            owner,
            evidence,
        )
    except OSError:
        _invalidate_after_io_error(journal)
        ticket = None

    if ticket is None:
        yield
        return

    try:
        yield
    except BaseException:
        raise
    try:
        journal.resolve(ticket)
    except OSError:
        _invalidate_after_io_error(journal)
        return


def _valid_digest(value: object, *, nullable: bool = False) -> bool:
    if nullable and value is None:
        return True
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _parse_event(value: object) -> dict[str, Any] | None:
    if type(value) is not dict or not _EVENT_KEYS.issubset(value):
        return None
    generation = value.get("generation")
    sequence = value.get("sequence")
    task_id = value.get("task_id")
    operation_id = value.get("submission_operation_id")
    membership_sequence = value.get("membership_sequence")
    owner = value.get("owner")
    state = value.get("state")
    evidence = value.get("evidence")
    if (
        type(generation) is not int
        or generation < 1
        or type(sequence) is not int
        or sequence < 1
        or type(task_id) is not str
        or not task_id
        or type(operation_id) is not str
        or not operation_id
        or type(membership_sequence) is not int
        or membership_sequence < 1
        or owner not in _OWNERS
        or state not in {"in_flight", "committed", "aborted"}
        or type(evidence) is not dict
        or not _EVIDENCE_KEYS.issubset(evidence)
    ):
        return None
    if (
        not _valid_digest(evidence.get("task_before"))
        or not _valid_digest(evidence.get("attempt_before"), nullable=True)
        or type(evidence.get("task_revision_before")) is not int
        or evidence["task_revision_before"] < 0
        or type(evidence.get("ready_generation_before")) is not int
        or evidence["ready_generation_before"] < 0
        or (
            evidence.get("attempt_number") is not None
            and (type(evidence["attempt_number"]) is not int or evidence["attempt_number"] < 1)
        )
        or type(evidence.get("details")) is not dict
    ):
        return None
    return value


def _same_event_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in _EVENT_KEYS if key != "state" and key != "evidence")


def _read_cleanup_operation(
    cfg: RootConfig,
    event: dict[str, Any],
    task: TaskRecord | None,
) -> tuple[str | None, dict[str, Any] | None, bool, bool]:
    details = event["evidence"]["details"]
    operation_id = details.get("cleanup_operation_id")
    if type(operation_id) is not str or not operation_id:
        return None, None, False, False
    path = locate_operation_path(cfg, "cleanup", event["task_id"])
    operation = read_json(path)
    cleanup = operation.get("cleanup")
    if not isinstance(cleanup, dict):
        return None, None, False, False
    if (
        cleanup.get("operation_id") != operation_id
        or cleanup.get("task_id") != event["task_id"]
        or cleanup.get("submission_operation_id") != event["submission_operation_id"]
    ):
        return None, cleanup, False, False
    group_name = cleanup.get("group_name")
    if task is not None and group_name != task.group_name:
        return None, cleanup, False, False
    if group_name is not None and type(group_name) is not str:
        return None, cleanup, False, False
    return group_name, cleanup, True, cleanup.get("state") == "completed"


def _ticket(event: dict[str, Any]) -> RecheckTicket:
    return RecheckTicket(event["generation"], event["sequence"])


def _resolve(
    journal: GroupRechecks,
    event: dict[str, Any],
    outcome: str,
) -> bool:
    position = journal.snapshot()
    if position is None or position.generation != event["generation"]:
        return True
    try:
        journal.resolve(_ticket(event), outcome=outcome)
    except (OSError, ValueError, KeyError):
        latest = journal.snapshot()
        if latest is None or latest.generation != event["generation"]:
            return True
        try:
            current = _parse_event(journal.read(latest, event["sequence"]))
        except (OSError, ValueError, KeyError):
            return False
        return current is not None and current["state"] in {"committed", "aborted"}
    return True


def _exact_before_state(
    cfg: RootConfig,
    task: TaskRecord,
    evidence: dict[str, Any],
) -> bool:
    _attempt, attempt_digest = _read_attempt(cfg, task.task_id, evidence["attempt_number"])
    return _task_digest(task) == evidence["task_before"] and attempt_digest == evidence["attempt_before"]


def _detail_positive_int(details: dict[str, Any], name: str) -> int | None:
    value = details.get(name)
    if type(value) is not int or value < 1:
        return None
    return value


def _detail_nonnegative_int(details: dict[str, Any], name: str) -> int | None:
    value = details.get(name)
    if type(value) is not int or value < 0:
        return None
    return value


def _detail_identifier(details: dict[str, Any], name: str) -> str | None:
    value = details.get(name)
    if type(value) is not str or not value:
        return None
    return value


def _transition_identity(event: dict[str, Any], *, launch: bool = False) -> tuple[str, int, int, str | None] | None:
    details = event["evidence"]["details"]
    attempt_id = _detail_identifier(details, "attempt_id")
    attempt_number = _detail_positive_int(details, "attempt_number")
    fencing_token = _detail_nonnegative_int(details, "fencing_token")
    launch_id = _detail_identifier(details, "launch_id") if launch else None
    if attempt_id is None or attempt_number is None or fencing_token is None or (launch and launch_id is None):
        return None
    return attempt_id, attempt_number, fencing_token, launch_id


def _task_revision_advanced(task: TaskRecord, evidence: dict[str, Any]) -> bool:
    revision = task.meta.get("revision")
    return type(revision) is int and revision > evidence["task_revision_before"]


def _same_attempt_identity(
    task: TaskRecord,
    attempt: AttemptRecord,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
) -> bool:
    claim = task.claim_control.get("active_claim") or {}
    return (
        attempt.task_id == task.task_id
        and attempt.attempt_id == attempt_id
        and attempt.attempt_number == attempt_number
        and attempt.current_fencing_token == fencing_token
        and claim.get("attempt_id") == attempt_id
        and claim.get("attempt_number") == attempt_number
        and claim.get("fencing_token") == fencing_token
        and task.attempt_control.get("current_attempt_id") == attempt_id
        and task.attempt_control.get("current_attempt_number") == attempt_number
    )


def _coherent_terminal_attempt(
    task: TaskRecord,
    attempt: AttemptRecord,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
) -> bool:
    return (
        not task.claim_control.get("active_claim")
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == attempt_number
        and task.state.get("projection") in _TERMINAL_PHASES
        and attempt.task_id == task.task_id
        and attempt.attempt_id == attempt_id
        and attempt.attempt_number == attempt_number
        and (
            attempt.current_fencing_token == fencing_token
            or (
                type(attempt.current_fencing_token) is int
                and attempt.current_fencing_token > fencing_token
                and fencing_token in attempt.token_history
            )
        )
        and attempt.phase == task.state.get("projection")
    )


def _coherent_recovered_successor(
    task: TaskRecord,
    attempt: AttemptRecord,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    evidence: dict[str, Any],
) -> bool:
    claim = task.claim_control.get("active_claim") or {}
    history = attempt.token_history
    return (
        _task_revision_advanced(task, evidence)
        and task.state.get("projection") == "running"
        and task.state.get("reason") == "recovered_live_attempt"
        and task.attempt_control.get("current_attempt_id") == attempt_id
        and task.attempt_control.get("current_attempt_number") == attempt_number
        and attempt.task_id == task.task_id
        and attempt.attempt_id == attempt_id
        and attempt.attempt_number == attempt_number
        and attempt.phase == "running"
        and type(attempt.current_fencing_token) is int
        and attempt.current_fencing_token > fencing_token
        and isinstance(history, list)
        and fencing_token in history
        and claim.get("attempt_id") == attempt_id
        and claim.get("attempt_number") == attempt_number
        and claim.get("fencing_token") == attempt.current_fencing_token
        and claim.get("machine_name") == attempt.machine_name
    )


def _coherent_successor(
    task: TaskRecord,
    attempt: AttemptRecord,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    evidence: dict[str, Any],
) -> bool:
    """Retire an old obligation only against a coherent, identified successor."""
    if not _task_revision_advanced(task, evidence):
        return False
    if _coherent_terminal_attempt(task, attempt, attempt_id, attempt_number, fencing_token):
        return True
    if _coherent_recovered_successor(task, attempt, attempt_id, attempt_number, fencing_token, evidence):
        return True
    return (
        task.state.get("projection") == "blocked"
        and task.state.get("reason") == "orphaned_attempt_requires_recovery"
        and not task.claim_control.get("active_claim")
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == attempt_number
        and attempt.task_id == task.task_id
        and attempt.attempt_id == attempt_id
        and attempt.attempt_number == attempt_number
        and attempt.phase == "orphaned"
        and type(attempt.current_fencing_token) is int
        and (
            attempt.current_fencing_token == fencing_token
            or (attempt.current_fencing_token > fencing_token and fencing_token in attempt.token_history)
        )
        and type(task.claim_control.get("fencing_epoch")) is int
        and task.claim_control["fencing_epoch"] >= attempt.current_fencing_token
    )


def _is_later_ready_supersession(
    task: TaskRecord,
    evidence: dict[str, Any],
    attempt_id: str,
    attempt_number: int,
) -> bool:
    if not _task_revision_advanced(task, evidence) or task.ready_generation <= evidence["ready_generation_before"]:
        return False
    current_number = task.attempt_control.get("current_attempt_number")
    current_id = task.attempt_control.get("current_attempt_id")
    return current_number != attempt_number or current_id != attempt_id


def _settle_withdrawn_claim(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
    identity: tuple[str, int, int, str | None],
) -> bool:
    """Settle a claim that its owner durably withdrew during materialization compensation."""

    _attempt_id, attempt_number, fencing_token, _launch_id = identity
    if (
        not _task_revision_advanced(task, event["evidence"])
        or task.state.get("projection") != "queued"
        or task.state.get("reason") != "attempt_materialization_failed"
        or task.claim_control.get("active_claim")
        or task.attempt_control.get("current_attempt_id") is not None
        or task.attempt_control.get("current_attempt_number") != attempt_number
        or type(task.claim_control.get("fencing_epoch")) is not int
        or task.claim_control["fencing_epoch"] < fencing_token
    ):
        return False
    from ..ready import commit_ready_publication

    if task.ready_generation <= 0 or not commit_ready_publication(cfg, task):
        return False
    return _resolve(journal, event, "committed")


def _settle_claim(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
) -> bool:
    evidence = event["evidence"]
    identity = _transition_identity(event)
    if identity is None:
        return False
    attempt_id, attempt_number, fencing_token, _launch_id = identity

    if _exact_before_state(cfg, task, evidence):
        proposed_attempt, _digest = _read_attempt(cfg, task.task_id, attempt_number)
        if proposed_attempt is None:
            return _resolve(journal, event, "aborted")
        return False

    if _settle_withdrawn_claim(cfg, journal, event, task, identity):
        return True

    attempt, _attempt_digest = _read_attempt(cfg, task.task_id, attempt_number)
    claim = task.claim_control.get("active_claim") or {}
    if (
        attempt is None
        and _task_revision_advanced(task, evidence)
        and task.state.get("projection") == "cancelled"
        and not claim
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == attempt_number
        and type(task.claim_control.get("fencing_epoch")) is int
        and task.claim_control["fencing_epoch"] >= fencing_token
    ):
        return _resolve(journal, event, "committed")
    if (
        attempt is not None
        and _task_revision_advanced(task, evidence)
        and task.state.get("projection") == "running"
        and _same_attempt_identity(task, attempt, attempt_id, attempt_number, fencing_token)
        and claim.get("machine_name") == attempt.machine_name
        and attempt.phase in _ACTIVE_ATTEMPT_PHASES
    ):
        from ..ready import retire_current_ready_generation

        retire_current_ready_generation(cfg, task)
        return _resolve(journal, event, "committed")

    if attempt is not None and _coherent_recovered_successor(
        task, attempt, attempt_id, attempt_number, fencing_token, evidence
    ):
        from ..ready import retire_current_ready_generation

        retire_current_ready_generation(cfg, task)
        return _resolve(journal, event, "committed")

    if (
        attempt is not None
        and _task_revision_advanced(task, evidence)
        and _coherent_successor(task, attempt, attempt_id, attempt_number, fencing_token, evidence)
    ):
        return _resolve(journal, event, "committed")

    if _is_later_ready_supersession(task, evidence, attempt_id, attempt_number):
        return _resolve(journal, event, "committed")
    return False


def _settle_launch(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
) -> bool:
    evidence = event["evidence"]
    identity = _transition_identity(event, launch=True)
    if identity is None:
        return False
    attempt_id, attempt_number, fencing_token, _proposed_launch_id = identity
    if _exact_before_state(cfg, task, evidence):
        return _resolve(journal, event, "aborted")

    attempt, _attempt_digest = _read_attempt(cfg, task.task_id, attempt_number)
    claim = task.claim_control.get("active_claim") or {}
    current_launch_id = claim.get("launch_id")
    attempt_launch_id = attempt.authorization.get("launch_id") if attempt is not None else None
    if (
        attempt is not None
        and _task_revision_advanced(task, evidence)
        and task.state.get("projection") == "running"
        and _same_attempt_identity(task, attempt, attempt_id, attempt_number, fencing_token)
        and claim.get("machine_name") == attempt.machine_name
        and claim.get("launch_state") in {"starting", "running"}
        and type(current_launch_id) is str
        and bool(current_launch_id)
        and current_launch_id == attempt_launch_id
        and attempt.phase in _LAUNCH_ATTEMPT_PHASES
    ):
        return _resolve(journal, event, "committed")

    if attempt is not None and _coherent_recovered_successor(
        task, attempt, attempt_id, attempt_number, fencing_token, evidence
    ):
        return _resolve(journal, event, "committed")

    if (
        attempt is not None
        and _task_revision_advanced(task, evidence)
        and _coherent_successor(task, attempt, attempt_id, attempt_number, fencing_token, evidence)
    ):
        return _resolve(journal, event, "committed")

    if _is_later_ready_supersession(task, evidence, attempt_id, attempt_number):
        return _resolve(journal, event, "committed")
    return False


def _settle_retry(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
) -> bool:
    evidence = event["evidence"]
    details = evidence["details"]
    reserved_generation = _detail_positive_int(details, "reserved_generation")
    if reserved_generation is None or reserved_generation <= evidence["ready_generation_before"]:
        return False
    if _exact_before_state(cfg, task, evidence):
        if task.ready_generation < reserved_generation:
            from ..ready import delete_ready_marker

            delete_ready_marker(cfg, task.task_id, reserved_generation)
        return _resolve(journal, event, "aborted")

    if (
        type(task.meta.get("revision")) is not int
        or task.meta["revision"] <= evidence["task_revision_before"]
        or task.ready_generation < reserved_generation
    ):
        return False
    if task.ready_generation == reserved_generation and task.state.get("projection") == "queued":
        from ..ready import commit_ready_publication, retire_previous_ready_generation

        if not commit_ready_publication(cfg, task):
            return False
        retire_previous_ready_generation(cfg, evidence["ready_generation_before"], task)
    return _resolve(journal, event, "committed")


def _deserialize_terminal_transition(
    event: dict[str, Any],
) -> tuple[Any, str | None, bool | None] | None:
    from ...lifecycle import TerminalTransition

    details = event["evidence"]["details"]
    raw = details.get("transition", details)
    if type(raw) is not dict:
        return None
    required = {
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "phase",
        "reason",
        "exit_code",
        "allowed_task_phases",
        "allowed_attempt_phases",
        "claim_mode",
    }
    if not required.issubset(raw):
        return None
    if (
        raw.get("task_id") != event["task_id"]
        or type(raw.get("attempt_id")) is not str
        or not raw["attempt_id"]
        or type(raw.get("attempt_number")) is not int
        or raw["attempt_number"] < 1
        or type(raw.get("fencing_token")) is not int
        or raw["fencing_token"] < 0
        or raw.get("phase") not in _TERMINAL_PHASES
        or type(raw.get("reason")) is not str
        or (raw.get("exit_code") is not None and type(raw["exit_code"]) is not int)
        or type(raw.get("allowed_task_phases")) is not list
        or type(raw.get("allowed_attempt_phases")) is not list
        or any(type(item) is not str for item in raw["allowed_task_phases"])
        or any(type(item) is not str for item in raw["allowed_attempt_phases"])
        or raw.get("claim_mode") not in {"active", "detached"}
        or (raw.get("termination_result") is not None and type(raw["termination_result"]) is not str)
        or ("allow_missing_attempt" in raw and type(raw["allow_missing_attempt"]) is not bool)
    ):
        return None
    cancellation_operation_id = details.get("cancellation_operation_id")
    if cancellation_operation_id is not None and type(cancellation_operation_id) is not str:
        return None
    terminate_running = details.get("terminate_running")
    if terminate_running is not None and type(terminate_running) is not bool:
        return None
    try:
        transition = TerminalTransition(
            raw["task_id"],
            raw["attempt_id"],
            raw["attempt_number"],
            raw["fencing_token"],
            raw["phase"],
            raw["reason"],
            raw["exit_code"],
            frozenset(raw["allowed_task_phases"]),
            frozenset(raw["allowed_attempt_phases"]),
            raw["claim_mode"],
            raw.get("termination_result"),
            raw.get("allow_missing_attempt", False),
        )
    except (TypeError, ValueError):
        return None
    return transition, cancellation_operation_id, terminate_running


def _restore_terminal_control(
    task: TaskRecord,
    cancellation_operation_id: str | None,
    terminate_running: bool | None,
) -> None:
    if cancellation_operation_id is None and terminate_running is None:
        return
    current_operation = task.control.get("cancellation_operation_id")
    if current_operation is not None and current_operation != cancellation_operation_id:
        return
    if cancellation_operation_id is not None:
        task.control["cancellation_operation_id"] = cancellation_operation_id
    if terminate_running is True:
        task.control["terminate_running"] = True
    elif terminate_running is False and not task.control.get("terminate_running"):
        task.control["terminate_running"] = False


def _settle_terminal(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
    results: list[TerminalCommitResult],
) -> bool:
    transition_data = _deserialize_terminal_transition(event)
    if transition_data is None:
        return False
    transition, cancellation_operation_id, terminate_running = transition_data
    evidence = event["evidence"]
    if _exact_before_state(cfg, task, evidence):
        return _resolve(journal, event, "aborted")

    attempt, _attempt_digest = _read_attempt(cfg, task.task_id, transition.attempt_number)
    active_claim = task.claim_control.get("active_claim") or {}
    if attempt is None:
        if (
            transition.allow_missing_attempt
            and transition.phase == "cancelled"
            and task.state.get("projection") == "cancelled"
            and not active_claim
        ):
            return _resolve(journal, event, "committed")
        return False
    if _coherent_successor(
        task, attempt, transition.attempt_id, transition.attempt_number, transition.fencing_token, evidence
    ):
        return _resolve(journal, event, "committed")
    if (
        attempt.task_id != task.task_id
        or attempt.attempt_id != transition.attempt_id
        or attempt.attempt_number != transition.attempt_number
        or attempt.current_fencing_token != transition.fencing_token
        or attempt.phase != transition.phase
    ):
        return False

    current_number = task.attempt_control.get("current_attempt_number")
    if (
        task.state.get("projection") == transition.phase
        and not active_claim
        and current_number == transition.attempt_number
    ):
        return _resolve(journal, event, "committed")

    current_epoch = task.claim_control.get("fencing_epoch")
    if type(current_epoch) is int and current_epoch > transition.fencing_token:
        return _resolve(journal, event, "committed")
    if type(current_number) is int and current_number > transition.attempt_number:
        return _resolve(journal, event, "committed")

    has_matching_claim = (
        active_claim.get("attempt_id") == transition.attempt_id
        and active_claim.get("fencing_token") == transition.fencing_token
    )
    can_commit = task.state.get("projection") in transition.allowed_task_phases and (
        (transition.claim_mode == "active" and has_matching_claim)
        or (transition.claim_mode == "detached" and not active_claim)
    )
    if not can_commit:
        return False
    _restore_terminal_control(task, cancellation_operation_id, terminate_running)
    from ...lifecycle import commit_terminal_transition_locked

    result = commit_terminal_transition_locked(cfg, task, transition)
    if result.outcome == "committed":
        results.append(result)
        return _resolve(journal, event, "committed")
    return False


def _settle_claim_loss(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
) -> bool:
    evidence = event["evidence"]
    if _exact_before_state(cfg, task, evidence):
        return _resolve(journal, event, "aborted")
    details = evidence["details"]
    expected_attempt_id = details.get("expected_attempt_id")
    expected_fencing_token = details.get("expected_fencing_token")
    if details.get("transition") == "claim_materialization_failed":
        reason = details.get("reason")
        if (
            type(expected_attempt_id) is str
            and bool(expected_attempt_id)
            and type(expected_fencing_token) is int
            and expected_fencing_token >= 0
            and type(reason) is str
            and bool(reason)
            and _task_revision_advanced(task, evidence)
            and task.state.get("projection") == "queued"
            and task.state.get("reason") == reason
            and not task.claim_control.get("active_claim")
            and task.attempt_control.get("current_attempt_id") is None
            and task.attempt_control.get("current_attempt_number") == evidence.get("attempt_number")
            and type(task.claim_control.get("fencing_epoch")) is int
            and task.claim_control["fencing_epoch"] >= expected_fencing_token
        ):
            from ..ready import commit_ready_publication

            if task.ready_generation <= 0 or not commit_ready_publication(cfg, task):
                return False
            return _resolve(journal, event, "committed")
        return False
    if details.get("is_invalid_evidence") is True:
        active_claim = task.claim_control.get("active_claim") or {}
        if (
            type(expected_attempt_id) is str
            and bool(expected_attempt_id)
            and type(expected_fencing_token) is int
            and expected_fencing_token >= 0
            and task.meta.get("revision", 0) > evidence["task_revision_before"]
            and task.state.get("projection") == "blocked"
            and task.state.get("reason") == "authority_mode_evidence_invalid"
            and active_claim.get("attempt_id") == expected_attempt_id
            and active_claim.get("fencing_token") == expected_fencing_token
        ):
            return _resolve(journal, event, "committed")
        return False
    if type(expected_attempt_id) is str and type(expected_fencing_token) is int and expected_fencing_token >= 0:
        successor, _digest = _read_attempt(cfg, task.task_id, evidence["attempt_number"])
        if successor is not None and _coherent_successor(
            task, successor, expected_attempt_id, evidence["attempt_number"], expected_fencing_token, evidence
        ):
            return _resolve(journal, event, "committed")
    if (
        type(expected_attempt_id) is not str
        or not expected_attempt_id
        or type(expected_fencing_token) is not int
        or expected_fencing_token < 0
        or task.meta.get("revision", 0) <= evidence["task_revision_before"]
        or task.claim_control.get("active_claim")
        or task.state.get("projection") not in {"queued", "blocked", *_TERMINAL_PHASES}
    ):
        return False
    attempt, _attempt_digest = _read_attempt(cfg, task.task_id, evidence["attempt_number"])
    if (
        attempt is None
        or attempt.attempt_id != expected_attempt_id
        or attempt.current_fencing_token != expected_fencing_token
        or (
            attempt.phase not in _CLAIM_LOSS_ATTEMPT_PHASES
            and not (
                details.get("transition") == "release"
                and task.state.get("projection") == "blocked"
                and task.state.get("reason") == details.get("reason")
            )
        )
    ):
        return False
    reserved_generation = _detail_positive_int(details, "reserved_generation")
    if reserved_generation is not None:
        if task.ready_generation < reserved_generation:
            return False
        if task.ready_generation == reserved_generation and task.state.get("projection") == "queued":
            from ..ready import commit_ready_publication, retire_previous_ready_generation

            if not commit_ready_publication(cfg, task):
                return False
            retire_previous_ready_generation(cfg, evidence["ready_generation_before"], task)
    return _resolve(journal, event, "committed")


def _settle_recovery(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
) -> bool:
    details = event["evidence"]["details"]
    expected_attempt_id = details.get("expected_attempt_id")
    expired_token = details.get("expired_token")
    if (
        type(expected_attempt_id) is not str
        or not expected_attempt_id
        or type(expired_token) is not int
        or expired_token < 0
    ):
        return False
    claim = task.claim_control.get("active_claim") or {}
    current_number = task.attempt_control.get("current_attempt_number")
    attempt, _attempt_digest = _read_attempt(cfg, task.task_id, current_number)
    if attempt is not None and _coherent_successor(
        task, attempt, expected_attempt_id, event["evidence"]["attempt_number"], expired_token, event["evidence"]
    ):
        return _resolve(journal, event, "committed")
    if (
        task.state.get("projection") != "running"
        or claim.get("attempt_id") != expected_attempt_id
        or type(claim.get("fencing_token")) is not int
        or claim["fencing_token"] <= expired_token
        or task.attempt_control.get("current_attempt_id") != expected_attempt_id
        or current_number != event["evidence"]["attempt_number"]
        or attempt is None
        or attempt.attempt_id != expected_attempt_id
        or attempt.phase != "running"
        or attempt.current_fencing_token != claim["fencing_token"]
    ):
        return False
    return _resolve(journal, event, "committed")


def _settle_cleanup(
    cfg: RootConfig,
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord | None,
) -> bool:
    _group_name, _cleanup, identity_valid, completed = _read_cleanup_operation(cfg, event, task)
    if not identity_valid or not completed:
        return False
    return _resolve(journal, event, "committed")


def _settle_task_cancel(
    journal: GroupRechecks,
    event: dict[str, Any],
    task: TaskRecord,
    cfg: RootConfig,
) -> bool:
    evidence = event["evidence"]
    if _exact_before_state(cfg, task, evidence):
        return _resolve(journal, event, "aborted")
    cancellation_requested_at = task.control.get("cancellation_requested_at")
    is_requested = type(cancellation_requested_at) is str and bool(cancellation_requested_at)
    is_terminal = task.state.get("projection") in _TERMINAL_PHASES and not task.claim_control.get("active_claim")
    # A later retry publishes a strictly newer ready generation under the same
    # Task authority fence. It supersedes this control mutation; the consumer
    # must still reclassify the current Task before advancing its cursor.
    is_superseded = task.ready_generation > evidence["ready_generation_before"]
    if task.meta.get("revision", 0) <= evidence["task_revision_before"] or not (
        is_requested or is_terminal or is_superseded
    ):
        return False
    if is_terminal:
        from ..ready import retire_current_ready_generation

        retire_current_ready_generation(cfg, task)
    return _resolve(journal, event, "committed")


def _settle(
    cfg: RootConfig,
    event: dict[str, Any],
    results: list[TerminalCommitResult],
) -> bool:
    parsed = _parse_event(event)
    if parsed is None:
        return False
    event = parsed

    try:
        task = load_task(cfg, event["task_id"])
    except FileNotFoundError:
        task = None

    group_name: str | None
    if task is None:
        cleanup = read_json(locate_operation_path(cfg, "cleanup", event["task_id"]))["cleanup"]
        witness = TaskRecord.from_dict(cleanup["group_discovery_outcome"]["task"])
        group_name = witness.group_name
        if (
            group_name is None
            or cleanup.get("state") != "completed"
            or cleanup.get("group_name") != group_name
            or cleanup.get("task_id") != event["task_id"]
            or cleanup.get("submission_operation_id") != event["submission_operation_id"]
            or witness.task_id != event["task_id"]
            or witness.submission_operation_id != event["submission_operation_id"]
            or witness.group_membership_sequence != event["membership_sequence"]
            or witness.state.get("projection") not in _TERMINAL_PHASES
            or witness.claim_control.get("active_claim")
        ):
            return False
    else:
        group_name = task.group_name
        if group_name is None:
            return False
        if (
            task.submission_operation_id != event["submission_operation_id"]
            or task.group_membership_sequence != event["membership_sequence"]
        ):
            return False

    if not is_group_authority_isolated(cfg.shared_root):
        return True
    journal = GroupRechecks(cfg.shared_root, group_name)
    position = journal.snapshot()
    if position is None:
        return True
    if position.generation != event["generation"]:
        return True
    try:
        persisted_event = _parse_event(journal.read(position, event["sequence"]))
    except (OSError, ValueError, KeyError):
        latest = journal.snapshot()
        if latest is None or latest.generation != event["generation"]:
            return True
        return False
    if persisted_event is None or not _same_event_identity(persisted_event, event):
        return False
    event = persisted_event
    if event["state"] in {"committed", "aborted"}:
        return True
    if event["state"] != "in_flight":
        return False

    if event["owner"] == "cleanup":
        return _settle_cleanup(cfg, journal, event, task)
    if task is None:
        # Completed cleanup permanently retires this exact Task identity. Its
        # witness supersedes older mutations without replaying any of them;
        # the consumer reads that witness separately for historical effects.
        if event["owner"] in {"retry", "availability"}:
            reserved_generation = _detail_positive_int(event["evidence"]["details"], "reserved_generation")
            if reserved_generation is None:
                return False
            from ..ready import delete_ready_marker

            delete_ready_marker(cfg, event["task_id"], reserved_generation)
        return _resolve(journal, event, "committed")
    if event["owner"] == "retry":
        return _settle_retry(cfg, journal, event, task)
    if event["owner"] == "availability":
        return _settle_retry(cfg, journal, event, task)
    if event["owner"] == "claim":
        return _settle_claim(cfg, journal, event, task)
    if event["owner"] == "launch":
        return _settle_launch(cfg, journal, event, task)
    if event["owner"] == "terminal":
        return _settle_terminal(cfg, journal, event, task, results)
    if event["owner"] == "claim_loss":
        return _settle_claim_loss(cfg, journal, event, task)
    if event["owner"] == "recovery":
        return _settle_recovery(cfg, journal, event, task)
    return _settle_task_cancel(journal, event, task, cfg)


def settle_task_change(
    cfg: RootConfig,
    event: dict[str, Any],
) -> tuple[bool, list[TerminalCommitResult]]:
    """Settle one in-flight Group change after proving its durable effect."""

    results: list[TerminalCommitResult] = []
    try:
        settled = _settle(cfg, event, results)
    except (OSError, ValueError, KeyError, TypeError):
        return False, results
    return settled, results


__all__ = ["record_task_change", "settle_task_change"]
