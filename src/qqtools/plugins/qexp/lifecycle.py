"""Shared terminal transition and lifecycle hook primitives for qexp."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from .config_types import RootConfig
from .events import write_notification_diagnostic
from .runtime.claims import archive_claim
from .runtime.paths import attempt_path
from .runtime.records import AttemptRecord, TaskRecord, utc_now
from .runtime.store import atomic_replace, read_json
from .runtime.tasks import save_task
from .runtime.ready import retire_current_ready_generation


@dataclass(frozen=True, slots=True)
class TaskLifecycleEvent:
    event_type: Literal["task_terminal"]
    task_id: str
    attempt_id: str
    attempt_number: int
    previous_task_phase: str
    phase: Literal["succeeded", "failed", "cancelled"]
    reason: str
    exit_code: int | None
    execution_machine_name: str
    dispatching_machine_name: str
    finished_at: str
    task_revision: int
    execution_started_at: str | None = None
    duration_ms: int | None = None


def _duration_ms(started_at: str | None, finished_at: str) -> int | None:
    if started_at is None:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((finish - start).total_seconds() * 1000))


@dataclass(frozen=True, slots=True)
class TerminalTransition:
    task_id: str
    attempt_id: str
    attempt_number: int
    fencing_token: int
    phase: Literal["succeeded", "failed", "cancelled"]
    reason: str
    exit_code: int | None
    allowed_task_phases: frozenset[str]
    allowed_attempt_phases: frozenset[str]
    claim_mode: Literal["active", "detached"]
    termination_result: str | None = None
    allow_missing_attempt: bool = False


@dataclass(frozen=True, slots=True)
class TerminalCommitResult:
    outcome: Literal["committed", "already_committed", "rejected"]
    event: TaskLifecycleEvent | None = None
    reservation_id: str | None = None
    reservation_machine_name: str | None = None
    reason: str | None = None


class TaskLifecycleHook(Protocol):
    name: str

    def handle(self, cfg: RootConfig, event: TaskLifecycleEvent) -> None:
        ...


def commit_terminal_transition_locked(
    cfg: RootConfig, task: TaskRecord, transition: TerminalTransition,
) -> TerminalCommitResult:
    """Commit an attempt-backed terminal projection while caller holds authority locks."""
    if task.task_id != transition.task_id:
        return TerminalCommitResult("rejected", reason="task_id_mismatch")
    if task.state["projection"] not in transition.allowed_task_phases:
        if task.state["projection"] == transition.phase:
            return TerminalCommitResult("already_committed", reason="already_terminal")
        return TerminalCommitResult("rejected", reason="invalid_task_source_phase")
    active_claim = task.claim_control.get("active_claim") or {}
    if transition.claim_mode == "active":
        if (active_claim.get("attempt_id") != transition.attempt_id
                or active_claim.get("fencing_token") != transition.fencing_token):
            return TerminalCommitResult("rejected", reason="stale_claim")
        reservation_id = active_claim.get("reservation_id")
        reservation_machine = active_claim.get("machine_name")
    else:
        if active_claim:
            return TerminalCommitResult("rejected", reason="active_claim_present")
        reservation_id = None
        reservation_machine = None

    path = attempt_path(cfg.shared_root, transition.task_id, transition.attempt_number)
    try:
        attempt = AttemptRecord.from_dict(read_json(path))
    except FileNotFoundError:
        if not _can_commit_missing_attempt(transition):
            return TerminalCommitResult("rejected", reason="attempt_missing")
        return _commit_missing_attempt_transition(
            cfg, task, transition, active_claim, reservation_id, reservation_machine
        )
    except (KeyError, ValueError):
        return TerminalCommitResult("rejected", reason="attempt_missing")
    if (attempt.attempt_id != transition.attempt_id
            or attempt.attempt_number != transition.attempt_number
            or attempt.current_fencing_token != transition.fencing_token):
        if attempt.phase == transition.phase and task.state["projection"] == transition.phase:
            return TerminalCommitResult("already_committed", reason="already_terminal")
        return TerminalCommitResult("rejected", reason="invalid_attempt_identity_or_phase")
    if attempt.phase not in transition.allowed_attempt_phases and attempt.phase != transition.phase:
        return TerminalCommitResult("rejected", reason="invalid_attempt_source_phase")
    if transition.claim_mode == "detached":
        reservation_id = attempt.reservation_id
        reservation_machine = attempt.machine_name

    previous_phase = task.state["projection"]
    finished_at = utc_now()
    attempt.phase = transition.phase
    attempt.result.update({"exit_code": transition.exit_code, "reason": transition.reason})
    attempt.timestamps["finished_at"] = finished_at
    if transition.termination_result is not None:
        attempt.termination.update({"acknowledged_at": finished_at,
                                    "result": transition.termination_result})
    atomic_replace(path, attempt.to_dict())
    if transition.claim_mode == "active" and active_claim:
        archive_claim(cfg, task.task_id, active_claim, transition.reason)
    task.claim_control["active_claim"] = None
    task.claim_control["fencing_epoch"] = max(
        task.claim_control.get("fencing_epoch", 0), transition.fencing_token
    )
    task.attempt_control["current_attempt_id"] = None
    task.attempt_control["current_attempt_number"] = transition.attempt_number
    task.state.update({"projection": transition.phase, "reason": transition.reason})
    if transition.termination_result is not None:
        task.control.update({"termination_acknowledged_at": finished_at,
                             "termination_result": transition.termination_result})
    task.meta["revision"] += 1
    task.meta["updated_at"] = finished_at
    save_task(cfg, task)
    retire_current_ready_generation(cfg, task)
    execution_started_at = (attempt.timestamps.get("process_created_at")
                            or attempt.timestamps.get("running_at"))
    event = TaskLifecycleEvent(
        event_type="task_terminal", task_id=task.task_id, attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number, previous_task_phase=previous_phase,
        phase=transition.phase, reason=transition.reason, exit_code=transition.exit_code,
        execution_machine_name=attempt.machine_name, dispatching_machine_name=cfg.machine_name,
        finished_at=finished_at, task_revision=task.meta["revision"],
        execution_started_at=execution_started_at,
        duration_ms=_duration_ms(execution_started_at, finished_at),
    )
    return TerminalCommitResult("committed", event, reservation_id, reservation_machine)


def _can_commit_missing_attempt(transition: TerminalTransition) -> bool:
    return (
        transition.allow_missing_attempt
        and transition.claim_mode == "active"
        and transition.phase == "cancelled"
    )


def _commit_missing_attempt_transition(
        cfg: RootConfig, task: TaskRecord, transition: TerminalTransition,
        active_claim: dict, reservation_id: str | None,
        reservation_machine: str | None) -> TerminalCommitResult:
    finished_at = utc_now()
    if active_claim:
        archive_claim(cfg, task.task_id, active_claim, transition.reason)
    task.claim_control["active_claim"] = None
    task.claim_control["fencing_epoch"] = max(
        task.claim_control.get("fencing_epoch", 0), transition.fencing_token
    )
    task.attempt_control["current_attempt_id"] = None
    task.attempt_control["current_attempt_number"] = transition.attempt_number
    task.state.update({"projection": transition.phase, "reason": transition.reason})
    task.meta["revision"] += 1
    task.meta["updated_at"] = finished_at
    save_task(cfg, task)
    retire_current_ready_generation(cfg, task)
    return TerminalCommitResult("committed", None, reservation_id, reservation_machine)


def _hooks() -> list[TaskLifecycleHook]:
    try:
        from .notifications import NotificationHook
        return [NotificationHook()]
    except Exception:
        return []


def dispatch_task_lifecycle_hooks_noexcept(cfg: RootConfig, event: TaskLifecycleEvent) -> None:
    """Run static lifecycle hooks, isolating every failure from qexp state transitions."""
    for hook in _hooks():
        try:
            hook.handle(cfg, event)
        except Exception:
            try:
                write_notification_diagnostic(cfg, "notification_failed", event,
                                              reason_code="hook_error", error_type="hook_error")
            except Exception:
                pass
