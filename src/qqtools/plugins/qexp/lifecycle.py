"""Shared terminal transition and lifecycle hook primitives for qexp."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Literal, Protocol

from .config_types import RootConfig
from .events import write_notification_diagnostic
from .runtime.claims import archive_claim
from .runtime.group_discovery.changes import record_task_change
from .runtime.paths import attempt_path
from .runtime.ready import retire_current_ready_generation
from .runtime.records import AttemptRecord, TaskRecord, utc_now
from .runtime.store import atomic_replace, fenced_mutations, read_json
from .runtime.tasks import save_task


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
    project_id: str | None = None
    project: str = ""
    task_name: str | None = None


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
    project_io_publication_id: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalCommitResult:
    outcome: Literal["committed", "already_committed", "rejected"]
    event: TaskLifecycleEvent | None = None
    reservation_id: str | None = None
    reservation_machine_name: str | None = None
    reason: str | None = None


class TaskLifecycleHook(Protocol):
    name: str

    def handle(self, cfg: RootConfig, event: TaskLifecycleEvent) -> None: ...


def _terminal_change_details(task: TaskRecord, transition: TerminalTransition) -> dict:
    """Serialize terminal transition authority alongside cancellation context."""
    details = asdict(transition)
    details["allowed_task_phases"] = sorted(transition.allowed_task_phases)
    details["allowed_attempt_phases"] = sorted(transition.allowed_attempt_phases)
    for key in ("cancellation_operation_id", "terminate_running"):
        if key in task.control:
            details[key] = task.control[key]
    return details


def _terminal_lifecycle_event(
    cfg: RootConfig,
    task: TaskRecord,
    attempt: AttemptRecord,
    transition: TerminalTransition,
    *,
    previous_phase: str,
    finished_at: str,
) -> TaskLifecycleEvent:
    execution_started_at = attempt.timestamps.get("process_created_at") or attempt.timestamps.get("running_at")
    project_id = None
    try:
        project_id = read_json(cfg.shared_root / "project" / "identity.json")["project"]["project_id"]
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return TaskLifecycleEvent(
        event_type="task_terminal",
        task_id=task.task_id,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        previous_task_phase=previous_phase,
        phase=transition.phase,
        reason=transition.reason,
        exit_code=transition.exit_code,
        execution_machine_name=attempt.machine_name,
        dispatching_machine_name=cfg.machine_name,
        finished_at=finished_at,
        task_revision=task.meta["revision"],
        execution_started_at=execution_started_at,
        duration_ms=_duration_ms(execution_started_at, finished_at),
        project=str(cfg.shared_root.expanduser().resolve()),
        task_name=task.name,
        project_id=project_id,
    )


def commit_terminal_transition_locked(
    cfg: RootConfig,
    task: TaskRecord,
    transition: TerminalTransition,
    *,
    mutation_fence: Callable[[], None] | None = None,
) -> TerminalCommitResult:
    """Commit an attempt-backed terminal projection while caller holds authority locks."""
    with fenced_mutations(cfg.shared_root, mutation_fence):
        return _commit_terminal_transition_locked(cfg, task, transition, mutation_fence=mutation_fence)


def _commit_terminal_transition_locked(
    cfg: RootConfig,
    task: TaskRecord,
    transition: TerminalTransition,
    *,
    mutation_fence: Callable[[], None] | None,
) -> TerminalCommitResult:
    if task.task_id != transition.task_id:
        return TerminalCommitResult("rejected", reason="task_id_mismatch")
    task_phase = task.state["projection"]
    active_claim = task.claim_control.get("active_claim") or {}
    if transition.claim_mode == "active":
        if active_claim:
            if (
                active_claim.get("attempt_id") != transition.attempt_id
                or active_claim.get("fencing_token") != transition.fencing_token
            ):
                return TerminalCommitResult("rejected", reason="stale_claim")
            reservation_id = active_claim.get("reservation_id")
            reservation_machine = active_claim.get("machine_name")
        else:
            reservation_id = None
            reservation_machine = None
            if task_phase != transition.phase:
                return TerminalCommitResult("rejected", reason="stale_claim")
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
            cfg,
            task,
            transition,
            active_claim,
            reservation_id,
            reservation_machine,
            mutation_fence,
        )
    except (KeyError, ValueError):
        return TerminalCommitResult("rejected", reason="attempt_missing")
    if (
        attempt.task_id != transition.task_id
        or attempt.attempt_id != transition.attempt_id
        or attempt.attempt_number != transition.attempt_number
        or attempt.current_fencing_token != transition.fencing_token
    ):
        return TerminalCommitResult("rejected", reason="invalid_attempt_identity_or_phase")
    attempt_target_matches = (
        attempt.phase == transition.phase
        and attempt.result.get("exit_code") == transition.exit_code
        and attempt.result.get("reason") == transition.reason
        and attempt.termination.get("result") == transition.termination_result
        and (
            transition.project_io_publication_id is None
            or attempt.termination.get("project_io_publication_id") == transition.project_io_publication_id
        )
    )
    task_target_matches = (
        task_phase == transition.phase
        and task.state.get("reason") == transition.reason
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == transition.attempt_number
        and not active_claim
        and task.control.get("termination_result") == transition.termination_result
        and (
            transition.project_io_publication_id is None
            or task.control.get("project_io_publication_id") == transition.project_io_publication_id
        )
    )
    if attempt_target_matches and task_target_matches:
        if transition.claim_mode == "active":
            reservation_id = attempt.reservation_id
            reservation_machine = attempt.machine_name
        if mutation_fence is None:
            return TerminalCommitResult("already_committed", None, reservation_id, reservation_machine)
        mutation_fence()
        retire_current_ready_generation(cfg, task)
        finished_at = attempt.timestamps.get("finished_at")
        if not isinstance(finished_at, str) or not finished_at:
            finished_at = task.meta.get("updated_at")
        if not isinstance(finished_at, str) or not finished_at:
            finished_at = utc_now()
        previous_phase = "running" if transition.claim_mode == "active" else "blocked"
        event = _terminal_lifecycle_event(
            cfg,
            task,
            attempt,
            transition,
            previous_phase=previous_phase,
            finished_at=finished_at,
        )
        return TerminalCommitResult("already_committed", event, reservation_id, reservation_machine)
    if task_phase not in transition.allowed_task_phases:
        return TerminalCommitResult("rejected", reason="invalid_task_source_phase")
    if attempt.phase != transition.phase and attempt.phase not in transition.allowed_attempt_phases:
        return TerminalCommitResult("rejected", reason="invalid_attempt_source_phase")
    if attempt.phase == transition.phase and not attempt_target_matches:
        return TerminalCommitResult("rejected", reason="terminal_conflict")
    if transition.claim_mode == "detached":
        reservation_id = attempt.reservation_id
        reservation_machine = attempt.machine_name

    previous_phase = task.state["projection"]
    finished_at = attempt.timestamps.get("finished_at") if attempt_target_matches else utc_now()
    if not isinstance(finished_at, str) or not finished_at:
        finished_at = utc_now()
    if not attempt_target_matches:
        attempt.phase = transition.phase
        attempt.result.update({"exit_code": transition.exit_code, "reason": transition.reason})
        if transition.phase == "cancelled" and task.control.get("cancellation_operation_id"):
            attempt.result["cancellation_operation_id"] = task.control["cancellation_operation_id"]
        attempt.timestamps["finished_at"] = finished_at
        if transition.termination_result is not None:
            attempt.termination.update({"acknowledged_at": finished_at, "result": transition.termination_result})
        if transition.project_io_publication_id is not None:
            attempt.termination["project_io_publication_id"] = transition.project_io_publication_id
    with record_task_change(
        cfg,
        task,
        "terminal",
        details=_terminal_change_details(task, transition),
        mutation_fence=mutation_fence,
    ):
        if not attempt_target_matches:
            atomic_replace(
                path,
                attempt.to_dict(),
                before_replace=(lambda _stat: mutation_fence()) if mutation_fence is not None else None,
            )
        if transition.claim_mode == "active" and active_claim:
            if mutation_fence is not None:
                mutation_fence()
            archive_claim(cfg, task.task_id, active_claim, transition.reason, mutation_fence=mutation_fence)
        task.claim_control["active_claim"] = None
        task.claim_control["fencing_epoch"] = max(task.claim_control.get("fencing_epoch", 0), transition.fencing_token)
        task.attempt_control["current_attempt_id"] = None
        task.attempt_control["current_attempt_number"] = transition.attempt_number
        task.state.update({"projection": transition.phase, "reason": transition.reason})
        if transition.termination_result is not None:
            task.control.update(
                {"termination_acknowledged_at": finished_at, "termination_result": transition.termination_result}
            )
        if transition.project_io_publication_id is not None:
            task.control["project_io_publication_id"] = transition.project_io_publication_id
        task.meta["revision"] += 1
        task.meta["updated_at"] = finished_at
        if mutation_fence is not None:
            mutation_fence()
        save_task(cfg, task, mutation_fence=mutation_fence)
        if mutation_fence is not None:
            mutation_fence()
        retire_current_ready_generation(cfg, task)
    event = _terminal_lifecycle_event(
        cfg,
        task,
        attempt,
        transition,
        previous_phase=previous_phase,
        finished_at=finished_at,
    )
    return TerminalCommitResult("committed", event, reservation_id, reservation_machine)


def _can_commit_missing_attempt(transition: TerminalTransition) -> bool:
    return transition.allow_missing_attempt and transition.claim_mode == "active" and transition.phase == "cancelled"


def _commit_missing_attempt_transition(
    cfg: RootConfig,
    task: TaskRecord,
    transition: TerminalTransition,
    active_claim: dict,
    reservation_id: str | None,
    reservation_machine: str | None,
    mutation_fence: Callable[[], None] | None = None,
) -> TerminalCommitResult:
    finished_at = utc_now()
    with record_task_change(
        cfg,
        task,
        "terminal",
        details=_terminal_change_details(task, transition),
        mutation_fence=mutation_fence,
    ):
        if active_claim:
            if mutation_fence is not None:
                mutation_fence()
            archive_claim(cfg, task.task_id, active_claim, transition.reason)
        task.claim_control["active_claim"] = None
        task.claim_control["fencing_epoch"] = max(task.claim_control.get("fencing_epoch", 0), transition.fencing_token)
        task.attempt_control["current_attempt_id"] = None
        task.attempt_control["current_attempt_number"] = transition.attempt_number
        task.state.update({"projection": transition.phase, "reason": transition.reason})
        task.meta["revision"] += 1
        task.meta["updated_at"] = finished_at
        if mutation_fence is not None:
            mutation_fence()
        save_task(cfg, task, mutation_fence=mutation_fence)
        if mutation_fence is not None:
            mutation_fence()
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
                write_notification_diagnostic(
                    cfg, "notification_failed", event, reason_code="hook_error", error_type="hook_error"
                )
            except Exception:
                pass
