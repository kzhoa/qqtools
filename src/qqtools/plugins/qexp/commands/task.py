"""Task command workflows for qexp."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config_types import RootConfig
from ..layout import (
    ensure_machine_layout,
    ensure_shared_layout,
    is_task_dependencies_root,
    validate_root_contract,
)
from ..manifest import parse_batch_manifest
from ..runtime.availability import (
    AvailabilityTransitionRequest,
    AvailabilityTransitionResult,
    apply_availability_transition,
)
from ..runtime.paths import attempt_path, shared_paths
from ..runtime.records import AttemptRecord, TaskRecord, utc_now, validate_group_name
from ..runtime.store import read_json
from ..runtime.submission import SubmissionResult, submit_specs
from ..runtime.tasks import load_task, save_task
from ..runtime.ready import (
    discard_ready_generation,
    prepare_ready_transition,
    reserve_ready_generation,
    retire_previous_ready_generation,
)
from ..runtime.active_operations import operation_exists
from ..runtime.dependencies import (
    is_committed_submission_task,
    normalize_dependency_ids,
    validate_group_dependencies,
)
from ..runtime.locks import group_lock, task_lock


def is_cleanup_blocked(task: TaskRecord) -> bool:
    return bool(task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"))


def has_cleanup_operation(cfg: RootConfig, task_id: str) -> bool:
    return operation_exists(cfg, "cleanup", task_id)


def reject_cleanup_blocked(cfg: RootConfig, task: TaskRecord, action: str) -> None:
    if is_cleanup_blocked(task) or has_cleanup_operation(cfg, task.task_id):
        raise ValueError(f"Task {task.task_id!r} is being cleaned and cannot be {action}.")


def submit(
    cfg: RootConfig,
    command: list[str],
    requested_gpus: int = 1,
    requested_cpus: int | None = None,
    task_id: str | None = None,
    name: str | None = None,
    group: str | None = None,
    working_dir: str | Path | None = None,
    home_machine: str | None = None,
    sharing_mode: str = "private",
    fallback_machines: str | list[str] = "group",
    offer_after_seconds: int | None = None,
    depends_on_task_ids: list[str] | None = None,
    idempotency_key: str | None = None,
) -> TaskRecord:
    validate_root_contract(cfg)
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    items = [
        {
            "task_id": task_id,
            "name": name,
            "command": list(command),
            "requested_gpus": requested_gpus,
            "requested_cpus": requested_cpus,
            "working_directory": str(Path(working_dir or Path.cwd()).resolve()),
            "home_machine": "current" if home_machine is None else home_machine,
            "sharing_mode": sharing_mode,
            "fallback_machines": fallback_machines,
            "offer_after_seconds": offer_after_seconds,
            "depends_on_task_ids": depends_on_task_ids or [],
        }
    ]
    return submit_specs(cfg, items, group_name=validate_group_name(group), idempotency_key=idempotency_key)[0]


def batch_submit(
    cfg: RootConfig,
    manifest_path: Path,
    *,
    group: str | None = None,
    idempotency_key: str | None = None,
    on_prepared: Callable[[str, str], None] | None = None,
) -> SubmissionResult:
    validate_root_contract(cfg)
    group_name = validate_group_name(group)
    normalized, workers = parse_batch_manifest(Path(manifest_path), group_name=group_name)
    return submit_specs(
        cfg,
        normalized,
        group_name=group_name,
        idempotency_key=idempotency_key,
        kind="bulk",
        worker_set=workers,
        on_prepared=on_prepared,
    )


def cancel(
    cfg: RootConfig,
    task_id: str,
    *,
    terminate_running: bool = True,
    reservation_runtime_root: Path | None = None,
) -> TaskRecord:
    from ..machine_runtime import resolve_execution_context
    from ..scheduler import cancel_task

    reservation_runtime_root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    return cancel_task(
        cfg,
        task_id,
        terminate_running=terminate_running,
        reservation_runtime_root=reservation_runtime_root,
    )


def retry(cfg: RootConfig, task_id: str, *, acknowledge_duplicate_risk: bool = False) -> TaskRecord:
    """Queue the next Attempt after a failed or orphaned current Attempt."""
    from ..scheduler import authority_locks

    initial = load_task(cfg, task_id)
    reserved_generation = initial.ready_generation + 1
    reference = reserve_ready_generation(
        cfg,
        task_id,
        reserved_generation,
        "home",
        initial.placement_policy["home_machine"],
    )
    is_committed = False
    try:
        with authority_locks(cfg, initial):
            task = load_task(cfg, task_id)
            reject_cleanup_blocked(cfg, task, "retried")
            if task.claim_control.get("active_claim"):
                raise ValueError("a Task with an active claim cannot be retried.")
            number = task.attempt_control.get("current_attempt_number")
            if number is None:
                raise ValueError("Task has no current Attempt to retry.")
            current_path = attempt_path(cfg.shared_root, task_id, number)
            current = AttemptRecord.from_dict(read_json(current_path))
            if task.state["projection"] == "failed" and current.phase == "failed":
                task.state = {"projection": "queued", "reason": None}
            elif task.state["projection"] == "blocked" and current.phase == "orphaned":
                superseded_at = utc_now()
                task.claim_control["fencing_epoch"] += 1
                from ..events import write_event

                write_event(
                    cfg,
                    "orphan_superseded_by_retry",
                    task_id=task_id,
                    details={
                        "attempt_id": current.attempt_id,
                        "fencing_token": current.current_fencing_token,
                        "operator": cfg.machine_name,
                        "timestamp": superseded_at,
                    },
                )
                task.state = {"projection": "queued", "reason": "orphan_superseded_by_retry"}
            else:
                raise ValueError(
                    "only a failed Task or a blocked Task with an orphaned current Attempt "
                    "can be retried."
                )
            task.control.update(
                {
                    "cancellation_requested_at": None,
                    "cancellation_operation_id": None,
                    "terminate_running": False,
                    "requested_by": None,
                    "termination_acknowledged_at": None,
                    "termination_result": None,
                }
            )
            task.placement_runtime.update(
                {
                    "queue_scope": "home",
                    "queued_home_at": utc_now(),
                    "offered_at": None,
                    "offer_reason": None,
                    "offered_by": None,
                }
            )
            task.attempt_control["current_attempt_id"] = None
            old_generation, _ = prepare_ready_transition(
                cfg, task, "retry", reference=reference
            )
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            is_committed = True
            retire_previous_ready_generation(cfg, old_generation, task)
            return task
    finally:
        if not is_committed:
            discard_ready_generation(cfg, task_id, reserved_generation)


def edit_dependencies(
    cfg: RootConfig,
    task_id: str,
    dependency_ids: list[str],
    *,
    action: str = "replace",
) -> TaskRecord:
    """Atomically replace, add, or remove dependencies from an unstarted Task."""
    validate_root_contract(cfg)
    if not is_task_dependencies_root(cfg):
        raise ValueError(
            "Task dependencies require an activated task-dependencies-v1 root."
        )
    initial = load_task(cfg, task_id)
    if initial.group_name is None:
        raise ValueError("ungrouped tasks cannot declare dependencies.")
    requested = normalize_dependency_ids(dependency_ids)
    with group_lock(cfg.shared_root, initial.group_name):
        with task_lock(cfg.shared_root, task_id):
            task = load_task(cfg, task_id)
            reject_cleanup_blocked(cfg, task, "have dependencies edited")
            if not is_committed_submission_task(cfg, task):
                raise ValueError(
                    "a Task whose submission is not committed cannot have dependencies edited."
                )
            if task.group_name != initial.group_name:
                raise RuntimeError("Task Group changed while editing dependencies.")
            if task.state["projection"] in {"succeeded", "failed", "cancelled"}:
                raise ValueError("terminal tasks cannot have dependencies edited.")
            if task.claim_control.get("active_claim") or task.attempt_control["next_attempt_number"] != 1:
                raise ValueError("a Task that has created an Attempt cannot have dependencies edited.")
            current = set(task.depends_on_task_ids)
            if action == "replace":
                updated = requested
            elif action == "add":
                updated = sorted(current | set(requested))
            elif action == "remove":
                updated = sorted(current - set(requested))
            else:
                raise ValueError(f"unknown dependency edit action {action!r}.")
            candidate = TaskRecord.from_dict(task.to_dict())
            candidate.depends_on_task_ids = updated
            validate_group_dependencies(cfg, task.group_name, [candidate])
            task.depends_on_task_ids = updated
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            return task


def share(
    cfg: RootConfig, task_id: str, *, after_seconds: int | None = None, helper_machines: list[str] | None = None
) -> AvailabilityTransitionResult:
    if after_seconds is not None and after_seconds < 0:
        raise ValueError("share --after must be non-negative.")
    action = "share_after" if after_seconds is not None else "share_now"
    return apply_availability_transition(
        cfg,
        AvailabilityTransitionRequest(
            action=action,
            task_id=task_id,
            helper_machines=helper_machines,
            after_seconds=after_seconds,
            reason="manual",
        ),
    )


def keep_local(cfg: RootConfig, task_id: str) -> AvailabilityTransitionResult:
    return apply_availability_transition(
        cfg, AvailabilityTransitionRequest(action="keep_local", task_id=task_id, reason="manual")
    )


def offer(cfg: RootConfig, task_id: str, *, reason: str = "manual") -> AvailabilityTransitionResult:
    action = "elapsed_offer" if reason == "elapsed" else "manual_offer"
    return apply_availability_transition(
        cfg, AvailabilityTransitionRequest(action=action, task_id=task_id, reason=reason)
    )
