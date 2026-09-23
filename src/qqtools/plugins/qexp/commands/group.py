"""Group command workflows for qexp."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import is_group_ready_members_root
from ..lifecycle import (
    TerminalCommitResult,
    TerminalTransition,
    commit_terminal_transition_locked,
    dispatch_task_lifecycle_hooks_noexcept,
)
from ..runtime.claims import archive_claim
from ..runtime.group_discovery.changes import record_task_change
from ..runtime.group_namespace import has_group_authority_cutover, is_group_authority_isolated, read_group
from ..runtime.locks import group_writer_lock, task_lock
from ..runtime.operation_store import (
    active_operation_path,
    archive_operation,
    iter_active_operation_paths,
    write_active_operation,
)
from ..runtime.paths import attempt_path, group_path, shared_paths, submission_path
from ..runtime.ready import (
    primary_projection_routes_for_group,
    primary_route_update_transaction,
    retire_current_ready_generation,
    sync_primary_ready_group,
)
from ..runtime.ready.group_members import assert_group_ready_members_writable
from ..runtime.records import (
    SCHEMA_VERSION,
    AttemptRecord,
    TaskRecord,
    new_group,
    new_id,
    new_worker_member,
    normalize_group_record,
    utc_now,
    validate_gpu_limit,
    validate_group_name,
    validate_identifier,
)
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.submission import finalize_submission_group, reconcile_submission
from ..runtime.tasks import load_task, save_task
from .group_cancel import advance_indexed_cancel, initialize_cancel_discovery
from .task import has_cleanup_operation, is_cleanup_blocked, retry
from .worker_removal import (
    WORKER_REMOVE_TYPE,
    begin_worker_removal_locked,
    can_bind_worker_removal,
    reconcile_worker_removal,
)


def _finalize_pending_submission_before_group_mutation(cfg: RootConfig, name: str, path: Path) -> None:
    """Reconcile an occupying Submission before taking a Group mutation lock."""
    owner: str | None = None
    with group_writer_lock(cfg, name):
        path = group_path(cfg.shared_root, name)
        if not path.exists():
            return
        data = read_json(path)
        normalize_group_record(data)
        pending = data["group"].get("pending_submission_commit") or {}
        owner = data["group"].get("creation_operation_id") or pending.get("operation_id")
    if owner is None:
        return
    state = reconcile_submission(cfg, owner, abort_incomplete=True)
    if state == "blocked":
        raise RuntimeError(f"Group {name!r} is blocked by Submission operation {owner!r}.")


def group_control(
    cfg: RootConfig,
    name: str,
    action: str,
    *,
    terminate_running: bool = False,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any]:
    from ..agent.context import resolve_execution_context

    reservation_runtime_root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    path = group_path(cfg.shared_root, validate_group_name(name) or name)
    _finalize_pending_submission_before_group_mutation(cfg, name, path)
    post_commit_results = []
    with group_writer_lock(cfg, name):
        path = group_path(cfg.shared_root, name)
        data = read_json(path)
        group = data["group"]
        pending = group.get("pending_submission_commit")
        if pending:
            operation_path = submission_path(cfg.shared_root, pending["operation_id"])
            if not operation_path.exists() or read_json(operation_path)["submission"].get("state") != "committed":
                raise RuntimeError(f"Group {name!r} has pending submission commit {pending['operation_id']!r}.")
            raise RuntimeError(f"Group {name!r} received a concurrent submission commit; retry the mutation.")
        indexed_cancel = False
        if action == "cancel":
            operation_id = new_id()
            high_watermark = group["next_membership_sequence"] - 1
            operation_path = active_operation_path(cfg, "group_control", operation_id)
            operation = {
                "meta": {
                    "schema_version": SCHEMA_VERSION,
                    "revision": 1,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "updated_by": {
                        "actor_type": "cli",
                        "machine_name": cfg.machine_name,
                        "process_id": str(os.getpid()),
                    },
                },
                "group_control": {
                    "operation_id": operation_id,
                    "operation_type": "cancel",
                    "group_name": name,
                    "state": "preparing",
                    "group_revision_at_start": data["meta"]["revision"],
                    "dispatch_epoch_at_start": group["dispatch_epoch"],
                    "membership_high_watermark": high_watermark,
                    "terminate_running": terminate_running,
                    "progress": {
                        "target_tasks": 0,
                        "already_terminal": 0,
                        "queued_cancelled": 0,
                        "prelaunch_cancelled": 0,
                        "running_allowed": 0,
                        "termination_pending": 0,
                        "termination_acknowledged": 0,
                        "blocked": 0,
                    },
                    "pending_machine_acknowledgements": {},
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "completed_at": None,
                    "blocked_reason": None,
                },
            }
            indexed_cancel = is_group_authority_isolated(cfg.shared_root)
            if indexed_cancel:
                initialize_cancel_discovery(cfg, operation["group_control"])
                from ..runtime.group_discovery.service import publish_group_locator_for_transition

                # QQTOOLS-COMPAT-0017: keep cancellation discoverable before
                # publishing the operation and its Group barrier.
                publish_group_locator_for_transition(cfg, name, "control", "group_operation")
            write_active_operation(cfg, "group_control", operation_id, operation)
            group["cancellation_barriers"].append(
                {
                    "operation_id": operation_id,
                    "membership_high_watermark": high_watermark,
                    "terminate_running": terminate_running,
                    "created_at": utc_now(),
                }
            )
            data["cancellation_operation"] = operation["group_control"]
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            atomic_replace(path, data)
            operation["group_control"]["state"] = "converging"
            operation["group_control"]["updated_at"] = utc_now()
            write_active_operation(cfg, "group_control", operation_id, operation)
            if not indexed_cancel:
                for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
                    task = TaskRecord.from_dict(read_json(task_file))
                    if task.group_name != name or (task.group_membership_sequence or 0) > high_watermark:
                        continue
                    operation["group_control"]["progress"]["target_tasks"] += 1
                    with task_lock(cfg.shared_root, task.task_id):
                        try:
                            task = load_task(cfg, task.task_id)
                        except FileNotFoundError:
                            operation["group_control"]["progress"]["already_terminal"] += 1
                            continue
                        progress_key, terminal = _apply_group_cancel_locked(cfg, task, operation["group_control"])
                        if terminal is not None:
                            post_commit_results.append(terminal)
                        if progress_key is not None:
                            operation["group_control"]["progress"][progress_key] += 1
                        if progress_key == "termination_pending":
                            claim = task.claim_control.get("active_claim") or {}
                            machine = claim.get("machine_name") or task.placement_policy["home_machine"]
                            operation["group_control"]["pending_machine_acknowledgements"].setdefault(
                                machine, []
                            ).append(task.task_id)
                progress = operation["group_control"]["progress"]
                if progress["blocked"]:
                    operation["group_control"].update(
                        {"state": "blocked", "blocked_reason": "task_cancellation_requires_resolution"}
                    )
                elif not terminate_running or progress["termination_pending"] == 0:
                    operation["group_control"].update({"state": "completed", "completed_at": utc_now()})
                else:
                    operation["group_control"]["state"] = "waiting_ack"
                operation["group_control"]["updated_at"] = utc_now()
                atomic_replace(operation_path, operation)
                if operation["group_control"]["state"] == "completed":
                    operation_path = archive_operation(cfg, "group_control", operation_id, operation)
            data["cancellation_operation"] = operation["group_control"]
        elif action == "seal":
            group["admission_state"] = "sealed"
        elif action == "reopen":
            group["admission_state"] = "open"
        elif action == "pause":
            group["dispatch_state"] = "paused"
            group["dispatch_epoch"] += 1
        elif action == "resume":
            group["dispatch_state"] = "active"
            group["dispatch_epoch"] += 1
        else:
            raise ValueError(f"unknown group action {action!r}.")
        data["meta"]["revision"] += 1
        data["meta"]["updated_at"] = utc_now()
        atomic_replace(path, data)
        result_data = data
    if indexed_cancel:
        advance_indexed_cancel(cfg, operation_path, reservation_runtime_root=reservation_runtime_root)
        result_data = read_group(cfg.shared_root, name)
        normalize_group_record(result_data)
    _dispatch_group_terminal_results(cfg, post_commit_results, reservation_runtime_root)
    return result_data


def _apply_group_cancel_locked(
    cfg: RootConfig, task: TaskRecord, control: dict[str, Any]
) -> tuple[str | None, TerminalCommitResult | None]:
    """Apply one cancellation effect under the existing Group and Task locks."""
    if is_cleanup_blocked(task) or has_cleanup_operation(cfg, task.task_id):
        return "already_terminal", None
    claim = task.claim_control.get("active_claim") or {}
    if task.state["projection"] == "queued" and not claim:
        with record_task_change(
            cfg,
            task,
            "task_cancel",
            details={"operation_id": control["operation_id"], "expected_effect": "queued_cancelled"},
        ):
            _set_group_cancel_request(cfg, task, control)
            task.state.update({"projection": "cancelled", "reason": "group_cancelled"})
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            retire_current_ready_generation(cfg, task)
        return "queued_cancelled", None
    if claim.get("launch_state") == "claimed":
        _set_group_cancel_request(cfg, task, control)
        result = commit_terminal_transition_locked(
            cfg,
            task,
            TerminalTransition(
                task.task_id,
                claim["attempt_id"],
                task.attempt_control["current_attempt_number"],
                claim["fencing_token"],
                "cancelled",
                "group_cancelled_before_launch",
                None,
                frozenset({"running"}),
                frozenset({"claimed"}),
                "active",
                allow_missing_attempt=True,
            ),
        )
        return ("prelaunch_cancelled" if result.outcome == "committed" else "blocked"), result
    if task.state["projection"] == "running":
        # Replaying an older default cancel must not downgrade a later termination.
        if not task.control.get("terminate_running") and (
            control["terminate_running"] or not task.control.get("cancellation_requested_at")
        ):
            with record_task_change(
                cfg,
                task,
                "task_cancel",
                details={"operation_id": control["operation_id"], "expected_effect": None},
            ):
                _set_group_cancel_request(cfg, task, control)
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
        return ("termination_pending" if control["terminate_running"] else "running_allowed"), None
    if task.state["projection"] in {"succeeded", "failed", "cancelled"}:
        if task.state["projection"] == "cancelled":
            retire_current_ready_generation(cfg, task)
        return "already_terminal", None
    return None, None


def _set_group_cancel_request(cfg: RootConfig, task: TaskRecord, control: dict[str, Any]) -> None:
    task.control.update(
        {
            "cancellation_requested_at": utc_now(),
            "cancellation_operation_id": control["operation_id"],
            "terminate_running": control["terminate_running"],
            "requested_by": cfg.machine_name,
        }
    )


def _dispatch_group_terminal_results(
    cfg: RootConfig, results: list[TerminalCommitResult], reservation_runtime_root: Path | None = None
) -> None:
    """Release local prelaunch occupancy and notify only after authority locks."""
    for result in results:
        if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
            from ..agent.context import resolve_execution_context
            from ..runtime.resources.reservations import release

            root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
            release(root, result.reservation_id, "group_cancelled_before_launch")
        if result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)


def reconcile_group_cancel_operations(
    cfg: RootConfig,
    group_name: str | None = None,
    *,
    include_legacy: bool = True,
    reservation_runtime_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Replay unfinished Group control effects and reconcile their status."""
    reconciled: list[dict[str, Any]] = []
    for operation_path in iter_active_operation_paths(cfg, "group_control", include_legacy=include_legacy):
        operation = read_json(operation_path)
        control = operation.get("group_control", {})
        operation_type = control.get("operation_type")
        if operation_type == WORKER_REMOVE_TYPE:
            result = reconcile_worker_removal(
                cfg,
                operation_path,
                group_name,
                reservation_runtime_root=reservation_runtime_root,
            )
            if result is not None:
                reconciled.append(result)
            continue
        if operation_type == "worker_remove":
            result = _reconcile_worker_remove_operation(cfg, operation_path, operation, group_name)
            if result is not None:
                reconciled.append(result)
            continue
        if operation_type != "cancel":
            continue
        if control.get("state") not in {"preparing", "converging", "waiting_ack", "blocked"}:
            continue
        name = control.get("group_name")
        if not name or (group_name is not None and name != group_name):
            continue
        if is_group_authority_isolated(cfg.shared_root):
            result = advance_indexed_cancel(cfg, operation_path, reservation_runtime_root=reservation_runtime_root)
            if result is not None:
                reconciled.append(result)
            continue
        post_commit_results = []
        with group_writer_lock(cfg, name):
            # The creator or another reconciler may have completed while we waited.
            if not operation_path.exists():
                continue
            operation = read_json(operation_path)
            control = operation.get("group_control", {})
            if control.get("state") not in {"preparing", "converging", "waiting_ack", "blocked"}:
                continue
            group_file = group_path(cfg.shared_root, name)
            if not group_file.exists():
                control.update({"state": "blocked", "completed_at": None, "blocked_reason": "group_missing"})
                control["updated_at"] = utc_now()
                operation["meta"]["revision"] += 1
                operation["meta"]["updated_at"] = utc_now()
                atomic_replace(operation_path, operation)
                reconciled.append(control)
                continue
            group_data = read_json(group_file)
            normalize_group_record(group_data)
            barriers = group_data["group"].get("cancellation_barriers", [])
            has_barrier = any(barrier.get("operation_id") == control["operation_id"] for barrier in barriers)
            if not has_barrier:
                control.update(
                    {
                        "state": "blocked",
                        "completed_at": None,
                        "blocked_reason": "cancellation_barrier_missing",
                        "updated_at": utc_now(),
                    }
                )
                operation["meta"]["revision"] += 1
                operation["meta"]["updated_at"] = utc_now()
                atomic_replace(operation_path, operation)
                archive_operation(cfg, "group_control", control["operation_id"], operation)
                snapshot = group_data.get("cancellation_operation") or {}
                if snapshot.get("operation_id") == control["operation_id"]:
                    group_data["cancellation_operation"] = control
                    group_data["meta"]["revision"] += 1
                    group_data["meta"]["updated_at"] = utc_now()
                    atomic_replace(group_file, group_data)
                reconciled.append(control)
                continue
            pending: dict[str, list[str]] = {}
            target_tasks = 0
            already_terminal = 0
            acknowledged = 0
            blocked = 0
            high_watermark = control["membership_high_watermark"]
            for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
                candidate = TaskRecord.from_dict(read_json(task_file))
                if candidate.group_name != name or (candidate.group_membership_sequence or 0) > high_watermark:
                    continue
                target_tasks += 1
                with task_lock(cfg.shared_root, candidate.task_id):
                    try:
                        task = load_task(cfg, candidate.task_id)
                    except FileNotFoundError:
                        already_terminal += 1
                        continue
                    if task.group_name != name or (task.group_membership_sequence or 0) > high_watermark:
                        target_tasks -= 1
                        continue
                    progress_key, terminal = _apply_group_cancel_locked(cfg, task, control)
                    if terminal is not None:
                        post_commit_results.append(terminal)
                    if progress_key == "blocked":
                        blocked += 1
                        continue
                if task.control.get("termination_acknowledged_at"):
                    acknowledged += 1
                    continue
                projection = task.state["projection"]
                if projection in {"succeeded", "failed", "cancelled"}:
                    already_terminal += 1
                elif control["terminate_running"] and projection == "running":
                    claim = task.claim_control.get("active_claim") or {}
                    machine = claim.get("machine_name") or task.placement_policy["home_machine"]
                    pending.setdefault(machine, []).append(task.task_id)
                elif control["terminate_running"] and projection == "blocked":
                    blocked += 1
            progress = control["progress"]
            progress.update(
                {
                    "target_tasks": target_tasks,
                    "already_terminal": already_terminal,
                    "termination_pending": sum(map(len, pending.values())),
                    "termination_acknowledged": acknowledged,
                    "blocked": blocked,
                }
            )
            control["pending_machine_acknowledgements"] = pending
            if blocked and not control["terminate_running"]:
                control.update(
                    {"state": "blocked", "completed_at": None, "blocked_reason": "orphaned_tasks_require_resolution"}
                )
            elif not control["terminate_running"]:
                control.update(
                    {
                        "state": "completed",
                        "completed_at": control.get("completed_at") or utc_now(),
                        "blocked_reason": None,
                    }
                )
            elif pending:
                control.update({"state": "waiting_ack", "completed_at": None, "blocked_reason": None})
            elif blocked:
                control.update(
                    {"state": "blocked", "completed_at": None, "blocked_reason": "orphaned_tasks_require_resolution"}
                )
            else:
                control.update({"state": "completed", "completed_at": utc_now(), "blocked_reason": None})
            control["updated_at"] = utc_now()
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            atomic_replace(operation_path, operation)
            if control["state"] == "completed":
                archive_operation(cfg, "group_control", control["operation_id"], operation)
            snapshot = group_data.get("cancellation_operation") or {}
            if snapshot.get("operation_id") == control["operation_id"]:
                group_data["cancellation_operation"] = control
                group_data["meta"]["revision"] += 1
                group_data["meta"]["updated_at"] = utc_now()
                atomic_replace(group_file, group_data)
            reconciled.append(control)
        _dispatch_group_terminal_results(cfg, post_commit_results, reservation_runtime_root)
    return reconciled


def _reconcile_worker_remove_operation(
    cfg: RootConfig, operation_path: Path, operation: dict[str, Any], group_name: str | None
) -> dict[str, Any] | None:
    control = operation.get("group_control", {})
    if control.get("state") not in {"converging", "waiting_ack", "blocked"}:
        return None
    name = control.get("group_name")
    machine = control.get("machine_name")
    if not name or not machine or (group_name is not None and name != group_name):
        return None
    with group_writer_lock(cfg, name):
        if not operation_path.exists():
            return None
        operation = read_json(operation_path)
        control = operation.get("group_control", {})
        if control.get("state") not in {"converging", "waiting_ack", "blocked"}:
            return None
        if control.get("group_name") != name or control.get("machine_name") != machine:
            return None
        if has_group_authority_cutover(cfg.shared_root):
            control.update(
                state="blocked",
                blocked_reason="legacy_worker_incarnation_unknown",
                completed_at=None,
                updated_at=utc_now(),
            )
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            group_file = group_path(cfg.shared_root, name)
            if group_file.exists():
                data = read_json(group_file)
                if (data.get("worker_control") or {}).get("operation_id") == control["operation_id"]:
                    data["worker_control"] = control
                    data["meta"]["revision"] += 1
                    data["meta"]["updated_at"] = utc_now()
                    atomic_replace(group_file, data)
            archive_operation(cfg, "group_control", control["operation_id"], operation)
            return control
        group_file = group_path(cfg.shared_root, name)
        if not group_file.exists():
            return None
        group_data = read_json(group_file)
        normalize_group_record(group_data)
        workers = group_data["group"].get("worker_set", {})
        worker = workers.get(machine)
        if worker is None:
            control.update(
                {
                    "state": "completed",
                    "completed_at": control.get("completed_at") or utc_now(),
                    "updated_at": utc_now(),
                }
            )
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            archive_operation(cfg, "group_control", control["operation_id"], operation)
            return control
        blockers: list[str] = []
        for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(task_file))
            if task.group_name != name:
                continue
            claim = task.claim_control.get("active_claim") or {}
            if claim.get("machine_name") == machine:
                if control.get("terminate_running"):
                    with task_lock(cfg.shared_root, task.task_id):
                        task = load_task(cfg, task.task_id)
                        current_claim = task.claim_control.get("active_claim") or {}
                        if current_claim.get("machine_name") == machine:
                            task.control.update(
                                {
                                    "cancellation_requested_at": utc_now(),
                                    "terminate_running": True,
                                    "requested_by": cfg.machine_name,
                                    "cancellation_operation_id": control["operation_id"],
                                }
                            )
                            task.meta["revision"] += 1
                            task.meta["updated_at"] = utc_now()
                            save_task(cfg, task)
                blockers.append(task.task_id)
            if task.state["projection"] == "queued" and task.placement_policy["home_machine"] == machine:
                blockers.append(task.task_id)
        control["blockers"] = sorted(set(blockers))
        if blockers:
            control.update({"state": "waiting_ack", "completed_at": None, "updated_at": utc_now()})
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            write_active_operation(cfg, "group_control", control["operation_id"], operation)
        else:
            if worker.get("state") != "removing":
                worker["state"] = "removing"
                worker["state_epoch"] += 1
                worker["remove_requested_at"] = worker.get("remove_requested_at") or utc_now()
                group_data["group"]["worker_set_epoch"] += 1
            control.update(
                {
                    "state": "completed",
                    "completed_at": control.get("completed_at") or utc_now(),
                    "updated_at": utc_now(),
                }
            )
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
        snapshot = group_data.get("worker_control") or {}
        if snapshot.get("operation_id") == control["operation_id"]:
            group_data["worker_control"] = control
        group_data["meta"]["revision"] += 1
        group_data["meta"]["updated_at"] = utc_now()
        atomic_replace(group_file, group_data)
        if control["state"] == "completed":
            archive_operation(cfg, "group_control", control["operation_id"], operation)
        return control


def create_group(cfg: RootConfig, name: str, workers: list[str] | None = None) -> dict[str, Any]:
    path = group_path(cfg.shared_root, validate_group_name(name) or name)
    _finalize_pending_submission_before_group_mutation(cfg, name, path)
    with group_writer_lock(cfg, name):
        path = group_path(cfg.shared_root, name)
        if path.exists():
            return read_json(path)
        data = new_group(name, cfg.machine_name)
        initial_workers = [cfg.machine_name] if workers is None else list(workers)
        seen: set[str] = set()
        for machine in initial_workers:
            validate_identifier(machine, "worker_machine")
            if machine in seen:
                raise ValueError(f"group workers must not contain duplicate machine {machine!r}.")
            seen.add(machine)
            data["group"]["worker_set"][machine] = new_worker_member()
        atomic_replace(path, data)
        return data


def show_group(cfg: RootConfig, name: str) -> dict[str, Any]:
    name = validate_group_name(name) or name
    reconcile_group_cancel_operations(cfg, name)
    result = read_group(cfg.shared_root, name)
    normalize_group_record(result)
    return result


def change_worker(
    cfg: RootConfig,
    group_name: str,
    machine: str,
    action: str,
    *,
    terminate_running: bool = False,
    role: str | None = None,
    gpu_limit_gpus: int | None = None,
    has_gpu_limit: bool = False,
) -> dict[str, Any]:
    validate_identifier(machine, "worker_machine")
    path = group_path(cfg.shared_root, validate_group_name(group_name) or group_name)
    _finalize_pending_submission_before_group_mutation(cfg, group_name, path)
    with group_writer_lock(cfg, group_name):
        path = group_path(cfg.shared_root, group_name)
        if is_group_ready_members_root(cfg):
            assert_group_ready_members_writable(cfg)
        data = read_json(path)
        normalize_group_record(data)
        pending = data["group"].get("pending_submission_commit")
        if pending:
            operation_path = submission_path(cfg.shared_root, pending["operation_id"])
            if not operation_path.exists() or read_json(operation_path)["submission"].get("state") != "committed":
                raise RuntimeError(f"Group {group_name!r} has pending submission commit {pending['operation_id']!r}.")
            raise RuntimeError(f"Group {group_name!r} received a concurrent submission commit; retry the mutation.")
        workers = data["group"]["worker_set"]
        previous_workers = {worker_name: dict(worker) for worker_name, worker in workers.items()}
        projection_routes = primary_projection_routes_for_group(cfg, group_name)
        worker_changed = True
        if has_gpu_limit:
            gpu_limit_gpus = validate_gpu_limit(gpu_limit_gpus, "gpu_limit_gpus")
        if action == "add":
            if role not in {None, "primary", "borrow"}:
                raise ValueError("role must be 'primary' or 'borrow'.")
            selected_role = role or "primary"
            current = workers.get(machine)
            if current is None:
                workers[machine] = new_worker_member(
                    scheduling_role=selected_role,
                    gpu_limit_gpus=gpu_limit_gpus,
                )
            else:
                if current["state"] == "draining":
                    raise ValueError(f"worker {machine!r} is draining; use 'resume' before adding it again.")
                if current["state"] == "removing" or current.get("removal_operation_id") is not None:
                    raise ValueError(
                        f"worker {machine!r} is being removed; wait for removal to finish instead of cancelling it."
                    )
                requested_role = role if role is not None else current["scheduling_role"]
                requested_limit = gpu_limit_gpus if has_gpu_limit else current["gpu_limit_gpus"]
                if current["scheduling_role"] != requested_role or current["gpu_limit_gpus"] != requested_limit:
                    raise ValueError(f"worker {machine!r} already exists with a different policy; use 'set'.")
                worker_changed = False

        elif machine not in workers:
            raise ValueError(f"machine {machine!r} is not a Worker Set member.")
        elif action == "set":
            if role is None and not has_gpu_limit:
                raise ValueError("machines set requires --role or --gpu-limit-gpus.")
            worker = workers[machine]
            selected_role = role or worker["scheduling_role"]
            if selected_role not in {"primary", "borrow"}:
                raise ValueError("role must be 'primary' or 'borrow'.")
            selected_limit = gpu_limit_gpus if has_gpu_limit else worker["gpu_limit_gpus"]
            selected_state = worker["state"]
            changed = (
                worker["scheduling_role"] != selected_role
                or worker["gpu_limit_gpus"] != selected_limit
                or worker["state"] != selected_state
            )
            worker_changed = changed
            worker.update(
                {
                    "scheduling_role": selected_role,
                    "gpu_limit_gpus": selected_limit,
                    "state": selected_state,
                }
            )
            if changed:
                worker["state_epoch"] += 1
        elif action == "resume":
            worker = workers[machine]
            if worker["state"] == "removing" or worker.get("removal_operation_id") is not None:
                raise ValueError(f"worker {machine!r} is being removed; wait for removal to finish before resuming it.")
            if worker["state"] == "active":
                worker_changed = False
            elif worker["state"] == "draining":
                worker["state"] = "active"
                worker_changed = True
            else:
                raise ValueError(f"worker {machine!r} has an invalid lifecycle state.")
        elif action == "drain":
            if workers[machine]["state"] == "removing":
                workers[machine].pop("removal_operation_id", None)
            workers[machine]["state"] = "draining"
            workers[machine]["state_epoch"] += 1
            workers[machine]["drain_requested_at"] = utc_now()
        elif action == "remove":
            if can_bind_worker_removal(cfg):
                return begin_worker_removal_locked(cfg, data, machine, terminate_running=terminate_running)
            if has_group_authority_cutover(cfg.shared_root):
                raise RuntimeError("Group authority migration is incomplete; retry after automatic recovery")
            operation_id = new_id()
            workers[machine]["state"] = "draining"
            workers[machine]["state_epoch"] += 1
            workers[machine]["drain_requested_at"] = utc_now()
            data["group"]["worker_set_epoch"] += 1
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            operation_path = active_operation_path(cfg, "group_control", operation_id)
            operation = {
                "meta": {
                    "schema_version": SCHEMA_VERSION,
                    "revision": 1,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "updated_by": {
                        "actor_type": "cli",
                        "machine_name": cfg.machine_name,
                        "process_id": str(os.getpid()),
                    },
                },
                "group_control": {
                    "operation_id": operation_id,
                    "operation_type": "worker_remove",
                    "group_name": group_name,
                    "machine_name": machine,
                    "state": "converging",
                    "terminate_running": terminate_running,
                    "blockers": [],
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "completed_at": None,
                },
            }
            from ..runtime.group_discovery.service import publish_group_locator_for_transition

            # QQTOOLS-COMPAT-0017: legacy removal remains dual-published while
            # the historical service is active; the locator is required once
            # the writer fence has been installed.
            publish_group_locator_for_transition(cfg, group_name, "control", "group_operation")
            write_active_operation(cfg, "group_control", operation_id, operation)
            blockers: list[str] = []
            for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
                task = TaskRecord.from_dict(read_json(task_file))
                if task.group_name != group_name:
                    continue
                claim = task.claim_control.get("active_claim") or {}
                if claim.get("machine_name") == machine:
                    if terminate_running:
                        with task_lock(cfg.shared_root, task.task_id):
                            task = load_task(cfg, task.task_id)
                            current_claim = task.claim_control.get("active_claim") or {}
                            if current_claim.get("machine_name") == machine:
                                task.control.update(
                                    {
                                        "cancellation_requested_at": utc_now(),
                                        "terminate_running": True,
                                        "requested_by": cfg.machine_name,
                                        "cancellation_operation_id": operation_id,
                                    }
                                )
                                task.meta["revision"] += 1
                                task.meta["updated_at"] = utc_now()
                                save_task(cfg, task)
                    blockers.append(task.task_id)
                if task.state["projection"] == "queued" and task.placement_policy["home_machine"] == machine:
                    blockers.append(task.task_id)
            operation["group_control"]["blockers"] = sorted(set(blockers))
            if blockers:
                operation["group_control"]["state"] = "waiting_ack"
            else:
                workers[machine]["state"] = "removing"
                workers[machine]["state_epoch"] += 1
                workers[machine]["remove_requested_at"] = utc_now()
                operation["group_control"].update({"state": "completed", "completed_at": utc_now()})
            operation["group_control"]["updated_at"] = utc_now()
            atomic_replace(operation_path, operation)
            if operation["group_control"]["state"] == "completed":
                archive_operation(cfg, "group_control", operation_id, operation)
            data["worker_control"] = operation["group_control"]
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            with primary_route_update_transaction(
                cfg, sorted(set(projection_routes + [("shared", machine), ("home", machine)]))
            ):
                atomic_replace(path, data)
                sync_primary_ready_group(cfg, group_name, previous_workers=previous_workers)
            return data
        else:
            raise ValueError(f"unknown Worker Set action {action!r}.")
        if not worker_changed:
            return data
        if worker_changed:
            data["group"]["worker_set_epoch"] += 1
            if action in {"add", "set", "resume"} and machine in workers:
                workers[machine]["state_epoch"] = data["group"]["worker_set_epoch"]
        data["meta"]["revision"] += 1
        data["meta"]["updated_at"] = utc_now()
        if worker_changed:
            # Keep borrow admission closed from the role mutation until the
            # corresponding primary candidate projection is published.
            with primary_route_update_transaction(
                cfg, sorted(set(projection_routes + [("shared", machine), ("home", machine)]))
            ):
                atomic_replace(path, data)
                sync_primary_ready_group(cfg, group_name, previous_workers=previous_workers)
        else:
            atomic_replace(path, data)
        return data


def group_retry_failed(cfg: RootConfig, name: str) -> dict[str, Any]:
    """Retry the current failed Tasks in a Group and classify skipped work.

    Group membership is selected while the Group writer fence is held.  The
    selected Task and Attempt revisions are then rechecked after releasing the
    Group fence, because ``retry`` owns the complete schema/Group/Task lock
    sequence and must not be called while a Group lock is held.
    """
    name = validate_group_name(name) or name
    _finalize_pending_submission_before_group_mutation(cfg, name, group_path(cfg.shared_root, name))

    # Each entry is the authoritative snapshot at one Group revision.  The
    # snapshot keeps retry from expanding to Tasks added after selection and
    # lets a concurrent Task transition be reported as ``other``.
    selected: list[dict[str, Any]] = []
    retried_task_ids: list[str] = []
    skipped: dict[str, list[str]] = {"blocked": [], "orphaned": [], "other": []}
    with group_writer_lock(cfg, name):
        group_data = read_group(cfg.shared_root, name)
        normalize_group_record(group_data)
        group = group_data["group"]
        if group.get("name") != name:
            raise ValueError(f"Group record name does not match requested Group {name!r}.")
        selection_revision = group_data["meta"].get("revision")
        if type(selection_revision) is not int or selection_revision < 1:
            raise ValueError("Group revision is invalid.")
        next_membership_sequence = group.get("next_membership_sequence")
        if type(next_membership_sequence) is not int or next_membership_sequence < 1:
            raise ValueError("Group membership sequence is invalid.")
        membership_high_watermark = next_membership_sequence - 1

        for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(task_file))
            membership_sequence = task.group_membership_sequence
            if (
                task.group_name != name
                or type(membership_sequence) is not int
                or membership_sequence < 1
                or membership_sequence > membership_high_watermark
            ):
                continue
            current_attempt = None
            current_attempt_number = task.attempt_control.get("current_attempt_number")
            current_attempt_id = task.attempt_control.get("current_attempt_id")
            if current_attempt_number is not None:
                try:
                    current_attempt = AttemptRecord.from_dict(
                        read_json(attempt_path(cfg.shared_root, task.task_id, current_attempt_number))
                    )
                except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
                    current_attempt = None

            if task.state["projection"] == "blocked":
                if (
                    current_attempt is not None
                    and (current_attempt_id is None or current_attempt_id == current_attempt.attempt_id)
                    and current_attempt.phase == "orphaned"
                ):
                    skipped["orphaned"].append(task.task_id)
                else:
                    skipped["blocked"].append(task.task_id)
                continue
            if (
                task.state["projection"] != "failed"
                or current_attempt is None
                or (current_attempt_id is not None and current_attempt_id != current_attempt.attempt_id)
                or current_attempt.phase != "failed"
            ):
                skipped["other"].append(task.task_id)
                continue
            selected.append(
                {
                    "task_id": task.task_id,
                    "task_revision": task.meta["revision"],
                    "group_membership_sequence": membership_sequence,
                    "attempt_number": current_attempt_number,
                    "attempt_id": current_attempt.attempt_id,
                    "group_revision": selection_revision,
                }
            )

    for selection in sorted(selected, key=lambda item: item["task_id"]):
        task_id = selection["task_id"]
        try:
            current_task = load_task(cfg, task_id)
            current_attempt_number = current_task.attempt_control.get("current_attempt_number")
            current_attempt_id = current_task.attempt_control.get("current_attempt_id")
            if (
                current_task.group_name != name
                or current_task.group_membership_sequence != selection["group_membership_sequence"]
                or current_task.meta["revision"] != selection["task_revision"]
                or current_task.state["projection"] != "failed"
                or current_attempt_number != selection["attempt_number"]
                or (current_attempt_id is not None and current_attempt_id != selection["attempt_id"])
            ):
                skipped["other"].append(task_id)
                continue
            current_attempt = AttemptRecord.from_dict(
                read_json(attempt_path(cfg.shared_root, task_id, current_attempt_number))
            )
            if current_attempt.attempt_id != selection["attempt_id"] or current_attempt.phase != "failed":
                skipped["other"].append(task_id)
                continue
            retry(cfg, task_id)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, RuntimeError):
            # A concurrent transition is deliberately not reclassified as a
            # newly eligible blocked/orphaned Task.
            skipped["other"].append(task_id)
        else:
            # ``retry`` only returns after its own authoritative transition.
            # Keep the result deterministic even if the Task list was not.
            selected_task_id = task_id
            if selected_task_id not in retried_task_ids:
                retried_task_ids.append(selected_task_id)

    retried_task_ids.sort()
    for task_ids in skipped.values():
        task_ids.sort()

    return {
        "group": name,
        "retried_task_ids": retried_task_ids,
        "retried_count": len(retried_task_ids),
        "skipped": skipped,
    }
