"""Group command workflows for qexp."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..lifecycle import (TerminalTransition, commit_terminal_transition_locked,
                         dispatch_task_lifecycle_hooks_noexcept)
from ..runtime.claims import archive_claim
from ..runtime.locks import group_lock, task_lock
from ..runtime.paths import attempt_path, group_path, shared_paths, submission_path
from ..runtime.records import (AttemptRecord, SCHEMA_VERSION, TaskRecord, new_group, new_id,
                               new_worker_member, normalize_group_record, utc_now,
                               validate_borrow_limit, validate_group_name, validate_identifier)
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.worker_encoding import write_group_record
from ..runtime.tasks import load_task, save_task
from ..runtime.ready import (
    primary_projection_transaction,
    primary_projection_routes_for_group,
    retire_current_ready_generation,
    sync_primary_ready_group,
)
from ..runtime.active_operations import (
    active_operation_path,
    archive_operation,
    iter_active_operation_paths,
    write_active_operation,
)
from ..runtime.submission import finalize_submission_group
from .task import has_cleanup_operation, is_cleanup_blocked, retry


def _finalize_pending_submission_before_group_mutation(
    cfg: RootConfig, name: str, path: Path
) -> None:
    """Finalize a committed submission before taking the Group mutation lock."""
    with group_lock(cfg.shared_root, name):
        if not path.exists():
            return
        data = read_json(path)
        normalize_group_record(data)
        pending = data["group"].get("pending_submission_commit") or {}
        if not pending:
            return
        operation_file = submission_path(cfg.shared_root, pending["operation_id"])
        if not operation_file.exists():
            raise RuntimeError(
                f"Group {name!r} has pending submission commit {pending['operation_id']!r}."
            )
        operation = read_json(operation_file)["submission"]
        if operation.get("state") != "committed":
            raise RuntimeError(
                f"Group {name!r} has pending submission commit {pending['operation_id']!r}."
            )
    finalize_submission_group(cfg, operation)


def group_control(
    cfg: RootConfig,
    name: str,
    action: str,
    *,
    terminate_running: bool = False,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any]:
    from ..machine_runtime import resolve_execution_context

    reservation_runtime_root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    path = group_path(cfg.shared_root, validate_group_name(name) or name)
    _finalize_pending_submission_before_group_mutation(cfg, name, path)
    post_commit_results = []
    with group_lock(cfg.shared_root, name):
        data = read_json(path)
        group = data["group"]
        pending = group.get("pending_submission_commit")
        if pending:
            operation_path = submission_path(cfg.shared_root, pending["operation_id"])
            if (
                not operation_path.exists()
                or read_json(operation_path)["submission"].get("state") != "committed"
            ):
                raise RuntimeError(
                    f"Group {name!r} has pending submission commit {pending['operation_id']!r}."
                )
            raise RuntimeError(
                f"Group {name!r} received a concurrent submission commit; retry the mutation."
            )
        if action == "cancel":
            operation_id = new_id()
            high_watermark = group["next_membership_sequence"] - 1
            operation_path = active_operation_path(
                cfg, "group_control", operation_id
            )
            operation = {"meta": {"schema_version": SCHEMA_VERSION, "revision": 1, "created_at": utc_now(),
                "updated_at": utc_now(), "updated_by": {"actor_type": "cli", "machine_name": cfg.machine_name,
                "process_id": str(os.getpid())}}, "group_control": {"operation_id": operation_id,
                "operation_type": "cancel", "group_name": name, "state": "preparing",
                "group_revision_at_start": data["meta"]["revision"], "dispatch_epoch_at_start": group["dispatch_epoch"],
                "membership_high_watermark": high_watermark, "terminate_running": terminate_running,
                "progress": {"target_tasks": 0, "already_terminal": 0, "queued_cancelled": 0,
                "prelaunch_cancelled": 0, "running_allowed": 0, "termination_pending": 0,
                "termination_acknowledged": 0, "blocked": 0}, "pending_machine_acknowledgements": {},
                "created_at": utc_now(), "updated_at": utc_now(), "completed_at": None, "blocked_reason": None}}
            write_active_operation(cfg, "group_control", operation_id, operation)
            group["cancellation_barriers"].append({"operation_id": operation_id,
                "membership_high_watermark": high_watermark, "terminate_running": terminate_running,
                "created_at": utc_now()})
            data["cancellation_operation"] = operation["group_control"]
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            write_group_record(cfg, path, data)
            operation["group_control"]["state"] = "converging"
            operation["group_control"]["updated_at"] = utc_now()
            write_active_operation(cfg, "group_control", operation_id, operation)
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
                    has_saved_task = False
                    if is_cleanup_blocked(task) or has_cleanup_operation(cfg, task.task_id):
                        operation["group_control"]["progress"]["already_terminal"] += 1
                        continue
                    claim = task.claim_control.get("active_claim") or {}
                    if task.state["projection"] == "queued" and not claim:
                        task.state.update({"projection": "cancelled", "reason": "group_cancelled"})
                        operation["group_control"]["progress"]["queued_cancelled"] += 1
                    elif claim.get("launch_state") == "claimed":
                        result = commit_terminal_transition_locked(
                            cfg, task, TerminalTransition(task.task_id, claim["attempt_id"],
                                task.attempt_control["current_attempt_number"], claim["fencing_token"],
                                "cancelled", "group_cancelled_before_launch", None,
                                frozenset({"running"}), frozenset({"claimed"}), "active",
                                allow_missing_attempt=True))
                        post_commit_results.append(result)
                        has_saved_task = result.outcome == "committed"
                        progress_key = (
                            "prelaunch_cancelled" if result.outcome == "committed" else "blocked"
                        )
                        operation["group_control"]["progress"][progress_key] += 1
                    elif task.state["projection"] == "running":
                        task.control.update({"cancellation_requested_at": utc_now(),
                                             "cancellation_operation_id": operation_id,
                                             "terminate_running": terminate_running,
                                             "requested_by": cfg.machine_name})
                        if terminate_running:
                            machine = (claim.get("machine_name") or task.placement_policy["home_machine"])
                            operation["group_control"]["pending_machine_acknowledgements"].setdefault(machine, []).append(task.task_id)
                            operation["group_control"]["progress"]["termination_pending"] += 1
                        else:
                            operation["group_control"]["progress"]["running_allowed"] += 1
                    elif task.state["projection"] in {"succeeded", "failed", "cancelled"}:
                        operation["group_control"]["progress"]["already_terminal"] += 1
                    if not has_saved_task:
                        task.meta["revision"] += 1
                        task.meta["updated_at"] = utc_now()
                        save_task(cfg, task)
                    if task.state["projection"] == "cancelled":
                        retire_current_ready_generation(cfg, task)
            progress = operation["group_control"]["progress"]
            if not terminate_running or progress["termination_pending"] == 0:
                operation["group_control"].update({"state": "completed", "completed_at": utc_now()})
            else:
                operation["group_control"]["state"] = "waiting_ack"
            operation["group_control"]["updated_at"] = utc_now()
            atomic_replace(operation_path, operation)
            if operation["group_control"]["state"] == "completed":
                operation_path = archive_operation(
                    cfg, "group_control", operation_id, operation
                )
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
        write_group_record(cfg, path, data)
        result_data = data
    for result in post_commit_results:
        if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
            from ..runtime.reservations import release
            release(reservation_runtime_root, result.reservation_id, "group_cancelled_before_launch")
        if result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
    return result_data


def reconcile_group_cancel_operations(
        cfg: RootConfig, group_name: str | None = None, *,
        include_legacy: bool = True) -> list[dict[str, Any]]:
    """Rebuild durable Group control operation status from current Task truth."""
    reconciled: list[dict[str, Any]] = []
    for operation_path in iter_active_operation_paths(
        cfg, "group_control", include_legacy=include_legacy
    ):
        operation = read_json(operation_path)
        control = operation.get("group_control", {})
        operation_type = control.get("operation_type")
        if operation_type == "worker_remove":
            result = _reconcile_worker_remove_operation(cfg, operation_path, operation, group_name)
            if result is not None:
                reconciled.append(result)
            continue
        if operation_type != "cancel":
            continue
        if control.get("state") not in {"converging", "waiting_ack", "blocked"}:
            continue
        name = control.get("group_name")
        if not name or (group_name is not None and name != group_name):
            continue
        with group_lock(cfg.shared_root, name):
            group_file = group_path(cfg.shared_root, name)
            if not group_file.exists():
                continue
            group_data = read_json(group_file)
            normalize_group_record(group_data)
            barriers = group_data["group"].get("cancellation_barriers", [])
            has_barrier = any(
                barrier.get("operation_id") == control["operation_id"] for barrier in barriers
            )
            if not has_barrier:
                control.update({"state": "blocked", "completed_at": None,
                                "blocked_reason": "cancellation_barrier_missing",
                                "updated_at": utc_now()})
                operation["meta"]["revision"] += 1
                operation["meta"]["updated_at"] = utc_now()
                atomic_replace(operation_path, operation)
                archive_operation(
                    cfg, "group_control", control["operation_id"], operation
                )
                snapshot = group_data.get("cancellation_operation") or {}
                if snapshot.get("operation_id") == control["operation_id"]:
                    group_data["cancellation_operation"] = control
                    group_data["meta"]["revision"] += 1
                    group_data["meta"]["updated_at"] = utc_now()
                    write_group_record(cfg, group_file, group_data)
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
                if (candidate.group_name != name
                        or (candidate.group_membership_sequence or 0) > high_watermark):
                    continue
                target_tasks += 1
                with task_lock(cfg.shared_root, candidate.task_id):
                    task = load_task(cfg, candidate.task_id)
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
            progress.update({"target_tasks": target_tasks,
                             "already_terminal": already_terminal,
                             "termination_pending": sum(map(len, pending.values())),
                             "termination_acknowledged": acknowledged,
                             "blocked": blocked})
            control["pending_machine_acknowledgements"] = pending
            if not control["terminate_running"]:
                control.update({"state": "completed", "completed_at":
                                control.get("completed_at") or utc_now(),
                                "blocked_reason": None})
            elif pending:
                control.update({"state": "waiting_ack", "completed_at": None,
                                "blocked_reason": None})
            elif blocked:
                control.update({"state": "blocked", "completed_at": None,
                                "blocked_reason": "orphaned_tasks_require_resolution"})
            else:
                control.update({"state": "completed", "completed_at": utc_now(),
                                "blocked_reason": None})
            control["updated_at"] = utc_now()
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            atomic_replace(operation_path, operation)
            if control["state"] == "completed":
                archive_operation(
                    cfg, "group_control", control["operation_id"], operation
                )
            snapshot = group_data.get("cancellation_operation") or {}
            if snapshot.get("operation_id") == control["operation_id"]:
                group_data["cancellation_operation"] = control
                group_data["meta"]["revision"] += 1
                group_data["meta"]["updated_at"] = utc_now()
                write_group_record(cfg, group_file, group_data)
            reconciled.append(control)
    return reconciled


def _reconcile_worker_remove_operation(
        cfg: RootConfig, operation_path: Path, operation: dict[str, Any],
        group_name: str | None) -> dict[str, Any] | None:
    control = operation.get("group_control", {})
    if control.get("state") not in {"converging", "waiting_ack"}:
        return None
    name = control.get("group_name")
    machine = control.get("machine_name")
    if not name or not machine or (group_name is not None and name != group_name):
        return None
    with group_lock(cfg.shared_root, name):
        group_file = group_path(cfg.shared_root, name)
        if not group_file.exists():
            return None
        group_data = read_json(group_file)
        normalize_group_record(group_data)
        workers = group_data["group"].get("worker_set", {})
        worker = workers.get(machine)
        if worker is None:
            control.update({"state": "completed", "completed_at": control.get("completed_at") or utc_now(),
                            "updated_at": utc_now()})
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
                            task.control.update({"cancellation_requested_at": utc_now(), "terminate_running": True,
                                                 "requested_by": cfg.machine_name,
                                                 "cancellation_operation_id": control["operation_id"]})
                            task.meta["revision"] += 1
                            task.meta["updated_at"] = utc_now()
                            save_task(cfg, task)
                blockers.append(task.task_id)
            if task.state["projection"] == "queued" and task.placement_policy["home_machine"] == machine:
                blockers.append(task.task_id)
        control["blockers"] = sorted(set(blockers))
        if blockers:
            control.update({"state": "waiting_ack", "completed_at": None,
                            "updated_at": utc_now()})
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            write_active_operation(cfg, "group_control", control["operation_id"], operation)
        else:
            if worker.get("state") != "removing":
                worker["state"] = "removing"
                worker["state_epoch"] += 1
                worker["remove_requested_at"] = worker.get("remove_requested_at") or utc_now()
                group_data["group"]["worker_set_epoch"] += 1
            control.update({"state": "completed", "completed_at": control.get("completed_at") or utc_now(),
                            "updated_at": utc_now()})
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
        snapshot = group_data.get("worker_control") or {}
        if snapshot.get("operation_id") == control["operation_id"]:
            group_data["worker_control"] = control
        group_data["meta"]["revision"] += 1
        group_data["meta"]["updated_at"] = utc_now()
        write_group_record(cfg, group_file, group_data)
        if control["state"] == "completed":
            archive_operation(cfg, "group_control", control["operation_id"], operation)
        return control


def create_group(cfg: RootConfig, name: str, workers: list[str] | None = None) -> dict[str, Any]:
    path = group_path(cfg.shared_root, validate_group_name(name) or name)
    with group_lock(cfg.shared_root, name):
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
        write_group_record(cfg, path, data)
        return data


def show_group(cfg: RootConfig, name: str) -> dict[str, Any]:
    name = validate_group_name(name) or name
    reconcile_group_cancel_operations(cfg, name)
    result = read_json(group_path(cfg.shared_root, name))
    normalize_group_record(result)
    return result


def change_worker(cfg: RootConfig, group_name: str, machine: str, action: str,
                  *, terminate_running: bool = False, role: str | None = None,
                  borrow_limit_gpus: int | None = None,
                  has_borrow_limit: bool = False) -> dict[str, Any]:
    path = group_path(cfg.shared_root, validate_group_name(group_name) or group_name)
    _finalize_pending_submission_before_group_mutation(cfg, group_name, path)
    with group_lock(cfg.shared_root, group_name):
        data = read_json(path)
        normalize_group_record(data)
        pending = data["group"].get("pending_submission_commit")
        if pending:
            operation_path = submission_path(cfg.shared_root, pending["operation_id"])
            if (
                not operation_path.exists()
                or read_json(operation_path)["submission"].get("state") != "committed"
            ):
                raise RuntimeError(
                    f"Group {group_name!r} has pending submission commit {pending['operation_id']!r}."
                )
            raise RuntimeError(
                f"Group {group_name!r} received a concurrent submission commit; retry the mutation."
            )
        workers = data["group"]["worker_set"]
        projection_routes = primary_projection_routes_for_group(cfg, group_name)
        worker_changed = True
        if has_borrow_limit:
            borrow_limit_gpus = validate_borrow_limit(borrow_limit_gpus, "max_gpus")
        if action == "add":
            validate_identifier(machine, "worker_machine")
            if role not in {None, "primary", "borrow"}:
                raise ValueError("role must be 'primary' or 'borrow'.")
            selected_role = role or "primary"
            if selected_role == "primary" and borrow_limit_gpus is not None:
                raise ValueError("primary Worker cannot have a borrow limit.")
            current = workers.get(machine)
            if current is None:
                workers[machine] = new_worker_member(
                    scheduling_role=selected_role,
                    borrow_limit_gpus=borrow_limit_gpus,
                )
            else:
                current_role = selected_role if role is not None else current["scheduling_role"]
                current_limit = borrow_limit_gpus if has_borrow_limit else (
                    None if role == "primary" else current["borrow_limit_gpus"]
                )
                if current_role == "primary":
                    current_limit = None
                current.update({
                    "scheduling_role": current_role,
                    "borrow_limit_gpus": current_limit,
                    "state": "borrow" if current_role == "borrow" else "active",
                })
                current["state_epoch"] += 1
        elif machine not in workers:
            raise ValueError(f"machine {machine!r} is not a Worker Set member.")
        elif action == "set":
            if role is None and not has_borrow_limit:
                raise ValueError("machines set requires --role or --max-gpus.")
            worker = workers[machine]
            selected_role = role or worker["scheduling_role"]
            if selected_role not in {"primary", "borrow"}:
                raise ValueError("role must be 'primary' or 'borrow'.")
            selected_limit = borrow_limit_gpus if has_borrow_limit else worker["borrow_limit_gpus"]
            if selected_role == "primary" and has_borrow_limit and selected_limit is not None:
                raise ValueError("primary Worker cannot have a borrow limit.")
            if selected_role == "borrow" and role == "borrow" and not has_borrow_limit:
                selected_limit = None
            if selected_role == "primary":
                selected_limit = None
            selected_state = worker["state"]
            if selected_state in {"active", "borrow"}:
                selected_state = "borrow" if selected_role == "borrow" else "active"
            changed = (
                worker["scheduling_role"] != selected_role
                or worker["borrow_limit_gpus"] != selected_limit
                or worker["state"] != selected_state
            )
            worker_changed = changed
            worker.update({
                "scheduling_role": selected_role,
                "borrow_limit_gpus": selected_limit,
                "state": selected_state,
            })
            if changed:
                worker["state_epoch"] += 1
        elif action == "drain":
            workers[machine]["state"] = "draining"
            workers[machine]["state_epoch"] += 1
            workers[machine]["drain_requested_at"] = utc_now()
        elif action == "remove":
            operation_id = new_id()
            workers[machine]["state"] = "draining"
            workers[machine]["state_epoch"] += 1
            workers[machine]["drain_requested_at"] = utc_now()
            data["group"]["worker_set_epoch"] += 1
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            operation_path = active_operation_path(
                cfg, "group_control", operation_id
            )
            operation = {"meta": {"schema_version": SCHEMA_VERSION, "revision": 1, "created_at": utc_now(),
                "updated_at": utc_now(), "updated_by": {"actor_type": "cli", "machine_name": cfg.machine_name,
                "process_id": str(os.getpid())}}, "group_control": {"operation_id": operation_id,
                "operation_type": "worker_remove", "group_name": group_name, "machine_name": machine,
                "state": "converging", "terminate_running": terminate_running, "blockers": [],
                "created_at": utc_now(), "updated_at": utc_now(), "completed_at": None}}
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
                                task.control.update({"cancellation_requested_at": utc_now(), "terminate_running": True,
                                                     "requested_by": cfg.machine_name,
                                                     "cancellation_operation_id": operation_id})
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
                archive_operation(
                    cfg, "group_control", operation_id, operation
                )
            data["worker_control"] = operation["group_control"]
            data["meta"]["revision"] += 1
            data["meta"]["updated_at"] = utc_now()
            with primary_projection_transaction(
                cfg, sorted(set(projection_routes + [("shared", machine), ("home", machine)]))
            ):
                write_group_record(cfg, path, data)
                sync_primary_ready_group(cfg, group_name)
            return data
        else:
            raise ValueError(f"unknown Worker Set action {action!r}.")
        if worker_changed:
            data["group"]["worker_set_epoch"] += 1
            if action in {"add", "set"} and machine in workers:
                workers[machine]["state_epoch"] = data["group"]["worker_set_epoch"]
        data["meta"]["revision"] += 1
        data["meta"]["updated_at"] = utc_now()
        if worker_changed:
            # Keep borrow admission closed from the role mutation until the
            # corresponding primary candidate projection is published.
            with primary_projection_transaction(
                cfg, sorted(set(projection_routes + [("shared", machine), ("home", machine)]))
            ):
                write_group_record(cfg, path, data)
                sync_primary_ready_group(cfg, group_name)
        else:
            write_group_record(cfg, path, data)
        return data


def group_retry_failed(cfg: RootConfig, name: str) -> list[TaskRecord]:
    result = []
    for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(task_file))
        if task.group_name == name and task.state["projection"] == "failed":
            result.append(retry(cfg, task.task_id))
    return result
