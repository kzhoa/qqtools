"""Cleanup command workflows for qexp."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.locks import group_lock, schema_lock, task_lock
from ..runtime.claims import reconcile_claim_archives
from ..runtime.paths import group_path, local_paths, shared_paths, task_path
from ..runtime.records import AttemptRecord, SCHEMA_VERSION, TaskRecord, new_id, utc_now
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.tasks import load_task, save_task
from ..runtime.availability import remove_deadline_index
from ..runtime.ready import retire_current_ready_generation
from ..runtime.active_operations import (
    active_operation_path,
    archive_operation,
    iter_active_operation_paths,
    locate_operation_path,
    write_active_operation,
)


def _machine_project_id(cfg: RootConfig, reservation_runtime_root: Path) -> str | None:
    if reservation_runtime_root == cfg.runtime_root:
        return None
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    if not identity_path.exists():
        return None
    value = read_json(identity_path).get("project", {}).get("project_id")
    return value if isinstance(value, str) else None


def _reservation_matches_task(
    reservation: dict[str, Any], task_id: str, machine_project_id: str | None
) -> bool:
    return (reservation.get("task_id") == task_id
            and (machine_project_id is None or reservation.get("project_id") == machine_project_id))


def _clean_blockers(
    cfg: RootConfig, task: TaskRecord, *, reservation_runtime_root: Path | None = None
) -> list[str]:
    reservation_runtime_root = reservation_runtime_root or cfg.runtime_root
    machine_project_id = _machine_project_id(cfg, reservation_runtime_root)
    blockers: list[str] = []
    if task.state["projection"] not in {"succeeded", "failed", "cancelled"}:
        blockers.append(f"task_state:{task.state['projection']}")
    if task.claim_control.get("active_claim"):
        blockers.append("active_claim")
    for operation_path in iter_json(shared_paths(cfg.shared_root)["group_control"]):
        control = read_json(operation_path).get("group_control", {})
        if control.get("group_name") != task.group_name:
            continue
        if control.get("operation_type") == "cancel" and task.group_name:
            group_file = group_path(cfg.shared_root, task.group_name)
            barriers = (read_json(group_file)["group"].get("cancellation_barriers", [])
                        if group_file.exists() else [])
            if not any(item.get("operation_id") == control.get("operation_id")
                       for item in barriers):
                blockers.append(f"group_control_barrier_missing:{control.get('operation_id')}")
                continue
        if control.get("state") == "completed":
            continue
        high_watermark = control.get("membership_high_watermark")
        if high_watermark is not None and (task.group_membership_sequence or 0) <= high_watermark:
            blockers.append(f"group_control:{control.get('operation_id')}")
    for manifest_path in iter_json(cfg.runtime_root / "processes"):
        process = read_json(manifest_path).get("process", {})
        if process.get("task_id") != task.task_id:
            continue
        if process.get("observed_state") not in {"exited", "missing", "quarantined"}:
            blockers.append(f"local_process:{process.get('attempt_id')}")
    for state in ("active", "provisional"):
        for reservation_path in iter_json(local_paths(reservation_runtime_root)[state]):
            reservation = read_json(reservation_path).get("reservation", {})
            if _reservation_matches_task(reservation, task.task_id, machine_project_id):
                blockers.append(f"local_{state}_reservation:{reservation.get('reservation_id')}")
    return blockers


def _cleanup_required_machines(cfg: RootConfig, task: TaskRecord) -> list[str]:
    machines = {cfg.machine_name, task.placement_policy["home_machine"]}
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task.task_id
    for path in iter_json(attempts_dir):
        machines.add(AttemptRecord.from_dict(read_json(path)).machine_name)
    return sorted(machine for machine in machines if machine)


def _start_cleanup_operation(cfg: RootConfig, task: TaskRecord) -> dict[str, Any]:
    from ..events import write_event
    operation_path = locate_operation_path(cfg, "cleanup", task.task_id)
    if operation_path.exists():
        operation = read_json(operation_path)
        cleanup = operation.get("cleanup", {})
        if cleanup.get("state") != "completed":
            task.control["cleanup_operation_id"] = cleanup.get("operation_id")
            task.control["cleanup_state"] = cleanup.get("state")
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
        return operation
    now = utc_now()
    operation = {"meta": {"schema_version": SCHEMA_VERSION, "revision": 1, "created_at": now,
        "updated_at": now, "updated_by": {"actor_type": "cli",
        "machine_name": cfg.machine_name, "process_id": str(os.getpid())}},
        "cleanup": {"operation_id": new_id(), "task_id": task.task_id,
        "state": "preparing", "group_name": task.group_name,
        "submission_operation_id": task.submission_operation_id,
        "terminal_state": task.state["projection"], "created_at": now,
        "required_machines": _cleanup_required_machines(cfg, task), "acknowledgements": {},
        "completed_at": None}}
    operation_path = active_operation_path(cfg, "cleanup", task.task_id)
    write_active_operation(cfg, "cleanup", task.task_id, operation)
    task.control["cleanup_operation_id"] = operation["cleanup"]["operation_id"]
    task.control["cleanup_state"] = "preparing"
    task.meta["revision"] += 1
    task.meta["updated_at"] = now
    save_task(cfg, task)
    retire_current_ready_generation(cfg, task)
    write_event(cfg, "task_cleanup_started", task_id=task.task_id, details={
        "operation_id": operation["cleanup"]["operation_id"],
        "group_name": task.group_name,
        "submission_operation_id": task.submission_operation_id,
        "terminal_state": task.state["projection"],
    })
    return operation


def _cleanup_local_resources(
    cfg: RootConfig, task_id: str, *, reservation_runtime_root: Path | None = None
) -> tuple[list[str], list[str]]:
    from ..machine_runtime import resolve_execution_context
    from ..runtime.reservations import release

    reservation_runtime_root = (
        reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    )
    machine_project_id = _machine_project_id(cfg, reservation_runtime_root)
    removed: list[str] = []
    blockers: list[str] = []
    attempt_ids: set[str] = set()
    for path in iter_json(shared_paths(cfg.shared_root)["attempts"] / task_id):
        try:
            attempt_ids.add(AttemptRecord.from_dict(read_json(path)).attempt_id)
        except (KeyError, TypeError, ValueError):
            continue
    for manifest_path in iter_json(cfg.runtime_root / "processes"):
        process = read_json(manifest_path).get("process", {})
        if process.get("task_id") != task_id:
            continue
        attempt_id = process.get("attempt_id")
        if isinstance(attempt_id, str):
            attempt_ids.add(attempt_id)
        if process.get("observed_state") not in {"exited", "missing", "quarantined"}:
            blockers.append(f"local_process:{process.get('attempt_id')}")
            continue
        manifest_path.unlink(missing_ok=True)
        removed.append(str(manifest_path))
    if blockers:
        return removed, blockers
    local = local_paths(cfg.runtime_root)
    for name, record_key in (
        ("registrations", "process_registration"),
        ("launch_intents", "launch_intent"),
    ):
        for path in iter_json(local[name]):
            record = read_json(path).get(record_key, {})
            if record.get("task_id") != task_id:
                continue
            attempt_id = record.get("attempt_id")
            if isinstance(attempt_id, str):
                attempt_ids.add(attempt_id)
            path.unlink(missing_ok=True)
            removed.append(str(path))
    for path in iter_json(local["observations"]):
        attempt_id = read_json(path).get("exit_observation", {}).get("attempt_id")
        if attempt_id not in attempt_ids:
            continue
        path.unlink(missing_ok=True)
        removed.append(str(path))
    for path in sorted(local["termination_decisions"].rglob("*.json")):
        decision = read_json(path).get("termination_decision", {})
        if decision.get("task_id") != task_id and decision.get("attempt_id") not in attempt_ids:
            continue
        path.unlink(missing_ok=True)
        removed.append(str(path))
    for path in iter_json(local["authority_diagnostics"]):
        attempt_id = read_json(path).get("authority_diagnostic", {}).get("attempt_id")
        if attempt_id not in attempt_ids:
            continue
        path.unlink(missing_ok=True)
        removed.append(str(path))
    paths = local_paths(reservation_runtime_root)
    for state in ("active", "provisional"):
        for reservation_path in list(iter_json(paths[state])):
            reservation = read_json(reservation_path).get("reservation", {})
            if not _reservation_matches_task(reservation, task_id, machine_project_id):
                continue
            release(reservation_runtime_root, reservation["reservation_id"], "task_cleanup")
            removed.append(str(reservation_path))
    for log_path in sorted((cfg.runtime_root / "logs").glob(f"{task_id}-*.log")):
        log_path.unlink(missing_ok=True)
        removed.append(str(log_path))
    return removed, []


def _finalize_cleanup_operation(cfg: RootConfig, operation: dict[str, Any]) -> list[str]:
    from ..events import write_event
    cleanup = operation["cleanup"]
    task_id = cleanup["task_id"]
    if not reconcile_claim_archives(cfg, task_id):
        return []
    removed: list[str] = []
    path = task_path(cfg.shared_root, task_id)
    if path.exists():
        path.unlink()
        removed.append(str(path))
    deadline_index = shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task_id}.json"
    if deadline_index.exists():
        remove_deadline_index(cfg, task_id)
        removed.append(str(deadline_index))
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task_id
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)
        removed.append(str(attempts_dir))
    logs_dir = shared_paths(cfg.shared_root)["logs"] / task_id
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
        removed.append(str(logs_dir))
    write_event(cfg, "task_cleaned", task_id=task_id, details={
        "operation_id": cleanup["operation_id"], "group_name": cleanup.get("group_name"),
        "submission_operation_id": cleanup.get("submission_operation_id"),
        "terminal_state": cleanup.get("terminal_state"),
    })
    cleanup.update({"state": "completed", "completed_at": utc_now()})
    operation["meta"]["revision"] += 1
    operation["meta"]["updated_at"] = utc_now()
    archive_operation(cfg, "cleanup", task_id, operation)
    return removed


def _finalize_cleanup_if_ready(cfg: RootConfig, operation_path: Path) -> list[str]:
    operation = read_json(operation_path)
    cleanup = operation.get("cleanup", {})
    task_id = cleanup.get("task_id")
    group_name = cleanup.get("group_name")
    if not task_id:
        return []

    def finalize_under_task_lock() -> list[str]:
        operation = read_json(operation_path)
        cleanup = operation["cleanup"]
        if cleanup.get("state") not in {"preparing", "waiting_ack"}:
            return []
        pending = cleanup.get("pending_machines")
        if pending is None:
            required = set(cleanup.get("required_machines", []))
            pending = sorted(required - set(cleanup.get("acknowledgements", {})))
            cleanup["pending_machines"] = pending
            atomic_replace(operation_path, operation)
        if pending:
            return []
        return _finalize_cleanup_operation(cfg, operation)

    with schema_lock(cfg.shared_root, blocking=False) as has_schema_lock:
        if not has_schema_lock:
            return []
        if group_name:
            with group_lock(cfg.shared_root, group_name, blocking=False) as has_group_lock:
                if not has_group_lock:
                    return []
                with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
                    if not has_task_lock:
                        return []
                    return finalize_under_task_lock()
        with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
            if not has_task_lock:
                return []
            return finalize_under_task_lock()


def reconcile_cleanup_operations(
    cfg: RootConfig, *, reservation_runtime_root: Path | None = None,
    include_legacy: bool = True,
) -> list[dict[str, Any]]:
    """Clean machine-local resources and finalize fully acknowledged cleanup operations."""
    results: list[dict[str, Any]] = []
    for operation_path in iter_active_operation_paths(
        cfg, "cleanup", include_legacy=include_legacy
    ):
        operation = read_json(operation_path)
        cleanup = operation.get("cleanup", {})
        if cleanup.get("state") not in {"preparing", "waiting_ack"}:
            continue
        task_id = cleanup.get("task_id")
        if not task_id:
            continue
        result = {"operation_id": cleanup.get("operation_id"), "task_id": task_id,
                  "state": cleanup.get("state", "waiting_ack"),
                  "pending_machines": cleanup.get("pending_machines", []),
                  "removed": [], "blockers": []}
        with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
            if not has_task_lock:
                result["blockers"] = ["task_lock_busy"]
                results.append(result)
                continue
            operation = read_json(operation_path)
            cleanup = operation["cleanup"]
            if cleanup.get("state") not in {"preparing", "waiting_ack"}:
                continue
            required = set(cleanup.get("required_machines", []))
            acknowledgements = cleanup.setdefault("acknowledgements", {})
            removed: list[str] = []
            blockers: list[str] = []
            if cfg.machine_name in required and cfg.machine_name not in acknowledgements:
                removed, blockers = _cleanup_local_resources(
                    cfg, task_id, reservation_runtime_root=reservation_runtime_root
                )
                if not blockers:
                    acknowledgements[cfg.machine_name] = {"acknowledged_at": utc_now(),
                                                          "removed": removed}
            cleanup["state"] = "waiting_ack"
            cleanup["pending_machines"] = sorted(required - set(acknowledgements))
            task_file = task_path(cfg.shared_root, task_id)
            if task_file.exists():
                task = load_task(cfg, task_id)
                if task.control.get("cleanup_operation_id") == cleanup.get("operation_id"):
                    task.control["cleanup_state"] = cleanup["state"]
                    task.meta["revision"] += 1
                    task.meta["updated_at"] = utc_now()
                    save_task(cfg, task)
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            atomic_replace(operation_path, operation)
            result = {"operation_id": cleanup["operation_id"], "task_id": task_id,
                      "state": cleanup["state"],
                      "pending_machines": cleanup.get("pending_machines", []),
                      "removed": removed, "blockers": blockers}
        if not result["pending_machines"]:
            result["removed"].extend(_finalize_cleanup_if_ready(cfg, operation_path))
            finalized_path = locate_operation_path(cfg, "cleanup", task_id)
            finalized = read_json(finalized_path).get("cleanup", {})
            result["state"] = finalized.get("state", result["state"])
            result["pending_machines"] = finalized.get("pending_machines", [])
        results.append(result)
    return results


def clean(
    cfg: RootConfig,
    *,
    task_id: str | None = None,
    group: str | None = None,
    older_than_days: int = 30,
    limit: int = 100,
    dry_run: bool = False,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Remove terminal Task truth exactly or under a bounded retention policy."""
    if reservation_runtime_root is None:
        from ..machine_runtime import resolve_execution_context

        context = resolve_execution_context(cfg)
        cfg = context.local_cfg
        reservation_runtime_root = context.reservation_root
    if task_id and group:
        raise ValueError("task_id and group cannot be used together.")
    if older_than_days < 0:
        raise ValueError("older_than_days must be non-negative.")
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if task_id:
        candidates = [load_task(cfg, task_id)]
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        candidates = []
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if group and task.group_name != group:
                continue
            updated_at = datetime.fromisoformat(task.meta["updated_at"].replace("Z", "+00:00"))
            if (task.state["projection"] in {"succeeded", "failed", "cancelled"}
                    and updated_at <= cutoff):
                candidates.append(task)
        candidates.sort(key=lambda item: (item.meta["updated_at"], item.task_id))
        candidates = candidates[:limit]
    result: dict[str, Any] = {"dry_run": dry_run,
                              "candidates": [task.task_id for task in candidates],
                              "removed": [], "skipped": {}}
    from ..scheduler import authority_locks
    for candidate in candidates:
        with schema_lock(cfg.shared_root):
            with authority_locks(cfg, candidate):
                task = load_task(cfg, candidate.task_id)
                blockers = _clean_blockers(
                    cfg, task, reservation_runtime_root=reservation_runtime_root
                )
                unsafe_blockers = [item for item in blockers if not item.startswith(
                    ("local_active_reservation:", "local_provisional_reservation:"))]
                if unsafe_blockers:
                    result["skipped"][task.task_id] = unsafe_blockers
                    continue
                if not dry_run:
                    operation = _start_cleanup_operation(cfg, task)
                    result.setdefault("operations", {})[task.task_id] = operation["cleanup"]
    if not dry_run:
        for reconciliation in reconcile_cleanup_operations(
                cfg, reservation_runtime_root=reservation_runtime_root
        ):
            if reconciliation["task_id"] in result["candidates"]:
                result["removed"].extend(reconciliation["removed"])
                result.setdefault("operations", {})[reconciliation["task_id"]] = reconciliation
    if task_id and result["skipped"] and not dry_run:
        blockers = result["skipped"][task_id]
        raise ValueError(f"Task {task_id!r} cannot be cleaned: {', '.join(blockers)}")
    return result
