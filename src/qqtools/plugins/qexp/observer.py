from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import fsqueue, manager
from .models import TASK_CANCELLED, TASK_DONE, TASK_FAILED, TASK_PENDING, TASK_RUNNING, qExpTask
from .tracker import probe_gpu_backend


PUBLIC_TASK_FIELDS = (
    "task_id",
    "name",
    "argv",
    "num_gpus",
    "workdir",
    "version",
    "created_at",
    "status",
    "scheduled_at",
    "assigned_gpus",
    "tmux_session",
    "tmux_window_id",
    "wrapper_pid",
    "process_group_id",
    "started_at",
    "finished_at",
    "exit_code",
    "exit_reason",
)


@dataclass(slots=True)
class qExpObservedTaskFile:
    path: Path
    state_dir: str
    mtime: float
    task: qExpTask | None
    error: str | None = None


def _format_state_label(state: str) -> str:
    return {
        TASK_PENDING: "Pending",
        TASK_RUNNING: "Running",
        TASK_DONE: "Done",
        TASK_FAILED: "Failed",
        TASK_CANCELLED: "Cancelled",
    }[state]


def _safe_stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _observe_task_file(path: Path) -> qExpObservedTaskFile:
    try:
        task = fsqueue.load_task(path)
        return qExpObservedTaskFile(
            path=path,
            state_dir=path.parent.name,
            mtime=_safe_stat_mtime(path),
            task=task,
        )
    except Exception as exc:
        return qExpObservedTaskFile(
            path=path,
            state_dir=path.parent.name,
            mtime=_safe_stat_mtime(path),
            task=None,
            error=str(exc),
        )


def _scan_task_files(root: Path | None = None) -> list[qExpObservedTaskFile]:
    records: list[qExpObservedTaskFile] = []
    for state in (TASK_PENDING, TASK_RUNNING, TASK_DONE, TASK_FAILED, TASK_CANCELLED):
        for path in fsqueue.get_state_task_paths(state, root):
            records.append(_observe_task_file(path))
    return records


def _serialize_public_task(task: qExpTask) -> dict[str, Any]:
    payload = task.to_dict()
    return {field: payload.get(field) for field in PUBLIC_TASK_FIELDS}


def _build_warning_message(record: qExpObservedTaskFile) -> str:
    return f"Skipped unreadable task file '{record.path}': {record.error}"


def _build_task_row(task: qExpTask) -> dict[str, Any]:
    assigned = "-" if not task.assigned_gpus else ",".join(str(gpu_id) for gpu_id in task.assigned_gpus)
    return {
        "state": _format_state_label(task.status),
        "state_key": task.status,
        "task_id": task.task_id,
        "name": task.name or "-",
        "gpus": task.num_gpus,
        "assigned": assigned,
        "created_at": task.created_at or "-",
        "exit_reason": task.exit_reason or "-",
        "task": _serialize_public_task(task),
    }


def _build_gpu_snapshot(running_tasks: list[qExpTask]) -> dict[str, Any]:
    backend_name, visible_gpu_ids = probe_gpu_backend()
    task_id_to_gpu_ids: dict[str, list[int]] = {}
    reserved_gpu_ids: set[int] = set()

    for task in running_tasks:
        if not task.assigned_gpus:
            continue
        gpu_ids = [int(gpu_id) for gpu_id in task.assigned_gpus]
        task_id_to_gpu_ids[task.task_id] = gpu_ids
        reserved_gpu_ids.update(gpu_ids)

    slots: list[dict[str, Any]] = []
    for gpu_id in visible_gpu_ids:
        owner_task_id = "-"
        for task_id, gpu_ids in task_id_to_gpu_ids.items():
            if gpu_id in gpu_ids:
                owner_task_id = task_id
                break
        slots.append(
            {
                "gpu_id": gpu_id,
                "state": "Reserved" if gpu_id in reserved_gpu_ids else "Free",
                "task_id": owner_task_id,
            }
        )

    return {
        "backend": backend_name,
        "visible_gpu_ids": list(visible_gpu_ids),
        "slots": slots,
    }


def _build_recent_events(observed_files: list[qExpObservedTaskFile]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in observed_files:
        if record.task is None or record.task.status not in {
            TASK_RUNNING,
            TASK_DONE,
            TASK_FAILED,
            TASK_CANCELLED,
        }:
            continue
        task = record.task
        events.append(
            {
                "timestamp": task.finished_at or task.started_at or task.scheduled_at or task.created_at or "-",
                "task_id": task.task_id,
                "state": _format_state_label(task.status),
                "state_key": task.status,
                "exit_reason": task.exit_reason or "-",
                "mtime": record.mtime,
            }
        )
    events.sort(key=lambda item: item["mtime"], reverse=True)
    return [
        {
            "timestamp": event["timestamp"],
            "task_id": event["task_id"],
            "state": event["state"],
            "state_key": event["state_key"],
            "exit_reason": event["exit_reason"],
        }
        for event in events[:3]
    ]


def _build_daemon_status(root: Path | None = None) -> dict[str, Any]:
    metadata = manager._read_pid_metadata(root)
    pid = metadata.get("pid") if isinstance(metadata, dict) else None
    is_lock_active = manager.has_active_daemon_lock_readonly(root)
    has_fresh_heartbeat = not manager.is_heartbeat_stale(root)
    is_process_alive = isinstance(pid, int) and manager._is_process_alive(pid)

    if is_lock_active and is_process_alive and has_fresh_heartbeat:
        state = "HEALTHY"
    elif is_lock_active and is_process_alive:
        state = "STALE (Unresponsive)"
    else:
        state = "STOPPED"

    return {
        "state": state,
        "is_active": is_lock_active and is_process_alive and has_fresh_heartbeat,
        "is_stale": is_lock_active and is_process_alive and not has_fresh_heartbeat,
        "pid": pid if isinstance(pid, int) else None,
        "started_at": metadata.get("started_at") if isinstance(metadata, dict) else None,
        "gpu_backend": metadata.get("gpu_backend") if isinstance(metadata, dict) else None,
        "visible_gpu_ids": metadata.get("visible_gpu_ids") if isinstance(metadata, dict) else None,
    }


def build_status_snapshot(*, root: Path | None = None) -> dict[str, Any]:
    observed_files = _scan_task_files(root)
    warnings = sorted(
        {
            _build_warning_message(record)
            for record in observed_files
            if record.task is None and record.error is not None
        }
    )

    tasks = [record.task for record in observed_files if record.task is not None]
    task_rows = [_build_task_row(task) for task in tasks]
    running_tasks = [task for task in tasks if task.status == TASK_RUNNING]

    counts = {
        "pending": sum(1 for task in tasks if task.status == TASK_PENDING),
        "running": sum(1 for task in tasks if task.status == TASK_RUNNING),
        "done": sum(1 for task in tasks if task.status == TASK_DONE),
        "failed": sum(1 for task in tasks if task.status == TASK_FAILED),
        "cancelled": sum(1 for task in tasks if task.status == TASK_CANCELLED),
    }

    return {
        "daemon": _build_daemon_status(root),
        "counts": counts,
        "tasks": task_rows,
        "pending_preview": [row for row in task_rows if row["state_key"] == TASK_PENDING][:5],
        "gpus": _build_gpu_snapshot(running_tasks),
        "events": _build_recent_events(observed_files),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
        },
        "warnings": warnings,
        "summary": {
            "invalid_task_files": len(warnings),
        },
    }
