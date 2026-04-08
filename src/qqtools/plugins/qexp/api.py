from __future__ import annotations

import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from . import fsqueue
from . import manager
from .models import TASK_CANCELLED, TASK_DONE, TASK_FAILED, TASK_PENDING, qExpTask
from .observer import build_status_snapshot

DEFAULT_CLEAN_OLDER_THAN_SECONDS = 7 * 24 * 60 * 60


def _generate_task_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def submit(
    argv: list[str],
    num_gpus: int,
    job_id: str | None = None,
    job_name: str | None = None,
    workdir: str | None = None,
    env: dict[str, Any] | None = None,
    root: Path | None = None,
) -> qExpTask:
    manager.run_preflight_checks()
    state_root = fsqueue.ensure_qexp_layout(root)
    task_id = job_id or _generate_task_id()

    existing_task = fsqueue.load_task_by_id(task_id, state_root)
    if existing_task is not None:
        if job_id is None:
            raise RuntimeError(
                f"Auto-generated task_id collision detected for '{task_id}', which should be impossible."
            )
        return existing_task

    task = qExpTask(
        task_id=task_id,
        name=job_name,
        workdir=workdir,
        argv=argv,
        num_gpus=num_gpus,
        env=env,
        status=TASK_PENDING,
    )
    fsqueue.save_task(task, state_root)

    if not manager.is_daemon_active(state_root):
        manager.start_daemon_background(state_root)
        time_wait_until = manager.DEFAULT_STARTUP_WAIT_SECONDS
        if time_wait_until > 0:
            import time

            time.sleep(time_wait_until)
        if not manager.is_daemon_active(state_root):
            raise RuntimeError(
                "qexp queued the task but daemon startup failed. Run `qexp daemon` "
                "for debugging."
            )

    return task


def _is_process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except OSError:
        return False
    return True


def cancel(
    task_id: str,
    *,
    root: Path | None = None,
    grace_seconds: float = 3.0,
    poll_interval_seconds: float = 0.1,
) -> qExpTask:
    state_root = fsqueue.ensure_qexp_layout(root)
    task = fsqueue.load_task_by_id(task_id, state_root)
    if task is None:
        raise FileNotFoundError(f"qexp task '{task_id}' does not exist.")

    if task.status == "pending":
        cancelled_path = fsqueue.cancel_pending_task(task_id, root=state_root)
        return fsqueue.load_task(cancelled_path)

    if task.status != "running":
        return task

    if task.process_group_id is None:
        raise RuntimeError(
            f"Running task '{task_id}' has no process_group_id yet; retry cancellation shortly."
        )

    try:
        os.killpg(task.process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return fsqueue.load_task_by_id(task_id, state_root) or task

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _is_process_group_alive(task.process_group_id):
            return fsqueue.load_task_by_id(task_id, state_root) or task
        time.sleep(poll_interval_seconds)

    try:
        os.killpg(task.process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass

    return fsqueue.load_task_by_id(task_id, state_root) or task


def get_logs_path(task_id: str, *, root: Path | None = None) -> Path:
    return fsqueue.get_log_path(task_id, root=root)


def read_logs(task_id: str, *, root: Path | None = None) -> str:
    log_path = get_logs_path(task_id, root=root)
    if not log_path.exists():
        raise FileNotFoundError(f"qexp log for task '{task_id}' does not exist.")
    return log_path.read_text(encoding="utf-8")


def get_status_snapshot(*, root: Path | None = None) -> dict[str, Any]:
    return build_status_snapshot(root=root)


def _is_task_old_enough(path: Path, older_than_seconds: int) -> bool:
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds >= older_than_seconds


def clean(
    *,
    root: Path | None = None,
    dry_run: bool = False,
    include_failed: bool = False,
    older_than_seconds: int = DEFAULT_CLEAN_OLDER_THAN_SECONDS,
) -> dict[str, Any]:
    state_root = fsqueue.ensure_qexp_layout(root)
    if older_than_seconds < 0:
        raise ValueError("older_than_seconds must be >= 0.")
    target_states = [TASK_DONE, TASK_CANCELLED]
    if include_failed:
        target_states.append(TASK_FAILED)

    deleted_task_ids: list[str] = []
    deleted_task_files: list[str] = []
    deleted_log_files: list[str] = []

    for state in target_states:
        for path in fsqueue.get_state_task_paths(state, state_root):
            if not _is_task_old_enough(path, older_than_seconds):
                continue
            task_id = path.stem
            log_path = fsqueue.get_log_path(task_id, state_root)
            deleted_task_ids.append(task_id)
            deleted_task_files.append(str(path))
            if log_path.exists():
                deleted_log_files.append(str(log_path))

            if dry_run:
                continue

            path.unlink(missing_ok=True)
            if log_path.exists():
                log_path.unlink()

    return {
        "dry_run": dry_run,
        "include_failed": include_failed,
        "older_than_seconds": older_than_seconds,
        "task_ids": deleted_task_ids,
        "deleted_task_files": deleted_task_files,
        "deleted_log_files": deleted_log_files,
    }
