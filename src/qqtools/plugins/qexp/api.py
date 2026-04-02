from __future__ import annotations

import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from . import fsqueue
from . import manager
from .models import TASK_PENDING, qExpTask


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
