from __future__ import annotations

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
                "qexp queued the task but daemon startup failed. Run `qexp daemon --foreground` "
                "for debugging."
            )

    return task
