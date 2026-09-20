"""Typed Task record persistence helpers."""

from __future__ import annotations

import os

from .paths import task_path
from .records import TaskRecord
from .store import atomic_replace, read_json
from .work_budget import diagnostic_increment, diagnostic_span


def load_task(cfg: object, task_id: str) -> TaskRecord:
    with diagnostic_span("task_json_read"):
        diagnostic_increment("task_json_read.records")
        value = read_json(task_path(cfg.shared_root, task_id))
        from ..layout import is_task_dependencies_root

        if is_task_dependencies_root(cfg) and "depends_on_task_ids" not in value.get("task", {}):
            raise ValueError("canonical Task is missing depends_on_task_ids.")
        return TaskRecord.from_dict(value)


def save_task(cfg: object, task: TaskRecord) -> None:
    from .observation.projection import publish_task
    from .ready import assert_ready_writer_compatible

    assert_ready_writer_compatible(cfg)
    path = task_path(cfg.shared_root, task.task_id)
    publish_task(cfg, task, lambda: atomic_replace(path, task.to_dict()))


def _unlink_task_truth(cfg: object, task_id: str) -> None:
    path = task_path(cfg.shared_root, task_id)
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def delete_task(cfg: object, task_id: str) -> None:
    """Durably delete Task truth and retire its observation memberships."""
    from .observation.projection import delete_task_truth
    from .ready import assert_ready_writer_compatible

    assert_ready_writer_compatible(cfg)
    delete_task_truth(cfg, task_id, lambda: _unlink_task_truth(cfg, task_id))
