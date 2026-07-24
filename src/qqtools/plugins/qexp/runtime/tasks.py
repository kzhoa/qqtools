"""Typed Task record persistence helpers."""
from __future__ import annotations

from .paths import task_path
from .records import TaskRecord
from .store import atomic_replace, read_json


def load_task(cfg: object, task_id: str) -> TaskRecord:
    return TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))


def save_task(cfg: object, task: TaskRecord) -> None:
    atomic_replace(task_path(cfg.shared_root, task.task_id), task.to_dict())
