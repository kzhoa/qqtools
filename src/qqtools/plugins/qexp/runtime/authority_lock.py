"""Shared authority lock order without scheduler dependencies."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..config_types import RootConfig
from .dependencies import dependency_locks
from .locks import schema_writer_lock
from .records import TaskRecord


@contextmanager
def authority_locks(cfg: RootConfig, task: TaskRecord) -> Iterator[None]:
    """Acquire the only permitted shared authority order."""
    with schema_writer_lock(cfg, require_narrow=True):
        with dependency_locks(cfg, task):
            yield
