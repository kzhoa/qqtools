"""Short-lived filesystem locks with the schema-defined lock order."""
from __future__ import annotations

import errno
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .paths import lock_path


@contextmanager
def exclusive(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    has_lock = False
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                yield False
                return
            raise
        has_lock = True
        yield True
    finally:
        if has_lock:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def schema_lock(root: Path, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "schema"), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def group_lock(root: Path, name: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "groups", name), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def task_lock(root: Path, task_id: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "tasks", task_id), blocking=blocking) as acquired:
        yield acquired
