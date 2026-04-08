from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from .layout import (
    RootConfig,
    batch_lock_path,
    migrate_lock_path,
    submit_lock_path,
)


class LockTimeout(Exception):
    """Raised when a lock cannot be acquired within the timeout."""


class FileLock:
    """Advisory file lock using fcntl.flock.

    Intended for protecting critical shared-root operations (submit, batch, migrate).
    """

    def __init__(self, lock_path: Path, timeout: float = 30.0) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LockTimeout(
                        f"Could not acquire lock {self.lock_path} "
                        f"within {self.timeout}s."
                    )
                time.sleep(0.05)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def is_stale(self, max_age_seconds: float = 300.0) -> bool:
        if not self.lock_path.is_file():
            return False
        try:
            mtime = self.lock_path.stat().st_mtime
            return (time.time() - mtime) > max_age_seconds
        except OSError:
            return False


def submit_lock(cfg: RootConfig, timeout: float = 30.0) -> FileLock:
    return FileLock(submit_lock_path(cfg), timeout=timeout)


def batch_lock(cfg: RootConfig, timeout: float = 30.0) -> FileLock:
    return FileLock(batch_lock_path(cfg), timeout=timeout)


def migrate_lock(cfg: RootConfig, timeout: float = 30.0) -> FileLock:
    return FileLock(migrate_lock_path(cfg), timeout=timeout)
