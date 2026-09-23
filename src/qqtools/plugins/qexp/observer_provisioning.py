"""Bounded, non-blocking observer provisioning jobs.

Observer creation is advisory work.  This module keeps it outside the machine
agent's launch, lease, and heartbeat path with one daemon worker and a bounded
queue.  A job is keyed by its project/Task/Attempt identity so an uncertain
window creation cannot be submitted twice for one Attempt.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread
from typing import Iterator

from .runtime.locks import exclusive

_MAX_PENDING_OBSERVER_JOBS = 16
_OBSERVER_QUEUE: Queue[tuple[tuple[str, ...], Callable[[], None]]] = Queue(maxsize=_MAX_PENDING_OBSERVER_JOBS)
_OBSERVER_LOCK = Lock()
_OBSERVER_KEYS: set[tuple[str, ...]] = set()
_OBSERVER_WORKER_STARTED = False


def observer_lock_path(shared_root: Path, task_id: str, attempt_id: str) -> Path:
    """Return a per-user local lock path; attachment needs no shared-root write."""
    identity = f"{Path(shared_root).resolve()}\x00{task_id}\x00{attempt_id}".encode("utf-8", "surrogatepass")
    digest = hashlib.sha256(identity).hexdigest()
    return Path(tempfile.gettempdir()) / f"qqtools-observer-locks-{os.getuid()}" / f"{digest}.lock"


@contextmanager
def observer_lock(shared_root: Path, task_id: str, attempt_id: str) -> Iterator[bool]:
    """Serialize viewer find/create/tag operations for one Task Attempt."""
    path = observer_lock_path(shared_root, task_id, attempt_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"unsafe observer lock directory: {path.parent}")
    with exclusive(path) as acquired:
        yield acquired


def _reset_after_fork() -> None:
    """Discard inherited jobs and worker state in a forked child."""
    global _OBSERVER_KEYS, _OBSERVER_LOCK, _OBSERVER_QUEUE, _OBSERVER_WORKER_STARTED
    _OBSERVER_QUEUE = Queue(maxsize=_MAX_PENDING_OBSERVER_JOBS)
    _OBSERVER_LOCK = Lock()
    _OBSERVER_KEYS = set()
    _OBSERVER_WORKER_STARTED = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _observer_worker() -> None:
    """Run queued jobs without keeping interpreter shutdown waiting on them."""
    while True:
        key, callback = _OBSERVER_QUEUE.get()
        try:
            callback()
        except Exception:
            # Callers own diagnostics because they know the project/Attempt
            # log.  Keep this worker alive for independent observer jobs.
            pass
        finally:
            with _OBSERVER_LOCK:
                _OBSERVER_KEYS.discard(key)
            _OBSERVER_QUEUE.task_done()


def _start_observer_worker() -> None:
    """Start the one process-wide worker exactly once."""
    global _OBSERVER_WORKER_STARTED
    if _OBSERVER_WORKER_STARTED:
        return
    _OBSERVER_WORKER_STARTED = True
    Thread(target=_observer_worker, name="qexp-observer-provisioner", daemon=True).start()


def submit_observer_job(key: tuple[str, ...], callback: Callable[[], None]) -> bool:
    """Queue one observer job without waiting for its callback.

    ``False`` means the Attempt already has a queued job or the bounded queue
    is full.  The caller should treat both outcomes as best-effort observer
    unavailability and continue the training launch.
    """
    if not key or not all(isinstance(part, str) and part for part in key):
        raise ValueError("observer job key must contain non-empty string parts")
    if not callable(callback):
        raise TypeError("observer job callback must be callable")
    with _OBSERVER_LOCK:
        if key in _OBSERVER_KEYS:
            return False
        if _OBSERVER_QUEUE.full():
            return False
        _OBSERVER_KEYS.add(key)
        try:
            _start_observer_worker()
            _OBSERVER_QUEUE.put_nowait((key, callback))
        except Full:
            _OBSERVER_KEYS.discard(key)
            return False
        except Exception:
            _OBSERVER_KEYS.discard(key)
            raise
    return True


__all__ = ["observer_lock", "observer_lock_path", "submit_observer_job"]
