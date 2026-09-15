"""Best-effort single-writer progress reporting for arbitrary applications.

Applications normally use only :func:`update` and :func:`flush`. The process
owns one bounded latest-value slot and one daemon file writer. Outside qexp this
module is a safe no-op. Progress never participates in task authority.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ._progress_protocol import MAX_PAYLOAD_BYTES, replace_advisory_snapshot, validate_payload

__all__ = ["update", "flush"]


def _is_primary() -> bool:
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return value == "0"
    return True


class _Reporter:
    """One process-local latest-value writer.

    Filesystem I/O happens only on the daemon writer. ``update`` uses a
    non-blocking state lock. The first accepted update may pay normal Python
    daemon-thread startup cost, but never waits for mailbox filesystem I/O.
    """

    def __init__(self, path: str | Path | None = None, *, interval_seconds: float = 1.0) -> None:
        self._path = Path(path) if path else None
        self._interval = max(0.1, float(interval_seconds))
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._closing = False
        self._last_attempt = float("-inf")
        self._last_stage: str | None = None

    def update(
        self,
        *,
        stage: str,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        message: str | None = None,
    ) -> bool:
        if self._path is None or os.getpid() != self._pid or not _is_primary():
            return False
        try:
            payload = validate_payload(
                {
                    "protocol_version": 1,
                    "update_id": uuid.uuid4().hex,
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "unit": unit,
                    "message": message,
                }
            )
            if not self._lock.acquire(blocking=False):
                return False
            try:
                if self._closing:
                    return False
                wake = self._pending is None or self._pending["stage"] != stage
                self._pending = payload
                if self._thread is None:
                    # Serialize startup with close. The state lock is never held
                    # across file I/O, and close waits on it only within timeout.
                    self._thread = threading.Thread(
                        target=self._run,
                        name="qexp-progress-writer",
                        daemon=True,
                    )
                    self._thread.start()
                if wake:
                    self._wake.set()
            finally:
                self._lock.release()
            return True
        except Exception:
            return False

    def _run(self) -> None:
        delay: float | None = None
        while True:
            self._wake.wait(delay)
            self._wake.clear()
            with self._lock:
                payload = self._pending
                if payload is None:
                    if self._closing:
                        return
                    delay = None
                    continue
                interval = 0.1 if payload["stage"] != self._last_stage else self._interval
                remaining = self._last_attempt + interval - time.monotonic()
                if remaining > 0 and not self._closing:
                    delay = remaining
                    continue
                self._pending = None
            try:
                assert self._path is not None
                replace_advisory_snapshot(self._path, payload, max_bytes=MAX_PAYLOAD_BYTES)
            except Exception:
                pass
            self._last_attempt = time.monotonic()
            self._last_stage = payload["stage"]
            with self._lock:
                if self._closing and self._pending is None:
                    return
                if self._pending is not None:
                    self._wake.set()
            delay = None

    def close(self, *, timeout: float = 0.1) -> None:
        """Best-effort close with a real upper bound on lock/join waiting."""
        if os.getpid() != self._pid:
            return
        try:
            budget = max(0.0, min(float(timeout), 1.0))
        except (TypeError, ValueError):
            budget = 0.1
        deadline = time.monotonic() + budget
        remaining = max(0.0, deadline - time.monotonic())
        try:
            if not self._lock.acquire(timeout=remaining):
                return
        except Exception:
            return
        try:
            self._closing = True
            self._wake.set()
            thread = self._thread
        finally:
            self._lock.release()
        if thread is None:
            return
        remaining = max(0.0, deadline - time.monotonic())
        try:
            thread.join(timeout=remaining)
        except RuntimeError:
            pass

    def replaceable(self) -> bool:
        """Whether a new singleton can be created without overlapping writers."""
        thread = self._thread
        if thread is None:
            return self._closing
        return not thread.is_alive()


_reporter: _Reporter | None = None
_reporter_lock = threading.Lock()


def _get_reporter() -> _Reporter | None:
    global _reporter
    if not os.environ.get("QEXP_PROGRESS_PATH") or not _is_primary():
        return None
    current = _reporter
    if current is not None and not current.replaceable():
        return current
    if not _reporter_lock.acquire(blocking=False):
        return None
    try:
        current = _reporter
        if current is not None and current.replaceable():
            _reporter = None
        if _reporter is None:
            _reporter = _Reporter(os.environ["QEXP_PROGRESS_PATH"])
        return _reporter
    finally:
        _reporter_lock.release()


def update(
    *,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    message: str | None = None,
) -> bool:
    """Offer progress to the current qexp Attempt, or safely do nothing."""
    try:
        reporter = _get_reporter()
        if reporter is None:
            return False
        return reporter.update(stage=stage, current=current, total=total, unit=unit, message=message)
    except Exception:
        return False


def flush(*, timeout: float = 0.1) -> None:
    """Close the process writer; reset only after its daemon has actually exited."""
    global _reporter
    try:
        budget = max(0.0, min(float(timeout), 1.0))
    except (TypeError, ValueError):
        budget = 0.1
    deadline = time.monotonic() + budget
    remaining = max(0.0, deadline - time.monotonic())
    try:
        if not _reporter_lock.acquire(timeout=remaining):
            return
    except Exception:
        return
    try:
        reporter = _reporter
        if reporter is not None:
            reporter.close(timeout=max(0.0, deadline - time.monotonic()))
            # A writer blocked in the filesystem remains the singleton sentinel.
            # Updates are rejected while it is closing. The next update replaces
            # it only after its daemon has really exited.
            if reporter.replaceable():
                _reporter = None
    finally:
        _reporter_lock.release()


atexit.register(flush)
