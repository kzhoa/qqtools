"""Best-effort, single-writer progress reporting for arbitrary applications.

Example::

    from qqtools.qexp import progress
    progress.update(stage="download", current=3, total=10, unit="file")

Updates are full replacements. Outside qexp this is a no-op. A single daemon
worker writes a latest-only mailbox; application threads never perform disk I/O.
Only the main process (global rank zero in distributed applications) should use
this API. Progress reports do not prove that an application is healthy.
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

__all__ = ["Reporter", "update", "flush"]


def _is_primary() -> bool:
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return value == "0"
    return True


class Reporter:
    """One bounded mailbox writer. ``close`` waits at most its timeout.

    The optional path is useful for adapters and tests; normal applications use
    ``update`` and the attempt-local path supplied by qexp. This is not an API
    for selecting another task. A reporter inherited through fork is disabled.
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
        self, *, stage: str, current: int | None = None, total: int | None = None,
        unit: str | None = None, message: str | None = None,
    ) -> bool:
        """Offer the latest state; return False if disabled, invalid or contended."""
        if self._path is None or os.getpid() != self._pid or not _is_primary():
            return False
        try:
            payload = validate_payload({
                "protocol_version": 1, "update_id": uuid.uuid4().hex,
                "stage": stage, "current": current, "total": total,
                "unit": unit, "message": message,
            })
            if not self._lock.acquire(blocking=False):
                return False
            try:
                if self._closing:
                    return False
                wake = self._pending is None or self._pending["stage"] != stage
                self._pending = payload
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="qexp-progress-writer", daemon=True)
                    self._thread.start()
                if wake:
                    self._wake.set()
            finally:
                self._lock.release()
            return True
        except Exception:
            # An optional observer may never turn a successful application into
            # a failure, including during interpreter shutdown or invalid input.
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
        """Best-effort final flush; never join a stuck writer indefinitely."""
        if os.getpid() != self._pid:
            return
        try:
            with self._lock:
                self._closing = True
                self._wake.set()
                thread = self._thread
            if thread is not None:
                thread.join(timeout=max(0.0, min(float(timeout), 1.0)))
        except Exception:
            pass


_reporter: Reporter | None = None
_reporter_lock = threading.Lock()


def update(
    *, stage: str, current: int | None = None, total: int | None = None,
    unit: str | None = None, message: str | None = None,
) -> bool:
    """Report progress for the current qexp attempt, or safely do nothing."""
    global _reporter
    if not os.environ.get("QEXP_PROGRESS_PATH") or not _is_primary():
        return False
    try:
        if _reporter is None:
            if not _reporter_lock.acquire(blocking=False):
                return False
            try:
                if _reporter is None:
                    _reporter = Reporter(os.environ["QEXP_PROGRESS_PATH"])
            finally:
                _reporter_lock.release()
        return _reporter.update(stage=stage, current=current, total=total, unit=unit, message=message)
    except Exception:
        return False


def flush(*, timeout: float = 0.1) -> None:
    """Flush and close the process reporter at the end of an application run."""
    if _reporter is not None:
        _reporter.close(timeout=timeout)


atexit.register(flush)
