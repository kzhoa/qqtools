"""Best-effort single-writer progress reporting for arbitrary applications.

Applications normally use only :func:`update` and :func:`flush`. The process
owns one bounded latest-value slot and one daemon file writer. Outside qexp this
module is a safe no-op. Progress never participates in task authority.
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ._progress_protocol import MAX_PAYLOAD_BYTES, replace_advisory_snapshot, validate_payload

__all__ = ["update", "flush"]

DEFAULT_PROGRESS_INTERVAL_SECONDS = 30


def _safe_interval(value: Any) -> float:
    if type(value) not in (int, float) or value < 1:
        return float(DEFAULT_PROGRESS_INTERVAL_SECONDS)
    try:
        interval = float(value)
    except OverflowError:
        interval = float("inf")
    if isinstance(value, float) and not math.isfinite(value):
        return float(DEFAULT_PROGRESS_INTERVAL_SECONDS)
    return interval


def _environment_interval() -> float:
    raw = os.environ.get("QEXP_PROGRESS_INTERVAL_SECONDS")
    if raw is None:
        return float(DEFAULT_PROGRESS_INTERVAL_SECONDS)
    text = raw.strip()
    try:
        value: int | float = int(text, 10)
    except (TypeError, ValueError):
        try:
            value = float(text)
        except (TypeError, ValueError):
            return float(DEFAULT_PROGRESS_INTERVAL_SECONDS)
    return _safe_interval(value)


def _payload_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(payload.get(name) for name in ("stage", "current", "total", "unit", "message"))


def _path_offset(path: Path, interval: float) -> float:
    digest = hashlib.sha256(os.fsencode(path)).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 * interval


def _next_deadline(deadline: float, now: float, interval: float) -> float:
    if not math.isfinite(deadline):
        return float("inf")
    target = now + interval
    if not math.isfinite(target):
        return float("inf")
    ratio = (target - deadline) / interval
    rounded = round(ratio)
    steps = max(
        1,
        rounded if math.isclose(ratio, rounded, rel_tol=1e-12, abs_tol=1e-12) else math.ceil(ratio),
    )
    candidate = deadline + steps * interval
    if not math.isfinite(candidate):
        return float("inf")
    return max(candidate, target)


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

    def __init__(
        self, path: str | Path | None = None, *, interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS
    ) -> None:
        self._path = Path(path) if path else None
        self._interval = _safe_interval(interval_seconds)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._close_requested = threading.Event()
        self._pending: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._last_success = float("-inf")
        self._next_due = float("-inf")
        self._last_written_key: tuple[Any, ...] | None = None
        self._initial_exception_available = True
        self._final_exception_available = True
        self._closing_failures = 0

    @property
    def path(self) -> Path | None:
        return self._path

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
                if self._close_requested.is_set():
                    return False
                key = _payload_key(payload)
                if key == self._last_written_key:
                    return True
                wake = self._pending is None or _payload_key(self._pending) != key
                # An update identifier is not a reason to rewrite an otherwise
                # identical semantic and message payload.
                if self._pending is None or _payload_key(self._pending) != key:
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
            safe_delay = None if delay is None else min(delay, threading.TIMEOUT_MAX)
            self._wake.wait(safe_delay)
            self._wake.clear()
            with self._lock:
                payload = self._pending
                if payload is None:
                    if self._close_requested.is_set():
                        return
                    delay = None
                    continue
                closing = self._close_requested.is_set()
                now = time.monotonic()
                if closing:
                    if not self._final_exception_available and _payload_key(payload) == self._last_written_key:
                        self._pending = None
                        return
                    due = self._final_exception_available
                else:
                    due = self._initial_exception_available or now >= self._next_due
                remaining = self._next_due - now
                if not due:
                    delay = max(0.0, remaining)
                    continue
                self._pending = None
            try:
                assert self._path is not None
                replace_advisory_snapshot(self._path, payload, max_bytes=MAX_PAYLOAD_BYTES)
            except Exception:
                with self._lock:
                    if closing:
                        self._closing_failures += 1
                        if self._closing_failures >= 3:
                            self._pending = None
                            return
                    # Keep the latest value for a later retry.  A failed
                    # replacement does not consume an initial/final exception,
                    # but final shutdown retries remain bounded.
                    if self._pending is None:
                        self._pending = payload
                delay = 0.05
                continue
            with self._lock:
                succeeded_at = time.monotonic()
                self._last_success = succeeded_at
                self._last_written_key = _payload_key(payload)
                if self._pending is not None and _payload_key(self._pending) == self._last_written_key:
                    self._pending = None
                if self._initial_exception_available:
                    assert self._path is not None
                    self._next_due = succeeded_at + self._interval + _path_offset(self._path, self._interval)
                elif not closing:
                    self._next_due = _next_deadline(self._next_due, succeeded_at, self._interval)
                self._initial_exception_available = False
                if closing:
                    self._final_exception_available = False
                else:
                    # A first ordinary write is also the initial lifecycle
                    # exception; future close may still use its final slot.
                    self._initial_exception_available = False
                has_pending = self._pending is not None
                should_exit = self._close_requested.is_set() and not has_pending
                if should_exit:
                    return
                if has_pending:
                    self._wake.set()
            with self._lock:
                # ``delay`` is recomputed from the successful write on the next
                # loop; this keeps ordinary cadence monotonic-clock based.
                delay = None

    def close(self, *, timeout: float = 0.1) -> None:
        """Request shutdown immediately and wait only within the supplied budget."""
        if os.getpid() != self._pid:
            return
        self._close_requested.set()
        self._wake.set()
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
            return self._close_requested.is_set()
        return not thread.is_alive()


_reporter: _Reporter | None = None
_reporter_lock = threading.Lock()


def _get_reporter() -> _Reporter | None:
    global _reporter
    path_value = os.environ.get("QEXP_PROGRESS_PATH")
    if not path_value or not _is_primary():
        return None
    desired_path = Path(path_value)
    current = _reporter
    if current is not None and not current.replaceable():
        # Never route a later Attempt to an older Attempt's still-running writer.
        if current.path != desired_path:
            return None
        return current
    if not _reporter_lock.acquire(blocking=False):
        return None
    try:
        current = _reporter
        if current is not None and current.replaceable():
            _reporter = None
        if _reporter is not None:
            return _reporter if _reporter.path == desired_path else None
        _reporter = _Reporter(desired_path, interval_seconds=_environment_interval())
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
            # The independent close-request Event guarantees it eventually exits.
            if reporter.replaceable():
                _reporter = None
    finally:
        _reporter_lock.release()


atexit.register(flush)
