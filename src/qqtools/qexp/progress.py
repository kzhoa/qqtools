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
from types import FunctionType
from typing import Any, Callable

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


_MessageAtom = str | int | None
_MessageParts = tuple[_MessageAtom, ...]
_MessageRenderer = Callable[[_MessageParts], str | None]


def _valid_text(value: object, limit: int, *, optional: bool = False) -> bool:
    if optional and value is None:
        return True
    if type(value) is not str or (not optional and not value) or len(value) > limit:
        return False
    try:
        if len(value.encode("utf-8")) > limit:
            return False
    except UnicodeEncodeError:
        return False
    return all(ord(character) >= 32 and ord(character) != 127 for character in value)


def _valid_progress_fields(
    stage: object,
    current: object,
    total: object,
    unit: object,
    message: object,
) -> bool:
    if not _valid_text(stage, 64):
        return False
    if current is not None and (type(current) is not int or not 0 <= current <= 2**63 - 1):
        return False
    if total is not None and (type(total) is not int or not 0 <= total <= 2**63 - 1):
        return False
    if current is not None and total is not None and current > total:
        return False
    return _valid_text(unit, 32, optional=True) and _valid_text(message, 1024, optional=True)


def _valid_message_parts(parts: object) -> bool:
    if type(parts) is not tuple or len(parts) > 16:
        return False
    for atom in parts:
        if atom is None:
            continue
        if type(atom) is int:
            if not 0 <= atom <= 2**63 - 1:
                return False
        elif not _valid_text(atom, 256):
            return False
    return True


def _same_pending_update(
    pending: dict[str, Any],
    *,
    stage: str,
    current: int | None,
    total: int | None,
    unit: str | None,
    message: str | None,
    message_parts: _MessageParts | None,
    render_message: _MessageRenderer | None,
) -> bool:
    if any(
        pending.get(name) != value
        for name, value in (("stage", stage), ("current", current), ("total", total), ("unit", unit))
    ):
        return False
    if message_parts is None:
        return pending.get("_message_parts") is None and pending.get("message") == message
    return pending.get("_message_parts") == message_parts and pending.get("_render_message") is render_message


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
        self._inflight: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._last_success = float("-inf")
        self._next_due = float("-inf")
        self._last_written_key: tuple[Any, ...] | None = None
        self._last_written_source: tuple[Any, ...] | None = None
        self._initial_exception_available = True
        self._final_exception_available = True
        self._closing_failures = 0
        self._managed = False

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
        _message_parts: _MessageParts | None = None,
        _render_message: _MessageRenderer | None = None,
    ) -> bool:
        if self._path is None or os.getpid() != self._pid or not _is_primary():
            return False
        try:
            if _message_parts is None:
                if _render_message is not None or not _valid_progress_fields(stage, current, total, unit, message):
                    return False
            else:
                if (
                    not _valid_message_parts(_message_parts)
                    or type(_render_message) is not FunctionType
                    or not _valid_progress_fields(stage, current, total, unit, None)
                ):
                    return False
            if not self._lock.acquire(blocking=False):
                return False
            try:
                if self._close_requested.is_set():
                    return False
                if self._pending is not None and _same_pending_update(
                    self._pending,
                    stage=stage,
                    current=current,
                    total=total,
                    unit=unit,
                    message=message,
                    message_parts=_message_parts,
                    render_message=_render_message,
                ):
                    return True
                if self._inflight is not None and _same_pending_update(
                    self._inflight,
                    stage=stage,
                    current=current,
                    total=total,
                    unit=unit,
                    message=message,
                    message_parts=_message_parts,
                    render_message=_render_message,
                ):
                    if self._pending is not None:
                        self._pending = None
                        self._wake.set()
                    return True
                matches_last_written = False
                if _message_parts is None and self._last_written_key is not None:
                    matches_last_written = self._last_written_key == (stage, current, total, unit, message)
                elif _message_parts is not None and self._last_written_source is not None:
                    last_stage, last_current, last_total, last_unit, _, last_parts, last_renderer = (
                        self._last_written_source
                    )
                    matches_last_written = (
                        (stage, current, total, unit) == (last_stage, last_current, last_total, last_unit)
                        and _message_parts == last_parts
                        and _render_message is last_renderer
                    )
                if matches_last_written:
                    # This fact is already represented by the snapshot. Clear a
                    # different pending fact so the next report cannot go stale.
                    if self._inflight is None:
                        if self._pending is not None:
                            self._pending = None
                            self._wake.set()
                        return True

                wake = self._pending is None or not _same_pending_update(
                    self._pending,
                    stage=stage,
                    current=current,
                    total=total,
                    unit=unit,
                    message=message,
                    message_parts=_message_parts,
                    render_message=_render_message,
                )
                # An update identifier is not a reason to rewrite an otherwise
                # identical semantic and message payload.
                self._pending = {
                    "protocol_version": 1,
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "unit": unit,
                    "message": message,
                    "_message_parts": _message_parts,
                    "_render_message": _render_message,
                }
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

    def request_managed_close(self) -> None:
        """Signal shutdown without taking either reporter lock or joining the writer."""
        if os.getpid() != self._pid:
            return
        self._close_requested.set()
        self._wake.set()

    def _build_payload(self, pending: dict[str, Any]) -> dict[str, Any]:
        cached = pending.get("_cached_payload")
        if cached is not None:
            return cached
        message = pending["message"]
        renderer = pending["_render_message"]
        if renderer is not None:
            message = renderer(pending["_message_parts"])
        return validate_payload(
            {
                "protocol_version": 1,
                "update_id": uuid.uuid4().hex,
                "stage": pending["stage"],
                "current": pending["current"],
                "total": pending["total"],
                "unit": pending["unit"],
                "message": message,
            }
        )

    def _pending_matches_last_written(self, pending: dict[str, Any]) -> bool:
        source = self._last_written_source
        if source is None:
            return False
        stage, current, total, unit, message, message_parts, renderer = source
        if (pending.get("stage"), pending.get("current"), pending.get("total"), pending.get("unit")) != (
            stage,
            current,
            total,
            unit,
        ):
            return False
        if message_parts is None:
            return pending.get("_message_parts") is None and pending.get("message") == message
        return pending.get("_message_parts") == message_parts and pending.get("_render_message") is renderer

    @staticmethod
    def _pending_source(pending: dict[str, Any]) -> tuple[Any, ...]:
        return (
            pending["stage"],
            pending["current"],
            pending["total"],
            pending["unit"],
            pending["message"],
            pending["_message_parts"],
            pending["_render_message"],
        )

    def _run(self) -> None:
        delay: float | None = None
        while True:
            safe_delay = None if delay is None else min(delay, threading.TIMEOUT_MAX)
            self._wake.wait(safe_delay)
            self._wake.clear()
            with self._lock:
                pending = self._pending
                if pending is None:
                    if self._close_requested.is_set():
                        return
                    delay = None
                    continue
                closing = self._close_requested.is_set()
                now = time.monotonic()
                if closing:
                    if not self._final_exception_available and self._pending_matches_last_written(pending):
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
                self._inflight = pending
            try:
                payload = self._build_payload(pending)
            except Exception:
                # Invalid or unrenderable primitive input is discarded without retry.
                with self._lock:
                    self._inflight = None
                    if self._close_requested.is_set() and self._pending is None:
                        return
                delay = None
                continue
            if not closing and _payload_key(payload) == self._last_written_key:
                with self._lock:
                    self._inflight = None
                    if self._pending is not None and self._pending_matches_last_written(self._pending):
                        self._pending = None
                    if self._close_requested.is_set() and self._pending is None:
                        return
                delay = None
                continue
            try:
                assert self._path is not None
                replace_advisory_snapshot(self._path, payload, max_bytes=MAX_PAYLOAD_BYTES)
            except Exception:
                pending["_cached_payload"] = payload
                with self._lock:
                    self._inflight = None
                    if closing:
                        self._closing_failures += 1
                        if self._closing_failures >= 3:
                            self._pending = None
                            return
                    # Keep the latest value for a later retry.  A failed
                    # replacement does not consume an initial/final exception,
                    # but final shutdown retries remain bounded.
                    if self._pending is None:
                        self._pending = pending
                delay = 0.05
                continue
            with self._lock:
                succeeded_at = time.monotonic()
                self._inflight = None
                self._last_success = succeeded_at
                self._last_written_key = _payload_key(payload)
                self._last_written_source = self._pending_source(pending)
                if self._pending is not None and self._pending_matches_last_written(self._pending):
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


def _offer_managed_progress(
    *,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    message_parts: _MessageParts,
    render_message: _MessageRenderer,
) -> bool:
    """Offer bounded primitive progress data for formatting by the writer."""
    try:
        reporter = _get_reporter()
        if reporter is None:
            return False
        accepted = reporter.update(
            stage=stage,
            current=current,
            total=total,
            unit=unit,
            _message_parts=message_parts,
            _render_message=render_message,
        )
        if accepted:
            reporter._managed = True
        return accepted
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


def _close_managed_progress() -> None:
    """Request automatic managed shutdown without waiting on either reporter lock."""
    reporter = _reporter
    if reporter is not None and reporter._managed:
        reporter.request_managed_close()


def _close_at_exit() -> None:
    reporter = _reporter
    if reporter is not None and reporter._managed:
        reporter.request_managed_close()
    else:
        flush()


atexit.register(_close_at_exit)
