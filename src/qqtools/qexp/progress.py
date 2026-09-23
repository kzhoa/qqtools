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
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Callable

from ._progress_protocol import MAX_PAYLOAD_BYTES, replace_advisory_snapshot, validate_payload
from ._progress_protocol_v2 import MAX_PAYLOAD_V2_BYTES, capture_metrics, fit_payload_v2, semantic_key_v2

__all__ = ["update", "flush"]

DEFAULT_PROGRESS_INTERVAL_SECONDS = 30
_INITIAL_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 60.0


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
    candidate: dict[str, Any],
) -> bool:
    if any(
        pending.get(name) != candidate.get(name)
        for name in ("stage", "current", "total", "unit", "_metrics", "_completeness")
    ):
        return False
    if candidate["_message_parts"] is None:
        return pending.get("_message_parts") is None and pending.get("message") == candidate["message"]
    return (
        pending.get("_message_parts") == candidate["_message_parts"]
        and pending.get("_render_message") is candidate["_render_message"]
    )


@dataclass
class _OutputState:
    path: Path | None
    last_success: float = float("-inf")
    next_due: float = float("-inf")
    retry_due: float = float("-inf")
    failure_delay_seconds: float = _INITIAL_RETRY_DELAY_SECONDS
    last_written_key: tuple[Any, ...] | None = None
    last_written_source: tuple[Any, ...] | None = None
    initial_exception_available: bool = True
    final_exception_available: bool = True
    closing_failures: int = 0
    closing_exhausted: bool = False


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
        self,
        path: str | Path | None = None,
        *,
        path_v2: str | Path | None = None,
        interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        self._path = Path(path) if path else None
        self._path_v2 = Path(path_v2) if path_v2 else None
        self._v1_state = _OutputState(self._path)
        self._v2_state = _OutputState(self._path_v2)
        self._channels = tuple(
            (name, state, max_bytes, key_func)
            for name, state, max_bytes, key_func in (
                ("v1", self._v1_state, MAX_PAYLOAD_BYTES, _payload_key),
                ("v2", self._v2_state, MAX_PAYLOAD_V2_BYTES, semantic_key_v2),
            )
            if state.path is not None
        )
        self._interval = _safe_interval(interval_seconds)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._close_requested = threading.Event()
        self._pending: dict[str, Any] | None = None
        self._inflight: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._managed = False

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def paths(self) -> tuple[Path | None, Path | None]:
        return self._path, self._path_v2

    # Keep the established v1 state attributes observable for existing callers
    # and focused diagnostics while storing each channel independently.
    @property
    def _last_success(self) -> float:
        return self._v1_state.last_success

    @_last_success.setter
    def _last_success(self, value: float) -> None:
        self._v1_state.last_success = value

    @property
    def _next_due(self) -> float:
        return self._v1_state.next_due

    @_next_due.setter
    def _next_due(self, value: float) -> None:
        self._v1_state.next_due = value

    @property
    def _retry_due(self) -> float:
        return self._v1_state.retry_due

    @_retry_due.setter
    def _retry_due(self, value: float) -> None:
        self._v1_state.retry_due = value

    @property
    def _failure_delay_seconds(self) -> float:
        return self._v1_state.failure_delay_seconds

    @_failure_delay_seconds.setter
    def _failure_delay_seconds(self, value: float) -> None:
        self._v1_state.failure_delay_seconds = value

    @property
    def _last_written_key(self) -> tuple[Any, ...] | None:
        return self._v1_state.last_written_key

    @_last_written_key.setter
    def _last_written_key(self, value: tuple[Any, ...] | None) -> None:
        self._v1_state.last_written_key = value

    @property
    def _last_written_source(self) -> tuple[Any, ...] | None:
        return self._v1_state.last_written_source

    @_last_written_source.setter
    def _last_written_source(self, value: tuple[Any, ...] | None) -> None:
        self._v1_state.last_written_source = value

    @property
    def _initial_exception_available(self) -> bool:
        return self._v1_state.initial_exception_available

    @_initial_exception_available.setter
    def _initial_exception_available(self, value: bool) -> None:
        self._v1_state.initial_exception_available = value

    @property
    def _final_exception_available(self) -> bool:
        return self._v1_state.final_exception_available

    @_final_exception_available.setter
    def _final_exception_available(self, value: bool) -> None:
        self._v1_state.final_exception_available = value

    @property
    def _closing_failures(self) -> int:
        return self._v1_state.closing_failures

    @_closing_failures.setter
    def _closing_failures(self, value: int) -> None:
        self._v1_state.closing_failures = value

    def update(
        self,
        *,
        stage: str,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        message: str | None = None,
        metrics: object = None,
        _metrics_source_count: int | None = None,
        _message_parts: _MessageParts | None = None,
        _render_message: _MessageRenderer | None = None,
    ) -> bool:
        if (self._path is None and self._path_v2 is None) or os.getpid() != self._pid or not _is_primary():
            return False
        try:
            if _message_parts is None:
                # Preserve the public string contract without retaining subclasses
                # or invoking their overridden conversion/encoding methods.
                stage = str.__str__(stage) if isinstance(stage, str) else stage
                unit = str.__str__(unit) if isinstance(unit, str) else unit
                message = str.__str__(message) if isinstance(message, str) else message
                if _render_message is not None or not _valid_progress_fields(stage, current, total, unit, message):
                    return False
            else:
                if (
                    not _valid_message_parts(_message_parts)
                    or type(_render_message) is not FunctionType
                    or not _valid_progress_fields(stage, current, total, unit, None)
                ):
                    return False
            captured_metrics: dict[str, int | float] | None = None
            completeness: dict[str, Any] | None = None
            if self._path_v2 is not None:
                captured_metrics, completeness = capture_metrics(metrics)
                if _metrics_source_count is not None:
                    if (
                        type(_metrics_source_count) is not int
                        or _metrics_source_count < len(captured_metrics)
                        or _metrics_source_count.bit_length() > 63
                    ):
                        return False
                    omitted = _metrics_source_count - len(captured_metrics)
                    reasons = list(completeness["reasons"])
                    if _metrics_source_count > 32 and "metric_limit" not in reasons:
                        reasons.append("metric_limit")
                    if omitted > 0 and not reasons:
                        reasons.append("invalid_metrics")
                    if omitted == 0 and reasons:
                        return False
                    completeness = {
                        "complete": omitted == 0 and not reasons,
                        "omitted_metrics": omitted,
                        "reasons": reasons,
                    }
            candidate = {
                "protocol_version": 1,
                "stage": stage,
                "current": current,
                "total": total,
                "unit": unit,
                "message": message,
                "_message_parts": _message_parts,
                "_render_message": _render_message,
                "_metrics": captured_metrics,
                "_completeness": completeness,
            }
            if not self._lock.acquire(blocking=False):
                return False
            try:
                if self._close_requested.is_set():
                    return False
                if self._pending is not None and _same_pending_update(self._pending, candidate):
                    return True
                if self._inflight is not None and _same_pending_update(self._inflight, candidate):
                    if self._pending is not None:
                        self._pending = None
                        self._wake.set()
                    return True
                matches_last_written = (
                    self._path is None
                    or self._pending_matches_source(
                        candidate,
                        self._v1_state.last_written_source,
                        include_metrics=False,
                    )
                ) and (
                    self._path_v2 is None
                    or self._pending_matches_source(
                        candidate,
                        self._v2_state.last_written_source,
                        include_metrics=True,
                    )
                )
                if matches_last_written:
                    # This fact is already represented by the snapshot. Clear a
                    # different pending fact so the next report cannot go stale.
                    if self._inflight is None:
                        if self._pending is not None:
                            self._pending = None
                            self._wake.set()
                        return True

                wake = self._pending is None or not _same_pending_update(self._pending, candidate)
                # An update identifier is not a reason to rewrite an otherwise
                # identical semantic and message payload.
                self._pending = candidate
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
        payloads = self._build_payloads(pending)
        payload = payloads["v1"]
        if payload is None:
            raise ValueError("no v1 progress channel is configured")
        return payload

    def _build_payloads(self, pending: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
        update_id = pending.get("_cached_update_id")
        if update_id is None:
            update_id = uuid.uuid4().hex
            pending["_cached_update_id"] = update_id
        message = pending.get("_cached_message")
        if "_cached_message" not in pending:
            renderer = pending["_render_message"]
            message = renderer(pending["_message_parts"]) if renderer is not None else pending["message"]
            pending["_cached_message"] = message

        payloads: dict[str, dict[str, Any] | None] = {"v1": None, "v2": None}
        if self._path is not None:
            payload = pending.get("_cached_payload_v1")
            if payload is None:
                payload = validate_payload(
                    {
                        "protocol_version": 1,
                        "update_id": update_id,
                        "stage": pending["stage"],
                        "current": pending["current"],
                        "total": pending["total"],
                        "unit": pending["unit"],
                        "message": message,
                    }
                )
                pending["_cached_payload_v1"] = payload
                # Retain the original private cache key for compatible diagnostics.
                pending["_cached_payload"] = payload
            payloads["v1"] = payload
        if self._path_v2 is not None:
            payload = pending.get("_cached_payload_v2")
            if payload is None:
                payload = fit_payload_v2(
                    update_id=update_id,
                    stage=pending["stage"],
                    current=pending["current"],
                    total=pending["total"],
                    unit=pending["unit"],
                    message=message,
                    metrics=pending["_metrics"],
                    completeness=pending["_completeness"],
                )
                # None means even the v2 base cannot fit and is a terminal discard.
                pending["_cached_payload_v2"] = payload
            payloads["v2"] = payload
        return payloads

    @staticmethod
    def _pending_matches_source(
        pending: dict[str, Any],
        source: tuple[Any, ...] | None,
        *,
        include_metrics: bool,
    ) -> bool:
        if source is None:
            return False
        stage, current, total, unit, message, message_parts, renderer, metrics, completeness = source
        if (pending.get("stage"), pending.get("current"), pending.get("total"), pending.get("unit")) != (
            stage,
            current,
            total,
            unit,
        ):
            return False
        if message_parts is None:
            if pending.get("_message_parts") is not None or pending.get("message") != message:
                return False
        elif pending.get("_message_parts") != message_parts or pending.get("_render_message") is not renderer:
            return False
        return not include_metrics or (
            pending.get("_metrics") == metrics and pending.get("_completeness") == completeness
        )

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
            pending["_metrics"],
            pending["_completeness"],
        )

    def _due_outputs(
        self,
        pending: dict[str, Any],
        *,
        now: float,
        closing: bool,
    ) -> tuple[
        list[tuple[str, _OutputState, int, Callable[[dict[str, Any]], tuple[Any, ...]]]],
        list[float],
        bool,
    ]:
        due_outputs: list[tuple[str, _OutputState, int, Callable[[dict[str, Any]], tuple[Any, ...]]]] = []
        deadlines: list[float] = []
        outstanding = False
        for name, state, max_bytes, key_func in self._channels:
            if self._pending_matches_source(
                pending,
                state.last_written_source,
                include_metrics=name == "v2",
            ):
                continue
            if closing and (state.closing_exhausted or not state.final_exception_available):
                continue
            outstanding = True
            if now < state.retry_due:
                deadlines.append(state.retry_due)
                continue
            if closing:
                due = state.final_exception_available
            else:
                due = state.initial_exception_available or now >= state.next_due
            if due:
                due_outputs.append((name, state, max_bytes, key_func))
            elif not closing:
                deadlines.append(state.next_due)
        return due_outputs, deadlines, outstanding

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
                self._pending = None
                self._inflight = pending

            source = self._pending_source(pending)
            with self._lock:
                closing = self._close_requested.is_set()
                due_outputs, deadlines, outstanding = self._due_outputs(
                    pending,
                    now=time.monotonic(),
                    closing=closing,
                )
                if not due_outputs:
                    if self._pending is None and outstanding:
                        self._pending = pending
                    self._inflight = None
                    has_pending = self._pending is not None
                    if closing and not has_pending:
                        return
                    if has_pending and self._pending is not pending:
                        self._wake.set()
                        delay = None
                    elif has_pending:
                        future_deadlines = [item for item in deadlines if item > time.monotonic()]
                        delay = max(0.0, min(future_deadlines) - time.monotonic()) if future_deadlines else None
                    else:
                        delay = None
                    continue

            try:
                payloads = self._build_payloads(pending)
            except Exception:
                with self._lock:
                    self._inflight = None
                    if self._pending is not None:
                        self._wake.set()
                    if self._close_requested.is_set() and self._pending is None:
                        return
                delay = None
                continue

            for name, state, max_bytes, key_func in due_outputs:
                payload = payloads[name]
                if payload is None:
                    with self._lock:
                        state.last_written_source = source
                    continue
                if key_func(payload) == state.last_written_key:
                    with self._lock:
                        state.last_written_source = source
                    continue
                closing = self._close_requested.is_set()
                try:
                    assert state.path is not None
                    replace_advisory_snapshot(state.path, payload, max_bytes=max_bytes)
                except Exception:
                    with self._lock:
                        retry_delay = state.failure_delay_seconds
                        state.failure_delay_seconds = min(
                            retry_delay * 2,
                            _MAX_RETRY_DELAY_SECONDS,
                        )
                        state.retry_due = time.monotonic() + retry_delay
                        if closing:
                            state.closing_failures += 1
                            if state.closing_failures >= 3:
                                state.closing_exhausted = True
                        if self._pending is None and not state.closing_exhausted:
                            self._pending = pending
                    continue

                with self._lock:
                    succeeded_at = time.monotonic()
                    state.last_success = succeeded_at
                    state.retry_due = float("-inf")
                    state.failure_delay_seconds = _INITIAL_RETRY_DELAY_SECONDS
                    state.last_written_key = key_func(payload)
                    state.last_written_source = source
                    if state.initial_exception_available:
                        assert state.path is not None
                        state.next_due = succeeded_at + self._interval + _path_offset(state.path, self._interval)
                    elif not closing:
                        state.next_due = _next_deadline(state.next_due, succeeded_at, self._interval)
                    state.initial_exception_available = False
                    if closing:
                        state.final_exception_available = False

            with self._lock:
                self._inflight = None
                closing = self._close_requested.is_set()
                _, deadlines, outstanding = self._due_outputs(
                    pending,
                    now=time.monotonic(),
                    closing=closing,
                )
                if self._pending is None and outstanding:
                    self._pending = pending
                has_pending = self._pending is not None
                if closing and not has_pending:
                    return
                if has_pending and self._pending is not pending:
                    self._wake.set()
                    delay = None
                elif has_pending:
                    future_deadlines = [item for item in deadlines if item > time.monotonic()]
                    delay = max(0.0, min(future_deadlines) - time.monotonic()) if future_deadlines else None
                else:
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
    path_v2_value = os.environ.get("QEXP_PROGRESS_V2_PATH")
    if (not path_value and not path_v2_value) or not _is_primary():
        return None
    desired_path = Path(path_value) if path_value else None
    desired_path_v2 = Path(path_v2_value) if path_v2_value else None
    current = _reporter
    if current is not None and not current.replaceable():
        # Never route a later Attempt to an older Attempt's still-running writer.
        if current.paths != (desired_path, desired_path_v2):
            return None
        return current
    if not _reporter_lock.acquire(blocking=False):
        return None
    try:
        current = _reporter
        if current is not None and current.replaceable():
            _reporter = None
        if _reporter is not None:
            return _reporter if _reporter.paths == (desired_path, desired_path_v2) else None
        _reporter = _Reporter(
            desired_path,
            path_v2=desired_path_v2,
            interval_seconds=_environment_interval(),
        )
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
    metrics: object = None,
) -> bool:
    """Offer progress to the current qexp Attempt, or safely do nothing."""
    try:
        reporter = _get_reporter()
        if reporter is None:
            return False
        return reporter.update(stage=stage, current=current, total=total, unit=unit, message=message, metrics=metrics)
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
    metrics: object = None,
    _metrics_source_count: int | None = None,
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
            metrics=metrics,
            _metrics_source_count=_metrics_source_count,
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
