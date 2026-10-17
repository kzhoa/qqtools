"""Thread-safe, bounded health state for the machine control plane."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ..runtime.records import utc_now
from ..runtime.store import atomic_replace, read_json_limited
from .diagnostics import bounded_exception_evidence

HEALTH_FILENAME = "control_plane_health.json"
HEALTH_SCHEMA_VERSION = 1
MAX_COUNT = (1 << 31) - 1
MAX_TEXT_LENGTH = 160
MAX_HEALTH_BYTES = 64 * 1024
_REASON_LENGTH = 96
_REASON_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-")
_STATES = frozenset({"healthy", "retrying", "degraded", "stalled", "stopped", "unknown"})
_PROJECT_PUBLICATION_STAGES = frozenset(
    {"intent", "queued", "started", "result_ready", "consumed", "backoff", "published"}
)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_SAFE_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_MAX_PROJECT_PUBLICATIONS = 8
_LANE_FIELDS = (
    "state",
    "thread_alive",
    "last_cycle_started_at",
    "last_cycle_completed_at",
    "last_success_at",
    "last_failure_at",
    "consecutive_failures",
    "last_failure",
    "cycle_sequence",
    "cycle_duration_seconds",
    "missed_deadlines",
    "effective_stale_after_seconds",
    "diagnostic_coverage_loss_count",
    "diagnostic_coverage_degraded",
    "last_liveness_failure",
)


def _bounded_count(value: object, *, default: int = 0) -> int:
    if type(value) is not int:
        return default
    return min(MAX_COUNT, max(0, value))


def _increment_count(value: object, amount: int = 1) -> int:
    current = _bounded_count(value)
    if type(amount) is not int or amount <= 0:
        return current
    return min(MAX_COUNT, current + amount)


def _bounded_text(value: object, *, limit: int = MAX_TEXT_LENGTH) -> str | None:
    if not isinstance(value, str):
        return None
    return value[:limit]


def _bounded_timestamp(value: object) -> str | None:
    return _bounded_text(value, limit=64)


def _finite_nonnegative(value: object) -> float | None:
    if type(value) not in {int, float} or isinstance(value, bool):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        return None
    return converted


def _reason_code(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and 0 < len(value) <= _REASON_LENGTH and all(char in _REASON_CHARS for char in value):
        return value
    return fallback


def _safe_identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER_RE.fullmatch(value) else None


def _safe_timestamp(value: object) -> str | None:
    bounded = _bounded_timestamp(value)
    if bounded is None:
        return None
    try:
        parsed = datetime.fromisoformat(bounded.replace("Z", "+00:00"))
    except ValueError:
        return None
    return bounded if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _sanitize_project_publication(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    required_identifiers = (
        "runtime_id",
        "agent_instance_id",
        "executor_epoch",
        "project_id",
        "registration_generation",
    )
    result: dict[str, object] = {}
    for key in required_identifiers:
        item = _safe_identifier(value.get(key))
        if item is None:
            return None
        result[key] = item
    registry_revision = value.get("registry_revision")
    if type(registry_revision) is not int or not 0 <= registry_revision <= MAX_COUNT:
        return None
    result["registry_revision"] = registry_revision
    if value.get("operation_kind") != "machine_snapshot_publish":
        return None
    result["operation_kind"] = "machine_snapshot_publish"
    request_id = value.get("request_id")
    if request_id is not None:
        request_id = _safe_identifier(request_id)
        if request_id is None:
            return None
    result["request_id"] = request_id
    stage = value.get("stage")
    if stage not in _PROJECT_PUBLICATION_STAGES:
        return None
    result["stage"] = stage
    for key in ("last_attempt_at", "last_published_at", "retry_at"):
        item = value.get(key)
        if item is not None:
            item = _safe_timestamp(item)
            if item is None:
                return None
        result[key] = item
    failures = value.get("consecutive_failures")
    if type(failures) is not int or not 0 <= failures <= MAX_COUNT:
        return None
    result["consecutive_failures"] = failures
    reason = value.get("reason_code")
    if reason is not None:
        if not isinstance(reason, str) or _SAFE_REASON_RE.fullmatch(reason) is None:
            return None
    result["reason_code"] = reason
    for key in ("worker_elapsed_seconds", "effective_stale_after_seconds"):
        item = value.get(key)
        if item is not None:
            item = _finite_nonnegative(item)
            if item is None:
                return None
        result[key] = item
    overdue = value.get("overdue")
    if type(overdue) is not bool:
        return None
    result["overdue"] = overdue
    return result


def _sanitize_project_publications(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return None
    items: list[dict[str, object]] = []
    for raw in raw_items[:_MAX_PROJECT_PUBLICATIONS]:
        item = _sanitize_project_publication(raw)
        if item is not None:
            items.append(item)
    total = _bounded_count(value.get("total"), default=len(items))
    total = max(total, len(items))
    omitted_count = _bounded_count(value.get("omitted_count"), default=max(0, total - len(items)))
    omitted_count = max(omitted_count, total - len(items))
    return {"total": total, "omitted_count": omitted_count, "items": items}


def _new_lane(threshold: float) -> dict[str, Any]:
    lane = {
        "state": "unknown",
        "thread_alive": None,
        "last_cycle_started_at": None,
        "last_cycle_completed_at": None,
        "last_success_at": None,
        "last_failure_at": None,
        "consecutive_failures": 0,
        "last_failure": None,
        "cycle_sequence": 0,
        "cycle_duration_seconds": None,
        "missed_deadlines": 0,
        "effective_stale_after_seconds": threshold,
        "diagnostic_coverage_loss_count": 0,
        "diagnostic_coverage_degraded": False,
        "last_liveness_failure": None,
        "_active_sequence": None,
        "_started_monotonic": None,
        "_last_completed_monotonic": None,
        "_last_success_monotonic": None,
        "_last_failure_monotonic": None,
        "_failure_started_monotonic": None,
        "_failure_reason": None,
        "_suppressed_identical_count": 0,
        "_maximum_consecutive_failures": 0,
        "_last_cycle_failed": None,
    }
    return lane


class ControlPlaneHealth:
    """Maintain bounded heartbeat and authority health for one agent instance."""

    def __init__(
        self,
        runtime_root: str | Path,
        *,
        instance_id: str,
        interval_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        utc_clock: Callable[[], str] = utc_now,
        private_log: Callable[[str], None] | None = None,
        publish_instance_evidence: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> None:
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("control-plane instance_id must be a non-empty string")
        if len(instance_id) > MAX_TEXT_LENGTH:
            raise ValueError("control-plane instance_id is too long")
        if (
            type(interval_seconds) not in {int, float}
            or isinstance(interval_seconds, bool)
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("control-plane interval must be finite and positive")
        if not callable(monotonic) or not callable(utc_clock):
            raise TypeError("control-plane clocks must be callable")
        self._runtime_root = Path(runtime_root)
        self._instance_id = instance_id
        self._interval_seconds = float(interval_seconds)
        self._stale_after_seconds = self._interval_seconds * 3.0
        self._monotonic = monotonic
        self._utc_clock = utc_clock
        self._private_log = private_log
        self._publish_instance_evidence = publish_instance_evidence
        self._lock = threading.RLock()
        self._evidence_publish_lock = threading.Lock()
        self._heartbeat = _new_lane(self._stale_after_seconds)
        self._authority = _new_lane(self._stale_after_seconds)
        self._project_snapshot_publications: dict[str, object] = {"total": 0, "omitted_count": 0, "items": []}
        self._coverage_loss_count = 0
        self._coverage_loss_channels: set[str] = set()
        self._pending_evidence_generation = 0
        self._published_evidence_generation = 0

    def begin_heartbeat_cycle(self) -> int:
        """Mark a heartbeat attempt before project-heartbeat work begins."""
        now_monotonic, timestamp = self._clock_sample()
        with self._lock:
            lane = self._heartbeat
            sequence = _increment_count(lane["cycle_sequence"])
            lane["cycle_sequence"] = sequence
            lane["last_cycle_started_at"] = timestamp
            lane["cycle_duration_seconds"] = None
            lane["_active_sequence"] = sequence
            lane["_started_monotonic"] = now_monotonic
            return sequence

    def complete_heartbeat_cycle(self, sequence: int, status: object) -> None:
        """Complete one heartbeat cycle; only ``queued`` is successful."""
        if status == "queued":
            transition, evidence = self._complete_success(self._heartbeat, sequence, lane_name="heartbeat")
        else:
            reason = _reason_code(status, fallback="heartbeat_status_failure")
            failure = {
                "reason_code": reason,
                "cycle_sequence": _bounded_count(sequence),
                "sanitized": True,
            }
            transition, evidence = self._complete_failure(
                self._heartbeat,
                sequence,
                failure,
                reason=reason,
                lane_name="heartbeat",
            )
        self._dispatch_callbacks(transition, evidence)

    def fail_heartbeat_cycle(self, sequence: int, exc: Exception) -> None:
        """Record an exception that ended one heartbeat cycle and re-raise it to callers."""
        reason = "heartbeat_cycle_exception"
        failure: dict[str, Any] = {
            "reason_code": reason,
            "cycle_sequence": _bounded_count(sequence),
            "sanitized": True,
        }
        try:
            bounded = bounded_exception_evidence(exc)
        except Exception:
            bounded = {"exception_type": type(exc).__name__[:MAX_TEXT_LENGTH]}
        if isinstance(bounded, dict):
            for key in ("exception_type", "filename", "function", "line"):
                if key in bounded:
                    failure[key] = bounded[key]
        transition, evidence = self._complete_failure(
            self._heartbeat,
            sequence,
            failure,
            reason=reason,
            lane_name="heartbeat",
        )
        self._dispatch_callbacks(transition, evidence)

    def record_heartbeat_exception(self, exc: Exception) -> None:
        """Record a loop-level exception without duplicating cycle-level accounting."""
        with self._lock:
            sequence = self._heartbeat.get("_active_sequence")
            if type(sequence) is not int:
                current = _bounded_count(self._heartbeat.get("cycle_sequence"))
                failure = self._heartbeat.get("last_failure")
                if (
                    isinstance(failure, dict)
                    and failure.get("reason_code") == "heartbeat_cycle_exception"
                    and failure.get("cycle_sequence") == current
                    and self._heartbeat.get("_last_cycle_failed") is True
                ):
                    return
                now_monotonic, timestamp = self._clock_sample()
                sequence = _increment_count(self._heartbeat.get("cycle_sequence"))
                self._heartbeat["cycle_sequence"] = sequence
                self._heartbeat["last_cycle_started_at"] = timestamp
                self._heartbeat["_active_sequence"] = sequence
                self._heartbeat["_started_monotonic"] = now_monotonic
        self.fail_heartbeat_cycle(sequence, exc)

    def record_missed_deadlines(self, count: int) -> None:
        """Accumulate heartbeat deadline slots skipped by normal deadline advancement."""
        if type(count) is not int or count <= 0:
            return
        with self._lock:
            self._heartbeat["missed_deadlines"] = _increment_count(self._heartbeat["missed_deadlines"], count)

    def record_authority_missed_deadlines(self, count: int) -> None:
        """Accumulate authority deadline slots skipped by normal deadline advancement."""
        if type(count) is not int or count <= 0:
            return
        with self._lock:
            self._authority["missed_deadlines"] = _increment_count(self._authority["missed_deadlines"], count)

    def active_heartbeat_sequence(self) -> int | None:
        """Return the in-flight heartbeat sequence, if one is being recorded."""
        with self._lock:
            sequence = self._heartbeat.get("_active_sequence")
            return sequence if type(sequence) is int else None

    def record_diagnostic_coverage_loss(self, channel: str = "diagnostic") -> None:
        """Record one failed private diagnostic callback without invoking callbacks again."""
        self._record_coverage_loss(notify=False, channel=channel)

    def record_diagnostic_coverage_recovery(self, channel: str = "diagnostic") -> None:
        """Mark one diagnostic channel healthy while retaining its cumulative loss count."""
        self._record_coverage_recovery(channel=channel)

    def monitor_heartbeat(self, heartbeat_thread_alive: bool) -> dict[str, object]:
        """Observe heartbeat liveness and return the current bounded health snapshot."""
        if type(heartbeat_thread_alive) is not bool:
            raise TypeError("heartbeat_thread_alive must be a boolean")
        now_monotonic, timestamp = self._clock_sample()
        transition: dict[str, object] | None = None
        evidence: dict[str, object] | None = None
        with self._lock:
            lane = self._heartbeat
            previous_state = lane.get("state")
            lane["thread_alive"] = heartbeat_thread_alive
            next_state = self._derive_state(lane, now_monotonic, heartbeat_thread_alive)
            lane["state"] = next_state
            previous_liveness_state = previous_state if previous_state in {"stalled", "stopped"} else None
            next_liveness_state = next_state if next_state in {"stalled", "stopped"} else None
            if next_liveness_state is not None and previous_state != next_liveness_state:
                reason = "heartbeat_thread_stopped" if next_liveness_state == "stopped" else "heartbeat_cycle_stalled"
                liveness_failure = {
                    "state": next_liveness_state,
                    "reason_code": reason,
                    "observed_at": timestamp,
                    "cycle_sequence": _bounded_count(lane.get("cycle_sequence")),
                    "thread_alive": heartbeat_thread_alive,
                }
                lane["last_liveness_failure"] = liveness_failure
                transition = self._transition_event(
                    event="liveness_transition",
                    lane_name="heartbeat",
                    sequence=_bounded_count(lane.get("cycle_sequence")),
                    reason=reason,
                    count=_bounded_count(lane.get("consecutive_failures")),
                    duration=None,
                    suppressed=0,
                    state=next_liveness_state,
                    observed_at=timestamp,
                )
            elif previous_liveness_state is not None and next_liveness_state is None:
                transition = self._transition_event(
                    event="recovered",
                    lane_name="heartbeat",
                    sequence=_bounded_count(lane.get("cycle_sequence")),
                    reason="recovered",
                    count=_bounded_count(lane.get("consecutive_failures")),
                    duration=lane.get("cycle_duration_seconds"),
                    suppressed=_bounded_count(lane.get("_suppressed_identical_count")),
                    state=next_state,
                    observed_at=timestamp,
                )
            if transition is not None:
                evidence = self._control_plane_evidence_locked()
        self._dispatch_callbacks(transition, evidence)
        return self.snapshot(heartbeat_thread_alive=heartbeat_thread_alive)

    def update_authority(self, sample: Mapping[str, object]) -> None:
        """Update authority health from one already-completed authority sample."""
        if not isinstance(sample, Mapping):
            return
        sequence = sample.get("sequence")
        if type(sequence) is not int or sequence < 0:
            with self._lock:
                sequence = _increment_count(self._authority["cycle_sequence"])
        status = sample.get("observation_status")
        if status == "completed":
            transition, evidence = self._complete_authority_success(sequence, sample)
        else:
            reason = _reason_code(status, fallback="authority_cycle_failure")
            failure = {
                "reason_code": reason,
                "cycle_sequence": _bounded_count(sequence),
                "sanitized": True,
            }
            transition, evidence = self._complete_authority_failure(sequence, sample, failure, reason)
        self._dispatch_callbacks(transition, evidence)

    def snapshot(self, *, heartbeat_thread_alive: bool | None) -> dict[str, object]:
        """Return an immutable bounded projection of both control-plane lanes."""
        now_monotonic = self._monotonic_value()
        with self._lock:
            heartbeat = self._public_lane(self._heartbeat, now_monotonic, heartbeat_thread_alive)
            authority = self._public_lane(self._authority, now_monotonic, None)
            state = self._aggregate_state(heartbeat["state"], authority["state"])
            if state not in {"stopped", "stalled"} and any(
                item.get("overdue") is True for item in self._project_snapshot_publications["items"]
            ):
                state = "degraded"
            return {
                "instance_id": self._instance_id,
                "state": state,
                "heartbeat": heartbeat,
                "authority": authority,
                "project_snapshot_publications": {
                    "total": self._project_snapshot_publications["total"],
                    "omitted_count": self._project_snapshot_publications["omitted_count"],
                    "items": [dict(item) for item in self._project_snapshot_publications["items"]],
                },
            }

    def update_project_snapshot_publications(self, value: Mapping[str, object]) -> None:
        """Replace the bounded controller-owned Project publication view."""
        sanitized = _sanitize_project_publications(value)
        if sanitized is None:
            raise ValueError("project snapshot publication health is invalid")
        with self._lock:
            previous = {
                self._project_publication_key(item): item
                for item in self._project_snapshot_publications["items"]
                if item.get("overdue") is True
            }
            self._project_snapshot_publications = sanitized
            current = {
                self._project_publication_key(item): item
                for item in self._project_snapshot_publications["items"]
                if item.get("overdue") is True
            }
        for key in current.keys() - previous.keys():
            self._log_project_publication_transition("project_snapshot_overdue", current[key])
        for key in previous.keys() - current.keys():
            self._log_project_publication_transition("project_snapshot_resolved", previous[key])

    @staticmethod
    def _project_publication_key(item: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(
            item.get(key)
            for key in (
                "runtime_id",
                "agent_instance_id",
                "registry_revision",
                "executor_epoch",
                "project_id",
                "registration_generation",
                "request_id",
            )
        )

    def _log_project_publication_transition(self, event: str, item: Mapping[str, object]) -> None:
        if self._private_log is None:
            return
        record = {
            "event": event,
            "instance_id": self._instance_id,
            "project_id": item.get("project_id"),
            "registration_generation": item.get("registration_generation"),
            "request_id": item.get("request_id"),
            "stage": item.get("stage"),
            "reason_code": item.get("reason_code"),
            "consecutive_failures": item.get("consecutive_failures"),
            "worker_elapsed_seconds": item.get("worker_elapsed_seconds"),
        }
        try:
            self._private_log(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        except Exception:
            self._record_coverage_loss(notify=False, channel="transition_log")
        else:
            self._record_coverage_recovery(channel="transition_log")

    def persist(self, *, heartbeat_thread_alive: bool | None) -> bool:
        """Best-effort replace-in-place persistence of the current health record."""
        self._record_coverage_recovery(channel="health_persistence")
        snapshot = self.snapshot(heartbeat_thread_alive=heartbeat_thread_alive)
        record = {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "instance_id": self._instance_id,
            "persisted_at": self._timestamp(),
            **snapshot,
        }
        try:
            result = atomic_replace(self._runtime_root / HEALTH_FILENAME, record)
            if result is None:
                raise OSError("control-plane health replacement was not witnessed")
            return True
        except Exception:
            self._record_coverage_loss(notify=True, channel="health_persistence")
            return False

    def _clock_sample(self) -> tuple[float, str]:
        return self._monotonic_value(), self._timestamp()

    def _monotonic_value(self) -> float:
        try:
            value = float(self._monotonic())
        except Exception:
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _timestamp(self) -> str:
        try:
            value = self._utc_clock()
        except Exception:
            value = utc_now()
        if isinstance(value, str) and value:
            return value[:64]
        return utc_now()

    def _complete_success(
        self,
        lane: dict[str, Any],
        sequence: int,
        *,
        lane_name: str,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        now_monotonic, timestamp = self._clock_sample()
        with self._lock:
            if lane.get("_active_sequence") != sequence:
                return None, None
            previous_failures = _bounded_count(lane["consecutive_failures"])
            duration = self._finish_cycle_locked(lane, sequence, now_monotonic, timestamp)
            lane["last_success_at"] = timestamp
            lane["_last_success_monotonic"] = now_monotonic
            lane["_failure_started_monotonic"] = None
            lane["consecutive_failures"] = 0
            lane["_last_cycle_failed"] = False
            previous_state = lane.get("state")
            lane["state"] = "healthy"
            transition = None
            if previous_failures or previous_state in {"retrying", "degraded", "stalled", "stopped"}:
                transition = self._transition_event(
                    event="recovered",
                    lane_name=lane_name,
                    sequence=sequence,
                    reason="recovered",
                    count=0,
                    duration=duration,
                    suppressed=_bounded_count(lane["_suppressed_identical_count"]),
                    state="healthy",
                    observed_at=timestamp,
                )
                lane["_suppressed_identical_count"] = 0
            evidence = self._control_plane_evidence_locked() if transition is not None else None
            if transition is None and previous_state != "healthy" and previous_failures:
                transition = self._transition_event(
                    event="recovered",
                    lane_name=lane_name,
                    sequence=sequence,
                    reason="recovered",
                    count=0,
                    duration=duration,
                    suppressed=0,
                    state="healthy",
                    observed_at=timestamp,
                )
                evidence = self._control_plane_evidence_locked()
            return transition, evidence

    def _complete_failure(
        self,
        lane: dict[str, Any],
        sequence: int,
        failure: dict[str, Any],
        *,
        reason: str,
        lane_name: str,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        now_monotonic, timestamp = self._clock_sample()
        with self._lock:
            if lane.get("_active_sequence") != sequence:
                return None, None
            previous_reason = lane.get("_failure_reason")
            prior_suppressed = _bounded_count(lane["_suppressed_identical_count"])
            previous_state = lane.get("state")
            cycle_started = lane.get("_started_monotonic")
            duration = self._finish_cycle_locked(lane, sequence, now_monotonic, timestamp)
            previous_failures = _bounded_count(lane["consecutive_failures"])
            lane["consecutive_failures"] = _increment_count(previous_failures)
            lane["_maximum_consecutive_failures"] = max(
                _bounded_count(lane["_maximum_consecutive_failures"]),
                _bounded_count(lane["consecutive_failures"]),
            )
            lane["last_failure_at"] = timestamp
            lane["_last_failure_monotonic"] = now_monotonic
            if previous_failures == 0:
                lane["_failure_started_monotonic"] = (
                    cycle_started if type(cycle_started) in {int, float} else now_monotonic
                )
            lane["last_failure"] = dict(failure)
            lane["_last_cycle_failed"] = True
            lane["_failure_reason"] = reason
            next_state = self._derive_state(lane, now_monotonic, thread_alive=True)
            lane["state"] = next_state
            transition = None
            if not previous_failures:
                event = "failure"
            elif previous_reason != reason:
                event = "failure_reason_changed"
            elif previous_state != next_state:
                event = "state_changed"
            else:
                event = None
            if event is not None:
                transition = self._transition_event(
                    event=event,
                    lane_name=lane_name,
                    sequence=sequence,
                    reason=reason,
                    count=lane["consecutive_failures"],
                    duration=duration,
                    suppressed=prior_suppressed,
                )
                lane["_suppressed_identical_count"] = 0
            elif previous_reason == reason:
                lane["_suppressed_identical_count"] = _increment_count(prior_suppressed)
            evidence = self._control_plane_evidence_locked() if transition is not None else None
            return transition, evidence

    def _finish_cycle_locked(
        self,
        lane: dict[str, Any],
        sequence: int,
        now_monotonic: float,
        timestamp: str,
    ) -> float | None:
        started = lane.get("_started_monotonic")
        duration = None
        if type(started) in {int, float} and math.isfinite(float(started)):
            duration = max(0.0, now_monotonic - float(started))
        lane["last_cycle_completed_at"] = timestamp
        lane["cycle_duration_seconds"] = duration
        lane["_last_completed_monotonic"] = now_monotonic
        lane["_active_sequence"] = None
        lane["_started_monotonic"] = None
        if lane["cycle_sequence"] < sequence:
            lane["cycle_sequence"] = _bounded_count(sequence)
        return duration

    def _complete_authority_success(
        self,
        sequence: int,
        sample: Mapping[str, object],
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        timestamp = _bounded_timestamp(sample.get("sampled_at")) or self._timestamp()
        duration = _finite_nonnegative(sample.get("work_seconds"))
        with self._lock:
            lane = self._authority
            previous_failures = _bounded_count(lane["consecutive_failures"])
            previous_state = lane.get("state")
            self._finish_authority_locked(lane, sequence, sample, timestamp, duration)
            lane["last_success_at"] = timestamp
            lane["_last_success_monotonic"] = self._monotonic_value()
            lane["_failure_started_monotonic"] = None
            lane["consecutive_failures"] = 0
            lane["state"] = "healthy"
            transition = None
            if previous_failures or previous_state not in {"unknown", "healthy"}:
                transition = self._transition_event(
                    event="recovered",
                    lane_name="authority",
                    sequence=sequence,
                    reason="recovered",
                    count=0,
                    duration=duration,
                    suppressed=_bounded_count(lane["_suppressed_identical_count"]),
                )
                lane["_suppressed_identical_count"] = 0
            evidence = self._control_plane_evidence_locked() if transition is not None else None
            return transition, evidence

    def _complete_authority_failure(
        self,
        sequence: int,
        sample: Mapping[str, object],
        failure: dict[str, Any],
        reason: str,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        timestamp = _bounded_timestamp(sample.get("sampled_at")) or self._timestamp()
        duration = _finite_nonnegative(sample.get("work_seconds"))
        with self._lock:
            lane = self._authority
            previous_failures = _bounded_count(lane["consecutive_failures"])
            previous_reason = lane.get("_failure_reason")
            prior_suppressed = _bounded_count(lane["_suppressed_identical_count"])
            previous_state = lane.get("state")
            self._finish_authority_locked(lane, sequence, sample, timestamp, duration)
            lane["consecutive_failures"] = _increment_count(previous_failures)
            lane["_maximum_consecutive_failures"] = max(
                _bounded_count(lane["_maximum_consecutive_failures"]),
                _bounded_count(lane["consecutive_failures"]),
            )
            lane["last_failure_at"] = timestamp
            lane["_last_failure_monotonic"] = self._monotonic_value()
            if previous_failures == 0:
                lane["_failure_started_monotonic"] = lane["_last_failure_monotonic"]
            lane["last_failure"] = dict(failure)
            lane["_failure_reason"] = reason
            next_state = self._derive_state(lane, self._monotonic_value(), thread_alive=None)
            lane["state"] = next_state
            if not previous_failures:
                event = "failure"
            elif previous_reason != reason:
                event = "failure_reason_changed"
            elif previous_state != next_state:
                event = "state_changed"
            else:
                event = None
            transition = (
                self._transition_event(
                    event=event,
                    lane_name="authority",
                    sequence=sequence,
                    reason=reason,
                    count=lane["consecutive_failures"],
                    duration=duration,
                    suppressed=prior_suppressed,
                )
                if event is not None
                else None
            )
            if event is not None:
                lane["_suppressed_identical_count"] = 0
            elif previous_reason == reason:
                lane["_suppressed_identical_count"] = _increment_count(prior_suppressed)
            evidence = self._control_plane_evidence_locked() if transition is not None else None
            return transition, evidence

    def _finish_authority_locked(
        self,
        lane: dict[str, Any],
        sequence: int,
        sample: Mapping[str, object],
        timestamp: str,
        duration: float | None,
    ) -> None:
        lane["cycle_sequence"] = max(_bounded_count(lane["cycle_sequence"]), _bounded_count(sequence))
        lane["last_cycle_completed_at"] = timestamp
        lane["cycle_duration_seconds"] = duration
        started_at = _bounded_timestamp(sample.get("cycle_started_at"))
        if started_at is not None:
            lane["last_cycle_started_at"] = started_at

    def _derive_state(self, lane: dict[str, Any], now_monotonic: float, thread_alive: bool | None) -> str:
        if lane is self._heartbeat and thread_alive is False:
            return "stopped"
        if lane is self._heartbeat and thread_alive is True:
            active_started = lane.get("_started_monotonic")
            if type(active_started) in {int, float} and math.isfinite(float(active_started)):
                if max(0.0, now_monotonic - float(active_started)) >= self._stale_after_seconds:
                    return "stalled"
            last_completed = lane.get("_last_completed_monotonic")
            if type(last_completed) in {int, float} and math.isfinite(float(last_completed)):
                if max(0.0, now_monotonic - float(last_completed)) >= self._stale_after_seconds:
                    return "stalled"
        if lane["cycle_sequence"] == 0:
            return "unknown"
        if _bounded_count(lane["consecutive_failures"]) == 0:
            return "healthy"
        baseline = lane.get("_last_success_monotonic")
        if type(baseline) not in {int, float} or not math.isfinite(float(baseline)):
            baseline = lane.get("_failure_started_monotonic")
        age = max(0.0, now_monotonic - float(baseline)) if type(baseline) in {int, float} else 0.0
        return "degraded" if age >= self._stale_after_seconds else "retrying"

    def _public_lane(
        self,
        lane: dict[str, Any],
        now_monotonic: float,
        thread_alive: bool | None,
    ) -> dict[str, object]:
        effective_thread_alive = thread_alive if isinstance(thread_alive, bool) else lane.get("thread_alive")
        state = self._derive_state(lane, now_monotonic, effective_thread_alive)
        result = {key: lane.get(key) for key in _LANE_FIELDS}
        result["state"] = state
        result["thread_alive"] = effective_thread_alive
        if isinstance(result["last_failure"], dict):
            result["last_failure"] = dict(result["last_failure"])
        if isinstance(result["last_liveness_failure"], dict):
            result["last_liveness_failure"] = dict(result["last_liveness_failure"])
        result["consecutive_failures"] = _bounded_count(result["consecutive_failures"])
        result["cycle_sequence"] = _bounded_count(result["cycle_sequence"])
        result["missed_deadlines"] = _bounded_count(result["missed_deadlines"])
        result["diagnostic_coverage_loss_count"] = self._coverage_loss_count
        result["diagnostic_coverage_degraded"] = bool(self._coverage_loss_channels)
        return result

    def _aggregate_state(self, heartbeat_state: object, authority_state: object) -> str:
        states = {heartbeat_state, authority_state}
        if "stopped" in states:
            return "stopped"
        if "stalled" in states:
            return "stalled"
        if "degraded" in states:
            return "degraded"
        if "retrying" in states:
            return "retrying"
        if heartbeat_state in {"healthy", "retrying", "degraded"}:
            return str(heartbeat_state)
        if authority_state in {"healthy", "retrying", "degraded"}:
            return str(authority_state)
        return "unknown"

    def _transition_event(
        self,
        *,
        event: str,
        lane_name: str,
        sequence: int,
        reason: str,
        count: int,
        duration: float | None,
        suppressed: int,
        state: str | None = None,
        observed_at: str | None = None,
    ) -> dict[str, object]:
        event = {
            "event": event,
            "instance_id": self._instance_id,
            "lane": lane_name,
            "cycle_sequence": _bounded_count(sequence),
            "reason_code": _reason_code(reason, fallback="unknown"),
            "consecutive_failures": _bounded_count(count),
            "duration_seconds": duration,
            "suppressed_identical_count": _bounded_count(suppressed),
        }
        if state in _STATES:
            event["state"] = state
        if observed_at is not None:
            event["observed_at"] = _bounded_timestamp(observed_at)
        return event

    def _control_plane_evidence_locked(self) -> dict[str, object]:
        heartbeat = self._heartbeat
        return {
            "control_plane": {
                "state": self._aggregate_state(
                    self._derive_state(heartbeat, self._monotonic_value(), heartbeat.get("thread_alive")),
                    self._derive_state(self._authority, self._monotonic_value(), None),
                ),
                "last_success_at": heartbeat.get("last_success_at"),
                "last_failure_at": heartbeat.get("last_failure_at"),
                "maximum_consecutive_failures": _bounded_count(heartbeat.get("_maximum_consecutive_failures")),
                "last_failure": (
                    dict(heartbeat["last_failure"]) if isinstance(heartbeat.get("last_failure"), dict) else None
                ),
                "last_liveness_failure": (
                    dict(heartbeat["last_liveness_failure"])
                    if isinstance(heartbeat.get("last_liveness_failure"), dict)
                    else None
                ),
                "coverage_loss_count": self._coverage_loss_count,
                "coverage_degraded": bool(self._coverage_loss_channels),
            }
        }

    def _dispatch_callbacks(
        self,
        transition: dict[str, object] | None,
        evidence: dict[str, object] | None,
    ) -> None:
        if transition is not None and self._private_log is not None:
            try:
                self._private_log(json.dumps(transition, sort_keys=True, separators=(",", ":")) + "\n")
            except Exception:
                self._record_coverage_loss(notify=False, channel="transition_log")
            else:
                self._record_coverage_recovery(channel="transition_log")
        if evidence is not None:
            self._queue_instance_evidence()
        self._publish_pending_instance_evidence()

    def _queue_instance_evidence(self) -> None:
        if self._publish_instance_evidence is None:
            return
        with self._lock:
            self._pending_evidence_generation = _increment_count(self._pending_evidence_generation)

    def _publish_pending_instance_evidence(self) -> None:
        callback = self._publish_instance_evidence
        if callback is None:
            return
        with self._evidence_publish_lock:
            self._record_coverage_recovery(channel="instance_evidence")
            with self._lock:
                generation = self._pending_evidence_generation
                if generation <= self._published_evidence_generation:
                    return
                evidence = self._control_plane_evidence_locked()
            try:
                published = callback(evidence) is True
            except Exception:
                published = False
            if not published:
                self._record_coverage_loss(notify=False, channel="instance_evidence")
                return
            with self._lock:
                self._published_evidence_generation = max(self._published_evidence_generation, generation)

    def _record_coverage_loss(self, *, notify: bool, channel: str) -> None:
        with self._lock:
            self._coverage_loss_count = _increment_count(self._coverage_loss_count)
            self._coverage_loss_channels.add(channel)
            self._heartbeat["diagnostic_coverage_loss_count"] = self._coverage_loss_count
            self._authority["diagnostic_coverage_loss_count"] = self._coverage_loss_count
        if notify:
            self._queue_instance_evidence()
            self._publish_pending_instance_evidence()

    def _record_coverage_recovery(self, *, channel: str) -> None:
        with self._lock:
            self._coverage_loss_channels.discard(channel)


def _unknown_lane(*, thread_alive: bool | None) -> dict[str, object]:
    return {
        "state": "unknown",
        "thread_alive": thread_alive if isinstance(thread_alive, bool) else None,
        "last_cycle_started_at": None,
        "last_cycle_completed_at": None,
        "last_success_at": None,
        "last_failure_at": None,
        "consecutive_failures": None,
        "last_failure": None,
        "cycle_sequence": None,
        "cycle_duration_seconds": None,
        "missed_deadlines": None,
        "effective_stale_after_seconds": None,
        "diagnostic_coverage_loss_count": None,
        "diagnostic_coverage_degraded": None,
        "last_liveness_failure": None,
    }


def _sanitize_failure(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    reason = _reason_code(value.get("reason_code"), fallback="unknown_failure")
    result: dict[str, object] = {"reason_code": reason}
    sequence = _bounded_count(value.get("cycle_sequence"), default=-1)
    if sequence >= 0:
        result["cycle_sequence"] = sequence
    if type(value.get("sanitized")) is bool:
        result["sanitized"] = value["sanitized"]
    for key in ("exception_type", "filename", "function"):
        text = _bounded_text(value.get(key))
        if text is not None:
            result[key] = Path(text).name if key == "filename" else text
    line = value.get("line")
    if type(line) is int and 0 <= line <= (1 << 63) - 1:
        result["line"] = line
    return result


def _sanitize_liveness_failure(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    state = value.get("state")
    if state not in {"stalled", "stopped"}:
        return None
    reason = _reason_code(value.get("reason_code"), fallback="unknown_liveness_failure")
    if reason not in {"heartbeat_cycle_stalled", "heartbeat_thread_stopped"}:
        return None
    observed_at = _bounded_timestamp(value.get("observed_at"))
    sequence = _bounded_count(value.get("cycle_sequence"), default=-1)
    thread_alive = value.get("thread_alive")
    if observed_at is None or sequence < 0 or type(thread_alive) is not bool:
        return None
    return {
        "state": state,
        "reason_code": reason,
        "observed_at": observed_at,
        "cycle_sequence": sequence,
        "thread_alive": thread_alive,
    }


def _sanitize_lane(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if value.get("state") not in _STATES:
        return None
    result: dict[str, object] = {}
    for key in _LANE_FIELDS:
        if key not in value:
            if key in {"diagnostic_coverage_degraded", "last_liveness_failure"}:
                result[key] = None
                continue
            return None
        item = value[key]
        if key == "state":
            if item not in _STATES:
                return None
        elif key == "thread_alive":
            if item is not None and type(item) is not bool:
                return None
        elif key in {"last_cycle_started_at", "last_cycle_completed_at", "last_success_at", "last_failure_at"}:
            if item is not None and _bounded_timestamp(item) != item:
                return None
        elif key in {"consecutive_failures", "cycle_sequence", "missed_deadlines", "diagnostic_coverage_loss_count"}:
            if item is not None and (type(item) is not int or not 0 <= item <= MAX_COUNT):
                return None
        elif key == "diagnostic_coverage_degraded":
            if item is not None and type(item) is not bool:
                return None
        elif key == "last_failure":
            if item is not None:
                item = _sanitize_failure(item)
                if item is None:
                    return None
        elif key == "last_liveness_failure":
            if item is not None:
                item = _sanitize_liveness_failure(item)
                if item is None:
                    return None
        elif key in {"cycle_duration_seconds", "effective_stale_after_seconds"}:
            if item is not None and _finite_nonnegative(item) is None:
                return None
        result[key] = item
    return result


def _unknown_projection(instance_id: str | None, *, thread_alive: bool | None) -> dict[str, object]:
    heartbeat = _unknown_lane(thread_alive=thread_alive)
    return {
        "instance_id": instance_id,
        "state": "unknown",
        "heartbeat": heartbeat,
        "authority": _unknown_lane(thread_alive=None),
        "project_snapshot_publications": {"total": 0, "omitted_count": 0, "items": []},
    }


def read_control_plane_health(
    runtime_root: str | Path,
    *,
    instance_id: str | None,
    thread_alive: bool | None = None,
) -> dict[str, object]:
    """Read matching live health, returning an honest unknown projection on any mismatch."""
    if not isinstance(instance_id, str) or not instance_id:
        return _unknown_projection(instance_id, thread_alive=thread_alive)
    path = Path(runtime_root) / HEALTH_FILENAME
    try:
        value = read_json_limited(path, max_bytes=MAX_HEALTH_BYTES, record_type="control_plane_health")
    except Exception:
        return _unknown_projection(instance_id, thread_alive=thread_alive)
    if value.get("schema_version") != HEALTH_SCHEMA_VERSION or value.get("instance_id") != instance_id:
        return _unknown_projection(instance_id, thread_alive=thread_alive)
    heartbeat = _sanitize_lane(value.get("heartbeat"))
    authority = _sanitize_lane(value.get("authority"))
    state = value.get("state")
    if heartbeat is None or authority is None or state not in _STATES:
        return _unknown_projection(instance_id, thread_alive=thread_alive)
    publications = _sanitize_project_publications(value.get("project_snapshot_publications"))
    if publications is None:
        publications = {"total": 0, "omitted_count": 0, "items": []}
    if thread_alive is not None and type(thread_alive) is bool:
        heartbeat["thread_alive"] = thread_alive
        if thread_alive is False:
            heartbeat["state"] = "stopped"
    aggregate = state
    if heartbeat["state"] == "stopped" or authority["state"] == "stopped":
        aggregate = "stopped"
    elif heartbeat["state"] == "stalled" or authority["state"] == "stalled":
        aggregate = "stalled"
    elif heartbeat["state"] == "degraded" or authority["state"] == "degraded":
        aggregate = "degraded"
    elif heartbeat["state"] == "retrying" or authority["state"] == "retrying":
        aggregate = "retrying"
    elif heartbeat["state"] in _STATES - {"unknown"}:
        aggregate = heartbeat["state"]
    elif authority["state"] in _STATES - {"unknown"}:
        aggregate = authority["state"]
    else:
        aggregate = "unknown"
    if aggregate not in {"stopped", "stalled"} and any(item.get("overdue") is True for item in publications["items"]):
        aggregate = "degraded"
    return {
        "instance_id": instance_id,
        "state": aggregate,
        "heartbeat": heartbeat,
        "authority": authority,
        "project_snapshot_publications": publications,
    }


__all__ = ["ControlPlaneHealth", "read_control_plane_health"]
