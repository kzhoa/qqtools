"""Machine-local progress for advisory Project snapshot publications."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .project_io_controller import _ValidationIdentity

SNAPSHOT_HEALTH_ITEM_LIMIT = 8
_COUNT_LIMIT = (1 << 31) - 1
_STAGE_ORDER = {
    "backoff": 0,
    "started": 1,
    "intent": 2,
    "queued": 3,
    "result_ready": 4,
    "consumed": 5,
    "published": 6,
}
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_RequestStage = Literal["queued", "started", "result_ready", "consumed", "published"]


@dataclass(slots=True)
class _PublicationProgress:
    stale_after: float
    stage: str = "intent"
    request_id: str | None = None
    attempt_started: float | None = None
    last_attempt_at: str | None = None
    published: float | None = None
    last_published_at: str | None = None
    consecutive_failures: int = 0
    retry_deadline: float | None = None
    reason_code: str | None = None
    worker_started: float | None = None
    worker_elapsed: float | None = None


class SnapshotPublicationHealth:
    """Track progress and expose bounded diagnostics for the current agent instance."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float],
        utc_clock: Callable[[], str],
        max_retry_seconds: float,
    ) -> None:
        self._monotonic = monotonic
        self._utc_clock = utc_clock
        self._max_retry_seconds = max_retry_seconds
        self._instance_id: str | None = None
        self._progress: dict[tuple[_ValidationIdentity, str], _PublicationProgress] = {}
        self._last_publication: dict[_ValidationIdentity, tuple[str, float]] = {}

    def reset_instance(self, instance_id: str) -> None:
        if self._instance_id != instance_id:
            self._progress.clear()
            self._last_publication.clear()
            self._instance_id = instance_id

    def prune(self, current_identities: Collection[_ValidationIdentity]) -> None:
        self._progress = {
            key: entry
            for key, entry in self._progress.items()
            if key[1] == self._instance_id and key[0] in current_identities
        }
        self._last_publication = {
            identity: state for identity, state in self._last_publication.items() if identity in current_identities
        }

    def discard_progress(self, identity: _ValidationIdentity, instance_id: str) -> None:
        self._progress.pop((identity, instance_id), None)

    def stage(self, identity: _ValidationIdentity, instance_id: str) -> str | None:
        entry = self._progress.get((identity, instance_id))
        return entry.stage if entry is not None else None

    def last_publication(self, identity: _ValidationIdentity) -> tuple[str, float] | None:
        return self._last_publication.get(identity)

    def record_publication(self, identity: _ValidationIdentity, signature: str, published_at: float) -> None:
        self._last_publication[identity] = (signature, published_at)

    def begin_intent(
        self,
        identity: _ValidationIdentity,
        instance_id: str,
        heartbeat_interval_seconds: int | float,
        *,
        new_attempt: bool = False,
    ) -> float:
        """Report an intent and return the original monotonic start of this attempt."""
        entry = self._entry(identity, instance_id, heartbeat_interval_seconds)
        now = self._now()
        attempt_started = entry.attempt_started
        if new_attempt or attempt_started is None:
            attempt_started = entry.attempt_started = now
            entry.last_attempt_at = self._timestamp()
            entry.request_id = None
            entry.worker_started = None
            entry.worker_elapsed = None
        entry.stage = "intent"
        entry.retry_deadline = None
        self._measure_worker(entry, now)
        return attempt_started

    def report_request(
        self,
        identity: _ValidationIdentity,
        instance_id: str,
        heartbeat_interval_seconds: int | float,
        request_id: str,
        *,
        stage: _RequestStage,
        reason: object = None,
    ) -> None:
        entry = self._entry(identity, instance_id, heartbeat_interval_seconds)
        self._ensure_attempt_started(entry)
        entry.request_id = request_id
        entry.stage = stage
        self._measure_worker(entry, self._now())
        if stage == "started":
            now = self._now()
            if entry.worker_started is None:
                entry.worker_started = now
            entry.retry_deadline = None
            self._measure_worker(entry, self._now())
        elif stage == "result_ready":
            if reason is not None:
                entry.reason_code = self._reason_code(reason, fallback="outcome_unknown")
            self._measure_worker(entry, self._now())
            entry.worker_started = None
        elif stage == "published":
            entry.last_published_at = self._timestamp()
            entry.published = self._now()
            entry.consecutive_failures = 0
            entry.retry_deadline = None
            entry.reason_code = None
            entry.worker_started = None

    def report_failure(
        self,
        identity: _ValidationIdentity,
        instance_id: str,
        heartbeat_interval_seconds: int | float,
        *,
        reason: object,
        request_id: str | None = None,
        retry_deadline: float | None = None,
    ) -> None:
        entry = self._entry(identity, instance_id, heartbeat_interval_seconds)
        self._ensure_attempt_started(entry)
        if request_id is not None:
            entry.request_id = request_id
        entry.stage = "backoff"
        entry.consecutive_failures = min(_COUNT_LIMIT, entry.consecutive_failures + 1)
        entry.reason_code = self._reason_code(reason, fallback="publication_failed")
        entry.retry_deadline = retry_deadline
        entry.worker_started = None

    def view(self, *, limit: int = SNAPSHOT_HEALTH_ITEM_LIMIT) -> dict[str, object]:
        """Return detached publication diagnostics in worst-first order."""
        if type(limit) is not int or not 1 <= limit <= SNAPSHOT_HEALTH_ITEM_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {SNAPSHOT_HEALTH_ITEM_LIMIT}.")
        now = self._now()
        entries: list[tuple[tuple[bool, int, float, str], dict[str, object]]] = []
        for (identity, instance_id), entry in self._progress.items():
            if instance_id != self._instance_id:
                continue
            elapsed = self._measure_worker(entry, now)
            is_overdue = elapsed is not None and elapsed >= entry.stale_after
            order = (
                not is_overdue,
                _STAGE_ORDER[entry.stage],
                entry.attempt_started if entry.attempt_started is not None else math.inf,
                identity.project_id,
            )
            entries.append(
                (
                    order,
                    {
                        "runtime_id": identity.runtime_id,
                        "agent_instance_id": instance_id,
                        "registry_revision": identity.registry_revision,
                        "executor_epoch": identity.executor_epoch,
                        "project_id": identity.project_id,
                        "registration_generation": identity.registration_generation,
                        "operation_kind": "machine_snapshot_publish",
                        "request_id": entry.request_id,
                        "stage": entry.stage,
                        "last_attempt_at": entry.last_attempt_at,
                        "last_published_at": entry.last_published_at,
                        "consecutive_failures": entry.consecutive_failures,
                        "retry_at": self._retry_timestamp(entry.retry_deadline, now),
                        "reason_code": entry.reason_code,
                        "worker_elapsed_seconds": entry.worker_elapsed,
                        "overdue": is_overdue,
                        "effective_stale_after_seconds": entry.stale_after,
                    },
                )
            )
        entries.sort(key=lambda item: item[0])
        total = min(_COUNT_LIMIT, len(entries))
        retained = [item for _, item in entries[:limit]]
        return {"total": total, "omitted_count": total - len(retained), "items": retained}

    def _entry(
        self, identity: _ValidationIdentity, instance_id: str, heartbeat_interval_seconds: int | float
    ) -> _PublicationProgress:
        key = (identity, instance_id)
        stale_after = float(heartbeat_interval_seconds) * 3.0
        entry = self._progress.get(key)
        if entry is None:
            entry = self._progress[key] = _PublicationProgress(stale_after)
        else:
            entry.stale_after = stale_after
        return entry

    def _ensure_attempt_started(self, entry: _PublicationProgress) -> None:
        if entry.attempt_started is None:
            entry.attempt_started = self._now()
            entry.last_attempt_at = self._timestamp()

    @staticmethod
    def _measure_worker(entry: _PublicationProgress, now: float) -> float | None:
        if entry.stage == "published" and entry.published is not None:
            baseline = entry.published
        elif entry.worker_started is not None:
            baseline = entry.worker_started
        else:
            baseline = entry.attempt_started
        if baseline is None:
            return None
        elapsed = max(0.0, now - baseline)
        if entry.worker_started is not None:
            entry.worker_elapsed = elapsed
        return elapsed

    def _now(self) -> float:
        try:
            value = float(self._monotonic())
        except Exception:
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _timestamp(self) -> str:
        try:
            value = self._utc_clock()
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value[:64]
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _retry_timestamp(self, deadline: float | None, now: float) -> str | None:
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            return None
        remaining = min(self._max_retry_seconds, max(0.0, deadline - now))
        try:
            base = datetime.fromisoformat(self._timestamp().replace("Z", "+00:00"))
            if base.tzinfo is None or base.utcoffset() is None:
                return None
            return (base + timedelta(seconds=remaining)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _reason_code(value: object, *, fallback: str) -> str:
        if isinstance(value, str) and _REASON_CODE.fullmatch(value):
            return value
        return fallback
