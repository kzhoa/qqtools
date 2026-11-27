"""Process-local diagnostics for the Idle Job timer and supervisor-issued stops."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .idle_job_diagnostics import validate_interval_detail


def _observation_reset_check(previous: str | None, current: str) -> str:
    if previous is None:
        return "observation_changed"
    try:
        before, after = json.loads(previous), json.loads(current)
        if (
            not isinstance(before, dict)
            or before.keys() != after.keys()
            or before["demand"] != "idle"
            or any(
                type(before[field]) is not int or before[field] < 0 for field in ("demand_revision", "config_revision")
            )
        ):
            return "observation_changed"
        previous_policy = json.loads(before["policy"])
        if not isinstance(previous_policy, dict) or "resolved_gpu_ids" not in previous_policy:
            return "observation_changed"
    except (TypeError, ValueError, RecursionError):
        return "observation_changed"
    for field, check in (
        ("demand_revision", "demand_revision_changed"),
        ("config_revision", "config_revision_changed"),
        ("policy", "gpu_policy_changed"),
    ):
        if before[field] != after[field]:
            return check
    return "observation_changed"


class IdleJobIntervalHistory:
    def __init__(self, monotonic: Callable[[], float], wall_clock: Callable[[], float]) -> None:
        self._now = monotonic
        self._wall_now = wall_clock
        self._interval_observed = False
        self._last_interval_reset: dict[str, Any] | None = None
        self._last_stop: dict[str, Any] | None = None
        self._last_stop_instance_id: str | None = None

    def _diagnostic_timestamp(self) -> str:
        return datetime.fromtimestamp(self._wall_now(), timezone.utc).isoformat().replace("+00:00", "Z")

    def record_reset(
        self,
        idle_since: float | None,
        check: str,
        *,
        previous_key: str | None,
        replacement_key: str | None = None,
        now: float | None = None,
    ) -> None:
        if idle_since is None or not self._interval_observed:
            return
        self._interval_observed = False
        self._last_interval_reset = None
        try:
            if replacement_key is not None:
                check = _observation_reset_check(previous_key, replacement_key)
            observed = self._now() if now is None else now
            elapsed = max(0.0, observed - idle_since)
            if math.isfinite(elapsed):
                self._last_interval_reset = {
                    "check": check,
                    "elapsed_seconds": elapsed,
                    "reset_at": self._diagnostic_timestamp(),
                }
        except Exception:
            # Optional evidence cannot interrupt interval invalidation.
            pass

    def record_stop(self, instance_id: str, check: str) -> None:
        if self._last_stop_instance_id == instance_id:
            return
        self._last_stop_instance_id = instance_id
        self._last_stop = None
        try:
            self._last_stop = {
                "check": check,
                "instance_id": instance_id,
                "stopped_at": self._diagnostic_timestamp(),
            }
        except Exception:
            # The persisted stop request remains authoritative.
            pass

    def at_publication(self, idle_since: float | None) -> dict[str, Any] | None:
        # An interval created and cleared within one decision was never published.
        self._interval_observed = idle_since is not None
        if idle_since is None and self._last_interval_reset is None and self._last_stop is None:
            return None
        elapsed = max(0.0, self._now() - idle_since) if idle_since is not None else 0.0
        return validate_interval_detail(
            {"idle_elapsed_seconds": elapsed, "last_reset": self._last_interval_reset, "last_stop": self._last_stop}
        )
