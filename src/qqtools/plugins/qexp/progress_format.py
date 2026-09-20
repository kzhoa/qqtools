"""Pure human presentation for qexp progress observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime.progress_types import ProgressObservation


_UNAVAILABLE_EXPLANATIONS = {
    "not_started": "pending (not started)",
    "no_snapshot": "unavailable (no report yet)",
    "cleanup": "unavailable (cleanup in progress)",
    "read_failed": "unavailable (read failed)",
    "invalid_snapshot": "unavailable (invalid snapshot)",
    "identity_mismatch": "unavailable (identity mismatch)",
    "unknown": "unavailable (unknown)",
}


def _parse_timestamp(stamp: object) -> datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return parsed


def _relative_age(stamp: datetime, now: datetime) -> str:
    try:
        seconds = (now - stamp).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if seconds < 0:
        return "unknown (clock difference)"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h ago"


def _timestamp_with_age(stamp: object, now: datetime) -> str:
    parsed = _parse_timestamp(stamp)
    if parsed is None:
        return "unknown"
    try:
        absolute = parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "unknown"
    return f"{absolute} ({_relative_age(parsed, now)})"


def _unavailable_details(observation: ProgressObservation) -> tuple[tuple[str, object], ...]:
    state = observation.get("observation_state", "unavailable")
    reason = observation.get("reason")
    return (
        ("Progress status", state),
        ("Progress", _UNAVAILABLE_EXPLANATIONS.get(reason, "unavailable")),
    )


def _available_details(observation: ProgressObservation, now: datetime) -> tuple[tuple[str, object], ...]:
    payload = observation["progress"]
    current, total = payload.get("current"), payload.get("total")
    count = "unknown" if current is None else str(current)
    if total is not None:
        count += f"/{total}"
    if payload.get("unit"):
        count += f" {payload['unit']}"
    if current is not None and total is not None and total > 0:
        count += f" ({100 * current / total:.1f}%)"
    return (
        ("Progress status", "available"),
        ("Stage", payload["stage"]),
        ("Progress", count),
        ("Message", payload.get("message")),
        ("Progress reported", _timestamp_with_age(observation["reported_at"], now)),
        ("Progress advanced", _timestamp_with_age(observation["advanced_at"], now)),
    )


def format_progress_details(
    observation: ProgressObservation | None,
    *,
    now: datetime | None = None,
) -> tuple[tuple[str, object], ...]:
    """Format one progress observation without querying or mutating qexp state."""
    if now is None:
        captured_now = datetime.now(timezone.utc)
    else:
        if now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        captured_now = now
    if observation is None:
        return (("Progress status", "unavailable"), ("Progress", "unavailable"))
    if observation.get("status") != "available":
        return _unavailable_details(observation)
    return _available_details(observation, captured_now)
