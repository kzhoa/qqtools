"""Pure human presentation for qexp progress observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Mapping, Sequence

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


def select_progress_observation(
    progress: Mapping[str, object] | None,
    progress_extended: Mapping[str, object] | None,
    selected_progress_version: int | None,
) -> Mapping[str, object] | None:
    """Resolve the selected whole observation, retaining a v1 absence reason when nothing is valid."""
    if selected_progress_version == 1:
        return progress
    if selected_progress_version == 2:
        return progress_extended
    for candidate in (progress, progress_extended):
        if isinstance(candidate, Mapping) and candidate.get("status") != "available":
            return candidate
    return None


def _unavailable_details(observation: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    state = observation.get("observation_state", "unavailable")
    reason = observation.get("reason")
    explanation = _UNAVAILABLE_EXPLANATIONS.get(reason) if isinstance(reason, str) else None
    return (
        ("Progress status", state),
        ("Progress", explanation or "unavailable"),
    )


def _available_details(observation: Mapping[str, object], now: datetime) -> tuple[tuple[str, object], ...]:
    payload = observation.get("progress")
    if not isinstance(payload, Mapping):
        return _unavailable_details(observation)
    current, total = payload.get("current"), payload.get("total")
    current_value = current if type(current) is int else None
    total_value = total if type(total) is int else None
    count = "unknown" if current_value is None else str(current_value)
    if total_value is not None:
        count += f"/{total_value}"
    unit = payload.get("unit")
    if isinstance(unit, str) and unit:
        count += f" {unit}"
    if current_value is not None and total_value is not None and total_value > 0:
        count += f" ({100 * current_value / total_value:.1f}%)"
    return (
        ("Progress status", "available"),
        ("Stage", payload["stage"]),
        ("Progress", count),
        ("Message", payload.get("message")),
        ("Progress reported", _timestamp_with_age(observation["reported_at"], now)),
        ("Progress advanced", _timestamp_with_age(observation["advanced_at"], now)),
    )


def _metric_summary(observation: Mapping[str, object], progress_version: int | None) -> str:
    if progress_version != 2:
        return "unavailable (v1 progress selected)" if progress_version == 1 else "unavailable"
    payload = observation.get("progress")
    if not isinstance(payload, Mapping):
        return "unavailable"
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return "unavailable"

    names = sorted(name for name in metrics if isinstance(name, str))
    summary = [f"{name}={metrics[name]}" for name in names[:3]]
    if not summary:
        summary.append("none reported")
    remaining = len(names) - 3
    if remaining > 0:
        summary.append(f"+{remaining} more")

    completeness = payload.get("completeness")
    if isinstance(completeness, Mapping) and completeness.get("complete") is False:
        reasons = completeness.get("reasons")
        reason_text = (
            ", ".join(item for item in reasons if isinstance(item, str))
            if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes))
            else ""
        )
        omitted = completeness.get("omitted_metrics")
        omission_text = f", {omitted} omitted" if type(omitted) is int else ""
        summary.append(f"incomplete: {reason_text or 'unknown reason'}{omission_text}")
    return ", ".join(summary)


def _metric_details(observation: Mapping[str, object], progress_version: int | None) -> tuple[tuple[str, object], ...]:
    if progress_version != 2:
        detail = "unavailable (v1 progress selected)" if progress_version == 1 else "unavailable"
        return (("Metrics", detail),)
    payload = observation.get("progress")
    if not isinstance(payload, Mapping):
        return (("Metrics", "unavailable"),)
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return (("Metrics", "unavailable"),)

    fields: list[tuple[str, object]] = []
    if metrics:
        fields.extend((f"Metric {name}", metrics[name]) for name in sorted(metrics) if isinstance(name, str))
    else:
        fields.append(("Metrics", "none reported"))
    completeness = payload.get("completeness")
    if isinstance(completeness, Mapping):
        is_complete = completeness.get("complete")
        complete_text = "complete" if is_complete is True else "incomplete" if is_complete is False else "unknown"
        omitted = completeness.get("omitted_metrics")
        reasons = completeness.get("reasons")
        reason_text = (
            ", ".join(item for item in reasons if isinstance(item, str))
            if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes))
            else ""
        )
        fields.extend(
            (
                ("Metric completeness", complete_text),
                ("Omitted metrics", omitted if type(omitted) is int else "unknown"),
                ("Completeness reasons", reason_text or "none"),
            )
        )
    else:
        fields.append(("Metric completeness", "unavailable"))
    return tuple(fields)


def format_progress_details(
    observation: Mapping[str, object] | ProgressObservation | None,
    *,
    progress_version: int | None = None,
    include_metrics: bool = False,
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
        fields: tuple[tuple[str, object], ...] = (("Progress status", "unavailable"), ("Progress", "unavailable"))
    elif observation.get("status") != "available":
        fields = _unavailable_details(observation)
    else:
        fields = _available_details(observation, captured_now)
    if include_metrics:
        fields += (
            _metric_details(observation, progress_version) if observation is not None else (("Metrics", "unavailable"),)
        )
    return fields


def format_progress_compact(
    observation: Mapping[str, object] | ProgressObservation | None,
    *,
    progress_version: int | None = None,
    now: datetime | None = None,
) -> tuple[tuple[str, object], ...]:
    """Format the finite compact progress summary without probing a terminal."""
    fields = format_progress_details(observation, now=now)
    if observation is None or observation.get("status") != "available":
        return fields + (("Metrics", "unavailable"),)
    return fields + (("Metrics", _metric_summary(observation, progress_version)),)
