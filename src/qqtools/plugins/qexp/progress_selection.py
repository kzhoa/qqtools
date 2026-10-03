"""Pure selection of one whole progress observation across protocol versions.

The observer reads and validates each channel independently.  This module only
decides which already validated candidate is suitable for presentation; it
performs no filesystem or runtime I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

_IDENTITY = (
    "task_id",
    "attempt_id",
    "attempt_number",
    "machine_name",
    "launch_id",
    "wrapper_pid",
    "wrapper_start_time_ticks",
)
_ACTIVITY_FIELDS = ("stage", "current", "total", "unit", "message")
_UNAVAILABLE_REASONS = ("identity_mismatch", "invalid_snapshot", "read_failed")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _candidate(
    value: Any, protocol_version: int, *, attempt_id: str | None, attempt_number: int | None
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("status") != "available" or value.get("observation_state") != "available":
        return None
    if type(value.get("protocol_version")) is not int or value["protocol_version"] != protocol_version:
        return None
    if attempt_id is not None and value.get("attempt_id") != attempt_id:
        return None
    if attempt_number is not None and value.get("attempt_number") != attempt_number:
        return None
    if _timestamp(value.get("reported_at")) is None:
        return None
    return value


def _activity(value: dict[str, Any], protocol_version: int) -> tuple[Any, ...] | None:
    progress = value.get("progress")
    if not isinstance(progress, dict):
        return None
    if protocol_version == 3:
        activity = progress.get("activity")
    else:
        activity = progress
    if not isinstance(activity, dict):
        return None
    if any(field not in activity for field in _ACTIVITY_FIELDS):
        return None
    return tuple(activity.get(field) for field in _ACTIVITY_FIELDS)


def _same_source(first: dict[str, Any], first_version: int, candidate: dict[str, Any], candidate_version: int) -> bool:
    """Return whether two candidates prove one producer update.

    Channel sequence numbers and timestamps are deliberately absent from this
    proof.  Every identity and fencing value is required because equal payloads
    from different Attempts must never be joined.
    """
    for field in _IDENTITY:
        if field not in first or field not in candidate or first[field] != candidate[field]:
            return False
    for field in ("fencing_token", "registration_generation"):
        if field not in first or field not in candidate or first[field] != candidate[field]:
            return False
    first_update = first.get("source_update_id")
    candidate_update = candidate.get("source_update_id")
    if (
        not isinstance(first_update, str)
        or not first_update
        or not isinstance(candidate_update, str)
        or not candidate_update
    ):
        return False
    if first_update != candidate_update:
        return False
    first_activity = _activity(first, first_version)
    candidate_activity = _activity(candidate, candidate_version)
    return first_activity is not None and first_activity == candidate_activity


def _candidates(
    progress: Any,
    progress_extended: Any,
    progress_scoped: Any,
    *,
    attempt_id: str | None,
    attempt_number: int | None,
) -> list[tuple[int, dict[str, Any], datetime]]:
    values = ((1, progress), (2, progress_extended), (3, progress_scoped))
    result: list[tuple[int, dict[str, Any], datetime]] = []
    for version, value in values:
        candidate = _candidate(value, version, attempt_id=attempt_id, attempt_number=attempt_number)
        if candidate is None:
            continue
        stamp = _timestamp(candidate.get("reported_at"))
        if stamp is not None:
            result.append((version, candidate, stamp))
    return result


def select_progress_protocol(
    progress: Any,
    progress_extended: Any,
    progress_scoped: Any,
    *,
    attempt_id: str | None = None,
    attempt_number: int | None = None,
) -> int | None:
    """Select the protocol for one complete, whole progress observation.

    The newest accepted report is the initial choice.  A candidate from a
    higher protocol may replace it only when its identity, fencing, generation,
    source update ID, and activity prove that it is the same producer update.
    """
    candidates = _candidates(
        progress,
        progress_extended,
        progress_scoped,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
    )
    if not candidates:
        return None
    initial = max(candidates, key=lambda item: (item[2], item[0]))
    selected_version, selected, _selected_at = initial
    for version, candidate, _stamp in candidates:
        if version > selected_version and _same_source(selected, selected_version, candidate, version):
            selected_version, selected = version, candidate
    return selected_version


def _failure_precedence(value: Any) -> int:
    if not isinstance(value, dict) or value.get("observation_state") != "unavailable":
        return -1
    reason = value.get("reason")
    try:
        return len(_UNAVAILABLE_REASONS) - _UNAVAILABLE_REASONS.index(reason)
    except ValueError:
        return 0


def selected_progress_observation(
    progress: Any,
    progress_extended: Any,
    progress_scoped: Any,
    selected_version: int | None,
    *,
    attempt_id: str | None = None,
    attempt_number: int | None = None,
) -> dict[str, Any] | None:
    """Return the selected candidate as one whole dictionary.

    When no candidate is available, an unavailable observation with the most
    actionable bounded reason is returned.  This keeps failure precedence
    deterministic while retaining ordinary pending/no-report observations when
    no channel failed.
    """
    by_version = {1: progress, 2: progress_extended, 3: progress_scoped}
    if selected_version in by_version:
        candidate = _candidate(
            by_version[selected_version],
            selected_version,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
        )
        if candidate is not None:
            return candidate

    observations = [value for value in (progress, progress_extended, progress_scoped) if isinstance(value, dict)]
    failures = [value for value in observations if _failure_precedence(value) >= 0]
    if failures:
        return max(failures, key=_failure_precedence)
    for state in ("pending", "no_report"):
        for value in observations:
            if value.get("observation_state") == state:
                return value
    return None


__all__ = ["select_progress_protocol", "selected_progress_observation"]
