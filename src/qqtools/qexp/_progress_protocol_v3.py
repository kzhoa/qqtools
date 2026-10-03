"""Bounded progress-v3 payload validation and advisory mailbox reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._progress_protocol import MAX_SNAPSHOT_BYTES, encode_json, read_advisory_snapshot
from ._progress_protocol_v2 import (
    _exact_dict_keys,
    _int64_nonnegative,
    _valid_identifier,
    _valid_metrics,
    _valid_text,
    _validated_completeness,
)

PROTOCOL_VERSION_V3 = 3
MAX_PAYLOAD_V3_BYTES = 8192
MAX_SNAPSHOT_V3_BYTES = MAX_SNAPSHOT_BYTES

_PAYLOAD_FIELDS = frozenset(("protocol_version", "update_id", "activity", "overall", "metrics", "completeness"))
_ACTIVITY_FIELDS = frozenset(("stage", "current", "total", "unit", "message"))
_OVERALL_FIELDS = frozenset(("current", "total", "unit", "label"))


def _validate_activity(value: object) -> dict[str, Any]:
    if type(value) is not dict or not _exact_dict_keys(value, _ACTIVITY_FIELDS):
        raise ValueError("invalid progress-v3 activity")
    stage = value["stage"]
    current = value["current"]
    total = value["total"]
    unit = value["unit"]
    message = value["message"]
    if (
        not _valid_text(stage, 64)
        or not _valid_text(unit, 32, optional=True)
        or not _valid_text(message, 1024, optional=True)
    ):
        raise ValueError("invalid progress-v3 activity text")
    for name, number in (("current", current), ("total", total)):
        if number is not None and not _int64_nonnegative(number):
            raise ValueError(f"invalid progress-v3 activity {name}")
    if current is not None and total is not None and current > total:
        raise ValueError("progress-v3 activity current exceeds total")
    return {
        "stage": stage,
        "current": current,
        "total": total,
        "unit": unit,
        "message": message,
    }


def _validate_overall(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict or not _exact_dict_keys(value, _OVERALL_FIELDS):
        raise ValueError("invalid progress-v3 overall")
    current = value["current"]
    total = value["total"]
    unit = value["unit"]
    label = value["label"]
    if not _int64_nonnegative(current):
        raise ValueError("invalid progress-v3 overall current")
    if total is not None and not _int64_nonnegative(total):
        raise ValueError("invalid progress-v3 overall total")
    if total is not None and current > total:
        raise ValueError("progress-v3 overall current exceeds total")
    if not _valid_text(unit, 32) or not _valid_text(label, 64, optional=True) or label == "":
        raise ValueError("invalid progress-v3 overall text")
    return {"current": current, "total": total, "unit": unit, "label": label}


def validate_payload_v3(value: Any) -> dict[str, Any]:
    """Validate and normalize one complete progress-v3 replacement payload."""
    if type(value) is not dict or not _exact_dict_keys(value, _PAYLOAD_FIELDS):
        raise ValueError("invalid progress-v3 fields")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != PROTOCOL_VERSION_V3:
        raise ValueError("unsupported progress protocol")
    update_id = value["update_id"]
    if not _valid_identifier(update_id):
        raise ValueError("invalid progress identity")
    activity = _validate_activity(value["activity"])
    overall = _validate_overall(value["overall"])
    metrics = _valid_metrics(value["metrics"])
    completeness = _validated_completeness(value["completeness"], metrics)
    result = {
        "protocol_version": PROTOCOL_VERSION_V3,
        "update_id": update_id,
        "activity": activity,
        "overall": overall,
        "metrics": metrics,
        "completeness": completeness,
    }
    encode_json(result, max_bytes=MAX_PAYLOAD_V3_BYTES)
    return result


def fit_payload_v3(
    *,
    update_id: str,
    stage: str,
    current: int | None,
    total: int | None,
    unit: str | None,
    message: str | None,
    overall: dict[str, Any] | None,
    metrics: dict[str, int | float],
    completeness: dict[str, Any],
) -> dict[str, Any] | None:
    """Trim captured metrics in reverse order if needed; return None if base cannot fit."""
    retained = dict(metrics)
    omitted = completeness["omitted_metrics"]
    reasons = list(completeness["reasons"])
    while True:
        payload = {
            "protocol_version": PROTOCOL_VERSION_V3,
            "update_id": update_id,
            "activity": {
                "stage": stage,
                "current": current,
                "total": total,
                "unit": unit,
                "message": message,
            },
            "overall": overall,
            "metrics": retained,
            "completeness": {
                "complete": completeness["complete"] and omitted == 0,
                "omitted_metrics": omitted,
                "reasons": reasons,
            },
        }
        try:
            return validate_payload_v3(payload)
        except ValueError as exc:
            if "byte limit" not in str(exc):
                raise
        if not retained:
            return None
        retained.popitem()
        omitted += 1
        if "size_limit" not in reasons:
            reasons.append("size_limit")


def read_payload_v3(path: Path) -> dict[str, Any]:
    """Read and strictly validate one bounded local progress-v3 payload."""
    value = read_advisory_snapshot(path, max_bytes=MAX_PAYLOAD_V3_BYTES)
    return validate_payload_v3(value)


def _nested_progress(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("progress")
    return nested if type(nested) is dict else payload


def semantic_key_v3(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Return a v3 content key for either a payload or a snapshot's nested progress."""
    nested = _nested_progress(payload)
    activity = nested.get("activity", {})
    overall = nested.get("overall")
    metrics = nested.get("metrics", {})
    completeness = nested.get("completeness", {})
    activity_key = tuple(activity.get(name) for name in ("stage", "current", "total", "unit", "message"))
    if type(overall) is dict:
        overall_key: Any = tuple(overall.get(name) for name in ("current", "total", "unit", "label"))
    else:
        overall_key = None
    metrics_key = tuple(sorted(metrics.items())) if type(metrics) is dict else None
    completeness_key = (
        completeness.get("complete"),
        completeness.get("omitted_metrics"),
        tuple(completeness.get("reasons", ())),
    )
    return activity_key, overall_key, metrics_key, completeness_key


__all__ = [
    "MAX_PAYLOAD_V3_BYTES",
    "MAX_SNAPSHOT_V3_BYTES",
    "PROTOCOL_VERSION_V3",
    "fit_payload_v3",
    "read_payload_v3",
    "semantic_key_v3",
    "validate_payload_v3",
]
