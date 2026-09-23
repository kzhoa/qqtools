"""Bounded progress-v2 payload validation and advisory mailbox reads."""

from __future__ import annotations

import math
import re
from itertools import islice
from pathlib import Path
from typing import Any

from ._progress_protocol import MAX_SNAPSHOT_BYTES, encode_json, read_advisory_snapshot

PROTOCOL_VERSION_V2 = 2
MAX_PAYLOAD_V2_BYTES = 8192
MAX_SNAPSHOT_V2_BYTES = MAX_SNAPSHOT_BYTES
MAX_METRICS = 32
_PAYLOAD_FIELDS = frozenset(
    (
        "protocol_version",
        "update_id",
        "stage",
        "current",
        "total",
        "unit",
        "message",
        "metrics",
        "completeness",
    )
)
_COMPLETENESS_FIELDS = frozenset(("complete", "omitted_metrics", "reasons"))
_METRIC_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{0,63}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_REASONS = frozenset(("invalid_metrics", "metric_limit", "size_limit"))


def _exact_dict_keys(value: dict[str, Any], expected: frozenset[str]) -> bool:
    if len(value) != len(expected):
        return False
    found: set[str] = set()
    for key in dict.__iter__(value):
        if type(key) is not str or key not in expected:
            return False
        found.add(key)
    return found == expected


def _int64_nonnegative(value: object) -> bool:
    if type(value) is not int or value < 0 or value.bit_length() > 63:
        return False
    return True


def _valid_identifier(value: object) -> bool:
    if type(value) is not str or value in {".", ".."}:
        return False
    return _IDENTIFIER.fullmatch(value) is not None


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


def _valid_metrics(value: object) -> dict[str, int | float]:
    if type(value) is not dict or len(value) > MAX_METRICS:
        raise ValueError("invalid progress metrics")
    result: dict[str, int | float] = {}
    for name, metric in dict.items(value):
        if type(name) is not str or len(name) > 64 or _METRIC_NAME.fullmatch(name) is None:
            raise ValueError("invalid progress metric name")
        if type(metric) is int:
            if metric.bit_length() > 64 or metric < -(2**63) or metric > 2**63 - 1:
                raise ValueError("invalid progress metric value")
        elif type(metric) is float:
            if not math.isfinite(metric):
                raise ValueError("invalid progress metric value")
        else:
            raise ValueError("invalid progress metric value")
        result[name] = metric
    return result


def _validated_completeness(value: object, metrics: dict[str, int | float]) -> dict[str, Any]:
    if type(value) is not dict or not _exact_dict_keys(value, _COMPLETENESS_FIELDS):
        raise ValueError("invalid progress completeness")
    complete = value["complete"]
    omitted = value["omitted_metrics"]
    reasons = value["reasons"]
    if type(complete) is not bool:
        raise ValueError("invalid progress completeness flag")
    if omitted is not None and not _int64_nonnegative(omitted):
        raise ValueError("invalid omitted metric count")
    if type(reasons) is not list or len(reasons) > len(_REASONS):
        raise ValueError("invalid progress completeness reasons")
    normalized_reasons: list[str] = []
    for reason in reasons:
        if type(reason) is not str or reason not in _REASONS or reason in normalized_reasons:
            raise ValueError("invalid progress completeness reason")
        normalized_reasons.append(reason)
    if complete:
        if omitted != 0 or normalized_reasons:
            raise ValueError("inconsistent complete progress metrics")
    elif not normalized_reasons or (omitted is not None and omitted == 0):
        raise ValueError("inconsistent incomplete progress metrics")
    if omitted is None:
        if normalized_reasons != ["invalid_metrics"] or metrics:
            raise ValueError("unknown omitted metric count requires invalid metrics")
    if any(reason in normalized_reasons for reason in ("metric_limit", "size_limit")) and (
        omitted is None or omitted == 0
    ):
        raise ValueError("truncated metrics require a known positive omission count")
    return {"complete": complete, "omitted_metrics": omitted, "reasons": normalized_reasons}


def validate_payload_v2(value: Any) -> dict[str, Any]:
    """Validate and normalize one complete progress-v2 replacement payload."""
    if type(value) is not dict or not _exact_dict_keys(value, _PAYLOAD_FIELDS):
        raise ValueError("invalid progress-v2 fields")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != PROTOCOL_VERSION_V2:
        raise ValueError("unsupported progress protocol")
    update_id = value["update_id"]
    if not _valid_identifier(update_id):
        raise ValueError("invalid progress identity")
    stage = value["stage"]
    unit = value["unit"]
    message = value["message"]
    if (
        not _valid_text(stage, 64)
        or not _valid_text(unit, 32, optional=True)
        or not _valid_text(message, 1024, optional=True)
    ):
        raise ValueError("invalid progress text field")
    current = value["current"]
    total = value["total"]
    for name, number in (("current", current), ("total", total)):
        if number is not None and not _int64_nonnegative(number):
            raise ValueError(f"invalid progress {name}")
    if current is not None and total is not None and current > total:
        raise ValueError("progress current exceeds total")
    metrics = _valid_metrics(value["metrics"])
    completeness = _validated_completeness(value["completeness"], metrics)
    result = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "update_id": update_id,
        "stage": stage,
        "current": current,
        "total": total,
        "unit": unit,
        "message": message,
        "metrics": metrics,
        "completeness": completeness,
    }
    encode_json(result, max_bytes=MAX_PAYLOAD_V2_BYTES)
    return result


def capture_metrics(metrics: object) -> tuple[dict[str, int | float], dict[str, Any]]:
    """Copy bounded built-in scalar metrics and describe omitted input entries."""
    if metrics is None:
        return {}, {"complete": True, "omitted_metrics": 0, "reasons": []}
    if type(metrics) is not dict:
        return {}, {"complete": False, "omitted_metrics": None, "reasons": ["invalid_metrics"]}

    initial_size = dict.__len__(metrics)
    captured: dict[str, int | float] = {}
    invalid_metrics = False
    try:
        for name, metric in islice(dict.items(metrics), MAX_METRICS):
            if type(name) is not str or len(name) > 64 or _METRIC_NAME.fullmatch(name) is None:
                invalid_metrics = True
                continue
            if type(metric) is int:
                if metric.bit_length() > 64 or metric < -(2**63) or metric > 2**63 - 1:
                    invalid_metrics = True
                    continue
            elif type(metric) is float:
                if not math.isfinite(metric):
                    invalid_metrics = True
                    continue
            else:
                invalid_metrics = True
                continue
            captured[name] = metric
        if dict.__len__(metrics) != initial_size:
            raise RuntimeError("metrics dict changed size during capture")
    except RuntimeError as exc:
        raise RuntimeError("metrics dict changed during capture") from exc

    omitted = initial_size - len(captured)
    reasons: list[str] = []
    if invalid_metrics:
        reasons.append("invalid_metrics")
    if initial_size > MAX_METRICS:
        reasons.append("metric_limit")
    return captured, {"complete": omitted == 0, "omitted_metrics": omitted, "reasons": reasons}


def fit_payload_v2(
    *,
    update_id: str,
    stage: str,
    current: int | None,
    total: int | None,
    unit: str | None,
    message: str | None,
    metrics: dict[str, int | float],
    completeness: dict[str, Any],
) -> dict[str, Any] | None:
    """Trim captured metrics in reverse order if needed; return None if base cannot fit."""
    retained = dict(metrics)
    omitted = completeness["omitted_metrics"]
    reasons = list(completeness["reasons"])
    while True:
        payload = {
            "protocol_version": PROTOCOL_VERSION_V2,
            "update_id": update_id,
            "stage": stage,
            "current": current,
            "total": total,
            "unit": unit,
            "message": message,
            "metrics": retained,
            "completeness": {
                "complete": completeness["complete"] and omitted == 0,
                "omitted_metrics": omitted,
                "reasons": reasons,
            },
        }
        try:
            return validate_payload_v2(payload)
        except ValueError as exc:
            if "byte limit" not in str(exc):
                raise
        if not retained:
            return None
        retained.popitem()
        omitted += 1
        if "size_limit" not in reasons:
            reasons.append("size_limit")


def read_payload_v2(path: Path) -> dict[str, Any]:
    """Read and strictly validate one bounded local progress-v2 payload."""
    value = read_advisory_snapshot(path, max_bytes=MAX_PAYLOAD_V2_BYTES)
    return validate_payload_v2(value)


def semantic_key_v2(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Return the v2 content key, including metrics and their completeness."""
    return (
        payload.get("stage"),
        payload.get("current"),
        payload.get("total"),
        payload.get("unit"),
        payload.get("message"),
        tuple(sorted(payload.get("metrics", {}).items())),
        (
            payload.get("completeness", {}).get("complete"),
            payload.get("completeness", {}).get("omitted_metrics"),
            tuple(payload.get("completeness", {}).get("reasons", ())),
        ),
    )
