"""Pure validation and normalization for progress producer arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MAX_INT64 = 2**63 - 1


@dataclass(frozen=True, kw_only=True)
class Counter:
    """A producer-declared completed-work counter."""

    current: int
    unit: str
    total: int | None = None
    label: str | None = None


def _normalize_text(value: object) -> object:
    if isinstance(value, str):
        return str.__str__(value)
    return value


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


def _valid_int64(value: object, *, optional: bool = False) -> bool:
    return optional and value is None or type(value) is int and 0 <= value <= _MAX_INT64


def _normalize_counter(value: Counter | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "current": value.current,
        "total": value.total,
        "unit": _normalize_text(value.unit),
        "label": _normalize_text(value.label),
    }


def _validate_update(
    *,
    stage: object,
    current: object = None,
    total: object = None,
    unit: object = None,
    message: object = None,
    metrics: object = None,
    overall: object = None,
    normalize_activity: bool,
) -> tuple[str, ...]:
    del metrics
    if normalize_activity:
        stage = _normalize_text(stage)
        unit = _normalize_text(unit)
        message = _normalize_text(message)

    errors: list[str] = []
    if not _valid_text(stage, 64):
        errors.append("stage: must be a nonempty string of at most 64 UTF-8 bytes")
    if not _valid_int64(current, optional=True):
        errors.append("current: must be a nonnegative int64")
    if not _valid_int64(total, optional=True):
        errors.append("total: must be a nonnegative int64")
    if _valid_int64(current, optional=True) and _valid_int64(total, optional=True):
        if current is not None and total is not None and current > total:
            errors.append("current: must not exceed total")
    if not _valid_text(unit, 32, optional=True):
        errors.append("unit: must be None or a nonempty string of at most 32 UTF-8 bytes")
    if not _valid_text(message, 1024, optional=True):
        errors.append("message: must be None or a string of at most 1024 UTF-8 bytes")

    if overall is not None:
        if type(overall) is not Counter:
            errors.append("overall: must be a Counter or None")
        else:
            normalized = _normalize_counter(overall)
            assert normalized is not None
            overall_current = normalized["current"]
            overall_total = normalized["total"]
            overall_unit = normalized["unit"]
            overall_label = normalized["label"]
            if not _valid_int64(overall_current):
                errors.append("overall.current: must be a nonnegative int64")
            if not _valid_int64(overall_total, optional=True):
                errors.append("overall.total: must be None or a nonnegative int64")
            if _valid_int64(overall_current) and _valid_int64(overall_total, optional=True):
                if overall_total is not None and overall_current > overall_total:
                    errors.append("overall.current: must not exceed overall.total")
            if not _valid_text(overall_unit, 32):
                errors.append("overall.unit: must be a nonempty string of at most 32 UTF-8 bytes")
            if not _valid_text(overall_label, 64, optional=True) or overall_label == "":
                errors.append("overall.label: must be None or a nonempty string of at most 64 UTF-8 bytes")
    return tuple(errors)


def validate_update(
    *,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    message: str | None = None,
    metrics: object = None,
    overall: Counter | None = None,
) -> tuple[str, ...]:
    """Return bounded, field-qualified errors without reading runtime state."""
    return _validate_update(
        stage=stage,
        current=current,
        total=total,
        unit=unit,
        message=message,
        metrics=metrics,
        overall=overall,
        normalize_activity=True,
    )


def normalize_overall(value: Counter | None) -> dict[str, Any] | None:
    """Return the validated counter as plain JSON-compatible fields."""
    return _normalize_counter(value)


__all__ = ["Counter", "validate_update"]
