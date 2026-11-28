"""Shared scalar and key validation for Project I/O records."""

from typing import Any, Mapping

from ....runtime.records import validate_identifier


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has missing or unknown fields.")


def _require_text(value: object, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is invalid.")
    return value


def _require_identifier(value: object, label: str, *, maximum: int = 128) -> str:
    text = _require_text(value, label, maximum=maximum)
    try:
        validate_identifier(text, label)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid.") from exc
    return text
