"""Shared validation for serialized ready-cursor positions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runtime.records import validate_identifier

_CURSOR_FIELDS = frozenset({"catalog_page", "partition", "after_name", "revision"})
_MAX_CATALOG_PAGE = 1_000_000_000_000_000
_MAX_CURSOR_TEXT = 256


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def validate_ready_cursor_position(
    value: object,
    label: str,
    *,
    require_safe_marker_name: bool = False,
    reject_marker_nul: bool = True,
) -> dict[str, Any]:
    """Validate and detach one persisted ready-cursor position.

    Primary-probe continuation historically requires a path-segment marker,
    while the Project I/O envelope accepts any bounded nonempty marker text.
    The explicit policy keeps those persisted-format contracts unchanged.
    """
    if not isinstance(value, Mapping) or set(value) != _CURSOR_FIELDS:
        raise ValueError(f"{label} has missing or unknown fields.")

    page = value["catalog_page"]
    if page is not None:
        page = _nonnegative_int(page, f"{label}.catalog_page")
        if page > _MAX_CATALOG_PAGE:
            raise ValueError(f"{label}.catalog_page is too large.")

    partition = value["partition"]
    if partition is not None:
        if not isinstance(partition, str) or not partition or len(partition) > _MAX_CURSOR_TEXT:
            raise ValueError(f"{label}.partition is invalid.")
        try:
            validate_identifier(partition, f"{label}.partition")
        except ValueError as exc:
            raise ValueError(f"{label}.partition is invalid.") from exc

    after_name = value["after_name"]
    if after_name is not None:
        is_invalid = (
            not isinstance(after_name, str)
            or not after_name
            or len(after_name) > _MAX_CURSOR_TEXT
            or (reject_marker_nul and "\x00" in after_name)
        )
        if require_safe_marker_name:
            is_invalid = is_invalid or "/" in after_name or "\\" in after_name or after_name in {".", ".."}
        if is_invalid:
            raise ValueError(f"{label}.after_name is invalid.")

    return {
        "catalog_page": page,
        "partition": partition,
        "after_name": after_name,
        "revision": _nonnegative_int(value["revision"], f"{label}.revision"),
    }


def validate_ready_cursor_transition(
    observed_value: object,
    next_value: object,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a monotonic observed-to-next ready-cursor transition."""
    observed = validate_ready_cursor_position(observed_value, f"{label}.observed")
    next_position = validate_ready_cursor_position(next_value, f"{label}.next")
    if next_position["revision"] < observed["revision"]:
        raise ValueError(f"{label}.next revision precedes observed revision.")
    position_fields = ("catalog_page", "partition", "after_name")
    if (
        any(next_position[field] != observed[field] for field in position_fields)
        and next_position["revision"] <= observed["revision"]
    ):
        raise ValueError(f"{label}.next position changed without advancing its revision.")
    return observed, next_position


__all__ = ["validate_ready_cursor_position", "validate_ready_cursor_transition"]
