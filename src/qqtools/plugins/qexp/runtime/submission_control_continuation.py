"""Closed, disposable traversal state for Submission-control maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def submission_control_continuation(value: object = None) -> dict[str, Any]:
    if value is None:
        return {"pending_lane": True, "pending_offset": 0, "pending_inode": None, "pending_had_work": False}
    if not isinstance(value, Mapping) or set(value) != {
        "pending_lane",
        "pending_offset",
        "pending_inode",
        "pending_had_work",
    }:
        raise ValueError("Submission-control continuation fields are invalid.")
    for field in ("pending_lane", "pending_had_work"):
        if type(value[field]) is not bool:
            raise ValueError(f"Submission-control {field} must be boolean.")
    if type(value["pending_offset"]) is not int or not 0 <= value["pending_offset"] <= (1 << 63) - 1:
        raise ValueError("Submission-control pending offset is invalid.")
    inode = value["pending_inode"]
    if inode is not None and (type(inode) is not int or not 0 <= inode <= (1 << 64) - 1):
        raise ValueError("Submission-control pending inode is invalid.")
    if value["pending_offset"] and (inode is None or not value["pending_had_work"]):
        raise ValueError("Submission-control pending cursor lacks sweep provenance.")
    return dict(value)
