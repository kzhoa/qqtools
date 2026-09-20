"""Authoritative Group cancellation barrier checks for scheduling gates."""

from __future__ import annotations

from typing import Any

from ..config_types import RootConfig
from .operation_store import locate_operation_path
from .records import TaskRecord, validate_identifier
from .store import read_json

_ACTIVE_CANCEL_STATES = frozenset({"preparing", "converging", "waiting_ack", "blocked"})


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def has_active_cancellation(
    cfg: RootConfig,
    task: TaskRecord,
    group: dict[str, Any],
    *,
    include_default: bool = True,
) -> bool:
    """Return whether an applicable Group cancellation barrier blocks ``task``.

    The operation record is reread through its exact operation identifier for every
    applicable barrier; no Task or operation-history directory is enumerated. The
    hidden I/O cost is O(retained applicable barriers). A later consumer stage is
    responsible for bounding the retained barrier representation; this bug fix does
    not claim a full-scale gate. Callers must reread Group truth under the existing
    Group fence; this helper keeps no cached cancellation truth.

    A malformed applicable barrier or operation fails closed. When
    ``include_default`` is false, a validated nonterminating barrier is ignored,
    while malformed or unrecognized operation status still blocks.
    """
    if not isinstance(group, dict):
        return True
    group_record = group.get("group")
    if not isinstance(group_record, dict):
        return True
    barriers = group_record.get("cancellation_barriers", [])
    if not isinstance(barriers, list):
        return True
    if not barriers:
        return False

    if task.group_name is None:
        return False
    sequence = task.group_membership_sequence
    if not _is_nonnegative_int(sequence):
        return True
    group_name = task.group_name
    if not isinstance(group_name, str) or not group_name:
        return True
    record_name = group_record.get("name")
    if record_name is not None and record_name != group_name:
        return True

    for barrier in barriers:
        if not isinstance(barrier, dict):
            return True
        high_watermark = barrier.get("membership_high_watermark")
        if not _is_nonnegative_int(high_watermark):
            return True
        if sequence > high_watermark:
            continue

        operation_id = barrier.get("operation_id")
        try:
            validate_identifier(operation_id, "cancellation operation_id")
        except (TypeError, ValueError):
            return True
        terminate_running = barrier.get("terminate_running")
        if type(terminate_running) is not bool:
            return True

        try:
            operation = read_json(locate_operation_path(cfg, "group_control", operation_id))
            control = operation.get("group_control") if isinstance(operation, dict) else None
        except (KeyError, OSError, TypeError, ValueError):
            return True
        if not isinstance(control, dict):
            return True

        control_high_watermark = control.get("membership_high_watermark")
        control_terminate_running = control.get("terminate_running")
        if (
            control.get("operation_id") != operation_id
            or control.get("operation_type") != "cancel"
            or control.get("group_name") != group_name
            or not _is_nonnegative_int(control_high_watermark)
            or control_high_watermark != high_watermark
            or type(control_terminate_running) is not bool
            or control_terminate_running != terminate_running
        ):
            return True

        state = control.get("state")
        if not isinstance(state, str):
            return True
        if state == "completed":
            if isinstance(control.get("completed_at"), str) and control["completed_at"]:
                continue
            return True
        if state not in _ACTIVE_CANCEL_STATES:
            return True
        if not include_default and not terminate_running:
            continue
        return True

    return False
