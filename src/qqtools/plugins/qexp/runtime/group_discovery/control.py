"""Bounded shared Group rechecks and exact control-locator retirement."""

from __future__ import annotations

import stat
from dataclasses import dataclass, field
from typing import Any

from ..locks import group_writer_lock, task_lock
from ..operation_store import iter_active_operation_paths
from ..paths import shared_paths
from ..records import validate_group_name
from ..store import read_json_limited
from . import locator
from .changes import settle_task_change
from .rechecks import GroupRechecks

_MAX_OPERATIONS_PER_SLICE = 64
_MAX_OPERATION_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class GroupControlCursor:
    """Disposable traversal progress, never evidence of shared authority."""

    locator_generation: int | None = None
    recheck_generation: int | None = None
    recheck_sequence: int = 1
    operation_witness: tuple[int, int, int, int] | None = None
    operation_cursor: dict[str, Any] = field(default_factory=dict)

    def reset_operations(self) -> None:
        self.operation_witness = None
        self.operation_cursor.clear()


def _operation_witness(cfg: Any) -> tuple[int, int, int, int] | None:
    directory = shared_paths(cfg.shared_root)["group_control_active"]
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("active Group operation storage must be a real directory")
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns


def _advance_operation_absence(cfg: Any, group: str, cursor: GroupControlCursor) -> bool:
    """Prove a whole unchanged directory sweep, not merely one empty page."""
    before = _operation_witness(cfg)
    if before != cursor.operation_witness:
        cursor.reset_operations()
        cursor.operation_witness = before
    previous_cycle = cursor.operation_cursor.get("cycle", 0)
    try:
        for path in iter_active_operation_paths(
            cfg,
            "group_control",
            limit=_MAX_OPERATIONS_PER_SLICE,
            include_legacy=False,
            cursor=cursor.operation_cursor,
            strict_active_files=True,
        ):
            operation = read_json_limited(path, max_bytes=_MAX_OPERATION_BYTES, record_type="group_control_operation")
            control = operation.get("group_control")
            if not isinstance(control, dict):
                raise ValueError("active Group operation is malformed")
            if control.get("group_name") == group and control.get("state") not in {"completed", "superseded"}:
                # A later page must not forget outstanding work in an earlier page.
                cursor.operation_cursor.clear()
                return False
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        # The iterator advances before yielding. Never retain proof past an
        # unreadable entry, otherwise retry could mistake its suffix for absence.
        cursor.reset_operations()
        raise
    after = _operation_witness(cfg)
    if before != after:
        cursor.reset_operations()
        return False
    return cursor.operation_cursor.get("cycle", 0) > previous_cycle


def advance_group_control_rechecks(cfg: Any, group: str, generation: int, cursor: GroupControlCursor) -> str:
    """Settle at most one recheck or scan one bounded operation page.

    This transaction has no MachineRuntime, process or resource effects. The
    caller separately advances Group cancellation/removal transactions. Only a
    stable complete active-operation sweep and exact current journal retirement
    permit deleting the captured locator generation. Restarting a cursor merely
    repeats shared proof; it cannot manufacture a quiescent result.

    Returns:
        ``progress``, ``blocked``, ``stale``, or ``quiescent``. Only quiescent
        means the exact locator was acknowledged under its Group writer fence.
    """
    validate_group_name(group)
    if type(generation) is not int or generation < 1:
        raise ValueError("control locator generation must be a positive integer")
    if not isinstance(cursor, GroupControlCursor):
        raise TypeError("control continuation must be a GroupControlCursor")
    with group_writer_lock(cfg, group, blocking=False) as acquired:
        if not acquired:
            return "blocked"
        current = locator.read_group_locator(cfg.shared_root, group, "control")
        if current is None or current["generation"] != generation:
            return "stale"
        if cursor.locator_generation != generation:
            cursor.locator_generation = generation
            cursor.reset_operations()
        journal = GroupRechecks(cfg.shared_root, group)
        position = journal.snapshot()
        if position is None:
            cursor.recheck_generation = None
            cursor.recheck_sequence = 1
        else:
            retention = journal.retention(position)
            if cursor.recheck_generation != position.generation:
                cursor.recheck_generation = position.generation
                cursor.recheck_sequence = retention.deleted + 1
                cursor.reset_operations()
            cursor.recheck_sequence = max(cursor.recheck_sequence, retention.deleted + 1)
            if cursor.recheck_sequence <= position.tail:
                event = journal.read(position, cursor.recheck_sequence)
                if event.get("state") == "in_flight":
                    task_id = event.get("task_id")
                    if not isinstance(task_id, str):
                        return "blocked"
                    with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
                        if not has_task_lock:
                            return "blocked"
                        settled, _results = settle_task_change(cfg, event)
                    if not settled:
                        return "blocked"
                cursor.recheck_sequence += 1
                return "progress"
        if not _advance_operation_absence(cfg, group, cursor):
            return "progress"

        def retirement_ready() -> bool:
            latest = journal.snapshot()
            if _operation_witness(cfg) != cursor.operation_witness:
                return False
            if position is None:
                return latest is None
            return (
                latest is not None
                and latest.generation == position.generation
                and latest.tail == position.tail
                and cursor.recheck_generation == latest.generation
                and cursor.recheck_sequence > latest.tail
            )

        if locator.acknowledge_group_locator_locked(
            cfg, group, "control", generation, retirement_ready=retirement_ready
        ):
            return "quiescent"
        cursor.reset_operations()
        return "stale"
