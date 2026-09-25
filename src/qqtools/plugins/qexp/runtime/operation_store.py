"""Type-specific active durable-operation storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Literal

from .directory_capture import read_directory_entry
from .locks import schema_lock
from .paths import local_paths, shared_paths
from .store import atomic_replace, read_json

ActiveOperationKind = Literal["availability", "group_control", "cleanup"]


def _active_key(kind: ActiveOperationKind) -> str:
    return f"{kind}_active"


def active_operation_path(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
) -> Path:
    return shared_paths(cfg.shared_root)[_active_key(kind)] / f"{operation_key}.json"


def archived_operation_path(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
) -> Path:
    return shared_paths(cfg.shared_root)[kind] / f"{operation_key}.json"


def locate_operation_path(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
) -> Path:
    active = active_operation_path(cfg, kind, operation_key)
    return active if active.exists() else archived_operation_path(cfg, kind, operation_key)


def operation_exists(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
) -> bool:
    return (
        active_operation_path(cfg, kind, operation_key).exists()
        or archived_operation_path(cfg, kind, operation_key).exists()
    )


def write_active_operation(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
    value: dict,
) -> Path:
    # QQTOOLS-COMPAT-0020: operation-backed maintenance descriptors are prepared
    # before active operation truth and remain discoverable until handoff.
    from .maintenance_outbox import activate_work, prepare_work

    section = "availability_operation" if kind == "availability" else kind
    control = value.get(section, {}) if isinstance(value, dict) else {}
    operation_id = control.get("operation_id") if isinstance(control, dict) else None
    if not isinstance(operation_id, str) or not operation_id:
        operation_id = operation_key
    descriptor_kind = "group_cancel" if kind == "group_control" else kind
    descriptor = None
    if not _operation_is_terminal(kind, value):
        descriptor = prepare_work(
            cfg,
            kind=descriptor_kind,
            target_id=operation_id,
            work_generation=operation_id,
            phase="operation",
            cursor={"operation_key": operation_key},
        )
        if descriptor["state"] in {"completed", "intervention", "superseded"}:
            raise RuntimeError("terminal operation maintenance identity cannot be reopened.")
    path = active_operation_path(cfg, kind, operation_key)
    atomic_replace(path, value)
    stable = archived_operation_path(cfg, kind, operation_key)
    if not stable.exists() and not stable.is_symlink():
        try:
            stable.symlink_to(Path("active") / path.name)
        except FileExistsError:
            pass
    if descriptor is not None:
        activate_work(cfg, descriptor)
    return path


def archive_operation(
    cfg: object,
    kind: ActiveOperationKind,
    operation_key: str,
    value: dict,
) -> Path:
    """Publish terminal history before removing the active truth path."""
    archived = archived_operation_path(cfg, kind, operation_key)
    section = "availability_operation" if kind == "availability" else kind
    control = value.get(section, {}) if isinstance(value, dict) else {}
    operation_id = control.get("operation_id") if isinstance(control, dict) else None
    if not isinstance(operation_id, str) or not operation_id:
        operation_id = operation_key
    descriptor_kind = "group_cancel" if kind == "group_control" else kind
    from .maintenance_outbox import prepare_work, retire_work

    terminal = _operation_is_terminal(kind, value)

    prepare_work(
        cfg,
        kind=descriptor_kind,
        target_id=operation_id,
        work_generation=operation_id,
        phase="operation",
        cursor={"operation_key": operation_key},
    )
    atomic_replace(archived, value)
    active_operation_path(cfg, kind, operation_key).unlink(missing_ok=True)
    retire_work(
        cfg,
        kind=descriptor_kind,
        target_id=operation_id,
        work_generation=operation_id,
        state="completed" if terminal else "intervention",
        proof={
            "source": "archived_operation",
            "operation_key": operation_key,
            "state": _operation_state(kind, value),
        },
    )
    return archived


def iter_active_operation_paths(
    cfg: object,
    kind: ActiveOperationKind,
    *,
    limit: int = 64,
    include_legacy: bool = False,
    cursor: dict[str, Any] | None = None,
) -> Iterator[Path]:
    """Stream a bounded page of active truths and retained legacy operations."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("active operation limit must be a positive integer.")
    active = shared_paths(cfg.shared_root)[_active_key(kind)]
    archive = shared_paths(cfg.shared_root)[kind]
    cursor_path = local_paths(cfg.runtime_root)["maintenance_cursors"] / f"{kind}.json"
    if cursor is None:
        try:
            cursor_record = read_json(cursor_path).get("active_operation_cursor", {})
        except (FileNotFoundError, OSError, TypeError, ValueError):
            cursor_record = {}
    else:
        cursor_record = cursor
    if not isinstance(cursor_record, dict) or cursor_record.get("kind") not in {None, kind}:
        cursor_record = {}
    lane = cursor_record.get("lane", "active")
    if lane not in {"active", "legacy"}:
        lane = "active"
    active_offset = cursor_record.get("active_offset", 0)
    legacy_offset = cursor_record.get("legacy_offset", 0)
    cycle = cursor_record.get("cycle", 0)
    if type(active_offset) is not int or active_offset < 0:
        active_offset = 0
    if type(legacy_offset) is not int or legacy_offset < 0:
        legacy_offset = 0
    if type(cycle) is not int or cycle < 0:
        cycle = 0

    def save_cursor() -> None:
        value = {
            "kind": kind,
            "lane": lane,
            "active_offset": active_offset,
            "legacy_offset": legacy_offset,
            "cycle": cycle,
        }
        if cursor is None:
            atomic_replace(cursor_path, {"active_operation_cursor": value})
        else:
            cursor.clear()
            cursor.update(value)

    yielded = 0
    examined_entries = 0
    lane_transitions = 0
    while yielded < limit and examined_entries < limit:
        directory = active if lane == "active" else archive
        offset = active_offset if lane == "active" else legacy_offset
        if not directory.is_dir():
            name, next_offset = None, offset
        else:
            name, next_offset = read_directory_entry(directory, offset)
        if name is None:
            if lane == "active" and include_legacy:
                lane = "legacy"
                legacy_offset = 0
                save_cursor()
                lane_transitions += 1
                if lane_transitions <= 1:
                    continue
                break
            if lane == "legacy":
                lane = "active"
                active_offset = 0
                cycle += 1
                save_cursor()
                lane_transitions += 1
                if yielded == 0 and lane_transitions <= 2:
                    continue
                break
            active_offset = 0
            cycle += 1
            save_cursor()
            lane_transitions += 1
            if yielded == 0 and lane_transitions <= 1:
                continue
            break

        examined_entries += 1
        if lane == "active":
            active_offset = next_offset
        else:
            legacy_offset = next_offset
        save_cursor()
        if not name.endswith(".json"):
            continue
        path = directory / name
        if lane == "active":
            if path.is_file():
                yielded += 1
                yield path
            continue
        if path.is_symlink() or not path.is_file():
            continue
        # Stable aliases and interrupted-import copies refer to active truth.
        if (active / name).exists():
            continue
        try:
            record = read_json(path)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if _operation_is_terminal(kind, record):
            continue
        yielded += 1
        yield path


def _operation_state(kind: ActiveOperationKind, record: dict) -> object:
    key = "availability_operation" if kind == "availability" else kind
    return record.get(key, {}).get("state")


def _terminal_states(kind: ActiveOperationKind) -> frozenset[str]:
    if kind == "availability":
        return frozenset({"completed"})
    return frozenset({"completed"})


def _operation_is_terminal(kind: ActiveOperationKind, record: dict) -> bool:
    state = _operation_state(kind, record)
    if state in _terminal_states(kind):
        return True
    if (
        kind == "group_control"
        and record.get(kind, {}).get("operation_type") == "worker_remove"
        and state == "blocked"
        and record.get(kind, {}).get("blocked_reason") == "legacy_worker_incarnation_unknown"
    ):
        return True
    if kind == "group_control" and record.get(kind, {}).get("operation_type") == "worker_remove_v2":
        return state == "superseded"
    return False


def migrate_legacy_active_operations(cfg: object) -> None:
    """Perform the one-time schema-6 active-layout split under the schema lock."""
    migration = shared_paths(cfg.shared_root)["operations_migration"]
    if migration.exists():
        return
    with schema_lock(cfg.shared_root):
        if migration.exists():
            return
        for kind in ("availability", "group_control", "cleanup"):
            directory = shared_paths(cfg.shared_root)[kind]
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink() or not entry.is_file() or not entry.name.endswith(".json"):
                        continue
                    path = Path(entry.path)
                    try:
                        record = read_json(path)
                    except (OSError, KeyError, TypeError, ValueError):
                        continue
                    if _operation_is_terminal(kind, record):
                        continue
                    active = active_operation_path(cfg, kind, path.stem)
                    atomic_replace(active, record)
                    path.unlink(missing_ok=True)
        atomic_replace(migration, {"active_operations": {"version": 1}})
