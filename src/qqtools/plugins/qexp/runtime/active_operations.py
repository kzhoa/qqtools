"""Type-specific active durable-operation storage."""

from __future__ import annotations

import os
import heapq
from pathlib import Path
from typing import Iterator, Literal

from .paths import local_paths, shared_paths
from .store import atomic_replace, read_json
from .locks import schema_lock

ActiveOperationKind = Literal["availability", "group_control", "cleanup"]


def _active_key(kind: ActiveOperationKind) -> str:
    return f"{kind}_active"


def active_operation_path(
    cfg: object, kind: ActiveOperationKind, operation_key: str,
) -> Path:
    return shared_paths(cfg.shared_root)[_active_key(kind)] / f"{operation_key}.json"


def archived_operation_path(
    cfg: object, kind: ActiveOperationKind, operation_key: str,
) -> Path:
    return shared_paths(cfg.shared_root)[kind] / f"{operation_key}.json"


def locate_operation_path(
    cfg: object, kind: ActiveOperationKind, operation_key: str,
) -> Path:
    active = active_operation_path(cfg, kind, operation_key)
    return active if active.exists() else archived_operation_path(cfg, kind, operation_key)


def operation_exists(
    cfg: object, kind: ActiveOperationKind, operation_key: str,
) -> bool:
    return (
        active_operation_path(cfg, kind, operation_key).exists()
        or archived_operation_path(cfg, kind, operation_key).exists()
    )


def write_active_operation(
    cfg: object, kind: ActiveOperationKind, operation_key: str, value: dict,
) -> Path:
    path = active_operation_path(cfg, kind, operation_key)
    atomic_replace(path, value)
    stable = archived_operation_path(cfg, kind, operation_key)
    if not stable.exists() and not stable.is_symlink():
        try:
            stable.symlink_to(Path("active") / path.name)
        except FileExistsError:
            pass
    return path


def archive_operation(
    cfg: object, kind: ActiveOperationKind, operation_key: str, value: dict,
) -> Path:
    """Publish terminal history before removing the active truth path."""
    archived = archived_operation_path(cfg, kind, operation_key)
    atomic_replace(archived, value)
    active_operation_path(cfg, kind, operation_key).unlink(missing_ok=True)
    return archived


def iter_active_operation_paths(
    cfg: object,
    kind: ActiveOperationKind,
    *,
    limit: int = 64,
    include_legacy: bool = False,
) -> Iterator[Path]:
    """Stream at most ``limit`` active truths without materializing operation history."""
    if limit <= 0:
        raise ValueError("active operation limit must be positive.")
    yielded = 0
    active = shared_paths(cfg.shared_root)[_active_key(kind)]
    if active.exists():
        cursor_path = local_paths(cfg.runtime_root)["maintenance_cursors"] / f"{kind}.json"
        after_name = None
        if cursor_path.exists():
            try:
                after_name = read_json(cursor_path)["active_operation_cursor"].get(
                    "after_name"
                )
            except (OSError, KeyError, TypeError, ValueError):
                after_name = None
        with os.scandir(active) as entries:
            selected = heapq.nsmallest(
                limit,
                (
                    Path(entry.path)
                    for entry in entries
                    if entry.is_file()
                    and entry.name.endswith(".json")
                    and (after_name is None or entry.name > after_name)
                ),
                key=lambda path: path.name,
            )
        if not selected and after_name is not None:
            with os.scandir(active) as entries:
                selected = heapq.nsmallest(
                    limit,
                    (
                        Path(entry.path)
                        for entry in entries
                        if entry.is_file() and entry.name.endswith(".json")
                    ),
                    key=lambda path: path.name,
                )
        if selected:
            atomic_replace(cursor_path, {"active_operation_cursor": {
                "kind": kind,
                "after_name": selected[-1].name,
            }})
        for path in selected:
            yielded += 1
            yield path
    if not include_legacy or yielded >= limit:
        return
    archive = shared_paths(cfg.shared_root)[kind]
    with os.scandir(archive) as entries:
        for entry in entries:
            if yielded >= limit:
                return
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            path = Path(entry.path)
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
    if kind == "availability":
        return (
            state == "blocked"
            and bool(record.get("availability_operation", {}).get("blocked_reason"))
        )
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
                    if (
                        entry.is_symlink()
                        or not entry.is_file()
                        or not entry.name.endswith(".json")
                    ):
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
