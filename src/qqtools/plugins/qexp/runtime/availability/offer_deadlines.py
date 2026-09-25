"""Durable offer-deadline projection maintenance."""

from __future__ import annotations

import heapq
import os
import stat
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ...config_types import RootConfig
from ..directory_capture import read_directory_entry
from ..locks import schema_lock
from ..paths import local_paths, shared_paths
from ..project_activation import project_activation_transaction
from ..records import TaskRecord
from ..store import atomic_replace, iter_json, read_json, read_json_limited

_MAX_REBUILD_TASK_BYTES = 1_048_576


def _deadline_index_path(cfg: RootConfig, task_id: str) -> Path:
    return shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task_id}.json"


def _deadline_bucket(value: str) -> str:
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return deadline.astimezone(timezone.utc).strftime("%Y%m%d%H")


def _active_deadline_path(cfg: RootConfig, task: TaskRecord) -> Path:
    bucket = _deadline_bucket(task.placement_runtime["offer_eligible_at"])
    return (
        shared_paths(cfg.shared_root)["offer_deadlines_active"]
        / task.placement_policy["home_machine"]
        / bucket
        / f"{task.task_id}.json"
    )


def _deadline_cursor_path(cfg: RootConfig) -> Path:
    return local_paths(cfg.runtime_root)["maintenance_cursors"] / "offer_deadlines.json"


def _load_deadline_cursor(cfg: RootConfig) -> dict[str, Any]:
    try:
        value = read_json(_deadline_cursor_path(cfg)).get("offer_deadline_cursor", {})
        if isinstance(value, dict) and value.get("version") == 2:
            cursor = {
                "bucket_offset": value.get("bucket_offset", 0),
                "bucket_name": value.get("bucket_name"),
                "entry_offset": value.get("entry_offset", 0),
            }
            if (
                type(cursor["bucket_offset"]) is int
                and cursor["bucket_offset"] >= 0
                and (cursor["bucket_name"] is None or isinstance(cursor["bucket_name"], str))
                and type(cursor["entry_offset"]) is int
                and cursor["entry_offset"] >= 0
            ):
                return cursor
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass
    # QQTOOLS-COMPAT-0020: accept the former lexical checkpoint and start a
    # bounded directory-cursor sweep rather than rebuilding it by enumeration.
    return {"bucket_offset": 0, "bucket_name": None, "entry_offset": 0}


def _save_deadline_cursor(cfg: RootConfig, cursor: dict[str, Any]) -> None:
    atomic_replace(
        _deadline_cursor_path(cfg),
        {
            "offer_deadline_cursor": {
                "version": 2,
                "bucket_offset": cursor["bucket_offset"],
                "bucket_name": cursor["bucket_name"],
                "entry_offset": cursor["entry_offset"],
            }
        },
    )


def _deadline_sort_key(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _iter_bucket_paths(buckets: list[Path]) -> Iterator[Path]:
    for bucket in buckets:
        with os.scandir(bucket) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    yield Path(entry.path)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory_chain(directory: Path, root: Path) -> None:
    directory.relative_to(root)
    current = directory
    while True:
        _sync_directory(current)
        if current == root:
            return
        current = current.parent


def remove_deadline_index(cfg: RootConfig, task_id: str, *, _publish_activation: bool = True) -> None:
    stable = _deadline_index_path(cfg, task_id)
    if stable.exists() or stable.is_symlink():
        descriptor = None
        if _publish_activation:
            from ..maintenance_outbox import activate_work, prepare_target_work, retire_work

            descriptor = prepare_target_work(
                cfg,
                kind="deadline_index",
                target_id=task_id,
                phase="remove",
            )
        activation = (
            project_activation_transaction(cfg, "offer_deadline_update") if _publish_activation else nullcontext()
        )
        with activation:
            try:
                target = stable.resolve(strict=True)
            except FileNotFoundError:
                target = None
            stable.unlink(missing_ok=True)
            if target is not None:
                target.unlink(missing_ok=True)
                bucket = target.parent
                home = bucket.parent
                _sync_directory(bucket)
                try:
                    bucket.rmdir()
                except OSError:
                    pass
                _sync_directory(home)
                try:
                    home.rmdir()
                except OSError:
                    pass
                _sync_directory(home.parent)
            _sync_directory_chain(stable.parent, cfg.shared_root)
        if descriptor is not None:
            handoff = activate_work(cfg, descriptor)
            retire_work(
                cfg,
                kind="deadline_index",
                target_id=task_id,
                work_generation=handoff["identity"]["work_generation"],
                proof={"source": "deadline_index_removed", "task_id": task_id},
            )


def sync_deadline_index(cfg: RootConfig, task: TaskRecord) -> None:
    path = _deadline_index_path(cfg, task.task_id)
    if (
        task.placement_runtime.get("queue_scope") == "home"
        and task.placement_policy.get("sharing_mode") == "spillover"
        and task.placement_runtime.get("offer_eligible_at")
        and task.placement_runtime.get("offer_clock_evidence")
    ):
        desired = {
            "offer_deadline": {
                "task_id": task.task_id,
                "group_name": task.group_name,
                "home_machine": task.placement_policy["home_machine"],
                "offer_eligible_at": task.placement_runtime["offer_eligible_at"],
                "operation_id": task.placement_runtime.get("availability_operation_id"),
                "updated_at": task.meta["updated_at"],
            }
        }
        active = _active_deadline_path(cfg, task)
        if path.exists():
            try:
                if read_json(path) == desired:
                    return
            except (KeyError, TypeError, ValueError):
                pass
        from ..maintenance_outbox import activate_work, prepare_target_work, retire_work

        descriptor = prepare_target_work(
            cfg,
            kind="deadline_index",
            target_id=task.task_id,
            phase="sync",
        )
        with project_activation_transaction(cfg, "offer_deadline_update"):
            remove_deadline_index(cfg, task.task_id, _publish_activation=False)
            atomic_replace(active, desired)
            try:
                path.symlink_to(active.relative_to(path.parent))
            except FileExistsError:
                pass
            _sync_directory_chain(active.parent, cfg.shared_root)
            _sync_directory_chain(path.parent, cfg.shared_root)
        handoff = activate_work(cfg, descriptor)
        retire_work(
            cfg,
            kind="deadline_index",
            target_id=task.task_id,
            work_generation=handoff["identity"]["work_generation"],
            proof={"source": "deadline_index_synced", "task_id": task.task_id},
        )
        return
    if path.exists() or path.is_symlink():
        from ..maintenance_outbox import activate_work, prepare_target_work, retire_work

        descriptor = prepare_target_work(
            cfg,
            kind="deadline_index",
            target_id=task.task_id,
            phase="remove",
        )
        remove_deadline_index(cfg, task.task_id, _publish_activation=False)
        handoff = activate_work(cfg, descriptor)
        retire_work(
            cfg,
            kind="deadline_index",
            target_id=task.task_id,
            work_generation=handoff["identity"]["work_generation"],
            proof={"source": "deadline_index_removed", "task_id": task.task_id},
        )


def iter_due_deadline_paths(cfg: RootConfig, *, limit: int = 64) -> Iterator[Path]:
    """Yield bounded due records for this home machine from time buckets only."""
    if limit <= 0:
        raise ValueError("deadline limit must be positive.")
    home = shared_paths(cfg.shared_root)["offer_deadlines_active"] / cfg.machine_name
    if not home.exists():
        return
    current_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    cursor = _load_deadline_cursor(cfg)
    yielded = 0
    scanned = 0
    wrapped_home = False
    max_scan = max(64, limit * 8)
    while scanned < max_scan and yielded < limit:
        bucket_name = cursor["bucket_name"]
        if bucket_name is None:
            try:
                name, next_offset = read_directory_entry(home, cursor["bucket_offset"])
            except FileNotFoundError:
                return
            if name is None:
                if wrapped_home or yielded:
                    _save_deadline_cursor(cfg, cursor)
                    return
                cursor.update({"bucket_offset": 0, "bucket_name": None, "entry_offset": 0})
                wrapped_home = True
                continue
            scanned += 1
            cursor["bucket_offset"] = next_offset
            bucket = home / name
            if len(name) != 10 or not name.isascii() or not name.isdecimal() or name > current_bucket:
                _save_deadline_cursor(cfg, cursor)
                continue
            try:
                metadata = bucket.lstat()
            except FileNotFoundError:
                _save_deadline_cursor(cfg, cursor)
                continue
            if stat.S_ISDIR(metadata.st_mode):
                cursor["bucket_name"] = name
                cursor["entry_offset"] = 0
            elif stat.S_ISLNK(metadata.st_mode):
                raise OSError(f"offer deadline bucket is a symlink: {bucket}")
            _save_deadline_cursor(cfg, cursor)
            continue

        bucket = home / bucket_name
        try:
            name, next_offset = read_directory_entry(bucket, cursor["entry_offset"])
        except FileNotFoundError:
            name = None
            next_offset = cursor["entry_offset"]
        if name is None:
            cursor["bucket_name"] = None
            cursor["entry_offset"] = 0
            _save_deadline_cursor(cfg, cursor)
            continue
        scanned += 1
        cursor["entry_offset"] = next_offset
        path = bucket / name
        if name.endswith(".json"):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                _save_deadline_cursor(cfg, cursor)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError(f"offer deadline entry is a symlink: {path}")
            _save_deadline_cursor(cfg, cursor)
            if stat.S_ISREG(metadata.st_mode):
                yielded += 1
                yield path
        else:
            _save_deadline_cursor(cfg, cursor)


def iter_flat_deadline_paths(cfg: RootConfig, *, limit: int = 64) -> Iterator[Path]:
    """Yield bounded legacy flat deadline records that still need reconciliation."""
    if limit <= 0:
        raise ValueError("deadline limit must be positive.")
    root = shared_paths(cfg.shared_root)["offer_deadlines"]
    if not root.exists():
        return
    paths = heapq.nsmallest(
        limit,
        (
            Path(entry.path)
            for entry in os.scandir(root)
            if not entry.is_symlink()
            and entry.is_file(follow_symlinks=False)
            and entry.name.endswith(".json")
            and entry.name != "layout-v1.json"
        ),
        key=lambda path: path.name,
    )
    yield from paths


def migrate_legacy_deadline_indexes(cfg: RootConfig) -> None:
    """Move legacy flat deadline records into home/time buckets once."""
    marker = shared_paths(cfg.shared_root)["offer_deadlines_migration"]
    if marker.exists():
        return
    with schema_lock(cfg.shared_root):
        if marker.exists():
            return
        root = shared_paths(cfg.shared_root)["offer_deadlines"]
        for path in root.iterdir():
            if not path.is_file() or path.is_symlink() or path.name == marker.name:
                continue
            try:
                record = read_json(path)["offer_deadline"]
                home_machine = record["home_machine"]
                bucket = _deadline_bucket(record["offer_eligible_at"])
            except (KeyError, TypeError, ValueError):
                path.unlink(missing_ok=True)
                continue
            active = shared_paths(cfg.shared_root)["offer_deadlines_active"] / home_machine / bucket / path.name
            atomic_replace(active, {"offer_deadline": record})
            path.unlink(missing_ok=True)
            path.symlink_to(active.relative_to(path.parent))
        atomic_replace(marker, {"offer_deadline_layout": {"version": 1}})


def rebuild_deadline_indexes(cfg: RootConfig) -> int:
    rebuilt = 0
    indexed: set[str] = set()
    for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(task_file))
        before = _deadline_index_path(cfg, task.task_id).exists()
        sync_deadline_index(cfg, task)
        after = _deadline_index_path(cfg, task.task_id).exists()
        if after:
            indexed.add(task.task_id)
        if before != after or after:
            rebuilt += 1
    for index_file in iter_json(shared_paths(cfg.shared_root)["offer_deadlines"]):
        if index_file.stem not in indexed:
            remove_deadline_index(cfg, index_file.stem)
            rebuilt += 1
    return rebuilt


def advance_deadline_index_rebuild_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one Task or flat-index entry in a resumable full rebuild."""
    current = {"side": "tasks", "offset": 0} if cursor is None else dict(cursor)
    if (
        set(current) != {"side", "offset"}
        or current.get("side") not in {"tasks", "indexes", "complete"}
        or type(current.get("offset")) is not int
        or current["offset"] < 0
    ):
        raise ValueError("deadline-index rebuild cursor is invalid.")
    if current["side"] == "complete":
        return {"state": "completed", "cursor": current, "processed": 0, "rebuilt": 0}

    paths = shared_paths(cfg.shared_root)
    source_name = "tasks" if current["side"] == "tasks" else "offer_deadlines"
    source = paths[source_name]
    name, next_offset = read_directory_entry(source, current["offset"])
    if name is None:
        if current["side"] == "tasks":
            return {
                "state": "building",
                "cursor": {"side": "indexes", "offset": 0},
                "processed": 0,
                "rebuilt": 0,
                "transitioned": True,
            }
        return {
            "state": "completed",
            "cursor": {"side": "complete", "offset": 0},
            "processed": 0,
            "rebuilt": 0,
            "transitioned": True,
        }

    next_cursor = {"side": current["side"], "offset": next_offset}
    if current["side"] == "tasks":
        if not name.endswith(".json"):
            return {"state": "building", "cursor": next_cursor, "processed": 0, "rebuilt": 0, "examined": True}
        task_pathname = source / name
        task = TaskRecord.from_dict(
            read_json_limited(
                task_pathname,
                max_bytes=_MAX_REBUILD_TASK_BYTES,
                record_type="deadline_rebuild_task",
            )
        )
        before = _deadline_index_path(cfg, task.task_id).exists()
        sync_deadline_index(cfg, task)
        after = _deadline_index_path(cfg, task.task_id).exists()
        return {
            "state": "building",
            "cursor": next_cursor,
            "processed": 1,
            "rebuilt": int(before != after or after),
            "task_id": task.task_id,
        }

    if name == "layout-v1.json" or not name.endswith(".json"):
        return {"state": "building", "cursor": next_cursor, "processed": 0, "rebuilt": 0, "examined": True}
    index_path = source / name
    if index_path.is_dir():
        return {"state": "building", "cursor": next_cursor, "processed": 0, "rebuilt": 0, "examined": True}
    task_id = index_path.stem
    if not paths["tasks"].joinpath(f"{task_id}.json").exists():
        remove_deadline_index(cfg, task_id)
        return {"state": "building", "cursor": next_cursor, "processed": 1, "rebuilt": 1, "task_id": task_id}
    return {"state": "building", "cursor": next_cursor, "processed": 1, "rebuilt": 0, "task_id": task_id}
