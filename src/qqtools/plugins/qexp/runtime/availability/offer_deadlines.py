"""Durable offer-deadline projection maintenance."""
from __future__ import annotations

import heapq
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ...config_types import RootConfig
from ..locks import schema_lock
from ..paths import shared_paths
from ..records import TaskRecord
from ..store import atomic_replace, iter_json, read_json


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


def remove_deadline_index(cfg: RootConfig, task_id: str) -> None:
    stable = _deadline_index_path(cfg, task_id)
    if stable.exists() or stable.is_symlink():
        try:
            target = stable.resolve(strict=True)
        except FileNotFoundError:
            target = None
        stable.unlink(missing_ok=True)
        if target is not None:
            target.unlink(missing_ok=True)
            bucket = target.parent
            home = bucket.parent
            try:
                bucket.rmdir()
            except OSError:
                pass
            try:
                home.rmdir()
            except OSError:
                pass


def sync_deadline_index(cfg: RootConfig, task: TaskRecord) -> None:
    path = _deadline_index_path(cfg, task.task_id)
    if (task.placement_runtime.get("queue_scope") == "home"
            and task.placement_policy.get("sharing_mode") == "spillover"
            and task.placement_runtime.get("offer_eligible_at")
            and task.placement_runtime.get("offer_clock_evidence")):
        desired = {"offer_deadline": {"task_id": task.task_id,
            "group_name": task.group_name, "home_machine": task.placement_policy["home_machine"],
            "offer_eligible_at": task.placement_runtime["offer_eligible_at"],
            "operation_id": task.placement_runtime.get("availability_operation_id"),
            "updated_at": task.meta["updated_at"]}}
        active = _active_deadline_path(cfg, task)
        if path.exists():
            try:
                if read_json(path) == desired:
                    return
            except (KeyError, TypeError, ValueError):
                pass
        remove_deadline_index(cfg, task.task_id)
        atomic_replace(active, desired)
        try:
            path.symlink_to(active.relative_to(path.parent))
        except FileExistsError:
            pass
        return
    remove_deadline_index(cfg, task.task_id)


def iter_due_deadline_paths(cfg: RootConfig, *, limit: int = 64) -> Iterator[Path]:
    """Yield bounded due records for this home machine from time buckets only."""
    if limit <= 0:
        raise ValueError("deadline limit must be positive.")
    home = shared_paths(cfg.shared_root)["offer_deadlines_active"] / cfg.machine_name
    if not home.exists():
        return
    current_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    yielded = 0
    due_buckets = heapq.nsmallest(
        limit,
        (
            Path(entry.path)
            for entry in os.scandir(home)
            if entry.is_dir() and entry.name <= current_bucket
        ),
        key=lambda path: path.name,
    )
    for bucket in due_buckets:
        with os.scandir(bucket) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                if yielded >= limit:
                    return
                yielded += 1
                yield Path(entry.path)


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
            active = (
                shared_paths(cfg.shared_root)["offer_deadlines_active"]
                / home_machine
                / bucket
                / path.name
            )
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
            index_file.unlink(missing_ok=True)
            rebuilt += 1
    return rebuilt
