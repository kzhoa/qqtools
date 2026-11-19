"""Control-operation coverage for empty Groups created before service activation."""

from __future__ import annotations

import stat
from pathlib import Path

from .directory_capture import read_directory_entry
from .paths import shared_paths
from .records import validate_group_name, validate_identifier
from .responsibility_store import DurableIO
from .store import atomic_replace, check_mutation_fence


def _directory(root: Path, name: str) -> Path:
    return shared_paths(root)["indexes"] / "group-early-controls" / validate_group_name(name)


def begin_empty_group(root: Path, name: str) -> None:
    from .group_discovery.activation import is_group_service_active

    if is_group_service_active(root):
        return
    directory = _directory(root, name)
    check_mutation_fence(directory)
    directory.mkdir(parents=True, exist_ok=True)
    io = DurableIO()
    io.sync_directory(directory.parent, "group_control_coverage")
    io.sync_directory(directory.parent.parent, "group_control_coverage")


def prepare_control(root: Path, name: str, operation_id: str) -> None:
    directory = _directory(root, name)
    if directory.is_dir():
        path = directory / f"{validate_identifier(operation_id, 'operation_id')}.json"
        if not path.exists():
            atomic_replace(path, {"operation_id": operation_id})


def finish_control(root: Path, name: str, operation_id: str) -> None:
    path = _directory(root, name) / f"{validate_identifier(operation_id, 'operation_id')}.json"
    check_mutation_fence(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    DurableIO().sync_directory(path.parent, "group_control_coverage")


def has_empty_control_coverage(cfg: object, name: str) -> bool:
    from .maintenance_outbox import read_work
    from .operation_store import operation_exists

    directory = _directory(cfg.shared_root, name)
    try:
        if not stat.S_ISDIR(directory.lstat().st_mode):
            return False
    except FileNotFoundError:
        return False
    entry, _ = read_directory_entry(directory, 0)
    if entry is None:
        return True
    if not entry.endswith(".json"):
        return False
    operation_id = validate_identifier(entry[:-5], "operation_id")
    descriptor = read_work(cfg, kind="group_cancel", target_id=operation_id, work_generation=operation_id)
    if (descriptor is None and not operation_exists(cfg, "group_control", operation_id)) or (
        descriptor and descriptor["state"] in {"completed", "superseded"}
    ):
        finish_control(cfg.shared_root, name, operation_id)
    return False
