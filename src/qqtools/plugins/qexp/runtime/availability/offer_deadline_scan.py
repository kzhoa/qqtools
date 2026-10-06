"""Bounded, fair discovery of due offer-deadline records."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ...config_types import RootConfig
from ..directory_capture import read_directory_entry
from ..paths import local_paths, shared_paths
from ..store import atomic_replace, read_json

_MAX_OFFSET = (1 << 63) - 1
_MAX_SCAN_ATTEMPTS = 64
_HOME_SCAN_QUOTA = 8
_MAX_EMPTY_SWEEP_WRAPS = 2
_CURSOR_VERSION = 3


def _root_cursor_path(cfg: RootConfig) -> Path:
    return local_paths(cfg.runtime_root)["maintenance_cursors"] / "offer_deadlines.json"


def _home_cursor_path(cfg: RootConfig, home_name: str) -> Path:
    return local_paths(cfg.runtime_root)["maintenance_cursors"] / "offer_deadline_homes" / f"{home_name}.json"


def _valid_offset(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_OFFSET


def _valid_identity(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int and 0 <= item <= (1 << 64) - 1 for item in value)
    )


def _directory_identity(metadata: os.stat_result) -> list[int]:
    return [int(metadata.st_dev), int(metadata.st_ino)]


def _load_root_cursor(cfg: RootConfig) -> dict[str, int]:
    try:
        value = read_json(_root_cursor_path(cfg)).get("offer_deadline_cursor", {})
        if (
            isinstance(value, dict)
            and set(value) == {"version", "home_offset"}
            and value.get("version") == _CURSOR_VERSION
            and _valid_offset(value.get("home_offset"))
        ):
            return {"home_offset": value["home_offset"]}
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass
    # QQTOOLS-COMPAT-0020: obsolete lexical and home-only cursors reset to a
    # bounded root sweep; no shared deadline index enumeration is required.
    return {"home_offset": 0}


def _save_root_cursor(cfg: RootConfig, cursor: dict[str, int]) -> None:
    atomic_replace(
        _root_cursor_path(cfg),
        {"offer_deadline_cursor": {"version": _CURSOR_VERSION, "home_offset": cursor["home_offset"]}},
    )


def _new_home_cursor(identity: list[int]) -> dict[str, Any]:
    return {
        "version": _CURSOR_VERSION,
        "bucket_offset": 0,
        "bucket_name": None,
        "entry_offset": 0,
        "home_identity": list(identity),
        "bucket_identity": None,
    }


def _valid_bucket_name(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 10 or not value.isascii() or not value.isdecimal():
        return False
    try:
        datetime.strptime(value, "%Y%m%d%H")
    except ValueError:
        return False
    return True


def _valid_home_cursor(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "bucket_offset",
        "bucket_name",
        "entry_offset",
        "home_identity",
        "bucket_identity",
    }:
        return False
    if (
        value.get("version") != _CURSOR_VERSION
        or not _valid_offset(value.get("bucket_offset"))
        or not _valid_offset(value.get("entry_offset"))
        or not _valid_identity(value.get("home_identity"))
    ):
        return False
    bucket_name = value.get("bucket_name")
    bucket_identity = value.get("bucket_identity")
    if bucket_name is None:
        return value["entry_offset"] == 0 and bucket_identity is None
    return _valid_bucket_name(bucket_name) and _valid_identity(bucket_identity)


def _load_home_cursor(cfg: RootConfig, home_name: str, identity: list[int]) -> dict[str, Any]:
    try:
        value = read_json(_home_cursor_path(cfg, home_name)).get("offer_deadline_home_cursor", {})
        if _valid_home_cursor(value) and value["home_identity"] == identity:
            return value
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass
    return _new_home_cursor(identity)


def _save_home_cursor(cfg: RootConfig, home_name: str, cursor: dict[str, Any]) -> None:
    atomic_replace(_home_cursor_path(cfg, home_name), {"offer_deadline_home_cursor": cursor})


def _real_directory_identity(path: Path, label: str) -> list[int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError(f"offer deadline {label} is a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    return _directory_identity(metadata)


def _due_bucket(name: str, current_bucket: str) -> bool:
    return _valid_bucket_name(name) and name <= current_bucket


def _reset_selected_bucket(cursor: dict[str, Any]) -> None:
    cursor.update(bucket_name=None, bucket_identity=None, entry_offset=0)


def _replace_home_cursor(cursor: dict[str, Any], identity: list[int]) -> None:
    cursor.clear()
    cursor.update(_new_home_cursor(identity))


def _scan_home(
    home: Path,
    cursor: dict[str, Any],
    *,
    current_bucket: str,
    budget: int,
) -> tuple[Path | None, int, bool]:
    """Scan one home turn and return candidate, budget, and home-presence state."""
    attempts = 0
    while attempts < _HOME_SCAN_QUOTA and budget > 0:
        identity = _real_directory_identity(home, "home")
        if identity is None:
            return None, budget, False
        if cursor["home_identity"] != identity:
            _replace_home_cursor(cursor, identity)

        bucket_name = cursor["bucket_name"]
        if bucket_name is None:
            try:
                name, next_offset = read_directory_entry(home, cursor["bucket_offset"])
            except FileNotFoundError:
                return None, budget - 1, False
            budget -= 1
            attempts += 1
            if name is None:
                cursor.update(bucket_offset=0, bucket_name=None, bucket_identity=None, entry_offset=0)
                return None, budget, True
            cursor["bucket_offset"] = next_offset
            bucket = home / name
            try:
                metadata = bucket.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError(f"offer deadline bucket is a symlink: {bucket}")
            if not stat.S_ISDIR(metadata.st_mode) or not _due_bucket(name, current_bucket):
                continue
            cursor.update(
                bucket_name=name,
                bucket_identity=_directory_identity(metadata),
                entry_offset=0,
            )
            continue

        if not _due_bucket(bucket_name, current_bucket):
            _reset_selected_bucket(cursor)
            continue
        bucket = home / bucket_name
        try:
            metadata = bucket.lstat()
        except FileNotFoundError:
            _reset_selected_bucket(cursor)
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"offer deadline bucket is a symlink: {bucket}")
        if not stat.S_ISDIR(metadata.st_mode):
            _reset_selected_bucket(cursor)
            continue
        identity = _directory_identity(metadata)
        if cursor["bucket_identity"] != identity:
            cursor["bucket_identity"] = identity
            cursor["entry_offset"] = 0
        try:
            name, next_offset = read_directory_entry(bucket, cursor["entry_offset"])
        except FileNotFoundError:
            _reset_selected_bucket(cursor)
            budget -= 1
            attempts += 1
            continue
        budget -= 1
        attempts += 1
        if name is None:
            _reset_selected_bucket(cursor)
            continue
        cursor["entry_offset"] = next_offset
        path = bucket / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"offer deadline entry is a symlink: {path}")
        if name.endswith(".json") and stat.S_ISREG(metadata.st_mode):
            return path, budget, True
    return None, budget, True


def iter_due_deadline_paths(cfg: RootConfig, *, limit: int = 64) -> Iterator[Path]:
    """Yield advisory due records across homes within bounded directory work."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("deadline limit must be positive.")
    active_root = shared_paths(cfg.shared_root)["offer_deadlines_active"]
    cursor = _load_root_cursor(cfg)
    budget = max(_MAX_SCAN_ATTEMPTS, limit * 8)
    current_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    yielded = 0
    wraps = 0
    while budget > 0 and yielded < limit:
        previous_offset = cursor["home_offset"]
        try:
            name, next_offset = read_directory_entry(active_root, previous_offset)
        except FileNotFoundError:
            cursor["home_offset"] = 0
            _save_root_cursor(cfg, cursor)
            return
        budget -= 1
        if name is None:
            cursor["home_offset"] = 0
            _save_root_cursor(cfg, cursor)
            if yielded or wraps >= _MAX_EMPTY_SWEEP_WRAPS:
                return
            wraps += 1
            continue
        home = active_root / name
        identity = _real_directory_identity(home, "home")
        if identity is None:
            cursor["home_offset"] = next_offset
            _save_root_cursor(cfg, cursor)
            continue
        if budget == 0:
            cursor["home_offset"] = previous_offset
            _save_root_cursor(cfg, cursor)
            return
        cursor["home_offset"] = next_offset
        _save_root_cursor(cfg, cursor)
        home_cursor = _load_home_cursor(cfg, name, identity)
        candidate, budget, home_present = _scan_home(
            home,
            home_cursor,
            current_bucket=current_bucket,
            budget=budget,
        )
        if home_present:
            _save_home_cursor(cfg, name, home_cursor)
        if candidate is None:
            continue
        _save_root_cursor(cfg, cursor)
        yielded += 1
        yield candidate


__all__ = ["iter_due_deadline_paths"]
