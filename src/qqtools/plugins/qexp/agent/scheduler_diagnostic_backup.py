"""Verified backups of the bounded machine-local scheduler diagnostic store."""

from __future__ import annotations

import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_ATOMIC_TEMP_NAME = re.compile(r"\.(.+\.json)\.[a-z0-9_]{8}")
FIXED_DIAGNOSTIC_FILES = {
    Path("metadata.json"): 64 * 1024,
    Path("summary.json"): 256 * 1024,
    Path("history/index.json"): 256 * 1024,
}
MAX_DIAGNOSTIC_TEMPORARIES = 16


def temporary_target(name: str) -> str | None:
    match = _ATOMIC_TEMP_NAME.fullmatch(name)
    return match[1] if match else None


def _read_backup_file(directory_fd: int, name: str, limit: int) -> bytes:
    with os.fdopen(os.open(name, _FILE_FLAGS, dir_fd=directory_fd), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise ValueError("scheduler diagnostic backup file is invalid or oversized")
        data = handle.read(limit + 1)
        if len(data) != info.st_size:
            raise ValueError("scheduler diagnostic backup source changed")
        return data


def create_scheduler_diagnostic_backup(
    root: Path,
    *,
    max_bytes: int,
    record_limits: Mapping[Path, tuple[re.Pattern[str], int, int]],
) -> Path:
    """Copy and verify the scheduler-v1 layout while its writer lock is held."""
    directories = {Path("history"), *record_limits}
    total = 0
    temporaries = 0

    def copy_directory(source_fd: int, relative: Path, destination: Path) -> None:
        nonlocal total, temporaries
        expected = {path.name for path in (*FIXED_DIAGNOSTIC_FILES, *directories) if path.parent == relative}
        pattern, max_records, max_record_bytes = record_limits.get(relative, (None, 0, 0))
        seen_records = 0
        with os.scandir(source_fd) as entries:
            for entry in entries:
                name = entry.name
                path = relative / name
                temporary = temporary_target(name)
                limit_path = path
                if name in expected:
                    expected.remove(name)
                elif temporary is not None and (
                    relative / temporary in FIXED_DIAGNOSTIC_FILES
                    or (pattern is not None and pattern.fullmatch(temporary) is not None)
                ):
                    temporaries += 1
                    if temporaries > MAX_DIAGNOSTIC_TEMPORARIES:
                        raise ValueError("too many scheduler diagnostic atomic temporaries")
                    limit_path = relative / temporary
                elif pattern is None or pattern.fullmatch(name) is None or seen_records >= max_records:
                    raise ValueError("unexpected scheduler diagnostic backup entry")
                else:
                    seen_records += 1

                target = destination / name
                if path in directories:
                    if not entry.is_dir(follow_symlinks=False):
                        raise ValueError("scheduler diagnostic backup directory is invalid")
                    target.mkdir(mode=0o700)
                    child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=source_fd)
                    try:
                        copy_directory(child_fd, path, target)
                    finally:
                        os.close(child_fd)
                    continue

                if not entry.is_file(follow_symlinks=False):
                    raise ValueError("scheduler diagnostic backup file is invalid")
                limit = min(FIXED_DIAGNOSTIC_FILES.get(limit_path, max_record_bytes), max_bytes - total)
                data = _read_backup_file(source_fd, name, limit)
                target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(target_fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                with target.open("rb") as handle:
                    if handle.read(len(data) + 1) != data or _read_backup_file(source_fd, name, limit) != data:
                        raise ValueError("scheduler diagnostic backup verification failed")
                total += len(data)
        if expected:
            raise ValueError("scheduler diagnostic backup source is incomplete")
        destination_fd = os.open(destination, _DIRECTORY_FLAGS)
        try:
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)

    parent_fd = os.open(root.parent, _DIRECTORY_FLAGS)
    try:
        source_fd = os.open(root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            with tempfile.TemporaryDirectory(prefix=".scheduler-v1-backup-", dir=root.parent) as temporary:
                source_copy = Path(temporary)
                copy_directory(source_fd, Path(), source_copy)
                backup = root.parent / f"scheduler-v1-backup-{uuid.uuid4().hex}"
                os.rename(source_copy.name, backup.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
                return backup
        finally:
            os.close(source_fd)
    finally:
        os.close(parent_fd)
