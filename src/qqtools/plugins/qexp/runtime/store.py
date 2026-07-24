"""Atomic JSON and revisioned persistence primitives."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CASConflict(RuntimeError):
    """Raised when a revisioned update lost a race."""


def atomic_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def create_if_absent(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise CASConflict(f"Record already exists: {path}") from exc
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def typed_load(path: Path, loader: Callable[[dict[str, Any]], T]) -> T:
    return loader(read_json(path))


def typed_save(path: Path, value: T, dumper: Callable[[T], dict[str, Any]]) -> None:
    atomic_replace(path, dumper(value))


def cas_update(path: Path, expected_revision: int, value: dict[str, Any]) -> None:
    current = read_json(path)
    actual = current.get("meta", {}).get("revision")
    if actual != expected_revision:
        raise CASConflict(f"Revision conflict at {path}: expected {expected_revision}, got {actual}.")
    value.setdefault("meta", {})["revision"] = expected_revision + 1
    atomic_replace(path, value)


def iter_json(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())
