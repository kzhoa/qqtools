"""Bounded retained source enumeration with closed, descriptor-free cursors."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any, Mapping

from .directory_capture import read_directory_entry
from .paths import local_paths
from .responsibility_backfill import BATCH_SIZE, LANES, _relative_record
from .responsibility_source_read import SourceCaptureStale, SourceReadContext, validate_source_hold
from .responsibility_store import Unavailable


def source_scan_cursor(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a cursor without resolving or accessing any filesystem path."""
    if not isinstance(value, Mapping) or set(value) != {
        "directory_identity",
        "offset",
        "child",
        "child_identity",
        "child_offset",
    }:
        raise ValueError("source discovery cursor fields are invalid")
    for key in ("offset", "child_offset"):
        if type(value[key]) is not int or not 0 <= value[key] <= (1 << 63) - 1:
            raise ValueError("source discovery cursor offset is invalid")
    for key in ("directory_identity", "child_identity"):
        identity = value[key]
        if identity is not None and (
            not isinstance(identity, (list, tuple))
            or len(identity) != 2
            or any(type(number) is not int or not 0 <= number <= (1 << 64) - 1 for number in identity)
        ):
            raise ValueError("source discovery directory identity is invalid")
    child = value["child"]
    if child is not None and (
        not isinstance(child, str)
        or not child
        or child in {".", ".."}
        or "/" in child
        or "\\" in child
        or "\x00" in child
        or len(child.encode()) > 255
    ):
        raise ValueError("source discovery child name is invalid")
    if (
        (value["directory_identity"] is None and (value["offset"] or child is not None))
        or (child is None and (value["child_identity"] is not None or value["child_offset"]))
        or (child is not None and value["child_identity"] is None)
    ):
        raise ValueError("source discovery cursor is inconsistent")
    return {
        **value,
        "directory_identity": None if value["directory_identity"] is None else list(value["directory_identity"]),
        "child_identity": None if value["child_identity"] is None else list(value["child_identity"]),
    }


def initial_source_scan_cursor() -> dict[str, Any]:
    return {"directory_identity": None, "offset": 0, "child": None, "child_identity": None, "child_offset": 0}


def source_scan_result(value: Mapping[str, Any], lane: str) -> dict[str, Any]:
    """Validate bounded enumeration evidence at transport and local application."""
    if (
        lane not in LANES
        or not isinstance(value, Mapping)
        or set(value) != {"pending", "cursor", "at_end", "entries_visited"}
    ):
        raise ValueError("source discovery evidence fields are invalid")
    pending, visits = value["pending"], value["entries_visited"]
    if (
        not isinstance(pending, (list, tuple))
        or type(visits) is not int
        or not 0 <= visits <= BATCH_SIZE
        or len(pending) > visits
        or type(value["at_end"]) is not bool
    ):
        raise ValueError("source discovery work bounds are invalid")
    for relative in pending:
        if not isinstance(relative, str) or len(relative.encode()) > 512:
            raise ValueError("source discovery path is invalid")
        try:
            _relative_record(lane, relative)
        except Unavailable as exc:
            raise ValueError("source discovery path is invalid") from exc
    if len(set(pending)) != len(pending):
        raise ValueError("source discovery returned duplicate paths")
    cursor = source_scan_cursor(value["cursor"])
    if value["at_end"] and cursor["child"] is not None:
        raise ValueError("source discovery cannot finish while a child remains")
    return {**value, "pending": list(pending), "cursor": cursor}


def _directory_identity(path: Path) -> list[int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise Unavailable("source discovery requires a real directory")
    return [metadata.st_dev, metadata.st_ino]


def scan_retained_source(
    context: SourceReadContext, lane: str, cursor: Mapping[str, Any], *, limit: int = BATCH_SIZE
) -> dict[str, Any]:
    """Visit at most 64 names; return hints, never modify source or target.

    The local owner journals returned paths before any record read. Directory
    replacement cannot silently advance a cookie into a different namespace.
    Retention preserves records while live captured writers remain discoverable.
    """
    if lane not in LANES or type(limit) is not int or not 1 <= limit <= BATCH_SIZE:
        raise ValueError("invalid source discovery lane or visit limit")
    current = source_scan_cursor(cursor)
    validate_source_hold(context)
    directory = local_paths(context.source_root)[lane]
    try:
        identity = _directory_identity(directory)
    except FileNotFoundError:
        if current["directory_identity"] is not None:
            raise SourceCaptureStale("source discovery directory disappeared") from None
        return {"pending": [], "cursor": current, "at_end": True, "entries_visited": 0}
    if current["directory_identity"] is not None and current["directory_identity"] != identity:
        raise SourceCaptureStale("source discovery directory was replaced")
    current["directory_identity"] = identity
    pending = []
    at_end = False
    visited = 0
    for _ in range(limit):
        child = current["child"]
        parent = directory if child is None else directory / child
        if child is not None and _directory_identity(parent) != current["child_identity"]:
            raise SourceCaptureStale("source discovery child directory was replaced")
        offset_key = "offset" if child is None else "child_offset"
        name, offset = read_directory_entry(parent, current[offset_key])
        current[offset_key] = offset
        if name is None:
            if child is None:
                at_end = True
                break
            current.update(child=None, child_identity=None, child_offset=0)
            # Child EOF consumes a work unit too; a forest of empty directories
            # cannot turn one bounded request into an unbounded traversal.
            visited += 1
            continue
        visited += 1
        path = parent / name
        try:
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise Unavailable("source discovery cannot traverse a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if child is not None:
                raise Unavailable("source evidence directory exceeds the supported path depth")
            current.update(child=name, child_identity=[metadata.st_dev, metadata.st_ino], child_offset=0)
        elif name.endswith(".json"):
            if not stat.S_ISREG(metadata.st_mode):
                raise Unavailable("source discovery evidence is not a regular file")
            relative = str(path.relative_to(directory))
            _relative_record(lane, relative)
            pending.append(relative)
    if _directory_identity(directory) != identity:
        raise SourceCaptureStale("source discovery directory changed during enumeration")
    validate_source_hold(context)
    return {"pending": pending, "cursor": current, "at_end": at_end, "entries_visited": visited}
