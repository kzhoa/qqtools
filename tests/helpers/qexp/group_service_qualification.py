"""Isolated Group service scale and resource qualification helpers."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from qqtools.plugins.qexp.runtime.group_discovery import locator
from qqtools.plugins.qexp.runtime.group_namespace import group_directory
from qqtools.plugins.qexp.runtime.records import new_group


def populate_settled_group_history(cfg: Any, count: int) -> None:
    """Create valid, obligation-free Group truth outside a measured interval."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    directory = group_directory(cfg.shared_root)
    for index in range(count):
        name = f"retained-{index:06d}"
        record = new_group(name, cfg.machine_name)
        record["group"]["admission_state"] = "sealed" if index % 2 else "open"
        record["group"]["dispatch_state"] = "paused" if index % 3 == 0 else "active"
        path = directory / f"{name}.json"
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view) :]
        finally:
            os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_resources() -> dict[str, int]:
    """Read Linux process resource counts without adding a runtime dependency."""
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    rss_line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
    return {
        "rss_kib": int(rss_line.split()[1]),
        "descriptors": len(tuple(Path("/proc/self/fd").iterdir())),
        "threads": len(threading.enumerate()),
    }


def measure_locator_traversal(root: Path, lane: str, expected_group: str) -> dict[str, Any]:
    """Measure the bounded locator path without reading retained Group history."""
    group_root = group_directory(root)
    submissions = root / "operations/submissions"
    calls: list[str] = []
    original = locator.read_directory_entry

    def counted(path: Path, offset: int):
        candidate = Path(path)
        if candidate in {group_root, submissions}:
            raise AssertionError(f"active locator traversal touched history: {candidate}")
        calls.append(candidate.relative_to(root).as_posix())
        return original(candidate, offset)

    locator.read_directory_entry = counted
    started = time.monotonic()
    try:
        traversal = locator.GroupLocatorTraversal(root, lane)
        found = None
        for _ in range(locator.SHARD_COUNT + 2):
            found = traversal.advance()
            if found is not None:
                break
    finally:
        locator.read_directory_entry = original
    if found is None or found["identity"]["group"] != expected_group:
        raise AssertionError("expected locator was not found in one complete shard rotation")
    locator_path = locator.group_locator_path(root, expected_group, lane)
    return {
        "directory_reads": len(calls),
        "locator_bytes": locator_path.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "visited": calls,
        "resources": process_resources(),
    }


__all__ = ["measure_locator_traversal", "populate_settled_group_history", "process_resources"]
