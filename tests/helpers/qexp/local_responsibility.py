"""Fault-injection and profiling adapters for the production membership algorithm."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from qqtools.plugins.qexp.runtime.responsibility_store import (
    BUCKETS,
    DATA_NAME,
    ENTRY_BYTES,
    FORMAT,
    HEADER_BYTES,
    INITIAL_ENTRY_BYTES,
    MAX_COUNTER,
    PAGE_BYTES,
    PAGE_SIZE,
    TRANSACTION_BYTES,
    Conflict,
    ServiceTraversal,
    Unavailable,
    encode,
    identity_key,
)
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger as StorageLedger


class DurableIO:
    """Atomic replacement with a fixed scratch file under the bucket's leaf lock.

    Instrumentation deliberately duplicates the small production replace primitive:
    crash hooks must cover write, file sync, rename, unlink, and directory sync.
    No monkey-patching or importing production storage is necessary.
    """

    def __init__(self, hook: Callable[[str], None] | None = None) -> None:
        self.hook = hook
        self.counts: Counter[str] = Counter()
        self.seconds: Counter[str] = Counter()
        self.events: list[str] = []

    def event(self, label: str) -> None:
        # Retaining traces is opt-in so a benchmark does not grow with history.
        if self.hook is not None:
            self.events.append(label)
            self.hook(label)

    @contextmanager
    def lock(self, path: Path):
        started = time.perf_counter()
        with path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            acquired = time.perf_counter()
            self.seconds["lock_wait"] += acquired - started
            try:
                yield
            finally:
                self.seconds["lock_held"] += time.perf_counter() - acquired
                fcntl.flock(stream, fcntl.LOCK_UN)

    def read(self, path: Path, limit: int) -> dict:
        self.counts["reads"] += 1
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        self.counts["read_bytes"] += len(data)
        if len(data) > limit:
            raise Unavailable(f"record exceeds byte limit: {path}")
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise Unavailable(f"invalid JSON: {path}") from exc
        if not isinstance(value, dict):
            raise Unavailable(f"expected object: {path}")
        return value

    def sync(self, fd: int, kind: str, label: str) -> None:
        start = time.perf_counter()
        os.fsync(fd)
        self.seconds[kind] += time.perf_counter() - start
        self.counts[kind] += 1
        self.event(f"{label}:{kind}")

    def sync_directory(self, path: Path, label: str) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.sync(fd, "directory_fsync", label)
        finally:
            os.close(fd)

    def replace(self, path: Path, value: dict, *, should_sync_directory: bool = True) -> None:
        data = encode(value)
        if path.name == "pending":
            self.counts["max_transaction_bytes"] = max(self.counts["max_transaction_bytes"], len(data))
            self.counts["max_images"] = max(self.counts["max_images"], len(value["images"]))
        scratch = path.parent / "scratch"
        with scratch.open("wb") as stream:
            stream.write(data)
            stream.flush()
            self.counts["writes"] += 1
            self.counts["write_bytes"] += len(data)
            self.event(f"{path.name}:write")
            self.sync(stream.fileno(), "file_fsync", path.name)
        os.replace(scratch, path)
        self.counts["renames"] += 1
        self.event(f"{path.name}:rename")
        if should_sync_directory:
            self.sync_directory(path.parent, path.name)

    def delete(self, path: Path, *, should_sync_directory: bool = True) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        else:
            self.counts["unlinks"] += 1
            self.event(f"{path.name}:unlink")
        # Even an already absent name may reflect an earlier unsynced unlink.
        if should_sync_directory:
            self.sync_directory(path.parent, path.name)


class Ledger(StorageLedger):
    """Use instrumented I/O while retaining the exact production algorithm."""

    io_type = DurableIO


__all__ = [
    "BUCKETS",
    "PAGE_SIZE",
    "HEADER_BYTES",
    "PAGE_BYTES",
    "ENTRY_BYTES",
    "INITIAL_ENTRY_BYTES",
    "TRANSACTION_BYTES",
    "MAX_COUNTER",
    "FORMAT",
    "DATA_NAME",
    "Conflict",
    "Unavailable",
    "encode",
    "identity_key",
    "ServiceTraversal",
    "Ledger",
    "DurableIO",
]
