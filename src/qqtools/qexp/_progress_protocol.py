"""Bounded progress-v1 messages and non-durable advisory JSON I/O.

This module uses only the standard library. Nothing here is suitable for
persisting qexp execution authority, leases, or terminal records.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 8192
MAX_SNAPSHOT_BYTES = 16384
SEMANTIC_FIELDS = ("stage", "current", "total", "unit")
_FIELDS = frozenset(("protocol_version", "update_id", *SEMANTIC_FIELDS, "message"))
_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def identifier(value: Any) -> str:
    if not isinstance(value, str) or value in {".", ".."} or not _ID.fullmatch(value):
        raise ValueError("invalid progress identity")
    return value


def _text(value: Any, name: str, limit: int, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or (not optional and not value):
        raise ValueError(f"invalid progress {name}")
    if len(value.encode("utf-8")) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"invalid progress {name}")
    return value


def encode_json(value: dict[str, Any], *, max_bytes: int) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("progress record exceeds byte limit")
    return encoded


def validate_payload(value: Any) -> dict[str, Any]:
    """Validate a complete replacement, never a patch or a metrics event."""
    if not isinstance(value, dict) or set(value) - _FIELDS:
        raise ValueError("invalid progress fields")
    if type(value.get("protocol_version")) is not int or value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported progress protocol")
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "update_id": identifier(value.get("update_id")),
        "stage": _text(value.get("stage"), "stage", 64),
        "current": value.get("current"),
        "total": value.get("total"),
        "unit": _text(value.get("unit"), "unit", 32, optional=True),
        "message": _text(value.get("message"), "message", 1024, optional=True),
    }
    for name in ("current", "total"):
        number = result[name]
        if number is not None and (type(number) is not int or not 0 <= number <= 2**63 - 1):
            raise ValueError(f"invalid progress {name}")
    if result["current"] is not None and result["total"] is not None and result["current"] > result["total"]:
        raise ValueError("progress current exceeds total")
    encode_json(result, max_bytes=MAX_PAYLOAD_BYTES)
    return result


def semantic_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(payload.get(name) for name in SEMANTIC_FIELDS)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate progress field")
        result[key] = value
    return result


def read_advisory_snapshot(path: Path, *, max_bytes: int = MAX_SNAPSHOT_BYTES) -> dict[str, Any]:
    """Bound reads; refuse symlinks, devices and FIFOs rather than blocking on them."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("invalid progress file")
        encoded = handle.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError("progress record exceeds byte limit")
    value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("progress record must be an object")
    return value


def replace_advisory_snapshot(path: Path, value: dict[str, Any], *, max_bytes: int = MAX_SNAPSHOT_BYTES) -> None:
    """Atomically replace a disposable snapshot, without fsync or parent creation.

    The channel owner provisions directories. A stale writer must not recreate
    a task directory removed by cleanup. Atomic visibility is not durability.
    """
    encoded = encode_json(value, max_bytes=max_bytes)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
