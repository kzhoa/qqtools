"""Strict decoding for durable Group discovery receipts."""

from __future__ import annotations

import json
from pathlib import Path

from .source_revision import SourceRevision

_STAT_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_RECEIPT_KEYS = frozenset(
    {
        "version",
        "source_path",
        "source_revision",
        "operation_id",
        "group",
        "status",
        "spool",
        "task_references",
        "sequence_references",
        "task_count",
    }
)
_FILE_KEYS = frozenset({"path", "size", "revision"})
_STAT_ORDER = ("device", "inode", "size", "mtime_ns", "ctime_ns")
_STAT_MAX = (1 << 64) - 1


def decode_receipt(
    data: bytes,
    source: Path,
    operation_id: str,
    group: str,
    scratch: Path,
) -> dict[str, object]:
    """Decode and validate one whole-source confirmation receipt."""

    try:
        value = json.loads(data, object_pairs_hook=_reject_duplicate_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("receipt is not valid UTF-8 JSON") from exc
    if type(value) is not dict or set(value) != _RECEIPT_KEYS:
        raise ValueError("receipt fields are invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported receipt version")
    if value["source_path"] != str(source):
        raise ValueError("receipt source path does not match the caller")
    if value["operation_id"] != operation_id or value["group"] != group:
        raise ValueError("receipt context does not match the caller")
    if value["status"] not in {"qualified", "irrelevant", "ambiguous"}:
        raise ValueError("receipt status is invalid")
    revision = _decode_revision(value["source_revision"], "source_revision")
    spool = _decode_file_record(value["spool"], "projection/events.jsonl", scratch)
    task = _decode_optional_file(value["task_references"], "qualification/task.refs", scratch)
    sequence = _decode_optional_file(value["sequence_references"], "qualification/sequence.refs", scratch)
    if value["status"] == "qualified":
        if task is None or sequence is None:
            raise ValueError("qualified receipt is missing references")
    elif task is not None or sequence is not None:
        raise ValueError("non-qualified receipt must not carry references")
    count = value["task_count"]
    if type(count) is not int or count < 0:
        raise ValueError("receipt task count is invalid")
    return {
        "version": 1,
        "source_path": str(source),
        "source_revision": revision,
        "operation_id": operation_id,
        "group": group,
        "status": value["status"],
        "spool": spool,
        "task_references": task,
        "sequence_references": sequence,
        "task_count": count,
    }


def _decode_optional_file(value: object, expected_path: str, scratch: Path) -> dict[str, object] | None:
    if value is None:
        return None
    return _decode_file_record(value, expected_path, scratch)


def _decode_file_record(value: object, expected_path: str, scratch: Path) -> dict[str, object]:
    if type(value) is not dict or set(value) != _FILE_KEYS:
        raise ValueError("receipt file record fields are invalid")
    path = value["path"]
    if type(path) is not str or path != expected_path or Path(path).is_absolute():
        raise ValueError("receipt file path is invalid")
    try:
        if (scratch / path).relative_to(scratch) != Path(path):
            raise ValueError("receipt file path escapes scratch")
    except ValueError as exc:
        raise ValueError("receipt file path escapes scratch") from exc
    size = value["size"]
    if type(size) is not int or size < 0:
        raise ValueError("receipt file size is invalid")
    revision = _decode_revision(value["revision"], "file revision")
    if revision.size != size:
        raise ValueError("receipt file size does not match its revision")
    return {"path": path, "size": size, "revision": revision}


def _decode_revision(value: object, label: str) -> SourceRevision:
    if type(value) is not dict or set(value) != _STAT_KEYS:
        raise ValueError(f"receipt {label} fields are invalid")
    numbers: list[int] = []
    for name in _STAT_ORDER:
        item = value[name]
        if type(item) is not int or item < 0 or item > _STAT_MAX:
            raise ValueError(f"receipt {label}.{name} is invalid")
        numbers.append(item)
    return SourceRevision(*numbers)


def _reject_duplicate_pairs(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("receipt contains duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"receipt contains non-JSON constant {value}")


__all__ = ["decode_receipt"]
