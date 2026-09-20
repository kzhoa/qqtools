"""Pure checkpoint contracts for provisional qexp runtime source extraction.

These helpers validate and encode parser, projection, source, and spool
checkpoint state.  They perform no filesystem I/O and do not certify Group
membership, establish authority, or publish runtime state.
"""

from __future__ import annotations

import json
from pathlib import Path

from .source_revision import SourceRevision

_CHECKPOINT_VERSION = 1
_CHECKPOINT_KEYS = frozenset(
    {
        "version",
        "source_path",
        "source_revision",
        "expected_operation_id",
        "expected_group",
        "spool",
        "scanner",
        "projection",
    }
)
_SOURCE_REVISION_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_SPOOL_KEYS = frozenset({"device", "inode", "size", "events"})


def _validate_envelope(
    value: object,
    source_path: Path,
    expected_operation_id: str,
    expected_group: str,
) -> dict[str, object]:
    envelope = _exact_dict(value, _CHECKPOINT_KEYS, "checkpoint")
    if _exact_int(envelope["version"], "checkpoint.version") != _CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    checkpoint_path = _exact_str(envelope["source_path"], "checkpoint.source_path")
    if not Path(checkpoint_path).is_absolute() or checkpoint_path != str(source_path):
        raise ValueError("checkpoint source path does not match the caller")
    operation = _exact_str(envelope["expected_operation_id"], "checkpoint.expected_operation_id")
    group = _exact_str(envelope["expected_group"], "checkpoint.expected_group")
    if operation != expected_operation_id or group != expected_group:
        raise ValueError("checkpoint context does not match the caller")
    revision_value = _exact_dict(envelope["source_revision"], _SOURCE_REVISION_KEYS, "checkpoint.source_revision")
    revision = SourceRevision(
        _exact_int(revision_value["device"], "source_revision.device"),
        _exact_int(revision_value["inode"], "source_revision.inode"),
        _exact_int(revision_value["size"], "source_revision.size"),
        _exact_int(revision_value["mtime_ns"], "source_revision.mtime_ns"),
        _exact_int(revision_value["ctime_ns"], "source_revision.ctime_ns"),
    )
    spool = _exact_dict(envelope["spool"], _SPOOL_KEYS, "checkpoint.spool")
    spool_values = {
        name: _exact_int(spool[name], f"spool.{name}", minimum=0) for name in ("device", "inode", "size", "events")
    }
    scanner = envelope["scanner"]
    projection = envelope["projection"]
    if type(scanner) is not dict or type(projection) is not dict:
        raise ValueError("checkpoint parser snapshots must be objects")
    return {
        "source_revision": revision,
        "spool": spool_values,
        "scanner": scanner,
        "projection": projection,
    }


def _validate_cross_consistency(
    scanner: dict[str, object],
    projection: dict[str, object],
    revision: SourceRevision,
) -> None:
    raw_mode = scanner.get("raw_mode")
    offset = scanner.get("offset")
    token_kind = scanner.get("token_kind")
    token_id = scanner.get("token_id")
    fragment_start = scanner.get("fragment_start")
    next_token_id = scanner.get("next_token_id")
    if raw_mode is not True:
        raise ValueError("projection session requires raw scanner snapshots")
    if type(offset) is not int or offset < 0 or offset > revision.size:
        raise ValueError("scanner offset exceeds the source revision")
    if type(next_token_id) is not int or next_token_id < 0:
        raise ValueError("scanner next_token_id is invalid")
    projection_state = projection.get("state")
    if type(projection_state) is not dict:
        raise ValueError("projection snapshot state is invalid")
    last_token_id = projection_state.get("last_token_id")
    if type(last_token_id) is not int:
        raise ValueError("projection last_token_id is invalid")
    active = projection.get("active")
    summary = projection.get("summary")
    scanner_complete = scanner.get("is_complete")
    if type(scanner_complete) is not bool:
        raise ValueError("scanner completion state is invalid")
    if scanner_complete and offset != revision.size:
        raise ValueError("completed scanner offset does not match source size")
    if scanner_complete != (summary is not None):
        raise ValueError("scanner and projection completion states disagree")
    if token_kind is None:
        if active is not None:
            raise ValueError("inactive scanner cannot have an active projection token")
        if last_token_id != next_token_id - 1:
            raise ValueError("scanner and projection token counts disagree")
        return
    if type(token_id) is not int or type(fragment_start) is not int:
        raise ValueError("active scanner token fields are invalid")
    if active is not None:
        if type(active) is not dict:
            raise ValueError("active projection token is invalid")
        if active.get("id") != token_id or active.get("kind") != token_kind:
            raise ValueError("scanner and projection active tokens disagree")
        if active.get("end") != fragment_start:
            raise ValueError("scanner and projection fragment offsets disagree")
    if last_token_id != token_id - 1:
        raise ValueError("scanner and projection token counts disagree")


def _snapshot_offset(snapshot: dict[str, object]) -> int:
    offset = snapshot.get("offset")
    if type(offset) is not int or offset < 0:
        raise ValueError("scanner snapshot offset is invalid")
    return offset


def _projection_context(snapshot: dict[str, object], name: str) -> str:
    value = snapshot.get(name)
    if type(value) is not str:
        raise ValueError(f"projection snapshot {name} is invalid")
    return value


def _revision_dict(revision: SourceRevision) -> dict[str, int]:
    return {
        "device": revision.device,
        "inode": revision.inode,
        "size": revision.size,
        "mtime_ns": revision.mtime_ns,
        "ctime_ns": revision.ctime_ns,
    }


def _encode_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("value cannot be encoded as deterministic JSON") from exc


def _exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must contain exactly its versioned fields")
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _exact_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("checkpoint contains duplicate object keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"checkpoint contains non-JSON constant {value}")
