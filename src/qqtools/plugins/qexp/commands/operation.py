"""Bounded, read-only inspection of durable qexp operation references."""

from __future__ import annotations

import base64
import binascii
import json
import re
import shlex
from typing import Any

from ..config_types import RootConfig
from ..layout import project_id
from ..runtime.operation_store import active_operation_path, archived_operation_path
from ..runtime.records import SCHEMA_VERSION, validate_group_name, validate_identifier
from ..runtime.store import read_json

OPERATION_REFERENCE_VERSION = 1
MAX_OPERATION_REFERENCE_BYTES = 4 * 1024
SUPPORTED_OPERATION_KINDS = frozenset({"group_cancel", "worker_remove", "cleanup"})

# Short aliases make the fixed contract easy to discover for callers that do
# not need to know the longer descriptive names.
REFERENCE_VERSION = OPERATION_REFERENCE_VERSION
MAX_REFERENCE_BYTES = MAX_OPERATION_REFERENCE_BYTES
SUPPORTED_KINDS = SUPPORTED_OPERATION_KINDS

_REFERENCE_FIELDS = frozenset({"v", "project", "kind", "key", "operation_id"})
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class OperationReferenceError(ValueError):
    """A reference token failed validation before durable-record lookup."""

    def __init__(self, *, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def _invalid_reference(message: str) -> OperationReferenceError:
    return OperationReferenceError(code="invalid_reference", message=message, exit_code=2)


def _validate_reference_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"{label} is not a safe identifier.")
    return validate_identifier(value, label)


def _canonical_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def create_operation_reference(cfg: RootConfig, kind: str, storage_key: str, operation_id: str) -> str:
    """Create a canonical opaque reference for one durable operation."""
    if kind not in SUPPORTED_OPERATION_KINDS:
        raise ValueError(f"unsupported operation reference kind {kind!r}.")
    if kind != "cleanup" and storage_key != operation_id:
        raise ValueError("Group operation references use operation_id as their storage key.")
    _validate_reference_identifier(storage_key, "storage_key")
    _validate_reference_identifier(operation_id, "operation_id")
    selected_project = _validate_reference_identifier(project_id(cfg.shared_root), "project")
    payload: dict[str, object] = {
        "v": OPERATION_REFERENCE_VERSION,
        "project": selected_project,
        "kind": kind,
        "key": storage_key,
        "operation_id": operation_id,
    }
    encoded = base64.urlsafe_b64encode(_canonical_payload(payload)).decode("ascii").rstrip("=")
    if len(encoded.encode("ascii")) > MAX_OPERATION_REFERENCE_BYTES:
        raise ValueError("operation reference exceeds the 4 KiB limit.")
    return encoded


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant {value!r} is not permitted.")


def _decode_reference(cfg: RootConfig, reference: str) -> dict[str, object]:
    if not isinstance(reference, str) or not reference:
        raise _invalid_reference("operation reference must be a non-empty string.")
    try:
        encoded = reference.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _invalid_reference("operation reference must contain URL-safe base64 characters.") from exc
    if len(encoded) > MAX_OPERATION_REFERENCE_BYTES:
        raise _invalid_reference("operation reference exceeds the 4 KiB limit.")
    if "=" in reference or _BASE64_PATTERN.fullmatch(reference) is None or len(reference) % 4 == 1:
        raise _invalid_reference("operation reference is not canonical URL-safe base64.")
    padded = encoded + b"=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(decoded).rstrip(b"=") != encoded:
            raise ValueError("non-canonical base64 encoding")
        text = decoded.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _invalid_reference("operation reference payload is not canonical JSON.") from exc
    if not isinstance(payload, dict) or frozenset(payload) != _REFERENCE_FIELDS:
        raise _invalid_reference("operation reference has missing or unknown fields.")
    if type(payload.get("v")) is not int or payload["v"] != OPERATION_REFERENCE_VERSION:
        raise _invalid_reference("operation reference version is unsupported.")
    if payload.get("kind") not in SUPPORTED_OPERATION_KINDS:
        raise _invalid_reference("operation reference kind is unsupported.")
    if any(type(payload.get(key)) is not str for key in ("project", "kind", "key", "operation_id")):
        raise _invalid_reference("operation reference identifiers must be strings.")
    try:
        selected_project = _validate_reference_identifier(payload["project"], "project")
        _validate_reference_identifier(payload["kind"], "kind")
        _validate_reference_identifier(payload["key"], "storage_key")
        _validate_reference_identifier(payload["operation_id"], "operation_id")
    except (TypeError, ValueError) as exc:
        raise _invalid_reference(str(exc)) from exc
    if payload["kind"] != "cleanup" and payload["key"] != payload["operation_id"]:
        raise _invalid_reference("Group operation reference key must equal operation_id.")
    if selected_project != project_id(cfg.shared_root):
        raise _invalid_reference("operation reference belongs to a different Project.")
    if _canonical_payload(payload) != decoded:
        raise _invalid_reference("operation reference JSON is not canonical.")
    return payload


def _operation_lookup_kind(kind: str) -> tuple[str, str]:
    if kind in {"group_cancel", "worker_remove"}:
        return "group_control", "group_control"
    return "cleanup", "cleanup"


def _read_operation_record(cfg: RootConfig, storage_kind: str, storage_key: str) -> dict[str, Any]:
    """Read active truth, then one exact archive path if it vanished."""
    active = active_operation_path(cfg, storage_kind, storage_key)  # type: ignore[arg-type]
    archive = archived_operation_path(cfg, storage_kind, storage_key)  # type: ignore[arg-type]
    try:
        return read_json(active)
    except FileNotFoundError:
        return read_json(archive)


def _validate_meta(meta: object) -> None:
    if not isinstance(meta, dict):
        raise ValueError("operation envelope metadata is malformed.")
    if meta.get("schema_version") != SCHEMA_VERSION or type(meta.get("revision")) is not int:
        raise ValueError("operation envelope metadata has an invalid schema or revision.")
    if meta["revision"] < 1:
        raise ValueError("operation envelope metadata has invalid revision.")
    for field in ("created_at", "updated_at"):
        if field in meta and not isinstance(meta[field], str):
            raise ValueError("operation envelope metadata has invalid timestamps.")
    updated_by = meta.get("updated_by")
    if updated_by is not None:
        if not isinstance(updated_by, dict) or not isinstance(updated_by.get("actor_type"), str):
            raise ValueError("operation envelope metadata has invalid writer identity.")
        _validate_reference_identifier(updated_by.get("machine_name"), "updated_by.machine_name")
        if not isinstance(updated_by.get("process_id"), str) or not updated_by["process_id"]:
            raise ValueError("operation envelope metadata has invalid process identity.")


def _validate_inner(
    record: object,
    *,
    kind: str,
    storage_key: str,
    operation_id: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    _, inner_key = _operation_lookup_kind(kind)
    if not isinstance(record, dict) or frozenset(record) != frozenset({"meta", inner_key}):
        raise ValueError("operation envelope is malformed.")
    _validate_meta(record.get("meta"))
    inner = record.get(inner_key)
    if not isinstance(inner, dict):
        raise ValueError("operation record is malformed.")
    if inner.get("operation_id") != operation_id:
        raise LookupError("operation record identity does not match the reference.")
    _validate_reference_identifier(inner.get("operation_id"), "operation_id")
    if not isinstance(inner.get("state"), str) or not inner["state"]:
        raise ValueError("operation state is malformed.")
    if kind == "cleanup":
        if inner.get("task_id") != storage_key:
            raise LookupError("cleanup Task identity does not match the reference.")
        _validate_reference_identifier(inner.get("task_id"), "task_id")
        target: dict[str, object] = {"task_id": inner["task_id"]}
    else:
        expected_types = {"group_cancel": {"cancel"}, "worker_remove": {"worker_remove", "worker_remove_v2"}}[kind]
        if inner.get("operation_type") not in expected_types:
            raise LookupError("operation type does not match the reference kind.")
        group_name = inner.get("group_name")
        try:
            validate_group_name(group_name)
        except (TypeError, ValueError) as exc:
            raise ValueError("operation group target is malformed.") from exc
        target = {"group_name": group_name}
        if kind == "worker_remove":
            _validate_reference_identifier(inner.get("machine_name"), "machine_name")
            target["machine_name"] = inner["machine_name"]
    if kind != "cleanup" and inner.get("group_name") is None:
        raise ValueError("operation group target is malformed.")
    return inner, target


def _pending_machines(inner: dict[str, Any], kind: str) -> list[str]:
    raw = inner.get("pending_machines")
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return sorted(set(raw))
    acknowledgements = inner.get("pending_machine_acknowledgements")
    if isinstance(acknowledgements, dict):
        return sorted(str(machine) for machine in acknowledgements)
    if kind == "cleanup":
        required = inner.get("required_machines")
        acknowledged = inner.get("acknowledgements")
        if isinstance(required, list) and isinstance(acknowledged, dict):
            return sorted(machine for machine in required if machine not in acknowledged)
    return []


def _result_base(payload: dict[str, object], reference: str) -> dict[str, object]:
    return {
        "schema_version": OPERATION_REFERENCE_VERSION,
        "reference": reference,
        "project": payload["project"],
        "kind": payload["kind"],
        "operation_id": payload["operation_id"],
        "storage_key": payload["key"],
        "target": None,
        "state": None,
        "progress": None,
        "blockers": [],
        "pending_machines": [],
        "outcome": None,
        "error": None,
        "lifecycle_outcome": None,
        "created_at": None,
        "updated_at": None,
        "completed_at": None,
        "next_action": None,
    }


def _lifecycle_outcome(state: object) -> str:
    if state == "completed":
        return "completed"
    if state == "blocked":
        return "blocked"
    if state in {"preparing", "converging", "waiting_ack"}:
        return "waiting_ack"
    return "pending"


def inspect_operation(cfg: RootConfig, reference: str) -> tuple[dict[str, object], int]:
    """Inspect one exact active/archive operation path without repairing it."""
    payload = _decode_reference(cfg, reference)
    result = _result_base(payload, reference)
    storage_kind, _ = _operation_lookup_kind(payload["kind"])
    try:
        record = _read_operation_record(cfg, storage_kind, payload["key"])
    except FileNotFoundError:
        result.update(
            {
                "outcome": "not_found",
                "lifecycle_outcome": "not_found",
                "error": {"code": "not_found", "message": "operation record was not found."},
            }
        )
        return result, 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result.update(
            {
                "outcome": "unreadable",
                "lifecycle_outcome": "unreadable",
                "error": {"code": "unreadable", "message": f"operation record is unreadable: {exc}"},
            }
        )
        return result, 1
    try:
        inner, target = _validate_inner(
            record,
            kind=payload["kind"],
            storage_key=payload["key"],
            operation_id=payload["operation_id"],
        )
    except LookupError as exc:
        result.update(
            {
                "outcome": "mismatched",
                "lifecycle_outcome": "mismatched",
                "error": {"code": "mismatched", "message": str(exc)},
            }
        )
        return result, 1
    except (TypeError, ValueError) as exc:
        result.update(
            {
                "outcome": "unreadable",
                "lifecycle_outcome": "unreadable",
                "error": {"code": "unreadable", "message": str(exc)},
            }
        )
        return result, 1
    blockers = inner.get("blockers")
    if not isinstance(blockers, list):
        blockers = []
    lifecycle_outcome = _lifecycle_outcome(inner["state"])
    meta = record["meta"]
    next_action = None
    if lifecycle_outcome in {"waiting_ack", "blocked", "pending"}:
        next_action = (
            "qexp admin operation show " + shlex.quote(reference) + " --project " + shlex.quote(str(cfg.project_root))
        )
    result.update(
        {
            "target": target,
            "state": inner["state"],
            "progress": inner.get("progress") if isinstance(inner.get("progress"), dict) else None,
            "blockers": blockers,
            "pending_machines": _pending_machines(inner, payload["kind"]),
            "outcome": "ok",
            "lifecycle_outcome": lifecycle_outcome,
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "completed_at": inner.get("completed_at"),
            "next_action": next_action,
        }
    )
    return result, 0


__all__ = [
    "MAX_OPERATION_REFERENCE_BYTES",
    "MAX_REFERENCE_BYTES",
    "OPERATION_REFERENCE_VERSION",
    "OperationReferenceError",
    "REFERENCE_VERSION",
    "SUPPORTED_KINDS",
    "SUPPORTED_OPERATION_KINDS",
    "create_operation_reference",
    "inspect_operation",
]
