"""Optional Group defaults for live progress on new submissions."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import group_namespace, locks
from .records import validate_group_name, validate_identifier
from .store import atomic_replace, read_json_limited

_MAX_RECORD_BYTES = 4096
_MAX_REVISION = (1 << 63) - 1
_MAX_SNAPSHOT_WAIT_SECONDS = 0.1
_APPLIES_TO = "new_submissions"


@dataclass(slots=True)
class _SnapshotRead:
    """State owned by one bounded policy-read worker invocation."""

    completed: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    discarded: bool = False


_snapshot_lock = threading.Lock()
_active_snapshot: _SnapshotRead | None = None


def _reset_snapshot_after_fork() -> None:
    """A child cannot inherit an outstanding read worker from its parent."""
    global _snapshot_lock, _active_snapshot
    _snapshot_lock = threading.Lock()
    _active_snapshot = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_snapshot_after_fork)


def _validated_name(name: str) -> str:
    validated = validate_group_name(name)
    if validated is None:
        raise ValueError("group name must not be null.")
    return validated


def _policy_path(root: Path, name: str) -> Path:
    return root / "group-observation" / f"{_validated_name(name)}.json"


def _validate_group_identity(value: Any, label: str = "group_identity") -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"name", "created_at", "creation_operation_id"}:
        raise ValueError(f"{label} must contain exactly name, created_at, and creation_operation_id.")

    name = _validated_name(value["name"])
    created_at = value["created_at"]
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"{label}.created_at must be a non-empty string.")

    creation_operation_id = value["creation_operation_id"]
    if creation_operation_id is not None:
        validate_identifier(creation_operation_id, f"{label}.creation_operation_id")

    return {
        "name": name,
        "created_at": created_at,
        "creation_operation_id": creation_operation_id,
    }


def group_identity(group: dict[str, Any]) -> dict[str, Any]:
    """Copy and validate the immutable identity fields from a published Group."""
    if type(group) is not dict:
        raise ValueError("Group must be a dictionary.")
    group_fields = group.get("group")
    meta_fields = group.get("meta")
    if type(group_fields) is not dict or type(meta_fields) is not dict:
        raise ValueError("Group must contain dictionary group and meta fields.")

    identity = {
        "name": group_fields.get("name"),
        "created_at": meta_fields.get("created_at"),
        "creation_operation_id": group_fields.get("creation_operation_id"),
    }
    return _validate_group_identity(identity, "Group identity")


def _validate_policy_record(value: Any) -> dict[str, Any]:
    required_fields = {"version", "revision", "group_identity", "live_progress"}
    if type(value) is not dict or set(value) != required_fields:
        raise ValueError(
            "Group live-progress policy must contain exactly version, revision, group_identity, and live_progress."
        )
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Group live-progress policy version must be integer 1.")

    revision = value["revision"]
    if type(revision) is not int or not 1 <= revision <= _MAX_REVISION:
        raise ValueError("Group live-progress policy revision must be a positive signed-64-bit integer.")

    if type(value["live_progress"]) is not bool:
        raise ValueError("Group live_progress must be a boolean.")

    return {
        "version": 1,
        "revision": revision,
        "group_identity": _validate_group_identity(value["group_identity"]),
        "live_progress": value["live_progress"],
    }


def read_policy_record(root: Path, name: str) -> dict[str, Any] | None:
    """Read and strictly validate one persisted Group policy record."""
    path = _policy_path(root, name)
    try:
        value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
    except FileNotFoundError:
        return None
    return _validate_policy_record(value)


def _bounded_diagnostic(value: object) -> str:
    try:
        message = str(value)
    except Exception:
        message = "policy read failed"
    safe_message = "".join(character if character.isprintable() else " " for character in message)
    safe_message = " ".join(safe_message.split())
    return (safe_message or "policy read failed")[:160]


def _exception_diagnostic(exc: Exception) -> str:
    return _bounded_diagnostic(f"{type(exc).__name__}: {_bounded_diagnostic(exc)}")


def _unavailable(reason: object) -> dict[str, Any]:
    return {"status": "unavailable", "reason": _bounded_diagnostic(reason)}


def _read_snapshot_worker(job: _SnapshotRead, root: Path, name: str) -> None:
    result: dict[str, Any] | None = None
    try:
        try:
            record = read_policy_record(root, name)
        except Exception as exc:
            result = _unavailable(_exception_diagnostic(exc))
        else:
            result = {"status": "missing"} if record is None else {"status": "available", "record": record}
    finally:
        with _snapshot_lock:
            if not job.discarded:
                job.result = result
            job.completed.set()


def read_policy_snapshot(root: Path, name: str, timeout: float = 0.1) -> dict[str, Any]:
    """Read one policy record with a process-local 100 ms submission wait budget."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
        raise ValueError("timeout must be a finite non-negative number.")
    if timeout < 0:
        raise ValueError("timeout must be a finite non-negative number.")
    wait_seconds = min(float(timeout), _MAX_SNAPSHOT_WAIT_SECONDS)

    global _active_snapshot
    with _snapshot_lock:
        if _active_snapshot is not None:
            if _active_snapshot.completed.is_set():
                _active_snapshot = None
            else:
                return _unavailable("policy read worker is occupied")

        job = _SnapshotRead()
        _active_snapshot = job
        try:
            worker = threading.Thread(
                target=_read_snapshot_worker,
                args=(job, root, name),
                name="qexp-group-policy-read",
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            if _active_snapshot is job:
                _active_snapshot = None
            return _unavailable(_exception_diagnostic(exc))

    if not job.completed.wait(wait_seconds):
        with _snapshot_lock:
            job.discarded = True
            job.result = None
        return _unavailable("policy read timed out")

    with _snapshot_lock:
        if job.result is None:
            return _unavailable("policy read worker returned no result")
        return job.result


def _default_inspection(identity: dict[str, Any], diagnostic: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "live_progress": False,
        "revision": 0,
        "source": "default",
        "applies_to": _APPLIES_TO,
        "group_identity": dict(identity),
    }
    if diagnostic is not None:
        result["diagnostic"] = _bounded_diagnostic(diagnostic)
    return result


def _configured_inspection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "live_progress": record["live_progress"],
        "revision": record["revision"],
        "source": "configured",
        "applies_to": _APPLIES_TO,
        "group_identity": dict(record["group_identity"]),
    }


def inspect_group_policy(root: Path, name: str) -> dict[str, Any]:
    """Inspect the policy associated with the currently published Group."""
    current_identity = group_identity(group_namespace.read_group(root, _validated_name(name)))
    record = read_policy_record(root, name)
    if record is None:
        return _default_inspection(current_identity)
    if record["group_identity"] != current_identity:
        return _default_inspection(current_identity, "stored policy identity does not match the published Group")
    return _configured_inspection(record)


def set_group_policy(root: Path, name: str, enabled: bool) -> dict[str, Any]:
    """Set the optional default for future submissions without changing Group truth."""
    validated_name = _validated_name(name)
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean.")

    lock_path = root / "locks" / "group-observation" / f"{validated_name}.lock"
    with locks.exclusive(lock_path) as acquired:
        if not acquired:
            raise RuntimeError("Could not acquire the Group live-progress policy lock.")

        current_identity = group_identity(group_namespace.read_group(root, validated_name))
        previous = read_policy_record(root, validated_name)
        if previous is not None and previous["group_identity"] == current_identity:
            if previous["revision"] == _MAX_REVISION:
                raise ValueError("Group live-progress policy revision is exhausted.")
            revision = previous["revision"] + 1
        else:
            revision = 1

        record = {
            "version": 1,
            "revision": revision,
            "group_identity": current_identity,
            "live_progress": enabled,
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("Group live-progress policy exceeds its 4096-byte limit.")

        published_identity = group_identity(group_namespace.read_group(root, validated_name))
        if published_identity != current_identity:
            raise RuntimeError("Published Group identity changed while setting its live-progress policy.")

        atomic_replace(_policy_path(root, validated_name), record)
        return _configured_inspection(record)
