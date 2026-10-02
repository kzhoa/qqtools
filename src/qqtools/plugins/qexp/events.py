"""Append-only event projection for diagnostics."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .runtime.paths import local_paths, shared_paths
from .runtime.records import new_id, validate_identifier
from .runtime.store import atomic_replace, iter_json, read_json

_LOCAL_EVENT_MAX_BYTES = 65_536
_LOCAL_EVENT_ID = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def write_notification_diagnostic(
    cfg: Any,
    event_type: str,
    event: Any,
    *,
    reason_code: str,
    error_type: str | None = None,
    notification_key: str | None = None,
    outcome: str | None = None,
    http_status: int | None = None,
    business_code: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Write a bounded notification diagnostic without exception or secret text."""
    details = {
        "attempt_id": event.attempt_id,
        "attempt_number": event.attempt_number,
        "phase": event.phase,
        "notifier": "feishu",
        "execution_machine_name": event.execution_machine_name,
        "dispatching_machine_name": event.dispatching_machine_name,
        "notification_key": notification_key or "",
        "outcome": outcome or event_type.removeprefix("notification_"),
        "reason_code": reason_code,
    }
    if isinstance(http_status, int):
        details["http_status"] = http_status
    if isinstance(business_code, str) and len(business_code) <= 64:
        details["business_code"] = business_code
    if isinstance(error_type, str):
        details["error_type"] = error_type
    if isinstance(duration_ms, int) and duration_ms >= 0:
        details["duration_ms"] = duration_ms
    write_diagnostic_event(cfg, event_type, task_id=event.task_id, attempt_id=event.attempt_id, details=details)


def write_event(
    cfg: Any, event_type: str, *, task_id: str | None = None, details: dict[str, Any] | None = None
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    event = {
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "task_id": task_id,
        "machine_name": cfg.machine_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    atomic_replace(shared_paths(cfg.shared_root)["events"] / now / f"{event['event_id']}.json", event)


def write_diagnostic_event(
    cfg: Any,
    event_type: str,
    *,
    task_id: str | None = None,
    attempt_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Publish diagnostics or preserve them locally when shared storage is unavailable."""
    try:
        write_event(cfg, event_type, task_id=task_id, details=details)
        return
    except OSError:
        pass
    event_id = new_id()
    event = {
        "event_id": event_id,
        "event_type": event_type,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "machine_name": cfg.machine_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    directory = local_paths(cfg.runtime_root)["events"] / (attempt_id or "machine")
    atomic_replace(directory / f"{event_id}.json", event)


def flush_local_event(
    cfg: Any,
    *,
    bucket: str,
    filename: str,
    event_id: str,
    sha256: str,
    machine_name: str,
    before_shared_write: Callable[[], bool],
) -> dict[str, str | None]:
    """Publish one digest-bound local diagnostic without retiring its source."""
    expected = {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "invalid_event"}
    if (
        not isinstance(bucket, str)
        or not bucket
        or "/" in bucket
        or "\\" in bucket
        or bucket in {".", ".."}
        or not isinstance(event_id, str)
        or not _LOCAL_EVENT_ID.fullmatch(event_id)
        or filename != f"{event_id}.json"
        or not isinstance(sha256, str)
        or not _SHA256_DIGEST.fullmatch(sha256)
    ):
        return expected

    runtime_root = Path(cfg.runtime_root)
    root = local_paths(runtime_root)["events"]
    directory = root / bucket
    path = directory / filename
    for current in reversed((*runtime_root.parents, runtime_root)):
        try:
            metadata = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            return {"outcome": "missing", "event_id": event_id, "sha256": sha256, "reason": "source_missing"}
        if not stat.S_ISDIR(metadata.st_mode):
            return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "source_untrusted"}
    for current in (runtime_root, root, directory):
        try:
            metadata = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            return {"outcome": "missing", "event_id": event_id, "sha256": sha256, "reason": "source_missing"}
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "source_untrusted"}

    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return {"outcome": "missing", "event_id": event_id, "sha256": sha256, "reason": "source_missing"}
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "source_untrusted"}

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return {"outcome": "missing", "event_id": event_id, "sha256": sha256, "reason": "source_missing"}
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "source_untrusted"}
        raise
    with os.fdopen(descriptor, "rb") as handle:
        opened_metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_uid != os.geteuid()
            or opened_metadata.st_nlink != 1
            or (opened_metadata.st_dev, opened_metadata.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "source_untrusted"}
        encoded = handle.read(_LOCAL_EVENT_MAX_BYTES + 1)

    if len(encoded) > _LOCAL_EVENT_MAX_BYTES:
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "invalid_event"}
    if hashlib.sha256(encoded).hexdigest() != sha256:
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "digest_mismatch"}

    try:
        event = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_event_object_without_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
        event_day = _validate_local_event(event, bucket=bucket, event_id=event_id, machine_name=machine_name)
    except _LocalEventIdentityError:
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "identity_mismatch"}
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "invalid_event"}

    if not before_shared_write():
        return {"outcome": "stale", "event_id": event_id, "sha256": sha256, "reason": "binding_fence"}

    published = atomic_replace(shared_paths(cfg.shared_root)["events"] / event_day / f"{event_id}.json", event)
    if published is None:
        raise OSError("durability could not be established for the shared diagnostic event.")
    return {"outcome": "flushed", "event_id": event_id, "sha256": sha256, "reason": None}


class _LocalEventIdentityError(ValueError):
    """The event is valid JSON but does not belong to the requested local identity."""


def _event_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("local event contains a duplicate JSON field.")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"local event contains invalid JSON constant {value}.")


def _validate_local_event(
    event: object,
    *,
    bucket: str,
    event_id: str,
    machine_name: str,
) -> str:
    required_fields = {"event_id", "event_type", "task_id", "machine_name", "timestamp", "details"}
    if not isinstance(event, dict) or frozenset(event) not in {
        frozenset(required_fields),
        frozenset(required_fields | {"attempt_id"}),
    }:
        raise ValueError("local event fields are invalid.")

    source_event_id = event["event_id"]
    event_type = event["event_type"]
    source_machine = event["machine_name"]
    timestamp = event["timestamp"]
    details = event["details"]
    if (
        not isinstance(source_event_id, str)
        or not _LOCAL_EVENT_ID.fullmatch(source_event_id)
        or not isinstance(event_type, str)
        or not event_type
        or len(event_type) > 128
        or "\x00" in event_type
        or not isinstance(source_machine, str)
        or not source_machine
        or len(source_machine) > 128
        or "\x00" in source_machine
        or not isinstance(timestamp, str)
        or not timestamp
        or len(timestamp) > 40
        or not isinstance(details, dict)
        or not _is_finite_json(details)
    ):
        raise ValueError("local event fields are invalid.")

    try:
        validate_identifier(source_event_id, "event_id")
        validate_identifier(source_machine, "machine_name")
    except ValueError as exc:
        raise ValueError("local event identifiers are invalid.") from exc

    for field in ("task_id", "attempt_id"):
        if field not in event:
            continue
        value = event[field]
        if value is not None and (not isinstance(value, str) or not value or len(value) > 128 or "\x00" in value):
            raise ValueError(f"local event {field} is invalid.")
        if value is not None:
            try:
                validate_identifier(value, f"event {field}")
            except ValueError as exc:
                raise ValueError(f"local event {field} is invalid.") from exc

    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp)
    except ValueError as exc:
        raise ValueError("local event timestamp is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("local event timestamp must use UTC.")

    attempt_id = event.get("attempt_id")
    expected_bucket = attempt_id if attempt_id is not None else "machine"
    if source_event_id != event_id or source_machine != machine_name or bucket != expected_bucket:
        raise _LocalEventIdentityError("local event identity does not match the request.")
    return parsed.strftime("%Y-%m-%d")


def _is_finite_json(value: object) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in value.items())
    return False


def flush_local_events(cfg: Any) -> int:
    """Idempotently copy durable local diagnostics to the shared event stream."""
    flushed = 0
    root = local_paths(cfg.runtime_root)["events"]
    for directory in sorted(root.glob("*")):
        if not directory.is_dir():
            continue
        for path in iter_json(directory):
            event = read_json(path)
            try:
                now = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).strftime("%Y-%m-%d")
                atomic_replace(shared_paths(cfg.shared_root)["events"] / now / f"{event['event_id']}.json", event)
            except (OSError, ValueError, KeyError):
                continue
            path.unlink(missing_ok=True)
            flushed += 1
    return flushed
