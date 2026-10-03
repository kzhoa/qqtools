"""Attempt-bound scoped progress observations for the v3 advisory channel."""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import identifier, read_advisory_snapshot, replace_advisory_snapshot
from qqtools.qexp._progress_protocol_v3 import (
    MAX_PAYLOAD_V3_BYTES,
    MAX_SNAPSHOT_V3_BYTES,
    semantic_key_v3,
    validate_payload_v3,
)

from .progress import _IDENTITY, _valid_interval
from .progress_v2 import _ENVELOPE_FIELDS as _V2_ENVELOPE_FIELDS
from .progress_v2 import ProgressV2Projector

_CONTEXT_FIELDS = frozenset(("protocol_version", "interval_seconds", *_IDENTITY))
_ENVELOPE_FIELDS = _V2_ENVELOPE_FIELDS
_PROGRESS_FIELDS = frozenset(("activity", "overall", "metrics", "completeness"))
_LOCAL_MAILBOX_NAME = "latest-v3.json"
_CONTEXT_DIRECTORY = "progress-v3-contexts"
_OBSERVED_DIRECTORY = "progress-v3-observed"
_SHARED_DIRECTORY = "progress-v3"
_COORDINATOR_DIRECTORY = "v3"


def _local_path(cfg: Any, directory: str, attempt_id: str) -> Path:
    return Path(cfg.runtime_root) / directory / f"{identifier(attempt_id)}.json"


def _local_mailbox_path(runtime_root: Path, attempt_id: str) -> Path:
    return Path(runtime_root) / "progress" / identifier(attempt_id) / _LOCAL_MAILBOX_NAME


def _context_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, _CONTEXT_DIRECTORY, attempt_id)


def _observed_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, _OBSERVED_DIRECTORY, attempt_id)


def _shared_path(cfg: Any, task_id: str, attempt_id: str) -> Path:
    return Path(cfg.shared_root) / _SHARED_DIRECTORY / identifier(task_id) / f"{identifier(attempt_id)}.json"


def _validate_context(value: Any, attempt_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise ValueError("invalid progress v3 context fields")
    if type(value.get("protocol_version")) is not int or value["protocol_version"] != 3:
        raise ValueError("unsupported progress v3 context")
    identifier(value.get("task_id"))
    identifier(value.get("attempt_id"))
    if value["attempt_id"] != attempt_id:
        raise ValueError("progress v3 context identity mismatch")
    if type(value.get("attempt_number")) is not int or value["attempt_number"] < 1:
        raise ValueError("invalid progress v3 Attempt number")
    if not isinstance(value.get("machine_name"), str) or not value["machine_name"]:
        raise ValueError("invalid progress v3 machine identity")
    if value.get("launch_id") is not None and not isinstance(value["launch_id"], str):
        raise ValueError("invalid progress v3 launch identity")
    if type(value.get("wrapper_pid")) is not int or value["wrapper_pid"] < 1:
        raise ValueError("invalid progress v3 wrapper process identity")
    if type(value.get("wrapper_start_time_ticks")) is not int or value["wrapper_start_time_ticks"] < 0:
        raise ValueError("invalid progress v3 wrapper start time")
    _valid_interval(value.get("interval_seconds"))
    return dict(value)


def _validate_projection_v3(
    value: Any,
    identity: dict[str, Any],
    *,
    require_token: bool = False,
    require_generation: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int or value["protocol_version"] != 3:
        raise ValueError("unsupported progress v3 snapshot")
    if set(value) != _ENVELOPE_FIELDS or not isinstance(value.get("progress"), dict):
        raise ValueError("invalid progress v3 snapshot fields")
    _validate_context(
        {"protocol_version": 3, "interval_seconds": 1, **{key: value[key] for key in _IDENTITY}},
        value["attempt_id"],
    )
    if any(type(value[key]) is not type(identity.get(key)) or value[key] != identity.get(key) for key in _IDENTITY):
        raise ValueError("progress v3 snapshot identity mismatch")
    identifier(value.get("registration_generation"))
    if require_generation and value["registration_generation"] != identity.get("registration_generation"):
        raise ValueError("superseded progress v3 registration generation")
    if type(value.get("fencing_token")) is not int:
        raise ValueError("invalid progress v3 fencing token")
    if require_token and value["fencing_token"] != identity.get("fencing_token"):
        raise ValueError("superseded progress v3 snapshot")
    identifier(value.get("source_update_id"))
    if type(value.get("sequence")) is not int or value["sequence"] < 1:
        raise ValueError("invalid progress v3 sequence")
    for key in ("reported_at", "advanced_at"):
        if not isinstance(value.get(key), str):
            raise ValueError("invalid progress v3 timestamp")
        try:
            stamp = datetime.fromisoformat(value[key].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid progress v3 timestamp") from exc
        if stamp.tzinfo is None:
            raise ValueError("progress v3 timestamp must have a timezone")
    if set(value["progress"]) != _PROGRESS_FIELDS:
        raise ValueError("invalid nested progress v3 fields")
    normalized = validate_payload_v3(
        {"protocol_version": 3, "update_id": value["source_update_id"], **value["progress"]}
    )
    result = dict(value)
    result["progress"] = {key: item for key, item in normalized.items() if key not in {"protocol_version", "update_id"}}
    return result


def _same_content(payload: dict[str, Any], projection: dict[str, Any]) -> bool:
    return semantic_key_v3(payload) == semantic_key_v3(projection.get("progress", projection))


def prepare_progress_v3_channel(
    cfg: Any,
    task: Any,
    attempt: Any,
    *,
    wrapper_start_time_ticks: int | None,
    interval_seconds: int | float,
) -> str | None:
    """Provision a v3 mailbox and its frozen identity using local I/O only."""
    try:
        if type(wrapper_start_time_ticks) is not int or wrapper_start_time_ticks < 0:
            return None
        if attempt.task_id != task.task_id:
            return None
        frozen_interval = _valid_interval(interval_seconds)
        context = {
            "protocol_version": 3,
            "interval_seconds": frozen_interval,
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "machine_name": cfg.machine_name,
            "launch_id": attempt.authorization["launch_id"],
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": wrapper_start_time_ticks,
        }
        attempt_id = identifier(attempt.attempt_id)
        _validate_context(context, attempt_id)
        runtime_root = Path(cfg.runtime_root)
        mailbox = _local_mailbox_path(runtime_root, attempt_id)
        mailbox.parent.mkdir(parents=True, exist_ok=True)
        context_root = runtime_root / _CONTEXT_DIRECTORY
        observed_root = runtime_root / _OBSERVED_DIRECTORY
        context_root.mkdir(parents=True, exist_ok=True)
        observed_root.mkdir(parents=True, exist_ok=True)
        path = context_root / f"{attempt_id}.json"
        if path.exists():
            existing = _validate_context(read_advisory_snapshot(path), attempt_id)
            if any(existing.get(key) != context.get(key) for key in _IDENTITY):
                return None
            if existing["interval_seconds"] != frozen_interval:
                return None
            return str(mailbox)
        replace_advisory_snapshot(path, context)
        return str(mailbox)
    except Exception:
        return None


def has_local_progress_v3_mailbox(runtime_root: Path) -> bool:
    """Look for a regular v3 producer mailbox without touching shared state."""
    root = Path(runtime_root) / "progress"
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    mailbox = Path(entry.path) / _LOCAL_MAILBOX_NAME
                    if stat.S_ISREG(mailbox.lstat().st_mode):
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


class ProgressV3Projector(ProgressV2Projector):
    """Bounded latest-only v3 ingestion with the v2 cadence state machine."""

    def _context_path(self, attempt_id: str) -> Path:
        return _context_path(self.cfg, attempt_id)

    def _observed_path(self, attempt_id: str) -> Path:
        return _observed_path(self.cfg, attempt_id)

    def _shared_path(self, task_id: str, attempt_id: str) -> Path:
        return _shared_path(self.cfg, task_id, attempt_id)

    def _mailbox_path(self, attempt_id: str) -> Path:
        return _local_mailbox_path(self.cfg.runtime_root, attempt_id)

    def _context_directory(self) -> str:
        return _CONTEXT_DIRECTORY

    def _observed_directory(self) -> str:
        return _OBSERVED_DIRECTORY

    def _mailbox_name(self) -> str:
        return _LOCAL_MAILBOX_NAME

    def _coordinator_directory(self) -> str:
        return _COORDINATOR_DIRECTORY

    def _validate_context(self, value: Any, attempt_id: str) -> dict[str, Any]:
        return _validate_context(value, attempt_id)

    def _validate_payload(self, value: Any) -> dict[str, Any]:
        return validate_payload_v3(value)

    def _validate_projection(
        self,
        value: Any,
        identity: dict[str, Any],
        *,
        require_token: bool = False,
        require_generation: bool = False,
    ) -> dict[str, Any]:
        return _validate_projection_v3(
            value,
            identity,
            require_token=require_token,
            require_generation=require_generation,
        )

    def _same_content(self, payload: dict[str, Any], projection: dict[str, Any]) -> bool:
        return _same_content(payload, projection)

    def _protocol_version(self) -> int:
        return 3

    def _advanced_content_key(self, payload: dict[str, Any]) -> tuple[Any, ...]:
        activity = payload.get("activity") or {}
        overall = payload.get("overall")
        overall_key = None if overall is None else tuple(overall.get(key) for key in ("current", "total", "unit"))
        return (
            *(activity.get(key) for key in ("stage", "current", "total", "unit")),
            overall_key,
        )

    def _payload_max_bytes(self) -> int:
        return MAX_PAYLOAD_V3_BYTES

    def _snapshot_max_bytes(self) -> int:
        return MAX_SNAPSHOT_V3_BYTES


__all__ = [
    "ProgressV3Projector",
    "_context_path",
    "_local_mailbox_path",
    "_local_path",
    "_observed_path",
    "_shared_path",
    "_validate_context",
    "_validate_projection_v3",
    "prepare_progress_v3_channel",
]
