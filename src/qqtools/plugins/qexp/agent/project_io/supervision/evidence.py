"""Read and identify local runner evidence shared by publication and reconciliation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ....runtime.authority_scan import validate_evidence_path
from ....runtime.records import validate_identifier
from ....runtime.store import read_json_limited
from ...context import ProjectBinding
from ...process_reservation_recovery import registration_reservation
from ...project_io_process import PROCESS_IDENTITY_FIELDS

SOURCE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError)


@dataclass(frozen=True, slots=True)
class RunnerSource:
    """One validated local registration or launch intent."""

    record: dict[str, Any]
    parameters: dict[str, Any]
    identity: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """One validated local process manifest and its full canonical identity."""

    record: dict[str, Any]
    identity: str
    parameters: dict[str, Any]


def canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def is_valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _read_record(path: Path, project_root: Path, key: str, record_type: str) -> dict[str, Any] | None:
    if not validate_evidence_path(path, project_root):
        return None
    envelope = read_json_limited(path, max_bytes=65_536, record_type=record_type)
    if set(envelope) != {key}:
        return None
    record = envelope[key]
    if (
        not isinstance(record, dict)
        or type(record.get("protocol_version")) is not int
        or record["protocol_version"] != 1
    ):
        return None
    return record


def _attempt_number(record: Mapping[str, Any], path: Path) -> int | None:
    task_id = record.get("task_id")
    attempt_id = record.get("attempt_id")
    if not isinstance(task_id, str) or not isinstance(attempt_id, str) or path.stem != attempt_id:
        return None
    try:
        validate_identifier(task_id, "task_id")
        validate_identifier(attempt_id, "attempt_id")
    except ValueError:
        return None
    prefix = f"{task_id}-attempt-"
    suffix = attempt_id.removeprefix(prefix)
    if not suffix.isascii() or not suffix.isdecimal():
        return None
    number = int(suffix)
    return number if number >= 1 and attempt_id == f"{prefix}{number}" else None


def _process_identity(record: Mapping[str, Any]) -> dict[str, int | None] | None:
    process_identity: dict[str, int | None] = {}
    for field in PROCESS_IDENTITY_FIELDS:
        if field not in record:
            return None
        value = record[field]
        positive = field in {"wrapper_pid", "process_group_id"}
        if value is not None and (type(value) is not int or value < (1 if positive else 0)):
            return None
        process_identity[field] = value

    return process_identity


def read_process_record(
    path: Path,
    project_root: Path,
    binding: ProjectBinding,
) -> ProcessRecord | None:
    """Read one exact local process manifest for natural-exit supervision."""
    record = _read_record(path, project_root, "process", "process_manifest")
    if record is None:
        return None
    attempt_number = _attempt_number(record, path)
    if attempt_number is None:
        return None
    task_id, attempt_id = record["task_id"], record["attempt_id"]
    fencing_token = record.get("fencing_token")
    reservation_id = record.get("reservation_id")
    if (
        record.get("machine_name") != binding.machine_name
        or type(fencing_token) is not int
        or fencing_token < 1
        or record.get("observed_state") not in {"running", "exited"}
    ):
        return None
    if reservation_id is not None:
        try:
            validate_identifier(reservation_id, "process reservation_id")
        except ValueError:
            return None

    process_identity = _process_identity(record)
    if process_identity is None:
        return None

    parameters = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "reservation_id": reservation_id,
        "process_identity": process_identity,
    }
    return ProcessRecord(
        record=record,
        identity=canonical_record(record),
        parameters=parameters,
    )


def read_registration(
    path: Path,
    project_root: Path,
    binding: ProjectBinding,
    *,
    allow_unassigned_reservation: bool = False,
) -> RunnerSource | None:
    """Read one bounded runner registration and derive its canonical Attempt number."""
    record = _read_record(path, project_root, "process_registration", "process_registration")
    if record is None:
        return None
    attempt_number = _attempt_number(record, path)
    if attempt_number is None:
        return None
    task_id, attempt_id = record["task_id"], record["attempt_id"]
    fencing_token = record.get("fencing_token")
    reservation_id = registration_reservation(record, project_root)
    process_created_at = record.get("process_created_at")
    if record.get("machine_name") != binding.machine_name or type(fencing_token) is not int or fencing_token < 1:
        return None
    if not is_valid_utc_timestamp(process_created_at):
        return None
    if reservation_id is None:
        if not allow_unassigned_reservation:
            return None
    elif isinstance(reservation_id, str):
        try:
            validate_identifier(reservation_id, "process reservation_id")
        except ValueError:
            return None
    else:
        return None
    process_identity = _process_identity(record)
    if process_identity is None:
        return None

    parameters = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "reservation_id": reservation_id,
        "process_identity": process_identity,
        "process_created_at": process_created_at,
    }
    identity = (
        task_id,
        attempt_id,
        attempt_number,
        fencing_token,
        binding.machine_name,
        process_created_at,
        *(process_identity[field] for field in PROCESS_IDENTITY_FIELDS),
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return RunnerSource(record=record, parameters=parameters, identity=identity)


def read_launch_intent(
    path: Path,
    project_root: Path,
) -> RunnerSource | None:
    """Read the smaller pre-child Runner envelope without inventing registration fields."""
    record = _read_record(path, project_root, "launch_intent", "launch_intent")
    if record is None or _attempt_number(record, path) is None:
        return None
    task_id, attempt_id = record["task_id"], record["attempt_id"]
    fencing_token = record.get("fencing_token")
    if (
        type(fencing_token) is not int
        or fencing_token < 1
        or "wrapper_pid" not in record
        or "wrapper_start_time_ticks" not in record
    ):
        return None
    wrapper_pid = record.get("wrapper_pid")
    wrapper_ticks = record.get("wrapper_start_time_ticks")
    if wrapper_pid is not None and (type(wrapper_pid) is not int or wrapper_pid < 1):
        return None
    if wrapper_ticks is not None and (type(wrapper_ticks) is not int or wrapper_ticks < 0):
        return None
    identity = (
        task_id,
        attempt_id,
        fencing_token,
        wrapper_pid,
        wrapper_ticks,
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return RunnerSource(record=record, parameters={}, identity=identity)
