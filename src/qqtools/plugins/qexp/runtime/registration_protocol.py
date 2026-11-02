"""Registration protocol validation and bounded replacement reservations."""

from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..lease import parse_utc
from .paths import machine_registration_path, shared_paths

TARGET_REGISTRATION_VERSION = 3
REGISTRATION_PROTOCOL = "qexp-registration-replacement-v1"
RECOVERY_PROTOCOL = "qexp-local-responsibility-v1"
# QQTOOLS-COMPAT-0024: only the pinned source package formats may enter conversion.
SOURCE_REGISTRATION_VERSIONS = frozenset({1, 2})
RESERVATION_VERSION = 1
MAX_REGISTRATION_RECORD_BYTES = 64 * 1024


def _require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"project machine registration has an invalid {field}.")


def _validate_identity(record: dict[str, Any], project_id: str, cfg: RootConfig) -> None:
    if record.get("project_id") != project_id or record.get("shared_root") != str(cfg.shared_root):
        raise RuntimeError("project machine registration does not match Project identity.")
    if record.get("machine_name") != cfg.machine_name:
        raise RuntimeError("project machine registration does not match the requested logical name.")
    for field in ("generation", "runtime_instance_id", "runtime_root", "eligibility_expires_at"):
        _require_nonempty_string(record.get(field), field)
    try:
        expiry = parse_utc(record["eligibility_expires_at"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("project machine registration has an invalid eligibility expiry.") from exc
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        raise RuntimeError("project machine registration has an invalid eligibility expiry.")
    if record.get("state") not in {"eligible", "superseded"}:
        raise RuntimeError("project machine registration has an unsupported state.")


def validate_registration(
    record: dict[str, Any],
    project_id: str,
    cfg: RootConfig,
    *,
    allow_source: bool = False,
) -> None:
    """Validate one registration envelope against its Project and machine name."""
    if not isinstance(record, dict):
        raise RuntimeError("project machine registration is malformed.")
    if len(json.dumps(record, ensure_ascii=True).encode("utf-8")) > MAX_REGISTRATION_RECORD_BYTES:
        raise RuntimeError("project machine registration exceeds its serialization limit.")
    version = record.get("version")
    if type(version) is not int:
        raise RuntimeError("project machine registration uses an unsupported protocol version.")
    if version in SOURCE_REGISTRATION_VERSIONS:
        if not allow_source:
            raise RuntimeError("project machine registration uses an unsupported protocol version.")
        if version == 1:
            protocol_version = record.get("protocol_version")
            if protocol_version is not None and (type(protocol_version) is not int or protocol_version != version):
                raise RuntimeError("project machine registration has an unsupported source protocol.")
            if record.get("recovery_protocol") not in {None, ""}:
                raise RuntimeError("project machine registration has an unsupported recovery protocol.")
        elif (
            type(record.get("protocol_version")) is not int
            or record.get("protocol_version") != version
            or record.get("recovery_protocol") != RECOVERY_PROTOCOL
        ):
            raise RuntimeError("project machine registration has an unsupported recovery protocol.")
        _validate_identity(record, project_id, cfg)
        return
    if version != TARGET_REGISTRATION_VERSION:
        raise RuntimeError("project machine registration uses an unsupported protocol version.")
    if (
        type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != TARGET_REGISTRATION_VERSION
        or record.get("registration_protocol") != REGISTRATION_PROTOCOL
    ):
        raise RuntimeError("project machine registration has an unsupported registration protocol.")
    if record.get("recovery_protocol") not in {None, RECOVERY_PROTOCOL}:
        raise RuntimeError("project machine registration has an unsupported recovery protocol.")
    _validate_identity(record, project_id, cfg)


def convert_registration(record: dict[str, Any], project_id: str, cfg: RootConfig) -> dict[str, Any]:
    """Return an idempotent target-protocol copy of a source or target record."""
    if not isinstance(record, dict):
        raise RuntimeError("project machine registration is malformed.")
    version = record.get("version")
    if version == TARGET_REGISTRATION_VERSION:
        validate_registration(record, project_id, cfg)
        return copy.deepcopy(record)
    validate_registration(record, project_id, cfg, allow_source=True)
    converted = copy.deepcopy(record)
    converted.update(
        {
            "version": TARGET_REGISTRATION_VERSION,
            "protocol_version": TARGET_REGISTRATION_VERSION,
            "registration_protocol": REGISTRATION_PROTOCOL,
            "recovery_protocol": RECOVERY_PROTOCOL if version == 2 else None,
        }
    )
    validate_registration(converted, project_id, cfg)
    return converted


def has_recovery_protocol(record: dict[str, Any]) -> bool:
    """Return whether a validated source or target marker grants recovery readiness."""
    if not isinstance(record, dict):
        return False
    version = record.get("version")
    if version == 2:
        return record.get("protocol_version") == 2 and record.get("recovery_protocol") == RECOVERY_PROTOCOL
    if version == TARGET_REGISTRATION_VERSION:
        return (
            record.get("protocol_version") == TARGET_REGISTRATION_VERSION
            and record.get("registration_protocol") == REGISTRATION_PROTOCOL
            and record.get("recovery_protocol") == RECOVERY_PROTOCOL
        )
    return False


def reservation_path(cfg: RootConfig) -> Path:
    """Return the exclusive reservation path for one requested machine name."""
    return shared_paths(cfg.shared_root)["project"] / "registration-reservations" / f"{cfg.machine_name}.json"


def _read_protocol_record(path: Path) -> dict[str, Any] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"project registration record is not a regular file: {path}")
    if metadata.st_size > MAX_REGISTRATION_RECORD_BYTES:
        raise RuntimeError(f"project registration record exceeds {MAX_REGISTRATION_RECORD_BYTES} bytes: {path}")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"project registration record cannot be read: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_REGISTRATION_RECORD_BYTES:
            raise RuntimeError(f"project registration record is invalid: {path}")
        chunks = []
        remaining = MAX_REGISTRATION_RECORD_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(encoded) > MAX_REGISTRATION_RECORD_BYTES:
        raise RuntimeError(f"project registration record exceeds {MAX_REGISTRATION_RECORD_BYTES} bytes: {path}")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"project registration record is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"project registration record is malformed: {path}")
    return value


def read_registration_document(cfg: RootConfig) -> dict[str, Any] | None:
    """Read the bounded registration envelope without following a record symlink."""
    return _read_protocol_record(machine_registration_path(cfg.shared_root, cfg.machine_name))


def _validate_reservation(value: dict[str, Any], cfg: RootConfig) -> dict[str, Any]:
    envelope = value.get("registration_reservation")
    if (
        not isinstance(envelope, dict)
        or type(envelope.get("version")) is not int
        or envelope["version"] != RESERVATION_VERSION
    ):
        raise RuntimeError("project registration reservation is malformed.")
    required = (
        "transaction_id",
        "project_id",
        "shared_root",
        "runtime_id",
        "runtime_root",
        "old_machine_name",
        "old_generation",
        "requested_machine_name",
        "target_generation",
    )
    for field in required:
        _require_nonempty_string(envelope.get(field), f"reservation {field}")
    if envelope["shared_root"] != str(cfg.shared_root) or envelope["requested_machine_name"] != cfg.machine_name:
        raise RuntimeError("project registration reservation does not match the requested Project name.")
    return envelope


def assert_name_unreserved(cfg: RootConfig, *, transaction_id: str | None = None) -> None:
    """Reject a malformed or competing reservation for the requested machine name."""
    value = _read_protocol_record(reservation_path(cfg))
    if value is None:
        return
    envelope = _validate_reservation(value, cfg)
    if transaction_id is not None and envelope["transaction_id"] == transaction_id:
        return
    raise RuntimeError(f"logical machine name {cfg.machine_name!r} is reserved by another registration replacement.")


__all__ = [
    "MAX_REGISTRATION_RECORD_BYTES",
    "RECOVERY_PROTOCOL",
    "REGISTRATION_PROTOCOL",
    "RESERVATION_VERSION",
    "SOURCE_REGISTRATION_VERSIONS",
    "TARGET_REGISTRATION_VERSION",
    "assert_name_unreserved",
    "convert_registration",
    "has_recovery_protocol",
    "reservation_path",
    "read_registration_document",
    "validate_registration",
]
