"""Machine-local Idle Job ownership and training-admission fencing."""

from __future__ import annotations

import os
import re
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from ..locks import exclusive
from ..paths import local_paths
from ..store import atomic_replace, read_json_limited, require_json_size

IDLE_JOB_RECORD_MAX_BYTES = 64 * 1024
_INSTANCE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_OWNERSHIP_FIELDS = frozenset(
    {
        "version",
        "runtime_id",
        "boot_id",
        "instance_id",
        "phase",
        "config_revision",
        "gpu_ids",
        "parent_pid",
        "parent_start_ticks",
        "created_at",
    }
)
_DEMAND_FIELDS = frozenset({"version", "revision"})
_PHASES = frozenset({"launching", "running", "stopping"})


class IdleJobAdmissionBlocked(ValueError):
    """Training admission is blocked until an owned Idle Job is cleaned up."""


def _runtime_root(value: Path | Any) -> Path:
    root = value if isinstance(value, Path) else getattr(value, "root", value)
    return Path(root).expanduser().resolve()


def _idle_job_root(root: Path | Any) -> Path:
    return _runtime_root(root) / "idle-job"


def _idle_job_ownership_path(root: Path | Any) -> Path:
    return _idle_job_root(root) / "ownership.json"


def _require_directory(path: Path, label: str) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise OSError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory: {path}")
    return True


def _read_record(path: Path, label: str) -> dict[str, Any] | None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OSError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return read_json_limited(path, max_bytes=IDLE_JOB_RECORD_MAX_BYTES, record_type=label)


def _require_bounded_text(value: object, label: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} must be a bounded nonempty string.")
    return value


def _require_uuid_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _INSTANCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase UUID hex string.")
    try:
        parsed = uuid.UUID(hex=value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase UUID hex string.") from exc
    if parsed.hex != value:
        raise ValueError(f"{label} must be a lowercase UUID hex string.")
    return value


def _require_uuid(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a canonical UUID.")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID.") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID.")
    return value


def _require_utc_text(value: object, label: str) -> str:
    text = _require_bounded_text(value, label, maximum=128)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp.")
    return text


def _validate_ownership(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"idle_job"} or not isinstance(value["idle_job"], dict):
        raise ValueError("Idle Job ownership envelope is malformed.")
    record = value["idle_job"]
    if set(record) != _OWNERSHIP_FIELDS:
        raise ValueError("Idle Job ownership fields are malformed.")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("Idle Job ownership version is invalid.")
    runtime_id = _require_bounded_text(record["runtime_id"], "Idle Job runtime_id")
    boot_id = _require_uuid(record["boot_id"], "Idle Job boot_id")
    instance_id = _require_uuid_hex(record["instance_id"], "Idle Job instance_id")
    if not isinstance(record["phase"], str) or record["phase"] not in _PHASES:
        raise ValueError("Idle Job ownership phase is invalid.")
    config_revision = record["config_revision"]
    if type(config_revision) is not int or config_revision < 0:
        raise ValueError("Idle Job config_revision must be a non-negative integer.")
    gpu_ids = record["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("Idle Job gpu_ids must be distinct non-negative integers.")
    parent_pid = record["parent_pid"]
    if type(parent_pid) is not int or parent_pid < 1:
        raise ValueError("Idle Job parent_pid must be a positive integer.")
    parent_start_ticks = record["parent_start_ticks"]
    if type(parent_start_ticks) is not int or parent_start_ticks < 0:
        raise ValueError("Idle Job parent_start_ticks must be a non-negative integer.")
    created_at = _require_utc_text(record["created_at"], "Idle Job created_at")
    return {
        "version": 1,
        "runtime_id": runtime_id,
        "boot_id": boot_id,
        "instance_id": instance_id,
        "phase": record["phase"],
        "config_revision": config_revision,
        "gpu_ids": list(gpu_ids),
        "parent_pid": parent_pid,
        "parent_start_ticks": parent_start_ticks,
        "created_at": created_at,
    }


def _ownership_envelope(record: dict[str, Any]) -> dict[str, Any]:
    validated = _validate_ownership({"idle_job": record})
    envelope = {"idle_job": validated}
    require_json_size(
        envelope,
        max_bytes=IDLE_JOB_RECORD_MAX_BYTES,
        record_type="idle_job_ownership",
    )
    return envelope


def _validate_demand(value: object) -> int:
    if not isinstance(value, dict) or set(value) != {"idle_job_demand"}:
        raise ValueError("Idle Job demand envelope is malformed.")
    record = value["idle_job_demand"]
    if not isinstance(record, dict) or set(record) != _DEMAND_FIELDS:
        raise ValueError("Idle Job demand fields are malformed.")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("Idle Job demand version is invalid.")
    revision = record["revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("Idle Job demand revision must be a non-negative integer.")
    return revision


def _demand_envelope(revision: int) -> dict[str, Any]:
    if type(revision) is not int or revision < 0:
        raise ValueError("Idle Job demand revision must be a non-negative integer.")
    envelope = {"idle_job_demand": {"version": 1, "revision": revision}}
    require_json_size(
        envelope,
        max_bytes=IDLE_JOB_RECORD_MAX_BYTES,
        record_type="idle_job_demand",
    )
    return envelope


@contextmanager
def idle_job_guard(root: Path | Any) -> Iterator[None]:
    """Hold the machine-wide fence for Idle Job and training admission."""
    paths = local_paths(_runtime_root(root))
    with exclusive(paths["locks"] / "idle-job-admission.lock"):
        yield


def idle_job_instance_path(root: Path | Any, instance_id: str) -> Path:
    """Return the derived stop/evidence directory for one validated instance."""
    validated = _require_uuid_hex(instance_id, "Idle Job instance_id")
    return _idle_job_root(root) / "instances" / validated


def read_idle_job_ownership(root: Path | Any) -> dict[str, Any] | None:
    """Read and validate the permanent Idle Job ownership record."""
    idle_root = _idle_job_root(root)
    if not _require_directory(idle_root, "Idle Job root"):
        return None
    value = _read_record(idle_root / "ownership.json", "idle_job_ownership")
    if value is None:
        return None
    return _validate_ownership(value)


def write_idle_job_ownership(root: Path | Any, record: dict[str, Any]) -> None:
    """Validate and atomically publish ownership; caller must hold idle_job_guard."""
    envelope = _ownership_envelope(record)
    atomic_replace(_idle_job_ownership_path(root), envelope)


def clear_idle_job_ownership(root: Path | Any, instance_id: str) -> None:
    """Delete ownership only when its current identity exactly matches instance_id."""
    validated = _require_uuid_hex(instance_id, "Idle Job instance_id")
    current = read_idle_job_ownership(root)
    if current is None or current["instance_id"] != validated:
        return
    path = _idle_job_ownership_path(root)
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Idle Job ownership must be a regular file: {path}")
    path.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_idle_job_demand(root: Path | Any) -> int:
    """Read the monotonic Idle Job training-demand revision."""
    idle_root = _idle_job_root(root)
    if not _require_directory(idle_root, "Idle Job root"):
        return 0
    value = _read_record(idle_root / "demand.json", "idle_job_demand")
    if value is None:
        return 0
    return _validate_demand(value)


def require_idle_job_released(root: Path | Any) -> None:
    """Record training demand and block admission while Idle Job ownership remains."""
    idle_root = _idle_job_root(root)
    if not _require_directory(idle_root, "Idle Job root"):
        return
    revision = read_idle_job_demand(root)
    atomic_replace(
        idle_root / "demand.json",
        _demand_envelope(revision + 1),
    )
    ownership = read_idle_job_ownership(root)
    if ownership is None:
        return
    stop_path = idle_job_instance_path(root, ownership["instance_id"]) / "stop.json"
    atomic_replace(stop_path, {"reason": "training_demand"})
    stopping = dict(ownership)
    stopping["phase"] = "stopping"
    write_idle_job_ownership(root, stopping)
    raise IdleJobAdmissionBlocked("Idle Job cleanup pending; training admission blocked")


__all__ = [
    "IDLE_JOB_RECORD_MAX_BYTES",
    "IdleJobAdmissionBlocked",
    "clear_idle_job_ownership",
    "idle_job_guard",
    "idle_job_instance_path",
    "read_idle_job_demand",
    "read_idle_job_ownership",
    "require_idle_job_released",
    "write_idle_job_ownership",
]
