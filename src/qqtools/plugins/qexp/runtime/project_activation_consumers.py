"""Durable contiguous progress for Project activation consumers."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .locks import exclusive
from .project_activation import (
    _increment_membership_revision_locked,
    _MissingActivationEvent,
    read_project_activation,
    read_project_activation_events,
    read_project_activation_snapshot,
)
from .records import utc_now
from .store import atomic_replace, read_json_limited

_MAX_RECORD_BYTES = 16 * 1024
_MAX_ACK_EVENTS = 256
_ACTIVATION_DIRECTORY = "project-activation-v1"
_CONSUMER_FIELDS = frozenset({"version", "identity", "process_fence", "ack", "state", "updated_at"})
_IDENTITY_FIELDS = frozenset({"runtime_id", "project_id", "registration_generation"})
_ACK_FIELDS = frozenset({"epoch", "sequence"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)


def consumer_progress_path(root: Path, runtime_id: str, registration_generation: str) -> Path:
    """Return the durable consumer progress path for one registration generation."""
    runtime_id = _validate_identifier(runtime_id, "runtime_id")
    registration_generation = _validate_identifier(registration_generation, "registration_generation")
    return (
        Path(root) / "operations" / _ACTIVATION_DIRECTORY / "consumers" / runtime_id / f"{registration_generation}.json"
    )


def read_consumer_progress(
    root: Path,
    *,
    runtime_id: str,
    project_id: str,
    registration_generation: str,
) -> dict[str, Any] | None:
    """Read and validate one bounded consumer progress record without locking."""
    identity = _validate_identity(runtime_id, project_id, registration_generation)
    root = Path(root)
    path = consumer_progress_path(root, runtime_id, registration_generation)
    if not _validate_directory_chain(root, path.parent, create=False):
        return None
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    _require_regular_record(path, metadata)
    value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES, record_type="project_activation_consumer")
    return _validate_consumer_record(value, path=path, root=root, identity=identity)


def register_consumer(
    root: Path,
    *,
    runtime_id: str,
    project_id: str,
    registration_generation: str,
    process_fence: str,
) -> dict[str, Any]:
    """Fence a consumer process, preserving progress only within the current epoch."""
    identity = _validate_identity(runtime_id, project_id, registration_generation)
    process_fence = _validate_identifier(process_fence, "process_fence")
    root = Path(root)
    path = consumer_progress_path(root, runtime_id, registration_generation)

    with exclusive(root / "locks" / "project-activation-v1.lock"):
        try:
            checkpoint = read_project_activation(root)
        except _MissingActivationEvent:
            # Register a fenced but unacknowledged consumer so the owning binding
            # remains visible and resident while its activation journal is repaired.
            checkpoint = None
        current_epoch = _checkpoint_epoch(checkpoint, project_id)
        existing = _read_consumer_record(root, path, identity)
        if existing is not None and existing["project_activation_consumer"]["state"] == "retired":
            raise ValueError("A retired Project activation consumer generation cannot be registered again.")
        if existing is None:
            _increment_membership_revision_locked(root, project_id)

        acknowledgement = None
        if existing is not None:
            previous = existing["project_activation_consumer"]["ack"]
            if previous is not None and current_epoch is not None and previous["epoch"] == current_epoch:
                acknowledgement = previous.copy()

        record = _build_record(identity, process_fence, acknowledgement, "active")
        return _write_consumer_record(root, path, identity, record)


def ack_consumer(
    root: Path,
    *,
    runtime_id: str,
    project_id: str,
    registration_generation: str,
    process_fence: str,
    epoch: str,
    sequence: int,
    reconstructed_floor: int | None = None,
    require_current: bool = False,
) -> dict[str, Any]:
    """Persist contiguous progress after local active/waiting state is durably handed off."""
    identity = _validate_identity(runtime_id, project_id, registration_generation)
    process_fence = _validate_identifier(process_fence, "process_fence")
    epoch = _validate_epoch(epoch)
    sequence = _validate_sequence(sequence)
    if reconstructed_floor is not None:
        reconstructed_floor = _validate_sequence(reconstructed_floor)
    root = Path(root)
    path = consumer_progress_path(root, runtime_id, registration_generation)

    with exclusive(root / "locks" / "project-activation-v1.lock"):
        record = _read_consumer_record(root, path, identity)
        if record is None:
            raise ValueError("Project activation consumer is not registered.")
        progress = record["project_activation_consumer"]
        _require_active_fence(progress, process_fence)

        checkpoint = read_project_activation(root)
        current = _checkpoint_progress(checkpoint, project_id)
        if current is None or current["epoch"] != epoch or sequence > current["sequence"]:
            raise ValueError("Project activation acknowledgement exceeds or mismatches the current checkpoint.")
        if require_current and sequence != current["sequence"]:
            raise ValueError("Project activation changed before the consumer could enter dormancy.")

        snapshot = read_project_activation_snapshot(root)
        snapshot_record = snapshot["project_activation_snapshot"] if snapshot is not None else None
        current_floor = 0
        if snapshot_record is not None and snapshot_record["epoch"] == epoch:
            current_floor = snapshot_record["floor_sequence"]
        if reconstructed_floor is not None:
            if snapshot_record is None or snapshot_record["epoch"] != epoch:
                raise ValueError("Project activation reconstruction does not match the current snapshot epoch.")
            if reconstructed_floor != current_floor:
                raise ValueError("Project activation reconstruction floor does not match the current snapshot.")

        previous = progress["ack"]
        previous_sequence = 0
        if previous is not None:
            if previous["epoch"] != epoch:
                raise ValueError("Project activation epoch changed; register the consumer before acknowledging it.")
            if sequence < previous["sequence"]:
                raise ValueError("Project activation acknowledgement cannot move backwards.")
            previous_sequence = previous["sequence"]
        replay_start = previous_sequence
        if current_floor > previous_sequence:
            if reconstructed_floor is None:
                raise ValueError("Project activation consumer requires reconstruction below the retention floor.")
            if sequence < current_floor:
                raise ValueError("Project activation acknowledgement cannot stop below the reconstruction floor.")
            replay_start = current_floor
        if sequence - replay_start > _MAX_ACK_EVENTS:
            raise ValueError("Project activation acknowledgement exceeds one bounded replay batch.")
        if sequence > replay_start:
            events = read_project_activation_events(
                root, epoch=epoch, after_sequence=replay_start, limit=sequence - replay_start
            )
            if len(events) != sequence - replay_start:
                raise ValueError("Project activation acknowledgement has a missing event.")

        updated = _build_record(identity, process_fence, {"epoch": epoch, "sequence": sequence}, "active")
        return _write_consumer_record(root, path, identity, updated)


def retire_consumer(
    root: Path,
    *,
    runtime_id: str,
    project_id: str,
    registration_generation: str,
    process_fence: str,
) -> dict[str, Any]:
    """Retire a fenced generation after registration-removal safety is established by the caller."""
    identity = _validate_identity(runtime_id, project_id, registration_generation)
    process_fence = _validate_identifier(process_fence, "process_fence")
    root = Path(root)
    path = consumer_progress_path(root, runtime_id, registration_generation)

    with exclusive(root / "locks" / "project-activation-v1.lock"):
        record = _read_consumer_record(root, path, identity)
        if record is None:
            raise ValueError("Project activation consumer is not registered.")
        progress = record["project_activation_consumer"]
        if progress["state"] == "retired":
            if progress["process_fence"] != process_fence:
                raise ValueError("Project activation consumer process fence is stale.")
            return record
        _require_active_fence(progress, process_fence)
        _increment_membership_revision_locked(root, project_id)
        retired = _build_record(identity, process_fence, progress["ack"], "retired")
        return _write_consumer_record(root, path, identity, retired)


def retire_consumer_after_registration_removal(
    root: Path,
    *,
    runtime_id: str,
    project_id: str,
    registration_generation: str,
) -> dict[str, Any] | None:
    """Retire only this consumer after its exact machine binding was durably removed.

    The internal caller must hold the machine registry guard and have established that
    this exact `(runtime_id, project_id, registration_generation, shared_root)` identity
    is absent from the committed registry. This function does not establish that proof;
    it only applies it to the matching consumer record under the Project activation lock.
    Missing and already-retired consumers are successful so an interrupted removal can
    safely retry from its durable machine-local intent.
    """
    identity = _validate_identity(runtime_id, project_id, registration_generation)
    root = Path(root)
    path = consumer_progress_path(root, runtime_id, registration_generation)

    with _existing_activation_lock(root):
        record = _read_consumer_record(root, path, identity)
        if record is None:
            return None
        progress = record["project_activation_consumer"]
        if progress["state"] == "retired":
            return record
        _increment_membership_revision_locked(root, project_id)
        retired = _build_record(identity, progress["process_fence"], progress["ack"], "retired")
        return _write_consumer_record(root, path, identity, retired)


@contextmanager
def _existing_activation_lock(root: Path) -> Iterator[None]:
    """Lock an existing activation root without recreating a vanished mount."""
    root = Path(root)
    try:
        root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Project activation root is unavailable: {root}") from exc
    lock_descriptor: int | None = None
    locked = False
    try:
        opened_root = os.fstat(root_descriptor)
        if not stat.S_ISDIR(opened_root.st_mode):
            raise RuntimeError(f"Project activation root is not a real directory: {root}")
        locks = root / "locks"
        try:
            locks_metadata = locks.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Project activation lock directory is unavailable: {locks}") from exc
        if not stat.S_ISDIR(locks_metadata.st_mode):
            raise RuntimeError(f"Project activation lock directory is not a real directory: {locks}")
        try:
            lock_descriptor = os.open(
                locks / "project-activation-v1.lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Project activation root is unavailable: {root}") from exc
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise RuntimeError("Project activation lock must be a regular file.")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        locked = True
        _require_same_root(root, opened_root)
        yield
        _require_same_root(root, opened_root)
    finally:
        if locked and lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        os.close(root_descriptor)


def _require_same_root(root: Path, opened: os.stat_result) -> None:
    try:
        current = root.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Project activation root is unavailable: {root}") from exc
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino, current.st_ctime_ns) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_ctime_ns,
    ):
        raise RuntimeError(f"Project activation root changed while retiring a consumer: {root}")


def _validate_identifier(value: object, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be 1 to 128 ASCII letters, digits, underscores, or hyphens.")
    return value


def _validate_identity(runtime_id: str, project_id: str, registration_generation: str) -> dict[str, str]:
    return {
        "runtime_id": _validate_identifier(runtime_id, "runtime_id"),
        "project_id": _validate_identifier(project_id, "project_id"),
        "registration_generation": _validate_identifier(registration_generation, "registration_generation"),
    }


def _validate_epoch(value: object) -> str:
    if type(value) is not str:
        raise ValueError("Project activation epoch must be a canonical nonzero UUID hex value.")
    try:
        parsed = uuid.UUID(hex=value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("Project activation epoch must be a canonical nonzero UUID hex value.") from exc
    if parsed.int == 0 or value != parsed.hex:
        raise ValueError("Project activation epoch must be a canonical nonzero UUID hex value.")
    return value


def _validate_sequence(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Project activation acknowledgement sequence must be a nonnegative integer.")
    return value


def _validate_timestamp(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("Project activation consumer timestamp is invalid.")
    try:
        if len(value.encode("utf-8")) > 128:
            raise ValueError("Project activation consumer timestamp is invalid.")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Project activation consumer timestamp is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Project activation consumer timestamp must be UTC.")
    return value


def _validate_consumer_record(
    value: object,
    *,
    path: Path,
    root: Path,
    identity: dict[str, str],
) -> dict[str, Any]:
    expected_path = consumer_progress_path(
        root,
        identity["runtime_id"],
        identity["registration_generation"],
    )
    if path != expected_path:
        raise ValueError("Project activation consumer record path does not match its identity.")
    if type(value) is not dict or set(value) != {"project_activation_consumer"}:
        raise ValueError("Project activation consumer record has missing or unknown fields.")
    progress = value["project_activation_consumer"]
    if type(progress) is not dict or set(progress) != _CONSUMER_FIELDS:
        raise ValueError("Project activation consumer progress has missing or unknown fields.")
    if type(progress["version"]) is not int or progress["version"] != 1:
        raise ValueError("Project activation consumer version is unsupported.")
    stored_identity = progress["identity"]
    if type(stored_identity) is not dict or set(stored_identity) != _IDENTITY_FIELDS:
        raise ValueError("Project activation consumer identity is invalid.")
    for key, expected in identity.items():
        if type(stored_identity[key]) is not str or stored_identity[key] != expected:
            raise ValueError("Project activation consumer identity does not match the requested identity.")
    _validate_identifier(progress["process_fence"], "process_fence")
    acknowledgement = progress["ack"]
    if acknowledgement is not None:
        if type(acknowledgement) is not dict or set(acknowledgement) != _ACK_FIELDS:
            raise ValueError("Project activation consumer acknowledgement is invalid.")
        _validate_epoch(acknowledgement["epoch"])
        _validate_sequence(acknowledgement["sequence"])
    if type(progress["state"]) is not str or progress["state"] not in {"active", "retired"}:
        raise ValueError("Project activation consumer state is invalid.")
    _validate_timestamp(progress["updated_at"])
    return value


def _validate_directory_chain(root: Path, directory: Path, *, create: bool) -> bool:
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        if create:
            raise RuntimeError(f"Project activation root is missing: {root}") from None
        return False
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"Project activation root is not a real directory: {root}")
    try:
        parts = directory.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("Project activation consumer directory must be below the Project root.") from exc

    current = root
    for part in parts:
        current /= part
        if create:
            try:
                current.mkdir()
            except FileExistsError:
                pass
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if create:
                raise RuntimeError(f"Project activation consumer path is missing: {current}") from None
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project activation consumer path is not a real directory: {current}")
    return True


def _require_regular_record(path: Path, metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECORD_BYTES:
        raise RuntimeError(f"Project activation consumer record is not a bounded regular file: {path}")


def _read_consumer_record(root: Path, path: Path, identity: dict[str, str]) -> dict[str, Any] | None:
    if not _validate_directory_chain(root, path.parent, create=False):
        return None
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    _require_regular_record(path, metadata)
    value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES, record_type="project_activation_consumer")
    return _validate_consumer_record(value, path=path, root=root, identity=identity)


def _checkpoint_progress(checkpoint: dict[str, Any] | None, project_id: str) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    activation = checkpoint["project_activation"]
    if activation["identity"]["project_id"] != project_id:
        raise ValueError("Project activation checkpoint identity does not match the requested Project.")
    return {"epoch": activation["epoch"], "sequence": activation["sequence"]}


def _checkpoint_epoch(checkpoint: dict[str, Any] | None, project_id: str) -> str | None:
    current = _checkpoint_progress(checkpoint, project_id)
    return None if current is None else current["epoch"]


def _require_active_fence(progress: dict[str, Any], process_fence: str) -> None:
    if progress["state"] != "active":
        raise ValueError("Project activation consumer is retired.")
    if progress["process_fence"] != process_fence:
        raise ValueError("Project activation consumer process fence is stale.")


def _build_record(
    identity: dict[str, str],
    process_fence: str,
    acknowledgement: dict[str, Any] | None,
    state: str,
) -> dict[str, Any]:
    return {
        "project_activation_consumer": {
            "version": 1,
            "identity": identity.copy(),
            "process_fence": process_fence,
            "ack": None if acknowledgement is None else acknowledgement.copy(),
            "state": state,
            "updated_at": utc_now(),
        }
    }


def _write_consumer_record(
    root: Path,
    path: Path,
    identity: dict[str, str],
    record: dict[str, Any],
) -> dict[str, Any]:
    if not _validate_directory_chain(root, path.parent, create=True):
        raise RuntimeError(f"Project activation consumer directory is unavailable: {path.parent}")
    _validate_consumer_record(record, path=path, root=root, identity=identity)
    encoded_size = len(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    if encoded_size > _MAX_RECORD_BYTES:
        raise ValueError(f"Project activation consumer record exceeds {_MAX_RECORD_BYTES} bytes.")
    replaced = atomic_replace(path, record)
    if replaced is None or not stat.S_ISREG(replaced.st_mode) or replaced.st_size > _MAX_RECORD_BYTES:
        raise RuntimeError(f"Project activation consumer record could not be durably replaced: {path}")
    _sync_directory_chain_to_root(path.parent, root)
    return record


def _sync_directory_chain_to_root(directory: Path, root: Path) -> None:
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("Project activation consumer directory must be below the Project root.") from exc
    current = directory
    while True:
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project activation consumer path is not a real directory: {current}")
        descriptor = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if current == root:
            return
        current = current.parent
