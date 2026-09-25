"""Durable, project-scoped activation checkpoint for dormant bindings."""

from __future__ import annotations

import json
import os
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .locks import exclusive
from .paths import shared_paths
from .records import utc_now
from .store import atomic_replace, require_json_size

ACTIVATION_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024
_MAX_SEQUENCE = (1 << 63) - 1
_CHECKPOINT_FIELDS = frozenset({"version", "identity", "epoch", "sequence", "reason", "updated_at"})
_EVENT_FIELDS = frozenset({"version", "identity", "epoch", "sequence", "reason", "updated_at"})
_PENDING_FIELDS = frozenset({"version", "identity", "epoch", "sequence", "reason", "updated_at"})
_SNAPSHOT_FIELDS = frozenset(
    {
        "version",
        "identity",
        "epoch",
        "previous_floor_sequence",
        "floor_sequence",
        "checkpoint_sequence",
        "membership_revision",
        "coverage",
        "updated_at",
    }
)
_MEMBERSHIP_FIELDS = frozenset({"version", "identity", "revision", "updated_at"})
_MAX_EVENT_BATCH = 256
_COMPACTION_BATCH = 256
_MAX_ORPHAN_EPOCH_SCAN = 64
_RECONSTRUCTION_COVERAGE = {
    "kind": "authoritative_indexes",
    "service_lanes": ["scheduler", "authority", "group", "observation", "submission"],
    "work_states": ["active", "waiting"],
}


class _MissingActivationEvent(RuntimeError):
    """A checkpoint needs conservative epoch reconstruction before replay."""


def activation_checkpoint_path(root: Path) -> Path:
    """Return the fixed durable activation checkpoint path for a Project."""
    return Path(root) / "operations" / "project-activation-v1" / "checkpoint.json"


def activation_pending_path(root: Path) -> Path:
    """Return the fixed durable pending activation path for a Project."""
    return Path(root) / "operations" / "project-activation-v1" / "pending.json"


def activation_snapshot_path(root: Path) -> Path:
    """Return the fixed durable activation reconstruction snapshot path."""
    return Path(root) / "operations" / "project-activation-v1" / "snapshot.json"


def activation_membership_path(root: Path) -> Path:
    """Return the fixed durable activation consumer membership revision path."""
    return Path(root) / "operations" / "project-activation-v1" / "membership.json"


def activation_event_path(root: Path, epoch: str, sequence: int) -> Path:
    """Return the fixed durable path for one Project activation event."""
    epoch = _validate_epoch(epoch)
    if type(sequence) is not int or not 1 <= sequence <= _MAX_SEQUENCE:
        raise ValueError("Project activation event sequence is invalid.")
    return Path(root) / "operations" / "project-activation-v1" / "events" / epoch / f"{sequence:020d}.json"


def _read_regular_json(path: Path) -> dict[str, Any]:
    """Read one bounded JSON object without following links or special files."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RECORD_BYTES:
        raise RuntimeError(f"Project activation record is not a bounded regular file: {path}")

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RECORD_BYTES:
            raise RuntimeError(f"Project activation record is not a bounded regular file: {path}")
        encoded = bytearray()
        while len(encoded) <= MAX_RECORD_BYTES:
            chunk = os.read(descriptor, min(8192, MAX_RECORD_BYTES + 1 - len(encoded)))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) > MAX_RECORD_BYTES:
            raise RuntimeError(f"Project activation record exceeds {MAX_RECORD_BYTES} bytes: {path}")
    finally:
        os.close(descriptor)

    value = json.loads(encoded.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError(f"Project activation record must be a JSON object: {path}")
    return value


def _project_id(root: Path) -> str:
    identity_path = shared_paths(root)["project"] / "identity.json"
    try:
        value = _read_regular_json(identity_path)
        project = value.get("project")
        project_id = project.get("project_id") if type(project) is dict else None
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Project identity is malformed: {identity_path}") from exc
    if type(project_id) is not str or not project_id:
        raise RuntimeError(f"Project identity is malformed: {identity_path}")
    return project_id


def _validate_reason(reason: object) -> str:
    if type(reason) is not str or not reason.strip():
        raise ValueError("Project activation reason must be a nonempty string.")
    try:
        encoded = reason.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Project activation reason must be valid UTF-8.") from exc
    if len(encoded) > 128 or any(ord(character) < 0x20 or ord(character) == 0x7F for character in reason):
        raise ValueError("Project activation reason must be at most 128 bytes without control characters.")
    return reason


def _validate_timestamp(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("Project activation timestamp is invalid.")
    try:
        if len(value.encode("utf-8")) > 128:
            raise ValueError("Project activation timestamp is invalid.")
    except UnicodeEncodeError as exc:
        raise ValueError("Project activation timestamp is invalid.") from exc
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Project activation timestamp is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Project activation timestamp must include a timezone.")
    return value


def _validate_epoch(value: object) -> str:
    if type(value) is not str:
        raise ValueError("Project activation epoch is invalid.")
    try:
        parsed = uuid.UUID(hex=value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Project activation epoch is invalid.") from exc
    if parsed.int == 0 or value != parsed.hex:
        raise ValueError("Project activation epoch must be a canonical nonzero UUID hex value.")
    return value


def _validate_checkpoint(value: object, project_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"project_activation"}:
        raise ValueError("Project activation checkpoint has missing or unknown fields.")
    checkpoint = value["project_activation"]
    if type(checkpoint) is not dict or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("Project activation record has missing or unknown fields.")
    if type(checkpoint["version"]) is not int or checkpoint["version"] != ACTIVATION_VERSION:
        raise ValueError("Project activation version is unsupported.")
    identity = checkpoint["identity"]
    if type(identity) is not dict or set(identity) != {"project_id"}:
        raise ValueError("Project activation identity is invalid.")
    if type(identity["project_id"]) is not str or identity["project_id"] != project_id:
        raise ValueError("Project activation identity does not match the current Project.")
    _validate_epoch(checkpoint["epoch"])
    sequence = checkpoint["sequence"]
    if type(sequence) is not int or not 1 <= sequence <= _MAX_SEQUENCE:
        raise ValueError("Project activation sequence is invalid.")
    _validate_reason(checkpoint["reason"])
    _validate_timestamp(checkpoint["updated_at"])
    return value


def _validate_event(
    value: object,
    *,
    project_id: str,
    epoch: str,
    sequence: int,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"project_activation_event"}:
        raise ValueError("Project activation event has missing or unknown fields.")
    event = value["project_activation_event"]
    if type(event) is not dict or set(event) != _EVENT_FIELDS:
        raise ValueError("Project activation event record has missing or unknown fields.")
    if type(event["version"]) is not int or event["version"] != ACTIVATION_VERSION:
        raise ValueError("Project activation event version is unsupported.")
    identity = event["identity"]
    if type(identity) is not dict or set(identity) != {"project_id"}:
        raise ValueError("Project activation event identity is invalid.")
    if type(identity["project_id"]) is not str or identity["project_id"] != project_id:
        raise ValueError("Project activation event identity does not match the current Project.")
    if _validate_epoch(event["epoch"]) != epoch:
        raise ValueError("Project activation event epoch does not match the requested epoch.")
    if type(event["sequence"]) is not int or event["sequence"] != sequence:
        raise ValueError("Project activation event sequence is not contiguous.")
    _validate_reason(event["reason"])
    _validate_timestamp(event["updated_at"])
    return value


def _validate_snapshot(value: object, project_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"project_activation_snapshot"}:
        raise ValueError("Project activation snapshot has missing or unknown fields.")
    snapshot = value["project_activation_snapshot"]
    if type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("Project activation snapshot has missing or unknown fields.")
    if type(snapshot["version"]) is not int or snapshot["version"] != ACTIVATION_VERSION:
        raise ValueError("Project activation snapshot version is unsupported.")
    identity = snapshot["identity"]
    if type(identity) is not dict or set(identity) != {"project_id"}:
        raise ValueError("Project activation snapshot identity is invalid.")
    if type(identity["project_id"]) is not str or identity["project_id"] != project_id:
        raise ValueError("Project activation snapshot identity does not match the current Project.")
    _validate_epoch(snapshot["epoch"])
    floor = snapshot["floor_sequence"]
    previous_floor = snapshot["previous_floor_sequence"]
    checkpoint_sequence = snapshot["checkpoint_sequence"]
    revision = snapshot["membership_revision"]
    if type(floor) is not int or not 0 <= floor <= _MAX_SEQUENCE:
        raise ValueError("Project activation snapshot retention floor is invalid.")
    if type(previous_floor) is not int or not 0 <= previous_floor <= floor:
        raise ValueError("Project activation snapshot previous floor is invalid.")
    if floor - previous_floor > _COMPACTION_BATCH:
        raise ValueError("Project activation snapshot compaction batch exceeds its bound.")
    if type(checkpoint_sequence) is not int or not 1 <= checkpoint_sequence <= _MAX_SEQUENCE:
        raise ValueError("Project activation snapshot checkpoint sequence is invalid.")
    if floor > checkpoint_sequence:
        raise ValueError("Project activation snapshot floor exceeds its checkpoint sequence.")
    if type(revision) is not int or not 0 <= revision <= _MAX_SEQUENCE:
        raise ValueError("Project activation snapshot membership revision is invalid.")
    if snapshot["coverage"] != _RECONSTRUCTION_COVERAGE:
        raise ValueError("Project activation snapshot reconstruction coverage is unsupported.")
    _validate_timestamp(snapshot["updated_at"])
    return value


def _validate_membership(value: object, project_id: str) -> int:
    if type(value) is not dict or set(value) != {"project_activation_membership"}:
        raise ValueError("Project activation membership record has missing or unknown fields.")
    membership = value["project_activation_membership"]
    if type(membership) is not dict or set(membership) != _MEMBERSHIP_FIELDS:
        raise ValueError("Project activation membership record has missing or unknown fields.")
    if type(membership["version"]) is not int or membership["version"] != ACTIVATION_VERSION:
        raise ValueError("Project activation membership version is unsupported.")
    identity = membership["identity"]
    if type(identity) is not dict or set(identity) != {"project_id"}:
        raise ValueError("Project activation membership identity is invalid.")
    if type(identity["project_id"]) is not str or identity["project_id"] != project_id:
        raise ValueError("Project activation membership identity does not match the current Project.")
    revision = membership["revision"]
    if type(revision) is not int or not 0 <= revision <= _MAX_SEQUENCE:
        raise ValueError("Project activation membership revision is invalid.")
    _validate_timestamp(membership["updated_at"])
    return revision


def _validate_pending(value: object, project_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"project_activation_pending"}:
        raise ValueError("Project activation pending record has missing or unknown fields.")
    activation = value["project_activation_pending"]
    if type(activation) is not dict or set(activation) != _PENDING_FIELDS:
        raise ValueError("Project activation pending record has missing or unknown fields.")
    _validate_checkpoint({"project_activation": activation}, project_id)
    return value


def _assert_checkpoint_event(root: Path, checkpoint: dict[str, Any]) -> None:
    """Require the checkpoint's exact committed event to be durably readable."""
    activation = checkpoint["project_activation"]
    epoch = activation["epoch"]
    sequence = activation["sequence"]
    path = activation_event_path(root, epoch, sequence)
    try:
        event = _read_regular_json(path)
    except FileNotFoundError as exc:
        snapshot = _read_snapshot_record(root, activation["identity"]["project_id"])
        if _snapshot_covers_checkpoint(root, checkpoint, snapshot):
            return
        raise _MissingActivationEvent(f"Project activation checkpoint points to a missing event: {path}") from exc
    _validate_event_directories(root, epoch)
    _validate_event(
        event,
        project_id=activation["identity"]["project_id"],
        epoch=epoch,
        sequence=sequence,
    )
    if event["project_activation_event"] != activation:
        raise RuntimeError(f"Project activation checkpoint does not match its event: {path}")


def _read_checkpoint_record(root: Path, project_id: str) -> dict[str, Any] | None:
    _validate_checkpoint_directories(root)
    path = activation_checkpoint_path(root)
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        return None
    return _validate_checkpoint(value, project_id)


def _read_snapshot_record(root: Path, project_id: str) -> dict[str, Any] | None:
    _validate_checkpoint_directories(root)
    path = activation_snapshot_path(root)
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        return None
    return _validate_snapshot(value, project_id)


def _snapshot_covers_checkpoint(
    root: Path,
    checkpoint: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
) -> bool:
    current = checkpoint["project_activation"]
    if snapshot is None:
        snapshot = _read_snapshot_record(root, current["identity"]["project_id"])
    if snapshot is None:
        return False
    record = snapshot["project_activation_snapshot"]
    sequence = current["sequence"]
    return (
        record["identity"]["project_id"] == current["identity"]["project_id"]
        and record["epoch"] == current["epoch"]
        and record["floor_sequence"] >= sequence
        and record["checkpoint_sequence"] == sequence
    )


def read_project_activation_snapshot(root: Path) -> dict[str, Any] | None:
    """Read and validate the current bounded activation reconstruction snapshot."""
    root = Path(root)
    project_id = _project_id(root)
    snapshot = _read_snapshot_record(root, project_id)
    if snapshot is None:
        return None
    checkpoint = _read_checkpoint_record(root, project_id)
    if checkpoint is None:
        raise RuntimeError("Project activation snapshot has no matching checkpoint.")
    snapshot_record = snapshot["project_activation_snapshot"]
    current = checkpoint["project_activation"]
    if snapshot_record["epoch"] == current["epoch"] and (
        snapshot_record["checkpoint_sequence"] > current["sequence"]
        or snapshot_record["floor_sequence"] > current["sequence"]
    ):
        raise RuntimeError("Project activation snapshot is ahead of the current checkpoint.")
    return snapshot


def _read_project_activation(root: Path, project_id: str) -> dict[str, Any] | None:
    checkpoint = _read_checkpoint_record(root, project_id)
    if checkpoint is None:
        if _read_snapshot_record(root, project_id) is not None:
            raise RuntimeError("Project activation snapshot has no matching checkpoint.")
        return None
    _assert_checkpoint_event(root, checkpoint)
    return checkpoint


def read_project_activation(root: Path) -> dict[str, Any] | None:
    """Read the bounded checkpoint and verify its Project identity."""
    root = Path(root)
    _validate_checkpoint_directories(root)
    path = activation_checkpoint_path(root)
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        try:
            activation_snapshot_path(root).lstat()
        except FileNotFoundError:
            return None
        if _read_snapshot_record(root, _project_id(root)) is not None:
            raise RuntimeError("Project activation snapshot has no matching checkpoint.")
        return None
    checkpoint = _validate_checkpoint(value, _project_id(root))
    _assert_checkpoint_event(root, checkpoint)
    return checkpoint


def _validate_checkpoint_directories(root: Path) -> None:
    for directory in (root / "operations", root / "operations" / "project-activation-v1"):
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project activation path is not a real directory: {directory}")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_real_directory_chain(root: Path, directory: Path) -> None:
    """Create activation directories below root, rejecting links and special files."""
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"Project activation root is not a real directory: {root}")
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("Project activation directory must be below the Project root.") from exc

    missing: list[Path] = []
    candidate = directory
    while candidate != root:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
        else:
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"Project activation path is not a real directory: {candidate}")
        candidate = candidate.parent

    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            pass
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project activation path is not a real directory: {path}")


def _validate_event_directories(root: Path, epoch: str) -> None:
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"Project activation root is not a real directory: {root}")
    directories = (
        root / "operations",
        root / "operations" / "project-activation-v1",
        root / "operations" / "project-activation-v1" / "events",
        root / "operations" / "project-activation-v1" / "events" / epoch,
    )
    for directory in directories:
        try:
            metadata = directory.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"Project activation event directory is missing: {directory}") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project activation path is not a real directory: {directory}")


def _sync_directory_chain_to_root(directory: Path, root: Path) -> None:
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("Project activation directory must be below the Project root.") from exc
    current = directory
    while True:
        _sync_directory(current)
        if current == root:
            return
        current = current.parent


def read_project_activation_events(
    root: Path,
    *,
    epoch: str,
    after_sequence: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Read a bounded consecutive suffix of Project activation events."""
    root = Path(root)
    epoch = _validate_epoch(epoch)
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("Project activation event after_sequence must be a nonnegative integer.")
    if type(limit) is not int or limit <= 0:
        raise ValueError("Project activation event limit must be a positive integer.")
    limit = min(limit, _MAX_EVENT_BATCH)

    checkpoint = read_project_activation(root)
    if checkpoint is None:
        if after_sequence == 0:
            return []
        raise ValueError("Project activation checkpoint is missing for the requested suffix.")

    current = checkpoint["project_activation"]
    if current["epoch"] != epoch:
        raise ValueError("Project activation checkpoint epoch changed; bootstrap is required.")
    current_sequence = current["sequence"]
    if after_sequence > current_sequence:
        raise ValueError("Project activation event after_sequence exceeds the checkpoint sequence.")

    snapshot = _read_snapshot_record(root, current["identity"]["project_id"])
    if snapshot is not None:
        snapshot_record = snapshot["project_activation_snapshot"]
        if snapshot_record["epoch"] == epoch:
            if snapshot_record["checkpoint_sequence"] > current_sequence:
                raise RuntimeError("Project activation snapshot is ahead of the current checkpoint.")
            floor = snapshot_record["floor_sequence"]
            if after_sequence < floor:
                raise ValueError(
                    f"Project activation event cursor is below the retention floor {floor}; reconstruction is required."
                )

    count = min(limit, current_sequence - after_sequence)
    if count == 0:
        return []
    _validate_event_directories(root, epoch)

    events: list[dict[str, Any]] = []
    for sequence in range(after_sequence + 1, after_sequence + count + 1):
        path = activation_event_path(root, epoch, sequence)
        try:
            value = _read_regular_json(path)
        except FileNotFoundError as exc:
            raise ValueError(f"Project activation event is missing: {path}") from exc
        events.append(
            _validate_event(
                value,
                project_id=current["identity"]["project_id"],
                epoch=epoch,
                sequence=sequence,
            )
        )
    return events


def _read_pending(root: Path, project_id: str) -> dict[str, Any] | None:
    _validate_checkpoint_directories(root)
    path = activation_pending_path(root)
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        return None
    return _validate_pending(value, project_id)


def _write_durable_record(path: Path, value: dict[str, Any], *, record_type: str) -> None:
    require_json_size(value, max_bytes=MAX_RECORD_BYTES, record_type=record_type)
    if atomic_replace(path, value) is None:
        raise RuntimeError(f"Project activation record could not be durably verified: {path}")


def _write_membership_revision_locked(root: Path, project_id: str, revision: int) -> None:
    path = activation_membership_path(root)
    _ensure_real_directory_chain(root, path.parent)
    value = {
        "project_activation_membership": {
            "version": ACTIVATION_VERSION,
            "identity": {"project_id": project_id},
            "revision": revision,
            "updated_at": utc_now(),
        }
    }
    _validate_membership(value, project_id)
    _write_durable_record(path, value, record_type="project_activation_membership")
    _sync_directory_chain_to_root(path.parent, root)


def _membership_revision_locked(root: Path, project_id: str) -> int:
    """Read the membership revision, initializing the durable zero record if absent."""
    path = activation_membership_path(root)
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        _write_membership_revision_locked(root, project_id, 0)
        return 0
    return _validate_membership(value, project_id)


def _increment_membership_revision_locked(root: Path, project_id: str) -> int:
    revision = _membership_revision_locked(root, project_id)
    if revision >= _MAX_SEQUENCE:
        raise OverflowError("Project activation membership revision is exhausted.")
    revision += 1
    _write_membership_revision_locked(root, project_id, revision)
    return revision


def _ensure_pending_event(root: Path, activation: dict[str, Any]) -> None:
    """Ensure a pending activation has one exact, durable event record."""
    epoch = activation["epoch"]
    sequence = activation["sequence"]
    path = activation_event_path(root, epoch, sequence)
    _ensure_real_directory_chain(root, path.parent)
    _validate_event_directories(root, epoch)
    expected = {"project_activation_event": activation.copy()}
    try:
        value = _read_regular_json(path)
    except FileNotFoundError:
        _write_durable_record(path, expected, record_type="project_activation_event")
        value = _read_regular_json(path)
    _validate_event(value, project_id=activation["identity"]["project_id"], epoch=epoch, sequence=sequence)
    if value != expected:
        raise RuntimeError(f"Project activation pending record conflicts with its event: {path}")
    _sync_directory_chain_to_root(path.parent, root)


def _assert_no_orphan_event(root: Path, checkpoint: dict[str, Any] | None) -> None:
    """Reject an event beyond the checkpoint when no pending record owns it."""
    if checkpoint is not None:
        activation = checkpoint["project_activation"]
        if activation["sequence"] >= _MAX_SEQUENCE:
            return
        path = activation_event_path(root, activation["epoch"], activation["sequence"] + 1)
        try:
            path.lstat()
        except FileNotFoundError:
            return
        raise RuntimeError(f"Project activation has an event without a pending record: {path}")

    events_root = Path(root) / "operations" / "project-activation-v1" / "events"
    try:
        metadata = events_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Project activation path is not a real directory: {events_root}")
    with os.scandir(events_root) as epochs:
        for epoch_index, epoch_entry in enumerate(epochs):
            if epoch_index >= _MAX_ORPHAN_EPOCH_SCAN:
                raise RuntimeError("Project activation orphan epoch scan exceeded its bounded limit.")
            if not epoch_entry.is_dir(follow_symlinks=False):
                raise RuntimeError(f"Project activation event path is not a real directory: {epoch_entry.path}")
            with os.scandir(epoch_entry.path) as events:
                if next(events, None) is not None:
                    raise RuntimeError("Project activation event exists without a checkpoint or pending record.")


def _remove_pending(root: Path) -> None:
    path = activation_pending_path(root)
    path.unlink()
    _sync_directory(path.parent)


def _reconstruct_missing_event_locked(
    root: Path,
    project_id: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Rotate a pre-journal checkpoint into a replayable conservative epoch."""
    previous = checkpoint["project_activation"]
    activation = {
        "version": ACTIVATION_VERSION,
        "identity": {"project_id": project_id},
        "epoch": _validate_epoch(uuid.uuid4().hex),
        "sequence": 1,
        "reason": "checkpoint_reconstruction",
        "updated_at": utc_now(),
    }
    if previous["identity"]["project_id"] != project_id:
        raise RuntimeError("Project activation reconstruction identity changed.")
    pending = {"project_activation_pending": activation.copy()}
    pending_path = activation_pending_path(root)
    _ensure_real_directory_chain(root, pending_path.parent)
    _write_durable_record(pending_path, pending, record_type="project_activation_pending")
    _sync_directory_chain_to_root(pending_path.parent, root)
    _ensure_pending_event(root, activation)
    committed = {"project_activation": activation.copy()}
    checkpoint_path = activation_checkpoint_path(root)
    _write_durable_record(checkpoint_path, committed, record_type="project_activation")
    _sync_directory_chain_to_root(checkpoint_path.parent, root)
    _remove_pending(root)
    return committed


def _event_journal_is_absent(root: Path) -> bool:
    path = root / "operations" / "project-activation-v1" / "events"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return True
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Project activation event path is not a real directory: {path}")
    return False


def _recover_pending_locked(root: Path, project_id: str) -> dict[str, Any] | None:
    """Recover or validate activation state while holding the writer lock."""
    checkpoint = _read_checkpoint_record(root, project_id)
    pending = _read_pending(root, project_id)
    if pending is None:
        if checkpoint is not None:
            try:
                _assert_checkpoint_event(root, checkpoint)
            except _MissingActivationEvent:
                if not _event_journal_is_absent(root):
                    raise
                return _reconstruct_missing_event_locked(root, project_id, checkpoint)
        elif _read_snapshot_record(root, project_id) is not None:
            raise RuntimeError("Project activation snapshot has no matching checkpoint.")
        _assert_no_orphan_event(root, checkpoint)
        return checkpoint

    activation = pending["project_activation_pending"]
    current_activation = checkpoint.get("project_activation") if checkpoint is not None else None
    if current_activation is None:
        if activation["sequence"] != 1:
            raise RuntimeError("Project activation pending sequence is not contiguous with a missing checkpoint.")
    elif activation == current_activation:
        pass
    elif (
        activation["reason"] == "checkpoint_reconstruction"
        and activation["epoch"] != current_activation["epoch"]
        and activation["sequence"] == 1
    ):
        pass
    elif (
        activation["epoch"] == current_activation["epoch"]
        and current_activation["sequence"] < _MAX_SEQUENCE
        and activation["sequence"] == current_activation["sequence"] + 1
    ):
        pass
    else:
        raise RuntimeError("Project activation pending record is not contiguous with the checkpoint.")

    _ensure_pending_event(root, activation)
    committed = {"project_activation": activation.copy()}
    if current_activation != activation:
        checkpoint_path = activation_checkpoint_path(root)
        _write_durable_record(checkpoint_path, committed, record_type="project_activation")
        _sync_directory_chain_to_root(checkpoint_path.parent, root)
    _remove_pending(root)
    return committed


def recover_project_activation(root: Path) -> dict[str, Any] | None:
    """Recover a pending Project activation under its shared writer lock."""
    root = Path(root)
    with exclusive(root / "locks" / "project-activation-v1.lock"):
        return _recover_pending_locked(root, _project_id(root))


@contextmanager
def project_activation_transaction(cfg: object, reason: str) -> Iterator[dict[str, Any]]:
    """Publish recoverable activation evidence across one authoritative mutation."""
    reason = _validate_reason(reason)
    root = Path(cfg.shared_root)
    with exclusive(root / "locks" / "project-activation-v1.lock"):
        project_id = _project_id(root)
        current = _recover_pending_locked(root, project_id)
        previous = current["project_activation"] if current is not None else None
        if previous is None:
            epoch = _validate_epoch(uuid.uuid4().hex)
            sequence = 1
        else:
            if previous["sequence"] >= _MAX_SEQUENCE:
                raise OverflowError("Project activation sequence is exhausted.")
            epoch = previous["epoch"]
            sequence = previous["sequence"] + 1

        activation = {
            "version": ACTIVATION_VERSION,
            "identity": {"project_id": project_id},
            "epoch": epoch,
            "sequence": sequence,
            "reason": reason,
            "updated_at": utc_now(),
        }
        pending = {"project_activation_pending": activation}
        event = {"project_activation_event": activation.copy()}
        require_json_size(pending, max_bytes=MAX_RECORD_BYTES, record_type="project_activation_pending")
        require_json_size(event, max_bytes=MAX_RECORD_BYTES, record_type="project_activation_event")

        pending_path = activation_pending_path(root)
        event_path = activation_event_path(root, epoch, sequence)
        _ensure_real_directory_chain(root, pending_path.parent)
        _ensure_real_directory_chain(root, event_path.parent)
        _sync_directory_chain_to_root(event_path.parent, root)
        try:
            pending_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("Project activation pending record appeared during publication.")
        _write_durable_record(pending_path, pending, record_type="project_activation_pending")
        _sync_directory_chain_to_root(pending_path.parent, root)
        _ensure_pending_event(root, activation)
        record = {"project_activation": activation.copy()}

        try:
            yield record
        except BaseException as body_error:
            try:
                committed = _recover_pending_locked(root, project_id)
                if committed != record:
                    raise RuntimeError("Project activation transaction did not commit its pending record.")
                _maybe_compact_activation_locked(root, project_id, committed)
            except BaseException as finalize_error:
                raise body_error from finalize_error
            raise
        else:
            committed = _recover_pending_locked(root, project_id)
            if committed != record:
                raise RuntimeError("Project activation transaction did not commit its pending record.")
            _maybe_compact_activation_locked(root, project_id, committed)


def publish_project_activation(cfg: object, reason: str) -> dict[str, Any]:
    """Commit one Project activation checkpoint and event under its writer lock."""
    with project_activation_transaction(cfg, reason) as record:
        pass
    return record


def _snapshot_value(
    *,
    project_id: str,
    epoch: str,
    previous_floor_sequence: int,
    floor_sequence: int,
    checkpoint_sequence: int,
    membership_revision: int,
) -> dict[str, Any]:
    return {
        "project_activation_snapshot": {
            "version": ACTIVATION_VERSION,
            "identity": {"project_id": project_id},
            "epoch": epoch,
            "previous_floor_sequence": previous_floor_sequence,
            "floor_sequence": floor_sequence,
            "checkpoint_sequence": checkpoint_sequence,
            "membership_revision": membership_revision,
            "coverage": {
                "kind": _RECONSTRUCTION_COVERAGE["kind"],
                "service_lanes": _RECONSTRUCTION_COVERAGE["service_lanes"].copy(),
                "work_states": _RECONSTRUCTION_COVERAGE["work_states"].copy(),
            },
            "updated_at": utc_now(),
        }
    }


def _write_snapshot_locked(root: Path, value: dict[str, Any], project_id: str) -> dict[str, Any]:
    path = activation_snapshot_path(root)
    _ensure_real_directory_chain(root, path.parent)
    _validate_snapshot(value, project_id)
    _write_durable_record(path, value, record_type="project_activation_snapshot")
    _sync_directory_chain_to_root(path.parent, root)
    return value


def _delete_compacted_prefix_locked(
    root: Path,
    *,
    project_id: str,
    epoch: str,
    after_sequence: int,
    through_sequence: int,
    allow_missing: bool,
) -> None:
    if through_sequence <= after_sequence:
        return
    if through_sequence - after_sequence > _COMPACTION_BATCH:
        raise RuntimeError("Project activation compaction exceeds its bounded deletion batch.")
    _validate_event_directories(root, epoch)
    directory = activation_event_path(root, epoch, through_sequence).parent
    for sequence in range(after_sequence + 1, through_sequence + 1):
        path = activation_event_path(root, epoch, sequence)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"Project activation event disappeared during compaction: {path}") from None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RECORD_BYTES:
            raise RuntimeError(f"Project activation event is not a bounded regular file: {path}")
        # Validate before unlinking so compaction never masks a corrupted retained event.
        value = _read_regular_json(path)
        _validate_event(value, project_id=project_id, epoch=epoch, sequence=sequence)
        path.unlink()
    _sync_directory_chain_to_root(directory, root)


def _finish_compaction_deletion_locked(
    root: Path,
    project_id: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    record = snapshot["project_activation_snapshot"]
    if record["previous_floor_sequence"] == record["floor_sequence"]:
        return snapshot
    _delete_compacted_prefix_locked(
        root,
        project_id=project_id,
        epoch=record["epoch"],
        after_sequence=record["previous_floor_sequence"],
        through_sequence=record["floor_sequence"],
        allow_missing=True,
    )
    completed = {
        "project_activation_snapshot": {
            **record,
            "previous_floor_sequence": record["floor_sequence"],
            "updated_at": utc_now(),
        }
    }
    return _write_snapshot_locked(root, completed, project_id)


def _compact_activation_locked(
    root: Path,
    project_id: str,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    current = checkpoint["project_activation"]
    epoch = current["epoch"]
    sequence = current["sequence"]
    existing = _read_snapshot_record(root, project_id)
    floor = 0
    if existing is not None and existing["project_activation_snapshot"]["epoch"] == epoch:
        prior = existing["project_activation_snapshot"]
        if prior["checkpoint_sequence"] > sequence or prior["floor_sequence"] > sequence:
            raise RuntimeError("Project activation snapshot is ahead of the current checkpoint.")
        if prior["previous_floor_sequence"] < prior["floor_sequence"]:
            # Complete an interrupted snapshot-before-delete operation in its own
            # bounded pass. A later pass can advance the floor again.
            return _finish_compaction_deletion_locked(root, project_id, existing)
        floor = prior["floor_sequence"]

    target_floor = min(sequence, floor + _COMPACTION_BATCH)
    for event_sequence in range(floor + 1, target_floor + 1):
        path = activation_event_path(root, epoch, event_sequence)
        try:
            event = _read_regular_json(path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Project activation event is missing before compaction: {path}") from exc
        _validate_event(event, project_id=project_id, epoch=epoch, sequence=event_sequence)

    revision = _membership_revision_locked(root, project_id)
    if existing is not None:
        prior = existing["project_activation_snapshot"]
        if (
            prior["epoch"] == epoch
            and prior["floor_sequence"] == target_floor
            and prior["checkpoint_sequence"] == sequence
            and prior["membership_revision"] == revision
        ):
            return existing

    snapshot = _snapshot_value(
        project_id=project_id,
        epoch=epoch,
        previous_floor_sequence=floor,
        floor_sequence=target_floor,
        checkpoint_sequence=sequence,
        membership_revision=revision,
    )
    durable_snapshot = _write_snapshot_locked(root, snapshot, project_id)
    _delete_compacted_prefix_locked(
        root,
        project_id=project_id,
        epoch=epoch,
        after_sequence=floor,
        through_sequence=target_floor,
        allow_missing=False,
    )
    return _finish_compaction_deletion_locked(root, project_id, durable_snapshot)


def _current_snapshot_floor_locked(root: Path, project_id: str, epoch: str) -> int:
    snapshot = _read_snapshot_record(root, project_id)
    if snapshot is None:
        return 0
    record = snapshot["project_activation_snapshot"]
    return record["floor_sequence"] if record["epoch"] == epoch else 0


def _maybe_compact_activation_locked(
    root: Path,
    project_id: str,
    checkpoint: dict[str, Any],
) -> None:
    current = checkpoint["project_activation"]
    snapshot = _read_snapshot_record(root, project_id)
    pending_deletion = False
    floor = 0
    if snapshot is not None:
        record = snapshot["project_activation_snapshot"]
        if record["epoch"] == current["epoch"]:
            if record["checkpoint_sequence"] > current["sequence"] or record["floor_sequence"] > current["sequence"]:
                raise RuntimeError("Project activation snapshot is ahead of the current checkpoint.")
            floor = record["floor_sequence"]
            pending_deletion = record["previous_floor_sequence"] < floor
    if pending_deletion or current["sequence"] - floor >= _COMPACTION_BATCH:
        _compact_activation_locked(root, project_id, checkpoint)


def compact_project_activation(root: Path) -> dict[str, Any] | None:
    """Compact one bounded event prefix into an authoritative-index snapshot."""
    root = Path(root)
    with exclusive(root / "locks" / "project-activation-v1.lock"):
        project_id = _project_id(root)
        checkpoint = _recover_pending_locked(root, project_id)
        return _compact_activation_locked(root, project_id, checkpoint)
