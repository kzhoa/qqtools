"""Crash-consistent isolation of Group truth from released cached writers."""

from __future__ import annotations

import os
import stat
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any

from .protocol_compatibility import GROUP_AUTHORITY_CAPABILITY, SUBMISSION_GROUP_PUBLICATION_CAPABILITY
from .records import normalize_group_record, utc_now, validate_identifier
from .responsibility_store import DurableIO
from .store import atomic_replace, read_json, read_json_limited

_GROUP_DIRECTORY = "groups-v2"
_IDENTITY_FILE = ".authority-identity"
_internal_provisional_read = ContextVar("qexp_internal_provisional_group_read", default=False)


class GroupNotPublished(RuntimeError):
    """A provisional Group is present but its Submission is not committed."""


class GroupPublicationUnavailable(RuntimeError):
    """The Submission proof required to publish a Group cannot be established."""


@contextmanager
def provisional_group_reader():
    """Temporarily permit a fenced transaction to inspect its own Group stage."""
    token = _internal_provisional_read.set(True)
    try:
        yield
    finally:
        _internal_provisional_read.reset(token)


def journal_path(root: Path) -> Path:
    return root / "schema" / "group-authority.json"


def _identity(path: Path) -> dict[str, int] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"Group authority path is not a real directory: {path}")
    return {"device": value.st_dev, "inode": value.st_ino}


def _directory_identity(path: Path, *, storage: Any | None = None) -> dict[str, Any] | None:
    if _identity(path) is None:
        return None
    try:
        value = _read_limited(path / _IDENTITY_FILE, max_bytes=4096, storage=storage)["group_directory"]
    except FileNotFoundError as exc:
        raise RuntimeError("Group authority directory identity is missing") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
        or not isinstance(value.get("directory_id"), str)
        or len(value["directory_id"]) != 32
        or not isinstance(value.get("project_id"), str)
        or not isinstance(value.get("shared_root"), str)
    ):
        raise RuntimeError("Group authority directory identity is invalid")
    return value


def _read_limited(path: Path, *, max_bytes: int, storage: Any | None = None) -> dict[str, Any]:
    if storage is not None:
        return storage.read_json_limited(path, max_bytes=max_bytes)
    return read_json_limited(path, max_bytes=max_bytes)


def _read_journal(root: Path, *, storage: Any | None = None) -> dict[str, Any] | None:
    try:
        value = _read_limited(journal_path(root), max_bytes=8192, storage=storage)["group_authority"]
    except FileNotFoundError:
        return None
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("shared_root") != str(root)
        or value.get("source") != "groups"
        or value.get("destination") != _GROUP_DIRECTORY
        or value.get("phase") not in {"prepared", "moved", "completed"}
        or not isinstance(value.get("project_id"), str)
        or not isinstance(value.get("source_identity"), dict)
        or value["source_identity"].get("project_id") != value["project_id"]
        or value["source_identity"].get("shared_root") != str(root)
    ):
        raise RuntimeError("Group authority journal is invalid")
    return value


def group_directory(root: Path, *, storage: Any | None = None) -> Path:
    """Select current truth, including an interrupted atomic directory move."""
    journal = _read_journal(root, storage=storage)
    if journal is None:
        try:
            schema = _read_limited(root / "schema/version.json", max_bytes=16384, storage=storage)["schema"]
        except FileNotFoundError:
            schema = {}
        if GROUP_AUTHORITY_CAPABILITY in schema.get("required_capabilities", []):
            raise RuntimeError("Group authority capability has no migration journal")
        return root / "groups"
    destination = root / _GROUP_DIRECTORY
    identity = journal["source_identity"]
    target = _directory_identity(destination, storage=storage)
    if target is not None:
        if target != identity:
            raise RuntimeError("Group authority destination identity changed")
        return destination
    source = root / "groups"
    if journal["phase"] == "prepared" and _directory_identity(source, storage=storage) == identity:
        # The rename can occur between the two stats. Prefer its destination.
        target = _directory_identity(destination, storage=storage)
        if target is not None:
            if target != identity:
                raise RuntimeError("Group authority destination identity changed")
            return destination
        return source
    # Retry the destination after a concurrent source rename.
    if _directory_identity(destination, storage=storage) == identity:
        return destination
    raise RuntimeError("Group authority directory is missing or changed")


def read_group_raw(root: Path, name: str) -> dict[str, Any]:
    """Read one Group without applying Submission publication filtering."""
    directory = group_directory(root)
    try:
        value = read_json(directory / f"{name}.json")
    except FileNotFoundError:
        current = group_directory(root)
        if current == directory:
            raise
        return read_json(current / f"{name}.json")
    current = group_directory(root)
    return value if current == directory else read_json(current / f"{name}.json")


def _submission_state(root: Path, operation_id: str) -> str:
    """Read one bounded Submission proof without requiring a RootConfig object."""
    from .submission_control import read_submission_state

    try:
        return read_submission_state(root, operation_id)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise GroupPublicationUnavailable(
            f"Group publication proof for operation {operation_id!r} is unavailable."
        ) from exc


def group_visibility(root: Path, name: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the one published Group projection shared by all ordinary readers.

    A Group created by a Submission is published only after its exact operation
    reaches ``committed``.  Existing Groups remain visible while filtering
    Worker additions owned by an uncommitted Submission.
    """
    value = deepcopy(value if value is not None else read_group_raw(root, name))
    try:
        normalize_group_record(value)
    except (TypeError, ValueError, KeyError) as exc:
        raise GroupPublicationUnavailable(f"Group {name!r} is malformed.") from exc
    group = value["group"]
    creation_operation_id = group.get("creation_operation_id")
    if creation_operation_id is not None:
        try:
            validate_identifier(creation_operation_id, "creation_operation_id")
        except (TypeError, ValueError) as exc:
            raise GroupPublicationUnavailable(f"Group {name!r} has invalid creation provenance.") from exc
        state = _submission_state(root, creation_operation_id)
        if state != "committed":
            raise GroupNotPublished(
                f"Group {name!r} is not published; Submission operation {creation_operation_id!r} is {state}."
            )

    states: dict[str, str] = {}
    workers = group.get("worker_set", {})
    for machine, worker in list(workers.items()):
        operation_id = worker.get("added_by_operation")
        if operation_id is None:
            continue
        try:
            validate_identifier(operation_id, "added_by_operation")
        except (TypeError, ValueError) as exc:
            raise GroupPublicationUnavailable(
                f"Group {name!r} has invalid Worker publication provenance for {machine!r}."
            ) from exc
        state = states.setdefault(operation_id, _submission_state(root, operation_id))
        if state != "committed":
            del workers[machine]
    return value


def read_group(root: Path, name: str) -> dict[str, Any]:
    """Read one published Group, retrying a concurrent namespace cutover."""
    if _internal_provisional_read.get():
        return read_group_raw(root, name)
    directory = group_directory(root)
    try:
        value = read_group_raw(root, name)
    except FileNotFoundError:
        raise
    current = group_directory(root)
    if current != directory:
        value = read_group_raw(root, name)
    return group_visibility(root, name, value)


def iter_published_groups(root: Path) -> list[dict[str, Any]]:
    """Enumerate ordinary Group projections, excluding unpublished records."""
    result: list[dict[str, Any]] = []
    for path in sorted(group_directory(root).glob("*.json")):
        try:
            result.append(read_group(root, path.stem))
        except GroupNotPublished:
            continue
    return result


def has_group_authority_cutover(root: Path) -> bool:
    return _read_journal(root) is not None


def is_group_authority_isolated(root: Path, *, storage: Any | None = None) -> bool:
    journal = _read_journal(root, storage=storage)
    if journal is None or journal["phase"] != "completed":
        return False
    if group_directory(root, storage=storage) != root / _GROUP_DIRECTORY:
        raise RuntimeError("Group authority completed without its canonical directory")
    return True


def group_authority_identity(root: Path, *, storage: Any | None = None) -> dict[str, Any]:
    journal = _read_journal(root, storage=storage)
    if journal is None or not is_group_authority_isolated(root, storage=storage):
        raise RuntimeError("Group authority namespace is not active")
    return {"project_id": journal["project_id"], "directory_identity": journal["source_identity"]}


def inspect_group_authority(cfg: object) -> dict[str, Any]:
    """Report fixed activation metadata without migration or history reads."""
    from .ready import state

    try:
        journal = _read_journal(cfg.shared_root)
        directory = group_directory(cfg.shared_root)
        result = {"state": "waiting", "directory": directory.name, "diagnostic_only": True}
        if journal is not None:
            result["state"] = journal["phase"]
            if journal["phase"] == "completed":
                state.assert_ready_writer_compatible(cfg)
                if state.read_state_record(cfg)[1]["writer_capability"] != state.CURRENT_READY_WRITER_CAPABILITY:
                    raise RuntimeError("Group authority lost its Task writer floor")
                result["state"] = "active"
        return result
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {"state": "unavailable", "detail": str(exc), "diagnostic_only": True}


def activate_group_authority_locked(cfg: object) -> bool:
    """Advance fixed metadata under the caller's exclusive schema fence.

    The caller must hold registration authority and have completed retained local
    writer capture. Returning False defers activation while ready rebuild is busy.
    """
    from ..layout import LOCAL_RECOVERY_CAPABILITY
    from .locks import exclusive
    from .ready import state

    root = cfg.shared_root
    schema_path = root / "schema/version.json"
    schema = read_json(schema_path)
    required = schema["schema"]["required_capabilities"]
    if LOCAL_RECOVERY_CAPABILITY not in required:
        raise RuntimeError("Group authority isolation requires recovery admission")
    io = DurableIO()
    journal = _read_journal(root)
    project_id = read_json(root / "project/identity.json")["project"]["project_id"]
    if journal is not None and journal["project_id"] != project_id:
        raise RuntimeError("Group authority journal belongs to another project")
    if journal is not None and journal["phase"] == "completed":
        if GROUP_AUTHORITY_CAPABILITY not in required or not is_group_authority_isolated(root):
            raise RuntimeError("Group authority completed without its capability")
        if SUBMISSION_GROUP_PUBLICATION_CAPABILITY not in required:
            # QQTOOLS-COMPAT-0015: qualified Group-authority activation fences
            # every reader before a marked Group can be emitted.
            required.append(SUBMISSION_GROUP_PUBLICATION_CAPABILITY)
            atomic_replace(schema_path, schema)
        state.assert_ready_writer_compatible(cfg)
        if state.read_state_record(cfg)[1]["writer_capability"] != state.CURRENT_READY_WRITER_CAPABILITY:
            raise RuntimeError("Group authority lost its Task writer floor")
        io.sync_directory(root, "group_authority_move")
        io.sync_directory(root / "schema", "group_authority_completed")
        return True
    if GROUP_AUTHORITY_CAPABILITY in required and journal is None:
        raise RuntimeError("Group authority capability has no migration journal")
    with exclusive(state.state_lock_path(cfg), blocking=False) as acquired:
        if not acquired:
            return False
        value, ready = state.read_state_record(cfg)
        if ready["state"] != "active":
            return False
        if ready["writer_capability"] not in state.SUPPORTED_READY_WRITERS:
            raise RuntimeError("Group authority cannot replace an unknown Task writer floor")
        capabilities = schema["schema"]["writer_capabilities"]
        if state.CURRENT_READY_WRITER_CAPABILITY not in capabilities:
            capabilities.append(state.CURRENT_READY_WRITER_CAPABILITY)
            atomic_replace(schema_path, schema)
        if ready["writer_capability"] != state.CURRENT_READY_WRITER_CAPABILITY:
            ready["writer_capability"] = state.CURRENT_READY_WRITER_CAPABILITY
            state.commit_state_under_lock(root / "indexes/ready/state.json", value, ready)
        io.sync_directory(root / "indexes/ready", "group_authority_task_floor")
    source, destination = root / "groups", root / _GROUP_DIRECTORY
    if journal is None:
        if _identity(destination) is not None:
            raise RuntimeError("Group authority destination exists without a journal")
        if _identity(source) is None:
            raise RuntimeError("Group authority source directory is missing")
        marker = source / _IDENTITY_FILE
        if not marker.exists():
            atomic_replace(
                marker,
                {
                    "group_directory": {
                        "version": 1,
                        "directory_id": uuid.uuid4().hex,
                        "project_id": project_id,
                        "shared_root": str(root),
                    }
                },
            )
        identity = _directory_identity(source)
        if identity["project_id"] != project_id or identity["shared_root"] != str(root):
            raise RuntimeError("Group authority source belongs to another project")
        io.sync_directory(source, "group_authority_identity")
        journal = {
            "version": 1,
            "project_id": project_id,
            "shared_root": str(root),
            "source": "groups",
            "destination": _GROUP_DIRECTORY,
            "source_identity": identity,
            "phase": "prepared",
            "prepared_at": utc_now(),
        }
        atomic_replace(journal_path(root), {"group_authority": journal})
    identity = journal["source_identity"]
    target_identity = _directory_identity(destination)
    if target_identity is None:
        if journal["phase"] != "prepared" or _directory_identity(source) != identity:
            raise RuntimeError("Group authority source identity changed before move")
        os.rename(source, destination)
    elif target_identity != identity:
        raise RuntimeError("Group authority destination identity changed")
    io.sync_directory(root, "group_authority_move")
    journal["phase"] = "moved"
    atomic_replace(journal_path(root), {"group_authority": journal})
    # Cached released writers may recreate groups/. It is never selected again.
    # Keep it as a shadow; no recursive cleanup belongs in activation.
    capabilities_changed = False
    if GROUP_AUTHORITY_CAPABILITY not in required:
        required.append(GROUP_AUTHORITY_CAPABILITY)
        capabilities_changed = True
    if SUBMISSION_GROUP_PUBLICATION_CAPABILITY not in required:
        # QQTOOLS-COMPAT-0015: activation is one schema-manifest transition.
        required.append(SUBMISSION_GROUP_PUBLICATION_CAPABILITY)
        capabilities_changed = True
    if capabilities_changed:
        atomic_replace(schema_path, schema)
    io.sync_directory(root / "schema", "group_authority_capability")
    journal["phase"] = "completed"
    journal["completed_at"] = utc_now()
    atomic_replace(journal_path(root), {"group_authority": journal})
    return True
