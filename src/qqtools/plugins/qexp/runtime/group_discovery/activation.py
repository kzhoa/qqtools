"""Fenced activation and bounded bootstrap for Group service locators."""

from __future__ import annotations

import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qqtools.version import __version__

from ...config_types import RootConfig
from ...lease import parse_utc
from ..directory_capture import read_directory_entry
from ..group_namespace import group_authority_identity, group_directory
from ..locks import group_lock, schema_lock
from ..paths import shared_paths
from ..protocol_compatibility import GROUP_SERVICE_CAPABILITY, SUPPORTED_REQUIRED_CAPABILITIES
from ..store import atomic_replace, read_json, read_json_limited, require_json_size
from . import locator

# QQTOOLS-COMPAT-0017: dual publication and historical-discovery overlap exists
# only while a Project moves through its fenced bootstrap activation window.

PROTOCOL = "group-service-v1"
WRITER_FLOOR = "1.3.22"
ACTIVATION_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024
_STATES = frozenset({"preparing", "fenced", "building", "active", "degraded"})
_PHASES = ("groups", "submissions", "active_namespaces", "stable_pass", "complete")
_CURSOR_NAMES = ("groups", "submissions", "submission_control", "group_control", "cleanup", "discovery_debt")
_CURSOR_FIELDS = frozenset({"directory_revision", "cookie", "complete"})
_REVISION_FIELDS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_RECORD_FIELDS = frozenset(
    {
        "version",
        "revision",
        "protocol",
        "identity",
        "state",
        "activation_epoch",
        "writer_floor",
        "layout",
        "bootstrap",
        "diagnostic",
        "updated_at",
    }
)
_IDENTITY_FIELDS = frozenset({"project_id", "group_directory_identity"})


def activation_path(root: Path) -> Path:
    return Path(root) / "schema" / "group-service.json"


def read_group_service_activation_record(root: Path, *, storage: Any | None = None) -> dict[str, Any] | None:
    """Read and validate the activation record, optionally through migration storage."""
    return _read_record(Path(root), storage=storage)


def group_service_registration_diagnostic(root: Path, *, storage: Any | None = None) -> dict[str, str] | None:
    """Return why a registered writer prevents the Group service fence."""
    return _registration_participant_diagnostic(Path(root), storage=storage)


def _utc_now() -> str:
    from ..records import utc_now

    return utc_now()


def _identity(root: Path, *, storage: Any | None = None) -> dict[str, Any]:
    authority = group_authority_identity(root, storage=storage)
    return {
        "project_id": authority["project_id"],
        "group_directory_identity": dict(authority["directory_identity"]),
    }


def _uuid_token(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc
    if parsed.int == 0 or value not in {parsed.hex, str(parsed)}:
        raise ValueError(f"{label} must be a canonical non-zero UUID")
    return value


def _revision(path: Path) -> dict[str, int] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"Group service bootstrap source is not a real directory: {path}")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _new_cursors() -> dict[str, dict[str, Any]]:
    return {name: {"directory_revision": None, "cookie": 0, "complete": False} for name in _CURSOR_NAMES}


def _validate_cursor(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CURSOR_FIELDS:
        raise RuntimeError(f"Group service bootstrap cursor {label} has an invalid shape")
    directory_revision = value["directory_revision"]
    if directory_revision is not None and (
        type(directory_revision) is not dict
        or set(directory_revision) != _REVISION_FIELDS
        or any(type(item) is not int or item < 0 for item in directory_revision.values())
    ):
        raise RuntimeError(f"Group service bootstrap cursor {label} has an invalid directory revision")
    if type(value["cookie"]) is not int or not 0 <= value["cookie"] <= (1 << 63) - 1:
        raise RuntimeError(f"Group service bootstrap cursor {label} has an invalid cookie")
    if type(value["complete"]) is not bool:
        raise RuntimeError(f"Group service bootstrap cursor {label} has an invalid completion flag")
    return value


def _validate_bootstrap(value: object) -> dict[str, Any]:
    fields = {"generation", "phase", "cursors", "stable_pass", "post_fence", "completed_at"}
    if type(value) is not dict or set(value) != fields:
        raise RuntimeError("Group service bootstrap record has an invalid shape")
    _uuid_token(value["generation"], "bootstrap generation")
    if value["phase"] not in _PHASES:
        raise RuntimeError("Group service bootstrap phase is invalid")
    cursors = value["cursors"]
    if type(cursors) is not dict or set(cursors) != set(_CURSOR_NAMES):
        raise RuntimeError("Group service bootstrap cursors are invalid")
    for name, cursor in cursors.items():
        _validate_cursor(cursor, name)
    if type(value["stable_pass"]) is not int or value["stable_pass"] < 0:
        raise RuntimeError("Group service bootstrap stable pass is invalid")
    if type(value["post_fence"]) is not bool:
        raise RuntimeError("Group service bootstrap fence marker is invalid")
    if value["completed_at"] is not None and not isinstance(value["completed_at"], str):
        raise RuntimeError("Group service bootstrap completion time is invalid")
    return value


def _validate_record(root: Path, value: object, *, storage: Any | None = None) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _RECORD_FIELDS:
        raise RuntimeError("Group service activation record has an invalid shape")
    if type(value["version"]) is not int or value["version"] != ACTIVATION_VERSION:
        raise RuntimeError("Group service activation record version is unsupported")
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise RuntimeError("Group service activation revision is invalid")
    if value["protocol"] != PROTOCOL or value["state"] not in _STATES:
        raise RuntimeError("Group service activation protocol or state is invalid")
    if type(value["identity"]) is not dict or set(value["identity"]) != _IDENTITY_FIELDS:
        raise RuntimeError("Group service activation identity is invalid")
    if value["identity"] != _identity(root, storage=storage):
        raise RuntimeError("Group service activation identity does not match current authority")
    _uuid_token(value["activation_epoch"], "activation epoch")
    if value["writer_floor"] != WRITER_FLOOR:
        raise RuntimeError("Group service writer floor is invalid")
    layout = value["layout"]
    if (
        type(layout) is not dict
        or set(layout) != {"revision", "manifest_digest"}
        or type(layout["revision"]) is not int
        or layout["revision"] != 1
        or not isinstance(layout["manifest_digest"], str)
        or len(layout["manifest_digest"]) != 64
    ):
        raise RuntimeError("Group service activation layout evidence is invalid")
    _validate_bootstrap(value["bootstrap"])
    diagnostic = value["diagnostic"]
    if diagnostic is not None and (
        type(diagnostic) is not dict
        or set(diagnostic) != {"code", "detail"}
        or not all(isinstance(diagnostic[key], str) and len(diagnostic[key]) <= 512 for key in diagnostic)
    ):
        raise RuntimeError("Group service activation diagnostic is invalid")
    if not isinstance(value["updated_at"], str):
        raise RuntimeError("Group service activation timestamp is invalid")
    return value


def _read_record(root: Path, *, storage: Any | None = None) -> dict[str, Any] | None:
    path = activation_path(root)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RECORD_BYTES:
        raise RuntimeError("Group service activation record is not a bounded regular file")
    value = (
        storage.read_json_limited(path, max_bytes=MAX_RECORD_BYTES)
        if storage is not None
        else read_json_limited(path, max_bytes=MAX_RECORD_BYTES, record_type="group_service_activation")
    )
    return _validate_record(root, value, storage=storage)


def _new_record(root: Path, layout: dict[str, Any], *, storage: Any | None = None) -> dict[str, Any]:
    return {
        "version": ACTIVATION_VERSION,
        "revision": 1,
        "protocol": PROTOCOL,
        "identity": _identity(root, storage=storage),
        "state": "preparing",
        "activation_epoch": uuid.uuid4().hex,
        "writer_floor": WRITER_FLOOR,
        "layout": {"revision": layout["revision"], "manifest_digest": layout["manifest_digest"]},
        "bootstrap": {
            "generation": uuid.uuid4().hex,
            "phase": "groups",
            "cursors": _new_cursors(),
            "stable_pass": 0,
            "post_fence": False,
            "completed_at": None,
        },
        "diagnostic": None,
        "updated_at": _utc_now(),
    }


def _write_record(root: Path, record: dict[str, Any], *, storage: Any | None = None) -> dict[str, Any]:
    _validate_record(root, record, storage=storage)
    require_json_size(record, max_bytes=MAX_RECORD_BYTES, record_type="group_service_activation")
    path = activation_path(root)
    if storage is None:
        atomic_replace(path, record)
    else:
        storage.atomic_replace(path, record)
    persisted = _read_record(root, storage=storage)
    if persisted is None:
        raise RuntimeError("Group service activation record disappeared while being written")
    if persisted != record:
        raise RuntimeError("Group service activation record changed while being written")
    return persisted


def _version_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    pieces = value.split(".")
    if not pieces or any(not piece.isdecimal() for piece in pieces):
        return None
    try:
        return tuple(int(piece) for piece in pieces)
    except ValueError:
        return None


def _meets_writer_floor(value: object) -> bool:
    candidate = _version_tuple(value)
    floor = _version_tuple(WRITER_FLOOR)
    return candidate is not None and floor is not None and candidate >= floor


def writer_floor_satisfied() -> bool:
    """Return whether this installed writer may participate in Group service activation."""
    return _meets_writer_floor(__version__)


def _registration_participant_diagnostic(
    root: Path,
    *,
    storage: Any | None = None,
) -> dict[str, str] | None:
    if not _meets_writer_floor(__version__):
        return {"code": "writer_floor_not_met", "detail": f"installed qqtools {__version__} is below {WRITER_FLOOR}"}
    machines = Path(root) / "machines"
    metadata = machines.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        return {"code": "registration_unavailable", "detail": "machines namespace is not a real directory"}
    with os.scandir(machines) as entries:
        for entry in entries:
            if entry.name in {".", ".."}:
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            path = Path(entry.path) / "registration.json"
            try:
                registration_file = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(registration_file.st_mode) or registration_file.st_size > 16 * 1024:
                return {"code": "registration_unavailable", "detail": f"registration for {entry.name!r} is unavailable"}
            try:
                registration_record = (
                    storage.read_json_limited(path, max_bytes=16 * 1024) if storage is not None else read_json(path)
                )
                registration = registration_record.get("registration")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return {"code": "registration_unavailable", "detail": f"registration for {entry.name!r}: {exc}"}
            if type(registration) is not dict:
                return {"code": "registration_unavailable", "detail": f"registration for {entry.name!r} is malformed"}
            state = registration.get("state")
            if state == "superseded":
                continue
            if state != "eligible":
                return {
                    "code": "registration_unavailable",
                    "detail": f"registration for {entry.name!r} has invalid state {state!r}",
                }
            try:
                expires_at = parse_utc(registration["eligibility_expires_at"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                return {
                    "code": "registration_unavailable",
                    "detail": f"registration for {entry.name!r} has invalid eligibility expiry: {exc}",
                }
            if expires_at <= datetime.now(timezone.utc):
                continue
            generation = registration.get("generation")
            try:
                _uuid_token(generation, f"registration generation for {entry.name}")
            except ValueError as exc:
                return {"code": "registration_unavailable", "detail": str(exc)}
            if registration.get("machine_name") != entry.name or not _meets_writer_floor(
                registration.get("client_version")
            ):
                return {
                    "code": "participant_below_writer_floor",
                    "detail": f"eligible machine {entry.name!r} generation {generation} has not reported {WRITER_FLOOR}",
                }
    return None


def _install_capability(root: Path, *, storage: Any | None = None) -> None:
    schema_path = shared_paths(root)["schema"] / "version.json"
    schema_file = storage.read_json_limited(schema_path, max_bytes=16 * 1024) if storage else read_json(schema_path)
    schema = schema_file.get("schema")
    if type(schema) is not dict:
        raise RuntimeError("qexp schema/version.json is malformed")
    capabilities = schema.get("required_capabilities")
    if type(capabilities) is not list or not all(isinstance(item, str) for item in capabilities):
        raise RuntimeError("qexp schema/version.json has malformed required capabilities")
    unknown = sorted(set(capabilities) - SUPPORTED_REQUIRED_CAPABILITIES)
    if unknown:
        raise RuntimeError(f"qexp root requires unsupported capabilities: {', '.join(unknown)}")
    if GROUP_SERVICE_CAPABILITY not in capabilities:
        schema["required_capabilities"] = [*capabilities, GROUP_SERVICE_CAPABILITY]
        if storage is None:
            atomic_replace(schema_path, schema_file)
        else:
            storage.atomic_replace(schema_path, schema_file)


def _bootstrap_paths(cfg: RootConfig, *, storage: Any | None = None) -> dict[str, Path]:
    paths = shared_paths(cfg.shared_root)
    return {
        "groups": group_directory(cfg.shared_root, storage=storage),
        "submissions": paths["submissions"],
        "submission_control": cfg.shared_root / "indexes" / "submission-control" / "pending",
        "group_control": paths["group_control_active"],
        "cleanup": paths["cleanup_active"],
        "discovery_debt": cfg.shared_root / "operations" / "group-discovery" / "active",
    }


def _cursor_phase(name: str) -> str:
    if name == "groups":
        return "groups"
    if name == "submissions":
        return "submissions"
    return "active_namespaces"


def _publish_locked(
    cfg: RootConfig,
    group: str,
    lane: str,
    reason: str,
    *,
    storage: Any | None = None,
) -> None:
    with group_lock(cfg.shared_root, group):
        locator.publish_group_locator_locked(cfg, group, lane, reason, storage=storage)


def _process_entry(cfg: RootConfig, namespace: str, path: Path, name: str, *, storage: Any | None = None) -> None:
    if not name.endswith(".json"):
        return
    if namespace == "groups":
        group_name = name[:-5]
        # Conservatively enqueue all obligations once. Active consumers recheck
        # Group authority and quickly acknowledge empty lanes.
        _publish_locked(cfg, group_name, "membership", "bootstrap", storage=storage)
        _publish_locked(cfg, group_name, "control", "bootstrap", storage=storage)
        _publish_locked(cfg, group_name, "maintenance", "bootstrap", storage=storage)
        return
    if namespace == "submissions":
        operation = storage.read_json_limited(path, max_bytes=MAX_RECORD_BYTES) if storage else read_json(path)
        submission = operation.get("submission")
        if type(submission) is not dict:
            raise RuntimeError(f"Submission source is malformed during activation bootstrap: {path.name}")
        group_name = submission.get("target_group")
        if group_name and submission.get("state") in {"prepared", "committing", "blocked"}:
            _publish_locked(cfg, group_name, "membership", "bootstrap", storage=storage)
        return
    if namespace == "group_control":
        record = storage.read_json_limited(path, max_bytes=MAX_RECORD_BYTES) if storage else read_json(path)
        operation = record.get("group_control")
        group_name = operation.get("group_name") if isinstance(operation, dict) else None
        if isinstance(group_name, str):
            _publish_locked(cfg, group_name, "control", "bootstrap", storage=storage)
        return
    if namespace == "cleanup":
        record = storage.read_json_limited(path, max_bytes=MAX_RECORD_BYTES) if storage else read_json(path)
        operation = record.get("cleanup")
        group_name = operation.get("group_name") if isinstance(operation, dict) else None
        if isinstance(group_name, str):
            _publish_locked(cfg, group_name, "control", "bootstrap", storage=storage)
            _publish_locked(cfg, group_name, "maintenance", "bootstrap", storage=storage)
        return
    if namespace == "submission_control":
        operation_path = shared_paths(cfg.shared_root)["submissions"] / name
        if not operation_path.exists():
            raise RuntimeError(f"Submission-control owner has no Submission source: {name}")
        record = (
            storage.read_json_limited(operation_path, max_bytes=MAX_RECORD_BYTES)
            if storage
            else read_json(operation_path)
        )
        operation = record.get("submission")
        group_name = operation.get("target_group") if isinstance(operation, dict) else None
        if isinstance(group_name, str):
            _publish_locked(cfg, group_name, "membership", "bootstrap", storage=storage)


def _advance_cursor(
    cfg: RootConfig,
    record: dict[str, Any],
    namespace: str,
    *,
    storage: Any | None = None,
) -> dict[str, Any]:
    path = _bootstrap_paths(cfg, storage=storage)[namespace]
    bootstrap = record["bootstrap"]
    cursor = dict(bootstrap["cursors"][namespace])
    current_revision = _revision(path)
    if cursor["directory_revision"] != current_revision:
        cursor.update(directory_revision=current_revision, cookie=0, complete=False)
    if current_revision is None:
        cursor.update(cookie=0, complete=True)
    elif not cursor["complete"]:
        name, next_cookie = read_directory_entry(path, cursor["cookie"])
        if _revision(path) != current_revision:
            cursor.update(directory_revision=_revision(path), cookie=0, complete=False)
        elif name is None:
            cursor.update(cookie=next_cookie, complete=True)
        else:
            _process_entry(cfg, namespace, path / name, name, storage=storage)
            cursor["cookie"] = next_cookie
    bootstrap["cursors"][namespace] = cursor
    record["revision"] += 1
    record["updated_at"] = _utc_now()
    return _write_record(cfg.shared_root, record, storage=storage)


def _start_building(record: dict[str, Any]) -> dict[str, Any]:
    record["state"] = "building"
    bootstrap = record["bootstrap"]
    bootstrap["generation"] = uuid.uuid4().hex
    bootstrap["phase"] = "groups"
    bootstrap["cursors"] = _new_cursors()
    bootstrap["stable_pass"] = 0
    bootstrap["post_fence"] = True
    bootstrap["completed_at"] = None
    record["diagnostic"] = None
    return record


def _transition_locked(
    cfg: RootConfig,
    record: dict[str, Any],
    target: str,
    *,
    storage: Any | None = None,
    registration_checked: bool = False,
    layout_prevalidated: bool = False,
) -> dict[str, Any]:
    current = record["state"]
    allowed = {
        "preparing": {"fenced"},
        "fenced": {"building"},
        "building": {"active"},
        "active": {"degraded"},
        "degraded": {"building"},
    }
    if target not in allowed.get(current, set()):
        raise RuntimeError(f"invalid Group service activation transition {current!r} -> {target!r}")
    if target == "fenced":
        if not registration_checked:
            diagnostic = _registration_participant_diagnostic(cfg.shared_root, storage=storage)
            if diagnostic is not None:
                raise RuntimeError(f"Group service writer fence is not ready: {diagnostic['detail']}")
        layout = (
            locator.read_group_service_layout_marker(cfg.shared_root, storage=storage)
            if layout_prevalidated
            else locator.ensure_group_service_layout(cfg, storage=storage)
        )
        if record["layout"] != {"revision": layout["revision"], "manifest_digest": layout["manifest_digest"]}:
            record["layout"] = {"revision": layout["revision"], "manifest_digest": layout["manifest_digest"]}
        _install_capability(cfg.shared_root, storage=storage)
    elif target == "building":
        if current == "degraded":
            layout = (
                locator.read_group_service_layout_marker(cfg.shared_root, storage=storage)
                if layout_prevalidated
                else locator.ensure_group_service_layout(cfg, storage=storage)
            )
            record["layout"] = {
                "revision": layout["revision"],
                "manifest_digest": layout["manifest_digest"],
            }
        if current == "fenced" and not record["bootstrap"]["post_fence"]:
            record["bootstrap"]["post_fence"] = True
        record = _start_building(record)
    elif target == "active":
        bootstrap = record["bootstrap"]
        schema_record = (
            storage.read_json_limited(shared_paths(cfg.shared_root)["schema"] / "version.json", max_bytes=16 * 1024)
            if storage is not None
            else read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")
        )
        schema = schema_record.get("schema", {})
        layout = (
            locator.read_group_service_layout_marker(cfg.shared_root, storage=storage)
            if layout_prevalidated
            else locator.read_group_service_layout(cfg.shared_root, storage=storage)
        )
        if (
            not bootstrap["post_fence"]
            or bootstrap["phase"] != "complete"
            or bootstrap["stable_pass"] < 1
            or bootstrap["completed_at"] is None
            or GROUP_SERVICE_CAPABILITY not in schema.get("required_capabilities", [])
            or record["layout"] != {"revision": layout["revision"], "manifest_digest": layout["manifest_digest"]}
            or not _meets_writer_floor(__version__)
        ):
            raise RuntimeError("Group service activation transition lacks complete post-fence evidence")
    elif target == "degraded":
        record["diagnostic"] = {"code": "activation_degraded", "detail": "Group service authority is unavailable"}
    record["state"] = target
    record["revision"] += 1
    record["updated_at"] = _utc_now()
    return _write_record(cfg.shared_root, record, storage=storage)


def transition_group_service_state(cfg: RootConfig, target: str) -> dict[str, Any]:
    """Apply one permitted activation transition under the schema fence."""
    if target not in _STATES:
        raise ValueError(f"unsupported Group service activation state: {target!r}")
    with schema_lock(cfg.shared_root):
        record = _read_record(cfg.shared_root)
        if record is None:
            raise RuntimeError("Group service activation transition has no preparing record")
        return _transition_locked(cfg, record, target)


def transition_group_service_state_locked(
    cfg: RootConfig,
    target: str,
    *,
    storage: Any | None = None,
) -> dict[str, Any]:
    """Apply one permitted activation transition when the caller owns the schema fence."""
    if target not in _STATES:
        raise ValueError(f"unsupported Group service activation state: {target!r}")
    record = _read_record(cfg.shared_root, storage=storage)
    if record is None:
        raise RuntimeError("Group service activation transition has no preparing record")
    return _transition_locked(cfg, record, target, storage=storage)


def _record_preparing(
    cfg: RootConfig,
    *,
    storage: Any | None = None,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if layout is None:
        layout = locator.ensure_group_service_layout(cfg, storage=storage)
    record = _new_record(cfg.shared_root, layout, storage=storage)
    return _write_record(cfg.shared_root, record, storage=storage)


def prepare_group_service_activation(
    cfg: RootConfig,
    *,
    storage: Any,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the durable preparing record under the project schema fence."""
    with schema_lock(cfg.shared_root):
        record = _read_record(cfg.shared_root, storage=storage)
        return record if record is not None else _record_preparing(cfg, storage=storage, layout=layout)


def _next_phase(bootstrap: dict[str, Any]) -> bool:
    cursors = bootstrap["cursors"]
    phase = bootstrap["phase"]
    if phase == "groups" and cursors["groups"]["complete"]:
        bootstrap["phase"] = "submissions"
        return True
    if phase == "submissions" and cursors["submissions"]["complete"]:
        bootstrap["phase"] = "active_namespaces"
        return True
    if phase == "active_namespaces" and all(cursors[name]["complete"] for name in _CURSOR_NAMES[2:]):
        bootstrap["phase"] = "stable_pass"
        return True
    return False


def _stable_pass(cfg: RootConfig, record: dict[str, Any], *, storage: Any | None = None) -> dict[str, Any]:
    bootstrap = record["bootstrap"]
    paths = _bootstrap_paths(cfg, storage=storage)
    for name in _CURSOR_NAMES:
        cursor = bootstrap["cursors"][name]
        current = _revision(paths[name])
        if current != cursor["directory_revision"]:
            cursor.update(directory_revision=current, cookie=0, complete=current is None)
            bootstrap["phase"] = _cursor_phase(name)
            record["revision"] += 1
            record["updated_at"] = _utc_now()
            return _write_record(cfg.shared_root, record, storage=storage)
    bootstrap["stable_pass"] += 1
    bootstrap["phase"] = "complete"
    bootstrap["completed_at"] = _utc_now()
    record["revision"] += 1
    record["updated_at"] = _utc_now()
    return _write_record(cfg.shared_root, record, storage=storage)


def advance_group_service_activation_locked(
    cfg: RootConfig,
    *,
    storage: Any | None = None,
    layout_prevalidated: bool = False,
) -> dict[str, Any]:
    """Advance one activation slice when the caller already owns the schema fence."""
    record = _read_record(cfg.shared_root, storage=storage)
    if record is None:
        record = _record_preparing(cfg, storage=storage)
    if record["state"] == "preparing":
        diagnostic = _registration_participant_diagnostic(cfg.shared_root, storage=storage)
        if diagnostic is not None:
            if record["diagnostic"] != diagnostic:
                record["diagnostic"] = diagnostic
                record["revision"] += 1
                record["updated_at"] = _utc_now()
                record = _write_record(cfg.shared_root, record, storage=storage)
            return record
        try:
            return _transition_locked(
                cfg,
                record,
                "fenced",
                storage=storage,
                registration_checked=True,
                layout_prevalidated=layout_prevalidated,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            diagnostic = {"code": "fence_unavailable", "detail": str(exc)[:512]}
            if record["diagnostic"] != diagnostic:
                record["diagnostic"] = diagnostic
                record["revision"] += 1
                record["updated_at"] = _utc_now()
                return _write_record(cfg.shared_root, record, storage=storage)
            return record
    if record["state"] == "fenced":
        return _transition_locked(cfg, record, "building", storage=storage)
    if record["state"] == "degraded":
        return _transition_locked(
            cfg,
            record,
            "building",
            storage=storage,
            layout_prevalidated=layout_prevalidated,
        )
    if record["state"] != "building":
        if record["state"] == "active" and not is_group_service_active(cfg.shared_root, storage=storage):
            return _transition_locked(cfg, record, "degraded", storage=storage)
        return record

    bootstrap = record["bootstrap"]
    if not bootstrap["post_fence"]:
        raise RuntimeError("Group service bootstrap was started before the writer fence")
    if _next_phase(bootstrap):
        record["revision"] += 1
        record["updated_at"] = _utc_now()
        return _write_record(cfg.shared_root, record, storage=storage)
    if bootstrap["phase"] == "stable_pass":
        return _stable_pass(cfg, record, storage=storage)
    if bootstrap["phase"] == "complete":
        return _transition_locked(
            cfg,
            record,
            "active",
            storage=storage,
            layout_prevalidated=layout_prevalidated,
        )
    cursor_name = {
        "groups": "groups",
        "submissions": "submissions",
        "active_namespaces": next(name for name in _CURSOR_NAMES[2:] if not bootstrap["cursors"][name]["complete"]),
    }[bootstrap["phase"]]
    return _advance_cursor(cfg, record, cursor_name, storage=storage)


def advance_group_service_activation(cfg: RootConfig) -> dict[str, Any]:
    """Advance activation by one bounded bootstrap entry or one phase change."""
    with schema_lock(cfg.shared_root):
        return advance_group_service_activation_locked(cfg)


def is_group_service_active(root: Path, *, storage: Any | None = None) -> bool:
    """Return whether active state, post-fence proof, layout, and capability agree."""
    root = Path(root)
    try:
        record = _read_record(root, storage=storage)
        if record is None or record["state"] != "active":
            return False
        bootstrap = record["bootstrap"]
        schema_record = (
            storage.read_json_limited(shared_paths(root)["schema"] / "version.json", max_bytes=16 * 1024)
            if storage is not None
            else read_json(shared_paths(root)["schema"] / "version.json")
        )
        schema = schema_record.get("schema", {})
        layout = locator.read_group_service_layout_marker(root, storage=storage)
        return (
            bootstrap["post_fence"]
            and bootstrap["phase"] == "complete"
            and bootstrap["stable_pass"] >= 1
            and bootstrap["completed_at"] is not None
            and GROUP_SERVICE_CAPABILITY in schema.get("required_capabilities", [])
            and record["layout"] == {"revision": layout["revision"], "manifest_digest": layout["manifest_digest"]}
            and _meets_writer_floor(__version__)
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


__all__ = [
    "PROTOCOL",
    "WRITER_FLOOR",
    "activation_path",
    "advance_group_service_activation",
    "advance_group_service_activation_locked",
    "group_service_registration_diagnostic",
    "is_group_service_active",
    "prepare_group_service_activation",
    "read_group_service_activation_record",
    "transition_group_service_state",
    "transition_group_service_state_locked",
    "writer_floor_satisfied",
]
