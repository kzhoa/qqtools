"""Private, revisioned notification policy storage."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .runtime.locks import exclusive
from .runtime.store import CASConflict, atomic_replace

_SCHEMA_VERSION = 1
_MIN_TIMEOUT_SECONDS = 0.5
_MAX_TIMEOUT_SECONDS = 30.0
_MAX_POLICY_BYTES = 65_536
_PROJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_POLICY_FIELDS = frozenset({"schema_version", "revision", "override"})
_OVERRIDE_FIELDS = frozenset({"enabled", "timeout_seconds", "destination"})
_DESTINATION_COMMON_FIELDS = frozenset({"provider", "source", "signing"})
_PRIVATE_DESTINATION_FIELDS = _DESTINATION_COMMON_FIELDS | {"credential_id"}
_ENV_DESTINATION_FIELDS = _DESTINATION_COMMON_FIELDS | {"webhook_env"}
_LEGACY_FIELDS = frozenset({"fingerprint", "imported_revision", "status"})
_PRESERVE_LEGACY = object()


def _runtime_root_path(runtime_root: Path) -> Path:
    if not isinstance(runtime_root, Path):
        raise ValueError("runtime_root must be a pathlib.Path.")
    expanded = runtime_root.expanduser()
    if "\x00" in os.fspath(expanded):
        raise ValueError("runtime_root is invalid.")
    return Path(os.path.abspath(expanded))


def _scope_path(root: Path, scope: str, project_id: str | None) -> Path:
    if scope == "global":
        if project_id is not None:
            raise ValueError("project_id must be omitted for global notification policy.")
        return root / "notifications" / "policy-global.json"
    if scope != "project":
        raise ValueError("scope must be 'global' or 'project'.")
    if not isinstance(project_id, str) or _PROJECT_ID_RE.fullmatch(project_id) is None:
        raise ValueError("project_id must be a safe stable identifier.")
    return root / "notifications" / f"policy-project-{project_id}.json"


def policy_path(runtime_root: Path, scope: str, project_id: str | None = None) -> Path:
    """Return the storage path for a global policy or stable project override."""
    root = _runtime_root_path(runtime_root)
    return _scope_path(root, scope, project_id)


def _validate_signing(value: Any) -> str | dict[str, str]:
    if value == "unsigned":
        return "unsigned"
    if not isinstance(value, dict) or set(value) != {"env"}:
        raise ValueError("destination.signing must be 'unsigned' or an environment reference.")
    name = value["env"]
    if not isinstance(name, str) or _ENV_NAME_RE.fullmatch(name) is None:
        raise ValueError("destination.signing.env must be a valid environment variable name.")
    return {"env": name}


def _validate_destination(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("destination must be a complete object.")
    source = value.get("source")
    if source == "private_file":
        if set(value) != _PRIVATE_DESTINATION_FIELDS:
            raise ValueError("private_file destination has missing or unknown fields.")
        credential_id = value["credential_id"]
        if not isinstance(credential_id, str) or re.fullmatch(r"[a-f0-9]{32}", credential_id) is None:
            raise ValueError("destination.credential_id must be 32 lowercase UUID hex characters.")
    elif source == "env":
        if set(value) != _ENV_DESTINATION_FIELDS:
            raise ValueError("env destination has missing or unknown fields.")
        webhook_env = value["webhook_env"]
        if not isinstance(webhook_env, str) or _ENV_NAME_RE.fullmatch(webhook_env) is None:
            raise ValueError("destination.webhook_env must be a valid environment variable name.")
    else:
        raise ValueError("destination.source must be 'private_file' or 'env'.")
    if value.get("provider") != "feishu":
        raise ValueError("destination.provider must be 'feishu'.")
    result = dict(value)
    result["signing"] = _validate_signing(value["signing"])
    return result


def validate_override(override: dict) -> dict:
    """Validate and return a sparse canonical notification override."""
    if not isinstance(override, dict):
        raise ValueError("notification override must be an object.")
    unknown = set(override) - _OVERRIDE_FIELDS
    if unknown:
        raise ValueError("notification override contains unknown fields.")
    result: dict[str, Any] = {}
    if "enabled" in override:
        if type(override["enabled"]) is not bool:
            raise ValueError("notification override.enabled must be boolean.")
        result["enabled"] = override["enabled"]
    if "timeout_seconds" in override:
        timeout = override["timeout_seconds"]
        if type(timeout) not in (int, float):
            raise ValueError("notification override.timeout_seconds must be a finite number.")
        if type(timeout) is float and not math.isfinite(timeout):
            raise ValueError("notification override.timeout_seconds must be finite.")
        if not _MIN_TIMEOUT_SECONDS <= timeout <= _MAX_TIMEOUT_SECONDS:
            raise ValueError("notification override.timeout_seconds must be between 0.5 and 30 seconds.")
        result["timeout_seconds"] = timeout
    if "destination" in override:
        result["destination"] = _validate_destination(override["destination"])
    return result


def _validate_runtime_root(root: Path, *, missing_ok: bool) -> bool:
    try:
        value = os.lstat(root)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise ValueError("runtime_root must be an existing directory.") from None
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid():
        raise ValueError("runtime_root must be a directory owned by the current user.")
    return True


def _open_private_directory(path: Path, *, create: bool) -> int | None:
    created = False
    try:
        original = os.lstat(path)
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        original = os.lstat(path)
    if not created and (
        not stat.S_ISDIR(original.st_mode) or original.st_uid != os.geteuid() or stat.S_IMODE(original.st_mode) != 0o700
    ):
        raise ValueError(f"notification storage directory is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode) or current.st_uid != os.geteuid():
            raise ValueError(f"notification storage directory is unsafe: {path}")
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            parent_fd = os.open(path.parent, parent_flags | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            current = os.fstat(descriptor)
        elif (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError(f"notification storage directory changed while opening: {path}")
        if stat.S_IMODE(current.st_mode) != 0o700:
            raise ValueError(f"notification storage directory is unsafe: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _private_file_stat(value: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular owner-only file.")


def _read_private_json(path: Path, scope: str) -> dict[str, Any] | None:
    try:
        original = os.lstat(path)
    except FileNotFoundError:
        return None
    _private_file_stat(original, "notification policy")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        _private_file_stat(current, "notification policy")
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError("notification policy changed while opening.")
        if current.st_size > _MAX_POLICY_BYTES:
            raise ValueError("notification policy record is too large.")
        encoded = bytearray()
        while len(encoded) <= _MAX_POLICY_BYTES:
            chunk = os.read(descriptor, min(8192, _MAX_POLICY_BYTES + 1 - len(encoded)))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) > _MAX_POLICY_BYTES:
            raise ValueError("notification policy record is too large.")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(encoded).decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise ValueError("notification policy JSON is malformed.") from None
    if not isinstance(value, dict):
        raise ValueError("notification policy record has missing or unknown fields.")
    fields = set(value)
    expected_fields = _POLICY_FIELDS | ({"legacy"} if scope == "project" and "legacy" in fields else set())
    if fields != expected_fields:
        raise ValueError("notification policy record has missing or unknown fields.")
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("notification policy schema_version is unsupported.")
    revision = value["revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("notification policy revision is invalid.")
    override = value["override"]
    if override is not None:
        try:
            override = validate_override(override)
        except (TypeError, ValueError):
            raise ValueError("notification policy override is malformed.") from None
    result = {"schema_version": _SCHEMA_VERSION, "revision": revision, "override": override}
    if "legacy" in value:
        result["legacy"] = _validate_legacy_metadata(value["legacy"])
    return result


def _validate_legacy_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LEGACY_FIELDS:
        raise ValueError("notification policy legacy metadata is malformed.")
    fingerprint = value["fingerprint"]
    imported_revision = value["imported_revision"]
    status = value["status"]
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValueError("notification policy legacy fingerprint is invalid.")
    if type(imported_revision) is not int or imported_revision < 0:
        raise ValueError("notification policy legacy imported_revision is invalid.")
    if not isinstance(status, str) or status not in {"ready", "legacy_conflict", "source_invalid"}:
        raise ValueError("notification policy legacy status is invalid.")
    return {
        "fingerprint": fingerprint,
        "imported_revision": imported_revision,
        "status": status,
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _default_record() -> dict[str, Any]:
    return {"schema_version": _SCHEMA_VERSION, "revision": 0, "override": None}


def _load_policy_at_path(path: Path, scope: str) -> dict[str, Any]:
    record = _read_private_json(path, scope)
    return _default_record() if record is None else record


@contextmanager
def policy_guard(runtime_root: Path, *, blocking: bool = True) -> Iterator[None]:
    """Lock notification policy access and establish its private storage directory."""
    root = _runtime_root_path(runtime_root)
    _validate_runtime_root(root, missing_ok=False)
    directory = _open_private_directory(root / "notifications", create=True)
    if directory is None:
        raise ValueError("notification storage directory is unavailable.")
    os.close(directory)
    lock_path = root / "notifications" / "policy.lock"
    _prepare_lock_file(lock_path)
    with exclusive(lock_path, blocking=blocking) as acquired:
        if not acquired:
            raise RuntimeError("Notification policy is busy; retry after the current update.")
        yield


def load_policy_unlocked(runtime_root: Path, scope: str, project_id: str | None = None) -> dict[str, Any]:
    """Load one policy while the caller holds policy_guard(runtime_root)."""
    root = _runtime_root_path(runtime_root)
    path = _scope_path(root, scope, project_id)
    if not _validate_runtime_root(root, missing_ok=True):
        return _default_record()
    directory = _open_private_directory(root / "notifications", create=False)
    if directory is None:
        return _default_record()
    os.close(directory)
    return _load_policy_at_path(path, scope)


def load_policy(runtime_root: Path, scope: str, project_id: str | None = None) -> dict[str, Any]:
    """Load a strict policy record, returning built-in defaults only when absent."""
    root = _runtime_root_path(runtime_root)
    _scope_path(root, scope, project_id)
    if not _validate_runtime_root(root, missing_ok=True):
        return _default_record()
    with policy_guard(root):
        return load_policy_unlocked(root, scope, project_id)


def _prepare_lock_file(path: Path) -> None:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(path, flags)
        created = False
    try:
        if created:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        _private_file_stat(os.fstat(descriptor), "notification policy lock")
    finally:
        os.close(descriptor)


def _check_policy_target(path: Path) -> None:
    try:
        value = os.lstat(path)
    except FileNotFoundError:
        return
    _private_file_stat(value, "notification policy")


def _write_policy(path: Path, value: dict[str, Any]) -> None:
    _check_policy_target(path)

    def verify_temporary_file(temporary_stat: os.stat_result) -> None:
        _private_file_stat(temporary_stat, "temporary notification policy")
        _check_policy_target(path)

    persisted = atomic_replace(path, value, before_replace=verify_temporary_file)
    if persisted is None:
        raise OSError("notification policy replacement could not be verified.")
    _private_file_stat(persisted, "notification policy")


def replace_policy_unlocked(
    runtime_root: Path,
    scope: str,
    expected_revision: int,
    override: dict | None,
    project_id: str | None = None,
    *,
    legacy: dict[str, Any] | None | object = _PRESERVE_LEGACY,
) -> dict[str, Any]:
    """Replace one policy while policy_guard(runtime_root) is held by the caller."""
    root = _runtime_root_path(runtime_root)
    path = _scope_path(root, scope, project_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer.")
    candidate = None if override is None else validate_override(override)
    if legacy is not _PRESERVE_LEGACY:
        if scope != "project":
            raise ValueError("legacy metadata is supported only for project policy.")
        if legacy is not None:
            legacy = _validate_legacy_metadata(legacy)
    _validate_runtime_root(root, missing_ok=False)
    directory = _open_private_directory(root / "notifications", create=False)
    if directory is None:
        raise ValueError("notification storage directory is unavailable.")
    os.close(directory)
    current = _load_policy_at_path(path, scope)
    actual_revision = current["revision"]
    if actual_revision != expected_revision:
        raise CASConflict(
            f"Notification policy revision conflict: expected {expected_revision}, got {actual_revision}."
        )
    updated = {
        "schema_version": _SCHEMA_VERSION,
        "revision": expected_revision + 1,
        "override": candidate,
    }
    if legacy is _PRESERVE_LEGACY:
        if "legacy" in current:
            updated["legacy"] = current["legacy"]
    elif legacy is not None:
        updated["legacy"] = legacy
    _write_policy(path, updated)
    return updated


def replace_policy(
    runtime_root: Path,
    scope: str,
    expected_revision: int,
    override: dict | None,
    project_id: str | None = None,
    *,
    legacy: dict[str, Any] | None | object = _PRESERVE_LEGACY,
) -> dict[str, Any]:
    """Compare and atomically replace one policy, retaining resets as tombstones."""
    root = _runtime_root_path(runtime_root)
    _scope_path(root, scope, project_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer.")
    if override is not None:
        validate_override(override)
    if legacy is not _PRESERVE_LEGACY:
        if scope != "project":
            raise ValueError("legacy metadata is supported only for project policy.")
        if legacy is not None:
            _validate_legacy_metadata(legacy)
    if not _validate_runtime_root(root, missing_ok=False):
        raise ValueError("runtime_root must be an existing directory.")
    with policy_guard(root):
        return replace_policy_unlocked(
            root,
            scope,
            expected_revision,
            override,
            project_id,
            legacy=legacy,
        )
