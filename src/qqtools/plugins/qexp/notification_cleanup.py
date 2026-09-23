"""Conservative retention and cleanup for private notification credentials."""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import stat
import time
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .notification_credentials import credential_path, read_webhook
from .notification_policy import load_policy_unlocked, policy_guard
from .runtime.store import atomic_replace

_CREDENTIAL_FILENAME_RE = re.compile(r"([a-f0-9]{32})\.json", re.ASCII)
_CREATED_AT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", re.ASCII)
_PROJECT_POLICY_RE = re.compile(r"policy-project-([A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json", re.ASCII)
_RETENTION_FILENAME = "retention.json"
_RETENTION_SCHEMA_VERSION = 1
_MAX_CREDENTIAL_BYTES = 65_536
_MAX_RETENTION_BYTES = 4_194_304
_MAX_CLEANUP_LIMIT = 1024
_STATUS_SCAN_LIMIT = 4096
_ORPHAN_RETENTION_SECONDS = 24 * 60 * 60
_REFERENCED_RETENTION_SECONDS = 7 * 24 * 60 * 60
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS |= getattr(os, "O_CLOEXEC", 0)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _runtime_root_path(runtime_root: Path) -> Path:
    if not isinstance(runtime_root, Path):
        raise ValueError("runtime_root must be a pathlib.Path.")
    expanded = runtime_root.expanduser()
    if "\x00" in os.fspath(expanded):
        raise ValueError("runtime_root is invalid.")
    return Path(os.path.abspath(expanded))


def _open_owned_directory(parent_fd: int, name: str, *, optional: bool = False) -> int | None:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if optional:
            return None
        raise
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid() or stat.S_IMODE(value.st_mode) != 0o700:
            raise ValueError("notification storage directory is unsafe.")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_notifications_directory(root: Path, stack: ExitStack) -> int:
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    stack.callback(os.close, root_fd)
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.geteuid():
        raise ValueError("runtime_root must be a directory owned by the current user.")
    notifications_fd = _open_owned_directory(root_fd, "notifications")
    if notifications_fd is None:
        raise ValueError("notification storage directory is unavailable.")
    stack.callback(os.close, notifications_fd)
    return notifications_fd


def _private_file_stat(value: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular owner-only file.")


def _stat_witness(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _read_descriptor(descriptor: int, maximum: int, label: str) -> bytes:
    content = bytearray()
    while len(content) <= maximum:
        chunk = os.read(descriptor, min(8192, maximum + 1 - len(content)))
        if not chunk:
            break
        content.extend(chunk)
    if len(content) > maximum:
        raise ValueError(f"{label} is too large.")
    return bytes(content)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _decode_json(encoded: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError):
        raise ValueError(f"{label} is malformed.") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} is malformed.")
    return value


def _timestamp(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("retention timestamp is invalid.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("retention timestamp is invalid.") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError("retention timestamp is invalid.")
    return result


def _created_at_timestamp(value: Any) -> float:
    if not isinstance(value, str) or _CREATED_AT_RE.fullmatch(value) is None:
        raise ValueError("credential created_at is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        result = parsed.timestamp()
    except (OSError, OverflowError, ValueError):
        raise ValueError("credential created_at is invalid.") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("credential created_at must be UTC.")
    if not math.isfinite(result) or result < 0:
        raise ValueError("credential created_at is invalid.")
    return result


def _empty_retention() -> dict[str, Any]:
    return {
        "schema_version": _RETENTION_SCHEMA_VERSION,
        "last_cleanup_at": None,
        "cursor": None,
        "credentials": {},
    }


def _validate_retention(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"schema_version", "last_cleanup_at", "cursor", "credentials"}:
        raise ValueError("notification retention metadata has unknown or missing fields.")
    if type(value["schema_version"]) is not int or value["schema_version"] != _RETENTION_SCHEMA_VERSION:
        raise ValueError("notification retention metadata version is unsupported.")

    last_cleanup_at = value["last_cleanup_at"]
    if last_cleanup_at is not None:
        last_cleanup_at = _timestamp(last_cleanup_at)
    cursor = value["cursor"]
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("notification retention cursor is invalid.")
    if cursor is not None and re.fullmatch(r"[a-f0-9]{32}", cursor, re.ASCII) is None:
        raise ValueError("notification retention cursor is invalid.")

    credentials = value["credentials"]
    if not isinstance(credentials, dict):
        raise ValueError("notification retention credentials are invalid.")
    validated_credentials: dict[str, dict[str, Any]] = {}
    for credential_id, record in credentials.items():
        if not isinstance(credential_id, str) or re.fullmatch(r"[a-f0-9]{32}", credential_id, re.ASCII) is None:
            raise ValueError("notification retention credential ID is invalid.")
        if not isinstance(record, dict) or set(record) != {"ever_referenced", "unreferenced_since"}:
            raise ValueError("notification retention credential record is invalid.")
        ever_referenced = record["ever_referenced"]
        if type(ever_referenced) is not bool:
            raise ValueError("notification retention reference state is invalid.")
        unreferenced_since = record["unreferenced_since"]
        if unreferenced_since is not None:
            unreferenced_since = _timestamp(unreferenced_since)
        if not ever_referenced and unreferenced_since is not None:
            raise ValueError("notification retention reference state is inconsistent.")
        validated_credentials[credential_id] = {
            "ever_referenced": ever_referenced,
            "unreferenced_since": unreferenced_since,
        }

    return {
        "schema_version": _RETENTION_SCHEMA_VERSION,
        "last_cleanup_at": last_cleanup_at,
        "cursor": cursor,
        "credentials": validated_credentials,
    }


def _read_retention(notifications_fd: int) -> dict[str, Any]:
    try:
        initial_stat = os.stat(_RETENTION_FILENAME, dir_fd=notifications_fd, follow_symlinks=False)
    except FileNotFoundError:
        return _empty_retention()
    _private_file_stat(initial_stat, "notification retention metadata")
    descriptor = os.open(_RETENTION_FILENAME, _READ_FLAGS, dir_fd=notifications_fd)
    try:
        opened_stat = os.fstat(descriptor)
        _private_file_stat(opened_stat, "notification retention metadata")
        if _stat_witness(initial_stat) != _stat_witness(opened_stat):
            raise ValueError("notification retention metadata changed while opening.")
        if opened_stat.st_size > _MAX_RETENTION_BYTES:
            raise ValueError("notification retention metadata is too large.")
        encoded = _read_descriptor(descriptor, _MAX_RETENTION_BYTES, "notification retention metadata")
        final_stat = os.fstat(descriptor)
        _private_file_stat(final_stat, "notification retention metadata")
        published_stat = os.stat(_RETENTION_FILENAME, dir_fd=notifications_fd, follow_symlinks=False)
        if _stat_witness(opened_stat) != _stat_witness(final_stat) or _stat_witness(final_stat) != _stat_witness(
            published_stat
        ):
            raise ValueError("notification retention metadata changed while reading.")
    finally:
        os.close(descriptor)
    return _validate_retention(_decode_json(encoded, "notification retention metadata"))


def _write_retention(root: Path, notifications_fd: int, value: dict[str, Any]) -> None:
    notifications_stat = os.fstat(notifications_fd)
    if (
        not stat.S_ISDIR(notifications_stat.st_mode)
        or notifications_stat.st_uid != os.geteuid()
        or stat.S_IMODE(notifications_stat.st_mode) != 0o700
    ):
        raise ValueError("notification storage directory is unsafe.")

    path = root / "notifications" / _RETENTION_FILENAME

    def verify_before_replace(temporary_stat: os.stat_result) -> None:
        _private_file_stat(temporary_stat, "temporary notification retention metadata")
        try:
            current_stat = os.lstat(path)
        except FileNotFoundError:
            return
        _private_file_stat(current_stat, "notification retention metadata")

    persisted = atomic_replace(path, value, before_replace=verify_before_replace)
    if persisted is None:
        raise OSError("notification retention metadata replacement could not be verified.")
    _private_file_stat(persisted, "notification retention metadata")


def _policy_references(root: Path, notifications_fd: int, *, scan_limit: int | None = None) -> tuple[set[str], bool]:
    project_ids: set[str] = set()
    with os.scandir(os.dup(notifications_fd)) as entries:
        for inspected, entry in enumerate(entries, start=1):
            if scan_limit is not None and inspected > scan_limit:
                raise ValueError("notification status scan limit exceeded")
            filename = entry.name
            if filename == "policy-global.json":
                continue
            if not filename.startswith("policy-"):
                continue
            match = _PROJECT_POLICY_RE.fullmatch(filename)
            if match is None:
                raise ValueError("notification policy filename is unknown.")
            project_ids.add(match.group(1))

    records = [load_policy_unlocked(root, "global")]
    records.extend(load_policy_unlocked(root, "project", project_id) for project_id in sorted(project_ids))
    referenced_ids: set[str] = set()
    unresolved_conflict = False
    for record in records:
        legacy = record.get("legacy")
        if isinstance(legacy, dict) and legacy.get("status") in {"legacy_conflict", "source_invalid"}:
            unresolved_conflict = True
        override = record["override"]
        if override is None:
            continue
        destination = override.get("destination")
        if isinstance(destination, dict) and destination.get("source") == "private_file":
            referenced_ids.add(destination["credential_id"])
    return referenced_ids, unresolved_conflict


def _keep_smallest(heap: list[tuple[int, str]], credential_id: str, limit: int) -> None:
    item = (-int(credential_id, 16), credential_id)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _select_credential_ids(
    credentials_fd: int, cursor: str | None, limit: int, *, scan_limit: int | None = None
) -> list[str]:
    after_cursor: list[tuple[int, str]] = []
    before_cursor: list[tuple[int, str]] = []
    cursor_value = -1 if cursor is None else int(cursor, 16)
    with os.scandir(os.dup(credentials_fd)) as entries:
        for inspected, entry in enumerate(entries, start=1):
            if scan_limit is not None and inspected > scan_limit:
                raise ValueError("notification status scan limit exceeded")
            match = _CREDENTIAL_FILENAME_RE.fullmatch(entry.name)
            if match is None:
                continue
            credential_id = match.group(1)
            credential_value = int(credential_id, 16)
            if cursor is None or credential_value > cursor_value:
                _keep_smallest(after_cursor, credential_id, limit)
            elif len(after_cursor) < limit:
                _keep_smallest(before_cursor, credential_id, limit)
    selected = sorted(credential_id for _, credential_id in after_cursor)
    if len(selected) < limit:
        wrapped = sorted(credential_id for _, credential_id in before_cursor)
        selected.extend(wrapped[: limit - len(selected)])
    return selected


def _read_credential_metadata(
    root: Path,
    credentials_fd: int,
    credential_id: str,
) -> tuple[float, tuple[int, int, int, int, int]]:
    filename = credential_path(root, credential_id).name
    initial_stat = os.stat(filename, dir_fd=credentials_fd, follow_symlinks=False)
    _private_file_stat(initial_stat, "notification credential")
    descriptor = os.open(filename, _READ_FLAGS, dir_fd=credentials_fd)
    try:
        opened_stat = os.fstat(descriptor)
        _private_file_stat(opened_stat, "notification credential")
        if _stat_witness(initial_stat) != _stat_witness(opened_stat):
            raise ValueError("notification credential changed while opening.")
        if opened_stat.st_size > _MAX_CREDENTIAL_BYTES:
            raise ValueError("notification credential is too large.")
        encoded = _read_descriptor(descriptor, _MAX_CREDENTIAL_BYTES, "notification credential")
        final_stat = os.fstat(descriptor)
        _private_file_stat(final_stat, "notification credential")
        published_stat = os.stat(filename, dir_fd=credentials_fd, follow_symlinks=False)
        if _stat_witness(opened_stat) != _stat_witness(final_stat) or _stat_witness(final_stat) != _stat_witness(
            published_stat
        ):
            raise ValueError("notification credential changed while reading.")
    finally:
        os.close(descriptor)

    value = _decode_json(encoded, "notification credential metadata")
    if (
        set(value) != {"schema_version", "credential_id", "created_at", "webhook"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["credential_id"] != credential_id
        or not isinstance(value["webhook"], str)
    ):
        raise ValueError("notification credential metadata is malformed.")
    return _created_at_timestamp(value["created_at"]), _stat_witness(final_stat)


def _result(retained: int, eligible: int, deleted: int, diagnostic: str | None) -> dict[str, int | str | None]:
    return {
        "retained": retained,
        "eligible": eligible,
        "deleted": deleted,
        "diagnostic": diagnostic,
    }


def mark_reference(runtime_root: Path, credential_id: str) -> None:
    """Durably mark a credential as activated before publishing a policy reference.

    The caller must hold ``policy_guard(runtime_root)`` through the subsequent policy CAS.

    Args:
        runtime_root: Machine runtime root containing private notification storage.
        credential_id: Canonical private credential identifier.

    Raises:
        ValueError: If the credential or retention metadata is unsafe or unavailable.
    """
    root = _runtime_root_path(runtime_root)
    try:
        credential_path(root, credential_id)
        with ExitStack() as stack:
            notifications_fd = _open_notifications_directory(root, stack)
            retention = _read_retention(notifications_fd)
            read_webhook(root, credential_id)
            state = retention["credentials"].get(
                credential_id,
                {"ever_referenced": False, "unreferenced_since": None},
            )
            state["ever_referenced"] = True
            state["unreferenced_since"] = None
            retention["credentials"][credential_id] = state
            _write_retention(root, notifications_fd, retention)
    except (OSError, TypeError, ValueError, RecursionError):
        raise ValueError("notification reference could not be recorded safely.") from None


def _cleanup_locked(root: Path, now: float, limit: int, *, dry_run: bool = False) -> dict[str, int | str | None]:
    with ExitStack() as stack:
        notifications_fd = _open_notifications_directory(root, stack)
        try:
            referenced_ids, unresolved_conflict = _policy_references(
                root, notifications_fd, scan_limit=_STATUS_SCAN_LIMIT if dry_run else None
            )
        except (OSError, TypeError, ValueError, RecursionError):
            return _result(0, 0, 0, "notification policy scan is unsafe or exceeds status limit; inspection deferred.")

        try:
            retention = _read_retention(notifications_fd)
        except (OSError, TypeError, ValueError, RecursionError):
            return _result(0, 0, 0, "notification retention metadata is unsafe or invalid; cleanup deferred.")

        last_cleanup_at = retention["last_cleanup_at"]
        if last_cleanup_at is not None and now < last_cleanup_at:
            return _result(0, 0, 0, "system clock moved backward; credential cleanup deferred.")

        credentials_fd = _open_owned_directory(notifications_fd, "credentials", optional=True)
        if credentials_fd is not None:
            stack.callback(os.close, credentials_fd)
            try:
                credential_ids = _select_credential_ids(
                    credentials_fd, retention["cursor"], limit, scan_limit=_STATUS_SCAN_LIMIT if dry_run else None
                )
            except (OSError, ValueError):
                return _result(
                    0, 0, 0, "private credential scan is unsafe or exceeds status limit; inspection deferred."
                )
        else:
            credential_ids = []

        metadata: dict[str, tuple[float, tuple[int, int, int, int, int]]] = {}
        try:
            if credentials_fd is not None:
                for credential_id in credential_ids:
                    metadata[credential_id] = _read_credential_metadata(root, credentials_fd, credential_id)
        except (OSError, TypeError, ValueError, RecursionError):
            return _result(
                len(credential_ids),
                0,
                0,
                "credential metadata is unsafe or invalid; cleanup deferred.",
            )

        updated_credentials = dict(retention["credentials"])
        eligible_ids: list[str] = []
        for credential_id, (created_at, _) in metadata.items():
            state = updated_credentials.get(
                credential_id,
                {"ever_referenced": False, "unreferenced_since": None},
            )
            if unresolved_conflict or credential_id in referenced_ids:
                state = {"ever_referenced": True, "unreferenced_since": None}
            elif state["ever_referenced"]:
                if state["unreferenced_since"] is None:
                    state["unreferenced_since"] = now
                if (
                    now >= state["unreferenced_since"]
                    and now - state["unreferenced_since"] >= _REFERENCED_RETENTION_SECONDS
                ):
                    eligible_ids.append(credential_id)
            elif now >= created_at and now - created_at >= _ORPHAN_RETENTION_SECONDS:
                eligible_ids.append(credential_id)
            if state["ever_referenced"]:
                updated_credentials[credential_id] = state

        updated_retention = {
            "schema_version": _RETENTION_SCHEMA_VERSION,
            "last_cleanup_at": now,
            "cursor": credential_ids[-1] if credential_ids else None,
            "credentials": updated_credentials,
        }
        if dry_run:
            return _result(len(credential_ids), len(eligible_ids), 0, None)
        try:
            _write_retention(root, notifications_fd, updated_retention)
        except (OSError, TypeError, ValueError, RecursionError):
            return _result(
                len(credential_ids),
                len(eligible_ids),
                0,
                "notification retention metadata could not be safely persisted; cleanup deferred.",
            )

        deleted = 0
        if credentials_fd is not None:
            for credential_id in eligible_ids:
                filename = credential_path(root, credential_id).name
                _, witness = metadata[credential_id]
                try:
                    current_stat = os.stat(filename, dir_fd=credentials_fd, follow_symlinks=False)
                    _private_file_stat(current_stat, "notification credential")
                    if _stat_witness(current_stat) != witness:
                        raise ValueError("notification credential changed before deletion.")
                    os.unlink(filename, dir_fd=credentials_fd)
                    deleted += 1
                    os.fsync(credentials_fd)
                except (OSError, TypeError, ValueError, RecursionError):
                    return _result(
                        len(credential_ids) - deleted,
                        len(eligible_ids),
                        deleted,
                        "credential deletion could not be safely completed; cleanup deferred.",
                    )

        return _result(
            len(credential_ids) - deleted,
            len(eligible_ids),
            deleted,
            None,
        )


def cleanup_credentials(
    runtime_root: Path,
    *,
    now: float | None = None,
    limit: int = 32,
) -> dict[str, int | str | None]:
    """Retain reachable credentials and delete only aged private orphans.

    Args:
        runtime_root: Machine runtime root containing private notification storage.
        now: Optional Unix timestamp for deterministic maintenance decisions.
        limit: Maximum number of credential records inspected in this pass.

    Returns:
        Per-pass retained, eligible, and deleted counts plus a sanitized diagnostic.
    """
    root = _runtime_root_path(runtime_root)
    if type(limit) is not int or not 1 <= limit <= _MAX_CLEANUP_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {_MAX_CLEANUP_LIMIT}.")
    current_time = _timestamp(time.time() if now is None else now)
    try:
        with policy_guard(root):
            return _cleanup_locked(root, current_time, limit)
    except (OSError, TypeError, ValueError, RecursionError):
        return _result(0, 0, 0, "private notification storage is unsafe or unavailable; cleanup deferred.")


def credential_retention_status(runtime_root: Path, *, limit: int = 8) -> dict[str, int | str | None]:
    """Inspect one bounded retention slice without publishing or deleting anything."""
    root = _runtime_root_path(runtime_root)
    if type(limit) is not int or not 1 <= limit <= _MAX_CLEANUP_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {_MAX_CLEANUP_LIMIT}.")
    try:
        if not (root / "notifications").exists():
            return _result(0, 0, 0, None)
        with policy_guard(root):
            return _cleanup_locked(root, _timestamp(time.time()), limit, dry_run=True)
    except (OSError, TypeError, ValueError, RecursionError):
        return _result(0, 0, 0, "private notification storage is unsafe or unavailable; inspection deferred.")


__all__ = ["cleanup_credentials", "credential_retention_status", "mark_reference"]
