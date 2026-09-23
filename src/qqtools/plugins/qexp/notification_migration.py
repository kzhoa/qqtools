"""QQTOOLS-COMPAT-0016: Bounded machine-level notification migration."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

from .agent.bindings import ProjectBinding
from .agent.context import MachineRuntime
from .config_types import RootConfig
from .notification_cleanup import cleanup_credentials
from .notification_policy import load_policy
from .notification_reconciliation import LegacyConflictError, reconcile_legacy
from .runtime.store import atomic_replace

_CURSOR_SCHEMA_VERSION = 1
_MAX_CURSOR_BYTES = 4096
_CURSOR_NAME = "migration-cursor.json"


class NotificationMaintenanceWorker:
    """Advance a bounded notification migration and retention pass off the scheduler thread."""

    def __init__(self, runtime: MachineRuntime) -> None:
        self.runtime = runtime
        self._stop = Event()
        self._thread = Thread(target=self._run, name="qexp-notification-maintenance", daemon=True)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            if not self._stop.is_set():
                advance_notification_migration(self.runtime, limit=4)
            if not self._stop.is_set():
                cleanup_credentials(self.runtime.root, limit=8)
        except (OSError, RuntimeError, ValueError):
            pass
        finally:
            self.runtime.notification_next_pass_at = time.monotonic() + 5.0


def _private_file_stat(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise ValueError("notification migration cursor must be an owner-only regular file.")


def _verify_private_directory(path: Path) -> os.stat_result:
    try:
        original = os.lstat(path)
    except FileNotFoundError:
        raise ValueError("notification migration storage directory is unavailable.") from None
    if not stat.S_ISDIR(original.st_mode) or original.st_uid != os.geteuid():
        raise ValueError("notification migration storage directory is unsafe.")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o700
            or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)
        ):
            raise ValueError("notification migration storage directory is unsafe.")
        return current
    finally:
        os.close(descriptor)


def _ensure_private_directory(runtime_root: Path) -> Path:
    root_info = os.lstat(runtime_root)
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.geteuid():
        raise ValueError("machine runtime root must be an owner-owned directory.")

    path = runtime_root / "notifications"
    created = False
    try:
        os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass

    original = os.lstat(path)
    if not created and (
        not stat.S_ISDIR(original.st_mode) or original.st_uid != os.geteuid() or stat.S_IMODE(original.st_mode) != 0o700
    ):
        raise ValueError("notification migration storage directory is unsafe.")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode) or current.st_uid != os.geteuid():
            raise ValueError("notification migration storage directory is unsafe.")
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError("notification migration storage directory changed while opening.")
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            parent_flags |= getattr(os, "O_CLOEXEC", 0)
            parent_descriptor = os.open(runtime_root, parent_flags)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
            current = os.fstat(descriptor)
        if stat.S_IMODE(current.st_mode) != 0o700:
            raise ValueError("notification migration storage directory is unsafe.")
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    _verify_private_directory(path)
    return path


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate cursor field")
        result[key] = value
    return result


def _read_cursor(path: Path) -> str | None:
    _verify_private_directory(path.parent)
    try:
        original = os.lstat(path)
    except FileNotFoundError:
        return None
    _private_file_stat(original)

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        _private_file_stat(current)
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError("notification migration cursor changed while opening.")
        encoded = bytearray()
        while len(encoded) <= _MAX_CURSOR_BYTES:
            chunk = os.read(descriptor, min(1024, _MAX_CURSOR_BYTES + 1 - len(encoded)))
            if not chunk:
                break
            encoded.extend(chunk)
    finally:
        os.close(descriptor)

    if len(encoded) > _MAX_CURSOR_BYTES:
        return None
    try:
        record = json.loads(bytes(encoded).decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "next_project_id"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != _CURSOR_SCHEMA_VERSION
    ):
        return None
    next_project_id = record["next_project_id"]
    if next_project_id is not None and not isinstance(next_project_id, str):
        return None
    return next_project_id


def _validate_cursor_target(path: Path) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    _private_file_stat(current)


def _write_cursor(path: Path, next_project_id: str | None) -> None:
    _verify_private_directory(path.parent)
    _validate_cursor_target(path)

    def validate_before_replace(temporary_stat: os.stat_result) -> None:
        _private_file_stat(temporary_stat)
        _verify_private_directory(path.parent)
        _validate_cursor_target(path)

    persisted = atomic_replace(
        path,
        {"schema_version": _CURSOR_SCHEMA_VERSION, "next_project_id": next_project_id},
        before_replace=validate_before_replace,
    )
    if persisted is None:
        raise OSError("notification migration cursor replacement could not be verified.")
    _private_file_stat(persisted)
    _verify_private_directory(path.parent)


def _ordered_bindings(bindings: list[ProjectBinding]) -> list[ProjectBinding]:
    return sorted(bindings, key=lambda binding: binding.project_id)


def _add_ids_and_count(result: dict[str, Any], name: str, project_ids: list[str]) -> None:
    result[name] = {"ids": project_ids, "count": len(project_ids)}


def _matches_binding(binding: ProjectBinding, cfg: RootConfig) -> bool:
    return cfg.shared_root == binding.shared_root


def advance_notification_migration(runtime: MachineRuntime, *, limit: int = 4) -> dict[str, Any]:
    """Reconcile a bounded round-robin slice of registered projects.

    Args:
        runtime: The initialized machine runtime whose registered projects are migrated.
        limit: Maximum number of projects to attempt in this call.

    Returns:
        Sanitized IDs and counts for this pass, plus the next registered project ID.
        ``imported`` includes successful no-op reconciliations; ``pending`` includes
        projects not confirmed successful in this bounded pass, including unvisited IDs.

    Raises:
        ValueError: If ``limit`` is not a positive integer or cursor storage is unsafe.
        OSError: If cursor storage cannot be read or durably replaced.

    Call this from the machine maintenance worker, not the scheduling thread.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer.")

    _, bindings = runtime.load_registry()
    ordered_bindings = _ordered_bindings(bindings)
    ordered_ids = [binding.project_id for binding in ordered_bindings]
    inspected_ids: list[str] = []
    imported_ids: list[str] = []
    conflict_ids: list[str] = []
    inaccessible_ids: list[str] = []

    if not ordered_bindings:
        next_cursor = None
    else:
        directory = _ensure_private_directory(runtime.root)
        cursor_path = directory / _CURSOR_NAME
        cursor_id = _read_cursor(cursor_path)
        try:
            start_index = ordered_ids.index(cursor_id) if cursor_id is not None else 0
        except ValueError:
            start_index = 0
        attempt_count = min(limit, len(ordered_bindings))
        selected = [ordered_bindings[(start_index + offset) % len(ordered_bindings)] for offset in range(attempt_count)]

        for binding in selected:
            project_id = binding.project_id
            inspected_ids.append(project_id)
            try:
                cfg = binding.root_config()
                if not _matches_binding(binding, cfg):
                    raise ValueError("registered project root changed during notification migration.")
                reconciled_id = reconcile_legacy(runtime, cfg)
                if reconciled_id != project_id:
                    raise ValueError("reconciled project identity no longer matches its registration.")
            except LegacyConflictError:
                conflict_ids.append(project_id)
            except Exception:
                inaccessible_ids.append(project_id)
            else:
                imported_ids.append(project_id)

        next_index = (start_index + attempt_count) % len(ordered_bindings)
        next_cursor = ordered_ids[next_index]
        _write_cursor(cursor_path, next_cursor)

    pending_set = set(ordered_ids) - set(imported_ids)
    pending_ids = [project_id for project_id in ordered_ids if project_id in pending_set]
    result: dict[str, Any] = {"next_cursor": next_cursor}
    _add_ids_and_count(result, "inspected", inspected_ids)
    _add_ids_and_count(result, "imported", imported_ids)
    _add_ids_and_count(result, "pending", pending_ids)
    _add_ids_and_count(result, "conflicts", conflict_ids)
    _add_ids_and_count(result, "inaccessible", inaccessible_ids)
    return result


def notification_migration_status(runtime: MachineRuntime) -> dict[str, Any]:
    """Report project migration state without reconciling or activating projects.

    Args:
        runtime: The machine runtime whose registered projects are inspected.

    Returns:
        Registered, ready, pending, conflicting, and inaccessible project IDs and counts.
    """
    _, bindings = runtime.load_registry()
    ordered_bindings = _ordered_bindings(bindings)
    registered_ids = [binding.project_id for binding in ordered_bindings]
    ready_ids: list[str] = []
    pending_ids: list[str] = []
    conflict_ids: list[str] = []
    inaccessible_ids: list[str] = []

    for binding in ordered_bindings:
        project_id = binding.project_id
        try:
            cfg = binding.root_config()
            if not _matches_binding(binding, cfg):
                raise ValueError("registered project root changed during notification status inspection.")
            policy = load_policy(runtime.root, "project", project_id)
        except Exception:
            pending_ids.append(project_id)
            inaccessible_ids.append(project_id)
            continue

        legacy = policy.get("legacy")
        if not isinstance(legacy, dict) or legacy.get("status") != "ready":
            pending_ids.append(project_id)
            if isinstance(legacy, dict) and legacy.get("status") == "legacy_conflict":
                conflict_ids.append(project_id)
            if isinstance(legacy, dict) and legacy.get("status") == "source_invalid":
                inaccessible_ids.append(project_id)
            continue
        ready_ids.append(project_id)

    result: dict[str, Any] = {}
    _add_ids_and_count(result, "registered", registered_ids)
    _add_ids_and_count(result, "ready", ready_ids)
    _add_ids_and_count(result, "pending", pending_ids)
    _add_ids_and_count(result, "conflicts", conflict_ids)
    _add_ids_and_count(result, "inaccessible", inaccessible_ids)
    return result
