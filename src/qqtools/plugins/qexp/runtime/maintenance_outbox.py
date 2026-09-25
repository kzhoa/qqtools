"""Durable per-project descriptors for bounded maintenance work."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..layout import project_id
from .directory_capture import read_directory_entry
from .locks import exclusive
from .project_activation import project_activation_transaction
from .records import new_id, utc_now, validate_identifier
from .store import atomic_replace, read_json_limited, require_json_size

DESCRIPTOR_SCHEMA_VERSION = 1
DESCRIPTOR_MAX_BYTES = 64 * 1024
_QUEUE_SLOT_SCHEMA_VERSION = 1
_QUEUE_SLOT_MAX_BYTES = DESCRIPTOR_MAX_BYTES + 4096
_ACTIVE_STATES = frozenset({"prepared", "pending", "running", "waiting"})
_TERMINAL_STATES = frozenset({"completed", "intervention", "superseded"})
_IDENTITY_FIELDS = frozenset({"project_id", "kind", "target_id", "work_generation"})
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "queue_position",
        "progress_revision",
        "phase",
        "cursor",
        "state",
        "due_at",
        "retry_count",
        "created_at",
        "updated_at",
        "meaningful_progress_at",
        "failure",
        "retirement_proof",
    }
)


def maintenance_root(cfg: object) -> Path:
    return Path(cfg.shared_root) / "operations" / "maintenance-v1"


def _active_root(cfg: object) -> Path:
    return maintenance_root(cfg) / "active"


def _retired_root(cfg: object) -> Path:
    return maintenance_root(cfg) / "retired"


def _index_root(cfg: object) -> Path:
    return maintenance_root(cfg) / "index"


def _state_path(cfg: object) -> Path:
    return maintenance_root(cfg) / "queue.json"


def _service_path(cfg: object) -> Path:
    return maintenance_root(cfg) / "service.json"


def _slot_root(cfg: object) -> Path:
    return maintenance_root(cfg) / "queue" / "slots"


def _slot_path(cfg: object, queue_position: int) -> Path:
    return _slot_root(cfg) / f"{queue_position:020d}.json"


def _queue_index_path(cfg: object) -> Path:
    return maintenance_root(cfg) / "queue-index.sqlite3"


def _target_index_path(cfg: object, kind: str, target_id: str) -> Path:
    identity = {"project_id": project_id(Path(cfg.shared_root)), "kind": kind, "target_id": target_id}
    return maintenance_root(cfg) / "targets" / f"{_identity_key(identity)}.json"


def _lock_path(cfg: object) -> Path:
    return Path(cfg.shared_root) / "locks" / "maintenance-outbox-v1.lock"


def _identity(cfg: object, kind: str, target_id: str, work_generation: str) -> dict[str, str]:
    if not isinstance(kind, str) or not kind or len(kind) > 32 or not kind.replace("_", "").isalnum():
        raise ValueError("maintenance kind is invalid.")
    validate_identifier(target_id, "maintenance target_id")
    validate_identifier(work_generation, "maintenance work_generation")
    return {
        "project_id": project_id(Path(cfg.shared_root)),
        "kind": kind,
        "target_id": target_id,
        "work_generation": work_generation,
    }


def _identity_key(identity: dict[str, str]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _activation_reason(identity: dict[str, str], progress_revision: int) -> str:
    return f"mw:{_identity_key(identity)}:{progress_revision}"


def _record_path(cfg: object, identity: dict[str, str], location: str) -> Path:
    key = _identity_key(identity)
    if location == "active":
        index_path = _index_root(cfg) / f"{key}.json"
        try:
            index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
        except FileNotFoundError:
            return _active_root(cfg) / "missing.json"
        if index.get("identity") != identity or index.get("location") not in {"active", "retired"}:
            raise ValueError("maintenance descriptor index is invalid.")
        base = _active_root(cfg) if index["location"] == "active" else _retired_root(cfg)
        filename = index.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("maintenance descriptor index filename is invalid.")
        return base / filename
    return _retired_root(cfg) / f"{key}.json"


def _read_record(path: Path, identity: dict[str, str]) -> dict[str, Any]:
    record = read_json_limited(path, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_descriptor")
    return _validate_record(record, identity)


def _validate_record(record: object, identity: dict[str, str]) -> dict[str, Any]:
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        raise ValueError("maintenance descriptor has missing or unknown fields.")
    if record.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION or record.get("identity") != identity:
        raise ValueError("maintenance descriptor identity or schema is invalid.")
    if type(record.get("queue_position")) is not int or record["queue_position"] <= 0:
        raise ValueError("maintenance descriptor queue position is invalid.")
    if type(record.get("progress_revision")) is not int or record["progress_revision"] <= 0:
        raise ValueError("maintenance descriptor progress revision is invalid.")
    if record.get("state") not in _ACTIVE_STATES | _TERMINAL_STATES:
        raise ValueError("maintenance descriptor state is invalid.")
    if not isinstance(record.get("phase"), str) or not isinstance(record.get("cursor"), dict):
        raise ValueError("maintenance descriptor progress is invalid.")
    if type(record.get("retry_count")) is not int or record["retry_count"] < 0:
        raise ValueError("maintenance descriptor retry count is invalid.")
    if record.get("failure") is not None and type(record["failure"]) is not dict:
        raise ValueError("maintenance descriptor failure evidence is invalid.")
    if record.get("retirement_proof") is not None and type(record["retirement_proof"]) is not dict:
        raise ValueError("maintenance descriptor retirement proof is invalid.")
    return record


def _read_slot(cfg: object, queue_position: int) -> dict[str, Any] | None:
    try:
        value = read_json_limited(
            _slot_path(cfg, queue_position),
            max_bytes=_QUEUE_SLOT_MAX_BYTES,
            record_type="maintenance_queue_slot",
        )
    except FileNotFoundError:
        return None
    slot = value.get("maintenance_queue_slot") if type(value) is dict else None
    if type(slot) is not dict or set(slot) != {
        "schema_version",
        "identity",
        "queue_position",
        "filename",
        "initial_record",
    }:
        raise ValueError("maintenance queue slot has missing or unknown fields.")
    if slot.get("schema_version") != _QUEUE_SLOT_SCHEMA_VERSION:
        raise ValueError("maintenance queue slot schema is unsupported.")
    identity = slot.get("identity")
    if type(identity) is not dict or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("maintenance queue slot identity is invalid.")
    if type(slot.get("queue_position")) is not int or slot["queue_position"] != queue_position:
        raise ValueError("maintenance queue slot position is invalid.")
    filename = slot.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("maintenance queue slot filename is invalid.")
    initial_record = _validate_record(slot.get("initial_record"), identity)
    if initial_record.get("queue_position") != queue_position:
        raise ValueError("maintenance queue slot record position is invalid.")
    return slot


def _write_slot(cfg: object, record: dict[str, Any], filename: str) -> None:
    queue_position = record["queue_position"]
    path = _slot_path(cfg, queue_position)
    if path.exists():
        slot = _read_slot(cfg, queue_position)
        if slot is None or slot["identity"] != record["identity"] or slot["filename"] != filename:
            raise ValueError("maintenance queue position is already owned by another descriptor.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    slot = {
        "maintenance_queue_slot": {
            "schema_version": _QUEUE_SLOT_SCHEMA_VERSION,
            "identity": record["identity"],
            "queue_position": queue_position,
            "filename": filename,
            "initial_record": record,
        }
    }
    require_json_size(slot, max_bytes=_QUEUE_SLOT_MAX_BYTES, record_type="maintenance_queue_slot")
    atomic_replace(path, slot)


def _write_record(path: Path, record: dict[str, Any]) -> None:
    require_json_size(record, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_descriptor")
    atomic_replace(path, record)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _queue_state(cfg: object) -> dict[str, int]:
    try:
        value = read_json_limited(_state_path(cfg), max_bytes=4096, record_type="maintenance_queue")
    except FileNotFoundError:
        return {"next_position": 1, "revision": 0, "slot_version": _QUEUE_SLOT_SCHEMA_VERSION}
    state = value.get("maintenance_queue") if type(value) is dict else None
    if (
        type(state) is not dict
        or type(state.get("next_position")) is not int
        or state["next_position"] <= 0
        or type(state.get("revision")) is not int
        or state["revision"] < 0
        or state.get("slot_version", 0) not in {0, _QUEUE_SLOT_SCHEMA_VERSION}
    ):
        raise ValueError("maintenance queue state is invalid.")
    return {
        "next_position": state["next_position"],
        "revision": state["revision"],
        "slot_version": state.get("slot_version", 0),
    }


def _queue_connection(cfg: object) -> sqlite3.Connection:
    """Open the durable active-slot index while the outbox lock is held.

    JSON descriptors and identity indexes remain authoritative. SQLite is a
    rebuildable, ordered index over only active obligations, so selection does
    not revisit queue positions that were retired long ago.
    """
    path = _queue_index_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS queue_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                next_position INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                cursor_position INTEGER NOT NULL,
                migration_phase INTEGER NOT NULL,
                migration_offset INTEGER NOT NULL,
                active_count INTEGER NOT NULL DEFAULT 0,
                cycle_remaining INTEGER NOT NULL DEFAULT 0,
                cycle_min_due TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS queue_slots (
                queue_position INTEGER PRIMARY KEY,
                identity_key TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL,
                filename TEXT NOT NULL,
                initial_record_json TEXT NOT NULL
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(queue_meta)")}
        added_active_count = "active_count" not in columns
        added_cycle_remaining = "cycle_remaining" not in columns
        if added_active_count or added_cycle_remaining:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if added_active_count:
                    connection.execute("ALTER TABLE queue_meta ADD COLUMN active_count INTEGER NOT NULL DEFAULT 0")
                if added_cycle_remaining:
                    connection.execute("ALTER TABLE queue_meta ADD COLUMN cycle_remaining INTEGER NOT NULL DEFAULT 0")
                if added_active_count:
                    connection.execute("DELETE FROM queue_slots")
                    connection.execute(
                        "UPDATE queue_meta SET active_count = 0, migration_phase = 0, migration_offset = 0"
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        row = connection.execute("SELECT singleton FROM queue_meta WHERE singleton = 1").fetchone()
        if row is None:
            legacy = _queue_state(cfg)
            if _slot_root(cfg).is_dir() or _active_root(cfg).is_dir() or legacy["next_position"] > 1:
                migration_phase = 0
            else:
                migration_phase = 2
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO queue_meta(
                    singleton, next_position, revision, cursor_position,
                    migration_phase, migration_offset, active_count, cycle_remaining, cycle_min_due
                ) VALUES (1, ?, ?, 0, ?, 0, 0, 0, NULL)""",
                (legacy["next_position"], legacy["revision"], migration_phase),
            )
            connection.execute("COMMIT")
        return connection
    except sqlite3.DatabaseError as exc:
        try:
            connection.close()  # type: ignore[possibly-undefined]
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise RuntimeError("maintenance active-slot index is corrupt; explicit recovery is required.") from exc


def _queue_register(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    filename: str,
    *,
    advance_allocator: bool = False,
) -> None:
    identity = record["identity"]
    key = _identity_key(identity)
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    initial_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = connection.execute(
            "SELECT identity_key, filename FROM queue_slots WHERE queue_position = ?",
            (record["queue_position"],),
        ).fetchone()
        if existing is not None and existing != (key, filename):
            raise ValueError("maintenance queue position is already owned by another descriptor.")
        inserted = connection.execute(
            """INSERT OR IGNORE INTO queue_slots(
                queue_position, identity_key, identity_json, filename, initial_record_json
            ) VALUES (?, ?, ?, ?, ?)""",
            (record["queue_position"], key, identity_json, filename, initial_json),
        ).rowcount
        if inserted:
            connection.execute("UPDATE queue_meta SET active_count = active_count + 1 WHERE singleton = 1")
        if advance_allocator:
            connection.execute(
                """UPDATE queue_meta SET next_position = ?, revision = revision + 1
                   WHERE singleton = 1""",
                (record["queue_position"] + 1,),
            )
        else:
            connection.execute(
                """UPDATE queue_meta SET next_position = MAX(next_position, ?)
                   WHERE singleton = 1""",
                (record["queue_position"] + 1,),
            )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _queue_allocate(
    connection: sqlite3.Connection,
    identity: dict[str, str],
    now: str,
    phase: str,
    cursor: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, bool]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute("SELECT next_position, migration_phase FROM queue_meta WHERE singleton = 1").fetchone()
        if row is None:
            raise RuntimeError("maintenance queue metadata is missing.")
        position = int(row[0])
        restore_completed_migration = int(row[1]) == 2
        record = _new_record(identity, position, now, phase, cursor)
        filename = f"{position:020d}-{_identity_key(identity)}.json"
        connection.execute(
            """UPDATE queue_meta SET next_position = ?, revision = revision + 1,
                   migration_phase = CASE WHEN migration_phase = 2 THEN 0 ELSE migration_phase END,
                   migration_offset = CASE WHEN migration_phase = 2 THEN 0 ELSE migration_offset END
               WHERE singleton = 1""",
            (position + 1,),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return record, filename, restore_completed_migration


def _finish_queue_allocation(connection: sqlite3.Connection, restore_migration: bool) -> None:
    if not restore_migration:
        return
    connection.execute("UPDATE queue_meta SET migration_phase = 2, migration_offset = 0 WHERE singleton = 1")


def _queue_remove(connection: sqlite3.Connection, identity: dict[str, str], position: int) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        deleted = connection.execute(
            "DELETE FROM queue_slots WHERE queue_position = ? AND identity_key = ?",
            (position, _identity_key(identity)),
        ).rowcount
        if deleted:
            connection.execute("UPDATE queue_meta SET active_count = active_count - 1 WHERE singleton = 1")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _adopt_retirement(
    cfg: object,
    connection: sqlite3.Connection,
    identity: dict[str, str],
    index_path: Path,
    active_filename: str,
) -> dict[str, Any] | None:
    """Finish a terminal descriptor commit interrupted before its index flip."""
    retired_path = _retired_root(cfg) / f"{_identity_key(identity)}.json"
    try:
        record = _read_record(retired_path, identity)
    except FileNotFoundError:
        return None
    if record["state"] not in _TERMINAL_STATES:
        raise ValueError("maintenance retirement proof is not terminal.")
    atomic_replace(
        index_path,
        {"identity": identity, "location": "retired", "filename": retired_path.name},
    )
    _unlink_durable(_active_root(cfg) / active_filename)
    _unlink_durable(_slot_path(cfg, record["queue_position"]))
    _queue_remove(connection, identity, record["queue_position"])
    return record


def _intervene_lost_progress(
    cfg: object,
    connection: sqlite3.Connection,
    identity: dict[str, str],
    index_path: Path,
    filename: str,
    initial_record: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when a committed active descriptor loses its latest cursor."""
    record = dict(initial_record)
    now = utc_now()
    record.update(
        {
            "state": "intervention",
            "progress_revision": record["progress_revision"] + 1,
            "updated_at": now,
            "meaningful_progress_at": now,
            "failure": {"code": "maintenance_progress_lost", "type": "Intervention"},
            "retirement_proof": {"code": "maintenance_progress_lost"},
        }
    )
    retired_path = _retired_root(cfg) / f"{_identity_key(identity)}.json"
    _write_record(retired_path, record)
    atomic_replace(
        index_path,
        {"identity": identity, "location": "retired", "filename": retired_path.name},
    )
    _unlink_durable(_slot_path(cfg, record["queue_position"]))
    _queue_remove(connection, identity, record["queue_position"])
    return record


def prepare_work(
    cfg: object,
    *,
    kind: str,
    target_id: str,
    work_generation: str,
    phase: str = "pending",
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Durably write discoverable prepare evidence before the authoritative mutation."""
    identity = _identity(cfg, kind, target_id, work_generation)
    with exclusive(_lock_path(cfg)):
        return _prepare_locked(cfg, identity, phase=phase, cursor=cursor)


def _prepare_locked(
    cfg: object,
    identity: dict[str, str],
    *,
    phase: str,
    cursor: dict[str, Any] | None,
) -> dict[str, Any]:
    connection = _queue_connection(cfg)
    try:
        return _prepare_with_queue(cfg, identity, phase=phase, cursor=cursor, connection=connection)
    finally:
        connection.close()


def _prepare_with_queue(
    cfg: object,
    identity: dict[str, str],
    *,
    phase: str,
    cursor: dict[str, Any] | None,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    now = utc_now()
    key = _identity_key(identity)
    index_path = _index_root(cfg) / f"{key}.json"
    try:
        index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
    except FileNotFoundError:
        index = None
    if index is not None:
        if index.get("identity") != identity or index.get("location") not in {"active", "retired"}:
            raise ValueError("maintenance descriptor index is invalid.")
        base = _active_root(cfg) if index["location"] == "active" else _retired_root(cfg)
        filename = index.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("maintenance descriptor index filename is invalid.")
        path = base / filename
        if index["location"] == "retired":
            return _read_record(path, identity)
        adopted = _adopt_retirement(cfg, connection, identity, index_path, filename)
        if adopted is not None:
            return adopted
        if path.exists():
            record = _read_record(path, identity)
            _write_slot(cfg, record, filename)
            _queue_register(connection, record, filename)
            return record
        position_text = filename.partition("-")[0]
        if not position_text.isdigit():
            raise ValueError("maintenance descriptor queue filename is invalid.")
        position = int(position_text)
        slot = _read_slot(cfg, position)
        if slot is not None and (slot["identity"] != identity or slot["filename"] != filename):
            raise ValueError("maintenance queue slot does not match its descriptor index.")
        if slot is not None:
            return _intervene_lost_progress(
                cfg,
                connection,
                identity,
                index_path,
                filename,
                slot["initial_record"],
            )
        else:
            row = connection.execute(
                "SELECT initial_record_json FROM queue_slots WHERE queue_position = ? AND identity_key = ?",
                (position, key),
            ).fetchone()
            if row is None:
                raise RuntimeError("active maintenance descriptor and recovery evidence are missing.")
            initial = _validate_record(json.loads(row[0]), identity)
            return _intervene_lost_progress(cfg, connection, identity, index_path, filename, initial)

    retired = _retired_root(cfg) / f"{key}.json"
    if retired.exists():
        return _read_record(retired, identity)
    queued = connection.execute(
        """SELECT queue_position, filename, initial_record_json FROM queue_slots
           WHERE identity_key = ?""",
        (key,),
    ).fetchone()
    if queued is not None:
        position, filename, initial_json = queued
        record = _validate_record(json.loads(initial_json), identity)
        if record["queue_position"] != position:
            raise ValueError("maintenance active-slot position is invalid.")
        _write_slot(cfg, record, filename)
        active_path = _active_root(cfg) / filename
        if not active_path.exists():
            _write_record(active_path, record)
        _index_root(cfg).mkdir(parents=True, exist_ok=True)
        atomic_replace(index_path, {"identity": identity, "location": "active", "filename": filename})
        return _read_record(active_path, identity)
    record, filename, restore_migration = _queue_allocate(connection, identity, now, phase, cursor)
    path = _active_root(cfg) / filename
    _index_root(cfg).mkdir(parents=True, exist_ok=True)
    queue_meta = connection.execute("SELECT next_position, revision FROM queue_meta WHERE singleton = 1").fetchone()
    atomic_replace(
        _state_path(cfg),
        {
            "maintenance_queue": {
                "next_position": queue_meta[0],
                "revision": queue_meta[1],
                "slot_version": _QUEUE_SLOT_SCHEMA_VERSION,
            }
        },
    )
    # Keep a reconstructable copy before the active-slot DB commit. The DB
    # migration marker remains incomplete across this window so a crash can
    # rediscover the JSON slot without scanning operation history.
    _write_slot(cfg, record, filename)
    _queue_register(connection, record, filename)
    _finish_queue_allocation(connection, restore_migration)
    _write_record(path, record)
    atomic_replace(index_path, {"identity": identity, "location": "active", "filename": filename})
    return record


def _new_record(
    identity: dict[str, str],
    position: int,
    now: str,
    phase: str,
    cursor: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "identity": identity,
        "queue_position": position,
        "progress_revision": 1,
        "phase": phase,
        "cursor": dict(cursor or {}),
        "state": "prepared",
        "due_at": now,
        "retry_count": 0,
        "created_at": now,
        "updated_at": now,
        "meaningful_progress_at": now,
        "failure": None,
        "retirement_proof": None,
    }


def prepare_target_work(
    cfg: object,
    *,
    kind: str,
    target_id: str,
    phase: str = "projection",
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach to active target work or allocate one durable new generation."""
    target_identity = _identity(cfg, kind, target_id, "target-generation-placeholder")
    target_path = _target_index_path(cfg, kind, target_id)
    with exclusive(_lock_path(cfg)):
        try:
            pointer = read_json_limited(target_path, max_bytes=4096, record_type="maintenance_target")
        except FileNotFoundError:
            pointer = None
        if pointer is not None:
            generation = pointer.get("work_generation")
            if not isinstance(generation, str):
                raise ValueError("maintenance target generation is invalid.")
            identity = {**target_identity, "work_generation": generation}
            try:
                index = read_json_limited(
                    _index_root(cfg) / f"{_identity_key(identity)}.json",
                    max_bytes=4096,
                    record_type="maintenance_index",
                )
                if (
                    index.get("identity") != identity
                    or index.get("location") not in {"active", "retired"}
                    or not isinstance(index.get("filename"), str)
                    or Path(index["filename"]).name != index["filename"]
                ):
                    raise ValueError("maintenance descriptor index is invalid.")
                if index["location"] == "active":
                    record_path = _active_root(cfg) / index["filename"]
                    if record_path.exists():
                        record = _read_record(record_path, identity)
                        if record["state"] not in _TERMINAL_STATES:
                            return record
                    else:
                        return _prepare_locked(cfg, identity, phase=phase, cursor=cursor)
                else:
                    _read_record(_retired_root(cfg) / index["filename"], identity)
            except FileNotFoundError:
                if pointer is not None:
                    return _prepare_locked(cfg, identity, phase=phase, cursor=cursor)
        generation = new_id()
        identity = {**target_identity, "work_generation": generation}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace(target_path, {"work_generation": generation})
        record = _prepare_locked(cfg, identity, phase=phase, cursor=cursor)
        return record


def read_work(
    cfg: object,
    *,
    kind: str,
    target_id: str,
    work_generation: str,
) -> dict[str, Any] | None:
    """Read one descriptor by its complete identity."""
    identity = _identity(cfg, kind, target_id, work_generation)
    try:
        with exclusive(_lock_path(cfg)):
            index_path = _index_root(cfg) / f"{_identity_key(identity)}.json"
            index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
            if index.get("identity") != identity or index.get("location") not in {"active", "retired"}:
                raise ValueError("maintenance descriptor index is invalid.")
            base = _active_root(cfg) if index["location"] == "active" else _retired_root(cfg)
            filename = index.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("maintenance descriptor index filename is invalid.")
            if index["location"] == "active":
                connection = _queue_connection(cfg)
                try:
                    adopted = _adopt_retirement(cfg, connection, identity, index_path, filename)
                finally:
                    connection.close()
                if adopted is not None:
                    return adopted
            return _read_record(base / filename, identity)
    except FileNotFoundError:
        return None


def _change_work(
    cfg: object,
    record: dict[str, Any],
    *,
    state: str | None = None,
    phase: str | None = None,
    cursor: dict[str, Any] | None = None,
    due_at: str | None = None,
    retry_count: int | None = None,
    failure: dict[str, Any] | None = None,
    retirement_proof: dict[str, Any] | None = None,
    meaningful_progress: bool = False,
) -> dict[str, Any]:
    identity = record["identity"]
    if state is not None:
        if state not in _ACTIVE_STATES | _TERMINAL_STATES:
            raise ValueError("maintenance descriptor state is invalid.")
        record["state"] = state
    if phase is not None:
        record["phase"] = phase
    if cursor is not None:
        record["cursor"] = dict(cursor)
    if due_at is not None:
        record["due_at"] = due_at
    if retry_count is not None:
        record["retry_count"] = retry_count
    if failure is not None:
        record["failure"] = dict(failure)
    if retirement_proof is not None:
        record["retirement_proof"] = dict(retirement_proof)
    now = utc_now()
    record["progress_revision"] += 1
    record["updated_at"] = now
    if meaningful_progress:
        record["meaningful_progress_at"] = now
    key = _identity_key(identity)
    index_path = _index_root(cfg) / f"{key}.json"
    index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
    current_base = _active_root(cfg) if index.get("location") == "active" else _retired_root(cfg)
    old_path = current_base / index["filename"]
    if record["state"] in _TERMINAL_STATES:
        _retired_root(cfg).mkdir(parents=True, exist_ok=True)
        new_path = _retired_root(cfg) / f"{key}.json"
        _write_record(new_path, record)
        index = {"identity": identity, "location": "retired", "filename": new_path.name}
        atomic_replace(index_path, index)
        if current_base == _active_root(cfg):
            _unlink_durable(old_path)
        _unlink_durable(_slot_path(cfg, record["queue_position"]))
        connection = _queue_connection(cfg)
        try:
            _queue_remove(connection, identity, record["queue_position"])
        finally:
            connection.close()
        return record
    _write_record(old_path, record)
    if index.get("location") != "active":
        raise ValueError("terminal maintenance descriptor cannot be reopened.")
    return record


def _read_current_locked(cfg: object, identity: dict[str, str]) -> dict[str, Any]:
    index = read_json_limited(
        _index_root(cfg) / f"{_identity_key(identity)}.json",
        max_bytes=4096,
        record_type="maintenance_index",
    )
    if index.get("identity") != identity or index.get("location") not in {"active", "retired"}:
        raise ValueError("maintenance descriptor index identity is invalid.")
    base = _active_root(cfg) if index["location"] == "active" else _retired_root(cfg)
    return _read_record(base / index["filename"], identity)


def _publish_revision_change(cfg: object, identity: dict[str, str], **changes: Any) -> dict[str, Any]:
    """Commit activation intent before the descriptor revision it advertises."""
    while True:
        with exclusive(_lock_path(cfg)):
            baseline = _read_current_locked(cfg, identity)
            if baseline["state"] in _TERMINAL_STATES:
                return baseline
            baseline_revision = baseline["progress_revision"]
        reason = _activation_reason(identity, baseline_revision + 1)
        retry = False
        result = baseline
        with project_activation_transaction(cfg, reason):
            with exclusive(_lock_path(cfg)):
                current = _read_current_locked(cfg, identity)
                if current["state"] in _TERMINAL_STATES:
                    result = current
                elif current["progress_revision"] != baseline_revision:
                    retry = True
                else:
                    result = _change_work(cfg, current, **changes)
        if not retry:
            return result


def activate_work(cfg: object, record: dict[str, Any]) -> dict[str, Any]:
    """Mark prepared work pending and publish its exact revision through activation."""
    return _publish_revision_change(
        cfg,
        record["identity"],
        state="pending",
        due_at=utc_now(),
        retry_count=0,
        failure=None,
        meaningful_progress=True,
    )


def update_work(
    cfg: object,
    *,
    kind: str,
    target_id: str,
    work_generation: str,
    state: str | None = None,
    phase: str | None = None,
    cursor: dict[str, Any] | None = None,
    due_at: str | None = None,
    retry_count: int | None = None,
    failure: dict[str, Any] | None = None,
    retirement_proof: dict[str, Any] | None = None,
    meaningful_progress: bool = False,
    publish_activation: bool = True,
) -> dict[str, Any]:
    """Commit one descriptor revision and optionally wake Project consumers."""
    identity = _identity(cfg, kind, target_id, work_generation)
    changes = {
        "state": state,
        "phase": phase,
        "cursor": cursor,
        "due_at": due_at,
        "retry_count": retry_count,
        "failure": failure,
        "retirement_proof": retirement_proof,
        "meaningful_progress": meaningful_progress,
    }
    if publish_activation:
        return _publish_revision_change(cfg, identity, **changes)
    with exclusive(_lock_path(cfg)):
        record = _read_current_locked(cfg, identity)
        if record["state"] in _TERMINAL_STATES:
            return record
        return _change_work(cfg, record, **changes)


def _migrate_slot_entry(cfg: object, connection: sqlite3.Connection, name: str) -> None:
    if not name.endswith(".json") or not name[:-5].isdigit():
        return
    position = int(name[:-5])
    slot = _read_slot(cfg, position)
    if slot is None:
        return
    identity = slot["identity"]
    key = _identity_key(identity)
    index_path = _index_root(cfg) / f"{key}.json"
    try:
        index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
    except FileNotFoundError:
        index = None
    if index is not None and (index.get("identity") != identity or index.get("location") not in {"active", "retired"}):
        raise ValueError("maintenance descriptor index is invalid during queue reconstruction.")
    if index is not None and index["location"] == "retired":
        return
    filename = index.get("filename") if index is not None else slot["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("maintenance descriptor filename is invalid during queue reconstruction.")
    active_path = _active_root(cfg) / filename
    if active_path.exists():
        record = _read_record(active_path, identity)
    elif index is not None:
        _intervene_lost_progress(cfg, connection, identity, index_path, filename, slot["initial_record"])
        return
    else:
        record = slot["initial_record"]
        _write_record(active_path, record)
    if record["queue_position"] != position:
        raise ValueError("maintenance descriptor queue position is invalid during reconstruction.")
    _index_root(cfg).mkdir(parents=True, exist_ok=True)
    atomic_replace(index_path, {"identity": identity, "location": "active", "filename": filename})
    _write_slot(cfg, slot["initial_record"], filename)
    _queue_register(connection, slot["initial_record"], filename)


def _migrate_active_entry(cfg: object, connection: sqlite3.Connection, name: str) -> None:
    if not name.endswith(".json"):
        return
    path = _active_root(cfg) / name
    try:
        value = read_json_limited(path, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_descriptor")
    except FileNotFoundError:
        return
    identity = value.get("identity") if type(value) is dict else None
    if type(identity) is not dict:
        raise ValueError("maintenance descriptor identity is invalid during queue reconstruction.")
    record = _read_record(path, identity)
    key = _identity_key(identity)
    index_path = _index_root(cfg) / f"{key}.json"
    try:
        index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
    except FileNotFoundError:
        index = None
    if index is not None and (index.get("identity") != identity or index.get("location") not in {"active", "retired"}):
        raise ValueError("maintenance descriptor index is invalid during queue reconstruction.")
    if index is not None and index["location"] == "retired":
        _unlink_durable(path)
        return
    if index is None:
        _index_root(cfg).mkdir(parents=True, exist_ok=True)
        atomic_replace(index_path, {"identity": identity, "location": "active", "filename": name})
    _write_slot(cfg, record, name)
    _queue_register(connection, record, name)


def _advance_reconstruction(cfg: object, connection: sqlite3.Connection, max_scan: int) -> bool:
    """Boundedly rebuild the active-slot index from slots, then legacy records."""
    row = connection.execute("SELECT migration_phase, migration_offset FROM queue_meta WHERE singleton = 1").fetchone()
    if row is None:
        raise RuntimeError("maintenance queue reconstruction metadata is missing.")
    phase, offset = int(row[0]), int(row[1])
    scanned = 0
    while phase < 2 and scanned < max_scan:
        root = _slot_root(cfg) if phase == 0 else _active_root(cfg)
        if not root.is_dir():
            name, next_offset = None, 0
        else:
            name, next_offset = read_directory_entry(root, offset)
        if name is None:
            phase += 1
            offset = 0
            continue
        scanned += 1
        if phase == 0:
            _migrate_slot_entry(cfg, connection, name)
        else:
            _migrate_active_entry(cfg, connection, name)
        offset = next_offset
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "UPDATE queue_meta SET migration_phase = ?, migration_offset = ? WHERE singleton = 1",
            (phase, offset),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return phase >= 2


def _candidate_record(
    cfg: object,
    connection: sqlite3.Connection,
    row: tuple[int, str, str, str],
) -> tuple[dict[str, Any] | None, int]:
    position, identity_json, filename, initial_json = row
    identity = json.loads(identity_json)
    if type(identity) is not dict or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("maintenance active-slot identity is invalid.")
    initial_record = _validate_record(json.loads(initial_json), identity)
    if initial_record["queue_position"] != position:
        raise ValueError("maintenance active-slot initial position is invalid.")
    key = _identity_key(identity)
    index_path = _index_root(cfg) / f"{key}.json"
    try:
        index = read_json_limited(index_path, max_bytes=4096, record_type="maintenance_index")
    except FileNotFoundError:
        index = None
    if index is not None:
        if index.get("identity") != identity or index.get("location") not in {"active", "retired"}:
            raise ValueError("maintenance descriptor index is invalid.")
        if index["location"] == "retired":
            _unlink_durable(_slot_path(cfg, position))
            _queue_remove(connection, identity, position)
            return None, position
        if index.get("filename") != filename:
            raise ValueError("maintenance active-slot filename disagrees with its identity index.")
        adopted = _adopt_retirement(cfg, connection, identity, index_path, filename)
        if adopted is not None:
            return None, position
    else:
        _index_root(cfg).mkdir(parents=True, exist_ok=True)
        atomic_replace(index_path, {"identity": identity, "location": "active", "filename": filename})
    slot = _read_slot(cfg, position)
    if slot is None:
        _write_slot(cfg, initial_record, filename)
    elif slot["identity"] != identity or slot["filename"] != filename:
        raise ValueError("maintenance queue slot disagrees with its active-slot index.")
    path = _active_root(cfg) / filename
    if not path.exists():
        if index is not None:
            _intervene_lost_progress(cfg, connection, identity, index_path, filename, initial_record)
            return None, position
        _write_record(path, initial_record)
        atomic_replace(index_path, {"identity": identity, "location": "active", "filename": filename})
        return initial_record, position
    record = _read_record(path, identity)
    if record["queue_position"] != position:
        raise ValueError("maintenance descriptor position disagrees with its active-slot index.")
    return record, position


def select_due_work(cfg: object, *, max_scan: int = 64) -> dict[str, Any]:
    """Select one due descriptor in persisted queue order with bounded discovery."""
    if type(max_scan) is not int or max_scan <= 0:
        raise ValueError("max_scan must be a positive integer.")
    with exclusive(_lock_path(cfg)):
        connection = _queue_connection(cfg)
        try:
            if not _advance_reconstruction(cfg, connection, max_scan):
                return {"descriptor": None, "next_due_at": None, "more": True}
            meta = connection.execute(
                """SELECT cursor_position, cycle_min_due, active_count, cycle_remaining
                   FROM queue_meta WHERE singleton = 1"""
            ).fetchone()
            if meta is None:
                raise RuntimeError("maintenance queue service metadata is missing.")
            cursor_position = int(meta[0])
            cycle_min_due = meta[1]
            active_count = int(meta[2])
            cycle_remaining = int(meta[3])
            if active_count == 0:
                connection.execute(
                    "UPDATE queue_meta SET cursor_position = 0, cycle_min_due = NULL WHERE singleton = 1"
                )
                return {"descriptor": None, "next_due_at": None, "more": False}
            if cycle_remaining <= 0:
                cycle_remaining = active_count
            else:
                cycle_remaining = min(cycle_remaining, active_count)
            scan_limit = min(max_scan, cycle_remaining)
            columns = "queue_position, identity_json, filename, initial_record_json"
            rows = connection.execute(
                f"SELECT {columns} FROM queue_slots WHERE queue_position > ? ORDER BY queue_position LIMIT ?",
                (cursor_position, scan_limit),
            ).fetchall()
            if len(rows) < scan_limit:
                wrapped = connection.execute(
                    f"SELECT {columns} FROM queue_slots WHERE queue_position <= ? ORDER BY queue_position LIMIT ?",
                    (cursor_position, scan_limit - len(rows)),
                ).fetchall()
                rows.extend(wrapped)
            if not rows:
                connection.execute(
                    "UPDATE queue_meta SET cursor_position = 0, cycle_min_due = NULL WHERE singleton = 1"
                )
                return {"descriptor": None, "next_due_at": None, "more": False}
            scanned_count = 0

            now = datetime.now(timezone.utc)
            scanned_min_due = cycle_min_due
            last_position = cursor_position
            for row in rows:
                record, position = _candidate_record(cfg, connection, row)
                scanned_count += 1
                last_position = position
                if record is None:
                    continue
                if record["state"] in _TERMINAL_STATES:
                    _change_work(
                        cfg,
                        record,
                        state=record["state"],
                        retirement_proof=record.get("retirement_proof") or {"source": "terminal_descriptor_recovery"},
                    )
                    continue
                due_at = record["due_at"]
                try:
                    due = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if due.tzinfo is None or due.utcoffset() is None:
                    continue
                if due > now:
                    if scanned_min_due is None or due_at < scanned_min_due:
                        scanned_min_due = due_at
                    continue
                connection.execute(
                    """UPDATE queue_meta SET cursor_position = ?, cycle_remaining = ?, cycle_min_due = NULL
                       WHERE singleton = 1""",
                    (position, max(0, cycle_remaining - scanned_count)),
                )
                return {"descriptor": record, "next_due_at": scanned_min_due, "more": False}

            remaining_after = max(0, cycle_remaining - scanned_count)
            completed_cycle = remaining_after == 0
            connection.execute(
                """UPDATE queue_meta SET cursor_position = ?, cycle_remaining = ?, cycle_min_due = ?
                   WHERE singleton = 1""",
                (last_position, remaining_after, None if completed_cycle else scanned_min_due),
            )
            return {"descriptor": None, "next_due_at": scanned_min_due, "more": not completed_cycle}
        finally:
            connection.close()


def retire_work(
    cfg: object,
    *,
    kind: str,
    target_id: str,
    work_generation: str,
    state: str = "completed",
    proof: dict[str, Any],
) -> dict[str, Any]:
    """Persist authoritative completion or intervention evidence and leave hot service."""
    if state not in {"completed", "intervention", "superseded"}:
        raise ValueError("maintenance retirement state is invalid.")
    return update_work(
        cfg,
        kind=kind,
        target_id=target_id,
        work_generation=work_generation,
        state=state,
        retirement_proof=proof,
        meaningful_progress=True,
        publish_activation=False,
    )


def maintenance_status(cfg: object, *, max_scan: int = 64) -> dict[str, Any]:
    """Return bounded active descriptor counts and the next due timestamp."""
    if type(max_scan) is not int or max_scan <= 0:
        raise ValueError("max_scan must be a positive integer.")
    root = _active_root(cfg)
    if not root.is_dir():
        return {"active_count": 0, "scanned": 0, "has_more": False, "next_due_at": None}
    offset = 0
    count = 0
    next_due: str | None = None
    for _ in range(max_scan):
        name, offset = read_directory_entry(root, offset)
        if name is None:
            return {"active_count": count, "scanned": count, "has_more": False, "next_due_at": next_due}
        if not name.endswith(".json"):
            continue
        count += 1
        record = read_json_limited(root / name, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_descriptor")
        if type(record) is not dict:
            raise ValueError(f"maintenance descriptor is invalid: {root / name}")
        due_at = record.get("due_at") if type(record) is dict else None
        if isinstance(due_at, str) and (next_due is None or due_at < next_due):
            next_due = due_at
    return {"active_count": count, "scanned": count, "has_more": True, "next_due_at": next_due}


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "activate_work",
    "maintenance_status",
    "prepare_work",
    "read_work",
    "retire_work",
    "select_due_work",
    "update_work",
]
