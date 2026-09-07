"""Resumable rebuild orchestration for Group ready-member projection."""

from __future__ import annotations

import ctypes
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from ..locks import exclusive
from ..paths import shared_paths
from ..records import TaskRecord, utc_now, validate_identifier
from ..store import atomic_replace, read_json, read_json_limited, require_json_size
from .group_members import (
    GROUP_MEMBER_PAGE_SIZE,
    GROUP_READY_MEMBERS_CAPABILITY,
    GROUP_READY_MEMBERS_VERSION,
    assert_group_ready_member_matches_task,
    group_ready_member_digest,
    group_ready_member_groups_root,
    group_ready_member_identity,
    group_ready_members_root,
    group_ready_members_state_lock_path,
    group_ready_members_state_path,
    has_group_ready_member,
    load_group_ready_member_page,
    mark_group_ready_members_degraded,
    publish_group_ready_member,
    read_group_ready_member_audit_state,
    read_group_ready_member_directory,
    read_group_ready_member_page_entries,
    read_group_ready_member_writable_index,
    read_group_ready_members,
    read_group_ready_members_state,
    update_group_ready_member_digest,
)
from .index import (
    begin_primary_ready_index_rebuild,
    complete_primary_ready_index_rebuild,
    rebuild_primary_ready_candidate,
    task_should_have_ready_marker,
)
from .records import ReadyMarkerRef
from .routes import reference_for_generation


class _Dirent(ctypes.Structure):
    """Linux dirent layout used to persist a bounded migration cursor."""

    _fields_ = [
        ("d_ino", ctypes.c_ulong),
        ("d_off", ctypes.c_long),
        ("d_reclen", ctypes.c_ushort),
        ("d_type", ctypes.c_ubyte),
        ("d_name", ctypes.c_char * 256),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.fdopendir.argtypes = [ctypes.c_int]
_LIBC.fdopendir.restype = ctypes.c_void_p
_LIBC.closedir.argtypes = [ctypes.c_void_p]
_LIBC.closedir.restype = ctypes.c_int
_LIBC.readdir.argtypes = [ctypes.c_void_p]
_LIBC.readdir.restype = ctypes.POINTER(_Dirent)
_LIBC.seekdir.argtypes = [ctypes.c_void_p, ctypes.c_long]
_LIBC.seekdir.restype = None
_LIBC.telldir.argtypes = [ctypes.c_void_p]
_LIBC.telldir.restype = ctypes.c_long

_MAX_BUILD_PAGE_BYTES = 32 * 1024
_MAX_BUILD_IDENTIFIER_BYTES = 256
_MAX_GLOBAL_STATE_BYTES = 64 * 1024

# Upper bounds for the current implementation's physical operations per
# logical maintenance item.  The additive term covers phase setup/finalization
# (state fences, directory handles, and cursor commits).  These values are
# intentionally conservative and are exercised by instrumentation tests.
PHASE_IO_BOUNDS: dict[str, dict[str, int]] = {
    "capture-tasks": {"read": 4, "write": 2, "metadata": 6, "constant": 8},
    "audit-tasks": {"read": 8, "write": 0, "metadata": 2, "constant": 4},
    "capture-groups": {"read": 4, "write": 2, "metadata": 6, "constant": 8},
    "audit-members": {"read": 16, "write": 0, "metadata": 2, "constant": 8},
    "archive-cleanup": {"read": 3, "write": 0, "metadata": 4, "constant": 8},
}


class BudgetExhausted(RuntimeError):
    """Raised when a maintenance slice has no remaining work budget."""


class WorkBudget:
    """Shared bounded-work counter for one maintenance invocation."""

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("work budget must be a positive integer")
        self.limit = limit
        self.consumed = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.consumed

    def consume(self, kind: str) -> None:
        if self.remaining <= 0:
            raise BudgetExhausted(f"maintenance work budget exhausted before {kind}")
        self.consumed += 1


def _writable_page_digest(page: int) -> str:
    """Return the bounded-audit set digest contribution for one member page."""
    return hashlib.sha256(str(page).encode()).hexdigest()


def _update_writable_page_digest(current: str, page: int) -> str:
    if not isinstance(current, str) or len(current) != 64:
        raise ValueError("Group ready-member writable-index digest is invalid.")
    return f"{int(current, 16) ^ int(_writable_page_digest(page), 16):064x}"


def _archive_cleanup_lock_path(cfg: object) -> Path:
    """Return the single-owner fence for bounded archive reclamation."""
    return shared_paths(cfg.shared_root)["ready_locks"] / "group-members-archive-cleanup.lock"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _park_groups_under_lock(cfg: object, value: dict[str, Any]) -> dict[str, Any]:
    """Finish one prepared Group-tree cutover without enumerating old files."""
    record = value["group_ready_members"]
    park = record.get("park")
    if not isinstance(park, dict) or park.get("state") != "prepared":
        return record
    build_id = park.get("build_id")
    if not isinstance(build_id, str) or len(build_id) != 32:
        raise ValueError("Group ready-member park state is invalid.")
    root = group_ready_members_root(cfg)
    groups = group_ready_member_groups_root(cfg)
    replaced_root = root / "replaced-groups"
    archive = replaced_root / build_id
    if archive.parent != replaced_root or archive.is_symlink() or groups.is_symlink():
        raise ValueError("Group ready-member park path is invalid.")
    replaced_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root)
    _fsync_directory(replaced_root)
    if not archive.exists() and groups.exists():
        if groups.stat().st_dev != replaced_root.stat().st_dev:
            raise ValueError("Group ready-member park requires one filesystem.")
        os.replace(groups, archive)
        _fsync_directory(root)
        _fsync_directory(replaced_root)
    if archive.exists() and not archive.is_dir():
        raise ValueError("Group ready-member archive is invalid.")
    groups.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root)
    record["state"] = "building"
    record["projection_id"] = park["next_projection_id"]
    record["archive_count"] += 1
    record["park"] = {"state": "completed", "build_id": build_id, "archive": f"replaced-groups/{build_id}"}
    record["build"] = {
        "build_id": build_id,
        "cursor": {"page": 0, "offset": 0},
        "processed": 0,
        "phase": "backfill",
        "watermark": {"is_complete": False},
        "started_at": utc_now(),
    }
    record["revision"] += 1
    record["updated_at"] = utc_now()
    atomic_replace(group_ready_members_state_path(cfg), value)
    return record


def begin_group_ready_members_build(cfg: object, *, is_repair: bool = False) -> dict[str, Any]:
    """Start or recover one durable, resumable member-projection build."""
    root = group_ready_members_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    path = group_ready_members_state_path(cfg)
    with exclusive(group_ready_members_state_lock_path(cfg)):
        if path.exists():
            record = read_group_ready_members_state(cfg)
            if isinstance(record.get("park"), dict) and record["park"].get("state") == "prepared":
                return _park_groups_under_lock(cfg, read_json_limited(path, max_bytes=_MAX_GLOBAL_STATE_BYTES))
            if record["state"] in {"active", "building", "degraded"} and not is_repair:
                return record
            if record["state"] == "building":
                return record
        if is_repair and path.exists():
            value = read_json_limited(path, max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            build_id = uuid.uuid4().hex
            record["state"] = "degraded"
            record["park"] = {
                "state": "prepared",
                "build_id": build_id,
                "next_projection_id": uuid.uuid4().hex,
                "archive": f"replaced-groups/{build_id}",
            }
            record["revision"] += 1
            record["updated_at"] = utc_now()
            atomic_replace(path, value)
            return _park_groups_under_lock(cfg, value)
        group_ready_member_groups_root(cfg).mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "required_capability": GROUP_READY_MEMBERS_CAPABILITY,
            "state": "building",
            "revision": 0,
            "projection_id": uuid.uuid4().hex,
            "audit": None,
            "archive_count": 0,
            "archive_cleanup": None,
            "audit_cleanup": None,
            "build": {
                "build_id": uuid.uuid4().hex,
                "cursor": {"page": 0, "offset": 0},
                "processed": 0,
                "phase": "backfill",
                "watermark": {"is_complete": False},
                "started_at": utc_now(),
            },
            "degraded_reasons": [],
            "updated_at": utc_now(),
        }
        atomic_replace(path, {"group_ready_members": record})
        return record


def _should_index(task: TaskRecord) -> bool:
    return bool(task.group_name and task.ready_generation > 0 and task.state["projection"] == "queued")


def _commit_audit(cfg: object, audit_id: str, audit: dict[str, Any]) -> dict[str, Any]:
    with exclusive(group_ready_members_state_lock_path(cfg)):
        value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
        record = value["group_ready_members"]
        current = record.get("audit")
        if not isinstance(current, dict) or current.get("audit_id") != audit_id:
            return record
        record["audit"] = audit
        if audit.get("state") == "completed" and not isinstance(record.get("audit_cleanup"), dict):
            record["audit_cleanup"] = {
                "audit_id": audit_id,
                "phase": "watermark",
                "processed": 0,
            }
        record["revision"] += 1
        record["updated_at"] = utc_now()
        atomic_replace(group_ready_members_state_path(cfg), value)
        return record


def advance_group_ready_members_audit(
    cfg: object,
    *,
    max_work_items: int = GROUP_MEMBER_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance exactly one resumable verification slice for an active projection."""
    if type(max_work_items) is not int or not 1 <= max_work_items <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_work_items must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    with exclusive(group_ready_members_state_lock_path(cfg)):
        value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
        record = value["group_ready_members"]
        if record.get("state") != "active":
            return record
        audit = record.get("audit")
        should_start_new_audit = (
            not isinstance(audit, dict)
            or audit.get("projection_id") != record.get("projection_id")
            or audit.get("state") == "completed"
        )
        if should_start_new_audit:
            audit = {
                "audit_id": uuid.uuid4().hex,
                "projection_id": record["projection_id"],
                "state": "building",
                "phase": "capture-tasks",
                "processed": 0,
                "task_watermark": {"is_complete": False},
                "updated_at": utc_now(),
            }
            record["audit"] = audit
            record["revision"] += 1
            atomic_replace(group_ready_members_state_path(cfg), value)
    audit_id = audit["audit_id"]
    try:
        phase = audit["phase"]
        if audit.get("state") != "building" or phase not in {
            "capture-tasks",
            "audit-tasks",
            "capture-groups",
            "audit-members",
        }:
            raise ValueError("Group ready-member audit state is invalid.")
        if phase == "capture-tasks":
            watermark = _capture_build_watermark(
                cfg, build_id=audit_id, watermark=audit["task_watermark"], max_entries=max_work_items, collection="audit-tasks"
            )
            audit["task_watermark"] = watermark
            if watermark.get("is_complete"):
                audit.update({"phase": "audit-tasks", "task_cursor": {"page": 0, "offset": 0}})
        elif phase == "audit-tasks":
            cursor = dict(audit["task_cursor"])
            processed = 0
            while processed < max_work_items and cursor["page"] < audit["task_watermark"]["page_count"]:
                items = _load_build_page(cfg, audit_id, cursor["page"], collection="audit-tasks")
                if cursor["offset"] >= len(items):
                    cursor.update({"page": cursor["page"] + 1, "offset": 0})
                    continue
                _audit_task_member(cfg, items[cursor["offset"]])
                processed += 1
                _advance_page_cursor(cursor, len(items))
            audit["task_cursor"] = cursor
            audit["processed"] += processed
            if cursor["page"] >= audit["task_watermark"]["page_count"]:
                audit.update({"phase": "capture-groups", "group_watermark": {"is_complete": False}})
        elif phase == "capture-groups":
            watermark = _capture_build_watermark(
                cfg,
                build_id=audit_id,
                watermark=audit["group_watermark"],
                max_entries=max_work_items,
                source_path=group_ready_member_groups_root(cfg),
                collection="audit-groups",
                should_include_json=False,
            )
            audit["group_watermark"] = watermark
            if watermark.get("is_complete"):
                audit.update(
                    {
                        "phase": "audit-members",
                        "member_cursor": {
                            "group_page": 0,
                            "group_offset": 0,
                            "group_page_count": watermark["page_count"],
                            "group_name": None,
                            "directory_page": None,
                            "directory_offset": 0,
                            "member_page": None,
                            "entry_offset": 0,
                            "pending_entry": None,
                            "pending_error": None,
                            "seen_count": 0,
                            "seen_digest": group_ready_member_digest([]),
                        },
                    }
                )
        elif phase == "audit-members":
            cursor, processed, complete = _advance_member_audit(
                cfg, audit_id, dict(audit["member_cursor"]), max_work_items
            )
            audit["member_cursor"] = cursor
            audit["processed"] += processed
            if complete:
                audit.update({"phase": "completed", "state": "completed", "completed_at": utc_now()})
        else:
            raise ValueError("Group ready-member audit phase is invalid.")
        audit["updated_at"] = utc_now()
        return _commit_audit(cfg, audit_id, audit)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"audit_failed:{type(exc).__name__}:{exc}")
        raise RuntimeError(f"Group ready-member audit failed: {exc}") from exc


def audit_group_ready_members(cfg: object, *, max_work_items: int = GROUP_MEMBER_PAGE_SIZE) -> dict[str, Any]:
    """Backward-compatible entry point for one bounded member-audit slice."""
    return advance_group_ready_members_audit(cfg, max_work_items=max_work_items)


def _audit_task_member(
    cfg: object,
    task_id: str,
) -> tuple[TaskRecord | None, ReadyMarkerRef | None]:
    """Audit one captured Task; a cleaned Task is a legal watermark tombstone."""
    try:
        task = TaskRecord.from_dict(read_json(shared_paths(cfg.shared_root)["tasks"] / f"{task_id}.json"))
    except FileNotFoundError:
        return None, None
    if not task.group_name or task.ready_generation <= 0:
        return task, None
    if not _should_index(task):
        if has_group_ready_member(cfg, task.group_name, task.task_id, task.ready_generation):
            raise ValueError(f"Group ready-member is stale for Task {task.task_id!r}.")
        return task, None

    reference = reference_for_generation(cfg, task.task_id, task.ready_generation)
    if reference is None:
        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
    assert_group_ready_member_matches_task(cfg, task, reference)
    return task, reference


def _audit_member_entry(cfg: object, group_name: str, entry: dict[str, Any]) -> None:
    """Reject an actual member that no longer has matching authoritative truth."""
    task_id = entry.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("Group ready-member Task identity is invalid.")
    task = TaskRecord.from_dict(read_json(shared_paths(cfg.shared_root)["tasks"] / f"{task_id}.json"))
    if task.group_name != group_name or not _should_index(task):
        raise ValueError(f"Group {group_name!r} has a stale ready-member entry.")

    reference = reference_for_generation(cfg, task.task_id, task.ready_generation)
    if reference is None:
        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
    assert_group_ready_member_matches_task(cfg, task, reference)


def _build_root(cfg: object, build_id: str) -> Path:
    validate_identifier(build_id, "group ready-member build id")
    return group_ready_members_root(cfg) / "builds" / build_id


def _build_page_path(
    cfg: object,
    build_id: str,
    page: int,
    *,
    collection: str = "watermark",
) -> Path:
    return _build_root(cfg, build_id) / collection / f"{page:016d}.json"


def _write_build_page(
    cfg: object,
    build_id: str,
    page: int,
    task_ids: list[str],
    *,
    collection: str = "watermark",
) -> None:
    if any(len(task_id.encode("utf-8")) > _MAX_BUILD_IDENTIFIER_BYTES for task_id in task_ids):
        raise ValueError("projection_encoding_unsupported:build_page_identifier")
    value = {
        "group_ready_member_build_page": {
                "schema_version": GROUP_READY_MEMBERS_VERSION,
                "build_id": build_id,
                "page": page,
                "task_ids": list(task_ids),
        }
    }
    require_json_size(value, max_bytes=_MAX_BUILD_PAGE_BYTES, record_type="build_page")
    atomic_replace(_build_page_path(cfg, build_id, page, collection=collection), value)


def _load_build_page(
    cfg: object,
    build_id: str,
    page: int,
    *,
    collection: str = "watermark",
) -> list[str]:
    record = read_json_limited(
        _build_page_path(cfg, build_id, page, collection=collection), max_bytes=_MAX_BUILD_PAGE_BYTES
    )["group_ready_member_build_page"]
    task_ids = record.get("task_ids")
    if (
        record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("build_id") != build_id
        or record.get("page") != page
        or not isinstance(task_ids, list)
        or len(task_ids) > GROUP_MEMBER_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in task_ids)
        or any(len(task_id.encode("utf-8")) > _MAX_BUILD_IDENTIFIER_BYTES for task_id in task_ids)
    ):
        raise ValueError("Group ready-member build watermark page is invalid.")
    return task_ids


def _capture_build_watermark(
    cfg: object,
    *,
    build_id: str,
    watermark: dict[str, Any],
    max_entries: int,
    source_path: Path | None = None,
    collection: str = "watermark",
    should_include_json: bool = True,
) -> dict[str, Any]:
    """Capture at most one I/O-budgeted directory slice into durable Task pages."""
    capture = watermark.get("capture")
    directory_path = source_path or shared_paths(cfg.shared_root)["tasks"]
    directory = directory_path.stat()
    if capture is None:
        capture = {
            "device": directory.st_dev,
            "inode": directory.st_ino,
            "offset": 0,
            "page": 0,
            "pending_task_ids": [],
            "task_count": 0,
        }
    if (
        not isinstance(capture, dict)
        or capture.get("device") != directory.st_dev
        or capture.get("inode") != directory.st_ino
        or type(capture.get("offset")) is not int
        or capture["offset"] < 0
        or type(capture.get("page")) is not int
        or capture["page"] < 0
        or not isinstance(capture.get("pending_task_ids"), list)
        or len(capture["pending_task_ids"]) >= GROUP_MEMBER_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in capture["pending_task_ids"])
        or type(capture.get("task_count")) is not int
        or capture["task_count"] < 0
    ):
        raise ValueError("Group ready-member build capture cursor is invalid.")
    directory_fd = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
    directory_handle = _LIBC.fdopendir(directory_fd)
    if not directory_handle:
        os.close(directory_fd)
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), directory_path)
    try:
        if capture["offset"]:
            _LIBC.seekdir(directory_handle, capture["offset"])
        scanned_entries = 0
        is_complete = False
        while scanned_entries < max_entries:
            ctypes.set_errno(0)
            entry = _LIBC.readdir(directory_handle)
            if not entry:
                error = ctypes.get_errno()
                if error:
                    raise OSError(error, os.strerror(error), directory_path)
                is_complete = True
                break
            name = os.fsdecode(entry.contents.d_name)
            capture["offset"] = _LIBC.telldir(directory_handle)
            if name in {".", ".."}:
                continue
            scanned_entries += 1
            if should_include_json and not name.endswith(".json"):
                continue
            capture["pending_task_ids"].append(name[:-5] if should_include_json else name)
            capture["task_count"] += 1
            if len(capture["pending_task_ids"]) == GROUP_MEMBER_PAGE_SIZE:
                _write_build_page(
                    cfg,
                    build_id,
                    capture["page"],
                    capture["pending_task_ids"],
                    collection=collection,
                )
                capture["page"] += 1
                capture["pending_task_ids"] = []
    finally:
        _LIBC.closedir(directory_handle)
    if is_complete:
        if capture["pending_task_ids"]:
            _write_build_page(
                cfg,
                build_id,
                capture["page"],
                capture["pending_task_ids"],
                collection=collection,
            )
            capture["page"] += 1
            capture["pending_task_ids"] = []
        return {
            "page_count": capture["page"],
            "task_count": capture["task_count"],
            "captured_at": utc_now(),
            "is_complete": True,
        }
    return {"is_complete": False, "capture": capture}


def _commit_build_record(
    cfg: object,
    *,
    build_id: str,
    cursor: dict[str, int] | None = None,
    processed: int = 0,
    build_updates: dict[str, Any] | None = None,
    should_activate: bool = False,
) -> dict[str, Any]:
    """Commit build progress without allowing a concurrent degraded gate to reopen."""
    with exclusive(group_ready_members_state_lock_path(cfg)):
        value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
        record = value["group_ready_members"]
        if record.get("state") != "building" or record.get("build", {}).get("build_id") != build_id:
            return record
        build = record["build"]
        if cursor is not None:
            build["cursor"] = dict(cursor)
            build["processed"] += processed
        if build_updates is not None:
            build.update(build_updates)
        if should_activate:
            build["phase"] = "completed"
            build["completed_at"] = utc_now()
            record["state"] = "active"
        record["revision"] += 1
        record["updated_at"] = utc_now()
        atomic_replace(group_ready_members_state_path(cfg), value)
        return record


def _advance_page_cursor(cursor: dict[str, int], item_count: int) -> None:
    cursor["offset"] += 1
    if cursor["offset"] >= item_count:
        cursor["page"] += 1
        cursor["offset"] = 0


def _advance_writable_index_audit(
    cfg: object,
    group_name: str,
    cursor: dict[str, Any],
    budget: int,
) -> int:
    """Consume bounded FIFO/free-page validation after the member-page scan."""
    if budget <= 0:
        return 0
    work = WorkBudget(budget)
    next_index_page = cursor.get("next_writable_index_page")
    index_count = cursor.get("writable_index_count")
    if (
        type(next_index_page) is not int
        or next_index_page < 0
        or type(index_count) is not int
        or index_count < 0
    ):
        raise ValueError("Group ready-member writable index state is invalid.")
    processed = 0
    while work.remaining:
        index_page = cursor.get("writable_index_page")
        entries = cursor.get("writable_index_entries")
        if index_page is not None:
            if type(index_page) is not int or index_page < 0:
                raise ValueError("Group ready-member writable index cursor is invalid.")
            if entries is None:
                if cursor.get("writable_index_steps", 0) >= next_index_page:
                    raise ValueError("Group ready-member writable index chain is invalid.")
                record = read_group_ready_member_writable_index(cfg, group_name, index_page)
                if not record["member_pages"]:
                    raise ValueError("Group ready-member writable index queue is invalid.")
                cursor.update(
                    {
                        "writable_index_entries": list(record["member_pages"]),
                        "writable_index_offset": 0,
                        "writable_index_next_page": record["next_page"],
                        "writable_index_last_page": index_page,
                        "writable_index_steps": cursor.get("writable_index_steps", 0) + 1,
                    }
                )
                work.consume("writable-index-page")
                processed = work.consumed
                continue
            offset = cursor.get("writable_index_offset")
            if type(offset) is not int or offset < 0 or not isinstance(entries, list):
                raise ValueError("Group ready-member writable index cursor is invalid.")
            if offset >= len(entries):
                cursor["writable_index_page"] = cursor.pop("writable_index_next_page", None)
                cursor["writable_index_entries"] = None
                cursor["writable_index_offset"] = 0
                continue
            member_page = entries[offset]
            if type(member_page) is not int or member_page < 0:
                raise ValueError("Group ready-member writable index member page is invalid.")
            state, catalog, partition = load_group_ready_member_page(cfg, group_name, member_page)
            page_entries = read_group_ready_member_page_entries(group_name, state, catalog, partition)
            if (
                len(page_entries) >= GROUP_MEMBER_PAGE_SIZE
                or member_page == cursor.get("writable_current_page")
                or not catalog["group_ready_member_catalog"].get("writable_indexed", False)
            ):
                raise ValueError("Group ready-member writable index coverage is invalid.")
            cursor["writable_seen_count"] += 1
            cursor["writable_seen_digest"] = _update_writable_page_digest(
                cursor["writable_seen_digest"], member_page
            )
            cursor["writable_index_offset"] = offset + 1
            work.consume("writable-member-page")
            processed = work.consumed
            continue
        free_page = cursor.get("writable_free_page")
        if free_page is not None:
            if type(free_page) is not int or free_page < 0:
                raise ValueError("Group ready-member writable free cursor is invalid.")
            if cursor.get("writable_free_steps", 0) >= next_index_page:
                raise ValueError("Group ready-member writable free chain is invalid.")
            record = read_group_ready_member_writable_index(cfg, group_name, free_page)
            if record["member_pages"]:
                raise ValueError("Group ready-member writable free page is not empty.")
            cursor["writable_free_page"] = record["next_page"]
            cursor["writable_free_steps"] = cursor.get("writable_free_steps", 0) + 1
            work.consume("writable-free-page")
            processed = work.consumed
            continue
        if (
            cursor.get("writable_index_last_page") != cursor.get("writable_tail")
            or cursor.get("writable_seen_count") != index_count
            or cursor.get("writable_seen_count") != cursor.get("writable_expected_count")
            or cursor.get("writable_seen_digest") != cursor.get("writable_expected_digest")
        ):
            raise ValueError("Group ready-member writable index coverage is invalid.")
        cursor["writable_audit_complete"] = True
        return processed
    return processed


def _advance_member_audit(
    cfg: object,
    build_id: str,
    cursor: dict[str, Any],
    max_entries: int,
) -> tuple[dict[str, Any], int, bool]:
    """Consume bounded actual-member records through durable Group and page cursors."""

    def reset_group_audit(state: dict[str, Any]) -> None:
        group = state["group_ready_members"]
        revision = group.get("membership_revision")
        if (
            type(group.get("directory_page_count")) is not int
            or group["directory_page_count"] < 0
            or type(revision) is not int
            or revision < 0
        ):
            raise ValueError("Group ready-member audit state is invalid.")
        cursor.update(
            {
                "directory_page": group.get("directory_head"),
                "directory_offset": 0,
                "member_page": None,
                "entry_offset": 0,
                "pending_entry": None,
                "seen_count": 0,
                "seen_digest": group_ready_member_digest([]),
                "membership_revision": revision,
                "writable_index_page": group.get("writable_index_head"),
                "writable_index_entries": None,
                "writable_index_offset": 0,
                "writable_index_last_page": None,
                "writable_index_steps": 0,
                "writable_free_page": group.get("writable_index_free_head"),
                "writable_free_steps": 0,
                "writable_seen_count": 0,
                "writable_seen_digest": "0" * 64,
                "writable_expected_count": 0,
                "writable_expected_digest": "0" * 64,
                "writable_audit_complete": False,
                "writable_current_page": group.get("writable_member_page"),
                "writable_tail": group.get("writable_index_tail"),
                "writable_index_count": group.get("writable_index_count"),
                "next_writable_index_page": group.get("next_writable_index_page"),
            }
        )

    processed = 0
    while processed < max_entries:
        group_name = cursor.get("group_name")
        pending_error = cursor.get("pending_error")
        if pending_error is not None:
            if not isinstance(pending_error, str) or group_name is None:
                raise ValueError("Group ready-member audit pending error is invalid.")
            state, _catalog, _partition = load_group_ready_member_page(cfg, group_name)
            processed += 1
            if cursor.get("membership_revision") != state["group_ready_members"].get("membership_revision"):
                cursor["pending_error"] = None
                reset_group_audit(state)
                continue
            raise ValueError(pending_error)
        if group_name is None:
            group_page = cursor.get("group_page", 0)
            group_offset = cursor.get("group_offset", 0)
            group_page_count = cursor.get("group_page_count")
            if (
                type(group_page) is not int
                or type(group_offset) is not int
                or type(group_page_count) is not int
                or group_page < 0
                or group_offset < 0
            ):
                raise ValueError("Group ready-member audit cursor is invalid.")
            if group_page >= group_page_count:
                return cursor, processed, True
            groups = _load_build_page(cfg, build_id, group_page, collection="audit-groups")
            if group_offset >= len(groups):
                cursor["group_page"] += 1
                cursor["group_offset"] = 0
                continue
            group_key = groups[group_offset]
            cursor["group_offset"] += 1
            state = read_group_ready_member_audit_state(cfg, group_key)
            cursor.update(
                {
                    "group_name": state["group_name"],
                    "directory_page": state.get("directory_head"),
                    "directory_offset": 0,
                    "member_page": None,
                    "entry_offset": 0,
                    "pending_entry": None,
                    "pending_error": None,
                    "seen_count": 0,
                    "seen_digest": group_ready_member_digest([]),
                    "membership_revision": state["membership_revision"],
                    "writable_index_page": state.get("writable_index_head"),
                    "writable_index_entries": None,
                    "writable_index_offset": 0,
                    "writable_index_last_page": None,
                    "writable_index_steps": 0,
                    "writable_free_page": state.get("writable_index_free_head"),
                    "writable_free_steps": 0,
                    "writable_seen_count": 0,
                    "writable_seen_digest": "0" * 64,
                    "writable_expected_count": 0,
                    "writable_expected_digest": "0" * 64,
                    "writable_audit_complete": False,
                    "writable_current_page": state.get("writable_member_page"),
                    "writable_tail": state.get("writable_index_tail"),
                    "writable_index_count": state.get("writable_index_count"),
                    "next_writable_index_page": state.get("next_writable_index_page"),
                }
            )
            processed += 1
            continue
        directory_page = cursor.get("directory_page")
        directory_offset = cursor.get("directory_offset")
        if type(directory_offset) is not int or directory_offset < 0:
            raise ValueError("Group ready-member audit cursor is invalid.")
        if directory_page is None:
            if not cursor.get("writable_audit_complete", False):
                processed += _advance_writable_index_audit(
                    cfg,
                    group_name,
                    cursor,
                    max_entries - processed,
                )
                if not cursor.get("writable_audit_complete", False):
                    return cursor, processed, False
            if processed >= max_entries:
                return cursor, processed, False
            state, _catalog, _partition = load_group_ready_member_page(cfg, group_name)
            processed += 1
            group = state["group_ready_members"]
            if cursor.get("membership_revision") != group.get("membership_revision"):
                reset_group_audit(state)
                continue
            if cursor.get("seen_count") != group.get("member_count") or cursor.get("seen_digest") != group.get(
                "membership_digest"
            ):
                raise ValueError(f"Group {group_name!r} ready-member count or digest is invalid.")
            cursor.update(
                {
                    "group_name": None,
                    "directory_page": None,
                    "directory_offset": 0,
                    "member_page": None,
                    "entry_offset": 0,
                    "pending_entry": None,
                    "pending_error": None,
                    "seen_count": 0,
                    "seen_digest": group_ready_member_digest([]),
                }
            )
            continue
        member_page = cursor.get("member_page")
        if member_page is None:
            if processed >= max_entries:
                return cursor, processed, False
            directory = read_group_ready_member_directory(cfg, group_name, directory_page)
            processed += 1
            member_pages = directory["member_pages"]
            if directory_offset >= len(member_pages):
                cursor["directory_page"] = directory["next_page"]
                cursor["directory_offset"] = 0
                cursor["entry_offset"] = 0
                continue
            member_page = member_pages[directory_offset]
            cursor["member_page"] = member_page
            # A directory read is itself one budgeted maintenance item.  Do
            # not begin the following member-page validation after consuming
            # the final unit in this slice.
            if processed >= max_entries:
                return cursor, processed, False
        if type(member_page) is not int or member_page < 0:
            raise ValueError("Group ready-member audit member page is invalid.")
        pending_entry = cursor.get("pending_entry")
        offset = cursor.get("entry_offset")
        if type(offset) is not int or offset < 0:
            raise ValueError("Group ready-member audit cursor is invalid.")
        if pending_entry is None:
            # Loading the bounded member page is independently budgeted.  The
            # selected entry is persisted so an N=1 continuation can audit it
            # without reopening the page in the same slice.
            state, catalog, partition = load_group_ready_member_page(cfg, group_name, member_page)
            entries = read_group_ready_member_page_entries(group_name, state, catalog, partition)
            processed += 1
            catalog_record = catalog["group_ready_member_catalog"]
            is_indexed = catalog_record.get("writable_indexed")
            if not isinstance(is_indexed, bool):
                raise ValueError("Group ready-member writable-index flag is invalid.")
            should_be_indexed = (
                len(entries) < GROUP_MEMBER_PAGE_SIZE
                and member_page != cursor.get("writable_current_page")
            )
            if is_indexed != should_be_indexed:
                raise ValueError("Group ready-member writable-index coverage is invalid.")
            if should_be_indexed:
                cursor["writable_expected_count"] += 1
                cursor["writable_expected_digest"] = _update_writable_page_digest(
                    cursor["writable_expected_digest"], member_page
                )
            if offset >= len(entries):
                cursor["directory_offset"] += 1
                cursor["member_page"] = None
                cursor["entry_offset"] = 0
                continue
            cursor["pending_entry"] = entries[offset]
            if processed >= max_entries:
                return cursor, processed, False
            pending_entry = entries[offset]
        if not isinstance(pending_entry, dict):
            raise ValueError("Group ready-member audit pending entry is invalid.")
        if processed >= max_entries:
            return cursor, processed, False
        entry = pending_entry
        processed += 1
        try:
            _audit_member_entry(cfg, group_name, entry)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            # Do the revision read in the next slice.  It is an independently
            # charged restart check and avoids hidden I/O after an N=1 entry
            # validation has exhausted its budget.
            cursor["pending_error"] = "Group ready-member audit entry is inconsistent."
            return cursor, processed, False
        cursor["seen_digest"] = update_group_ready_member_digest(cursor["seen_digest"], entry)
        cursor["seen_count"] += 1
        cursor["entry_offset"] += 1
        cursor["pending_entry"] = None
    return cursor, processed, False


def advance_group_ready_members_build(
    cfg: object,
    *,
    max_tasks: int = GROUP_MEMBER_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance one bounded backfill, audit, or candidate-rebuild slice."""
    if type(max_tasks) is not int or not 1 <= max_tasks <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_tasks must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    record = begin_group_ready_members_build(cfg)
    if record["state"] != "building":
        return record
    try:
        with exclusive(group_ready_members_state_lock_path(cfg)):
            value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            build = record.get("build")
            if record.get("state") != "building" or not isinstance(build, dict):
                return record
            build_id = build.get("build_id")
            if not isinstance(build_id, str):
                raise ValueError("Group ready-member build state is invalid.")
            phase = build.get("phase")
            watermark = build.get("watermark")
            if not isinstance(watermark, dict):
                raise ValueError("Group ready-member build watermark is invalid.")
        if not watermark.get("is_complete"):
            captured = _capture_build_watermark(
                cfg,
                build_id=build_id,
                watermark=watermark,
                max_entries=max_tasks,
            )
            return _commit_build_record(
                cfg,
                build_id=build_id,
                build_updates={"watermark": captured},
            )
        if phase == "backfill":
            cursor = dict(build.get("cursor", {}))
            page_count = watermark.get("page_count")
            if (
                type(page_count) is not int
                or not isinstance(cursor.get("page"), int)
                or not isinstance(cursor.get("offset"), int)
            ):
                raise ValueError("Group ready-member build cursor is invalid.")
            processed = 0
            while processed < max_tasks and cursor["page"] < page_count:
                items = _load_build_page(cfg, build_id, cursor["page"])
                if cursor["offset"] >= len(items):
                    cursor.update({"page": cursor["page"] + 1, "offset": 0})
                    continue
                try:
                    task = TaskRecord.from_dict(
                        read_json(shared_paths(cfg.shared_root)["tasks"] / f"{items[cursor['offset']]}.json")
                    )
                except FileNotFoundError:
                    task = None
                if task is not None and _should_index(task):
                    reference = reference_for_generation(cfg, task.task_id, task.ready_generation)
                    if reference is None:
                        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
                    publish_group_ready_member(cfg, task, reference)
                processed += 1
                _advance_page_cursor(cursor, len(items))
            updates: dict[str, Any] = {}
            if cursor["page"] >= page_count:
                updates = {"phase": "audit-task-capture"}
            return _commit_build_record(
                cfg,
                build_id=build_id,
                cursor=cursor,
                processed=processed,
                build_updates=updates,
            )
        if phase == "audit-task-capture":
            current = build.get("audit_task_watermark", {"is_complete": False})
            if not isinstance(current, dict):
                raise ValueError("Group ready-member audit watermark is invalid.")
            # Empty the old candidate projection before fixing the final Task
            # watermark.  Every Task or Group mutation after that fence will
            # dual-write the cleared rebuild, including work outside the
            # watermark that is captured below.
            begin_primary_ready_index_rebuild(cfg, build_id)
            captured = _capture_build_watermark(
                cfg, build_id=build_id, watermark=current, max_entries=max_tasks, collection="audit-tasks"
            )
            updates = {"audit_task_watermark": captured}
            if captured.get("is_complete"):
                updates.update(
                    {
                        "phase": "audit-tasks",
                        "audit_task_cursor": {"page": 0, "offset": 0},
                    }
                )
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        if phase == "audit-tasks":
            audit = build.get("audit_task_watermark", {})
            cursor = dict(build.get("audit_task_cursor", {}))
            page_count = audit.get("page_count")
            if not audit.get("is_complete") or type(page_count) is not int:
                raise ValueError("Group ready-member audit Task cursor is invalid.")
            processed = 0
            while processed < max_tasks and cursor["page"] < page_count:
                items = _load_build_page(cfg, build_id, cursor["page"], collection="audit-tasks")
                if cursor["offset"] >= len(items):
                    cursor.update({"page": cursor["page"] + 1, "offset": 0})
                    continue
                _audit_task_member(cfg, items[cursor["offset"]])
                processed += 1
                _advance_page_cursor(cursor, len(items))
            updates = {"audit_task_cursor": cursor}
            if cursor["page"] >= page_count:
                updates["phase"] = "audit-group-capture"
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        if phase == "audit-group-capture":
            current = build.get("audit_group_watermark", {"is_complete": False})
            if not isinstance(current, dict):
                raise ValueError("Group ready-member audit Group watermark is invalid.")
            captured = _capture_build_watermark(
                cfg,
                build_id=build_id,
                watermark=current,
                max_entries=max_tasks,
                source_path=group_ready_member_groups_root(cfg),
                collection="audit-groups",
                should_include_json=False,
            )
            updates = {"audit_group_watermark": captured}
            if captured.get("is_complete"):
                updates.update(
                    {
                        "phase": "audit-members",
                        "audit_member_cursor": {
                            "group_page": 0,
                            "group_offset": 0,
                            "group_page_count": captured["page_count"],
                            "group_name": None,
                            "directory_page": None,
                            "directory_offset": 0,
                            "member_page": None,
                            "entry_offset": 0,
                            "pending_entry": None,
                            "seen_count": 0,
                            "seen_digest": group_ready_member_digest([]),
                        },
                    }
                )
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        if phase == "audit-members":
            cursor, _processed, complete = _advance_member_audit(
                cfg,
                build_id,
                dict(build.get("audit_member_cursor", {})),
                max_tasks,
            )
            updates = {"audit_member_cursor": cursor}
            if complete:
                updates.update(
                    {
                        "phase": "primary-rebuild",
                        "primary_cursor": {"page": 0, "offset": 0},
                    }
                )
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        if phase == "primary-rebuild":
            audit = build.get("audit_task_watermark", {})
            cursor = dict(build.get("primary_cursor", {}))
            page_count = audit.get("page_count")
            if not audit.get("is_complete") or type(page_count) is not int:
                raise ValueError("Primary candidate rebuild cursor is invalid.")
            begin_primary_ready_index_rebuild(cfg, build_id)
            processed = 0
            while processed < max_tasks and cursor["page"] < page_count:
                items = _load_build_page(cfg, build_id, cursor["page"], collection="audit-tasks")
                if cursor["offset"] >= len(items):
                    cursor.update({"page": cursor["page"] + 1, "offset": 0})
                    continue
                try:
                    task = TaskRecord.from_dict(
                        read_json(shared_paths(cfg.shared_root)["tasks"] / f"{items[cursor['offset']]}.json")
                    )
                except FileNotFoundError:
                    task = None
                if task is not None and task_should_have_ready_marker(task):
                    reference = reference_for_generation(cfg, task.task_id, task.ready_generation)
                    if reference is None:
                        raise ValueError(f"Primary candidate marker is missing for Task {task.task_id!r}.")
                    rebuild_primary_ready_candidate(cfg, build_id, task, reference)
                processed += 1
                _advance_page_cursor(cursor, len(items))
            updates = {"primary_cursor": cursor}
            if cursor["page"] >= page_count:
                complete_primary_ready_index_rebuild(cfg, build_id)
                return _commit_build_record(
                    cfg,
                    build_id=build_id,
                    build_updates=updates,
                    should_activate=True,
                )
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        raise ValueError("Group ready-member build phase is invalid.")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"build_failed:{type(exc).__name__}:{exc}")
        return read_group_ready_members_state(cfg)


def repair_group_ready_members(
    cfg: object,
    *,
    max_work_items: int = GROUP_MEMBER_PAGE_SIZE,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Rebuild damaged derived membership from durable Task and ready truth."""
    from ...layout import is_group_ready_members_root

    if max_tasks is not None:
        max_work_items = max_tasks

    if not is_group_ready_members_root(cfg):
        return {"state": "legacy", "required_capability": GROUP_READY_MEMBERS_CAPABILITY}
    try:
        current = read_group_ready_members_state(cfg)
        if current["state"] == "active":
            return advance_group_ready_members_audit(cfg, max_work_items=max_work_items)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"doctor_repair:{type(exc).__name__}")
    else:
        if current["state"] == "building":
            return advance_group_ready_members_build(cfg, max_tasks=max_work_items)
    begin_group_ready_members_build(cfg, is_repair=True)
    return advance_group_ready_members_build(cfg, max_tasks=max_work_items)


def cleanup_group_ready_member_archives(
    cfg: object,
    *,
    max_work_items: int = GROUP_MEMBER_PAGE_SIZE,
) -> dict[str, Any]:
    """Delete a bounded archive slice outside the schema lock."""
    if type(max_work_items) is not int or not 1 <= max_work_items <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_work_items must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    from ...layout import is_group_ready_members_root

    if not is_group_ready_members_root(cfg):
        return {"state": "completed", "archive_count": 0, "processed": 0}
    audit_result = _cleanup_completed_audit_scratch(cfg, max_work_items)
    if audit_result is not None:
        return audit_result
    root = group_ready_members_root(cfg) / "replaced-groups"
    # The member-state lock remains short.  This dedicated lock makes the
    # selected cursor and all unlink operations a single-owner transaction.
    with exclusive(_archive_cleanup_lock_path(cfg)):
        with exclusive(group_ready_members_state_lock_path(cfg)):
            value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            cleanup = record.get("archive_cleanup")
        if not isinstance(cleanup, dict):
            archive = _first_archive_directory(root)
            if archive is None:
                return _complete_empty_archive_cleanup(cfg)
            cleanup = {
                "build_id": archive.name,
                "group_key": None,
                "processed": 0,
                "phase": "select-group",
            }
            with exclusive(group_ready_members_state_lock_path(cfg)):
                value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
                record = value["group_ready_members"]
                active = record.get("archive_cleanup")
                if isinstance(active, dict):
                    cleanup = active
                else:
                    record["archive_cleanup"] = cleanup
                    record["revision"] += 1
                    record["updated_at"] = utc_now()
                    atomic_replace(group_ready_members_state_path(cfg), value)
        archive = root / cleanup["build_id"]
        if archive.parent != root or archive.is_symlink():
            raise ValueError("Group ready-member archive cleanup target is invalid.")
        if archive.exists():
            cleanup, processed = _advance_archive_cleanup(archive, dict(cleanup), max_work_items)
        else:
            processed = 0
        with exclusive(group_ready_members_state_lock_path(cfg)):
            value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            active = record.get("archive_cleanup")
            if isinstance(active, dict) and active.get("build_id") == cleanup["build_id"]:
                if archive.exists():
                    cleanup["processed"] = active.get("processed", 0) + processed
                    record["archive_cleanup"] = cleanup
                else:
                    record["archive_count"] = max(0, record.get("archive_count", 0) - 1)
                    record["archive_cleanup"] = None
                record["revision"] += 1
                record["updated_at"] = utc_now()
                atomic_replace(group_ready_members_state_path(cfg), value)
            return {
                "state": "building" if archive.exists() else "completed",
                "archive_count": record.get("archive_count", 0),
                "processed": processed,
                "build_id": cleanup["build_id"],
            }


def _complete_empty_archive_cleanup(cfg: object) -> dict[str, Any]:
    """Record that no archive exists without scanning while holding the state lock."""
    with exclusive(group_ready_members_state_lock_path(cfg)):
        value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
        record = value["group_ready_members"]
        if not isinstance(record.get("archive_cleanup"), dict):
            record["archive_count"] = 0
            record["revision"] += 1
            record["updated_at"] = utc_now()
            atomic_replace(group_ready_members_state_path(cfg), value)
        return {"state": "completed", "archive_count": record["archive_count"], "processed": 0}


def _cleanup_completed_audit_scratch(
    cfg: object, max_work_items: int
) -> dict[str, Any] | None:
    """Reclaim one completed audit's fixed-layout capture pages in bounded slices."""
    with exclusive(_archive_cleanup_lock_path(cfg)):
        with exclusive(group_ready_members_state_lock_path(cfg)):
            value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            cleanup = record.get("audit_cleanup")
        if not isinstance(cleanup, dict):
            candidate = _first_completed_audit_scratch(cfg, record)
            if candidate is None:
                return None
            cleanup = {"audit_id": candidate.name, "phase": "watermark", "processed": 0}
            with exclusive(group_ready_members_state_lock_path(cfg)):
                value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
                record = value["group_ready_members"]
                active = record.get("audit_cleanup")
                if isinstance(active, dict):
                    cleanup = active
                else:
                    record["audit_cleanup"] = cleanup
                    record["revision"] += 1
                    record["updated_at"] = utc_now()
                    atomic_replace(group_ready_members_state_path(cfg), value)
        audit_id = cleanup.get("audit_id")
        phase = cleanup.get("phase")
        if not isinstance(audit_id, str) or phase not in {
            "watermark",
            "audit-tasks",
            "audit-groups",
            "root",
        }:
            raise ValueError("Group ready-member audit cleanup state is invalid.")
        root = _build_root(cfg, audit_id)
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ValueError("Group ready-member audit cleanup target is invalid.")
        processed = _advance_audit_scratch_cleanup(root, cleanup, max_work_items)
        with exclusive(group_ready_members_state_lock_path(cfg)):
            value = read_json_limited(group_ready_members_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            active = record.get("audit_cleanup")
            if isinstance(active, dict) and active.get("audit_id") == audit_id:
                if root.exists():
                    cleanup["processed"] = active.get("processed", 0) + processed
                    record["audit_cleanup"] = cleanup
                else:
                    record["audit_cleanup"] = None
                record["revision"] += 1
                record["updated_at"] = utc_now()
                atomic_replace(group_ready_members_state_path(cfg), value)
            return {
                "state": "building" if root.exists() else "completed",
                "archive_count": record.get("archive_count", 0),
                "processed": processed,
                "audit_id": audit_id,
            }


def _advance_audit_scratch_cleanup(root: Path, cleanup: dict[str, Any], max_work_items: int) -> int:
    """Delete no more than one budgeted set of fixed audit capture entries."""
    processed = 0
    phase = cleanup["phase"]
    while processed < max_work_items:
        if phase in {"watermark", "audit-tasks", "audit-groups"}:
            directory = root / phase
            processed += _delete_archive_directory_entries(directory, max_work_items - processed)
            if processed >= max_work_items:
                break
            if directory.exists():
                directory.rmdir()
                processed += 1
            phase = {
                "watermark": "audit-tasks",
                "audit-tasks": "audit-groups",
                "audit-groups": "root",
            }[phase]
            cleanup["phase"] = phase
            continue
        if root.exists():
            root.rmdir()
            processed += 1
        break
    return processed


def _first_completed_audit_scratch(cfg: object, record: dict[str, Any]) -> Path | None:
    """Choose one completed audit scratch directory without retaining an ID queue."""
    root = group_ready_members_root(cfg) / "builds"
    if not root.is_dir():
        return None
    build = record.get("build")
    active_build_id = build.get("build_id") if isinstance(build, dict) else None
    audit = record.get("audit")
    active_audit_id = None
    if isinstance(audit, dict) and audit.get("state") == "building":
        active_audit_id = audit.get("audit_id")
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ValueError("Group ready-member build root contains an unsafe entry.")
            if entry.name not in {active_build_id, active_audit_id}:
                validate_identifier(entry.name, "group ready-member audit id")
                return Path(entry.path)
    return None


def _first_archive_directory(root: Path) -> Path | None:
    """Return one archive candidate without materializing the directory."""
    if not root.is_dir():
        return None
    with os.scandir(root) as entries:
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ValueError("Group ready-member archive root contains an unsafe entry.")
            return path
    return None


def _advance_archive_cleanup(
    archive: Path,
    cleanup: dict[str, Any],
    max_work_items: int,
) -> tuple[dict[str, Any], int]:
    """Remove a bounded flat-layout archive slice without recursive traversal."""
    phase = cleanup.get("phase")
    group_key = cleanup.get("group_key")
    allowed_phases = {
        "select-group",
        "locators",
        "catalog",
        "partitions",
        "directories",
        "writable-pages",
        "state",
        "group",
    }
    if phase not in allowed_phases:
        raise ValueError("Group ready-member archive cleanup state is invalid.")
    if group_key is not None and (
        not isinstance(group_key, str) or "/" in group_key or group_key in {"", ".", ".."}
    ):
        raise ValueError("Group ready-member archive cleanup group is invalid.")
    processed = 0
    while processed < max_work_items:
        if phase == "select-group":
            group = _first_archive_directory(archive)
            if group is None:
                archive.rmdir()
                processed += 1
                break
            cleanup.update({"group_key": group.name, "phase": "locators"})
            group_key = group.name
            phase = "locators"
            processed += 1
            continue
        if group_key is None:
            raise ValueError("Group ready-member archive cleanup group is invalid.")
        group_root = archive / group_key
        if group_root.parent != archive or group_root.is_symlink() or not group_root.is_dir():
            raise ValueError("Group ready-member archive cleanup target is invalid.")
        if phase in {"locators", "catalog", "partitions", "directories", "writable-pages"}:
            directory = group_root / phase
            processed += _delete_archive_directory_entries(directory, max_work_items - processed)
            if processed >= max_work_items:
                break
            if directory.exists():
                directory.rmdir()
                processed += 1
            next_phase = {
                "locators": "catalog",
                "catalog": "partitions",
                "partitions": "directories",
                "directories": "writable-pages",
                "writable-pages": "state",
            }[phase]
            cleanup["phase"] = next_phase
            phase = next_phase
            continue
        if phase == "state":
            state_path = group_root / "state.json"
            if state_path.exists():
                if state_path.is_symlink() or not state_path.is_file():
                    raise ValueError("Group ready-member archive contains an unsafe state entry.")
                state_path.unlink()
                processed += 1
            cleanup["phase"] = "group"
            phase = "group"
            continue
        group_root.rmdir()
        processed += 1
        cleanup.update({"group_key": None, "phase": "select-group"})
        group_key = None
        phase = "select-group"
    return cleanup, processed


def _delete_archive_directory_entries(directory: Path, budget: int) -> int:
    """Delete up to budget regular files from one known flat archive directory."""
    if budget <= 0 or not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Group ready-member archive contains an unsafe directory.")
    processed = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if processed >= budget:
                break
            path = Path(entry.path)
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError("Group ready-member archive contains an unsafe entry.")
            path.unlink()
            processed += 1
    return processed
