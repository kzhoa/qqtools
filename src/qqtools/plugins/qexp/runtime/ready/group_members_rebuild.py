"""Resumable rebuild orchestration for Group ready-member projection."""

from __future__ import annotations

import ctypes
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..locks import exclusive
from ..paths import shared_paths
from ..records import TaskRecord, utc_now, validate_identifier
from ..store import atomic_replace, iter_json, read_json
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
    read_group_ready_member_page_entries,
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


def begin_group_ready_members_build(cfg: object, *, is_repair: bool = False) -> dict[str, Any]:
    """Start one durable, resumable backfill for legacy roots."""
    root = group_ready_members_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    group_ready_member_groups_root(cfg).mkdir(parents=True, exist_ok=True)
    path = group_ready_members_state_path(cfg)
    with exclusive(group_ready_members_state_lock_path(cfg)):
        if path.exists():
            record = read_group_ready_members_state(cfg)
            if record["state"] == "active" and not is_repair:
                return record
            if record["state"] in {"building", "degraded"} and not is_repair:
                return record
        if is_repair:
            groups = group_ready_member_groups_root(cfg)
            if groups.exists():
                shutil.rmtree(groups)
            groups.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "required_capability": GROUP_READY_MEMBERS_CAPABILITY,
            "state": "building",
            "revision": 0,
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


def audit_group_ready_members(cfg: object) -> dict[str, Any]:
    """Verify member entries against authoritative queued Task and ready-marker truth."""
    state = read_group_ready_members_state(cfg)
    if state["state"] not in {"building", "active"}:
        raise ValueError("Group ready-member projection is not writable.")
    expected: dict[str, dict[str, tuple[TaskRecord, ReadyMarkerRef]]] = {}
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if not _should_index(task):
            continue
        reference = reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is None:
            raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
        identity = group_ready_member_identity(task.group_name or "", task.task_id, task.ready_generation)
        expected.setdefault(task.group_name or "", {})[identity] = (task, reference)

    actual: dict[str, dict[str, dict[str, Any]]] = {}
    groups_root = group_ready_member_groups_root(cfg)
    for state_path in sorted(groups_root.glob("*/state.json")):
        group = read_json(state_path)["group_ready_members"]
        group_name = group.get("group_name")
        if not isinstance(group_name, str):
            raise ValueError("Group ready-member state is invalid.")
        entries = read_group_ready_members(cfg, group_name)
        actual[group_name] = {entry["identity"]: entry for entry in entries}

    for group_name in sorted(set(expected) | set(actual)):
        expected_entries = expected.get(group_name, {})
        actual_entries = actual.get(group_name, {})
        if set(expected_entries) != set(actual_entries):
            raise ValueError(f"Group {group_name!r} ready-member Task coverage is invalid.")
        for identity, entry in actual_entries.items():
            task, reference = expected_entries[identity]
            if (
                entry.get("task_id") != task.task_id
                or entry.get("generation") != task.ready_generation
                or entry.get("queue_scope") != reference.queue_scope
                or entry.get("home_machine") != reference.home_machine
                or entry.get("partition") != reference.partition
                or entry.get("catalog_page") != reference.catalog_page
                or entry.get("marker_name") != reference.marker_name
                or entry.get("lane") != (task.spec.lane or "gpu")
                or entry.get("submission_operation_id") != task.submission_operation_id
            ):
                raise ValueError(f"Group {group_name!r} ready-member entry is stale.")
    return state


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
    atomic_replace(
        _build_page_path(cfg, build_id, page, collection=collection),
        {
            "group_ready_member_build_page": {
                "schema_version": GROUP_READY_MEMBERS_VERSION,
                "build_id": build_id,
                "page": page,
                "task_ids": list(task_ids),
            }
        },
    )


def _load_build_page(
    cfg: object,
    build_id: str,
    page: int,
    *,
    collection: str = "watermark",
) -> list[str]:
    record = read_json(_build_page_path(cfg, build_id, page, collection=collection))["group_ready_member_build_page"]
    task_ids = record.get("task_ids")
    if (
        record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("build_id") != build_id
        or record.get("page") != page
        or not isinstance(task_ids, list)
        or len(task_ids) > GROUP_MEMBER_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in task_ids)
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
        value = read_json(group_ready_members_state_path(cfg))
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


def _advance_member_audit(
    cfg: object,
    build_id: str,
    cursor: dict[str, Any],
    max_entries: int,
) -> tuple[dict[str, Any], int, bool]:
    """Consume bounded actual-member records through durable Group and page cursors."""

    def reset_group_audit(state: dict[str, Any]) -> None:
        group = state["group_ready_members"]
        pages = group.get("catalog_pages")
        revision = group.get("membership_revision")
        if (
            not isinstance(pages, list)
            or not all(type(page) is int and page >= 0 for page in pages)
            or pages != sorted(set(pages))
            or type(revision) is not int
            or revision < 0
        ):
            raise ValueError("Group ready-member audit state is invalid.")
        cursor.update(
            {
                "pages": pages,
                "page_index": 0,
                "entry_offset": 0,
                "seen_count": 0,
                "seen_digest": group_ready_member_digest([]),
                "membership_revision": revision,
            }
        )

    processed = 0
    while processed < max_entries:
        group_name = cursor.get("group_name")
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
            pages = state["catalog_pages"]
            cursor.update(
                {
                    "group_name": state["group_name"],
                    "pages": pages,
                    "page_index": 0,
                    "entry_offset": 0,
                    "seen_count": 0,
                    "seen_digest": group_ready_member_digest([]),
                    "membership_revision": state["membership_revision"],
                }
            )
            processed += 1
            continue
        pages = cursor.get("pages")
        page_index = cursor.get("page_index")
        if not isinstance(pages, list) or type(page_index) is not int:
            raise ValueError("Group ready-member audit cursor is invalid.")
        if page_index >= len(pages):
            state, _catalog, _partition = load_group_ready_member_page(cfg, group_name)
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
                    "pages": [],
                    "page_index": 0,
                    "entry_offset": 0,
                    "seen_count": 0,
                    "seen_digest": group_ready_member_digest([]),
                }
            )
            continue
        state, catalog, partition = load_group_ready_member_page(cfg, group_name, pages[page_index])
        entries = read_group_ready_member_page_entries(group_name, state, catalog, partition)
        offset = cursor.get("entry_offset")
        if type(offset) is not int or offset < 0:
            raise ValueError("Group ready-member audit cursor is invalid.")
        if offset >= len(entries):
            cursor["page_index"] += 1
            cursor["entry_offset"] = 0
            continue
        entry = entries[offset]
        try:
            _audit_member_entry(cfg, group_name, entry)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            state, _catalog, _partition = load_group_ready_member_page(cfg, group_name)
            if cursor.get("membership_revision") != state["group_ready_members"].get("membership_revision"):
                reset_group_audit(state)
                continue
            raise
        cursor["seen_digest"] = update_group_ready_member_digest(cursor["seen_digest"], entry)
        cursor["seen_count"] += 1
        cursor["entry_offset"] += 1
        processed += 1
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
            value = read_json(group_ready_members_state_path(cfg))
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
                            "pages": [],
                            "page_index": 0,
                            "entry_offset": 0,
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


def repair_group_ready_members(cfg: object, *, max_tasks: int = GROUP_MEMBER_PAGE_SIZE) -> dict[str, Any]:
    """Rebuild damaged derived membership from durable Task and ready truth."""
    from ...layout import is_group_ready_members_root

    if not is_group_ready_members_root(cfg):
        return {"state": "legacy", "required_capability": GROUP_READY_MEMBERS_CAPABILITY}
    try:
        current = read_group_ready_members_state(cfg)
        if current["state"] == "active":
            audit_group_ready_members(cfg)
            return current
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"doctor_repair:{type(exc).__name__}")
    else:
        if current["state"] == "building":
            return advance_group_ready_members_build(cfg, max_tasks=max_tasks)
    begin_group_ready_members_build(cfg, is_repair=True)
    return advance_group_ready_members_build(cfg, max_tasks=max_tasks)
