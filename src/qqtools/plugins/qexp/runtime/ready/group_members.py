"""Derived live-ready membership indexed by Group."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..locks import exclusive
from ..paths import shared_paths
from ..records import TaskRecord, utc_now, validate_identifier
from ..store import atomic_replace, read_json

if TYPE_CHECKING:
    from .index import ReadyMarkerRef
    from ..records import TaskRecord

GROUP_READY_MEMBERS_CAPABILITY = "group-ready-members-v1"
GROUP_READY_MEMBERS_VERSION = 1
GROUP_MEMBER_PAGE_SIZE = 64
_MEMBERSHIP_DIGEST_ALGORITHM = "xor-sha256-v1"


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


def _root(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_group_members"]


def _state_path(cfg: object) -> Path:
    return _root(cfg) / "state.json"


def _state_lock_path(cfg: object) -> Path:
    """Return the short global fence for the member-projection state."""
    return shared_paths(cfg.shared_root)["ready_locks"] / "group-members-state.lock"


def _group_key(group_name: str) -> str:
    validate_identifier(group_name, "group_name")
    return hashlib.sha256(group_name.encode()).hexdigest()


def _group_root(cfg: object, group_name: str) -> Path:
    return shared_paths(cfg.shared_root)["ready_group_member_groups"] / _group_key(group_name)


def _group_state_path(cfg: object, group_name: str) -> Path:
    return _group_root(cfg, group_name) / "state.json"


def _catalog_path(cfg: object, group_name: str, page: int = 0) -> Path:
    return _group_root(cfg, group_name) / "catalog" / f"{page:016d}.json"


def _partition_path(cfg: object, group_name: str, page: int = 0) -> Path:
    return _group_root(cfg, group_name) / "partitions" / f"{page:016d}.json"


def _identity(group_name: str, task_id: str, generation: int) -> str:
    return hashlib.sha256(f"{group_name}\0{task_id}\0{generation}".encode()).hexdigest()


def _locator_path(cfg: object, group_name: str, identity: str) -> Path:
    return _group_root(cfg, group_name) / "locators" / f"{identity}.json"


def _digest(entries: list[dict[str, Any]]) -> str:
    """Return a reversible set digest so exact retirement stays page-bounded."""
    digest = 0
    for item in entries:
        value = (item["identity"], item["task_id"], item["generation"], item["member_revision"])
        digest ^= int.from_bytes(
            hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).digest(), "big"
        )
    return f"{digest:064x}"


def _update_digest(current: str, entry: dict[str, Any]) -> str:
    if not isinstance(current, str) or len(current) != 64:
        raise ValueError("Group ready-member digest is invalid.")
    return f"{int(current, 16) ^ int(_digest([entry]), 16):064x}"


def initialize_group_ready_members(cfg: object) -> None:
    """Create the empty canonical projection used by newly initialized roots."""
    root = _root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    shared_paths(cfg.shared_root)["ready_group_member_groups"].mkdir(parents=True, exist_ok=True)
    path = _state_path(cfg)
    if not path.exists():
        atomic_replace(path, {"group_ready_members": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "required_capability": GROUP_READY_MEMBERS_CAPABILITY,
            "state": "active", "revision": 0, "build": None,
            "degraded_reasons": [], "updated_at": utc_now(),
        }})


def read_group_ready_members_state(cfg: object) -> dict[str, Any]:
    """Read and validate the global membership gate."""
    record = read_json(_state_path(cfg))["group_ready_members"]
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("required_capability") != GROUP_READY_MEMBERS_CAPABILITY
        or record.get("state") not in {"building", "active", "degraded"}
        or type(record.get("revision")) is not int
        or record["revision"] < 0
    ):
        raise ValueError("group ready-member state is invalid.")
    return record


def group_ready_members_state(cfg: object) -> str:
    try:
        return read_group_ready_members_state(cfg)["state"]
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return "degraded"


def is_group_ready_member_projection_usable(cfg: object) -> bool:
    """Return whether the installed member protocol is safe for primary probes."""
    from ...layout import is_group_ready_members_root

    return not is_group_ready_members_root(cfg) or group_ready_members_state(cfg) == "active"


def mark_group_ready_members_degraded(cfg: object, reason: str) -> None:
    """Persist the fail-closed state before returning a projection error."""
    with exclusive(_state_lock_path(cfg)):
        try:
            value = read_json(_state_path(cfg))
            record = value["group_ready_members"]
            read_group_ready_members_state(cfg)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            return
        reasons = record.setdefault("degraded_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        record["state"] = "degraded"
        record["revision"] += 1
        record["updated_at"] = utc_now()
        atomic_replace(_state_path(cfg), value)


def assert_group_ready_members_writable(cfg: object) -> None:
    """Reject grouped authoritative mutation when an installed projection is unsafe."""
    try:
        state = read_group_ready_members_state(cfg)["state"]
    except (AttributeError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("group ready-member state is invalid; grouped mutation is disabled.") from exc
    if state not in {"building", "active"}:
        raise RuntimeError("group ready-member projection is degraded; grouped mutation is disabled.")


def _empty_group(group_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = {"group_ready_member_catalog": {
        "schema_version": GROUP_READY_MEMBERS_VERSION, "group_name": group_name,
        "page": 0, "identities": [], "revision": 0,
    }}
    partition = {"group_ready_member_partition": {
        "schema_version": GROUP_READY_MEMBERS_VERSION, "group_name": group_name,
        "page": 0, "entries": [], "revision": 0,
    }}
    state = {"group_ready_members": {
        "schema_version": GROUP_READY_MEMBERS_VERSION, "group_name": group_name,
        "membership_revision": 0, "member_count": 0, "catalog_head": 0,
        "catalog_tail": 0, "catalog_pages": [], "available_catalog_pages": [],
        "free_catalog_page": 0, "membership_digest": _digest([]),
        "digest_algorithm": _MEMBERSHIP_DIGEST_ALGORITHM, "updated_at": utc_now(),
    }}
    return state, catalog, partition


def _empty_page(group_name: str, page: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _state, catalog, partition = _empty_group(group_name)
    catalog["group_ready_member_catalog"]["page"] = page
    partition["group_ready_member_partition"]["page"] = page
    return catalog, partition


def _load_group(
    cfg: object, group_name: str, page: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_path = _group_state_path(cfg, group_name)
    if not state_path.exists():
        return _empty_group(group_name)
    state = read_json(state_path)
    catalog = read_json(_catalog_path(cfg, group_name, page))
    partition = read_json(_partition_path(cfg, group_name, page))
    return state, catalog, partition


def _entries(group_name: str, state: dict[str, Any], catalog: dict[str, Any], partition: dict[str, Any]) -> list[dict[str, Any]]:
    group = state.get("group_ready_members")
    page = catalog.get("group_ready_member_catalog")
    slots = partition.get("group_ready_member_partition")
    if not all(isinstance(item, dict) for item in (group, page, slots)):
        raise ValueError("Group ready-member records are invalid.")
    if group.get("group_name") != group_name or page.get("group_name") != group_name or slots.get("group_name") != group_name:
        raise ValueError("Group ready-member identity is invalid.")
    entries = slots.get("entries")
    identities = page.get("identities")
    if not isinstance(entries, list) or not isinstance(identities, list):
        raise ValueError("Group ready-member page is invalid.")
    if len(entries) != len(identities):
        raise ValueError("Group ready-member count is invalid.")
    if [entry.get("identity") for entry in entries] != identities:
        raise ValueError("Group ready-member digest is invalid.")
    seen: set[str] = set()
    for entry in entries:
        identity = entry.get("identity")
        if not isinstance(identity, str) or identity in seen:
            raise ValueError("Group ready-member identity is invalid.")
        seen.add(identity)
        if identity != _identity(group_name, entry.get("task_id", ""), entry.get("generation")):
            raise ValueError("Group ready-member identity does not match its reference.")
    return entries


def _read_entries(cfg: object, group_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state, _catalog, _partition = _load_group(cfg, group_name)
    group = state["group_ready_members"]
    tail = group.get("catalog_tail")
    pages = group.get("catalog_pages")
    available_pages = group.get("available_catalog_pages")
    free_page = group.get("free_catalog_page")
    if (
        type(tail) is not int
        or tail < 0
        or not isinstance(pages, list)
        or not all(type(page) is int and page >= 0 for page in pages)
        or pages != sorted(set(pages))
        or not isinstance(available_pages, list)
        or not all(type(page) is int and page in pages for page in available_pages)
        or available_pages != sorted(set(available_pages))
        or type(free_page) is not int
        or free_page < 0
        or free_page in pages
        or group.get("digest_algorithm") != _MEMBERSHIP_DIGEST_ALGORITHM
        or type(group.get("member_count")) is not int
        or group["member_count"] < 0
    ):
        raise ValueError("Group ready-member catalog tail is invalid.")
    if pages and (group.get("catalog_head") != pages[0] or tail != pages[-1]):
        raise ValueError("Group ready-member catalog range is invalid.")
    if not pages and (group.get("catalog_head") != 0 or tail != 0):
        raise ValueError("Group ready-member empty catalog range is invalid.")
    entries: list[dict[str, Any]] = []
    writable_pages: set[int] = set()
    for page in pages:
        _state, catalog, partition = _load_group(cfg, group_name, page)
        page_entries = _entries(group_name, state, catalog, partition)
        entries.extend(page_entries)
        if len(page_entries) < GROUP_MEMBER_PAGE_SIZE:
            writable_pages.add(page)
    if set(available_pages) != writable_pages:
        raise ValueError("Group ready-member writable page coverage is invalid.")
    if len(entries) != group.get("member_count") or _digest(entries) != group.get("membership_digest"):
        raise ValueError("Group ready-member count or digest is invalid.")
    return state, entries


def read_group_ready_members(cfg: object, group_name: str) -> list[dict[str, Any]]:
    """Return validated live ready references for exactly one Group."""
    try:
        state, entries = _read_entries(cfg, group_name)
        for entry in entries:
            _validate_entry_locator(cfg, group_name, state, entry)
        locators_root = _group_root(cfg, group_name) / "locators"
        locator_identities: set[str] = set()
        if locators_root.exists():
            for path in locators_root.iterdir():
                if not path.is_file() or path.suffix != ".json":
                    raise ValueError("Group ready-member locator directory is invalid.")
                locator_identities.add(path.stem)
        if locator_identities != {entry["identity"] for entry in entries}:
            raise ValueError("Group ready-member locator coverage is invalid.")
        return entries
    except (AttributeError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"member_invalid:{group_name}:{type(exc).__name__}")
        raise RuntimeError(f"Group {group_name!r} ready-member projection is invalid.") from exc


def _validate_entry_locator(
    cfg: object,
    group_name: str,
    state: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Return the exact page entry identified by one validated locator."""
    locator = read_json(_locator_path(cfg, group_name, entry["identity"]))[
        "group_ready_member_locator"
    ]
    if not isinstance(locator, dict):
        raise ValueError("Group ready-member locator is invalid.")
    page = locator.get("page")
    slot = locator.get("slot")
    group = state["group_ready_members"]
    pages = group.get("catalog_pages")
    if (
        locator.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or locator.get("identity") != entry["identity"]
        or locator.get("group_name") != group_name
        or locator.get("member_revision") != entry["member_revision"]
        or type(page) is not int
        or page < 0
        or type(slot) is not int
        or slot < 0
        or not isinstance(pages, list)
        or page not in pages
    ):
        raise ValueError("Group ready-member locator is invalid.")
    _state, catalog, partition = _load_group(cfg, group_name, page)
    page_entries = _entries(group_name, state, catalog, partition)
    if slot >= len(page_entries):
        raise ValueError("Group ready-member locator slot is invalid.")
    located = page_entries[slot]
    if (
        located.get("identity") != entry["identity"]
        or located.get("member_revision") != entry["member_revision"]
    ):
        raise ValueError("Group ready-member locator does not point to its entry.")
    return located, page


def _write_group(
    cfg: object, group_name: str, state: dict[str, Any], catalog: dict[str, Any], partition: dict[str, Any],
) -> None:
    page = catalog["group_ready_member_catalog"]["page"]
    atomic_replace(_partition_path(cfg, group_name, page), partition)
    atomic_replace(_catalog_path(cfg, group_name, page), catalog)
    atomic_replace(_group_state_path(cfg, group_name), state)
    # This lock only covers the global read-modify-write.  It intentionally is
    # not held while acquiring a Group lock, preserving schema -> Group -> Task.
    with exclusive(_state_lock_path(cfg)):
        value = read_json(_state_path(cfg))
        record = value["group_ready_members"]
        if record.get("state") not in {"building", "active"}:
            raise RuntimeError("group ready-member projection is degraded; publication is disabled.")
        record["revision"] += 1
        record["updated_at"] = utc_now()
        atomic_replace(_state_path(cfg), value)


def _rewrite_page_locators(
    cfg: object,
    group_name: str,
    page: int,
    entries: list[dict[str, Any]],
) -> None:
    """Keep bounded-page locator slots exact after an entry is removed."""
    for slot, entry in enumerate(entries):
        atomic_replace(_locator_path(cfg, group_name, entry["identity"]), {
            "group_ready_member_locator": {
                "schema_version": GROUP_READY_MEMBERS_VERSION,
                "identity": entry["identity"],
                "group_name": group_name,
                "page": page,
                "slot": slot,
                "member_revision": entry["member_revision"],
            }
        })


def publish_group_ready_member(cfg: object, task: TaskRecord, reference: ReadyMarkerRef) -> None:
    """Publish one grouped ready generation after its marker is durable."""
    if not task.group_name:
        return
    from ...layout import is_group_ready_members_root

    if not is_group_ready_members_root(cfg):
        return
    group_name = task.group_name
    identity = _identity(group_name, task.task_id, reference.generation)
    try:
        assert_group_ready_members_writable(cfg)
        state, _catalog, _partition = _load_group(cfg, group_name)
        record = state["group_ready_members"]
        if (
            record.get("digest_algorithm") != _MEMBERSHIP_DIGEST_ALGORITHM
            or type(record.get("member_count")) is not int
            or record["member_count"] < 0
        ):
            raise ValueError("Group ready-member state is invalid.")
        locator_path = _locator_path(cfg, group_name, identity)
        if locator_path.exists():
            locator = read_json(locator_path)["group_ready_member_locator"]
            if not isinstance(locator, dict):
                raise ValueError("Group ready-member locator is invalid.")
            located, _page = _validate_entry_locator(
                cfg, group_name, state,
                {"identity": identity, "member_revision": locator.get("member_revision")},
            )
            if located.get("task_id") != task.task_id or located.get("generation") != reference.generation:
                raise ValueError("Group ready-member identity conflicts with an existing entry.")
            return
        pages = record.get("catalog_pages")
        available_pages = record.get("available_catalog_pages")
        free_page = record.get("free_catalog_page")
        if (
            not isinstance(pages, list)
            or not isinstance(available_pages, list)
            or type(free_page) is not int
            or free_page < 0
        ):
            raise ValueError("Group ready-member catalog tail is invalid.")
        if available_pages:
            page = available_pages[0]
            _state, catalog, partition = _load_group(cfg, group_name, page)
            page_entries = _entries(group_name, state, catalog, partition)
        else:
            if pages:
                page = free_page if free_page not in pages else record["catalog_tail"] + 1
            else:
                page = free_page
            catalog, partition = _empty_page(group_name, page)
            page_entries = []
            pages.append(page)
            pages.sort()
            record["catalog_head"] = pages[0]
            record["catalog_tail"] = pages[-1]
            record["free_catalog_page"] = record["catalog_tail"] + 1
        entry = {
            "identity": identity, "task_id": task.task_id, "generation": reference.generation,
            "queue_scope": reference.queue_scope, "home_machine": reference.home_machine,
            "partition": reference.partition, "catalog_page": reference.catalog_page,
            "marker_name": reference.marker_name, "lane": task.spec.lane or "gpu",
            "submission_operation_id": task.submission_operation_id,
            "source_revision": task.meta["revision"], "target_revision": task.meta["revision"] + 1,
            "member_revision": 1,
        }
        page_entries.append(entry)
        catalog["group_ready_member_catalog"]["identities"] = [item["identity"] for item in page_entries]
        catalog["group_ready_member_catalog"]["revision"] += 1
        partition["group_ready_member_partition"]["entries"] = page_entries
        partition["group_ready_member_partition"]["revision"] += 1
        if len(page_entries) == GROUP_MEMBER_PAGE_SIZE:
            available_pages.remove(page)
        elif page not in available_pages:
            available_pages.append(page)
            available_pages.sort()
        record["member_count"] += 1
        record["membership_digest"] = _update_digest(record["membership_digest"], entry)
        record["membership_revision"] += 1
        record["updated_at"] = utc_now()
        atomic_replace(locator_path, {"group_ready_member_locator": {
            "schema_version": GROUP_READY_MEMBERS_VERSION, "identity": identity,
            "group_name": group_name, "page": page, "slot": len(page_entries) - 1,
            "member_revision": entry["member_revision"],
        }})
        _write_group(cfg, group_name, state, catalog, partition)
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(
            cfg, f"member_publish_failed:{group_name}:{type(exc).__name__}"
        )
        raise RuntimeError(
            f"Group {group_name!r} ready-member publication failed; grouped mutation is disabled."
        ) from exc


def retire_group_ready_member(cfg: object, group_name: str, task_id: str, generation: int) -> bool:
    """Remove only an exact live generation, preserving slot-reuse safety."""
    if group_ready_members_state(cfg) not in {"building", "active"}:
        return False
    identity = _identity(group_name, task_id, generation)
    locator_path = _locator_path(cfg, group_name, identity)
    if not locator_path.exists():
        return False
    try:
        state, _catalog, _partition = _load_group(cfg, group_name)
        record = state["group_ready_members"]
        if (
            record.get("digest_algorithm") != _MEMBERSHIP_DIGEST_ALGORITHM
            or type(record.get("member_count")) is not int
            or record["member_count"] <= 0
        ):
            raise ValueError("Group ready-member state is invalid.")
        locator = read_json(locator_path)["group_ready_member_locator"]
        if not isinstance(locator, dict):
            raise ValueError("Group ready-member locator is invalid.")
        located, page = _validate_entry_locator(
            cfg, group_name, state,
            {"identity": identity, "member_revision": locator.get("member_revision")},
        )
        if located.get("task_id") != task_id or located.get("generation") != generation:
            return False
        _state, catalog, partition = _load_group(cfg, group_name, page)
        page_entries = _entries(group_name, state, catalog, partition)
        page_entries.remove(located)
        catalog["group_ready_member_catalog"]["identities"] = [
            item["identity"] for item in page_entries
        ]
        catalog["group_ready_member_catalog"]["revision"] += 1
        partition["group_ready_member_partition"]["entries"] = page_entries
        partition["group_ready_member_partition"]["revision"] += 1
        pages = record.get("catalog_pages")
        available_pages = record.get("available_catalog_pages")
        free_page = record.get("free_catalog_page")
        if (
            not isinstance(pages, list)
            or page not in pages
            or not isinstance(available_pages, list)
            or type(free_page) is not int
            or free_page < 0
        ):
            raise ValueError("Group ready-member page state is invalid.")
        discarded_free_page: int | None = None
        if page_entries:
            if page not in available_pages:
                available_pages.append(page)
                available_pages.sort()
        else:
            pages.remove(page)
            if page in available_pages:
                available_pages.remove(page)
            if pages:
                record["catalog_head"] = pages[0]
                record["catalog_tail"] = pages[-1]
                if free_page in pages:
                    free_page = page
                elif page < free_page:
                    discarded_free_page = free_page
                    free_page = page
                else:
                    discarded_free_page = page
            else:
                record["catalog_head"] = 0
                record["catalog_tail"] = 0
                if page != 0:
                    discarded_free_page = page
                free_page = 0
            record["free_catalog_page"] = free_page
        record["member_count"] -= 1
        record["membership_digest"] = _update_digest(record["membership_digest"], located)
        record["membership_revision"] += 1
        record["updated_at"] = utc_now()
        _rewrite_page_locators(cfg, group_name, page, page_entries)
        _write_group(cfg, group_name, state, catalog, partition)
        if discarded_free_page is not None:
            _catalog_path(cfg, group_name, discarded_free_page).unlink(missing_ok=True)
            _partition_path(cfg, group_name, discarded_free_page).unlink(missing_ok=True)
        locator_path.unlink(missing_ok=True)
        return True
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"member_retire_invalid:{group_name}:{type(exc).__name__}")
        raise RuntimeError(
            f"Group {group_name!r} ready-member projection is invalid; retirement is disabled."
        ) from exc


def begin_group_ready_members_build(cfg: object, *, is_repair: bool = False) -> dict[str, Any]:
    """Start one durable, resumable backfill for legacy roots."""
    root = _root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    shared_paths(cfg.shared_root)["ready_group_member_groups"].mkdir(parents=True, exist_ok=True)
    path = _state_path(cfg)
    with exclusive(_state_lock_path(cfg)):
        if path.exists():
            record = read_group_ready_members_state(cfg)
            if record["state"] == "active" and not is_repair:
                return record
            if record["state"] in {"building", "degraded"} and not is_repair:
                return record
        if is_repair:
            groups = shared_paths(cfg.shared_root)["ready_group_member_groups"]
            if groups.exists():
                shutil.rmtree(groups)
            groups.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "required_capability": GROUP_READY_MEMBERS_CAPABILITY,
            "state": "building", "revision": 0,
            "build": {
                "build_id": uuid.uuid4().hex, "cursor": {"page": 0, "offset": 0},
                "processed": 0, "phase": "backfill", "watermark": {"is_complete": False},
                "started_at": utc_now(),
            },
            "degraded_reasons": [], "updated_at": utc_now(),
        }
        atomic_replace(path, {"group_ready_members": record})
        return record


def _should_index(task: TaskRecord) -> bool:
    return bool(task.group_name and task.ready_generation > 0 and task.state["projection"] == "queued")


def audit_group_ready_members(cfg: object) -> dict[str, Any]:
    """Verify member entries against authoritative queued Task and ready-marker truth."""
    from .index import _reference_for_generation
    from ..records import TaskRecord
    from ..store import iter_json

    state = read_group_ready_members_state(cfg)
    if state["state"] not in {"building", "active"}:
        raise ValueError("Group ready-member projection is not writable.")
    expected: dict[str, dict[str, tuple[TaskRecord, ReadyMarkerRef]]] = {}
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if not _should_index(task):
            continue
        reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is None:
            raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
        identity = _identity(task.group_name or "", task.task_id, task.ready_generation)
        expected.setdefault(task.group_name or "", {})[identity] = (task, reference)

    actual: dict[str, dict[str, dict[str, Any]]] = {}
    groups_root = shared_paths(cfg.shared_root)["ready_group_member_groups"]
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


def _assert_member_matches_task(
    cfg: object, task: TaskRecord, reference: ReadyMarkerRef,
) -> None:
    """Check one exact member locator against its authoritative Task and marker."""
    if not task.group_name:
        raise ValueError("Ungrouped Task cannot have a ready-member entry.")
    group_name = task.group_name
    identity = _identity(group_name, task.task_id, task.ready_generation)
    locator_path = _locator_path(cfg, group_name, identity)
    if not locator_path.exists():
        raise ValueError(f"Group ready-member is missing for Task {task.task_id!r}.")
    state, _catalog, _partition = _load_group(cfg, group_name)
    locator = read_json(locator_path)["group_ready_member_locator"]
    if not isinstance(locator, dict):
        raise ValueError("Group ready-member locator is invalid.")
    entry, _page = _validate_entry_locator(
        cfg, group_name, state,
        {"identity": identity, "member_revision": locator.get("member_revision")},
    )
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


def _audit_task_member(
    cfg: object, task_id: str,
) -> tuple[TaskRecord | None, ReadyMarkerRef | None]:
    """Audit one captured Task; a cleaned Task is a legal watermark tombstone."""
    try:
        task = TaskRecord.from_dict(
            read_json(shared_paths(cfg.shared_root)["tasks"] / f"{task_id}.json")
        )
    except FileNotFoundError:
        return None, None
    if not task.group_name or task.ready_generation <= 0:
        return task, None
    identity = _identity(task.group_name, task.task_id, task.ready_generation)
    if not _should_index(task):
        if _locator_path(cfg, task.group_name, identity).exists():
            raise ValueError(f"Group ready-member is stale for Task {task.task_id!r}.")
        return task, None
    from .index import _reference_for_generation

    reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
    if reference is None:
        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
    _assert_member_matches_task(cfg, task, reference)
    return task, reference


def _audit_member_entry(cfg: object, group_name: str, entry: dict[str, Any]) -> None:
    """Reject an actual member that no longer has matching authoritative truth."""
    task_id = entry.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("Group ready-member Task identity is invalid.")
    task = TaskRecord.from_dict(
        read_json(shared_paths(cfg.shared_root)["tasks"] / f"{task_id}.json")
    )
    if task.group_name != group_name or not _should_index(task):
        raise ValueError(f"Group {group_name!r} has a stale ready-member entry.")
    from .index import _reference_for_generation

    reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
    if reference is None:
        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
    _assert_member_matches_task(cfg, task, reference)


def _build_root(cfg: object, build_id: str) -> Path:
    validate_identifier(build_id, "group ready-member build id")
    return _root(cfg) / "builds" / build_id


def _build_page_path(
    cfg: object, build_id: str, page: int, *, collection: str = "watermark",
) -> Path:
    return _build_root(cfg, build_id) / collection / f"{page:016d}.json"


def _write_build_page(
    cfg: object, build_id: str, page: int, task_ids: list[str], *, collection: str = "watermark",
) -> None:
    atomic_replace(_build_page_path(cfg, build_id, page, collection=collection), {
        "group_ready_member_build_page": {
        "schema_version": GROUP_READY_MEMBERS_VERSION, "build_id": build_id,
        "page": page, "task_ids": list(task_ids),
        }
    })


def _load_build_page(
    cfg: object, build_id: str, page: int, *, collection: str = "watermark",
) -> list[str]:
    record = read_json(
        _build_page_path(cfg, build_id, page, collection=collection)
    )["group_ready_member_build_page"]
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
                    cfg, build_id, capture["page"], capture["pending_task_ids"],
                    collection=collection,
                )
                capture["page"] += 1
                capture["pending_task_ids"] = []
    finally:
        _LIBC.closedir(directory_handle)
    if is_complete:
        if capture["pending_task_ids"]:
            _write_build_page(
                cfg, build_id, capture["page"], capture["pending_task_ids"],
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
    with exclusive(_state_lock_path(cfg)):
        value = read_json(_state_path(cfg))
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
        atomic_replace(_state_path(cfg), value)
        return record


def _advance_page_cursor(cursor: dict[str, int], item_count: int) -> None:
    cursor["offset"] += 1
    if cursor["offset"] >= item_count:
        cursor["page"] += 1
        cursor["offset"] = 0


def _advance_member_audit(
    cfg: object, build_id: str, cursor: dict[str, Any], max_entries: int,
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
        cursor.update({
            "pages": pages,
            "page_index": 0,
            "entry_offset": 0,
            "seen_count": 0,
            "seen_digest": _digest([]),
            "membership_revision": revision,
        })

    processed = 0
    while processed < max_entries:
        group_name = cursor.get("group_name")
        if group_name is None:
            group_page = cursor.get("group_page", 0)
            group_offset = cursor.get("group_offset", 0)
            group_page_count = cursor.get("group_page_count")
            if (
                type(group_page) is not int or type(group_offset) is not int
                or type(group_page_count) is not int or group_page < 0 or group_offset < 0
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
            state = read_json(
                shared_paths(cfg.shared_root)["ready_group_member_groups"]
                / group_key / "state.json"
            )["group_ready_members"]
            pages = state.get("catalog_pages")
            if (
                not isinstance(state.get("group_name"), str)
                or not isinstance(pages, list)
                or not all(type(page) is int and page >= 0 for page in pages)
                or pages != sorted(set(pages))
                or type(state.get("membership_revision")) is not int
                or state["membership_revision"] < 0
            ):
                raise ValueError("Group ready-member audit state is invalid.")
            cursor.update({"group_name": state["group_name"], "pages": pages, "page_index": 0,
                           "entry_offset": 0, "seen_count": 0, "seen_digest": _digest([]),
                           "membership_revision": state["membership_revision"]})
            processed += 1
            continue
        pages = cursor.get("pages")
        page_index = cursor.get("page_index")
        if not isinstance(pages, list) or type(page_index) is not int:
            raise ValueError("Group ready-member audit cursor is invalid.")
        if page_index >= len(pages):
            state, _catalog, _partition = _load_group(cfg, group_name)
            group = state["group_ready_members"]
            if cursor.get("membership_revision") != group.get("membership_revision"):
                reset_group_audit(state)
                continue
            if (
                cursor.get("seen_count") != group.get("member_count")
                or cursor.get("seen_digest") != group.get("membership_digest")
            ):
                raise ValueError(f"Group {group_name!r} ready-member count or digest is invalid.")
            cursor.update({"group_name": None, "pages": [], "page_index": 0,
                           "entry_offset": 0, "seen_count": 0, "seen_digest": _digest([])})
            continue
        state, catalog, partition = _load_group(cfg, group_name, pages[page_index])
        entries = _entries(group_name, state, catalog, partition)
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
            state, _catalog, _partition = _load_group(cfg, group_name)
            if cursor.get("membership_revision") != state["group_ready_members"].get(
                "membership_revision"
            ):
                reset_group_audit(state)
                continue
            raise
        cursor["seen_digest"] = _update_digest(cursor["seen_digest"], entry)
        cursor["seen_count"] += 1
        cursor["entry_offset"] += 1
        processed += 1
    return cursor, processed, False


def advance_group_ready_members_build(
    cfg: object, *, max_tasks: int = GROUP_MEMBER_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance one bounded backfill, audit, or candidate-rebuild slice."""
    if type(max_tasks) is not int or not 1 <= max_tasks <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_tasks must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    record = begin_group_ready_members_build(cfg)
    if record["state"] != "building":
        return record
    from .index import (
        _reference_for_generation, _task_should_have_ready_marker,
        begin_primary_ready_index_rebuild, complete_primary_ready_index_rebuild,
        rebuild_primary_ready_candidate,
    )
    from ..records import TaskRecord
    try:
        with exclusive(_state_lock_path(cfg)):
            value = read_json(_state_path(cfg))
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
                cfg, build_id=build_id, watermark=watermark, max_entries=max_tasks,
            )
            return _commit_build_record(
                cfg, build_id=build_id, build_updates={"watermark": captured},
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
                    task = TaskRecord.from_dict(read_json(
                        shared_paths(cfg.shared_root)["tasks"] / f"{items[cursor['offset']]}.json"
                    ))
                except FileNotFoundError:
                    task = None
                if task is not None and _should_index(task):
                    reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
                    if reference is None:
                        raise ValueError(f"Group ready-member marker is missing for Task {task.task_id!r}.")
                    publish_group_ready_member(cfg, task, reference)
                processed += 1
                _advance_page_cursor(cursor, len(items))
            updates: dict[str, Any] = {}
            if cursor["page"] >= page_count:
                updates = {"phase": "audit-task-capture"}
            return _commit_build_record(
                cfg, build_id=build_id, cursor=cursor, processed=processed, build_updates=updates,
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
            captured = _capture_build_watermark(cfg, build_id=build_id, watermark=current,
                                                max_entries=max_tasks, collection="audit-tasks")
            updates = {"audit_task_watermark": captured}
            if captured.get("is_complete"):
                updates.update({
                    "phase": "audit-tasks", "audit_task_cursor": {"page": 0, "offset": 0},
                })
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
                cfg, build_id=build_id, watermark=current, max_entries=max_tasks,
                source_path=shared_paths(cfg.shared_root)["ready_group_member_groups"],
                collection="audit-groups", should_include_json=False,
            )
            updates = {"audit_group_watermark": captured}
            if captured.get("is_complete"):
                updates.update({"phase": "audit-members", "audit_member_cursor": {
                    "group_page": 0, "group_offset": 0, "group_page_count": captured["page_count"],
                    "group_name": None, "pages": [], "page_index": 0, "entry_offset": 0,
                    "seen_count": 0, "seen_digest": _digest([]),
                }})
            return _commit_build_record(cfg, build_id=build_id, build_updates=updates)
        if phase == "audit-members":
            cursor, _processed, complete = _advance_member_audit(
                cfg, build_id, dict(build.get("audit_member_cursor", {})), max_tasks,
            )
            updates = {"audit_member_cursor": cursor}
            if complete:
                updates.update({
                    "phase": "primary-rebuild", "primary_cursor": {"page": 0, "offset": 0},
                })
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
                    task = TaskRecord.from_dict(read_json(
                        shared_paths(cfg.shared_root)["tasks"] / f"{items[cursor['offset']]}.json"
                    ))
                except FileNotFoundError:
                    task = None
                if task is not None and _task_should_have_ready_marker(task):
                    reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
                    if reference is None:
                        raise ValueError(f"Primary candidate marker is missing for Task {task.task_id!r}.")
                    rebuild_primary_ready_candidate(cfg, build_id, task, reference)
                processed += 1
                _advance_page_cursor(cursor, len(items))
            updates = {"primary_cursor": cursor}
            if cursor["page"] >= page_count:
                complete_primary_ready_index_rebuild(cfg, build_id)
                return _commit_build_record(
                    cfg, build_id=build_id, build_updates=updates, should_activate=True,
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
