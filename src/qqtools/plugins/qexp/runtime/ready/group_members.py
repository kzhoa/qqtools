"""Derived live-ready membership indexed by Group."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..locks import exclusive
from ..paths import shared_paths
from ..records import TaskRecord, utc_now, validate_identifier
from ..store import atomic_replace, read_json, read_json_limited, require_json_size

if TYPE_CHECKING:
    from ..records import TaskRecord
    from .records import ReadyMarkerRef

GROUP_READY_MEMBERS_CAPABILITY = "group-ready-members-v1"
GROUP_READY_MEMBERS_VERSION = 1
GROUP_MEMBER_PAGE_SIZE = 64
_MEMBERSHIP_DIGEST_ALGORITHM = "xor-sha256-v1"
_MAX_PROJECTION_IDENTIFIER_BYTES = 256
_MAX_GROUP_HEADER_BYTES = 8 * 1024
_MAX_DIRECTORY_PAGE_BYTES = 8 * 1024
_MAX_MEMBER_PAGE_BYTES = 64 * 1024
_MAX_LOCATOR_BYTES = 4 * 1024
_MAX_GLOBAL_STATE_BYTES = 64 * 1024
_MAX_DEGRADED_REASONS = 16
_MAX_DEGRADED_REASON_BYTES = 512


def _root(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_group_members"]


def _state_path(cfg: object) -> Path:
    return _root(cfg) / "state.json"


def _state_lock_path(cfg: object) -> Path:
    """Return the short global fence for the member-projection state."""
    return shared_paths(cfg.shared_root)["ready_locks"] / "group-members-state.lock"


def _write_global_state(cfg: object, value: dict[str, Any]) -> None:
    """Persist the global projection state only within its encoding budget."""
    require_json_size(value, max_bytes=_MAX_GLOBAL_STATE_BYTES, record_type="member_global_state")
    atomic_replace(_state_path(cfg), value)


def _group_key(group_name: str) -> str:
    validate_identifier(group_name, "group_name")
    return hashlib.sha256(group_name.encode()).hexdigest()


def _group_root(cfg: object, group_name: str) -> Path:
    return shared_paths(cfg.shared_root)["ready_group_member_groups"] / _group_key(group_name)


def _group_state_path(cfg: object, group_name: str) -> Path:
    return _group_root(cfg, group_name) / "state.json"


def _catalog_path(cfg: object, group_name: str, page: int = 0) -> Path:
    return _group_root(cfg, group_name) / "catalog" / f"{page:016d}.json"


def _directory_path(cfg: object, group_name: str, page: int = 0) -> Path:
    return _group_root(cfg, group_name) / "directories" / f"{page:016d}.json"


def _writable_index_path(cfg: object, group_name: str, page: int) -> Path:
    return _group_root(cfg, group_name) / "writable-pages" / f"{page:016d}.json"


def _partition_path(cfg: object, group_name: str, page: int = 0) -> Path:
    return _group_root(cfg, group_name) / "partitions" / f"{page:016d}.json"


def _validate_projection_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_PROJECTION_IDENTIFIER_BYTES:
        raise ValueError(f"projection_encoding_unsupported:{label}")
    return value


def _identity(group_name: str, task_id: str, generation: int) -> str:
    return hashlib.sha256(f"{group_name}\0{task_id}\0{generation}".encode()).hexdigest()


def _locator_path(cfg: object, group_name: str, identity: str) -> Path:
    return _group_root(cfg, group_name) / "locators" / f"{identity}.json"


def _digest(entries: list[dict[str, Any]]) -> str:
    """Return a reversible set digest so exact retirement stays page-bounded."""
    digest = 0
    for item in entries:
        value = (item["identity"], item["task_id"], item["generation"], item["member_revision"])
        digest ^= int.from_bytes(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).digest(), "big")
    return f"{digest:064x}"


def _update_digest(current: str, entry: dict[str, Any]) -> str:
    if not isinstance(current, str) or len(current) != 64:
        raise ValueError("Group ready-member digest is invalid.")
    return f"{int(current, 16) ^ int(_digest([entry]), 16):064x}"


# Explicit package-internal collaboration surface for bounded rebuild audits.
group_ready_members_root = _root
group_ready_members_state_path = _state_path
group_ready_members_state_lock_path = _state_lock_path
group_ready_member_identity = _identity
group_ready_member_locator_path = _locator_path
group_ready_member_digest = _digest
update_group_ready_member_digest = _update_digest


def group_ready_member_groups_root(cfg: object) -> Path:
    """Return the online Group-member storage root for bounded rebuild capture."""
    return shared_paths(cfg.shared_root)["ready_group_member_groups"]


def initialize_group_ready_members(cfg: object) -> None:
    """Create the empty canonical projection used by newly initialized roots."""
    root = _root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    shared_paths(cfg.shared_root)["ready_group_member_groups"].mkdir(parents=True, exist_ok=True)
    path = _state_path(cfg)
    if not path.exists():
        value = {
            "group_ready_members": {
                "schema_version": GROUP_READY_MEMBERS_VERSION,
                "required_capability": GROUP_READY_MEMBERS_CAPABILITY,
                "state": "active",
                "revision": 0,
                "projection_id": hashlib.sha256(f"{utc_now()}:{id(cfg)}".encode()).hexdigest()[:32],
                "build": None,
                "audit": None,
                "archive_count": 0,
                "archive_cleanup": None,
                "audit_cleanup": None,
                "degraded_reasons": [],
                "updated_at": utc_now(),
            }
        }
        _write_global_state(cfg, value)


def read_group_ready_members_state(cfg: object) -> dict[str, Any]:
    """Read and validate the global membership gate."""
    record = read_json_limited(_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)["group_ready_members"]
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("required_capability") != GROUP_READY_MEMBERS_CAPABILITY
        or record.get("state") not in {"building", "active", "degraded"}
        or type(record.get("revision")) is not int
        or record["revision"] < 0
        or not isinstance(record.get("projection_id"), str)
        or len(record["projection_id"]) != 32
        or type(record.get("archive_count")) is not int
        or record["archive_count"] < 0
        or not isinstance(record.get("degraded_reasons"), list)
        or len(record["degraded_reasons"]) > _MAX_DEGRADED_REASONS
        or not all(
            isinstance(reason, str) and len(reason.encode("utf-8")) <= _MAX_DEGRADED_REASON_BYTES
            for reason in record["degraded_reasons"]
        )
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
            value = read_json_limited(_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
            record = value["group_ready_members"]
            read_group_ready_members_state(cfg)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            return
        if not isinstance(reason, str):
            reason = type(reason).__name__
        encoded_reason = reason.encode("utf-8")
        if len(encoded_reason) > _MAX_DEGRADED_REASON_BYTES:
            digest = hashlib.sha256(encoded_reason).hexdigest()[:16]
            prefix = encoded_reason[: _MAX_DEGRADED_REASON_BYTES - len(digest) - 1]
            reason = f"{prefix.decode('utf-8', errors='ignore')}:{digest}"
        reasons = record.setdefault("degraded_reasons", [])
        if reason not in reasons:
            if len(reasons) >= _MAX_DEGRADED_REASONS:
                reasons.pop(0)
            reasons.append(reason)
        record["state"] = "degraded"
        record["revision"] += 1
        record["updated_at"] = utc_now()
        _write_global_state(cfg, value)


def assert_group_ready_members_writable(cfg: object) -> None:
    """Reject grouped authoritative mutation when an installed projection is unsafe."""
    try:
        state = read_group_ready_members_state(cfg)["state"]
    except (AttributeError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("group ready-member state is invalid; grouped mutation is disabled.") from exc
    if state not in {"building", "active"}:
        raise RuntimeError("group ready-member projection is degraded; grouped mutation is disabled.")


def _empty_group(group_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = {
        "group_ready_member_catalog": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "group_name": group_name,
            "page": 0,
            "identities": [],
            "revision": 0,
            "writable_indexed": False,
        }
    }
    partition = {
        "group_ready_member_partition": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "group_name": group_name,
            "page": 0,
            "entries": [],
            "revision": 0,
        }
    }
    state = {
        "group_ready_members": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "group_name": group_name,
            "membership_revision": 0,
            "member_count": 0,
            "directory_head": None,
            "directory_tail": None,
            "directory_page_count": 0,
            "next_directory_page": 0,
            "next_member_page": 0,
            "writable_member_page": None,
            "writable_index_head": None,
            "writable_index_tail": None,
            "writable_index_free_head": None,
            "next_writable_index_page": 0,
            "writable_index_count": 0,
            "membership_digest": _digest([]),
            "digest_algorithm": _MEMBERSHIP_DIGEST_ALGORITHM,
            "updated_at": utc_now(),
        }
    }
    return state, catalog, partition


def _empty_page(group_name: str, page: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _state, catalog, partition = _empty_group(group_name)
    catalog["group_ready_member_catalog"]["page"] = page
    partition["group_ready_member_partition"]["page"] = page
    return catalog, partition


def _directory_record(group_name: str, page: int, member_pages: list[int], next_page: int | None) -> dict[str, Any]:
    value = {
        "group_ready_member_directory": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "group_name": group_name,
            "page": page,
            "member_pages": member_pages,
            "next_page": next_page,
        }
    }
    require_json_size(value, max_bytes=_MAX_DIRECTORY_PAGE_BYTES, record_type="member_directory")
    return value


def _writable_index_record(
    group_name: str, page: int, member_pages: list[int], next_page: int | None
) -> dict[str, Any]:
    value = {
        "group_ready_member_writable_index": {
            "schema_version": GROUP_READY_MEMBERS_VERSION,
            "group_name": group_name,
            "page": page,
            "member_pages": member_pages,
            "next_page": next_page,
        }
    }
    require_json_size(value, max_bytes=_MAX_DIRECTORY_PAGE_BYTES, record_type="member_writable_index")
    return value


def _read_writable_index(cfg: object, group_name: str, page: int) -> dict[str, Any]:
    value = read_json_limited(_writable_index_path(cfg, group_name, page), max_bytes=_MAX_DIRECTORY_PAGE_BYTES)
    record = value.get("group_ready_member_writable_index")
    pages = record.get("member_pages") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("group_name") != group_name
        or record.get("page") != page
        or not isinstance(pages, list)
        or len(pages) > GROUP_MEMBER_PAGE_SIZE
        or not all(type(item) is int and item >= 0 for item in pages)
        or type(record.get("next_page")) not in {int, type(None)}
    ):
        raise ValueError("Group ready-member writable index page is invalid.")
    return record


def _acquire_writable_index_page(cfg: object, group_name: str, group: dict[str, Any]) -> int:
    """Take one recycled writable-index page or allocate one at historical peak."""
    free = group.get("writable_index_free_head")
    if free is None:
        page = group.get("next_writable_index_page")
        if type(page) is not int or page < 0:
            raise ValueError("Group ready-member writable index state is invalid.")
        group["next_writable_index_page"] = page + 1
        return page
    if type(free) is not int or free < 0:
        raise ValueError("Group ready-member writable index free list is invalid.")
    record = _read_writable_index(cfg, group_name, free)
    group["writable_index_free_head"] = record["next_page"]
    return free


def _enqueue_writable_member_page(cfg: object, group_name: str, group: dict[str, Any], member_page: int) -> None:
    """Append one non-current member page to the bounded reusable-page queue."""
    tail = group.get("writable_index_tail")
    if tail is None:
        index_page = _acquire_writable_index_page(cfg, group_name, group)
        atomic_replace(
            _writable_index_path(cfg, group_name, index_page),
            _writable_index_record(group_name, index_page, [member_page], None),
        )
        group["writable_index_head"] = index_page
        group["writable_index_tail"] = index_page
    else:
        if type(tail) is not int or tail < 0:
            raise ValueError("Group ready-member writable index tail is invalid.")
        record = _read_writable_index(cfg, group_name, tail)
        pages = list(record["member_pages"])
        if len(pages) < GROUP_MEMBER_PAGE_SIZE:
            pages.append(member_page)
            atomic_replace(
                _writable_index_path(cfg, group_name, tail),
                _writable_index_record(group_name, tail, pages, record["next_page"]),
            )
        else:
            index_page = _acquire_writable_index_page(cfg, group_name, group)
            atomic_replace(
                _writable_index_path(cfg, group_name, index_page),
                _writable_index_record(group_name, index_page, [member_page], None),
            )
            atomic_replace(
                _writable_index_path(cfg, group_name, tail),
                _writable_index_record(group_name, tail, pages, index_page),
            )
            group["writable_index_tail"] = index_page
    group["writable_index_count"] += 1


def _dequeue_writable_member_page(cfg: object, group_name: str, group: dict[str, Any]) -> int | None:
    """Take one member page from the reusable-page queue without scanning history."""
    head = group.get("writable_index_head")
    if head is None:
        return None
    if type(head) is not int or head < 0:
        raise ValueError("Group ready-member writable index head is invalid.")
    record = _read_writable_index(cfg, group_name, head)
    pages = list(record["member_pages"])
    if not pages:
        raise ValueError("Group ready-member writable index queue is invalid.")
    member_page = pages.pop(0)
    if pages:
        atomic_replace(
            _writable_index_path(cfg, group_name, head),
            _writable_index_record(group_name, head, pages, record["next_page"]),
        )
    else:
        next_head = record["next_page"]
        group["writable_index_head"] = next_head
        if group.get("writable_index_tail") == head:
            group["writable_index_tail"] = None
        atomic_replace(
            _writable_index_path(cfg, group_name, head),
            _writable_index_record(group_name, head, [], group.get("writable_index_free_head")),
        )
        group["writable_index_free_head"] = head
    group["writable_index_count"] -= 1
    return member_page


def _read_directory(cfg: object, group_name: str, page: int) -> dict[str, Any]:
    value = read_json_limited(_directory_path(cfg, group_name, page), max_bytes=_MAX_DIRECTORY_PAGE_BYTES)
    record = value.get("group_ready_member_directory")
    pages = record.get("member_pages") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or record.get("group_name") != group_name
        or record.get("page") != page
        or not isinstance(pages, list)
        or not 0 < len(pages) <= GROUP_MEMBER_PAGE_SIZE
        or not all(type(item) is int and item >= 0 for item in pages)
        or type(record.get("next_page")) not in {int, type(None)}
    ):
        raise ValueError("Group ready-member directory page is invalid.")
    return record


def _iter_member_pages(cfg: object, group_name: str, group: dict[str, Any]):
    """Yield member pages through fixed-size linked directory records."""
    page = group.get("directory_head")
    seen: set[int] = set()
    while page is not None:
        if type(page) is not int or page < 0 or page in seen:
            raise ValueError("Group ready-member directory chain is invalid.")
        seen.add(page)
        directory = _read_directory(cfg, group_name, page)
        yield from directory["member_pages"]
        page = directory["next_page"]


def _append_member_page(cfg: object, group_name: str, group: dict[str, Any], member_page: int) -> None:
    """Append one member-page reference without materializing historical page IDs."""
    tail = group["directory_tail"]
    if tail is None:
        directory_page = group["next_directory_page"]
        atomic_replace(_directory_path(cfg, group_name, directory_page), _directory_record(group_name, directory_page, [member_page], None))
        group.update({"directory_head": directory_page, "directory_tail": directory_page, "directory_page_count": 1, "next_directory_page": directory_page + 1})
        return
    directory = _read_directory(cfg, group_name, tail)
    pages = list(directory["member_pages"])
    if len(pages) < GROUP_MEMBER_PAGE_SIZE:
        pages.append(member_page)
        atomic_replace(_directory_path(cfg, group_name, tail), _directory_record(group_name, tail, pages, None))
        return
    directory_page = group["next_directory_page"]
    atomic_replace(_directory_path(cfg, group_name, directory_page), _directory_record(group_name, directory_page, [member_page], None))
    atomic_replace(_directory_path(cfg, group_name, tail), _directory_record(group_name, tail, pages, directory_page))
    group.update({"directory_tail": directory_page, "directory_page_count": group["directory_page_count"] + 1, "next_directory_page": directory_page + 1})


def _load_group(
    cfg: object,
    group_name: str,
    page: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_path = _group_state_path(cfg, group_name)
    if not state_path.exists():
        return _empty_group(group_name)
    state = read_json_limited(state_path, max_bytes=_MAX_GROUP_HEADER_BYTES)
    catalog = read_json_limited(_catalog_path(cfg, group_name, page), max_bytes=_MAX_DIRECTORY_PAGE_BYTES)
    partition = read_json_limited(_partition_path(cfg, group_name, page), max_bytes=_MAX_MEMBER_PAGE_BYTES)
    return state, catalog, partition


def _entries(
    group_name: str, state: dict[str, Any], catalog: dict[str, Any], partition: dict[str, Any]
) -> list[dict[str, Any]]:
    group = state.get("group_ready_members")
    page = catalog.get("group_ready_member_catalog")
    slots = partition.get("group_ready_member_partition")
    if not all(isinstance(item, dict) for item in (group, page, slots)):
        raise ValueError("Group ready-member records are invalid.")
    if (
        group.get("group_name") != group_name
        or page.get("group_name") != group_name
        or slots.get("group_name") != group_name
    ):
        raise ValueError("Group ready-member identity is invalid.")
    entries = slots.get("entries")
    identities = page.get("identities")
    if not isinstance(entries, list) or not isinstance(identities, list):
        raise ValueError("Group ready-member page is invalid.")
    if len(entries) > GROUP_MEMBER_PAGE_SIZE or len(identities) > GROUP_MEMBER_PAGE_SIZE:
        raise ValueError("Group ready-member page exceeds its fixed record budget.")
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
    directory_page_count = group.get("directory_page_count")
    if (
        type(directory_page_count) is not int
        or directory_page_count < 0
        or type(group.get("next_directory_page")) is not int
        or type(group.get("next_member_page")) is not int
        or type(group.get("next_writable_index_page")) is not int
        or type(group.get("writable_index_count")) is not int
        or group["writable_index_count"] < 0
        or group.get("digest_algorithm") != _MEMBERSHIP_DIGEST_ALGORITHM
        or type(group.get("member_count")) is not int
        or group["member_count"] < 0
    ):
        raise ValueError("Group ready-member catalog tail is invalid.")
    entries: list[dict[str, Any]] = []
    writable_pages: list[int] = []
    for page in _iter_member_pages(cfg, group_name, group):
        _state, catalog, partition = _load_group(cfg, group_name, page)
        page_entries = _entries(group_name, state, catalog, partition)
        entries.extend(page_entries)
        if len(page_entries) < GROUP_MEMBER_PAGE_SIZE:
            writable_pages.append(page)
    if group.get("writable_member_page") is not None and group["writable_member_page"] not in writable_pages:
        raise ValueError("Group ready-member writable page is invalid.")
    _validate_writable_index(cfg, group_name, group, writable_pages)
    if len(entries) != group.get("member_count") or _digest(entries) != group.get("membership_digest"):
        raise ValueError("Group ready-member count or digest is invalid.")
    return state, entries


def _validate_writable_index(
    cfg: object,
    group_name: str,
    group: dict[str, Any],
    writable_pages: list[int],
) -> None:
    """Validate the bounded FIFO/free chains and their member-page coverage."""
    for field in (
        "writable_index_head",
        "writable_index_tail",
        "writable_index_free_head",
    ):
        value = group.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Group ready-member writable index pointer is invalid.")
    next_index_page = group.get("next_writable_index_page")
    count = group.get("writable_index_count")
    current_page = group.get("writable_member_page")
    if (
        type(next_index_page) is not int
        or next_index_page < 0
        or type(count) is not int
        or count < 0
        or (group.get("writable_index_head") is None) != (group.get("writable_index_tail") is None)
    ):
        raise ValueError("Group ready-member writable index state is invalid.")

    queued_member_pages: set[int] = set()
    active_index_pages: set[int] = set()
    index_page = group.get("writable_index_head")
    last_index_page: int | None = None
    while index_page is not None:
        if index_page >= next_index_page or index_page in active_index_pages:
            raise ValueError("Group ready-member writable index chain is invalid.")
        active_index_pages.add(index_page)
        record = _read_writable_index(cfg, group_name, index_page)
        for member_page in record["member_pages"]:
            if (
                member_page >= group.get("next_member_page", 0)
                or member_page in queued_member_pages
                or member_page == current_page
            ):
                raise ValueError("Group ready-member writable index coverage is invalid.")
            _state, catalog, partition = _load_group(cfg, group_name, member_page)
            entries = _entries(group_name, {"group_ready_members": group}, catalog, partition)
            if len(entries) >= GROUP_MEMBER_PAGE_SIZE or not catalog["group_ready_member_catalog"].get(
                "writable_indexed", False
            ):
                raise ValueError("Group ready-member writable index points to an invalid page.")
            queued_member_pages.add(member_page)
        last_index_page = index_page
        index_page = record["next_page"]
    if last_index_page != group.get("writable_index_tail") or len(queued_member_pages) != count:
        raise ValueError("Group ready-member writable index count is invalid.")

    free_index_pages: set[int] = set()
    index_page = group.get("writable_index_free_head")
    while index_page is not None:
        if (
            index_page >= next_index_page
            or index_page in active_index_pages
            or index_page in free_index_pages
        ):
            raise ValueError("Group ready-member writable index free chain is invalid.")
        free_index_pages.add(index_page)
        record = _read_writable_index(cfg, group_name, index_page)
        if record["member_pages"]:
            raise ValueError("Group ready-member writable index free page is not empty.")
        index_page = record["next_page"]

    for member_page in writable_pages:
        _state, catalog, _partition = _load_group(cfg, group_name, member_page)
        is_indexed = bool(catalog["group_ready_member_catalog"].get("writable_indexed", False))
        if member_page == current_page:
            if is_indexed:
                raise ValueError("Group ready-member writable page is also indexed.")
        elif is_indexed != (member_page in queued_member_pages):
            raise ValueError("Group ready-member writable index coverage is invalid.")


def validate_group_ready_member_writable_index(cfg: object, group_name: str) -> None:
    """Validate all durable writable-page index records for one Group."""
    state, _catalog, _partition = _load_group(cfg, group_name)
    group = state["group_ready_members"]
    writable_pages: list[int] = []
    for member_page in _iter_member_pages(cfg, group_name, group):
        _state, catalog, partition = _load_group(cfg, group_name, member_page)
        if len(_entries(group_name, state, catalog, partition)) < GROUP_MEMBER_PAGE_SIZE:
            writable_pages.append(member_page)
    _validate_writable_index(cfg, group_name, group, writable_pages)


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
    locator = read_json_limited(
        _locator_path(cfg, group_name, entry["identity"]), max_bytes=_MAX_LOCATOR_BYTES
    )["group_ready_member_locator"]
    if not isinstance(locator, dict):
        raise ValueError("Group ready-member locator is invalid.")
    page = locator.get("page")
    slot = locator.get("slot")
    group = state["group_ready_members"]
    if (
        locator.get("schema_version") != GROUP_READY_MEMBERS_VERSION
        or locator.get("identity") != entry["identity"]
        or locator.get("group_name") != group_name
        or locator.get("member_revision") != entry["member_revision"]
        or type(page) is not int
        or page < 0
        or type(slot) is not int
        or slot < 0
        or type(group.get("next_member_page")) is not int
        or page >= group["next_member_page"]
    ):
        raise ValueError("Group ready-member locator is invalid.")
    _state, catalog, partition = _load_group(cfg, group_name, page)
    page_entries = _entries(group_name, state, catalog, partition)
    if slot >= len(page_entries):
        raise ValueError("Group ready-member locator slot is invalid.")
    located = page_entries[slot]
    if located.get("identity") != entry["identity"] or located.get("member_revision") != entry["member_revision"]:
        raise ValueError("Group ready-member locator does not point to its entry.")
    return located, page


def _write_group(
    cfg: object,
    group_name: str,
    state: dict[str, Any],
    catalog: dict[str, Any],
    partition: dict[str, Any],
) -> None:
    page = catalog["group_ready_member_catalog"]["page"]
    require_json_size(partition, max_bytes=_MAX_MEMBER_PAGE_BYTES, record_type="member_page")
    require_json_size(catalog, max_bytes=_MAX_DIRECTORY_PAGE_BYTES, record_type="member_catalog")
    require_json_size(state, max_bytes=_MAX_GROUP_HEADER_BYTES, record_type="member_header")
    atomic_replace(_partition_path(cfg, group_name, page), partition)
    atomic_replace(_catalog_path(cfg, group_name, page), catalog)
    atomic_replace(_group_state_path(cfg, group_name), state)
    # This lock only covers the global read-modify-write.  It intentionally is
    # not held while acquiring a Group lock, preserving schema -> Group -> Task.
    with exclusive(_state_lock_path(cfg)):
        value = read_json_limited(_state_path(cfg), max_bytes=_MAX_GLOBAL_STATE_BYTES)
        record = value["group_ready_members"]
        if record.get("state") not in {"building", "active"}:
            raise RuntimeError("group ready-member projection is degraded; publication is disabled.")
        record["revision"] += 1
        record["updated_at"] = utc_now()
        _write_global_state(cfg, value)


def _rewrite_page_locators(
    cfg: object,
    group_name: str,
    page: int,
    entries: list[dict[str, Any]],
) -> None:
    """Keep bounded-page locator slots exact after an entry is removed."""
    for slot, entry in enumerate(entries):
        value = {
            "group_ready_member_locator": {
                    "schema_version": GROUP_READY_MEMBERS_VERSION,
                    "identity": entry["identity"],
                    "group_name": group_name,
                    "page": page,
                    "slot": slot,
                    "member_revision": entry["member_revision"],
            }
        }
        require_json_size(value, max_bytes=_MAX_LOCATOR_BYTES, record_type="member_locator")
        atomic_replace(_locator_path(cfg, group_name, entry["identity"]), value)


# These operations validate online storage; rebuild orchestration never reaches private helpers.
load_group_ready_member_page = _load_group
read_group_ready_member_page_entries = _entries
validate_group_ready_member_locator = _validate_entry_locator
read_group_ready_member_directory = _read_directory
read_group_ready_member_writable_index = _read_writable_index


def read_group_ready_member_audit_state(cfg: object, group_key: str) -> dict[str, Any]:
    """Read the minimal validated Group state required by a bounded audit."""
    state = read_json_limited(
        group_ready_member_groups_root(cfg) / group_key / "state.json", max_bytes=_MAX_GROUP_HEADER_BYTES
    )["group_ready_members"]
    if (
        not isinstance(state.get("group_name"), str)
        or type(state.get("directory_page_count")) is not int
        or state["directory_page_count"] < 0
        or type(state.get("membership_revision")) is not int
        or state["membership_revision"] < 0
    ):
        raise ValueError("Group ready-member audit state is invalid.")
    return state


def has_group_ready_member(cfg: object, group_name: str, task_id: str, generation: int) -> bool:
    """Return whether the exact online member locator exists."""
    return _locator_path(cfg, group_name, _identity(group_name, task_id, generation)).exists()


def assert_group_ready_member_matches_task(
    cfg: object,
    task: TaskRecord,
    reference: ReadyMarkerRef,
) -> None:
    """Validate one exact online member locator against Task and ready-marker truth."""
    if not task.group_name:
        raise ValueError("Ungrouped Task cannot have a ready-member entry.")
    group_name = task.group_name
    identity = _identity(group_name, task.task_id, task.ready_generation)
    locator_path = _locator_path(cfg, group_name, identity)
    if not locator_path.exists():
        raise ValueError(f"Group ready-member is missing for Task {task.task_id!r}.")
    state, _catalog, _partition = _load_group(cfg, group_name)
    locator = read_json_limited(locator_path, max_bytes=_MAX_LOCATOR_BYTES)["group_ready_member_locator"]
    if not isinstance(locator, dict):
        raise ValueError("Group ready-member locator is invalid.")
    entry, _page = _validate_entry_locator(
        cfg,
        group_name,
        state,
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
            locator = read_json_limited(
                locator_path, max_bytes=_MAX_LOCATOR_BYTES
            )["group_ready_member_locator"]
            if not isinstance(locator, dict):
                raise ValueError("Group ready-member locator is invalid.")
            located, _page = _validate_entry_locator(
                cfg,
                group_name,
                state,
                {"identity": identity, "member_revision": locator.get("member_revision")},
            )
            if located.get("task_id") != task.task_id or located.get("generation") != reference.generation:
                raise ValueError("Group ready-member identity conflicts with an existing entry.")
            return
        page = record.get("writable_member_page")
        if page is not None:
            if type(page) is not int or page < 0:
                raise ValueError("Group ready-member writable page is invalid.")
            _state, catalog, partition = _load_group(cfg, group_name, page)
            page_entries = _entries(group_name, state, catalog, partition)
        else:
            page = _dequeue_writable_member_page(cfg, group_name, record)
            if page is None:
                page = record.get("next_member_page")
                if type(page) is not int or page < 0:
                    raise ValueError("Group ready-member next page is invalid.")
                catalog, partition = _empty_page(group_name, page)
                page_entries = []
                _append_member_page(cfg, group_name, record, page)
                record["next_member_page"] = page + 1
            else:
                _state, catalog, partition = _load_group(cfg, group_name, page)
                page_entries = _entries(group_name, state, catalog, partition)
                if len(page_entries) >= GROUP_MEMBER_PAGE_SIZE:
                    raise ValueError("Group ready-member writable index points to a full page.")
                catalog["group_ready_member_catalog"]["writable_indexed"] = False
        entry = {
            "identity": identity,
            "task_id": task.task_id,
            "generation": reference.generation,
            "queue_scope": reference.queue_scope,
            "home_machine": reference.home_machine,
            "partition": reference.partition,
            "catalog_page": reference.catalog_page,
            "marker_name": reference.marker_name,
            "lane": task.spec.lane or "gpu",
            "submission_operation_id": task.submission_operation_id,
            "source_revision": task.meta["revision"],
            "target_revision": task.meta["revision"] + 1,
            "member_revision": 1,
        }
        for label in ("task_id", "queue_scope", "home_machine", "partition", "marker_name"):
            _validate_projection_identifier(entry[label], label)
        if entry["submission_operation_id"] is not None:
            _validate_projection_identifier(entry["submission_operation_id"], "submission_operation_id")
        page_entries.append(entry)
        catalog["group_ready_member_catalog"]["identities"] = [item["identity"] for item in page_entries]
        catalog["group_ready_member_catalog"]["revision"] += 1
        partition["group_ready_member_partition"]["entries"] = page_entries
        partition["group_ready_member_partition"]["revision"] += 1
        record["writable_member_page"] = page if len(page_entries) < GROUP_MEMBER_PAGE_SIZE else None
        record["member_count"] += 1
        record["membership_digest"] = _update_digest(record["membership_digest"], entry)
        record["membership_revision"] += 1
        record["updated_at"] = utc_now()
        locator_value = {
            "group_ready_member_locator": {
                    "schema_version": GROUP_READY_MEMBERS_VERSION,
                    "identity": identity,
                    "group_name": group_name,
                    "page": page,
                    "slot": len(page_entries) - 1,
                    "member_revision": entry["member_revision"],
            }
        }
        require_json_size(locator_value, max_bytes=_MAX_LOCATOR_BYTES, record_type="member_locator")
        atomic_replace(locator_path, locator_value)
        _write_group(cfg, group_name, state, catalog, partition)
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"member_publish_failed:{group_name}:{type(exc).__name__}")
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
        locator = read_json_limited(locator_path, max_bytes=_MAX_LOCATOR_BYTES)["group_ready_member_locator"]
        if not isinstance(locator, dict):
            raise ValueError("Group ready-member locator is invalid.")
        located, page = _validate_entry_locator(
            cfg,
            group_name,
            state,
            {"identity": identity, "member_revision": locator.get("member_revision")},
        )
        if located.get("task_id") != task_id or located.get("generation") != generation:
            return False
        _state, catalog, partition = _load_group(cfg, group_name, page)
        page_entries = _entries(group_name, state, catalog, partition)
        page_entries.remove(located)
        catalog["group_ready_member_catalog"]["identities"] = [item["identity"] for item in page_entries]
        catalog["group_ready_member_catalog"]["revision"] += 1
        partition["group_ready_member_partition"]["entries"] = page_entries
        partition["group_ready_member_partition"]["revision"] += 1
        current_page = record.get("writable_member_page")
        if current_page is None:
            record["writable_member_page"] = page
        elif current_page != page:
            if not catalog["group_ready_member_catalog"].get("writable_indexed", False):
                _enqueue_writable_member_page(cfg, group_name, record, page)
                catalog["group_ready_member_catalog"]["writable_indexed"] = True
        record["member_count"] -= 1
        record["membership_digest"] = _update_digest(record["membership_digest"], located)
        record["membership_revision"] += 1
        record["updated_at"] = utc_now()
        _rewrite_page_locators(cfg, group_name, page, page_entries)
        _write_group(cfg, group_name, state, catalog, partition)
        locator_path.unlink(missing_ok=True)
        return True
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mark_group_ready_members_degraded(cfg, f"member_retire_invalid:{group_name}:{type(exc).__name__}")
        raise RuntimeError(f"Group {group_name!r} ready-member projection is invalid; retirement is disabled.") from exc
