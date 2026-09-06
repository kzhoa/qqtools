"""Durable generation-safe ready liveness projection."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..locks import exclusive, schema_lock, schema_writer_lock
from ..paths import group_path, ready_state_path, shared_paths, submission_path, task_path
from ..records import TaskRecord, normalize_group_record, utc_now, validate_identifier
from ..store import atomic_replace, iter_json, read_json
from ..work_budget import SliceBudget
from . import primary_candidates, routes, state
from .group_members import (
    group_ready_members_state,
    is_group_ready_member_projection_usable,
    publish_group_ready_member,
    read_group_ready_members,
    retire_group_ready_member,
)
from .records import ReadyMarkerRef, ReadyScope

READY_BUILD_PAGE_SIZE = 64
ReadyClassification = Literal["claimable", "temporarily_unavailable", "permanently_stale", "corrupt"]


@dataclass(frozen=True, slots=True)
class ReadyClassificationResult:
    classification: ReadyClassification
    reason: str
    task: TaskRecord | None = None


@dataclass(frozen=True, slots=True)
class ReadyCursor:
    project_id: str
    machine_name: str
    queue_scope: ReadyScope
    catalog_page: int | None
    partition: str | None
    after_name: str | None
    revision: int


@dataclass(frozen=True, slots=True)
class ReadyPeek:
    """One lazily discovered ready marker and the cursor after it."""

    reference: ReadyMarkerRef | None
    cursor: ReadyCursor
    wrapped: bool = False
    exhausted: bool = False
    unresolved: bool = False


def _build_root(cfg: object, build_id: str) -> Path:
    validate_identifier(build_id, "ready build id")
    return shared_paths(cfg.shared_root)["ready_builds"] / build_id


def _build_page_path(cfg: object, build_id: str, page: int) -> Path:
    return _build_root(cfg, build_id) / "watermark" / f"{page:016d}.json"


def _write_build_page(
    cfg: object,
    build_id: str,
    page: int,
    task_ids: list[str],
) -> None:
    atomic_replace(
        _build_page_path(cfg, build_id, page),
        {
            "ready_build_page": {
                "schema_version": state.READY_PROTOCOL_VERSION,
                "build_id": build_id,
                "page": page,
                "task_ids": list(task_ids),
            }
        },
    )


def _capture_build_watermark(cfg: object, record: dict[str, Any]) -> None:
    """Stream one immutable legacy inventory into bounded durable pages."""
    build = record["build"]
    build_id = build["build_id"]
    page = 0
    task_count = 0
    task_ids: list[str] = []
    tasks = shared_paths(cfg.shared_root)["tasks"]
    with os.scandir(tasks) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            task_ids.append(entry.name[:-5])
            task_count += 1
            if len(task_ids) == READY_BUILD_PAGE_SIZE:
                _write_build_page(cfg, build_id, page, task_ids)
                page += 1
                task_ids = []
    if task_ids:
        _write_build_page(cfg, build_id, page, task_ids)
        page += 1
    build["watermark"] = {
        "page_count": page,
        "task_count": task_count,
        "captured_at": utc_now(),
        "is_complete": True,
    }
    build["phase"] = "backfill"


def _reset_ready_projection_for_repair(cfg: object, build_id: str) -> None:
    """Move the damaged advisory projection aside before a truth-based rebuild."""
    paths = shared_paths(cfg.shared_root)
    archive = _build_root(cfg, build_id) / "replaced-projection"
    archive.mkdir(parents=True, exist_ok=True)
    targets = {
        "home": paths["ready_home"],
        "shared": paths["ready_shared"],
        "catalogs": paths["ready_catalogs"],
        "reservations": paths["ready_reservations"],
        "cursors": paths["ready_cursors"],
        "allocators": paths["ready"] / "allocators",
    }
    for name, target in targets.items():
        archived = archive / name
        if target.exists():
            os.replace(target, archived)
        target.mkdir(parents=True, exist_ok=True)


def begin_ready_index_build(cfg: object, *, is_repair: bool = False) -> dict[str, Any]:
    """Start or resume the single durable ready-index build."""
    state.ensure_ready_layout(cfg)
    current_state = state.read_ready_index_state(cfg)
    if current_state == "active" or (current_state == "degraded" and not is_repair):
        return state.read_ready_index_status(cfg)
    if current_state in {"absent", "degraded"}:
        with schema_lock(cfg.shared_root):
            with exclusive(state.state_lock_path(cfg)):
                path = ready_state_path(cfg.shared_root)
                value, record = state.read_state_record(cfg)
                current_state = record["state"]
                if current_state == "active" or (current_state == "degraded" and not is_repair):
                    return record
                if current_state in {"absent", "degraded"}:
                    state.install_writer_capability_gate(cfg)
                    build_id = uuid.uuid4().hex
                    if current_state == "degraded" and is_repair:
                        _reset_ready_projection_for_repair(cfg, build_id)
                    record["state"] = "building"
                    record["writer_capability"] = state.READY_WRITER_CAPABILITY
                    record["build"] = {
                        "build_id": build_id,
                        "phase": "inventory",
                        "is_repair": is_repair,
                        "watermark": {
                            "page_count": 0,
                            "task_count": 0,
                            "captured_at": None,
                            "is_complete": False,
                        },
                        "cursor": {"page": 0, "offset": 0},
                        "audit_cursor": {"page": 0, "offset": 0},
                        "processed": 0,
                        "repaired": 0,
                        "stale_removed": 0,
                        "started_at": utc_now(),
                        "completed_at": None,
                    }
                    state.commit_state_under_lock(path, value, record)
    with exclusive(state.state_lock_path(cfg)):
        path = ready_state_path(cfg.shared_root)
        value, record = state.read_state_record(cfg)
        current_state = record["state"]
        if current_state == "active":
            return record
        if current_state == "degraded" and not is_repair:
            return record
        build = record.get("build")
        if not isinstance(build, dict):
            raise RuntimeError("ready index build state is missing.")
        if not build.get("watermark", {}).get("is_complete"):
            _capture_build_watermark(cfg, record)
            state.commit_state_under_lock(path, value, record)
        return record


def is_primary_ready_index_active(cfg: object) -> bool:
    """Return whether the primary-only candidate projection is usable."""
    return primary_candidates.is_projection_active(cfg) and is_group_ready_member_projection_usable(cfg)


def begin_primary_ready_index_rebuild(cfg: object, build_id: str) -> None:
    """Initialize the ready layout before starting a candidate-only rebuild."""
    state.ensure_ready_layout(cfg)
    primary_candidates.begin_primary_ready_index_rebuild(cfg, build_id)


def _cursor_path(root: Path, project_id: str, machine_name: str, scope: ReadyScope) -> Path:
    validate_identifier(project_id, "project_id")
    validate_identifier(machine_name, "machine_name")
    return shared_paths(root)["ready_cursors"] / f"{project_id}.{machine_name}.{scope}.json"


def _default_cursor(project_id: str, machine_name: str, scope: ReadyScope) -> ReadyCursor:
    return ReadyCursor(project_id, machine_name, scope, 0, None, None, 0)


def load_ready_cursor(
    cfg: object,
    project_id: str,
    queue_scope: ReadyScope,
) -> ReadyCursor:
    """Load advisory candidate progress, falling back conservatively on damage."""
    path = _cursor_path(cfg.shared_root, project_id, cfg.machine_name, queue_scope)
    if not path.exists():
        return _default_cursor(project_id, cfg.machine_name, queue_scope)
    try:
        record = read_json(path)["cursor"]
        if set(record) != {
            "schema_version",
            "project_id",
            "machine_name",
            "queue_scope",
            "catalog_page",
            "partition",
            "after_name",
            "revision",
        }:
            raise ValueError("ready cursor schema is invalid.")
        if (
            record["schema_version"] != state.READY_PROTOCOL_VERSION
            or record["project_id"] != project_id
            or record["machine_name"] != cfg.machine_name
            or record["queue_scope"] != queue_scope
            or not isinstance(record["revision"], int)
            or record["revision"] < 0
        ):
            raise ValueError("ready cursor identity is invalid.")
        page = record["catalog_page"]
        if page is not None:
            page = int(page)
        return ReadyCursor(
            project_id,
            cfg.machine_name,
            queue_scope,
            page,
            record["partition"],
            record["after_name"],
            record["revision"],
        )
    except (KeyError, TypeError, ValueError):
        return _default_cursor(project_id, cfg.machine_name, queue_scope)


def _save_ready_cursor(cfg: object, cursor: ReadyCursor) -> None:
    atomic_replace(
        _cursor_path(cfg.shared_root, cursor.project_id, cursor.machine_name, cursor.queue_scope),
        {
            "cursor": {
                "schema_version": state.READY_PROTOCOL_VERSION,
                "project_id": cursor.project_id,
                "machine_name": cursor.machine_name,
                "queue_scope": cursor.queue_scope,
                "catalog_page": (None if cursor.catalog_page is None else str(cursor.catalog_page)),
                "partition": cursor.partition,
                "after_name": cursor.after_name,
                "revision": cursor.revision,
            }
        },
    )


def _reference_from_slot(
    cfg: object,
    scope: ReadyScope,
    catalog_page: int,
    partition: str,
    marker_name: str,
) -> ReadyMarkerRef:
    stem = marker_name[:-5] if marker_name.endswith(".json") else marker_name
    task_id, separator, generation_value = stem.rpartition(".")
    generation = int(generation_value) if separator else -1
    home_machine = cfg.machine_name
    if scope == "shared":
        provisional = ReadyMarkerRef(task_id, generation, scope, home_machine, partition, catalog_page, marker_name)
        try:
            marker = read_json(routes.marker_path(cfg.shared_root, provisional))["ready_marker"]
            if isinstance(marker.get("home_machine"), str):
                home_machine = marker["home_machine"]
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            pass
    return ReadyMarkerRef(task_id, generation, scope, home_machine, partition, catalog_page, marker_name)


def _is_partition_referenced_under_route_lock(
    cfg: object,
    route_key: str,
    page_path: Path,
    page_number: int,
    partition_name: str,
) -> bool:
    """Return whether a partition remains referenced after a route-stable recheck."""
    lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
        try:
            catalog = read_json(page_path)["ready_catalog"]
            partitions = catalog["partitions"]
            if (
                catalog.get("schema_version") != state.READY_PROTOCOL_VERSION
                or catalog.get("route") != route_key
                or catalog.get("page") != page_number
                or not isinstance(partitions, list)
                or not all(isinstance(item, str) for item in partitions)
            ):
                raise ValueError("ready catalog is invalid.")
            return partition_name in partitions
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return True


def next_ready_marker(
    cfg: object,
    project_id: str,
    queue_scope: ReadyScope,
    excluded_identities: set[str] | None = None,
) -> tuple[ReadyMarkerRef | None, bool]:
    """Return and durably advance past one marker without unbounded enumeration."""
    cursor = load_ready_cursor(cfg, project_id, queue_scope)
    route_key = routes.route_key(queue_scope, cfg.machine_name)
    page_number = cursor.catalog_page or 0
    page_path = routes.catalog_path(cfg.shared_root, route_key, page_number)
    if not page_path.exists():
        if page_number == 0:
            return None, False
        _save_ready_cursor(
            cfg,
            ReadyCursor(
                project_id,
                cfg.machine_name,
                queue_scope,
                0,
                None,
                None,
                cursor.revision + 1,
            ),
        )
        return None, True
    try:
        catalog = read_json(page_path)["ready_catalog"]
        partitions = catalog["partitions"]
        successor = catalog.get("successor")
        if not isinstance(partitions, list) or not all(isinstance(item, str) for item in partitions):
            raise ValueError("ready catalog partitions are invalid.")
        if successor is not None and (not isinstance(successor, int) or successor < 0):
            raise ValueError("ready catalog successor is invalid.")
    except (KeyError, TypeError, ValueError):
        state.mark_ready_index_degraded(cfg, f"catalog_invalid:{route_key}:{page_number}")
        return None, False
    if cursor.partition in partitions:
        partition_index = partitions.index(cursor.partition)
        partition_name = cursor.partition
        after_name = cursor.after_name
    elif partitions:
        partition_index = 0
        partition_name = partitions[0]
        after_name = None
    else:
        next_page = successor if successor is not None else 0
        has_wrapped = successor is None
        _save_ready_cursor(
            cfg,
            ReadyCursor(
                project_id,
                cfg.machine_name,
                queue_scope,
                next_page,
                None,
                None,
                cursor.revision + 1,
            ),
        )
        return None, has_wrapped
    partition_path = routes.partition_record_path(cfg.shared_root, queue_scope, cfg.machine_name, partition_name)
    try:
        partition = read_json(partition_path)["ready_partition"]
        slots = partition["slots"]
        if not isinstance(slots, list) or not all(isinstance(name, str) for name in slots):
            raise ValueError("ready partition slots are invalid.")
        names = sorted(slots)
    except FileNotFoundError:
        if _is_partition_referenced_under_route_lock(cfg, route_key, page_path, page_number, partition_name):
            state.mark_ready_index_degraded(cfg, f"partition_missing:{route_key}:{partition_name}")
            return None, False
        names = []
    except (KeyError, TypeError, ValueError):
        state.mark_ready_index_degraded(cfg, f"partition_invalid:{route_key}:{partition_name}")
        return None, False
    for marker_name in names:
        if after_name is not None and marker_name <= after_name:
            continue
        reference = _reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name)
        if excluded_identities is not None and reference.identity in excluded_identities:
            _save_ready_cursor(
                cfg,
                ReadyCursor(
                    project_id,
                    cfg.machine_name,
                    queue_scope,
                    page_number,
                    partition_name,
                    marker_name,
                    cursor.revision + 1,
                ),
            )
            return None, True
        _save_ready_cursor(
            cfg,
            ReadyCursor(
                project_id,
                cfg.machine_name,
                queue_scope,
                page_number,
                partition_name,
                marker_name,
                cursor.revision + 1,
            ),
        )
        return reference, False
    if partition_index + 1 < len(partitions):
        next_page = page_number
        next_partition = partitions[partition_index + 1]
        has_wrapped = False
    elif successor is not None:
        next_page = successor
        next_partition = None
        has_wrapped = False
    else:
        next_page = 0
        next_partition = None
        has_wrapped = True
    _save_ready_cursor(
        cfg,
        ReadyCursor(
            project_id,
            cfg.machine_name,
            queue_scope,
            next_page,
            next_partition,
            None,
            cursor.revision + 1,
        ),
    )
    return None, has_wrapped


def peek_ready_marker(
    cfg: object,
    project_id: str,
    queue_scope: ReadyScope,
    cursor: ReadyCursor | None,
    budget: SliceBudget,
) -> ReadyPeek:
    """Read at most one candidate through a caller-owned, bounded cursor.

    Catalogs and partitions are followed by their durable successor links.  No
    directory-wide glob or complete reference list is built, and every file
    read is preceded by an operation-budget check.
    """
    current = cursor or _default_cursor(project_id, cfg.machine_name, queue_scope)
    route_key = routes.route_key(queue_scope, cfg.machine_name)
    page_number = current.catalog_page or 0
    partition_name = current.partition
    after_name = current.after_name
    progress_cursor = current
    while True:
        page_path = routes.catalog_path(cfg.shared_root, route_key, page_number)
        if not budget.can_start_operation():
            return ReadyPeek(None, progress_cursor, exhausted=True)
        budget.consume_operation()
        if not page_path.exists():
            if page_number != 0:
                page_number = 0
                partition_name = None
                after_name = None
                continue
            return ReadyPeek(None, current)
        if not budget.can_start_operation():
            return ReadyPeek(None, progress_cursor, exhausted=True)
        budget.consume_operation()
        try:
            catalog = read_json(page_path)["ready_catalog"]
            partitions = catalog["partitions"]
            successor = catalog.get("successor")
            if (
                catalog.get("schema_version") != state.READY_PROTOCOL_VERSION
                or catalog.get("route") != route_key
                or catalog.get("page") != page_number
                or not isinstance(partitions, list)
                or len(partitions) > routes.READY_CATALOG_PAGE_SIZE
                or not all(isinstance(item, str) for item in partitions)
                or (successor is not None and (not isinstance(successor, int) or successor < 0))
            ):
                raise ValueError("ready catalog is invalid.")
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            state.mark_ready_index_degraded(cfg, f"catalog_invalid:{route_key}:{page_number}")
            return ReadyPeek(None, current, unresolved=True)

        if partition_name in partitions:
            partition_index = partitions.index(partition_name)
        else:
            partition_index = 0
            partition_name = None
            after_name = None
        while partition_index < len(partitions):
            partition_name = partitions[partition_index]
            partition_path = routes.partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            if not budget.can_start_operation():
                return ReadyPeek(None, progress_cursor, exhausted=True)
            budget.consume_operation()
            try:
                partition = read_json(partition_path)["ready_partition"]
                slots = partition["slots"]
                if (
                    partition.get("schema_version") != state.READY_PROTOCOL_VERSION
                    or partition.get("route") != route_key
                    or partition.get("partition") != partition_name
                    or not isinstance(slots, list)
                    or len(slots) > routes.READY_PARTITION_SLOTS
                    or not all(isinstance(name, str) for name in slots)
                ):
                    raise ValueError("ready partition is invalid.")
                names = sorted(slots)
            except FileNotFoundError:
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                if _is_partition_referenced_under_route_lock(cfg, route_key, page_path, page_number, partition_name):
                    state.mark_ready_index_degraded(cfg, f"partition_missing:{route_key}:{partition_name}")
                    return ReadyPeek(None, current, unresolved=True)
                names = []
            except (KeyError, TypeError, ValueError):
                state.mark_ready_index_degraded(cfg, f"partition_invalid:{route_key}:{partition_name}")
                return ReadyPeek(None, current, unresolved=True)
            for marker_name in names:
                if after_name is not None and marker_name <= after_name:
                    continue
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                reference = _reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name)
                next_cursor = ReadyCursor(
                    project_id,
                    cfg.machine_name,
                    queue_scope,
                    page_number,
                    partition_name,
                    marker_name,
                    current.revision + 1,
                )
                return ReadyPeek(reference, next_cursor)
            progress_cursor = ReadyCursor(
                project_id,
                cfg.machine_name,
                queue_scope,
                page_number,
                partition_name,
                names[-1] if names else "",
                current.revision + 1,
            )
            partition_index += 1
            after_name = None

        if successor is not None:
            progress_cursor = ReadyCursor(
                project_id,
                cfg.machine_name,
                queue_scope,
                successor,
                None,
                None,
                current.revision + 1,
            )
            page_number = successor
            partition_name = None
            after_name = None
            continue
        return ReadyPeek(
            None,
            ReadyCursor(project_id, cfg.machine_name, queue_scope, 0, None, None, current.revision + 1),
            wrapped=True,
        )


def peek_primary_ready_marker(
    cfg: object,
    project_id: str,
    queue_scope: ReadyScope,
    cursor: ReadyCursor | None,
    budget: SliceBudget,
) -> ReadyPeek:
    """Read one primary-only candidate without touching borrow markers."""
    current = cursor or _default_cursor(project_id, cfg.machine_name, queue_scope)
    if not primary_candidates.is_projection_active(cfg):
        return ReadyPeek(None, current, unresolved=True)
    route_key = routes.route_key(queue_scope, cfg.machine_name)
    primary_route = primary_candidates.route_key(queue_scope, cfg.machine_name)
    page_number = current.catalog_page or 0
    partition_name = current.partition
    after_name = current.after_name
    progress_cursor = current
    while True:
        page_path = routes.catalog_path(cfg.shared_root, route_key, page_number)
        if not budget.can_start_operation():
            return ReadyPeek(None, progress_cursor, exhausted=True)
        budget.consume_operation()
        try:
            catalog = read_json(page_path)["ready_catalog"]
            partitions = catalog["partitions"]
            successor = catalog.get("successor")
            if not isinstance(partitions, list) or not all(isinstance(item, str) for item in partitions):
                raise ValueError("primary catalog is invalid.")
        except FileNotFoundError:
            if page_number == 0 and not routes.allocator_path(cfg.shared_root, route_key).exists():
                return ReadyPeek(None, current)
            return ReadyPeek(None, current, unresolved=True)
        except (KeyError, TypeError, ValueError):
            return ReadyPeek(None, current, unresolved=True)
        partition_index = partitions.index(partition_name) if partition_name in partitions else 0
        if partition_name not in partitions:
            after_name = None
        while partition_index < len(partitions):
            partition_name = partitions[partition_index]
            if not budget.can_start_operation():
                return ReadyPeek(None, progress_cursor, exhausted=True)
            budget.consume_operation()
            try:
                slots = read_json(
                    routes.partition_record_path(cfg.shared_root, queue_scope, cfg.machine_name, partition_name)
                )["ready_partition"]["slots"]
                if not isinstance(slots, list) or not all(isinstance(item, str) for item in slots):
                    raise ValueError("primary partition is invalid.")
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                return ReadyPeek(None, current, unresolved=True)
            for marker_name in sorted(slots):
                if after_name is not None and marker_name <= after_name:
                    continue
                progress_cursor = ReadyCursor(
                    project_id,
                    cfg.machine_name,
                    queue_scope,
                    page_number,
                    partition_name,
                    marker_name,
                    current.revision + 1,
                )
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                candidate_path = primary_candidates.candidate_path(
                    cfg, primary_route, marker_name.removesuffix(".json")
                )
                if not candidate_path.exists():
                    continue
                try:
                    candidate = read_json(candidate_path)["primary_ready_candidate"]
                    required = {
                        "schema_version",
                        "task_id",
                        "generation",
                        "queue_scope",
                        "home_machine",
                        "partition",
                        "catalog_page",
                        "marker_name",
                    }
                    if (
                        set(candidate) != required
                        or candidate.get("schema_version") != primary_candidates.PRIMARY_READY_PROTOCOL_VERSION
                        or candidate.get("queue_scope") != queue_scope
                        or not isinstance(candidate.get("task_id"), str)
                        or not candidate["task_id"]
                        or type(candidate.get("generation")) is not int
                        or candidate["generation"] <= 0
                        or not isinstance(candidate.get("home_machine"), str)
                        or not isinstance(candidate.get("partition"), str)
                        or type(candidate.get("catalog_page")) is not int
                        or candidate["catalog_page"] < 0
                        or candidate.get("marker_name") != marker_name
                    ):
                        raise ValueError("primary candidate is invalid.")
                    reference = ReadyMarkerRef(
                        candidate["task_id"],
                        candidate["generation"],
                        queue_scope,
                        candidate["home_machine"],
                        candidate["partition"],
                        candidate["catalog_page"],
                        candidate["marker_name"],
                    )
                except (KeyError, TypeError, ValueError, OSError):
                    return ReadyPeek(None, current, unresolved=True)
                return ReadyPeek(reference, progress_cursor)
            partition_index += 1
            after_name = None
        if successor is None:
            return ReadyPeek(
                None,
                ReadyCursor(
                    project_id,
                    cfg.machine_name,
                    queue_scope,
                    0,
                    None,
                    None,
                    current.revision + 1,
                ),
                wrapped=True,
            )
        page_number, partition_name, after_name = successor, None, None


def ready_index_revision(cfg: object, queue_scope: ReadyScope) -> str:
    """Return a read-only revision fingerprint for one machine route."""
    route_key = routes.route_key(queue_scope, cfg.machine_name)
    catalog_root = shared_paths(cfg.shared_root)["ready_catalogs"] / route_key
    parts: list[object] = [state.read_ready_index_status(cfg).get("revision")]
    for page_path in sorted(catalog_root.glob("*.json")):
        catalog = read_json(page_path)["ready_catalog"]
        parts.append((catalog["page"], catalog["revision"], tuple(catalog["partitions"])))
        for partition_name in catalog["partitions"]:
            partition_path = routes.partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            partition = read_json(partition_path)["ready_partition"]
            parts.append((partition_name, partition["revision"], tuple(partition["slots"])))
    return repr(parts)


def iter_ready_marker_refs(cfg: object, queue_scope: ReadyScope) -> list[ReadyMarkerRef]:
    """Enumerate ready markers without advancing any dispatch cursor."""
    route_key = routes.route_key(queue_scope, cfg.machine_name)
    catalog_root = shared_paths(cfg.shared_root)["ready_catalogs"] / route_key
    references: list[ReadyMarkerRef] = []
    for page_path in sorted(catalog_root.glob("*.json")):
        catalog = read_json(page_path)["ready_catalog"]
        page_number = catalog["page"]
        for partition_name in catalog["partitions"]:
            partition_path = routes.partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            partition = read_json(partition_path)["ready_partition"]
            for marker_name in sorted(partition["slots"]):
                references.append(_reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name))
    return references


def delete_stale_ready_marker(cfg: object, reference: ReadyMarkerRef) -> bool:
    """Recheck authoritative generation immediately before exact stale deletion."""
    result = classify_ready_marker(cfg, reference)
    if result.classification != "permanently_stale":
        return False
    return delete_ready_marker(cfg, reference.task_id, reference.generation)


def _has_primary_ready_demand(cfg: object, task: TaskRecord) -> bool:
    """Return whether a marker can represent primary demand on this project."""
    if not task.group_name:
        return True
    group = read_json(group_path(cfg.shared_root, task.group_name))
    normalize_group_record(group)
    workers = group["group"]["worker_set"]
    if task.placement_runtime["queue_scope"] == "home":
        worker = workers.get(task.placement_policy["home_machine"])
        return worker is not None and worker["scheduling_role"] == "primary"
    return any(worker["scheduling_role"] == "primary" for worker in workers.values())


def rebuild_primary_ready_index(cfg: object) -> None:
    """Rebuild primary candidates from authoritative queued Task records."""
    state.ensure_ready_layout(cfg)
    with primary_candidates.projection_rebuild_lock(cfg):
        build_id = uuid.uuid4().hex
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": False,
                    "updated_at": utc_now(),
                }
            },
        )
        primary_candidates.park_projection_under_lock(cfg, build_id)
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": True,
                    "updated_at": utc_now(),
                }
            },
        )
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if not _task_should_have_ready_marker(task):
                continue
            reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
            if reference is not None:
                primary_candidates.sync_candidate_under_lock(cfg, task, reference, should_require_active=False)
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "active",
                    "completed_build_id": build_id,
                    "updated_at": utc_now(),
                }
            },
        )


def sync_primary_ready_group(
    cfg: object,
    group_name: str,
    *,
    previous_workers: dict[str, dict[str, Any]],
) -> None:
    """Refresh primary candidates affected by a Group worker-role mutation."""
    with primary_candidates.projection_rebuild_lock(cfg):
        if not primary_candidates.accepts_updates_under_lock(cfg):
            return
        if group_ready_members_state(cfg) in {"building", "active"}:
            for entry in read_group_ready_members(cfg, group_name):
                reference = ReadyMarkerRef(
                    entry["task_id"],
                    entry["generation"],
                    entry["queue_scope"],
                    entry["home_machine"],
                    entry["partition"],
                    entry["catalog_page"],
                    entry["marker_name"],
                )
                primary_candidates.sync_member_candidate_under_lock(cfg, group_name, reference, previous_workers)
            return
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if task.group_name != group_name or not _task_should_have_ready_marker(task):
                continue
            reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
            if reference is not None:
                primary_candidates.sync_candidate_under_lock(cfg, task, reference)


def primary_projection_routes_for_group(
    cfg: object,
    group_name: str,
) -> list[tuple[ReadyScope, str]]:
    """Return every authoritative route whose candidates a Group sync can alter."""
    primary_routes: set[tuple[ReadyScope, str]] = set()
    if group_ready_members_state(cfg) == "active":
        for entry in read_group_ready_members(cfg, group_name):
            primary_routes.add((entry["queue_scope"], entry["home_machine"]))
        return sorted(primary_routes)
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if task.group_name != group_name or not _task_should_have_ready_marker(task):
            continue
        reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is not None:
            primary_routes.add((reference.queue_scope, reference.home_machine))
    return sorted(primary_routes)


def write_ready_marker(
    cfg: object,
    task: TaskRecord,
    *,
    generation: int,
    source_transition: str,
    source_revision: int,
    target_revision: int,
    reference: ReadyMarkerRef | None = None,
) -> ReadyMarkerRef:
    """Durably write a target generation before its Task transition commits."""
    if generation <= task.ready_generation:
        raise ValueError("target ready generation must exceed current generation.")
    scope = task.placement_runtime["queue_scope"]
    if reference is None:
        reference = routes.reserve_slot(cfg, task.task_id, generation, scope, task.placement_policy["home_machine"])
    reservation_path = routes.reservation_path(cfg.shared_root, task.task_id, generation)
    if not reservation_path.exists():
        raise RuntimeError("ready reservation disappeared before marker publication.")
    if (
        reference.task_id != task.task_id
        or reference.generation != generation
        or reference.queue_scope != scope
        or reference.home_machine != task.placement_policy["home_machine"]
    ):
        raise ValueError("ready reservation does not match the target Task route.")
    marker_value = {
        "schema_version": state.READY_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "generation": generation,
        "source_transition": source_transition,
        "source_revision": source_revision,
        "target_revision": target_revision,
        "queue_scope": scope,
        "home_machine": task.placement_policy["home_machine"],
        "group_name": task.group_name,
        "submission_operation_id": task.submission_operation_id,
        "created_at": utc_now(),
    }
    if task.spec.lane is None:
        marker_value["requested_gpus"] = task.spec.requested_gpus
    elif task.spec.is_cpu_only:
        marker_value.update({"lane": "cpu", "requested_cpus": task.spec.requested_cpus})
    else:
        marker_value.update({"lane": "gpu", "requested_gpus": task.spec.requested_gpus})
    marker = {"ready_marker": marker_value}
    if _has_primary_ready_demand(cfg, task):
        with routes.primary_route_update_transaction(
            cfg, [(scope, reference.home_machine)], lanes=(task.spec.lane or "gpu",)
        ):
            atomic_replace(routes.marker_path(cfg.shared_root, reference), marker)
            publish_group_ready_member(cfg, task, reference)
            primary_candidates.sync_candidate(cfg, task, reference)
    else:
        atomic_replace(routes.marker_path(cfg.shared_root, reference), marker)
        publish_group_ready_member(cfg, task, reference)
    return reference


def delete_ready_marker(cfg: object, task_id: str, generation: int) -> bool:
    """Delete only the exact generation and its slot reservation."""
    path = routes.reservation_path(cfg.shared_root, task_id, generation)
    if not path.exists():
        return False
    try:
        record = read_json(path)["ready_reservation"]
    except FileNotFoundError:
        return False
    reference = ReadyMarkerRef(
        task_id,
        generation,
        record["queue_scope"],
        record["home_machine"],
        record["partition"],
        record["catalog_page"],
        record["marker_name"],
    )
    is_primary = True
    lane = "gpu"
    group_name: str | None = None
    try:
        marker = read_json(routes.marker_path(cfg.shared_root, reference))["ready_marker"]
        lane = marker.get("lane", "gpu")
        group_name = marker.get("group_name")
        if group_name:
            group = read_json(group_path(cfg.shared_root, group_name))
            normalize_group_record(group)
            workers = group["group"]["worker_set"]
            if reference.queue_scope == "home":
                worker = workers.get(reference.home_machine)
                is_primary = worker is not None and worker["scheduling_role"] == "primary"
            else:
                is_primary = any(worker["scheduling_role"] == "primary" for worker in workers.values())
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        is_primary = True
    route_key = routes.route_key(reference.queue_scope, reference.home_machine)
    lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
        if is_primary:
            allocator_path, allocator = routes.load_or_create_allocator_under_lock(cfg.shared_root, route_key)
            allocator["ready_allocator"]["primary_state"] = "updating"
            atomic_replace(allocator_path, allocator)
        routes.marker_path(cfg.shared_root, reference).unlink(missing_ok=True)
        partition_path = routes.partition_record_path(
            cfg.shared_root,
            reference.queue_scope,
            reference.home_machine,
            reference.partition,
        )
        if partition_path.exists():
            partition_record = read_json(partition_path)
            partition = partition_record["ready_partition"]
            partition["slots"] = [name for name in partition["slots"] if name != reference.marker_name]
            partition["revision"] += 1
            if not partition["slots"] and partition.get("sealed"):
                partition_path.unlink(missing_ok=True)
                catalog_path = routes.catalog_path(cfg.shared_root, route_key, reference.catalog_page)
                if catalog_path.exists():
                    page = read_json(catalog_path)
                    catalog = page["ready_catalog"]
                    catalog["partitions"] = [item for item in catalog["partitions"] if item != reference.partition]
                    catalog["revision"] += 1
                    atomic_replace(catalog_path, page)
                allocator_path = routes.allocator_path(cfg.shared_root, route_key)
                if allocator_path.exists():
                    allocator = read_json(allocator_path)
                    control = allocator["ready_allocator"]
                    control["revision"] += 1
                    if control.get("current_partition") == reference.partition:
                        control["current_partition"] = None
                    atomic_replace(allocator_path, allocator)
            else:
                atomic_replace(partition_path, partition_record)
                allocator_path = routes.allocator_path(cfg.shared_root, route_key)
                if allocator_path.exists():
                    allocator = read_json(allocator_path)
                    allocator["ready_allocator"]["revision"] += 1
                    atomic_replace(allocator_path, allocator)
        path.unlink(missing_ok=True)
        if group_name:
            retire_group_ready_member(cfg, group_name, task_id, generation)
        primary_candidates.remove_candidate_everywhere(cfg, reference.identity)
        if is_primary:
            allocator_path, allocator = routes.load_or_create_allocator_under_lock(cfg.shared_root, route_key)
            control = allocator["ready_allocator"]
            control["primary_revision"] = control.get("primary_revision", control["revision"]) + 1
            routes.ensure_primary_lane_revisions(control)
            control["primary_lane_revisions"][lane] += 1
            control["primary_state"] = "active"
            atomic_replace(allocator_path, allocator)
    return True


def prepare_ready_transition(
    cfg: object,
    task: TaskRecord,
    source_transition: str,
    *,
    target_revision: int | None = None,
    reference: ReadyMarkerRef | None = None,
) -> tuple[int, int]:
    """Write a new marker and return the old/new generation pair."""
    old_generation = task.ready_generation
    new_generation = old_generation + 1
    write_ready_marker(
        cfg,
        task,
        generation=new_generation,
        source_transition=source_transition,
        source_revision=task.meta["revision"],
        target_revision=target_revision or task.meta["revision"] + 1,
        reference=reference,
    )
    task.ready_generation = new_generation
    return old_generation, new_generation


def retire_previous_ready_generation(cfg: object, old_generation: int, task: TaskRecord) -> None:
    if old_generation > 0 and old_generation != task.ready_generation:
        try:
            delete_ready_marker(cfg, task.task_id, old_generation)
        except (OSError, KeyError, TypeError, ValueError):
            return


def discard_ready_generation(cfg: object, task_id: str, generation: int) -> None:
    """Best-effort cleanup for a transition that did not commit Task truth."""
    try:
        delete_ready_marker(cfg, task_id, generation)
    except (OSError, KeyError, TypeError, ValueError):
        return


def retire_current_ready_generation(cfg: object, task: TaskRecord) -> None:
    if task.ready_generation > 0:
        try:
            delete_ready_marker(cfg, task.task_id, task.ready_generation)
        except (OSError, KeyError, TypeError, ValueError):
            return


def _is_ready_publication_pending(
    cfg: object,
    reference: ReadyMarkerRef,
    task: TaskRecord,
) -> bool:
    """Return whether a valid future generation is still being published."""
    if reference.generation <= task.ready_generation:
        return False
    reservation = routes.reference_for_generation(cfg, reference.task_id, reference.generation)
    if reservation is None:
        return False
    if (
        reservation.queue_scope != reference.queue_scope
        or reservation.partition != reference.partition
        or reservation.catalog_page != reference.catalog_page
        or reservation.marker_name != reference.marker_name
    ):
        return False
    return reference.queue_scope == "shared" or reservation.home_machine == reference.home_machine


def _recheck_missing_ready_marker(
    cfg: object,
    reference: ReadyMarkerRef,
) -> dict[str, Any] | ReadyClassificationResult:
    """Recheck a missing marker against Task truth while its route is stable."""
    route_key = routes.route_key(reference.queue_scope, reference.home_machine)
    lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
        task_file = task_path(cfg.shared_root, reference.task_id)
        if not task_file.exists():
            return ReadyClassificationResult("permanently_stale", "task_missing")
        try:
            task = TaskRecord.from_dict(read_json(task_file))
        except (KeyError, TypeError, ValueError):
            return ReadyClassificationResult("corrupt", "task_invalid")
        if _is_ready_publication_pending(cfg, reference, task):
            return ReadyClassificationResult("temporarily_unavailable", "marker_publication_pending", task)
        if reference.generation != task.ready_generation:
            return ReadyClassificationResult("permanently_stale", "generation_superseded", task)
        if reference.queue_scope != task.placement_runtime.get(
            "queue_scope"
        ) or reference.home_machine != task.placement_policy.get("home_machine"):
            return ReadyClassificationResult("corrupt", "route_mismatch", task)
        if task.state.get("projection") != "queued" or task.claim_control.get("active_claim"):
            return ReadyClassificationResult("permanently_stale", "task_not_queued", task)
        if (
            task.control.get("cleanup_operation_id")
            or task.control.get("cleanup_state")
            or task.control.get("cancellation_requested_at")
        ):
            return ReadyClassificationResult("permanently_stale", "task_controlled", task)
        try:
            return read_json(routes.marker_path(cfg.shared_root, reference))["ready_marker"]
        except FileNotFoundError:
            partition_path = routes.partition_record_path(
                cfg.shared_root,
                reference.queue_scope,
                reference.home_machine,
                reference.partition,
            )
            try:
                slots = read_json(partition_path)["ready_partition"]["slots"]
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                return ReadyClassificationResult("corrupt", "marker_missing_unindexed", task)
            if isinstance(slots, list) and reference.marker_name in slots:
                return ReadyClassificationResult("corrupt", "marker_missing_indexed", task)
            return ReadyClassificationResult("corrupt", "marker_missing_unindexed", task)
        except (KeyError, TypeError, ValueError):
            return ReadyClassificationResult("corrupt", "marker_invalid", task)


def classify_ready_marker(
    cfg: object,
    reference: ReadyMarkerRef,
) -> ReadyClassificationResult:
    """Classify one advisory marker against authoritative Task and Submission truth."""
    if reference.generation <= 0 or not reference.task_id:
        return ReadyClassificationResult("corrupt", "marker_identity_invalid")
    try:
        marker = read_json(routes.marker_path(cfg.shared_root, reference))["ready_marker"]
    except FileNotFoundError:
        marker = _recheck_missing_ready_marker(cfg, reference)
        if isinstance(marker, ReadyClassificationResult):
            return marker
    except (KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "marker_invalid")
    try:
        common = {
            "schema_version",
            "task_id",
            "generation",
            "source_transition",
            "source_revision",
            "target_revision",
            "queue_scope",
            "home_machine",
            "group_name",
            "submission_operation_id",
            "created_at",
        }
        is_legacy = set(marker) == common | {"requested_gpus"}
        is_gpu = set(marker) == common | {"lane", "requested_gpus"} and marker.get("lane") == "gpu"
        is_cpu = set(marker) == common | {"lane", "requested_cpus"} and marker.get("lane") == "cpu"
        if not (is_legacy or is_gpu or is_cpu) or marker["schema_version"] != state.READY_PROTOCOL_VERSION:
            raise ValueError("ready marker schema is invalid.")
        if (
            marker["task_id"] != reference.task_id
            or marker["generation"] != reference.generation
            or marker["queue_scope"] != reference.queue_scope
            or marker["home_machine"] != reference.home_machine
        ):
            raise ValueError("ready marker identity is inconsistent.")
    except (KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "marker_invalid")
    task_file = task_path(cfg.shared_root, reference.task_id)
    if not task_file.exists():
        return ReadyClassificationResult("permanently_stale", "task_missing")
    try:
        task = TaskRecord.from_dict(read_json(task_file))
    except (KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "task_invalid")
    if _is_ready_publication_pending(cfg, reference, task):
        return ReadyClassificationResult("temporarily_unavailable", "marker_publication_pending", task)
    if reference.generation != task.ready_generation:
        return ReadyClassificationResult("permanently_stale", "generation_superseded", task)
    if reference.queue_scope != task.placement_runtime.get(
        "queue_scope"
    ) or reference.home_machine != task.placement_policy.get("home_machine"):
        return ReadyClassificationResult("corrupt", "route_mismatch", task)
    if task.state.get("projection") != "queued" or task.claim_control.get("active_claim"):
        return ReadyClassificationResult("permanently_stale", "task_not_queued", task)
    if (
        task.control.get("cleanup_operation_id")
        or task.control.get("cleanup_state")
        or task.control.get("cancellation_requested_at")
    ):
        return ReadyClassificationResult("permanently_stale", "task_controlled", task)
    operation_id = task.submission_operation_id
    if not operation_id:
        return ReadyClassificationResult("corrupt", "submission_identity_missing", task)
    operation_file = submission_path(cfg.shared_root, operation_id)
    if not operation_file.exists():
        return ReadyClassificationResult("corrupt", "submission_missing", task)
    try:
        submission_state = read_json(operation_file)["submission"]["state"]
    except (KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "submission_invalid", task)
    if submission_state in {"preparing", "committing", "blocked"}:
        return ReadyClassificationResult("temporarily_unavailable", f"submission_{submission_state}", task)
    if submission_state == "aborted":
        return ReadyClassificationResult("permanently_stale", "submission_aborted", task)
    if submission_state != "committed":
        return ReadyClassificationResult("corrupt", "submission_state_invalid", task)
    if task.group_name:
        path = group_path(cfg.shared_root, task.group_name)
        if not path.exists():
            return ReadyClassificationResult("corrupt", "group_missing", task)
        try:
            group = read_json(path)["group"]
        except (KeyError, TypeError, ValueError):
            return ReadyClassificationResult("corrupt", "group_invalid", task)
        if group.get("dispatch_state") != "active":
            return ReadyClassificationResult("temporarily_unavailable", "group_paused", task)
    from ..dependencies import dependency_gate

    gate = dependency_gate(cfg, task)
    if gate.state == "invalid":
        return ReadyClassificationResult("corrupt", "dependency_invalid", task)
    if gate.state != "ready":
        return ReadyClassificationResult("temporarily_unavailable", f"dependency_{gate.state}", task)
    return ReadyClassificationResult("claimable", "eligible_truth", task)


def _task_should_have_ready_marker(task: TaskRecord) -> bool:
    return (
        task.state.get("projection") == "queued"
        and not task.claim_control.get("active_claim")
        and not task.control.get("cleanup_operation_id")
        and not task.control.get("cleanup_state")
        and not task.control.get("cancellation_requested_at")
    )


def _repair_task_ready_projection(cfg: object, task_id: str) -> tuple[int, int]:
    """Repair one Task projection under its authority lock."""
    from ..locks import task_writer_lock
    from ..tasks import load_task, save_task

    repaired = 0
    stale_removed = 0
    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        return repaired, stale_removed
    with task_writer_lock(cfg, task_id, initial.group_name):
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            return repaired, stale_removed
        reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
        classification = (
            classify_ready_marker(cfg, reference).classification
            if reference is not None and routes.is_reference_indexed(cfg, reference)
            else None
        )
        if _task_should_have_ready_marker(task):
            if classification in {"claimable", "temporarily_unavailable"}:
                return repaired, stale_removed
            old_generation, _new_generation = prepare_ready_transition(cfg, task, "ready_index_rebuild")
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            retire_previous_ready_generation(cfg, old_generation, task)
            return 1, int(old_generation > 0)
        if reference is not None and delete_ready_marker(cfg, task.task_id, task.ready_generation):
            stale_removed += 1
        return repaired, stale_removed


def _load_build_page(cfg: object, build_id: str, page: int) -> list[str]:
    record = read_json(_build_page_path(cfg, build_id, page))["ready_build_page"]
    task_ids = record.get("task_ids")
    if (
        record.get("schema_version") != state.READY_PROTOCOL_VERSION
        or record.get("build_id") != build_id
        or record.get("page") != page
        or not isinstance(task_ids, list)
        or len(task_ids) > READY_BUILD_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in task_ids)
    ):
        raise ValueError("ready build watermark page is invalid.")
    return task_ids


def _advance_build_cursor(
    cursor: dict[str, Any],
    *,
    item_count: int,
    page_count: int,
) -> bool:
    cursor["offset"] += 1
    if cursor["offset"] < item_count:
        return False
    cursor["page"] += 1
    cursor["offset"] = 0
    return cursor["page"] >= page_count


def _active_incompatible_writers(cfg: object) -> list[str]:
    """Return recently active machine agents that did not advertise ready-v1."""
    incompatible: list[str] = []
    machines = shared_paths(cfg.shared_root)["machines"]
    try:
        entries = os.scandir(machines)
    except FileNotFoundError:
        return incompatible
    now = datetime.now(timezone.utc)
    with entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            path = Path(entry.path) / "state" / "agent.json"
            try:
                agent = read_json(path)["agent"]
                heartbeat = datetime.fromisoformat(agent["heartbeat_at"].replace("Z", "+00:00"))
                interval = float(agent["heartbeat_interval_seconds"])
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if agent.get("observed_state") not in {"active", "idle"}:
                continue
            if (now - heartbeat).total_seconds() > max(30.0, interval * 3.0):
                continue
            if agent.get("writer_capability") != state.READY_WRITER_CAPABILITY:
                incompatible.append(entry.name)
    return sorted(incompatible)


def _audit_task_ready_projection(cfg: object, task_id: str) -> str | None:
    try:
        task = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
    except FileNotFoundError:
        return None
    except (KeyError, TypeError, ValueError):
        return f"task_invalid:{task_id}"
    reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
    if _task_should_have_ready_marker(task):
        if reference is None:
            return f"marker_missing:{task_id}"
        if not routes.is_reference_indexed(cfg, reference):
            return f"marker_unindexed:{task_id}"
        result = classify_ready_marker(cfg, reference)
        if result.classification not in {"claimable", "temporarily_unavailable"}:
            return f"marker_{result.classification}:{task_id}:{result.reason}"
    elif reference is not None:
        return f"marker_stale:{task_id}"
    return None


def ready_task_projection_issue(cfg: object, task_id: str) -> str | None:
    """Return the ready projection defect for one authoritative Task, if any."""
    return _audit_task_ready_projection(cfg, task_id)


def advance_ready_index_build(
    cfg: object,
    *,
    max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance at most ``max_tasks`` durable rebuild or audit records."""
    if type(max_tasks) is not int or not 1 <= max_tasks <= READY_BUILD_PAGE_SIZE:
        raise ValueError(f"max_tasks must be between 1 and {READY_BUILD_PAGE_SIZE}.")
    begin_ready_index_build(cfg)
    # A rebuild can repair Task truth.  Hold the schema fence before the ready
    # state lock so its nested Task writer follows Schema -> state -> Group -> Task.
    with schema_writer_lock(cfg):
        with exclusive(state.state_lock_path(cfg)):
            path = ready_state_path(cfg.shared_root)
            value, record = state.read_state_record(cfg)
            if record["state"] != "building":
                return record
            build = record.get("build")
            if not isinstance(build, dict):
                state.degrade_state_record(record, "build_state_missing")
                state.commit_state_under_lock(path, value, record)
                return record
            watermark = build.get("watermark", {})
            page_count = watermark.get("page_count")
            if type(page_count) is not int or page_count < 0 or not watermark.get("is_complete"):
                state.degrade_state_record(record, "build_watermark_invalid")
                state.commit_state_under_lock(path, value, record)
                return record
            phase = build.get("phase")
            cursor_name = {
                "backfill": "cursor",
                "audit": "audit_cursor",
                "primary-rebuild": "primary_cursor",
            }.get(phase)
            if cursor_name is None:
                state.degrade_state_record(record, f"build_phase_invalid:{phase}")
                state.commit_state_under_lock(path, value, record)
                return record
            cursor = build.get(cursor_name)
            if not isinstance(cursor, dict):
                state.degrade_state_record(record, f"build_cursor_invalid:{cursor_name}")
                state.commit_state_under_lock(path, value, record)
                return record
            processed_now = 0
            try:
                if phase == "primary-rebuild":
                    begin_primary_ready_index_rebuild(cfg, build["build_id"])
                while processed_now < max_tasks and cursor["page"] < page_count:
                    task_ids = _load_build_page(cfg, build["build_id"], cursor["page"])
                    if cursor["offset"] >= len(task_ids):
                        cursor["page"] += 1
                        cursor["offset"] = 0
                        continue
                    task_id = task_ids[cursor["offset"]]
                    if phase == "backfill":
                        repaired, stale_removed = _repair_task_ready_projection(cfg, task_id)
                        build["repaired"] += repaired
                        build["stale_removed"] += stale_removed
                        build["processed"] += 1
                    elif phase == "audit":
                        issue = _audit_task_ready_projection(cfg, task_id)
                        if issue is not None:
                            state.degrade_state_record(record, issue)
                            break
                    else:
                        try:
                            task = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
                        except FileNotFoundError:
                            task = None
                        if task is not None and _task_should_have_ready_marker(task):
                            reference = routes.reference_for_generation(
                                cfg,
                                task.task_id,
                                task.ready_generation,
                            )
                            if reference is None:
                                state.degrade_state_record(record, f"marker_missing:{task.task_id}")
                                break
                            primary_candidates.rebuild_primary_ready_candidate(cfg, build["build_id"], task, reference)
                    processed_now += 1
                    _advance_build_cursor(cursor, item_count=len(task_ids), page_count=page_count)
                if record["state"] == "building" and cursor["page"] >= page_count:
                    if phase == "backfill":
                        build["phase"] = "audit"
                    elif phase == "audit":
                        build["phase"] = "primary-rebuild"
                        build["primary_cursor"] = {"page": 0, "offset": 0}
                    else:
                        primary_candidates.complete_primary_ready_index_rebuild(cfg, build["build_id"])
                        state.assert_ready_writer_compatible(cfg)
                        incompatible = _active_incompatible_writers(cfg)
                        if incompatible:
                            state.degrade_state_record(
                                record,
                                "incompatible_active_writers:" + ",".join(incompatible),
                            )
                        else:
                            record["state"] = "active"
                            record["degraded_reasons"] = []
                            build["phase"] = "completed"
                            build["completed_at"] = utc_now()
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                state.degrade_state_record(record, f"build_failed:{type(exc).__name__}:{exc}")
            state.commit_state_under_lock(path, value, record)
            return record


def repair_ready_index(
    cfg: object,
    *,
    max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Start degraded recovery and advance one bounded repair slice."""
    begin_ready_index_build(cfg, is_repair=True)
    return advance_ready_index_build(cfg, max_tasks=max_tasks)
