"""Cursor-backed traversal of durable ready routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..locks import exclusive
from ..paths import shared_paths
from ..records import validate_identifier
from ..store import atomic_replace, read_json
from ..work_budget import SliceBudget
from . import primary_candidates, routes, state
from .records import ReadyMarkerRef, ReadyScope


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
