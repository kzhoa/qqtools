"""Cursor-backed traversal of durable ready routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..locks import exclusive
from ..paths import shared_paths
from ..records import validate_identifier
from ..store import atomic_replace, read_json
from ..work_budget import SliceBudget
from . import primary_candidates, routes, state
from .diagnostics import ReadyDiagnostic, storage_diagnostic
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


_CATALOG_FIELDS = frozenset({"schema_version", "route", "page", "partitions", "successor", "revision"})
_PARTITION_FIELDS = frozenset(
    {"schema_version", "route", "partition", "catalog_page", "slots", "sealed", "successor", "revision"}
)
_NO_INVALID_PARTITION = object()


class _CatalogValidationError(ValueError):
    """Carry field-level catalog differences to the persisted diagnostic."""

    def __init__(
        self,
        *,
        missing_fields: list[str] | None = None,
        unexpected_fields: list[str] | None = None,
        mismatch_fields: list[str] | None = None,
        invalid_partition: object = _NO_INVALID_PARTITION,
    ) -> None:
        self.missing_fields = sorted(set(missing_fields or []))
        self.unexpected_fields = sorted(set(unexpected_fields or []))
        self.mismatch_fields = sorted(set(mismatch_fields or []))
        self.invalid_partition = invalid_partition
        super().__init__("ready catalog validation failed")


class _PartitionValidationError(ValueError):
    """Carry field-level partition differences to the persisted diagnostic."""

    def __init__(
        self,
        *,
        missing_fields: list[str] | None = None,
        unexpected_fields: list[str] | None = None,
        mismatch_fields: list[str] | None = None,
    ) -> None:
        self.missing_fields = sorted(set(missing_fields or []))
        self.unexpected_fields = sorted(set(unexpected_fields or []))
        self.mismatch_fields = sorted(set(mismatch_fields or []))
        super().__init__("ready partition validation failed")


def _read_and_validate_catalog(
    page_path: Path,
    route_key: str,
    page_number: int,
) -> tuple[list[str], int | None]:
    """Read one catalog and return its validated partition names and successor."""
    value = read_json(page_path)
    if not isinstance(value, dict):
        raise _CatalogValidationError(mismatch_fields=["ready_catalog"])
    if "ready_catalog" not in value:
        raise _CatalogValidationError(
            missing_fields=["ready_catalog"],
            unexpected_fields=sorted(set(value)),
        )
    catalog = value["ready_catalog"]
    if not isinstance(catalog, dict):
        raise _CatalogValidationError(mismatch_fields=["ready_catalog"])

    missing_fields = sorted(_CATALOG_FIELDS - set(catalog))
    unexpected_fields = sorted((set(value) - {"ready_catalog"}) | (set(catalog) - _CATALOG_FIELDS))
    mismatch_fields: list[str] = []
    invalid_partition: object = _NO_INVALID_PARTITION
    if "schema_version" in catalog and (
        type(catalog["schema_version"]) is not int or catalog["schema_version"] != state.READY_PROTOCOL_VERSION
    ):
        mismatch_fields.append("schema_version")
    if "route" in catalog and catalog["route"] != route_key:
        mismatch_fields.append("route")
    if "page" in catalog and (type(catalog["page"]) is not int or catalog["page"] != page_number):
        mismatch_fields.append("page")
    if "revision" in catalog and (type(catalog["revision"]) is not int or catalog["revision"] < 0):
        mismatch_fields.append("revision")

    partitions = catalog.get("partitions")
    if "partitions" in catalog:
        if not isinstance(partitions, list):
            mismatch_fields.append("partitions")
        elif len(partitions) > routes.READY_CATALOG_PAGE_SIZE:
            mismatch_fields.append("partitions")
        elif any(not isinstance(partition, str) or _invalid_identifier(partition) for partition in partitions):
            mismatch_fields.append("partition")
            invalid_partition = next(
                partition
                for partition in partitions
                if not isinstance(partition, str) or _invalid_identifier(partition)
            )

    successor = catalog.get("successor")
    if "successor" in catalog and successor is not None and (type(successor) is not int or successor < 0):
        mismatch_fields.append("successor")

    if missing_fields or unexpected_fields or mismatch_fields:
        raise _CatalogValidationError(
            missing_fields=missing_fields,
            unexpected_fields=unexpected_fields,
            mismatch_fields=mismatch_fields,
            invalid_partition=invalid_partition,
        )
    return partitions, successor


def _invalid_identifier(value: str) -> bool:
    try:
        validate_identifier(value, "ready partition")
    except (TypeError, ValueError):
        return True
    return False


def _catalog_diagnostic(
    route_key: str,
    page_number: int,
    stage: str,
    exception: BaseException,
) -> ReadyDiagnostic:
    """Build a catalog diagnostic without exposing validation exception text."""
    invalid_partition = getattr(exception, "invalid_partition", _NO_INVALID_PARTITION)
    protocol_exception = ValueError() if isinstance(exception, _CatalogValidationError) else exception
    if invalid_partition is not _NO_INVALID_PARTITION:
        return storage_diagnostic(
            "partition_invalid",
            route=route_key,
            location=invalid_partition,
            stage=stage,
            exception=protocol_exception,
            unexpected_fields=getattr(exception, "unexpected_fields", None),
            missing_fields=getattr(exception, "missing_fields", None),
            mismatch_fields=getattr(exception, "mismatch_fields", None),
        )
    return storage_diagnostic(
        "catalog_invalid",
        route=route_key,
        location=page_number,
        stage=stage,
        exception=protocol_exception,
        unexpected_fields=getattr(exception, "unexpected_fields", None),
        missing_fields=getattr(exception, "missing_fields", None),
        mismatch_fields=getattr(exception, "mismatch_fields", None),
    )


def _read_and_validate_partition(
    partition_path: Path,
    route_key: str,
    page_number: int,
    partition_name: str,
) -> list[str]:
    """Read one partition and return its validated marker slots."""
    value = read_json(partition_path)
    if not isinstance(value, dict):
        raise _PartitionValidationError(mismatch_fields=["ready_partition"])
    if "ready_partition" not in value:
        raise _PartitionValidationError(
            missing_fields=["ready_partition"],
            unexpected_fields=sorted(set(value)),
        )
    partition = value["ready_partition"]
    if not isinstance(partition, dict):
        raise _PartitionValidationError(mismatch_fields=["ready_partition"])

    missing_fields = sorted(_PARTITION_FIELDS - set(partition))
    unexpected_fields = sorted((set(value) - {"ready_partition"}) | (set(partition) - _PARTITION_FIELDS))
    mismatch_fields: list[str] = []
    if "schema_version" in partition and (
        type(partition["schema_version"]) is not int or partition["schema_version"] != state.READY_PROTOCOL_VERSION
    ):
        mismatch_fields.append("schema_version")
    if "route" in partition and partition["route"] != route_key:
        mismatch_fields.append("route")
    if "partition" in partition and partition["partition"] != partition_name:
        mismatch_fields.append("partition")
    if "catalog_page" in partition and (
        type(partition["catalog_page"]) is not int or partition["catalog_page"] != page_number
    ):
        mismatch_fields.append("catalog_page")
    slots = partition.get("slots")
    if "slots" in partition and (
        not isinstance(slots, list)
        or len(slots) > routes.READY_PARTITION_SLOTS
        or not all(isinstance(name, str) for name in slots)
    ):
        mismatch_fields.append("slots")
    if "sealed" in partition and type(partition["sealed"]) is not bool:
        mismatch_fields.append("sealed")
    successor = partition.get("successor")
    if (
        "successor" in partition
        and successor is not None
        and (not isinstance(successor, str) or _invalid_identifier(successor))
    ):
        mismatch_fields.append("successor")
    if "revision" in partition and (type(partition["revision"]) is not int or partition["revision"] < 0):
        mismatch_fields.append("revision")

    if missing_fields or unexpected_fields or mismatch_fields:
        raise _PartitionValidationError(
            missing_fields=missing_fields,
            unexpected_fields=unexpected_fields,
            mismatch_fields=mismatch_fields,
        )
    return slots


def _partition_diagnostic(
    route_key: str,
    partition_name: str,
    exception: BaseException,
) -> ReadyDiagnostic:
    """Build a partition diagnostic without exposing validation exception text."""
    protocol_exception = ValueError() if isinstance(exception, _PartitionValidationError) else exception
    return storage_diagnostic(
        "partition_invalid",
        route=route_key,
        location=partition_name,
        stage="partition_schema",
        exception=protocol_exception,
        unexpected_fields=getattr(exception, "unexpected_fields", None),
        missing_fields=getattr(exception, "missing_fields", None),
        mismatch_fields=getattr(exception, "mismatch_fields", None),
    )


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
) -> ReadyMarkerRef | None:
    stem = marker_name[:-5] if marker_name.endswith(".json") else marker_name
    task_id, separator, generation_value = stem.rpartition(".")
    try:
        validate_identifier(task_id, "slot task_id")
        generation = int(generation_value) if separator else -1
    except (TypeError, ValueError):
        return None
    if generation <= 0 or marker_name != f"{task_id}.{generation}.json":
        return None
    home_machine = cfg.machine_name
    if scope == "shared":
        provisional = ReadyMarkerRef(task_id, generation, scope, home_machine, partition, catalog_page, marker_name)
        try:
            marker = read_json(routes.marker_path(cfg.shared_root, provisional))["ready_marker"]
            if isinstance(marker.get("home_machine"), str):
                home_machine = validate_identifier(marker["home_machine"], "marker home_machine")
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
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
            partitions, _successor = _read_and_validate_catalog(page_path, route_key, page_number)
            return partition_name in partitions
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
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
    try:
        page_path.stat()
    except FileNotFoundError:
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
    except OSError as exc:
        state.mark_ready_index_degraded(
            cfg,
            storage_diagnostic(
                "catalog_invalid",
                route=route_key,
                location=page_number,
                stage="catalog_read",
                exception=exc,
            ),
        )
        return None, False
    try:
        partitions, successor = _read_and_validate_catalog(page_path, route_key, page_number)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        state.mark_ready_index_degraded(
            cfg,
            _catalog_diagnostic(route_key, page_number, "catalog_schema", exc),
        )
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
        names = sorted(_read_and_validate_partition(partition_path, route_key, page_number, partition_name))
    except FileNotFoundError as exc:
        if _is_partition_referenced_under_route_lock(cfg, route_key, page_path, page_number, partition_name):
            state.mark_ready_index_degraded(
                cfg,
                storage_diagnostic(
                    "partition_missing",
                    route=route_key,
                    location=partition_name,
                    stage="partition_read",
                    exception=exc,
                ),
            )
            return None, False
        names = []
    except (KeyError, OSError, TypeError, ValueError) as exc:
        state.mark_ready_index_degraded(cfg, _partition_diagnostic(route_key, partition_name, exc))
        return None, False
    for marker_name in names:
        if after_name is not None and marker_name <= after_name:
            continue
        reference = _reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name)
        if reference is None:
            state.mark_ready_index_degraded(
                cfg,
                storage_diagnostic(
                    "partition_invalid",
                    route=route_key,
                    location=partition_name,
                    stage="slot_identity",
                    mismatch_fields=["marker_name"],
                ),
            )
            return None, False
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
        try:
            page_path.stat()
        except FileNotFoundError:
            if page_number != 0:
                page_number = 0
                partition_name = None
                after_name = None
                continue
            return ReadyPeek(None, current)
        except OSError as exc:
            state.mark_ready_index_degraded(
                cfg,
                storage_diagnostic(
                    "catalog_invalid",
                    route=route_key,
                    location=page_number,
                    stage="catalog_read",
                    exception=exc,
                ),
            )
            return ReadyPeek(None, current, unresolved=True)
        if not budget.can_start_operation():
            return ReadyPeek(None, progress_cursor, exhausted=True)
        budget.consume_operation()
        try:
            partitions, successor = _read_and_validate_catalog(page_path, route_key, page_number)
        except FileNotFoundError as exc:
            state.mark_ready_index_degraded(
                cfg,
                storage_diagnostic(
                    "catalog_invalid",
                    route=route_key,
                    location=page_number,
                    stage="catalog_read",
                    exception=exc,
                ),
            )
            return ReadyPeek(None, current, unresolved=True)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            state.mark_ready_index_degraded(
                cfg,
                _catalog_diagnostic(route_key, page_number, "catalog_schema", exc),
            )
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
                names = sorted(_read_and_validate_partition(partition_path, route_key, page_number, partition_name))
            except FileNotFoundError as exc:
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                if _is_partition_referenced_under_route_lock(cfg, route_key, page_path, page_number, partition_name):
                    state.mark_ready_index_degraded(
                        cfg,
                        storage_diagnostic(
                            "partition_missing",
                            route=route_key,
                            location=partition_name,
                            stage="partition_read",
                            exception=exc,
                        ),
                    )
                    return ReadyPeek(None, current, unresolved=True)
                names = []
            except (KeyError, OSError, TypeError, ValueError) as exc:
                state.mark_ready_index_degraded(cfg, _partition_diagnostic(route_key, partition_name, exc))
                return ReadyPeek(None, current, unresolved=True)
            for marker_name in names:
                if after_name is not None and marker_name <= after_name:
                    continue
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                reference = _reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name)
                if reference is None:
                    state.mark_ready_index_degraded(
                        cfg,
                        storage_diagnostic(
                            "partition_invalid",
                            route=route_key,
                            location=partition_name,
                            stage="slot_identity",
                            mismatch_fields=["marker_name"],
                        ),
                    )
                    return ReadyPeek(None, current, unresolved=True)
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
                reference = _reference_from_slot(cfg, queue_scope, page_number, partition_name, marker_name)
                if reference is None:
                    state.mark_ready_index_degraded(
                        cfg,
                        storage_diagnostic(
                            "partition_invalid",
                            route=route_key,
                            location=partition_name,
                            stage="slot_identity",
                            mismatch_fields=["marker_name"],
                        ),
                    )
                    continue
                references.append(reference)
    return references
