"""Durable generation-safe ready liveness projection."""

from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .locks import exclusive, schema_lock
from .paths import group_path, ready_state_path, shared_paths, submission_path, task_path
from .records import TaskRecord, normalize_group_record, utc_now, validate_identifier
from .store import atomic_replace, iter_json, read_json
from .work_budget import SliceBudget

READY_PROTOCOL_VERSION = 1
READY_WRITER_CAPABILITY = "ready-v1"
PRIMARY_READY_PROTOCOL_VERSION = 1
READY_PARTITION_SLOTS = 64
READY_CATALOG_PAGE_SIZE = 64
READY_BUILD_PAGE_SIZE = 64
ReadyScope = Literal["home", "shared"]
ReadyClassification = Literal[
    "claimable", "temporarily_unavailable", "permanently_stale", "corrupt"
]
ReadyIndexState = Literal["absent", "building", "active", "degraded"]


class ReadyProbeBudgetExhausted(RuntimeError):
    """Raised when a bounded ready revision read cannot start safely."""


@dataclass(frozen=True, slots=True)
class ReadyMarkerRef:
    task_id: str
    generation: int
    queue_scope: ReadyScope
    home_machine: str
    partition: str
    catalog_page: int
    marker_name: str

    @property
    def identity(self) -> str:
        return f"{self.task_id}.{self.generation}"


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


def ensure_ready_layout(cfg: object) -> None:
    """Create the additive ready layout without activating ready-only scheduling."""
    paths = shared_paths(cfg.shared_root)
    for name in (
        "ready", "ready_home", "ready_shared", "ready_catalogs", "ready_reservations",
        "ready_cursors", "ready_builds", "ready_locks",
        "ready_primary",
    ):
        paths[name].mkdir(parents=True, exist_ok=True)
    state_path = ready_state_path(cfg.shared_root)
    if not state_path.exists():
        atomic_replace(state_path, {
            "ready_index": {
                "schema_version": READY_PROTOCOL_VERSION,
                "state": "absent",
                "writer_capability": None,
                "revision": 0,
                "build": None,
                "updated_at": utc_now(),
                "degraded_reasons": [],
            }
        })


def read_ready_index_state(cfg: object) -> ReadyIndexState:
    """Return the scheduling gate for this project's ready projection."""
    path = ready_state_path(cfg.shared_root)
    if not path.exists():
        return "absent"
    try:
        record = read_json(path)["ready_index"]
        state = record["state"]
        if record["schema_version"] != READY_PROTOCOL_VERSION:
            return "degraded"
        if state not in {"absent", "building", "active", "degraded"}:
            return "degraded"
        if (
            state in {"building", "active"}
            and record.get("writer_capability") != READY_WRITER_CAPABILITY
        ):
            return "degraded"
        return state
    except (KeyError, TypeError, ValueError):
        return "degraded"


def _state_lock_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_locks"] / "state.lock"


def _read_ready_state_record(cfg: object) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(ready_state_path(cfg.shared_root))
    record = value["ready_index"]
    if record.get("schema_version") != READY_PROTOCOL_VERSION:
        raise ValueError("ready index state schema is unsupported.")
    if record.get("state") not in {"absent", "building", "active", "degraded"}:
        raise ValueError("ready index state is invalid.")
    record.setdefault("writer_capability", None)
    record.setdefault("revision", 0)
    record.setdefault("build", None)
    record.setdefault("degraded_reasons", [])
    if type(record["revision"]) is not int or record["revision"] < 0:
        raise ValueError("ready index revision is invalid.")
    return value, record


def read_ready_index_status(cfg: object) -> dict[str, Any]:
    """Return the durable build, cursor, watermark, and degradation status."""
    try:
        _value, record = _read_ready_state_record(cfg)
        return record
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return {
            "schema_version": READY_PROTOCOL_VERSION,
            "state": "degraded",
            "writer_capability": None,
            "revision": 0,
            "build": None,
            "degraded_reasons": ["state_invalid"],
            "updated_at": None,
        }


def _commit_ready_state(path: Path, value: dict[str, Any], record: dict[str, Any]) -> None:
    record["revision"] += 1
    record["updated_at"] = utc_now()
    atomic_replace(path, value)


def _schema_capability_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def _install_writer_capability_gate(cfg: object) -> None:
    """Make pre-ready schema readers reject the root before they can mutate Tasks."""
    path = _schema_capability_path(cfg)
    value = read_json(path)
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise RuntimeError("qexp schema/version.json is malformed.")
    capabilities = schema.get("writer_capabilities")
    if capabilities is None:
        schema["writer_capabilities"] = [READY_WRITER_CAPABILITY]
    elif (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or READY_WRITER_CAPABILITY not in capabilities
    ):
        raise RuntimeError("qexp schema writer capability gate is incompatible.")
    else:
        return
    atomic_replace(path, value)


def assert_ready_writer_compatible(
    cfg: object, writer_capability: str | None = READY_WRITER_CAPABILITY,
) -> None:
    """Reject an incompatible writer before authoritative Task mutation."""
    state = read_ready_index_state(cfg)
    if state == "absent":
        return
    try:
        _value, record = _read_ready_state_record(cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ready index state is invalid; Task mutation is disabled.") from exc
    required = record.get("writer_capability")
    if required != READY_WRITER_CAPABILITY or writer_capability != required:
        raise RuntimeError(
            f"ready index requires writer capability {required!r}; "
            f"writer declared {writer_capability!r}."
        )
    try:
        schema = read_json(_schema_capability_path(cfg))["schema"]
        capabilities = schema["writer_capabilities"]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ready writer schema capability gate is missing.") from exc
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or READY_WRITER_CAPABILITY not in capabilities
    ):
        raise RuntimeError("ready writer schema capability gate is incompatible.")


def _build_root(cfg: object, build_id: str) -> Path:
    validate_identifier(build_id, "ready build id")
    return shared_paths(cfg.shared_root)["ready_builds"] / build_id


def _build_page_path(cfg: object, build_id: str, page: int) -> Path:
    return _build_root(cfg, build_id) / "watermark" / f"{page:016d}.json"


def _write_build_page(
    cfg: object, build_id: str, page: int, task_ids: list[str],
) -> None:
    atomic_replace(
        _build_page_path(cfg, build_id, page),
        {"ready_build_page": {
            "schema_version": READY_PROTOCOL_VERSION,
            "build_id": build_id,
            "page": page,
            "task_ids": list(task_ids),
        }},
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
    ensure_ready_layout(cfg)
    state = read_ready_index_state(cfg)
    if state == "active" or (state == "degraded" and not is_repair):
        return read_ready_index_status(cfg)
    if state in {"absent", "degraded"}:
        with schema_lock(cfg.shared_root):
            with exclusive(_state_lock_path(cfg)):
                path = ready_state_path(cfg.shared_root)
                value, record = _read_ready_state_record(cfg)
                state = record["state"]
                if state == "active" or (state == "degraded" and not is_repair):
                    return record
                if state in {"absent", "degraded"}:
                    _install_writer_capability_gate(cfg)
                    build_id = uuid.uuid4().hex
                    if state == "degraded" and is_repair:
                        _reset_ready_projection_for_repair(cfg, build_id)
                    record["state"] = "building"
                    record["writer_capability"] = READY_WRITER_CAPABILITY
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
                    _commit_ready_state(path, value, record)
    with exclusive(_state_lock_path(cfg)):
        path = ready_state_path(cfg.shared_root)
        value, record = _read_ready_state_record(cfg)
        state = record["state"]
        if state == "active":
            return record
        if state == "degraded" and not is_repair:
            return record
        build = record.get("build")
        if not isinstance(build, dict):
            raise RuntimeError("ready index build state is missing.")
        if not build.get("watermark", {}).get("is_complete"):
            _capture_build_watermark(cfg, record)
            _commit_ready_state(path, value, record)
        return record


def mark_ready_index_degraded(cfg: object, reason: str) -> None:
    """Fail closed after detecting a corrupt active projection."""
    path = ready_state_path(cfg.shared_root)
    try:
        with exclusive(_state_lock_path(cfg)):
            value, record = _read_ready_state_record(cfg)
            _degrade_ready_record(record, reason)
            _commit_ready_state(path, value, record)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return


def _degrade_ready_record(record: dict[str, Any], reason: str) -> None:
    reasons = record.get("degraded_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    if reason not in reasons:
        reasons.append(reason)
    record["state"] = "degraded"
    record["degraded_reasons"] = reasons


def _route_key(scope: ReadyScope, home_machine: str) -> str:
    validate_identifier(home_machine, "home_machine")
    return f"home.{home_machine}" if scope == "home" else "shared"


def _primary_route_key(scope: ReadyScope, machine: str) -> str:
    validate_identifier(machine, "machine")
    return f"{scope}.{machine}"


def _primary_index_state_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_primary"] / "state.json"


def _primary_route_path(cfg: object, route_key: str) -> Path:
    return shared_paths(cfg.shared_root)["ready_primary"] / "routes" / route_key


def _primary_candidate_path(cfg: object, route_key: str, identity: str) -> Path:
    validate_identifier(identity.replace(".", "-"), "primary candidate identity")
    return _primary_route_path(cfg, route_key) / f"{identity}.json"


def _primary_index_is_active(cfg: object) -> bool:
    try:
        value = read_json(_primary_index_state_path(cfg))["primary_ready_index"]
        return (
            value.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and value.get("state") == "active"
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def is_primary_ready_index_active(cfg: object) -> bool:
    """Return whether the primary-only candidate projection is usable."""
    return _primary_index_is_active(cfg)


def _route_directory(root: Path, scope: ReadyScope, home_machine: str) -> Path:
    paths = shared_paths(root)
    if scope == "home":
        return paths["ready_home"] / home_machine
    return paths["ready_shared"]


def _reservation_path(root: Path, task_id: str, generation: int) -> Path:
    return shared_paths(root)["ready_reservations"] / f"{task_id}.{generation}.json"


def _allocator_path(root: Path, route_key: str) -> Path:
    return shared_paths(root)["ready"] / "allocators" / f"{route_key}.json"


def _catalog_path(root: Path, route_key: str, page: int) -> Path:
    return shared_paths(root)["ready_catalogs"] / route_key / f"{page:016d}.json"


def _partition_record_path(
    root: Path, scope: ReadyScope, home_machine: str, partition: str,
) -> Path:
    return _route_directory(root, scope, home_machine) / partition / "partition.json"


def _marker_path(root: Path, reference: ReadyMarkerRef) -> Path:
    return (
        _route_directory(root, reference.queue_scope, reference.home_machine)
        / reference.partition
        / reference.marker_name
    )


def _cursor_path(root: Path, project_id: str, machine_name: str, scope: ReadyScope) -> Path:
    validate_identifier(project_id, "project_id")
    validate_identifier(machine_name, "machine_name")
    return shared_paths(root)["ready_cursors"] / f"{project_id}.{machine_name}.{scope}.json"


def _default_cursor(project_id: str, machine_name: str, scope: ReadyScope) -> ReadyCursor:
    return ReadyCursor(project_id, machine_name, scope, 0, None, None, 0)


def load_ready_cursor(
    cfg: object, project_id: str, queue_scope: ReadyScope,
) -> ReadyCursor:
    """Load advisory candidate progress, falling back conservatively on damage."""
    path = _cursor_path(cfg.shared_root, project_id, cfg.machine_name, queue_scope)
    if not path.exists():
        return _default_cursor(project_id, cfg.machine_name, queue_scope)
    try:
        record = read_json(path)["cursor"]
        if set(record) != {
            "schema_version", "project_id", "machine_name", "queue_scope",
            "catalog_page", "partition", "after_name", "revision",
        }:
            raise ValueError("ready cursor schema is invalid.")
        if (
            record["schema_version"] != READY_PROTOCOL_VERSION
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
            project_id, cfg.machine_name, queue_scope, page, record["partition"],
            record["after_name"], record["revision"],
        )
    except (KeyError, TypeError, ValueError):
        return _default_cursor(project_id, cfg.machine_name, queue_scope)


def _save_ready_cursor(cfg: object, cursor: ReadyCursor) -> None:
    atomic_replace(
        _cursor_path(
            cfg.shared_root, cursor.project_id, cursor.machine_name, cursor.queue_scope
        ),
        {"cursor": {
            "schema_version": READY_PROTOCOL_VERSION,
            "project_id": cursor.project_id,
            "machine_name": cursor.machine_name,
            "queue_scope": cursor.queue_scope,
            "catalog_page": (
                None if cursor.catalog_page is None else str(cursor.catalog_page)
            ),
            "partition": cursor.partition,
            "after_name": cursor.after_name,
            "revision": cursor.revision,
        }},
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
        provisional = ReadyMarkerRef(
            task_id, generation, scope, home_machine, partition, catalog_page, marker_name
        )
        try:
            marker = read_json(_marker_path(cfg.shared_root, provisional))["ready_marker"]
            if isinstance(marker.get("home_machine"), str):
                home_machine = marker["home_machine"]
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            pass
    return ReadyMarkerRef(
        task_id, generation, scope, home_machine, partition, catalog_page, marker_name
    )


def next_ready_marker(
    cfg: object,
    project_id: str,
    queue_scope: ReadyScope,
    excluded_identities: set[str] | None = None,
) -> tuple[ReadyMarkerRef | None, bool]:
    """Return and durably advance past one marker without unbounded enumeration."""
    cursor = load_ready_cursor(cfg, project_id, queue_scope)
    route_key = _route_key(queue_scope, cfg.machine_name)
    page_number = cursor.catalog_page or 0
    page_path = _catalog_path(cfg.shared_root, route_key, page_number)
    if not page_path.exists():
        if page_number == 0:
            return None, False
        _save_ready_cursor(
            cfg,
            ReadyCursor(
                project_id, cfg.machine_name, queue_scope, 0, None, None,
                cursor.revision + 1,
            ),
        )
        return None, True
    try:
        catalog = read_json(page_path)["ready_catalog"]
        partitions = catalog["partitions"]
        successor = catalog.get("successor")
        if not isinstance(partitions, list) or not all(
            isinstance(item, str) for item in partitions
        ):
            raise ValueError("ready catalog partitions are invalid.")
        if successor is not None and (
            not isinstance(successor, int) or successor < 0
        ):
            raise ValueError("ready catalog successor is invalid.")
    except (KeyError, TypeError, ValueError):
        mark_ready_index_degraded(cfg, f"catalog_invalid:{route_key}:{page_number}")
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
                project_id, cfg.machine_name, queue_scope, next_page, None, None,
                cursor.revision + 1,
            ),
        )
        return None, has_wrapped
    partition_path = _partition_record_path(
        cfg.shared_root, queue_scope, cfg.machine_name, partition_name
    )
    try:
        partition = read_json(partition_path)["ready_partition"]
        slots = partition["slots"]
        if not isinstance(slots, list) or not all(isinstance(name, str) for name in slots):
            raise ValueError("ready partition slots are invalid.")
        names = sorted(slots)
    except FileNotFoundError:
        lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
        with exclusive(lock_path):
            try:
                current_catalog = read_json(page_path)["ready_catalog"]
                current_partitions = current_catalog["partitions"]
                if (
                    current_catalog.get("schema_version") != READY_PROTOCOL_VERSION
                    or current_catalog.get("route") != route_key
                    or current_catalog.get("page") != page_number
                    or not isinstance(current_partitions, list)
                    or not all(isinstance(item, str) for item in current_partitions)
                ):
                    raise ValueError("ready catalog is invalid.")
                is_still_referenced = partition_name in current_partitions
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                is_still_referenced = True
        if is_still_referenced:
            mark_ready_index_degraded(
                cfg, f"partition_missing:{route_key}:{partition_name}"
            )
            return None, False
        names = []
    except (KeyError, TypeError, ValueError):
        mark_ready_index_degraded(cfg, f"partition_invalid:{route_key}:{partition_name}")
        return None, False
    for marker_name in names:
        if after_name is not None and marker_name <= after_name:
            continue
        reference = _reference_from_slot(
            cfg, queue_scope, page_number, partition_name, marker_name
        )
        if excluded_identities is not None and reference.identity in excluded_identities:
            _save_ready_cursor(
                cfg,
                ReadyCursor(
                    project_id, cfg.machine_name, queue_scope, page_number,
                    partition_name, marker_name, cursor.revision + 1,
                ),
            )
            return None, True
        _save_ready_cursor(
            cfg,
            ReadyCursor(
                project_id, cfg.machine_name, queue_scope, page_number, partition_name,
                marker_name, cursor.revision + 1,
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
            project_id, cfg.machine_name, queue_scope, next_page, next_partition, None,
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
    route_key = _route_key(queue_scope, cfg.machine_name)
    page_number = current.catalog_page or 0
    partition_name = current.partition
    after_name = current.after_name
    progress_cursor = current
    while True:
        page_path = _catalog_path(cfg.shared_root, route_key, page_number)
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
                catalog.get("schema_version") != READY_PROTOCOL_VERSION
                or catalog.get("route") != route_key
                or catalog.get("page") != page_number
                or not isinstance(partitions, list)
                or len(partitions) > READY_CATALOG_PAGE_SIZE
                or not all(isinstance(item, str) for item in partitions)
                or (successor is not None and (not isinstance(successor, int) or successor < 0))
            ):
                raise ValueError("ready catalog is invalid.")
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            mark_ready_index_degraded(cfg, f"catalog_invalid:{route_key}:{page_number}")
            return ReadyPeek(None, current, unresolved=True)

        if partition_name in partitions:
            partition_index = partitions.index(partition_name)
        else:
            partition_index = 0
            partition_name = None
            after_name = None
        while partition_index < len(partitions):
            partition_name = partitions[partition_index]
            partition_path = _partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            if not budget.can_start_operation():
                return ReadyPeek(None, progress_cursor, exhausted=True)
            budget.consume_operation()
            try:
                partition = read_json(partition_path)["ready_partition"]
                slots = partition["slots"]
                if (
                    partition.get("schema_version") != READY_PROTOCOL_VERSION
                    or partition.get("route") != route_key
                    or partition.get("partition") != partition_name
                    or not isinstance(slots, list)
                    or len(slots) > READY_PARTITION_SLOTS
                    or not all(isinstance(name, str) for name in slots)
                ):
                    raise ValueError("ready partition is invalid.")
                names = sorted(slots)
            except FileNotFoundError:
                mark_ready_index_degraded(
                    cfg, f"partition_missing:{route_key}:{partition_name}"
                )
                return ReadyPeek(None, current, unresolved=True)
            except (KeyError, TypeError, ValueError):
                mark_ready_index_degraded(
                    cfg, f"partition_invalid:{route_key}:{partition_name}"
                )
                return ReadyPeek(None, current, unresolved=True)
            for marker_name in names:
                if after_name is not None and marker_name <= after_name:
                    continue
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                reference = _reference_from_slot(
                    cfg, queue_scope, page_number, partition_name, marker_name
                )
                next_cursor = ReadyCursor(
                    project_id, cfg.machine_name, queue_scope, page_number,
                    partition_name, marker_name, current.revision + 1,
                )
                return ReadyPeek(reference, next_cursor)
            progress_cursor = ReadyCursor(
                project_id, cfg.machine_name, queue_scope, page_number,
                partition_name, names[-1] if names else "", current.revision + 1,
            )
            partition_index += 1
            after_name = None

        if successor is not None:
            progress_cursor = ReadyCursor(
                project_id, cfg.machine_name, queue_scope, successor, None, None,
                current.revision + 1,
            )
            page_number = successor
            partition_name = None
            after_name = None
            continue
        return ReadyPeek(
            None,
            ReadyCursor(project_id, cfg.machine_name, queue_scope, 0, None, None,
                        current.revision + 1),
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
    if not _primary_index_is_active(cfg):
        return ReadyPeek(None, current, unresolved=True)
    route_key = _route_key(queue_scope, cfg.machine_name)
    primary_route = _primary_route_key(queue_scope, cfg.machine_name)
    page_number = current.catalog_page or 0
    partition_name = current.partition
    after_name = current.after_name
    progress_cursor = current
    while True:
        page_path = _catalog_path(cfg.shared_root, route_key, page_number)
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
            if page_number == 0 and not _allocator_path(cfg.shared_root, route_key).exists():
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
                slots = read_json(_partition_record_path(
                    cfg.shared_root, queue_scope, cfg.machine_name, partition_name
                ))["ready_partition"]["slots"]
                if not isinstance(slots, list) or not all(isinstance(item, str) for item in slots):
                    raise ValueError("primary partition is invalid.")
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                return ReadyPeek(None, current, unresolved=True)
            for marker_name in sorted(slots):
                if after_name is not None and marker_name <= after_name:
                    continue
                progress_cursor = ReadyCursor(
                    project_id, cfg.machine_name, queue_scope, page_number,
                    partition_name, marker_name, current.revision + 1,
                )
                if not budget.can_start_operation():
                    return ReadyPeek(None, progress_cursor, exhausted=True)
                budget.consume_operation()
                candidate_path = _primary_candidate_path(
                    cfg, primary_route, marker_name.removesuffix(".json")
                )
                if not candidate_path.exists():
                    continue
                try:
                    candidate = read_json(candidate_path)["primary_ready_candidate"]
                    required = {
                        "schema_version", "task_id", "generation", "queue_scope",
                        "home_machine", "partition", "catalog_page", "marker_name",
                    }
                    if (
                        set(candidate) != required
                        or candidate.get("schema_version") != PRIMARY_READY_PROTOCOL_VERSION
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
                        candidate["task_id"], candidate["generation"], queue_scope,
                        candidate["home_machine"], candidate["partition"],
                        candidate["catalog_page"], candidate["marker_name"],
                    )
                except (KeyError, TypeError, ValueError, OSError):
                    return ReadyPeek(None, current, unresolved=True)
                return ReadyPeek(reference, progress_cursor)
            partition_index += 1
            after_name = None
        if successor is None:
            return ReadyPeek(None, ReadyCursor(
                project_id, cfg.machine_name, queue_scope, 0, None, None,
                current.revision + 1,
            ), wrapped=True)
        page_number, partition_name, after_name = successor, None, None


def ready_index_revision(cfg: object, queue_scope: ReadyScope) -> str:
    """Return a read-only revision fingerprint for one machine route."""
    route_key = _route_key(queue_scope, cfg.machine_name)
    catalog_root = shared_paths(cfg.shared_root)["ready_catalogs"] / route_key
    parts: list[object] = [read_ready_index_status(cfg).get("revision")]
    for page_path in sorted(catalog_root.glob("*.json")):
        catalog = read_json(page_path)["ready_catalog"]
        parts.append((catalog["page"], catalog["revision"], tuple(catalog["partitions"])))
        for partition_name in catalog["partitions"]:
            partition_path = _partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            partition = read_json(partition_path)["ready_partition"]
            parts.append((partition_name, partition["revision"], tuple(partition["slots"])))
    return repr(parts)


def ready_index_route_revision(
    cfg: object, queue_scope: ReadyScope, budget: SliceBudget | None = None,
    *, primary_only: bool = False,
) -> int:
    """Read the constant-size route watermark used by bounded probes."""
    route_key = _route_key(queue_scope, cfg.machine_name)
    path = _allocator_path(cfg.shared_root, route_key)
    if budget is not None:
        if not budget.can_start_operation():
            raise ReadyProbeBudgetExhausted
        budget.consume_operation()
    if not path.exists():
        return 0
    if budget is not None:
        if not budget.can_start_operation():
            raise ReadyProbeBudgetExhausted
        budget.consume_operation()
    try:
        allocator = read_json(path)["ready_allocator"]
        revision_key = "primary_revision" if primary_only else "revision"
        if (
            allocator.get("schema_version") != READY_PROTOCOL_VERSION
            or allocator.get("route") != route_key
            or type(allocator.get(revision_key)) is not int
            or allocator[revision_key] < 0
            or (
                primary_only
                and allocator.get("primary_state", "active") != "active"
            )
        ):
            raise ValueError("ready allocator is invalid.")
        return allocator[revision_key]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ready allocator is unreadable: {route_key}") from exc


@contextmanager
def primary_projection_transaction(
    cfg: object, routes: list[tuple[ReadyScope, str]],
):
    """Publish primary candidates only after all affected routes are complete.

    A route remains non-admissible while its projection is being changed.  A
    failed mutation is deliberately left degraded rather than exposing a
    potentially incomplete candidate set to a borrow probe.
    """
    ensure_ready_layout(cfg)
    route_keys = sorted({_route_key(scope, machine) for scope, machine in routes})
    locks = []
    allocators: list[tuple[Path, dict[str, Any]]] = []
    try:
        for route_key in route_keys:
            lock = exclusive(shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock")
            lock.__enter__()
            locks.append(lock)
            allocator_path, allocator = _load_or_create_allocator(cfg.shared_root, route_key)
            allocator["ready_allocator"]["primary_state"] = "updating"
            atomic_replace(allocator_path, allocator)
            allocators.append((allocator_path, allocator))
        try:
            yield
        except BaseException:
            for allocator_path, allocator in allocators:
                allocator["ready_allocator"]["primary_state"] = "degraded"
                atomic_replace(allocator_path, allocator)
            raise
        else:
            for allocator_path, allocator in allocators:
                control = allocator["ready_allocator"]
                control["primary_revision"] = control.get("primary_revision", control["revision"]) + 1
                control["primary_state"] = "active"
                atomic_replace(allocator_path, allocator)
    finally:
        for lock in reversed(locks):
            lock.__exit__(None, None, None)


def bump_primary_ready_revision(
    cfg: object, queue_scope: ReadyScope, home_machine: str,
) -> None:
    """Advance the independent revision for primary-demand changes."""
    with primary_projection_transaction(cfg, [(queue_scope, home_machine)]):
        pass


def iter_ready_marker_refs(cfg: object, queue_scope: ReadyScope) -> list[ReadyMarkerRef]:
    """Enumerate ready markers without advancing any dispatch cursor."""
    route_key = _route_key(queue_scope, cfg.machine_name)
    catalog_root = shared_paths(cfg.shared_root)["ready_catalogs"] / route_key
    references: list[ReadyMarkerRef] = []
    for page_path in sorted(catalog_root.glob("*.json")):
        catalog = read_json(page_path)["ready_catalog"]
        page_number = catalog["page"]
        for partition_name in catalog["partitions"]:
            partition_path = _partition_record_path(
                cfg.shared_root, queue_scope, cfg.machine_name, partition_name
            )
            partition = read_json(partition_path)["ready_partition"]
            for marker_name in sorted(partition["slots"]):
                references.append(
                    _reference_from_slot(
                        cfg, queue_scope, page_number, partition_name, marker_name
                    )
                )
    return references


def delete_stale_ready_marker(cfg: object, reference: ReadyMarkerRef) -> bool:
    """Recheck authoritative generation immediately before exact stale deletion."""
    result = classify_ready_marker(cfg, reference)
    if result.classification != "permanently_stale":
        return False
    return delete_ready_marker(cfg, reference.task_id, reference.generation)


def _new_allocator(route_key: str) -> dict[str, Any]:
    return {
        "ready_allocator": {
            "schema_version": READY_PROTOCOL_VERSION,
            "route": route_key,
            "next_partition": 1,
            "current_partition": None,
            "current_catalog_page": 0,
            "revision": 1,
            "primary_revision": 1,
            "primary_state": "active",
        }
    }


def _load_or_create_allocator(root: Path, route_key: str) -> tuple[Path, dict[str, Any]]:
    path = _allocator_path(root, route_key)
    if path.exists():
        value = read_json(path)
        control = value["ready_allocator"]
        if "primary_revision" not in control:
            control["primary_revision"] = control["revision"]
            control["primary_state"] = "active"
            atomic_replace(path, value)
        elif "primary_state" not in control:
            control["primary_state"] = "active"
            atomic_replace(path, value)
        return path, value
    value = _new_allocator(route_key)
    atomic_replace(path, value)
    return path, value


def _append_catalog_partition(
    root: Path, route_key: str, allocator: dict[str, Any], partition: str,
) -> int:
    control = allocator["ready_allocator"]
    page_number = control["current_catalog_page"]
    path = _catalog_path(root, route_key, page_number)
    if path.exists():
        page = read_json(path)
    else:
        page = {"ready_catalog": {
            "schema_version": READY_PROTOCOL_VERSION,
            "route": route_key,
            "page": page_number,
            "partitions": [],
            "successor": None,
            "revision": 1,
        }}
    catalog = page["ready_catalog"]
    if len(catalog["partitions"]) >= READY_CATALOG_PAGE_SIZE:
        successor = page_number + 1
        catalog["successor"] = successor
        catalog["revision"] += 1
        atomic_replace(path, page)
        page_number = successor
        control["current_catalog_page"] = page_number
        path = _catalog_path(root, route_key, page_number)
        page = {"ready_catalog": {
            "schema_version": READY_PROTOCOL_VERSION,
            "route": route_key,
            "page": page_number,
            "partitions": [],
            "successor": None,
            "revision": 1,
        }}
        catalog = page["ready_catalog"]
    catalog["partitions"].append(partition)
    catalog["revision"] += 1
    atomic_replace(path, page)
    return page_number


def _reserve_slot(
    cfg: object, task_id: str, generation: int, scope: ReadyScope, home_machine: str,
) -> ReadyMarkerRef:
    ensure_ready_layout(cfg)
    root = cfg.shared_root
    reservation_path = _reservation_path(root, task_id, generation)
    if reservation_path.exists():
        raise RuntimeError(
            f"ready generation {task_id}.{generation} already has an in-progress writer."
        )
    route_key = _route_key(scope, home_machine)
    lock_path = shared_paths(root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
        if reservation_path.exists():
            raise RuntimeError(
                f"ready generation {task_id}.{generation} already has an in-progress writer."
            )
        allocator_path, allocator = _load_or_create_allocator(root, route_key)
        control = allocator["ready_allocator"]
        partition = control["current_partition"]
        partition_record = None
        if partition is not None:
            path = _partition_record_path(root, scope, home_machine, partition)
            if path.exists():
                partition_record = read_json(path)
        if (
            partition_record is None
            or partition_record["ready_partition"].get("sealed")
            or len(partition_record["ready_partition"]["slots"]) >= READY_PARTITION_SLOTS
        ):
            partition = f"{control['next_partition']:016d}"
            control["next_partition"] += 1
            control["current_partition"] = partition
            partition_record = {"ready_partition": {
                "schema_version": READY_PROTOCOL_VERSION,
                "route": route_key,
                "partition": partition,
                "slots": [],
                "sealed": False,
                "successor": None,
                "revision": 1,
            }}
            catalog_page = _append_catalog_partition(root, route_key, allocator, partition)
            partition_record["ready_partition"]["catalog_page"] = catalog_page
        else:
            catalog_page = partition_record["ready_partition"]["catalog_page"]
        marker_name = f"{task_id}.{generation}.json"
        partition_record["ready_partition"]["slots"].append(marker_name)
        if len(partition_record["ready_partition"]["slots"]) >= READY_PARTITION_SLOTS:
            partition_record["ready_partition"]["sealed"] = True
        partition_record["ready_partition"]["revision"] += 1
        atomic_replace(
            _partition_record_path(root, scope, home_machine, partition), partition_record
        )
        control["revision"] += 1
        atomic_replace(allocator_path, allocator)
        reference = ReadyMarkerRef(
            task_id, generation, scope, home_machine, partition, catalog_page, marker_name
        )
        atomic_replace(reservation_path, {"ready_reservation": {
            "schema_version": READY_PROTOCOL_VERSION,
            "task_id": task_id,
            "generation": generation,
            "queue_scope": scope,
            "home_machine": home_machine,
            "partition": partition,
            "catalog_page": catalog_page,
            "marker_name": marker_name,
            "created_at": utc_now(),
        }})
        return reference


def reserve_ready_generation(
    cfg: object,
    task_id: str,
    generation: int,
    queue_scope: ReadyScope,
    home_machine: str,
) -> ReadyMarkerRef:
    """Reserve index capacity before acquiring shared Task or Group locks."""
    if generation <= 0:
        raise ValueError("ready generation must be positive.")
    validate_identifier(task_id, "task_id")
    if queue_scope not in {"home", "shared"}:
        raise ValueError("ready queue_scope must be home or shared.")
    return _reserve_slot(cfg, task_id, generation, queue_scope, home_machine)


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


def _primary_machines_for_task(cfg: object, task: TaskRecord) -> list[str]:
    """Return machines whose primary probe must see this task marker."""
    if not task.group_name:
        return [task.placement_policy["home_machine"]]
    group = read_json(group_path(cfg.shared_root, task.group_name))
    normalize_group_record(group)
    workers = group["group"]["worker_set"]
    if task.placement_runtime["queue_scope"] == "home":
        machine = task.placement_policy["home_machine"]
        worker = workers.get(machine)
        return [machine] if worker and worker["scheduling_role"] == "primary" else []
    return sorted(
        machine for machine, worker in workers.items()
        if worker["scheduling_role"] == "primary"
    )


def _primary_candidate_value(reference: ReadyMarkerRef) -> dict[str, Any]:
    return {"primary_ready_candidate": {
        "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
        "task_id": reference.task_id,
        "generation": reference.generation,
        "queue_scope": reference.queue_scope,
        "home_machine": reference.home_machine,
        "partition": reference.partition,
        "catalog_page": reference.catalog_page,
        "marker_name": reference.marker_name,
    }}


def _remove_primary_candidate_everywhere(cfg: object, identity: str) -> None:
    routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    if not routes.exists():
        return
    for route in os.scandir(routes):
        if not route.is_dir():
            continue
        _primary_candidate_path(cfg, route.name, identity).unlink(missing_ok=True)


def _sync_primary_candidate(cfg: object, task: TaskRecord, reference: ReadyMarkerRef) -> None:
    """Mirror one authoritative marker into the primary-only candidate projection."""
    if not _primary_index_is_active(cfg):
        return
    _remove_primary_candidate_everywhere(cfg, reference.identity)
    for machine in _primary_machines_for_task(cfg, task):
        route_key = _primary_route_key(reference.queue_scope, machine)
        route = _primary_route_path(cfg, route_key)
        route.mkdir(parents=True, exist_ok=True)
        atomic_replace(
            _primary_candidate_path(cfg, route_key, reference.identity),
            _primary_candidate_value(reference),
        )


def _clear_primary_candidate_projection(cfg: object) -> None:
    routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    if routes.exists():
        for route in os.scandir(routes):
            if route.is_dir():
                shutil.rmtree(route.path)
    routes.mkdir(parents=True, exist_ok=True)


def rebuild_primary_ready_index(cfg: object) -> None:
    """Rebuild primary candidates from authoritative queued Task records."""
    ensure_ready_layout(cfg)
    atomic_replace(
        _primary_index_state_path(cfg),
        {"primary_ready_index": {
            "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
            "state": "rebuilding",
            "updated_at": utc_now(),
        }},
    )
    _clear_primary_candidate_projection(cfg)
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if not _task_should_have_ready_marker(task):
            continue
        reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is None:
            continue
        for machine in _primary_machines_for_task(cfg, task):
            route_key = _primary_route_key(reference.queue_scope, machine)
            route = _primary_route_path(cfg, route_key)
            route.mkdir(parents=True, exist_ok=True)
            atomic_replace(
                _primary_candidate_path(cfg, route_key, reference.identity),
                _primary_candidate_value(reference),
            )
    atomic_replace(
        _primary_index_state_path(cfg),
        {"primary_ready_index": {
            "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
            "state": "active",
            "updated_at": utc_now(),
        }},
    )


def sync_primary_ready_group(cfg: object, group_name: str) -> None:
    """Refresh primary candidates affected by a Group worker-role mutation."""
    if not _primary_index_is_active(cfg):
        return
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if task.group_name != group_name or not _task_should_have_ready_marker(task):
            continue
        reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is not None:
            _sync_primary_candidate(cfg, task, reference)


def primary_projection_routes_for_group(
    cfg: object, group_name: str,
) -> list[tuple[ReadyScope, str]]:
    """Return every authoritative route whose candidates a Group sync can alter."""
    routes: set[tuple[ReadyScope, str]] = set()
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if task.group_name != group_name or not _task_should_have_ready_marker(task):
            continue
        reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
        if reference is not None:
            routes.add((reference.queue_scope, reference.home_machine))
    return sorted(routes)


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
        reference = _reserve_slot(
            cfg, task.task_id, generation, scope, task.placement_policy["home_machine"]
        )
    reservation_path = _reservation_path(cfg.shared_root, task.task_id, generation)
    if not reservation_path.exists():
        raise RuntimeError("ready reservation disappeared before marker publication.")
    if (
        reference.task_id != task.task_id
        or reference.generation != generation
        or reference.queue_scope != scope
        or reference.home_machine != task.placement_policy["home_machine"]
    ):
        raise ValueError("ready reservation does not match the target Task route.")
    marker = {"ready_marker": {
        "schema_version": READY_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "generation": generation,
        "source_transition": source_transition,
        "source_revision": source_revision,
        "target_revision": target_revision,
        "queue_scope": scope,
        "home_machine": task.placement_policy["home_machine"],
        "group_name": task.group_name,
        "requested_gpus": task.spec.requested_gpus,
        "submission_operation_id": task.submission_operation_id,
        "created_at": utc_now(),
    }}
    if _has_primary_ready_demand(cfg, task):
        with primary_projection_transaction(cfg, [(scope, reference.home_machine)]):
            atomic_replace(_marker_path(cfg.shared_root, reference), marker)
            _sync_primary_candidate(cfg, task, reference)
    else:
        atomic_replace(_marker_path(cfg.shared_root, reference), marker)
    return reference


def delete_ready_marker(cfg: object, task_id: str, generation: int) -> bool:
    """Delete only the exact generation and its slot reservation."""
    path = _reservation_path(cfg.shared_root, task_id, generation)
    if not path.exists():
        return False
    record = read_json(path)["ready_reservation"]
    reference = ReadyMarkerRef(
        task_id, generation, record["queue_scope"], record["home_machine"],
        record["partition"], record["catalog_page"], record["marker_name"],
    )
    is_primary = True
    try:
        marker = read_json(_marker_path(cfg.shared_root, reference))["ready_marker"]
        group_name = marker.get("group_name")
        if group_name:
            group = read_json(group_path(cfg.shared_root, group_name))
            normalize_group_record(group)
            workers = group["group"]["worker_set"]
            if reference.queue_scope == "home":
                worker = workers.get(reference.home_machine)
                is_primary = worker is not None and worker["scheduling_role"] == "primary"
            else:
                is_primary = any(
                    worker["scheduling_role"] == "primary" for worker in workers.values()
                )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        is_primary = True
    route_key = _route_key(reference.queue_scope, reference.home_machine)
    lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
        if is_primary:
            allocator_path, allocator = _load_or_create_allocator(
                cfg.shared_root, route_key
            )
            allocator["ready_allocator"]["primary_state"] = "updating"
            atomic_replace(allocator_path, allocator)
        _marker_path(cfg.shared_root, reference).unlink(missing_ok=True)
        partition_path = _partition_record_path(
            cfg.shared_root,
            reference.queue_scope,
            reference.home_machine,
            reference.partition,
        )
        if partition_path.exists():
            partition_record = read_json(partition_path)
            partition = partition_record["ready_partition"]
            partition["slots"] = [
                name for name in partition["slots"] if name != reference.marker_name
            ]
            partition["revision"] += 1
            if not partition["slots"] and partition.get("sealed"):
                partition_path.unlink(missing_ok=True)
                catalog_path = _catalog_path(
                    cfg.shared_root, route_key, reference.catalog_page
                )
                if catalog_path.exists():
                    page = read_json(catalog_path)
                    catalog = page["ready_catalog"]
                    catalog["partitions"] = [
                        item for item in catalog["partitions"]
                        if item != reference.partition
                    ]
                    catalog["revision"] += 1
                    atomic_replace(catalog_path, page)
                allocator_path = _allocator_path(cfg.shared_root, route_key)
                if allocator_path.exists():
                    allocator = read_json(allocator_path)
                    control = allocator["ready_allocator"]
                    control["revision"] += 1
                    if control.get("current_partition") == reference.partition:
                        control["current_partition"] = None
                    atomic_replace(allocator_path, allocator)
            else:
                atomic_replace(partition_path, partition_record)
                allocator_path = _allocator_path(cfg.shared_root, route_key)
                if allocator_path.exists():
                    allocator = read_json(allocator_path)
                    allocator["ready_allocator"]["revision"] += 1
                    atomic_replace(allocator_path, allocator)
        path.unlink(missing_ok=True)
        _remove_primary_candidate_everywhere(cfg, reference.identity)
        if is_primary:
            allocator_path, allocator = _load_or_create_allocator(
                cfg.shared_root, route_key
            )
            control = allocator["ready_allocator"]
            control["primary_revision"] = (
                control.get("primary_revision", control["revision"]) + 1
            )
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


def classify_ready_marker(
    cfg: object, reference: ReadyMarkerRef,
) -> ReadyClassificationResult:
    """Classify one advisory marker against authoritative Task and Submission truth."""
    if reference.generation <= 0 or not reference.task_id:
        return ReadyClassificationResult("corrupt", "marker_identity_invalid")
    try:
        marker = read_json(_marker_path(cfg.shared_root, reference))["ready_marker"]
        required = {
            "schema_version", "task_id", "generation", "source_transition",
            "source_revision", "target_revision", "queue_scope", "home_machine",
            "group_name", "requested_gpus", "submission_operation_id", "created_at",
        }
        if set(marker) != required or marker["schema_version"] != READY_PROTOCOL_VERSION:
            raise ValueError("ready marker schema is invalid.")
        if (
            marker["task_id"] != reference.task_id
            or marker["generation"] != reference.generation
            or marker["queue_scope"] != reference.queue_scope
            or marker["home_machine"] != reference.home_machine
        ):
            raise ValueError("ready marker identity is inconsistent.")
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "marker_invalid")
    task_file = task_path(cfg.shared_root, reference.task_id)
    if not task_file.exists():
        return ReadyClassificationResult("permanently_stale", "task_missing")
    try:
        task = TaskRecord.from_dict(read_json(task_file))
    except (KeyError, TypeError, ValueError):
        return ReadyClassificationResult("corrupt", "task_invalid")
    if reference.generation != task.ready_generation:
        return ReadyClassificationResult(
            "permanently_stale", "generation_superseded", task
        )
    if (
        reference.queue_scope != task.placement_runtime.get("queue_scope")
        or reference.home_machine != task.placement_policy.get("home_machine")
    ):
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
        return ReadyClassificationResult(
            "temporarily_unavailable", f"submission_{submission_state}", task
        )
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
    return ReadyClassificationResult("claimable", "eligible_truth", task)


def _reference_for_generation(
    cfg: object, task_id: str, generation: int,
) -> ReadyMarkerRef | None:
    path = _reservation_path(cfg.shared_root, task_id, generation)
    try:
        record = read_json(path)["ready_reservation"]
        reference = ReadyMarkerRef(
            task_id,
            generation,
            record["queue_scope"],
            record["home_machine"],
            record["partition"],
            record["catalog_page"],
            record["marker_name"],
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    if (
        record.get("schema_version") != READY_PROTOCOL_VERSION
        or record.get("task_id") != task_id
        or record.get("generation") != generation
        or record.get("queue_scope") not in {"home", "shared"}
        or not isinstance(record.get("home_machine"), str)
        or not isinstance(record.get("partition"), str)
        or type(record.get("catalog_page")) is not int
        or record["catalog_page"] < 0
        or record.get("marker_name") != f"{task_id}.{generation}.json"
    ):
        return None
    return reference


def _is_ready_reference_indexed(cfg: object, reference: ReadyMarkerRef) -> bool:
    """Verify the exact reservation is reachable through its partition and catalog."""
    route_key = _route_key(reference.queue_scope, reference.home_machine)
    try:
        partition = read_json(
            _partition_record_path(
                cfg.shared_root,
                reference.queue_scope,
                reference.home_machine,
                reference.partition,
            )
        )["ready_partition"]
        catalog = read_json(
            _catalog_path(cfg.shared_root, route_key, reference.catalog_page)
        )["ready_catalog"]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    slots = partition.get("slots")
    partitions = catalog.get("partitions")
    return (
        isinstance(slots, list)
        and all(isinstance(item, str) for item in slots)
        and isinstance(partitions, list)
        and all(isinstance(item, str) for item in partitions)
        and partition.get("schema_version") == READY_PROTOCOL_VERSION
        and partition.get("route") == route_key
        and partition.get("partition") == reference.partition
        and partition.get("catalog_page") == reference.catalog_page
        and reference.marker_name in slots
        and catalog.get("schema_version") == READY_PROTOCOL_VERSION
        and catalog.get("route") == route_key
        and catalog.get("page") == reference.catalog_page
        and reference.partition in partitions
    )


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
    from .locks import task_lock
    from .tasks import load_task, save_task

    repaired = 0
    stale_removed = 0
    with task_lock(cfg.shared_root, task_id):
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            return repaired, stale_removed
        reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
        classification = (
            classify_ready_marker(cfg, reference).classification
            if reference is not None and _is_ready_reference_indexed(cfg, reference)
            else None
        )
        if _task_should_have_ready_marker(task):
            if classification in {"claimable", "temporarily_unavailable"}:
                return repaired, stale_removed
            old_generation, _new_generation = prepare_ready_transition(
                cfg, task, "ready_index_rebuild"
            )
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            retire_previous_ready_generation(cfg, old_generation, task)
            return 1, int(old_generation > 0)
        if reference is not None and delete_ready_marker(
            cfg, task.task_id, task.ready_generation
        ):
            stale_removed += 1
        return repaired, stale_removed


def _load_build_page(cfg: object, build_id: str, page: int) -> list[str]:
    record = read_json(_build_page_path(cfg, build_id, page))["ready_build_page"]
    task_ids = record.get("task_ids")
    if (
        record.get("schema_version") != READY_PROTOCOL_VERSION
        or record.get("build_id") != build_id
        or record.get("page") != page
        or not isinstance(task_ids, list)
        or len(task_ids) > READY_BUILD_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in task_ids)
    ):
        raise ValueError("ready build watermark page is invalid.")
    return task_ids


def _advance_build_cursor(
    cursor: dict[str, Any], *, item_count: int, page_count: int,
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
                heartbeat = datetime.fromisoformat(
                    agent["heartbeat_at"].replace("Z", "+00:00")
                )
                interval = float(agent["heartbeat_interval_seconds"])
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if agent.get("observed_state") not in {"active", "idle"}:
                continue
            if (now - heartbeat).total_seconds() > max(30.0, interval * 3.0):
                continue
            if agent.get("writer_capability") != READY_WRITER_CAPABILITY:
                incompatible.append(entry.name)
    return sorted(incompatible)


def _audit_task_ready_projection(cfg: object, task_id: str) -> str | None:
    try:
        task = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
    except FileNotFoundError:
        return None
    except (KeyError, TypeError, ValueError):
        return f"task_invalid:{task_id}"
    reference = _reference_for_generation(cfg, task.task_id, task.ready_generation)
    if _task_should_have_ready_marker(task):
        if reference is None:
            return f"marker_missing:{task_id}"
        if not _is_ready_reference_indexed(cfg, reference):
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
    cfg: object, *, max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance at most ``max_tasks`` durable rebuild or audit records."""
    if type(max_tasks) is not int or not 1 <= max_tasks <= READY_BUILD_PAGE_SIZE:
        raise ValueError(
            f"max_tasks must be between 1 and {READY_BUILD_PAGE_SIZE}."
        )
    begin_ready_index_build(cfg)
    with exclusive(_state_lock_path(cfg)):
        path = ready_state_path(cfg.shared_root)
        value, record = _read_ready_state_record(cfg)
        if record["state"] != "building":
            return record
        build = record.get("build")
        if not isinstance(build, dict):
            _degrade_ready_record(record, "build_state_missing")
            _commit_ready_state(path, value, record)
            return record
        watermark = build.get("watermark", {})
        page_count = watermark.get("page_count")
        if type(page_count) is not int or page_count < 0 or not watermark.get("is_complete"):
            _degrade_ready_record(record, "build_watermark_invalid")
            _commit_ready_state(path, value, record)
            return record
        phase = build.get("phase")
        cursor_name = "cursor" if phase == "backfill" else "audit_cursor"
        if phase not in {"backfill", "audit"}:
            _degrade_ready_record(record, f"build_phase_invalid:{phase}")
            _commit_ready_state(path, value, record)
            return record
        cursor = build.get(cursor_name)
        if not isinstance(cursor, dict):
            _degrade_ready_record(record, f"build_cursor_invalid:{cursor_name}")
            _commit_ready_state(path, value, record)
            return record
        processed_now = 0
        try:
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
                else:
                    issue = _audit_task_ready_projection(cfg, task_id)
                    if issue is not None:
                        _degrade_ready_record(record, issue)
                        break
                processed_now += 1
                _advance_build_cursor(
                    cursor, item_count=len(task_ids), page_count=page_count
                )
            if record["state"] == "building" and cursor["page"] >= page_count:
                if phase == "backfill":
                    build["phase"] = "audit"
                else:
                    assert_ready_writer_compatible(cfg)
                    incompatible = _active_incompatible_writers(cfg)
                    if incompatible:
                        _degrade_ready_record(
                            record,
                            "incompatible_active_writers:" + ",".join(incompatible),
                        )
                    else:
                        record["state"] = "active"
                        record["degraded_reasons"] = []
                        build["phase"] = "completed"
                        build["completed_at"] = utc_now()
                        rebuild_primary_ready_index(cfg)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _degrade_ready_record(record, f"build_failed:{type(exc).__name__}:{exc}")
        _commit_ready_state(path, value, record)
        return record


def repair_ready_index(
    cfg: object, *, max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Start degraded recovery and advance one bounded repair slice."""
    begin_ready_index_build(cfg, is_repair=True)
    return advance_ready_index_build(cfg, max_tasks=max_tasks)
