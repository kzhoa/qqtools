"""Durable storage routes for the ready projection."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..locks import exclusive
from ..paths import shared_paths
from ..records import utc_now, validate_identifier
from ..store import atomic_replace, read_json
from ..work_budget import SliceBudget
from . import state
from .records import ReadyMarkerRef, ReadyScope

READY_PARTITION_SLOTS = 64
READY_CATALOG_PAGE_SIZE = 64


class ReadyProbeBudgetExhausted(RuntimeError):
    """Raised when a bounded ready revision read cannot start safely."""


def route_key(scope: ReadyScope, home_machine: str) -> str:
    validate_identifier(home_machine, "home_machine")
    return f"home.{home_machine}" if scope == "home" else "shared"


def route_directory(root: Path, scope: ReadyScope, home_machine: str) -> Path:
    paths = shared_paths(root)
    if scope == "home":
        return paths["ready_home"] / home_machine
    return paths["ready_shared"]


def reservation_path(root: Path, task_id: str, generation: int) -> Path:
    return shared_paths(root)["ready_reservations"] / f"{task_id}.{generation}.json"


def allocator_path(root: Path, route: str) -> Path:
    return shared_paths(root)["ready"] / "allocators" / f"{route}.json"


def catalog_path(root: Path, route: str, page: int) -> Path:
    return shared_paths(root)["ready_catalogs"] / route / f"{page:016d}.json"


def partition_record_path(root: Path, scope: ReadyScope, home_machine: str, partition: str) -> Path:
    return route_directory(root, scope, home_machine) / partition / "partition.json"


def marker_path(root: Path, reference: ReadyMarkerRef) -> Path:
    return (
        route_directory(root, reference.queue_scope, reference.home_machine)
        / reference.partition
        / reference.marker_name
    )


def _new_allocator(route: str) -> dict[str, Any]:
    return {
        "ready_allocator": {
            "schema_version": state.READY_PROTOCOL_VERSION,
            "route": route,
            "next_partition": 1,
            "current_partition": None,
            "current_catalog_page": 0,
            "revision": 1,
            "primary_revision": 1,
            "primary_state": "active",
            "primary_lane_revisions": {"gpu": 1, "cpu": 1},
        }
    }


def ensure_primary_lane_revisions(control: dict[str, Any]) -> bool:
    """Backfill lane-local primary watermarks from the legacy watermark."""
    revisions = control.get("primary_lane_revisions")
    if isinstance(revisions, dict) and all(
        type(revisions.get(lane)) is int and revisions[lane] >= 0 for lane in ("gpu", "cpu")
    ):
        return False
    revision = control.get("primary_revision", control["revision"])
    control["primary_lane_revisions"] = {"gpu": revision, "cpu": revision}
    return True


def load_or_create_allocator_under_lock(root: Path, route: str) -> tuple[Path, dict[str, Any]]:
    """Load a route allocator while the caller holds that route's lock."""
    path = allocator_path(root, route)
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
        if ensure_primary_lane_revisions(control):
            atomic_replace(path, value)
        return path, value
    value = _new_allocator(route)
    atomic_replace(path, value)
    return path, value


def _append_catalog_partition(root: Path, route: str, allocator: dict[str, Any], partition: str) -> int:
    control = allocator["ready_allocator"]
    page_number = control["current_catalog_page"]
    path = catalog_path(root, route, page_number)
    if path.exists():
        page = read_json(path)
    else:
        page = {
            "ready_catalog": {
                "schema_version": state.READY_PROTOCOL_VERSION,
                "route": route,
                "page": page_number,
                "partitions": [],
                "successor": None,
                "revision": 1,
            }
        }
    catalog = page["ready_catalog"]
    if len(catalog["partitions"]) >= READY_CATALOG_PAGE_SIZE:
        successor = page_number + 1
        catalog["successor"] = successor
        catalog["revision"] += 1
        atomic_replace(path, page)
        page_number = successor
        control["current_catalog_page"] = page_number
        path = catalog_path(root, route, page_number)
        page = {
            "ready_catalog": {
                "schema_version": state.READY_PROTOCOL_VERSION,
                "route": route,
                "page": page_number,
                "partitions": [],
                "successor": None,
                "revision": 1,
            }
        }
        catalog = page["ready_catalog"]
    catalog["partitions"].append(partition)
    catalog["revision"] += 1
    atomic_replace(path, page)
    return page_number


def reserve_slot(cfg: object, task_id: str, generation: int, scope: ReadyScope, home_machine: str) -> ReadyMarkerRef:
    """Reserve one route slot; validates neither Task authority nor public inputs."""
    state.ensure_ready_layout(cfg)
    root = cfg.shared_root
    reservation = reservation_path(root, task_id, generation)
    if reservation.exists():
        raise RuntimeError(f"ready generation {task_id}.{generation} already has an in-progress writer.")
    route = route_key(scope, home_machine)
    with exclusive(shared_paths(root)["ready_locks"] / f"{route}.lock"):
        if reservation.exists():
            raise RuntimeError(f"ready generation {task_id}.{generation} already has an in-progress writer.")
        allocator_file, allocator = load_or_create_allocator_under_lock(root, route)
        control = allocator["ready_allocator"]
        partition = control["current_partition"]
        partition_record = None
        if partition is not None:
            path = partition_record_path(root, scope, home_machine, partition)
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
            partition_record = {
                "ready_partition": {
                    "schema_version": state.READY_PROTOCOL_VERSION,
                    "route": route,
                    "partition": partition,
                    "slots": [],
                    "sealed": False,
                    "successor": None,
                    "revision": 1,
                }
            }
            catalog_page = _append_catalog_partition(root, route, allocator, partition)
            partition_record["ready_partition"]["catalog_page"] = catalog_page
        else:
            catalog_page = partition_record["ready_partition"]["catalog_page"]
        marker_name = f"{task_id}.{generation}.json"
        partition_record["ready_partition"]["slots"].append(marker_name)
        if len(partition_record["ready_partition"]["slots"]) >= READY_PARTITION_SLOTS:
            partition_record["ready_partition"]["sealed"] = True
        partition_record["ready_partition"]["revision"] += 1
        atomic_replace(partition_record_path(root, scope, home_machine, partition), partition_record)
        control["revision"] += 1
        atomic_replace(allocator_file, allocator)
        reference = ReadyMarkerRef(task_id, generation, scope, home_machine, partition, catalog_page, marker_name)
        atomic_replace(
            reservation,
            {
                "ready_reservation": {
                    "schema_version": state.READY_PROTOCOL_VERSION,
                    "task_id": task_id,
                    "generation": generation,
                    "queue_scope": scope,
                    "home_machine": home_machine,
                    "partition": partition,
                    "catalog_page": catalog_page,
                    "marker_name": marker_name,
                    "created_at": utc_now(),
                }
            },
        )
        return reference


def reserve_ready_generation(
    cfg: object, task_id: str, generation: int, queue_scope: ReadyScope, home_machine: str
) -> ReadyMarkerRef:
    """Reserve index capacity before acquiring shared Task or Group locks."""
    if generation <= 0:
        raise ValueError("ready generation must be positive.")
    validate_identifier(task_id, "task_id")
    if queue_scope not in {"home", "shared"}:
        raise ValueError("ready queue_scope must be home or shared.")
    return reserve_slot(cfg, task_id, generation, queue_scope, home_machine)


def reference_for_generation(cfg: object, task_id: str, generation: int) -> ReadyMarkerRef | None:
    path = reservation_path(cfg.shared_root, task_id, generation)
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
        record.get("schema_version") != state.READY_PROTOCOL_VERSION
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


def is_reference_indexed(cfg: object, reference: ReadyMarkerRef) -> bool:
    """Verify the exact reservation is reachable through its partition and catalog."""
    route = route_key(reference.queue_scope, reference.home_machine)
    try:
        partition = read_json(
            partition_record_path(cfg.shared_root, reference.queue_scope, reference.home_machine, reference.partition)
        )["ready_partition"]
        catalog = read_json(catalog_path(cfg.shared_root, route, reference.catalog_page))["ready_catalog"]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    slots, partitions = partition.get("slots"), catalog.get("partitions")
    return (
        isinstance(slots, list)
        and all(isinstance(item, str) for item in slots)
        and isinstance(partitions, list)
        and all(isinstance(item, str) for item in partitions)
        and partition.get("schema_version") == state.READY_PROTOCOL_VERSION
        and partition.get("route") == route
        and partition.get("partition") == reference.partition
        and partition.get("catalog_page") == reference.catalog_page
        and reference.marker_name in slots
        and catalog.get("schema_version") == state.READY_PROTOCOL_VERSION
        and catalog.get("route") == route
        and catalog.get("page") == reference.catalog_page
        and reference.partition in partitions
    )


def ready_index_route_revision(
    cfg: object,
    queue_scope: ReadyScope,
    budget: SliceBudget | None = None,
    *,
    primary_only: bool = False,
    lane: str = "gpu",
) -> int:
    """Read the constant-size route watermark used by bounded probes."""
    route = route_key(queue_scope, cfg.machine_name)
    path = allocator_path(cfg.shared_root, route)
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
        if primary_only:
            ensure_primary_lane_revisions(allocator)
            revision = allocator["primary_lane_revisions"].get(lane)
        else:
            revision = allocator.get(revision_key)
        if (
            allocator.get("schema_version") != state.READY_PROTOCOL_VERSION
            or allocator.get("route") != route
            or type(revision) is not int
            or revision < 0
            or (primary_only and allocator.get("primary_state", "active") != "active")
        ):
            raise ValueError("ready allocator is invalid.")
        return revision
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ready allocator is unreadable: {route}") from exc


@contextmanager
def primary_route_update_transaction(
    cfg: object, routes: list[tuple[ReadyScope, str]], *, lanes: tuple[str, ...] = ("gpu", "cpu")
):
    """Fence affected route watermarks while a primary projection is updated."""
    state.ensure_ready_layout(cfg)
    route_keys = sorted({route_key(scope, machine) for scope, machine in routes})
    locks = []
    allocators: list[tuple[Path, dict[str, Any]]] = []
    try:
        for route in route_keys:
            lock = exclusive(shared_paths(cfg.shared_root)["ready_locks"] / f"{route}.lock")
            lock.__enter__()
            locks.append(lock)
            allocator_file, allocator = load_or_create_allocator_under_lock(cfg.shared_root, route)
            ensure_primary_lane_revisions(allocator["ready_allocator"])
            allocator["ready_allocator"]["primary_state"] = "updating"
            atomic_replace(allocator_file, allocator)
            allocators.append((allocator_file, allocator))
        try:
            yield
        except BaseException:
            for allocator_file, allocator in allocators:
                allocator["ready_allocator"]["primary_state"] = "degraded"
                atomic_replace(allocator_file, allocator)
            raise
        else:
            for allocator_file, allocator in allocators:
                control = allocator["ready_allocator"]
                control["primary_revision"] = control.get("primary_revision", control["revision"]) + 1
                ensure_primary_lane_revisions(control)
                for lane in lanes:
                    control["primary_lane_revisions"][lane] += 1
                control["primary_state"] = "active"
                atomic_replace(allocator_file, allocator)
    finally:
        for lock in reversed(locks):
            lock.__exit__(None, None, None)


def bump_primary_ready_revision(cfg: object, queue_scope: ReadyScope, home_machine: str, *, lane: str = "gpu") -> None:
    """Advance the independent revision for primary-demand changes."""
    with primary_route_update_transaction(cfg, [(queue_scope, home_machine)], lanes=(lane,)):
        pass
