"""Primary-worker candidate projection storage and synchronization."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..locks import exclusive
from ..paths import group_path, shared_paths
from ..records import TaskRecord, normalize_group_record, utc_now, validate_identifier
from ..store import atomic_replace, read_json
from .records import ReadyMarkerRef, ReadyScope

PRIMARY_READY_PROTOCOL_VERSION = 1


def route_key(scope: ReadyScope, machine: str) -> str:
    validate_identifier(machine, "machine")
    return f"{scope}.{machine}"


def projection_state_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_primary"] / "state.json"


def _projection_rebuild_lock_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["ready_locks"] / "primary-projection-rebuild.lock"


@contextmanager
def projection_rebuild_lock(cfg: object):
    """Serialize primary projection mutations with a full rebuild."""
    with exclusive(_projection_rebuild_lock_path(cfg)):
        yield


def route_path(cfg: object, route: str) -> Path:
    return shared_paths(cfg.shared_root)["ready_primary"] / "routes" / route


def candidate_path(cfg: object, route: str, identity: str) -> Path:
    validate_identifier(identity.replace(".", "-"), "primary candidate identity")
    return route_path(cfg, route) / f"{identity}.json"


def is_projection_active(cfg: object) -> bool:
    try:
        value = read_json(projection_state_path(cfg))["primary_ready_index"]
        return value.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION and value.get("state") == "active"
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def rebuild_record(cfg: object) -> dict[str, Any] | None:
    """Return the optional durable owner record for a candidate rebuild."""
    path = projection_state_path(cfg)
    if not path.exists():
        return None
    value = read_json(path).get("primary_ready_index")
    return value if isinstance(value, dict) else None


def accepts_updates_under_lock(cfg: object) -> bool:
    """Return whether a caller holding the projection lock may write candidates."""
    primary = rebuild_record(cfg)
    return bool(
        primary is not None
        and primary.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
        and (
            primary.get("state") == "active"
            or (primary.get("state") == "rebuilding" and primary.get("cleared") is True)
        )
    )


def _primary_machines_for_task(cfg: object, task: TaskRecord) -> list[str]:
    if not task.group_name:
        return [task.placement_policy["home_machine"]]
    group = read_json(group_path(cfg.shared_root, task.group_name))
    normalize_group_record(group)
    workers = group["group"]["worker_set"]
    if task.placement_runtime["queue_scope"] == "home":
        machine = task.placement_policy["home_machine"]
        worker = workers.get(machine)
        return [machine] if worker and worker["scheduling_role"] == "primary" else []
    return sorted(machine for machine, worker in workers.items() if worker["scheduling_role"] == "primary")


def _candidate_value(reference: ReadyMarkerRef) -> dict[str, Any]:
    return {
        "primary_ready_candidate": {
            "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
            "task_id": reference.task_id,
            "generation": reference.generation,
            "queue_scope": reference.queue_scope,
            "home_machine": reference.home_machine,
            "partition": reference.partition,
            "catalog_page": reference.catalog_page,
            "marker_name": reference.marker_name,
        }
    }


def remove_candidate_everywhere_under_lock(cfg: object, identity: str) -> None:
    """Remove a candidate while the caller holds the projection lock."""
    routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    if not routes.exists():
        return
    for route in os.scandir(routes):
        if route.is_dir():
            candidate_path(cfg, route.name, identity).unlink(missing_ok=True)


def remove_candidate_everywhere(cfg: object, identity: str) -> None:
    with projection_rebuild_lock(cfg):
        remove_candidate_everywhere_under_lock(cfg, identity)


def _remove_candidate_from_machines_under_lock(
    cfg: object,
    reference: ReadyMarkerRef,
    machines: set[str],
) -> None:
    for machine in sorted(machines):
        candidate_route = route_key(reference.queue_scope, machine)
        candidate_path(cfg, candidate_route, reference.identity).unlink(missing_ok=True)


def sync_candidate_under_lock(
    cfg: object,
    task: TaskRecord,
    reference: ReadyMarkerRef,
    *,
    should_require_active: bool = True,
) -> None:
    """Synchronize one candidate while the caller holds the projection lock."""
    if should_require_active and not accepts_updates_under_lock(cfg):
        return
    remove_candidate_everywhere_under_lock(cfg, reference.identity)
    for machine in _primary_machines_for_task(cfg, task):
        candidate_route = route_key(reference.queue_scope, machine)
        route = route_path(cfg, candidate_route)
        route.mkdir(parents=True, exist_ok=True)
        atomic_replace(candidate_path(cfg, candidate_route, reference.identity), _candidate_value(reference))


def sync_candidate(cfg: object, task: TaskRecord, reference: ReadyMarkerRef) -> None:
    with projection_rebuild_lock(cfg):
        sync_candidate_under_lock(cfg, task, reference)


def _primary_member_machines(
    reference: ReadyMarkerRef,
    workers: dict[str, dict[str, Any]],
) -> set[str]:
    if reference.queue_scope == "home":
        worker = workers.get(reference.home_machine)
        return {reference.home_machine} if worker and worker.get("scheduling_role") == "primary" else set()
    return {machine for machine, worker in workers.items() if worker.get("scheduling_role") == "primary"}


def sync_member_candidate_under_lock(
    cfg: object,
    group_name: str,
    reference: ReadyMarkerRef,
    previous_workers: dict[str, dict[str, Any]],
) -> None:
    """Refresh a member candidate while the caller holds the projection lock."""
    group = read_json(group_path(cfg.shared_root, group_name))
    normalize_group_record(group)
    current_workers = group["group"]["worker_set"]
    previous_machines = _primary_member_machines(reference, previous_workers)
    current_machines = _primary_member_machines(reference, current_workers)
    _remove_candidate_from_machines_under_lock(cfg, reference, previous_machines - current_machines)
    for machine in sorted(current_machines - previous_machines):
        candidate_route = route_key(reference.queue_scope, machine)
        route_path(cfg, candidate_route).mkdir(parents=True, exist_ok=True)
        atomic_replace(candidate_path(cfg, candidate_route, reference.identity), _candidate_value(reference))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def park_projection_under_lock(cfg: object, build_id: str) -> None:
    """Replace active routes with an empty tree while holding the projection lock."""
    validate_identifier(build_id, "primary ready rebuild id")
    primary_root = shared_paths(cfg.shared_root)["ready_primary"]
    routes = primary_root / "routes"
    replaced_root = primary_root / "replaced-routes"
    replaced = replaced_root / build_id
    replaced_root.mkdir(parents=True, exist_ok=True)
    if not replaced.exists():
        if routes.exists():
            os.replace(routes, replaced)
        else:
            replaced.mkdir()
    routes.mkdir(parents=True, exist_ok=True)
    _fsync_directory(replaced_root)
    _fsync_directory(primary_root)


def begin_primary_ready_index_rebuild(cfg: object, build_id: str) -> None:
    """Start or resume one owner-fenced, empty-first candidate rebuild."""
    validate_identifier(build_id, "primary ready rebuild id")
    with projection_rebuild_lock(cfg):
        current = rebuild_record(cfg)
        if (
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "rebuilding"
            and current.get("build_id") == build_id
            and current.get("cleared") is True
        ):
            return
        if (
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "active"
            and current.get("completed_build_id") == build_id
        ):
            return
        atomic_replace(
            projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": False,
                    "updated_at": utc_now(),
                }
            },
        )
        park_projection_under_lock(cfg, build_id)
        atomic_replace(
            projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": True,
                    "updated_at": utc_now(),
                }
            },
        )


def rebuild_primary_ready_candidate(
    cfg: object,
    build_id: str,
    task: TaskRecord,
    reference: ReadyMarkerRef,
) -> None:
    """Publish one candidate into a rebuild already cleared by its owner."""
    with projection_rebuild_lock(cfg):
        current = rebuild_record(cfg)
        is_rebuilding = bool(
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "rebuilding"
            and current.get("build_id") == build_id
            and current.get("cleared") is True
        )
        is_completed = bool(
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "active"
            and current.get("completed_build_id") == build_id
        )
        if not is_rebuilding and not is_completed:
            raise RuntimeError("primary candidate rebuild ownership is invalid.")
        for machine in _primary_machines_for_task(cfg, task):
            candidate_route = route_key(reference.queue_scope, machine)
            route_path(cfg, candidate_route).mkdir(parents=True, exist_ok=True)
            atomic_replace(candidate_path(cfg, candidate_route, reference.identity), _candidate_value(reference))


def complete_primary_ready_index_rebuild(cfg: object, build_id: str) -> None:
    """Make a fully populated owner-fenced candidate projection observable."""
    with projection_rebuild_lock(cfg):
        current = rebuild_record(cfg)
        if (
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "active"
            and current.get("completed_build_id") == build_id
        ):
            return
        if not (
            current is not None
            and current.get("schema_version") == PRIMARY_READY_PROTOCOL_VERSION
            and current.get("state") == "rebuilding"
            and current.get("build_id") == build_id
            and current.get("cleared") is True
        ):
            raise RuntimeError("primary candidate rebuild cannot be completed by this owner.")
        atomic_replace(
            projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "active",
                    "completed_build_id": build_id,
                    "updated_at": utc_now(),
                }
            },
        )
