"""Durable generation-safe ready liveness projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .locks import exclusive
from .paths import group_path, ready_state_path, shared_paths, submission_path, task_path
from .records import SCHEMA_VERSION, TaskRecord, utc_now, validate_identifier
from .store import atomic_replace, read_json

READY_PROTOCOL_VERSION = 1
READY_PARTITION_SLOTS = 64
READY_CATALOG_PAGE_SIZE = 64
ReadyScope = Literal["home", "shared"]
ReadyClassification = Literal[
    "claimable", "temporarily_unavailable", "permanently_stale", "corrupt"
]
ReadyIndexState = Literal["absent", "building", "active", "degraded"]


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


def ensure_ready_layout(cfg: object) -> None:
    """Create the additive ready layout without activating ready-only scheduling."""
    paths = shared_paths(cfg.shared_root)
    for name in (
        "ready", "ready_home", "ready_shared", "ready_catalogs", "ready_reservations",
        "ready_cursors", "ready_locks",
    ):
        paths[name].mkdir(parents=True, exist_ok=True)
    state_path = ready_state_path(cfg.shared_root)
    if not state_path.exists():
        atomic_replace(state_path, {
            "ready_index": {
                "schema_version": READY_PROTOCOL_VERSION,
                "state": "absent",
                "writer_capability": None,
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
        return state
    except (KeyError, TypeError, ValueError):
        return "degraded"


def mark_ready_index_degraded(cfg: object, reason: str) -> None:
    """Fail closed after detecting a corrupt active projection."""
    path = ready_state_path(cfg.shared_root)
    try:
        value = read_json(path)
        record = value["ready_index"]
        reasons = record.get("degraded_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        if reason not in reasons:
            reasons.append(reason)
        record["state"] = "degraded"
        record["degraded_reasons"] = reasons
        record["updated_at"] = utc_now()
        atomic_replace(path, value)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return


def _route_key(scope: ReadyScope, home_machine: str) -> str:
    validate_identifier(home_machine, "home_machine")
    return f"home.{home_machine}" if scope == "home" else "shared"


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
        }
    }


def _load_or_create_allocator(root: Path, route_key: str) -> tuple[Path, dict[str, Any]]:
    path = _allocator_path(root, route_key)
    if path.exists():
        return path, read_json(path)
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
    _marker_path(cfg.shared_root, reference).unlink(missing_ok=True)
    route_key = _route_key(reference.queue_scope, reference.home_machine)
    lock_path = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    with exclusive(lock_path):
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
                    if control.get("current_partition") == reference.partition:
                        control["current_partition"] = None
                        control["revision"] += 1
                        atomic_replace(allocator_path, allocator)
            else:
                atomic_replace(partition_path, partition_record)
        path.unlink(missing_ok=True)
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
