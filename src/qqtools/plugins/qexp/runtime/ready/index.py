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
from .diagnostics import ReadyDiagnostic, classification_diagnostic, diagnostic, exception_fields, safe_identifier
from .group_members import (
    group_ready_members_state,
    is_group_ready_member_projection_usable,
    publish_group_ready_member,
    read_group_ready_members,
    retire_group_ready_member,
)
from .records import ReadyMarkerRef, ReadyScope

ReadyClassification = Literal["claimable", "temporarily_unavailable", "permanently_stale", "corrupt"]


@dataclass(frozen=True, slots=True)
class ReadyClassificationResult:
    classification: ReadyClassification
    reason: str
    task: TaskRecord | None = None
    diagnostic: ReadyDiagnostic | None = None


def _classification_result(
    cfg: object,
    reference: ReadyMarkerRef | None,
    classification: ReadyClassification,
    reason: str,
    task: TaskRecord | None = None,
    *,
    marker: dict[str, Any] | None = None,
    exception: BaseException | None = None,
    diagnostic_value: ReadyDiagnostic | None = None,
) -> ReadyClassificationResult:
    """Build a classification while retaining a typed degradation observation."""
    if classification != "corrupt":
        return ReadyClassificationResult(classification, reason, task, diagnostic_value)
    if diagnostic_value is None:
        diagnostic_value = classification_diagnostic(
            reason,
            reference,
            task=task,
            marker=marker,
            exception=exception,
        )
    return ReadyClassificationResult(classification, reason, task, diagnostic_value)


def is_primary_ready_index_active(cfg: object) -> bool:
    """Return whether the primary-only candidate projection is usable."""
    return primary_candidates.is_projection_active(cfg) and is_group_ready_member_projection_usable(cfg)


def begin_primary_ready_index_rebuild(cfg: object, build_id: str) -> None:
    """Initialize the ready layout before starting a candidate-only rebuild."""
    state.ensure_ready_layout(cfg)
    primary_candidates.begin_primary_ready_index_rebuild(cfg, build_id)


def rebuild_primary_ready_candidate(
    cfg: object,
    build_id: str,
    task: TaskRecord,
    reference: ReadyMarkerRef,
) -> None:
    """Publish one owner-fenced primary candidate during an incremental rebuild."""
    primary_candidates.rebuild_primary_ready_candidate(cfg, build_id, task, reference)


def complete_primary_ready_index_rebuild(cfg: object, build_id: str) -> None:
    """Activate one fully populated owner-fenced primary candidate rebuild."""
    primary_candidates.complete_primary_ready_index_rebuild(cfg, build_id)


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
            if task.group_name != group_name or not task_should_have_ready_marker(task):
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
        if task.group_name != group_name or not task_should_have_ready_marker(task):
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
            return _classification_result(cfg, reference, "permanently_stale", "task_missing")
        try:
            task = TaskRecord.from_dict(read_json(task_file))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return _classification_result(cfg, reference, "corrupt", "task_invalid", exception=exc)
        if _is_ready_publication_pending(cfg, reference, task):
            return _classification_result(cfg, reference, "temporarily_unavailable", "marker_publication_pending", task)
        if reference.generation != task.ready_generation:
            return _classification_result(cfg, reference, "permanently_stale", "generation_superseded", task)
        if reference.queue_scope != task.placement_runtime.get(
            "queue_scope"
        ) or reference.home_machine != task.placement_policy.get("home_machine"):
            return _classification_result(cfg, reference, "corrupt", "route_mismatch", task)
        if task.state.get("projection") != "queued" or task.claim_control.get("active_claim"):
            return _classification_result(cfg, reference, "permanently_stale", "task_not_queued", task)
        if (
            task.control.get("cleanup_operation_id")
            or task.control.get("cleanup_state")
            or task.control.get("cancellation_requested_at")
        ):
            return _classification_result(cfg, reference, "permanently_stale", "task_controlled", task)
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
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
                if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
                    return _classification_result(cfg, reference, "corrupt", "marker_invalid", task, exception=exc)
                return _classification_result(
                    cfg,
                    reference,
                    "corrupt",
                    "marker_missing_unindexed",
                    task,
                    diagnostic_value=diagnostic(
                        "marker_missing",
                        stage="marker_truth",
                        task_id=reference.task_id,
                        generation=reference.generation,
                        indexed=False,
                        task_projection=task.state.get("projection", "unobserved"),
                        active_claim=bool(task.claim_control.get("active_claim")),
                    ),
                )
            if isinstance(slots, list) and reference.marker_name in slots:
                return _classification_result(
                    cfg,
                    reference,
                    "corrupt",
                    "marker_missing_indexed",
                    task,
                    diagnostic_value=diagnostic(
                        "marker_missing",
                        stage="marker_truth",
                        task_id=reference.task_id,
                        generation=reference.generation,
                        indexed=True,
                        task_projection=task.state.get("projection", "unobserved"),
                        active_claim=bool(task.claim_control.get("active_claim")),
                    ),
                )
            return _classification_result(
                cfg,
                reference,
                "corrupt",
                "marker_missing_unindexed",
                task,
                diagnostic_value=diagnostic(
                    "marker_missing",
                    stage="marker_truth",
                    task_id=reference.task_id,
                    generation=reference.generation,
                    indexed=False,
                    task_projection=task.state.get("projection", "unobserved"),
                    active_claim=bool(task.claim_control.get("active_claim")),
                ),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return _classification_result(cfg, reference, "corrupt", "marker_invalid", task, exception=exc)


def classify_ready_marker(
    cfg: object,
    reference: ReadyMarkerRef,
) -> ReadyClassificationResult:
    """Classify one advisory marker against authoritative Task and Submission truth."""
    if reference.generation <= 0 or not reference.task_id:
        return _classification_result(cfg, reference, "corrupt", "marker_identity_invalid")
    try:
        envelope = read_json(routes.marker_path(cfg.shared_root, reference))
        marker = envelope["ready_marker"]
        if not isinstance(marker, dict):
            raise TypeError("ready marker envelope is invalid")
    except FileNotFoundError:
        marker = _recheck_missing_ready_marker(cfg, reference)
        if isinstance(marker, ReadyClassificationResult):
            return marker
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _classification_result(
            cfg,
            reference,
            "corrupt",
            "marker_invalid",
            diagnostic_value=diagnostic(
                "record_invalid",
                object="marker",
                stage="marker_parse",
                issue_code="marker_invalid",
                **({"task_id": reference.task_id} if reference.task_id else {}),
                **({"generation": reference.generation} if reference.generation >= 0 else {}),
                **exception_fields(exc),
            ),
        )
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
            raise LookupError("ready marker identity is inconsistent.")
    except LookupError:
        expected_identity = {
            "task_id": reference.task_id,
            "generation": reference.generation,
            "queue_scope": reference.queue_scope,
            "home_machine": reference.home_machine,
        }
        mismatch_fields = [key for key, expected in expected_identity.items() if marker.get(key) != expected]
        identity_fields: dict[str, Any] = {}
        for key in mismatch_fields:
            identity_fields[f"expected_{key}"] = expected_identity[key]
            observed = marker.get(key)
            if key == "task_id" and isinstance(observed, str):
                safe_task_id = safe_identifier(observed)
                if safe_task_id != "unobserved":
                    identity_fields["observed_task_id"] = safe_task_id
            elif key == "generation" and type(observed) is int and observed >= 0:
                identity_fields["observed_generation"] = observed
            elif key == "queue_scope" and observed in {"home", "shared"}:
                identity_fields["observed_queue_scope"] = observed
            elif key == "home_machine" and isinstance(observed, str) and observed:
                safe_home_machine = safe_identifier(observed)
                if safe_home_machine != "unobserved":
                    identity_fields["observed_home_machine"] = safe_home_machine
        return _classification_result(
            cfg,
            reference,
            "corrupt",
            "marker_invalid",
            diagnostic_value=diagnostic(
                "identity_mismatch",
                stage="marker_identity",
                task_id=reference.task_id,
                generation=reference.generation,
                mismatch_fields=mismatch_fields,
                **identity_fields,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _classification_result(cfg, reference, "corrupt", "marker_invalid", marker=marker, exception=exc)
    task_file = task_path(cfg.shared_root, reference.task_id)
    if not task_file.exists():
        return _classification_result(cfg, reference, "permanently_stale", "task_missing")
    try:
        task = TaskRecord.from_dict(read_json(task_file))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _classification_result(cfg, reference, "corrupt", "task_invalid", exception=exc)
    if _is_ready_publication_pending(cfg, reference, task):
        return _classification_result(cfg, reference, "temporarily_unavailable", "marker_publication_pending", task)
    if reference.generation != task.ready_generation:
        return _classification_result(cfg, reference, "permanently_stale", "generation_superseded", task)
    if reference.queue_scope != task.placement_runtime.get(
        "queue_scope"
    ) or reference.home_machine != task.placement_policy.get("home_machine"):
        return _classification_result(cfg, reference, "corrupt", "route_mismatch", task, marker=marker)
    if task.state.get("projection") != "queued" or task.claim_control.get("active_claim"):
        return _classification_result(cfg, reference, "permanently_stale", "task_not_queued", task)
    if (
        task.control.get("cleanup_operation_id")
        or task.control.get("cleanup_state")
        or task.control.get("cancellation_requested_at")
    ):
        return _classification_result(cfg, reference, "permanently_stale", "task_controlled", task)
    operation_id = task.submission_operation_id
    if not operation_id:
        return _classification_result(cfg, reference, "corrupt", "submission_identity_missing", task)
    operation_file = submission_path(cfg.shared_root, operation_id)
    if not operation_file.exists():
        return _classification_result(cfg, reference, "corrupt", "submission_missing", task)
    try:
        submission_state = read_json(operation_file)["submission"]["state"]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _classification_result(cfg, reference, "corrupt", "submission_invalid", task, exception=exc)
    if submission_state in {"preparing", "committing", "blocked"}:
        return _classification_result(cfg, reference, "temporarily_unavailable", f"submission_{submission_state}", task)
    if submission_state == "aborted":
        return _classification_result(cfg, reference, "permanently_stale", "submission_aborted", task)
    if submission_state != "committed":
        return _classification_result(cfg, reference, "corrupt", "submission_state_invalid", task)
    if task.group_name:
        path = group_path(cfg.shared_root, task.group_name)
        if not path.exists():
            return _classification_result(cfg, reference, "corrupt", "group_missing", task)
        try:
            group = read_json(path)["group"]
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return _classification_result(cfg, reference, "corrupt", "group_invalid", task, exception=exc)
        if group.get("dispatch_state") != "active":
            return _classification_result(cfg, reference, "temporarily_unavailable", "group_paused", task)
    from ..dependencies import dependency_gate

    try:
        gate = dependency_gate(cfg, task)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _classification_result(cfg, reference, "corrupt", "dependency_invalid", task, exception=exc)
    if gate.state == "invalid":
        return _classification_result(cfg, reference, "corrupt", "dependency_invalid", task)
    if gate.state != "ready":
        return _classification_result(cfg, reference, "temporarily_unavailable", f"dependency_{gate.state}", task)
    return _classification_result(cfg, reference, "claimable", "eligible_truth", task)


def task_should_have_ready_marker(task: TaskRecord) -> bool:
    return (
        task.state.get("projection") == "queued"
        and not task.claim_control.get("active_claim")
        and not task.control.get("cleanup_operation_id")
        and not task.control.get("cleanup_state")
        and not task.control.get("cancellation_requested_at")
    )
