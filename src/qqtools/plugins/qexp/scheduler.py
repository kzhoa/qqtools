"""Fenced scheduling pipeline: reserve, claim, materialize, authorize, launch, reconcile."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping

from .config_types import RootConfig
from .domain.policies import group_allows, task_machine_matches
from .events import write_diagnostic_event, write_event
from .executor import Executor, append_launch_failure_diagnostic, launch_failure_handle, launch_failure_reason
from .gpu_policy import GpuReservationPolicy
from .infrastructure.clock import clock_evidence as _clock_evidence
from .infrastructure.process import terminate_process_group as _terminate_process_group
from .launch_policy import validate_launch_handoff_timeout_seconds
from .layout import load_machine_registration
from .lease import (
    AuthorityResolution,
    AuthorityResolutionOutcome,
    ClockObservation,
    LeaseFailureDetails,
    LeaseRenewalOutcome,
    LeaseRenewalResult,
    clock_capability,
    lease_expiry,
    load_lease_policy,
    persist_clock_observation,
    reclaim_allowed_at,
)
from .lifecycle import TerminalTransition, commit_terminal_transition_locked, dispatch_task_lifecycle_hooks_noexcept
from .runtime.authority_lock import authority_locks
from .runtime.authority_scan import is_path_present
from .runtime.claims import archive_claim
from .runtime.dependencies import dependency_gate, dependency_locks
from .runtime.durable_ownership import (
    OwnershipTransitionConflict,
    process_identity_matches,
    transition_attempt_ownership,
)
from .runtime.group_cancellation import has_active_cancellation
from .runtime.group_discovery.changes import record_task_change
from .runtime.group_namespace import read_group
from .runtime.locks import group_lock, schema_writer_lock, task_lock
from .runtime.operation_store import operation_exists
from .runtime.paths import attempt_path, group_path, local_paths, shared_paths, task_path
from .runtime.process_evidence import inspect_group_identity, inspect_wrapper_identity
from .runtime.ready import (
    ReadyMarkerRef,
    advance_ready_index_build,
    classification_diagnostic,
    classify_ready_marker,
    commit_ready_publication,
    delete_stale_ready_marker,
    mark_ready_index_degraded,
    next_ready_marker,
    prepare_ready_transition,
    read_ready_index_state,
    read_ready_index_status,
    retire_current_ready_generation,
    retire_previous_ready_generation,
)
from .runtime.ready import routes as ready_routes
from .runtime.records import AttemptRecord, TaskRecord, TaskSpec, normalize_group_record, utc_now, validate_identifier
from .runtime.resources.cpu_lane import attach_cpu, has_active_cpu_reservation, release_cpu, reserve_cpu
from .runtime.resources.reservations import ReservationIdentity, attach, release, reserve, reserve_admitted
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.submission_control import SubmissionControlUnavailable, read_submission_state
from .runtime.tasks import load_task, save_task
from .runtime.terminal_evidence import load_settled_terminal_attempt
from .runtime.work_budget import AdaptiveBatchSizer, SliceBudget, diagnostic_increment, diagnostic_span

LEASE_SECONDS = 120
TERMINATION_GRACE_SECONDS = 5.0
TERMINATION_POLL_SECONDS = 0.05
_LAUNCH_ID = re.compile(r"[0-9a-f]{32}")


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _bounded_lease_is_current(claim: Mapping[str, Any], attempt: AttemptRecord) -> bool:
    claim_expiry = claim.get("lease_expires_at")
    attempt_expiry = attempt.lease.get("expires_at")
    if claim_expiry != attempt_expiry or not _valid_utc_timestamp(claim_expiry):
        return False
    holder_bound = claim.get("clock_error_bound_seconds")
    if type(holder_bound) not in {int, float} or holder_bound < 0 or not float(holder_bound) < float("inf"):
        return False
    try:
        expires_at = datetime.fromisoformat(
            claim_expiry[:-1] + "+00:00" if claim_expiry.endswith("Z") else claim_expiry
        )
    except ValueError:
        return False
    return expires_at - timedelta(seconds=float(holder_bound)) > datetime.now(timezone.utc)


class BorrowAdmissionRequired(RuntimeError):
    """Raised when a borrow claim is attempted without machine-wide admission."""


def _registration_is_current(record: object) -> bool:
    if not isinstance(record, dict) or record.get("state") != "eligible":
        return False
    try:
        expires_at = datetime.fromisoformat(record["eligibility_expires_at"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return expires_at > datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class _BorrowAdmissionRevision:
    project_id: str
    cfg: RootConfig
    queue_scope: str
    revision: int


@dataclass(frozen=True, slots=True)
class _BorrowAdmissionGrant:
    """Machine-agent capability proving a stable no-primary-demand observation."""

    runtime_root: Path
    revisions: tuple[_BorrowAdmissionRevision, ...]
    lane: str = "gpu"

    def is_valid(self, runtime_root: Path) -> bool:
        if self.runtime_root.resolve() != Path(runtime_root).resolve():
            return False
        from .runtime.ready import is_primary_ready_index_active, ready_index_route_revision

        try:
            return all(
                is_primary_ready_index_active(item.cfg)
                and ready_index_route_revision(item.cfg, item.queue_scope, primary_only=True, lane=self.lane)
                == item.revision
                for item in self.revisions
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False


def _reservation_root(cfg: RootConfig, value: Path | None = None) -> Path:
    return value if value is not None else cfg.runtime_root


def _eligible(cfg: RootConfig, task: TaskRecord) -> bool:
    if task.state["projection"] != "queued" or task.claim_control.get("active_claim"):
        return False
    if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
        return False
    if operation_exists(cfg, "cleanup", task.task_id):
        return False
    if task.control.get("cancellation_requested_at"):
        return False
    operation_id = task.submission_operation_id
    if not operation_id:
        return False
    try:
        submission_state = read_submission_state(cfg, operation_id)
    except (FileNotFoundError, SubmissionControlUnavailable):
        return False
    if submission_state != "committed":
        return False
    if not task.group_name:
        return task_machine_matches(task, cfg.machine_name)
    try:
        group = read_group(cfg.shared_root, task.group_name)
        if has_active_cancellation(cfg, task, group):
            return False
        return group_allows(group, task, cfg.machine_name)
    except FileNotFoundError:
        return False


def _task_worker_role(cfg: RootConfig, task: TaskRecord) -> str | None:
    if not task.group_name:
        return None
    try:
        group = read_group(cfg.shared_root, task.group_name)
    except FileNotFoundError:
        return None
    normalize_group_record(group)
    worker = group["group"]["worker_set"].get(cfg.machine_name)
    return worker.get("scheduling_role") if worker else None


def _admission_role_matches(cfg: RootConfig, task: TaskRecord, admission_role: str | None) -> bool:
    """Return whether a task belongs in the requested admission layer."""
    if admission_role == "borrow":
        return bool(task.group_name and _task_worker_role(cfg, task) == "borrow")
    if admission_role == "primary":
        return not task.group_name or _task_worker_role(cfg, task) == "primary"
    if admission_role is not None and task.group_name:
        return _task_worker_role(cfg, task) == admission_role
    return True


def _claim(
    cfg: RootConfig,
    task_id: str,
    assigned_gpus: list[int],
    *,
    lease_seconds: int,
    authority_mode: str,
    clock_evidence: dict[str, Any] | None,
    reservation_runtime_root: Path,
    project_id: str | None,
    admission_role: str | None,
    borrow_admission_grant: _BorrowAdmissionGrant | None,
    gpu_policy: GpuReservationPolicy | None,
) -> AttemptRecord | None:
    task = load_task(cfg, task_id)
    if task.placement_policy["home_machine"] != cfg.machine_name and not _eligible(cfg, task):
        return None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if not _eligible(cfg, task):
            return None
        if not dependency_gate(cfg, task).is_ready:
            return None
        group: dict[str, Any] | None = None
        worker: dict[str, Any] | None = None
        if task.group_name:
            group = read_group(cfg.shared_root, task.group_name)
            normalize_group_record(group)
            worker = group["group"]["worker_set"].get(cfg.machine_name)
            if worker is None:
                return None
            if admission_role is not None and worker["scheduling_role"] != admission_role:
                return None
            if worker["scheduling_role"] == "borrow" and (
                borrow_admission_grant is None or not borrow_admission_grant.is_valid(reservation_runtime_root)
            ):
                raise BorrowAdmissionRequired("borrow claims require a current machine-agent admission grant.")
        attempt_number = task.attempt_control["next_attempt_number"]
        attempt_id = f"{task.task_id}-attempt-{attempt_number}"
        token = task.claim_control["fencing_epoch"] + 1
        if task.spec.is_cpu_only:
            reservation = reserve_cpu(
                reservation_runtime_root,
                task_id,
                task.spec.requested_cpus or 0,
                attempt_id=attempt_id,
                fencing_token=token,
                project_id=project_id,
                shared_root=str(cfg.shared_root),
                machine_name=cfg.machine_name,
                group_name=task.group_name,
            )
        elif group is not None and worker is not None:
            reservation = reserve_admitted(
                reservation_runtime_root,
                task_id,
                assigned_gpus,
                project_id=project_id or "",
                group_name=task.group_name or "",
                machine_name=cfg.machine_name,
                gpu_limit_gpus=worker["gpu_limit_gpus"],
                worker_scheduling_role=worker["scheduling_role"],
                group_worker_set_epoch=group["group"]["worker_set_epoch"],
                worker_state_epoch=worker["state_epoch"],
                attempt_id=attempt_id,
                fencing_token=token,
                shared_root=str(cfg.shared_root),
                admitted_as_borrow=worker["scheduling_role"] == "borrow",
                gpu_policy=gpu_policy,
            )
        else:
            reservation = reserve(
                reservation_runtime_root,
                task_id,
                assigned_gpus,
                attempt_id=attempt_id,
                fencing_token=token,
                project_id=project_id,
                shared_root=str(cfg.shared_root),
                machine_name=cfg.machine_name,
                gpu_policy=gpu_policy,
            )
        attempt = AttemptRecord.claimed(
            task,
            cfg.machine_name,
            reservation["reservation"].get("gpu_ids", []),
            reservation["reservation"]["reservation_id"],
            token,
            authority_mode=authority_mode,
            clock_evidence=clock_evidence,
            lease_seconds=lease_seconds,
            attempt_id=attempt_id,
        )
        claim = {
            "claim_id": attempt_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "machine_name": cfg.machine_name,
            "reservation_id": attempt.reservation_id,
            "queue_origin": task.placement_runtime["queue_scope"],
            "fencing_token": token,
            "claimed_at": utc_now(),
            "authority_mode": authority_mode,
            "clock_error_bound_seconds": (clock_evidence or {}).get("clock_error_bound_seconds"),
            "clock_provider": (clock_evidence or {}).get("provider"),
            "clock_observation_id": (clock_evidence or {}).get("observation_id"),
            "lease_expires_at": attempt.lease["expires_at"],
            "launch_state": "claimed",
            "launch_authorized_at": None,
            "group_dispatch_epoch": None,
            "group_worker_set_epoch": None,
        }
        if task.group_name and group is not None and worker is not None:
            claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
            claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
            claim["worker_state_epoch"] = worker["state_epoch"]
            claim["worker_scheduling_role"] = worker["scheduling_role"]
            claim["gpu_limit_gpus"] = worker["gpu_limit_gpus"]
            claim["admitted_as_borrow"] = worker["scheduling_role"] == "borrow"
            attempt.authorization["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
            attempt.authorization["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
            attempt.authorization["worker_state_epoch"] = worker["state_epoch"]
            attempt.authorization["worker_scheduling_role"] = worker["scheduling_role"]
            attempt.authorization["gpu_limit_gpus"] = worker["gpu_limit_gpus"]
            attempt.authorization["admitted_as_borrow"] = worker["scheduling_role"] == "borrow"
        with record_task_change(
            cfg,
            task,
            "claim",
            details={
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": token,
            },
        ):
            task.claim_control.update({"fencing_epoch": token, "active_claim": claim})
            task.attempt_control.update(
                {
                    "current_attempt_id": attempt_id,
                    "current_attempt_number": attempt_number,
                    "next_attempt_number": attempt_number + 1,
                }
            )
            task.state["projection"] = "running"
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            try:
                atomic_replace(attempt_path(cfg.shared_root, task_id, attempt_number), attempt.to_dict())
                if task.spec.is_cpu_only:
                    attach_cpu(reservation_runtime_root, attempt.reservation_id, attempt.attempt_id, token)
                else:
                    attach(reservation_runtime_root, attempt.reservation_id, attempt.attempt_id, token)
            except Exception:
                _release_claim_locked(
                    cfg,
                    task,
                    token,
                    "attempt_materialization_failed",
                    reservation_runtime_root=reservation_runtime_root,
                )
                raise
            retire_current_ready_generation(cfg, task)
            return attempt


def _release_claim_locked(
    cfg: RootConfig,
    task: TaskRecord,
    token: int,
    reason: str,
    *,
    reservation_runtime_root: Path | None = None,
) -> None:
    claim = task.claim_control.get("active_claim") or {}
    if claim.get("fencing_token") != token:
        return
    with record_task_change(
        cfg,
        task,
        "claim_loss",
        details={
            "expected_attempt_id": claim.get("attempt_id"),
            "expected_fencing_token": token,
            "transition": "claim_materialization_failed",
            "reason": reason,
        },
    ):
        archive_claim(cfg, task.task_id, claim, reason)
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": "queued", "reason": reason})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
    reservation_root = reservation_runtime_root or cfg.runtime_root
    if task.spec.is_cpu_only:
        release_cpu(reservation_root, claim["reservation_id"], reason)
    else:
        release(reservation_root, claim["reservation_id"], reason)


def _has_active_launch_reservation(
    task: TaskRecord,
    claim: dict[str, Any],
    reservation_runtime_root: Path,
) -> bool:
    if not task.spec.is_cpu_only:
        return True
    reservation_id = claim.get("reservation_id")
    attempt_id = claim.get("attempt_id")
    fencing_token = claim.get("fencing_token")
    return (
        isinstance(reservation_id, str)
        and isinstance(attempt_id, str)
        and isinstance(fencing_token, int)
        and has_active_cpu_reservation(
            reservation_runtime_root,
            reservation_id,
            task_id=task.task_id,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
        )
    )


def _release_task_reservation(
    task: TaskRecord, reservation_runtime_root: Path, reservation_id: str, reason: str
) -> None:
    """Release the reservation backend selected by the Task lane."""
    if task.spec.is_cpu_only:
        release_cpu(reservation_runtime_root, reservation_id, reason)
    else:
        release(reservation_runtime_root, reservation_id, reason)


def claim_task(
    cfg: RootConfig,
    task_id: str,
    assigned_gpus: list[int],
    *,
    lease_seconds: int | None = None,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
    admission_role: str | None = None,
    borrow_admission_grant: _BorrowAdmissionGrant | None = None,
    gpu_policy: GpuReservationPolicy | None = None,
) -> AttemptRecord | None:
    task = load_task(cfg, task_id)
    if task.placement_policy["home_machine"] != cfg.machine_name and not _eligible(cfg, task):
        return None
    policy = load_lease_policy(cfg)
    capability = clock_capability(cfg, policy)
    authority_mode = "bounded_lease" if capability.is_healthy else "holder_bound"
    evidence = None
    if capability.observation:
        persist_clock_observation(cfg, capability.observation)
        evidence = _clock_evidence(capability.observation)
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    if lease_seconds is None:
        lease_seconds = policy.ttl_seconds
    return _claim(
        cfg,
        task_id,
        assigned_gpus,
        lease_seconds=lease_seconds,
        authority_mode=authority_mode,
        clock_evidence=evidence,
        reservation_runtime_root=reservation_runtime_root,
        project_id=project_id,
        admission_role=admission_role,
        borrow_admission_grant=borrow_admission_grant,
        gpu_policy=gpu_policy,
    )


def claim_project_io_candidate(
    cfg: RootConfig,
    candidate: dict[str, Any],
    offer: dict[str, Any],
    *,
    project_id: str,
    registration_generation: str,
    admission_role: str,
    source_revisions: dict[str, int],
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Commit one observed candidate using its already-owned local resource offer.

    This path writes only Project Task and Attempt truth.  In particular, it
    never consults or changes either machine-local reservation backend.
    """
    no_claim = lambda reason: {"outcome": "no_claim", "reason": reason}
    task_id = candidate.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return no_claim("offer_mismatch")
    if admission_role not in {"primary", "borrow"}:
        return no_claim("admission_role_changed")
    if candidate.get("admission_role") != admission_role or (
        candidate.get("group_name") is not None and offer.get("worker_scheduling_role") != admission_role
    ):
        return no_claim("admission_role_changed")
    if admission_role == "borrow" and (candidate.get("group_name") is None or offer.get("group_name") is None):
        return no_claim("admission_role_changed")
    expected_source_revisions = {
        "ready_index": source_revisions.get("ready_index"),
        "task": candidate.get("task_revision"),
        "ready_catalog": candidate.get("catalog_revision"),
        "ready_partition": candidate.get("ready_revision"),
    }
    if candidate.get("group_revision") is not None:
        expected_source_revisions["group"] = candidate.get("group_revision")
    if source_revisions != expected_source_revisions:
        return no_claim("ready_changed")
    ready_status = read_ready_index_status(cfg)
    if ready_status.get("state") != "active" or ready_status.get("revision") != source_revisions["ready_index"]:
        return no_claim("ready_changed")
    if (
        offer.get("project_id") != project_id
        or offer.get("shared_root") != str(cfg.shared_root)
        or offer.get("registration_generation") != registration_generation
        or offer.get("task_id") != task_id
        or offer.get("lane") != candidate.get("lane")
        or offer.get("attempt_id") != candidate.get("attempt_id")
        or offer.get("attempt_number") != candidate.get("attempt_number")
        or offer.get("fencing_token") != candidate.get("fencing_token")
        or offer.get("group_name") != candidate.get("group_name")
        or offer.get("admitted_as_borrow") is not (admission_role == "borrow")
        or any(
            offer.get(field) != candidate.get(field)
            for field in (
                "group_dispatch_epoch",
                "group_worker_set_epoch",
                "worker_state_epoch",
                "worker_scheduling_role",
                "gpu_limit_gpus",
            )
        )
        or not isinstance(offer.get("reservation_id"), str)
    ):
        return no_claim("offer_mismatch")

    policy = load_lease_policy(cfg)
    capability = clock_capability(cfg, policy)
    authority_mode = "bounded_lease" if capability.is_healthy else "holder_bound"
    clock_evidence = None
    if capability.observation:
        mutation_fence()
        persist_clock_observation(cfg, capability.observation)
        clock_evidence = _clock_evidence(capability.observation)

    initial = load_task(cfg, task_id)
    if initial.group_name != candidate.get("group_name"):
        return no_claim("group_changed")
    with authority_locks(cfg, initial):
        registration_envelope = load_machine_registration(cfg) or {}
        registration = registration_envelope.get("registration")
        if (
            not isinstance(registration, dict)
            or registration.get("project_id") != project_id
            or registration.get("shared_root") != str(cfg.shared_root)
            or registration.get("machine_name") != cfg.machine_name
            or registration.get("generation") != registration_generation
            or not _registration_is_current(registration)
        ):
            return no_claim("candidate_stale")
        task = load_task(cfg, task_id)
        if task.group_name != initial.group_name:
            return no_claim("group_changed")
        if task.meta.get("revision") != candidate.get("task_revision"):
            return no_claim("task_changed")
        if task.ready_generation != candidate.get("ready_generation"):
            return no_claim("ready_changed")
        if (
            task.attempt_control.get("next_attempt_number") != candidate.get("attempt_number")
            or task.claim_control.get("fencing_epoch", -1) + 1 != candidate.get("fencing_token")
            or candidate.get("attempt_id") != f"{task_id}-attempt-{candidate.get('attempt_number')}"
        ):
            return no_claim("task_changed")
        if (
            (candidate.get("lane") == "cpu") != task.spec.is_cpu_only
            or (candidate.get("lane") == "gpu" and task.spec.requested_gpus != candidate.get("requested_gpus"))
            or (candidate.get("lane") == "cpu" and (task.spec.requested_cpus or 0) != candidate.get("requested_cpus"))
        ):
            return no_claim("task_changed")
        if not _eligible(cfg, task):
            return no_claim("candidate_stale")
        if not dependency_gate(cfg, task).is_ready:
            return no_claim("dependency_not_ready")

        reference = ReadyMarkerRef(
            task_id=task_id,
            generation=candidate["ready_generation"],
            queue_scope=candidate["ready_scope"],
            home_machine=candidate["home_machine"],
            partition=candidate["partition"],
            catalog_page=candidate["catalog_page"],
            marker_name=candidate["marker_name"],
        )
        if (
            reference.identity != candidate.get("ready_identity")
            or reference.marker_name != f"{task_id}.{reference.generation}.json"
            or reference.queue_scope != task.placement_runtime.get("queue_scope")
            or reference.home_machine != task.placement_policy.get("home_machine")
        ):
            return no_claim("ready_changed")
        classification = classify_ready_marker(cfg, reference, read_only=True)
        if classification.classification != "claimable" or classification.task is None:
            return no_claim(
                "dependency_not_ready" if classification.reason.startswith("dependency_") else "ready_changed"
            )
        if classification.task.meta.get("revision") != candidate.get("task_revision"):
            return no_claim("task_changed")
        route = ready_routes.route_key(reference.queue_scope, cfg.machine_name)
        catalog = read_json(ready_routes.catalog_path(cfg.shared_root, route, reference.catalog_page))["ready_catalog"]
        partition = read_json(
            ready_routes.partition_record_path(
                cfg.shared_root,
                reference.queue_scope,
                cfg.machine_name,
                reference.partition,
            )
        )["ready_partition"]
        if (
            catalog.get("route") != route
            or catalog.get("page") != reference.catalog_page
            or reference.partition not in catalog.get("partitions", [])
            or partition.get("route") != route
            or partition.get("partition") != reference.partition
            or partition.get("catalog_page") != reference.catalog_page
            or reference.marker_name not in partition.get("slots", [])
        ):
            return no_claim("ready_changed")
        if catalog.get("revision") != candidate.get("catalog_revision") or partition.get("revision") != candidate.get(
            "ready_revision"
        ):
            return no_claim("ready_changed")

        group: dict[str, Any] | None = None
        worker: dict[str, Any] | None = None
        if task.group_name:
            try:
                group_value = normalize_group_record(read_group(cfg.shared_root, task.group_name))
            except FileNotFoundError:
                return no_claim("group_changed")
            group = group_value["group"]
            group_meta = group_value.get("meta")
            worker = group["worker_set"].get(cfg.machine_name)
            if (
                not isinstance(group_meta, dict)
                or group_meta.get("revision") != candidate.get("group_revision")
                or group.get("dispatch_epoch") != candidate.get("group_dispatch_epoch")
                or group.get("worker_set_epoch") != candidate.get("group_worker_set_epoch")
            ):
                return no_claim("group_changed")
            if worker is None or worker.get("state") != "active" or worker.get("scheduling_role") != admission_role:
                return no_claim("admission_role_changed")
            if (
                worker.get("state_epoch", 0) != candidate.get("worker_state_epoch")
                or worker.get("scheduling_role") != candidate.get("worker_scheduling_role")
                or worker.get("gpu_limit_gpus") != candidate.get("gpu_limit_gpus")
            ):
                return no_claim("group_changed")
            if task.spec.is_cpu_only is False and worker.get("gpu_limit_gpus") is not None:
                if task.spec.requested_gpus > worker["gpu_limit_gpus"]:
                    return no_claim("group_changed")
        elif admission_role != "primary" or candidate.get("admission_role") != "primary":
            return no_claim("admission_role_changed")
        if not task_machine_matches(task, cfg.machine_name):
            return no_claim("candidate_stale")
        if task.group_name and group is not None and has_active_cancellation(cfg, task, group_value):
            return no_claim("candidate_stale")

        gpu_ids = list(offer["gpu_ids"])
        if (task.spec.is_cpu_only and (gpu_ids or offer.get("cpu_slots") != (task.spec.requested_cpus or 0))) or (
            not task.spec.is_cpu_only and (len(gpu_ids) != task.spec.requested_gpus or offer.get("cpu_slots") != 0)
        ):
            return no_claim("offer_mismatch")
        attempt_number = candidate["attempt_number"]
        attempt_file = attempt_path(cfg.shared_root, task_id, attempt_number)
        if attempt_file.exists():
            return no_claim("task_changed")
        token = candidate["fencing_token"]
        attempt = AttemptRecord.claimed(
            task,
            cfg.machine_name,
            gpu_ids,
            offer["reservation_id"],
            token,
            authority_mode=authority_mode,
            clock_evidence=clock_evidence,
            lease_seconds=policy.ttl_seconds,
            attempt_id=candidate["attempt_id"],
        )
        claim = {
            "claim_id": attempt.attempt_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt_number,
            "machine_name": cfg.machine_name,
            "reservation_id": offer["reservation_id"],
            "queue_origin": task.placement_runtime["queue_scope"],
            "fencing_token": token,
            "claimed_at": utc_now(),
            "authority_mode": authority_mode,
            "clock_error_bound_seconds": (clock_evidence or {}).get("clock_error_bound_seconds"),
            "clock_provider": (clock_evidence or {}).get("provider"),
            "clock_observation_id": (clock_evidence or {}).get("observation_id"),
            "lease_expires_at": attempt.lease["expires_at"],
            "launch_state": "claimed",
            "launch_authorized_at": None,
            "group_dispatch_epoch": None,
            "group_worker_set_epoch": None,
        }
        if task.group_name and group is not None and worker is not None:
            claim.update(
                {
                    "group_dispatch_epoch": group["dispatch_epoch"],
                    "group_worker_set_epoch": group["worker_set_epoch"],
                    "worker_state_epoch": worker["state_epoch"],
                    "worker_scheduling_role": worker["scheduling_role"],
                    "gpu_limit_gpus": worker["gpu_limit_gpus"],
                    "admitted_as_borrow": admission_role == "borrow",
                }
            )
            attempt.authorization.update(
                {
                    "group_dispatch_epoch": group["dispatch_epoch"],
                    "group_worker_set_epoch": group["worker_set_epoch"],
                    "worker_state_epoch": worker["state_epoch"],
                    "worker_scheduling_role": worker["scheduling_role"],
                    "gpu_limit_gpus": worker["gpu_limit_gpus"],
                    "admitted_as_borrow": admission_role == "borrow",
                }
            )
        attempt_persisted = False
        mutation_fence()
        with record_task_change(
            cfg,
            task,
            "claim",
            details={"attempt_id": attempt.attempt_id, "attempt_number": attempt_number, "fencing_token": token},
        ):
            mutation_fence()
            task.claim_control.update({"fencing_epoch": token, "active_claim": claim})
            task.attempt_control.update(
                {
                    "current_attempt_id": attempt.attempt_id,
                    "current_attempt_number": attempt_number,
                    "next_attempt_number": attempt_number + 1,
                }
            )
            task.state["projection"] = "running"
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            try:
                atomic_replace(attempt_file, attempt.to_dict())
                attempt_persisted = True
            except Exception:
                _rollback_project_io_claim_locked(cfg, task, token, attempt_file, attempt_persisted)
                raise
        mutation_fence()
        retire_current_ready_generation(cfg, task)
        return {
            "outcome": "claimed",
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": token,
            "reservation_id": offer["reservation_id"],
            "gpu_ids": gpu_ids,
            "cpu_slots": offer["cpu_slots"],
        }


def _rollback_project_io_claim_locked(
    cfg: RootConfig,
    task: TaskRecord,
    token: int,
    attempt_file: Path,
    attempt_persisted: bool,
) -> None:
    """Undo only the shared claim artifacts; the executor retains its offer."""
    claim = task.claim_control.get("active_claim") or {}
    if claim.get("fencing_token") != token:
        return
    with record_task_change(
        cfg,
        task,
        "claim_loss",
        details={
            "expected_attempt_id": claim.get("attempt_id"),
            "expected_fencing_token": token,
            "transition": "claim_materialization_failed",
            "reason": "attempt_materialization_failed",
        },
    ):
        archive_claim(cfg, task.task_id, claim, "attempt_materialization_failed")
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": "queued", "reason": "attempt_materialization_failed"})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
    if attempt_persisted:
        try:
            attempt_file.unlink(missing_ok=True)
        except OSError:
            pass


def _has_local_launch_evidence(cfg: RootConfig, attempt_id: str) -> bool:
    paths = local_paths(cfg.runtime_root)
    return any(
        is_path_present(paths[directory] / f"{attempt_id}.json")
        for directory in ("launch_intents", "registrations", "processes", "observations")
    )


def _persist_starting_pair_locked(
    cfg: RootConfig,
    task: TaskRecord,
    attempt: AttemptRecord,
    *,
    launch_id: str,
    authorized_at: str,
    mutation_fence: Callable[[], None] | None = None,
    launch_handoff_timeout_seconds: int | float | None = None,
    task_writer: Callable[[TaskRecord], None] | None = None,
) -> None:
    if task_writer is None:
        task_writer = lambda value: save_task(cfg, value)

    def write_attempt(value: AttemptRecord) -> None:
        target = attempt_path(cfg.shared_root, value.task_id, value.attempt_number)
        if mutation_fence is None:
            atomic_replace(target, value.to_dict())
        else:
            atomic_replace(target, value.to_dict(), before_replace=lambda _stat: mutation_fence())

    result = transition_attempt_ownership(
        cfg,
        task,
        attempt,
        target_phase="starting",
        launch_id=launch_id,
        authorized_at=authorized_at,
        launch_handoff_timeout_seconds=launch_handoff_timeout_seconds,
        mutation_fence=mutation_fence,
        task_writer=task_writer,
        attempt_writer=write_attempt,
    )
    # Replay repair returns a projected copy because its Task is already
    # durable. Keep the caller's Attempt object equally current.
    for field in (
        "phase",
        "lease",
        "authority_mode",
        "authorization",
        "timestamps",
    ):
        setattr(attempt, field, copy.deepcopy(getattr(result.attempt, field)))


def resume_starting_attempt(
    cfg: RootConfig,
    task_id: str,
    *,
    reservation_runtime_root: Path | None = None,
    expected_reservation: ReservationIdentity | None = None,
) -> AttemptRecord | None:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    if task.task_id != task_id:
        return None
    cancel_result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if task.task_id != task_id:
            return None
        claim = task.claim_control.get("active_claim") or {}
        attempt_id = claim.get("attempt_id")
        attempt_number = task.attempt_control.get("current_attempt_number")
        fencing_token = claim.get("fencing_token")
        if (
            task.state.get("projection") != "running"
            or claim.get("machine_name") != cfg.machine_name
            or claim.get("launch_state") not in {"claimed", "starting"}
            or not isinstance(attempt_id, str)
            or type(attempt_number) is not int
            or not isinstance(fencing_token, int)
            or _has_local_launch_evidence(cfg, attempt_id)
        ):
            return None
        if expected_reservation is not None and (
            expected_reservation.task_id != task_id
            or expected_reservation.attempt_id != attempt_id
            or expected_reservation.fencing_token != fencing_token
            or expected_reservation.reservation_id != claim.get("reservation_id")
        ):
            return None
        try:
            attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, attempt_number)))
        except (FileNotFoundError, KeyError, ValueError):
            return None
        if (
            attempt.task_id != task_id
            or type(attempt.attempt_number) is not int
            or attempt.attempt_number != attempt_number
            or attempt.attempt_id != attempt_id
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != cfg.machine_name
            or attempt.phase not in {"claimed", "starting"}
            or not isinstance(attempt.authorization, dict)
        ):
            return None

        # A Task-first authorization may have reached durable storage while
        # the companion Attempt write was interrupted.  Replay that exact
        # receipt before evaluating ordinary mode/gate state; this is the
        # only path allowed to repair the stranded Attempt projection.
        if claim.get("launch_state") == "starting":
            try:
                _persist_starting_pair_locked(
                    cfg,
                    task,
                    attempt,
                    launch_id=claim.get("launch_id"),
                    authorized_at=claim.get("launch_authorized_at"),
                    launch_handoff_timeout_seconds=claim.get("launch_handoff_timeout_seconds"),
                )
            except (OwnershipTransitionConflict, TypeError, ValueError):
                return None
            claim = task.claim_control.get("active_claim") or {}
        launch_id = claim.get("launch_id")
        has_committed_authorization = (
            claim.get("launch_state") == "starting"
            and attempt.phase == "starting"
            and attempt.authority_mode == "holder_bound"
            and isinstance(launch_id, str)
            and bool(launch_id)
            and attempt.authorization.get("launch_id") == launch_id
        )
        if task.control.get("cancellation_requested_at") and (
            not has_committed_authorization or task.control.get("terminate_running")
        ):
            cancel_result = _cancel_prelaunch_locked(cfg, task, "cancelled_before_launch", {"claimed", "starting"})
        elif task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return None
        elif operation_exists(cfg, "cleanup", task.task_id):
            return None
        elif not _has_active_launch_reservation(task, claim, reservation_runtime_root):
            cancel_result = _cancel_prelaunch_locked(cfg, task, "launch_reservation_lost", {"claimed", "starting"})
        else:
            if task.group_name:
                group = read_group(cfg.shared_root, task.group_name)
                if has_active_cancellation(cfg, task, group, include_default=not has_committed_authorization):
                    cancel_result = _cancel_prelaunch_locked(
                        cfg, task, "group_cancelled_before_launch", {"claimed", "starting"}
                    )
                elif not has_committed_authorization and not group_allows(group, task, cfg.machine_name):
                    cancel_result = _cancel_prelaunch_locked(
                        cfg, task, "worker_or_dispatch_changed", {"claimed", "starting"}
                    )
            if cancel_result is None:
                if not (claim.get("authority_mode") != "bounded_lease" or clock_capability(cfg).is_healthy):
                    return None
                if claim.get("authority_mode") == "bounded_lease" and not _bounded_lease_is_current(claim, attempt):
                    return None
                launch_id = launch_id if has_committed_authorization else uuid.uuid4().hex
                authorized_at = claim.get("launch_authorized_at") if has_committed_authorization else utc_now()
                change_context = (
                    nullcontext()
                    if has_committed_authorization
                    else record_task_change(
                        cfg,
                        task,
                        "launch",
                        details={
                            "attempt_id": attempt_id,
                            "attempt_number": attempt_number,
                            "fencing_token": fencing_token,
                            "launch_id": launch_id,
                        },
                    )
                )
                with change_context:
                    _persist_starting_pair_locked(
                        cfg,
                        task,
                        attempt,
                        launch_id=launch_id,
                        authorized_at=authorized_at,
                        launch_handoff_timeout_seconds=claim.get("launch_handoff_timeout_seconds"),
                    )
                return attempt
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            _release_task_reservation(
                task,
                reservation_runtime_root,
                cancel_result.reservation_id,
                cancel_result.reason or "cancelled_before_launch",
            )
        if cancel_result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, cancel_result.event)
    return None


def authorize_project_io_launch(
    cfg: RootConfig,
    claim_identity: Mapping[str, Any],
    *,
    launch_handoff_timeout_seconds: int | float,
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Authorize one exact claim without touching local resource reservations or launching work."""
    fields = {"task_id", "attempt_id", "attempt_number", "fencing_token", "reservation_id"}
    if not isinstance(claim_identity, Mapping) or set(claim_identity) != fields:
        raise ValueError("launch authorization identity has missing or unknown fields.")
    identity = dict(claim_identity)
    for field in ("task_id", "attempt_id", "reservation_id"):
        validate_identifier(identity[field], f"claim_identity.{field}")
    for field in ("attempt_number", "fencing_token"):
        if type(identity[field]) is not int or identity[field] < 1:
            raise ValueError(f"claim_identity.{field} must be a positive integer.")
    task_id = identity["task_id"]
    attempt_number = identity["attempt_number"]
    attempt_id = identity["attempt_id"]
    fencing_token = identity["fencing_token"]
    reservation_id = identity["reservation_id"]
    if attempt_id != f"{task_id}-attempt-{attempt_number}":
        raise ValueError("launch authorization attempt identity is inconsistent.")
    timeout_seconds = validate_launch_handoff_timeout_seconds(launch_handoff_timeout_seconds)

    def denied(reason: str) -> dict[str, Any]:
        return {"outcome": "denied", "claim_identity": identity, "reason": reason}

    def authorized(launch_id: str, resolved_timeout: int | float) -> dict[str, Any]:
        return {
            "outcome": "authorized",
            "claim_identity": identity,
            "launch_id": launch_id,
            "launch_handoff_timeout_seconds": validate_launch_handoff_timeout_seconds(resolved_timeout),
        }

    try:
        initial = load_task(cfg, task_id)
    except (FileNotFoundError, KeyError, ValueError):
        return denied("claim_not_current")
    if initial.task_id != task_id:
        return denied("claim_not_current")

    with authority_locks(cfg, initial):
        try:
            task = load_task(cfg, task_id)
        except (FileNotFoundError, KeyError, ValueError):
            return denied("claim_not_current")
        claim = task.claim_control.get("active_claim") or {}
        if (
            task.state.get("projection") != "running"
            or task.attempt_control.get("current_attempt_id") != attempt_id
            or task.attempt_control.get("current_attempt_number") != attempt_number
            or claim.get("attempt_id") != attempt_id
            or claim.get("attempt_number") != attempt_number
            or claim.get("fencing_token") != fencing_token
            or claim.get("reservation_id") != reservation_id
            or claim.get("machine_name") != cfg.machine_name
            or claim.get("launch_state") not in {"claimed", "starting"}
        ):
            return denied("claim_not_current")

        try:
            attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, attempt_number)))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return denied("attempt_not_current")
        if (
            attempt.task_id != task_id
            or attempt.attempt_number != attempt_number
            or attempt.attempt_id != attempt_id
            or attempt.current_fencing_token != fencing_token
            or attempt.reservation_id != reservation_id
            or attempt.machine_name != cfg.machine_name
            or attempt.phase not in {"claimed", "starting"}
        ):
            return denied("attempt_not_current")

        authority_mode = claim.get("authority_mode")
        if authority_mode not in {"bounded_lease", "holder_bound"}:
            return denied("authority_unavailable")

        # Once the Task-side launch gate commits, later pause, cancellation,
        # membership, or clock changes cannot revoke it. Exact replay either
        # returns the durable identity or repairs only its matching Attempt.
        if claim.get("launch_state") == "starting":
            launch_id = claim.get("launch_id")
            authorized_at = claim.get("launch_authorized_at")
            if not isinstance(launch_id, str) or not launch_id:
                return denied("attempt_not_current")
            if not _valid_utc_timestamp(authorized_at):
                return denied("attempt_not_current")
            stored_timeout = claim.get("launch_handoff_timeout_seconds")
            if stored_timeout is None:
                stored_timeout = attempt.authorization.get("launch_handoff_timeout_seconds")
            try:
                if stored_timeout is not None:
                    stored_timeout = validate_launch_handoff_timeout_seconds(stored_timeout)
            except (TypeError, ValueError):
                return denied("attempt_not_current")

            def write_task(value: TaskRecord) -> None:
                atomic_replace(
                    task_path(cfg.shared_root, task_id),
                    value.to_dict(),
                    before_replace=lambda _stat: mutation_fence(),
                )

            try:
                _persist_starting_pair_locked(
                    cfg,
                    task,
                    attempt,
                    launch_id=launch_id,
                    authorized_at=authorized_at,
                    launch_handoff_timeout_seconds=stored_timeout,
                    mutation_fence=mutation_fence,
                    task_writer=write_task,
                )
            except (OwnershipTransitionConflict, TypeError, ValueError):
                return denied("attempt_not_current")
            refreshed_claim = task.claim_control.get("active_claim") or {}
            resolved_timeout = refreshed_claim.get("launch_handoff_timeout_seconds", stored_timeout)
            if resolved_timeout is None:
                resolved_timeout = timeout_seconds
            return authorized(launch_id, resolved_timeout)
        if attempt.authority_mode != authority_mode:
            return denied("authority_unavailable")
        if claim.get("launch_state") != "claimed" or attempt.phase != "claimed":
            return denied("attempt_not_current")

        if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return denied("cleanup_pending")
        try:
            if operation_exists(cfg, "cleanup", task_id):
                return denied("cleanup_pending")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return denied("cleanup_pending")
        if task.control.get("cancellation_requested_at"):
            return denied("cancellation_pending")

        group: dict[str, Any] | None = None
        if task.group_name:
            try:
                group = read_group(cfg.shared_root, task.group_name)
                if has_active_cancellation(cfg, task, group):
                    return denied("cancellation_pending")
                if not group_allows(group, task, cfg.machine_name):
                    return denied("group_gate_closed")
            except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError):
                return denied("group_gate_closed")

        if authority_mode == "bounded_lease":
            if not _bounded_lease_is_current(claim, attempt):
                return denied("authority_unavailable")
            try:
                if not clock_capability(cfg).is_healthy:
                    return denied("clock_unhealthy")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return denied("clock_unhealthy")

        authorized_at = utc_now()
        launch_id = uuid.uuid4().hex
        details = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "launch_id": launch_id,
        }
        # Group change journaling can make its own shared writes. Fence before
        # entering it, then fence immediately before the Task and Attempt writes.
        mutation_fence()
        with record_task_change(cfg, task, "launch", details=details, mutation_fence=mutation_fence):
            if group is not None:
                claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
                claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]

            def write_task(value: TaskRecord) -> None:
                atomic_replace(
                    task_path(cfg.shared_root, task_id),
                    value.to_dict(),
                    before_replace=lambda _stat: mutation_fence(),
                )

            try:
                _persist_starting_pair_locked(
                    cfg,
                    task,
                    attempt,
                    launch_id=launch_id,
                    authorized_at=authorized_at,
                    launch_handoff_timeout_seconds=timeout_seconds,
                    mutation_fence=mutation_fence,
                    task_writer=write_task,
                )
            except OwnershipTransitionConflict:
                return denied("attempt_not_current")
        return authorized(launch_id, timeout_seconds)


def authorize_launch(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    *,
    reservation_runtime_root: Path | None = None,
    write_guard: Callable[[], ContextManager[bool]] | None = None,
) -> bool:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    if task.task_id != task_id:
        return False
    cancel_result = None

    @contextmanager
    def _authority_write_guard() -> Iterator[bool]:
        if write_guard is None:
            yield True
            return
        with write_guard() as is_eligible:
            yield is_eligible

    with _authority_write_guard() as is_eligible:
        if not is_eligible:
            return False
        with authority_locks(cfg, task):
            task = load_task(cfg, task_id)
            if task.task_id != task_id:
                return False
            claim = task.claim_control.get("active_claim") or {}
            if claim.get("machine_name") != cfg.machine_name:
                return False
            if claim.get("authority_mode") not in {"bounded_lease", "holder_bound"}:
                return False
            if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
                return False
            attempt_number = task.attempt_control.get("current_attempt_number")
            if type(attempt_number) is not int:
                return False
            try:
                attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, attempt_number)))
            except (FileNotFoundError, KeyError, ValueError):
                return False
            if (
                attempt.task_id != task_id
                or type(attempt.attempt_number) is not int
                or attempt.attempt_number != attempt_number
                or attempt.attempt_id != attempt_id
                or attempt.current_fencing_token != fencing_token
                or attempt.machine_name != cfg.machine_name
            ):
                return False
            if claim.get("launch_state") == "starting":
                launch_id = claim.get("launch_id")
                authorized_at = claim.get("launch_authorized_at")
                if not isinstance(launch_id, str) or not launch_id or not _valid_utc_timestamp(authorized_at):
                    return False
                try:
                    _persist_starting_pair_locked(
                        cfg,
                        task,
                        attempt,
                        launch_id=launch_id,
                        authorized_at=authorized_at,
                        launch_handoff_timeout_seconds=claim.get("launch_handoff_timeout_seconds"),
                    )
                except (OwnershipTransitionConflict, TypeError, ValueError):
                    return False
                return True
            if attempt.authority_mode != claim.get("authority_mode"):
                return False
            if claim.get("authority_mode") == "bounded_lease":
                if not clock_capability(cfg).is_healthy or not _bounded_lease_is_current(claim, attempt):
                    return False
            if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
                return False
            if operation_exists(cfg, "cleanup", task.task_id):
                return False
            if (
                claim.get("launch_state") != "claimed"
                or attempt.phase != "claimed"
                or task.control.get("cancellation_requested_at")
            ):
                cancel_result = _cancel_prelaunch_locked(cfg, task, "launch_gate_lost")
            elif not _has_active_launch_reservation(task, claim, reservation_runtime_root):
                cancel_result = _cancel_prelaunch_locked(cfg, task, "launch_reservation_lost")
            elif task.group_name:
                group = read_group(cfg.shared_root, task.group_name)
                if has_active_cancellation(cfg, task, group):
                    cancel_result = _cancel_prelaunch_locked(cfg, task, "group_cancelled_before_launch")
                elif not group_allows(group, task, cfg.machine_name):
                    cancel_result = _cancel_prelaunch_locked(cfg, task, "worker_or_dispatch_changed")
                else:
                    claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
                    claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
            if cancel_result is None:
                authorized_at = utc_now()
                launch_id = uuid.uuid4().hex
                with record_task_change(
                    cfg,
                    task,
                    "launch",
                    details={
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        "fencing_token": fencing_token,
                        "launch_id": launch_id,
                    },
                ):
                    try:
                        _persist_starting_pair_locked(
                            cfg,
                            task,
                            attempt,
                            launch_id=launch_id,
                            authorized_at=authorized_at,
                        )
                    except OwnershipTransitionConflict:
                        return False
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            _release_task_reservation(
                task,
                reservation_runtime_root,
                cancel_result.reservation_id,
                cancel_result.reason or "cancelled",
            )
        if cancel_result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, cancel_result.event)
        return False
    return True


def _cancel_prelaunch_locked(cfg: RootConfig, task: TaskRecord, reason: str, attempt_phases: set[str] | None = None):
    claim = task.claim_control.get("active_claim") or {}
    attempt_id = task.attempt_control.get("current_attempt_id")
    if not attempt_id:
        return None
    return commit_terminal_transition_locked(
        cfg,
        task,
        TerminalTransition(
            task.task_id,
            attempt_id,
            task.attempt_control["current_attempt_number"],
            claim.get("fencing_token", 0),
            "cancelled",
            reason,
            None,
            frozenset({"running"}),
            frozenset(attempt_phases or {"claimed"}),
            "active",
            allow_missing_attempt=True,
        ),
    )


def cancel_task(
    cfg: RootConfig,
    task_id: str,
    *,
    terminate_running: bool = True,
    reservation_runtime_root: Path | None = None,
) -> TaskRecord:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    cancel_result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if (
            task.control.get("cleanup_operation_id")
            or task.control.get("cleanup_state")
            or operation_exists(cfg, "cleanup", task_id)
        ):
            raise ValueError(f"Task {task_id!r} is being cleaned and cannot be cancelled.")
        if task.state["projection"] in {"succeeded", "failed", "cancelled"}:
            return task
        claim = task.claim_control.get("active_claim") or {}
        has_saved_task = False
        task.control.update(
            {
                "cancellation_requested_at": utc_now(),
                "terminate_running": terminate_running,
                "requested_by": cfg.machine_name,
            }
        )
        # Prelaunch cancellation has its own terminal recovery owner. An outer
        # task_cancel record would block that owner at the ordered journal head.
        if claim and claim.get("launch_state") == "claimed":
            cancel_result = _cancel_prelaunch_locked(cfg, task, "cancelled_before_launch")
            has_saved_task = cancel_result is not None and cancel_result.outcome == "committed"
            result_task = load_task(cfg, task_id) if cancel_result is None else None
        else:
            result_task = None
        if not has_saved_task:
            with record_task_change(
                cfg,
                task,
                "task_cancel",
                details={"terminate_running": terminate_running, "expected_effect": "task_cancel"},
            ):
                if not claim and task.state["projection"] == "queued":
                    task.state.update({"projection": "cancelled", "reason": "cancelled_by_user"})
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
                if task.state["projection"] == "cancelled":
                    retire_current_ready_generation(cfg, task)
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            _release_task_reservation(
                task,
                reservation_runtime_root,
                cancel_result.reservation_id,
                cancel_result.reason or "cancelled_before_launch",
            )
        if cancel_result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, cancel_result.event)
        return load_task(cfg, task_id)
    return result_task or load_task(cfg, task_id)


def _load_current_attempt(cfg: RootConfig, task: TaskRecord) -> AttemptRecord | None:
    number = task.attempt_control.get("current_attempt_number")
    if not isinstance(number, int):
        return None
    try:
        return AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, number)))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def run_dispatch_cycle(
    cfg: RootConfig,
    *,
    available_gpus: list[int] | None = None,
    available_cpus: int | None = None,
    executor: Executor | None = None,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
    ready_cursor_namespace: str | None = None,
    admission_role: str | None = None,
    borrow_admission_grant: _BorrowAdmissionGrant | None = None,
    preflight: Callable[[TaskSpec], bool] | None = None,
    preflight_rejected: Callable[[TaskRecord], None] | None = None,
    before_claim: Callable[[], bool] | None = None,
    claim_guard: Callable[[], ContextManager[bool]] | None = None,
    max_new_claims: int | None = None,
    on_claim: Callable[[str], None] | None = None,
    should_recover_starting: bool = True,
    work_budget: SliceBudget | None = None,
    batch_sizer: AdaptiveBatchSizer | None = None,
    inspected_ready: set[tuple[str, str, str]] | None = None,
    lane: str | None = None,
    gpu_policy: GpuReservationPolicy | None = None,
) -> list[str]:
    with diagnostic_span("run_dispatch_cycle"):
        return _run_dispatch_cycle(
            cfg,
            available_gpus=available_gpus,
            available_cpus=available_cpus,
            executor=executor,
            reservation_runtime_root=reservation_runtime_root,
            project_id=project_id,
            ready_cursor_namespace=ready_cursor_namespace,
            admission_role=admission_role,
            borrow_admission_grant=borrow_admission_grant,
            preflight=preflight,
            preflight_rejected=preflight_rejected,
            before_claim=before_claim,
            claim_guard=claim_guard,
            max_new_claims=max_new_claims,
            on_claim=on_claim,
            should_recover_starting=should_recover_starting,
            work_budget=work_budget,
            batch_sizer=batch_sizer,
            inspected_ready=inspected_ready,
            lane=lane,
            gpu_policy=gpu_policy,
        )


def _run_dispatch_cycle(
    cfg: RootConfig,
    *,
    available_gpus: list[int] | None = None,
    available_cpus: int | None = None,
    executor: Executor | None = None,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
    ready_cursor_namespace: str | None = None,
    admission_role: str | None = None,
    borrow_admission_grant: _BorrowAdmissionGrant | None = None,
    preflight: Callable[[TaskSpec], bool] | None = None,
    preflight_rejected: Callable[[TaskRecord], None] | None = None,
    before_claim: Callable[[], bool] | None = None,
    claim_guard: Callable[[], ContextManager[bool]] | None = None,
    max_new_claims: int | None = None,
    on_claim: Callable[[str], None] | None = None,
    should_recover_starting: bool = True,
    work_budget: SliceBudget | None = None,
    batch_sizer: AdaptiveBatchSizer | None = None,
    inspected_ready: set[tuple[str, str, str]] | None = None,
    lane: str | None = None,
    gpu_policy: GpuReservationPolicy | None = None,
) -> list[str]:
    if lane not in {None, "cpu", "gpu"}:
        raise ValueError("lane must be cpu, gpu, or None.")
    if borrow_admission_grant is not None and lane is not None and borrow_admission_grant.lane != lane:
        raise BorrowAdmissionRequired("borrow admission grant belongs to a different resource lane.")
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    available = list(available_gpus or [])
    available_cpu_slots = available_cpus or 0
    executor = executor or Executor()
    launched: list[str] = []
    new_claims = 0
    if admission_role == "borrow" and borrow_admission_grant is None:
        raise BorrowAdmissionRequired("borrow dispatch requires a current machine-agent admission grant.")
    if should_recover_starting:
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = load_task(cfg, path.stem)
            attempt = resume_starting_attempt(cfg, task.task_id, reservation_runtime_root=reservation_runtime_root)
            if attempt is None:
                continue
            try:
                executor.launch_attempt(cfg, task.task_id, attempt)
                launched.append(task.task_id)
            except Exception as exc:
                reason = launch_failure_reason(exc)
                append_launch_failure_diagnostic(cfg, task.task_id, attempt.attempt_id, exc)
                if fail_attempt(
                    cfg,
                    task.task_id,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                    reason,
                    should_require_unstarted=True,
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    handle = launch_failure_handle(exc)
                    if handle is not None:
                        try:
                            executor.cleanup_launch(handle)
                        except Exception as cleanup_error:
                            append_launch_failure_diagnostic(
                                cfg,
                                task.task_id,
                                attempt.attempt_id,
                                cleanup_error,
                            )
    ready_state = read_ready_index_state(cfg)
    if (available or available_cpu_slots) and project_id is None and ready_state in {"absent", "building"}:
        advance_ready_index_build(cfg)
        ready_state = read_ready_index_state(cfg)
    if ready_state == "degraded":
        diagnostic_increment("scheduler.ready.skipped_degraded")
        return launched
    if ready_state == "active":
        budget = work_budget or SliceBudget()
        sizer = batch_sizer or AdaptiveBatchSizer(budget.policy)
        inspected = inspected_ready if inspected_ready is not None else set()
        cursor_project_id = ready_cursor_namespace or project_id or "standalone"
        scopes = ("home", "shared")
        scope_identities: dict[str, set[str]] = {scope: set() for scope in scopes}
        batch_limit = sizer.batch_size

        def iter_candidates() -> Iterator[tuple[ReadyMarkerRef, str]]:
            # Advance the durable cursor only when the caller is ready to process a candidate.
            candidate_count = 0
            while candidate_count < batch_limit and budget.can_start_record(operations=3):
                made_progress = False
                for scope in scopes:
                    if candidate_count >= batch_limit or not budget.can_start_record(operations=3):
                        break
                    reference, has_wrapped = next_ready_marker(
                        cfg,
                        cursor_project_id,
                        scope,
                        scope_identities[scope],
                    )
                    if has_wrapped:
                        diagnostic_increment("scheduler.ready.cursor_wraps")
                    if reference is None:
                        continue
                    identity = (cursor_project_id, scope, reference.identity)
                    if identity in inspected:
                        made_progress = True
                        continue
                    scope_identities[scope].add(reference.identity)
                    budget.consume_record(operations=3)
                    candidate_count += 1
                    made_progress = True
                    yield reference, scope
                if not made_progress:
                    break

        for reference, _scope in iter_candidates():
            inspected.add((cursor_project_id, _scope, reference.identity))
            started_ns = time.monotonic_ns()
            diagnostic_increment("scheduler.ready.markers_inspected")
            result = classify_ready_marker(cfg, reference)
            if result.classification == "corrupt":
                diagnostic_increment("scheduler.ready.corrupt")
                mark_ready_index_degraded(
                    cfg,
                    result.diagnostic or classification_diagnostic(result.reason, reference, task=result.task),
                )
                break
            if result.classification == "permanently_stale":
                diagnostic_increment("scheduler.ready.stale")
                delete_stale_ready_marker(cfg, reference)
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            task = result.task
            if result.classification == "temporarily_unavailable" or task is None:
                diagnostic_increment("scheduler.ready.temporarily_unavailable")
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if not _eligible(cfg, task):
                diagnostic_increment("scheduler.ready.machine_ineligible")
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if lane == "cpu" and not task.spec.is_cpu_only:
                continue
            if lane == "gpu" and task.spec.is_cpu_only:
                continue
            if not _admission_role_matches(cfg, task, admission_role):
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if borrow_admission_grant is None and task.group_name and _task_worker_role(cfg, task) == "borrow":
                diagnostic_increment("scheduler.borrow.skipped_without_admission")
                raise BorrowAdmissionRequired("borrow dispatch requires a current machine-agent admission grant.")
            if preflight is not None and not preflight(task.spec):
                if preflight_rejected is not None:
                    preflight_rejected(task)
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if task.spec.is_cpu_only:
                if available_cpu_slots < (task.spec.requested_cpus or 0):
                    diagnostic_increment("scheduler.ready.insufficient_cpu_capacity")
                    sizer.observe(max(1, time.monotonic_ns() - started_ns))
                    continue
            elif len(available) < task.spec.requested_gpus:
                diagnostic_increment("scheduler.ready.insufficient_capacity")
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if max_new_claims is not None and new_claims >= max_new_claims:
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if task.spec.is_cpu_only:
                gpus = []
                available_cpu_slots -= task.spec.requested_cpus or 0
            else:
                gpus, available = (
                    available[: task.spec.requested_gpus],
                    available[task.spec.requested_gpus :],
                )
            try:
                if claim_guard is not None:
                    with claim_guard() as is_permitted:
                        if not is_permitted:
                            if task.spec.is_cpu_only:
                                available_cpu_slots += task.spec.requested_cpus or 0
                            else:
                                available = gpus + available
                            break
                        with diagnostic_span("scheduler.claim"):
                            attempt = claim_task(
                                cfg,
                                task.task_id,
                                gpus,
                                reservation_runtime_root=reservation_runtime_root,
                                project_id=project_id,
                                admission_role=admission_role,
                                borrow_admission_grant=borrow_admission_grant,
                                gpu_policy=gpu_policy,
                            )
                else:
                    if before_claim is not None and not before_claim():
                        if task.spec.is_cpu_only:
                            available_cpu_slots += task.spec.requested_cpus or 0
                        else:
                            available = gpus + available
                        break
                    with diagnostic_span("scheduler.claim"):
                        attempt = claim_task(
                            cfg,
                            task.task_id,
                            gpus,
                            reservation_runtime_root=reservation_runtime_root,
                            project_id=project_id,
                            admission_role=admission_role,
                            borrow_admission_grant=borrow_admission_grant,
                            gpu_policy=gpu_policy,
                        )
            except ValueError:
                if task.spec.is_cpu_only:
                    available_cpu_slots += task.spec.requested_cpus or 0
                else:
                    available = gpus + available
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if attempt is None:
                if task.spec.is_cpu_only:
                    available_cpu_slots += task.spec.requested_cpus or 0
                else:
                    available = gpus + available
                diagnostic_increment("scheduler.ready.claim_races")
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            new_claims += 1
            if on_claim is not None:
                on_claim(task.task_id)
            try:
                with diagnostic_span("scheduler.authorize"):
                    is_authorized = authorize_launch(
                        cfg,
                        task.task_id,
                        attempt.attempt_id,
                        attempt.current_fencing_token,
                        reservation_runtime_root=reservation_runtime_root,
                        write_guard=claim_guard,
                    )
                if is_authorized:
                    authorized = _load_current_attempt(cfg, load_task(cfg, task.task_id))
                    if authorized is not None:
                        executor.launch_attempt(cfg, task.task_id, authorized)
                        launched.append(task.task_id)
                    elif task.spec.is_cpu_only:
                        available_cpu_slots += task.spec.requested_cpus or 0
                    else:
                        available = gpus + available
                else:
                    if task.spec.is_cpu_only:
                        available_cpu_slots += task.spec.requested_cpus or 0
                    else:
                        available = gpus + available
            except Exception as exc:
                reason = launch_failure_reason(exc)
                append_launch_failure_diagnostic(cfg, task.task_id, attempt.attempt_id, exc)
                if fail_attempt(
                    cfg,
                    task.task_id,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                    reason,
                    should_require_unstarted=True,
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    handle = launch_failure_handle(exc)
                    if handle is not None:
                        try:
                            executor.cleanup_launch(handle)
                        except Exception as cleanup_error:
                            append_launch_failure_diagnostic(
                                cfg,
                                task.task_id,
                                attempt.attempt_id,
                                cleanup_error,
                            )
            sizer.observe(max(1, time.monotonic_ns() - started_ns))
            if max_new_claims is not None and new_claims >= max_new_claims:
                break
        return launched

    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        if max_new_claims is not None and new_claims >= max_new_claims:
            break
        task = load_task(cfg, path.stem)
        if not _eligible(cfg, task):
            continue
        if lane == "cpu" and not task.spec.is_cpu_only:
            continue
        if lane == "gpu" and task.spec.is_cpu_only:
            continue
        if not _admission_role_matches(cfg, task, admission_role):
            continue
        if borrow_admission_grant is None and task.group_name and _task_worker_role(cfg, task) == "borrow":
            diagnostic_increment("scheduler.borrow.skipped_without_admission")
            raise BorrowAdmissionRequired("borrow dispatch requires a current machine-agent admission grant.")
        if preflight is not None and not preflight(task.spec):
            if preflight_rejected is not None:
                preflight_rejected(task)
            continue
        if task.spec.is_cpu_only:
            if available_cpu_slots < (task.spec.requested_cpus or 0):
                continue
        elif len(available) < task.spec.requested_gpus:
            continue
        if task.spec.is_cpu_only:
            gpus = []
            available_cpu_slots -= task.spec.requested_cpus or 0
        else:
            gpus, available = available[: task.spec.requested_gpus], available[task.spec.requested_gpus :]
        try:
            if claim_guard is not None:
                with claim_guard() as is_permitted:
                    if not is_permitted:
                        if task.spec.is_cpu_only:
                            available_cpu_slots += task.spec.requested_cpus or 0
                        else:
                            available = gpus + available
                        break
                    attempt = claim_task(
                        cfg,
                        task.task_id,
                        gpus,
                        reservation_runtime_root=reservation_runtime_root,
                        project_id=project_id,
                        admission_role=admission_role,
                        borrow_admission_grant=borrow_admission_grant,
                        gpu_policy=gpu_policy,
                    )
            else:
                if before_claim is not None and not before_claim():
                    if task.spec.is_cpu_only:
                        available_cpu_slots += task.spec.requested_cpus or 0
                    else:
                        available = gpus + available
                    break
                attempt = claim_task(
                    cfg,
                    task.task_id,
                    gpus,
                    reservation_runtime_root=reservation_runtime_root,
                    project_id=project_id,
                    admission_role=admission_role,
                    borrow_admission_grant=borrow_admission_grant,
                    gpu_policy=gpu_policy,
                )
        except ValueError:
            if task.spec.is_cpu_only:
                available_cpu_slots += task.spec.requested_cpus or 0
            else:
                available = gpus + available
            continue
        if attempt is None:
            if task.spec.is_cpu_only:
                available_cpu_slots += task.spec.requested_cpus or 0
            else:
                available = gpus + available
            continue
        new_claims += 1
        if on_claim is not None:
            on_claim(task.task_id)
        try:
            if not authorize_launch(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                reservation_runtime_root=reservation_runtime_root,
                write_guard=claim_guard,
            ):
                if task.spec.is_cpu_only:
                    available_cpu_slots += task.spec.requested_cpus or 0
                else:
                    available = gpus + available
                continue
            authorized = _load_current_attempt(cfg, load_task(cfg, task.task_id))
            if authorized is None:
                if task.spec.is_cpu_only:
                    available_cpu_slots += task.spec.requested_cpus or 0
                else:
                    available = gpus + available
                continue
            executor.launch_attempt(cfg, task.task_id, authorized)
            launched.append(task.task_id)
        except Exception as exc:
            reason = launch_failure_reason(exc)
            append_launch_failure_diagnostic(cfg, task.task_id, attempt.attempt_id, exc)
            if fail_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                reason,
                should_require_unstarted=True,
                reservation_runtime_root=reservation_runtime_root,
            ):
                handle = launch_failure_handle(exc)
                if handle is not None:
                    try:
                        executor.cleanup_launch(handle)
                    except Exception as cleanup_error:
                        append_launch_failure_diagnostic(
                            cfg,
                            task.task_id,
                            attempt.attempt_id,
                            cleanup_error,
                        )
    return launched


def has_eligible_local_work(cfg: RootConfig) -> bool:
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        if _eligible(cfg, load_task(cfg, path.stem)):
            return True
    return False


def renew_attempt_lease(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    lease_seconds: int | None = None,
    *,
    process_identity: Mapping[str, Any] | None = None,
) -> LeaseRenewalResult:
    """Renew an Attempt lease without collapsing errors into fencing decisions."""
    try:
        policy = load_lease_policy(cfg)
        task = load_task(cfg, task_id)
        with authority_locks(cfg, task):
            task = load_task(cfg, task_id)
            claim = task.claim_control.get("active_claim") or {}
            if claim.get("machine_name") != cfg.machine_name:
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id, observed_token=claim.get("fencing_token")
                )
            if claim.get("attempt_id") != attempt_id:
                if task.state.get("projection") == "blocked":
                    return LeaseRenewalResult(LeaseRenewalOutcome.ORPHANED_RECOVERY_REQUIRED, attempt_id)
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id, observed_token=claim.get("fencing_token")
                )
            if claim.get("fencing_token") != fencing_token:
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id, observed_token=claim.get("fencing_token")
                )
            path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
            attempt = AttemptRecord.from_dict(read_json(path))
            if attempt.current_fencing_token != fencing_token:
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id, observed_token=attempt.current_fencing_token
                )
            if claim.get("authority_mode") != attempt.authority_mode:
                return LeaseRenewalResult(LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id)
            if task.control.get("terminate_running"):
                if process_identity is not None and not process_identity_matches(attempt, process_identity):
                    return LeaseRenewalResult(LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id)
                return LeaseRenewalResult(LeaseRenewalOutcome.TERMINATION_REQUESTED, attempt_id)
            if claim.get("termination_decision_id"):
                # A historical stop marker is not a new cancellation request.
                # Its exact local decision must be reconciled separately.
                return LeaseRenewalResult(LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id)
            if (
                process_identity is not None
                and claim.get("authority_mode") == "bounded_lease"
                and attempt.authority_mode == "bounded_lease"
                and claim.get("launch_state") == "running"
                and attempt.phase == "running"
                and claim.get("launch_id") == attempt.authorization.get("launch_id")
                and isinstance(claim.get("launch_id"), str)
                and bool(claim.get("launch_id"))
                and _valid_utc_timestamp(claim.get("launch_authorized_at"))
                and process_identity_matches(attempt, process_identity)
                and not task.control.get("cancellation_requested_at")
                and not task.control.get("terminate_running")
                and not any(
                    claim.get(field) is not None
                    for field in (
                        "termination_decision_id",
                        "termination_decision_token",
                        "termination_committed_at",
                        "termination_commitment_digest",
                    )
                )
                and not any(
                    attempt.termination.get(field) is not None
                    for field in ("requested_at", "requested_by_operation_id", "decision_id", "decision_token")
                )
            ):
                try:
                    transition_attempt_ownership(
                        cfg,
                        task,
                        attempt,
                        target_phase="running",
                        launch_id=claim["launch_id"],
                        authorized_at=claim["launch_authorized_at"],
                        task_writer=lambda value: save_task(cfg, value),
                        attempt_writer=lambda value: atomic_replace(
                            attempt_path(cfg.shared_root, value.task_id, value.attempt_number), value.to_dict()
                        ),
                    )
                except OwnershipTransitionConflict:
                    return LeaseRenewalResult(LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id)
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.NOT_REQUIRED,
                    attempt_id,
                    observed_token=fencing_token,
                )
            if attempt.authority_mode == "holder_bound":
                return LeaseRenewalResult(LeaseRenewalOutcome.NOT_REQUIRED, attempt_id, observed_token=fencing_token)
            if process_identity is not None and not process_identity_matches(attempt, process_identity):
                return LeaseRenewalResult(LeaseRenewalOutcome.AUTHORITY_CHANGED, attempt_id)
            capability = clock_capability(cfg, policy)
            if not capability.is_healthy or capability.observation is None:
                return LeaseRenewalResult(
                    LeaseRenewalOutcome.RETRYABLE_ERROR,
                    attempt_id,
                    error=LeaseFailureDetails("ClockHealthError", capability.reason),
                )
            persist_clock_observation(cfg, capability.observation)
            evidence = _clock_evidence(capability.observation)
            expires = (
                lease_expiry(policy)
                if lease_seconds is None
                else (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            claim["lease_expires_at"] = expires
            claim.update(
                {
                    "clock_error_bound_seconds": evidence["clock_error_bound_seconds"],
                    "clock_provider": evidence["provider"],
                    "clock_observation_id": evidence["observation_id"],
                }
            )
            attempt.lease.update({"renewed_at": utc_now(), "expires_at": expires, "clock_evidence": evidence})
            atomic_replace(path, attempt.to_dict())
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            return LeaseRenewalResult(
                LeaseRenewalOutcome.RENEWED, attempt_id, observed_token=fencing_token, lease_expires_at=expires
            )
    except (OSError, ValueError, RuntimeError) as exc:
        return LeaseRenewalResult(
            LeaseRenewalOutcome.RETRYABLE_ERROR,
            attempt_id,
            error=LeaseFailureDetails(type(exc).__name__, str(exc), getattr(exc, "errno", None)),
        )


def renew_project_io_attempt_lease(
    cfg: RootConfig,
    *,
    request_id: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    expected_task_revision: int,
    expected_attempt_digest: str,
    mutation_fence: Callable[[], None],
    replay_only: bool = False,
) -> dict[str, Any]:
    """Renew one exact observed Attempt with replayable per-write fencing.

    ``request_id`` becomes the immutable clock-observation identity.  That
    marker lets an exact request resume after the clock-plan or Task write and
    recognize a fully committed Task/Attempt pair without adding shared-schema
    fields.  The Task write is deliberately metadata-only: renewal cannot
    change observation routes, so it bypasses ``save_task`` and its unrelated
    projection/outbox transaction.  ``replay_only`` is for stale executor
    requests and may only finish a Task commit already marked by this request.
    """

    def stale(reason: str, *, task_revision: int | None, attempt_digest: str | None) -> dict[str, Any]:
        return {
            "outcome": "observed_stale",
            "reason": reason,
            "lease_expires_at": None,
            "renew_after_seconds": None,
            "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
        }

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        path = attempt_path(cfg.shared_root, task_id, attempt_number)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def observation_id(record: Mapping[str, Any] | None) -> object:
        return None if not isinstance(record, Mapping) else record.get("observation_id")

    def validate_clock_observation(observation: ClockObservation) -> None:
        if observation.observation_id != request_id:
            raise ValueError("renewal clock observation identity is inconsistent.")
        if (
            not isinstance(observation.provider, str)
            or not observation.provider
            or len(observation.provider) > 128
            or not _valid_utc_timestamp(observation.observed_at)
            or not isinstance(observation.boot_id, str)
            or not observation.boot_id
            or len(observation.boot_id) > 128
        ):
            raise ValueError("renewal clock observation identity is malformed.")
        numeric = (
            observation.monotonic_observed_at,
            observation.lower_error_seconds,
            observation.upper_error_seconds,
            observation.max_drift_rate,
            observation.provider_margin_seconds,
        )
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric):
            raise ValueError("renewal clock observation contains a non-finite value.")
        if (
            observation.monotonic_observed_at < 0
            or observation.lower_error_seconds > observation.upper_error_seconds
            or observation.max_drift_rate < 0
            or observation.provider_margin_seconds < 0
        ):
            raise ValueError("renewal clock observation bounds are invalid.")

    def validate_plan(value: object) -> tuple[str, str, float]:
        if not isinstance(value, Mapping) or set(value) != {
            "renewed_at",
            "lease_expires_at",
            "renew_after_seconds",
        }:
            raise ValueError("renewal clock record has no immutable plan.")
        renewed_at = value["renewed_at"]
        expires = value["lease_expires_at"]
        if not _valid_utc_timestamp(renewed_at) or not _valid_utc_timestamp(expires):
            raise ValueError("renewal plan timestamps are malformed.")
        renewed_time = datetime.fromisoformat(renewed_at.replace("Z", "+00:00"))
        expiry_time = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expiry_time <= renewed_time:
            raise ValueError("renewal plan expiry must follow renewal time.")
        renew_after = value["renew_after_seconds"]
        if type(renew_after) not in {int, float} or not math.isfinite(renew_after) or not 0 < renew_after <= 86_400:
            raise ValueError("renewal plan cadence is invalid.")
        return renewed_at, expires, float(renew_after)

    initial = load_task(cfg, task_id)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        attempt, attempt_digest = read_attempt()
        task_revision = task.meta.get("revision")
        if type(task_revision) is not int or task_revision < 0:
            raise ValueError("Task revision is malformed.")
        claim = task.claim_control.get("active_claim") or {}
        if (
            task.attempt_control.get("current_attempt_id") != attempt_id
            or task.attempt_control.get("current_attempt_number") != attempt_number
            or claim.get("attempt_id") != attempt_id
            or claim.get("attempt_number") != attempt_number
            or claim.get("fencing_token") != fencing_token
            or claim.get("machine_name") != cfg.machine_name
        ):
            return stale("task_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if (
            attempt is None
            or attempt.task_id != task_id
            or attempt.attempt_id != attempt_id
            or attempt.attempt_number != attempt_number
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != cfg.machine_name
        ):
            return stale("attempt_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if claim.get("reservation_id") != reservation_id or attempt.reservation_id != reservation_id:
            return stale("reservation_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if any(
            attempt.process.get(field) != process_identity.get(field)
            for field in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        ):
            return stale("process_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if attempt.phase != "running":
            return stale("attempt_not_running", task_revision=task_revision, attempt_digest=attempt_digest)

        # An exact process observation can adopt a legacy running pair into
        # holder-bound ownership. Replay-only requests may repair only a
        # receipt whose Task commit already exists; they never create a new
        # conversion after a worker restart.
        if (
            claim.get("authority_mode") in {"bounded_lease", "holder_bound"}
            and attempt.authority_mode == "bounded_lease"
            and claim.get("launch_state") == "running"
            and claim.get("launch_id") == attempt.authorization.get("launch_id")
            and isinstance(claim.get("launch_id"), str)
            and bool(claim.get("launch_id"))
            and _valid_utc_timestamp(claim.get("launch_authorized_at"))
            and process_identity_matches(attempt, process_identity)
            and not task.control.get("terminate_running")
            and not task.control.get("cancellation_requested_at")
            and not any(
                claim.get(field) is not None
                for field in (
                    "termination_decision_id",
                    "termination_decision_token",
                    "termination_committed_at",
                    "termination_commitment_digest",
                )
            )
            and not any(
                attempt.termination.get(field) is not None
                for field in ("requested_at", "requested_by_operation_id", "decision_id", "decision_token")
            )
        ):

            def write_task(value: TaskRecord) -> None:
                atomic_replace(
                    task_path(cfg.shared_root, task_id),
                    value.to_dict(),
                    before_replace=lambda _stat: mutation_fence(),
                )

            try:
                transition_attempt_ownership(
                    cfg,
                    task,
                    attempt,
                    target_phase="running",
                    launch_id=claim["launch_id"],
                    authorized_at=claim["launch_authorized_at"],
                    mutation_fence=mutation_fence,
                    task_writer=write_task,
                    attempt_writer=lambda value: atomic_replace(
                        attempt_path(cfg.shared_root, value.task_id, value.attempt_number), value.to_dict()
                    ),
                    replay_only=replay_only,
                    expected_task_revision=expected_task_revision,
                    expected_attempt_digest=expected_attempt_digest,
                )
            except OwnershipTransitionConflict:
                return stale("attempt_changed", task_revision=task_revision, attempt_digest=attempt_digest)
            committed_task = load_task(cfg, task_id)
            committed_attempt_file = attempt_path(cfg.shared_root, task_id, attempt_number)
            committed_attempt_digest = hashlib.sha256(committed_attempt_file.read_bytes()).hexdigest()
            return {
                "outcome": "not_required",
                "reason": None,
                "lease_expires_at": None,
                "renew_after_seconds": None,
                "committed_revisions": {
                    "task": committed_task.meta["revision"],
                    "attempt_digest": committed_attempt_digest,
                },
            }

        attempt_clock = attempt.lease.get("clock_evidence")
        task_applied = claim.get("clock_observation_id") == request_id
        attempt_applied = observation_id(attempt_clock) == request_id
        if attempt_applied and not task_applied:
            raise RuntimeError("Attempt renewal marker exists without authoritative Task evidence.")
        if replay_only and not task_applied:
            return stale("task_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if not task_applied and (task_revision != expected_task_revision or attempt_digest != expected_attempt_digest):
            return stale("attempt_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if not task_applied and (
            task.control.get("terminate_running") is True
            or task.control.get("cancellation_requested_at") is not None
            or claim.get("termination_decision_id") is not None
            or attempt.termination.get("requested_at") is not None
            or attempt.termination.get("requested_by_operation_id") is not None
        ):
            return {
                "outcome": "termination_requested",
                "reason": "termination_pending",
                "lease_expires_at": None,
                "renew_after_seconds": None,
                "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
            }
        if claim.get("authority_mode") != attempt.authority_mode:
            return stale("attempt_changed", task_revision=task_revision, attempt_digest=attempt_digest)
        if attempt.authority_mode == "holder_bound":
            return {
                "outcome": "not_required",
                "reason": None,
                "lease_expires_at": None,
                "renew_after_seconds": None,
                "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
            }

        observation_path = shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{request_id}.json"
        try:
            clock_record = read_json(observation_path)
            if set(clock_record) != {"clock_observation", "renewal_plan"}:
                raise ValueError("renewal clock record fields are invalid.")
            stored_observation = ClockObservation.from_dict(clock_record["clock_observation"])
            validate_clock_observation(stored_observation)
            renewed_at, expires, renew_after_seconds = validate_plan(clock_record["renewal_plan"])
        except FileNotFoundError:
            if task_applied:
                raise RuntimeError("Authoritative Task renewal is missing its clock observation.")
            policy = load_lease_policy(cfg)
            capability = clock_capability(cfg, policy)
            if not capability.is_healthy or capability.observation is None:
                raise RuntimeError(f"clock capability is unavailable: {capability.reason}")
            source = capability.observation
            stored_observation = ClockObservation(
                observation_id=request_id,
                provider=source.provider,
                observed_at=source.observed_at,
                monotonic_observed_at=source.monotonic_observed_at,
                boot_id=source.boot_id,
                lower_error_seconds=source.lower_error_seconds,
                upper_error_seconds=source.upper_error_seconds,
                max_drift_rate=source.max_drift_rate,
                provider_margin_seconds=source.provider_margin_seconds,
            )
            validate_clock_observation(stored_observation)
            renewed_at = utc_now()
            expires = lease_expiry(policy)
            renew_after_seconds = policy.renew_interval_seconds
            validate_plan(
                {
                    "renewed_at": renewed_at,
                    "lease_expires_at": expires,
                    "renew_after_seconds": renew_after_seconds,
                }
            )
            mutation_fence()
            atomic_replace(
                observation_path,
                {
                    "clock_observation": stored_observation.to_dict(),
                    "renewal_plan": {
                        "renewed_at": renewed_at,
                        "lease_expires_at": expires,
                        "renew_after_seconds": renew_after_seconds,
                    },
                },
            )
        else:
            if not task_applied:
                policy = load_lease_policy(cfg)
                capability = clock_capability(cfg, policy)
                now_mono = time.monotonic()
                age = now_mono - stored_observation.monotonic_observed_at
                if (
                    not capability.is_healthy
                    or capability.observation is None
                    or capability.observation.boot_id != stored_observation.boot_id
                    or age < 0
                    or age > policy.clock_observation_max_age_seconds
                    or stored_observation.bound_at(now_mono) > policy.max_clock_skew_seconds
                ):
                    raise RuntimeError("renewal clock observation is no longer current.")

        if not task_applied:
            remaining_deadline = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            conservative_commit_time = datetime.now(timezone.utc) + timedelta(
                seconds=stored_observation.bound_at(time.monotonic()) + policy.renewal_commit_margin_seconds
            )
            if remaining_deadline <= conservative_commit_time:
                return stale(
                    "renewal_plan_expired",
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                )

        if task_applied:
            evidence = {
                "clock_error_bound_seconds": claim.get("clock_error_bound_seconds"),
                "provider": claim.get("clock_provider"),
                "observation_id": claim.get("clock_observation_id"),
            }
            if (
                claim.get("lease_expires_at") != expires
                or evidence["observation_id"] != request_id
                or evidence["provider"] != stored_observation.provider
                or type(evidence["clock_error_bound_seconds"]) not in {int, float}
                or not math.isfinite(evidence["clock_error_bound_seconds"])
                or evidence["clock_error_bound_seconds"] < 0
            ):
                raise RuntimeError("Authoritative Task renewal evidence is inconsistent.")
            if attempt_applied:
                if (
                    attempt.lease.get("expires_at") != expires
                    or attempt.lease.get("renewed_at") != renewed_at
                    or attempt_clock != evidence
                ):
                    raise RuntimeError("Committed renewal evidence is inconsistent.")
                return {
                    "outcome": "renewed",
                    "reason": None,
                    "lease_expires_at": expires,
                    "renew_after_seconds": renew_after_seconds,
                    "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
                }
        else:
            evidence = _clock_evidence(stored_observation)
            if (
                type(evidence.get("clock_error_bound_seconds")) not in {int, float}
                or not math.isfinite(evidence["clock_error_bound_seconds"])
                or evidence["clock_error_bound_seconds"] < 0
            ):
                raise ValueError("renewal clock evidence bound is invalid.")

            before = copy.deepcopy(task.to_dict())
            claim["lease_expires_at"] = expires
            claim.update(
                {
                    "clock_error_bound_seconds": evidence["clock_error_bound_seconds"],
                    "clock_provider": evidence["provider"],
                    "clock_observation_id": evidence["observation_id"],
                }
            )
            task.meta["revision"] = task_revision + 1
            task.meta["updated_at"] = renewed_at
            after = copy.deepcopy(task.to_dict())
            for value in (before, after):
                value["meta"].pop("revision", None)
                value["meta"].pop("updated_at", None)
                active_claim = value["task"]["claim_control"].get("active_claim") or {}
                for field in (
                    "lease_expires_at",
                    "clock_error_bound_seconds",
                    "clock_provider",
                    "clock_observation_id",
                ):
                    active_claim.pop(field, None)
            if before != after:
                raise RuntimeError("renewal attempted an observation-visible Task mutation.")
            mutation_fence()
            atomic_replace(task_path(cfg.shared_root, task_id), task.to_dict())
            task_revision += 1

        attempt.lease.update({"renewed_at": renewed_at, "expires_at": expires, "clock_evidence": evidence})
        mutation_fence()
        attempt_file = attempt_path(cfg.shared_root, task_id, attempt_number)
        atomic_replace(attempt_file, attempt.to_dict())
        attempt_digest = hashlib.sha256(attempt_file.read_bytes()).hexdigest()
        return {
            "outcome": "renewed",
            "reason": None,
            "lease_expires_at": expires,
            "renew_after_seconds": renew_after_seconds,
            "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
        }


def resolve_execution_authority(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    decision_id: str,
    *,
    reservation_runtime_root: Path | None = None,
    defer_recovery: bool = False,
) -> AuthorityResolution:
    """Perform the final authoritative renewal/recovery decision for a live process."""
    try:
        result = renew_attempt_lease(cfg, task_id, attempt_id, fencing_token)
    except Exception as exc:  # defensive: resolver is the sole fail-closed authority boundary
        return AuthorityResolution(
            AuthorityResolutionOutcome.AUTHORITY_UNAVAILABLE,
            decision_id,
            attempt_id,
            fencing_token,
            reason=type(exc).__name__,
        )
    if result.outcome in {LeaseRenewalOutcome.RENEWED, LeaseRenewalOutcome.NOT_REQUIRED}:
        return AuthorityResolution(
            AuthorityResolutionOutcome.RENEWED,
            decision_id,
            attempt_id,
            fencing_token,
            fencing_token,
            result.lease_expires_at,
        )
    if result.outcome is LeaseRenewalOutcome.ORPHANED_RECOVERY_REQUIRED:
        if defer_recovery:
            return AuthorityResolution(
                AuthorityResolutionOutcome.AUTHORITY_UNAVAILABLE,
                decision_id,
                attempt_id,
                fencing_token,
                reason="recovery_deferred",
            )
        from .runtime.attempt_recovery import recover_running_attempt

        token = recover_running_attempt(
            cfg,
            task_id,
            attempt_id,
            fencing_token,
            reservation_runtime_root=reservation_runtime_root,
        )
        if token is not None:
            return AuthorityResolution(
                AuthorityResolutionOutcome.RECOVERED, decision_id, attempt_id, fencing_token, token
            )
        return AuthorityResolution(
            AuthorityResolutionOutcome.TERMINATION_REQUIRED,
            decision_id,
            attempt_id,
            fencing_token,
            reason="lease_recovery_rejected",
        )
    if result.outcome is LeaseRenewalOutcome.RETRYABLE_ERROR:
        return AuthorityResolution(
            AuthorityResolutionOutcome.AUTHORITY_UNAVAILABLE,
            decision_id,
            attempt_id,
            fencing_token,
            reason=result.error.error_type if result.error else None,
        )
    return AuthorityResolution(
        AuthorityResolutionOutcome.TERMINATION_REQUIRED,
        decision_id,
        attempt_id,
        fencing_token,
        reason=result.outcome.value,
    )


def commit_shared_termination(
    cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, decision_id: str
) -> bool:
    """Fence Recovery before an externally visible termination signal is issued."""
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        existing = claim.get("termination_decision_id")
        if existing not in {None, decision_id}:
            return False
        claim.update(
            {
                "termination_decision_id": decision_id,
                "termination_decision_token": fencing_token,
                "termination_committed_at": utc_now(),
            }
        )
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True


def commit_project_io_shared_termination(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    decision_id: str,
    decision_token: int,
    authority_outcome: str,
    reason: str,
    process_identity: Mapping[str, Any],
    expected_task_revision: int,
    expected_attempt_digest: str,
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Commit one exact Project-I/O termination marker under shared authority locks."""

    def evidence(
        outcome: str,
        stale_reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
        shared_commitment: str | None,
    ) -> dict[str, Any]:
        value = {
            "outcome": outcome,
            "reason": stale_reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": reservation_id,
            "decision_id": decision_id,
            "decision_token": decision_token,
            "authority_outcome": authority_outcome,
            "decision_reason": reason,
            "process_identity": dict(process_identity),
            "source_revisions": {"task": expected_task_revision, "attempt_digest": expected_attempt_digest},
            "committed_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "shared_commitment": shared_commitment,
            "authority_granted": False,
            "local_effects": [],
        }
        return value

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        path = attempt_path(cfg.shared_root, task_id, attempt_number)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    commitment_identity = {
        "machine_name": machine_name,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "reservation_id": reservation_id,
        "decision_id": decision_id,
        "decision_token": decision_token,
        "authority_outcome": authority_outcome,
        "reason": reason,
        "process_identity": dict(process_identity),
        "source_revisions": {"task": expected_task_revision, "attempt_digest": expected_attempt_digest},
    }
    commitment_digest = hashlib.sha256(
        json.dumps(
            commitment_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest, None)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        if task.depends_on_task_ids != initial.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while Project-I/O locks were acquired.")
        attempt, attempt_digest = read_attempt()
        if attempt is None:
            return evidence("stale", "attempt_missing", task, None, None, None)
        claim = task.claim_control.get("active_claim") or {}
        attempt_identity_matches = (
            task.task_id == task_id
            and attempt.task_id == task_id
            and attempt.attempt_id == attempt_id
            and attempt.attempt_number == attempt_number
            and attempt.current_fencing_token == fencing_token
            and attempt.machine_name == machine_name
        )
        if not attempt_identity_matches:
            return evidence("stale", "claim_not_current", task, attempt, attempt_digest, None)
        attempt_process_matches = all(
            attempt.process.get(field) == process_identity.get(field) for field in process_identity
        )
        claim_identity_matches = (
            task.attempt_control.get("current_attempt_id") == attempt_id
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and claim.get("attempt_id") == attempt_id
            and claim.get("attempt_number") == attempt_number
            and claim.get("fencing_token") == fencing_token
            and claim.get("machine_name") == machine_name
        )
        if not claim_identity_matches and not claim:
            terminal_pair = (
                task.attempt_control.get("current_attempt_id") is None
                and task.attempt_control.get("current_attempt_number") == attempt_number
                and task.state.get("projection") in {"succeeded", "failed", "cancelled"}
                and task.state.get("projection") == attempt.phase
                and task.state.get("reason") == attempt.result.get("reason")
            )
            if terminal_pair and attempt_process_matches and attempt.reservation_id == reservation_id:
                claim_paths = shared_paths(cfg.shared_root)
                for archive_dir in ("claim_archive", "claim_pending"):
                    path = claim_paths[archive_dir] / task_id / f"{fencing_token}.json"
                    try:
                        archived = read_json(path).get("claim_archive", {}).get("claim", {})
                    except FileNotFoundError:
                        continue
                    if (
                        archived.get("attempt_id") == attempt_id
                        and archived.get("attempt_number") == attempt_number
                        and archived.get("fencing_token") == fencing_token
                    ):
                        existing_decision_id = archived.get("termination_decision_id")
                        existing_decision_token = archived.get("termination_decision_token")
                        if (
                            existing_decision_id == decision_id
                            and existing_decision_token == decision_token
                            and archived.get("termination_commitment_digest") == commitment_digest
                        ):
                            return evidence("already_committed", None, task, attempt, attempt_digest, "committed")
                        if existing_decision_id is not None or existing_decision_token is not None:
                            return evidence("stale", "decision_conflict", task, attempt, attempt_digest, None)
            return evidence("stale", "claim_not_current", task, attempt, attempt_digest, None)
        if not claim_identity_matches:
            return evidence("stale", "claim_not_current", task, attempt, attempt_digest, None)
        existing_decision_id = claim.get("termination_decision_id")
        existing_decision_token = claim.get("termination_decision_token")
        if existing_decision_id is not None or existing_decision_token is not None:
            if (
                existing_decision_id == decision_id
                and existing_decision_token == decision_token
                and claim.get("termination_commitment_digest") == commitment_digest
            ):
                if (
                    claim.get("reservation_id") != reservation_id
                    or attempt.reservation_id != reservation_id
                    or not attempt_process_matches
                ):
                    return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest, None)
                return evidence("already_committed", None, task, attempt, attempt_digest, "committed")
            return evidence("stale", "decision_conflict", task, attempt, attempt_digest, None)
        if task.meta.get("revision") != expected_task_revision or attempt_digest != expected_attempt_digest:
            return evidence("stale", "source_revision_mismatch", task, attempt, attempt_digest, None)
        if task.state.get("projection") != "running":
            return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest, None)
        if claim.get("reservation_id") != reservation_id or attempt.reservation_id != reservation_id:
            return evidence("stale", "reservation_mismatch", task, attempt, attempt_digest, None)
        if attempt.phase != "running":
            return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest, None)
        if any(attempt.process.get(field) != process_identity.get(field) for field in process_identity):
            return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest, None)
        claim.update(
            {
                "termination_decision_id": decision_id,
                "termination_decision_token": decision_token,
                "termination_committed_at": utc_now(),
                "termination_commitment_digest": commitment_digest,
            }
        )
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        mutation_fence()
        save_task(cfg, task, mutation_fence=mutation_fence)
        return evidence("committed", None, task, attempt, attempt_digest, "committed")


def retire_project_io_timeout_decision(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    decision_id: str,
    decision_token: int,
    authority_outcome: str,
    reason: str,
    process_identity: Mapping[str, Any],
    source_revisions: Mapping[str, Any],
    retirement: Mapping[str, Any],
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Conditionally replace one historical timeout stop reference with a receipt."""

    def evidence(
        outcome: str,
        stale_reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
    ) -> dict[str, Any]:
        value = {
            "outcome": outcome,
            "reason": stale_reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": reservation_id,
            "decision_id": decision_id,
            "decision_token": decision_token,
            "authority_outcome": authority_outcome,
            "decision_reason": reason,
            "process_identity": dict(process_identity),
            "source_revisions": dict(source_revisions),
            "committed_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "shared_commitment": "committed" if outcome in {"committed", "already_committed"} else None,
            "authority_granted": False,
            "local_effects": [],
        }
        value["retirement_receipt"] = dict(retirement)
        return value

    required = {
        "receipt_version",
        "operation",
        "operation_id",
        "decision_id",
        "decision_digest",
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "machine_name",
        "process_identity",
        "source_state",
        "source_shared_commitment",
        "source_authority_outcome",
        "source_reason",
    }
    if not isinstance(retirement, Mapping) or not required.issubset(retirement):
        raise ValueError("timeout retirement receipt is incomplete")
    if (
        retirement.get("operation") != "timeout_decision_retirement"
        or retirement.get("decision_id") != decision_id
        or retirement.get("task_id") != task_id
        or retirement.get("attempt_id") != attempt_id
        or retirement.get("attempt_number") != attempt_number
        or retirement.get("fencing_token") != fencing_token
        or retirement.get("machine_name") != machine_name
        or retirement.get("process_identity") != process_identity
        or retirement.get("source_authority_outcome") != "holder_safe_deadline_elapsed"
        or retirement.get("source_reason") != "holder_safe_deadline_elapsed"
    ):
        raise ValueError("timeout retirement receipt identity is inconsistent")

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        path = attempt_path(cfg.shared_root, task_id, attempt_number)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def exact_attempt(attempt: AttemptRecord | None) -> bool:
        return (
            attempt is not None
            and attempt.task_id == task_id
            and attempt.attempt_id == attempt_id
            and attempt.attempt_number == attempt_number
            and attempt.current_fencing_token == fencing_token
            and attempt.machine_name == machine_name
            and attempt.reservation_id == reservation_id
            and all(attempt.process.get(field) == process_identity.get(field) for field in process_identity)
        )

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        attempt, attempt_digest = read_attempt()
        if not exact_attempt(attempt):
            return evidence("stale", "attempt_identity_mismatch", task, attempt, attempt_digest)
        assert attempt is not None
        claim = task.claim_control.get("active_claim") or {}
        active_matches = (
            task.attempt_control.get("current_attempt_id") == attempt_id
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and claim.get("attempt_id") == attempt_id
            and claim.get("attempt_number") == attempt_number
            and claim.get("fencing_token") == fencing_token
            and claim.get("machine_name") == machine_name
        )
        existing = (
            claim.get("termination_decision_retirement")
            if active_matches
            else attempt.meta.get("timeout_decision_retirement")
        )
        if existing is not None:
            if existing.get("operation_id") != retirement.get("operation_id"):
                return evidence("stale", "decision_conflict", task, attempt, attempt_digest)
            return evidence("already_committed", None, task, attempt, attempt_digest)
        if task.meta.get("revision") != source_revisions.get("task"):
            return evidence("stale", "task_changed", task, attempt, attempt_digest)
        if attempt_digest != source_revisions.get("attempt_digest"):
            return evidence("stale", "attempt_changed", task, attempt, attempt_digest)
        active_stop = (
            claim.get("termination_decision_id") == decision_id
            and claim.get("termination_decision_token") == decision_token
        )
        if active_matches and not active_stop:
            # A pending local decision has no active shared stop reference; the
            # receipt itself is still the conditional audit commit.
            other_stop = claim.get("termination_decision_id") or claim.get("termination_decision_token")
            if other_stop is not None:
                return evidence("stale", "decision_conflict", task, attempt, attempt_digest)
        shared_receipt = {
            **dict(retirement),
            "state": "shared_reconciled",
            "shared_at": utc_now(),
            "original_shared_commitment": retirement.get("source_shared_commitment"),
        }
        mutation_fence()
        if active_matches:
            claim["termination_decision_retirement"] = shared_receipt
            if active_stop:
                for field in (
                    "termination_decision_id",
                    "termination_decision_token",
                    "termination_committed_at",
                    "termination_commitment_digest",
                ):
                    claim.pop(field, None)
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task, mutation_fence=mutation_fence)
            committed_task = task
        else:
            # A successor or an archived claim won the Task pointer.  Annotate
            # only the historical Attempt; never alter successor control truth.
            attempt.meta["timeout_decision_retirement"] = shared_receipt
            atomic_replace(
                attempt_path(cfg.shared_root, task_id, attempt_number),
                attempt.to_dict(),
                before_replace=lambda _stat: mutation_fence(),
            )
            committed_task = task
        refreshed_attempt, refreshed_digest = read_attempt()
        return evidence("committed", None, committed_task, refreshed_attempt, refreshed_digest)


def recover_project_io_orphaned_attempt(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    binding_signature: list[str] | tuple[str, ...],
    expected_task_revision: int | None,
    expected_attempt_digest: str | None,
    mutation_fence: Callable[[], None],
    replay_only: bool = False,
) -> dict[str, Any]:
    """Run the shared-only orphan recovery transaction for Project I/O."""
    from .runtime.attempt_recovery import recover_orphaned_attempt_shared

    return recover_orphaned_attempt_shared(
        cfg,
        machine_name=machine_name,
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        expired_token=fencing_token,
        reservation_id=reservation_id,
        process_identity=process_identity,
        binding_signature=binding_signature,
        expected_task_revision=expected_task_revision,
        expected_attempt_digest=expected_attempt_digest,
        mutation_fence=mutation_fence,
        replay_only=replay_only,
    )


def _continue_committed_termination(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int) -> bool:
    """Let an agent finish a runner's shared termination commitment after a crash."""
    from .runtime.termination import attempt_control_lock, commit_signal, decision_path, send_signals, update_decision

    with attempt_control_lock(cfg, attempt_id):
        task = load_task(cfg, task_id)
        with authority_locks(cfg, task):
            task = load_task(cfg, task_id)
            claim = task.claim_control.get("active_claim") or {}
            decision_id = claim.get("termination_decision_id")
            if (
                claim.get("attempt_id") != attempt_id
                or claim.get("fencing_token") != fencing_token
                or claim.get("termination_decision_token") != fencing_token
                or not isinstance(decision_id, str)
            ):
                return False
        path = decision_path(cfg, attempt_id, decision_id)
        if not path.exists():
            return False
        decision = read_json(path).get("termination_decision", {})
        if decision.get("decision_token") != fencing_token:
            return False
        if decision.get("state") == "pending":
            update_decision(cfg, attempt_id, decision_id, shared_commitment="committed")
            commit_signal(cfg, attempt_id, decision_id)
        if decision.get("state") not in {"confirmed", "superseded"}:
            send_signals(cfg, attempt_id, decision_id)
    return True


def expire_claim(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    *,
    reservation_runtime_root: Path | None = None,
) -> bool:
    """Converge an expired claim without creating replacement execution authority."""
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    policy = load_lease_policy(cfg)
    capability = clock_capability(cfg, policy)
    if not capability.is_healthy or capability.observation is None:
        return False
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        if claim.get("launch_state") != "claimed":
            return False
        if claim.get("authority_mode") != "bounded_lease":
            return False
        holder_bound = claim.get("clock_error_bound_seconds")
        expires = claim.get("lease_expires_at")
        if not isinstance(holder_bound, (int, float)) or not isinstance(expires, str):
            with record_task_change(
                cfg,
                task,
                "claim_loss",
                details={
                    "expected_attempt_id": attempt_id,
                    "expected_fencing_token": fencing_token,
                    "is_invalid_evidence": True,
                },
            ):
                task.state.update({"projection": "blocked", "reason": "authority_mode_evidence_invalid"})
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
                return False
        reclaimer_bound = capability.observation.bound_at(time.monotonic())
        if datetime.now(timezone.utc) < reclaim_allowed_at(expires, holder_bound, reclaimer_bound):
            return False
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        attempt = AttemptRecord.from_dict(read_json(path))
        orphan_work = None
        if claim.get("launch_state") != "claimed":
            from .runtime.maintenance_outbox import prepare_work

            orphan_work = prepare_work(
                cfg,
                kind="orphan_recovery",
                target_id=attempt_id,
                work_generation=f"fencing-{fencing_token}",
                phase="attempt_recovery",
                cursor={
                    "stage": "process",
                    "task_id": task_id,
                    "attempt_number": attempt.attempt_number,
                    "attempt_id": attempt_id,
                    "fencing_token": fencing_token,
                    "attempt_process_identity": {
                        key: attempt.process.get(key) for key in ("process_group_id", "process_group_start_time_ticks")
                    },
                    "was_terminated": bool(task.control.get("terminate_running")),
                },
            )
        with record_task_change(
            cfg,
            task,
            "claim_loss",
            details={
                "expected_attempt_id": attempt_id,
                "expected_fencing_token": fencing_token,
                "reserved_generation": task.ready_generation + 1 if claim.get("launch_state") == "claimed" else None,
            },
        ):
            old_ready_generation = None
            if claim.get("launch_state") == "claimed":
                task.state.update({"projection": "queued", "reason": "lease_expired_before_launch"})
                old_ready_generation, _ = prepare_ready_transition(cfg, task, "prelaunch_expiry")
                attempt.phase = "cancelled"
                attempt.result["reason"] = "lease_expired_before_launch"
                _release_task_reservation(
                    task, reservation_runtime_root, claim["reservation_id"], "lease_expired_before_launch"
                )
            else:
                attempt.phase = "orphaned"
                attempt.result.update({"exit_code": None, "signal": None, "category": None, "reason": None})
                decision_id = claim.get("termination_decision_id")
                if isinstance(decision_id, str):
                    attempt.termination.update(
                        {"decision_id": decision_id, "decision_token": claim.get("termination_decision_token")}
                    )
                attempt.timestamps["orphaned_at"] = utc_now()
                task.state.update({"projection": "blocked", "reason": "orphaned_attempt_requires_recovery"})
            if attempt.phase != "orphaned":
                attempt.timestamps["finished_at"] = utc_now()
            atomic_replace(path, attempt.to_dict())
            archive_claim(cfg, task_id, claim, "lease_expired")
            task.claim_control["active_claim"] = None
            task.attempt_control["current_attempt_id"] = None
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            if old_ready_generation is not None:
                commit_ready_publication(cfg, task)
            if old_ready_generation is not None:
                retire_previous_ready_generation(cfg, old_ready_generation, task)
            if attempt.phase == "orphaned":
                try:
                    write_event(
                        cfg,
                        "attempt_orphaned",
                        task_id=task_id,
                        details={
                            "attempt_id": attempt_id,
                            "fencing_token": fencing_token,
                            "reason": "lease_expired_process_unknown",
                        },
                    )
                except OSError:
                    pass
                if orphan_work is not None and orphan_work["state"] not in {
                    "completed",
                    "intervention",
                    "superseded",
                }:
                    try:
                        from .runtime.maintenance_outbox import activate_work

                        activate_work(cfg, orphan_work)
                    except (OSError, RuntimeError, ValueError):
                        # The descriptor is already durable and active; a
                        # later resident turn can recover this handoff.
                        pass
            return True


def fail_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    reason: str,
    *,
    reservation_runtime_root: Path | None = None,
    should_require_unstarted: bool = False,
) -> bool:
    """Commit failure; launch compensation must require absent local execution evidence.

    The guarded check shares the runner's launch-intent authority fence. Evidence
    or a foreign holder retains authority and capacity for owning-agent recovery.
    Inaccessible evidence raises rather than being treated as proof of absence.
    """
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        if should_require_unstarted and (
            claim.get("machine_name") != cfg.machine_name or _has_local_launch_evidence(cfg, attempt_id)
        ):
            diagnostic_increment("scheduler.launch_failure.deferred")
            return False
        result = commit_terminal_transition_locked(
            cfg,
            task,
            TerminalTransition(
                task_id,
                attempt_id,
                task.attempt_control["current_attempt_number"],
                fencing_token,
                "failed",
                reason,
                None,
                frozenset({"running"}),
                frozenset({"claimed", "starting", "running"}),
                "active",
            ),
        )
    if result.outcome != "committed":
        return False
    if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
        _release_task_reservation(task, reservation_runtime_root, result.reservation_id, reason)
    if result.event:
        dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
    return True


def finalize_agent_supervised_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    *,
    was_terminated: bool,
    reservation_runtime_root: Path | None = None,
) -> bool:
    """Publish terminal truth after the agent confirms a recovered process is absent."""
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            return False
        reason = "terminated_by_agent" if was_terminated else "process_exited_without_status"
        phase = "cancelled" if was_terminated else "failed"
        result = commit_terminal_transition_locked(
            cfg,
            task,
            TerminalTransition(
                task_id,
                attempt_id,
                number,
                fencing_token,
                phase,
                reason,
                None,
                frozenset({"running"}),
                frozenset({"claimed", "starting", "running"}),
                "active",
                "terminated" if was_terminated else None,
            ),
        )
    if result.outcome != "committed":
        return False
    if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
        _release_task_reservation(task, reservation_runtime_root, result.reservation_id, reason)
    if result.event:
        dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
    return True


def finalize_orphaned_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    *,
    exit_code: int | None,
    was_terminated: bool,
    reservation_runtime_root: Path | None = None,
) -> bool:
    """Resolve a blocked orphan after local evidence confirms its process is absent."""
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if task.state["projection"] != "blocked" or task.claim_control.get("active_claim"):
            return False
        number = task.attempt_control.get("current_attempt_number")
        if number is None or task.attempt_control["next_attempt_number"] != number + 1:
            return False
        path = attempt_path(cfg.shared_root, task_id, number)
        attempt = AttemptRecord.from_dict(read_json(path))
        if (
            attempt.attempt_id != attempt_id
            or attempt.phase not in {"orphaned", "running", "succeeded", "failed", "cancelled"}
            or attempt.current_fencing_token != fencing_token
        ):
            return False
        if attempt.phase in {"succeeded", "failed", "cancelled"}:
            phase = attempt.phase
            reason = attempt.result.get("reason") or "recovered_terminal_attempt"
            exit_code = attempt.result.get("exit_code")
            termination_result = attempt.termination.get("result")
        elif was_terminated:
            phase = "cancelled"
            reason = "termination_process_already_exited"
            termination_result = "already_exited"
        elif exit_code == 0:
            phase = "succeeded"
            reason = "completed"
            termination_result = None
        elif exit_code is not None:
            phase = "failed"
            reason = "nonzero_exit"
            termination_result = None
        else:
            phase = "failed"
            reason = "process_exited_without_status"
            termination_result = None
        result = commit_terminal_transition_locked(
            cfg,
            task,
            TerminalTransition(
                task_id,
                attempt_id,
                number,
                fencing_token,
                phase,
                reason,
                exit_code,
                frozenset({"blocked"}),
                frozenset({"orphaned", "running", "succeeded", "failed", "cancelled"}),
                "detached",
                termination_result,
            ),
        )
    if result.outcome != "committed":
        return False
    if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
        _release_task_reservation(task, reservation_runtime_root, result.reservation_id, reason)
    if result.event:
        dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
    return True


def _has_superseded_ownership(task: TaskRecord, attempt: AttemptRecord) -> bool:
    """Prove historical ownership from the committed operation, never age."""
    receipt = task.attempt_control.get("last_supersession")
    if not isinstance(receipt, dict) or receipt.get("attempt_id") != attempt.attempt_id:
        receipt = attempt.authorization.get("supersession_receipt")
    return (
        isinstance(receipt, dict)
        and receipt.get("task_id") == task.task_id == attempt.task_id
        and receipt.get("attempt_id") == attempt.attempt_id
        and receipt.get("attempt_number") == attempt.attempt_number
        and receipt.get("fencing_token") == attempt.current_fencing_token
        and receipt.get("old_machine") == attempt.machine_name
        and receipt.get("reservation_id") == attempt.reservation_id
        and receipt.get("terminate_old_process") is False
        and receipt.get("duplicate_risk_acknowledged") is True
        and task.attempt_control.get("current_attempt_id") != attempt.attempt_id
        and (task.claim_control.get("active_claim") or {}).get("attempt_id") != attempt.attempt_id
        and task.claim_control.get("fencing_epoch", 0) > attempt.current_fencing_token
    )


def observe_project_io_terminal_state(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Read one exact active or detached Attempt snapshot under authority locks."""

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        try:
            raw = attempt_path(cfg.shared_root, task_id, attempt_number).read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def evidence(
        outcome: str,
        reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reason": reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": None if attempt is None else attempt.reservation_id,
            "process_identity": dict(process_identity),
            "mode": mode,
            "task_phase": None if task is None else task.state.get("projection"),
            "attempt_phase": None if attempt is None else attempt.phase,
            "attempt_result_reason": None if attempt is None else attempt.result.get("reason"),
            "attempt_exit_code": None if attempt is None else attempt.result.get("exit_code"),
            "execution_machine_name": None if attempt is None else attempt.machine_name,
            "reservation_machine_name": None if attempt is None else attempt.machine_name,
            "termination_result": None if attempt is None else attempt.termination.get("result"),
            "cancel_requested": False if task is None else bool(task.control.get("terminate_running")),
            "source_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "authority_granted": False,
            "local_effects": [],
        }

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        if task.depends_on_task_ids != initial.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while Project-I/O locks were acquired.")
        attempt, attempt_digest = read_attempt()
        if attempt is None:
            return evidence("stale", "attempt_missing", task, None, None)
        if (
            attempt.task_id != task_id
            or attempt.attempt_id != attempt_id
            or attempt.attempt_number != attempt_number
            or attempt.current_fencing_token != fencing_token
        ):
            return evidence("stale", "attempt_identity_mismatch", task, attempt, attempt_digest)
        if attempt.machine_name != machine_name:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        if attempt.reservation_id != reservation_id:
            return evidence("stale", "reservation_mismatch", task, attempt, attempt_digest)
        if any(attempt.process.get(field) != process_identity.get(field) for field in process_identity):
            return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest)

        task_phase = task.state.get("projection")
        attempt_phase = attempt.phase
        if _has_superseded_ownership(task, attempt) and attempt_phase in {"starting", "running"}:
            observed = evidence("historical_current", None, task, attempt, attempt_digest)
            observed["cancel_requested"] = False
            return observed
        terminal_pair = (
            task_phase in {"succeeded", "failed", "cancelled"}
            and attempt_phase == task_phase
            and task.state.get("reason") == attempt.result.get("reason")
            and task.control.get("termination_result") == attempt.termination.get("result")
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("current_attempt_id") is None
            and not task.claim_control.get("active_claim")
        )
        if terminal_pair:
            return evidence("already_terminal", None, task, attempt, attempt_digest)

        if attempt_phase in {"succeeded", "failed", "cancelled"}:
            settled_attempt = load_settled_terminal_attempt(
                cfg,
                task_id,
                attempt_id,
                attempt_number=attempt_number,
            )
            if settled_attempt is not None:
                return evidence("settled_terminal", None, task, attempt, attempt_digest)

        if mode == "active":
            if (
                task.attempt_control.get("current_attempt_id") != attempt_id
                or task.attempt_control.get("current_attempt_number") != attempt_number
            ):
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            claim = task.claim_control.get("active_claim") or {}
            if (
                task_phase != "running"
                or claim.get("attempt_id") != attempt_id
                or claim.get("attempt_number") != attempt_number
                or claim.get("fencing_token") != fencing_token
                or claim.get("machine_name") != machine_name
                or claim.get("reservation_id") != reservation_id
            ):
                return evidence("stale", "claim_not_current", task, attempt, attempt_digest)
            if attempt_phase != "running":
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        elif mode == "detached_orphan":
            if task_phase != "blocked" or task.claim_control.get("active_claim"):
                return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)
            if (
                task.attempt_control.get("current_attempt_id") is not None
                or task.attempt_control.get("current_attempt_number") != attempt_number
            ):
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            if task.attempt_control.get("next_attempt_number") != attempt_number + 1:
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            if attempt_phase != "orphaned":
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        else:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        return evidence("current", None, task, attempt, attempt_digest)


def publish_project_io_terminal_transition(
    cfg: RootConfig,
    *,
    request_id: str,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    mode: str,
    phase: str,
    reason: str,
    exit_code: int | None,
    termination_result: str | None,
    expected_task_revision: int,
    expected_attempt_digest: str,
    transition_digest: str,
    mutation_fence: Callable[[], None],
) -> dict[str, Any]:
    """Publish one digest-bound terminal transition and its shared lifecycle evidence."""

    def read_attempt() -> tuple[AttemptRecord | None, str | None]:
        try:
            raw = attempt_path(cfg.shared_root, task_id, attempt_number).read_bytes()
        except FileNotFoundError:
            return None, None
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Attempt record must contain a JSON object.")
        return AttemptRecord.from_dict(value), hashlib.sha256(raw).hexdigest()

    def evidence(
        outcome: str,
        stale_reason: str | None,
        task: TaskRecord | None,
        attempt: AttemptRecord | None,
        attempt_digest: str | None,
        result: object | None = None,
    ) -> dict[str, Any]:
        event = getattr(result, "event", None)
        event_value = None if event is None else asdict(event)
        if event_value is not None and event_value["task_name"] is not None:
            # Bound the wire projection, including JSON escaping, while keeping
            # the full user name in shared truth and ordinary lifecycle events.
            name = event_value["task_name"]
            low, high = 0, min(len(name), 16_384)
            while low < high:
                middle = (low + high + 1) // 2
                if len(json.dumps(name[:middle], ensure_ascii=False).encode("utf-8")) <= 16_384:
                    low = middle
                else:
                    high = middle - 1
            event_value["task_name"] = name[:low]
        reservation = getattr(result, "reservation_id", None)
        reservation_machine = getattr(result, "reservation_machine_name", None)
        return {
            "outcome": outcome,
            "reason": stale_reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": reservation
            if result is not None
            else (None if attempt is None else attempt.reservation_id),
            "reservation_machine_name": reservation_machine
            if result is not None
            else (None if attempt is None else attempt.machine_name),
            "process_identity": dict(process_identity),
            "mode": mode,
            "phase": phase,
            "transition_reason": reason,
            "exit_code": exit_code,
            "termination_result": termination_result,
            "source_revisions": {"task": expected_task_revision, "attempt_digest": expected_attempt_digest},
            "committed_revisions": {
                "task": None if task is None else task.meta.get("revision"),
                "attempt_digest": attempt_digest,
            },
            "lifecycle_event": event_value,
            "transition_digest": transition_digest,
            "authority_granted": False,
            "local_effects": [],
        }

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        attempt, attempt_digest = read_attempt()
        return evidence("stale", "task_missing", None, attempt, attempt_digest)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        if task.depends_on_task_ids != initial.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while Project-I/O locks were acquired.")
        attempt, attempt_digest = read_attempt()
        if attempt is None:
            return evidence("stale", "attempt_missing", task, None, None)
        if (
            attempt.task_id != task_id
            or attempt.attempt_id != attempt_id
            or attempt.attempt_number != attempt_number
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != machine_name
        ):
            return evidence("stale", "attempt_identity_mismatch", task, attempt, attempt_digest)
        if attempt.reservation_id != reservation_id:
            return evidence("stale", "reservation_mismatch", task, attempt, attempt_digest)
        if any(attempt.process.get(field) != process_identity.get(field) for field in process_identity):
            return evidence("stale", "process_identity_mismatch", task, attempt, attempt_digest)

        task_phase = task.state.get("projection")
        active_claim = task.claim_control.get("active_claim") or {}
        target_attempt_matches = (
            attempt.phase == phase
            and attempt.result.get("reason") == reason
            and attempt.result.get("exit_code") == exit_code
            and attempt.termination.get("result") == termination_result
            and attempt.termination.get("project_io_publication_id") == request_id
        )
        target_task_matches = (
            task_phase == phase
            and task.state.get("reason") == reason
            and task.attempt_control.get("current_attempt_id") is None
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and not active_claim
            and task.control.get("termination_result") == termination_result
            and task.control.get("project_io_publication_id") == request_id
        )
        if _has_superseded_ownership(task, attempt):
            if target_attempt_matches:
                return evidence("historical_already_committed", None, task, attempt, attempt_digest)
            if (
                mode != "active"
                or phase not in {"succeeded", "failed"}
                or termination_result is not None
                or exit_code is None
                or phase != ("succeeded" if exit_code == 0 else "failed")
                or reason != ("completed" if exit_code == 0 else "nonzero_exit")
            ):
                return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)
            if task.meta.get("revision") != expected_task_revision or attempt_digest != expected_attempt_digest:
                return evidence("stale", "source_revision_mismatch", task, attempt, attempt_digest)
            if attempt.phase not in {"starting", "running"}:
                return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
            attempt.phase = phase
            attempt.result.update(exit_code=exit_code, reason=reason, signal=-exit_code if exit_code < 0 else None)
            attempt.timestamps["finished_at"] = utc_now()
            attempt.termination.update(result=None, project_io_publication_id=request_id)
            attempt.meta["revision"] += 1
            attempt.meta["updated_at"] = utc_now()
            mutation_fence()
            atomic_replace(
                attempt_path(cfg.shared_root, task_id, attempt_number),
                attempt.to_dict(),
                before_replace=lambda _stat: mutation_fence(),
            )
            committed_attempt, committed_digest = read_attempt()
            return evidence("historical_committed", None, task, committed_attempt, committed_digest)
        if target_attempt_matches and target_task_matches:
            replay_transition = TerminalTransition(
                task_id,
                attempt_id,
                attempt_number,
                fencing_token,
                phase,
                reason,
                exit_code,
                frozenset({phase}),
                frozenset({phase}),
                "active" if mode == "active" else "detached",
                termination_result,
                project_io_publication_id=request_id,
            )
            committed = commit_terminal_transition_locked(
                cfg,
                task,
                replay_transition,
                mutation_fence=mutation_fence,
            )
            if committed.outcome == "already_committed":
                _attempt, current_digest = read_attempt()
                task = load_task(cfg, task_id)
                return evidence("already_committed", None, task, _attempt, current_digest, committed)
            return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)
        if (attempt.phase == phase and not target_attempt_matches) or (task_phase == phase and not target_task_matches):
            return evidence("stale", "terminal_conflict", task, attempt, attempt_digest)

        if mode == "active":
            current_identity = (
                task.attempt_control.get("current_attempt_id") == attempt_id
                and task.attempt_control.get("current_attempt_number") == attempt_number
            )
        elif mode == "detached_orphan":
            current_identity = (
                task_phase == "blocked"
                and not active_claim
                and task.attempt_control.get("current_attempt_id") is None
                and task.attempt_control.get("current_attempt_number") == attempt_number
                and task.attempt_control.get("next_attempt_number") == attempt_number + 1
            )
        else:
            current_identity = False
        if not current_identity:
            return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
        if mode == "active":
            if (
                task_phase != "running"
                or active_claim.get("attempt_id") != attempt_id
                or active_claim.get("attempt_number") != attempt_number
                or active_claim.get("fencing_token") != fencing_token
                or active_claim.get("machine_name") != machine_name
                or active_claim.get("reservation_id") != reservation_id
            ):
                return evidence("stale", "claim_not_current", task, attempt, attempt_digest)
            allowed_attempt_phases = frozenset({"running"})
        elif mode == "detached_orphan":
            if task_phase != "blocked" or task.claim_control.get("active_claim"):
                return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)
            if task.attempt_control.get("next_attempt_number") != attempt_number + 1:
                return evidence("stale", "task_identity_mismatch", task, attempt, attempt_digest)
            allowed_attempt_phases = frozenset({"orphaned"})
        else:
            return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)

        partial_attempt_matches = target_attempt_matches and (
            task_phase in ({"running"} if mode == "active" else {"blocked"})
        )
        if not partial_attempt_matches and (
            task.meta.get("revision") != expected_task_revision or attempt_digest != expected_attempt_digest
        ):
            return evidence("stale", "source_revision_mismatch", task, attempt, attempt_digest)
        if attempt.phase not in allowed_attempt_phases and not partial_attempt_matches:
            return evidence("stale", "attempt_phase_mismatch", task, attempt, attempt_digest)
        if task_phase not in ({"running"} if mode == "active" else {"blocked"}):
            return evidence("stale", "task_phase_mismatch", task, attempt, attempt_digest)

        transition = TerminalTransition(
            task_id,
            attempt_id,
            attempt_number,
            fencing_token,
            phase,
            reason,
            exit_code,
            frozenset({"running"} if mode == "active" else {"blocked"}),
            allowed_attempt_phases,
            "active" if mode == "active" else "detached",
            termination_result,
            project_io_publication_id=request_id,
        )
        committed = commit_terminal_transition_locked(
            cfg,
            task,
            transition,
            mutation_fence=mutation_fence,
        )
        if committed.outcome not in {"committed", "already_committed"}:
            stale_reason = {
                "attempt_missing": "attempt_missing",
                "stale_claim": "claim_not_current",
                "active_claim_present": "claim_not_current",
                "invalid_attempt_identity_or_phase": "attempt_identity_mismatch",
                "invalid_attempt_source_phase": "attempt_phase_mismatch",
                "invalid_task_source_phase": "task_phase_mismatch",
            }.get(committed.reason, "terminal_conflict")
            _attempt, current_digest = read_attempt()
            current_task = load_task(cfg, task_id)
            return evidence("stale", stale_reason, current_task, _attempt, current_digest)
        current_task = load_task(cfg, task_id)
        current_attempt, current_digest = read_attempt()
        return evidence(
            committed.outcome,
            None,
            current_task,
            current_attempt,
            current_digest,
            committed,
        )


def reconcile_running_tasks(
    cfg: RootConfig, *, executor: Executor | None = None, reservation_runtime_root: Path | None = None
) -> list[str]:
    """Reconcile local manifests and persistent cancellation intents."""
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    reconciled: list[str] = []
    process_dir = cfg.runtime_root / "processes"
    for manifest in iter_json(process_dir):
        data = read_json(manifest).get("process", {})
        task_id = data.get("task_id")
        attempt_id = data.get("attempt_id")
        token = data.get("fencing_token")
        if not task_id or not attempt_id or token is None:
            continue
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != token:
            recovered = None
            pid = data.get("process_group_id")
            number = task.attempt_control.get("current_attempt_number")
            attempt = None
            if number is not None:
                path = attempt_path(cfg.shared_root, task_id, number)
                if path.exists():
                    attempt = AttemptRecord.from_dict(read_json(path))
            evidence_state = (
                inspect_group_identity(attempt.process, data).state
                if attempt is not None and attempt.attempt_id == attempt_id
                else "unknown"
            )
            if task.state["projection"] == "blocked" and evidence_state == "alive":
                from .runtime.attempt_recovery import recover_running_attempt

                recovered = recover_running_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    token,
                    manifest=data,
                    reservation_runtime_root=reservation_runtime_root,
                )
            elif task.state["projection"] == "blocked" and attempt is not None and evidence_state == "absent":
                was_terminated = bool(task.control.get("terminate_running"))
                if finalize_orphaned_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    attempt.current_fencing_token,
                    exit_code=data.get("exit_code"),
                    was_terminated=was_terminated,
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    data.update(
                        {
                            "fencing_token": attempt.current_fencing_token,
                            "observed_state": "exited",
                            "termination_confirmed_at": utc_now() if was_terminated else None,
                        }
                    )
                    atomic_replace(manifest, {"process": data})
                    reconciled.append(task_id)
                    recovered = attempt.current_fencing_token
            elif evidence_state == "alive" and claim.get("attempt_id") == attempt_id:
                if attempt and attempt.current_fencing_token == claim.get("fencing_token"):
                    data.update(
                        {
                            "fencing_token": attempt.current_fencing_token,
                            "recovered_at": utc_now(),
                            "observed_state": "running",
                            "supervisor": ("runner" if inspect_wrapper_identity(data).state == "alive" else "agent"),
                        }
                    )
                    atomic_replace(manifest, {"process": data})
                    recovered = attempt.current_fencing_token
            if recovered is None:
                if pid and evidence_state == "alive":
                    _terminate_process_group(pid)
            continue
        supervisor = "runner" if inspect_wrapper_identity(data).state == "alive" else "agent"
        if data.get("supervisor") != supervisor:
            data["supervisor"] = supervisor
            atomic_replace(manifest, {"process": data})
        if supervisor != "agent":
            continue
        if isinstance(claim.get("termination_decision_id"), str):
            if _continue_committed_termination(cfg, task_id, attempt_id, token):
                reconciled.append(task_id)
            continue
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            continue
        path = attempt_path(cfg.shared_root, task_id, number)
        attempt = AttemptRecord.from_dict(read_json(path))
        evidence_state = inspect_group_identity(attempt.process, data).state
        if evidence_state == "unknown":
            continue
        pid = data.get("process_group_id")
        is_process_alive = evidence_state == "alive"
        was_terminated = bool(task.control.get("terminate_running"))
        if is_process_alive and was_terminated:
            is_process_alive = not _terminate_process_group(pid)
        if not is_process_alive:
            if finalize_agent_supervised_attempt(
                cfg,
                task_id,
                attempt_id,
                token,
                was_terminated=was_terminated,
                reservation_runtime_root=reservation_runtime_root,
            ):
                data.update(
                    {
                        "observed_state": "exited",
                        "exit_code": None,
                        "termination_confirmed_at": utc_now() if was_terminated else None,
                    }
                )
                atomic_replace(manifest, {"process": data})
                reconciled.append(task_id)
            continue
        renewal = renew_attempt_lease(cfg, task_id, attempt_id, token)
        if renewal is True or (
            isinstance(renewal, LeaseRenewalResult) and renewal.outcome is LeaseRenewalOutcome.RENEWED
        ):
            reconciled.append(task_id)
        elif (
            isinstance(renewal, LeaseRenewalResult)
            and renewal.outcome is LeaseRenewalOutcome.TERMINATION_REQUESTED
            and _continue_committed_termination(cfg, task_id, attempt_id, token)
        ):
            reconciled.append(task_id)
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = load_task(cfg, path.stem)
        claim = task.claim_control.get("active_claim") or {}
        expires = claim.get("lease_expires_at")
        if not claim or not expires:
            continue
        if claim.get("authority_mode") == "bounded_lease" and datetime.fromisoformat(
            expires.replace("Z", "+00:00")
        ) <= datetime.now(timezone.utc):
            if expire_claim(
                cfg,
                task.task_id,
                claim["attempt_id"],
                claim["fencing_token"],
                reservation_runtime_root=reservation_runtime_root,
            ):
                reconciled.append(task.task_id)
    return reconciled
