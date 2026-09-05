"""Fenced scheduling pipeline: reserve, claim, materialize, authorize, launch, reconcile."""

from __future__ import annotations

import os
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator

from .events import write_diagnostic_event, write_event
from .lifecycle import TerminalTransition, commit_terminal_transition_locked, dispatch_task_lifecycle_hooks_noexcept
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
from .executor import Executor
from .config_types import RootConfig
from .runtime.locks import group_lock, task_lock
from .runtime.claims import archive_claim
from .runtime.paths import attempt_path, group_path, local_paths, shared_paths, submission_path
from .runtime.records import AttemptRecord, TaskRecord, TaskSpec, normalize_group_record, utc_now
from .runtime.ready import (
    advance_ready_index_build,
    ReadyMarkerRef,
    classify_ready_marker,
    delete_stale_ready_marker,
    mark_ready_index_degraded,
    next_ready_marker,
    prepare_ready_transition,
    read_ready_index_state,
    retire_current_ready_generation,
    retire_previous_ready_generation,
)
from .runtime.active_operations import operation_exists
from .runtime.reservations import ReservationIdentity, attach, release, reserve, reserve_admitted
from .runtime.cpu_lane import attach_cpu, reserve_cpu
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.tasks import load_task, save_task
from .runtime.work_budget import (
    AdaptiveBatchSizer,
    SliceBudget,
    diagnostic_increment,
    diagnostic_span,
)

LEASE_SECONDS = 120
TERMINATION_GRACE_SECONDS = 5.0
TERMINATION_POLL_SECONDS = 0.05


class BorrowAdmissionRequired(RuntimeError):
    """Raised when a borrow claim is attempted without machine-wide admission."""


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
        from .runtime.ready import (
            is_primary_ready_index_active,
            ready_index_route_revision,
        )

        try:
            return all(
                is_primary_ready_index_active(item.cfg)
                and ready_index_route_revision(
                    item.cfg, item.queue_scope, primary_only=True
                ) == item.revision
                for item in self.revisions
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False


def _reservation_root(cfg: RootConfig, value: Path | None = None) -> Path:
    return value if value is not None else cfg.runtime_root


def _is_process_group_alive(process_group_id: int | None) -> bool:
    if not process_group_id:
        return False
    try:
        os.killpg(process_group_id, 0)
    except OSError:
        return False
    return True


def _is_process_alive(process_id: int | None) -> bool:
    if not process_id:
        return False
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def _process_start_time_ticks(process_id: int | None) -> int | None:
    if not process_id:
        return None
    try:
        stat = (Path("/proc") / str(process_id) / "stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


def _terminate_process_group(process_group_id: int, *, grace_seconds: float = TERMINATION_GRACE_SECONDS) -> bool:
    """Terminate a process group and return only after its absence is confirmed."""
    if not _is_process_group_alive(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except OSError:
        return not _is_process_group_alive(process_group_id)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _is_process_group_alive(process_group_id):
            return True
        time.sleep(TERMINATION_POLL_SECONDS)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _is_process_group_alive(process_group_id):
            return True
        time.sleep(TERMINATION_POLL_SECONDS)
    return not _is_process_group_alive(process_group_id)


def _manifest_supervisor(data: dict[str, Any]) -> str:
    wrapper_pid = data.get("wrapper_pid")
    expected_start = data.get("wrapper_start_time_ticks")
    is_runner = (
        _is_process_alive(wrapper_pid)
        and expected_start is not None
        and _process_start_time_ticks(wrapper_pid) == expected_start
    )
    return "runner" if is_runner else "agent"


def _process_evidence_state(attempt: AttemptRecord, data: dict[str, Any]) -> str:
    """Classify local process evidence without conflating mismatch and absence."""
    recorded_group = attempt.process.get("process_group_id")
    manifest_group = data.get("process_group_id")
    if not manifest_group or not recorded_group:
        return "unverifiable"
    if recorded_group != manifest_group:
        return "mismatch"
    expected_start = data.get("process_group_start_time_ticks")
    recorded_start = attempt.process.get("process_group_start_time_ticks")
    if expected_start is None or recorded_start is None:
        return "unverifiable"
    if recorded_start != expected_start:
        return "mismatch"
    current_start = _process_start_time_ticks(manifest_group)
    if current_start == expected_start:
        return "alive" if _is_process_group_alive(manifest_group) else "unverifiable"
    if current_start is not None:
        return "mismatch"
    return "unverifiable" if _is_process_group_alive(manifest_group) else "absent"


@contextmanager
def authority_locks(cfg: RootConfig, task: TaskRecord) -> Iterator[None]:
    """Acquire the only permitted shared authority order."""
    if task.group_name:
        with group_lock(cfg.shared_root, task.group_name):
            with task_lock(cfg.shared_root, task.task_id):
                yield
    else:
        with task_lock(cfg.shared_root, task.task_id):
            yield


def _group_allows(group: dict[str, Any], task: TaskRecord, machine: str) -> bool:
    normalize_group_record(group)
    if group["group"]["dispatch_state"] != "active":
        return False
    worker = group["group"]["worker_set"].get(machine)
    if not worker or worker["state"] != "active":
        return False
    if task.placement_runtime["queue_scope"] == "home":
        return task.placement_policy["home_machine"] == machine
    fallback = task.placement_policy["fallback_constraint"]
    return fallback == "group" or machine in fallback


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
    operation_file = submission_path(cfg.shared_root, operation_id)
    if not operation_file.exists():
        return False
    if read_json(operation_file).get("submission", {}).get("state") != "committed":
        return False
    if not task.group_name:
        return task.placement_policy["home_machine"] == cfg.machine_name
    group_file = group_path(cfg.shared_root, task.group_name)
    return group_file.exists() and _group_allows(read_json(group_file), task, cfg.machine_name)


def _task_worker_role(cfg: RootConfig, task: TaskRecord) -> str | None:
    if not task.group_name:
        return None
    path = group_path(cfg.shared_root, task.group_name)
    if not path.exists():
        return None
    group = read_json(path)
    normalize_group_record(group)
    worker = group["group"]["worker_set"].get(cfg.machine_name)
    return worker.get("scheduling_role") if worker else None


def _clock_evidence(observation: ClockObservation) -> dict[str, Any]:
    return {
        "clock_error_bound_seconds": observation.bound_at(time.monotonic()),
        "provider": observation.provider,
        "observation_id": observation.observation_id,
    }


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
) -> AttemptRecord | None:
    task = load_task(cfg, task_id)
    if task.placement_policy["home_machine"] != cfg.machine_name and not _eligible(cfg, task):
        return None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if not _eligible(cfg, task):
            return None
        group: dict[str, Any] | None = None
        worker: dict[str, Any] | None = None
        if task.group_name:
            group = read_json(group_path(cfg.shared_root, task.group_name))
            normalize_group_record(group)
            worker = group["group"]["worker_set"].get(cfg.machine_name)
            if worker is None:
                return None
            if admission_role is not None and worker["scheduling_role"] != admission_role:
                return None
            if (
                worker["scheduling_role"] == "borrow"
                and (
                    borrow_admission_grant is None
                    or not borrow_admission_grant.is_valid(reservation_runtime_root)
                )
            ):
                raise BorrowAdmissionRequired(
                    "borrow claims require a current machine-agent admission grant."
                )
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
                cfg, task, token, "attempt_materialization_failed",
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
    archive_claim(cfg, task.task_id, claim, reason)
    task.claim_control["active_claim"] = None
    task.attempt_control["current_attempt_id"] = None
    task.state.update({"projection": "queued", "reason": reason})
    task.meta["revision"] += 1
    task.meta["updated_at"] = utc_now()
    save_task(cfg, task)
    release(reservation_runtime_root or cfg.runtime_root, claim["reservation_id"], reason)


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
    )


def _has_local_launch_evidence(cfg: RootConfig, attempt_id: str) -> bool:
    paths = local_paths(cfg.runtime_root)
    return any(
        (paths[directory] / f"{attempt_id}.json").exists()
        for directory in ("launch_intents", "registrations", "processes")
    )


def resume_starting_attempt(
    cfg: RootConfig,
    task_id: str,
    *,
    reservation_runtime_root: Path | None = None,
    expected_reservation: ReservationIdentity | None = None,
) -> AttemptRecord | None:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    cancel_result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        attempt_id = claim.get("attempt_id")
        attempt_number = task.attempt_control.get("current_attempt_number")
        fencing_token = claim.get("fencing_token")
        if (
            task.state.get("projection") != "running"
            or claim.get("machine_name") != cfg.machine_name
            or claim.get("launch_state") != "starting"
            or not isinstance(attempt_id, str)
            or not isinstance(attempt_number, int)
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
            attempt.attempt_id != attempt_id
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != cfg.machine_name
            or attempt.phase not in {"claimed", "starting"}
        ):
            return None
        if task.control.get("cancellation_requested_at"):
            cancel_result = _cancel_prelaunch_locked(cfg, task, "cancelled_before_launch", {"claimed", "starting"})
        elif task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return None
        elif operation_exists(cfg, "cleanup", task.task_id):
            return None
        elif task.group_name and not _group_allows(
            read_json(group_path(cfg.shared_root, task.group_name)), task, cfg.machine_name
        ):
            cancel_result = _cancel_prelaunch_locked(cfg, task, "worker_or_dispatch_changed", {"claimed", "starting"})
        elif not (claim.get("authority_mode") != "bounded_lease" or clock_capability(cfg).is_healthy):
            return None
        else:
            launch_id = uuid.uuid4().hex
            authorized_at = utc_now()
            claim["launch_id"] = launch_id
            claim["launch_authorized_at"] = authorized_at
            task.meta["revision"] += 1
            task.meta["updated_at"] = authorized_at
            save_task(cfg, task)
            attempt.phase = "starting"
            attempt.authorization["launch_id"] = launch_id
            attempt.timestamps["launch_authorized_at"] = authorized_at
            atomic_replace(attempt_path(cfg.shared_root, task_id, attempt_number), attempt.to_dict())
            return attempt
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            release(reservation_runtime_root, cancel_result.reservation_id, cancel_result.reason or "cancelled_before_launch")
        if cancel_result.event:
            dispatch_task_lifecycle_hooks_noexcept(cfg, cancel_result.event)
    return None


def authorize_launch(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    *,
    reservation_runtime_root: Path | None = None,
) -> bool:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    cancel_result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("machine_name") != cfg.machine_name:
            return False
        if claim.get("authority_mode") not in {"bounded_lease", "holder_bound"}:
            return False
        if claim.get("authority_mode") == "bounded_lease" and not clock_capability(cfg).is_healthy:
            return False
        if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return False
        if operation_exists(cfg, "cleanup", task.task_id):
            return False
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        attempt_number = task.attempt_control.get("current_attempt_number")
        if not isinstance(attempt_number, int):
            return False
        try:
            attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, attempt_number)))
        except (FileNotFoundError, KeyError, ValueError):
            return False
        if (
            attempt.attempt_id != attempt_id
            or attempt.current_fencing_token != fencing_token
            or attempt.machine_name != cfg.machine_name
        ):
            return False
        if claim.get("launch_state") == "starting":
            return False
        elif (
            claim.get("launch_state") != "claimed"
            or attempt.phase != "claimed"
            or task.control.get("cancellation_requested_at")
        ):
            cancel_result = _cancel_prelaunch_locked(cfg, task, "launch_gate_lost")
        elif task.group_name:
            group = read_json(group_path(cfg.shared_root, task.group_name))
            if not _group_allows(group, task, cfg.machine_name):
                cancel_result = _cancel_prelaunch_locked(cfg, task, "worker_or_dispatch_changed")
            else:
                claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
                claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
        if cancel_result is None:
            authorized_at = utc_now()
            launch_id = uuid.uuid4().hex
            claim["launch_state"] = "starting"
            claim["launch_authorized_at"] = authorized_at
            claim["launch_id"] = launch_id
            task.meta["revision"] += 1
            task.meta["updated_at"] = authorized_at
            save_task(cfg, task)
            attempt.phase = "starting"
            attempt.authorization["launch_id"] = launch_id
            attempt.timestamps["launch_authorized_at"] = authorized_at
            atomic_replace(attempt_path(cfg.shared_root, task_id, attempt_number), attempt.to_dict())
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            release(reservation_runtime_root, cancel_result.reservation_id, cancel_result.reason or "cancelled")
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
        if claim and claim.get("launch_state") == "claimed":
            cancel_result = _cancel_prelaunch_locked(cfg, task, "cancelled_before_launch")
            has_saved_task = cancel_result is not None and cancel_result.outcome == "committed"
            result_task = load_task(cfg, task_id) if cancel_result is None else None
        else:
            result_task = None
        if not claim and task.state["projection"] == "queued":
            task.state.update({"projection": "cancelled", "reason": "cancelled_by_user"})
        if not has_saved_task:
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
        if task.state["projection"] == "cancelled":
            retire_current_ready_generation(cfg, task)
    if cancel_result is not None:
        if cancel_result.reservation_id and cancel_result.reservation_machine_name == cfg.machine_name:
            release(reservation_runtime_root, cancel_result.reservation_id, cancel_result.reason or "cancelled_before_launch")
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
        raise BorrowAdmissionRequired(
            "borrow dispatch requires a current machine-agent admission grant."
        )
    if should_recover_starting:
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = load_task(cfg, path.stem)
            attempt = resume_starting_attempt(
                cfg, task.task_id, reservation_runtime_root=reservation_runtime_root
            )
            if attempt is None:
                continue
            try:
                executor.launch_attempt(cfg, task.task_id, attempt)
                launched.append(task.task_id)
            except Exception:
                fail_attempt(
                    cfg,
                    task.task_id,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                    "executor_launch_failed",
                    reservation_runtime_root=reservation_runtime_root,
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
        candidates: list[tuple[ReadyMarkerRef, str]] = []
        scope_identities: dict[str, set[str]] = {scope: set() for scope in scopes}
        while (
            len(candidates) < sizer.batch_size
            and budget.can_start_record(operations=3)
        ):
            made_progress = False
            for scope in scopes:
                if (
                    len(candidates) >= sizer.batch_size
                    or not budget.can_start_record(operations=3)
                ):
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
                candidates.append((reference, scope))
                made_progress = True
            if not made_progress:
                break
        for reference, _scope in candidates:
            inspected.add((cursor_project_id, _scope, reference.identity))
            started_ns = time.monotonic_ns()
            diagnostic_increment("scheduler.ready.markers_inspected")
            result = classify_ready_marker(cfg, reference)
            if result.classification == "corrupt":
                diagnostic_increment("scheduler.ready.corrupt")
                mark_ready_index_degraded(cfg, f"marker_corrupt:{reference.identity}")
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
            if (
                admission_role is not None
                and task.group_name
                and _task_worker_role(cfg, task) != admission_role
            ):
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if (
                borrow_admission_grant is None
                and task.group_name
                and _task_worker_role(cfg, task) == "borrow"
            ):
                diagnostic_increment("scheduler.borrow.skipped_without_admission")
                raise BorrowAdmissionRequired(
                    "borrow dispatch requires a current machine-agent admission grant."
                )
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
                            available = gpus + available
                            break
                        attempt = claim_task(
                            cfg, task.task_id, gpus,
                            reservation_runtime_root=reservation_runtime_root,
                            project_id=project_id,
                            admission_role=admission_role,
                            borrow_admission_grant=borrow_admission_grant,
                        )
                else:
                    if before_claim is not None and not before_claim():
                        available = gpus + available
                        break
                    attempt = claim_task(
                        cfg, task.task_id, gpus,
                        reservation_runtime_root=reservation_runtime_root,
                        project_id=project_id,
                        admission_role=admission_role,
                        borrow_admission_grant=borrow_admission_grant,
                    )
            except ValueError:
                available = gpus + available
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            if attempt is None:
                available = gpus + available
                diagnostic_increment("scheduler.ready.claim_races")
                sizer.observe(max(1, time.monotonic_ns() - started_ns))
                continue
            new_claims += 1
            if on_claim is not None:
                on_claim(task.task_id)
            try:
                if authorize_launch(
                    cfg,
                    task.task_id,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    authorized = _load_current_attempt(cfg, load_task(cfg, task.task_id))
                    if authorized is not None:
                        executor.launch_attempt(cfg, task.task_id, authorized)
                        launched.append(task.task_id)
            except Exception:
                fail_attempt(
                    cfg,
                    task.task_id,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                    "executor_launch_failed",
                    reservation_runtime_root=reservation_runtime_root,
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
        if (
            admission_role is not None
            and task.group_name
            and _task_worker_role(cfg, task) != admission_role
        ):
            continue
        if (
            borrow_admission_grant is None
            and task.group_name
            and _task_worker_role(cfg, task) == "borrow"
        ):
            diagnostic_increment("scheduler.borrow.skipped_without_admission")
            raise BorrowAdmissionRequired(
                "borrow dispatch requires a current machine-agent admission grant."
            )
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
                        available = gpus + available
                        break
                    attempt = claim_task(
                        cfg, task.task_id, gpus,
                        reservation_runtime_root=reservation_runtime_root, project_id=project_id,
                        admission_role=admission_role,
                        borrow_admission_grant=borrow_admission_grant,
                    )
            else:
                if before_claim is not None and not before_claim():
                    available = gpus + available
                    break
                attempt = claim_task(
                    cfg, task.task_id, gpus,
                    reservation_runtime_root=reservation_runtime_root, project_id=project_id,
                    admission_role=admission_role,
                    borrow_admission_grant=borrow_admission_grant,
                )
        except ValueError:
            available = gpus + available
            continue
        if attempt is None:
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
            ):
                available = gpus + available
                continue
            authorized = _load_current_attempt(cfg, load_task(cfg, task.task_id))
            if authorized is None:
                available = gpus + available
                continue
            executor.launch_attempt(cfg, task.task_id, authorized)
            launched.append(task.task_id)
        except Exception:
            fail_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                "executor_launch_failed",
                reservation_runtime_root=reservation_runtime_root,
            )
    return launched


def has_eligible_local_work(cfg: RootConfig) -> bool:
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        if _eligible(cfg, load_task(cfg, path.stem)):
            return True
    return False


def renew_attempt_lease(
    cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, lease_seconds: int | None = None
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
            if task.control.get("terminate_running"):
                return LeaseRenewalResult(LeaseRenewalOutcome.TERMINATION_REQUESTED, attempt_id)
            if claim.get("termination_decision_id"):
                return LeaseRenewalResult(LeaseRenewalOutcome.TERMINATION_REQUESTED, attempt_id)
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
            if attempt.authority_mode == "holder_bound":
                return LeaseRenewalResult(LeaseRenewalOutcome.NOT_REQUIRED, attempt_id, observed_token=fencing_token)
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


def resolve_execution_authority(
    cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, decision_id: str,
    *, reservation_runtime_root: Path | None = None,
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
        from .runtime.recovery import recover_running_attempt

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
        if claim.get("authority_mode") != "bounded_lease":
            return False
        holder_bound = claim.get("clock_error_bound_seconds")
        expires = claim.get("lease_expires_at")
        if not isinstance(holder_bound, (int, float)) or not isinstance(expires, str):
            task.state.update({"projection": "blocked", "reason": "authority_mode_evidence_invalid"})
            save_task(cfg, task)
            return False
        reclaimer_bound = capability.observation.bound_at(time.monotonic())
        if datetime.now(timezone.utc) < reclaim_allowed_at(expires, holder_bound, reclaimer_bound):
            return False
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        attempt = AttemptRecord.from_dict(read_json(path))
        old_ready_generation = None
        if claim.get("launch_state") == "claimed":
            task.state.update({"projection": "queued", "reason": "lease_expired_before_launch"})
            old_ready_generation, _ = prepare_ready_transition(
                cfg, task, "prelaunch_expiry"
            )
            attempt.phase = "cancelled"
            attempt.result["reason"] = "lease_expired_before_launch"
            release(reservation_runtime_root, claim["reservation_id"], "lease_expired_before_launch")
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
        return True


def fail_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    reason: str,
    *,
    reservation_runtime_root: Path | None = None,
) -> bool:
    reservation_runtime_root = _reservation_root(cfg, reservation_runtime_root)
    task = load_task(cfg, task_id)
    result = None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
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
        release(reservation_runtime_root, result.reservation_id, reason)
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
        release(reservation_runtime_root, result.reservation_id, reason)
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
            or attempt.phase not in {"orphaned", "running"}
            or attempt.current_fencing_token != fencing_token
        ):
            return False
        if was_terminated:
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
                frozenset({"orphaned", "running"}),
                "detached",
                termination_result,
            ),
        )
    if result.outcome != "committed":
        return False
    if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
        release(reservation_runtime_root, result.reservation_id, reason)
    if result.event:
        dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
    return True


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
            evidence_state = _process_evidence_state(attempt, data) if attempt is not None else "unverifiable"
            if task.state["projection"] == "blocked" and evidence_state == "alive":
                from .runtime.recovery import recover_running_attempt

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
                            "supervisor": _manifest_supervisor(data),
                        }
                    )
                    atomic_replace(manifest, {"process": data})
                    recovered = attempt.current_fencing_token
            if recovered is None:
                if pid and evidence_state == "alive":
                    _terminate_process_group(pid)
            continue
        supervisor = _manifest_supervisor(data)
        if data.get("supervisor") != supervisor:
            data["supervisor"] = supervisor
            atomic_replace(manifest, {"process": data})
        if supervisor != "agent":
            continue
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            continue
        path = attempt_path(cfg.shared_root, task_id, number)
        attempt = AttemptRecord.from_dict(read_json(path))
        evidence_state = _process_evidence_state(attempt, data)
        if evidence_state in {"mismatch", "unverifiable"}:
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
