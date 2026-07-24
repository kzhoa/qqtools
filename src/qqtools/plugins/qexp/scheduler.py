"""Fenced scheduling pipeline: reserve, claim, materialize, authorize, launch, reconcile."""
from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .events import write_event
from .executor import Executor
from .config_types import RootConfig
from .runtime.locks import group_lock, task_lock
from .runtime.paths import attempt_path, group_path, shared_paths, submission_path
from .runtime.records import AttemptRecord, TaskRecord, utc_now
from .runtime.reservations import attach, release, reserve
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.tasks import load_task, save_task

LEASE_SECONDS = 60
TERMINATION_GRACE_SECONDS = 5.0
TERMINATION_POLL_SECONDS = 0.05


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


def _terminate_process_group(
        process_group_id: int, *, grace_seconds: float = TERMINATION_GRACE_SECONDS) -> bool:
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
    is_runner = (_is_process_alive(wrapper_pid)
                 and expected_start is not None
                 and _process_start_time_ticks(wrapper_pid) == expected_start)
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
    if (shared_paths(cfg.shared_root)["cleanup"] / f"{task.task_id}.json").exists():
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


def _claim(cfg: RootConfig, task_id: str, reservation: dict[str, Any], *, lease_seconds: int) -> AttemptRecord | None:
    task = load_task(cfg, task_id)
    if not _eligible(cfg, task):
        return None
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if not _eligible(cfg, task):
            return None
        attempt_number = task.attempt_control["next_attempt_number"]
        attempt_id = f"{task.task_id}-attempt-{attempt_number}"
        token = task.claim_control["fencing_epoch"] + 1
        attempt = AttemptRecord.claimed(task, cfg.machine_name, reservation["reservation"]["gpu_ids"],
                                        reservation["reservation"]["reservation_id"], token,
                                        lease_seconds=lease_seconds, attempt_id=attempt_id)
        claim = {"claim_id": attempt_id, "attempt_id": attempt_id, "attempt_number": attempt_number,
                 "machine_name": cfg.machine_name, "reservation_id": attempt.reservation_id,
                 "queue_origin": task.placement_runtime["queue_scope"], "fencing_token": token,
                 "claimed_at": utc_now(), "lease_expires_at": attempt.lease["expires_at"],
                 "launch_state": "claimed", "launch_authorized_at": None,
                 "group_dispatch_epoch": None, "group_worker_set_epoch": None}
        if task.group_name:
            group = read_json(group_path(cfg.shared_root, task.group_name))
            claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
            claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
            worker = group["group"]["worker_set"][cfg.machine_name]
            claim["worker_state_epoch"] = worker["state_epoch"]
            attempt.authorization["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
            attempt.authorization["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
            attempt.authorization["worker_state_epoch"] = worker["state_epoch"]
        task.claim_control.update({"fencing_epoch": token, "active_claim": claim})
        task.attempt_control.update({"current_attempt_id": attempt_id,
                                     "current_attempt_number": attempt_number,
                                     "next_attempt_number": attempt_number + 1})
        task.state["projection"] = "running"
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        try:
            atomic_replace(attempt_path(cfg.shared_root, task_id, attempt_number), attempt.to_dict())
            attach(cfg.runtime_root, attempt.reservation_id, attempt.attempt_id, token)
        except Exception:
            _release_claim_locked(cfg, task, token, "attempt_materialization_failed")
            raise
        return attempt


def _release_claim_locked(cfg: RootConfig, task: TaskRecord, token: int, reason: str) -> None:
    claim = task.claim_control.get("active_claim") or {}
    if claim.get("fencing_token") != token:
        return
    task.claim_control["active_claim"] = None
    task.attempt_control["current_attempt_id"] = None
    task.state.update({"projection": "queued", "reason": reason})
    task.meta["revision"] += 1
    task.meta["updated_at"] = utc_now()
    save_task(cfg, task)
    release(cfg.runtime_root, claim["reservation_id"], reason)


def claim_task(cfg: RootConfig, task_id: str, assigned_gpus: list[int], *, lease_seconds: int = LEASE_SECONDS) -> AttemptRecord | None:
    reservation = reserve(cfg.runtime_root, task_id, assigned_gpus)
    try:
        attempt = _claim(cfg, task_id, reservation, lease_seconds=lease_seconds)
    except Exception:
        release(cfg.runtime_root, reservation["reservation"]["reservation_id"], "claim_failed")
        raise
    if attempt is None:
        release(cfg.runtime_root, reservation["reservation"]["reservation_id"], "claim_lost")
    return attempt


def authorize_launch(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int) -> bool:
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return False
        if (shared_paths(cfg.shared_root)["cleanup"] / f"{task.task_id}.json").exists():
            return False
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        if claim.get("launch_state") != "claimed" or task.control.get("cancellation_requested_at"):
            _cancel_prelaunch_locked(cfg, task, "launch_gate_lost")
            return False
        if task.group_name:
            group = read_json(group_path(cfg.shared_root, task.group_name))
            if not _group_allows(group, task, cfg.machine_name):
                _cancel_prelaunch_locked(cfg, task, "worker_or_dispatch_changed")
                return False
            claim["group_dispatch_epoch"] = group["group"]["dispatch_epoch"]
            claim["group_worker_set_epoch"] = group["group"]["worker_set_epoch"]
        claim["launch_state"] = "starting"
        claim["launch_authorized_at"] = utc_now()
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True


def _cancel_prelaunch_locked(cfg: RootConfig, task: TaskRecord, reason: str) -> None:
    claim = task.claim_control.get("active_claim") or {}
    attempt_id = task.attempt_control.get("current_attempt_id")
    if attempt_id:
        path = attempt_path(cfg.shared_root, task.task_id, task.attempt_control["current_attempt_number"])
        if path.exists():
            attempt = AttemptRecord.from_dict(read_json(path))
            if attempt.phase == "claimed":
                attempt.phase = "cancelled"
                attempt.result["reason"] = reason
                attempt.timestamps["finished_at"] = utc_now()
                atomic_replace(path, attempt.to_dict())
    if claim and claim.get("machine_name") == cfg.machine_name:
        release(cfg.runtime_root, claim["reservation_id"], reason)
    task.claim_control["active_claim"] = None
    task.attempt_control["current_attempt_id"] = None
    task.state.update({"projection": "cancelled", "reason": reason})
    task.meta["revision"] += 1
    task.meta["updated_at"] = utc_now()
    save_task(cfg, task)


def cancel_task(cfg: RootConfig, task_id: str, *, terminate_running: bool = True) -> TaskRecord:
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if (task.control.get("cleanup_operation_id") or task.control.get("cleanup_state")
                or (shared_paths(cfg.shared_root)["cleanup"] / f"{task_id}.json").exists()):
            raise ValueError(f"Task {task_id!r} is being cleaned and cannot be cancelled.")
        if task.state["projection"] in {"succeeded", "failed", "cancelled"}:
            return task
        claim = task.claim_control.get("active_claim") or {}
        task.control.update({"cancellation_requested_at": utc_now(), "terminate_running": terminate_running,
                             "requested_by": cfg.machine_name})
        if claim and claim.get("launch_state") == "claimed":
            _cancel_prelaunch_locked(cfg, task, "cancelled_before_launch")
            return load_task(cfg, task_id)
        if not claim and task.state["projection"] == "queued":
            task.state.update({"projection": "cancelled", "reason": "cancelled_by_user"})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return task


def run_dispatch_cycle(cfg: RootConfig, *, available_gpus: list[int] | None = None,
                       executor: Executor | None = None) -> list[str]:
    available = list(available_gpus or [])
    executor = executor or Executor()
    launched: list[str] = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = load_task(cfg, path.stem)
        if not _eligible(cfg, task) or len(available) < task.spec.requested_gpus:
            continue
        gpus, available = available[:task.spec.requested_gpus], available[task.spec.requested_gpus:]
        try:
            attempt = claim_task(cfg, task.task_id, gpus)
        except ValueError:
            available = gpus + available
            continue
        if attempt is None:
            available = gpus + available
            continue
        try:
            executor.launch_attempt(cfg, task.task_id, attempt)
            launched.append(task.task_id)
        except Exception:
            fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "executor_launch_failed")
    return launched


def has_eligible_local_work(cfg: RootConfig) -> bool:
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        if _eligible(cfg, load_task(cfg, path.stem)):
            return True
    return False


def renew_attempt_lease(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int,
                        lease_seconds: int = LEASE_SECONDS) -> bool:
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        claim["lease_expires_at"] = expires
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        attempt = AttemptRecord.from_dict(read_json(path))
        if attempt.current_fencing_token != fencing_token:
            return False
        attempt.lease.update({"renewed_at": utc_now(), "expires_at": expires})
        atomic_replace(path, attempt.to_dict())
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True


def expire_claim(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int) -> bool:
    """Converge an expired claim without creating replacement execution authority."""
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        attempt = AttemptRecord.from_dict(read_json(path))
        if claim.get("launch_state") == "claimed":
            attempt.phase = "cancelled"
            attempt.result["reason"] = "lease_expired_before_launch"
            release(cfg.runtime_root, claim["reservation_id"], "lease_expired_before_launch")
            task.state.update({"projection": "queued", "reason": "lease_expired_before_launch"})
        else:
            attempt.phase = "orphaned"
            attempt.result.update({"exit_code": None, "signal": None, "category": None,
                                   "reason": None})
            attempt.timestamps["orphaned_at"] = utc_now()
            task.state.update({"projection": "blocked", "reason": "orphaned_attempt_requires_recovery"})
        if attempt.phase != "orphaned":
            attempt.timestamps["finished_at"] = utc_now()
        atomic_replace(path, attempt.to_dict())
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        if attempt.phase == "orphaned":
            try:
                write_event(cfg, "attempt_orphaned", task_id=task_id, details={
                    "attempt_id": attempt_id, "fencing_token": fencing_token,
                    "reason": "lease_expired_process_unknown",
                })
            except OSError:
                pass
        return True


def fail_attempt(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, reason: str) -> bool:
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        attempt = AttemptRecord.from_dict(read_json(path))
        attempt.phase = "failed"
        attempt.result["reason"] = reason
        attempt.timestamps["finished_at"] = utc_now()
        atomic_replace(path, attempt.to_dict())
        release(cfg.runtime_root, claim["reservation_id"], reason)
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": "failed", "reason": reason})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True


def finalize_agent_supervised_attempt(cfg: RootConfig, task_id: str, attempt_id: str,
                                      fencing_token: int, *, was_terminated: bool) -> bool:
    """Publish terminal truth after the agent confirms a recovered process is absent."""
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return False
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            return False
        path = attempt_path(cfg.shared_root, task_id, number)
        attempt = AttemptRecord.from_dict(read_json(path))
        if attempt.attempt_id != attempt_id or attempt.current_fencing_token != fencing_token:
            return False
        reason = "terminated_by_agent" if was_terminated else "process_exited_without_status"
        phase = "cancelled" if was_terminated else "failed"
        attempt.phase = phase
        attempt.result.update({"exit_code": None, "reason": reason})
        attempt.timestamps["finished_at"] = utc_now()
        if was_terminated:
            attempt.termination.update({"acknowledged_at": utc_now(), "result": "terminated"})
        atomic_replace(path, attempt.to_dict())
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": phase, "reason": reason})
        if was_terminated:
            task.control.update({"termination_acknowledged_at": utc_now(),
                                 "termination_result": "terminated"})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        release(cfg.runtime_root, claim["reservation_id"], reason)
        return True


def finalize_orphaned_attempt(
        cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int,
        *, exit_code: int | None, was_terminated: bool) -> bool:
    """Resolve a blocked orphan after local evidence confirms its process is absent."""
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        if task.state["projection"] != "blocked" or task.claim_control.get("active_claim"):
            return False
        number = task.attempt_control.get("current_attempt_number")
        if number is None or task.attempt_control["next_attempt_number"] != number + 1:
            return False
        path = attempt_path(cfg.shared_root, task_id, number)
        attempt = AttemptRecord.from_dict(read_json(path))
        if (attempt.attempt_id != attempt_id or attempt.phase not in {"orphaned", "running"}
                or attempt.current_fencing_token != fencing_token):
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
        attempt.phase = phase
        attempt.result.update({"exit_code": exit_code, "reason": reason})
        attempt.timestamps["finished_at"] = utc_now()
        if was_terminated:
            attempt.termination.update({"acknowledged_at": utc_now(),
                                        "result": termination_result})
        atomic_replace(path, attempt.to_dict())
        task.claim_control["fencing_epoch"] = max(
            task.claim_control["fencing_epoch"], attempt.current_fencing_token
        )
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": phase, "reason": reason})
        if was_terminated:
            task.control.update({"termination_acknowledged_at": utc_now(),
                                 "termination_result": termination_result})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        release(cfg.runtime_root, attempt.reservation_id, reason)
        return True


def reconcile_running_tasks(cfg: RootConfig, *, executor: Executor | None = None) -> list[str]:
    """Reconcile local manifests and persistent cancellation intents."""
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
            evidence_state = (_process_evidence_state(attempt, data)
                              if attempt is not None else "unverifiable")
            if task.state["projection"] == "blocked" and evidence_state == "alive":
                from .runtime.recovery import recover_running_attempt
                recovered = recover_running_attempt(cfg, task_id, attempt_id, token, manifest=data)
            elif (task.state["projection"] == "blocked" and attempt is not None
                  and evidence_state == "absent"):
                was_terminated = bool(task.control.get("terminate_running"))
                if finalize_orphaned_attempt(
                        cfg, task_id, attempt_id, attempt.current_fencing_token,
                        exit_code=data.get("exit_code"), was_terminated=was_terminated):
                    data.update({"fencing_token": attempt.current_fencing_token,
                                 "observed_state": "exited",
                                 "termination_confirmed_at":
                                     utc_now() if was_terminated else None})
                    atomic_replace(manifest, {"process": data})
                    reconciled.append(task_id)
                    recovered = attempt.current_fencing_token
            elif evidence_state == "alive" and claim.get("attempt_id") == attempt_id:
                if attempt and attempt.current_fencing_token == claim.get("fencing_token"):
                    data.update({"fencing_token": attempt.current_fencing_token,
                                 "recovered_at": utc_now(), "observed_state": "running",
                                 "supervisor": _manifest_supervisor(data)})
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
                    cfg, task_id, attempt_id, token, was_terminated=was_terminated):
                data.update({"observed_state": "exited", "exit_code": None,
                             "termination_confirmed_at": utc_now() if was_terminated else None})
                atomic_replace(manifest, {"process": data})
                reconciled.append(task_id)
            continue
        if renew_attempt_lease(cfg, task_id, attempt_id, token):
            reconciled.append(task_id)
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = load_task(cfg, path.stem)
        claim = task.claim_control.get("active_claim") or {}
        expires = claim.get("lease_expires_at")
        if not claim or not expires:
            continue
        expires_at = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expires_at <= datetime.now(timezone.utc):
            if expire_claim(cfg, task.task_id, claim["attempt_id"], claim["fencing_token"]):
                reconciled.append(task.task_id)
    return reconciled
