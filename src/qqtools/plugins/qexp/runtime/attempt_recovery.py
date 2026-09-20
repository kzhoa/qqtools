"""Fenced recovery CAS for locally verified orphaned Attempts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from ..config_types import RootConfig
from ..lease import clock_capability, lease_expiry, load_lease_policy, persist_clock_observation
from ..scheduler import _manifest_supervisor, authority_locks
from .group_discovery.changes import record_task_change
from .group_namespace import read_group
from .paths import attempt_path, group_path, local_paths
from .records import AttemptRecord, normalize_group_record, utc_now
from .resources.reservations import retag
from .store import atomic_replace, read_json
from .tasks import load_task, save_task
from .termination import attempt_control_lock, recovery_check_steps


def recovery_steps(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    expired_token: int,
    manifest: dict[str, object] | None = None,
    reservation_runtime_root: Path | None = None,
    *,
    cooperative: bool = True,
) -> Generator[None, None, int | None]:
    """Restore authority only for a locally verified live orphaned process."""

    def reject(reason: str) -> None:
        path = cfg.runtime_root / "authority-diagnostics" / f"{attempt_id}.json"
        try:
            from .store import atomic_replace

            atomic_replace(path, {"authority_diagnostic": {"attempt_id": attempt_id, "reason": reason}})
        except OSError:
            pass

    reservation_root = reservation_runtime_root or cfg.runtime_root
    manifest_path = cfg.runtime_root / "processes" / f"{attempt_id}.json"
    with attempt_control_lock(cfg, attempt_id, blocking=not cooperative) as acquired:
        if not acquired:
            reject("recovery_lock_busy")
            return None
        if (yield from recovery_check_steps(cfg, attempt_id)):
            reject("recovery_termination_blocked")
            return None
        policy = load_lease_policy(cfg)
        capability = clock_capability(cfg, policy)
        if not capability.is_healthy or capability.observation is None:
            reject("recovery_clock_unhealthy")
            return None
        if manifest is None or cooperative:
            if not manifest_path.exists():
                return None
            record = read_json(manifest_path)
            manifest = record.get("process") if isinstance(record, dict) else None
        if not isinstance(manifest, dict):
            raise ValueError("local process manifest must be an object")
        if (
            manifest.get("task_id") != task_id
            or manifest.get("attempt_id") != attempt_id
            or manifest.get("fencing_token") != expired_token
        ):
            reject("recovery_manifest_identity_mismatch")
            return None
        task = load_task(cfg, task_id)
        with authority_locks(cfg, task):
            task = load_task(cfg, task_id)
            if task.state["projection"] != "blocked" or task.claim_control.get("active_claim"):
                reject("recovery_task_state_invalid")
                return None
            number = task.attempt_control.get("current_attempt_number")
            if number is None:
                return None
            path = attempt_path(cfg.shared_root, task_id, number)
            attempt = AttemptRecord.from_dict(read_json(path))
            if attempt.attempt_id != attempt_id:
                return None
            if attempt.authority_mode != "bounded_lease":
                return None
            if attempt.termination.get("decision_id"):
                return None
            if cooperative:
                from ..scheduler import _process_evidence_state

                if (local_paths(cfg.runtime_root)["observations"] / f"{attempt_id}.json").exists():
                    return None
                if _process_evidence_state(attempt, manifest) != "alive":
                    reject("recovery_process_not_alive")
                    return None
            is_partial_recovery = attempt.phase == "running" and attempt.current_fencing_token > expired_token
            if not is_partial_recovery and (
                attempt.phase != "orphaned" or attempt.current_fencing_token != expired_token
            ):
                reject("recovery_attempt_state_invalid")
                return None
            if task.group_name:
                group = read_group(cfg.shared_root, task.group_name)
                normalize_group_record(group)
                worker = group["group"]["worker_set"].get(cfg.machine_name)
                if not worker or worker["state"] not in {"active", "draining"}:
                    reject("recovery_worker_state_invalid")
                    return None
                if task.control.get("terminate_running"):
                    return None
                if (
                    worker["state"] == "draining"
                    and attempt.authorization.get("worker_state_epoch", -1) >= worker["state_epoch"]
                ):
                    return None
                for barrier in group["group"].get("cancellation_barriers", []):
                    if (
                        barrier.get("terminate_running")
                        and (task.group_membership_sequence or 0) <= barrier["membership_high_watermark"]
                    ):
                        return None
            token = attempt.current_fencing_token if is_partial_recovery else task.claim_control["fencing_epoch"] + 1
            expires = lease_expiry(policy)
            persist_clock_observation(cfg, capability.observation)
            evidence = {
                "clock_error_bound_seconds": capability.observation.bound_at(time.monotonic()),
                "provider": capability.observation.provider,
                "observation_id": capability.observation.observation_id,
            }
            with record_task_change(
                cfg,
                task,
                "recovery",
                details={"expected_attempt_id": attempt_id, "expired_token": expired_token},
            ):
                if not retag(reservation_root, attempt.reservation_id, attempt_id, token):
                    reject("recovery_reservation_retag_failed")
                    return None
                task.claim_control.update(
                    {
                        "fencing_epoch": token,
                        "active_claim": {
                            "claim_id": attempt_id,
                            "attempt_id": attempt_id,
                            "attempt_number": number,
                            "machine_name": cfg.machine_name,
                            "reservation_id": attempt.reservation_id,
                            "queue_origin": task.placement_runtime["queue_scope"],
                            "fencing_token": token,
                            "claimed_at": utc_now(),
                            "authority_mode": "bounded_lease",
                            "clock_error_bound_seconds": evidence["clock_error_bound_seconds"],
                            "clock_provider": evidence["provider"],
                            "clock_observation_id": evidence["observation_id"],
                            "lease_expires_at": expires,
                            "launch_state": "running",
                            "launch_authorized_at": attempt.timestamps.get("launch_authorized_at"),
                            "group_dispatch_epoch": attempt.authorization.get("group_dispatch_epoch"),
                            "group_worker_set_epoch": attempt.authorization.get("group_worker_set_epoch"),
                        },
                    }
                )
                task.state.update({"projection": "running", "reason": "recovered_live_attempt"})
                task.attempt_control["current_attempt_id"] = attempt_id
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                attempt.current_fencing_token = token
                if not is_partial_recovery:
                    attempt.token_history.append(token)
                attempt.phase = "running"
                attempt.result.update({"exit_code": None, "signal": None, "category": None, "reason": None})
                attempt.timestamps["finished_at"] = None
                attempt.timestamps["recovered_at"] = utc_now()
                attempt.lease.update({"renewed_at": utc_now(), "expires_at": expires, "clock_evidence": evidence})
                atomic_replace(path, attempt.to_dict())
                save_task(cfg, task)
                manifest = dict(manifest)
                manifest.update(
                    {
                        "fencing_token": token,
                        "recovered_at": utc_now(),
                        "observed_state": "running",
                        "supervisor": _manifest_supervisor(manifest),
                    }
                )
                atomic_replace(manifest_path, {"process": manifest})
                return token


def recover_running_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    expired_token: int,
    manifest: dict[str, object] | None = None,
    reservation_runtime_root: Path | None = None,
) -> int | None:
    """Synchronous recovery for doctor and standalone reconciliation callers."""
    steps = recovery_steps(
        cfg, task_id, attempt_id, expired_token, manifest, reservation_runtime_root, cooperative=False
    )
    try:
        while True:
            next(steps)
    except StopIteration as result:
        return result.value
    finally:
        steps.close()
