"""Fenced recovery CAS for locally verified orphaned Attempts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Generator, Mapping

from ..config_types import RootConfig
from ..lease import clock_capability, lease_expiry, load_lease_policy, persist_clock_observation
from ..scheduler import authority_locks
from .durable_ownership import transition_attempt_ownership
from .group_discovery.changes import record_task_change
from .group_namespace import read_group
from .paths import attempt_path, group_path, local_paths
from .process_evidence import inspect_group_identity, inspect_wrapper_identity
from .records import AttemptRecord, normalize_group_record, utc_now
from .resources.reservations import retag
from .store import atomic_replace, read_json
from .tasks import load_task, save_task
from .termination import attempt_control_lock, recovery_check_steps


def _restore_durable_ownership(
    cfg: RootConfig, task: Any, attempt: AttemptRecord, *, mutation_fence: Callable[[], None] | None = None
) -> None:
    """Convert a recovered exact legacy launch through the Task-first receipt."""
    claim = task.claim_control.get("active_claim") or {}
    launch_id = attempt.authorization.get("launch_id")
    authorized_at = attempt.timestamps.get("launch_authorized_at")
    if not isinstance(launch_id, str) or not launch_id or not isinstance(authorized_at, str):
        # Older incomplete evidence remains readable, but cannot invent a
        # launch identity or reusable durable ownership.
        return
    if claim.get("launch_id") != launch_id:
        return
    transition_attempt_ownership(
        cfg,
        task,
        attempt,
        target_phase="running",
        launch_id=launch_id,
        authorized_at=authorized_at,
        mutation_fence=mutation_fence,
        task_writer=lambda value: save_task(cfg, value, mutation_fence=mutation_fence),
    )


def recover_orphaned_attempt_shared(
    cfg: RootConfig,
    *,
    machine_name: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    expired_token: int,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    binding_signature: list[str] | tuple[str, ...],
    expected_task_revision: int | None,
    expected_attempt_digest: str | None,
    mutation_fence: Callable[[], None],
    replay_only: bool = False,
) -> dict[str, Any]:
    """Recover a live orphan using shared Project truth only.

    This is deliberately separate from :func:`recovery_steps`: the typed
    Project-I/O worker may execute this CAS while the controller retains all
    MachineRuntime reservation and manifest effects for its local phase.
    ``None`` source revisions are permitted only for the first machine-local
    evidence request; the worker binds the exact pair while holding the
    authority lock. Replay accepts only the same target token and identity.
    """

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

    def evidence(
        outcome: str,
        reason: str | None,
        *,
        source: Mapping[str, Any] | None,
        task_revision: int | None,
        attempt_digest: str | None,
        recovered_token: int | None,
        expires_at: str | None,
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reason": reason,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": expired_token,
            "reservation_id": reservation_id,
            "process_identity": dict(process_identity),
            "binding_signature": list(binding_signature),
            "source_revisions": {
                "task": None if source is None else source.get("task"),
                "attempt_digest": None if source is None else source.get("attempt_digest"),
            },
            "committed_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
            "recovered_fencing_token": recovered_token,
            "lease_expires_at": expires_at,
            "authority_granted": False,
            "local_effects": [],
        }

    stable_identity = {
        "machine_name": machine_name,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": expired_token,
        "reservation_id": reservation_id,
        "process_identity": dict(process_identity),
        "binding_signature": list(binding_signature),
    }
    recovery_id = hashlib.sha256(
        json.dumps(stable_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        return evidence(
            "stale",
            "task_missing",
            source={"task": expected_task_revision, "attempt_digest": expected_attempt_digest},
            task_revision=None,
            attempt_digest=None,
            recovered_token=None,
            expires_at=None,
        )
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        task_revision = task.meta.get("revision")
        if type(task_revision) is not int or task_revision < 0:
            raise ValueError("Task revision is malformed.")
        attempt, attempt_digest = read_attempt()
        observed_source = {"task": task_revision, "attempt_digest": attempt_digest}
        if attempt is None:
            return evidence(
                "stale",
                "attempt_missing",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=None,
                recovered_token=None,
                expires_at=None,
            )
        identity_matches = (
            task.task_id == task_id
            and attempt.task_id == task_id
            and attempt.attempt_id == attempt_id
            and attempt.attempt_number == attempt_number
            and attempt.machine_name == machine_name
            and attempt.reservation_id == reservation_id
        )
        process_matches = all(
            attempt.process.get(field) == process_identity.get(field)
            for field in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        )
        if not identity_matches:
            return evidence(
                "stale",
                "recovery_identity_mismatch",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )
        if not process_matches:
            return evidence(
                "stale",
                "process_identity_mismatch",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )

        marker = attempt.authorization.get("project_io_orphan_recovery")
        marker_source = marker.get("source_revisions") if isinstance(marker, dict) else None
        marker_token = marker.get("recovered_fencing_token") if isinstance(marker, dict) else None
        marker_expiry = marker.get("lease_expires_at") if isinstance(marker, dict) else None
        marker_launch_id = marker.get("launch_id") if isinstance(marker, dict) else None
        marker_matches = (
            isinstance(marker, dict)
            and marker.get("recovery_id") == recovery_id
            and marker.get("expired_fencing_token") == expired_token
            and list(marker.get("binding_signature", ())) == list(binding_signature)
            and isinstance(marker_source, dict)
            and set(marker_source) == {"task", "attempt_digest"}
            and type(marker_source.get("task")) is int
            and isinstance(marker_source.get("attempt_digest"), str)
            and type(marker_token) is int
            and marker_token > expired_token
            and isinstance(marker_expiry, str)
            and bool(marker_expiry)
            and ("launch_id" not in marker or marker_launch_id == attempt.authorization.get("launch_id"))
            and attempt.phase == "running"
            and attempt.current_fencing_token == marker_token
            and attempt.timestamps.get("recovered_at") is not None
        )
        claim = task.claim_control.get("active_claim") or {}
        already_recovered = (
            marker_matches
            and task.state.get("projection") == "running"
            and task.attempt_control.get("current_attempt_id") == attempt_id
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("next_attempt_number") == attempt_number + 1
            and task.claim_control.get("fencing_epoch") == marker_token
            and claim.get("attempt_id") == attempt_id
            and claim.get("attempt_number") == attempt_number
            and claim.get("machine_name") == machine_name
            and claim.get("reservation_id") == reservation_id
            and claim.get("fencing_token") == marker_token
            and claim.get("project_io_orphan_recovery_id") == recovery_id
        )
        if already_recovered:
            _restore_durable_ownership(cfg, task, attempt, mutation_fence=mutation_fence)
            _, attempt_digest = read_attempt()
            return evidence(
                "already_recovered",
                None,
                source=marker_source,
                task_revision=task.meta["revision"],
                attempt_digest=attempt_digest,
                recovered_token=marker_token,
                expires_at=marker_expiry,
            )

        exact_blocked_target = (
            task.state.get("projection") == "blocked"
            and not claim
            and task.attempt_control.get("current_attempt_id") is None
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("next_attempt_number") == attempt_number + 1
            and task.claim_control.get("fencing_epoch") == expired_token
        )
        partial = marker_matches and exact_blocked_target
        if not exact_blocked_target:
            return evidence(
                "stale",
                "recovery_target_mismatch",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )
        if replay_only and not partial:
            return evidence(
                "stale",
                "recovery_target_mismatch",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )

        if not partial and (
            task.control.get("terminate_running") is True
            or task.control.get("cancellation_requested_at") is not None
            or attempt.termination.get("requested_at") is not None
            or attempt.termination.get("requested_by_operation_id") is not None
        ):
            return evidence(
                "stale",
                "recovery_termination_blocked",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )
        if not partial and attempt.authority_mode != "bounded_lease":
            return evidence(
                "stale",
                "recovery_target_mismatch",
                source=observed_source,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                recovered_token=None,
                expires_at=None,
            )
        if not partial and task.group_name:
            group = read_group(cfg.shared_root, task.group_name)
            normalize_group_record(group)
            worker = group["group"]["worker_set"].get(machine_name)
            if not worker or worker.get("state") not in {"active", "draining"}:
                return evidence(
                    "stale",
                    "recovery_worker_state_invalid",
                    source=observed_source,
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                    recovered_token=None,
                    expires_at=None,
                )
            if worker.get("state") == "draining" and attempt.authorization.get("worker_state_epoch", -1) >= worker.get(
                "state_epoch", -1
            ):
                return evidence(
                    "stale",
                    "recovery_worker_state_invalid",
                    source=observed_source,
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                    recovered_token=None,
                    expires_at=None,
                )
            for barrier in group["group"].get("cancellation_barriers", []):
                if barrier.get("terminate_running") and (task.group_membership_sequence or 0) <= barrier.get(
                    "membership_high_watermark", -1
                ):
                    return evidence(
                        "stale",
                        "recovery_termination_blocked",
                        source=observed_source,
                        task_revision=task_revision,
                        attempt_digest=attempt_digest,
                        recovered_token=None,
                        expires_at=None,
                    )

        if partial:
            token = marker_token
            expires = marker_expiry
            source = marker_source
            clock_evidence = attempt.lease.get("clock_evidence")
            if not isinstance(clock_evidence, dict):
                raise RuntimeError("Partial orphan recovery has invalid clock evidence.")
        else:
            if expected_task_revision is not None and (
                task_revision != expected_task_revision or attempt_digest != expected_attempt_digest
            ):
                return evidence(
                    "stale",
                    "source_revision_mismatch",
                    source=observed_source,
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                    recovered_token=None,
                    expires_at=None,
                )
            if attempt.phase != "orphaned" or attempt.current_fencing_token != expired_token:
                return evidence(
                    "stale",
                    "attempt_phase_mismatch",
                    source=observed_source,
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                    recovered_token=None,
                    expires_at=None,
                )
            policy = load_lease_policy(cfg)
            capability = clock_capability(cfg, policy)
            if not capability.is_healthy or capability.observation is None:
                return evidence(
                    "stale",
                    "recovery_clock_unhealthy",
                    source=observed_source,
                    task_revision=task_revision,
                    attempt_digest=attempt_digest,
                    recovered_token=None,
                    expires_at=None,
                )
            token = expired_token + 1
            expires = lease_expiry(policy)
            mutation_fence()
            persist_clock_observation(cfg, capability.observation)
            clock_evidence = {
                "clock_error_bound_seconds": capability.observation.bound_at(time.monotonic()),
                "provider": capability.observation.provider,
                "observation_id": capability.observation.observation_id,
            }
            source = observed_source
            attempt.current_fencing_token = token
            if token not in attempt.token_history:
                attempt.token_history.append(token)
            attempt.phase = "running"
            attempt.result.update({"exit_code": None, "signal": None, "category": None, "reason": None})
            attempt.timestamps["finished_at"] = None
            attempt.timestamps["recovered_at"] = utc_now()
            attempt.lease.update({"renewed_at": utc_now(), "expires_at": expires, "clock_evidence": clock_evidence})
            attempt.authorization["project_io_orphan_recovery"] = {
                "recovery_id": recovery_id,
                "expired_fencing_token": expired_token,
                "binding_signature": list(binding_signature),
                "source_revisions": dict(source),
                "recovered_fencing_token": token,
                "lease_expires_at": expires,
                "launch_id": attempt.authorization.get("launch_id"),
            }
            mutation_fence()
            atomic_replace(attempt_path(cfg.shared_root, task_id, attempt_number), attempt.to_dict())
            attempt, attempt_digest = read_attempt()
            if attempt is None:
                raise RuntimeError("Recovered Attempt disappeared after publication.")

        with record_task_change(
            cfg,
            task,
            "recovery",
            details={"expected_attempt_id": attempt_id, "expired_token": expired_token},
            mutation_fence=mutation_fence,
        ):
            active_claim = {
                "claim_id": attempt_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "machine_name": machine_name,
                "reservation_id": reservation_id,
                "queue_origin": task.placement_runtime["queue_scope"],
                "fencing_token": token,
                "claimed_at": utc_now(),
                "authority_mode": "bounded_lease",
                "clock_error_bound_seconds": clock_evidence["clock_error_bound_seconds"],
                "clock_provider": clock_evidence["provider"],
                "clock_observation_id": clock_evidence["observation_id"],
                "lease_expires_at": expires,
                "launch_state": "running",
                "launch_id": attempt.authorization.get("launch_id"),
                "launch_authorized_at": attempt.timestamps.get("launch_authorized_at"),
                "launch_handoff_timeout_seconds": attempt.authorization.get("launch_handoff_timeout_seconds"),
                "group_dispatch_epoch": attempt.authorization.get("group_dispatch_epoch"),
                "group_worker_set_epoch": attempt.authorization.get("group_worker_set_epoch"),
                "project_io_orphan_recovery_id": recovery_id,
            }
            task.claim_control.update({"fencing_epoch": token, "active_claim": active_claim})
            task.state.update({"projection": "running", "reason": "recovered_live_attempt"})
            task.attempt_control["current_attempt_id"] = attempt_id
            task.attempt_control["current_attempt_number"] = attempt_number
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            mutation_fence()
            save_task(cfg, task, mutation_fence=mutation_fence)

        _restore_durable_ownership(cfg, task, attempt, mutation_fence=mutation_fence)
        _, committed_digest = read_attempt()
        return evidence(
            "already_recovered" if partial else "recovered",
            None,
            source=source,
            task_revision=task.meta["revision"],
            attempt_digest=committed_digest,
            recovered_token=token,
            expires_at=expires,
        )


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
                if (local_paths(cfg.runtime_root)["observations"] / f"{attempt_id}.json").exists():
                    return None
                if inspect_group_identity(attempt.process, manifest).state != "alive":
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
                            "launch_id": attempt.authorization.get("launch_id"),
                            "launch_authorized_at": attempt.timestamps.get("launch_authorized_at"),
                            "launch_handoff_timeout_seconds": attempt.authorization.get(
                                "launch_handoff_timeout_seconds"
                            ),
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
                _restore_durable_ownership(cfg, task, attempt)
                manifest = dict(manifest)
                manifest.update(
                    {
                        "fencing_token": token,
                        "recovered_at": utc_now(),
                        "observed_state": "running",
                        "supervisor": ("runner" if inspect_wrapper_identity(manifest).state == "alive" else "agent"),
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
