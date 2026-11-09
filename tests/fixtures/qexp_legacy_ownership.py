"""Persist supported historical ownership evidence independently of current writers."""

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.lease import ClockObservation
from qqtools.plugins.qexp.runtime.claims import archive_claim
from qqtools.plugins.qexp.runtime.group_discovery.changes import record_task_change
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord, utc_now
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task


def persist_legacy_orphan(cfg, task_id: str) -> AttemptRecord:
    """Build the bounded-lease orphan image produced by a supported old writer."""
    task = load_task(cfg, task_id)
    claim = dict(task.claim_control["active_claim"])
    path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
    attempt = AttemptRecord.from_dict(read_json(path))
    history = claim.pop("ownership_transition", {}).get("source_active_lease_evidence", {})
    if isinstance(history.get("claim"), dict):
        claim.update(history["claim"])
    if isinstance(history.get("attempt"), dict):
        attempt.lease = history["attempt"]
    claim["authority_mode"] = attempt.authority_mode = "bounded_lease"
    attempt.phase = "orphaned"
    attempt.result.update(exit_code=None, signal=None, category=None, reason=None)
    attempt.timestamps["orphaned_at"] = utc_now()
    with record_task_change(
        cfg,
        task,
        "claim_loss",
        details={
            "expected_attempt_id": attempt.attempt_id,
            "expected_fencing_token": attempt.current_fencing_token,
            "reserved_generation": None,
        },
    ):
        atomic_replace(path, attempt.to_dict())
        archive_claim(cfg, task_id, claim, "lease_expired")
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update(projection="blocked", reason="orphaned_attempt_requires_recovery")
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
    return attempt


def persist_legacy_termination_decision(
    cfg: RootConfig,
    *,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    process: dict[str, Any],
    authority_outcome: str,
    reason: str,
    decision_id: str | None = None,
) -> dict[str, Any]:
    """QQTOOLS-COMPAT-0022: persist an old decision independently of current writers."""
    from qqtools.plugins.qexp.runtime.records import new_id
    from qqtools.plugins.qexp.runtime.termination import decision_path

    decision_id = decision_id or new_id()
    decision = {
        "decision_id": decision_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "decision_token": fencing_token,
        "authority_outcome": authority_outcome,
        "reason": reason,
        "state": "pending",
        "shared_commitment": "pending",
        "shared_reconciliation": "pending",
        "process_group_id": process.get("process_group_id"),
        "process_group_start_time_ticks": process.get("process_group_start_time_ticks"),
        "signal_attempts": [],
        "observed_exit_code": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_replace(decision_path(cfg, attempt_id, decision_id), {"termination_decision": decision})
    return decision


def persist_legacy_renewal(
    cfg: RootConfig,
    task_id: str,
    request_id: str,
    observation: ClockObservation,
    *,
    commit_task: bool = True,
) -> dict[str, Any]:
    """QQTOOLS-COMPAT-0021: persist a released writer's interrupted renewal."""
    plan = {
        "renewed_at": utc_now(),
        "lease_expires_at": (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat(),
        "renew_after_seconds": 10.0,
    }
    atomic_replace(
        shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{request_id}.json",
        {
            "clock_observation": {**observation.to_dict(), "observation_id": request_id},
            "renewal_plan": plan,
        },
    )
    if commit_task:
        task = load_task(cfg, task_id)
        task.claim_control["active_claim"].update(
            clock_observation_id=request_id,
            clock_provider=observation.provider,
            clock_error_bound_seconds=observation.bound_at(time.monotonic()),
            lease_expires_at=plan["lease_expires_at"],
        )
        task.meta["revision"] += 1
        task.meta["updated_at"] = plan["renewed_at"]
        atomic_replace(shared_paths(cfg.shared_root)["tasks"] / f"{task_id}.json", task.to_dict())
    return plan
