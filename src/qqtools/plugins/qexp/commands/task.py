"""Task command workflows for qexp."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Callable

import yaml

from ..config_types import RootConfig
from ..layout import (ensure_machine_layout, ensure_shared_layout,
                      validate_root_contract)
from ..runtime.locks import task_lock
from ..runtime.paths import attempt_path, group_path, shared_paths
from ..runtime.records import AttemptRecord, TaskRecord, utc_now, validate_group_name
from ..runtime.store import read_json
from ..runtime.submission import submit_specs
from ..runtime.tasks import load_task, save_task
from ..lease import (ClockObservation, clock_capability, new_timed_offer_proof,
                     persist_clock_observation, timed_offer_deadline_upper)


def is_cleanup_blocked(task: TaskRecord) -> bool:
    return bool(task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"))


def has_cleanup_operation(cfg: RootConfig, task_id: str) -> bool:
    return (shared_paths(cfg.shared_root)["cleanup"] / f"{task_id}.json").exists()


def reject_cleanup_blocked(cfg: RootConfig, task: TaskRecord, action: str) -> None:
    if is_cleanup_blocked(task) or has_cleanup_operation(cfg, task.task_id):
        raise ValueError(f"Task {task.task_id!r} is being cleaned and cannot be {action}.")


def submit(cfg: RootConfig, command: list[str], requested_gpus: int = 1, task_id: str | None = None,
           name: str | None = None, group: str | None = None, working_dir: str | Path | None = None,
           sharing_mode: str = "private", fallback_machines: str | list[str] = "group",
           offer_after_seconds: int | None = None, idempotency_key: str | None = None) -> TaskRecord:
    validate_root_contract(cfg)
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    items = [{"task_id": task_id, "name": name, "command": list(command),
              "requested_gpus": requested_gpus, "working_directory": str(Path(working_dir or Path.cwd()).resolve()),
              "sharing_mode": sharing_mode, "fallback_machines": fallback_machines,
              "offer_after_seconds": offer_after_seconds}]
    return submit_specs(cfg, items, group_name=validate_group_name(group), idempotency_key=idempotency_key)[0]


def batch_submit(cfg: RootConfig, manifest_path: Path, *, group: str | None = None,
                 idempotency_key: str | None = None,
                 on_prepared: Callable[[str, str], None] | None = None) -> list[TaskRecord]:
    validate_root_contract(cfg)
    raw = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    if "name" in raw.get("group", {}):
        raise ValueError("manifest group.name is invalid; pass --group explicitly.")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest must contain a non-empty tasks list.")
    defaults = raw.get("defaults", {}) or {}
    placement = defaults.get("placement", {}) or {}
    sharing = placement.get("sharing", {}) or {}
    normalized: list[dict[str, Any]] = []
    for entry in tasks:
        if not isinstance(entry, dict) or not entry.get("command"):
            raise ValueError("each manifest task requires command.")
        normalized.append({"task_id": entry.get("task_id"), "name": entry.get("name"),
            "command": list(entry["command"]), "requested_gpus": entry.get("requested_gpus", defaults.get("requested_gpus", 1)),
            "working_directory": entry.get("working_directory", defaults.get("working_directory", str(Path.cwd()))),
            "sharing_mode": entry.get("sharing_mode", sharing.get("mode", "private")),
            "fallback_machines": entry.get("fallback_machines", sharing.get("fallback_machines", "group")),
            "offer_after_seconds": entry.get("offer_after_seconds", (sharing.get("offer") or {}).get("after_seconds"))})
    workers = (raw.get("workers") or raw.get("group", {}).get("workers") or
               (raw.get("defaults", {}).get("placement", {}) or {}).get("workers"))
    return submit_specs(cfg, normalized, group_name=validate_group_name(group),
                        idempotency_key=idempotency_key, kind="bulk",
                        worker_set=list(workers or []), on_prepared=on_prepared)


def cancel(cfg: RootConfig, task_id: str, *, terminate_running: bool = True) -> TaskRecord:
    from ..scheduler import cancel_task
    return cancel_task(cfg, task_id, terminate_running=terminate_running)


def retry(
        cfg: RootConfig, task_id: str, *, acknowledge_duplicate_risk: bool = False) -> TaskRecord:
    """Queue the next Attempt, optionally accepting unresolved orphan duplication risk."""
    from ..scheduler import authority_locks
    initial = load_task(cfg, task_id)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        reject_cleanup_blocked(cfg, task, "retried")
        if task.claim_control.get("active_claim"):
            raise ValueError("a Task with an active claim cannot be retried.")
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            raise ValueError("Task has no current Attempt to retry.")
        current_path = attempt_path(cfg.shared_root, task_id, number)
        current = AttemptRecord.from_dict(read_json(current_path))
        if acknowledge_duplicate_risk:
            if task.state["projection"] != "blocked" or current.phase != "orphaned":
                raise ValueError(
                    "--acknowledge-duplicate-risk requires a blocked Task with an "
                    "orphaned current Attempt."
                )
            acknowledged_at = utc_now()
            from ..events import write_event
            write_event(
                cfg,
                "duplicate_risk_acknowledged",
                task_id=task_id,
                details={"attempt_id": current.attempt_id,
                         "fencing_token": current.current_fencing_token,
                         "reason": "operator_acknowledged_duplicate_risk"},
            )
            task.claim_control["fencing_epoch"] += 1
            task.control.update({"duplicate_risk_acknowledged_at": acknowledged_at,
                                 "duplicate_risk_acknowledged_by": cfg.machine_name,
                                 "duplicate_risk_attempt_id": current.attempt_id})
            task.state = {"projection": "queued", "reason": "duplicate_risk_acknowledged"}
        else:
            if task.state["projection"] != "failed" or current.phase != "failed":
                raise ValueError(
                    "only a failed Task with a failed current Attempt can be retried."
                )
            task.state = {"projection": "queued", "reason": None}
        task.control.update({"cancellation_requested_at": None, "cancellation_operation_id": None,
                             "terminate_running": False, "requested_by": None,
                             "termination_acknowledged_at": None, "termination_result": None})
        task.placement_runtime.update({"queue_scope": "home", "queued_home_at": utc_now(), "offered_at": None,
                                       "offer_reason": None, "offered_by": None})
        task.attempt_control["current_attempt_id"] = None
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return task


def _clock_evidence(cfg: RootConfig) -> tuple[ClockObservation, datetime, float]:
    capability = clock_capability(cfg)
    if not capability.is_healthy or capability.observation is None:
        raise ValueError("timed sharing requires a healthy clock capability; use immediate share instead.")
    persist_clock_observation(cfg, capability.observation)
    # Taking monotonic time after wall time makes the evaluator lower bound conservative.
    return capability.observation, datetime.now(timezone.utc), time.monotonic()


def _elapsed_offer_is_proven(cfg: RootConfig, task: TaskRecord) -> bool:
    proof = task.placement_runtime.get("offer_clock_evidence")
    deadline = task.placement_runtime.get("offer_eligible_at")
    if not isinstance(proof, dict) or not isinstance(deadline, str):
        return False
    try:
        observation, now, monotonic_now = _clock_evidence(cfg)
    except ValueError:
        return False
    evaluator_lower = now - timedelta(seconds=observation.bound_at(monotonic_now))
    try:
        deadline_upper = timed_offer_deadline_upper(deadline, proof)
    except (TypeError, ValueError, OverflowError):
        return False
    return evaluator_lower >= deadline_upper


def _transition(cfg: RootConfig, task_id: str, *, action: str, reason: str,
                helper_machines: list[str] | None = None, after_seconds: int | None = None) -> TaskRecord:
    from ..scheduler import authority_locks

    initial = load_task(cfg, task_id)
    with authority_locks(cfg, initial):
        task = load_task(cfg, task_id)
        reject_cleanup_blocked(cfg, task, action)
        if action == "offer" and reason == "elapsed":
            if task.placement_policy["home_machine"] != cfg.machine_name:
                return task
            if not _elapsed_offer_is_proven(cfg, task):
                return task
        if (action == "offer" and task.state["projection"] == "queued"
                and task.placement_runtime["queue_scope"] == "shared"
                and task.placement_policy["sharing_mode"] == "spillover"):
            return task
        if task.state["projection"] != "queued" or task.claim_control.get("active_claim"):
            raise ValueError("placement can only change while a Task is queued and unclaimed.")
        if task.control.get("cancellation_requested_at"):
            raise ValueError("cancelled Tasks cannot change placement.")
        if action == "keep_local":
            if task.group_name is None and task.placement_policy["sharing_mode"] == "private":
                return task
            task.placement_policy.update({"sharing_mode": "private", "fallback_constraint": "group",
                                         "offer_after_seconds": None})
            task.placement_runtime.update({"queue_scope": "home", "queued_home_at": utc_now(),
                                           "offer_eligible_at": None, "offer_clock_evidence": None,
                                           "offered_at": None, "offer_reason": None, "offered_by": cfg.machine_name})
        else:
            if action == "share" and task.group_name is None:
                raise ValueError(
                    f"Task {task_id!r} is local-only because it does not belong to a Group. "
                    "Submit the work to a Group to let other machines help."
                )
            if action == "offer" and task.placement_policy["sharing_mode"] != "spillover":
                raise ValueError("private Tasks cannot be offered to shared workers; use task share.")
            fallback: str | list[str] = task.placement_policy["fallback_constraint"]
            if helper_machines is not None:
                if len(set(helper_machines)) != len(helper_machines):
                    raise ValueError("shared helper machines must be unique.")
                if task.placement_policy["home_machine"] in helper_machines:
                    raise ValueError("the home machine is already eligible and cannot be a helper.")
                if task.group_name:
                    workers = read_json(group_path(cfg.shared_root, task.group_name))["group"]["worker_set"]
                    invalid = [machine for machine in helper_machines
                               if workers.get(machine, {}).get("state") != "active"]
                    if invalid:
                        raise ValueError(f"shared helpers are not active Group workers: {invalid}")
                fallback = helper_machines or "group"
            if action == "share":
                task.placement_policy["sharing_mode"] = "spillover"
                task.placement_policy["fallback_constraint"] = fallback
            if after_seconds is not None:
                if task.placement_runtime["queue_scope"] != "home":
                    raise ValueError("share --after requires a home-queued Task; use keep-local first.")
                observation, wall_now, monotonic_now = _clock_evidence(cfg)
                deadline, proof = new_timed_offer_proof(
                    observation, after_seconds, wall_now=wall_now, monotonic_now=monotonic_now
                )
                task.placement_policy["offer_after_seconds"] = after_seconds
                task.placement_runtime.update({
                    "queue_scope": "home", "queued_home_at": utc_now(),
                    "offer_eligible_at": deadline, "offer_clock_evidence": proof, "offered_at": None,
                    "offer_reason": None, "offered_by": cfg.machine_name,
                })
            else:
                task.placement_runtime.update({"queue_scope": "shared", "offered_at": utc_now(),
                                               "offer_reason": reason, "offered_by": cfg.machine_name})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return task


def share(cfg: RootConfig, task_id: str, *, after_seconds: int | None = None,
          helper_machines: list[str] | None = None) -> TaskRecord:
    if after_seconds is not None and after_seconds < 0:
        raise ValueError("share --after must be non-negative.")
    return _transition(cfg, task_id, action="share", reason="manual", helper_machines=helper_machines,
                       after_seconds=after_seconds)


def keep_local(cfg: RootConfig, task_id: str) -> TaskRecord:
    return _transition(cfg, task_id, action="keep_local", reason="manual")


def offer(cfg: RootConfig, task_id: str, *, reason: str = "manual") -> TaskRecord:
    return _transition(cfg, task_id, action="offer", reason=reason)
