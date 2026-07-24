"""Task command workflows for qexp."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from ..layout import (RootConfig, ensure_machine_layout, ensure_shared_layout,
                      validate_root_contract)
from ..runtime.locks import task_lock
from ..runtime.paths import attempt_path, shared_paths
from ..runtime.records import AttemptRecord, TaskRecord, utc_now, validate_group_name
from ..runtime.store import read_json
from ..runtime.submission import submit_specs
from ..runtime.tasks import load_task, save_task


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


def offer(cfg: RootConfig, task_id: str, *, reason: str = "manual") -> TaskRecord:
    with task_lock(cfg.shared_root, task_id):
        task = load_task(cfg, task_id)
        reject_cleanup_blocked(cfg, task, "offered")
        if task.placement_policy["sharing_mode"] != "spillover":
            raise ValueError("private Tasks cannot be offered to shared workers.")
        if task.state["projection"] != "queued" or task.placement_runtime["queue_scope"] != "home":
            return task
        task.placement_runtime.update({"queue_scope": "shared", "offered_at": utc_now(),
                                       "offer_reason": reason, "offered_by": cfg.machine_name})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return task
