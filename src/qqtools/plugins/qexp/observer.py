"""Truth-derived qexp projections."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .runtime.paths import group_path, machine_path, shared_paths, submission_path, task_path
from .runtime.records import TaskRecord, normalize_group_record
from .runtime.reservations import reservation_snapshot
from .runtime.store import iter_json, read_json
from .runtime.tasks import load_task


def _task_view(task: TaskRecord) -> dict[str, Any]:
    claim = task.claim_control.get("active_claim") or {}
    return {"task_id": task.task_id, "name": task.name, "group": task.group_name,
            "phase": task.state["projection"], "reason": task.state.get("reason"),
            "gpus": task.spec.requested_gpus, "home_machine": task.placement_policy["home_machine"],
            "queue_scope": task.placement_runtime["queue_scope"], "current_attempt_id": task.attempt_control["current_attempt_id"],
            "claim_machine": claim.get("machine_name")}


def list_tasks(cfg: RootConfig, *, phase: str | None = None, group: str | None = None,
               limit: int = 50) -> list[dict[str, Any]]:
    result = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        view = _task_view(task)
        if phase and view["phase"] != phase:
            continue
        if group and view["group"] != group:
            continue
        result.append(view)
    return result[:limit]


def inspect_task(cfg: RootConfig, task_id: str) -> dict[str, Any]:
    task = load_task(cfg, task_id)
    result = task.to_dict()
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task_id
    result["attempts"] = [read_json(path) for path in iter_json(attempts_dir)]
    operation_id = task.submission_operation_id
    if operation_id:
        operation_path = submission_path(cfg.shared_root, operation_id)
        if operation_path.exists():
            submission = read_json(operation_path).get("submission", {})
            result["submission"] = {
                "operation_id": operation_id,
                "original_submitting_machine": submission.get("original_submitting_machine"),
            }
    return result


def list_groups(cfg: RootConfig) -> list[dict[str, Any]]:
    groups = []
    for path in iter_json(shared_paths(cfg.shared_root)["groups"]):
        group = read_json(path)
        normalize_group_record(group)
        groups.append(group)
    return groups


def list_group_machines(
    cfg: RootConfig, name: str, *, reservation_runtime_root: Path | None = None
) -> dict[str, Any]:
    """Return normalized Worker roles, usage, limits, and machine observations."""
    group = read_json(group_path(cfg.shared_root, name))
    normalize_group_record(group)
    machines = []
    for machine, worker in sorted(group["group"]["worker_set"].items()):
        summary_path = (
            shared_paths(cfg.shared_root)["machines"] / machine / "state" / "summary.json"
        )
        reservations: list[dict[str, Any]] = []
        agent_state = "unknown"
        if summary_path.exists():
            try:
                summary = read_json(summary_path).get("summary", {})
                reservations = [
                    item for item in summary.get("machine_reservations", [])
                    if isinstance(item, dict)
                ]
                agent_state = summary.get("agent_state", "unknown")
            except (OSError, KeyError, TypeError, ValueError):
                agent_state = "unknown"
        if machine == cfg.machine_name and reservation_runtime_root is not None:
            try:
                reservations = list(reservation_snapshot(reservation_runtime_root).reservations)
                reservations = [
                    item for item in reservations
                    if item.get("shared_root") in {None, str(cfg.shared_root)}
                ]
            except (OSError, KeyError, TypeError, ValueError):
                pass
        usage = sum(
            len(item.get("gpu_ids", []))
            for item in reservations
            if item.get("group_name") == name
        )
        limit = worker["borrow_limit_gpus"]
        state = worker["state"]
        if worker["scheduling_role"] == "borrow":
            if limit is not None and usage > limit:
                state = "over_limit"
            elif limit is not None and usage >= limit:
                state = "full"
        machines.append({
            "machine_name": machine,
            "scheduling_role": worker["scheduling_role"],
            "gpu_usage": usage,
            "borrow_limit_gpus": limit,
            "state": state,
            "agent": "registered" if agent_state in {"active", "idle"} else agent_state,
        })
    return {"group": name, "machines": machines}


def list_machines(cfg: RootConfig) -> list[dict[str, Any]]:
    machines = shared_paths(cfg.shared_root)["machines"]
    return [_machine_view(cfg, path) for path in sorted(machines.glob("*/machine.json"))]


def _machine_view(cfg: RootConfig, machine_path_value: Any) -> dict[str, Any]:
    record = read_json(machine_path_value)
    machine_name = record.get("machine", {}).get("machine_name")
    if not isinstance(machine_name, str):
        return record
    state_dir = shared_paths(cfg.shared_root)["machines"] / machine_name / "state"
    state = {name: read_json(path) for name in ("agent", "gpu", "summary")
             if (path := state_dir / f"{name}.json").exists()}
    agent = state.get("agent", {})
    heartbeat = agent.get("heartbeat_at") if isinstance(agent, dict) else None
    interval = agent.get("heartbeat_interval_seconds") if isinstance(agent, dict) else None
    if isinstance(heartbeat, str) and isinstance(interval, (int, float)):
        try:
            elapsed = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))).total_seconds()
            state["freshness"] = "stale" if elapsed > interval * 3 else "fresh"
        except ValueError:
            state["freshness"] = "unknown"
    else:
        state["freshness"] = "unknown"
    record["state"] = state
    return record


def top_view(cfg: RootConfig, *, all_machines: bool = False) -> dict[str, Any]:
    tasks = list_tasks(cfg, limit=10**9)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["phase"]] = counts.get(task["phase"], 0) + 1
    return {"counts": counts, "tasks": tasks if all_machines else [t for t in tasks if t["home_machine"] == cfg.machine_name],
            "machines": list_machines(cfg)}
