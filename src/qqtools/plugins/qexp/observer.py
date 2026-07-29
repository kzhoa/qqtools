"""Truth-derived qexp projections."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config_types import RootConfig
from .runtime.paths import group_path, machine_path, shared_paths, task_path
from .runtime.records import TaskRecord
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
    return result


def list_groups(cfg: RootConfig) -> list[dict[str, Any]]:
    return [read_json(path) for path in iter_json(shared_paths(cfg.shared_root)["groups"])]


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
