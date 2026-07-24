"""Truth-derived qexp projections."""
from __future__ import annotations

from typing import Any

from .layout import RootConfig
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
    return [read_json(path) for path in sorted(machines.glob("*/machine.json"))]


def top_view(cfg: RootConfig, *, all_machines: bool = False) -> dict[str, Any]:
    tasks = list_tasks(cfg, limit=10**9)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["phase"]] = counts.get(task["phase"], 0) + 1
    return {"counts": counts, "tasks": tasks if all_machines else [t for t in tasks if t["home_machine"] == cfg.machine_name],
            "machines": list_machines(cfg)}
