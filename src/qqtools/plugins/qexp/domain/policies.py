"""Pure scheduling and admission policies.

These functions operate on already loaded records and have no persistence or
runtime dependencies, so they can be imported by orchestration layers safely.
"""

from __future__ import annotations

from typing import Any

from ..runtime.records import TaskRecord, normalize_group_record


def group_allows(group: dict[str, Any], task: TaskRecord, machine: str) -> bool:
    """Return whether *machine* may receive *task* from a group."""
    normalize_group_record(group)
    record = group["group"]
    if record["dispatch_state"] != "active":
        return False
    worker = record["worker_set"].get(machine)
    if not worker or worker["state"] != "active":
        return False
    if task.placement_runtime["queue_scope"] == "home":
        return task.placement_policy["home_machine"] == machine
    fallback = task.placement_policy["fallback_constraint"]
    return fallback == "group" or machine in fallback


def task_machine_matches(task: TaskRecord, machine: str) -> bool:
    """Return whether placement policy permits this machine for a task."""
    if task.placement_runtime["queue_scope"] == "home":
        return task.placement_policy["home_machine"] == machine
    fallback = task.placement_policy["fallback_constraint"]
    return fallback == "group" or machine in fallback
