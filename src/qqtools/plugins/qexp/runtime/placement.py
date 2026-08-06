"""Home-first placement policy helpers."""
from __future__ import annotations

from typing import Any


def is_machine_eligible(task: Any, machine_name: str, *, now: datetime | None = None) -> bool:
    if task.placement_policy["sharing_mode"] == "private":
        return task.placement_policy["home_machine"] == machine_name
    if task.placement_runtime["queue_scope"] == "home":
        return task.placement_policy["home_machine"] == machine_name
    fallback = task.placement_policy["fallback_constraint"]
    return fallback == "group" or machine_name in fallback


def offer_due(task: Any) -> bool:
    """Return whether a home agent should evaluate a persisted elapsed-offer proof."""
    return bool(task.placement_runtime.get("offer_eligible_at")
                and task.placement_runtime.get("offer_clock_evidence")
                and task.placement_runtime["queue_scope"] == "home")
