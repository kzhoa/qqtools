"""Home-first placement policy helpers."""

from __future__ import annotations

from typing import Any


def offer_due(task: Any) -> bool:
    """Return whether a home agent should evaluate a persisted elapsed-offer proof."""
    return bool(
        task.placement_runtime.get("offer_eligible_at")
        and task.placement_runtime.get("offer_clock_evidence")
        and task.placement_runtime["queue_scope"] == "home"
    )
