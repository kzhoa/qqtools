"""Apply individually confirmed Group changes after Project takeover."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ..commands.group import add_confirmed_primary_worker
from ..config_types import RootConfig
from ..runtime.paths import machine_project_paths
from ..runtime.takeover_registration import TakeoverRegistration, takeover_successor_guard
from .takeover_plan import TakeoverGroupApproval
from .takeover_task_preview import observe_takeover_group_selection


def apply_takeover_group(
    transition: TakeoverRegistration,
    approval: TakeoverGroupApproval,
    plan_id: str,
    *,
    before_mutation: Callable[[], None],
) -> Literal["applied", "stale"]:
    """Apply a Group approval for the observed successor under a current mutation fence."""
    machine = transition.target.machine_name
    if approval.machine_name != machine:
        raise ValueError("Project takeover Group approval names a different target machine.")
    if not callable(before_mutation):
        raise ValueError("Project takeover Group changes require a current mutation fence.")
    root = transition.shared_root
    partition = machine_project_paths(transition.successor_runtime_root, transition.project_id)["root"]
    cfg = RootConfig.from_canonical_paths(root, root.parent, machine, partition)

    def approval_is_current() -> bool:
        try:
            selections = observe_takeover_group_selection(cfg, (approval.group_name,))
        except ValueError:
            return False
        (selection,) = selections
        return TakeoverGroupApproval.from_dict(selection) == approval

    with takeover_successor_guard(transition, before_mutation=before_mutation) as owns_registration:
        if not owns_registration:
            return "stale"
        return add_confirmed_primary_worker(
            cfg, approval.group_name, machine, plan_id, approval_is_current=approval_is_current
        )
