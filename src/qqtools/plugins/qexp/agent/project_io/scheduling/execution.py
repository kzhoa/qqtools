"""Worker-side Project mutations for due offers and ready-index builds."""

from collections.abc import Callable
from typing import Any

from ....config_types import RootConfig
from ....runtime.ready import advance_ready_index_build
from ....runtime.store import fenced_mutations


def run(kind: str, cfg: RootConfig, before_shared_mutation: Callable[[], None]) -> dict[str, Any]:
    if kind == "scheduler_due_offer":
        from ....project_maintenance import advance_due_offer

        progress = advance_due_offer(cfg, before_shared_mutation=before_shared_mutation)
        return {"outcome": progress.outcome, "reason": progress.reason, "task_id": progress.task_id}
    if kind != "scheduler_ready_index_build":
        raise ValueError(f"Unknown scheduling operation {kind!r}.")
    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        status = advance_ready_index_build(cfg, max_tasks=64, bounded_initialization=True)
    state = status.get("state")
    revision = status.get("revision")
    build = status.get("build") if state == "building" else None
    build_id = build.get("build_id") if isinstance(build, dict) else None
    phase = build.get("phase") if isinstance(build, dict) else None
    return {"state": state, "revision": revision, "build_id": build_id, "phase": phase}
