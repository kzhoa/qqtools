"""Task claim and fencing transitions."""
from __future__ import annotations

from typing import Any

from ..config_types import RootConfig
from .locks import task_lock
from .records import utc_now
from .tasks import load_task, save_task


def release_claim(cfg: RootConfig, task_id: str, fencing_token: int, reason: str) -> bool:
    with task_lock(cfg.shared_root, task_id):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("fencing_token") != fencing_token:
            return False
        task.claim_control["active_claim"] = None
        if task.state["projection"] == "running":
            task.state.update({"projection": "blocked", "reason": reason})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True


def renew_lease(cfg: RootConfig, task_id: str, fencing_token: int, expires_at: str) -> bool:
    with task_lock(cfg.shared_root, task_id):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("fencing_token") != fencing_token:
            return False
        claim["lease_expires_at"] = expires_at
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
        return True
