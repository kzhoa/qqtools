"""Task claim and fencing transitions."""
from __future__ import annotations

from typing import Any

from ..config_types import RootConfig
from .locks import task_lock
from .paths import shared_paths
from .records import utc_now
from .store import CASConflict, create_if_absent, iter_json, read_json
from .tasks import load_task, save_task


def archive_claim(cfg: RootConfig, task_id: str, claim: dict[str, Any], reason: str) -> bool:
    """Persist one immutable terminal record for a fenced claim."""
    token = claim.get("fencing_token")
    if not isinstance(token, int):
        raise ValueError("claim must contain an integer fencing_token before archival.")
    paths = shared_paths(cfg.shared_root)
    path = paths["claim_archive"] / task_id / f"{token}.json"
    pending_path = paths["claim_pending"] / task_id / f"{token}.json"
    record = {"claim_archive": {"task_id": task_id, "attempt_id": claim.get("attempt_id"),
                                 "fencing_token": token, "reason": reason,
                                 "archived_at": utc_now(), "claim": dict(claim)}}
    try:
        create_if_absent(path, record)
    except CASConflict:
        return True
    except OSError:
        try:
            create_if_absent(pending_path, record)
        except (CASConflict, OSError):
            pass
        return False
    return True


def reconcile_claim_archives(cfg: RootConfig, task_id: str | None = None) -> bool:
    """Retry pending claim archives and report whether the requested scope is clear."""
    paths = shared_paths(cfg.shared_root)
    pending_root = paths["claim_pending"]
    task_directories = [pending_root / task_id] if task_id else sorted(pending_root.glob("*"))
    has_pending = False
    for task_directory in task_directories:
        for pending_path in iter_json(task_directory):
            record = read_json(pending_path)
            archive = record.get("claim_archive", {})
            archive_task_id = archive.get("task_id")
            token = archive.get("fencing_token")
            if not isinstance(archive_task_id, str) or not isinstance(token, int):
                has_pending = True
                continue
            path = paths["claim_archive"] / archive_task_id / f"{token}.json"
            try:
                create_if_absent(path, record)
            except CASConflict:
                pass
            except OSError:
                has_pending = True
                continue
            pending_path.unlink(missing_ok=True)
    if task_id:
        return not has_pending and not any(iter_json(pending_root / task_id))
    return not has_pending


def release_claim(cfg: RootConfig, task_id: str, fencing_token: int, reason: str) -> bool:
    with task_lock(cfg.shared_root, task_id):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("fencing_token") != fencing_token:
            return False
        archive_claim(cfg, task_id, claim, reason)
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
