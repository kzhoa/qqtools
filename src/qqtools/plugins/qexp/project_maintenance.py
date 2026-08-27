"""Project-scoped durable maintenance shared by qexp agents."""
from __future__ import annotations

from pathlib import Path

from .commands.cleanup import reconcile_cleanup_operations
from .commands.group import reconcile_group_cancel_operations
from .commands.task import offer
from .config_types import RootConfig
from .events import flush_local_events
from .runtime.availability import (
    elapsed_offer_is_proven,
    reconcile_availability_operations,
    remove_deadline_index,
    sync_deadline_index,
)
from .runtime.claims import reconcile_claim_archives
from .runtime.paths import attempt_path, shared_paths
from .runtime.placement import offer_due
from .runtime.records import AttemptRecord
from .runtime.reservations import active_reservations, release, retag
from .runtime.store import iter_json, read_json
from .runtime.tasks import load_task


def maintain_project(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
) -> None:
    """Converge one project's durable state before its next dispatch attempt."""
    reservation_root = reservation_runtime_root or cfg.runtime_root
    if project_id is None and reservation_root != cfg.runtime_root:
        raise ValueError("a shared reservation runtime requires a project_id.")
    flush_local_events(cfg)
    reconcile_claim_archives(cfg)
    reconcile_project_reservations(
        cfg,
        reservation_runtime_root=reservation_root,
        project_id=project_id,
    )
    reconcile_group_cancel_operations(cfg)
    reconcile_cleanup_operations(cfg, reservation_runtime_root=reservation_root)
    reconcile_availability_operations(cfg)
    offer_due_tasks(cfg)


def reconcile_project_reservations(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
) -> None:
    """Release only this project's stale reservations from its resource backend."""
    reservation_root = reservation_runtime_root or cfg.runtime_root
    if project_id is None and reservation_root != cfg.runtime_root:
        raise ValueError("a shared reservation runtime requires a project_id.")
    for reservation in active_reservations(reservation_root):
        if project_id is not None and reservation.get("project_id") != project_id:
            continue
        try:
            task = load_task(cfg, reservation["task_id"])
        except FileNotFoundError:
            release(reservation_root, reservation["reservation_id"], "task_missing")
            continue
        claim = task.claim_control.get("active_claim") or {}
        if (
            claim.get("reservation_id") != reservation["reservation_id"]
            or claim.get("fencing_token") != reservation.get("fencing_token")
        ):
            if task.state["projection"] == "blocked":
                continue
            number = task.attempt_control.get("current_attempt_number")
            if number is not None:
                attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
                if attempt_file.exists():
                    attempt = AttemptRecord.from_dict(read_json(attempt_file))
                    if (
                        claim.get("attempt_id") == attempt.attempt_id
                        and claim.get("fencing_token") == attempt.current_fencing_token
                        and retag(
                            reservation_root,
                            reservation["reservation_id"],
                            attempt.attempt_id,
                            attempt.current_fencing_token,
                        )
                    ):
                        continue
            release(reservation_root, reservation["reservation_id"], "claim_missing")


def offer_due_tasks(cfg: RootConfig) -> None:
    """Move elapsed home-only work into its configured shared queue."""
    seen: set[str] = set()
    candidate_paths = list(iter_json(shared_paths(cfg.shared_root)["offer_deadlines"]))
    candidate_paths.extend(iter_json(shared_paths(cfg.shared_root)["tasks"]))
    for path in candidate_paths:
        task_id = path.stem
        if task_id in seen:
            continue
        seen.add(task_id)
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            if path.parent == shared_paths(cfg.shared_root)["offer_deadlines"]:
                remove_deadline_index(cfg, task_id)
            continue
        sync_deadline_index(cfg, task)
        if task.placement_policy["home_machine"] != cfg.machine_name or not offer_due(task):
            continue
        if not elapsed_offer_is_proven(cfg, task):
            continue
        try:
            offer(cfg, task.task_id, reason="elapsed")
        except (ValueError, FileNotFoundError):
            continue
