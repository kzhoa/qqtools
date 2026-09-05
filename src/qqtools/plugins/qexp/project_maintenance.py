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
    iter_flat_deadline_paths,
    iter_due_deadline_paths,
    migrate_legacy_deadline_indexes,
)
from .runtime.claims import reconcile_claim_archives
from .runtime.paths import attempt_path, shared_paths
from .runtime.placement import offer_due
from .runtime.records import AttemptRecord
from .runtime.reservations import (
    ReservationIdentity,
    active_reservations,
    release_if_matches,
    retag_if_matches,
)
from .runtime.store import iter_json, read_json
from .runtime.tasks import load_task
from .runtime.work_budget import diagnostic_span
from .runtime.active_operations import migrate_legacy_active_operations


def maintain_project(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
    should_reconcile_reservations: bool = True,
) -> None:
    """Converge one project's durable state before its next dispatch attempt."""
    with diagnostic_span("maintain_project"):
        reservation_root = reservation_runtime_root or cfg.runtime_root
        if project_id is None and reservation_root != cfg.runtime_root:
            raise ValueError("a shared reservation runtime requires a project_id.")
        flush_local_events(cfg)
        migrate_legacy_active_operations(cfg)
        migrate_legacy_deadline_indexes(cfg)
        reconcile_claim_archives(cfg)
        if should_reconcile_reservations:
            reconcile_project_reservations(
                cfg,
                reservation_runtime_root=reservation_root,
                project_id=project_id,
            )
        reconcile_group_cancel_operations(cfg, include_legacy=False)
        reconcile_cleanup_operations(
            cfg,
            reservation_runtime_root=reservation_root,
            include_legacy=False,
        )
        reconcile_availability_operations(cfg, include_legacy=False)
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
        reconcile_reservation(cfg, reservation, reservation_runtime_root=reservation_root)


def reconcile_reservation(
    cfg: RootConfig,
    reservation: dict[str, object],
    *,
    reservation_runtime_root: Path,
) -> str:
    """Reconcile one snapshotted reservation through an identity-fenced mutation."""
    try:
        identity = ReservationIdentity.from_record(reservation)
    except ValueError:
        return "isolated"
    try:
        task = load_task(cfg, identity.task_id)
    except FileNotFoundError:
        is_released = release_if_matches(
            reservation_runtime_root,
            identity,
            "task_missing",
        )
        return "released" if is_released else "changed"
    claim = task.claim_control.get("active_claim") or {}
    if (
        claim.get("reservation_id") == identity.reservation_id
        and claim.get("attempt_id") == identity.attempt_id
        and claim.get("fencing_token") == identity.fencing_token
    ):
        return "retained"
    if task.state["projection"] == "blocked":
        return "isolated"
    number = task.attempt_control.get("current_attempt_number")
    if isinstance(number, int):
        attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
        if attempt_file.exists():
            attempt = AttemptRecord.from_dict(read_json(attempt_file))
            if (
                claim.get("attempt_id") == attempt.attempt_id
                and identity.attempt_id == attempt.attempt_id
                and claim.get("fencing_token") == attempt.current_fencing_token
                and retag_if_matches(
                    reservation_runtime_root,
                    identity,
                    attempt.attempt_id,
                    attempt.current_fencing_token,
                )
            ):
                return "retagged"
    is_released = release_if_matches(
        reservation_runtime_root,
        identity,
        "claim_missing",
    )
    return "released" if is_released else "changed"


def offer_due_tasks(cfg: RootConfig) -> None:
    """Move elapsed home-only work into its configured shared queue."""
    with diagnostic_span("offer_due_tasks"):
        migrate_legacy_deadline_indexes(cfg)
        for path in iter_flat_deadline_paths(cfg):
            task_id = path.stem
            try:
                task = load_task(cfg, task_id)
            except FileNotFoundError:
                remove_deadline_index(cfg, task_id)
                continue
            remove_deadline_index(cfg, task_id)
            sync_deadline_index(cfg, task)
        for path in iter_due_deadline_paths(cfg):
            task_id = path.stem
            try:
                task = load_task(cfg, task_id)
            except FileNotFoundError:
                remove_deadline_index(cfg, task_id)
                continue
            sync_deadline_index(cfg, task)
            if task.state["projection"] != "queued" or task.claim_control.get("active_claim"):
                continue
            if task.placement_policy["home_machine"] != cfg.machine_name or not offer_due(task):
                continue
            if not elapsed_offer_is_proven(cfg, task):
                continue
            try:
                offer(cfg, task.task_id, reason="elapsed")
            except (ValueError, FileNotFoundError):
                continue
