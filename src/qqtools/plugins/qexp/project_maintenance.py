"""Project-scoped durable maintenance shared by qexp agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .commands.task import offer
from .config_types import RootConfig
from .events import flush_local_events
from .runtime.availability import (
    elapsed_offer_is_proven,
    iter_due_deadline_paths,
    remove_deadline_index,
    sync_deadline_index,
)
from .runtime.paths import attempt_path, shared_paths
from .runtime.placement import offer_due
from .runtime.records import AttemptRecord
from .runtime.resources.reservations import (
    ReservationIdentity,
    active_reservations,
    release_if_matches,
    retag_if_matches,
)
from .runtime.store import fenced_mutations, iter_json, read_json
from .runtime.tasks import load_task
from .runtime.work_budget import diagnostic_span


@dataclass(frozen=True, slots=True)
class ReservationReconciliation:
    """Shared-truth classification for one exact machine reservation."""

    outcome: Literal["retained", "retag", "release", "isolated"]
    reason: Literal["task_missing", "claim_missing"] | None = None
    attempt_id: str | None = None
    fencing_token: int | None = None


@dataclass(frozen=True, slots=True)
class DueOfferProgress:
    """Bounded result of processing one durable offer-deadline entry."""

    outcome: Literal["offered", "noop"]
    reason: Literal[
        "offered",
        "no_due_deadline",
        "task_missing",
        "task_not_queued",
        "claim_active",
        "not_home_machine",
        "deadline_not_due",
        "elapsed_offer_unproven",
        "offer_rejected",
    ]
    task_id: str | None


def classify_reservation(
    cfg: RootConfig,
    identity: ReservationIdentity,
) -> ReservationReconciliation:
    """Classify an exact reservation using Project truth without local effects."""
    try:
        task = load_task(cfg, identity.task_id)
    except FileNotFoundError:
        return ReservationReconciliation("release", reason="task_missing")

    claim = task.claim_control.get("active_claim")
    if not isinstance(claim, dict):
        if task.state.get("projection") == "blocked":
            return ReservationReconciliation("isolated")
        return ReservationReconciliation("release", reason="claim_missing")

    reservation_id = claim.get("reservation_id")
    attempt_id = claim.get("attempt_id")
    fencing_token = claim.get("fencing_token")
    if (
        reservation_id == identity.reservation_id
        and attempt_id == identity.attempt_id
        and fencing_token == identity.fencing_token
    ):
        return ReservationReconciliation("retained")
    if task.state.get("projection") == "blocked":
        return ReservationReconciliation("isolated")

    current_number = task.attempt_control.get("current_attempt_number")
    if (
        reservation_id == identity.reservation_id
        and attempt_id == identity.attempt_id
        and isinstance(fencing_token, int)
        and not isinstance(fencing_token, bool)
        and fencing_token > identity.fencing_token
        and isinstance(current_number, int)
        and not isinstance(current_number, bool)
        and current_number > 0
    ):
        attempt_file = attempt_path(cfg.shared_root, task.task_id, current_number)
        try:
            attempt = AttemptRecord.from_dict(read_json(attempt_file))
        except FileNotFoundError:
            return ReservationReconciliation("isolated")
        if (
            attempt.task_id == identity.task_id
            and attempt.attempt_id == identity.attempt_id
            and attempt.attempt_number == current_number
            and attempt.reservation_id == identity.reservation_id
            and attempt.current_fencing_token == fencing_token
        ):
            return ReservationReconciliation(
                "retag",
                attempt_id=attempt.attempt_id,
                fencing_token=fencing_token,
            )
        return ReservationReconciliation("isolated")

    malformed_claim = (
        not isinstance(reservation_id, str)
        or not isinstance(attempt_id, str)
        or not isinstance(fencing_token, int)
        or isinstance(fencing_token, bool)
        or fencing_token <= 0
    )
    if malformed_claim:
        return ReservationReconciliation("isolated")
    return ReservationReconciliation("release", reason="claim_missing")


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
        if should_reconcile_reservations:
            reconcile_project_reservations(
                cfg,
                reservation_runtime_root=reservation_root,
                project_id=project_id,
            )
        # Keep due offers ahead of admission, but only consume one durable
        # deadline partition entry per cycle. Maintenance descriptors own the
        # remaining recovery work after primary dispatch has had its turn.
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
    result = classify_reservation(cfg, identity)
    if result.outcome == "retained":
        return "retained"
    if result.outcome == "isolated":
        return "isolated"
    if result.outcome == "retag":
        assert result.attempt_id is not None
        assert result.fencing_token is not None
        is_retagged = retag_if_matches(
            reservation_runtime_root,
            identity,
            result.attempt_id,
            result.fencing_token,
        )
        return "retagged" if is_retagged else "changed"
    assert result.reason is not None
    is_released = release_if_matches(reservation_runtime_root, identity, result.reason)
    return "released" if is_released else "changed"


def offer_due_tasks(cfg: RootConfig) -> None:
    """Move elapsed home-only work into its configured shared queue."""
    with diagnostic_span("offer_due_tasks"):
        advance_due_offer(cfg)


def advance_due_offer(
    cfg: RootConfig,
    *,
    before_shared_mutation: Callable[[], None] | None = None,
) -> DueOfferProgress:
    """Process at most one due deadline using the existing offer transaction."""
    path = next(iter_due_deadline_paths(cfg, limit=1), None)
    if path is None:
        return DueOfferProgress("noop", "no_due_deadline", None)
    task_id = path.stem
    try:
        task = load_task(cfg, task_id)
    except FileNotFoundError:
        with fenced_mutations(cfg.shared_root, before_shared_mutation):
            remove_deadline_index(cfg, task_id)
        return DueOfferProgress("noop", "task_missing", task_id)

    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        sync_deadline_index(cfg, task)
    if task.state["projection"] != "queued":
        return DueOfferProgress("noop", "task_not_queued", task_id)
    if task.claim_control.get("active_claim"):
        return DueOfferProgress("noop", "claim_active", task_id)
    if task.placement_policy["home_machine"] != cfg.machine_name:
        return DueOfferProgress("noop", "not_home_machine", task_id)
    if not offer_due(task):
        return DueOfferProgress("noop", "deadline_not_due", task_id)
    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        elapsed_proven = elapsed_offer_is_proven(cfg, task)
    if not elapsed_proven:
        return DueOfferProgress("noop", "elapsed_offer_unproven", task_id)
    try:
        with fenced_mutations(cfg.shared_root, before_shared_mutation):
            result = offer(cfg, task.task_id, reason="elapsed")
    except (ValueError, FileNotFoundError):
        return DueOfferProgress("noop", "offer_rejected", task_id)
    if result.resulting_state != "shared":
        return DueOfferProgress("noop", "elapsed_offer_unproven", task_id)
    return DueOfferProgress("offered", "offered", task_id)
