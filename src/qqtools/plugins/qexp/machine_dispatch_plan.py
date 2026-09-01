"""Pure planning rules for one machine dispatch cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AdmissionRole = Literal["primary", "borrow"]
PrimaryDemandState = Literal[
    "runnable_now",
    "waiting_for_aggregation",
    "no_primary_demand",
    "unresolved",
]
PrimaryCandidateOutcome = Literal["skip", "runnable_now", "waiting_for_aggregation"]


@dataclass(frozen=True, slots=True)
class PrimaryCandidateObservation:
    """Validated candidate facts used to reduce primary demand."""

    is_eligible: bool
    has_primary_group_worker: bool
    working_directory_reason: str | None
    requested_gpus: int
    visible_gpu_count: int
    free_gpu_count: int
    group_gpu_limit: int | None = None
    group_gpu_usage: int = 0

    def __post_init__(self) -> None:
        if self.requested_gpus < 1:
            raise ValueError("requested GPU count must be positive.")
        if self.visible_gpu_count < 0 or self.free_gpu_count < 0:
            raise ValueError("GPU counts must not be negative.")
        if self.free_gpu_count > self.visible_gpu_count:
            raise ValueError("free GPU count must not exceed visible GPU count.")
        if self.group_gpu_limit is not None and self.group_gpu_limit < 0:
            raise ValueError("group GPU limit must not be negative.")
        if self.group_gpu_usage < 0:
            raise ValueError("group GPU usage must not be negative.")


@dataclass(frozen=True, slots=True)
class PrimaryCandidateDecision:
    """The pure primary-demand result for one ready candidate."""

    outcome: PrimaryCandidateOutcome
    reason: str | None = None


def evaluate_primary_candidate(
    observation: PrimaryCandidateObservation,
) -> PrimaryCandidateDecision:
    """Reduce validated candidate facts without reading runtime state."""
    if not observation.is_eligible:
        return PrimaryCandidateDecision("skip", "placement_rejected")
    if not observation.has_primary_group_worker:
        return PrimaryCandidateDecision("skip")
    if observation.working_directory_reason is not None:
        return PrimaryCandidateDecision(
            "skip", f"working_directory:{observation.working_directory_reason}"
        )
    if (
        observation.group_gpu_limit is not None
        and observation.requested_gpus
        > observation.group_gpu_limit - observation.group_gpu_usage
    ):
        return PrimaryCandidateDecision("skip", "group_gpu_limit_reached")
    if observation.requested_gpus > observation.visible_gpu_count:
        return PrimaryCandidateDecision("skip", "exceeds_machine_capacity")
    if observation.requested_gpus <= observation.free_gpu_count:
        return PrimaryCandidateDecision("runnable_now")
    return PrimaryCandidateDecision("waiting_for_aggregation")


@dataclass(frozen=True, slots=True)
class MachineDispatchSnapshot:
    """Validated inputs required to choose dispatch order and admission roles."""

    enabled_project_ids: tuple[str, ...]
    cursor_project_id: str | None
    has_free_capacity: bool
    primary_demand_state: PrimaryDemandState

    def __post_init__(self) -> None:
        if any(not project_id for project_id in self.enabled_project_ids):
            raise ValueError("enabled project IDs must not be empty.")
        if len(set(self.enabled_project_ids)) != len(self.enabled_project_ids):
            raise ValueError("enabled project IDs must be unique.")
        if self.primary_demand_state not in {
            "runnable_now",
            "waiting_for_aggregation",
            "no_primary_demand",
            "unresolved",
        }:
            raise ValueError("primary demand state is invalid.")


@dataclass(frozen=True, slots=True)
class MachineDispatchPlan:
    """Side-effect-free order and admission effects for one dispatch cycle."""

    ordered_project_ids: tuple[str, ...]
    admission_roles: tuple[AdmissionRole, ...]


@dataclass(frozen=True, slots=True)
class CursorUpdateEffect:
    """The cursor write to perform after dispatch observations are reduced."""

    project_id: str


def order_dispatch_project_ids(
    enabled_project_ids: tuple[str, ...],
    cursor_project_id: str | None,
) -> tuple[str, ...]:
    """Return enabled projects in stable cursor order."""
    ordered = tuple(sorted(enabled_project_ids))
    if cursor_project_id is None:
        return ordered
    try:
        index = ordered.index(cursor_project_id)
    except ValueError:
        return ordered
    return ordered[index:] + ordered[:index]


def build_machine_dispatch_plan(snapshot: MachineDispatchSnapshot) -> MachineDispatchPlan:
    """Plan fair admission layers without reading state or performing effects."""
    ordered_project_ids = order_dispatch_project_ids(
        snapshot.enabled_project_ids,
        snapshot.cursor_project_id,
    )
    if not snapshot.has_free_capacity or not ordered_project_ids:
        return MachineDispatchPlan(ordered_project_ids, ())
    admission_roles: tuple[AdmissionRole, ...] = ("primary",)
    if snapshot.primary_demand_state == "no_primary_demand":
        admission_roles += ("borrow",)
    return MachineDispatchPlan(ordered_project_ids, admission_roles)


def reduce_dispatch_cursor(
    plan: MachineDispatchPlan,
    last_successful_project_id: str | None,
) -> CursorUpdateEffect | None:
    """Choose the next cursor after the adapter reports the final winning project."""
    if not plan.ordered_project_ids:
        return None
    if last_successful_project_id is None:
        next_index = 1 % len(plan.ordered_project_ids)
    else:
        try:
            winner_index = plan.ordered_project_ids.index(last_successful_project_id)
        except ValueError as exc:
            raise ValueError("last successful project is not in the dispatch plan.") from exc
        next_index = (winner_index + 1) % len(plan.ordered_project_ids)
    return CursorUpdateEffect(plan.ordered_project_ids[next_index])
