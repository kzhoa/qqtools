from __future__ import annotations

import pytest

from qqtools.plugins.qexp.machine_dispatch_plan import (
    MachineDispatchPlan,
    MachineDispatchSnapshot,
    PrimaryCandidateObservation,
    build_machine_dispatch_plan,
    evaluate_primary_candidate,
    order_dispatch_project_ids,
    reduce_dispatch_cursor,
)


def test_order_dispatch_project_ids_uses_cursor_and_lexical_order() -> None:
    assert order_dispatch_project_ids(("project-b", "project-a", "project-c"), "project-b") == (
        "project-b",
        "project-c",
        "project-a",
    )
    assert order_dispatch_project_ids(("project-b", "project-a"), "removed-project") == (
        "project-a",
        "project-b",
    )


@pytest.mark.parametrize(
    ("has_free_capacity", "primary_demand_state", "expected_roles"),
    [
        (False, "no_primary_demand", ()),
        (True, "runnable_now", ("primary",)),
        (True, "waiting_for_aggregation", ("primary",)),
        (True, "unresolved", ("primary",)),
        (True, "no_primary_demand", ("primary", "borrow")),
    ],
)
def test_machine_dispatch_plan_admits_borrow_only_without_primary_demand(
    has_free_capacity: bool,
    primary_demand_state: str,
    expected_roles: tuple[str, ...],
) -> None:
    snapshot = MachineDispatchSnapshot(
        enabled_project_ids=("project-b", "project-a"),
        cursor_project_id="project-b",
        has_free_capacity=has_free_capacity,
        primary_demand_state=primary_demand_state,
    )

    plan = build_machine_dispatch_plan(snapshot)

    assert plan.ordered_project_ids == ("project-b", "project-a")
    assert plan.admission_roles == expected_roles


def test_reduce_dispatch_cursor_advances_after_winner_or_empty_cycle() -> None:
    plan = MachineDispatchPlan(("project-b", "project-c", "project-a"), ("primary",))

    assert reduce_dispatch_cursor(plan, "project-c").project_id == "project-a"
    assert reduce_dispatch_cursor(plan, None).project_id == "project-c"


def test_reduce_dispatch_cursor_handles_single_project_and_unknown_winner() -> None:
    plan = MachineDispatchPlan(("project-a",), ())

    assert reduce_dispatch_cursor(plan, None).project_id == "project-a"
    with pytest.raises(ValueError, match="not in the dispatch plan"):
        reduce_dispatch_cursor(plan, "project-b")


def test_reduce_dispatch_cursor_skips_empty_dispatch_plan() -> None:
    assert reduce_dispatch_cursor(MachineDispatchPlan((), ()), None) is None


def test_machine_dispatch_snapshot_rejects_ambiguous_project_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        MachineDispatchSnapshot(
            enabled_project_ids=("project-a", "project-a"),
            cursor_project_id=None,
            has_free_capacity=True,
            primary_demand_state="unresolved",
        )


@pytest.mark.parametrize(
    ("observation", "expected_outcome", "expected_reason"),
    [
        (
            PrimaryCandidateObservation(False, True, None, 1, 2, 2),
            "skip",
            "placement_rejected",
        ),
        (PrimaryCandidateObservation(True, False, None, 1, 2, 2), "skip", None),
        (
            PrimaryCandidateObservation(True, True, "missing", 1, 2, 2),
            "skip",
            "working_directory:missing",
        ),
        (
            PrimaryCandidateObservation(True, True, None, 2, 2, 2, 2, 1),
            "skip",
            "group_gpu_limit_reached",
        ),
        (
            PrimaryCandidateObservation(True, True, None, 3, 2, 2),
            "skip",
            "exceeds_machine_capacity",
        ),
        (PrimaryCandidateObservation(True, True, None, 1, 2, 2), "runnable_now", None),
        (PrimaryCandidateObservation(True, True, None, 2, 2, 1), "waiting_for_aggregation", None),
    ],
)
def test_evaluate_primary_candidate_reduces_admission_observations(
    observation: PrimaryCandidateObservation,
    expected_outcome: str,
    expected_reason: str | None,
) -> None:
    decision = evaluate_primary_candidate(observation)

    assert decision.outcome == expected_outcome
    assert decision.reason == expected_reason


def test_primary_candidate_observation_rejects_impossible_gpu_counts() -> None:
    with pytest.raises(ValueError, match="free GPU count"):
        PrimaryCandidateObservation(True, True, None, 1, 1, 2)
