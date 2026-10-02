from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from qqtools.plugins.qexp.agent.project_io_arbiter import ProjectIOArbiter, ServiceIntent


def _owner(project: str, generation: str = "generation-1") -> tuple[str, str, str]:
    return ("runtime", project, generation)


def _intent(project: str, service: str, *, work: str = "first", generation: str = "generation-1") -> ServiceIntent:
    operation = {
        "authority": "authority_renewal",
        "primary": "scheduler_observe",
        "background": "machine_snapshot_publish",
    }[service]
    return ServiceIntent(_owner(project, generation), service, operation, (work,))


def test_weighted_service_opportunities_do_not_starve_primary_or_background():
    arbiter = ProjectIOArbiter()
    intents = [_intent("a", "authority"), _intent("p", "primary"), _intent("b", "background")]
    arbiter.reconcile([item.owner for item in intents])
    selected = [arbiter.select(intents, limit=1)[0].service_class for _ in range(12)]
    assert selected == ["authority", "primary", "authority", "background"] * 3


def test_binding_rotation_survives_registry_reorder_and_new_arrivals():
    arbiter = ProjectIOArbiter()
    intents = [_intent(name, "authority") for name in ("a", "b", "c")]
    arbiter.reconcile([item.owner for item in intents])
    assert arbiter.select(intents, limit=1) == (intents[0],)
    newcomer = _intent("new", "authority")
    arbiter.reconcile([newcomer.owner, intents[0].owner, intents[2].owner, intents[1].owner])
    candidates = [newcomer, *reversed(intents)]
    assert [arbiter.select(candidates, limit=1)[0] for _ in range(4)] == [intents[1], intents[2], intents[0], newcomer]


@pytest.mark.parametrize("service", ["authority", "primary", "background"])
def test_two_blocked_bindings_leave_both_healthy_owners_selectable(service):
    arbiter = ProjectIOArbiter()
    intents = [_intent(name, service) for name in ("hung-a", "hung-b", "healthy-a", "healthy-b")]
    arbiter.reconcile([item.owner for item in intents])
    assert arbiter.select(intents, blocked=[item.owner for item in intents[:2]], limit=2) == tuple(intents[2:])


def test_owner_is_selected_once_across_service_classes_and_duplicates():
    arbiter = ProjectIOArbiter()
    authority = _intent("same", "authority")
    primary = _intent("same", "primary")
    background = _intent("other", "background")
    arbiter.reconcile([authority.owner, background.owner])
    assert arbiter.select([authority, _intent("same", "authority", work="second"), primary, background]) == (
        authority,
        background,
    )


def test_generation_replacement_rejects_stale_candidates_without_implicit_admission():
    arbiter = ProjectIOArbiter()
    old = _intent("a", "primary")
    new = _intent("a", "primary", generation="generation-2")
    other = _intent("not-registered", "primary")
    arbiter.reconcile([old.owner])
    arbiter.reconcile([new.owner])
    assert arbiter.select([old, other]) == ()
    assert arbiter.select([old, new, other], blocked=[old.owner]) == (new,)


def test_empty_and_saturated_turns_do_not_spend_class_opportunity():
    arbiter = ProjectIOArbiter()
    intents = [_intent("a", "authority"), _intent("p", "primary"), _intent("b", "background")]
    arbiter.reconcile([item.owner for item in intents])
    assert arbiter.select(intents, limit=1) == (intents[0],)
    assert arbiter.select([], limit=4) == ()
    assert arbiter.select(intents, limit=0) == ()
    assert arbiter.select(intents, blocked=[item.owner for item in intents]) == ()
    assert arbiter.select(intents, limit=1) == (intents[1],)


def test_roster_larger_than_candidate_budget_progresses_when_producer_rotates_discovery():
    arbiter = ProjectIOArbiter()
    intents = [_intent(f"project-{index}", "primary") for index in range(65)]
    arbiter.reconcile([item.owner for item in intents])
    counts = Counter()
    # The caller owns bounded discovery. Omitted work is revisited, never queued
    # unboundedly inside the arbiter or dropped from its authoritative source.
    for turn in range(130):
        offset = turn % len(intents)
        window = (intents[offset:] + intents[:offset])[:64]
        counts.update(item.owner for item in arbiter.select(window, limit=4))
    assert len(counts) == 65
    assert min(counts.values()) > 0


def test_continuous_new_bindings_do_not_jump_ahead_of_existing_due_work():
    arbiter = ProjectIOArbiter()
    incumbents = [_intent(f"old-{index}", "authority") for index in range(4)]
    candidates = list(incumbents)
    arbiter.reconcile([item.owner for item in candidates])
    selected = []
    for index in range(4):
        candidates.insert(0, _intent(f"new-{index}", "authority"))
        arbiter.reconcile([item.owner for item in candidates])
        selected.extend(arbiter.select(candidates, limit=1))
    assert selected == incumbents


def test_dependency_can_inherit_authority_class_without_changing_operation_kind():
    intent = ServiceIntent(_owner("a"), "authority", "validate_binding", ())
    arbiter = ProjectIOArbiter()
    arbiter.reconcile([intent.owner, intent.owner])
    assert arbiter.select([intent]) == (intent,)
    with pytest.raises(FrozenInstanceError):
        intent.operation_kind = "scheduler_claim"


@pytest.mark.parametrize("limit", [-1, 5, True, 1.0, "1"])
def test_invalid_limit_does_not_change_selection_state(limit):
    arbiter = ProjectIOArbiter()
    intents = [_intent("a", "authority"), _intent("p", "primary")]
    arbiter.reconcile([item.owner for item in intents])
    with pytest.raises(ValueError):
        arbiter.select(intents, limit=limit)
    assert arbiter.select(intents, limit=1) == (intents[0],)


def test_candidate_limit_is_checked_before_deduplication_or_grant():
    arbiter = ProjectIOArbiter()
    intent = _intent("a", "authority")
    arbiter.reconcile([intent.owner])
    with pytest.raises(ValueError):
        arbiter.select([intent] * 65)
    assert arbiter.select([intent] * 64) == (intent,)


@pytest.mark.parametrize("owner", [("runtime", "project"), ("runtime", "project", ""), ["r", "p", "g"], (1, "p", "g")])
def test_invalid_reconcile_is_atomic(owner):
    arbiter = ProjectIOArbiter()
    intents = [_intent("a", "primary"), _intent("b", "primary")]
    arbiter.reconcile([item.owner for item in intents])
    assert arbiter.select(intents, limit=1) == (intents[0],)
    with pytest.raises(ValueError):
        arbiter.reconcile([_owner("new"), owner])
    assert arbiter.select(intents, limit=1) == (intents[1],)


def test_invalid_blocked_identity_and_candidate_do_not_spend_opportunities():
    arbiter = ProjectIOArbiter()
    intent = _intent("a", "authority")
    arbiter.reconcile([intent.owner])
    with pytest.raises(ValueError):
        arbiter.select([intent], blocked=[("not-an-owner",)])
    with pytest.raises(ValueError):
        arbiter.select([intent, object()])
    assert arbiter.select([intent]) == (intent,)


@pytest.mark.parametrize(
    "changes",
    [
        {"owner": ("r", "p")},
        {"service_class": "unknown"},
        {"operation_kind": "arbitrary_python_call"},
        {"work_key": ["mutable"]},
        {"work_key": ("",)},
        {"work_key": (1,)},
    ],
)
def test_invalid_intents_are_rejected(changes):
    fields = dict(owner=_owner("a"), service_class="authority", operation_kind="authority_service", work_key=())
    fields.update(changes)
    with pytest.raises(ValueError):
        ServiceIntent(**fields)
