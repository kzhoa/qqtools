"""Exercise machine-local takeover journal ownership and crash replay."""

from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.takeover_intent import (
    TakeoverIntent,
    TakeoverIntentConflict,
    finish_takeover_intent,
    read_takeover_intent,
    read_takeover_result,
    save_takeover_intent,
)
from qqtools.plugins.qexp.agent.takeover_plan import TakeoverGroupApproval
from qqtools.plugins.qexp.runtime.registration_protocol import MAX_REGISTRATION_RECORD_BYTES
from qqtools.plugins.qexp.runtime.store import atomic_replace, json_encoded_size
from qqtools.plugins.qexp.runtime.takeover_registration import RegistrationWitness, TakeoverRegistration

pytestmark = pytest.mark.integration


@pytest.fixture
def intent(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    transition = TakeoverRegistration(
        project_id="project-id",
        shared_root=tmp_path / "project" / ".qexp",
        transaction_id="1" * 32,
        successor_runtime_id="2" * 64,
        successor_runtime_root=root,
        successor_generation="3" * 32,
        target=RegistrationWitness(
            machine_name="g2",
            generation="4" * 32,
            runtime_id="5" * 64,
            runtime_root=str(tmp_path / "old runtime"),
            eligibility_expires_at="2000-01-01T00:00:00Z",
            state="eligible",
        ),
        departing=None,
    )
    return TakeoverIntent(plan_id="6" * 64, transition=transition, original_enabled=True)


def test_intent_preserves_exact_phase_and_cancellation_cas(intent):
    root = intent.transition.successor_runtime_root
    no_longer_current = replace(intent, phase="reserved")
    save_takeover_intent(root, intent, expected=None)
    assert read_takeover_intent(root) == intent
    assert TakeoverIntent.from_dict(intent.to_dict()) == intent

    with pytest.raises(TakeoverIntentConflict):
        save_takeover_intent(root, replace(intent, phase="draining"), expected=intent)
    with pytest.raises(TakeoverIntentConflict):
        save_takeover_intent(
            root, replace(intent, phase="cancelling", is_cancel_requested=True), expected=no_longer_current
        )

    cancelling_requested = replace(intent, is_cancel_requested=True)
    save_takeover_intent(root, cancelling_requested, expected=intent)
    with pytest.raises(TakeoverIntentConflict):
        save_takeover_intent(root, intent, expected=cancelling_requested)
    cancelling = replace(cancelling_requested, phase="cancelling")
    save_takeover_intent(root, cancelling, expected=cancelling_requested)
    cancelled = replace(cancelling, phase="cancelled")
    save_takeover_intent(root, cancelled, expected=cancelling)
    assert read_takeover_intent(root) == cancelled


def test_intent_never_replaces_a_legacy_registration_journal(intent):
    root = intent.transition.successor_runtime_root
    active = root / "project-registration-replacement.json"
    atomic_replace(active, {"project_registration_replacement": {"phase": "draining"}})
    original = active.read_bytes()

    assert read_takeover_intent(root) is None
    with pytest.raises(TakeoverIntentConflict):
        save_takeover_intent(root, intent, expected=None)

    assert active.read_bytes() == original


def test_terminal_receipt_survives_crash_before_active_retirement(intent, monkeypatch):
    root = intent.transition.successor_runtime_root
    save_takeover_intent(root, intent, expected=None)
    aborting = replace(intent, phase="aborting_stale")
    save_takeover_intent(root, aborting, expected=intent)
    stale = replace(aborting, phase="stale")
    save_takeover_intent(root, stale, expected=aborting)
    active = root / "project-registration-replacement.json"
    unlink = Path.unlink

    def fail_active_unlink(path, *args, **kwargs):
        if path == active:
            raise OSError("injected crash before retiring the active intent")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", fail_active_unlink)
        with pytest.raises(OSError, match="injected crash"):
            finish_takeover_intent(root, stale)

    assert read_takeover_intent(root) == stale
    assert read_takeover_result(root, stale.plan_id) == stale

    finish_takeover_intent(root, stale)
    assert read_takeover_intent(root) is None
    assert read_takeover_result(root, stale.plan_id) == stale
    finish_takeover_intent(root, stale)


def test_intent_rejects_a_terminal_receipt_placed_under_another_plan(intent):
    root = intent.transition.successor_runtime_root
    save_takeover_intent(root, intent, expected=None)
    aborting = replace(intent, phase="aborting_stale")
    save_takeover_intent(root, aborting, expected=intent)
    stale = replace(aborting, phase="stale")
    save_takeover_intent(root, stale, expected=aborting)
    other = replace(stale, plan_id="7" * 64)
    terminal = root / "takeover-results" / f"{stale.plan_id}.json"
    atomic_replace(terminal, other.to_dict())
    before = terminal.read_bytes()

    with pytest.raises((ValueError, TakeoverIntentConflict)):
        finish_takeover_intent(root, stale)

    assert read_takeover_intent(root) == stale
    assert terminal.read_bytes() == before


@pytest.mark.parametrize(
    ("terminal", "phases"),
    [
        ("stale", ("aborting_stale", "stale")),
        ("cancelled", ("cancelling", "cancelled")),
        (
            "completed",
            (
                "reserved",
                "draining",
                "verifying_drain",
                "committing",
                "shared_committed",
                "publishing_binding",
                "releasing",
                "completed",
            ),
        ),
    ],
)
def test_terminal_plan_cannot_reopen_the_active_slot(intent, terminal, phases):
    root = intent.transition.successor_runtime_root
    save_takeover_intent(root, intent, expected=None)
    current = intent
    for phase in phases:
        updated = replace(current, phase=phase, is_cancel_requested=terminal == "cancelled")
        save_takeover_intent(root, updated, expected=current)
        current = updated
    finish_takeover_intent(root, current)

    with pytest.raises(TakeoverIntentConflict, match="terminal result"):
        save_takeover_intent(root, intent, expected=None)

    assert read_takeover_intent(root) is None
    assert read_takeover_result(root, intent.plan_id) == current


def test_intent_rejects_inputs_that_cannot_be_carried_by_project_io(intent):
    for update in (
        {"project_id": "p" * 129},
        {"target": replace(intent.transition.target, generation="g" * 60_000)},
    ):
        with pytest.raises(ValueError):
            replace(intent, transition=replace(intent.transition, **update))

    assert read_takeover_intent(intent.transition.successor_runtime_root) is None


def test_intent_sizes_the_json_encoded_path_repeated_by_project_io(intent):
    shared_root = Path("/").joinpath(*(["\x01" * 200] * 15), ".qexp")
    transition = replace(
        intent.transition,
        shared_root=shared_root,
        target=replace(intent.transition.target, generation="g" * 33_000),
    )

    with pytest.raises(ValueError, match="takeover_transition"):
        replace(intent, transition=transition)


def test_group_approvals_survive_replay_and_cannot_expand_after_acceptance(intent):
    group = TakeoverGroupApproval(
        action="add_primary_worker",
        group_name="training",
        machine_name=intent.transition.target.machine_name,
        group_revision=4,
        worker_set_epoch=1,
        affected_task_count=2,
        affected_tasks_digest="f" * 64,
    )
    approved = replace(intent, group_approvals=(group,))
    root = approved.transition.successor_runtime_root
    save_takeover_intent(root, approved, expected=None)

    assert read_takeover_intent(root) == approved

    changed = replace(approved, group_approvals=(replace(group, group_revision=5),))
    with pytest.raises(TakeoverIntentConflict, match="Group approval changed"):
        save_takeover_intent(root, changed, expected=approved)

    assert read_takeover_intent(root) == approved


def test_intent_leaves_space_for_the_largest_future_phase(intent):
    approval = TakeoverGroupApproval("add_primary_worker", "a", "g2", 1, 1, 1, "f" * 64)
    accepted = replace(intent, group_approvals=(approval,))
    headroom = MAX_REGISTRATION_RECORD_BYTES - json_encoded_size(accepted.to_dict())
    near_limit = replace(approval, group_name="a" * (headroom - 2))

    with pytest.raises(ValueError, match="project_takeover"):
        replace(intent, group_approvals=(near_limit,))
