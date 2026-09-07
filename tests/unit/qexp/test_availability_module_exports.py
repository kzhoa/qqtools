from __future__ import annotations


def test_availability_facade_exports_transition_owner_objects() -> None:
    from qqtools.plugins.qexp.runtime import availability
    from qqtools.plugins.qexp.runtime.availability import offer_deadlines, transitions

    assert availability.AvailabilityTransitionRequest is transitions.AvailabilityTransitionRequest
    assert availability.AvailabilityTransitionResult is transitions.AvailabilityTransitionResult
    assert availability.apply_availability_transition is transitions.apply_availability_transition
    assert availability.clock_evidence is transitions.clock_evidence
    assert availability.elapsed_offer_is_proven is transitions.elapsed_offer_is_proven
    assert availability.reconcile_availability_operations is transitions.reconcile_availability_operations
    assert availability.remove_deadline_index is offer_deadlines.remove_deadline_index
    assert availability.sync_deadline_index is offer_deadlines.sync_deadline_index
    assert availability.iter_due_deadline_paths is offer_deadlines.iter_due_deadline_paths
    assert availability.iter_flat_deadline_paths is offer_deadlines.iter_flat_deadline_paths
    assert availability.migrate_legacy_deadline_indexes is offer_deadlines.migrate_legacy_deadline_indexes
    assert availability.rebuild_deadline_indexes is offer_deadlines.rebuild_deadline_indexes


def test_offer_deadline_owner_does_not_depend_on_availability_transitions() -> None:
    from pathlib import Path

    source = Path(__file__).parents[3] / "src/qqtools/plugins/qexp/runtime/availability/offer_deadlines.py"

    assert "transitions" not in source.read_text(encoding="utf-8")
