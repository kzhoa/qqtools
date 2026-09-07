from __future__ import annotations


def test_availability_facade_exports_transition_owner_objects() -> None:
    from qqtools.plugins.qexp.runtime import availability
    from qqtools.plugins.qexp.runtime.availability import transitions

    assert availability.AvailabilityTransitionRequest is transitions.AvailabilityTransitionRequest
    assert availability.AvailabilityTransitionResult is transitions.AvailabilityTransitionResult
    assert availability.apply_availability_transition is transitions.apply_availability_transition
    assert availability.clock_evidence is transitions.clock_evidence
    assert availability.elapsed_offer_is_proven is transitions.elapsed_offer_is_proven
    assert availability.reconcile_availability_operations is transitions.reconcile_availability_operations
