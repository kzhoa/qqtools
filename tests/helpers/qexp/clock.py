"""Deterministic evaluator clock inputs for persisted timed-offer proofs."""

from datetime import timedelta

from qqtools.plugins.qexp.lease import ClockObservation, timed_offer_deadline_upper
from qqtools.plugins.qexp.runtime.availability import transitions
from qqtools.plugins.qexp.runtime.tasks import load_task


def set_offer_evaluation_time(monkeypatch, cfg, task_id, *, seconds_after_deadline=1):
    """Position the evaluator's UTC lower bound relative to the creator's upper bound."""
    task = load_task(cfg, task_id)
    proof = task.placement_runtime["offer_clock_evidence"]
    deadline = task.placement_runtime["offer_eligible_at"]
    observation = ClockObservation.from_dict(proof["creator_observation"])
    monotonic_now = proof["deadline_monotonic_at"] + max(0, seconds_after_deadline)
    now = timed_offer_deadline_upper(deadline, proof) + timedelta(
        seconds=seconds_after_deadline + observation.bound_at(monotonic_now)
    )
    monkeypatch.setattr(transitions, "clock_evidence", lambda _cfg: (observation, now, monotonic_now))
