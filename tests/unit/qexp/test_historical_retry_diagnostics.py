"""Historical cause diagnostics never grant execution authority."""

import pytest

from qqtools.plugins.qexp.runtime.historical_retry import HistoricalRetryError, reject_cancelled_retry
from qqtools.plugins.qexp.runtime.records import AttemptRecord, TaskRecord, TaskSpec


def _terminal_pair():
    task = TaskRecord.new(task_id="task-1", machine="gpu-1", spec=TaskSpec(["echo", "ok"], "/tmp", 1))
    attempt = AttemptRecord.claimed(
        task, "gpu-1", [0], "reservation-1", 1, authority_mode="holder_bound", clock_evidence=None
    )
    task.state = {"projection": "cancelled", "reason": "terminated_by_agent"}
    task.attempt_control.update(current_attempt_id=attempt.attempt_id, current_attempt_number=1)
    attempt.phase = "cancelled"
    attempt.result.update(reason="terminated_by_agent", exit_code=None)
    return task, attempt


@pytest.mark.parametrize("exit_code", [None, -15, -9, 0])
def test_exit_observations_do_not_prove_historical_intent(exit_code):
    task, attempt = _terminal_pair()
    attempt.result["exit_code"] = exit_code
    before = (task.to_dict(), attempt.to_dict())
    with pytest.raises(HistoricalRetryError) as caught:
        reject_cancelled_retry(task, attempt)
    assert caught.value.reason == "unsupported_evidence"
    assert attempt.attempt_id in str(caught.value)
    assert (task.to_dict(), attempt.to_dict()) == before


@pytest.mark.parametrize("case", ["active_claim", "missing_attempt", "running_attempt", "successor", "wrong_task"])
def test_unsettled_or_conflicting_history_is_diagnosed(case):
    task, attempt = _terminal_pair()
    expected = "termination_evidence_conflict"
    if case == "active_claim":
        task.claim_control["active_claim"] = {"attempt_id": attempt.attempt_id}
        expected = "termination_recovery_pending"
    elif case == "missing_attempt":
        attempt = None
        expected = "termination_recovery_pending"
    elif case == "running_attempt":
        attempt.phase = "running"
        expected = "termination_recovery_pending"
    elif case == "successor":
        task.attempt_control["current_attempt_id"] = "successor"
    else:
        attempt.task_id = "other-task"
    with pytest.raises(HistoricalRetryError) as caught:
        reject_cancelled_retry(task, attempt)
    assert caught.value.reason == expected


def test_user_cancellation_is_not_eligible():
    task, attempt = _terminal_pair()
    attempt.result["reason"] = "cancelled_before_launch"
    with pytest.raises(HistoricalRetryError) as caught:
        reject_cancelled_retry(task, attempt)
    assert caught.value.reason == "retry_not_eligible"
