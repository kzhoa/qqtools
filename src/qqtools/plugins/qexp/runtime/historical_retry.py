"""Finite diagnostics for cancelled Attempts without historical retry authority.

Terminal cleanup and retry eligibility are separate. Current writers never
terminate launch-authorized work on a lease deadline. Retained deadline results
lack immutable cancellation history, so no receipt or exit code grants retry.
"""

from __future__ import annotations

from .records import AttemptRecord, TaskRecord


class HistoricalRetryError(ValueError):
    """A cancelled Attempt cannot enter the ordinary retry transaction."""

    def __init__(self, reason: str, task_id: str, attempt_id: str | None, detail: str):
        super().__init__(f"Task {task_id!r}, Attempt {attempt_id!r}: {detail}")
        self.reason = reason
        self.task_id = task_id
        self.attempt_id = attempt_id


def reject_cancelled_retry(task: TaskRecord, attempt: AttemptRecord | None) -> None:
    """Reject using authoritative records; missing cause is never permission."""
    attempt_id = task.attempt_control.get("current_attempt_id")
    if attempt is not None:
        attempt_id = attempt.attempt_id
    if (
        task.claim_control.get("active_claim")
        or attempt is None
        or attempt.phase not in {"succeeded", "failed", "cancelled"}
    ):
        raise HistoricalRetryError(
            "termination_recovery_pending",
            task.task_id,
            attempt_id,
            "terminal ownership is unresolved; preserve the execution evidence and inspect agent recovery",
        )
    if (
        task.attempt_control.get("current_attempt_id") != attempt.attempt_id
        or task.attempt_control.get("current_attempt_number") != attempt.attempt_number
        or attempt.task_id != task.task_id
        or attempt.phase != "cancelled"
    ):
        raise HistoricalRetryError(
            "termination_evidence_conflict",
            task.task_id,
            attempt_id,
            "Task and Attempt terminal identities conflict; preserve both records and investigate",
        )
    if attempt.result.get("reason") == "terminated_by_agent":
        raise HistoricalRetryError(
            "unsupported_evidence",
            task.task_id,
            attempt_id,
            "retained agent termination lacks immutable cancellation history; cleanup does not authorize retry",
        )
    raise HistoricalRetryError(
        "retry_not_eligible",
        task.task_id,
        attempt_id,
        "cancelled execution is not eligible for ordinary retry; inspect the retained cancellation result",
    )
