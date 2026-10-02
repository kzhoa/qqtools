"""Read-only proofs for settled shared Task and Attempt records."""

from __future__ import annotations

from collections.abc import Mapping

from ..config_types import RootConfig
from .paths import attempt_path
from .records import AttemptRecord
from .store import read_json
from .tasks import load_task


def canonical_attempt_number(task_id: str, attempt_id: str) -> int | None:
    """Return the immutable number encoded by an exact canonical Attempt ID."""
    prefix = f"{task_id}-attempt-"
    if not attempt_id.startswith(prefix):
        return None
    suffix = attempt_id[len(prefix) :]
    if not suffix.isascii() or not suffix.isdecimal():
        return None
    number = int(suffix)
    return number if suffix == str(number) else None


def load_settled_terminal_attempt(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    *,
    attempt_number: int | None = None,
    recovery_locator: Mapping[str, object] | None = None,
) -> AttemptRecord | None:
    """Resolve settled current or historical truth using only shared Task and Attempt records."""
    task = load_task(cfg, task_id)
    claim = task.claim_control.get("active_claim") or {}
    if not isinstance(claim, dict) or claim.get("attempt_id") == attempt_id:
        return None
    current_id = task.attempt_control.get("current_attempt_id")
    current_number = task.attempt_control.get("current_attempt_number")
    number = current_number if attempt_number is None else attempt_number
    # Scheduler identities encode their immutable number. An opaque historical
    # identity may instead have a captured direct locator. Either hint must
    # still match authoritative Task/Attempt truth below before it is used.
    canonical_number = canonical_attempt_number(task_id, attempt_id)
    if canonical_number is not None:
        number = canonical_number
        if attempt_number is not None and number != attempt_number:
            return None
    elif attempt_number is None and recovery_locator is not None:
        if recovery_locator["task_id"] not in (None, task_id):
            return None
        if recovery_locator["attempt_number"] is not None:
            number = recovery_locator["attempt_number"]
    if type(number) is not int or type(current_number) is not int or not 1 <= number <= current_number:
        return None
    if number == current_number:
        # A terminal Attempt can precede its Task commit after a crash. Only
        # committed Task terminal truth or a completed retry transition proves
        # that this Attempt no longer owns the Task's terminal publication.
        if claim or task.state.get("projection") not in {"succeeded", "failed", "cancelled", "queued"}:
            return None
        if current_id not in {None, attempt_id}:
            return None
        if task.state.get("projection") == "queued" and current_id is not None:
            return None
    next_number = task.attempt_control.get("next_attempt_number")
    if type(next_number) is not int or number >= next_number:
        return None
    attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, number)))
    if (
        attempt.task_id != task_id
        or attempt.attempt_id != attempt_id
        or attempt.attempt_number != number
        or attempt.machine_name != cfg.machine_name
        or attempt.phase not in {"succeeded", "failed", "cancelled"}
    ):
        return None
    return attempt
