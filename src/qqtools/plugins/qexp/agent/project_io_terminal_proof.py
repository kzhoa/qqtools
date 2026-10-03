"""Pure construction and validation helpers for supervised terminal proofs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .project_io_protocol import authority_terminal_transition_digest

_TERMINAL_IDENTITY_FIELDS = (
    "task_id",
    "attempt_id",
    "attempt_number",
    "fencing_token",
    "reservation_id",
    "process_identity",
    "mode",
)


def has_exact_terminal_revisions(value: object) -> bool:
    """Return whether a value is an exact Task/Attempt revision witness."""
    if not isinstance(value, Mapping) or set(value) != {"task", "attempt_digest"}:
        return False
    task_revision = value.get("task")
    attempt_digest = value.get("attempt_digest")
    return (
        type(task_revision) is int
        and task_revision >= 0
        and isinstance(attempt_digest, str)
        and len(attempt_digest) == 64
        and all(character in "0123456789abcdef" for character in attempt_digest)
    )


def terminal_transition_target(
    *,
    mode: object,
    exit_code: int,
    cancel_requested: object,
) -> tuple[str, str, str | None] | None:
    """Select the exact terminal result permitted by one supervision mode."""
    if type(cancel_requested) is not bool:
        return None
    if mode == "active":
        if cancel_requested and exit_code != 0:
            return "cancelled", "termination_process_already_exited", "already_exited"
        phase = "succeeded" if exit_code == 0 else "failed"
        reason = "completed" if exit_code == 0 else "nonzero_exit"
        return phase, reason, "already_exited" if cancel_requested else None
    if mode == "detached_orphan":
        if cancel_requested:
            return "cancelled", "termination_process_already_exited", "already_exited"
        phase = "succeeded" if exit_code == 0 else "failed"
        reason = "completed" if exit_code == 0 else "nonzero_exit"
        return phase, reason, None
    return None


def build_terminal_transition(
    *,
    parameters: Mapping[str, Any],
    machine_name: str,
    exit_code: int,
    source_revisions: Mapping[str, Any],
    cancel_requested: object,
) -> dict[str, Any] | None:
    """Build one exact terminal publication from already-validated evidence."""
    target = terminal_transition_target(
        mode=parameters.get("mode"),
        exit_code=exit_code,
        cancel_requested=cancel_requested,
    )
    if target is None or not has_exact_terminal_revisions(source_revisions):
        return None
    phase, reason, termination_result = target
    digest = authority_terminal_transition_digest(
        mode=parameters["mode"],
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=machine_name,
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=phase,
        reason=reason,
        exit_code=exit_code,
        termination_result=termination_result,
    )
    return {
        **parameters,
        "phase": phase,
        "reason": reason,
        "exit_code": exit_code,
        "termination_result": termination_result,
        "source_revisions": dict(source_revisions),
        "transition_digest": digest,
    }


def matches_terminal_observation_identity(
    evidence: Mapping[str, Any],
    parameters: Mapping[str, Any],
    machine_name: str,
) -> bool:
    """Match shared observation evidence to one exact local candidate."""
    return (
        evidence.get("machine_name") == machine_name
        and all(evidence.get(field) == parameters[field] for field in _TERMINAL_IDENTITY_FIELDS)
        and evidence.get("authority_granted") is False
        and isinstance(evidence.get("local_effects"), (list, tuple))
        and not evidence["local_effects"]
        and has_exact_terminal_revisions(evidence.get("source_revisions"))
        and type(evidence.get("cancel_requested")) is bool
    )


def matches_terminal_lifecycle_event(
    event: object,
    committed_revisions: object,
    parameters: Mapping[str, Any],
    *,
    phase: str,
    reason: str,
    exit_code: int | None,
) -> bool:
    """Match a lifecycle event to the committed terminal Task revision."""
    if not isinstance(event, Mapping) or not has_exact_terminal_revisions(committed_revisions):
        return False
    return all(
        event.get(field) == value
        for field, value in (
            ("task_id", parameters["task_id"]),
            ("attempt_id", parameters["attempt_id"]),
            ("attempt_number", parameters["attempt_number"]),
            ("phase", phase),
            ("reason", reason),
            ("exit_code", exit_code),
            ("task_revision", committed_revisions["task"]),
        )
    )


__all__ = [
    "build_terminal_transition",
    "has_exact_terminal_revisions",
    "matches_terminal_lifecycle_event",
    "matches_terminal_observation_identity",
    "terminal_transition_target",
]
