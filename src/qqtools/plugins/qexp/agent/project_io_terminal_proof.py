"""Pure construction and validation helpers for supervised terminal proofs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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

_PROCESS_IDENTITY_FIELDS = (
    "wrapper_pid",
    "wrapper_start_time_ticks",
    "process_group_id",
    "process_group_start_time_ticks",
)

_TERMINAL_OBSERVATION_OUTCOMES = frozenset({"already_terminal", "settled_terminal"})
_TERMINAL_PUBLICATION_OUTCOMES = frozenset(
    {"committed", "already_committed", "historical_committed", "historical_already_committed"}
)


@dataclass(frozen=True, slots=True)
class TerminalObservationProof:
    """Immutable proof of an accepted terminal observation."""

    outcome: str
    mode: str
    task_id: str
    attempt_id: str
    attempt_number: int
    fencing_token: int
    machine_name: str
    reservation_id: str | None
    process_identity: tuple[tuple[str, int | None], ...]
    phase: str
    reason: str
    exit_code: int
    termination_result: str | None
    task_revision: int
    attempt_digest: str
    transition_digest: str


@dataclass(frozen=True, slots=True)
class TerminalPublicationProof:
    """Immutable proof of an accepted terminal publication."""

    outcome: str
    mode: str
    task_id: str
    attempt_id: str
    attempt_number: int
    fencing_token: int
    machine_name: str
    reservation_id: str | None
    process_identity: tuple[tuple[str, int | None], ...]
    phase: str
    reason: str
    exit_code: int
    termination_result: str | None
    task_revision: int
    attempt_digest: str
    transition_digest: str


TerminalProof = TerminalObservationProof | TerminalPublicationProof


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


def _immutable_process_identity(parameters: Mapping[str, Any]) -> tuple[tuple[str, int | None], ...] | None:
    process_identity = parameters.get("process_identity")
    if not isinstance(process_identity, Mapping) or set(process_identity) != set(_PROCESS_IDENTITY_FIELDS):
        return None
    values = tuple((field, process_identity[field]) for field in _PROCESS_IDENTITY_FIELDS)
    if any(value is not None and type(value) is not int for _field, value in values):
        return None
    return values


def _terminal_proof_kwargs(
    *,
    parameters: Mapping[str, Any],
    machine_name: str,
    exit_code: int,
    source_revisions: object,
    phase: str,
    reason: str,
    termination_result: str | None,
) -> dict[str, Any] | None:
    if not isinstance(parameters, Mapping) or not has_exact_terminal_revisions(source_revisions):
        return None
    process_identity = _immutable_process_identity(parameters)
    if process_identity is None or type(exit_code) is not int:
        return None
    try:
        mode = parameters["mode"]
        task_id = parameters["task_id"]
        attempt_id = parameters["attempt_id"]
        attempt_number = parameters["attempt_number"]
        fencing_token = parameters["fencing_token"]
        reservation_id = parameters["reservation_id"]
    except KeyError:
        return None
    if (
        not isinstance(mode, str)
        or not isinstance(task_id, str)
        or not isinstance(attempt_id, str)
        or type(attempt_number) is not int
        or type(fencing_token) is not int
        or (reservation_id is not None and not isinstance(reservation_id, str))
        or not isinstance(machine_name, str)
        or not isinstance(phase, str)
        or not isinstance(reason, str)
        or (termination_result is not None and not isinstance(termination_result, str))
    ):
        return None
    transition_digest = authority_terminal_transition_digest(
        mode=mode,
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        fencing_token=fencing_token,
        machine_name=machine_name,
        reservation_id=reservation_id,
        process_identity=dict(process_identity),
        phase=phase,
        reason=reason,
        exit_code=exit_code,
        termination_result=termination_result,
    )
    return {
        "mode": mode,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "machine_name": machine_name,
        "reservation_id": reservation_id,
        "process_identity": process_identity,
        "phase": phase,
        "reason": reason,
        "exit_code": exit_code,
        "termination_result": termination_result,
        "task_revision": source_revisions["task"],
        "attempt_digest": source_revisions["attempt_digest"],
        "transition_digest": transition_digest,
    }


def build_terminal_observation_proof(
    *,
    evidence: Mapping[str, Any] | None,
    parameters: Mapping[str, Any],
    machine_name: str,
    exit_code: int,
) -> TerminalObservationProof | None:
    """Build an immutable proof from one exact terminal observation."""
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(evidence, Mapping)
        or not matches_terminal_observation_identity(evidence, parameters, machine_name)
    ):
        return None
    outcome = evidence.get("outcome")
    if outcome not in _TERMINAL_OBSERVATION_OUTCOMES:
        return None
    if parameters.get("mode") == "active" and outcome == "settled_terminal":
        # Preserve the existing active settled-history rule: either prior
        # termination marker can accompany an otherwise exact settled exit.
        target = terminal_transition_target(
            mode=parameters.get("mode"),
            exit_code=exit_code,
            cancel_requested=False,
        )
        expected_termination: tuple[str | None, ...] = (None, "already_exited")
    elif parameters.get("mode") == "detached_orphan":
        # Once the shared Attempt is terminal, its exact phase/result fields
        # are the durable record of whether termination was requested. The
        # Task flag may have changed while settled history remained intact.
        expected_cancel = evidence.get("attempt_phase") == "cancelled"
        target = terminal_transition_target(
            mode=parameters.get("mode"),
            exit_code=exit_code,
            cancel_requested=expected_cancel,
        )
        expected_termination = () if target is None else (target[2],)
    else:
        target = terminal_transition_target(
            mode=parameters.get("mode"),
            exit_code=exit_code,
            cancel_requested=evidence["cancel_requested"],
        )
        expected_termination = () if target is None else (target[2],)
    if target is None:
        return None
    phase, reason, _termination_result = target
    if evidence.get("attempt_phase") != phase:
        return None
    if outcome == "already_terminal" and evidence.get("task_phase") != phase:
        return None
    if outcome == "settled_terminal" and evidence.get("task_phase") is None:
        return None
    termination_result = evidence.get("termination_result")
    if not (
        evidence.get("execution_machine_name") == machine_name
        and evidence.get("reservation_machine_name") == machine_name
        and evidence.get("attempt_result_reason") == reason
        and evidence.get("attempt_exit_code") == exit_code
        and termination_result in expected_termination
    ):
        return None
    values = _terminal_proof_kwargs(
        parameters=parameters,
        machine_name=machine_name,
        exit_code=exit_code,
        source_revisions=evidence.get("source_revisions"),
        phase=phase,
        reason=reason,
        termination_result=termination_result,
    )
    if values is None:
        return None
    return TerminalObservationProof(outcome=outcome, **values)


def build_terminal_publication_proof(
    *,
    evidence: Mapping[str, Any] | None,
    parameters: Mapping[str, Any],
    machine_name: str,
    exit_code: int,
    transition: Mapping[str, Any] | None,
) -> TerminalPublicationProof | None:
    """Build an immutable proof from one exact terminal publication."""
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(transition, Mapping)
        or evidence.get("outcome") not in _TERMINAL_PUBLICATION_OUTCOMES
        or evidence.get("machine_name") != machine_name
        or evidence.get("authority_granted") is not False
        or not isinstance(evidence.get("local_effects"), (list, tuple))
        or evidence["local_effects"]
    ):
        return None
    if not isinstance(parameters, Mapping):
        return None
    identity_fields = (
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "reservation_id",
        "process_identity",
        "mode",
    )
    try:
        if not all(evidence.get(field) == parameters[field] for field in identity_fields):
            return None
        if not all(transition.get(field) == parameters[field] for field in identity_fields):
            return None
    except KeyError:
        return None
    if not all(
        evidence.get(evidence_field) == transition.get(transition_field)
        for evidence_field, transition_field in (
            ("phase", "phase"),
            ("transition_reason", "reason"),
            ("exit_code", "exit_code"),
            ("termination_result", "termination_result"),
            ("transition_digest", "transition_digest"),
            ("source_revisions", "source_revisions"),
        )
    ):
        return None
    transition_termination_result = transition.get("termination_result")
    if transition_termination_result not in {None, "already_exited"}:
        return None
    target = terminal_transition_target(
        mode=parameters.get("mode"),
        exit_code=exit_code,
        cancel_requested=transition_termination_result == "already_exited",
    )
    if target is None:
        return None
    phase, reason, expected_termination_result = target
    if (
        transition.get("phase") != phase
        or transition.get("reason") != reason
        or transition.get("exit_code") != exit_code
        or transition_termination_result != expected_termination_result
    ):
        return None
    source_revisions = evidence.get("source_revisions")
    if not has_exact_terminal_revisions(source_revisions):
        return None
    values = _terminal_proof_kwargs(
        parameters=parameters,
        machine_name=machine_name,
        exit_code=exit_code,
        source_revisions=source_revisions,
        phase=phase,
        reason=reason,
        termination_result=transition_termination_result,
    )
    if values is None or transition.get("transition_digest") != values["transition_digest"]:
        return None
    reservation_machine = evidence.get("reservation_machine_name")
    reservation_id = parameters.get("reservation_id")
    if reservation_machine not in {None, machine_name} or (
        reservation_id is not None and reservation_machine != machine_name
    ):
        return None
    historical = evidence.get("outcome") in {"historical_committed", "historical_already_committed"}
    if historical and (
        parameters.get("mode") != "active"
        or transition_termination_result is not None
        or evidence.get("lifecycle_event") is not None
    ):
        return None
    if not historical and not matches_terminal_lifecycle_event(
        evidence.get("lifecycle_event"),
        evidence.get("committed_revisions"),
        parameters,
        phase=phase,
        reason=reason,
        exit_code=exit_code,
    ):
        return None
    return TerminalPublicationProof(outcome=evidence["outcome"], **values)


def terminal_proof_matches_candidate(
    proof: object,
    *,
    parameters: Mapping[str, Any],
    machine_name: str,
    exit_code: int,
) -> bool:
    """Return whether an immutable proof matches the current terminal candidate."""
    if type(proof) is TerminalObservationProof:
        if proof.outcome not in _TERMINAL_OBSERVATION_OUTCOMES:
            return False
    elif type(proof) is TerminalPublicationProof:
        if proof.outcome not in _TERMINAL_PUBLICATION_OUTCOMES:
            return False
    else:
        return False
    if not isinstance(parameters, Mapping):
        return False
    if type(proof) is TerminalObservationProof and proof.outcome == "settled_terminal" and proof.mode == "active":
        target = terminal_transition_target(
            mode=parameters.get("mode"),
            exit_code=exit_code,
            cancel_requested=False,
        )
    else:
        target = terminal_transition_target(
            mode=parameters.get("mode"),
            exit_code=exit_code,
            cancel_requested=proof.termination_result == "already_exited",
        )
    if target is None or proof.phase != target[0] or proof.reason != target[1]:
        return False
    if (
        not (
            type(proof) is TerminalObservationProof
            and proof.outcome == "settled_terminal"
            and parameters.get("mode") == "active"
        )
        and proof.termination_result != target[2]
    ):
        return False
    values = _terminal_proof_kwargs(
        parameters=parameters,
        machine_name=machine_name,
        exit_code=exit_code,
        source_revisions={"task": proof.task_revision, "attempt_digest": proof.attempt_digest},
        phase=proof.phase,
        reason=proof.reason,
        termination_result=proof.termination_result,
    )
    return values is not None and all(getattr(proof, field) == value for field, value in values.items())


__all__ = [
    "build_terminal_observation_proof",
    "build_terminal_publication_proof",
    "build_terminal_transition",
    "has_exact_terminal_revisions",
    "matches_terminal_lifecycle_event",
    "matches_terminal_observation_identity",
    "terminal_proof_matches_candidate",
    "terminal_transition_target",
]
