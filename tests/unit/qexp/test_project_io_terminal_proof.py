from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from qqtools.plugins.qexp.agent.project_io_terminal_proof import (
    TerminalObservationProof,
    TerminalPublicationProof,
    TerminationObservationProof,
    build_terminal_observation_proof,
    build_terminal_publication_proof,
    build_terminal_transition,
    build_termination_observation_proof,
    has_exact_terminal_revisions,
    matches_terminal_lifecycle_event,
    matches_terminal_observation_identity,
    terminal_proof_matches_candidate,
    terminal_transition_target,
)


def _parameters(mode: str = "active") -> dict[str, object]:
    return {
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 2,
        "reservation_id": "reservation-a",
        "process_identity": {
            "wrapper_pid": 101,
            "wrapper_start_time_ticks": 5,
            "process_group_id": 101,
            "process_group_start_time_ticks": 5,
        },
        "mode": mode,
    }


def _revisions() -> dict[str, object]:
    return {"task": 3, "attempt_digest": "d" * 64}


def test_terminal_transition_builder_selects_exact_active_outcome() -> None:
    parameters = _parameters()
    transition = build_terminal_transition(
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=7,
        source_revisions=_revisions(),
        cancel_requested=True,
    )

    assert transition is not None
    assert transition["phase"] == "cancelled"
    assert transition["reason"] == "termination_process_already_exited"
    assert transition["termination_result"] == "already_exited"
    assert isinstance(transition["transition_digest"], str)


def test_terminal_target_rejects_unknown_mode_and_boolean_like_integer() -> None:
    assert terminal_transition_target(mode="unknown", exit_code=0, cancel_requested=False) is None
    assert terminal_transition_target(mode="active", exit_code=0, cancel_requested=1) is None


def test_terminal_proof_helpers_require_exact_identity_and_revision_witness() -> None:
    parameters = _parameters("detached_orphan")
    evidence = {
        "machine_name": "gpu-1",
        **parameters,
        "authority_granted": False,
        "local_effects": [],
        "source_revisions": _revisions(),
        "cancel_requested": False,
    }

    assert matches_terminal_observation_identity(evidence, parameters, "gpu-1")
    evidence["process_identity"] = {**parameters["process_identity"], "wrapper_pid": 202}
    assert not matches_terminal_observation_identity(evidence, parameters, "gpu-1")
    assert has_exact_terminal_revisions(_revisions())
    assert not has_exact_terminal_revisions({"task": 3, "attempt_digest": "short"})


def test_lifecycle_event_binds_committed_task_revision() -> None:
    parameters = _parameters()
    event = {
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "attempt_number": parameters["attempt_number"],
        "phase": "succeeded",
        "reason": "completed",
        "exit_code": 0,
        "task_revision": 3,
    }

    assert matches_terminal_lifecycle_event(
        event,
        _revisions(),
        parameters,
        phase="succeeded",
        reason="completed",
        exit_code=0,
    )
    event["task_revision"] = 4
    assert not matches_terminal_lifecycle_event(
        event,
        _revisions(),
        parameters,
        phase="succeeded",
        reason="completed",
        exit_code=0,
    )


def _terminal_observation_evidence(parameters: dict[str, object]) -> dict[str, object]:
    return {
        "outcome": "already_terminal",
        "machine_name": "gpu-1",
        **parameters,
        "authority_granted": False,
        "local_effects": [],
        "source_revisions": _revisions(),
        "cancel_requested": False,
        "task_phase": "succeeded",
        "attempt_phase": "succeeded",
        "execution_machine_name": "gpu-1",
        "reservation_machine_name": "gpu-1",
        "attempt_result_reason": "completed",
        "attempt_exit_code": 0,
        "termination_result": None,
    }


def test_observation_proof_is_immutable_and_rejects_candidate_mismatch() -> None:
    parameters = _parameters()
    proof = build_terminal_observation_proof(
        evidence=_terminal_observation_evidence(parameters),
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
    )

    assert isinstance(proof, TerminalObservationProof)
    assert terminal_proof_matches_candidate(
        proof,
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
    )
    assert all(not isinstance(getattr(proof, field.name), (dict, list, set)) for field in fields(proof))
    with pytest.raises(FrozenInstanceError):
        proof.outcome = "settled_terminal"  # type: ignore[misc]

    mismatched = _parameters()
    mismatched["process_identity"] = {
        **parameters["process_identity"],  # type: ignore[dict-item]
        "wrapper_pid": 202,
    }
    assert not terminal_proof_matches_candidate(
        proof,
        parameters=mismatched,
        machine_name="gpu-1",
        exit_code=0,
    )
    assert not terminal_proof_matches_candidate(
        True,
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
    )


def test_publication_proof_preserves_distinct_provenance() -> None:
    parameters = _parameters()
    transition = build_terminal_transition(
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
        source_revisions=_revisions(),
        cancel_requested=False,
    )
    assert transition is not None
    evidence = {
        "outcome": "committed",
        "machine_name": "gpu-1",
        **parameters,
        "authority_granted": False,
        "local_effects": [],
        "phase": "succeeded",
        "transition_reason": "completed",
        "exit_code": 0,
        "termination_result": None,
        "transition_digest": transition["transition_digest"],
        "source_revisions": _revisions(),
        "committed_revisions": _revisions(),
        "reservation_machine_name": "gpu-1",
        "lifecycle_event": {
            "task_id": parameters["task_id"],
            "attempt_id": parameters["attempt_id"],
            "attempt_number": parameters["attempt_number"],
            "phase": "succeeded",
            "reason": "completed",
            "exit_code": 0,
            "task_revision": 3,
        },
    }

    proof = build_terminal_publication_proof(
        evidence=evidence,
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
        transition=transition,
    )

    assert isinstance(proof, TerminalPublicationProof)
    assert not isinstance(proof, TerminalObservationProof)
    assert proof.outcome == "committed"
    assert terminal_proof_matches_candidate(
        proof,
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=0,
    )


def _confirmed_signal_decision(parameters: dict[str, object]) -> dict[str, object]:
    identity = parameters["process_identity"]
    return {
        "decision_id": "decision-a",
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "decision_token": parameters["fencing_token"],
        "authority_outcome": "holder_safe_deadline_elapsed",
        "reason": "holder_safe_deadline_elapsed",
        "state": "confirmed",
        "shared_commitment": "committed",
        "confirmation": "identity_absent",
        "process_group_id": identity["process_group_id"],
        "process_group_start_time_ticks": identity["process_group_start_time_ticks"],
        "signal_attempts": [{"signal": "SIGTERM", "at": "2026-10-09T00:00:00Z"}],
    }


def _terminated_evidence(parameters: dict[str, object]) -> dict[str, object]:
    return {
        **_terminal_observation_evidence(parameters),
        "attempt_phase": "cancelled",
        "task_phase": "cancelled",
        "attempt_result_reason": "terminated_by_agent",
        "attempt_exit_code": None,
        "termination_result": "terminated",
    }


@pytest.mark.parametrize(
    "outcome,task_phase",
    [("already_terminal", "cancelled"), ("settled_terminal", "queued"), ("settled_terminal", "running")],
)
@pytest.mark.parametrize("cancel_requested", [False, True])
def test_termination_proof_preserves_shared_null_and_separate_runner_exit(outcome, task_phase, cancel_requested):
    parameters = _parameters()
    evidence = _terminated_evidence(parameters)
    evidence.update(outcome=outcome, task_phase=task_phase, cancel_requested=cancel_requested)
    proof = build_termination_observation_proof(
        evidence=evidence,
        parameters=parameters,
        machine_name="gpu-1",
        exit_code=-15,
        decision=_confirmed_signal_decision(parameters),
    )
    assert isinstance(proof, TerminationObservationProof)
    assert proof.shared_exit_code is None
    assert proof.exit_code == -15
    assert (proof.phase, proof.reason, proof.termination_result) == ("cancelled", "terminated_by_agent", "terminated")
    assert proof.decision_id == "decision-a"
    assert terminal_proof_matches_candidate(proof, parameters=parameters, machine_name="gpu-1", exit_code=-15)
    assert not terminal_proof_matches_candidate(proof, parameters=parameters, machine_name="gpu-1", exit_code=0)
    assert not terminal_proof_matches_candidate(
        proof, parameters={**parameters, "fencing_token": 3}, machine_name="gpu-1", exit_code=-15
    )
    with pytest.raises(FrozenInstanceError):
        proof.decision_id = "other"


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision_id", ""),
        ("task_id", "other"),
        ("attempt_id", "other"),
        ("decision_token", 3),
        ("decision_token", True),
        ("state", "pending"),
        ("shared_commitment", "pending"),
        ("confirmation", "identity_present"),
        ("process_group_id", 202),
        ("process_group_start_time_ticks", 6),
        ("signal_attempts", []),
        ("signal_attempts", [{"signal": "unknown"}]),
        ("reason", "cancellation_requested"),
    ],
)
def test_termination_proof_rejects_unproven_or_conflicting_decision(field, value):
    parameters = _parameters()
    decision = _confirmed_signal_decision(parameters)
    decision[field] = value
    assert (
        build_termination_observation_proof(
            evidence=_terminated_evidence(parameters),
            parameters=parameters,
            machine_name="gpu-1",
            exit_code=-15,
            decision=decision,
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "current"),
        ("attempt_phase", "failed"),
        ("task_phase", "queued"),
        ("attempt_result_reason", "nonzero_exit"),
        ("attempt_exit_code", -15),
        ("termination_result", "already_exited"),
        ("fencing_token", 3),
        ("authority_granted", True),
        ("execution_machine_name", "other"),
        ("reservation_machine_name", "other"),
        ("source_revisions", {"task": 3, "attempt_digest": "short"}),
    ],
)
def test_termination_proof_does_not_relax_shared_result_or_identity_checks(field, value):
    parameters = _parameters()
    evidence = _terminated_evidence(parameters)
    evidence[field] = value
    assert (
        build_termination_observation_proof(
            evidence=evidence,
            parameters=parameters,
            machine_name="gpu-1",
            exit_code=-15,
            decision=_confirmed_signal_decision(parameters),
        )
        is None
    )
