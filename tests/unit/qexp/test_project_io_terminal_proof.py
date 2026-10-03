from __future__ import annotations

from qqtools.plugins.qexp.agent.project_io_terminal_proof import (
    build_terminal_transition,
    has_exact_terminal_revisions,
    matches_terminal_lifecycle_event,
    matches_terminal_observation_identity,
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
