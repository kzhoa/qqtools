from __future__ import annotations

import pytest

from tests.qexp_architecture import (
    InjectedProtocolError,
    ProtocolDisposition,
    ProtocolPoint,
    QexpReferenceModel,
    ReferenceCommand,
    ReferenceScenarioRunner,
    ScenarioRunner,
    SimulatedParticipantCrash,
    SimulatedProtocolPause,
    SimulatedRuntime,
    TraceEnvelope,
)


def test_simulated_runtime_cas_conflict_preserves_newer_revision() -> None:
    runtime = SimulatedRuntime(seed=7)

    assert runtime.cas("task", None, {"state": "queued"}) is not None
    snapshot = runtime.read("task")
    assert snapshot is not None
    assert runtime.cas("task", snapshot.revision, {"state": "running"}) is not None
    assert runtime.cas("task", snapshot.revision, {"state": "cancelled"}) is None
    assert runtime.read("task").value == {"state": "running"}


def test_simulated_runtime_records_and_injects_protocol_fault() -> None:
    runtime = SimulatedRuntime(seed=9)
    runtime.inject_fault(ProtocolPoint.ATOMIC_REPLACE)

    with pytest.raises(InjectedProtocolError):
        runtime.yield_at(ProtocolPoint.ATOMIC_REPLACE, participant="machine-a")

    replay = TraceEnvelope.from_json(runtime.trace.to_json())
    assert replay.events[-1].payload == {
        "point": ProtocolPoint.ATOMIC_REPLACE,
        "disposition": ProtocolDisposition.CONTINUE,
    }
    assert replay.events[-1].participant == "machine-a"


def test_trace_redacts_nested_secret_values() -> None:
    runtime = SimulatedRuntime()
    runtime.trace.record("action", {"command": {"api_key": "secret-value"}}, runtime.clock)

    assert "secret-value" not in runtime.trace.to_json()


def test_scenario_runner_shrinks_to_minimal_failing_trace() -> None:
    def harmless(runtime: SimulatedRuntime) -> None:
        runtime.advance(1)

    def fail(runtime: SimulatedRuntime) -> None:
        runtime.advance(1)
        raise AssertionError("expected failure")

    def fails(actions):
        try:
            ScenarioRunner(SimulatedRuntime()).run(actions)
        except AssertionError:
            return True
        return False

    minimized = ScenarioRunner.shrink([harmless, fail, harmless], fails)
    assert minimized == [fail]


def test_reference_model_rejects_claim_before_submission_commit() -> None:
    model = QexpReferenceModel()
    model.create_submission("submission-1")
    model.stage_task("task-1", "submission-1")

    assert model.claim("task-1") is None
    model.commit_submission("submission-1")
    assert model.claim("task-1") == 1
    assert model.claim("task-1") is None


def test_reference_model_fences_stale_launch_and_terminal_writes() -> None:
    model = QexpReferenceModel()
    model.create_submission("submission-1")
    model.stage_task("task-1", "submission-1")
    model.commit_submission("submission-1")
    token = model.claim("task-1")

    assert token == 1
    assert model.authorize_launch("task-1", 0) is None
    assert model.authorize_launch("task-1", token).kind == "launch"
    assert model.publish_running("task-1", token) is True
    assert model.publish_terminal("task-1", 0, cancelled=False) is False
    assert model.task_state("task-1") == "running"


def test_reference_model_cancel_linearizes_against_launch_authorization() -> None:
    model = QexpReferenceModel()
    model.create_submission("submission-1")
    model.stage_task("task-1", "submission-1")
    model.commit_submission("submission-1")
    token = model.claim("task-1")

    assert token == 1
    assert model.cancel("task-1") is None
    assert model.authorize_launch("task-1", token) is None
    assert model.task_state("task-1") == "cancelled"


def test_simulated_runtime_schedules_a_cas_conflict_without_overwriting_truth() -> None:
    runtime = SimulatedRuntime()
    runtime.schedule(ProtocolPoint.CAS, ProtocolDisposition.CONFLICT)

    assert runtime.cas("task", None, {"state": "queued"}) is None
    assert runtime.read("task") is None


def test_simulated_runtime_exposes_pause_and_crash_at_explicit_boundaries() -> None:
    runtime = SimulatedRuntime()
    runtime.schedule(ProtocolPoint.LOCK_ACQUIRE, ProtocolDisposition.PAUSE)
    runtime.schedule(ProtocolPoint.PROCESS_CREATE, ProtocolDisposition.CRASH)

    with pytest.raises(SimulatedProtocolPause, match="lock_acquire"):
        runtime.acquire("scheduler")
    with pytest.raises(SimulatedParticipantCrash, match="process_create"):
        runtime.yield_at(ProtocolPoint.PROCESS_CREATE, participant="machine-a")


def test_reference_scenario_trace_replays_serialized_commands_with_the_same_seed() -> None:
    commands = [
        ReferenceCommand("create_submission", submission_id="submission-1"),
        ReferenceCommand("stage_task", task_id="task-1", submission_id="submission-1"),
        ReferenceCommand("commit_submission", submission_id="submission-1"),
        ReferenceCommand("claim", task_id="task-1"),
        ReferenceCommand("authorize_launch", task_id="task-1", fencing_token=1),
        ReferenceCommand("publish_running", task_id="task-1", fencing_token=1),
    ]
    runner = ReferenceScenarioRunner(seed=41)

    model = runner.run(commands)
    replayed = ReferenceScenarioRunner.replay(TraceEnvelope.from_json(runner.trace.to_json()))

    assert runner.trace.seed == 41
    assert model.task_state("task-1") == replayed.task_state("task-1") == "running"


def test_reference_scenario_rejects_a_trace_command_with_an_invalid_fencing_token() -> None:
    model = QexpReferenceModel()

    with pytest.raises(ValueError, match="positive fencing_token"):
        model.execute(ReferenceCommand("authorize_launch", task_id="task-1", fencing_token=0))


def test_reference_scenario_generator_is_seeded_and_only_emits_valid_actions() -> None:
    first = ReferenceScenarioRunner(seed=17)
    second = ReferenceScenarioRunner(seed=17)

    generated = first.generate(maximum_actions=8)

    assert generated == second.generate(maximum_actions=8)
    assert len(generated) <= 8
    assert first.run(generated).task_state("task-1") in {
        "claimed",
        "starting",
        "running",
        "cancelled",
        "finished",
    }
