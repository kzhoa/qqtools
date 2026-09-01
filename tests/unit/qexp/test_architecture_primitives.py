from __future__ import annotations

import pytest

from tests.qexp_architecture import (
    InjectedProtocolError,
    ProtocolPoint,
    ScenarioRunner,
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
    assert replay.events[-1].payload == {"point": ProtocolPoint.ATOMIC_REPLACE}
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
