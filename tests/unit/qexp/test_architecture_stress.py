from __future__ import annotations

import pytest

from tests.qexp_architecture import (
    CrashWindow,
    ProtocolDisposition,
    ProtocolPoint,
    ReferenceScenarioRunner,
    SimulatedParticipantCrash,
    SimulatedRuntime,
    TraceEnvelope,
    plan_crash_window_recovery,
)


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("seed", range(64))
def test_reference_model_generated_scenarios_replay_for_expanded_seed_set(seed: int) -> None:
    runner = ReferenceScenarioRunner(seed=seed)
    commands = runner.generate(maximum_actions=12)

    model = runner.run(commands)
    replayed = ReferenceScenarioRunner.replay(TraceEnvelope.from_json(runner.trace.to_json()))

    assert model.task_state("task-1") == replayed.task_state("task-1")


@pytest.mark.parametrize("point", list(ProtocolPoint))
def test_simulated_protocol_crash_matrix_reaches_every_stable_boundary(
    point: ProtocolPoint,
) -> None:
    runtime = SimulatedRuntime()
    runtime.schedule(point, ProtocolDisposition.CRASH)

    with pytest.raises(SimulatedParticipantCrash, match=point):
        runtime.yield_at(point, participant="machine-a")


def test_crash_window_recovery_table_covers_every_runtime_spec_window() -> None:
    assert {plan_crash_window_recovery(window).invariant for window in CrashWindow} == {
        "no_operation_or_task",
        "no_claimable_task",
        "no_partial_visibility",
        "worker_inactive_until_commit",
        "task_remains_queued",
        "no_launch_without_attempt",
        "no_process_without_authorization",
        "no_untracked_process",
        "one_attempt_identity",
        "terminal_truth_converges",
        "no_automatic_launch",
        "no_automatic_replacement",
        "attempt_identity_preserved",
        "stale_token_cannot_mutate_truth",
        "queued_work_not_stranded",
        "one_group_lock_order_wins",
        "forbidden_recovery_is_quarantined",
        "pending_machine_set_preserved",
    }
