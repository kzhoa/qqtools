"""A failed handoff observation must not release an already accepted launch."""

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, runner, submit
from qqtools.plugins.qexp.agent import dispatch_loop
from qqtools.plugins.qexp.executor import LaunchHandoff
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    claim_task,
    fail_attempt,
    resume_starting_attempt,
    run_dispatch_cycle,
)

pytestmark = pytest.mark.integration


def prepare(tmp_path, *, is_cpu=False):
    cfg = init_shared_root(tmp_path / ".qexp", "local", runtime_root=tmp_path / "runtime")
    if is_cpu:
        set_cpu_lane_capacity(cfg.runtime_root, capacity=1)
    task = submit(cfg, ["true"], requested_gpus=0 if is_cpu else 1, requested_cpus=1 if is_cpu else None)
    return cfg, task


def authorized_attempt(cfg, task, *, is_cpu=False):
    attempt = claim_task(cfg, task.task_id, [] if is_cpu else [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    return AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))


def assert_reserved(cfg, task, *, is_cpu=False):
    claim = load_task(cfg, task.task_id).claim_control["active_claim"]
    path = local_paths(cfg.runtime_root)["cpu_active" if is_cpu else "active"] / f"{claim['reservation_id']}.json"
    assert read_json(path)["reservation"]["attempt_id"] == claim["attempt_id"]


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("is_recovery", [False, True])
def test_dispatch_handoff_failure_preserves_runner_after_intent(tmp_path, monkeypatch, is_cpu, is_recovery):
    cfg, task = prepare(tmp_path, is_cpu=is_cpu)
    if is_recovery:
        authorized_attempt(cfg, task, is_cpu=is_cpu)
    entered, resume = Event(), Event()
    workers, outcomes, spawned = [], [], []

    def pause_after_intent(*_args, **_kwargs):
        entered.set()
        if not resume.wait(10):
            outcomes.append(TimeoutError("test did not release the runner"))

    def spawn(*_args, **_kwargs):
        spawned.append(load_task(cfg, task.task_id).state["projection"])
        return SimpleNamespace(pid=99999989, wait=lambda: 0)

    class FailedHandoff:
        def launch_attempt(self, local_cfg, task_id, attempt):
            def run():
                try:
                    outcomes.append(
                        runner.run_attempt(
                            local_cfg,
                            task_id,
                            attempt.attempt_id,
                            attempt.current_fencing_token,
                            attempt.authorization["launch_id"],
                            popen_factory=spawn,
                        )
                    )
                except BaseException as exc:
                    outcomes.append(exc)

            thread = Thread(target=run)
            workers.append(thread)
            thread.start()
            assert entered.wait(5)
            raise RuntimeError("handoff observation failed after intent publication")

    monkeypatch.setattr(runner, "prepare_progress_channel", pause_after_intent)
    arguments = {"available_cpus": 1} if is_cpu else {"available_gpus": [0]}
    try:
        assert run_dispatch_cycle(cfg, executor=FailedHandoff(), **arguments) == []
        current = load_task(cfg, task.task_id)
        assert current.state["projection"] == "running"
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "starting"
        assert_reserved(cfg, task, is_cpu=is_cpu)
        assert not spawned
        assert run_dispatch_cycle(cfg, executor=FailedHandoff(), **arguments) == []
        assert len(workers) == 1
    finally:
        resume.set()
        for thread in workers:
            thread.join(5)
            assert not thread.is_alive()
    assert spawned == ["running"]
    assert outcomes == [0]
    identity = current.attempt_control["current_attempt_id"]
    assert read_json(runner.observation_path(cfg, identity))["exit_observation"]["observed_exit_code"] == 0
    assert load_task(cfg, task.task_id).attempt_control["current_attempt_number"] == 1


@pytest.mark.parametrize("lane", ["launch_intents", "registrations", "processes", "observations"])
def test_batched_handoff_failure_retains_existing_launch_evidence(tmp_path, lane):
    cfg, task = prepare(tmp_path)
    attempt = authorized_attempt(cfg, task)
    path = local_paths(cfg.runtime_root)[lane] / f"{attempt.attempt_id}.json"
    atomic_replace(path, {})  # Even ambiguous evidence requires recovery, not a negative assumption.
    handoff = LaunchHandoff(attempt.attempt_id, runner.launch_intent_path(cfg, attempt.attempt_id), 0)

    class FailedHandoff:
        def wait_for_launch_handoffs(self, _handoffs):
            return {handoff: RuntimeError("handoff failed")}

    batch = dispatch_loop._LaunchHandoffBatch(FailedHandoff(), cfg.runtime_root)
    batch._pending.append(
        dispatch_loop._PendingLaunchHandoff(
            cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "project", handoff
        )
    )
    result = {"project": {"launched": [task.task_id], "status": "dispatched"}}
    batch.finish(result)
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    assert_reserved(cfg, task)
    assert path.exists()
    assert result["project"]["status"] == "error"


def test_failure_before_intent_prevents_delayed_runner_and_replay(tmp_path):
    cfg, task = prepare(tmp_path)
    attempt = authorized_attempt(cfg, task)
    assert fail_attempt(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        "executor_launch_failed",
        should_require_unstarted=True,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("compensated Attempt created a process")

    with pytest.raises(RuntimeError, match="not authorized"):
        runner.run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            attempt.authorization["launch_id"],
            popen_factory=forbidden,
        )
    assert not runner.launch_intent_path(cfg, attempt.attempt_id).exists()
    assert load_task(cfg, task.task_id).state["projection"] == "failed"
    assert not list(local_paths(cfg.runtime_root)["active"].glob("*.json"))
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=SimpleNamespace(launch_attempt=forbidden)) == []
    assert load_task(cfg, task.task_id).attempt_control["current_attempt_number"] == 1


@pytest.mark.parametrize("evidence", ["inaccessible", "dangling_link", "final_observation"])
def test_uncertain_or_final_evidence_cannot_be_negative_launch_proof(tmp_path, monkeypatch, evidence):
    cfg, task = prepare(tmp_path)
    attempt = authorized_attempt(cfg, task)
    path = runner.launch_intent_path(cfg, attempt.attempt_id)
    if evidence == "inaccessible":
        original = Path.stat

        def inaccessible(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("launch evidence cannot be inspected")
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", inaccessible)
    elif evidence == "dangling_link":
        path.symlink_to(tmp_path / "missing")
    else:
        runner._publish_exit_observation(cfg, attempt.attempt_id, 0, task_id=task.task_id)

    def compensate():
        return fail_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            "executor_launch_failed",
            should_require_unstarted=True,
        )

    if evidence == "inaccessible":
        with pytest.raises(PermissionError, match="cannot be inspected"):
            compensate()
        with pytest.raises(PermissionError, match="cannot be inspected"):
            resume_starting_attempt(cfg, task.task_id)
    else:
        assert not compensate()
        assert resume_starting_attempt(cfg, task.task_id) is None
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    assert_reserved(cfg, task)


def test_another_machine_cannot_compensate_from_its_empty_local_inbox(tmp_path):
    cfg, task = prepare(tmp_path)
    attempt = authorized_attempt(cfg, task)
    other = replace(cfg, machine_name="other", runtime_root=tmp_path / "other-runtime")
    assert not fail_attempt(
        other,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        "executor_launch_failed",
        should_require_unstarted=True,
    )
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    assert_reserved(cfg, task)
