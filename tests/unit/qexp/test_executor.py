import time
from pathlib import Path
from threading import Timer
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.executor import Executor, LaunchHandle, LaunchHandoff
from qqtools.plugins.qexp.launch_policy import set_launch_handoff_policy
from qqtools.plugins.qexp.runtime.records import SCHEMA_VERSION, AttemptRecord


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid

    def wait(self):
        return 0


def _attempt() -> AttemptRecord:
    return AttemptRecord.from_dict(
        {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "revision": 1,
                "created_at": "2026-07-24T00:00:00Z",
                "updated_at": "2026-07-24T00:00:00Z",
                "updated_by": {"actor_type": "test", "machine_name": "gpu-1", "process_id": "0"},
            },
            "attempt": {
                "attempt_id": "task-1-attempt-1",
                "task_id": "task-1",
                "attempt_number": 1,
                "phase": "claimed",
                "machine_name": "gpu-1",
                "assigned_gpus": [0],
                "reservation_id": "res-1",
                "current_fencing_token": 7,
                "token_history": [7],
                "lease": {
                    "claimed_at": "2026-07-24T00:00:00Z",
                    "renewed_at": "2026-07-24T00:00:00Z",
                    "expires_at": "2026-07-24T00:01:00Z",
                    "clock_evidence": {
                        "clock_error_bound_seconds": 0.1,
                        "clock_provider": "chrony",
                        "clock_observation_id": "test-observation",
                    },
                },
                "authority_mode": "bounded_lease",
                "authorization": {
                    "group_name": None,
                    "group_dispatch_epoch": None,
                    "group_worker_set_epoch": None,
                    "launch_id": "launch-1",
                },
                "process": {
                    "wrapper_pid": None,
                    "process_group_id": None,
                    "tmux_reference": None,
                    "local_process_manifest": "",
                    "log_references": [],
                },
                "termination": {
                    "requested_by_operation_id": None,
                    "requested_at": None,
                    "acknowledged_at": None,
                    "result": None,
                },
                "timestamps": {
                    "launch_authorized_at": None,
                    "process_created_at": None,
                    "running_at": None,
                    "orphaned_at": None,
                    "recovered_at": None,
                    "finished_at": None,
                },
                "result": {"exit_code": None, "signal": None, "category": None, "reason": None},
            },
        }
    )


def _cfg(tmp_path: Path) -> RootConfig:
    return RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")


def test_executor_uses_tmux_when_available(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QEXP_CURRENT_AGENT_ENV", "current")
    created: list[tuple[str, str, str, str]] = []

    def create_window(
        task_id: str,
        session_name: str,
        start_directory: str,
        initial_command: str,
    ) -> str:
        created.append((task_id, session_name, start_directory, initial_command))
        return "@7"

    def spawn_runner(_argv, **kwargs):
        assert kwargs["env"]["QEXP_CURRENT_AGENT_ENV"] == "current"
        intent = tmp_path / "rt" / "launch-intents" / "task-1-attempt-1.json"
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.touch()
        return _FakeProcess(4321)

    executor = Executor(
        create_window=create_window,
        send_command=lambda *_args: (_ for _ in ()).throw(AssertionError("runner command must not be injected")),
        destroy_window=lambda window_id: None,
        check_window=lambda window_id: True,
        tmux_available=lambda: True,
        spawn_runner=spawn_runner,
        observer_decision=lambda _cfg, _task_id: {"enabled": True, "source": "task_override"},
    )

    result = executor.launch_attempt(_cfg(tmp_path), "task-1", _attempt())

    assert result == "@7"
    assert created[0][:3] == ("task-1", "experiments", str(tmp_path))
    assert "tail" in created[0][3]
    assert "--pid=4321" in created[0][3]


def test_executor_falls_back_to_detached_runner_without_tmux(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QEXP_CURRENT_AGENT_ENV", "detached-current")
    spawned: list[dict[str, object]] = []
    diagnostics = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.executor.append_launch_diagnostic",
        lambda *_args: diagnostics.append(_args[-1]) or True,
    )

    def fake_spawn(argv, **kwargs):
        spawned.append({"argv": argv, **kwargs})
        intent = tmp_path / "rt" / "launch-intents" / "task-1-attempt-1.json"
        intent.parent.mkdir(parents=True)
        intent.touch()
        return _FakeProcess(4321)

    executor = Executor(
        create_window=lambda *args: (_ for _ in ()).throw(AssertionError("tmux path should not be used")),
        send_command=lambda *args: (_ for _ in ()).throw(AssertionError("tmux path should not be used")),
        destroy_window=lambda window_id: None,
        check_window=lambda window_id: False,
        tmux_available=lambda: False,
        spawn_runner=fake_spawn,
        observer_decision=lambda _cfg, _task_id: {"enabled": True, "source": "task_override"},
    )
    cfg = _cfg(tmp_path)

    result = executor.launch_attempt(cfg, "task-1", _attempt())

    assert result == "pid:4321"
    assert len(spawned) == 1
    assert spawned[0]["argv"] == executor.build_runner_argv(cfg, "task-1", "task-1-attempt-1", 7, "launch-1")
    assert spawned[0]["cwd"] == str(cfg.project_root)
    assert spawned[0]["env"]["QEXP_CURRENT_AGENT_ENV"] == "detached-current"
    assert spawned[0]["start_new_session"] is True
    assert diagnostics == ["tmux launch observer unavailable: tmux/libtmux unavailable"]


def test_disabled_observer_returns_before_any_tmux_probe_or_command(tmp_path: Path, monkeypatch):
    diagnostics = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.executor.append_launch_diagnostic",
        lambda *_args: diagnostics.append(_args[-1]) or True,
    )
    executor = Executor(
        create_window=lambda *_args: pytest.fail("disabled observer must not create a window"),
        tmux_available=lambda: pytest.fail("disabled observer must not probe tmux"),
        observer_decision=lambda _cfg, _task_id: {
            "enabled": False,
            "source": "default",
            "diagnostic_reason": "malformed stored override",
        },
    )
    handle = LaunchHandle("detached", _FakeProcess(4321), runner_process=_FakeProcess(4321))

    executor.attach_observer(_cfg(tmp_path), "task-1", "attempt-1", handle)

    assert handle.backend == "detached"
    assert handle.observer_window_id is None
    assert diagnostics == ["tmux observer policy unavailable: malformed stored override"]


def test_observer_decision_is_resolved_per_task_without_executor_cache(tmp_path: Path):
    decisions = {"visible": True, "quiet": False}
    probes = []
    windows = []

    executor = Executor(
        create_window=lambda task_id, *_args: windows.append(task_id) or f"@{len(windows)}",
        tmux_available=lambda: probes.append(None) or True,
        observer_decision=lambda _cfg, task_id: {"enabled": decisions[task_id], "source": "task_override"},
    )

    executor.attach_observer(
        _cfg(tmp_path),
        "visible",
        "visible-attempt",
        LaunchHandle("detached", _FakeProcess(1001), runner_process=_FakeProcess(1001)),
    )
    executor.attach_observer(
        _cfg(tmp_path),
        "quiet",
        "quiet-attempt",
        LaunchHandle("detached", _FakeProcess(1002), runner_process=_FakeProcess(1002)),
    )

    assert windows == ["visible"]
    assert len(probes) == 1


def test_executor_accepts_handoff_delayed_beyond_previous_two_second_limit(tmp_path: Path):
    cfg = _cfg(tmp_path)
    set_launch_handoff_policy(cfg, 3)
    intent = tmp_path / "rt" / "launch-intents" / "task-1-attempt-1.json"
    timer: Timer | None = None

    def fake_spawn(_argv, **_kwargs):
        nonlocal timer

        def publish_intent():
            intent.parent.mkdir(parents=True, exist_ok=True)
            intent.touch()

        timer = Timer(2.1, publish_intent)
        timer.start()
        return _FakeProcess(4321)

    executor = Executor(tmux_available=lambda: False, spawn_runner=fake_spawn)

    try:
        assert executor.launch_attempt(cfg, "task-1", _attempt()) == "pid:4321"
    finally:
        if timer is not None:
            timer.join()


def test_executor_rejects_runner_without_launch_handoff(tmp_path: Path):
    executor = Executor(tmux_available=lambda: False, spawn_runner=lambda *_args, **_kwargs: _FakeProcess(4321))

    with pytest.raises(RuntimeError, match="did not publish launch intent"):
        executor._wait_for_launch_intent(_cfg(tmp_path), "task-1-attempt-1", timeout_seconds=0.01)


def test_executor_reports_only_failed_handoffs_with_duplicate_attempt_ids(tmp_path: Path):
    published = tmp_path / "project-a" / "published.json"
    published.parent.mkdir()
    published.touch()
    published_handoff = LaunchHandoff("shared-attempt", published, 0.0)
    missing_handoff = LaunchHandoff("shared-attempt", tmp_path / "project-b" / "missing.json", 0.0)

    failures = Executor.wait_for_launch_handoffs([published_handoff, missing_handoff])

    assert list(failures) == [missing_handoff]
    assert str(failures[missing_handoff]) == "runner did not publish launch intent for 'shared-attempt'"


def test_launch_batch_isolates_duplicate_attempt_ids_across_projects(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import dispatch_loop

    successful_path = tmp_path / "project-a" / "intent.json"
    successful_path.parent.mkdir()
    successful_path.touch()
    successful = LaunchHandoff("shared-attempt", successful_path, 0.0)
    failed = LaunchHandoff("shared-attempt", tmp_path / "project-b" / "intent.json", 0.0)

    class _BatchExecutor:
        def wait_for_launch_handoffs(self, _handoffs):
            raise AssertionError("machine launch handoffs must never block")

    cfg_a = RootConfig(tmp_path / "a" / ".qexp", tmp_path / "a", "gpu-1", tmp_path / "a" / "rt")
    cfg_b = RootConfig(tmp_path / "b" / ".qexp", tmp_path / "b", "gpu-1", tmp_path / "b" / "rt")
    batch = dispatch_loop._LaunchHandoffBatch(_BatchExecutor(), tmp_path / "machine")
    batch._pending = [
        dispatch_loop._PendingLaunchHandoff(cfg_a, "task", "shared-attempt", 7, "project-a", successful),
        dispatch_loop._PendingLaunchHandoff(cfg_b, "task", "shared-attempt", 8, "project-b", failed),
    ]
    failed_attempts = []

    def record_failure(cfg, task_id, attempt_id, fencing_token, reason, **_kwargs):
        failed_attempts.append((cfg.shared_root, task_id, attempt_id, fencing_token, reason))
        return True

    monkeypatch.setattr(dispatch_loop, "fail_attempt", record_failure)
    results = {
        "project-a": {"launched": ["task"], "status": "dispatched"},
        "project-b": {"launched": ["task"], "status": "dispatched"},
    }

    batch.finish(results)

    assert failed_attempts == [(cfg_b.shared_root, "task", "shared-attempt", 8, "executor_launch_handoff_timeout")]
    assert results["project-a"] == {"launched": ["task"], "status": "dispatched"}
    assert results["project-b"]["launched"] == []
    assert results["project-b"]["status"] == "error"
    assert results["project-b"]["error"] == "runner did not publish launch intent for 'shared-attempt'"


def test_launch_batch_cleans_exact_handle_only_after_compensation_commits(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import dispatch_loop

    handoff = LaunchHandoff("attempt", tmp_path / "missing.json", 0.0)
    process = _FakeProcess(4321)
    handle = LaunchHandle("detached", process)
    cleaned = []

    class _BatchExecutor:
        def cleanup_launch(self, candidate):
            cleaned.append(candidate)

    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")
    batch = dispatch_loop._LaunchHandoffBatch(_BatchExecutor(), tmp_path / "machine")
    pending = dispatch_loop._PendingLaunchHandoff(cfg, "task", "attempt", 7, "project", handoff, handle)
    result = {"project": {"launched": ["task"], "status": "dispatched"}}

    monkeypatch.setattr(dispatch_loop, "fail_attempt", lambda *_args, **_kwargs: False)
    batch._pending = [pending]
    batch.finish(result)
    assert cleaned == []

    result = {"project": {"launched": ["task"], "status": "dispatched"}}
    monkeypatch.setattr(dispatch_loop, "fail_attempt", lambda *_args, **_kwargs: True)
    batch._pending = [pending]
    batch.finish(result)
    assert cleaned == [handle]


def test_machine_launch_batch_retains_future_handoff_across_cycles(tmp_path: Path):
    from qqtools.plugins.qexp.agent import dispatch_loop

    runtime = MachineRuntime(tmp_path / "machine")
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")
    intent = tmp_path / "intent.json"
    handoff = LaunchHandoff("attempt", intent, time.monotonic() + 60)
    attached = []

    class _BatchExecutor:
        def wait_for_launch_handoffs(self, _handoffs):
            raise AssertionError("machine launch handoffs must never block")

        def attach_observer(self, *args):
            attached.append(args)

    pending = dispatch_loop._PendingLaunchHandoff(cfg, "task", "attempt", 7, "project", handoff)
    runtime.pending_launch_handoffs[("project", "attempt")] = pending

    dispatch_loop._LaunchHandoffBatch(_BatchExecutor(), runtime).finish()
    assert runtime.pending_launch_handoffs == {("project", "attempt"): pending}

    intent.touch()
    dispatch_loop._LaunchHandoffBatch(_BatchExecutor(), runtime).finish()
    assert runtime.pending_launch_handoffs == {}
    assert attached == []  # There is no exact launch handle to observe.


def test_machine_launch_batch_retries_compensation_errors(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import dispatch_loop

    clock = [10.0]
    monkeypatch.setattr(dispatch_loop.time, "monotonic", lambda: clock[0])
    runtime = MachineRuntime(tmp_path / "machine")
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")
    handoff = LaunchHandoff("attempt", tmp_path / "missing.json", 0.0)
    pending = dispatch_loop._PendingLaunchHandoff(cfg, "task", "attempt", 7, "project", handoff)
    runtime.pending_launch_handoffs[("project", "attempt")] = pending
    calls = []

    def fail_once(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise OSError("authority temporarily unavailable")
        return False

    monkeypatch.setattr(dispatch_loop, "fail_attempt", fail_once)
    batch = dispatch_loop._LaunchHandoffBatch(Executor(), runtime)
    batch.finish()
    assert set(runtime.pending_launch_handoffs) == {("project", "attempt")}
    assert runtime.pending_launch_handoffs[("project", "attempt")].retry_failures == 1
    assert 0 < runtime.pending_launch_wait_seconds(5.0) <= 0.1

    clock[0] += 0.11
    batch.finish()
    assert runtime.pending_launch_handoffs == {}
    assert len(calls) == 2


def test_starting_recovery_skips_attempt_with_pending_handoff(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import helpers

    runtime = MachineRuntime(tmp_path / "machine")
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")
    identity = SimpleNamespace(
        project_id="project",
        task_id="task",
        attempt_id="attempt",
        fencing_token=7,
    )
    monkeypatch.setattr(helpers.ReservationIdentity, "from_record", lambda _record: identity)
    monkeypatch.setattr(
        helpers,
        "resume_starting_attempt",
        lambda *_args, **_kwargs: pytest.fail("pending attempt must not be recovered"),
    )

    recovered = helpers._recover_starting_reservations(
        runtime,
        {"project": cfg},
        ({"reservation": "opaque"},),
        Executor(),
        excluded_pending={("project", "attempt")},
    )

    assert recovered == {}


def test_starting_recovery_registers_nonblocking_handoff(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import helpers

    runtime = MachineRuntime(tmp_path / "machine")
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")
    identity = SimpleNamespace(
        project_id="project",
        task_id="task",
        attempt_id="attempt",
        fencing_token=7,
    )
    attempt = SimpleNamespace(attempt_id="attempt", current_fencing_token=7)
    launched = []
    monkeypatch.setattr(helpers.ReservationIdentity, "from_record", lambda _record: identity)
    monkeypatch.setattr(helpers, "resume_starting_attempt", lambda *_args, **_kwargs: attempt)

    class _BlockingExecutor:
        def launch_attempt(self, *_args, **_kwargs):
            pytest.fail("machine recovery must not use the synchronous launch API")

    recovered = helpers._recover_starting_reservations(
        runtime,
        {"project": cfg},
        ({"reservation": "opaque"},),
        _BlockingExecutor(),
        launch_recovered=lambda *args: launched.append(args),
    )

    assert recovered == {"project": ["task"]}
    assert launched == [(cfg, "task", attempt, "project")]


def test_machine_runtime_wait_is_bounded_by_earliest_pending_deadline(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp.agent import context

    runtime = MachineRuntime(tmp_path / "machine")
    runtime.pending_launch_handoffs[("project-a", "attempt-a")] = SimpleNamespace(
        handoff=LaunchHandoff("attempt-a", tmp_path / "a.json", 103.0)
    )
    runtime.pending_launch_handoffs[("project-b", "attempt-b")] = SimpleNamespace(
        handoff=LaunchHandoff("attempt-b", tmp_path / "b.json", 107.0)
    )
    monkeypatch.setattr(context.time, "monotonic", lambda: 100.0)

    assert runtime.pending_launch_wait_seconds(5.0) == 3.0
