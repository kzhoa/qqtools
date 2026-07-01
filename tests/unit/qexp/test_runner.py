from __future__ import annotations

import os
import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from qqtools.plugins.qexp.indexes import update_index_on_phase_change
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.models import (
    Meta,
    PHASE_CANCELLED,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    PHASE_SUCCEEDED,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    utc_now_iso,
)
from qqtools.plugins.qexp.runner import (
    build_child_command,
    build_child_environment,
    classify_exit,
    run_task,
)
from qqtools.plugins.qexp.storage import cas_update_task, load_task, save_task
from qqtools.plugins.qexp.indexes import update_index_on_submit


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


def _make_starting_task(cfg, task_id: str = "t1") -> Task:
    now = utc_now_iso()
    task = Task(
        meta=Meta.new("dev1"),
        task_id=task_id,
        name=None,
        group=None,
        batch_id=None,
        machine_name="dev1",
        attempt=1,
        spec=TaskSpec(
            command=["echo", "hello"],
            requested_gpus=1,
            working_dir=str(cfg.project_root),
        ),
        status=TaskStatus(phase=PHASE_STARTING),
        runtime=TaskRuntime(assigned_gpus=[0]),
        timestamps=TaskTimestamps(created_at=now, queued_at=now, started_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )
    save_task(cfg, task)
    update_index_on_submit(cfg, task)
    # Fix index: put in starting, not queued
    update_index_on_phase_change(cfg, task_id, PHASE_QUEUED, PHASE_STARTING)
    return task


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestBuildChildEnvironment:
    def test_sets_cuda_visible_devices(self):
        env = build_child_environment([0, 2])
        assert env["CUDA_VISIBLE_DEVICES"] == "0,2"

    def test_empty_gpus(self):
        env = build_child_environment([])
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_extra_env(self):
        env = build_child_environment([1], extra_env={"FOO": "bar"})
        assert env["FOO"] == "bar"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_inherits_parent_env(self):
        env = build_child_environment([])
        assert "PATH" in env


class TestBuildChildCommand:
    def test_wraps_in_bash(self):
        cmd = build_child_command(["python", "train.py", "--lr", "0.01"])
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        assert "exec" in cmd[2]
        assert "python" in cmd[2]
        assert "train.py" in cmd[2]


class TestClassifyExit:
    def test_success(self):
        phase, reason = classify_exit(0)
        assert phase == PHASE_SUCCEEDED
        assert reason is None

    def test_nonzero(self):
        phase, reason = classify_exit(1)
        assert phase == PHASE_FAILED
        assert reason == "nonzero_exit"

    def test_sigterm(self):
        phase, reason = classify_exit(-signal.SIGTERM)
        assert phase == PHASE_CANCELLED
        assert reason == "cancelled_by_signal"

    def test_sigkill(self):
        phase, reason = classify_exit(-signal.SIGKILL)
        assert phase == PHASE_CANCELLED
        assert reason == "cancelled_by_signal"

    def test_other_signal(self):
        phase, reason = classify_exit(-signal.SIGSEGV)
        assert phase == PHASE_FAILED
        assert reason == "nonzero_exit"


# ---------------------------------------------------------------------------
# Integration tests for run_task
# ---------------------------------------------------------------------------


class _FakePopen:
    """Simulate subprocess.Popen for testing."""

    def __init__(self, exit_code: int = 0, output: bytes = b"test output\n"):
        self._exit_code = exit_code
        self._output = output
        self.pid = 12345

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # Create a fake stdout with read1 method
        self.stdout = MagicMock()
        call_count = [0]

        def fake_read1(size):
            if call_count[0] == 0:
                call_count[0] += 1
                return self._output
            return b""

        self.stdout.read1 = fake_read1
        return self

    def wait(self):
        return self._exit_code

    def poll(self):
        return self._exit_code


class TestRunTask:
    def test_successful_task(self, cfg):
        _make_starting_task(cfg)
        popen = _FakePopen(exit_code=0, output=b"ok\n")

        rc = run_task(cfg, "t1", popen_factory=popen)

        assert rc == 0
        loaded = load_task(cfg, "t1")
        assert loaded.status.phase == PHASE_SUCCEEDED
        assert loaded.result.exit_code == 0
        assert loaded.timestamps.finished_at is not None

    def test_failed_task(self, cfg):
        _make_starting_task(cfg)
        popen = _FakePopen(exit_code=1)

        rc = run_task(cfg, "t1", popen_factory=popen)

        assert rc == 1
        loaded = load_task(cfg, "t1")
        assert loaded.status.phase == PHASE_FAILED
        assert loaded.result.exit_code == 1
        assert loaded.status.reason == "nonzero_exit"

    def test_wrong_phase_raises(self, cfg):
        # Create a task in queued phase instead of starting
        now = utc_now_iso()
        task = Task(
            meta=Meta.new("dev1"),
            task_id="t-bad",
            name=None,
            group=None,
            batch_id=None,
            machine_name="dev1",
            attempt=1,
            spec=TaskSpec(
                command=["echo"],
                requested_gpus=1,
                working_dir=str(cfg.project_root),
            ),
            status=TaskStatus(phase=PHASE_QUEUED),
            runtime=TaskRuntime(),
            timestamps=TaskTimestamps(created_at=now, queued_at=now),
            result=TaskResult(),
            lineage=TaskLineage(),
        )
        save_task(cfg, task)

        with pytest.raises(RuntimeError, match="starting"):
            run_task(cfg, "t-bad")

    def test_sets_wrapper_pid(self, cfg):
        _make_starting_task(cfg)
        popen = _FakePopen(exit_code=0)

        run_task(cfg, "t1", popen_factory=popen)

        loaded = load_task(cfg, "t1")
        # wrapper_pid was set during running phase
        # After finalization it's still there
        assert loaded.runtime.wrapper_pid is not None

    def test_creates_log_file(self, cfg):
        _make_starting_task(cfg)
        popen = _FakePopen(exit_code=0, output=b"logged output\n")

        run_task(cfg, "t1", popen_factory=popen)

        from qqtools.plugins.qexp.layout import runtime_log_path
        log_path = runtime_log_path(cfg, "t1")
        assert log_path.is_file()
        assert b"logged output" in log_path.read_bytes()

    def test_launch_failure_marks_failed(self, cfg):
        _make_starting_task(cfg)

        def bad_popen(*args, **kwargs):
            raise OSError("cannot launch")

        with pytest.raises(OSError, match="cannot launch"):
            run_task(cfg, "t1", popen_factory=bad_popen)

        loaded = load_task(cfg, "t1")
        assert loaded.status.phase == PHASE_FAILED
        assert "launch_failed" in (loaded.status.reason or "")

    def test_launches_child_in_task_working_dir(self, cfg):
        _make_starting_task(cfg)
        popen = _FakePopen(exit_code=0)

        run_task(cfg, "t1", popen_factory=popen)

        assert popen.kwargs["cwd"] == str(cfg.project_root)
