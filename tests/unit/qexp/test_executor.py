from __future__ import annotations

import sys

import pytest

from qqtools.plugins.qexp.executor import Executor
from qqtools.plugins.qexp.layout import init_shared_root


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


class TestBuildRunnerCommand:
    def test_contains_module_path(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "task-abc")
        assert "qqtools.plugins.qexp.runner" in cmd

    def test_contains_task_id(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "task-abc")
        assert "task-abc" in cmd

    def test_contains_shared_root(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "t1")
        assert str(cfg.shared_root) in cmd

    def test_contains_machine(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "t1")
        assert "dev1" in cmd

    def test_contains_runtime_root(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "t1")
        assert str(cfg.runtime_root) in cmd

    def test_uses_current_python(self, cfg):
        executor = Executor()
        cmd = executor.build_runner_command(cfg, "t1")
        assert sys.executable.split("/")[-1] in cmd


class TestLaunchInWindow:
    def test_creates_window_and_sends_command(self, cfg):
        created = []
        sent = []
        task = type("TaskStub", (), {"spec": type("SpecStub", (), {"working_dir": "/tmp/from-task"})()})()

        def fake_create(task_id, session, start_directory):
            created.append((task_id, session, start_directory))
            return f"@fake-{task_id}"

        def fake_send(window_id, command):
            sent.append((window_id, command))

        executor = Executor(
            create_window=fake_create,
            send_command=fake_send,
            task_loader=lambda _cfg, _task_id: task,
        )
        window_id = executor.launch_in_window(cfg, "t1", session_name="test-session")

        assert window_id == "@fake-t1"
        assert len(created) == 1
        assert created[0] == ("t1", "test-session", "/tmp/from-task")
        assert len(sent) == 1
        assert sent[0][0] == "@fake-t1"
        assert "qqtools.plugins.qexp.runner" in sent[0][1]

    def test_launch_task_routes_group_to_session(self, cfg):
        created = []
        task = type("TaskStub", (), {"spec": type("SpecStub", (), {"working_dir": "/tmp/from-task"})()})()

        def fake_create(task_id, session, start_directory):
            created.append((task_id, session, start_directory))
            return f"@fake-{task_id}"

        executor = Executor(
            create_window=fake_create,
            send_command=lambda window_id, command: None,
            task_loader=lambda _cfg, _task_id: task,
        )
        executor.launch_task(cfg, "t1", "contract_n_4and6")
        assert created == [("t1", "contract_n_4and6", "/tmp/from-task")]

    def test_cleanup_window(self):
        destroyed = []

        def fake_destroy(window_id):
            destroyed.append(window_id)

        executor = Executor(destroy_window=fake_destroy)
        executor.cleanup_window("@w123")
        assert destroyed == ["@w123"]
