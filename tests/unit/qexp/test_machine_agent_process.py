from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.process import spawn_machine_agent_process
from qqtools.plugins.qexp.agent.setup import initialize_machine


def test_machine_agent_default_stderr_captures_startup_errors_without_a_pipe(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        stderr = None

        def poll(self) -> None:
            return None

    def start_process(*_args, **kwargs) -> FakeProcess:
        captured.update(kwargs)
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid}}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.subprocess.Popen", start_process)

    assert spawn_machine_agent_process(runtime).pid == FakeProcess.pid
    assert captured["stderr"] is not None


def test_machine_agent_startup_error_includes_child_failure(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")

    class FakeProcess:
        pid = 123

        def poll(self) -> int:
            return 1

    def start_process(*_args, **kwargs) -> FakeProcess:
        kwargs["stderr"].write("RuntimeError: machine scheduler authority is already held.\n")
        kwargs["stderr"].flush()
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.subprocess.Popen", start_process)

    with pytest.raises(RuntimeError, match="machine scheduler authority is already held"):
        spawn_machine_agent_process(runtime)


def test_machine_agent_process_accepts_explicit_loop_interval(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

    def start_process(command, **_kwargs) -> FakeProcess:
        captured["command"] = command
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid}}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.subprocess.Popen", start_process)

    spawn_machine_agent_process(runtime, loop_interval=0.1)

    command = captured["command"]
    assert command[-2:] == ["--loop-interval", "0.1"]


def test_machine_agent_process_rejects_non_positive_loop_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loop_interval must be positive"):
        spawn_machine_agent_process(MachineRuntime(tmp_path / "machine-runtime"), loop_interval=0)
