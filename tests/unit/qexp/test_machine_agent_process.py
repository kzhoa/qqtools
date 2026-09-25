from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import MachineAgentStartError
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

    def start_process(command, **kwargs) -> FakeProcess:
        captured.update(kwargs)
        instance_id = command[command.index("--instance-id") + 1]
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid, "instance_id": instance_id}}),
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
        kwargs["stderr"].write(b"RuntimeError: machine scheduler authority is already held.\n")
        kwargs["stderr"].flush()
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.subprocess.Popen", start_process)

    with pytest.raises(RuntimeError, match="machine scheduler authority is already held"):
        spawn_machine_agent_process(runtime)


def test_machine_agent_spawn_oserror_is_named(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.process.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("spawn denied")),
    )

    with pytest.raises(MachineAgentStartError, match="spawn denied"):
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
        instance_id = command[command.index("--instance-id") + 1]
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid, "instance_id": instance_id}}),
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


def test_handshake_timeout_is_recorded_before_bounded_signal_escalation(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    log_root = tmp_path / "tmp"
    log_root.mkdir()
    monkeypatch.setattr("qqtools.plugins.qexp.agent.diagnostics._TMP_ROOT", log_root)
    events: list[tuple[str, object]] = []

    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            events.append(("signal", signum))

        def wait(self, timeout=None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired("agent", timeout)
            return -signal.SIGKILL

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.subprocess.Popen", lambda *_args, **_kwargs: FakeProcess())
    monotonic_values = iter((0.0, 6.0))

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return next(monotonic_values)

        @staticmethod
        def sleep(_seconds: float) -> None:
            raise AssertionError("expired handshake should not sleep")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.time", FakeTime)

    def capture_evidence(*_args, **kwargs) -> bool:
        events.append(("evidence", dict(kwargs["evidence"])))
        return True

    monkeypatch.setattr("qqtools.plugins.qexp.agent.process.update_diagnostic_evidence", capture_evidence)

    with pytest.raises(MachineAgentStartError, match="did not acquire scheduler authority"):
        spawn_machine_agent_process(runtime)

    first_timeout = next(
        index
        for index, event in enumerate(events)
        if event[0] == "evidence" and event[1].get("startup_outcome") == "timed_out"
    )
    first_signal = next(index for index, event in enumerate(events) if event[0] == "signal")
    assert first_timeout < first_signal
    assert [event[1] for event in events if event[0] == "signal"] == [signal.SIGTERM, signal.SIGKILL]
    final_evidence = [event[1] for event in events if event[0] == "evidence"][-1]
    assert final_evidence["wait_status"] == -signal.SIGKILL
    assert [attempt["signal"] for attempt in final_evidence["signal_attempts"]] == [
        signal.SIGTERM,
        signal.SIGKILL,
    ]


def test_detached_launcher_passes_degraded_reconciliation_to_child(tmp_path: Path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    captured: dict[str, object] = {}

    from qqtools.plugins.qexp.agent import process as agent_process

    original_prepare = agent_process.prepare_agent_diagnostics

    def degraded_prepare(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        prepared.capture_health = "degraded"
        prepared.error = "reconciliation_unavailable"
        prepared.reconciliation_degraded = True
        return prepared

    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

    def start_process(command, **_kwargs) -> FakeProcess:
        captured["command"] = command
        instance_id = command[command.index("--instance-id") + 1]
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid, "instance_id": instance_id}}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(agent_process, "prepare_agent_diagnostics", degraded_prepare)
    monkeypatch.setattr(agent_process.subprocess, "Popen", start_process)

    assert spawn_machine_agent_process(runtime).pid == FakeProcess.pid
    command = captured["command"]
    assert "--initial-reconciliation-degraded" in command
    assert command[command.index("--initial-capture-error") + 1] == "reconciliation_unavailable"
