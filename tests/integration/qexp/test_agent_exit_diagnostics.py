from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.config import set_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.diagnostics import DEFAULT_LOG_MAX_BYTES, MIN_LOG_MAX_BYTES, summarize_diagnostic_record
from qqtools.plugins.qexp.agent.lifecycle import ensure_machine_agent_started, get_machine_agent_status
from qqtools.plugins.qexp.agent.process import spawn_machine_agent_process
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.layout import machine_state_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _wait_for(predicate, *, timeout: float = 15.0, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{description} did not converge within {timeout:.1f}s")


def _child_environment(checkout_subprocess_env: dict[str, str]) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": checkout_subprocess_env["PYTHONPATH"]}


def _latest_record(runtime: MachineRuntime) -> dict:
    records = [read_json(path) for path in runtime.paths["diagnostic_instances"].glob("*.json")]
    assert records
    return max(records, key=lambda value: value["startup_sequence"])


def _foreground_command(
    runtime: MachineRuntime,
    *,
    raise_after_start: bool,
    fail_pid_unlink: bool = False,
    fail_initial_log_open: bool = False,
    degrade_reconciliation: bool = False,
) -> list[str]:
    source_lines = [
        "from pathlib import Path",
        "from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop",
    ]
    if fail_pid_unlink:
        source_lines.extend(
            (
                "original_unlink = Path.unlink",
                "def unlink(path, *args, **kwargs):",
                "    if path.name == 'machine-agent.pid':",
                "        raise PermissionError('injected PID removal failure')",
                "    return original_unlink(path, *args, **kwargs)",
                "Path.unlink = unlink",
            )
        )
    if fail_initial_log_open:
        source_lines.extend(
            (
                "from qqtools.plugins.qexp.agent import diagnostics",
                "original_log_open = diagnostics._open_log_file",
                "log_open_attempts = 0",
                "def fail_first_log_open(path):",
                "    global log_open_attempts",
                "    log_open_attempts += 1",
                "    if log_open_attempts == 1:",
                "        raise OSError('injected transient log open failure')",
                "    return original_log_open(path)",
                "diagnostics._open_log_file = fail_first_log_open",
            )
        )
    if degrade_reconciliation:
        source_lines.extend(
            (
                "from qqtools.plugins.qexp.agent import diagnostics",
                "diagnostics.reconcile_agent_diagnostics = lambda *_args, **_kwargs: {",
                "    'available': True, 'reconciled': False, 'coverage': {'evicted_count': 0}",
                "}",
            )
        )
    if raise_after_start:
        source_lines.extend(
            (
                "from qqtools.plugins.qexp.agent import lifecycle",
                "def fail_cycle(_self):",
                "    raise RuntimeError('post-handshake boom')",
                "lifecycle._MachineControlPlane.start = fail_cycle",
            )
        )
    source_lines.append(f"run_machine_agent_loop({str(runtime.root)!r}, loop_interval=0.05, available_gpus=[])")
    return [sys.executable, "-c", "\n".join(source_lines)]


def test_exception_after_active_handshake_keeps_traceback_and_specific_reason(
    tmp_path: Path, checkout_subprocess_env: dict[str, str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    process = subprocess.run(
        _foreground_command(runtime, raise_after_start=True, fail_pid_unlink=True),
        env=_child_environment(checkout_subprocess_env),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert process.returncode != 0
    record = _latest_record(runtime)
    summary = summarize_diagnostic_record(record)
    assert summary["reason"] == "unhandled_exception"
    assert summary["source"] == "agent"
    assert summary["cleanup_outcome"] == "failed"
    assert record["writers"]["agent"]["primary_exception"]["exception_type"] == "RuntimeError"
    assert record["writers"]["agent"]["cleanup_steps"]["identity_checked_pid_removal"] == {
        "outcome": "failed",
        "error_type": "PermissionError",
    }
    log_path = Path(record["log_path"])
    assert log_path.is_file()
    assert "post-handshake boom" in log_path.read_text(encoding="utf-8")


def test_foreground_sigterm_records_signal_and_completed_cleanup(
    tmp_path: Path, checkout_subprocess_env: dict[str, str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    process = subprocess.Popen(
        _foreground_command(runtime, raise_after_start=False),
        env=_child_environment(checkout_subprocess_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(
            lambda: (
                (runtime.paths["agent"] / "status.json").exists()
                and read_json(runtime.paths["agent"] / "status.json").get("machine_agent", {}).get("state") == "active"
            ),
            description="foreground active status",
        )
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    record = _latest_record(runtime)
    summary = summarize_diagnostic_record(record)
    assert summary["reason"] == "stopped_by_signal"
    assert summary["handled_signal"] == signal.SIGTERM
    assert summary["cleanup_outcome"] == "succeeded"
    status = read_json(runtime.paths["agent"] / "status.json")["machine_agent"]
    assert status["state"] == "stopped"
    assert status["stop_reason"] == "stopped_by_signal"


def test_project_reason_survives_later_pid_cleanup_failure(
    tmp_path: Path, checkout_subprocess_env: dict[str, str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    project = tmp_path / "project"
    initialize_project(project)
    result = register_projects(runtime, [project])
    assert result["projects"][0]["status"] == "registered"
    binding = runtime.load_registry()[1][0]

    process = subprocess.Popen(
        _foreground_command(runtime, raise_after_start=False, fail_pid_unlink=True),
        env=_child_environment(checkout_subprocess_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(
            lambda: (
                (runtime.paths["agent"] / "status.json").exists()
                and read_json(runtime.paths["agent"] / "status.json").get("machine_agent", {}).get("state") == "active"
            ),
            description="registered foreground active status",
        )
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    record = _latest_record(runtime)
    summary = summarize_diagnostic_record(record)
    assert summary["reason"] == "stopped_by_signal"
    assert summary["cleanup_outcome"] == "failed"
    assert record["writers"]["agent"]["cleanup_steps"]["project_stop_publications"]["outcome"] == "succeeded"
    assert record["writers"]["agent"]["cleanup_steps"]["identity_checked_pid_removal"]["outcome"] == "failed"
    local_status = read_json(runtime.paths["agent"] / "status.json")["machine_agent"]
    assert local_status["stop_reason"] == "stopped_by_signal"
    project_status = read_json(machine_state_path(binding.root_config(), "agent.json"))["agent"]
    assert project_status["stop_reason"] == "stopped_by_signal"


def test_initial_log_open_failure_recovers_without_agent_restart(
    tmp_path: Path, checkout_subprocess_env: dict[str, str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    process = subprocess.Popen(
        _foreground_command(runtime, raise_after_start=False, fail_initial_log_open=True),
        env=_child_environment(checkout_subprocess_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(
            lambda: (
                bool(list(runtime.paths["diagnostic_instances"].glob("*.json")))
                and _latest_record(runtime).get("capture_health") == "healthy"
                and "capture_error" not in _latest_record(runtime)
            ),
            description="log capture recovery",
        )
        record = _latest_record(runtime)
        assert Path(record["log_path"]).is_file()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_log_health_cannot_hide_degraded_reconciliation(
    tmp_path: Path, checkout_subprocess_env: dict[str, str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    process = subprocess.Popen(
        _foreground_command(runtime, raise_after_start=False, degrade_reconciliation=True),
        env=_child_environment(checkout_subprocess_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(
            lambda: (
                bool(list(runtime.paths["diagnostic_instances"].glob("*.json")))
                and _latest_record(runtime).get("phase") == "active"
            ),
            description="agent activation with degraded reconciliation",
        )
        record = _latest_record(runtime)
        assert record["capture_health"] == "degraded"
        assert record["capture_error"] == "reconciliation_unavailable"
        assert Path(record["log_path"]).is_file()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_detached_raw_descriptors_follow_repeated_rotation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    set_agent_config(runtime, log_max_bytes=MIN_LOG_MAX_BYTES)
    process = spawn_machine_agent_process(runtime, loop_interval=0.05, available_gpus=[])
    try:
        set_agent_config(runtime, log_max_bytes=DEFAULT_LOG_MAX_BYTES)
        replacement, idempotent_status = ensure_machine_agent_started(runtime)
        assert replacement is None
        assert idempotent_status["pid"] == process.pid
        diagnostics = get_machine_agent_status(runtime)["diagnostics"]
        assert diagnostics["configured_log_max_bytes"] == DEFAULT_LOG_MAX_BYTES
        assert diagnostics["effective_log_max_bytes"] == MIN_LOG_MAX_BYTES

        status = read_json(runtime.paths["agent"] / "status.json")["machine_agent"]
        record = read_json(runtime.paths["diagnostic_instances"] / f"{status['instance_id']}.json")
        log_path = Path(record["log_path"])
        for rotation in range(2):
            with Path(f"/proc/{process.pid}/fd/1").open("wb", buffering=0) as raw_stdout:
                raw_stdout.write(bytes([65 + rotation]) * (MIN_LOG_MAX_BYTES + 4096))
            _wait_for(
                lambda: len(list(log_path.parent.glob("_agent_*.log"))) >= rotation + 1,
                description=f"rotation {rotation + 1}",
            )
        with Path(f"/proc/{process.pid}/fd/2").open("wb", buffering=0) as raw_stderr:
            raw_stderr.write(b"after-rotation\n")
        _wait_for(
            lambda: b"after-rotation" in log_path.read_bytes(),
            description="post-rotation stderr",
        )
        assert len(list(log_path.parent.glob("_agent_*.log"))) == 2
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=15)


def test_sigkill_is_reconciled_as_unknown_without_invented_signal(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    first = spawn_machine_agent_process(runtime, loop_interval=0.05, available_gpus=[])
    first_status = read_json(runtime.paths["agent"] / "status.json")["machine_agent"]
    first.kill()
    assert first.wait(timeout=10) == -signal.SIGKILL

    successor = spawn_machine_agent_process(runtime, loop_interval=0.05, available_gpus=[])
    try:
        record = read_json(runtime.paths["diagnostic_instances"] / f"{first_status['instance_id']}.json")
        summary = summarize_diagnostic_record(record)
        assert summary["reason"] == "abnormal_exit_unknown"
        assert summary["source"] == "observer"
        assert summary.get("signal") is None
        assert summary.get("exit_code") is None
        assert summary.get("exited_at") is None
    finally:
        successor.terminate()
        successor.wait(timeout=15)
