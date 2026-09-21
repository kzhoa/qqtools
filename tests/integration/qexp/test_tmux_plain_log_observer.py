"""Real-pane evidence for the Attempt-bound tmux log observer."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
from qqtools.plugins.qexp.layout import shared_attempt_log_path
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.tmux import is_libtmux_available
from tests.helpers.qexp.resources import _has_active_unix_socket

pytestmark = [pytest.mark.integration, pytest.mark.machine_lab]


def _wait_for(predicate, description: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{description} did not converge within {timeout:.1f}s")


def _tmux(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=5, check=False)


def _observer_window(task_id: str) -> str | None:
    result = _tmux("list-windows", "-a", "-F", "#{window_id}\t#{window_name}")
    for line in result.stdout.splitlines():
        window_id, _, name = line.partition("\t")
        if name == task_id:
            return window_id
    return None


def _capture(window_id: str) -> str:
    result = _tmux("capture-pane", "-p", "-t", window_id, "-S", "-200")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_enabled_tmux_observer_streams_plain_log_without_owning_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qexp_resource_scope
) -> None:
    if sys.platform != "linux":
        pytest.fail("plain-log observer evidence requires Linux")
    if shutil.which("tmux") is None or not is_libtmux_available():
        pytest.fail("plain-log observer evidence requires tmux and libtmux")

    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0")
    project_root = tmp_path / "project with spaces"
    cfg = init_shared_root(project_root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, "gpu-1")
    ready = tmp_path / "ready"
    append = tmp_path / "append"
    finish = tmp_path / "finish"
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time\nfrom pathlib import Path\n"
            f"ready,append,finish=map(Path,{[str(ready), str(append), str(finish)]!r})\n"
            "print('stdout-α',flush=True)\nprint('stderr-traceback-marker',file=sys.stderr,flush=True)\n"
            "ready.touch()\n"
            "deadline=time.monotonic()+30\n"
            "while not append.exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(91)\n"
            "    time.sleep(.02)\n"
            "print('stdout-after-append',flush=True)\n"
            "while not finish.exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(92)\n"
            "    time.sleep(.02)\n"
        ),
    ]
    task = submit(cfg, command, working_dir=project_root, tmux_override=True)
    agent = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
    try:
        _wait_for(ready.exists, "training start")
        _wait_for(lambda: load_task(cfg, task.task_id).state["projection"] == "running", "running truth")
        stored = load_task(cfg, task.task_id)
        attempt_id = stored.attempt_control["current_attempt_id"]
        attempt_number = stored.attempt_control["current_attempt_number"]
        assert isinstance(attempt_id, str) and isinstance(attempt_number, int)
        attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt_number)
        _wait_for(
            lambda: all(
                isinstance(read_json(attempt_file)["attempt"]["process"].get(key), int)
                for key in ("wrapper_pid", "process_group_id")
            ),
            "published process identity",
        )
        attempt_before = read_json(attempt_file)["attempt"]
        process_identity = (
            attempt_before["process"]["wrapper_pid"],
            attempt_before["process"]["process_group_id"],
        )

        window: str | None = None

        def pane_has_initial_output() -> bool:
            nonlocal window
            window = _observer_window(task.task_id)
            return window is not None and {"stdout-α", "stderr-traceback-marker"} <= set(_capture(window).splitlines())

        _wait_for(pane_has_initial_output, "initial pane output")
        assert window is not None
        append.touch()
        _wait_for(lambda: "stdout-after-append" in _capture(window), "appended pane output")

        assert _tmux("kill-window", "-t", window).returncode == 0
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        attempt_after_close = read_json(attempt_file)["attempt"]
        assert (
            attempt_after_close["process"]["wrapper_pid"],
            attempt_after_close["process"]["process_group_id"],
        ) == process_identity

        finish.touch()
        _wait_for(
            lambda: load_task(cfg, task.task_id).state["projection"] == "succeeded",
            "terminal reconciliation after observer close",
        )
        log_path = shared_attempt_log_path(cfg, task.task_id, attempt_id)
        text = log_path.read_text(encoding="utf-8")
        assert "stdout-α" in text
        assert "stderr-traceback-marker" in text
        assert "stdout-after-append" in text
    finally:
        append.touch()
        finish.touch()
        try:
            if get_machine_agent_status(runtime)["is_running"]:
                stop_machine_agent(runtime, timeout=5)
        finally:
            if agent.poll() is None:
                agent.kill()
            agent.wait(timeout=5)
            for socket_path in qexp_resource_scope.tmux_root.rglob("*"):
                if socket_path.is_socket():
                    subprocess.run(
                        ["tmux", "-S", str(socket_path), "kill-server"],
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
            _wait_for(
                lambda: not _has_active_unix_socket(qexp_resource_scope.tmux_root),
                "tmux cleanup",
                timeout=5,
            )
