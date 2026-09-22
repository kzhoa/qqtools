"""Real CLI-process evidence for read-only continuous Task observation."""

from __future__ import annotations

import errno
import os
import pty
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
from qqtools.plugins.qexp.progress_policy import set_progress_policy
from qqtools.plugins.qexp.runtime.progress import shared_progress_path
from qqtools.plugins.qexp.runtime.tasks import load_task
from tests.helpers.qexp.resources import _has_active_unix_socket

pytestmark = [pytest.mark.integration, pytest.mark.machine_lab]


def _wait_for(predicate, description: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{description} did not converge within {timeout:.1f}s")


def _read_available(fd: int) -> bytes:
    chunks = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _viewer_command(cfg, runtime: MachineRuntime, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "qqtools.plugins.qexp.cli",
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(runtime.root),
        *args,
    ]


def _start_watch(command: list[str], env: dict[str, str]) -> tuple[subprocess.Popen, int]:
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    return process, master


def test_independent_viewers_do_not_control_training_and_default_viewers_exit_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkout_subprocess_env,
    qexp_resource_scope,
) -> None:
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0")
    project_root = tmp_path / "project"
    cfg = init_shared_root(project_root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    set_progress_policy(cfg, 1)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, "gpu-1")
    ready, finish = tmp_path / "ready", tmp_path / "finish"
    command = [
        sys.executable,
        "-c",
        (
            "import time\nfrom pathlib import Path\nfrom qqtools.qexp import progress\n"
            f"ready,finish=map(Path,{[str(ready), str(finish)]!r})\n"
            "progress.update(stage='train',current=1,total=2,message='viewer-live')\n"
            "progress.flush()\nprint('log-before-viewers',flush=True)\nready.touch()\n"
            "deadline=time.monotonic()+30\n"
            "while not finish.exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(93)\n"
            "    time.sleep(.02)\n"
            "print('log-after-viewers',flush=True)\n"
        ),
    ]
    task = submit(cfg, command, working_dir=project_root)
    agent = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
    log_viewer = watch_viewer = None
    live_watch_fd = terminal_watch_fd = None
    env = os.environ.copy()
    env.update(checkout_subprocess_env)
    try:
        _wait_for(ready.exists, "training output")
        _wait_for(lambda: load_task(cfg, task.task_id).state["projection"] == "running", "running truth")
        stored = load_task(cfg, task.task_id)
        attempt_id = stored.attempt_control["current_attempt_id"]
        assert isinstance(attempt_id, str)
        _wait_for(
            lambda: shared_progress_path(cfg.shared_root, task.task_id, attempt_id).exists(),
            "projected application progress",
        )

        log_viewer = subprocess.Popen(
            _viewer_command(cfg, runtime, "task", "logs", task.task_id, "--follow", "--interval-seconds", "1"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        assert log_viewer.stdout is not None
        os.set_blocking(log_viewer.stdout.fileno(), False)
        live_log = bytearray()

        watch_viewer, live_watch_fd = _start_watch(
            _viewer_command(cfg, runtime, "task", "show", task.task_id, "--watch", "--interval-seconds", "1"),
            env,
        )
        live_watch = bytearray()

        def viewers_observed() -> bool:
            live_log.extend(_read_available(log_viewer.stdout.fileno()))
            live_watch.extend(_read_available(live_watch_fd))
            return b"log-before-viewers" in live_log and b"viewer-live" in live_watch

        _wait_for(viewers_observed, "live log and progress viewers")
        for viewer in (log_viewer, watch_viewer):
            viewer.send_signal(signal.SIGINT)
        assert log_viewer.wait(timeout=5) == 130
        assert watch_viewer.wait(timeout=5) == 130
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        assert load_task(cfg, task.task_id).attempt_control["current_attempt_id"] == attempt_id

        finish.touch()
        _wait_for(lambda: load_task(cfg, task.task_id).state["projection"] == "succeeded", "terminal truth")

        terminal_logs = subprocess.run(
            _viewer_command(cfg, runtime, "task", "logs", task.task_id, "--follow", "--interval-seconds", "1"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=env,
            timeout=10,
            check=False,
        )
        assert terminal_logs.returncode == 0, terminal_logs.stderr.decode(errors="replace")
        assert b"log-before-viewers" in terminal_logs.stdout
        assert b"log-after-viewers" in terminal_logs.stdout

        terminal_watch, terminal_watch_fd = _start_watch(
            _viewer_command(cfg, runtime, "task", "show", task.task_id, "--watch", "--interval-seconds", "1"),
            env,
        )
        assert terminal_watch.wait(timeout=10) == 0
        terminal_watch_output = _read_available(terminal_watch_fd)
        assert task.task_id.encode() in terminal_watch_output
        assert b"succeeded" in terminal_watch_output
    finally:
        finish.touch()
        for viewer in (log_viewer, watch_viewer):
            if viewer is not None and viewer.poll() is None:
                viewer.kill()
                viewer.wait(timeout=5)
        for fd in (live_watch_fd, terminal_watch_fd):
            if fd is not None:
                os.close(fd)
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
