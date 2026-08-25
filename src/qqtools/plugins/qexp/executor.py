"""tmux launch wrapper for fenced Attempts."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .config_types import RootConfig
from .runtime.records import AttemptRecord
from .tmux import create_window_for_task, is_tmux_launch_available, kill_window, send_command_to_window, window_exists


@dataclass(slots=True)
class Executor:
    create_window: Callable[[str, str, str | None], str] = create_window_for_task
    send_command: Callable[[str, str], None] = send_command_to_window
    destroy_window: Callable[[str | None], None] = kill_window
    check_window: Callable[[str | None], bool] = window_exists
    tmux_available: Callable[[], bool] = is_tmux_launch_available
    spawn_runner: Callable[..., Any] = subprocess.Popen

    def build_runner_command(
        self, cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, launch_id: str
    ) -> str:
        parts = [
            shlex.quote(part) for part in self.build_runner_argv(cfg, task_id, attempt_id, fencing_token, launch_id)
        ]
        return " ".join(parts)

    def build_runner_argv(
        self, cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int, launch_id: str
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "qqtools.plugins.qexp.runner",
            "--shared-root",
            str(cfg.shared_root),
            "--machine",
            cfg.machine_name,
            "--task-id",
            task_id,
            "--attempt-id",
            attempt_id,
            "--fencing-token",
            str(fencing_token),
            "--launch-id",
            launch_id,
            "--runtime-root",
            str(cfg.runtime_root),
        ]

    def launch_attempt(
        self, cfg: RootConfig, task_id: str, attempt: AttemptRecord, session_name: str = "experiments"
    ) -> str:
        launch_id = attempt.authorization.get("launch_id")
        if not isinstance(launch_id, str):
            raise RuntimeError("Attempt has no launch authorization.")
        if self.tmux_available():
            window_id = self.create_window(task_id, session_name, None)
            self.send_command(
                window_id,
                self.build_runner_command(cfg, task_id, attempt.attempt_id, attempt.current_fencing_token, launch_id),
            )
            return window_id

        process = self.spawn_runner(
            self.build_runner_argv(cfg, task_id, attempt.attempt_id, attempt.current_fencing_token, launch_id),
            cwd=str(cfg.project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"pid:{process.pid}"

    def cleanup_window(self, window_id: str | None) -> None:
        self.destroy_window(window_id)
