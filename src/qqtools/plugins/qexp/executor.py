"""qexp task executor — builds runner commands and launches them in tmux."""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .tmux import (
    create_window_for_task,
    kill_window,
    send_command_to_window,
    window_exists,
)
from .layout import RootConfig
from .models import tmux_session_for_group


@dataclass(slots=True)
class Executor:
    """Builds and launches runner commands in tmux windows."""

    create_window: Callable[[str, str], str] = create_window_for_task
    send_command: Callable[[str, str], None] = send_command_to_window
    destroy_window: Callable[[str | None], None] = kill_window
    check_window: Callable[[str | None], bool] = window_exists

    def build_runner_command(
        self,
        cfg: RootConfig,
        task_id: str,
    ) -> str:
        """Build the shell command that invokes the qexp runner."""
        parts = [
            shlex.quote(sys.executable),
            "-m",
            "qqtools.plugins.qexp.runner",
            "--shared-root",
            shlex.quote(str(cfg.shared_root)),
            "--machine",
            shlex.quote(cfg.machine_name),
            "--task-id",
            shlex.quote(task_id),
            "--runtime-root",
            shlex.quote(str(cfg.runtime_root)),
        ]
        return " ".join(parts)

    def launch_in_window(
        self,
        cfg: RootConfig,
        task_id: str,
        session_name: str | None = None,
    ) -> str:
        """Create a tmux window and launch the runner inside it.

        Returns the tmux window_id.
        """
        resolved_session = session_name or "experiments"
        window_id = self.create_window(task_id, resolved_session)
        command = self.build_runner_command(cfg, task_id)
        self.send_command(window_id, command)
        return window_id

    def launch_task(self, cfg: RootConfig, task_id: str, group: str | None) -> str:
        return self.launch_in_window(
            cfg,
            task_id,
            session_name=tmux_session_for_group(group),
        )

    def cleanup_window(self, window_id: str | None) -> None:
        """Kill a tmux window if it exists."""
        self.destroy_window(window_id)
