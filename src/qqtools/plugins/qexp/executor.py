"""tmux launch wrapper for fenced Attempts."""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from typing import Callable

from .layout import RootConfig
from .runtime.records import AttemptRecord
from .tmux import create_window_for_task, kill_window, send_command_to_window, window_exists


@dataclass(slots=True)
class Executor:
    create_window: Callable[[str, str, str | None], str] = create_window_for_task
    send_command: Callable[[str, str], None] = send_command_to_window
    destroy_window: Callable[[str | None], None] = kill_window
    check_window: Callable[[str | None], bool] = window_exists

    def build_runner_command(self, cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int) -> str:
        parts = [shlex.quote(sys.executable), "-m", "qqtools.plugins.qexp.runner", "--shared-root",
                 shlex.quote(str(cfg.shared_root)), "--machine", shlex.quote(cfg.machine_name),
                 "--task-id", shlex.quote(task_id), "--attempt-id", shlex.quote(attempt_id),
                 "--fencing-token", str(fencing_token), "--runtime-root", shlex.quote(str(cfg.runtime_root))]
        return " ".join(parts)

    def launch_attempt(self, cfg: RootConfig, task_id: str, attempt: AttemptRecord, session_name: str = "experiments") -> str:
        window_id = self.create_window(task_id, session_name, None)
        self.send_command(window_id, self.build_runner_command(cfg, task_id, attempt.attempt_id, attempt.current_fencing_token))
        return window_id

    def cleanup_window(self, window_id: str | None) -> None:
        self.destroy_window(window_id)
