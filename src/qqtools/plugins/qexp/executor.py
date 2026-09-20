"""tmux launch wrapper for fenced Attempts."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config_types import RootConfig
from .runtime.paths import local_paths
from .runtime.records import AttemptRecord
from .runtime.work_budget import diagnostic_span
from .tmux import create_window_for_task, is_tmux_launch_available, kill_window, send_command_to_window, window_exists


@dataclass(frozen=True, slots=True)
class LaunchHandoff:
    """A runner launch awaiting its durable intent publication."""

    attempt_id: str
    intent_path: Path
    deadline: float


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
        with diagnostic_span("executor.launch.initiate"):
            reference, handoff = self.initiate_attempt(cfg, task_id, attempt, session_name)
        with diagnostic_span("executor.launch.handoff"):
            failures = self.wait_for_launch_handoffs([handoff])
        if failure := failures.get(handoff):
            raise failure
        return reference

    def initiate_attempt(
        self, cfg: RootConfig, task_id: str, attempt: AttemptRecord, session_name: str = "experiments"
    ) -> tuple[str, LaunchHandoff]:
        """Start a runner and return without waiting for its durable handoff."""
        launch_id = attempt.authorization.get("launch_id")
        if not isinstance(launch_id, str):
            raise RuntimeError("Attempt has no launch authorization.")
        from .runtime.responsibility import require_launch_responsibility

        require_launch_responsibility(cfg, task_id, attempt.attempt_id, attempt.attempt_number)
        if self.tmux_available():
            window_id = self.create_window(task_id, session_name, None)
            self.send_command(
                window_id,
                self.build_runner_command(cfg, task_id, attempt.attempt_id, attempt.current_fencing_token, launch_id),
            )
            return window_id, self._launch_handoff(cfg, attempt.attempt_id)

        process = self.spawn_runner(
            self.build_runner_argv(cfg, task_id, attempt.attempt_id, attempt.current_fencing_token, launch_id),
            cwd=str(cfg.project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"pid:{process.pid}", self._launch_handoff(cfg, attempt.attempt_id)

    @staticmethod
    def _launch_handoff(cfg: RootConfig, attempt_id: str, timeout_seconds: float = 2.0) -> LaunchHandoff:
        path = local_paths(cfg.runtime_root)["launch_intents"] / f"{attempt_id}.json"
        return LaunchHandoff(attempt_id, path, time.monotonic() + timeout_seconds)

    @staticmethod
    def wait_for_launch_handoffs(
        handoffs: list[LaunchHandoff],
    ) -> dict[LaunchHandoff, RuntimeError]:
        """Wait for handoffs by durable path identity, not project-local Attempt ID."""
        remaining = list(handoffs)
        failures: dict[LaunchHandoff, RuntimeError] = {}
        while remaining:
            now = time.monotonic()
            for handoff in list(remaining):
                if handoff.intent_path.exists():
                    remaining.remove(handoff)
                elif now >= handoff.deadline:
                    failures[handoff] = RuntimeError(f"runner did not publish launch intent for {handoff.attempt_id!r}")
                    remaining.remove(handoff)
            if remaining:
                time.sleep(0.01)
        return failures

    @staticmethod
    def _wait_for_launch_intent(cfg: RootConfig, attempt_id: str, timeout_seconds: float = 2.0) -> None:
        """Wait until the runner has durably claimed the launch handoff."""
        handoff = Executor._launch_handoff(cfg, attempt_id, timeout_seconds)
        failures = Executor.wait_for_launch_handoffs([handoff])
        if failure := failures.get(handoff):
            raise failure

    def cleanup_window(self, window_id: str | None) -> None:
        self.destroy_window(window_id)
