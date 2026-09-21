"""tmux and detached launch wrappers for fenced Attempts."""

from __future__ import annotations

import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config_types import RootConfig
from .launch_policy import resolve_launch_handoff_policy, validate_launch_handoff_timeout_seconds
from .layout import shared_attempt_log_path
from .runtime.paths import local_paths
from .runtime.records import AttemptRecord
from .runtime.work_budget import diagnostic_span
from .task_observation import resolve_task_tmux_observation
from .tmux import create_window_for_task, is_tmux_launch_available, kill_window, send_command_to_window, window_exists

_LAUNCH_FAILURE_CODES = frozenset(
    {
        "executor_launch_backend_failed",
        "executor_launch_handoff_timeout",
        "executor_launch_failed",
    }
)
_DIAGNOSTIC_LIMIT = 512


@dataclass(slots=True, eq=False)
class LaunchHandle:
    """Exact local resource returned by a launch backend."""

    backend: str
    reference: Any
    runner_process: Any | None = None
    observer_window_id: str | None = None

    @property
    def backend_kind(self) -> str:
        """Return the backend name using the explicit contract spelling."""
        return self.backend

    @property
    def kind(self) -> str:
        """Return the backend kind."""
        return self.backend

    @property
    def handle(self) -> Any:
        """Return the exact backend object or reference."""
        return self.reference

    @property
    def object(self) -> Any:
        """Return the exact backend object or reference."""
        return self.reference

    @property
    def value(self) -> Any:
        """Compatibility alias for callers that call the reference a value."""
        return self.reference

    @property
    def display_reference(self) -> str:
        """Return the historical launch reference string."""
        if self.observer_window_id is not None:
            return self.observer_window_id
        process = self.runner_process if self.runner_process is not None else self.reference
        return f"pid:{process.pid}"

    def __str__(self) -> str:
        return self.display_reference

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.display_reference == other
        if not isinstance(other, LaunchHandle):
            return NotImplemented
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True, slots=True)
class LaunchHandoff:
    """A runner launch awaiting its durable handoff publication."""

    attempt_id: str
    intent_path: Path
    deadline: float
    handle: LaunchHandle | None = None


class ExecutorLaunchError(RuntimeError):
    """Base class for failures while creating or confirming a runner."""

    code = "executor_launch_failed"

    def __init__(self, message: str, *, handle: LaunchHandle | None = None) -> None:
        super().__init__(message)
        self.handle = handle
        self.reason_code = self.code


class ExecutorLaunchBackendError(ExecutorLaunchError):
    """The selected local launch backend could not create its runner."""

    code = "executor_launch_backend_failed"


class ExecutorLaunchHandoffTimeout(ExecutorLaunchError):
    """The runner did not publish its durable launch intent before the deadline."""

    code = "executor_launch_handoff_timeout"


class ExecutorLaunchFailure(ExecutorLaunchError):
    """An executor launch failure without a more specific classification."""


# Readable aliases for callers that prefer the failure-oriented class names.
ExecutorLaunchBackendFailure = ExecutorLaunchBackendError
ExecutorLaunchHandoffTimeoutError = ExecutorLaunchHandoffTimeout


def launch_failure_reason(exception: BaseException) -> str:
    """Map a launch exception to the stable Attempt result reason."""
    if isinstance(exception, ExecutorLaunchError):
        code = getattr(exception, "code", None)
        if code in _LAUNCH_FAILURE_CODES:
            return code
        reason_code = getattr(exception, "reason_code", None)
        if reason_code in _LAUNCH_FAILURE_CODES:
            return reason_code
    return "executor_launch_failed"


def launch_failure_handle(exception: BaseException) -> LaunchHandle | None:
    """Return the exact handle carried by a typed launch failure, if any."""
    handle = getattr(exception, "handle", None)
    return handle if isinstance(handle, LaunchHandle) else None


def _bounded_text(value: object, limit: int = _DIAGNOSTIC_LIMIT) -> str:
    text = str(value).replace("\x00", "\\0").replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit]


def append_launch_diagnostic(cfg: RootConfig, task_id: str, attempt_id: str, message: object) -> bool:
    """Best-effort append of one bounded launch diagnostic to the Attempt log."""
    try:
        path = shared_attempt_log_path(cfg, task_id, attempt_id)
        with path.open("ab") as log:
            log.write(f"[qexp launch diagnostic] {_bounded_text(message)}\n".encode("utf-8", "replace"))
            log.flush()
    except Exception:
        return False
    return True


def append_launch_failure_diagnostic(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    exception: BaseException,
) -> bool:
    """Append a bounded reason, type, and original launch error text."""
    reason = launch_failure_reason(exception)
    details = f"code={reason}; error_type={type(exception).__name__}; error={exception}"
    cause = exception.__cause__
    if cause is not None:
        details += f"; cause={type(cause).__name__}: {cause}"
    return append_launch_diagnostic(cfg, task_id, attempt_id, details)


def _start_runner_reaper(process: Any) -> None:
    """Reap one direct runner child without tying its lifetime to dispatch."""
    wait = getattr(process, "wait", None)
    if not callable(wait):
        raise TypeError("direct runner process must provide wait()")
    threading.Thread(
        target=wait,
        name=f"qexp-runner-reaper-{process.pid}",
        daemon=True,
    ).start()


@dataclass(slots=True)
class Executor:
    create_window: Callable[..., str] = create_window_for_task
    # Kept as an injected compatibility seam; runner bootstrap uses window_shell
    # and deliberately never types into a pane.
    send_command: Callable[[str, str], None] = send_command_to_window
    destroy_window: Callable[[str | None], None] = kill_window
    check_window: Callable[[str | None], bool] = window_exists
    tmux_available: Callable[[], bool] = is_tmux_launch_available
    observer_decision: Callable[[RootConfig, str], dict[str, Any]] = resolve_task_tmux_observation
    spawn_runner: Callable[..., Any] = subprocess.Popen

    def build_runner_command(
        self,
        cfg: RootConfig,
        task_id: str,
        attempt_id: str,
        fencing_token: int,
        launch_id: str,
    ) -> str:
        """Build the historical shell representation for diagnostics."""
        parts = self.build_runner_argv(cfg, task_id, attempt_id, fencing_token, launch_id)
        log_path = shared_attempt_log_path(cfg, task_id, attempt_id)
        return f"exec {' '.join(shlex.quote(part) for part in parts)} >> {shlex.quote(str(log_path))} 2>&1"

    @staticmethod
    def build_observer_command(cfg: RootConfig, task_id: str, attempt_id: str, process_id: int) -> str:
        """Build a non-interactive tmux log observer bound to the runner PID."""
        tail = shutil.which("tail")
        if tail is None:
            raise RuntimeError("tail is unavailable for tmux launch observation")
        log_path = shared_attempt_log_path(cfg, task_id, attempt_id)
        parts = [tail, f"--pid={process_id}", "-n", "+1", "-F", str(log_path)]
        return f"exec {' '.join(shlex.quote(part) for part in parts)}"

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
            handle, handoff = self.initiate_attempt(cfg, task_id, attempt, session_name)
        with diagnostic_span("executor.launch.handoff"):
            failures = self.wait_for_launch_handoffs([handoff])
        if failure := failures.get(handoff):
            if launch_failure_handle(failure) is None and isinstance(failure, ExecutorLaunchError):
                failure.handle = handle
            raise failure
        self.attach_observer(cfg, task_id, attempt.attempt_id, handle, session_name)
        return handle.display_reference

    def initiate_attempt(
        self, cfg: RootConfig, task_id: str, attempt: AttemptRecord, session_name: str = "experiments"
    ) -> tuple[LaunchHandle, LaunchHandoff]:
        """Start a runner and return its exact backend handle without waiting."""
        launch_id = attempt.authorization.get("launch_id")
        if not isinstance(launch_id, str):
            raise RuntimeError("Attempt has no launch authorization.")
        from .runtime.responsibility import require_launch_responsibility

        require_launch_responsibility(cfg, task_id, attempt.attempt_id, attempt.attempt_number)
        policy = resolve_launch_handoff_policy(cfg)
        timeout_seconds = validate_launch_handoff_timeout_seconds(policy["timeout_seconds"])
        if policy.get("diagnostic_reason"):
            append_launch_diagnostic(
                cfg,
                task_id,
                attempt.attempt_id,
                f"launch handoff policy resolution failed: {policy['diagnostic_reason']}",
            )

        try:
            log_path = shared_attempt_log_path(cfg, task_id, attempt.attempt_id)
            with log_path.open("ab") as log:
                process = self.spawn_runner(
                    self.build_runner_argv(
                        cfg,
                        task_id,
                        attempt.attempt_id,
                        attempt.current_fencing_token,
                        launch_id,
                    ),
                    cwd=str(cfg.project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    env=os.environ.copy(),
                    start_new_session=True,
                )
        except Exception as exc:
            raise ExecutorLaunchBackendError(f"direct runner launch failed: {exc}") from exc
        handle = LaunchHandle("detached", process, runner_process=process)
        try:
            _start_runner_reaper(process)
        except Exception as exc:
            raise ExecutorLaunchBackendError(
                f"direct runner reaper failed: {exc}",
                handle=handle,
            ) from exc

        return handle, self._launch_handoff(cfg, attempt.attempt_id, timeout_seconds, handle)

    def attach_observer(
        self,
        cfg: RootConfig,
        task_id: str,
        attempt_id: str,
        handle: LaunchHandle,
        session_name: str = "experiments",
    ) -> None:
        """Best-effort tmux observation after the runner has accepted its handoff."""
        try:
            decision = self.observer_decision(cfg, task_id)
            if not isinstance(decision, dict) or type(decision.get("enabled")) is not bool:
                raise ValueError("observer decision is malformed")
        except Exception as exc:
            decision = {
                "enabled": False,
                "diagnostic_reason": f"{type(exc).__name__}: {exc}",
            }
        diagnostic_reason = decision.get("diagnostic_reason")
        if diagnostic_reason is not None:
            append_launch_diagnostic(
                cfg,
                task_id,
                attempt_id,
                f"tmux observer policy unavailable: {diagnostic_reason}",
            )
        if not decision["enabled"]:
            return
        try:
            if not self.tmux_available():
                append_launch_diagnostic(
                    cfg, task_id, attempt_id, "tmux launch observer unavailable: tmux/libtmux unavailable"
                )
                return
            process = handle.runner_process if handle.runner_process is not None else handle.reference
            command = self.build_observer_command(cfg, task_id, attempt_id, process.pid)
            window_id = self.create_window(task_id, session_name, str(cfg.project_root), command)
            if not isinstance(window_id, str) or not window_id:
                raise RuntimeError("tmux did not return a window id")
            handle.backend = "tmux"
            handle.reference = window_id
            handle.observer_window_id = window_id
        except Exception as exc:
            append_launch_diagnostic(
                cfg,
                task_id,
                attempt_id,
                f"tmux launch observer unavailable: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _launch_handoff(
        cfg: RootConfig,
        attempt_id: str,
        timeout_seconds: int | float | None = None,
        handle: LaunchHandle | None = None,
    ) -> LaunchHandoff:
        if timeout_seconds is None:
            timeout_seconds = resolve_launch_handoff_policy(cfg)["timeout_seconds"]
            timeout_seconds = validate_launch_handoff_timeout_seconds(timeout_seconds)
        else:
            try:
                is_finite = type(timeout_seconds) in (int, float) and math.isfinite(float(timeout_seconds))
            except OverflowError:
                is_finite = False
            if not is_finite:
                raise ValueError("launch handoff timeout must be a finite positive number.")
            if timeout_seconds <= 0:
                raise ValueError("launch handoff timeout must be greater than zero.")
        path = local_paths(cfg.runtime_root)["launch_intents"] / f"{attempt_id}.json"
        return LaunchHandoff(attempt_id, path, time.monotonic() + float(timeout_seconds), handle)

    @staticmethod
    def wait_for_launch_handoffs(
        handoffs: list[LaunchHandoff],
    ) -> dict[LaunchHandoff, ExecutorLaunchError]:
        """Wait for handoffs by durable path identity, not project-local Attempt ID."""
        remaining = list(handoffs)
        failures: dict[LaunchHandoff, ExecutorLaunchError] = {}
        while remaining:
            now = time.monotonic()
            for handoff in list(remaining):
                if handoff.intent_path.exists():
                    remaining.remove(handoff)
                elif now >= handoff.deadline:
                    failures[handoff] = ExecutorLaunchHandoffTimeout(
                        f"runner did not publish launch intent for {handoff.attempt_id!r}",
                        handle=handoff.handle,
                    )
                    remaining.remove(handoff)
            if remaining:
                time.sleep(0.01)
        return failures

    @staticmethod
    def _wait_for_launch_intent(
        cfg: RootConfig,
        attempt_id: str,
        timeout_seconds: int | float | None = None,
    ) -> None:
        """Wait until the runner has durably claimed the launch handoff."""
        handoff = Executor._launch_handoff(cfg, attempt_id, timeout_seconds)
        failures = Executor.wait_for_launch_handoffs([handoff])
        if failure := failures.get(handoff):
            raise failure

    def cleanup_launch(self, launch_handle: LaunchHandle) -> None:
        """Best-effort cleanup for exactly the handle returned by a launch."""
        if not isinstance(launch_handle, LaunchHandle):
            raise TypeError("launch cleanup requires an exact LaunchHandle.")
        if launch_handle.backend == "tmux":
            if not isinstance(launch_handle.observer_window_id, str):
                raise TypeError("tmux LaunchHandle reference must be a window id string.")
            self.destroy_window(launch_handle.observer_window_id)
        elif launch_handle.backend != "detached":
            raise ValueError(f"unknown launch backend {launch_handle.backend!r}.")
        process = launch_handle.runner_process if launch_handle.runner_process is not None else launch_handle.reference
        poll = getattr(process, "poll", None)
        terminate = getattr(process, "terminate", None)
        if not callable(poll) or not callable(terminate):
            raise TypeError("LaunchHandle runner must be a Popen-like object.")
        if poll() is None:
            terminate()

    def cleanup_launch_handle(self, launch_handle: LaunchHandle) -> None:
        """Alias for the exact-handle cleanup API."""
        self.cleanup_launch(launch_handle)

    def cleanup_handle(self, launch_handle: LaunchHandle) -> None:
        """Alias for the exact-handle cleanup API."""
        self.cleanup_launch(launch_handle)

    def cleanup_window(self, window_id: str | None) -> None:
        """Retain the narrow legacy tmux cleanup seam for callers outside launch paths."""
        self.destroy_window(window_id)


__all__ = [
    "Executor",
    "ExecutorLaunchBackendError",
    "ExecutorLaunchBackendFailure",
    "ExecutorLaunchError",
    "ExecutorLaunchFailure",
    "ExecutorLaunchHandoffTimeout",
    "ExecutorLaunchHandoffTimeoutError",
    "LaunchHandle",
    "LaunchHandoff",
    "append_launch_diagnostic",
    "append_launch_failure_diagnostic",
    "launch_failure_handle",
    "launch_failure_reason",
]
