"""v2 task wrapper process.

Invoked by the executor inside a tmux window::

    python -m qqtools.plugins.qexp.v2.runner \
        --shared-root /mnt/share/qexp \
        --machine gpu2a \
        --task-id abc123def456 \
        [--runtime-root ~/.qqtools/qexp-runtime]

Lifecycle:
    1. Load task, validate phase == starting
    2. CAS: starting → running (set wrapper_pid, process_group_id)
    3. Spawn child process with CUDA_VISIBLE_DEVICES
    4. Stream stdout+stderr to log file and stdout
    5. On exit: classify result, CAS: running → succeeded/failed/cancelled
    6. Write scheduling event
    7. Release claim
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .events import write_event
from .indexes import update_index_on_phase_change
from .layout import (
    RootConfig,
    load_root_config,
    runtime_log_path,
)
from .models import (
    PHASE_CANCELLED,
    PHASE_FAILED,
    PHASE_RUNNING,
    PHASE_STARTING,
    PHASE_SUCCEEDED,
    utc_now_iso,
)
from .storage import (
    CASConflict,
    cas_update_task,
    load_task,
    release_claim,
)


# ---------------------------------------------------------------------------
# Child process helpers
# ---------------------------------------------------------------------------


def build_child_environment(
    assigned_gpus: list[int],
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if assigned_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned_gpus)
    if extra_env:
        env.update(extra_env)
    return env


def build_child_command(command: list[str]) -> list[str]:
    """Wrap command for exec via bash."""
    import shlex

    joined = shlex.join(command)
    return ["bash", "-c", f"exec {joined}"]


def classify_exit(return_code: int) -> tuple[str, str | None]:
    """Return (phase, terminal_reason) based on child exit code."""
    if return_code == 0:
        return PHASE_SUCCEEDED, None
    if return_code < 0 and abs(return_code) in (signal.SIGTERM, signal.SIGKILL):
        return PHASE_CANCELLED, "cancelled_by_signal"
    return PHASE_FAILED, "nonzero_exit"


def stream_output(stream: Any, log_handle: Any) -> None:
    """Read child stdout/stderr and tee to log file + stdout."""
    if stream is None:
        return
    while True:
        chunk = stream.read1(8192)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        log_handle.write(chunk)
        log_handle.flush()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_task(
    cfg: RootConfig,
    task_id: str,
    *,
    popen_factory: Any = subprocess.Popen,
) -> int:
    """Execute a single task through its full lifecycle.

    Returns the child process exit code (0 on success).
    """
    task = load_task(cfg, task_id)

    if task.status.phase != PHASE_STARTING:
        raise RuntimeError(
            f"Runner expected task in 'starting' phase, got {task.status.phase!r}."
        )

    # CAS: starting → running
    old_rev = task.meta.revision
    task.status.phase = PHASE_RUNNING
    task.runtime.wrapper_pid = os.getpid()
    task.timestamps.started_at = utc_now_iso()
    try:
        task = cas_update_task(cfg, task, old_rev)
        update_index_on_phase_change(cfg, task_id, PHASE_STARTING, PHASE_RUNNING)
    except CASConflict:
        raise RuntimeError(f"CAS conflict transitioning task {task_id} to running.")

    write_event(cfg, "task_started", task_id=task_id)

    # Prepare log
    log_path = runtime_log_path(cfg, task_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build child
    child_env = build_child_environment(task.runtime.assigned_gpus)
    child_cmd = build_child_command(task.spec.command)

    child: subprocess.Popen | None = None
    return_code: int = -1

    # Forward SIGTERM to child process group
    def _forward_signal(sig: int, frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, sig)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    with log_path.open("ab") as log_handle:
        try:
            child = popen_factory(
                child_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
        except Exception as exc:
            _finalize_task(
                cfg, task, PHASE_FAILED,
                exit_code=None,
                terminal_reason=f"launch_failed:{type(exc).__name__}",
            )
            raise

        # Update process_group_id
        old_rev = task.meta.revision
        task.runtime.process_group_id = child.pid
        try:
            task = cas_update_task(cfg, task, old_rev)
        except CASConflict:
            pass  # Best-effort; non-critical field

        # Stream output
        stream_output(child.stdout, log_handle)
        return_code = child.wait()

    # Finalize
    terminal_phase, terminal_reason = classify_exit(return_code)
    _finalize_task(
        cfg, task, terminal_phase,
        exit_code=return_code,
        terminal_reason=terminal_reason,
    )

    return return_code


def _finalize_task(
    cfg: RootConfig,
    task: Any,
    terminal_phase: str,
    exit_code: int | None,
    terminal_reason: str | None,
) -> None:
    """Write terminal state, event, and release claim."""
    old_phase = task.status.phase
    old_rev = task.meta.revision

    task.status.phase = terminal_phase
    task.status.reason = terminal_reason
    task.result.exit_code = exit_code
    task.result.terminal_reason = terminal_reason
    task.timestamps.finished_at = utc_now_iso()

    try:
        cas_update_task(cfg, task, old_rev)
        update_index_on_phase_change(cfg, task.task_id, old_phase, terminal_phase)
    except CASConflict:
        pass  # Best-effort on finalization

    # Write event
    event_type = {
        PHASE_SUCCEEDED: "task_finished",
        PHASE_FAILED: "task_failed",
        PHASE_CANCELLED: "task_cancelled",
    }.get(terminal_phase, "task_finished")
    write_event(cfg, event_type, task_id=task.task_id, details={
        "exit_code": exit_code,
        "terminal_reason": terminal_reason,
    })

    # Release claim
    release_claim(cfg, task.task_id, terminal_reason or "completed")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="qexp v2 task wrapper")
    parser.add_argument("--shared-root", required=True, type=str)
    parser.add_argument("--machine", required=True, type=str)
    parser.add_argument("--task-id", required=True, type=str)
    parser.add_argument("--runtime-root", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_root_config(args.shared_root, args.machine, args.runtime_root)
    return run_task(cfg, args.task_id)


if __name__ == "__main__":
    raise SystemExit(main())
