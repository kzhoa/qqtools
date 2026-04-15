"""v2 agent — on-demand or persistent agent with tmux-based lifecycle.

The agent is the single process responsible for dispatching queued tasks
on this machine. It runs in a tmux window and writes heartbeats to
state/agent.json so other commands can observe its status.

Lifecycle:
    1. Preflight: verify Linux, tmux, libtmux, GPU backend
    2. Acquire runtime lock (fcntl on agent.pid)
    3. Initialize GPU tracker
    4. Main loop: refresh tracker → reconcile → dispatch → heartbeat
    5. On idle timeout (on_demand) or signal: clean exit
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import shlex
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .events import write_event
from .executor import Executor
from .lifecycle import (
    build_machine_workset,
    build_starting_agent_snapshot,
    build_stopped_agent_snapshot,
    build_summary_snapshot,
    derive_agent_state,
    read_agent_snapshot,
    summarize_agent_snapshot,
    write_agent_snapshot,
    write_summary_snapshot,
)
from .layout import (
    RootConfig,
    gpu_state_path,
    runtime_pid_path,
)
from .models import (
    AGENT_MODE_ON_DEMAND,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_DRAINING,
    AGENT_STATE_FAILED,
    AGENT_STATE_IDLE,
    AGENT_STATE_STARTING,
    AGENT_STATE_STALE,
    AGENT_STATE_STOPPED,
    AgentSnapshot,
    MachineWorkset,
    utc_now_iso,
)
from .storage import read_json, write_atomic_json

log = logging.getLogger(__name__)

IDLE_TIMEOUT_DEFAULT = 600  # seconds
LOOP_INTERVAL_DEFAULT = 5.0
HEARTBEAT_STALE_SECONDS = 30


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class PreflightResult:
    __slots__ = ("gpu_backend", "visible_gpu_ids")

    def __init__(self, gpu_backend: str, visible_gpu_ids: list[int]) -> None:
        self.gpu_backend = gpu_backend
        self.visible_gpu_ids = visible_gpu_ids


def run_preflight_checks() -> PreflightResult:
    """Verify runtime prerequisites: Linux, tmux, libtmux, GPU backend."""
    if platform.system() != "Linux":
        raise RuntimeError("qexp agent is only supported on Linux.")

    from ..tmux import is_tmux_executable_available, require_libtmux

    if not is_tmux_executable_available():
        raise RuntimeError("tmux is required. Install it and try again.")

    require_libtmux()

    from ..tracker import probe_gpu_backend

    backend, gpu_ids = probe_gpu_backend()
    if backend is None or not gpu_ids:
        raise RuntimeError(
            "No GPUs detected. Ensure pynvml or nvidia-smi is available."
        )

    return PreflightResult(gpu_backend=backend, visible_gpu_ids=gpu_ids)


# ---------------------------------------------------------------------------
# Agent state file I/O
# ---------------------------------------------------------------------------


def read_agent_state(cfg: RootConfig) -> dict[str, Any] | None:
    snapshot = read_agent_snapshot(cfg)
    if snapshot is None:
        return None
    return summarize_agent_snapshot(snapshot)


def is_agent_running(cfg: RootConfig) -> bool:
    state = read_agent_state(cfg)
    if state is None:
        return False
    pid = state.get("pid")
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_agent_running_from_snapshot(snapshot: AgentSnapshot | None) -> bool:
    if snapshot is None or snapshot.pid is None:
        return False
    try:
        os.kill(snapshot.pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# GPU and summary snapshots
# ---------------------------------------------------------------------------


def write_gpu_state(cfg: RootConfig, tracker: Any) -> None:
    payload = {
        "visible_gpu_ids": list(tracker.visible_gpu_ids),
        "reserved_gpu_ids": sorted(tracker.reserved_gpu_ids),
        "task_to_gpu_ids": {
            tid: list(gpus)
            for tid, gpus in tracker.task_id_to_gpu_ids.items()
        },
        "updated_at": utc_now_iso(),
    }
    write_atomic_json(gpu_state_path(cfg), payload)


def write_heartbeat(cfg: RootConfig) -> None:
    snapshot = read_agent_snapshot(cfg)
    if snapshot is None:
        return
    snapshot.last_heartbeat = utc_now_iso()
    write_agent_snapshot(cfg, snapshot)


# ---------------------------------------------------------------------------
# Runtime lock (prevents multiple agents on same machine)
# ---------------------------------------------------------------------------


class _RuntimeLock:
    """Exclusive lock on runtime_root/agent.pid via fcntl."""

    def __init__(self, cfg: RootConfig) -> None:
        self.pid_path = runtime_pid_path(cfg)
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.pid_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        # Write PID
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


def start_agent_record(
    cfg: RootConfig,
    persistent: bool = False,
    idle_timeout: int = IDLE_TIMEOUT_DEFAULT,
) -> None:
    snapshot = build_starting_agent_snapshot(
        cfg,
        persistent=persistent,
        pid=os.getpid(),
        idle_timeout_seconds=idle_timeout,
        started_at=utc_now_iso(),
    )
    write_agent_snapshot(cfg, snapshot)


def stop_agent_record(cfg: RootConfig, reason: str = "clean_exit") -> None:
    previous = read_agent_snapshot(cfg)
    if previous is None:
        previous = build_starting_agent_snapshot(
            cfg,
            persistent=False,
            pid=os.getpid(),
            started_at=utc_now_iso(),
        )
    snapshot = build_stopped_agent_snapshot(previous, reason=reason, stopped_at=utc_now_iso())
    write_agent_snapshot(cfg, snapshot)


def get_agent_status(cfg: RootConfig, probe_local_pid: bool = True) -> dict[str, Any]:
    snapshot = read_agent_snapshot(cfg)
    if snapshot is None:
        return {"agent_state": AGENT_STATE_STOPPED, "pid": None, "is_running": False}
    running = (
        is_agent_running_from_snapshot(snapshot)
        if probe_local_pid
        else snapshot.agent_state not in {
            AGENT_STATE_STOPPED,
            AGENT_STATE_STALE,
            AGENT_STATE_FAILED,
        }
    )
    result = summarize_agent_snapshot(snapshot)
    result["is_running"] = running
    if probe_local_pid and not running and snapshot.agent_state in (
        AGENT_STATE_STARTING,
        AGENT_STATE_ACTIVE,
        AGENT_STATE_DRAINING,
        AGENT_STATE_IDLE,
    ):
        result["agent_state"] = "stale"
    return result


def run_agent_loop(
    cfg: RootConfig,
    persistent: bool = False,
    loop_interval: float = LOOP_INTERVAL_DEFAULT,
    idle_timeout: int = IDLE_TIMEOUT_DEFAULT,
    # Injection points for testing
    dispatch_fn: Any = None,
    reconcile_fn: Any = None,
    tracker_factory: Any = None,
) -> int:
    """Run the agent main loop.

    In production (tracker_factory=None), performs full preflight,
    creates a real GPU tracker, and dispatches via tmux+executor.

    For testing, inject stubs via dispatch_fn/reconcile_fn/tracker_factory.
    """
    from .scheduler import Scheduler, run_dispatch_cycle, reconcile_running_tasks

    # Runtime lock
    runtime_lock = _RuntimeLock(cfg)
    if not runtime_lock.acquire():
        raise RuntimeError(
            f"Another agent is already running for machine {cfg.machine_name}."
        )

    tracker = None
    scheduler = None

    if tracker_factory is not None:
        tracker = tracker_factory()
    elif dispatch_fn is None:
        # Production mode: preflight + real tracker
        preflight = run_preflight_checks()
        from ..tracker import qExpTracker
        tracker = qExpTracker(
            visible_gpu_ids=preflight.visible_gpu_ids,
            backend_name=preflight.gpu_backend,
            gpu_probe=lambda: (preflight.gpu_backend, preflight.visible_gpu_ids),
        )
        scheduler = Scheduler(tracker=tracker)

    start_agent_record(cfg, persistent=persistent, idle_timeout=idle_timeout)
    agent_snapshot = read_agent_snapshot(cfg)
    write_event(cfg, "agent_started")
    shutdown = False
    exit_reason = "clean_exit"

    def _handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown
        shutdown = True

    prev_term = signal.signal(signal.SIGTERM, _handle_signal)
    prev_int = signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not shutdown:
            try:
                if scheduler is not None:
                    # Full production dispatch
                    tracker.refresh_visibility()
                    # Rebuild reservations from v2 storage
                    _rebuild_tracker_from_v2(cfg, tracker)
                    write_gpu_state(cfg, tracker)

                    failed = scheduler.reconcile_running_tasks(cfg)
                    if failed:
                        _rebuild_tracker_from_v2(cfg, tracker)

                    launched = scheduler.run_dispatch_cycle(cfg)
                elif dispatch_fn is not None:
                    launched = dispatch_fn(cfg, tracker_factory)
                    if reconcile_fn is not None:
                        reconcile_fn(cfg)
                else:
                    launched = run_dispatch_cycle(cfg, tracker)
                    reconcile_running_tasks(cfg)
            except Exception:
                log.exception("Agent cycle error")
            cycle_now = utc_now_iso()
            workset = build_machine_workset(cfg, updated_at=cycle_now)
            agent_snapshot = _build_live_agent_snapshot(
                cfg=cfg,
                previous=agent_snapshot,
                workset=workset,
                persistent=persistent,
                idle_timeout=idle_timeout,
            )
            write_agent_snapshot(cfg, agent_snapshot)
            try:
                write_summary_snapshot(cfg, build_summary_snapshot(workset))
            except Exception:
                pass

            if (
                not persistent
                and agent_snapshot.agent_state == AGENT_STATE_IDLE
                and agent_snapshot.idle_deadline_at is not None
                and _iso_is_due(agent_snapshot.idle_deadline_at, cycle_now)
            ):
                log.info("Idle timeout reached (%ds), exiting.", idle_timeout)
                exit_reason = "idle_timeout"
                break

            time.sleep(loop_interval)
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)
        reason = "signal" if shutdown else exit_reason
        stop_agent_record(cfg, reason=reason)
        write_event(cfg, "agent_stopped", details={"reason": reason})
        runtime_lock.release()

    return 0


def _rebuild_tracker_from_v2(cfg: RootConfig, tracker: Any) -> None:
    """Rebuild tracker reservations from v2 running tasks (not v1 fsqueue)."""
    from .storage import iter_all_tasks
    from .models import PHASE_RUNNING, PHASE_DISPATCHING, PHASE_STARTING

    tracker.reserved_gpu_ids = set()
    tracker.task_id_to_gpu_ids = {}

    for task in iter_all_tasks(cfg):
        if task.status.phase not in (PHASE_RUNNING, PHASE_DISPATCHING, PHASE_STARTING):
            continue
        if task.machine_name != cfg.machine_name:
            continue
        if task.runtime.assigned_gpus:
            gpu_ids = list(task.runtime.assigned_gpus)
            tracker.task_id_to_gpu_ids[task.task_id] = gpu_ids
            tracker.reserved_gpu_ids.update(gpu_ids)


def _build_live_agent_snapshot(
    cfg: RootConfig,
    previous: AgentSnapshot | None,
    workset: MachineWorkset,
    persistent: bool,
    idle_timeout: int,
) -> AgentSnapshot:
    workset_updated_at = workset.updated_at
    agent_state = derive_agent_state(workset)
    previous_state = previous.agent_state if previous is not None else None

    idle_started_at = None
    idle_deadline_at = None
    if agent_state == AGENT_STATE_IDLE:
        if previous_state == AGENT_STATE_IDLE and previous is not None:
            idle_started_at = previous.idle_started_at or workset_updated_at
        else:
            idle_started_at = workset_updated_at
        if not persistent:
            idle_deadline_at = _iso_add_seconds(idle_started_at, idle_timeout)

    drain_started_at = None
    if agent_state == AGENT_STATE_DRAINING:
        if previous_state == AGENT_STATE_DRAINING and previous is not None:
            drain_started_at = previous.drain_started_at or workset_updated_at
        else:
            drain_started_at = workset_updated_at

    started_at = previous.started_at if previous is not None else workset_updated_at
    last_transition_at = (
        previous.last_transition_at
        if previous is not None and previous_state == agent_state
        else workset_updated_at
    )

    return AgentSnapshot(
        schema_version=previous.schema_version if previous is not None else "3.0",
        machine_name=cfg.machine_name,
        agent_mode="persistent" if persistent else AGENT_MODE_ON_DEMAND,
        agent_state=agent_state,
        pid=os.getpid(),
        started_at=started_at,
        last_heartbeat=workset_updated_at,
        last_transition_at=last_transition_at,
        idle_timeout_seconds=0 if persistent else idle_timeout,
        idle_started_at=idle_started_at,
        idle_deadline_at=idle_deadline_at,
        drain_started_at=drain_started_at,
        last_exit_reason=None,
        workset=workset,
    )


def _iso_add_seconds(value: str, seconds: int) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        (dt + timedelta(seconds=seconds))
        .astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _iso_is_due(deadline_at: str, current_at: str) -> bool:
    deadline = datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
    current = datetime.fromisoformat(current_at.replace("Z", "+00:00"))
    return current >= deadline


# ---------------------------------------------------------------------------
# Wake agent (called by submit/retry)
# ---------------------------------------------------------------------------


def wake_agent_if_needed(cfg: RootConfig) -> bool:
    """If agent is not running, start it in a tmux background window.

    Returns True if agent was already running or successfully started.
    """
    if is_agent_running(cfg):
        return True

    try:
        _start_agent_background(cfg)
        return True
    except Exception:
        log.warning(
            "Could not auto-start agent for %s. "
            "Start manually with 'qexp agent start'.",
            cfg.machine_name,
            exc_info=True,
        )
        return False


def _start_agent_background(cfg: RootConfig) -> str:
    """Launch the agent in a tmux window (background)."""
    from ..tmux import ensure_managed_session, send_command_to_window

    session = ensure_managed_session(
        "qqtools_internal",
        "internal",
        initial_window_name="agent",
    )

    window = session.windows.get(window_name="agent")
    if window is None:
        window = session.new_window(window_name="agent", attach=False)

    command = " ".join([
        shlex.quote(sys.executable),
        "-m",
        "qqtools.plugins.qexp.v2.agent",
        "--shared-root",
        shlex.quote(str(cfg.shared_root)),
        "--machine",
        shlex.quote(cfg.machine_name),
        "--runtime-root",
        shlex.quote(str(cfg.runtime_root)),
    ])

    send_command_to_window(str(window.window_id), command)
    return str(window.window_id)


# ---------------------------------------------------------------------------
# CLI entry point (for tmux background launch)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .layout import load_root_config

    parser = argparse.ArgumentParser(description="qexp v2 agent")
    parser.add_argument("--shared-root", required=True, type=str)
    parser.add_argument("--machine", required=True, type=str)
    parser.add_argument("--runtime-root", type=str, default=None)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=None,
        help="Override idle timeout in seconds",
    )
    args = parser.parse_args(argv)

    cfg = load_root_config(args.shared_root, args.machine, args.runtime_root)
    idle = args.idle_timeout if args.idle_timeout is not None else IDLE_TIMEOUT_DEFAULT
    return run_agent_loop(cfg, persistent=args.persistent, idle_timeout=idle)


if __name__ == "__main__":
    raise SystemExit(main())
