"""Local qexp agent lifecycle control plane."""
from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable

from .agent import get_agent_status
from .agent_process import spawn_agent_process
from .config_types import RootConfig
from .events import write_event
from .layout import runtime_pid_path
from .machine_config import load_machine_policy
from .runtime.locks import exclusive
from .scheduler import has_eligible_local_work


def ensure_local_agent_active(cfg: RootConfig, *, reason: str) -> bool:
    action, _ = start_local_agent(cfg, reason=reason, require_eligible_work=True)
    return action == "started"


def start_local_agent(
        cfg: RootConfig, *, reason: str, require_eligible_work: bool) -> tuple[str, dict[str, object]]:
    """Start a detached local agent unless one is already active."""
    policy = load_machine_policy(cfg)
    write_event(
        cfg,
        "agent_activation_requested",
        details={"reason": reason, "agent_mode": policy.agent_mode},
    )
    if get_agent_status(cfg).get("is_running"):
        return "already_running", get_agent_status(cfg)
    if require_eligible_work and not has_eligible_local_work(cfg):
        return "no_eligible_work", get_agent_status(cfg)

    activation_lock = cfg.runtime_root / "locks" / "activation.lock"
    with exclusive(activation_lock, blocking=False) as acquired:
        if not acquired:
            return "already_running", get_agent_status(cfg)
        status = get_agent_status(cfg)
        if status.get("is_running"):
            return "already_running", status
        if require_eligible_work and not has_eligible_local_work(cfg):
            return "no_eligible_work", get_agent_status(cfg)
        process = spawn_agent_process(cfg)
        write_event(
            cfg,
            "agent_activation_started",
            details={"reason": reason, "agent_mode": policy.agent_mode, "pid": process.pid},
        )
        return "started", _active_status(cfg, process.pid)


def run_local_agent_foreground(
        cfg: RootConfig, *, reason: str, on_started: Callable[[dict[str, object]], None]) -> None:
    """Run one agent in the current process after claiming lifecycle ownership."""
    policy = load_machine_policy(cfg)
    activation_lock = cfg.runtime_root / "locks" / "activation.lock"
    with exclusive(activation_lock, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError("qexp agent startup is already in progress.")
        if get_agent_status(cfg).get("is_running"):
            raise RuntimeError("qexp agent is already running; use 'qexp agent status'.")
        write_event(
            cfg,
            "agent_activation_started",
            details={"reason": reason, "agent_mode": policy.agent_mode, "pid": os.getpid()},
        )
        on_started(_active_status(cfg, os.getpid()))
        from .agent import run_agent_loop
        run_agent_loop(cfg, exit_when_idle=policy.exit_when_idle)


def stop_local_agent(
        cfg: RootConfig, *, timeout_seconds: float = 2.0) -> tuple[str, dict[str, object]]:
    """Stop the local coordination process without terminating task processes."""
    status = get_agent_status(cfg)
    pid_path = runtime_pid_path(cfg)
    if not status.get("is_running"):
        pid_path.unlink(missing_ok=True)
        return "already_stopped", _stopped_status(cfg)
    pid = status["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and get_agent_status(cfg).get("is_running"):
        time.sleep(0.05)
    if get_agent_status(cfg).get("is_running"):
        raise RuntimeError(f"qexp agent pid {pid} did not stop within {timeout_seconds:g} seconds.")
    pid_path.unlink(missing_ok=True)
    return "stopped", _stopped_status(cfg)


def restart_local_agent(cfg: RootConfig) -> tuple[str, dict[str, object]]:
    """Stop the current agent, then start one detached replacement."""
    previous_pid = get_agent_status(cfg).get("pid")
    stop_local_agent(cfg)
    action, status = start_local_agent(cfg, reason="restart", require_eligible_work=False)
    if action != "started":
        raise RuntimeError("qexp agent restart could not start a replacement process.")
    status["previous_pid"] = previous_pid
    return "restarted", status


def _active_status(cfg: RootConfig, pid: int) -> dict[str, object]:
    return {
        "machine_name": cfg.machine_name,
        "agent_state": "active",
        "pid": pid,
        "is_running": True,
    }


def _stopped_status(cfg: RootConfig) -> dict[str, object]:
    return {
        "machine_name": cfg.machine_name,
        "agent_state": "stopped",
        "pid": None,
        "is_running": False,
    }
