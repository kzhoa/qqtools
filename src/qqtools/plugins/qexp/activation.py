"""Global qexp agent lifecycle helpers."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable

from .agent.context import MachineRuntime, ProjectBinding
from .agent.lifecycle import (
    MachineAgentStartBlockedError,
    MachineAgentStartError,
    ensure_machine_agent_started,
    get_machine_agent_status,
    restart_machine_agent,
    run_machine_agent_loop,
    stop_machine_agent,
)
from .agent.setup import pending_machine_replacement
from .config_types import RootConfig
from .events import write_event
from .machine_config import is_legacy_agent_project


class AgentActivationError(RuntimeError):
    """An expected registration or process failure blocked agent activation."""

    def __init__(self, message: str, *, next_action: str | None = None) -> None:
        super().__init__(message)
        self.next_action = next_action


def managed_project_agent_status(
    cfg: RootConfig, *, machine_runtime: MachineRuntime | None = None
) -> tuple[MachineRuntime, ProjectBinding, dict[str, object]] | None:
    """Return global-agent ownership details for a registered project."""
    runtime = machine_runtime or MachineRuntime()
    binding = runtime.matching_binding(cfg)
    if binding is None:
        return None
    status = get_machine_agent_status(runtime)
    return (
        runtime,
        binding,
        {
            **status,
            "managed_by_machine": True,
            "project_id": binding.project_id,
            "project_state": runtime.binding_state(binding),
        },
    )


def _registration_error(cfg: RootConfig) -> AgentActivationError:
    if is_legacy_agent_project(cfg):
        command = shlex.join(
            [
                "qexp",
                "admin",
                "migrate",
                "agent",
                "--project",
                str(cfg.shared_root),
                "--machine",
                cfg.machine_name,
            ]
        )
        return AgentActivationError(f"legacy project metadata detected; run '{command}'.", next_action=command)
    return AgentActivationError(
        "project is not registered; run 'qexp project register <PATH>'.",
        next_action="qexp project register <PATH>",
    )


def _ensure_machine_agent_started(
    runtime: MachineRuntime,
) -> tuple[bool, dict[str, object]]:
    """Start the machine agent once through the shared lifecycle lock."""
    process, status = ensure_machine_agent_started(runtime)
    return process is not None, status


def _agent_start_command(runtime: MachineRuntime) -> str:
    return shlex.join(["qexp", "--machine-runtime-root", str(runtime.root), "agent", "start"])


def _replacement_resume_command(runtime: MachineRuntime) -> str:
    transaction = pending_machine_replacement(runtime)
    if transaction is None:
        return _agent_start_command(runtime)
    command = [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "init",
        "--machine",
        transaction["target_name"],
        "--agent-mode",
        transaction["policy"],
        "--yes",
    ]
    if transaction["detach_old_runtime"]:
        command.append("--detach-old-runtime")
    return shlex.join(command)


def ensure_managed_project_agent_active(
    cfg: RootConfig, *, machine_runtime: MachineRuntime | None = None
) -> tuple[str, dict[str, object]] | None:
    managed = managed_project_agent_status(cfg, machine_runtime=machine_runtime)
    if managed is None:
        return None
    runtime, _binding, status = managed
    if status["is_running"]:
        return "already_running", status
    try:
        is_started, status = _ensure_machine_agent_started(runtime)
    except MachineAgentStartBlockedError as exc:
        raise AgentActivationError(
            f"machine agent could not be started: {exc}", next_action=_replacement_resume_command(runtime)
        ) from exc
    except MachineAgentStartError as exc:
        raise AgentActivationError(
            f"machine agent could not be started: {exc}", next_action=_agent_start_command(runtime)
        ) from exc
    return ("started" if is_started else "already_running"), status


def ensure_local_agent_active(cfg: RootConfig, *, reason: str, machine_runtime: MachineRuntime | None = None) -> bool:
    """Ensure the current project is served by the sole machine agent."""
    del reason
    runtime = machine_runtime or MachineRuntime()
    if runtime.matching_binding(cfg) is None:
        raise _registration_error(cfg)
    try:
        is_started, status = _ensure_machine_agent_started(runtime)
    except MachineAgentStartBlockedError as exc:
        raise AgentActivationError(
            f"machine agent could not be started: {exc}", next_action=_replacement_resume_command(runtime)
        ) from exc
    except MachineAgentStartError as exc:
        raise AgentActivationError(
            f"machine agent could not be started: {exc}", next_action=_agent_start_command(runtime)
        ) from exc
    if is_started:
        warnings = status.get("warnings", [])
        if warnings and isinstance(warnings[0], dict):
            message = warnings[0].get("message")
            if isinstance(message, str) and message:
                print(f"Warning: {message}", file=sys.stderr)
    return is_started


def start_local_agent(
    cfg: RootConfig,
    *,
    reason: str,
    require_eligible_work: bool,
    machine_runtime: MachineRuntime | None = None,
) -> tuple[str, dict[str, object]]:
    """Compatibility wrapper that starts the unique machine agent."""
    del require_eligible_work
    runtime = machine_runtime or MachineRuntime()
    if runtime.matching_binding(cfg) is None:
        raise _registration_error(cfg)
    is_started, status = _ensure_machine_agent_started(runtime)
    if not is_started:
        return "already_running", status
    write_event(
        cfg,
        "agent_activation_started",
        details={"reason": reason, "agent_mode": "machine", "pid": status["pid"]},
    )
    return "started", status


def run_local_agent_foreground(
    cfg: RootConfig,
    *,
    reason: str,
    on_started: Callable[[dict[str, object]], None],
    machine_runtime: MachineRuntime | None = None,
) -> None:
    """Run the unique machine agent in the foreground."""
    runtime = machine_runtime or MachineRuntime()
    if runtime.matching_binding(cfg) is None:
        raise _registration_error(cfg)
    if get_machine_agent_status(runtime)["is_running"]:
        raise RuntimeError("machine agent is already running; use 'qexp agent status'.")
    write_event(
        cfg,
        "agent_activation_started",
        details={"reason": reason, "agent_mode": "machine", "pid": os.getpid()},
    )
    on_started({"agent_state": "starting", "is_running": False, "machine_runtime_root": str(runtime.root)})
    run_machine_agent_loop(runtime)


def stop_local_agent(
    cfg: RootConfig, *, timeout_seconds: float = 10.0, machine_runtime: MachineRuntime | None = None
) -> tuple[str, dict[str, object]]:
    """Stop the unique machine agent without terminating task processes."""
    del cfg
    runtime = machine_runtime or MachineRuntime()
    stopped = stop_machine_agent(runtime, timeout=timeout_seconds)
    return ("stopped" if stopped else "already_stopped"), get_machine_agent_status(runtime)


def restart_local_agent(
    cfg: RootConfig, *, machine_runtime: MachineRuntime | None = None
) -> tuple[str, dict[str, object]]:
    """Restart the unique machine agent."""
    del cfg
    runtime = machine_runtime or MachineRuntime()
    process = restart_machine_agent(runtime)
    return "restarted", {
        **get_machine_agent_status(runtime),
        "pid": process.pid,
        "previous_pid": getattr(process, "previous_pid", None),
    }
