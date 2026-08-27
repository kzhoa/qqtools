"""Global qexp agent lifecycle helpers."""
from __future__ import annotations

import os
from collections.abc import Callable

from .config_types import RootConfig
from .events import write_event
from .machine_agent import (
    ensure_machine_agent_started,
    get_machine_agent_status,
    restart_machine_agent,
    run_machine_agent_loop,
    stop_machine_agent,
)
from .machine_config import is_legacy_agent_project
from .machine_runtime import MachineRuntime, ProjectBinding


def managed_project_agent_status(
    cfg: RootConfig, *, machine_runtime: MachineRuntime | None = None
) -> tuple[MachineRuntime, ProjectBinding, dict[str, object]] | None:
    """Return global-agent ownership details for a registered project."""
    runtime = machine_runtime or MachineRuntime()
    binding = runtime.matching_binding(cfg)
    if binding is None:
        return None
    status = get_machine_agent_status(runtime)
    return runtime, binding, {
        **status,
        "managed_by_machine": True,
        "project_id": binding.project_id,
        "project_state": runtime.binding_state(binding),
    }


def _registration_error(cfg: RootConfig) -> RuntimeError:
    if is_legacy_agent_project(cfg):
        return RuntimeError("legacy project metadata detected; run 'qexp agent migrate-project'.")
    return RuntimeError("project is not registered; run 'qexp agent add-project'.")


def _ensure_machine_agent_started(
    runtime: MachineRuntime,
) -> tuple[bool, dict[str, object]]:
    """Start the machine agent once through the shared lifecycle lock."""
    process, status = ensure_machine_agent_started(runtime)
    return process is not None, status


def ensure_managed_project_agent_active(
    cfg: RootConfig, *, machine_runtime: MachineRuntime | None = None
) -> tuple[str, dict[str, object]] | None:
    managed = managed_project_agent_status(cfg, machine_runtime=machine_runtime)
    if managed is None:
        return None
    runtime, _binding, status = managed
    if status["is_running"]:
        return "already_running", status
    is_started, status = _ensure_machine_agent_started(runtime)
    return ("started" if is_started else "already_running"), status


def ensure_local_agent_active(
    cfg: RootConfig, *, reason: str, machine_runtime: MachineRuntime | None = None
) -> bool:
    """Ensure the current project is served by the sole machine agent."""
    del reason
    runtime = machine_runtime or MachineRuntime()
    if runtime.matching_binding(cfg) is None:
        raise _registration_error(cfg)
    is_started, _status = _ensure_machine_agent_started(runtime)
    return is_started


def start_local_agent(
    cfg: RootConfig, *, reason: str, require_eligible_work: bool,
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
    cfg: RootConfig, *, reason: str, on_started: Callable[[dict[str, object]], None],
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
    return "restarted", {**get_machine_agent_status(runtime), "pid": process.pid}
