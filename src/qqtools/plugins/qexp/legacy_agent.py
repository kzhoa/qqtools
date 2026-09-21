"""Legacy-agent inspection helpers and GPU discovery."""

from __future__ import annotations

import os
from typing import Any

from .config_types import RootConfig
from .gpu_policy import discover_gpu_inventory, parse_gpu_id_list
from .layout import runtime_pid_path
from .lease import clock_capability


def get_agent_status(cfg: RootConfig, probe_local_pid: bool = True) -> dict[str, Any]:
    pid_path = runtime_pid_path(cfg)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else None
    except (OSError, ValueError):
        pid = None
    running = bool(pid and (not probe_local_pid or _pid_alive(pid)))
    capability = clock_capability(cfg)
    return {
        "machine_name": cfg.machine_name,
        "agent_state": "active" if running else "stopped",
        "pid": pid,
        "is_running": running,
        "clock_capability": capability.status,
        "clock_reason": capability.reason,
        "clock_provider": capability.observation.provider if capability.observation else None,
        "scheduling_capability": "full" if capability.is_healthy else "local-safe",
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _visible_gpus(cfg: RootConfig) -> list[int]:
    """Return the legacy visible helper result for compatibility callers.

    Machine dispatch uses :func:`discover_gpu_inventory` directly so a raw
    inventory observation is never short-circuited by the environment policy.
    """
    value = os.environ.get("QEXP_VISIBLE_GPUS", "")
    if value:
        try:
            return list(parse_gpu_id_list(value))
        except ValueError:
            return []
    discovery = discover_gpu_inventory()
    return list(discovery.gpu_ids or ())


def run_agent_loop(*_args: object, **_kwargs: object) -> None:
    """Reject the removed standalone agent runtime."""
    raise RuntimeError("standalone agent runtime was removed; use 'qexp agent add-project' then 'qexp agent start'.")


def start_agent(*_args: object, **_kwargs: object) -> None:
    """Reject the removed standalone agent runtime."""
    run_agent_loop()
