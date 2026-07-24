"""Machine policy loading and normalization for qexp agent lifecycle."""
from __future__ import annotations

from .config_types import MachinePolicy
from .models import AGENT_MODE_DAEMON, AGENT_MODE_ON_DEMAND

AGENT_MODES = frozenset({AGENT_MODE_ON_DEMAND, AGENT_MODE_DAEMON})


def normalize_agent_mode(agent_mode: str | None) -> str:
    if agent_mode in {None, "", AGENT_MODE_ON_DEMAND}:
        return AGENT_MODE_ON_DEMAND
    if agent_mode == AGENT_MODE_DAEMON:
        return AGENT_MODE_DAEMON
    raise ValueError(f"Unsupported qexp agent mode {agent_mode!r}.")


def resolve_machine_policy(agent_mode: str | None) -> MachinePolicy:
    normalized = normalize_agent_mode(agent_mode)
    if normalized == AGENT_MODE_ON_DEMAND:
        return MachinePolicy(agent_mode=normalized, autostart_local=True, exit_when_idle=True)
    return MachinePolicy(agent_mode=normalized, autostart_local=False, exit_when_idle=False)
