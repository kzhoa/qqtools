"""Revisioned machine-global agent configuration.

The effective binding registry remains the authority for project names and
generations.  This module only owns the machine-wide default name and
residency policy, so changing either value cannot rewrite a Project binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import AGENT_MODE_DAEMON, AGENT_MODE_ON_DEMAND
from ..runtime.store import atomic_replace, read_json
from .diagnostics import DEFAULT_LOG_MAX_BYTES, format_log_size, parse_log_size

AGENT_CONFIG_VERSION = 1
AGENT_MODES = frozenset({AGENT_MODE_DAEMON, AGENT_MODE_ON_DEMAND})
DEFAULT_AGENT_MODE = AGENT_MODE_DAEMON


def validate_agent_name(name: str) -> str:
    """Validate and return one logical machine name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("agent name must be a non-empty string.")
    value = name.strip()
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError("agent name must not contain path separators or '..'.")
    return value


def validate_agent_mode(agent_mode: str) -> str:
    """Validate one global residency policy."""
    if not isinstance(agent_mode, str) or agent_mode not in AGENT_MODES:
        raise ValueError(f"agent mode must be one of: {AGENT_MODE_DAEMON}, {AGENT_MODE_ON_DEMAND}.")
    return agent_mode


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Durable global agent policy."""

    name: str
    agent_mode: str = DEFAULT_AGENT_MODE
    revision: int = 0
    provenance: str = "default"
    log_max_bytes: int = DEFAULT_LOG_MAX_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_agent_name(self.name))
        object.__setattr__(self, "agent_mode", validate_agent_mode(self.agent_mode))
        object.__setattr__(self, "log_max_bytes", parse_log_size(self.log_max_bytes))
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("agent config revision must be a non-negative integer.")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("agent config provenance must be a non-empty string.")

    @property
    def exit_when_idle(self) -> bool:
        return self.agent_mode == AGENT_MODE_ON_DEMAND

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": AGENT_CONFIG_VERSION,
            "name": self.name,
            "agent_mode": self.agent_mode,
            "log_max_bytes": self.log_max_bytes,
            "revision": self.revision,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentConfig":
        if not isinstance(value, dict):
            raise ValueError("agent config must be an object.")
        version = value.get("version", AGENT_CONFIG_VERSION)
        if version != AGENT_CONFIG_VERSION:
            raise ValueError(f"unsupported agent config version {version!r}.")
        try:
            return cls(
                name=value["name"],
                agent_mode=value.get("agent_mode", DEFAULT_AGENT_MODE),
                log_max_bytes=value.get("log_max_bytes", DEFAULT_LOG_MAX_BYTES),
                revision=value.get("revision", 0),
                provenance=value.get("provenance", "default"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("agent config is malformed.") from exc


GlobalAgentConfig = AgentConfig


def _as_runtime(runtime: Any):
    if hasattr(runtime, "paths") and hasattr(runtime, "root"):
        return runtime
    from .context import MachineRuntime

    return MachineRuntime(runtime)


def _config_path(runtime: Any) -> Path:
    return _as_runtime(runtime).paths["global_config"]


def _stored_config(runtime: Any) -> AgentConfig | None:
    path = _config_path(runtime)
    if not path.exists():
        return None
    value = read_json(path)
    raw = value.get("agent_config")
    if raw is None:
        # Permit a short-lived early development shape while retaining strict
        # field validation for all values that are actually consumed.
        raw = value.get("config")
    if not isinstance(raw, dict):
        raise RuntimeError("machine global agent config is malformed.")
    try:
        return AgentConfig.from_dict(raw)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _write_locked(runtime: Any, config: AgentConfig) -> AgentConfig:
    atomic_replace(_config_path(runtime), {"agent_config": config.to_dict()})
    return config


def load_agent_config(runtime: Any, *, require_initialized: bool = True) -> AgentConfig:
    """Load the canonical machine-global agent configuration."""
    machine_runtime = _as_runtime(runtime)
    if require_initialized:
        machine_runtime.require_identity()
    stored = _stored_config(machine_runtime)
    if stored is not None:
        return stored
    raise RuntimeError("qexp machine global config is missing; run 'qexp init --machine NAME'.")


def read_agent_config(runtime: Any, *, require_initialized: bool = True) -> AgentConfig:
    """Compatibility alias for callers that use read terminology."""
    return load_agent_config(runtime, require_initialized=require_initialized)


def initialize_agent_config(runtime: Any, name: str, *, agent_mode: str | None = None) -> AgentConfig:
    """Create or replace global config as part of a machine initialization."""
    machine_runtime = _as_runtime(runtime)
    validated_name = validate_agent_name(name)
    selected_mode = validate_agent_mode(agent_mode) if agent_mode is not None else None
    with machine_runtime.config_guard():
        previous = _stored_config(machine_runtime)
        if selected_mode is None:
            if previous is not None:
                selected_mode = previous.agent_mode
                provenance = previous.provenance
            else:
                selected_mode = DEFAULT_AGENT_MODE
                provenance = "default"
        else:
            provenance = "init_explicit"
        revision = previous.revision + 1 if previous is not None else 0
        return _write_locked(
            machine_runtime,
            AgentConfig(
                validated_name,
                selected_mode,
                log_max_bytes=previous.log_max_bytes if previous is not None else DEFAULT_LOG_MAX_BYTES,
                revision=revision,
                provenance=provenance,
            ),
        )


def set_agent_config(
    runtime: Any,
    *,
    name: str | None = None,
    agent_mode: str | None = None,
    log_max_bytes: str | int | None = None,
) -> AgentConfig:
    """Publish a revisioned global agent configuration update."""
    machine_runtime = _as_runtime(runtime)
    machine_runtime.require_initialized()
    if name is None and agent_mode is None and log_max_bytes is None:
        raise ValueError("set_agent_config requires name, agent_mode, or log_max_bytes.")
    with machine_runtime.config_guard():
        current = _stored_config(machine_runtime)
        if current is None:
            current = load_agent_config(machine_runtime)
        updated = AgentConfig(
            validate_agent_name(name) if name is not None else current.name,
            validate_agent_mode(agent_mode) if agent_mode is not None else current.agent_mode,
            log_max_bytes=parse_log_size(log_max_bytes) if log_max_bytes is not None else current.log_max_bytes,
            revision=current.revision + 1,
            provenance="configured",
        )
        return _write_locked(machine_runtime, updated)


def agent_config_payload(runtime: Any) -> dict[str, Any]:
    """Return stable facts used by both human and JSON CLI output."""
    machine_runtime = _as_runtime(runtime)
    config = load_agent_config(machine_runtime)
    return {
        "machine_runtime_root": str(machine_runtime.root),
        "agent_name": config.name,
        "agent_mode": config.agent_mode,
        "log_max_bytes": config.log_max_bytes,
        "log_max_size": format_log_size(config.log_max_bytes),
        "revision": config.revision,
        "provenance": config.provenance,
        "runtime_id": machine_runtime.instance_id,
    }


__all__ = [
    "AGENT_CONFIG_VERSION",
    "AGENT_MODES",
    "AgentConfig",
    "GlobalAgentConfig",
    "agent_config_payload",
    "initialize_agent_config",
    "load_agent_config",
    "read_agent_config",
    "set_agent_config",
    "validate_agent_mode",
    "validate_agent_name",
]
