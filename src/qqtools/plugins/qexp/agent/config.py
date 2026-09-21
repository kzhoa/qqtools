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

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_agent_name(self.name))
        object.__setattr__(self, "agent_mode", validate_agent_mode(self.agent_mode))
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


def _legacy_mode(value: Any) -> str:
    # QQTOOLS-COMPAT-0013: old per-binding records omitted mode, where the
    # historical default was on_demand.  Do not guess on malformed values.
    if value in {None, ""}:
        return AGENT_MODE_ON_DEMAND
    return validate_agent_mode(value)


def _migrate_legacy_mode(runtime: Any, *, name: str, explicit_mode: str | None) -> tuple[str, str]:
    if explicit_mode is not None:
        return validate_agent_mode(explicit_mode), "init_explicit"
    try:
        _revision, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        bindings = []
    selected = [binding for binding in bindings if binding.enabled] or list(bindings)
    modes: list[str] = []
    from ..layout import load_machine_record

    for binding in selected:
        try:
            record = load_machine_record(binding.root_config()) or {}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"cannot migrate agent mode for Project {binding.project_id!r}: {exc}") from exc
        machine = record.get("machine") if isinstance(record, dict) else None
        mode = machine.get("agent_mode") if isinstance(machine, dict) else None
        modes.append(_legacy_mode(mode))
    if not modes:
        return DEFAULT_AGENT_MODE, "default"
    if AGENT_MODE_DAEMON in modes:
        return AGENT_MODE_DAEMON, "legacy_enabled_bindings" if any(
            binding.enabled for binding in bindings
        ) else "legacy_bindings"
    return AGENT_MODE_ON_DEMAND, "legacy_enabled_bindings" if any(
        binding.enabled for binding in bindings
    ) else "legacy_bindings"


def _write_locked(runtime: Any, config: AgentConfig) -> AgentConfig:
    atomic_replace(_config_path(runtime), {"agent_config": config.to_dict()})
    return config


def load_agent_config(runtime: Any, *, require_initialized: bool = True) -> AgentConfig:
    """Load global config, performing one compatibility migration when needed."""
    machine_runtime = _as_runtime(runtime)
    stored = _stored_config(machine_runtime)
    if stored is not None:
        return stored
    if require_initialized and not machine_runtime.has_identity:
        raise RuntimeError("qexp machine runtime is uninitialized; run 'qexp init --machine NAME'.")
    # A pre-feature registry is the sole read-time migration exception.  It
    # preserves its effective runtime ID before creating the new config.
    has_registry = machine_runtime.paths["registry"].exists()
    if has_registry:
        try:
            machine_runtime.load_registry()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("machine registry is malformed; refusing global policy migration.") from exc
        machine_runtime.ensure_compatibility_identity()
    elif require_initialized:
        raise RuntimeError("qexp machine global config is missing; run 'qexp init --machine NAME'.")
    if not machine_runtime.has_identity and not has_registry:
        raise RuntimeError("qexp machine runtime is uninitialized; run 'qexp init --machine NAME'.")
    if not has_registry and not require_initialized:
        raise RuntimeError("qexp machine global agent config is missing.")
    try:
        _revision, bindings = machine_runtime.load_registry()
        first_name = bindings[0].machine_name if bindings else None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        first_name = None
    name = validate_agent_name(first_name or "machine")
    mode, provenance = _migrate_legacy_mode(machine_runtime, name=name, explicit_mode=None)
    config = AgentConfig(name=name, agent_mode=mode, revision=0, provenance=provenance)
    with machine_runtime.config_guard():
        stored = _stored_config(machine_runtime)
        return stored if stored is not None else _write_locked(machine_runtime, config)


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
                selected_mode, provenance = _migrate_legacy_mode(
                    machine_runtime, name=validated_name, explicit_mode=None
                )
        else:
            provenance = "init_explicit"
        revision = previous.revision + 1 if previous is not None else 0
        return _write_locked(
            machine_runtime,
            AgentConfig(validated_name, selected_mode, revision=revision, provenance=provenance),
        )


def set_agent_config(
    runtime: Any,
    *,
    name: str | None = None,
    agent_mode: str | None = None,
) -> AgentConfig:
    """Publish a revisioned name/mode update without replacing identity."""
    machine_runtime = _as_runtime(runtime)
    machine_runtime.require_initialized()
    if name is None and agent_mode is None:
        raise ValueError("set_agent_config requires name or agent_mode.")
    with machine_runtime.config_guard():
        current = _stored_config(machine_runtime)
        if current is None:
            current = load_agent_config(machine_runtime)
        updated = AgentConfig(
            validate_agent_name(name) if name is not None else current.name,
            validate_agent_mode(agent_mode) if agent_mode is not None else current.agent_mode,
            revision=current.revision + 1,
            provenance="configured",
        )
        return _write_locked(machine_runtime, updated)


def agent_config_payload(runtime: Any) -> dict[str, Any]:
    """Return stable facts used by both human and JSON CLI output."""
    machine_runtime = _as_runtime(runtime)
    config = load_agent_config(machine_runtime)
    return {
        "agent_name": config.name,
        "agent_mode": config.agent_mode,
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
