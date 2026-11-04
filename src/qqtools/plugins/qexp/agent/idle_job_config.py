"""Machine-local configuration and validation for the optional Idle Job."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config_types import MAX_IDLE_JOB_STOP_TIMEOUT_SECONDS
from ..runtime.store import atomic_replace, read_json_limited

IDLE_JOB_CONFIG_VERSION = 1
IDLE_JOB_CONFIG_MAX_BYTES = 65_536
MAX_IDLE_JOB_COMMAND_LENGTH = 16_384
MAX_IDLE_JOB_GPU_SELECTORS = 256

_CONFIG_FIELDS = frozenset(
    {
        "version",
        "enabled",
        "command",
        "gpu_ids",
        "idle_delay_seconds",
        "stop_timeout_seconds",
        "working_directory",
        "revision",
    }
)
_MUTABLE_FIELDS = frozenset(_CONFIG_FIELDS - {"version", "revision"})
_SHELL_OPERATOR_TOKENS = frozenset({"|", "||", "&&", ";", ">", ">>", "<"})
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_CUDA_VISIBLE_DEVICES_PREFIX = "CUDA_VISIBLE_DEVICES="


def _validate_delay(value: object, *, field_name: str, positive: bool, maximum: float | None = None) -> float:
    """Validate and normalize one JSON-compatible delay value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number; bool is not accepted.")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    if positive:
        if normalized <= 0:
            raise ValueError(f"{field_name} must be greater than zero.")
    elif normalized < 0:
        raise ValueError(f"{field_name} must be nonnegative.")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} must be at most {maximum:g} seconds.")
    return normalized


def _canonical_working_directory(value: object) -> str | None:
    """Validate and canonicalize an explicitly configured working directory."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("working_directory must be a string or None.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("working_directory must be an absolute existing directory.")
    # Persisted configuration must remain readable and repairable after unmount.
    # Availability is checked when setting a path and by the launch adapter.
    return str(path)


def _validate_gpu_selectors(value: object) -> tuple[int, ...]:
    """Validate the persisted GPU selector sequence without resolving -1."""
    if not isinstance(value, tuple):
        raise TypeError("gpu_ids must be a tuple of integers.")
    selectors = tuple(value)
    if not selectors:
        raise ValueError("gpu_ids must contain at least one selector.")
    if len(selectors) > MAX_IDLE_JOB_GPU_SELECTORS:
        raise ValueError(f"gpu_ids must contain at most {MAX_IDLE_JOB_GPU_SELECTORS} selectors.")
    for selector in selectors:
        if type(selector) is not int:
            raise TypeError("gpu_ids selectors must be integers; bool is not accepted.")
        if selector < -1:
            raise ValueError("gpu_ids selectors must be -1 or nonnegative integers.")
    return selectors


def _is_cuda_assignment(token: str) -> bool:
    return token.startswith(_CUDA_VISIBLE_DEVICES_PREFIX)


def _is_env_assignment(token: str) -> bool:
    return _ENV_ASSIGNMENT_RE.match(token) is not None


def _raise_cuda_assignment() -> None:
    raise ValueError("CUDA_VISIBLE_DEVICES overrides are unsupported; use gpu_ids to select GPUs.")


def _check_nested_env_split_string(token: str) -> None:
    """Reject CUDA assignments hidden in an env split-string argument."""
    try:
        nested_tokens = shlex.split(token)
    except ValueError:
        return
    if any(_is_cuda_assignment(nested) for nested in nested_tokens):
        _raise_cuda_assignment()


def command_argv(command: str) -> tuple[str, ...]:
    """Parse an Idle Job command for direct, shell-free execution.

    The returned tokens are suitable for ``subprocess`` execution with
    ``shell=False``. Shell operators and shell assignment prefixes are rejected
    so the configured command cannot silently acquire shell semantics.
    """
    if not isinstance(command, str):
        raise TypeError("Idle Job command must be a string.")
    if "\x00" in command:
        raise ValueError("Idle Job command must not contain NUL characters.")
    try:
        command_size = len(command.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("Idle Job command must be valid UTF-8 text.") from exc
    if command_size > MAX_IDLE_JOB_COMMAND_LENGTH:
        raise ValueError(f"Idle Job command must be at most {MAX_IDLE_JOB_COMMAND_LENGTH} UTF-8 bytes.")
    if "\n" in command or "\r" in command:
        raise ValueError("Idle Job command must be one line; invoke a script explicitly for shell scripts.")
    try:
        argv = tuple(shlex.split(command, comments=False, posix=True))
    except ValueError as exc:
        raise ValueError(f"Idle Job command has invalid quoting: {exc}.") from exc
    if not argv:
        raise ValueError("Idle Job command must not be empty.")

    if any(_is_cuda_assignment(token) for token in argv):
        _raise_cuda_assignment()
    if any(token in _SHELL_OPERATOR_TOKENS for token in argv):
        raise ValueError("Idle Job command must not contain shell operators; invoke a script explicitly.")

    first_token = argv[0]
    if _is_env_assignment(first_token):
        raise ValueError("Shell environment prefixes are unsupported; use env NAME=value executable.")

    if Path(first_token).name == "env":
        for index, token in enumerate(argv[1:], start=1):
            if token == "-S" or token.startswith("-S") and token != "-":
                if token != "-S":
                    _check_nested_env_split_string(token[2:])
                if index + 1 < len(argv):
                    _check_nested_env_split_string(argv[index + 1])
                raise ValueError("env -S/--split-string is unsupported; use env NAME=value executable.")
            if token == "--split-string" or token.startswith("--split-string="):
                if token == "--split-string" and index + 1 < len(argv):
                    _check_nested_env_split_string(argv[index + 1])
                elif token.startswith("--split-string="):
                    _check_nested_env_split_string(token.partition("=")[2])
                raise ValueError("env -S/--split-string is unsupported; use env NAME=value executable.")

    return argv


@dataclass(frozen=True, slots=True)
class IdleJobConfig:
    """Revisioned machine-local configuration for an optional Idle Job."""

    enabled: bool = False
    command: str = ""
    gpu_ids: tuple[int, ...] = (-1,)
    idle_delay_seconds: float = 60
    stop_timeout_seconds: float = 10
    working_directory: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        if not isinstance(self.command, str):
            raise TypeError("command must be a string.")
        if self.command or self.enabled:
            command_argv(self.command)
        object.__setattr__(self, "gpu_ids", _validate_gpu_selectors(self.gpu_ids))
        object.__setattr__(
            self,
            "idle_delay_seconds",
            _validate_delay(self.idle_delay_seconds, field_name="idle_delay_seconds", positive=False),
        )
        object.__setattr__(
            self,
            "stop_timeout_seconds",
            _validate_delay(
                self.stop_timeout_seconds,
                field_name="stop_timeout_seconds",
                positive=True,
                maximum=MAX_IDLE_JOB_STOP_TIMEOUT_SECONDS,
            ),
        )
        object.__setattr__(self, "working_directory", _canonical_working_directory(self.working_directory))
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a nonnegative integer.")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact versioned persistence payload for this config."""
        return {
            "version": IDLE_JOB_CONFIG_VERSION,
            "enabled": self.enabled,
            "command": self.command,
            "gpu_ids": list(self.gpu_ids),
            "idle_delay_seconds": self.idle_delay_seconds,
            "stop_timeout_seconds": self.stop_timeout_seconds,
            "working_directory": self.working_directory,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IdleJobConfig":
        """Decode one strict versioned config payload."""
        if not isinstance(value, dict):
            raise ValueError("idle job config must be an object.")
        if set(value) != _CONFIG_FIELDS:
            raise ValueError("idle job config has unknown or missing fields.")
        if type(value["version"]) is not int or value["version"] != IDLE_JOB_CONFIG_VERSION:
            raise ValueError(f"unsupported idle job config version {value['version']!r}.")
        gpu_ids = value["gpu_ids"]
        if not isinstance(gpu_ids, list):
            raise ValueError("idle job config gpu_ids must be a JSON array.")
        return cls(
            enabled=value["enabled"],
            command=value["command"],
            gpu_ids=tuple(gpu_ids),
            idle_delay_seconds=value["idle_delay_seconds"],
            stop_timeout_seconds=value["stop_timeout_seconds"],
            working_directory=value["working_directory"],
            revision=value["revision"],
        )


def resolve_idle_job_gpus(
    selectors: tuple[int, ...] | list[int],
    eligible_gpu_ids: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Resolve selectors against a discovered, explicitly allowed GPU inventory."""
    if not isinstance(selectors, (tuple, list)):
        raise ValueError("Idle Job gpu_ids must be a tuple or list of selectors.")
    if not selectors:
        raise ValueError("Idle Job gpu_ids must contain at least one selector.")
    if len(selectors) > MAX_IDLE_JOB_GPU_SELECTORS:
        raise ValueError(f"Idle Job gpu_ids must contain at most {MAX_IDLE_JOB_GPU_SELECTORS} selectors.")
    if eligible_gpu_ids is None or not isinstance(eligible_gpu_ids, (tuple, list)) or not eligible_gpu_ids:
        raise ValueError("Idle Job GPU inventory is unavailable; refusing unrestricted GPU selection.")

    eligible = tuple(eligible_gpu_ids)
    for gpu_id in eligible:
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("eligible GPU IDs must be nonnegative integers; bool is not accepted.")
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible GPU IDs must not contain duplicates.")
    allowed = set(eligible)
    final_gpu_id = sorted(allowed)[-1]

    resolved: list[int] = []
    seen: set[int] = set()
    for selector in selectors:
        if type(selector) is not int:
            raise ValueError("Idle Job GPU selectors must be integers; bool is not accepted.")
        if selector < -1:
            raise ValueError("Idle Job GPU selectors must be -1 or nonnegative integers.")
        gpu_id = final_gpu_id if selector == -1 else selector
        if gpu_id not in allowed:
            raise ValueError(f"Idle Job GPU selector {selector} is not in the discovered allowed inventory.")
        if gpu_id not in seen:
            seen.add(gpu_id)
            resolved.append(gpu_id)
    return tuple(resolved)


def validate_idle_job_mode(config: IdleJobConfig, agent_mode: str) -> None:
    """Reject an enabled Idle Job when the agent is configured on-demand."""
    if not isinstance(config, IdleJobConfig):
        raise TypeError("config must be an IdleJobConfig.")
    if type(agent_mode) is not str or agent_mode not in {"daemon", "on_demand"}:
        raise ValueError("agent_mode must be 'daemon' or 'on_demand'.")
    if config.enabled and agent_mode == "on_demand":
        raise ValueError("enabled Idle Job requires daemon agent mode; on_demand mode is incompatible.")


def _runtime_root(runtime: Any) -> Path:
    """Return a runtime root without initializing or laying out the runtime."""
    if isinstance(runtime, (str, Path)):
        return Path(runtime)
    root = getattr(runtime, "root", None)
    if root is None:
        raise TypeError("runtime must provide a root path.")
    return Path(root)


def _config_path(runtime: Any) -> Path:
    return _runtime_root(runtime) / "idle-job" / "config.json"


def load_idle_job_config(runtime: Any) -> IdleJobConfig:
    """Read the machine-local Idle Job config without requiring initialization."""
    path = _config_path(runtime)
    try:
        envelope = read_json_limited(
            path,
            max_bytes=IDLE_JOB_CONFIG_MAX_BYTES,
            record_type="idle_job_config",
        )
    except FileNotFoundError:
        return IdleJobConfig()
    if set(envelope) != {"idle_job"}:
        raise ValueError("idle job config envelope is malformed.")
    raw_config = envelope.get("idle_job")
    if not isinstance(raw_config, dict):
        raise ValueError("idle job config envelope must contain an object under 'idle_job'.")
    return IdleJobConfig.from_dict(raw_config)


def _as_mutable_runtime(runtime: Any) -> Any:
    if all(hasattr(runtime, attribute) for attribute in ("root", "require_initialized", "config_guard")):
        return runtime
    from .context import MachineRuntime

    return MachineRuntime(runtime)


def set_idle_job_config(runtime: Any, **changes: Any) -> IdleJobConfig:
    """Validate and atomically publish a revisioned Idle Job config update."""
    machine_runtime = _as_mutable_runtime(runtime)
    machine_runtime.require_initialized()
    if not changes:
        raise ValueError("set_idle_job_config requires at least one mutable field.")
    unknown = set(changes) - _MUTABLE_FIELDS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unknown or immutable Idle Job config field(s): {names}.")

    with machine_runtime.config_guard() as acquired:
        if acquired is False:
            raise RuntimeError("could not acquire the machine config lock.")
        current = load_idle_job_config(machine_runtime)
        updated = replace(current, **changes, revision=current.revision + 1)
        if updated.working_directory is not None and (updated.enabled or "working_directory" in changes):
            if not Path(updated.working_directory).is_dir():
                raise ValueError("working_directory must be an absolute existing directory.")
        from .config import load_agent_config

        agent_config = load_agent_config(machine_runtime)
        validate_idle_job_mode(updated, agent_config.agent_mode)
        atomic_replace(_config_path(machine_runtime), {"idle_job": updated.to_dict()})
        return updated


__all__ = [
    "IDLE_JOB_CONFIG_MAX_BYTES",
    "IDLE_JOB_CONFIG_VERSION",
    "IdleJobConfig",
    "command_argv",
    "load_idle_job_config",
    "resolve_idle_job_gpus",
    "set_idle_job_config",
    "validate_idle_job_mode",
]
