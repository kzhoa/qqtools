"""Core typed output contracts and the closed qexp renderer registry."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar


class OutputKind(str, Enum):
    """Finite set of structured qexp CLI output families."""

    TASK_LIST = "task-list"
    TASK_PAGE = "task-page"
    TASK_SHOW = "task-show"
    TASK_WATCH = "task-watch"
    TASK_CANCEL = "task-cancel"
    TASK_RETRY = "task-retry"
    DEPENDENCIES = "dependencies"
    AVAILABILITY = "availability"
    GROUP_LIST = "group-list"
    GROUP_SHOW = "group-show"
    GROUP_STATE_CHANGE = "group-state-change"
    GROUP_RETRY = "group-retry"
    GROUP_CANCEL = "group-cancel"
    GROUP_WORKER_CHANGE = "group-worker-change"
    GROUP_MACHINES = "group-machines"
    MACHINES = "machines"
    AGENT_OPERATION = "agent-operation"
    AGENT_STATUS = "agent-status"
    AGENT_CONFIG = "agent-config"
    AGENT_READINESS = "agent-readiness"
    MACHINE_INIT = "machine-init"
    PROJECT_OPERATION = "project-operation"
    PROJECT_REGISTER = "project-register"
    PROJECT_LIST = "project-list"
    CPU_LANE = "cpu-lane"
    GPU_POLICY = "gpu-policy"
    UPGRADE_REGISTRY_STATUS = "upgrade-registry-status"
    UPGRADE_ADVANCE = "upgrade-advance"
    UPGRADE_PROJECT = "upgrade-project"
    UPGRADE_REPAIR = "upgrade-repair"
    SCHEMA6_UPGRADE = "schema6-upgrade"
    SUBMISSION = "submission"
    CONTEXT = "context"
    DOCTOR_VERIFY = "doctor-verify"
    DOCTOR_REPAIR = "doctor-repair"
    MACHINE_IDENTITY_DIAGNOSIS = "machine-identity-diagnosis"
    CLEAN = "clean"
    CONFIG = "config"
    TASK_WAIT = "task-wait"
    STATUS = "status"
    MACHINE_SHOW = "machine-show"
    OPERATION = "operation"


T = TypeVar("T")
Validator = Callable[[Any], None]
Renderer = Callable[[Any, Mapping[str, object]], str]


@dataclass(frozen=True, slots=True)
class CliOutput(Generic[T]):
    """Canonical CLI payload plus optional non-JSON presentation context."""

    kind: OutputKind
    payload: T
    presentation: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputContract:
    """Validator and human projection for one canonical output family."""

    validator: Validator
    renderer: Renderer
    payload_type: type[Any] | None = None


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_registry(*contract_maps: Mapping[OutputKind, OutputContract]) -> dict[OutputKind, OutputContract]:
    """Combine family contracts and enforce the closed OutputKind registry."""
    registry: dict[OutputKind, OutputContract] = {}
    for contract_map in contract_maps:
        for kind, contract in contract_map.items():
            if kind in registry:
                raise RuntimeError(f"CLI output renderer registered more than once for {kind.value!r}")
            registry[kind] = contract
    expected = set(OutputKind)
    actual = set(registry)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        missing_text = ", ".join(sorted(kind.value for kind in missing)) or "none"
        extra_text = ", ".join(sorted(getattr(kind, "value", str(kind)) for kind in extra)) or "none"
        raise RuntimeError(f"CLI output renderer registry is incomplete (missing: {missing_text}; extra: {extra_text})")
    return registry


from . import admin, agent, config, group, machine, setup, submission, task

_REGISTRY: dict[OutputKind, OutputContract] = _build_registry(
    task.CONTRACTS,
    group.CONTRACTS,
    machine.CONTRACTS,
    setup.CONTRACTS,
    agent.CONTRACTS,
    config.CONTRACTS,
    admin.CONTRACTS,
    submission.CONTRACTS,
)


def render(output: CliOutput[Any], output_format: str) -> str:
    """Render one validated finite output without changing its canonical payload."""
    if not isinstance(output, CliOutput):
        raise TypeError(f"render expects CliOutput, got {type(output).__name__}")
    contract = _contract_for(output.kind)
    contract.validator(output.payload)
    if output_format == "json":
        return json.dumps(output.payload, default=_json_default)
    if output_format != "human":
        raise ValueError(f"unsupported output format {output_format!r}")
    if not isinstance(output.presentation, Mapping):
        raise TypeError("CLI output presentation must be a mapping")
    return contract.renderer(output.payload, output.presentation)


def _contract_for(kind: Any) -> OutputContract:
    if not isinstance(kind, OutputKind):
        raise TypeError(f"unknown CLI output kind {kind!r}")
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"no renderer registered for CLI output kind {kind.value!r}") from exc


__all__ = ["CliOutput", "OutputContract", "OutputKind", "render"]
