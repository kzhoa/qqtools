"""Closed JSON contract for isolated Group-service census requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..runtime.group_discovery.probe import validate_group_service_probe_state
from ..runtime.records import validate_group_name

_GROUP_SERVICE_OPERATION_KINDS = frozenset({"group_service_probe", "group_service_advance"})


@dataclass(frozen=True, slots=True)
class GroupServiceRequestSpec:
    """Closed, validated request parameters for one Group-service operation."""

    operation_kind: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.operation_kind) is not str or self.operation_kind not in _GROUP_SERVICE_OPERATION_KINDS:
            raise ValueError(f"unsupported Group-service request operation: {self.operation_kind!r}")
        validator = (
            group_service_probe_parameters
            if self.operation_kind == "group_service_probe"
            else group_service_advance_parameters
        )
        object.__setattr__(self, "parameters", validator(self.parameters))


def group_service_probe_parameters(value: object) -> dict[str, Any]:
    """Validate one Group-service census parameter object."""
    if not isinstance(value, Mapping) or set(value) != {"machine_name", "probe_state"}:
        raise ValueError("Group-service probe parameters have missing or unknown fields.")
    machine_name = value["machine_name"]
    if not isinstance(machine_name, str) or not machine_name:
        raise ValueError("Group-service probe machine_name is invalid.")
    return {"machine_name": machine_name, "probe_state": validate_group_service_probe_state(value["probe_state"])}


def group_service_probe_request(machine_name: str, probe_state: Mapping[str, Any]) -> GroupServiceRequestSpec:
    """Build one closed Group-service probe request specification."""
    return GroupServiceRequestSpec(
        operation_kind="group_service_probe",
        parameters=group_service_probe_parameters({"machine_name": machine_name, "probe_state": probe_state}),
    )


def group_service_probe_evidence(value: object) -> dict[str, Any]:
    """Validate one closed Group-service census result."""
    if not isinstance(value, Mapping) or set(value) != {"state", "probe_state", "candidate"}:
        raise ValueError("Group-service probe evidence has missing or unknown fields.")
    state = value["state"]
    if state not in {"quiescent", "active", "pending"}:
        raise ValueError("Group-service probe state is invalid.")
    candidate_value = value["candidate"]
    candidate = None if candidate_value is None else group_service_candidate(candidate_value)
    if (state == "active") != (candidate is not None):
        raise ValueError("Group-service active evidence must name exactly one candidate.")
    return {
        "state": state,
        "probe_state": validate_group_service_probe_state(value["probe_state"]),
        "candidate": candidate,
    }


def group_service_candidate(value: object) -> dict[str, Any]:
    """Validate one exact candidate captured by the census."""
    if not isinstance(value, Mapping) or set(value) != {"group", "lane", "generation"}:
        raise ValueError("Group-service candidate is invalid.")
    group = value["group"]
    validate_group_name(group)
    lane = value["lane"]
    if lane not in {"legacy", "control", "membership", "maintenance"}:
        raise ValueError("Group-service candidate lane is invalid.")
    generation = value["generation"]
    if lane == "legacy":
        if generation is not None:
            raise ValueError("legacy Group-service candidate cannot carry a generation.")
    elif type(generation) is not int or generation < 1:
        raise ValueError("locator Group-service candidate requires a positive generation.")
    return {"group": group, "lane": lane, "generation": generation}


def _continuation_for(candidate: Mapping[str, Any], value: object) -> dict[str, Any] | None:
    from ..runtime.group_discovery.advance import control_continuation, discovery_continuation

    lane = candidate["lane"]
    if lane in {"legacy", "membership"}:
        return discovery_continuation(value)
    if lane == "control":
        return control_continuation(value)
    if value is not None:
        raise ValueError("maintenance Group-service work cannot carry a continuation.")
    return None


def initial_group_service_continuation(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    from ..runtime.group_discovery.advance import initial_control_continuation, initial_discovery_continuation

    lane = candidate["lane"]
    if lane in {"legacy", "membership"}:
        return initial_discovery_continuation()
    if lane == "control":
        return initial_control_continuation()
    return None


def group_service_advance_parameters(value: object) -> dict[str, Any]:
    """Validate one exact bounded Group transaction request."""

    if not isinstance(value, Mapping) or set(value) != {
        "machine_name",
        "candidate",
        "continuation",
        "probe_state",
    }:
        raise ValueError("Group-service advance parameters have missing or unknown fields.")
    machine_name = value["machine_name"]
    if not isinstance(machine_name, str) or not machine_name:
        raise ValueError("Group-service advance machine_name is invalid.")
    candidate = group_service_candidate(value["candidate"])
    return {
        "machine_name": machine_name,
        "candidate": candidate,
        "continuation": _continuation_for(candidate, value["continuation"]),
        "probe_state": validate_group_service_probe_state(value["probe_state"]),
    }


def group_service_advance_request(
    machine_name: str,
    candidate: Mapping[str, Any],
    continuation: Mapping[str, Any] | None,
    probe_state: Mapping[str, Any],
) -> GroupServiceRequestSpec:
    """Build one closed Group-service advance request specification."""
    return GroupServiceRequestSpec(
        operation_kind="group_service_advance",
        parameters=group_service_advance_parameters(
            {
                "machine_name": machine_name,
                "candidate": candidate,
                "continuation": continuation,
                "probe_state": probe_state,
            }
        ),
    )


def group_service_advance_evidence(value: object, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one bounded Group transaction result."""

    if not isinstance(value, Mapping) or set(value) != {"state", "continuation", "probe"}:
        raise ValueError("Group-service advance evidence has missing or unknown fields.")
    if value["state"] not in {"progress", "quiescent", "blocked", "stale"}:
        raise ValueError("Group-service advance state is invalid.")
    continuation = None if value["state"] == "stale" else _continuation_for(candidate, value["continuation"])
    probe = value["probe"]
    if probe is not None:
        if candidate["lane"] != "legacy" or value["state"] not in {"quiescent", "stale"}:
            raise ValueError("Only settled legacy Group work can carry a follow-up census.")
        probe = group_service_probe_evidence(probe)
    return {"state": value["state"], "continuation": continuation, "probe": probe}


__all__ = [
    "GroupServiceRequestSpec",
    "group_service_advance_evidence",
    "group_service_advance_parameters",
    "group_service_advance_request",
    "group_service_candidate",
    "group_service_probe_evidence",
    "group_service_probe_parameters",
    "group_service_probe_request",
    "initial_group_service_continuation",
]
