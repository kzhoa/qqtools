"""Closed JSON contract for isolated Group-service census requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runtime.group_discovery.advance import (
    control_continuation,
    discovery_continuation,
    initial_control_continuation,
    initial_discovery_continuation,
)
from ..runtime.group_discovery.probe import validate_group_service_probe_state
from ..runtime.records import validate_group_name


def group_service_probe_parameters(value: object) -> dict[str, Any]:
    """Validate one Group-service census parameter object."""
    if not isinstance(value, Mapping) or set(value) != {"machine_name", "probe_state"}:
        raise ValueError("Group-service probe parameters have missing or unknown fields.")
    machine_name = value["machine_name"]
    if not isinstance(machine_name, str) or not machine_name:
        raise ValueError("Group-service probe machine_name is invalid.")
    return {"machine_name": machine_name, "probe_state": validate_group_service_probe_state(value["probe_state"])}


def group_service_probe_evidence(value: object) -> dict[str, Any]:
    """Validate one closed Group-service census result."""
    if not isinstance(value, Mapping) or set(value) != {"state", "probe_state", "candidate"}:
        raise ValueError("Group-service probe evidence has missing or unknown fields.")
    state = value["state"]
    if state not in {"quiescent", "active", "pending"}:
        raise ValueError("Group-service probe state is invalid.")
    candidate = value["candidate"]
    if candidate is not None:
        if not isinstance(candidate, Mapping) or set(candidate) != {"group", "lane", "generation"}:
            raise ValueError("Group-service probe candidate is invalid.")
        group = candidate["group"]
        validate_group_name(group)
        lane = candidate["lane"]
        if lane not in {"legacy", "control", "membership", "maintenance"}:
            raise ValueError("Group-service probe candidate lane is invalid.")
        generation = candidate["generation"]
        if lane == "legacy":
            if generation is not None:
                raise ValueError("legacy Group-service candidate cannot carry a generation.")
        elif type(generation) is not int or generation < 1:
            raise ValueError("locator Group-service candidate requires a positive generation.")
        candidate = {"group": group, "lane": lane, "generation": generation}
    if (state == "active") != (candidate is not None):
        raise ValueError("Group-service active evidence must name exactly one candidate.")
    return {
        "state": state,
        "probe_state": validate_group_service_probe_state(value["probe_state"]),
        "candidate": candidate,
    }


def group_service_candidate(value: object) -> dict[str, Any]:
    """Validate one exact candidate captured by the census."""

    evidence = group_service_probe_evidence(
        {
            "state": "active",
            "probe_state": validate_group_service_probe_state(
                {"mode": "start", "revision": None, "offset": 0, "lane_index": 0, "shard": 0}
            ),
            "candidate": value,
        }
    )
    candidate = evidence["candidate"]
    if candidate is None:
        raise ValueError("Group-service candidate is required.")
    return candidate


def _continuation_for(candidate: Mapping[str, Any], value: object) -> dict[str, Any] | None:
    lane = candidate["lane"]
    if lane in {"legacy", "membership"}:
        return discovery_continuation(value)
    if lane == "control":
        return control_continuation(value)
    if value is not None:
        raise ValueError("maintenance Group-service work cannot carry a continuation.")
    return None


def initial_group_service_continuation(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
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
    "group_service_advance_evidence",
    "group_service_advance_parameters",
    "group_service_candidate",
    "group_service_probe_evidence",
    "group_service_probe_parameters",
    "initial_group_service_continuation",
]
