"""Closed, bounded transport for one progress-v1 or progress-v2 context."""

from __future__ import annotations

from typing import Any, Mapping

from qqtools.qexp._progress_protocol import identifier

from ..runtime.progress import _IDENTITY, _context_interval
from ..runtime.progress_projection import validate_progress_snapshot
from ..runtime.progress_v2 import _validate_context


def progress_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the frozen local producer identity, never an authority grant."""
    if not isinstance(value, Mapping):
        raise ValueError("progress context must be an object")
    context = dict(value)
    version = context.get("protocol_version")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("progress context version is invalid")
    attempt_id = identifier(context.get("attempt_id"))
    if version == 2:
        return _validate_context(context, attempt_id)
    required = {"protocol_version", *_IDENTITY}
    policy = {"reporting_policy_version", "interval_seconds"}
    if set(context) not in (required, required | policy):
        raise ValueError("progress v1 context fields are invalid")
    for name in ("task_id", "attempt_id", "machine_name"):
        identifier(context.get(name))
    if context.get("launch_id") is not None:
        identifier(context["launch_id"])
    for name in ("attempt_number", "wrapper_pid", "wrapper_start_time_ticks"):
        minimum = 0 if name == "wrapper_start_time_ticks" else 1
        if type(context.get(name)) is not int or context[name] < minimum:
            raise ValueError(f"progress {name} is invalid")
    _context_interval(context)
    return context


def progress_projection(
    value: Mapping[str, Any] | None, context: Mapping[str, Any], generation: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("progress projection must be an object or null")
    return validate_progress_snapshot(
        dict(value), {**context, "registration_generation": generation}, context["protocol_version"]
    )


def progress_evidence(value: Mapping[str, Any], parameters: Mapping[str, Any], generation: str) -> dict[str, Any]:
    """Require exact identity and publication evidence for the captured request."""
    evidence = dict(value)
    if set(evidence) != {"state", "binding", "snapshot"}:
        raise ValueError("progress evidence fields are invalid")
    state = evidence["state"]
    if state not in {"observed", "published", "blocked", "retired", "stale"}:
        raise ValueError("progress evidence state is invalid")
    if state in {"blocked", "retired", "stale"}:
        if evidence["binding"] is not None or evidence["snapshot"] is not None:
            raise ValueError("unavailable progress evidence cannot grant an identity or snapshot")
        return evidence
    context = parameters["context"]
    binding = evidence["binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        *_IDENTITY,
        "registration_generation",
        "fencing_token",
        "terminal",
    }:
        raise ValueError("progress binding fields are invalid")
    binding = dict(binding)
    if any(binding[key] != context[key] for key in _IDENTITY) or binding["registration_generation"] != generation:
        raise ValueError("progress binding differs from captured producer identity")
    if (
        type(binding["fencing_token"]) is not int
        or binding["fencing_token"] < 1
        or type(binding["terminal"]) is not bool
    ):
        raise ValueError("progress binding token or terminal state is invalid")
    if (state == "published") != (parameters["projection"] is not None):
        raise ValueError("progress result does not match the requested transaction")
    snapshot = progress_projection(evidence["snapshot"], context, generation)
    if state == "published":
        if (
            snapshot != parameters["projection"]
            or snapshot is None
            or snapshot["fencing_token"] != binding["fencing_token"]
        ):
            raise ValueError("progress publication differs from the exact requested projection")
    evidence["binding"] = binding
    evidence["snapshot"] = snapshot
    return evidence
