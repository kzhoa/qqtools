"""One isolated, restart-safe Group-service transaction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ...commands.group import reconcile_group_cancel_operations
from ..group_namespace import GroupNotPublished, GroupPublicationUnavailable, read_group
from ..records import validate_group_name
from . import activation, locator
from .control import GroupControlCursor, advance_group_control_rechecks
from .maintenance import GroupMaintenance
from .probe import probe_group_service
from .service import GroupDiscoveryService

_REVISION_KEYS = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
# RecoverableSource publishes only phase-boundary checkpoints. One isolated
# worker must therefore be allowed to finish that bounded parser unit; stopping
# earlier can make every fresh process replay the same prefix forever. The
# wall-clock bound remains well below the executor's overdue threshold.
_DISCOVERY_ADVANCES_PER_SLICE = 20_000
_DISCOVERY_SLICE_SECONDS = 2.0


def _revision(value: object, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _REVISION_KEYS:
        raise ValueError(f"{label} is invalid")
    result = dict(value)
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise ValueError(f"{label} is invalid")
    return result


def discovery_continuation(value: object) -> dict[str, Any]:
    """Validate the complete restart state of one discovery owner."""

    if not isinstance(value, Mapping) or set(value) != {
        "bootstrap_offset",
        "bootstrap_revision",
        "debt_offset",
        "debt_revision",
    }:
        raise ValueError("Group discovery continuation has missing or unknown fields")
    bootstrap_offset = value["bootstrap_offset"]
    debt_offset = value["debt_offset"]
    if any(type(offset) is not int or not 0 <= offset <= (1 << 63) - 1 for offset in (bootstrap_offset, debt_offset)):
        raise ValueError("Group discovery continuation offset is invalid")
    return {
        "bootstrap_offset": bootstrap_offset,
        "bootstrap_revision": _revision(value["bootstrap_revision"], "bootstrap revision"),
        "debt_offset": debt_offset,
        "debt_revision": _revision(value["debt_revision"], "debt revision"),
    }


def initial_discovery_continuation() -> dict[str, Any]:
    return {
        "bootstrap_offset": 0,
        "bootstrap_revision": None,
        "debt_offset": 0,
        "debt_revision": None,
    }


def control_continuation(value: object) -> dict[str, Any]:
    """Validate a serializable Group control cursor."""

    if not isinstance(value, Mapping) or set(value) != {
        "locator_generation",
        "recheck_generation",
        "recheck_sequence",
        "operation_witness",
        "operation_cursor",
    }:
        raise ValueError("Group control continuation has missing or unknown fields")
    for name in ("locator_generation", "recheck_generation"):
        item = value[name]
        if item is not None and (type(item) is not int or item < 1):
            raise ValueError(f"Group control {name} is invalid")
    sequence = value["recheck_sequence"]
    if type(sequence) is not int or sequence < 1:
        raise ValueError("Group control recheck_sequence is invalid")
    witness = value["operation_witness"]
    if witness is not None:
        if not isinstance(witness, list | tuple) or len(witness) != 4:
            raise ValueError("Group control operation_witness is invalid")
        witness = tuple(witness)
        if any(type(item) is not int or item < 0 for item in witness):
            raise ValueError("Group control operation_witness is invalid")
    cursor = value["operation_cursor"]
    if not isinstance(cursor, Mapping):
        raise ValueError("Group control operation_cursor is invalid")
    return {
        "locator_generation": value["locator_generation"],
        "recheck_generation": value["recheck_generation"],
        "recheck_sequence": sequence,
        "operation_witness": list(witness) if witness is not None else None,
        "operation_cursor": dict(cursor),
    }


def initial_control_continuation() -> dict[str, Any]:
    return {
        "locator_generation": None,
        "recheck_generation": None,
        "recheck_sequence": 1,
        "operation_witness": None,
        "operation_cursor": {},
    }


def _control_cursor(value: object) -> GroupControlCursor:
    state = control_continuation(value)
    witness = state["operation_witness"]
    return GroupControlCursor(
        locator_generation=state["locator_generation"],
        recheck_generation=state["recheck_generation"],
        recheck_sequence=state["recheck_sequence"],
        operation_witness=tuple(witness) if witness is not None else None,
        operation_cursor=state["operation_cursor"],
    )


def _control_state(cursor: GroupControlCursor) -> dict[str, Any]:
    return control_continuation(
        {
            "locator_generation": cursor.locator_generation,
            "recheck_generation": cursor.recheck_generation,
            "recheck_sequence": cursor.recheck_sequence,
            "operation_witness": list(cursor.operation_witness) if cursor.operation_witness is not None else None,
            "operation_cursor": cursor.operation_cursor,
        }
    )


def _locator_is_current(root: Path, group: str, lane: str, generation: int) -> bool:
    current = locator.read_group_locator(root, group, lane)
    return current is not None and current["generation"] == generation


def _close_discovery_owner(service: GroupDiscoveryService) -> None:
    """Bound cooperative cleanup; process exit remains the final descriptor fence."""

    service.request_close()
    for _ in range(64):
        if service.is_closed:
            return
        service.advance()


def advance_group_service(
    cfg: Any,
    candidate: Mapping[str, Any],
    continuation: object,
    probe_state: object,
) -> dict[str, Any]:
    """Advance exactly one bounded Group candidate without local effects."""

    group = candidate["group"]
    lane = candidate["lane"]
    generation = candidate["generation"]
    validate_group_name(group)
    root = Path(cfg.shared_root)

    if lane == "legacy":
        try:
            read_group(root, group)
        except (FileNotFoundError, GroupNotPublished, GroupPublicationUnavailable):
            return {
                "state": "stale",
                "continuation": None,
                "probe": probe_group_service(root, probe_state),
            }
        state = discovery_continuation(continuation)
        service = GroupDiscoveryService(root, group, continuation=state)
        result: dict[str, object] = {"state": "waiting"}
        deadline = time.monotonic() + _DISCOVERY_SLICE_SECONDS
        for _ in range(_DISCOVERY_ADVANCES_PER_SLICE):
            result = service.advance()
            if result.get("state") in {"complete", "error", "ambiguous", "closed"} or time.monotonic() >= deadline:
                break
        outcome = (
            "quiescent"
            if result.get("state") == "complete"
            else "blocked"
            if result.get("state")
            in {
                "error",
                "ambiguous",
            }
            else "progress"
        )
        state = service.continuation
        _close_discovery_owner(service)
        return {
            "state": outcome,
            "continuation": state,
            "probe": probe_group_service(root, probe_state) if outcome == "quiescent" else None,
        }

    if type(generation) is not int or generation < 1 or not activation.is_group_service_active(root):
        return {"state": "stale", "continuation": None, "probe": None}
    if not _locator_is_current(root, group, lane, generation):
        return {"state": "stale", "continuation": None, "probe": None}

    if lane == "membership":
        state = discovery_continuation(continuation)
        service = GroupDiscoveryService(
            root,
            group,
            mode="locator",
            locator_generation=generation,
            continuation=state,
        )
        result = {"state": "waiting"}
        deadline = time.monotonic() + _DISCOVERY_SLICE_SECONDS
        for _ in range(_DISCOVERY_ADVANCES_PER_SLICE):
            result = service.advance()
            if result.get("state") in {"complete", "error", "ambiguous", "closed"} or time.monotonic() >= deadline:
                break
        acknowledged = result.get("state") == "complete" and service.acknowledge_if_quiescent()
        outcome = (
            "quiescent"
            if acknowledged
            else "blocked"
            if result.get("state")
            in {
                "error",
                "ambiguous",
            }
            else "progress"
        )
        state = service.continuation
        _close_discovery_owner(service)
        return {"state": outcome, "continuation": state, "probe": None}

    if lane == "control":
        cursor = _control_cursor(continuation)
        reconciled = reconcile_group_cancel_operations(
            cfg,
            group,
            include_legacy=False,
            reservation_runtime_root=cfg.runtime_root,
            limit=1,
        )
        state = "progress" if reconciled else advance_group_control_rechecks(cfg, group, generation, cursor)
        return {
            "state": "quiescent"
            if state == "quiescent"
            else "stale"
            if state == "stale"
            else "blocked"
            if state == "blocked"
            else "progress",
            "continuation": _control_state(cursor),
            "probe": None,
        }

    if lane == "maintenance":
        maintenance = GroupMaintenance(root, group)
        result = maintenance.advance(locator_generation=generation)
        acknowledged = maintenance.acknowledge_if_quiescent(generation)
        maintenance.request_close()
        if not maintenance.is_closed:
            maintenance.advance()
        outcome = "quiescent" if acknowledged else "blocked" if result.get("state") == "error" else "progress"
        return {"state": outcome, "continuation": None, "probe": None}

    raise ValueError(f"unknown Group service lane: {lane!r}")


__all__ = [
    "advance_group_service",
    "control_continuation",
    "discovery_continuation",
    "initial_control_continuation",
    "initial_discovery_continuation",
]
