"""Bounded progress observation for Group-service activation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..directory_capture import read_directory_entry
from ..group_discovery.activation import (
    _CURSOR_NAMES,
    _bootstrap_paths,
    _revision,
    advance_group_service_activation_locked,
    is_bootstrap_source,
)
from .contracts import UpgradeContext

_STATE_VERSION = 1
_MAX_COOKIE = (1 << 63) - 1
_REVISION_FIELDS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_BINDING_FIELDS = frozenset({"activation_epoch", "bootstrap_generation", "namespace", "directory_revision"})
_TARGET_FIELDS = frozenset({"binding", "cookie", "complete"})
_STATE_FIELDS = frozenset(
    {
        "version",
        "epoch",
        "binding",
        "activation_revision",
        "activation_cookie",
        "activation_complete",
        "inventory_cookie",
        "inventory_count",
        "inventory_eof",
        "target",
        "target_reached",
        "target_ordinal",
        "completed_units",
        "count_since_target",
        "inventory_next",
        "last_progress_at",
    }
)
_BOUND_STAGES = (*_CURSOR_NAMES, "stable_pass")
_ACTIVE_NAMESPACES = _CURSOR_NAMES[2:]
_STAGE_LABELS = {
    "groups": "Group bootstrap",
    "submissions": "Submission bootstrap",
    "submission_control": "Submission control bootstrap",
    "group_control": "Group control bootstrap",
    "cleanup": "Cleanup bootstrap",
    "discovery_debt": "Discovery debt bootstrap",
    "stable_pass": "Stable activation pass",
}


def _exact_int(value: object, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is invalid")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} is invalid")
    return value


def _exact_optional_string(value: object, label: str) -> str | None:
    if value is not None and type(value) is not str:
        raise ValueError(f"{label} is invalid")
    return value


def _revision_copy(value: object, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    if type(value) is not dict or frozenset(value) != _REVISION_FIELDS:
        raise ValueError(f"{label} has an invalid shape")
    return {key: _exact_int(value[key], f"{label}.{key}") for key in _REVISION_FIELDS}


def _binding_copy(value: object) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != _BINDING_FIELDS:
        raise ValueError("progress binding has an invalid shape")
    activation_epoch = value["activation_epoch"]
    generation = value["bootstrap_generation"]
    namespace = value["namespace"]
    if type(activation_epoch) is not str or not activation_epoch:
        raise ValueError("progress activation epoch is invalid")
    if type(generation) is not str or not generation:
        raise ValueError("progress bootstrap generation is invalid")
    if namespace not in _BOUND_STAGES:
        raise ValueError("progress namespace is invalid")
    return {
        "activation_epoch": activation_epoch,
        "bootstrap_generation": generation,
        "namespace": namespace,
        "directory_revision": _revision_copy(value["directory_revision"], "progress directory revision"),
    }


def _state_copy(value: object) -> dict[str, Any]:
    """Validate and copy compact internal progress state."""
    if type(value) is not dict or frozenset(value) != _STATE_FIELDS:
        raise ValueError("Group-service progress state has an invalid shape")
    if type(value["version"]) is not int or value["version"] != _STATE_VERSION:
        raise ValueError("unsupported Group-service progress state version")
    epoch = _exact_int(value["epoch"], "progress epoch", minimum=1)
    binding = _binding_copy(value["binding"])
    activation_revision = _exact_int(value["activation_revision"], "progress activation revision", minimum=1)
    activation_cookie = _exact_int(value["activation_cookie"], "progress activation cookie", maximum=_MAX_COOKIE)
    activation_complete = _exact_bool(value["activation_complete"], "progress activation completion")
    inventory_cookie = _exact_int(value["inventory_cookie"], "progress inventory cookie", maximum=_MAX_COOKIE)
    inventory_count = _exact_int(value["inventory_count"], "progress inventory count")
    inventory_eof = _exact_bool(value["inventory_eof"], "progress inventory eof")
    target = value["target"]
    if type(target) is not dict or frozenset(target) != _TARGET_FIELDS:
        raise ValueError("progress reconstruction target has an invalid shape")
    target_binding = _binding_copy(target["binding"])
    if target_binding != binding:
        raise ValueError("progress reconstruction target binding changed")
    target_cookie = _exact_int(target["cookie"], "progress target cookie", maximum=_MAX_COOKIE)
    target_complete = _exact_bool(target["complete"], "progress target completion")
    target_reached = _exact_bool(value["target_reached"], "progress target reached")
    target_ordinal = value["target_ordinal"]
    if target_ordinal is not None:
        target_ordinal = _exact_int(target_ordinal, "progress target ordinal")
    if target_reached != (target_ordinal is not None):
        raise ValueError("progress target reach and ordinal disagree")
    completed_units = value["completed_units"]
    if completed_units is not None:
        completed_units = _exact_int(completed_units, "progress completed units")
    count_since_target = _exact_int(value["count_since_target"], "progress count since target")
    inventory_next = _exact_bool(value["inventory_next"], "progress inventory turn")
    last_progress_at = _exact_optional_string(value["last_progress_at"], "progress timestamp")
    if target_cookie == 0 and not target_reached:
        raise ValueError("zero progress target must already be reached")
    if inventory_eof and inventory_cookie < 0:
        raise ValueError("progress inventory eof cookie is invalid")
    if completed_units is not None and not target_reached:
        raise ValueError("completed units require a reconstructed target")
    if target_ordinal is not None and target_ordinal > inventory_count:
        raise ValueError("progress target ordinal exceeds inventory count")
    if completed_units is not None and completed_units != target_ordinal + count_since_target:
        raise ValueError("progress completed units disagree with reconstruction")
    return {
        "version": _STATE_VERSION,
        "epoch": epoch,
        "binding": binding,
        "activation_revision": activation_revision,
        "activation_cookie": activation_cookie,
        "activation_complete": activation_complete,
        "inventory_cookie": inventory_cookie,
        "inventory_count": inventory_count,
        "inventory_eof": inventory_eof,
        "target": {
            "binding": target_binding,
            "cookie": target_cookie,
            "complete": target_complete,
        },
        "target_reached": target_reached,
        "target_ordinal": target_ordinal,
        "completed_units": completed_units,
        "count_since_target": count_since_target,
        "inventory_next": inventory_next,
        "last_progress_at": last_progress_at,
    }


def _cursor(record: dict[str, Any], namespace: str) -> dict[str, Any]:
    if namespace == "stable_pass":
        return {"directory_revision": None, "cookie": 0, "complete": record["bootstrap"]["phase"] == "complete"}
    cursor = record["bootstrap"]["cursors"][namespace]
    return {
        "directory_revision": deepcopy(cursor["directory_revision"]),
        "cookie": cursor["cookie"],
        "complete": cursor["complete"],
    }


def _stage_for(record: dict[str, Any]) -> str | None:
    bootstrap = record.get("bootstrap")
    if type(bootstrap) is not dict:
        return None
    phase = bootstrap.get("phase")
    if phase in {"groups", "submissions"}:
        return phase
    if phase == "active_namespaces":
        cursors = bootstrap.get("cursors")
        if type(cursors) is not dict:
            return None
        return next((name for name in _ACTIVE_NAMESPACES if not cursors[name]["complete"]), "stable_pass")
    if phase == "stable_pass":
        return "stable_pass"
    return None


def _remaining_stages(stage: str) -> list[str]:
    if stage == "groups":
        return ["submissions", *_ACTIVE_NAMESPACES, "stable_pass"]
    if stage == "submissions":
        return [*_ACTIVE_NAMESPACES, "stable_pass"]
    if stage in _ACTIVE_NAMESPACES:
        index = _ACTIVE_NAMESPACES.index(stage)
        return [*_ACTIVE_NAMESPACES[index + 1 :], "stable_pass"]
    return []


def _binding(record: dict[str, Any], namespace: str, directory_revision: dict[str, int] | None) -> dict[str, Any]:
    return {
        "activation_epoch": record["activation_epoch"],
        "bootstrap_generation": record["bootstrap"]["generation"],
        "namespace": namespace,
        "directory_revision": deepcopy(directory_revision),
    }


def _new_state(
    record: dict[str, Any],
    namespace: str,
    directory_revision: dict[str, int] | None,
    *,
    previous: dict[str, Any] | None = None,
    force_activation: bool = False,
    use_cursor: bool = True,
) -> dict[str, Any]:
    cursor = _cursor(record, namespace) if use_cursor else {"cookie": 0, "complete": False}
    binding = _binding(record, namespace, directory_revision)
    target_cookie = cursor["cookie"]
    target_complete = cursor["complete"]
    target_reached = target_cookie == 0
    target_ordinal = 0 if target_reached else None
    return {
        "version": _STATE_VERSION,
        "epoch": (previous["epoch"] + 1) if previous is not None else 1,
        "binding": binding,
        "activation_revision": record["revision"],
        "activation_cookie": cursor["cookie"],
        "activation_complete": cursor["complete"],
        "inventory_cookie": 0,
        "inventory_count": 0,
        "inventory_eof": directory_revision is None,
        "target": {"binding": deepcopy(binding), "cookie": target_cookie, "complete": target_complete},
        "target_reached": target_reached,
        "target_ordinal": target_ordinal,
        "completed_units": 0 if target_reached else None,
        "count_since_target": 0,
        "inventory_next": False if force_activation else (previous["inventory_next"] if previous else False),
        "last_progress_at": None,
    }


def _metadata_revision(context: UpgradeContext, path: Path) -> dict[str, int] | None:
    context.storage.account_metadata_ops(1)
    return _revision(path)


def _state_matches(
    state: dict[str, Any],
    record: dict[str, Any],
    namespace: str,
    directory_revision: dict[str, int] | None,
) -> bool:
    cursor = _cursor(record, namespace)
    binding = state["binding"]
    if binding != _binding(record, namespace, directory_revision):
        return False
    if state["activation_revision"] != record["revision"]:
        return False
    if state["activation_cookie"] != cursor["cookie"] or state["activation_complete"] != cursor["complete"]:
        return False
    if cursor["directory_revision"] is None and cursor["cookie"] == 0:
        return True
    return cursor["directory_revision"] == directory_revision


def _set_target_reached(state: dict[str, Any], ordinal: int) -> None:
    state["target_reached"] = True
    state["target_ordinal"] = ordinal
    state["completed_units"] = ordinal + state["count_since_target"]


def _reset_after_directory_change(
    record: dict[str, Any], namespace: str, directory_revision: dict[str, int] | None, previous: dict[str, Any]
) -> dict[str, Any]:
    state = _new_state(record, namespace, directory_revision, previous=previous, force_activation=True)
    state["inventory_eof"] = directory_revision is None
    return state


def _inventory_slice(
    context: UpgradeContext,
    record: dict[str, Any],
    state: dict[str, Any],
    namespace: str,
    path: Path,
    directory_revision: dict[str, int] | None,
) -> tuple[dict[str, Any], str]:
    if directory_revision is None:
        state["inventory_cookie"] = 0
        state["inventory_count"] = 0
        state["inventory_eof"] = True
        _set_target_reached(state, 0)
        state["inventory_next"] = False
        return state, "inventory"
    if state["inventory_eof"]:
        state["inventory_next"] = False
        return state, "recounting" if state["completed_units"] is None else "inventory"

    cookie = state["inventory_cookie"]
    target_cookie = state["target"]["cookie"]
    if not state["target_reached"] and cookie == target_cookie:
        _set_target_reached(state, state["inventory_count"])
    context.storage.account_metadata_ops(3)
    name, next_cookie = read_directory_entry(path, cookie)
    after_revision = _metadata_revision(context, path)
    if after_revision != directory_revision:
        return _reset_after_directory_change(record, namespace, after_revision, state), "restarted"
    if name is None:
        state["inventory_cookie"] = next_cookie
        state["inventory_eof"] = True
        if not state["target_reached"]:
            if next_cookie == target_cookie:
                _set_target_reached(state, state["inventory_count"])
            else:
                state["completed_units"] = None
        state["inventory_next"] = False
        return state, "recounting" if state["completed_units"] is None else "inventory"
    if is_bootstrap_source(name):
        state["inventory_count"] += 1
    state["inventory_cookie"] = next_cookie
    if not state["target_reached"] and next_cookie == target_cookie:
        _set_target_reached(state, state["inventory_count"])
    state["inventory_next"] = False
    return state, "recounting" if state["completed_units"] is None else "inventory"


def _checkpoint_identity(observation: dict[str, Any]) -> str:
    value = {
        "namespace": observation["source_checkpoint"]["namespace"],
        "operation_id": observation["source_checkpoint"]["operation_id"],
        "source_path": observation["source_checkpoint"]["source_path"],
        "source_revision": observation["source_checkpoint"]["source_revision"],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _binding_identity(binding: dict[str, Any]) -> str:
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _activation_source_observation(
    observation: dict[str, Any], namespace: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    current_item = observation.get("current_item")
    checkpoint = observation.get("source_checkpoint")
    if current_item is None:
        return None, None
    if (
        type(current_item) is not dict
        or current_item.get("kind") != "submission"
        or type(current_item.get("id")) is not str
        or type(current_item.get("completed_bytes")) is not int
        or current_item["completed_bytes"] < 0
        or type(current_item.get("total_bytes")) is not int
        or current_item["total_bytes"] < current_item["completed_bytes"]
        or namespace not in {"submissions", "submission_control"}
    ):
        raise ValueError("activation current-item observation is invalid")
    if type(checkpoint) is not dict or checkpoint.get("namespace") != namespace:
        raise ValueError("activation checkpoint observation is invalid")
    if checkpoint.get("operation_id") != current_item["id"] or type(checkpoint.get("source_path")) is not str:
        raise ValueError("activation checkpoint observation identity is invalid")
    source_revision = checkpoint.get("source_revision")
    _revision_copy(source_revision, "activation checkpoint source revision")
    if source_revision["size"] != current_item["total_bytes"]:
        raise ValueError("activation checkpoint size is inconsistent")
    item = {
        "kind": "submission",
        "id": current_item["id"],
        "completed_bytes": current_item["completed_bytes"],
        "total_bytes": current_item["total_bytes"],
    }
    if current_item.get("restarted") is True:
        item["restarted"] = True
    evidence = {
        "identity": _checkpoint_identity(observation),
        "completed_bytes": current_item["completed_bytes"],
        "previous_completed_bytes": observation.get("previous_completed_bytes"),
    }
    return item, evidence


def _activation_slice(
    context: UpgradeContext,
    record: dict[str, Any],
    state: dict[str, Any],
    namespace: str,
    directory_revision: dict[str, int] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, str]:
    before_revision = record["revision"]
    before_cursor = _cursor(record, namespace)
    observation: dict[str, Any] = {}
    next_record = advance_group_service_activation_locked(
        context.cfg,
        storage=context.storage,
        layout_prevalidated=True,
        observation=observation,
    )
    try:
        after_stage = _stage_for(next_record)
        after_path = _bootstrap_paths(context.cfg, storage=context.storage).get(namespace)
        after_directory_revision = (
            _metadata_revision(context, after_path)
            if after_path is not None and after_stage == namespace
            else directory_revision
        )
        if after_stage == namespace and after_directory_revision != directory_revision:
            restarted = _reset_after_directory_change(next_record, namespace, after_directory_revision, state)
            state.clear()
            state.update(restarted)
            state["inventory_next"] = False
            return next_record, None, None, "restarted"

        expected_revision = before_revision + 1
        if next_record["state"] == "building" and next_record["revision"] != expected_revision:
            restarted = _new_state(
                next_record,
                after_stage or namespace,
                after_directory_revision,
                previous=state,
                force_activation=True,
            )
            state.clear()
            state.update(restarted)
            return next_record, None, None, "restarted"

        after_cursor = _cursor(next_record, namespace)
        initial_cursor = before_cursor["directory_revision"] is None and before_cursor["cookie"] == 0
        restarted_cursor = observation.get("cursor_restarted") is True and not initial_cursor
        if restarted_cursor:
            # The activation slice started this revision at zero, even if its
            # pre-slice cursor still pointed into the previous directory.
            restarted = _new_state(next_record, namespace, after_directory_revision, previous=state, use_cursor=False)
            state.clear()
            state.update(restarted)

        if after_stage != namespace:
            # The completed namespace remains publishable for this invocation.  The
            # next call creates a new stage binding after the phase transition.
            state["inventory_next"] = True
            state["activation_revision"] = next_record["revision"]
            state["activation_cookie"] = after_cursor["cookie"]
            state["activation_complete"] = after_cursor["complete"]
            state["last_progress_at"] = next_record["updated_at"]
            return next_record, None, None, "complete"

        if observation.get("source_completed") is True:
            state["count_since_target"] += 1
            if state["target_reached"]:
                state["completed_units"] = (state["target_ordinal"] or 0) + state["count_since_target"]
        state["activation_revision"] = next_record["revision"]
        state["activation_cookie"] = after_cursor["cookie"]
        state["activation_complete"] = after_cursor["complete"]
        state["inventory_next"] = not state["inventory_eof"]
        stage_state = "complete" if after_cursor["complete"] else "processing"
        if restarted_cursor:
            stage_state = "restarted"
        current_item, checkpoint = _activation_source_observation(observation, namespace)
        meaningful = bool(observation.get("source_completed"))
        if current_item is not None:
            meaningful = meaningful or (
                current_item["completed_bytes"] > 0 and observation.get("current_item", {}).get("restarted") is not True
            )
        if meaningful:
            state["last_progress_at"] = next_record["updated_at"]
        return next_record, current_item, checkpoint, stage_state
    except (OSError, ValueError, TypeError, KeyError):
        # The authoritative activation has already committed.  Discard only the
        # observation assembled after that commit and let the next slice rebuild it.
        return next_record, None, None, "observation_error"


def _public_progress(
    state: dict[str, Any],
    stage: str,
    *,
    stage_state: str,
    current_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage == "stable_pass":
        completed_units = None
        inventoried_units = 0
        total_units = None
        total_kind = "unknown"
    else:
        completed_units = state["completed_units"]
        inventoried_units = state["inventory_count"]
        total_units = state["inventory_count"] if state["inventory_eof"] else None
        total_kind = "snapshot_exact" if state["inventory_eof"] else "dynamic"
    progress: dict[str, Any] = {
        "version": 1,
        "scope": "shared_project",
        "stage": stage,
        "stage_label": _STAGE_LABELS[stage],
        "stage_state": stage_state,
        "scan_epoch": state["epoch"],
        "completed_units": completed_units,
        "inventoried_units": inventoried_units,
        "total_units": total_units,
        "total_kind": total_kind,
        "unit": "sources",
        "remaining_stages": _remaining_stages(stage),
        "last_progress_at": state["last_progress_at"],
    }
    if current_item is not None:
        progress["current_item"] = current_item
    return progress


def _evidence(state: dict[str, Any], checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "version": 1,
        "identity": _binding_identity(state["binding"]),
        "activation_revision": state["activation_revision"],
        "activation_cursor": state["activation_cookie"],
        "source_checkpoint": checkpoint,
    }


def _load_progress_state(context: UpgradeContext) -> dict[str, Any] | None:
    item = context.journal["upgrade"]["migrations"]["group-service-v1"]
    try:
        return _state_copy(item.get("progress_state"))
    except (ValueError, TypeError, KeyError):
        return None


def _advance_stable_pass(context: UpgradeContext, record: dict[str, Any], prior: dict[str, Any] | None):
    state = (
        prior
        if prior is not None and _state_matches(prior, record, "stable_pass", None)
        else _new_state(record, "stable_pass", None, previous=prior)
    )
    next_record = advance_group_service_activation_locked(
        context.cfg, storage=context.storage, layout_prevalidated=True
    )
    stage = _stage_for(next_record)
    if stage in _CURSOR_NAMES:
        cursor = _cursor(next_record, stage)
        state = _new_state(next_record, stage, cursor["directory_revision"], previous=state)
        state["inventory_next"] = True
        stage_state = "restarted"
    else:
        stage = "stable_pass"
        state["activation_revision"] = next_record["revision"]
        state["activation_complete"] = next_record["bootstrap"]["phase"] == "complete"
        state["last_progress_at"] = next_record["updated_at"]
        stage_state = "complete" if state["activation_complete"] else "processing"
    return next_record, _public_progress(state, stage, stage_state=stage_state), _evidence(state, None), state


def advance_group_service_with_progress(
    context: UpgradeContext,
    record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Advance Group-service activation or one bounded observational inventory slice."""
    if record is None:
        from ..group_discovery.activation import read_group_service_activation_record

        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
    if record is None:
        raise RuntimeError("Group service activation record is unavailable")
    if record["state"] != "building":
        next_record = advance_group_service_activation_locked(
            context.cfg,
            storage=context.storage,
            layout_prevalidated=True,
        )
        return next_record, None, None, None
    stage = _stage_for(record)
    prior_state = _load_progress_state(context)
    if stage == "stable_pass":
        return _advance_stable_pass(context, record, prior_state)
    if stage is None:
        next_record = advance_group_service_activation_locked(
            context.cfg,
            storage=context.storage,
            layout_prevalidated=True,
        )
        return next_record, None, None, None

    try:
        paths = _bootstrap_paths(context.cfg, storage=context.storage)
        path = paths[stage]
        directory_revision = _metadata_revision(context, path)
    except (OSError, ValueError, TypeError, KeyError):
        # A failed observational probe cannot indefinitely withhold activation.
        next_record = advance_group_service_activation_locked(
            context.cfg, storage=context.storage, layout_prevalidated=True
        )
        return next_record, None, None, None
    if prior_state is None or not _state_matches(prior_state, record, stage, directory_revision):
        state = _new_state(
            record,
            stage,
            directory_revision,
            previous=prior_state,
            force_activation=prior_state is None,
        )
        if prior_state is not None and state["binding"] != prior_state["binding"]:
            state["inventory_next"] = prior_state["inventory_next"]
    else:
        state = prior_state

    if state["inventory_next"]:
        try:
            state, stage_state = _inventory_slice(context, record, state, stage, path, directory_revision)
        except (OSError, ValueError, TypeError, KeyError):
            state["inventory_next"] = False
            return record, None, None, state
        return record, _public_progress(state, stage, stage_state=stage_state), _evidence(state, None), state

    # Keep authoritative activation outside the observational error boundary.  A
    # projection failure must still enter the coordinator's normal safety path.
    next_record, current_item, checkpoint, stage_state = _activation_slice(
        context,
        record,
        state,
        stage,
        directory_revision,
    )
    if stage_state == "observation_error":
        return next_record, None, None, None
    progress = _public_progress(state, stage, stage_state=stage_state, current_item=current_item)
    return next_record, progress, _evidence(state, checkpoint), state


__all__ = ["advance_group_service_with_progress"]
