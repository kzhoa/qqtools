"""Incarnation-bound Worker removal after shared recovery admission fencing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import LOCAL_RECOVERY_CAPABILITY
from ..lifecycle import TerminalCommitResult
from ..runtime.group_namespace import group_authority_identity, is_group_authority_isolated
from ..runtime.locks import group_writer_lock
from ..runtime.operation_store import archive_operation, locate_operation_path, write_active_operation
from ..runtime.paths import group_path, shared_paths
from ..runtime.ready import (
    primary_projection_routes_for_group,
    primary_route_update_transaction,
    sync_primary_ready_group,
)
from ..runtime.records import SCHEMA_VERSION, new_id, normalize_group_record, utc_now, validate_identifier
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, read_json

WORKER_REMOVE_TYPE = "worker_remove_v2"
_PENDING_STATES = {"preparing", "converging", "waiting_ack", "blocked"}


def can_bind_worker_removal(cfg: RootConfig) -> bool:
    """Check the released-writer fence while the caller holds the schema lock."""
    path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    return LOCAL_RECOVERY_CAPABILITY in read_json(path)["schema"][
        "required_capabilities"
    ] and is_group_authority_isolated(cfg.shared_root)


def _persist_operation(cfg: RootConfig, operation: dict[str, Any]) -> None:
    control = operation["group_control"]
    control["updated_at"] = utc_now()
    operation["meta"]["revision"] += 1
    operation["meta"]["updated_at"] = control["updated_at"]
    if control["state"] in {"completed", "superseded"}:
        archive_operation(cfg, "group_control", control["operation_id"], operation)
    else:
        write_active_operation(cfg, "group_control", control["operation_id"], operation)


def _publish_group(cfg: RootConfig, data: dict[str, Any], previous_workers: dict[str, Any]) -> None:
    name = data["group"]["name"]
    data["meta"]["revision"] += 1
    data["meta"]["updated_at"] = utc_now()
    if previous_workers == data["group"]["worker_set"]:
        atomic_replace(group_path(cfg.shared_root, name), data)
        return
    routes = primary_projection_routes_for_group(cfg, name)
    with primary_route_update_transaction(cfg, routes):
        atomic_replace(group_path(cfg.shared_root, name), data)
        sync_primary_ready_group(cfg, name, previous_workers=previous_workers)


def begin_worker_removal_locked(
    cfg: RootConfig, data: dict[str, Any], machine: str, *, terminate_running: bool
) -> dict[str, Any]:
    """Prepare durable removal intent under the caller's schema and Group locks."""
    if not can_bind_worker_removal(cfg):
        raise RuntimeError("bound Worker removal requires local-recovery-v1 admission")
    # Complete an uncertain capability rename before publishing the new format.
    DurableIO().sync_directory(shared_paths(cfg.shared_root)["schema"], "worker_remove_admission")
    worker = data["group"]["worker_set"][machine]
    operation_id = worker.get("removal_operation_id")
    if operation_id:
        validate_identifier(operation_id, "removal_operation_id")
        operation = read_json(locate_operation_path(cfg, "group_control", operation_id))
        control = operation.get("group_control", {})
        if (
            control.get("operation_type") != WORKER_REMOVE_TYPE
            or control.get("operation_id") != operation_id
            or control.get("group_name") != data["group"]["name"]
            or control.get("machine_name") != machine
        ):
            raise RuntimeError("Worker removal binding is invalid")
        if control["state"] in {"completed", "superseded"}:
            # A later retry may have created new home-queue blockers. An explicit
            # removal after completion must establish a new intent and census.
            operation_id = None
        elif terminate_running and not control["terminate_running"] and control["state"] in _PENDING_STATES:
            control["terminate_running"] = True
            _persist_operation(cfg, operation)
    if not operation_id:
        operation_id = new_id()
        now = utc_now()
        operation = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
                "updated_by": {
                    "actor_type": "cli",
                    "machine_name": cfg.machine_name,
                    "process_id": str(os.getpid()),
                },
            },
            "group_control": {
                "operation_id": operation_id,
                "operation_type": WORKER_REMOVE_TYPE,
                "group_name": data["group"]["name"],
                "machine_name": machine,
                "state": "preparing",
                "worker_before": dict(worker),
                "authority": group_authority_identity(cfg.shared_root),
                "draining_state_epoch": max(data["group"]["worker_set_epoch"], worker["state_epoch"]) + 1,
                "terminate_running": terminate_running,
                "blockers": [],
                "blocked_reason": None,
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
            },
        }
        from .worker_removal_discovery import initialize_removal_discovery

        initialize_removal_discovery(cfg, operation["group_control"], data)
        write_active_operation(cfg, "group_control", operation_id, operation)
    _reconcile_locked(cfg, data, operation, allow_settlement=False)
    return {**data, "worker_control": _control_snapshot(operation["group_control"])}


def reconcile_worker_removal(
    cfg: RootConfig,
    operation_path: Path,
    group_name: str | None,
    *,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any] | None:
    """Replay only the Worker incarnation bound to a versioned operation."""
    try:
        control = read_json(operation_path).get("group_control", {})
    except FileNotFoundError:
        return None
    name = control.get("group_name")
    if not name or (group_name is not None and name != group_name):
        return None
    validate_identifier(name, "group_name")
    post_commit_results: list[TerminalCommitResult] = []
    result: dict[str, Any] | None = None
    with group_writer_lock(cfg, name):
        if not operation_path.exists():
            return None
        operation = read_json(operation_path)
        control = operation.get("group_control", {})
        if control.get("operation_id") != operation_path.stem:
            raise RuntimeError("Worker removal operation identity does not match its path")
        if control.get("operation_type") != WORKER_REMOVE_TYPE or control.get("group_name") != name:
            return None
        if control.get("state") in {"completed", "superseded"}:
            # The archive may have committed before active discovery retirement.
            _persist_operation(cfg, operation)
            return control
        if control.get("state") not in _PENDING_STATES:
            return None
        if not can_bind_worker_removal(cfg):
            raise RuntimeError("bound Worker removal lost local-recovery-v1 admission")
        path = group_path(cfg.shared_root, name)
        if not path.exists():
            control.update(state="blocked", blocked_reason="group_missing", completed_at=None)
            _persist_operation(cfg, operation)
            return control
        data = read_json(path)
        normalize_group_record(data)
        post_commit_results = _reconcile_locked(cfg, data, operation)
        result = control
    if post_commit_results:
        from .group import _dispatch_group_terminal_results

        _dispatch_group_terminal_results(cfg, post_commit_results, reservation_runtime_root)
    return result


def _has_valid_proof(control: dict[str, Any]) -> bool:
    before = control.get("worker_before")
    epoch = control.get("draining_state_epoch")
    try:
        validate_identifier(control.get("operation_id"), "operation_id")
        validate_identifier(control.get("machine_name"), "machine_name")
    except (TypeError, ValueError):
        return False
    return (
        isinstance(before, dict)
        and before.get("state") in {"active", "draining", "removing"}
        and type(before.get("state_epoch")) is int
        and before["state_epoch"] >= 0
        and type(epoch) is int
        and epoch > before["state_epoch"]
        and type(control.get("terminate_running")) is bool
    )


def _reconcile_locked(
    cfg: RootConfig,
    data: dict[str, Any],
    operation: dict[str, Any],
    *,
    allow_settlement: bool = True,
) -> list[TerminalCommitResult]:
    control = operation["group_control"]
    if control.get("state") in {"completed", "superseded"}:
        return []
    if not _has_valid_proof(control) or control.get("authority") != group_authority_identity(cfg.shared_root):
        control.update(state="blocked", blocked_reason="worker_removal_proof_invalid", completed_at=None)
        _persist_operation(cfg, operation)
        return []
    machine, operation_id = control["machine_name"], control["operation_id"]
    workers = data["group"]["worker_set"]
    worker = workers.get(machine)
    has_binding = (
        worker is not None
        and worker.get("removal_operation_id") == operation_id
        and worker.get("state") in {"draining", "removing"}
    )
    if not has_binding:
        if control["state"] != "preparing" or worker != control["worker_before"]:
            control.update(state="superseded", blocked_reason="worker_incarnation_changed", completed_at=utc_now())
            if (data.get("worker_control") or {}).get("operation_id") == operation_id:
                data["worker_control"] = _control_snapshot(control)
                _publish_group(cfg, data, {name: dict(value) for name, value in workers.items()})
            _persist_operation(cfg, operation)
            return []
        previous = {name: dict(value) for name, value in workers.items()}
        worker.update(
            state="draining",
            state_epoch=control["draining_state_epoch"],
            drain_requested_at=control["created_at"],
            removal_operation_id=operation_id,
        )
        data["group"]["worker_set_epoch"] = max(data["group"]["worker_set_epoch"] + 1, control["draining_state_epoch"])
        data["worker_control"] = _control_snapshot(control)
        _publish_group(cfg, data, previous)
    if has_binding and worker["state"] == "removing":
        # This bound incarnation already crossed the removal linearization point.
        # An archive failure must not resubscribe it to later independent work.
        control.update(
            state="completed",
            blocked_reason=None,
            completed_at=control.get("completed_at") or worker.get("remove_requested_at") or utc_now(),
        )
        if (data.get("worker_control") or {}).get("operation_id") == operation_id:
            data["worker_control"] = _control_snapshot(control)
        previous = {name: dict(value) for name, value in workers.items()}
        previous[machine] = dict(control["worker_before"])
        _publish_group(cfg, data, previous)
        _persist_operation(cfg, operation)
        return []
    if has_binding and control["state"] == "preparing":
        # A crash may have published draining but not repaired ready projections.
        previous = {name: dict(value) for name, value in workers.items()}
        previous[machine] = dict(control["worker_before"])
        _publish_group(cfg, data, previous)
    if "discovery" not in control:
        from .worker_removal_discovery import initialize_removal_discovery

        initialize_removal_discovery(cfg, control, data)
        _persist_operation(cfg, operation)
    if control["state"] == "preparing":
        control.update(state="converging", blocked_reason=None)
        _persist_operation(cfg, operation)
    from .worker_removal_discovery import advance_removal_discovery_locked

    ready, post_commit_results = advance_removal_discovery_locked(
        cfg,
        control,
        data,
        allow_settlement=allow_settlement,
    )
    previous = {name: dict(value) for name, value in workers.items()}
    if ready:
        if worker["state"] != "removing":
            data["group"]["worker_set_epoch"] += 1
            worker.update(
                state="removing",
                state_epoch=data["group"]["worker_set_epoch"],
                remove_requested_at=utc_now(),
            )
        control.update(state="completed", blocked_reason=None, completed_at=control.get("completed_at") or utc_now())
    control["updated_at"] = utc_now()
    if (data.get("worker_control") or {}).get("operation_id") == operation_id:
        data["worker_control"] = _control_snapshot(control)
    # The authoritative removing state must be durable before retiring recovery.
    _publish_group(cfg, data, previous)
    _persist_operation(cfg, operation)
    return post_commit_results


def _control_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded public Group snapshot for a full operation control."""
    snapshot = dict(control)
    discovery = control.get("discovery")
    if isinstance(discovery, dict):
        snapshot["discovery"] = {
            key: discovery[key]
            for key in (
                "version",
                "generation",
                "initial_high_watermark",
                "member_cursor",
                "journal_cursor",
                "policy_epoch",
            )
            if key in discovery
        }
    return snapshot
