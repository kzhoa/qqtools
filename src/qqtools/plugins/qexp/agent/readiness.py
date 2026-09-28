"""Bounded machine-agent readiness snapshots."""

from __future__ import annotations

import time
from math import isfinite
from pathlib import Path
from typing import Any, Callable

from ..machine_config import is_legacy_agent_project
from ..runtime.store import read_json
from .config import load_agent_config
from .context import MachineRuntime
from .helpers import _active_machine_identity
from .inventory import ProjectInventoryEntry, exact_binding_for_entry, reconcile_live_bindings_locked

DEFAULT_START_TIMEOUT_SECONDS = 30.0


def capture_readiness_snapshot(runtime: MachineRuntime | str | Path | None) -> dict[str, Any]:
    """Freeze effective enabled bindings selected by one start request."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    try:
        with machine_runtime.agent_lifecycle_guard():
            with machine_runtime.inventory_guard():
                with machine_runtime.registry_guard():
                    reconciliation = reconcile_live_bindings_locked(machine_runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return {
            "inventory_revision": None,
            "registry_revision": None,
            "project_ids": [],
            "entries": [],
            "reconciliation_blockers": ["registry_enablement_unknown"],
            "inventory_converged": False,
        }
    bindings = reconciliation.bindings
    enabled_ids = {binding.project_id for binding in bindings if binding.enabled}
    for entry in reconciliation.entries:
        binding, conflict = exact_binding_for_entry(entry, bindings)
        if conflict and entry.enabled:
            enabled_ids.add(entry.project_id)
        elif binding is None and entry.enabled:
            enabled_ids.add(entry.project_id)
    return {
        "inventory_revision": reconciliation.inventory_revision,
        "registry_revision": reconciliation.registry_revision,
        "project_ids": sorted(enabled_ids),
        "entries": [entry.to_dict() for entry in reconciliation.entries],
        "reconciliation_blockers": list(reconciliation.blockers),
        "inventory_converged": reconciliation.converged,
    }


def _status(runtime: MachineRuntime) -> dict[str, Any]:
    try:
        value = read_json(runtime.paths["agent"] / "status.json")
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}
    state = value.get("machine_agent")
    return state if isinstance(state, dict) else {}


def _captured_entries(snapshot: dict[str, Any]) -> list[ProjectInventoryEntry]:
    values = snapshot.get("entries", [])
    if not isinstance(values, list):
        return []
    parsed: list[ProjectInventoryEntry] = []
    for value in values:
        try:
            parsed.append(ProjectInventoryEntry.from_dict(value))
        except (TypeError, ValueError):
            continue
    return parsed


def evaluate_readiness(runtime: MachineRuntime | str | Path | None, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate current process, policy acknowledgement, authority and reconciliation."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    try:
        config = load_agent_config(machine_runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {
            "machine_runtime_root": str(machine_runtime.root),
            "ready": False,
            "reason": "agent_config_unavailable",
            "error": str(exc),
            "projects": [],
            "inventory_revision": None,
            "registry_revision": None,
            "inventory_converged": False,
            "captured_inventory_revision": snapshot.get("inventory_revision"),
        }
    try:
        with machine_runtime.agent_lifecycle_guard():
            with machine_runtime.inventory_guard():
                with machine_runtime.registry_guard():
                    reconciliation = reconcile_live_bindings_locked(machine_runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return {
            "machine_runtime_root": str(machine_runtime.root),
            "ready": False,
            "reason": "registry_enablement_unknown",
            "reasons": ["registry_enablement_unknown"],
            "runtime_id": machine_runtime.instance_id,
            "inventory_revision": None,
            "registry_revision": None,
            "captured_inventory_revision": snapshot.get("inventory_revision"),
            "captured_registry_revision": snapshot.get("registry_revision"),
            "inventory_converged": False,
            "projects": [],
        }
    revision, entries = reconciliation.inventory_revision, list(reconciliation.entries)
    captured_revision = snapshot.get("inventory_revision")
    captured_registry_revision = snapshot.get("registry_revision")
    entry_map = {entry.project_id: entry for entry in entries}
    status = _status(machine_runtime)
    identity = _active_machine_identity(machine_runtime)
    projects: list[dict[str, Any]] = []
    reasons: list[str] = []
    if not snapshot.get("project_ids"):
        reasons.append("no_enabled_projects")
    if revision != captured_revision:
        reasons.append("concurrent_inventory_change")
    if reconciliation.registry_revision != captured_registry_revision:
        reasons.append("concurrent_registry_change")
    if not reconciliation.converged:
        reasons.extend(reconciliation.blockers or ("inventory_mirror_incomplete",))
    if snapshot.get("reconciliation_blockers"):
        reasons.append("captured_enablement_reconciliation_incomplete")
    if identity is None:
        reasons.append("agent_not_ready")
    if status.get("runtime_id") != machine_runtime.instance_id:
        reasons.append("stale_runtime_generation")
    if status.get("policy_revision_requested") != config.revision:
        reasons.append("policy_request_not_observed")
    if status.get("policy_revision_acknowledged") != config.revision:
        reasons.append("policy_revision_not_acknowledged")
    reconciled = status.get("reconciled_project_ids")
    if not isinstance(reconciled, list):
        reconciled = []
    reconciled_ids = set(item for item in reconciled if isinstance(item, str))
    bindings = reconciliation.bindings
    bindings_by_project = {binding.project_id: binding for binding in bindings}
    for project_id in snapshot.get("project_ids", []):
        entry = entry_map.get(project_id)
        binding = bindings_by_project.get(project_id)
        project: dict[str, Any] = {"project_id": project_id, "status": "unknown", "reason": None}
        if entry is None:
            project.update(status="conflicting", reason="inventory_entry_removed")
        elif binding is None:
            _binding, conflict = exact_binding_for_entry(entry, bindings)
            project.update(
                status="conflicting" if conflict else "inventory_only",
                reason="identity_path_conflict" if conflict else "not_registered",
            )
        else:
            exact_binding, conflict = exact_binding_for_entry(entry, bindings)
            if conflict or exact_binding != binding:
                project.update(status="conflicting", reason="identity_path_conflict")
                projects.append(project)
                reasons.append(f"{project_id}:{project['reason']}")
                continue
            if not binding.enabled:
                project.update(status="disabled", reason="registry_disabled_after_capture")
                projects.append(project)
                reasons.append(f"{project_id}:{project['reason']}")
                continue
            if entry.enabled != binding.enabled:
                project.update(status="registered", reason="inventory_mirror_incomplete")
                projects.append(project)
                reasons.append(f"{project_id}:{project['reason']}")
                continue
            try:
                if is_legacy_agent_project(binding.root_config()):
                    project.update(status="conflicting", reason="legacy_project_requires_migration")
                    authority = None
                else:
                    authority = machine_runtime.registration_status(binding)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                authority = {"state": "invalid", "write_eligible": False, "error": str(exc)}
            if authority is None:
                pass
            elif not authority.get("write_eligible"):
                project.update(status="conflicting", reason=authority.get("state", "authority_unavailable"))
            elif project_id not in reconciled_ids:
                project.update(status="registered", reason="initial_reconciliation_pending")
            else:
                project.update(status="registered", reason=None)
            project["machine_name"] = binding.machine_name
            project["registration_generation"] = binding.registration_generation
        projects.append(project)
        if project["status"] != "registered":
            reasons.append(f"{project_id}:{project['reason']}")
    if not status.get("ready"):
        reasons.append("agent_not_ready")
    # Preserve first-seen order while avoiding repeated generic diagnostics.
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "machine_runtime_root": str(machine_runtime.root),
        "ready": not unique_reasons,
        "reason": None if not unique_reasons else unique_reasons[0],
        "reasons": unique_reasons,
        "runtime_id": machine_runtime.instance_id,
        "agent_state": "active" if identity is not None else status.get("state", "stopped"),
        "pid": identity[0] if identity is not None else status.get("pid"),
        "configured_agent_mode": config.agent_mode,
        "requested_policy_revision": config.revision,
        "observed_policy_revision": status.get("policy_revision_acknowledged"),
        "inventory_revision": revision,
        "registry_revision": reconciliation.registry_revision,
        "captured_inventory_revision": captured_revision,
        "captured_registry_revision": captured_registry_revision,
        "inventory_converged": reconciliation.converged,
        "enablement_blockers": list(reconciliation.blockers),
        "projects": projects,
    }


def wait_for_readiness(
    runtime: MachineRuntime | str | Path | None,
    snapshot: dict[str, Any],
    *,
    timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait until readiness or a monotonic deadline without stopping anything."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("agent start timeout must be a positive number of seconds.")
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    deadline = clock() + float(timeout_seconds)
    latest = evaluate_readiness(machine_runtime, snapshot)
    while not latest["ready"] and clock() < deadline:
        sleep(min(0.05, max(0.0, deadline - clock())))
        latest = evaluate_readiness(machine_runtime, snapshot)
    if not latest["ready"]:
        latest["timed_out"] = True
    else:
        latest["timed_out"] = False
    latest["timeout_seconds"] = float(timeout_seconds)
    return latest


__all__ = [
    "DEFAULT_START_TIMEOUT_SECONDS",
    "capture_readiness_snapshot",
    "evaluate_readiness",
    "wait_for_readiness",
]
