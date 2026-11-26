"""Bounded read-only Project and machine status projections."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent.bindings import ProjectBinding, decode_registry
from ..agent.config import AgentConfig
from ..agent.context import MachineRuntime, _effective_runtime_id
from ..agent.control_plane_health import HEALTH_FILENAME, HEALTH_SCHEMA_VERSION, MAX_HEALTH_BYTES
from ..agent.diagnostic_recovery import diagnostics_repair_command, read_scheduler_recovery
from ..agent.identity import decode_identity_record
from ..agent.inventory import ProjectInventoryEntry, decode_inventory
from ..agent.scheduler_diagnostics import SchedulerDiagnosticStore, unavailable_summary
from ..agent.snapshot_publication_health import SNAPSHOT_COVERAGE_FILENAME
from ..agent.takeover_intent import (
    _CANCELLABLE_PHASES,
    _INTENT_NAME,
    _LEGACY_KEY,
    TakeoverIntent,
    decode_takeover_issue,
    takeover_issue_path,
)
from ..config_types import RootConfig
from ..layout import machine_path, project_id
from ..runtime.authority_scan import validate_evidence_path
from ..runtime.observation.projection import _validate_state, observation_path
from ..runtime.paths import machine_registration_path, shared_paths
from ..runtime.records import validate_identifier
from ..runtime.registration_authority import registration_state
from ..runtime.registration_protocol import MAX_REGISTRATION_RECORD_BYTES, validate_registration
from ..upgrade_diagnostics import summarize_upgrade_journal

STATUS_SCHEMA_VERSION = 1
MAX_READ_ATTEMPTS = 32
MAX_RECORD_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MACHINE_DETAIL_MAX_READ_ATTEMPTS = 8


class _BoundedReadFailure(ValueError):
    """An exact-path read failed without allowing an unbounded fallback."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(reason)
        self.reason = reason
        self.path = path


class _ReadBudget:
    """Track exact-file reads and enforce the status payload limits."""

    def __init__(
        self,
        *,
        max_read_attempts: int = MAX_READ_ATTEMPTS,
        max_record_bytes: int = MAX_RECORD_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
        self.max_read_attempts = max_read_attempts
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self.read_attempts = 0
        self.bytes_read = 0

    def read_json(self, path: Path, *, max_bytes: int | None = None) -> dict[str, Any] | None:
        """Read one regular JSON file, returning ``None`` only when absent."""
        record_limit = self.max_record_bytes if max_bytes is None else min(self.max_record_bytes, max_bytes)
        if self.read_attempts >= self.max_read_attempts:
            raise _BoundedReadFailure("read_budget_exhausted", path)
        self.read_attempts += 1
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _BoundedReadFailure(f"lstat_failed:{type(exc).__name__}", path) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise _BoundedReadFailure("not_a_regular_file", path)
        if metadata.st_size > record_limit:
            raise _BoundedReadFailure("record_oversized", path)
        if self.bytes_read + metadata.st_size > self.max_total_bytes:
            raise _BoundedReadFailure("total_read_budget_exhausted", path)
        try:
            with path.open("rb") as handle:
                remaining = self.max_total_bytes - self.bytes_read
                encoded = handle.read(min(record_limit + 1, remaining))
                self.bytes_read += len(encoded)
                if len(encoded) == remaining and os.fstat(handle.fileno()).st_size > remaining:
                    raise _BoundedReadFailure("total_read_budget_exhausted", path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _BoundedReadFailure(f"read_failed:{type(exc).__name__}", path) from exc
        if len(encoded) > record_limit:
            raise _BoundedReadFailure("record_oversized", path)
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BoundedReadFailure("malformed_json", path) from exc
        if not isinstance(value, dict):
            raise _BoundedReadFailure("json_object_required", path)
        return value

    def as_dict(self) -> dict[str, int]:
        return {
            "read_attempts": self.read_attempts,
            "max_read_attempts": self.max_read_attempts,
            "bytes_read": self.bytes_read,
            "max_record_bytes": self.max_record_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


def _warning(component: str, failure: _BoundedReadFailure) -> str:
    return f"{component} unavailable ({failure.reason})."


def _optional_read(
    reader: _ReadBudget,
    path: Path,
    component: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        value = reader.read_json(path)
    except _BoundedReadFailure as exc:
        warnings.append(_warning(component, exc))
        return None
    if value is None:
        return None
    return value


def _project_identity(cfg: RootConfig, reader: _ReadBudget) -> str:
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    try:
        identity = reader.read_json(identity_path)
    except _BoundedReadFailure as exc:
        raise ValueError(f"Project identity is unavailable: {exc.reason}.") from exc
    if identity is None:
        raise ValueError(f"Project identity is missing: {identity_path}.")
    project = identity.get("project")
    stable_id = project.get("project_id") if isinstance(project, dict) else None
    if (
        not isinstance(stable_id, str)
        or not stable_id
        or stable_id == "."
        or any(token in stable_id for token in ("/", "\\", "..", "\x00"))
        or project.get("shared_root") != str(cfg.shared_root)
    ):
        raise ValueError("Project identity is malformed or does not match the selected Project.")
    return stable_id


def _matching_participation(
    cfg: RootConfig,
    selected_project_id: str,
    registry: tuple[ProjectBinding, ...] | None,
    inventory: list[ProjectInventoryEntry] | None,
    *,
    registry_failed: bool,
    inventory_failed: bool,
) -> tuple[str, str | None, ProjectBinding | None]:
    matches = [
        binding
        for binding in registry or ()
        if binding.project_id == selected_project_id or binding.shared_root == cfg.shared_root
    ]
    if any(binding.project_id == selected_project_id and binding.shared_root == cfg.shared_root for binding in matches):
        return "registered", None, matches[0] if len(matches) == 1 else None
    inventoried = any(
        entry.project_id == selected_project_id and entry.shared_root == cfg.shared_root for entry in inventory or []
    )
    if inventoried and not registry_failed:
        return "inventory_only", "inventory_entry_without_binding", None
    if registry_failed or inventory_failed:
        return "unavailable", "local_registry_or_inventory_unavailable", None
    return "unregistered", "no_local_binding_or_inventory", None


def _verified_local_binding(
    cfg: RootConfig, runtime: MachineRuntime, binding: ProjectBinding, reader: _ReadBudget
) -> dict[str, object] | None:
    """Observe the selected binding against its exact shared registration."""
    runtime_root = runtime.paths["root"]
    machine_name = binding.machine_name
    if (
        binding.runtime_root not in {None, str(runtime_root)}
        or not binding.registration_generation
        or machine_name == "."
        or "\x00" in machine_name
    ):
        return None
    try:
        identity = reader.read_json(runtime.paths["identity"])
        record = decode_identity_record(identity) if identity is not None else None
        if record is None:
            return None
        runtime_id = _effective_runtime_id(record["instance_id"], record.get("runtime_id"))
        if binding.runtime_instance_id != runtime_id:
            return None

        registration_path = machine_registration_path(cfg.shared_root, machine_name)
        if not validate_evidence_path(registration_path, cfg.shared_root):
            return None
        registration_raw = reader.read_json(registration_path, max_bytes=MAX_REGISTRATION_RECORD_BYTES)
        if registration_raw is None:
            return None
        registration = registration_raw.get("registration")
        registration_cfg = RootConfig.from_canonical_paths(
            cfg.shared_root, cfg.project_root, machine_name, runtime_root
        )
        validate_registration(registration, binding.project_id, registration_cfg, allow_source=True)
        write_eligibility = registration_state(registration)
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None
    if (
        registration["generation"] != binding.registration_generation
        or registration["runtime_instance_id"] != runtime_id
        or registration["runtime_root"] != str(runtime_root)
        or write_eligibility not in {"eligible", "expired"}
    ):
        return None
    return {
        "machine_name": machine_name,
        "registration_generation": binding.registration_generation,
        "enabled": binding.enabled,
        "write_eligibility": write_eligibility,
    }


def _agent_observation(
    value: dict[str, Any] | None,
    *,
    failed: bool,
    missing: bool,
) -> dict[str, object]:
    if value is None:
        reason = "record_missing" if missing and not failed else "record_unreadable"
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": reason}
    record = value.get("machine_agent")
    if not isinstance(record, dict):
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": "record_malformed"}
    state = record.get("state")
    if not isinstance(state, str) or not state:
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": "state_unavailable"}
    observed_at = next(
        (record.get(key) for key in ("heartbeat_at", "observed_at", "timestamp") if isinstance(record.get(key), str)),
        None,
    )
    reason = record.get("stop_reason") if state == "stopped" else None
    return {"scope": "machine", "state": state, "observed_at": observed_at, "reason": reason}


def _configured_agent_mode(reader: _ReadBudget, runtime: MachineRuntime, warnings: list[str]) -> str | None:
    """Read the durable mode without invoking the compatibility/migration loader."""
    path = runtime.paths["global_config"]
    try:
        value = reader.read_json(path)
    except _BoundedReadFailure as exc:
        warnings.append(_warning("agent configuration", exc))
        return None
    if value is None:
        warnings.append("agent configuration unavailable (record_missing).")
        return None
    raw = value.get("agent_config", value.get("config"))
    if not isinstance(raw, dict):
        warnings.append("agent configuration unavailable (record_malformed).")
        return None
    try:
        config = AgentConfig.from_dict(raw)
    except (TypeError, ValueError):
        warnings.append("agent configuration unavailable (record_malformed).")
        return None
    return config.agent_mode


def _task_observation(
    value: dict[str, Any] | None, *, failed: bool, missing: bool, cfg: RootConfig
) -> dict[str, object]:
    if value is None:
        return {
            "state": "unavailable" if failed else "absent",
            "reason": "record_unreadable" if failed else ("projection_missing" if missing else "state_unavailable"),
        }
    try:
        validated = _validate_state(cfg, value)
    except (TypeError, ValueError) as exc:
        return {"state": "unavailable", "reason": f"invalid_state:{type(exc).__name__}"}
    reason = "projection_dirty" if validated.get("dirty") else None
    return {"state": validated["state"], "reason": reason}


def _project_upgrade(
    cfg: RootConfig, selected_project_id: str, reader: _ReadBudget, warnings: list[str]
) -> dict[str, object]:
    path = shared_paths(cfg.shared_root)["upgrade_journal"]
    reason = "journal_missing"
    try:
        value = reader.read_json(path)
    except _BoundedReadFailure as exc:
        reason = exc.reason
        value = None
    except (ValueError, RecursionError):
        reason = "malformed_json"
        value = None
    if value is None:
        if reason != "journal_missing":
            warnings.append(f"Upgrade journal unavailable ({reason}).")
        return {"state": "unknown", "migration": None, "phase": None, "admission_blocked": None, "reason": reason}

    upgrade = summarize_upgrade_journal(value, project_id=selected_project_id, shared_root=cfg.shared_root)
    if upgrade["state"] == "unknown":
        warnings.append(f"Upgrade journal unavailable ({upgrade['reason']}).")
    elif upgrade["state"] == "repair_required":
        warning = (
            f"Project {selected_project_id} needs upgrade repair during {upgrade['migration']} {upgrade['phase']}."
        )
        if upgrade["admission_blocked"] is True:
            warning += " This migration blocks new training."
        elif upgrade["admission_blocked"] is False:
            warning += " This migration does not block new training."
        else:
            warning += " Effect on new training is unknown."
        if "additional_repairs" in upgrade:
            warning += " Additional migrations need repair."
        warnings.append(warning)
    return upgrade


def _project_takeover(
    cfg: RootConfig,
    selected_project_id: str,
    runtime: MachineRuntime,
    reader: _ReadBudget,
    warnings: list[str],
) -> dict[str, object] | None:
    """Observe the active local slot only for the selected Project and runtime."""
    path = runtime.root / _INTENT_NAME
    component = "Project takeover"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        warnings.append(f"{component} unavailable (lstat_failed:{type(exc).__name__}).")
        return None
    if metadata.st_size > MAX_REGISTRATION_RECORD_BYTES:
        warnings.append(f"{component} unavailable (record_oversized).")
        return None

    warning_count = len(warnings)
    try:
        value = _optional_read(reader, path, component, warnings)
        if value is None:
            if len(warnings) == warning_count:
                warnings.append(f"{component} unavailable (record_missing).")
            return None
        if value.keys() == {_LEGACY_KEY} and isinstance(value[_LEGACY_KEY], dict):
            return None
        intent = TakeoverIntent.from_dict(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        warnings.append(f"{component} unavailable (record_malformed).")
        return None

    transition = intent.transition
    if transition.project_id != selected_project_id or transition.shared_root != cfg.shared_root:
        return None
    if transition.successor_runtime_root != runtime.root:
        warnings.append(f"{component} unavailable (runtime_root_mismatch).")
        return None

    try:
        identity = reader.read_json(runtime.paths["identity"])
        record = decode_identity_record(identity) if identity is not None else None
        runtime_id = _effective_runtime_id(record["instance_id"], record.get("runtime_id")) if record else None
    except _BoundedReadFailure as exc:
        warnings.append(_warning(f"{component} runtime identity", exc))
        return None
    except (OSError, RuntimeError, ValueError):
        runtime_id = None
    if runtime_id is None:
        warnings.append(f"{component} unavailable (runtime_identity_unverified).")
        return None
    if transition.successor_runtime_id != runtime_id:
        warnings.append(f"{component} unavailable (runtime_identity_mismatch).")
        return None

    target = transition.target.machine_name
    command = [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "project",
        "takeover",
        str(transition.shared_root),
        f"--machine={target}",
    ]
    can_cancel = intent.phase in _CANCELLABLE_PHASES or intent.phase == "cancelling"
    return {
        "plan_id": intent.plan_id,
        "phase": intent.phase,
        "departing_machine": transition.departing.machine_name if transition.departing is not None else None,
        "target_machine": target,
        "is_cancel_requested": intent.is_cancel_requested,
        "drain_observation": intent.drain_observation,
        "operation_issue": decode_takeover_issue(
            _optional_read(reader, takeover_issue_path(runtime.root), "Project takeover issue", warnings), intent
        ),
        "operation_status_command": shlex.join([*command, f"--operation-status={intent.plan_id}"]),
        "cancel_command": shlex.join([*command, f"--cancel={intent.plan_id}"]) if can_cancel else None,
    }


def _follow_up_actions(
    project: Path,
    project_id: str,
    runtime: MachineRuntime,
    participation: str,
    agent: dict[str, object],
    observation: dict[str, object],
    diagnostics: dict[str, Any],
    warnings: list[str],
    *,
    upgrade: dict[str, object] | None = None,
    diagnostic_recovery: dict[str, Any] | None = None,
) -> list[str]:
    prefix = ["qexp", "--machine-runtime-root", str(runtime.root)]
    actions: list[str] = []

    def suggest(*args: str) -> None:
        command = shlex.join([*prefix, *args])
        if command not in actions:
            actions.append(command)

    if participation == "registered" and upgrade is not None and upgrade.get("state") == "repair_required":
        suggest("admin", "upgrade", "status", "--project", project_id)
    elif participation in {"unregistered", "inventory_only"}:
        suggest("project", "register", str(project))
    elif participation == "unavailable":
        # Project list can initialize a missing inventory. Inspect the files directly.
        actions.append(shlex.join(["ls", "-ld", "--", str(runtime.paths["registry"]), str(runtime.paths["inventory"])]))

    repair_blocked = diagnostic_recovery is not None and diagnostic_recovery.get("state") == "blocked"
    if repair_blocked:
        actions.append(diagnostics_repair_command(runtime))
    if diagnostics.get("reason") != "project_id_unavailable":
        diagnostic_args = ("agent", "diagnostics", "active", "--project-id", project_id)
        has_reason = False
        for sample in diagnostics["active_findings"]:
            identity = sample.get("identity")
            reason = identity.get("reason_code") if isinstance(identity, dict) else None
            if isinstance(reason, str) and reason:
                has_reason = True
                suggest(*diagnostic_args, "--reason", reason)
        if not repair_blocked and (
            diagnostics["status"] != "available"
            or diagnostics["coverage"] != "complete"
            or diagnostics["truncated"]
            or (diagnostics["active_count"] and not has_reason)
        ):
            suggest(*diagnostic_args)
    if observation["state"] != "active" or observation["reason"] == "projection_dirty":
        suggest("--project", str(project), "task", "list")
    modes_conflict = (
        agent["configured_mode"] and agent["observed_mode"] and agent["configured_mode"] != agent["observed_mode"]
    )
    if agent["configured_mode"] is None:
        suggest("config", "show", "agent")
    if agent["state"] in {"unavailable", "stopped"} or modes_conflict or (warnings and not actions):
        suggest("agent", "status")
    return actions


def project_status(
    cfg: RootConfig,
    *,
    selection_source: str,
    machine_runtime: MachineRuntime,
) -> dict[str, object]:
    """Return a bounded Project overview without Task or machine enumeration."""
    reader = _ReadBudget()
    selected_project_id = _project_identity(cfg, reader)
    warnings: list[str] = []

    registry_path = machine_runtime.paths["registry"]
    inventory_path = machine_runtime.paths["inventory"]
    agent_path = machine_runtime.paths["agent"] / "status.json"
    observation_state_path = observation_path(cfg) / "state.json"

    warning_count = len(warnings)
    registry_raw = _optional_read(reader, registry_path, "local registry", warnings)
    registry_failed = len(warnings) != warning_count
    warning_count = len(warnings)
    inventory_raw = _optional_read(reader, inventory_path, "local inventory", warnings)
    inventory_failed = len(warnings) != warning_count
    agent_missing = False
    try:
        agent_raw = reader.read_json(agent_path)
        agent_missing = agent_raw is None
    except _BoundedReadFailure as exc:
        warnings.append(_warning("local agent status", exc))
        agent_raw = None
        agent_failed = True
    else:
        agent_failed = False
        if agent_raw is None:
            warnings.append("local agent status unavailable (record_missing).")

    observation_missing = False
    try:
        observation_raw = reader.read_json(observation_state_path)
        observation_missing = observation_raw is None
    except _BoundedReadFailure as exc:
        warnings.append(_warning("Task observation", exc))
        observation_raw = None
        observation_failed = True
    else:
        observation_failed = False

    configured_agent_mode = _configured_agent_mode(reader, machine_runtime, warnings)

    registry: tuple[ProjectBinding, ...] | None = None
    inventory: list[ProjectInventoryEntry] | None = None
    if registry_raw is not None:
        try:
            _, registry = decode_registry(registry_raw)
        except RuntimeError as exc:
            registry_failed = True
            warnings.append(f"local registry unavailable (malformed_record:{type(exc).__name__}).")
    if inventory_raw is not None:
        try:
            _, inventory = decode_inventory(inventory_raw)
        except RuntimeError as exc:
            inventory_failed = True
            warnings.append(f"local inventory unavailable (malformed_record:{type(exc).__name__}).")

    participation_state, participation_reason, binding = _matching_participation(
        cfg,
        selected_project_id,
        registry,
        inventory,
        registry_failed=registry_failed,
        inventory_failed=inventory_failed,
    )
    local_agent = _agent_observation(agent_raw, failed=agent_failed, missing=agent_missing)
    agent_record = agent_raw.get("machine_agent") if agent_raw is not None else None
    observed_agent_mode = agent_record.get("observed_agent_mode") if isinstance(agent_record, dict) else None
    local_agent["configured_mode"] = configured_agent_mode
    local_agent["observed_mode"] = observed_agent_mode if isinstance(observed_agent_mode, str) else None
    task_observation = _task_observation(
        observation_raw,
        failed=observation_failed,
        missing=observation_missing,
        cfg=cfg,
    )
    if task_observation["state"] == "unavailable" and not any("Task observation" in item for item in warnings):
        warnings.append("Task observation unavailable (record_malformed).")
    if local_agent["state"] == "unavailable" and not any("local agent status" in item for item in warnings):
        warnings.append("local agent status unavailable (record_malformed).")

    upgrade = _project_upgrade(cfg, selected_project_id, reader, warnings)
    try:
        scheduler_diagnostics = SchedulerDiagnosticStore(machine_runtime).summary_view(
            project_id=selected_project_id,
            limit=8,
            read_json=reader.read_json,
        )
    except ValueError:
        scheduler_diagnostics = unavailable_summary(machine_runtime.root, "project_id_unavailable")
    binding_observation = (
        _verified_local_binding(cfg, machine_runtime, binding, reader) if binding is not None else None
    )
    binding_reason = (
        "local_binding_identity_unavailable"
        if participation_state == "registered" and binding_observation is None
        else None
    )
    takeover = _project_takeover(cfg, selected_project_id, machine_runtime, reader, warnings)
    heartbeat, project_snapshot, readiness, recovery = _scheduling_observations(
        machine_runtime,
        reader,
        agent_record,
        binding if binding_observation is not None else None,
        scheduler_diagnostics,
    )
    local_agent["heartbeat"] = heartbeat
    selected_diagnostics = scheduler_diagnostics
    selected_coverage = readiness["diagnostic_coverage"]["state"]
    if scheduler_diagnostics["coverage"] == "complete" and selected_coverage == "incomplete":
        selected_diagnostics = {**scheduler_diagnostics, "coverage": "incomplete"}
    elif selected_coverage == "complete" and scheduler_diagnostics["reason"] == "active_overflow":
        selected_diagnostics = {
            **scheduler_diagnostics,
            "coverage": "complete",
            "machine_coverage": scheduler_diagnostics["coverage"],
            "reason": None,
            "truncated": scheduler_diagnostics["active_count"] > len(scheduler_diagnostics["active_findings"]),
        }

    result: dict[str, object] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "partial" if warnings else "complete",
        "project": {
            "path": str(cfg.project_root),
            "project_id": selected_project_id,
            "selection_source": selection_source,
        },
        "local_participation": {
            "state": participation_state,
            "reason": participation_reason,
            "binding": binding_observation,
            "binding_reason": binding_reason,
        },
        "local_agent": local_agent,
        "configured_agent_mode": configured_agent_mode,
        "task_observation": task_observation,
        "upgrade": upgrade,
        "scheduler_diagnostics": selected_diagnostics,
        "diagnostic_recovery": recovery,
        "project_snapshot": project_snapshot,
        "scheduling_readiness": readiness,
        "totals": {"tasks": None, "machines": None, "reason": "not_available"},
        "takeover_preview_help": shlex.join(
            [
                "qexp",
                "--machine-runtime-root",
                str(machine_runtime.root),
                "project",
                "takeover",
                str(cfg.project_root),
                "--help",
            ]
        ),
        "next_actions": _follow_up_actions(
            cfg.project_root,
            selected_project_id,
            machine_runtime,
            participation_state,
            local_agent,
            task_observation,
            scheduler_diagnostics,
            warnings,
            upgrade=upgrade,
            diagnostic_recovery=recovery,
        ),
        "budget": reader.as_dict(),
        "warnings": warnings,
    }
    if takeover is not None:
        result["takeover"] = takeover
    return result


def _scheduling_observations(
    runtime: MachineRuntime,
    reader: _ReadBudget,
    agent: object,
    binding: ProjectBinding | None,
    diagnostics: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object], dict[str, Any], dict[str, Any]]:
    """Project bounded, independent heartbeat, snapshot and coverage evidence."""
    now = datetime.now(timezone.utc)

    def read_optional(path: Path, *, max_bytes: int = 4096) -> dict[str, Any]:
        try:
            return reader.read_json(path, max_bytes=max_bytes) or {}
        except (ValueError, RecursionError):
            return {}

    def timestamp(raw: object) -> datetime | None:
        if (
            not isinstance(raw, str)
            or len(raw) > 64
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?"
                r"(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
                raw,
            )
            is None
        ):
            return None
        try:
            return datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw)
        except ValueError:
            return None

    def freshness(seen_at: object, interval: object, *, intervals: int = 1) -> dict[str, object]:
        observed = timestamp(seen_at)
        unknown = {"observed_at": None, "freshness": "unknown"}
        if observed is None or observed > now or type(interval) not in {int, float} or interval <= 0:
            return unknown
        try:
            if not math.isfinite(interval):
                return unknown
        except OverflowError:
            return unknown
        age = (now - observed).total_seconds()
        return {"observed_at": seen_at, "freshness": "fresh" if age / intervals <= interval else "stale"}

    active_instance = None
    runtime_id = binding.runtime_instance_id if binding is not None else None
    if isinstance(agent, dict) and agent.get("state") == "active":
        instance = agent.get("instance_id")
        reported_runtime = agent.get("runtime_id")
        if isinstance(instance, str) and instance and isinstance(reported_runtime, str) and reported_runtime:
            if runtime_id is None:
                try:
                    identity = decode_identity_record(read_optional(runtime.paths["identity"]))
                    if identity is not None:
                        runtime_id = _effective_runtime_id(identity["instance_id"], identity.get("runtime_id"))
                except (RuntimeError, ValueError):
                    pass
            if runtime_id == reported_runtime:
                active_instance = instance

    heartbeat = {"observed_at": None, "freshness": "unknown"}
    snapshot = {"observed_at": None, "freshness": "unknown"}
    heartbeat_coverage_complete = False
    checkpoint = None
    if active_instance is not None:
        health = read_optional(runtime.root / HEALTH_FILENAME, max_bytes=MAX_HEALTH_BYTES)
        lane = health.get("heartbeat")
        if (
            type(health.get("schema_version")) is int
            and health["schema_version"] == HEALTH_SCHEMA_VERSION
            and health.get("instance_id") == active_instance
            and isinstance(lane, dict)
        ):
            heartbeat = freshness(lane.get("last_success_at"), lane.get("effective_stale_after_seconds"))
            heartbeat_coverage_complete = (
                heartbeat["freshness"] == "fresh"
                and lane.get("state") == "healthy"
                and lane.get("diagnostic_coverage_degraded") is False
            )
        if binding is not None:
            binding_root = runtime.project_paths(binding.project_id)["root"]
            checkpoint = read_optional(binding_root / SNAPSHOT_COVERAGE_FILENAME).get("snapshot_coverage")
            receipt = read_optional(binding_root / "snapshot-observation.json").get("snapshot_observation")
            if (
                isinstance(receipt, dict)
                and receipt.get("agent_instance_id") == active_instance
                and receipt.get("runtime_id") == runtime_id
                and receipt.get("registration_generation") == binding.registration_generation
            ):
                snapshot = freshness(receipt.get("snapshot_at"), receipt.get("heartbeat_interval_seconds"), intervals=3)

    recovery = read_scheduler_recovery(runtime, instance_id=active_instance, read_json=read_optional)
    coverage = diagnostics["coverage"] if binding is not None else "unknown"
    complete_coverage = diagnostics["status"] == "available" and coverage == "complete"
    retired_at = timestamp(diagnostics.get("last_retirement_at"))
    reported_progress_at = timestamp(recovery.get("last_progress_at"))
    if (
        recovery["state"] in {"idle", "awaiting_observation"}
        and active_instance is not None
        and retired_at is not None
        and (reported_progress_at is None or retired_at > reported_progress_at)
    ):
        recovery = {
            **recovery,
            "state": "awaiting_observation",
            "last_progress_at": diagnostics["last_retirement_at"],
        }

    if (
        binding is not None
        and isinstance(checkpoint, dict)
        and checkpoint.get("runtime_id") == runtime_id
        and checkpoint.get("agent_instance_id") == active_instance
        and checkpoint.get("registration_generation") == binding.registration_generation
    ):
        progress_at = timestamp(recovery.get("last_progress_at"))
        snapshot_at = timestamp(checkpoint.get("snapshot_at"))
        accepted_at = timestamp(checkpoint.get("accepted_at"))
        producer_at = timestamp(checkpoint.get("observed_at"))
        ordered = (
            snapshot_at is not None
            and accepted_at is not None
            and producer_at is not None
            and snapshot_at <= accepted_at < producer_at <= now
        )
        overflow = diagnostics.get("overflow", {})
        omitted_scopes = overflow.get("omitted_scopes")
        latest_omission = timestamp(overflow.get("latest_observed_at"))
        heartbeat_at = timestamp(heartbeat["observed_at"])
        # Current scoped observations can cover historical control-plane omissions.
        # Keep the machine-wide overflow marker and its uncertainty.
        scoped_overflow_rechecked = (
            ordered
            and diagnostics["status"] == "available"
            and diagnostics["reason"] == "active_overflow"
            and heartbeat_coverage_complete
            and snapshot["freshness"] == "fresh"
            and isinstance(omitted_scopes, list)
            and 0 < len(omitted_scopes) < 16
            and all(
                isinstance(scope, dict)
                and scope.get("producer") == "control_plane_health"
                and scope.get("scope_type") in {"agent_instance", "project_binding"}
                for scope in omitted_scopes
            )
            and (
                latest_omission is None
                or (snapshot_at > latest_omission and heartbeat_at is not None and heartbeat_at > latest_omission)
            )
        )
        if scoped_overflow_rechecked and diagnostics.get("producer_coverage_complete") is True:
            complete_coverage = True
            coverage = "complete"
        if (
            recovery["state"] == "awaiting_observation"
            and ordered
            and progress_at is not None
            and snapshot_at > progress_at
            and complete_coverage
        ):
            recovery = {
                **recovery,
                "state": "verified",
                "project_id": binding.project_id,
                "reason": None,
                "next_action": None,
            }

    if recovery.get("state") in {"pending", "progressing", "blocked", "awaiting_observation"}:
        complete_coverage = False
        coverage = "incomplete"

    reasons = []
    if binding is None:
        reasons.append("local_binding_unverified")
    for name, evidence in (("heartbeat", heartbeat), ("snapshot", snapshot)):
        if evidence["freshness"] == "stale":
            reasons.append(f"{name}_stale")
        elif evidence["freshness"] != "fresh":
            reasons.append(f"{name}_missing")
    if not complete_coverage:
        reasons.append("diagnostic_coverage_incomplete")
    readiness = {
        "state": "unknown" if reasons else "observed",
        "reasons": reasons,
        "diagnostic_coverage": {"scope": "project_binding", "state": coverage},
    }
    return heartbeat, snapshot, readiness, recovery


def _machine_state_path(cfg: RootConfig, name: str, state_name: str) -> Path:
    return shared_paths(cfg.shared_root)["machines"] / name / "state" / state_name


def _machine_optional_record(
    reader: _ReadBudget,
    path: Path,
    key: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    value = _optional_read(reader, path, f"machine {key}", warnings)
    if value is None:
        if not any(f"machine {key} unavailable" in item for item in warnings):
            warnings.append(f"machine {key} unavailable (record_missing).")
        return None
    record = value.get(key)
    if not isinstance(record, dict):
        warnings.append(f"machine {key} unavailable (record_malformed).")
        return None
    return record


def machine_detail(cfg: RootConfig, name: str) -> dict[str, object]:
    """Return one declared machine and exact local observation snapshots."""
    validate_identifier(name, "machine_name")
    reader = _ReadBudget(max_read_attempts=MACHINE_DETAIL_MAX_READ_ATTEMPTS)
    declaration_path = machine_path(cfg.shared_root, name)
    try:
        declaration_raw = reader.read_json(declaration_path)
    except _BoundedReadFailure as exc:
        raise ValueError(f"machine declaration is malformed: {exc.reason}.") from exc
    if declaration_raw is None:
        raise FileNotFoundError(declaration_path)
    declaration = declaration_raw.get("machine")
    expected_project_id = project_id(cfg.shared_root)
    if (
        not isinstance(declaration, dict)
        or declaration.get("machine_name") != name
        or declaration.get("project_id") != expected_project_id
        or declaration.get("shared_root") != str(cfg.shared_root)
    ):
        raise ValueError("machine declaration is malformed or belongs to another Project.")

    warnings: list[str] = []
    agent = _machine_optional_record(reader, _machine_state_path(cfg, name, "agent.json"), "agent", warnings)
    gpu = _machine_optional_record(reader, _machine_state_path(cfg, name, "gpu.json"), "gpu", warnings)
    summary = _machine_optional_record(reader, _machine_state_path(cfg, name, "summary.json"), "summary", warnings)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "machine_name": name,
        "project": {"path": str(cfg.project_root), "project_id": expected_project_id},
        "declaration": declaration,
        "agent": agent,
        "gpu": gpu,
        "summary": summary,
        "complete": not warnings,
        "warnings": warnings,
        "budget": reader.as_dict(),
    }


__all__ = [
    "MACHINE_DETAIL_MAX_READ_ATTEMPTS",
    "MAX_READ_ATTEMPTS",
    "MAX_RECORD_BYTES",
    "MAX_TOTAL_BYTES",
    "STATUS_SCHEMA_VERSION",
    "machine_detail",
    "project_status",
]
