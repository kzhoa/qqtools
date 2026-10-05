"""Pure agent lifecycle and resource output contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .admin import _validate_upgrade_registry
from .core import OutputContract, OutputKind
from .primitives import (
    _details,
    _mapping,
    _operation,
    _required,
    _required_bool,
    _required_int,
    _required_mapping,
    _required_sequence,
    _sequence,
    _table,
)

CpuLanePayload = Mapping[str, Any]


def _status_section(title: str, fields: Sequence[tuple[str, Any]]) -> str:
    """Render an indented section for human-readable agent status."""
    details = _details(fields)
    if not details:
        return ""
    return f"{title}\n" + "\n".join(f"  {line}" for line in details.splitlines())


def _status_mapping_fields(value: Any) -> list[tuple[str, Any]]:
    """Expand a diagnostic mapping into readable status fields."""
    if not isinstance(value, Mapping):
        return []
    fields: list[tuple[str, Any]] = []
    for key, item in value.items():
        label = str(key).replace("_", " ").capitalize()
        if label.endswith(" id"):
            label = f"{label[:-3]} ID"
        if isinstance(item, Mapping):
            for nested_key, nested_item in item.items():
                nested_label = str(nested_key).replace("_", " ").capitalize()
                fields.append((f"{label} / {nested_label}", nested_item))
        else:
            fields.append((label, item))
    return fields


def _project_readiness(project: Mapping[str, Any]) -> tuple[bool, Any, Any]:
    """Return the ready decision, displayed state, and actionable reason."""
    status = project.get("status")
    eligibility = project.get("eligibility")
    eligibility_state = eligibility.get("state") if isinstance(eligibility, Mapping) else None
    if "write_eligible" in project:
        ready = project.get("write_eligible") is True
    else:
        ready = status in {"registered", "ready", "complete"}
    state = status or eligibility_state or project.get("state")
    reason = project.get("reason", project.get("blocker"))
    if not ready and reason is None:
        reason = eligibility_state
    return ready, state, reason


def _render_agent_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    projects = result.get("projects")
    project_rows: list[tuple[Any, ...]] = []
    if isinstance(projects, Sequence) and not isinstance(projects, (str, bytes)):
        for project in projects:
            if not isinstance(project, Mapping):
                continue
            ready, state, reason = _project_readiness(project)
            project_rows.append((project.get("project_id"), state, reason, ready))
    ready_count = sum(row[3] for row in project_rows)
    readiness = result.get("ready")
    if readiness is None:
        readiness_text = "pending"
    else:
        readiness_text = "ready" if readiness else "not ready"
    migration = result.get("migration")
    migration_state = result.get("migration_state")
    migration_updated = None
    if isinstance(migration, Mapping):
        migration_state = migration_state or migration.get("state")
        migration_updated = migration.get("updated_at")
    fields: list[tuple[str, Any]] = [
        ("Configured mode", result.get("configured_agent_mode", result.get("agent_mode"))),
        ("Observed mode", result.get("observed_agent_mode")),
        ("Agent state", result.get("agent_state", result.get("status", result.get("state")))),
        ("PID", result.get("pid")),
        ("Previous PID", result.get("previous_pid")),
        ("Ready", readiness_text),
        ("Projects ready", f"{ready_count}/{len(project_rows)}" if project_rows else "0/0"),
        ("Migration state", migration_state),
        ("Migration updated", migration_updated),
        ("Reason", result.get("reason")),
        ("Message", result.get("message")),
        ("Error", result.get("error")),
        ("Operation reference", result.get("operation_reference")),
        ("Next", result.get("next_action", result.get("follow_up_command"))),
    ]
    warnings = result.get("warnings") or []
    if warnings:
        fields.append(
            ("Warnings", [item.get("message", item) if isinstance(item, Mapping) else item for item in warnings])
        )
    rendered = _operation(
        str(result.get("action") or "agent").replace("_", " "),
        str(result.get("outcome", result.get("agent_state", result.get("status", result.get("state"))))).replace(
            "_", " "
        ),
        fields,
    )
    blockers = [row[:3] for row in project_rows if not row[3]]
    if blockers:
        rendered += "\n\n" + _table(("Project", "State", "Reason"), blockers[:10], empty_message="No Project blockers.")
    return rendered


def _render_agent_status(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    rendered = _render_agent_operation(result, _presentation)
    if result.get("stop_reason") is not None:
        rendered = f"{rendered}\n\nStop reason: {result.get('stop_reason')}"
    project_io = result.get("project_io_isolation")
    if isinstance(project_io, Mapping) and project_io.get("envelope") != "healthy":
        project_io_details = _status_section(
            "Attention required",
            (
                ("Project I/O isolation status", project_io.get("envelope")),
                ("Blocking projects", project_io.get("blocking_project_ids")),
                ("Oldest overdue request", project_io.get("oldest_overdue_at")),
                ("Free executor slots", project_io.get("free_slot_count")),
                ("Executor epoch", project_io.get("executor_epoch")),
                (
                    "Workers (active / overdue / exit unverified)",
                    f"{project_io.get('active_worker_count')} / {project_io.get('overdue_worker_count')} / "
                    f"{project_io.get('exit_unverified_worker_count')}",
                ),
                ("Unreaped workers", project_io.get("unreaped_worker_count")),
            ),
        )
        if project_io_details:
            rendered = f"{rendered}\n\n{project_io_details}" if rendered else project_io_details
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        last_exit = diagnostics.get("last_exit")
        last_exit = last_exit if isinstance(last_exit, Mapping) else {}
        diagnostic_sections = [
            _status_section(
                "Diagnostics",
                (
                    ("State", diagnostics.get("state")),
                    ("Capture", diagnostics.get("capture_mode")),
                    ("Capture health", diagnostics.get("capture_health")),
                    ("Log", diagnostics.get("log_path")),
                    ("Persistent log available", diagnostics.get("log_available")),
                    ("Instance", diagnostics.get("instance_id")),
                    (
                        "Log rotation (configured)",
                        diagnostics.get("configured_log_max_size", diagnostics.get("configured_log_max_bytes")),
                    ),
                    (
                        "Log rotation (effective)",
                        diagnostics.get("effective_log_max_size", diagnostics.get("effective_log_max_bytes")),
                    ),
                ),
            ),
            _status_section("Current startup", _status_mapping_fields(diagnostics.get("summary"))),
            _status_section(
                "Previous exit (stale)" if diagnostics.get("last_exit_stale") else "Previous exit",
                (
                    ("Source", diagnostics.get("last_exit_source")),
                    ("Reason", last_exit.get("reason")),
                    ("Handled signal", last_exit.get("handled_signal")),
                    ("Cleanup", last_exit.get("cleanup_outcome")),
                    ("Instance", last_exit.get("instance_id")),
                    (
                        "Full details",
                        "qexp agent status --format json"
                        if last_exit or diagnostics.get("last_exit_source") is not None
                        else None,
                    ),
                ),
            ),
            _status_section("Diagnostic coverage", _status_mapping_fields(diagnostics.get("coverage"))),
        ]
        diagnostic_details = "\n\n".join(section for section in diagnostic_sections if section)
        if diagnostic_details:
            rendered = f"{rendered}\n\n{diagnostic_details}" if rendered else diagnostic_details
    scheduler = result.get("scheduler_diagnostics")
    if isinstance(scheduler, Mapping):
        scheduler_details = _status_section(
            "Scheduling diagnostics",
            (
                ("State", scheduler.get("status")),
                ("Coverage", scheduler.get("coverage")),
                ("Observed", scheduler.get("observed_at")),
                ("Active findings", scheduler.get("active_count")),
                ("Reason", scheduler.get("reason")),
            ),
        )
        if scheduler_details:
            rendered = f"{rendered}\n\n{scheduler_details}" if rendered else scheduler_details
    return rendered


def _render_scheduler_diagnostics(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    rows = []
    for item in result.get("items", []):
        if not isinstance(item, Mapping):
            continue
        identity = item.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        rows.append(
            (
                identity.get("producer"),
                identity.get("reason_code"),
                identity.get("project_id"),
                identity.get("scope_type"),
                item.get("last_observed_at", item.get("resolved_at")),
            )
        )
    rendered = _details(
        (
            ("Action", str(result.get("action", "scheduler diagnostics")).replace("_", " ")),
            ("Coverage", result.get("coverage")),
            ("Observed", result.get("observed_at")),
            ("Snapshot revision", result.get("snapshot_revision")),
            ("Total", result.get("total")),
            ("Reason", result.get("reason")),
            ("End of capture", result.get("end_of_capture")),
            ("Next cursor", result.get("next_cursor")),
        )
    )
    table = _table(
        ("Producer", "Reason", "Project", "Scope", "Observed/resolved"),
        rows,
        empty_message="No scheduler diagnostics.",
    )
    return f"{rendered}\n\n{table}" if rendered else table


def _render_agent_config(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Outcome", str(result.get("action", "shown")).replace("_", " ")),
            ("Agent name", result.get("agent_name")),
            ("Agent mode", result.get("agent_mode")),
            ("Log rotation trigger", result.get("log_max_size", result.get("log_max_bytes"))),
        )
    )


def _render_agent_readiness(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        "agent start",
        "ready" if result.get("ready") else "not ready",
        (
            ("MachineRuntime root", result.get("machine_runtime_root")),
            ("Runtime ID", result.get("runtime_id")),
            ("Agent state", result.get("agent_state")),
            ("Requested policy revision", result.get("requested_policy_revision")),
            ("Observed policy revision", result.get("observed_policy_revision")),
            ("Inventory revision", result.get("inventory_revision")),
            ("Reasons", result.get("reasons", result.get("reason"))),
            ("Projects", result.get("projects")),
        ),
    )


def _render_cpu_lane(result: CpuLanePayload, _presentation: Mapping[str, object]) -> str:
    lane = result["cpu_lane"]
    return _details(
        (
            ("Outcome", str(result.get("action", "shown")).replace("_", " ")),
            ("MachineRuntime root", result.get("machine_runtime_root")),
            ("Capacity", lane["capacity"]),
            ("Revision", lane["revision"]),
        )
    )


def _render_gpu_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    warnings = result.get("warnings") or []
    warning_values = [item.get("message", item) if isinstance(item, Mapping) else item for item in warnings]
    return _details(
        (
            ("Outcome", str(result.get("action", "shown")).replace("_", " ")),
            ("MachineRuntime root", result.get("machine_runtime_root")),
            ("Mode", result.get("mode")),
            ("Source", result.get("source")),
            ("Revision", result.get("revision")),
            ("Previous revision", result.get("previous_revision")),
            ("Current revision", result.get("current_revision")),
            ("Configured GPUs", result.get("configured_gpu_ids")),
            ("Discovered GPUs", result.get("discovered_gpu_ids")),
            ("Visible GPUs", result.get("visible_gpu_ids")),
            ("Reserved GPUs", result.get("reserved_gpu_ids")),
            ("Unreserved GPUs", result.get("unreserved_gpu_ids")),
            ("Draining GPUs", result.get("draining_gpu_ids")),
            ("Entered draining GPUs", result.get("entered_draining_gpu_ids")),
            ("Discovery status", result.get("discovery_status")),
            ("Visible status", result.get("visible_status")),
            ("Warnings", warning_values),
            ("Agent running", result.get("agent_running")),
        )
    )


def _validate_agent_operation(result: Any) -> None:
    value = _mapping(result, "agent-operation payload")
    action = _required(value, "action", "agent-operation payload")
    project_actions = {
        "project_added",
        "project_already_registered",
        "project_enabled",
        "project_disabled",
        "project_removed",
        "project_migrated",
    }
    lifecycle_actions = {"started", "already_running", "restarted", "stopped", "already_stopped"}
    if action in project_actions:
        for key in ("project_id", "shared_root", "machine_name", "enabled"):
            _required(value, key, "agent-operation payload")
        if action == "project_migrated":
            for key in ("agent_state", "pid", "is_running"):
                _required(value, key, "agent-operation payload")
            if "migration_state" in value and not isinstance(value["migration_state"], str):
                raise TypeError("agent-operation payload.migration_state must be a string")
            if "migration" in value:
                _mapping(value["migration"], "agent-operation payload.migration")
            if "error" in value:
                _mapping(value["error"], "agent-operation payload.error")
        return
    if action == "running":
        for key in ("agent_state", "is_running", "machine_runtime_root"):
            _required(value, key, "agent-operation payload")
        return
    if action in lifecycle_actions:
        for key in ("machine_runtime_root", "agent_state", "pid", "is_running"):
            _required(value, key, "agent-operation payload")
        return
    raise ValueError(f"agent-operation payload has unknown action {action!r}")


def _validate_project(value: Any, label: str) -> None:
    project = _mapping(value, label)
    for key in ("project_id", "shared_root", "machine_name", "enabled", "state", "eligibility", "write_eligible"):
        _required(project, key, label)
    _required_bool(project, "enabled", label)
    eligibility = _required_mapping(project, "eligibility", label)
    _required(eligibility, "state", f"{label}.eligibility")
    _required_bool(project, "write_eligible", label)


def _validate_agent_config(result: Any) -> None:
    value = _mapping(result, "agent-config payload")
    action = _required(value, "action", "agent-config payload")
    if action not in {"shown", "updated"}:
        raise ValueError("agent-config payload.action is invalid")
    for key in (
        "machine_runtime_root",
        "agent_name",
        "agent_mode",
        "revision",
        "provenance",
        "runtime_id",
        "log_max_bytes",
        "log_max_size",
    ):
        _required(value, key, "agent-config payload")
    _required_int(value, "revision", "agent-config payload")
    if type(value["log_max_bytes"]) is not int or value["log_max_bytes"] <= 0:
        raise TypeError("agent-config payload.log_max_bytes must be a positive integer")
    if not isinstance(value["log_max_size"], str):
        raise TypeError("agent-config payload.log_max_size must be a string")


def _validate_agent_readiness(result: Any) -> None:
    value = _mapping(result, "agent-readiness payload")
    _required(value, "machine_runtime_root", "agent-readiness payload")
    _required_bool(value, "ready", "agent-readiness payload")
    _required(value, "projects", "agent-readiness payload")
    _sequence(value["projects"], "agent-readiness payload.projects")


def _validate_agent_status(result: Any) -> None:
    value = _mapping(result, "agent-status payload")
    if _required(value, "action", "agent-status payload") != "status":
        raise ValueError("agent-status payload.action must be 'status'")
    for key in ("machine_runtime_root", "agent_state", "pid", "registry_revision"):
        _required(value, key, "agent-status payload")
    projects = _required_sequence(value, "projects", "agent-status payload")
    for index, project in enumerate(projects):
        _validate_project(project, f"agent-status payload.projects[{index}]")
    upgrade = _required_mapping(value, "upgrade", "agent-status payload")
    _validate_upgrade_registry(upgrade)
    if "gpu_policy" in value:
        _validate_gpu_policy_view(value["gpu_policy"], "agent-status payload.gpu_policy")
    if "warnings" in value:
        _sequence(value["warnings"], "agent-status payload.warnings")
    if "diagnostics" in value:
        diagnostics = _mapping(value["diagnostics"], "agent-status payload.diagnostics")
        if "log_available" in diagnostics and type(diagnostics["log_available"]) is not bool:
            raise TypeError("agent-status payload.diagnostics.log_available must be a boolean")
        for key in ("configured_log_max_bytes", "effective_log_max_bytes"):
            if (
                key in diagnostics
                and not (key == "effective_log_max_bytes" and diagnostics[key] is None)
                and (type(diagnostics[key]) is not int or diagnostics[key] <= 0)
            ):
                raise TypeError(f"agent-status payload.diagnostics.{key} must be a positive integer")
        for key in ("available", "summary_synthetic", "last_exit_historical", "last_exit_stale", "historical"):
            if key in diagnostics and type(diagnostics[key]) is not bool:
                raise TypeError(f"agent-status payload.diagnostics.{key} must be a boolean")
        for key in (
            "instance_id",
            "state",
            "log_path",
            "configured_log_max_size",
            "effective_log_max_size",
            "capture_mode",
            "capture_health",
            "last_exit_source",
        ):
            if key in diagnostics and diagnostics[key] is not None and not isinstance(diagnostics[key], str):
                raise TypeError(f"agent-status payload.diagnostics.{key} must be a string or null")
        for key in ("summary", "last_exit", "coverage"):
            if key in diagnostics and diagnostics[key] is not None:
                _mapping(diagnostics[key], f"agent-status payload.diagnostics.{key}")
    if "scheduler_diagnostics" in value:
        scheduler = _mapping(
            value["scheduler_diagnostics"],
            "agent-status payload.scheduler_diagnostics",
        )
        for key in ("schema_version", "status", "coverage", "truncated", "reason"):
            _required(scheduler, key, "agent-status payload.scheduler_diagnostics")
    if "project_io_isolation" in value:
        _validate_project_io_isolation(value["project_io_isolation"])


def _validate_project_io_isolation(value: Any) -> None:
    label = "agent-status payload.project_io_isolation"
    isolation = _mapping(value, label)
    expected_keys = {
        "protocol_version",
        "executor_epoch",
        "capacity",
        "active_worker_count",
        "overdue_worker_count",
        "exit_unverified_worker_count",
        "unreaped_worker_count",
        "free_slot_count",
        "supported_hang_limit",
        "envelope",
        "oldest_overdue_at",
        "blocking_project_ids",
    }
    if set(isolation) != expected_keys:
        raise ValueError(f"{label} has missing or unknown fields")
    if type(isolation["protocol_version"]) is not int or isolation["protocol_version"] != 1:
        raise ValueError(f"{label}.protocol_version is unsupported")
    epoch = isolation["executor_epoch"]
    if epoch is not None and (not isinstance(epoch, str) or not re.fullmatch(r"[0-9a-f]{32}", epoch)):
        raise TypeError(f"{label}.executor_epoch must be a lowercase hexadecimal ID or null")
    if type(isolation["capacity"]) is not int or isolation["capacity"] != 4:
        raise ValueError(f"{label}.capacity must be 4")
    if type(isolation["supported_hang_limit"]) is not int or isolation["supported_hang_limit"] != 2:
        raise ValueError(f"{label}.supported_hang_limit must be 2")
    for key in (
        "active_worker_count",
        "overdue_worker_count",
        "exit_unverified_worker_count",
        "unreaped_worker_count",
        "free_slot_count",
    ):
        count = isolation[key]
        if type(count) is not int or count < 0 or count > 4:
            raise TypeError(f"{label}.{key} must be an integer from 0 through 4")
    if isolation["envelope"] not in {"healthy", "degraded", "exceeded", "unknown"}:
        raise ValueError(f"{label}.envelope is invalid")
    oldest = isolation["oldest_overdue_at"]
    if oldest is not None:
        if not isinstance(oldest, str) or not oldest:
            raise TypeError(f"{label}.oldest_overdue_at must be a non-empty string or null")
        try:
            timestamp = datetime.fromisoformat(oldest[:-1] + "+00:00" if oldest.endswith("Z") else oldest)
        except ValueError as exc:
            raise ValueError(f"{label}.oldest_overdue_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{label}.oldest_overdue_at must include a timezone")
    projects = isolation["blocking_project_ids"]
    if not isinstance(projects, list) or len(projects) > 4:
        raise TypeError(f"{label}.blocking_project_ids must be a list of at most four IDs")
    if any(not isinstance(project_id, str) or not project_id or len(project_id) > 128 for project_id in projects):
        raise TypeError(f"{label}.blocking_project_ids contains an invalid ID")
    if len(set(projects)) != len(projects):
        raise ValueError(f"{label}.blocking_project_ids must not contain duplicates")


def _validate_scheduler_diagnostics(result: Any) -> None:
    value = _mapping(result, "scheduler-diagnostics payload")
    action = _required(value, "action", "scheduler-diagnostics payload")
    if action not in {"diagnostics_active", "diagnostics_history"}:
        raise ValueError("scheduler-diagnostics payload.action is invalid")
    for key in (
        "schema_version",
        "machine_runtime_root",
        "coverage",
        "items",
        "truncated",
    ):
        _required(value, key, "scheduler-diagnostics payload")
    _required_int(value, "schema_version", "scheduler-diagnostics payload")
    _required_sequence(value, "items", "scheduler-diagnostics payload")
    _required_bool(value, "truncated", "scheduler-diagnostics payload")
    if value["coverage"] not in {"complete", "incomplete", "unknown"}:
        raise ValueError("scheduler-diagnostics payload.coverage is invalid")
    if action == "diagnostics_history":
        _required_bool(value, "end_of_capture", "scheduler-diagnostics payload")


def _validate_cpu_lane(result: Any) -> None:
    value = _mapping(result, "cpu-lane payload")
    action = _required(value, "action", "cpu-lane payload")
    if action not in {"shown", "updated"}:
        raise ValueError("cpu-lane payload.action is invalid")
    _required(value, "machine_runtime_root", "cpu-lane payload")
    lane = _required_mapping(value, "cpu_lane", "cpu-lane payload")
    _required_int(lane, "capacity", "cpu-lane payload.cpu_lane")
    _required_int(lane, "revision", "cpu-lane payload.cpu_lane")


def _validate_gpu_policy_view(result: Any, label: str) -> None:
    value = _mapping(result, label)
    for key in (
        "mode",
        "source",
        "revision",
        "configured_gpu_ids",
        "discovered_gpu_ids",
        "visible_gpu_ids",
        "undiscovered_configured_gpu_ids",
        "reserved_gpu_ids",
        "unreserved_gpu_ids",
        "draining_gpu_ids",
        "discovery_status",
        "visible_status",
        "warnings",
        "agent_running",
    ):
        _required(value, key, label)
    if value["mode"] not in {"auto", "explicit"}:
        raise ValueError(f"{label}.mode is invalid")
    if value["source"] not in {"discovery", "environment", "persisted", "pending", "unavailable"}:
        raise ValueError(f"{label}.source is invalid")
    if type(value["revision"]) is not int or value["revision"] < 0:
        raise TypeError(f"{label}.revision must be a nonnegative integer")
    for key in (
        "configured_gpu_ids",
        "discovered_gpu_ids",
        "visible_gpu_ids",
        "undiscovered_configured_gpu_ids",
        "reserved_gpu_ids",
        "unreserved_gpu_ids",
        "draining_gpu_ids",
    ):
        ids = value[key]
        if ids is not None:
            values = _sequence(ids, f"{label}.{key}")
            if any(type(item) is not int or item < 0 for item in values):
                raise TypeError(f"{label}.{key} must contain nonnegative integers")
    _sequence(value["warnings"], f"{label}.warnings")
    _required_bool(value, "agent_running", label)


def _validate_gpu_policy_payload(result: Any) -> None:
    value = _mapping(result, "gpu-policy payload")
    action = _required(value, "action", "gpu-policy payload")
    if action not in {"shown", "updated", "reset"}:
        raise ValueError("gpu-policy payload.action is invalid")
    _required(value, "machine_runtime_root", "gpu-policy payload")
    _validate_gpu_policy_view(value, "gpu-policy payload")


CONTRACTS = {
    OutputKind.AGENT_OPERATION: OutputContract(_validate_agent_operation, _render_agent_operation),
    OutputKind.AGENT_STATUS: OutputContract(_validate_agent_status, _render_agent_status),
    OutputKind.AGENT_CONFIG: OutputContract(_validate_agent_config, _render_agent_config),
    OutputKind.AGENT_READINESS: OutputContract(_validate_agent_readiness, _render_agent_readiness),
    OutputKind.SCHEDULER_DIAGNOSTICS: OutputContract(_validate_scheduler_diagnostics, _render_scheduler_diagnostics),
    OutputKind.CPU_LANE: OutputContract(_validate_cpu_lane, _render_cpu_lane),
    OutputKind.GPU_POLICY: OutputContract(_validate_gpu_policy_payload, _render_gpu_policy),
}

__all__ = ["CONTRACTS"]
