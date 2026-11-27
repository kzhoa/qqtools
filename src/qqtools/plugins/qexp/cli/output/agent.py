"""Pure agent lifecycle and resource output contracts."""

from __future__ import annotations

import math
import re
import shlex
import textwrap
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config_types import MAX_IDLE_JOB_STOP_TIMEOUT_SECONDS
from ..project_presentation import encode_display_text
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
_HEALTH_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]{1,160}")
_HEALTH_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,95}")


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


def _render_affected_projects(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    raw_items = value.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return ""
    lines: list[str] = []
    for item in raw_items[:8]:
        if not isinstance(item, Mapping) or item.get("overdue") is not True:
            continue
        detail = str(item.get("stage"))
        if item.get("reason_code") is not None:
            detail += f" ({item.get('reason_code')})"
        lines.append(f"  {item.get('project_id')}: {detail}")
    omitted = value.get("omitted_count")
    if lines and type(omitted) is int and omitted > 0:
        lines.append(f"  … {omitted} additional publication entries omitted")
    return "Affected Projects\n" + "\n".join(lines) if lines else ""


def _render_registration_renewals(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    lines = []
    for item in value["items"]:
        blocker = item["blocking_reason"]
        cause = item["cause"]
        if blocker is None and cause is None:
            continue
        details = [f"registration {item['registration_generation']}"]
        if (outcome := item["last_outcome"]) is not None:
            details.append(outcome)
        if blocker is not None:
            details.append(blocker)
        if cause is not None:
            details.append(f"cause {cause}")
        if (retry_at := item["next_retry_at"]) is not None:
            details.append(f"next retry {retry_at}")
        lines.append(f"  {item['project_id']}: {', '.join(details)}")
    if not lines:
        return []
    if omitted := value["omitted_count"]:
        lines.append(f"  … {omitted} additional renewal entries omitted")
    return ["Project registration renewals", *lines]


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
    action = result.get("action")
    if action == "restarted":
        return _render_agent_restart(result)
    if action in {"started", "already_running"} and "inventory_revision" in result:
        lines = ["Agent started." if action == "started" else "Agent already running."]
        lines.extend(_agent_scope_lines(result))
        projects = result.get("projects")
        if projects:
            lines.append("Startup readiness verified for the enabled Projects.")
        else:
            lines.append("Machine-only daemon ready with no enabled Projects.")
        return "\n".join(lines)
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


_IDLE_JOB_WAIT_MESSAGES = {
    "dispatch_cycle_incomplete": "Waiting for the dispatch cycle to complete.",
    "enablement_unready": "Waiting for Project enablement to converge.",
    "bindings_unknown": "Waiting for valid Project registrations.",
    "launch_handoff_pending": "Waiting for pending training launches to finish handing off.",
    "upgrade_evidence_unknown": "Waiting for project upgrade information before checking for training.",
    "recovery_enrollment_pending": "Waiting for Project recovery enrollment to complete.",
    "maintenance_retry_pending": "Waiting for pending Project maintenance to retry.",
    "partition_unknown": "Waiting for the unregistered Project partition scan.",
    "recovery_unknown": "Waiting for Project recovery evidence.",
    "authority_unready": "Waiting for project scheduling to become ready before checking for training.",
    "demand_census_unknown": "Waiting for the training-demand census.",
    "executor_unready": "Waiting for the Project I/O executor to become ready.",
    "gpu_policy_unknown": "Waiting for current GPU policy and visibility information.",
    "cpu_policy_unknown": "Waiting for current CPU policy information.",
    "observation_failed": "Waiting after a training-demand observation failed.",
}


def _render_idle_job_status(status: Mapping[str, Any]) -> str:
    fields = {key: value for key, value in status.items() if key not in {"demand_detail", "interval_detail"}}
    rendered = _status_section("Idle Job", _status_mapping_fields(fields))
    interval = status.get("interval_detail")
    if isinstance(interval, Mapping):
        rendered += f"\n  Idle elapsed when last observed: {interval['idle_elapsed_seconds']:.1f} seconds"
        if (reset := interval.get("last_reset")) is not None:
            rendered += f"\n  Last idle reset: {reset['check']} after {reset['elapsed_seconds']:.1f} seconds at {reset['reset_at']}"
        if (stop := interval.get("last_stop")) is not None:
            rendered += (
                f"\n  Last stop request: {stop['check']} at {stop['stopped_at']} (instance {stop['instance_id']})"
            )
    if status.get("reason") == "idle_delay":
        return f"{rendered}\n  Training is idle. Waiting for the configured start delay."
    detail = status.get("demand_detail")
    if not isinstance(detail, Mapping):
        return rendered
    raw_projects = detail.get("projects")
    projects = (
        [
            project
            for project in raw_projects
            if isinstance(project, Mapping) and isinstance(project.get("project_id"), str)
        ]
        if isinstance(raw_projects, list)
        else []
    )
    names = [project["project_id"] for project in projects]
    live = detail.get("live") is True
    code = detail.get("code")
    count = detail.get("project_count")
    lines = []
    if code == "upgrade_blocks_idle":
        repairs = [
            project["project_id"]
            for project in projects
            if project.get("state") == "repair_required" and project.get("migration") and project.get("phase")
        ]
        if count == 1 and repairs:
            message = f"qexp cannot confirm training is idle while the {repairs[0]} upgrade needs repair."
        else:
            message = "qexp cannot confirm training is idle while Project upgrades are blocking the check."
        if live:
            action = "Stopping" if status.get("state") == "stopping" else "Waiting"
            message = f"{action}. {message}"
        if repairs and count != 1:
            lines.append(f"Upgrades needing repair: {', '.join(repairs)}")
    else:
        message = _IDLE_JOB_WAIT_MESSAGES.get(code, "Waiting to confirm whether training is idle.")
    if names and (code != "upgrade_blocks_idle" or count != 1 or not repairs):
        if type(count) is int and count > len(names):
            lines.append(f"Blocking Projects ({len(names)} of {count} shown): {', '.join(names)}")
        else:
            lines.append(f"Blocking Projects: {', '.join(names)}")
    if not live:
        message = f"Last observed: {message}"
    elapsed = detail.get("elapsed_seconds")
    if live and type(elapsed) in {int, float}:
        lines.append(f"Elapsed wait: {elapsed:.1f} seconds")
    for project in projects:
        if command := project.get("detail_command"):
            lines.append(f"Inspect: {command}")
    return "\n".join((rendered, *(f"  {line}" for line in (message, *lines))))


def _agent_scope_lines(result: Mapping[str, Any]) -> list[str]:
    root = result.get("machine_runtime_root")
    lines = [f"MachineRuntime root: {encode_display_text(str(root))}"] if root is not None else []
    configured = result.get("configured_agent_mode")
    observed = result.get("observed_agent_mode")
    if configured:
        mode = str(configured).replace("_", " ")
        readiness = result.get("project_readiness") or {}
        if (
            observed
            and observed != configured
            and result.get("agent_state") == "active"
            and readiness.get("reason") != "replacement_observation_unavailable"
        ):
            mode += f" configured, {str(observed).replace('_', ' ')} observed"
        lines.append(f"Mode: {mode}")
    return lines


def _machine_command(result: Mapping[str, Any], *args: str) -> str:
    root = result.get("machine_runtime_root")
    scope = ("--machine-runtime-root", str(root)) if root is not None else ()
    return shlex.join(("qexp", *scope, *args))


def _next_step(description: str, command: str) -> list[str]:
    return [*textwrap.wrap(description, width=80), f"  {command}"]


_STATUS_REASONS = {
    "agent_not_ready": "The agent has not reported startup readiness.",
    "authority_unavailable": "Current registration authority could not be verified.",
    "concurrent_inventory_change": "The Project inventory changed during startup.",
    "concurrent_registry_change": "The Project registry changed during startup.",
    "enablement_mirror_diverged": "The Project inventory and registry disagree about enablement.",
    "identity_path_conflict": "The Project ID and path disagree with local registration.",
    "initial_reconciliation_pending": "Initial Project reconciliation is pending.",
    "invalid": "The Project registration could not be verified.",
    "inventory_entry_removed": "The Project inventory entry was removed during startup.",
    "inventory_mirror_incomplete": "The Project inventory and registry have not converged.",
    "legacy_project_requires_migration": "The Project still uses a legacy registration.",
    "no_enabled_projects": "This agent mode requires an enabled Project registered here.",
    "not_registered": "The Project is in the inventory and needs registration.",
    "project_roster_unavailable": "The Project inventory or registry could not be read.",
    "registry_enablement_unknown": "The Project inventory or registry could not be read.",
    "replacement_observation_unavailable": "The replacement process has no matching status observation yet.",
    "snapshot_observation_expired": "The latest Project snapshot observation has expired.",
    "snapshot_observation_identity_mismatch": "The snapshot is from another agent or registration.",
    "snapshot_observation_invalid": "The Project snapshot observation is invalid.",
    "snapshot_observation_unavailable": "A current Project snapshot observation is unavailable.",
    "stale_agent_identity": "The status record does not match the current agent process.",
    "stale_runtime_generation": "The status record is from another runtime identity.",
    "superseded": "The local Project registration has been superseded.",
    "unregistered": "The current Project registration is missing.",
}


def _reason_text(value: object) -> str:
    reason = str(value)
    return _STATUS_REASONS.get(reason, reason.replace("_", " ").capitalize().removesuffix(".") + ".")


def _readiness_card(result: Mapping[str, Any], *, restarting: bool = False) -> tuple[list[str], tuple[str, str] | None]:
    readiness = result.get("project_readiness")
    if not isinstance(readiness, Mapping):
        if not restarting:
            return [], None
        readiness = {}
    state = readiness.get("state")
    enabled = readiness.get("enabled_count")
    if state == "machine_only":
        return ["Machine-only daemon ready with no enabled Projects."], None
    if state == "observed":
        scope = "1 enabled Project" if enabled == 1 else f"all {enabled} enabled Projects"
        return [f"Project readiness verified for {scope}."], None
    if state == "inactive" and not restarting and result.get("agent_state") == "stopped":
        return ["Project readiness is unavailable while the agent is stopped."], None

    lines = ["Project readiness has not been verified."]
    reason = readiness.get("reason")
    if state == "inventory_unknown":
        lines.append(_reason_text(reason or "project_roster_unavailable"))
    elif state == "inactive":
        lines.append("A current agent process identity could not be verified.")
    elif state == "pending":
        fresh = readiness.get("fresh_count")
        checked = readiness.get("checked_count")
        if checked:
            lines.append(f"Fresh observations for {fresh} of {checked} Projects checked.")
            if checked < enabled:
                lines.append(f"The sample covers {checked} of {enabled} enabled Projects.")
        if reason == "startup_readiness_unverified":
            lines.append("Startup readiness for the current Project inventory is pending.")
    affected = readiness.get("affected_projects") or ()
    for project in affected[:3]:
        lines.extend(
            (
                f"  Project: {encode_display_text(project['path'])}",
                *textwrap.wrap(
                    _reason_text(project["reason"]), width=80, initial_indent="    ", subsequent_indent="    "
                ),
            )
        )
    if len(affected) > 3:
        lines.append(f"  {len(affected) - 3} more checked Projects need attention.")
    if state == "inventory_unknown" and reason != "replacement_observation_unavailable":
        root = Path(result["machine_runtime_root"])
        return lines, (
            "Inspect the Project inventory and registry paths.",
            shlex.join(("ls", "-ld", "--", str(root / "inventory.json"), str(root / "registry.json"))),
        )
    if reason == "project_sample_limited" and (path := readiness.get("first_unchecked_project")):
        lines.extend(
            _next_step(
                "Inspect the first Project outside this sample.", _machine_command(result, "--project", path, "status")
            )
        )
        if not affected:
            return lines, None
    if restarting or not affected:
        return lines, ("Recheck process and Project readiness.", _machine_command(result, "agent", "status"))
    project = affected[0]
    return lines, (
        "Inspect the affected Project's local agent and scheduling evidence.",
        _machine_command(result, "--project", project["path"], "status"),
    )


def _diagnostic_recovery_card(result: Mapping[str, Any]) -> tuple[list[str], tuple[str, str] | None]:
    recovery = result.get("diagnostic_recovery")
    if not isinstance(recovery, Mapping):
        return [], None
    state = recovery.get("state")
    messages = {
        "pending": "Scheduler diagnostic recovery is pending for this runtime.",
        "progressing": "Scheduler diagnostic recovery is progressing for this runtime.",
        "blocked": "Scheduler diagnostic recovery is blocked for this runtime.",
    }
    if state not in messages:
        return [], None
    lines = [messages[state]]
    if state == "progressing":
        lines.append("Recovery can continue without another restart.")
    if reason := recovery.get("reason"):
        lines.append(_reason_text(reason))
    if state == "blocked":
        return lines, (
            "Inspect the diagnostic recovery blocker without changing local state.",
            _machine_command(result, "agent", "diagnostics", "repair", "--dry-run"),
        )
    return lines, None


def _render_agent_restart(result: Mapping[str, Any]) -> str:
    title = "Agent restarted." if result.get("previous_pid") is not None else "Agent started via restart."
    lines = [title, *_agent_scope_lines(result)]
    readiness, readiness_step = _readiness_card(result, restarting=True)
    lines.extend(readiness)
    view = result.get("project_readiness")
    if isinstance(view, Mapping) and view.get("reason") == "replacement_observation_unavailable":
        recovery, recovery_step = [], None
    else:
        recovery, recovery_step = _diagnostic_recovery_card(result)
    lines.extend(recovery)
    if step := recovery_step or readiness_step:
        lines.extend(_next_step(*step))
    return "\n".join(lines)


def _render_agent_status(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    active = result.get("agent_state") == "active"
    name = result.get("agent_name")
    if active:
        lines = ["Agent active."]
    elif result.get("pid") is not None:
        lines = ["Agent process state is unverified."]
    else:
        lines = ["Agent stopped."]
    if isinstance(name, str):
        lines.append(f"Machine name: {encode_display_text(name)}")
    lines.extend(_agent_scope_lines(result))
    if reason := result.get("stop_reason"):
        lines.append(f"Stop reason: {_reason_text(reason)}")
    readiness, readiness_step = _readiness_card(result)
    lines.extend(readiness)

    diagnostics_step = (
        "Inspect machine scheduler diagnostics for the reported blocker.",
        _machine_command(result, "agent", "diagnostics", "active"),
    )
    control_step = None
    publication_step = None
    control = result.get("control_plane_health")
    if isinstance(control, Mapping):
        heartbeat = control.get("heartbeat") or {}
        authority = control.get("authority") or {}
        affected = _render_affected_projects(control.get("project_snapshot_publications"))
        for label, lane in (("Heartbeat", heartbeat), ("Authority", authority)):
            state = lane.get("state")
            if state in {"degraded", "stalled", "retrying"} or (active and state in {"stopped", "unknown"}):
                lines.append(f"Control plane {label.lower()} is {state}.")
                failure = lane.get("last_failure") or {}
                if reason := failure.get("reason_code"):
                    lines.extend(
                        textwrap.wrap(_reason_text(reason), width=80, initial_indent="  ", subsequent_indent="  ")
                    )
                if state in {"degraded", "stalled", "stopped"}:
                    control_step = diagnostics_step
                else:
                    publication_step = diagnostics_step
        if affected:
            lines.extend(("Project snapshot publication is overdue.", affected))
            publication_step = diagnostics_step
        elif not control_step and control.get("state") in {"degraded", "stalled", "stopped"} and active:
            lines.append(f"Control plane is {control['state']}.")
            control_step = diagnostics_step
        if batch_error := control.get("snapshot_batch_error"):
            lines.append(f"Project snapshot batch error: {batch_error}")
        lines.extend(_render_registration_renewals(control.get("project_registration_renewals")))

    recovery, recovery_step = _diagnostic_recovery_card(result)
    lines.extend(recovery)
    warnings_step = None
    isolation = result.get("project_io_isolation")
    if isinstance(isolation, Mapping):
        envelope = isolation.get("envelope")
        if envelope in {"degraded", "exceeded"} or (active and envelope == "unknown"):
            lines.append(f"Project I/O isolation is {envelope}.")
            for project_id in (isolation.get("blocking_project_ids") or ())[:4]:
                lines.append(f"  Project ID: {encode_display_text(project_id)}")
            warnings_step = diagnostics_step

    scheduler = result.get("scheduler_diagnostics")
    if isinstance(scheduler, Mapping):
        active_count = scheduler.get("active_count")
        diagnostic_reason = scheduler.get("reason")
        if scheduler.get("coverage") == "incomplete" or (active and scheduler.get("status") == "unavailable"):
            lines.append("Scheduler diagnostic coverage is incomplete.")
            if diagnostic_reason:
                lines.append(f"  {_reason_text(diagnostic_reason)}")
            warnings_step = diagnostics_step
        elif isinstance(diagnostic_reason, str) and "clock" in diagnostic_reason:
            lines.append(_reason_text(diagnostic_reason))
            warnings_step = diagnostics_step
        if type(active_count) is int and active_count > 0:
            lines.append(f"Active scheduler findings: {active_count}")
            warnings_step = diagnostics_step

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        capture = diagnostics.get("capture_health")
        if active and capture in {"degraded", "unavailable"}:
            lines.append(f"Agent log capture is {capture}.")
            if log_path := diagnostics.get("log_path"):
                lines.append(f"  Log: {encode_display_text(log_path)}")
            warnings_step = warnings_step or (
                "Inspect the full machine status for log capture details.",
                _machine_command(result, "agent", "status", "--format", "json"),
            )
        last_exit = diagnostics.get("last_exit")
        if not active and isinstance(last_exit, Mapping) and last_exit.get("reason"):
            lines.append(f"Previous agent exit: {_reason_text(last_exit['reason'])}")

    project_warnings = []
    for project in result.get("projects") or ():
        eligibility = project.get("eligibility") or {}
        if project.get("state") in {"superseded", "conflicting"} or eligibility.get("state") in {
            "invalid",
            "expired",
            "revoked",
        }:
            project_warnings.append(project)
    for project in project_warnings[:3]:
        path = project.get("shared_root") or project.get("project_id")
        state = project.get("state")
        reason = state if state in {"superseded", "conflicting"} else project["eligibility"]["state"]
        displayed_path = str(path).removesuffix("/.qexp") or "/"
        if any(
            displayed_path == item["path"]
            for item in (result.get("project_readiness") or {}).get("affected_projects", ())[:3]
        ):
            continue
        lines.extend(
            (
                f"Project registration needs attention: {encode_display_text(displayed_path)}",
                f"  {_reason_text(reason)}",
            )
        )
    if project_warnings and (path := project_warnings[0].get("shared_root")):
        readiness_step = (
            "Inspect the affected Project's local registration.",
            _machine_command(result, "--project", path, "status"),
        )

    idle_step = None
    idle_job = result.get("idle_job")
    if isinstance(idle_job, Mapping) and idle_job.get("state") in {"waiting", "blocked", "failed", "stopping"}:
        for line in _render_idle_job_status(idle_job).splitlines():
            command = line.strip().removeprefix("Inspect: ")
            if command != line.strip():
                if idle_step is None:
                    try:
                        parts = shlex.split(command)
                    except ValueError:
                        continue
                    if (
                        parts[:1] == ["qexp"]
                        and parts[-2:-1] == ["--project"]
                        and parts[-5:-2] == ["admin", "upgrade", "status"]
                    ):
                        idle_step = (
                            "Inspect the Project upgrade affecting the Idle Job.",
                            _machine_command(result, "admin", "upgrade", "status", "--project", parts[-1]),
                        )
            else:
                lines.extend(textwrap.wrap(line, width=80, break_long_words=False, break_on_hyphens=False))
    for warning in (result.get("warnings") or ())[:3]:
        message = warning.get("message") if isinstance(warning, Mapping) else warning
        if message:
            lines.extend(textwrap.wrap(str(message), width=80, break_long_words=False, break_on_hyphens=False))
    if step := control_step or recovery_step or readiness_step or publication_step or idle_step or warnings_step:
        lines.extend(_next_step(*step))
    return "\n".join(lines)


def _render_scheduler_diagnostics(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    if result.get("action") == "diagnostics_repair":
        backup = result.get("backup_path")
        counts = (
            (label, result[key] if result[key] is not None else "unknown")
            for key, label in (
                ("eligible", "Eligible"),
                ("retired", "Retired"),
                ("deferred", "Deferred"),
                ("retained_current", "Retained current"),
                ("retained_uncertain", "Retained uncertain"),
            )
        )
        return _details(
            (
                ("Action", "scheduler diagnostic repair"),
                ("Scope", "machine runtime (all Projects)"),
                ("Machine runtime", encode_display_text(result["machine_runtime_root"])),
                ("State", result["state"].replace("_", " ")),
                ("Reason", result.get("reason")),
                *counts,
                ("Pending transition", result["pending_transition"]),
                ("Backup", encode_display_text(backup) if isinstance(backup, str) else None),
                ("Backup verified", result["backup_verified"]),
                ("Apply needed", result["apply_needed"] if result["apply_needed"] is not None else "unknown"),
                ("Next command", result.get("next_action")),
            )
        )
    items = [item for item in result.get("items", []) if isinstance(item, Mapping)]
    show_disposition = result.get("action") == "diagnostics_history" and any("disposition" in item for item in items)
    headers = ("Producer", "Reason", "Project", "Scope", "Observed/resolved")
    if show_disposition:
        headers += ("Disposition",)
    rows = []
    for item in items:
        identity = item.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        row = (
            identity.get("producer"),
            identity.get("reason_code"),
            identity.get("project_id"),
            identity.get("scope_type"),
            item.get("last_observed_at", item.get("resolved_at")),
        )
        rows.append((*row, item.get("disposition")) if show_disposition else row)
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
        headers,
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


def _render_agent_readiness(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    if result.get("ready") is True:
        lines = ["Agent startup readiness verified."]
    elif result.get("timed_out"):
        lines = ["Agent startup timed out before a ready state was verified."]
    else:
        lines = ["Agent start blocked. A ready agent has not been verified."]
    lines.extend(_agent_scope_lines(result))
    if result.get("agent_state") == "active":
        lines.append("The agent is reported running.")
    else:
        lines.append("The agent has not been observed running.")

    paths = presentation.get("project_paths")
    paths = paths if isinstance(paths, Mapping) else {}
    step = ("Inspect the latest agent state and Project readiness.", _machine_command(result, "agent", "status"))
    blockers = [item for item in result["projects"] if isinstance(item, Mapping) and item.get("reason")]
    if not blockers and (reason := result.get("reason")):
        lines.append(_reason_text(reason))
    for project in blockers[:3]:
        project_id = str(project.get("project_id"))
        path = paths.get(project_id)
        lines.append(f"Project: {encode_display_text(str(path or project_id))}")
        lines.append(f"  {_reason_text(project['reason'])}")
    if blockers:
        first = blockers[0]
        path = paths.get(first.get("project_id"))
        if path and first.get("reason") == "legacy_project_requires_migration" and first.get("machine_name"):
            name = first["machine_name"]
            machine = [f"--machine={name}"] if name.startswith("-") else ["--machine", name]
            step = (
                "Migrate this Project's existing agent registration explicitly.",
                _machine_command(result, "admin", "migrate", "agent", "--project", path, *machine),
            )
        elif path and first.get("status") == "inventory_only" and first.get("reason") == "not_registered":
            step = (
                "Register the enabled Project on this runtime.",
                _machine_command(result, "project", "register", path),
            )
        elif path:
            step = (
                "Inspect the affected Project's local agent and scheduling evidence.",
                _machine_command(result, "--project", path, "status"),
            )
    elif result.get("reason") == "no_enabled_projects":
        step = (
            "Enter the intended Project directory before enrolling it on this runtime.",
            _machine_command(result, "project", "register", "."),
        )
    lines.extend(_next_step(*step))
    return "\n".join(lines)


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


def _render_idle_job(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    config = result["idle_job"]
    return _details(
        (
            ("Outcome", str(result.get("action", "shown")).replace("_", " ")),
            ("MachineRuntime root", result.get("machine_runtime_root")),
            ("Enabled", config["enabled"]),
            ("Command", config["command"] or "-"),
            ("GPU IDs", config["gpu_ids"]),
            ("Idle delay seconds", config["idle_delay_seconds"]),
            ("Stop timeout seconds", config["stop_timeout_seconds"]),
            ("Working directory", config["working_directory"] or "-"),
            ("Revision", config["revision"]),
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
        if "project_readiness" in value:
            _validate_project_readiness(value["project_readiness"])
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


def _validate_project_readiness(value: Any) -> None:
    label = "project_readiness"
    view = _mapping(value, label)
    state = _required(view, "state", label)
    if state not in {"inventory_unknown", "machine_only", "pending", "observed", "inactive"}:
        raise ValueError(f"{label}.state is invalid")
    checked = _required_int(view, "checked_count", label)
    enabled = _required(view, "enabled_count", label)
    fresh = _required(view, "fresh_count", label)
    if (
        not 0 <= checked <= 8
        or (enabled is not None and (type(enabled) is not int or enabled < checked))
        or (fresh is not None and (type(fresh) is not int or not 0 <= fresh <= checked))
    ):
        raise ValueError(f"{label} contains inconsistent counts")
    if (state == "inventory_unknown") != (enabled is None) or (state in {"inventory_unknown", "inactive"}) != (
        fresh is None
    ):
        raise ValueError(f"{label} contains counts that do not match its state")
    if state == "machine_only" and enabled != 0:
        raise ValueError(f"{label}.state machine_only requires an empty roster")
    if state == "observed" and (not enabled or fresh != enabled):
        raise ValueError(f"{label}.state observed requires a fresh observation for every enabled Project")
    affected = _required_sequence(view, "affected_projects", label)
    if len(affected) > checked:
        raise ValueError(f"{label}.affected_projects exceeds the checked count")
    for item in affected:
        project = _mapping(item, f"{label}.affected_projects item")
        if not all(isinstance(project.get(key), str) and project[key] for key in ("path", "reason")):
            raise TypeError(f"{label}.affected_projects items require a path and reason")
    reason = _required(view, "reason", label)
    if reason is not None and not isinstance(reason, str):
        raise TypeError(f"{label}.reason must be a string or null")


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
    if "project_readiness" in value:
        _validate_project_readiness(value["project_readiness"])
    if "agent_name" in value and not isinstance(value["agent_name"], str):
        raise TypeError("agent-status payload.agent_name must be a string")
    if "gpu_policy" in value:
        _validate_gpu_policy_view(value["gpu_policy"], "agent-status payload.gpu_policy")
    if "warnings" in value:
        _sequence(value["warnings"], "agent-status payload.warnings")
    if "control_plane_health" in value:
        _validate_control_plane_health(value["control_plane_health"])
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


def _validate_control_plane_health(value: Any) -> None:
    label = "agent-status payload.control_plane_health"
    health = _mapping(value, label)
    state = _required(health, "state", label)
    if state not in {"healthy", "retrying", "degraded", "stalled", "stopped", "unknown"}:
        raise ValueError(f"{label}.state is invalid")
    if "instance_id" in health and health["instance_id"] is not None and not isinstance(health["instance_id"], str):
        raise TypeError(f"{label}.instance_id must be a string or null")
    if "project_snapshot_publications" in health:
        _validate_project_snapshot_publications(health["project_snapshot_publications"], label)
    if "project_registration_renewals" in health:
        _validate_project_registration_renewals(health["project_registration_renewals"], label)
    if "snapshot_batch_error" in health:
        error = health["snapshot_batch_error"]
        if not isinstance(error, str) or _HEALTH_IDENTIFIER_RE.fullmatch(error) is None:
            raise ValueError(f"{label}.snapshot_batch_error must be a bounded exception class")
    for lane_name in ("heartbeat", "authority"):
        if lane_name not in health:
            continue
        lane = _mapping(health[lane_name], f"{label}.{lane_name}")
        if "state" in lane and lane["state"] not in {
            "healthy",
            "retrying",
            "degraded",
            "stalled",
            "stopped",
            "unknown",
        }:
            raise ValueError(f"{label}.{lane_name}.state is invalid")
        if "thread_alive" in lane and lane["thread_alive"] is not None and type(lane["thread_alive"]) is not bool:
            raise TypeError(f"{label}.{lane_name}.thread_alive must be a boolean or null")
        if (
            "diagnostic_coverage_degraded" in lane
            and lane["diagnostic_coverage_degraded"] is not None
            and type(lane["diagnostic_coverage_degraded"]) is not bool
        ):
            raise TypeError(f"{label}.{lane_name}.diagnostic_coverage_degraded must be a boolean or null")
        for key in (
            "consecutive_failures",
            "cycle_sequence",
            "missed_deadlines",
            "diagnostic_coverage_loss_count",
        ):
            if key in lane and lane[key] is not None and (type(lane[key]) is not int or lane[key] < 0):
                raise TypeError(f"{label}.{lane_name}.{key} must be a nonnegative integer or null")
        for key in ("cycle_duration_seconds", "effective_stale_after_seconds"):
            if key in lane and lane[key] is not None:
                if type(lane[key]) not in {int, float} or isinstance(lane[key], bool) or not math.isfinite(lane[key]):
                    raise TypeError(f"{label}.{lane_name}.{key} must be finite or null")
                if lane[key] < 0:
                    raise ValueError(f"{label}.{lane_name}.{key} must be nonnegative")
        for key in (
            "last_cycle_started_at",
            "last_cycle_completed_at",
            "last_success_at",
            "last_failure_at",
        ):
            if key in lane and lane[key] is not None and not isinstance(lane[key], str):
                raise TypeError(f"{label}.{lane_name}.{key} must be a string or null")
        if "last_failure" in lane and lane["last_failure"] is not None:
            failure = _mapping(lane["last_failure"], f"{label}.{lane_name}.last_failure")
            if "reason_code" in failure and not isinstance(failure["reason_code"], str):
                raise TypeError(f"{label}.{lane_name}.last_failure.reason_code must be a string")
            if "sanitized" in failure and type(failure["sanitized"]) is not bool:
                raise TypeError(f"{label}.{lane_name}.last_failure.sanitized must be a boolean")
        if "last_liveness_failure" in lane and lane["last_liveness_failure"] is not None:
            failure_label = f"{label}.{lane_name}.last_liveness_failure"
            failure = _mapping(lane["last_liveness_failure"], failure_label)
            if failure.get("state") not in {"stalled", "stopped"}:
                raise ValueError(f"{failure_label}.state is invalid")
            if failure.get("reason_code") not in {"heartbeat_cycle_stalled", "heartbeat_thread_stopped"}:
                raise ValueError(f"{failure_label}.reason_code is invalid")
            if not isinstance(failure.get("observed_at"), str):
                raise TypeError(f"{failure_label}.observed_at must be a string")
            if type(failure.get("cycle_sequence")) is not int or failure["cycle_sequence"] < 0:
                raise TypeError(f"{failure_label}.cycle_sequence must be a nonnegative integer")
            if type(failure.get("thread_alive")) is not bool:
                raise TypeError(f"{failure_label}.thread_alive must be a boolean")


def _project_health_items(value: Any, label: str) -> Sequence[Any]:
    sample = _mapping(value, label)
    for key in ("total", "omitted_count"):
        item = _required(sample, key, label)
        if type(item) is not int or item < 0:
            raise TypeError(f"{label}.{key} must be a nonnegative integer")
    items = _required_sequence(sample, "items", label)
    if len(items) > 8:
        raise ValueError(f"{label}.items must contain at most 8 entries")
    return items


def _validate_project_registration_renewals(value: Any, health_label: str) -> None:
    label = f"{health_label}.project_registration_renewals"
    for index, raw in enumerate(_project_health_items(value, label)):
        item_label = f"{label}.items[{index}]"
        item = _mapping(raw, item_label)
        for key in ("project_id", "registration_generation"):
            field = _required(item, key, item_label)
            if not isinstance(field, str) or _HEALTH_IDENTIFIER_RE.fullmatch(field) is None:
                raise ValueError(f"{item_label}.{key} must be a bounded identifier")
        for key in ("last_outcome", "blocking_reason"):
            field = _required(item, key, item_label)
            if field is not None and (not isinstance(field, str) or _HEALTH_REASON_RE.fullmatch(field) is None):
                raise ValueError(f"{item_label}.{key} must be a bounded reason code or null")
        cause = _required(item, "cause", item_label)
        if cause is not None and (not isinstance(cause, str) or _HEALTH_IDENTIFIER_RE.fullmatch(cause) is None):
            raise ValueError(f"{item_label}.cause must be a bounded exception class or null")
        retry_at = _required(item, "next_retry_at", item_label)
        if retry_at is not None:
            if not isinstance(retry_at, str) or not 0 < len(retry_at) <= 64 or not retry_at.isprintable():
                raise ValueError(f"{item_label}.next_retry_at must be a bounded timestamp or null")
            try:
                timestamp = datetime.fromisoformat(retry_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{item_label}.next_retry_at must be an ISO-8601 timestamp") from exc
            if timestamp.utcoffset() is None:
                raise ValueError(f"{item_label}.next_retry_at must include a timezone")


def _validate_project_snapshot_publications(value: Any, health_label: str) -> None:
    label = f"{health_label}.project_snapshot_publications"
    items = _project_health_items(value, label)
    for index, raw in enumerate(items):
        item_label = f"{label}.items[{index}]"
        item = _mapping(raw, item_label)
        for key in (
            "runtime_id",
            "agent_instance_id",
            "executor_epoch",
            "project_id",
            "registration_generation",
        ):
            if not isinstance(_required(item, key, item_label), str):
                raise TypeError(f"{item_label}.{key} must be a string")
        revision = _required(item, "registry_revision", item_label)
        if type(revision) is not int or revision < 0:
            raise TypeError(f"{item_label}.registry_revision must be a nonnegative integer")
        if _required(item, "operation_kind", item_label) != "machine_snapshot_publish":
            raise ValueError(f"{item_label}.operation_kind is invalid")
        request_id = _required(item, "request_id", item_label)
        if request_id is not None and not isinstance(request_id, str):
            raise TypeError(f"{item_label}.request_id must be a string or null")
        if _required(item, "stage", item_label) not in {
            "intent",
            "queued",
            "started",
            "result_ready",
            "consumed",
            "backoff",
            "published",
        }:
            raise ValueError(f"{item_label}.stage is invalid")
        for key in ("last_attempt_at", "last_published_at", "retry_at", "reason_code"):
            field = _required(item, key, item_label)
            if field is not None and not isinstance(field, str):
                raise TypeError(f"{item_label}.{key} must be a string or null")
        failures = _required(item, "consecutive_failures", item_label)
        if type(failures) is not int or failures < 0:
            raise TypeError(f"{item_label}.consecutive_failures must be a nonnegative integer")
        for key in ("worker_elapsed_seconds", "effective_stale_after_seconds"):
            field = _required(item, key, item_label)
            if field is not None and (
                type(field) not in {int, float} or isinstance(field, bool) or not math.isfinite(field) or field < 0
            ):
                raise TypeError(f"{item_label}.{key} must be finite and nonnegative or null")
        if type(_required(item, "overdue", item_label)) is not bool:
            raise TypeError(f"{item_label}.overdue must be a boolean")


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

    envelope = isolation["envelope"]
    overdue_count = isolation["overdue_worker_count"]
    exit_unverified_count = isolation["exit_unverified_worker_count"]
    unreaped_count = isolation["unreaped_worker_count"]
    if envelope != "unknown" and epoch is None:
        raise ValueError(f"{label}.executor_epoch is required for a non-unknown envelope")
    if envelope == "healthy":
        for key in ("overdue_worker_count", "exit_unverified_worker_count", "unreaped_worker_count"):
            if isolation[key] != 0:
                raise ValueError(f"{label}.envelope=healthy is inconsistent with {label}.{key}")
        if oldest is not None:
            raise ValueError(f"{label}.envelope=healthy is inconsistent with {label}.oldest_overdue_at")
        if projects:
            raise ValueError(f"{label}.envelope=healthy is inconsistent with {label}.blocking_project_ids")
    elif (overdue_count > 0) != (oldest is not None):
        raise ValueError(
            f"{label}.oldest_overdue_at and {label}.overdue_worker_count must describe the same overdue state"
        )
    if exit_unverified_count > 0 and envelope != "unknown":
        raise ValueError(f"{label}.exit_unverified_worker_count requires {label}.envelope='unknown'")
    if envelope != "unknown":
        if overdue_count > isolation["supported_hang_limit"] and envelope != "exceeded":
            raise ValueError(
                f"{label}.overdue_worker_count exceeds {label}.supported_hang_limit but envelope is not 'exceeded'"
            )
        if envelope == "exceeded" and overdue_count <= isolation["supported_hang_limit"]:
            raise ValueError(
                f"{label}.envelope='exceeded' requires {label}.overdue_worker_count to exceed "
                f"{label}.supported_hang_limit"
            )
    occupied_slots = isolation["capacity"] - isolation["free_slot_count"]
    if isolation["active_worker_count"] + exit_unverified_count > occupied_slots:
        raise ValueError(
            f"{label}.active_worker_count and {label}.exit_unverified_worker_count exceed occupied slots "
            f"derived from {label}.free_slot_count"
        )
    if overdue_count > occupied_slots:
        raise ValueError(f"{label}.overdue_worker_count exceeds occupied slots derived from {label}.free_slot_count")
    if unreaped_count > occupied_slots:
        raise ValueError(f"{label}.unreaped_worker_count exceeds occupied slots derived from {label}.free_slot_count")


def _validate_scheduler_diagnostics(result: Any) -> None:
    value = _mapping(result, "scheduler-diagnostics payload")
    label = "scheduler-diagnostics payload"
    action = _required(value, "action", label)
    if action == "diagnostics_repair":
        if _required_int(value, "schema_version", label) != 1 or _required(value, "scope", label) != "machine_runtime":
            raise ValueError(f"{label} repair schema or scope is invalid")
        if not isinstance(_required(value, "machine_runtime_root", label), str):
            raise TypeError(f"{label}.machine_runtime_root must be a string")
        if _required(value, "state", label) not in {
            "pending",
            "progressing",
            "awaiting_observation",
            "blocked",
            "idle",
        }:
            raise ValueError(f"{label}.state is invalid")
        for key in ("reason", "backup_path", "next_action"):
            item = _required(value, key, label)
            if item is not None and not isinstance(item, str):
                raise TypeError(f"{label}.{key} must be a string or null")
        for key in ("eligible", "retired", "deferred", "retained_current", "retained_uncertain"):
            item = _required(value, key, label)
            if item is not None and (type(item) is not int or item < 0):
                raise ValueError(f"{label}.{key} must be a nonnegative integer or null")
        for key in ("pending_transition", "backup_verified"):
            _required_bool(value, key, label)
        apply_needed = _required(value, "apply_needed", label)
        if apply_needed is not None and type(apply_needed) is not bool:
            raise TypeError(f"{label}.apply_needed must be a boolean or null")
        return
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


def _validate_idle_job(result: Any) -> None:
    value = _mapping(result, "idle-job payload")
    action = _required(value, "action", "idle-job payload")
    if action not in {"shown", "updated"}:
        raise ValueError("idle-job payload.action is invalid")
    _required(value, "machine_runtime_root", "idle-job payload")
    config = _required_mapping(value, "idle_job", "idle-job payload")
    expected_keys = {
        "version",
        "enabled",
        "command",
        "gpu_ids",
        "idle_delay_seconds",
        "stop_timeout_seconds",
        "working_directory",
        "revision",
    }
    if set(config) != expected_keys:
        raise ValueError("idle-job payload.idle_job has missing or unknown fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("idle-job payload.idle_job.version is unsupported")
    if type(config["enabled"]) is not bool:
        raise TypeError("idle-job payload.idle_job.enabled must be a boolean")
    if not isinstance(config["command"], str):
        raise TypeError("idle-job payload.idle_job.command must be a string")
    gpu_ids = config["gpu_ids"]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise TypeError("idle-job payload.idle_job.gpu_ids must be a non-empty list")
    if any(type(gpu_id) is not int or gpu_id < -1 for gpu_id in gpu_ids):
        raise TypeError("idle-job payload.idle_job.gpu_ids must contain -1 or nonnegative integers")
    for key, is_positive in (("idle_delay_seconds", False), ("stop_timeout_seconds", True)):
        number = config[key]
        try:
            is_finite = math.isfinite(number)
        except (OverflowError, TypeError):
            is_finite = False
        if type(number) not in {int, float} or isinstance(number, bool) or not is_finite:
            raise TypeError(f"idle-job payload.idle_job.{key} must be finite")
        if (is_positive and number <= 0) or (not is_positive and number < 0):
            raise ValueError(f"idle-job payload.idle_job.{key} has an invalid value")
        if key == "stop_timeout_seconds" and number > MAX_IDLE_JOB_STOP_TIMEOUT_SECONDS:
            raise ValueError(f"idle-job payload.idle_job.{key} has an invalid value")
    working_directory = config["working_directory"]
    if working_directory is not None and not isinstance(working_directory, str):
        raise TypeError("idle-job payload.idle_job.working_directory must be a string or null")
    revision = config["revision"]
    if type(revision) is not int or revision < 0:
        raise TypeError("idle-job payload.idle_job.revision must be a nonnegative integer")


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
    OutputKind.IDLE_JOB: OutputContract(_validate_idle_job, _render_idle_job),
    OutputKind.GPU_POLICY: OutputContract(_validate_gpu_policy_payload, _render_gpu_policy),
}

__all__ = ["CONTRACTS"]
