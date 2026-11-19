"""Pure machine-observation output contracts."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import _details, _mapping, _required, _required_mapping, _section, _sequence, _table, _task_summary

_STATUS_FINDING_SAMPLE_LIMIT = 16
_STATUS_WARNING_LIMIT = 16
_SCHEDULER_REASON_COMMANDS = {
    "termination_exit_code_invalid": "qexp agent diagnostics active",
    "termination_exit_observation_mismatch": "qexp agent diagnostics active",
    "termination_process_identity_present": "qexp agent diagnostics active",
    "termination_reservation_release_pending": "qexp agent diagnostics active",
    "termination_reservation_conflict": "qexp agent diagnostics active",
    "project_snapshot_overdue": "qexp agent diagnostics active",
    "heartbeat_diagnostic_coverage_loss": "qexp agent diagnostics active",
    "registry_enablement_unknown": "qexp agent diagnostics active",
    "inventory_mirror_incomplete": "qexp agent diagnostics active",
    "enablement_mirror_diverged": "qexp agent diagnostics active",
}


def _machine_values(item: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]] = ()) -> tuple[Any, ...]:
    machine = item.get("machine", {}).get("machine_name")
    state = item.get("state", {})
    agent_record = state.get("agent", {})
    agent = agent_record.get("agent", agent_record)
    gpu_record = state.get("gpu", {})
    gpu = gpu_record.get("gpu", gpu_record)
    policy = gpu.get("gpu_policy", {})
    task_summary = item.get("task_summary")
    if task_summary is None and tasks:
        task_summary = _task_summary(tasks, machine=machine)
    freshness = state.get("freshness")
    gpu_is_current = freshness not in {"stale", "stopped", "missing"}
    reason = item.get("reason") or (policy.get("warnings") if policy.get("warnings") else None)
    if reason is None and freshness in {"stale", "stopped", "missing"}:
        reason = "resource observation is not current"
    return (
        machine,
        freshness,
        agent.get("observed_state", agent.get("agent_state")),
        gpu.get("free_gpu_ids", gpu.get("unreserved")) if gpu_is_current else None,
        gpu.get("reserved_gpu_ids", gpu.get("reserved")) if gpu_is_current else None,
        policy.get("draining_gpu_ids") if gpu_is_current else None,
        task_summary,
        reason,
    )


def _render_machines(result: Any, _presentation: Mapping[str, object]) -> str:
    machines = result["machines"] if isinstance(result, Mapping) else result
    return _table(
        (
            "Machine",
            "Availability",
            "Agent state",
            "GPU free",
            "GPU reserved",
            "Draining",
            "Task summary",
            "Reason",
        ),
        [_machine_values(item) for item in machines],
        empty_message="No Machines.",
    )


def _validate_machine(result: Any, label: str) -> None:
    value = _mapping(result, label)
    machine = _required_mapping(value, "machine", label)
    _required(machine, "machine_name", f"{label}.machine")
    _required_mapping(value, "state", label)


def _validate_machines(result: Any) -> None:
    machines = _sequence(result, "machines payload")
    for index, item in enumerate(machines):
        _validate_machine(item, f"machines payload[{index}]")


def _status_reason(value: Any) -> str | None:
    """Format one stable status reason without rendering arbitrary payload data."""
    if isinstance(value, str) and value:
        if value == "record_missing":
            return "no status record"
        if value == "record_unreadable":
            return "status record unreadable"
        return value.replace("_", " ")
    return None


def _status_action(actions: Sequence[Any], prefix: str, fallback: str) -> str:
    for action in actions[:32]:
        if isinstance(action, str) and action.startswith(prefix):
            return action
    return fallback


def _scheduler_command(reason: str | None, project_id: Any = None) -> str:
    command = _SCHEDULER_REASON_COMMANDS.get(reason or "", "qexp agent diagnostics active")
    if not isinstance(reason, str):
        return command
    command_parts = shlex.split(command)
    if isinstance(project_id, str) and project_id:
        command_parts.extend(("--project-id", project_id))
    command_parts.extend(("--reason", reason))
    return shlex.join(command_parts)


def _scheduler_scope(identity: Mapping[str, Any]) -> str | None:
    scope: list[str] = []
    project_id = identity.get("project_id")
    attempt_id = identity.get("attempt_id")
    if isinstance(project_id, str) and project_id:
        scope.append(f"Project {project_id}")
    if isinstance(attempt_id, str) and attempt_id:
        scope.append(f"Attempt {attempt_id}")
    scope_type = identity.get("scope_type")
    if not scope and isinstance(scope_type, str) and scope_type:
        scope.append(f"scope {scope_type}")
    return "; ".join(scope) if scope else None


def _scheduler_finding_text(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    identity = item.get("identity")
    if not isinstance(identity, Mapping):
        return None
    reason_code = identity.get("reason_code")
    reason = _status_reason(reason_code) if isinstance(reason_code, str) else None
    if reason is None:
        return None
    fields = [reason]
    scope = _scheduler_scope(identity)
    if scope:
        fields.append(scope)
    observed_at = item.get("last_observed_at") or item.get("observed_at") or item.get("first_observed_at")
    if isinstance(observed_at, str) and observed_at:
        fields.append(f"observed {observed_at}")
    fields.append(_scheduler_command(reason_code, identity.get("project_id")))
    return " — ".join(fields)


def _status_actions(
    result: Mapping[str, Any],
    *,
    participation_state: Any,
    agent_state: Any,
    configured_mode: Any,
    observed_mode: Any,
    task_state: Any,
    task_reason: Any,
    warnings: Sequence[Any],
    scheduler: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    scheduler = scheduler or {}
    active_findings = scheduler.get("active_findings")
    active_count = scheduler.get("active_count")
    has_findings = (type(active_count) is int and active_count > 0) or (
        isinstance(active_findings, Sequence)
        and not isinstance(active_findings, (str, bytes))
        and bool(active_findings)
    )
    coverage_incomplete = scheduler.get("coverage") == "incomplete"
    participation_issue = participation_state not in {"registered", "complete"}
    task_issue = task_state in {"absent", "building", "degraded", "dirty", "unavailable"} or (
        isinstance(task_reason, str)
        and any(marker in task_reason for marker in ("building", "degraded", "dirty", "unavailable"))
    )
    agent_issue = agent_state in {"unavailable", "stopped"}
    mode_mismatch = configured_mode is not None and observed_mode is not None and configured_mode != observed_mode
    warning_issue = bool(warnings)
    needs_actions = any(
        (participation_issue, task_issue, agent_issue, mode_mismatch, coverage_incomplete, warning_issue, has_findings)
    )
    if not needs_actions:
        return []

    raw_actions = result.get("next_actions")
    actions = raw_actions if isinstance(raw_actions, Sequence) and not isinstance(raw_actions, (str, bytes)) else ()
    project = result.get("project")
    project_path = project.get("path") if isinstance(project, Mapping) else "PATH"
    output: list[tuple[str, str]] = []
    if participation_issue:
        if participation_state in {"unregistered", "inventory_only"}:
            command = _status_action(
                actions, "qexp project register", shlex.join(("qexp", "project", "register", str(project_path)))
            )
            output.append(("To run Tasks on this machine", command))
        else:
            command = _status_action(actions, "qexp init", "qexp init")
            output.append(("Restore local enrollment", command))
    if task_issue:
        output.append(("Inspect Task observation", _status_action(actions, "qexp task list", "qexp task list")))
    if agent_issue or mode_mismatch:
        output.append(("Inspect local agent", _status_action(actions, "qexp agent status", "qexp agent status")))
    if coverage_incomplete:
        output.append(("Inspect scheduling diagnostics", "qexp agent diagnostics active"))
    if has_findings:
        seen_reasons: set[str] = set()
        if isinstance(active_findings, Sequence) and not isinstance(active_findings, (str, bytes)):
            for item in active_findings[:_STATUS_FINDING_SAMPLE_LIMIT]:
                if not isinstance(item, Mapping):
                    continue
                identity = item.get("identity")
                reason = identity.get("reason_code") if isinstance(identity, Mapping) else None
                if not isinstance(reason, str) or reason in seen_reasons:
                    continue
                seen_reasons.add(reason)
                project_id = identity.get("project_id") if isinstance(identity, Mapping) else None
                output.append((f"Inspect finding ({_status_reason(reason)})", _scheduler_command(reason, project_id)))
        if not seen_reasons:
            output.append(("Inspect active findings", "qexp agent diagnostics active"))
    if warning_issue and not (agent_issue or mode_mismatch):
        output.append(("Inspect local agent", _status_action(actions, "qexp agent status", "qexp agent status")))
    return output


def _render_status(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    project = result.get("project", {})
    participation = result.get("local_participation", {})
    agent = result.get("local_agent", {})
    observation = result.get("task_observation", {})
    participation_state = participation.get("state") if isinstance(participation, Mapping) else participation
    participation_reason = participation.get("reason") if isinstance(participation, Mapping) else None
    agent_state = agent.get("state") if isinstance(agent, Mapping) else agent
    configured_mode = agent.get("configured_mode") if isinstance(agent, Mapping) else None
    observed_mode = agent.get("observed_mode") if isinstance(agent, Mapping) else None
    task_state = observation.get("state") if isinstance(observation, Mapping) else observation
    task_reason = observation.get("reason") if isinstance(observation, Mapping) else None
    warnings = result.get("warnings")
    warnings = warnings if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) else ()
    scheduler = result.get("scheduler_diagnostics")
    scheduler = scheduler if isinstance(scheduler, Mapping) else None

    project_line = presentation.get("project_line")
    implicit = presentation.get("implicit_project") is True and isinstance(project_line, str)
    if implicit:
        project_header = project_line
    else:
        project_path = project.get("path") if isinstance(project, Mapping) else project
        project_header = f"Project: {project_path}"
    project_fields = (
        ("Project ID", project.get("project_id") if isinstance(project, Mapping) else None),
        (
            "Selection source",
            None if implicit else project.get("selection_source") if isinstance(project, Mapping) else None,
        ),
    )
    sections = [project_header]
    project_details = _section("Project details", project_fields)
    if project_details:
        sections.append(project_details)
    enrollment_summary = {
        "registered": "registered",
        "inventory_only": "known in local inventory but not registered",
        "unregistered": "not registered with the Project",
        "unavailable": "unavailable",
    }.get(participation_state, participation_state)
    sections.append(
        _section(
            "Local enrollment",
            (("Status", enrollment_summary), ("Reason", _status_reason(participation_reason))),
        )
    )
    observations = _section(
        "Local observations",
        (
            ("Local agent", agent_state),
            ("Agent reason", _status_reason(agent.get("reason") if isinstance(agent, Mapping) else None)),
            (
                "Local agent observed",
                agent.get("observed_at") if isinstance(agent, Mapping) else None,
            ),
            ("Configured agent mode", configured_mode if observed_mode != configured_mode else None),
            ("Observed agent mode", observed_mode if observed_mode != configured_mode else None),
            ("Task observation", task_state),
            ("Task observation reason", _status_reason(task_reason)),
        ),
    )
    if observations:
        sections.append(observations)

    diagnostic_fields: list[tuple[str, Any]] = []
    scheduler_material = scheduler is not None and (
        scheduler.get("status") != "available"
        or scheduler.get("coverage") != "complete"
        or scheduler.get("reason")
        or scheduler.get("active_count") not in (None, 0)
        or scheduler.get("active_findings")
    )
    if scheduler_material and scheduler is not None:
        diagnostic_fields.extend(
            (
                ("State", scheduler.get("status")),
                ("Coverage", scheduler.get("coverage")),
                ("Observed", scheduler.get("observed_at")),
                ("Reason", _status_reason(scheduler.get("reason"))),
            )
        )
        active_count = scheduler.get("active_count")
        samples = scheduler.get("active_findings")
        bounded_samples = (
            samples[:_STATUS_FINDING_SAMPLE_LIMIT]
            if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes))
            else ()
        )
        if scheduler.get("status") != "unavailable" or active_count is not None:
            diagnostic_fields.append(("Active findings", active_count))
            diagnostic_fields.append(("Displayed findings", len(bounded_samples)))
        for index, item in enumerate(bounded_samples, 1):
            finding = _scheduler_finding_text(item)
            if finding:
                diagnostic_fields.append((f"Finding {index}", finding))
        if scheduler.get("truncated") is True:
            diagnostic_fields.append(("Samples truncated", "yes"))
    if diagnostic_fields:
        sections.append(_section("Scheduling diagnostics", tuple(diagnostic_fields)))
    warning_rows = tuple(
        ("Warning", warning) for warning in warnings[:_STATUS_WARNING_LIMIT] if isinstance(warning, str)
    )
    if warning_rows:
        sections.append(_section("Warnings", warning_rows))

    actions = _status_actions(
        result,
        participation_state=participation_state,
        agent_state=agent_state,
        configured_mode=configured_mode,
        observed_mode=observed_mode,
        task_state=task_state,
        task_reason=task_reason,
        warnings=warnings,
        scheduler=scheduler,
    )
    if actions:
        sections.append(_section("Next actions", tuple(actions)))
    return "\n\n".join(section for section in sections if section)


def _render_machine_show(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    declaration = result.get("declaration")
    agent = result.get("agent")
    gpu = result.get("gpu")
    summary = result.get("summary")

    def section(label: str, value: Any, keys: Sequence[str], available: str) -> tuple[tuple[str, Any], ...]:
        if not isinstance(value, Mapping) or not value:
            return ((label, "unavailable"),)
        return (
            (label, available),
            *((key.replace("_", " ").capitalize(), value.get(key)) for key in keys),
        )

    return _details(
        (
            ("Machine", result.get("machine_name")),
            ("Project", result.get("project")),
            ("Complete", result.get("complete")),
        ),
        section("Declaration", declaration, ("machine_name", "agent_mode", "runtime_id"), "declared"),
        section("Agent", agent, ("observed_state", "heartbeat_at", "heartbeat_interval_seconds"), "observed"),
        section("GPU", gpu, ("visible_gpu_ids", "reserved_gpu_ids", "free_gpu_ids"), "observed"),
        section("Summary", summary, ("agent_state", "task_count", "machine_reservations"), "observed"),
        (("Warnings", result.get("warnings")),),
    )


def _validate_status(result: Any) -> None:
    value = _mapping(result, "status payload")
    for key in ("project", "local_participation", "local_agent", "task_observation", "next_actions"):
        _required(value, key, "status payload")
    if "scheduler_diagnostics" in value:
        _mapping(value["scheduler_diagnostics"], "status payload.scheduler_diagnostics")


def _validate_machine_show(result: Any) -> None:
    value = _mapping(result, "machine-show payload")
    for key in ("machine_name", "project", "declaration", "warnings", "complete"):
        _required(value, key, "machine-show payload")


def _validate_identity_diagnosis(result: Any) -> None:
    value = _mapping(result, "machine identity diagnosis payload")
    for key in (
        "outcome",
        "reason",
        "runtime_root",
        "selection_source",
        "affected_file",
        "evidence_sources",
        "checks",
        "planned_changes",
        "next_action",
        "configuration_scope",
    ):
        _required(value, key, "machine identity diagnosis payload")
    if value["outcome"] not in {"healthy", "blocked", "failed"}:
        raise ValueError("machine identity diagnosis outcome must be healthy, blocked, or failed.")
    if not isinstance(value["evidence_sources"], list) or not all(
        isinstance(source, str) for source in value["evidence_sources"]
    ):
        raise TypeError("machine identity diagnosis evidence_sources must be a list of strings.")
    if not isinstance(value["planned_changes"], list):
        raise TypeError("machine identity diagnosis planned_changes must be a list.")
    for index, item in enumerate(_sequence(value["checks"], "machine identity diagnosis checks")):
        check = _mapping(item, f"machine identity diagnosis checks[{index}]")
        for key in ("name", "status", "detail"):
            _required(check, key, f"machine identity diagnosis checks[{index}]")


def _render_identity_diagnosis(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    checks = [f"{item['name']}: {item['status']} — {item['detail']}" for item in result["checks"]]
    return _details(
        (
            ("Outcome", result["outcome"]),
            ("Reason", result["reason"]),
            ("Runtime root", result["runtime_root"]),
            ("Selection source", result["selection_source"]),
            ("Affected file", result["affected_file"]),
            ("Replacement phase", result.get("replacement_phase")),
            ("Replacement target", result.get("replacement_target")),
            ("Configuration", "not checked"),
        ),
        (("Evidence sources", result["evidence_sources"]),),
        (("Checks", checks),),
        (("Planned changes", result["planned_changes"] or "none"),),
        (("Next action", result["next_action"]),),
    )


CONTRACTS = {
    OutputKind.MACHINES: OutputContract(_validate_machines, _render_machines),
    OutputKind.STATUS: OutputContract(_validate_status, _render_status),
    OutputKind.MACHINE_SHOW: OutputContract(_validate_machine_show, _render_machine_show),
    OutputKind.MACHINE_IDENTITY_DIAGNOSIS: OutputContract(_validate_identity_diagnosis, _render_identity_diagnosis),
}

__all__ = ["CONTRACTS"]
