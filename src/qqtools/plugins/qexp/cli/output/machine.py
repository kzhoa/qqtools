"""Pure machine-observation output contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...takeover_contract import TAKEOVER_PHASES, TAKEOVER_PLAN_ID
from ..project_presentation import encode_display_text
from .core import OutputContract, OutputKind
from .primitives import (
    _details,
    _mapping,
    _required,
    _required_bool,
    _required_mapping,
    _section,
    _sequence,
    _table,
    _task_summary,
)

_STATUS_FINDING_SAMPLE_LIMIT = 16
_STATUS_WARNING_LIMIT = 16
_TAKEOVER_DRAIN_TEXT = {
    "waiting_for_work": "Running or reserved work is draining.",
    "drain_pending": "Local writer or Attempt obligations remain.",
    "drain_unavailable": "Drain evidence cannot be verified.",
    "drained": "Drain evidence verified.",
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
    return " — ".join(fields)


def _render_status_takeover(takeover: Mapping[str, Any]) -> str:
    departing = takeover["departing_machine"]
    summary = _section(
        "Project takeover",
        (
            ("Plan ID", takeover["plan_id"]),
            ("Phase", takeover["phase"]),
            ("Departing machine", encode_display_text(departing) if departing is not None else "none"),
            ("Target machine", encode_display_text(takeover["target_machine"])),
            ("Cancellation requested", takeover["is_cancel_requested"]),
            ("Drain observation", _TAKEOVER_DRAIN_TEXT.get(takeover["drain_observation"])),
            ("Current issue", takeover["operation_issue"]),
        ),
    )
    # Keep shell quoting intact for paths with embedded line breaks.
    commands = _details(
        (
            ("  Operation status", takeover["operation_status_command"]),
            ("  Request cancellation", takeover["cancel_command"]),
        )
    )
    return f"{summary}\n{commands}"


def _render_status(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    project = result.get("project", {})
    participation = result.get("local_participation", {})
    agent = result.get("local_agent", {})
    observation = result.get("task_observation", {})
    participation_state = participation.get("state") if isinstance(participation, Mapping) else participation
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
    enrollment = {
        "registered": "registered",
        "inventory_only": "inventory entry without a local binding",
        "unregistered": "not registered",
        "unavailable": "unknown (check the local registry and inventory)",
    }.get(participation_state, participation_state)
    lines = [project_header, f"Local enrollment: {enrollment}"]
    if participation_state == "registered" and isinstance(participation, Mapping) and "binding" in participation:
        binding = participation["binding"]
        if binding is None:
            lines.append("Local binding: unverified (local identity and shared registration could not be matched)")
        else:
            name = encode_display_text(binding["machine_name"])
            details = []
            if not binding["enabled"]:
                details.append("disabled")
            if binding["write_eligibility"] == "expired":
                details.append("write eligibility expired")
            lines.append(f"Local binding: {name}" + (f" ({', '.join(details)})" if details else ""))
    agent_reason = _status_reason(agent.get("reason")) if isinstance(agent, Mapping) else None
    lines.append(f"Local agent: {agent_state}" + (f" ({agent_reason})" if agent_reason else ""))
    if isinstance(agent, Mapping) and agent.get("observed_at"):
        lines.append(f"Agent observed: {agent['observed_at']}")
    if configured_mode != observed_mode:
        if configured_mode:
            lines.append(f"Configured agent mode: {configured_mode}")
        if observed_mode:
            lines.append(f"Observed agent mode: {observed_mode}")
    if task_state != "active" or task_reason:
        reason = _status_reason(task_reason)
        lines.append(f"Task observation index: {task_state}" + (f" ({reason})" if reason else ""))
    sections = ["\n".join(lines)]
    if "takeover" in result:
        sections.append(_render_status_takeover(result["takeover"]))

    diagnostic_fields: list[tuple[str, Any]] = []
    scheduler_material = scheduler is not None and (
        scheduler.get("status") != "available"
        or scheduler.get("coverage") != "complete"
        or scheduler.get("reason")
        or scheduler.get("active_count") not in (None, 0)
        or scheduler.get("active_findings")
        or scheduler.get("truncated")
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

    actions = result["next_actions"]
    if actions:
        # Registration is the first canonical action for an unbound Project.
        labels = ["Run"] * len(actions)
        if participation_state in {"unregistered", "inventory_only"}:
            labels[0] = "To run Tasks on this machine"
        sections.append(_section("Next actions", tuple(zip(labels, actions, strict=True))))
    sections.append(_details((("Project takeover preview help", result.get("takeover_preview_help")),)))
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


def _validate_status_takeover(takeover: Mapping[str, Any], label: str) -> None:
    plan_id = _required(takeover, "plan_id", label)
    if not isinstance(plan_id, str) or not TAKEOVER_PLAN_ID.fullmatch(plan_id):
        raise ValueError(f"{label}.plan_id must be 64 lowercase hexadecimal characters")
    phase = _required(takeover, "phase", label)
    if not isinstance(phase, str) or phase not in TAKEOVER_PHASES:
        raise ValueError(f"{label}.phase is invalid")
    for key in ("target_machine", "operation_status_command"):
        value = _required(takeover, key, label)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}.{key} must be a non-empty string")
    for key in ("departing_machine", "cancel_command", "operation_issue"):
        value = _required(takeover, key, label)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{label}.{key} must be a non-empty string or null")
    drain = _required(takeover, "drain_observation", label)
    if drain is not None and (not isinstance(drain, str) or drain not in _TAKEOVER_DRAIN_TEXT):
        raise ValueError(f"{label}.drain_observation is invalid")
    _required_bool(takeover, "is_cancel_requested", label)


def _validate_status(result: Any) -> None:
    value = _mapping(result, "status payload")
    for key in ("project", "local_participation", "local_agent", "task_observation", "next_actions"):
        _required(value, key, "status payload")
    participation = value["local_participation"]
    if isinstance(participation, Mapping):
        reason = participation.get("binding_reason")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError("status payload.local_participation.binding_reason must be a non-empty string or null")
        if participation.get("binding") is not None:
            label = "status payload.local_participation.binding"
            binding = _mapping(participation["binding"], label)
            for key in ("machine_name", "registration_generation"):
                text = _required(binding, key, label)
                if not isinstance(text, str) or not text:
                    raise ValueError(f"{label}.{key} must be a non-empty string")
            _required_bool(binding, "enabled", label)
            if _required(binding, "write_eligibility", label) not in ("eligible", "expired"):
                raise ValueError(f"{label}.write_eligibility must be eligible or expired")
    if "scheduler_diagnostics" in value:
        _mapping(value["scheduler_diagnostics"], "status payload.scheduler_diagnostics")
    if "takeover" in value:
        _validate_status_takeover(_required_mapping(value, "takeover", "status payload"), "status payload.takeover")
    if "takeover_preview_help" in value and (
        not isinstance(value["takeover_preview_help"], str) or not value["takeover_preview_help"]
    ):
        raise ValueError("status payload.takeover_preview_help must be a non-empty string")


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
