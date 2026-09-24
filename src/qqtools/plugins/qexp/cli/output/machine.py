"""Pure machine-observation output contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import _details, _mapping, _required, _required_mapping, _sequence, _table, _task_summary


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
    return (
        machine,
        state.get("freshness"),
        agent.get("observed_state", agent.get("agent_state")),
        gpu.get("free_gpu_ids", gpu.get("unreserved")),
        gpu.get("reserved_gpu_ids", gpu.get("reserved")),
        policy.get("draining_gpu_ids"),
        task_summary,
        item.get("reason") or (policy.get("warnings") if policy.get("warnings") else None),
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
    actions = result.get("next_actions") or []
    project_line = presentation.get("project_line")
    is_implicit_presentation = presentation.get("implicit_project") is True and isinstance(project_line, str)
    project_path = (
        project_line.removeprefix("Project: ")
        if is_implicit_presentation
        else (project.get("path") if isinstance(project, Mapping) else project)
    )
    sections = [
        (
            ("Outcome", result.get("status")),
            ("Project", project_path),
            ("Project ID", project.get("project_id") if isinstance(project, Mapping) else None),
            (
                "Selection source",
                None
                if is_implicit_presentation
                else project.get("selection_source")
                if isinstance(project, Mapping)
                else None,
            ),
        ),
        (
            ("Participation", participation_state),
            ("Participation reason", participation_reason),
            ("Agent", agent_state),
            ("Configured agent mode", configured_mode),
            ("Observed agent mode", observed_mode if observed_mode != configured_mode else None),
            ("Task observation", task_state),
            ("Task observation reason", task_reason),
        ),
    ]
    if actions and (participation_state not in {"registered", "complete"} or task_state not in {"ready", "available"}):
        sections.append((("Next actions", actions),))
    warnings = result.get("warnings") or []
    if warnings:
        sections.append((("Warnings", warnings),))
    return _details(*sections)


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
