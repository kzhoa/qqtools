"""Pure Task-family output contracts."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from ...progress_format import format_progress_compact, format_progress_details, select_progress_observation
from .core import OutputContract, OutputKind
from .primitives import (
    _details,
    _mapping,
    _operation,
    _queue_summary,
    _required,
    _required_bool,
    _required_mapping,
    _required_sequence,
    _sequence,
    _table,
    _task_summary,
    _value,
)


def _task_rows(tasks: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    include_queue = any(item.get("queue_scope") not in {None, "", "home"} for item in tasks)
    include_machine = any(item.get("claim_machine") not in {None, ""} for item in tasks)
    include_dependency = any(
        item.get("dependency_state") not in {None, "", "ready"} or item.get("depends_on_task_ids") for item in tasks
    )
    include_reason = any(item.get("reason") not in {None, ""} for item in tasks)
    rows: list[tuple[Any, ...]] = []
    for item in tasks:
        row: list[Any] = [
            item.get("task_id"),
            item.get("name"),
            item.get("phase"),
            item.get("gpus"),
            item.get("group"),
            item.get("home_machine"),
        ]
        if include_queue:
            row.append(item.get("queue_scope"))
        if include_machine:
            row.append(item.get("claim_machine"))
        if include_dependency:
            dependency_state = item.get("dependency_state")
            dependencies = item.get("depends_on_task_ids") or []
            row.append(dependency_state if not dependencies else f"{dependency_state}: {dependencies}")
        if include_reason:
            row.append(item.get("reason"))
        rows.append(tuple(row))
    return rows


def _render_task_table(tasks: Sequence[Mapping[str, Any]]) -> str:
    include_queue = any(item.get("queue_scope") not in {None, "", "home"} for item in tasks)
    include_machine = any(item.get("claim_machine") not in {None, ""} for item in tasks)
    include_dependency = any(
        item.get("dependency_state") not in {None, "", "ready"} or item.get("depends_on_task_ids") for item in tasks
    )
    include_reason = any(item.get("reason") not in {None, ""} for item in tasks)
    headers = ["Task ID", "Name", "State", "GPUs", "Group", "Home"]
    if include_queue:
        headers.append("Queue")
    if include_machine:
        headers.append("Machine")
    if include_dependency:
        headers.append("Dependency")
    if include_reason:
        headers.append("Reason")
    return _table(headers, _task_rows(tasks), empty_message="No Tasks.")


def _render_task_list(result: Any, _presentation: Mapping[str, object]) -> str:
    return _render_task_table(result)


def _render_task_page(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    if not result["items"] and result.get("next_cursor") is not None:
        rendered = "No matches in this page; more candidates remain."
    else:
        rendered = _render_task_table(result["items"])
    stop_reason = result["stop_reason"]
    lines = [rendered]
    if result.get("next_cursor") is not None or stop_reason not in {"exhausted", "complete", "completed"}:
        lines.append(f"Stop reason: {stop_reason}")
    elif not result["items"]:
        lines.append("End of results.")
    command = presentation.get("continuation_command")
    if command is not None:
        lines.append(f"Continue with: {_value(command)}")
    return "\n".join(lines)


def _task_progress(result: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, int | None]:
    """Resolve the whole progress candidate shared by one-shot and watch output."""
    progress = result.get("progress")
    progress_extended = result.get("progress_extended")
    selected_version = result.get("selected_progress_version")
    if "selected_progress_version" not in result:
        selected_version = 1 if isinstance(progress, Mapping) and progress.get("status") == "available" else None
    if type(selected_version) is not int:
        selected_version = None
    return (
        select_progress_observation(
            progress if isinstance(progress, Mapping) else None,
            progress_extended if isinstance(progress_extended, Mapping) else None,
            selected_version,
        ),
        selected_version,
    )


def _render_task_show(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    task = result.get("task", result)
    spec = task.get("spec", {})
    placement = task.get("placement_policy", {})
    runtime = task.get("placement_runtime", {})
    state = task.get("state", {})
    control = task.get("control", {})
    attempts = result.get("attempts", ())
    submission = result.get("submission", {})
    execution_machines = [
        f"#{attempt.get('attempt', {}).get('attempt_number')}:{attempt.get('attempt', {}).get('machine_name')}"
        for attempt in attempts
    ]
    command = spec.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        command = shlex.join(str(item) for item in command)
    attempt_records = [
        item.get("attempt", {})
        for item in attempts
        if isinstance(item, Mapping) and isinstance(item.get("attempt"), Mapping)
    ]
    current_attempt_id = task.get("attempt_control", {}).get("current_attempt_id")
    current_attempt = next(
        (attempt for attempt in attempt_records if attempt.get("attempt_id") == current_attempt_id),
        None,
    )
    if current_attempt is None:
        current_attempt = max(
            attempt_records,
            key=lambda attempt: attempt.get("attempt_number") if type(attempt.get("attempt_number")) is int else -1,
            default={},
        )
    terminal = state.get("projection") in {"succeeded", "failed", "cancelled"}
    progress_observation, progress_version = _task_progress(result)
    if presentation.get("details"):
        progress_fields = format_progress_details(
            progress_observation,
            progress_version=progress_version,
            include_metrics=True,
        )
    else:
        progress_fields = format_progress_compact(
            progress_observation,
            progress_version=progress_version,
        )
    return _details(
        (
            ("Task ID", task.get("task_id")),
            ("Name", task.get("name")),
            ("Command", command),
            ("GPUs", spec.get("requested_gpus")),
            ("Group", task.get("group_name")),
            ("Dependencies", task.get("depends_on_task_ids")),
            ("TMUX observer override", result.get("observation", {}).get("tmux_override")),
        ),
        (
            ("State", state.get("projection")),
            ("Original submitting machine", submission.get("original_submitting_machine")),
            ("Home machine", placement.get("home_machine")),
            ("Current Attempt", current_attempt.get("attempt_number") if current_attempt else None),
            ("Attempt phase", current_attempt.get("phase") if current_attempt else None),
            ("Execution machines", execution_machines if terminal else None),
            ("Queue scope", runtime.get("queue_scope")),
            ("Control", control.get("cancellation_operation_id")),
            ("Attempts", len(attempts)),
            ("Reason", state.get("reason")),
            ("Dependency gate", result.get("dependency_gate")),
            ("Exit code", current_attempt.get("exit_code") if terminal else None),
        ),
        progress_fields,
    )


def _render_task_watch(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    """Render one compact current Task frame without performing reads."""
    attempt = result.get("selected_attempt")
    attempt_values = (
        (
            ("Attempt ID", attempt.get("attempt_id")),
            ("Attempt number", attempt.get("attempt_number")),
            ("Attempt phase", attempt.get("phase")),
            ("Machine", attempt.get("machine_name")),
        )
        if isinstance(attempt, Mapping)
        else (("Attempt", "none"),)
    )
    observation = result.get("observation_state")
    observation_reason = result.get("observation_reason")
    if observation_reason is not None:
        observation = f"{observation} ({observation_reason})"
    progress_observation, progress_version = _task_progress(result)
    if presentation.get("details"):
        progress_fields = format_progress_details(
            progress_observation,
            progress_version=progress_version,
            include_metrics=True,
        )
    else:
        progress_fields = format_progress_compact(
            progress_observation,
            progress_version=progress_version,
        )
    return _details(
        (
            ("Task ID", result.get("task_id")),
            ("Name", result.get("name")),
            ("Phase", result.get("phase")),
            ("Reason", result.get("reason")),
        ),
        attempt_values,
        (("Observation", observation),),
        progress_fields,
    )


def _render_task_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action", "cancel"),
        result.get("outcome", result.get("operation_state", result.get("task_state"))),
        (
            ("Task ID", result.get("task_id")),
            ("Owning machine", result.get("owning_machine")),
            ("Queue scope", result.get("queue_scope")),
            ("Eligible machines", result.get("eligible_machines")),
            ("Pending acknowledgement", result.get("pending_acknowledgement")),
            ("Termination acknowledged at", result.get("termination_acknowledged_at")),
            ("Reason", result.get("reason")),
            ("Next", result.get("follow_up_command")),
        ),
    )


def _render_dependencies(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(tuple((str(label).replace("_", " ").capitalize(), value) for label, value in result.items()))


def _render_availability(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action"),
        str(result.get("outcome", "completed")).replace("_", " "),
        (
            ("Task ID", result.get("task_id")),
            ("Queue scope", result.get("resulting_state")),
            ("Eligible machines", result.get("eligible_helper_machines")),
            ("Home machine", result.get("home_machine")),
            ("Effective at", result.get("effective_at")),
            ("Operation ID", result.get("operation_id")),
            ("Reason", result.get("message")),
        ),
    )


def _render_task_wait(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    outcome = result.get("outcome")
    fields: list[tuple[str, Any]] = [("Task ID", result.get("task_id")), ("Outcome", outcome)]
    reason = result.get("reason")
    if reason:
        fields.append(("Reason", reason))
    if outcome in {"timeout", "timed_out"}:
        fields.append(("Result", "No mutation made."))
    else:
        fields.extend(
            (
                ("Selected Attempt number", result.get("selected_attempt_number")),
                ("Selected Attempt ID", result.get("selected_attempt_id")),
                ("Task exit code", result.get("task_exit_code")),
                ("Project", result.get("project")),
                ("Error", result.get("error")),
            )
        )
    return _details(tuple(fields))


def _validate_task_wait(result: Any) -> None:
    value = _mapping(result, "task-wait payload")
    for key in (
        "schema_version",
        "task_id",
        "project",
        "selected_attempt_number",
        "selected_attempt_id",
        "outcome",
        "reason",
        "task_exit_code",
        "error",
    ):
        _required(value, key, "task-wait payload")


def _validate_task_items(items: Sequence[Any], label: str) -> None:
    for index, item in enumerate(items):
        task = _mapping(item, f"{label}[{index}]")
        for key in (
            "task_id",
            "name",
            "phase",
            "reason",
            "gpus",
            "group",
            "home_machine",
            "queue_scope",
            "current_attempt_id",
            "claim_machine",
            "depends_on_task_ids",
            "dependency_state",
        ):
            _required(task, key, f"{label}[{index}]")
        _sequence(task["depends_on_task_ids"], f"{label}[{index}].depends_on_task_ids")


def _validate_task_list(result: Any) -> None:
    _validate_task_items(_sequence(result, "task-list payload"), "task-list payload")


def _validate_task_page(result: Any) -> None:
    page = _mapping(result, "task-page payload")
    items = _required_sequence(page, "items", "task-page payload")
    _validate_task_items(items, "task-page payload.items")
    stop_reason = _required(page, "stop_reason", "task-page payload")
    if not isinstance(stop_reason, str):
        raise TypeError("task-page payload.stop_reason must be a string")
    cursor = _required(page, "next_cursor", "task-page payload")
    if cursor is not None and not isinstance(cursor, str):
        raise TypeError("task-page payload.next_cursor must be a string or null")


def _validate_task_show(result: Any) -> None:
    value = _mapping(result, "task-show payload")
    task = _required_mapping(value, "task", "task-show payload")
    for key in ("task_id", "name", "group_name", "depends_on_task_ids"):
        _required(task, key, "task-show payload.task")
    spec = _required_mapping(task, "spec", "task-show payload.task")
    for key in ("command", "requested_gpus"):
        _required(spec, key, "task-show payload.task.spec")
    placement = _required_mapping(task, "placement_policy", "task-show payload.task")
    _required(placement, "home_machine", "task-show payload.task.placement_policy")
    runtime = _required_mapping(task, "placement_runtime", "task-show payload.task")
    _required(runtime, "queue_scope", "task-show payload.task.placement_runtime")
    state = _required_mapping(task, "state", "task-show payload.task")
    for key in ("projection", "reason"):
        _required(state, key, "task-show payload.task.state")
    control = _required_mapping(task, "control", "task-show payload.task")
    _required(control, "cancellation_operation_id", "task-show payload.task.control")
    _required_mapping(value, "dependency_gate", "task-show payload")
    _required_mapping(value, "progress", "task-show payload")
    if "progress_extended" in value:
        _mapping(value["progress_extended"], "task-show payload.progress_extended")
    selected_progress_version = value.get("selected_progress_version")
    if selected_progress_version is not None and (
        type(selected_progress_version) is not int or selected_progress_version not in {1, 2}
    ):
        raise ValueError("task-show payload.selected_progress_version must be 1, 2, or null")
    observation = _required_mapping(value, "observation", "task-show payload")
    override = _required(observation, "tmux_override", "task-show payload.observation")
    if override not in {"enabled", "disabled", "inherit"}:
        raise ValueError("task-show payload.observation.tmux_override is invalid")
    attempts = _required_sequence(value, "attempts", "task-show payload")
    for index, item in enumerate(attempts):
        attempt = _mapping(item, f"task-show payload.attempts[{index}]")
        attempt_value = _required_mapping(attempt, "attempt", f"task-show payload.attempts[{index}]")
        for key in ("attempt_number", "machine_name"):
            _required(attempt_value, key, f"task-show payload.attempts[{index}].attempt")


def _validate_task_watch(result: Any) -> None:
    """Validate the compact payload consumed by the refreshing Task view."""
    value = _mapping(result, "task-watch payload")
    for key in ("task_id", "name", "phase", "reason", "terminal", "observation_state", "observation_reason"):
        _required(value, key, "task-watch payload")
    _required(value, "revision", "task-watch payload")
    _required_bool(value, "terminal", "task-watch payload")
    _required_mapping(value, "progress", "task-watch payload")
    if "progress_extended" in value:
        _mapping(value["progress_extended"], "task-watch payload.progress_extended")
    selected_progress_version = value.get("selected_progress_version")
    if selected_progress_version is not None and (
        type(selected_progress_version) is not int or selected_progress_version not in {1, 2}
    ):
        raise ValueError("task-watch payload.selected_progress_version must be 1, 2, or null")
    attempt = _required(value, "selected_attempt", "task-watch payload")
    if attempt is not None:
        attempt_value = _mapping(attempt, "task-watch payload.selected_attempt")
        for key in ("attempt_id", "attempt_number", "phase", "machine_name", "log_path"):
            _required(attempt_value, key, "task-watch payload.selected_attempt")


def _validate_task_operation(result: Any) -> None:
    value = _mapping(result, "task-operation payload")
    _required(value, "task_id", "task-operation payload")
    _required(value, "action", "task-operation payload")
    _required(value, "outcome", "task-operation payload")
    if "operation_state" not in value and "task_state" not in value:
        raise ValueError("task-operation payload requires 'operation_state' or 'task_state'")


def _validate_dependencies(result: Any) -> None:
    value = _mapping(result, "dependencies payload")
    _required(value, "task_id", "dependencies payload")
    _required_sequence(value, "depends_on_task_ids", "dependencies payload")


def _validate_availability(result: Any) -> None:
    value = _mapping(result, "availability payload")
    for key in (
        "action",
        "outcome",
        "task_id",
        "resulting_state",
        "eligible_helper_machines",
        "message",
    ):
        _required(value, key, "availability payload")
    _sequence(value["eligible_helper_machines"], "availability payload.eligible_helper_machines")


CONTRACTS = {
    OutputKind.TASK_LIST: OutputContract(_validate_task_list, _render_task_list),
    OutputKind.TASK_PAGE: OutputContract(_validate_task_page, _render_task_page),
    OutputKind.TASK_SHOW: OutputContract(_validate_task_show, _render_task_show),
    OutputKind.TASK_WATCH: OutputContract(_validate_task_watch, _render_task_watch),
    OutputKind.TASK_CANCEL: OutputContract(_validate_task_operation, _render_task_operation),
    OutputKind.TASK_RETRY: OutputContract(_validate_task_operation, _render_task_operation),
    OutputKind.DEPENDENCIES: OutputContract(_validate_dependencies, _render_dependencies),
    OutputKind.AVAILABILITY: OutputContract(_validate_availability, _render_availability),
    OutputKind.TASK_WAIT: OutputContract(_validate_task_wait, _render_task_wait),
}

__all__ = ["CONTRACTS"]
