"""Pure CLI result rendering for qexp."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def render(kind: str, result: Any, output_format: str, *, tasks: Sequence[Mapping[str, Any]] = ()) -> str:
    if output_format == "json":
        return json.dumps(result, default=_json_default)
    return _render_human(kind, result, tasks=tasks)


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return "none" if not value else ", ".join(f"{key}={_value(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "none" if not value else ", ".join(_value(item) for item in value)
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "No results."
    text_rows = [[_value(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    render_row = lambda row: "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
    return "\n".join([render_row(headers), render_row(["-" * width for width in widths]), *map(render_row, text_rows)])


def _details(*sections: Sequence[tuple[str, Any]]) -> str:
    return "\n\n".join("\n".join(f"{label}: {_value(value)}" for label, value in section) for section in sections)


def _operation(action: str, status: str, fields: Sequence[tuple[str, Any]]) -> str:
    return _details((("Action", action), ("Status", status), *fields))


def _task_summary(tasks: Sequence[Mapping[str, Any]], *, group: str | None = None, machine: str | None = None) -> str:
    selected = [
        task
        for task in tasks
        if (group is None or task.get("group") == group) and (machine is None or task.get("home_machine") == machine)
    ]
    counts = Counter(task.get("phase") or "unknown" for task in selected)
    return _value(dict(sorted(counts.items())))


def _queue_summary(tasks: Sequence[Mapping[str, Any]], group: str | None) -> str:
    counts = Counter(task.get("queue_scope") or "unknown" for task in tasks if task.get("group") == group)
    return _value(dict(sorted(counts.items())))


def _group_values(result: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    group = result.get("group", {})
    name = group.get("name") or result.get("name")
    workers = {machine: worker.get("state") for machine, worker in group.get("worker_set", {}).items()}
    return (
        name,
        group.get("admission_state"),
        group.get("dispatch_state"),
        workers,
        _task_summary(tasks, group=name),
        _queue_summary(tasks, name),
        group.get("pending_submission_commit"),
        group.get("reason") or result.get("reason"),
    )


def _group_fields(result: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    labels = (
        "Group",
        "Admission",
        "Dispatch",
        "Workers",
        "Task summary",
        "Queue summary",
        "Control operation",
        "Reason",
    )
    return tuple(zip(labels, _group_values(result, tasks), strict=True))


def _render_human(kind: str, result: Any, *, tasks: Sequence[Mapping[str, Any]]) -> str:
    if kind == "task-list":
        return _table(
            (
                "Task ID",
                "Name",
                "State",
                "GPUs",
                "Group",
                "Home machine",
                "Queue scope",
                "Attempt",
                "Claimed machine",
                "Reason",
            ),
            [
                (
                    item.get("task_id"),
                    item.get("name"),
                    item.get("phase"),
                    item.get("gpus"),
                    item.get("group"),
                    item.get("home_machine"),
                    item.get("queue_scope"),
                    item.get("current_attempt_id"),
                    item.get("claim_machine"),
                    item.get("reason"),
                )
                for item in result
            ],
        )
    if kind == "task-show":
        task = result.get("task", result)
        spec = task.get("spec", {})
        placement = task.get("placement_policy", {})
        runtime = task.get("placement_runtime", {})
        state = task.get("state", {})
        control = task.get("control", {})
        attempts = result.get("attempts", ())
        submission = result.get("submission", {})
        execution_machines = [
            f"#{attempt.get('attempt', {}).get('attempt_number')}:"
            f"{attempt.get('attempt', {}).get('machine_name')}"
            for attempt in attempts
        ]
        return _details(
            (
                ("Task ID", task.get("task_id")),
                ("Name", task.get("name")),
                ("Command", spec.get("argv")),
                ("GPUs", spec.get("requested_gpus")),
                ("Group", task.get("group_name")),
            ),
            (
                ("State", state.get("projection")),
                ("Original submitting machine", submission.get("original_submitting_machine")),
                ("Home machine", placement.get("home_machine")),
                ("Execution machines", execution_machines),
                ("Queue scope", runtime.get("queue_scope")),
                ("Control", control.get("cancellation_operation_id")),
                ("Attempts", len(attempts)),
                ("Reason", state.get("reason")),
            ),
        )
    if kind == "task-operation":
        return _operation(
            result.get("action", "cancel"),
            result.get("operation_state", result.get("task_state")),
            (
                ("Task ID", result.get("task_id")),
                ("Queue scope", result.get("queue_scope")),
                ("Eligible machines", result.get("eligible_machines")),
                ("Pending acknowledgement", result.get("pending_acknowledgement")),
                ("Reason", result.get("reason")),
            ),
        )
    if kind == "availability":
        return _operation(
            result.get("action"),
            result.get("resulting_state"),
            (
                ("Task ID", result.get("task_id")),
                ("Queue scope", result.get("resulting_state")),
                ("Eligible machines", result.get("eligible_helper_machines")),
                ("Pending acknowledgement", result.get("pending_acknowledgement")),
                ("Reason", result.get("message")),
            ),
        )
    if kind == "group-list":
        return _table(
            (
                "Group",
                "Admission",
                "Dispatch",
                "Workers",
                "Task summary",
                "Queue summary",
                "Control operation",
                "Reason",
            ),
            [_group_values(item, tasks) for item in result],
        )
    if kind == "group-show":
        return _details(_group_fields(result, tasks))
    if kind == "group-operation":
        group = result.get("group", {})
        return _operation(
            result.get("action", "group"),
            result.get("status", group.get("dispatch_state")),
            (
                ("Group", group.get("name") or result.get("name")),
                ("Worker machine", result.get("worker_machine")),
                ("Task IDs", result.get("task_ids")),
                ("Pending machines", result.get("pending_machines")),
                ("Reason", result.get("reason")),
            ),
        )
    if kind == "group-machines":
        return _table(
            ("Machine", "Role", "GPU usage", "GPU limit", "State", "Agent"),
            [
                (
                    item.get("machine_name"),
                    item.get("scheduling_role"),
                    item.get("gpu_usage"),
                    (
                        item.get("borrow_limit_gpus")
                        if item.get("borrow_limit_gpus") is not None
                        else "unlimited"
                    ),
                    item.get("state"),
                    item.get("agent"),
                )
                for item in result.get("machines", ())
            ],
        )
    if kind in {"machines", "top"}:
        machines = result.get("machines", ()) if isinstance(result, Mapping) else result
        return _table(
            (
                "Machine",
                "Availability",
                "GPU visible",
                "GPU reserved",
                "GPU unreserved",
                "Agent state",
                "Task summary",
                "Reason",
            ),
            [
                (
                    item.get("machine", {}).get("machine_name"),
                    item.get("state", {}).get("freshness"),
                    item.get("state", {}).get("gpu", {}).get("visible"),
                    item.get("state", {}).get("gpu", {}).get("reserved"),
                    item.get("state", {}).get("gpu", {}).get("unreserved"),
                    item.get("state", {}).get("agent", {}).get("agent_state"),
                    _task_summary(tasks, machine=item.get("machine", {}).get("machine_name")),
                    item.get("reason"),
                )
                for item in machines
            ],
        )
    if kind == "agent":
        return _operation(
            result.get("action"),
            result.get("agent_state"),
            (
                ("Agent mode", result.get("agent_mode")),
                ("Machine", result.get("machine_name")),
                ("PID", result.get("pid")),
                ("Previous PID", result.get("previous_pid")),
                ("Reason", result.get("reason")),
            ),
        )
    if kind == "batch-submit":
        return _operation(
            "batch-submit",
            result.get("state"),
            (
                ("Operation ID", result.get("operation_id")),
                ("Idempotency key", result.get("idempotency_key")),
                ("Group", result.get("target_group")),
                ("Task count", len(result.get("task_ids", ()))),
                ("Task IDs", result.get("task_ids")),
                ("Reason", result.get("reason")),
            ),
        )
    if kind == "context":
        return _details(
            (
                ("Shared root", result.get("shared_root")),
                ("Machine", result.get("machine")),
                ("Runtime root", result.get("runtime_root")),
            )
        )
    if kind in {"doctor", "clean", "notifications", "lease-policy"}:
        action = result.get("action", kind)
        status = result.get("status", result.get("state", "completed"))
        return _operation(
            action,
            status,
            tuple(
                (label.replace("_", " ").capitalize(), value)
                for label, value in result.items()
                if label not in {"action", "status", "state"}
            ),
        )
    return _details(tuple((str(label).replace("_", " ").capitalize(), value) for label, value in result.items()))
