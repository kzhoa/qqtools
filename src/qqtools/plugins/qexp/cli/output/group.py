"""Pure Group-family output contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import (
    _details,
    _mapping,
    _operation,
    _queue_summary,
    _required,
    _required_mapping,
    _required_sequence,
    _sequence,
    _table,
    _task_summary,
)


def _group_values(result: Mapping[str, Any], presentation: Mapping[str, object]) -> tuple[Any, ...]:
    group = result.get("group", {})
    name = group.get("name") or result.get("name") or presentation.get("name")
    worker_set = group.get("worker_set", {})
    workers = (
        ", ".join(
            f"{machine}={worker.get('state')}"
            for machine, worker in sorted(worker_set.items())
            if isinstance(worker, Mapping)
        )
        if isinstance(worker_set, Mapping)
        else None
    )
    raw_tasks = result.get("tasks")
    if isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)):
        task_values = [item for item in raw_tasks if isinstance(item, Mapping)]
        task_summary = _task_summary(task_values, group=name)
        queue_summary = _queue_summary(task_values, name)
    else:
        task_summary = result.get("task_summary", "-")
        queue_summary = result.get("queue_summary", "-")
    return (
        name,
        group.get("admission_state"),
        group.get("dispatch_state"),
        workers,
        task_summary,
        queue_summary,
        group.get("pending_submission_commit"),
        group.get("reason") or result.get("reason"),
    )


def _group_fields(result: Mapping[str, Any], presentation: Mapping[str, object]) -> tuple[tuple[str, Any], ...]:
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
    return tuple(zip(labels, _group_values(result, presentation), strict=True))


def _render_group_list(result: Sequence[Mapping[str, Any]], _presentation: Mapping[str, object]) -> str:
    if not result:
        return "No Groups."
    return _table(
        (
            "Group",
            "Admission",
            "Dispatch",
            "Workers",
            "Task summary",
            "Control operation",
            "Reason",
        ),
        [
            (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[6],
                values[7],
            )
            for item in result
            for values in (_group_values(item, {}),)
        ],
        empty_message="No Groups.",
    )


def _render_group_show(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    return _details(_group_fields(result, presentation))


def _render_group_operation(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    group_value = result.get("group", {})
    group = group_value if isinstance(group_value, Mapping) else {}
    action = result.get("action", presentation.get("action", "group"))
    status = result.get("outcome", result.get("status", group.get("dispatch_state")))
    return _operation(
        str(action).replace("_", " "),
        str(status).replace("_", " ") if status is not None else None,
        (
            ("Group", group.get("name") or result.get("name") or presentation.get("name")),
            ("Worker machine", result.get("worker_machine")),
            ("Worker state", result.get("worker_state")),
            ("Scheduling role", result.get("scheduling_role")),
            (
                "GPU limit",
                "unlimited"
                if "gpu_limit_gpus" in result and result.get("gpu_limit_gpus") is None
                else result.get("gpu_limit_gpus"),
            ),
            ("Operation state", result.get("status")),
            ("Task IDs", result.get("task_ids")),
            ("Pending machines", result.get("pending_machines")),
            ("Blockers", result.get("blockers")),
            ("Reason", result.get("reason")),
            ("Operation reference", result.get("operation_reference")),
            ("Next", result.get("follow_up_command")),
        ),
    )


def _render_group_retry(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    count = result["retried_count"]
    if count == 0 and not any(result["skipped"].values()):
        return f"Group retry: no change\nGroup: {result['group']}\nNo eligible failed Tasks."
    skipped = result["skipped"]
    lines = [
        f"Group retry: {str(result['outcome']).replace('_', ' ')}",
        f"Group: {result['group']}",
        f"Retried: {count}",
        f"Task IDs: {result['retried_task_ids'][:10]}",
        f"Blocked: {len(skipped['blocked'])}",
        f"Orphaned: {len(skipped['orphaned'])}",
        f"Other skipped: {len(skipped['other'])}",
    ]
    if result.get("follow_up_command"):
        lines.append(f"Next: {result['follow_up_command']}")
    return "\n".join(lines)


def _render_group_machines(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        ("Machine", "Role", "GPU usage", "GPU limit", "State", "Agent"),
        [
            (
                item.get("machine_name"),
                item.get("scheduling_role"),
                item.get("gpu_usage"),
                item.get("gpu_limit_gpus") if item.get("gpu_limit_gpus") is not None else "unlimited",
                item.get("state"),
                item.get("agent"),
            )
            for item in result["machines"]
        ],
        empty_message="No Workers.",
    )


def _validate_group_record(value: Any, label: str) -> None:
    group_value = _mapping(value, label)
    group = _required_mapping(group_value, "group", label)
    _required(group, "name", f"{label}.group")
    _required_mapping(group, "worker_set", f"{label}.group")
    _required(group, "admission_state", f"{label}.group")
    _required(group, "dispatch_state", f"{label}.group")


def _validate_group_list(result: Any) -> None:
    groups = _sequence(result, "group-list payload")
    for index, item in enumerate(groups):
        _validate_group_record(item, f"group-list payload[{index}]")


def _validate_group_show(result: Any) -> None:
    _validate_group_record(result, "group-show payload")


def _validate_group_operation(result: Any) -> None:
    value = _mapping(result, "group-operation payload")
    if "group" not in value and "task_ids" not in value:
        raise ValueError("group-operation payload requires 'group' or 'task_ids'")
    if "group" in value:
        if isinstance(value["group"], Mapping):
            _validate_group_record(value, "group-operation payload")
        elif not isinstance(value["group"], str):
            raise TypeError("group-operation payload.group must be a Group record or name")
    if "task_ids" in value:
        _sequence(value["task_ids"], "group-operation payload.task_ids")


def _validate_group_retry(result: Any) -> None:
    value = _mapping(result, "group-retry payload")
    for key in ("action", "outcome", "group", "retried_task_ids", "retried_count", "skipped"):
        _required(value, key, "group-retry payload")
    _sequence(value["retried_task_ids"], "group-retry payload.retried_task_ids")
    skipped = _mapping(value["skipped"], "group-retry payload.skipped")
    for key in ("blocked", "orphaned", "other"):
        _sequence(_required(skipped, key, "group-retry payload.skipped"), f"group-retry payload.skipped.{key}")


def _validate_group_worker_change(result: Any) -> None:
    _validate_group_operation(result)
    value = _mapping(result, "group-worker-change payload")
    for key in (
        "action",
        "outcome",
        "worker_machine",
        "worker_state",
        "scheduling_role",
        "gpu_limit_gpus",
    ):
        _required(value, key, "group-worker-change payload")
    if "blockers" in value:
        _sequence(value["blockers"], "group-worker-change payload.blockers")


def _validate_group_machines(result: Any) -> None:
    value = _mapping(result, "group-machines payload")
    machines = _required_sequence(value, "machines", "group-machines payload")
    for index, item in enumerate(machines):
        machine = _mapping(item, f"group-machines payload.machines[{index}]")
        for key in ("machine_name", "scheduling_role", "gpu_usage", "gpu_limit_gpus", "state", "agent"):
            _required(machine, key, f"group-machines payload.machines[{index}]")


CONTRACTS = {
    OutputKind.GROUP_LIST: OutputContract(_validate_group_list, _render_group_list),
    OutputKind.GROUP_SHOW: OutputContract(_validate_group_show, _render_group_show),
    OutputKind.GROUP_STATE_CHANGE: OutputContract(_validate_group_operation, _render_group_operation),
    OutputKind.GROUP_RETRY: OutputContract(_validate_group_retry, _render_group_retry),
    OutputKind.GROUP_CANCEL: OutputContract(_validate_group_operation, _render_group_operation),
    OutputKind.GROUP_WORKER_CHANGE: OutputContract(_validate_group_worker_change, _render_group_operation),
    OutputKind.GROUP_MACHINES: OutputContract(_validate_group_machines, _render_group_machines),
}

__all__ = ["CONTRACTS"]
