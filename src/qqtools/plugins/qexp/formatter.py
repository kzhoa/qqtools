"""Pure CLI result rendering for qexp.

The formatter is deliberately a closed boundary.  Command handlers provide one
canonical payload and optional human-only presentation context; this module
never reads qexp state or calls a workflow while rendering that value.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypedDict, TypeVar

from .progress_format import format_progress_details


class OutputKind(str, Enum):
    """Finite set of structured qexp CLI output families."""

    TASK_LIST = "task-list"
    TASK_PAGE = "task-page"
    TASK_SHOW = "task-show"
    TASK_WATCH = "task-watch"
    TASK_OPERATION = "task-operation"
    DEPENDENCIES = "dependencies"
    AVAILABILITY = "availability"
    GROUP_LIST = "group-list"
    GROUP_SHOW = "group-show"
    GROUP_OPERATION = "group-operation"
    GROUP_MACHINES = "group-machines"
    MACHINES = "machines"
    TOP = "top"
    AGENT_OPERATION = "agent-operation"
    AGENT_STATUS = "agent-status"
    AGENT_PROJECT_LIST = "agent-project-list"
    AGENT_CONFIG = "agent-config"
    AGENT_READINESS = "agent-readiness"
    MACHINE_INIT = "machine-init"
    PROJECT_OPERATION = "project-operation"
    PROJECT_REGISTER = "project-register"
    PROJECT_LIST = "project-list"
    CPU_LANE = "cpu-lane"
    GPU_POLICY = "gpu-policy"
    UPGRADE_REGISTRY_STATUS = "upgrade-registry-status"
    UPGRADE_ADVANCE = "upgrade-advance"
    UPGRADE_PROJECT = "upgrade-project"
    UPGRADE_REPAIR = "upgrade-repair"
    SCHEMA6_UPGRADE = "schema6-upgrade"
    BATCH_SUBMIT = "batch-submit"
    CONTEXT = "context"
    PROGRESS_POLICY = "progress-policy"
    LAUNCH_HANDOFF_POLICY = "launch-handoff-policy"
    TMUX_POLICY = "tmux-policy"
    NOTIFICATIONS = "notifications"
    LEASE_POLICY = "lease-policy"
    DOCTOR_VERIFY = "doctor-verify"
    DOCTOR_REPAIR = "doctor-repair"
    CLEAN = "clean"


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CliOutput(Generic[T]):
    """Canonical CLI payload plus optional non-JSON presentation context."""

    kind: OutputKind
    payload: T
    presentation: Mapping[str, object] = field(default_factory=dict)


class CpuLaneValue(TypedDict):
    capacity: int
    revision: int


class CpuLanePayload(TypedDict):
    cpu_lane: CpuLaneValue


class ProjectListPayload(TypedDict):
    action: str
    projects: list[dict[str, Any]]


class TaskPagePayload(TypedDict):
    items: list[dict[str, Any]]
    next_cursor: str | None
    stop_reason: str


class UpgradeRegistryStatusPayload(TypedDict):
    projects: list[dict[str, Any]]


class UpgradeAdvancePayload(TypedDict):
    projects: list[dict[str, Any]]


Validator = Callable[[Any], None]
Renderer = Callable[[Any, Mapping[str, object]], str]


def render(output: CliOutput[Any], output_format: str) -> str:
    """Render one validated finite output without changing its canonical payload."""
    if not isinstance(output, CliOutput):
        raise TypeError(f"render expects CliOutput, got {type(output).__name__}")
    contract = _contract_for(output.kind)
    contract.validator(output.payload)
    if output_format == "json":
        return json.dumps(output.payload, default=_json_default)
    if output_format != "human":
        raise ValueError(f"unsupported output format {output_format!r}")
    if not isinstance(output.presentation, Mapping):
        raise TypeError("CLI output presentation must be a mapping")
    return contract.renderer(output.payload, output.presentation)


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


def _operation(action: Any, status: Any, fields: Sequence[tuple[str, Any]]) -> str:
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


def _task_rows(tasks: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [
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
            item.get("depends_on_task_ids"),
            item.get("dependency_state"),
            item.get("reason"),
        )
        for item in tasks
    ]


def _render_task_table(tasks: Sequence[Mapping[str, Any]]) -> str:
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
            "Dependencies",
            "Dependency gate",
            "Reason",
        ),
        _task_rows(tasks),
    )


def _group_values(result: Mapping[str, Any], presentation: Mapping[str, object]) -> tuple[Any, ...]:
    group = result.get("group", {})
    name = group.get("name") or result.get("name") or presentation.get("name")
    workers = {machine: worker.get("state") for machine, worker in group.get("worker_set", {}).items()}
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


def _render_task_list(result: Any, _presentation: Mapping[str, object]) -> str:
    return _render_task_table(result)


def _render_task_page(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    rendered = _render_task_table(result["items"])
    lines = [rendered, f"Stop reason: {result['stop_reason']}"]
    command = presentation.get("continuation_command")
    if command is not None:
        lines.append(f"Continue with: {_value(command)}")
    return "\n".join(lines)


def _render_task_show(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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
    return _details(
        (
            ("Task ID", task.get("task_id")),
            ("Name", task.get("name")),
            ("Command", spec.get("command")),
            ("GPUs", spec.get("requested_gpus")),
            ("Group", task.get("group_name")),
            ("Dependencies", task.get("depends_on_task_ids")),
            ("TMUX observer override", result.get("observation", {}).get("tmux_override")),
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
            ("Dependency gate", result.get("dependency_gate")),
        ),
        format_progress_details(result.get("progress")),
    )


def _render_task_watch(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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
    return _details(
        (
            ("Task ID", result.get("task_id")),
            ("Name", result.get("name")),
            ("Phase", result.get("phase")),
            ("Reason", result.get("reason")),
        ),
        attempt_values,
        (("Observation", observation),),
        format_progress_details(result.get("progress")),
    )


def _render_task_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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


def _render_dependencies(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(tuple((str(label).replace("_", " ").capitalize(), value) for label, value in result.items()))


def _render_availability(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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


def _render_group_list(result: Sequence[Mapping[str, Any]], _presentation: Mapping[str, object]) -> str:
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
        [_group_values(item, {}) for item in result],
    )


def _render_group_show(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    return _details(_group_fields(result, presentation))


def _render_group_operation(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    group = result.get("group", {})
    action = presentation.get("action", result.get("action", "group"))
    status = presentation.get("status", result.get("status", group.get("dispatch_state")))
    return _operation(
        action,
        status,
        (
            ("Group", group.get("name") or result.get("name") or presentation.get("name")),
            ("Worker machine", presentation.get("worker_machine", result.get("worker_machine"))),
            ("Task IDs", result.get("task_ids")),
            ("Pending machines", presentation.get("pending_machines", result.get("pending_machines"))),
            ("Reason", presentation.get("reason", result.get("reason"))),
        ),
    )


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
    )


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
        gpu.get("visible_gpu_ids", gpu.get("visible")),
        gpu.get("reserved_gpu_ids", gpu.get("reserved")),
        gpu.get("free_gpu_ids", gpu.get("unreserved")),
        policy.get("source"),
        policy.get("mode"),
        policy.get("draining_gpu_ids"),
        policy.get("warnings"),
        agent.get("observed_state", agent.get("agent_state")),
        task_summary,
        item.get("reason"),
    )


def _render_machines(result: Any, _presentation: Mapping[str, object]) -> str:
    machines = result["machines"] if isinstance(result, Mapping) else result
    return _table(
        (
            "Machine",
            "Availability",
            "GPU visible",
            "GPU reserved",
            "GPU unreserved",
            "GPU source",
            "GPU mode",
            "GPU draining",
            "GPU warnings",
            "Agent state",
            "Task summary",
            "Reason",
        ),
        [_machine_values(item) for item in machines],
    )


def _render_top(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        (
            "Machine",
            "Availability",
            "GPU visible",
            "GPU reserved",
            "GPU unreserved",
            "GPU source",
            "GPU mode",
            "GPU draining",
            "GPU warnings",
            "Agent state",
            "Task summary",
            "Reason",
        ),
        [_machine_values(item, result["tasks"]) for item in result["machines"]],
    )


def _render_agent_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action"),
        result.get("agent_state", result.get("status", result.get("state"))),
        (
            ("Agent mode", result.get("agent_mode")),
            ("Project ID", result.get("project_id")),
            ("Machine", result.get("machine_name")),
            ("PID", result.get("pid")),
            ("Previous PID", result.get("previous_pid")),
            ("Message", result.get("message")),
            ("Shared root", result.get("shared_root")),
            ("Enable command", result.get("enable_command")),
            ("Reason", result.get("reason")),
            ("GPU policy", result.get("gpu_policy")),
            ("Warnings", result.get("warnings", ())),
        ),
    )


def _render_agent_status(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    projects = result["projects"]
    upgrade = result["upgrade"]
    pending = upgrade.get("pending_project_ids", ()) if isinstance(upgrade, Mapping) else ()
    return _operation(
        result["action"],
        result["agent_state"],
        (
            ("PID", result["pid"]),
            ("Runtime ID", result.get("runtime_id")),
            ("Configured mode", result.get("configured_agent_mode")),
            ("Observed mode", result.get("observed_agent_mode")),
            ("Requested policy revision", result.get("requested_policy_revision")),
            ("Acknowledged policy revision", result.get("acknowledged_policy_revision")),
            ("Ready", result.get("ready")),
            ("Registry revision", result["registry_revision"]),
            ("Registered projects", len(projects)),
            ("GPU policy", result.get("gpu_policy")),
            ("Warnings", result.get("warnings", ())),
            ("Pending upgrades", pending),
            ("Waiting for first registration", result.get("waiting_for_first_registration")),
        ),
    )


def _render_agent_project_list(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        ("Project ID", "Shared root", "Machine", "Enabled", "State", "Eligibility", "Write eligible"),
        [
            (
                project.get("project_id"),
                project.get("shared_root"),
                project.get("machine_name"),
                project.get("enabled"),
                project.get("state"),
                project.get("eligibility", {}).get("state"),
                project.get("write_eligible"),
            )
            for project in result["projects"]
        ],
    )


def _render_machine_init(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action"),
        "completed",
        (
            ("Agent name", result.get("agent_name")),
            ("Agent mode", result.get("agent_mode")),
            ("Old runtime ID", result.get("old_runtime_id")),
            ("New runtime ID", result.get("new_runtime_id")),
            ("Detached", result.get("detached")),
            ("Archive", result.get("archive_path")),
            ("Obligations", result.get("obligations")),
        ),
    )


def _render_project_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action"),
        result.get("status", result.get("state", "completed")),
        (
            ("Project ID", result.get("project_id")),
            ("Shared root", result.get("shared_root")),
            ("Machine", result.get("machine_name")),
            ("Enabled", result.get("enabled")),
            ("Source", result.get("name_source")),
            ("Local only", result.get("local_only")),
            ("Reason", result.get("reason")),
        ),
    )


def _render_project_register(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    projects = result.get("projects", ())
    return _table(
        ("Project ID", "Shared root", "Machine", "Source", "Enabled", "Status", "Reason"),
        [
            (
                project.get("project_id"),
                project.get("shared_root"),
                project.get("machine_name"),
                project.get("name_source"),
                project.get("enabled"),
                project.get("status"),
                project.get("reason"),
            )
            for project in projects
        ],
    )


def _render_project_list(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        ("Project ID", "Shared root", "Machine", "Source", "Enabled", "Status", "Mount"),
        [
            (
                project.get("project_id"),
                project.get("shared_root"),
                project.get("machine_name"),
                project.get("name_source"),
                project.get("enabled"),
                project.get("status"),
                project.get("mount_available"),
            )
            for project in result.get("projects", ())
        ],
    )


def _render_agent_config(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Agent name", result.get("agent_name")),
            ("Agent mode", result.get("agent_mode")),
            ("Revision", result.get("revision")),
            ("Provenance", result.get("provenance")),
            ("Runtime ID", result.get("runtime_id")),
        )
    )


def _render_agent_readiness(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        "agent start",
        "ready" if result.get("ready") else "not ready",
        (
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
    return _details((("Capacity", lane["capacity"]), ("Revision", lane["revision"])))


def _render_gpu_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
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
            ("Warnings", result.get("warnings")),
            ("Agent running", result.get("agent_running")),
        )
    )


def _upgrade_rows(projects: Sequence[Mapping[str, Any]], *, nested: bool) -> list[tuple[Any, ...]]:
    rows = []
    for project in projects:
        status = project.get("upgrade", {}) if nested else project
        rows.append(
            (
                project.get("project_id"),
                status.get("phase"),
                status.get("state"),
                status.get("pending"),
                status.get("admission_blocked"),
                status.get("blockers"),
            )
        )
    return rows


def _render_upgrade_registry(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        ("Project", "Phase", "State", "Pending", "Admission blocked", "Blockers"),
        _upgrade_rows(result["projects"], nested=True),
    )


def _render_upgrade_advance(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _table(
        ("Project", "Phase", "State", "Pending", "Admission blocked", "Blockers"),
        _upgrade_rows(result["projects"], nested=False),
    )


def _render_upgrade_project(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        "upgrade",
        result.get("state", result.get("aggregate_state", "completed")),
        (
            ("Project", result.get("project_id")),
            ("Phase", result.get("phase")),
            ("Pending", result.get("pending")),
            ("Admission blocked", result.get("admission_blocked")),
            ("Blockers", result.get("blockers")),
        ),
    )


def _render_upgrade_repair(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    return _operation(
        presentation.get("action", "upgrade-repair"),
        result["state"],
        (
            ("Repair ID", result["repair_id"]),
            ("Target", result["target"]),
            ("Reason", result.get("error")),
        ),
    )


def _render_schema6(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(tuple((str(label).replace("_", " ").capitalize(), value) for label, value in result.items()))


def _render_batch_submit(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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


def _render_context(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return f"shared_root: {result.get('shared_root') or '<not set>'}"


def _render_progress_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Interval seconds", result.get("interval_seconds")),
            ("Source", result.get("source")),
            ("Applies to", result.get("applies_to")),
        ),
        (
            (
                "Timing",
                "Shorter intervals show fresh reports sooner but increase filesystem I/O; "
                "longer intervals reduce I/O and may delay visibility.",
            ),
        ),
    )


def _render_launch_handoff_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Timeout seconds", result.get("timeout_seconds")),
            ("Source", result.get("source")),
            ("Applies to", result.get("applies_to")),
        ),
        (("Timing", "The timeout is frozen for each new runner launch."),),
    )


def _render_tmux_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Project tmux observers enabled", result.get("enabled")),
            ("Source", result.get("source")),
            ("Applies to", "future observer decisions"),
        ),
        (("Scope", "This project only; existing windows and explicit Task choices are unchanged."),),
    )


def _render_named_operation(result: Mapping[str, Any], default_action: str) -> str:
    action = result.get("action", default_action)
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


def _render_notifications(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _render_named_operation(result, "notifications")


def _render_lease_policy(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _render_named_operation(result, "lease-policy")


def _render_doctor(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _render_named_operation(result, "doctor")


def _render_clean(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _render_named_operation(result, "clean")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence, got {type(value).__name__}")
    return value


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise ValueError(f"{label} is missing required key {key!r}")
    return value[key]


def _required_mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    return _mapping(_required(value, key, label), f"{label}.{key}")


def _required_sequence(value: Mapping[str, Any], key: str, label: str) -> Sequence[Any]:
    return _sequence(_required(value, key, label), f"{label}.{key}")


def _required_int(value: Mapping[str, Any], key: str, label: str) -> int:
    item = _required(value, key, label)
    if type(item) is not int:
        raise TypeError(f"{label}.{key} must be an integer")
    return item


def _required_bool(value: Mapping[str, Any], key: str, label: str) -> bool:
    item = _required(value, key, label)
    if type(item) is not bool:
        raise TypeError(f"{label}.{key} must be a boolean")
    return item


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
    attempt = _required(value, "selected_attempt", "task-watch payload")
    if attempt is not None:
        attempt_value = _mapping(attempt, "task-watch payload.selected_attempt")
        for key in ("attempt_id", "attempt_number", "phase", "machine_name", "log_path"):
            _required(attempt_value, key, "task-watch payload.selected_attempt")


def _validate_task_operation(result: Any) -> None:
    value = _mapping(result, "task-operation payload")
    _required(value, "task_id", "task-operation payload")
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
        "task_id",
        "resulting_state",
        "eligible_helper_machines",
        "message",
    ):
        _required(value, key, "availability payload")
    _sequence(value["eligible_helper_machines"], "availability payload.eligible_helper_machines")


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
        _validate_group_record(value, "group-operation payload")
    if "task_ids" in value:
        _sequence(value["task_ids"], "group-operation payload.task_ids")


def _validate_group_machines(result: Any) -> None:
    value = _mapping(result, "group-machines payload")
    machines = _required_sequence(value, "machines", "group-machines payload")
    for index, item in enumerate(machines):
        machine = _mapping(item, f"group-machines payload.machines[{index}]")
        for key in ("machine_name", "scheduling_role", "gpu_usage", "gpu_limit_gpus", "state", "agent"):
            _required(machine, key, f"group-machines payload.machines[{index}]")


def _validate_machine(result: Any, label: str) -> None:
    value = _mapping(result, label)
    machine = _required_mapping(value, "machine", label)
    _required(machine, "machine_name", f"{label}.machine")
    _required_mapping(value, "state", label)


def _validate_machines(result: Any) -> None:
    machines = _sequence(result, "machines payload")
    for index, item in enumerate(machines):
        _validate_machine(item, f"machines payload[{index}]")


def _validate_top(result: Any) -> None:
    value = _mapping(result, "top payload")
    machines = _required_sequence(value, "machines", "top payload")
    tasks = _required_sequence(value, "tasks", "top payload")
    _validate_task_items(tasks, "top payload.tasks")
    for index, item in enumerate(machines):
        _validate_machine(item, f"top payload.machines[{index}]")


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
        return
    if action == "running":
        for key in ("agent_state", "is_running", "machine_runtime_root"):
            _required(value, key, "agent-operation payload")
        return
    if action in lifecycle_actions:
        for key in ("agent_state", "pid", "is_running"):
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


def _validate_agent_project_list(result: Any) -> None:
    value = _mapping(result, "agent-project-list payload")
    if _required(value, "action", "agent-project-list payload") != "project_list":
        raise ValueError("agent-project-list payload.action must be 'project_list'")
    projects = _required_sequence(value, "projects", "agent-project-list payload")
    for index, project in enumerate(projects):
        _validate_project(project, f"agent-project-list payload.projects[{index}]")


def _validate_machine_init(result: Any) -> None:
    value = _mapping(result, "machine-init payload")
    if _required(value, "action", "machine-init payload") not in {"initialized", "reinitialized"}:
        raise ValueError("machine-init payload.action is invalid")
    for key in (
        "agent_name",
        "agent_mode",
        "old_runtime_id",
        "new_runtime_id",
        "detached",
        "archive_path",
        "obligations",
    ):
        _required(value, key, "machine-init payload")
    _required_bool(value, "detached", "machine-init payload")
    if not isinstance(value["new_runtime_id"], str) or len(value["new_runtime_id"]) != 64:
        raise ValueError("machine-init payload.new_runtime_id must be a 64-hex runtime ID")
    _required_sequence(value, "obligations", "machine-init payload")


def _validate_project_operation(result: Any) -> None:
    value = _mapping(result, "project-operation payload")
    _required(value, "action", "project-operation payload")
    if "project_id" in value:
        _required(value, "shared_root", "project-operation payload")


def _validate_project_register(result: Any) -> None:
    value = _mapping(result, "project-register payload")
    _required_int(value, "revision", "project-register payload")
    projects = _required_sequence(value, "projects", "project-register payload")
    for index, item in enumerate(projects):
        project = _mapping(item, f"project-register payload.projects[{index}]")
        for key in ("project_id", "shared_root", "enabled", "name_source", "status"):
            _required(project, key, f"project-register payload.projects[{index}]")


def _validate_project_list(result: Any) -> None:
    value = _mapping(result, "project-list payload")
    _required_int(value, "revision", "project-list payload")
    projects = _required_sequence(value, "projects", "project-list payload")
    for index, item in enumerate(projects):
        project = _mapping(item, f"project-list payload.projects[{index}]")
        for key in ("project_id", "shared_root", "enabled", "name_source", "status", "mount_available"):
            _required(project, key, f"project-list payload.projects[{index}]")
        _required_bool(project, "enabled", f"project-list payload.projects[{index}]")
        _required_bool(project, "mount_available", f"project-list payload.projects[{index}]")


def _validate_agent_config(result: Any) -> None:
    value = _mapping(result, "agent-config payload")
    for key in ("agent_name", "agent_mode", "revision", "provenance", "runtime_id"):
        _required(value, key, "agent-config payload")
    _required_int(value, "revision", "agent-config payload")


def _validate_agent_readiness(result: Any) -> None:
    value = _mapping(result, "agent-readiness payload")
    _required_bool(value, "ready", "agent-readiness payload")
    _required(value, "projects", "agent-readiness payload")
    _sequence(value["projects"], "agent-readiness payload.projects")


def _validate_agent_status(result: Any) -> None:
    value = _mapping(result, "agent-status payload")
    if _required(value, "action", "agent-status payload") != "status":
        raise ValueError("agent-status payload.action must be 'status'")
    for key in ("agent_state", "pid", "registry_revision"):
        _required(value, key, "agent-status payload")
    projects = _required_sequence(value, "projects", "agent-status payload")
    for index, project in enumerate(projects):
        _validate_project(project, f"agent-status payload.projects[{index}]")
    upgrade = _required_mapping(value, "upgrade", "agent-status payload")
    _validate_upgrade_registry(upgrade)
    if "gpu_policy" in value:
        _validate_gpu_policy(value["gpu_policy"])
    if "warnings" in value:
        _sequence(value["warnings"], "agent-status payload.warnings")


def _validate_cpu_lane(result: Any) -> None:
    value = _mapping(result, "cpu-lane payload")
    lane = _required_mapping(value, "cpu_lane", "cpu-lane payload")
    _required_int(lane, "capacity", "cpu-lane payload.cpu_lane")
    _required_int(lane, "revision", "cpu-lane payload.cpu_lane")


def _validate_gpu_policy(result: Any) -> None:
    value = _mapping(result, "gpu-policy payload")
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
        _required(value, key, "gpu-policy payload")
    if value["mode"] not in {"auto", "explicit"}:
        raise ValueError("gpu-policy payload.mode is invalid")
    if value["source"] not in {"discovery", "environment", "persisted", "pending", "unavailable"}:
        raise ValueError("gpu-policy payload.source is invalid")
    if type(value["revision"]) is not int or value["revision"] < 0:
        raise TypeError("gpu-policy payload.revision must be a nonnegative integer")
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
            values = _sequence(ids, f"gpu-policy payload.{key}")
            if any(type(item) is not int or item < 0 for item in values):
                raise TypeError(f"gpu-policy payload.{key} must contain nonnegative integers")
    _sequence(value["warnings"], "gpu-policy payload.warnings")
    _required_bool(value, "agent_running", "gpu-policy payload")


def _validate_upgrade_registry(result: Any) -> None:
    value = _mapping(result, "upgrade-registry-status payload")
    projects = _required_sequence(value, "projects", "upgrade-registry-status payload")
    for index, project in enumerate(projects):
        item = _mapping(project, f"upgrade-registry-status payload.projects[{index}]")
        _required(item, "project_id", f"upgrade-registry-status payload.projects[{index}]")
        status = _required_mapping(item, "upgrade", f"upgrade-registry-status payload.projects[{index}]")
        for key in ("phase", "state", "pending", "admission_blocked", "blockers"):
            _required(status, key, f"upgrade-registry-status payload.projects[{index}].upgrade")
        _required_bool(status, "pending", f"upgrade-registry-status payload.projects[{index}].upgrade")
        _required_bool(status, "admission_blocked", f"upgrade-registry-status payload.projects[{index}].upgrade")
        _sequence(status["blockers"], f"upgrade-registry-status payload.projects[{index}].upgrade.blockers")


def _validate_upgrade_advance(result: Any) -> None:
    value = _mapping(result, "upgrade-advance payload")
    projects = _required_sequence(value, "projects", "upgrade-advance payload")
    for index, project in enumerate(projects):
        item = _mapping(project, f"upgrade-advance payload.projects[{index}]")
        _required(item, "project_id", f"upgrade-advance payload.projects[{index}]")
        for key in ("phase", "state", "pending", "admission_blocked", "blockers"):
            _required(item, key, f"upgrade-advance payload.projects[{index}]")
        _required_bool(item, "pending", f"upgrade-advance payload.projects[{index}]")
        _required_bool(item, "admission_blocked", f"upgrade-advance payload.projects[{index}]")
        _sequence(item["blockers"], f"upgrade-advance payload.projects[{index}].blockers")


def _validate_upgrade_project(result: Any) -> None:
    value = _mapping(result, "upgrade-project payload")
    for key in ("project_id", "phase", "state", "pending", "admission_blocked", "blockers"):
        _required(value, key, "upgrade-project payload")
    _required_bool(value, "pending", "upgrade-project payload")
    _required_bool(value, "admission_blocked", "upgrade-project payload")
    _sequence(value["blockers"], "upgrade-project payload.blockers")


def _validate_upgrade_repair(result: Any) -> None:
    value = _mapping(result, "upgrade-repair payload")
    for key in ("repair_id", "target", "state"):
        _required(value, key, "upgrade-repair payload")


def _validate_schema6_upgrade(result: Any) -> None:
    value = _mapping(result, "schema6-upgrade payload")
    _required(value, "phase", "schema6-upgrade payload")


def _validate_batch_submit(result: Any) -> None:
    value = _mapping(result, "batch-submit payload")
    for key in ("operation_id", "idempotency_key", "target_group", "task_ids", "state"):
        _required(value, key, "batch-submit payload")
    _sequence(value["task_ids"], "batch-submit payload.task_ids")


def _validate_context(result: Any) -> None:
    value = _mapping(result, "context payload")
    _required(value, "shared_root", "context payload")


def _validate_progress_policy(result: Any) -> None:
    value = _mapping(result, "progress-policy payload")
    for key in ("interval_seconds", "source", "applies_to"):
        _required(value, key, "progress-policy payload")


def _validate_launch_handoff_policy(result: Any) -> None:
    value = _mapping(result, "launch-handoff-policy payload")
    for key in ("timeout_seconds", "source", "applies_to"):
        _required(value, key, "launch-handoff-policy payload")


def _validate_tmux_policy(result: Any) -> None:
    value = _mapping(result, "tmux-policy payload")
    _required_bool(value, "enabled", "tmux-policy payload")
    source = _required(value, "source", "tmux-policy payload")
    if source not in {"default", "configured"}:
        raise ValueError("tmux-policy payload.source is invalid")
    applies_to = _required(value, "applies_to", "tmux-policy payload")
    if applies_to != "new_observer_decisions":
        raise ValueError("tmux-policy payload.applies_to is invalid")


def _validate_notifications(result: Any) -> None:
    value = _mapping(result, "notifications payload")
    _required_bool(value, "enabled", "notifications payload")
    _required_mapping(value, "providers", "notifications payload")


def _validate_lease_policy(result: Any) -> None:
    value = _mapping(result, "lease-policy payload")
    _required_mapping(value, "lease_policy", "lease-policy payload")


def _validate_doctor_verify(result: Any) -> None:
    value = _mapping(result, "doctor-verify payload")
    for key in ("schema_version", "tasks_checked", "issues", "complete", "healthy"):
        _required(value, key, "doctor-verify payload")
    _sequence(value["issues"], "doctor-verify payload.issues")


def _validate_doctor_repair(result: Any) -> None:
    value = _mapping(result, "doctor-repair payload")
    for key in ("repaired", "blocked", "message"):
        _required(value, key, "doctor-repair payload")
    _sequence(value["repaired"], "doctor-repair payload.repaired")
    _sequence(value["blocked"], "doctor-repair payload.blocked")


def _validate_clean(result: Any) -> None:
    value = _mapping(result, "clean payload")
    for key in ("dry_run", "candidates", "removed", "skipped"):
        _required(value, key, "clean payload")
    _required_bool(value, "dry_run", "clean payload")
    _sequence(value["candidates"], "clean payload.candidates")
    _sequence(value["removed"], "clean payload.removed")
    _mapping(value["skipped"], "clean payload.skipped")


@dataclass(frozen=True, slots=True)
class _OutputContract:
    validator: Validator
    renderer: Renderer


_REGISTRY: dict[OutputKind, _OutputContract] = {
    OutputKind.TASK_LIST: _OutputContract(_validate_task_list, _render_task_list),
    OutputKind.TASK_PAGE: _OutputContract(_validate_task_page, _render_task_page),
    OutputKind.TASK_SHOW: _OutputContract(_validate_task_show, _render_task_show),
    OutputKind.TASK_WATCH: _OutputContract(_validate_task_watch, _render_task_watch),
    OutputKind.TASK_OPERATION: _OutputContract(_validate_task_operation, _render_task_operation),
    OutputKind.DEPENDENCIES: _OutputContract(_validate_dependencies, _render_dependencies),
    OutputKind.AVAILABILITY: _OutputContract(_validate_availability, _render_availability),
    OutputKind.GROUP_LIST: _OutputContract(_validate_group_list, _render_group_list),
    OutputKind.GROUP_SHOW: _OutputContract(_validate_group_show, _render_group_show),
    OutputKind.GROUP_OPERATION: _OutputContract(_validate_group_operation, _render_group_operation),
    OutputKind.GROUP_MACHINES: _OutputContract(_validate_group_machines, _render_group_machines),
    OutputKind.MACHINES: _OutputContract(_validate_machines, _render_machines),
    OutputKind.TOP: _OutputContract(_validate_top, _render_top),
    OutputKind.AGENT_OPERATION: _OutputContract(_validate_agent_operation, _render_agent_operation),
    OutputKind.AGENT_STATUS: _OutputContract(_validate_agent_status, _render_agent_status),
    OutputKind.AGENT_PROJECT_LIST: _OutputContract(_validate_agent_project_list, _render_agent_project_list),
    OutputKind.AGENT_CONFIG: _OutputContract(_validate_agent_config, _render_agent_config),
    OutputKind.AGENT_READINESS: _OutputContract(_validate_agent_readiness, _render_agent_readiness),
    OutputKind.MACHINE_INIT: _OutputContract(_validate_machine_init, _render_machine_init),
    OutputKind.PROJECT_OPERATION: _OutputContract(_validate_project_operation, _render_project_operation),
    OutputKind.PROJECT_REGISTER: _OutputContract(_validate_project_register, _render_project_register),
    OutputKind.PROJECT_LIST: _OutputContract(_validate_project_list, _render_project_list),
    OutputKind.CPU_LANE: _OutputContract(_validate_cpu_lane, _render_cpu_lane),
    OutputKind.GPU_POLICY: _OutputContract(_validate_gpu_policy, _render_gpu_policy),
    OutputKind.UPGRADE_REGISTRY_STATUS: _OutputContract(_validate_upgrade_registry, _render_upgrade_registry),
    OutputKind.UPGRADE_ADVANCE: _OutputContract(_validate_upgrade_advance, _render_upgrade_advance),
    OutputKind.UPGRADE_PROJECT: _OutputContract(_validate_upgrade_project, _render_upgrade_project),
    OutputKind.UPGRADE_REPAIR: _OutputContract(_validate_upgrade_repair, _render_upgrade_repair),
    OutputKind.SCHEMA6_UPGRADE: _OutputContract(_validate_schema6_upgrade, _render_schema6),
    OutputKind.BATCH_SUBMIT: _OutputContract(_validate_batch_submit, _render_batch_submit),
    OutputKind.CONTEXT: _OutputContract(_validate_context, _render_context),
    OutputKind.PROGRESS_POLICY: _OutputContract(_validate_progress_policy, _render_progress_policy),
    OutputKind.LAUNCH_HANDOFF_POLICY: _OutputContract(
        _validate_launch_handoff_policy,
        _render_launch_handoff_policy,
    ),
    OutputKind.TMUX_POLICY: _OutputContract(_validate_tmux_policy, _render_tmux_policy),
    OutputKind.NOTIFICATIONS: _OutputContract(_validate_notifications, _render_notifications),
    OutputKind.LEASE_POLICY: _OutputContract(_validate_lease_policy, _render_lease_policy),
    OutputKind.DOCTOR_VERIFY: _OutputContract(_validate_doctor_verify, _render_doctor),
    OutputKind.DOCTOR_REPAIR: _OutputContract(_validate_doctor_repair, _render_doctor),
    OutputKind.CLEAN: _OutputContract(_validate_clean, _render_clean),
}


def _contract_for(kind: Any) -> _OutputContract:
    if not isinstance(kind, OutputKind):
        raise TypeError(f"unknown CLI output kind {kind!r}")
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"no renderer registered for CLI output kind {kind.value!r}") from exc


if frozenset(_REGISTRY) != frozenset(OutputKind):
    raise RuntimeError("CLI output renderer registry is incomplete")


__all__ = [
    "CliOutput",
    "CpuLanePayload",
    "OutputKind",
    "ProjectListPayload",
    "TaskPagePayload",
    "UpgradeAdvancePayload",
    "UpgradeRegistryStatusPayload",
    "render",
]
