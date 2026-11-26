"""Pure machine setup, Project enrollment, and saved-context contracts."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from ...takeover_contract import TAKEOVER_CANCELLABLE_PHASES, TAKEOVER_PHASES, TAKEOVER_PLAN_ID
from ..project_presentation import encode_display_text
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


def _render_machine_init(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    root = str(result["machine_runtime_root"])
    replaced = result["action"] == "reinitialized"
    lines = [
        "Machine identity replaced." if replaced else "Machine identity initialized.",
        f"Machine name: {encode_display_text(str(result['agent_name']))}",
        f"MachineRuntime root: {encode_display_text(root)}",
        f"Agent mode: {result['agent_mode'].replace('_', ' ')}",
    ]
    if replaced:
        lines.append("The new runtime identity replaces the old local authority.")
        lines.append("Shared Project records, Tasks, and claims are preserved.")
        if result.get("detached"):
            lines.append("Old local evidence was detached in an isolated archive.")
        if archive := result.get("archive_path"):
            lines.append(f"Archive: {encode_display_text(archive)}")
        for obligation in result.get("obligations") or ():
            lines.append(f"  Unresolved obligation: {encode_display_text(obligation)}")
        return "\n".join(lines)
    lines.append("No Project was registered and no agent was started.")
    project = presentation.get("project_path")
    if isinstance(project, str):
        lines.extend((f"Project: {encode_display_text(project)}", "Next, register this Project on the new runtime."))
    else:
        project = "PROJECT_PATH"
        lines.extend(
            (
                "Replace PROJECT_PATH with the intended Project directory.",
                "For a new Project, initialize it first:",
                "  qexp project init PROJECT_PATH",
                "Register the initialized Project on this runtime:",
            )
        )
    command = shlex.join(("qexp", "--machine-runtime-root", root, "project", "register", project))
    return "\n".join((*lines, f"  {command}"))


def _render_project_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    action = str(result.get("action") or "project operation")
    status = result.get("status", result.get("state"))
    if action == "project_already_initialized":
        status = "already initialized"
    elif status is None:
        status = "completed"
    lines = [
        _operation(
            action.replace("_", " "),
            status,
            (
                ("Project ID", result.get("project_id")),
                ("Shared root", result.get("shared_root")),
                ("Machine", result.get("machine_name")),
                ("Enabled", result.get("enabled")),
                ("Requested enabled", result.get("requested_enabled")),
                ("Effective enabled", result.get("effective_enabled")),
                ("Registry revision", result.get("registry_revision")),
                ("Inventory revision", result.get("inventory_revision")),
                ("Inventory converged", result.get("inventory_converged")),
                ("Source", result.get("name_source")),
                ("Local only", result.get("local_only")),
                ("Reason", result.get("reason")),
            ),
        )
    ]
    if status == "partially_committed":
        lines.append(
            "Next: retry the same project enable/disable command; the registry choice is already effective "
            "and retry repairs the inventory mirror."
        )
    elif status == "outcome_unknown":
        lines.append(
            "Next: inspect a fresh qexp project list, then retry the same project enable/disable request "
            "to establish convergence. Do not issue the inverse request as rollback."
        )
    return "\n".join(lines)


def _render_project_registration_replacement(result: Mapping[str, Any]) -> str:
    lines = [
        _operation(
            "project registration replacement",
            result.get("status"),
            (
                ("Project ID", result.get("project_id")),
                ("Shared root", result.get("shared_root")),
                ("Runtime ID", result.get("runtime_id")),
                ("Old machine", result.get("old_machine_name")),
                ("Requested machine", result.get("requested_machine_name")),
                ("Phase", result.get("phase")),
                ("Generation", result.get("registration_generation")),
                ("Cancellable", result.get("cancellable")),
                ("Blockers", result.get("blockers")),
            ),
        )
    ]
    actions = result.get("next_actions") or result.get("follow_up_actions") or []
    if actions:
        lines.append("Next: " + " or ".join(str(action) for action in actions))
    return "\n".join(lines)


def _render_project_register(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    if result.get("action") == "project_registration_replacement":
        return _render_project_registration_replacement(result)
    projects = result.get("projects", ())
    successful = [project for project in projects if project.get("status") in {"registered", "disabled"}]
    failed = [project for project in projects if project not in successful]
    summary = f"Project register: {len(successful)} succeeded, {len(failed)} failed"
    if not projects:
        return summary
    table = _table(
        ("Project ID", "Shared root", "Machine", "Status", "Reason"),
        [
            (
                project.get("project_id"),
                project.get("shared_root"),
                project.get("machine_name"),
                project.get("status"),
                project.get("reason"),
            )
            for project in projects
        ],
        empty_message="No Project entries.",
    )
    lines = [summary, table]
    if failed:
        lines.append("Next: qexp project register PATH")
    return "\n".join(lines)


def _render_project_takeover(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    if result["action"] == "project_takeover_operation":
        return _render_takeover_operation(result, presentation)
    binding = result["current_binding"] or {}
    target = result["target_registration"]
    preview = _operation(
        "project takeover registration preview",
        result["status"],
        (
            ("Plan ID", result["plan_id"]),
            ("Project ID", result["project_id"]),
            ("Shared root", result["shared_root"]),
            ("Machine runtime root", result["machine_runtime_root"]),
            ("Runtime ID", result["runtime_id"]),
            ("Current local binding", binding.get("machine_name", "none")),
            ("Current registration generation", binding.get("registration_generation")),
            ("Current binding enabled", binding.get("enabled")),
            ("Target machine", target["machine_name"]),
            ("Target registration state", target["state"]),
            ("Target registration generation", target["generation"]),
            ("Target runtime ID", target["runtime_id"]),
            ("Target runtime root", target["runtime_root"]),
            ("Target eligibility expires at", target["eligibility_expires_at"]),
            ("Blockers", [blocker["message"] for blocker in result["blockers"]]),
        ),
    )
    changes = result["selected_group_changes"]
    selected_groups = _sequence(
        presentation.get("selected_groups", [change["group_name"] for change in changes]), "Selected Groups"
    )
    sections = [preview]
    if changes:
        sections.append(
            "Selected primary worker additions\n"
            + _table(
                ("Group", "Primary worker", "Affected Tasks"),
                [(change["group_name"], change["machine_name"], change["affected_task_count"]) for change in changes],
            )
        )
    sections.extend(
        (_render_takeover_recovery(result, selected_groups), _render_takeover_tasks(result, selected_groups))
    )
    if plan_id := result["plan_id"]:
        command = _takeover_command(
            result,
            target["machine_name"],
            f"--apply={plan_id}",
            *(f"--add-primary-worker={group}" for group in selected_groups),
        )
        sections.append("Confirm: " + command)
    return "\n\n".join(section for section in sections if section)


def _takeover_command(result: Mapping[str, Any], target_machine: str, *options: str) -> str:
    return shlex.join(
        [
            "qexp",
            "--machine-runtime-root",
            result["machine_runtime_root"],
            "project",
            "takeover",
            result["shared_root"],
            f"--machine={target_machine}",
            *options,
        ]
    )


def _takeover_preview_continuation(
    result: Mapping[str, Any], selected_groups: Sequence[str], option: str, value: str
) -> str:
    return _takeover_command(
        result,
        result["target_registration"]["machine_name"],
        "--plan",
        *(f"--add-primary-worker={group}" for group in selected_groups),
        f"{option}={value}",
    )


def _render_takeover_operation(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    plan_id = result["plan_id"]
    phase = result["phase"]
    target = result["target_machine"]
    approvals = result["group_approvals"]
    group_completion = (
        "partial completion"
        if phase == "completed" and any(group["outcome"] == "stale" for group in approvals)
        else None
    )
    lines = [
        _operation(
            "project takeover operation",
            phase,
            (
                ("Plan ID", plan_id),
                ("Cancellation requested", result["is_cancel_requested"]),
                ("Target machine", target),
                ("Departing machine", result["departing_machine"] or "none"),
                ("Group additions", group_completion),
                ("Agent activation error", result["agent_start_error"]),
                ("Current issue", result["operation_issue"]),
            ),
        )
    ]
    if approvals:
        lines.append(
            _table(
                ("Group", "Primary worker", "Affected Tasks", "Outcome"),
                [
                    (group["group_name"], group["machine_name"], group["affected_task_count"], group["outcome"])
                    for group in approvals
                ],
            )
        )
    for approval in approvals:
        if approval["outcome"] == "stale":
            lines.append(
                "Review changed Group: "
                + shlex.join(
                    [
                        "qexp",
                        "--machine-runtime-root",
                        result["machine_runtime_root"],
                        "--project",
                        result["shared_root"],
                        "group",
                        "show",
                        approval["group_name"],
                    ]
                )
            )
    lines.append("Operation status: " + _takeover_command(result, target, f"--operation-status={plan_id}"))
    if phase in TAKEOVER_CANCELLABLE_PHASES or phase == "cancelling":
        lines.append("Request cancellation: " + _takeover_command(result, target, f"--cancel={plan_id}"))
    retry_option = presentation.get("retry_option")
    if result["agent_start_error"] is not None and retry_option in ("--apply", "--cancel"):
        groups = (
            [f"--add-primary-worker={group['group_name']}" for group in approvals] if retry_option == "--apply" else []
        )
        lines.append("Retry: " + _takeover_command(result, target, f"{retry_option}={plan_id}", *groups))
    return "\n".join(lines)


def _render_takeover_recovery(result: Mapping[str, Any], selected_groups: Sequence[str]) -> str:
    recovery = result["departing_recovery"]
    if recovery is None:
        return ""
    count = recovery["count"]
    if count is None:
        return "Local recovery ledger: unavailable"
    lines = [f"Local recovery ledger records: {count}"]
    if recovery["records"]:
        lines.append(
            _table(
                ("Record", "Kind", "Task", "Attempt", "Cleanup receipt recorded"),
                [
                    (
                        item["identity"],
                        item["kind"].replace("_", " "),
                        item["task_id"],
                        item["attempt_number"],
                        item["cleanup_receipt_recorded"],
                    )
                    for item in recovery["records"]
                ],
            )
        )
    if cursor := recovery["next_cursor"]:
        lines.append(
            "More recovery records: "
            + _takeover_preview_continuation(result, selected_groups, "--recovery-cursor", cursor)
        )
    return "\n".join(lines)


def _render_takeover_tasks(result: Mapping[str, Any], selected_groups: Sequence[str]) -> str:
    impact = result["task_impact"]
    if impact is None:
        return ""
    if impact["status"] == "unavailable":
        return "Target Task impact: unavailable"

    rows = []
    for task in impact["tasks"]:
        attempt = task["attempt"]
        attempt_details = None
        if attempt is not None:
            attempt_details = (
                f"{attempt['attempt_id']} on {attempt['machine_name']!r}, "
                f"{attempt['phase']}, process {attempt['process_state'].replace('_', ' ')}"
            )
            if attempt["reservation_id"]:
                attempt_details += f", reservation {attempt['reservation_id']!r}"
            if attempt["stop_pending"]:
                attempt_details += ", stop pending"
            if attempt["exit_code"] is not None:
                attempt_details += f", exit {attempt['exit_code']}"
        if attempt_details is None and task["claim_machine"]:
            attempt_details = f"Claim on {task['claim_machine']!r}, process unknown"
        cancellation = task["cancellation"]
        cancellation_text = None
        if cancellation is not None:
            cancellation_text = "Acknowledged" if cancellation["acknowledged_at"] else "Requested"
            if cancellation["terminate_running"]:
                cancellation_text += ", includes stop"
        placement = task["placement"]
        placement_details = None
        if placement is not None:
            placement_details = placement["state"]
            if placement["reason"]:
                placement_details += f" ({placement['reason'].replace('_', ' ')})"
            if change := placement["group_change"]:
                placement_details += f". Proposed: add {change['machine_name']!r} as primary"
        rows.append(
            (
                task["task_id"],
                task["group_name"],
                task["phase"],
                task["disposition"].replace("_", " "),
                attempt_details,
                cancellation_text,
                placement_details,
            )
        )
    lines = [f"Target Tasks observed: {impact['count']}"]
    if rows:
        lines.append(
            _table(("Task", "Group", "Phase", "Disposition", "Attempt evidence", "Cancellation", "Placement"), rows)
        )
        if any(task["disposition"] == "review_interrupted_attempt" for task in impact["tasks"]):
            lines.append("Review interrupted Attempts before deciding whether to retry.")
    if after := impact["next_task"]:
        lines.append("More Tasks: " + _takeover_preview_continuation(result, selected_groups, "--tasks-after", after))
    return "\n".join(lines)


def _render_project_list(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    projects = result.get("projects", ())
    if not projects:
        return "No enrolled Projects.\nNext: qexp project register PATH"
    rows = [
        (
            project.get("project_id"),
            project.get("shared_root"),
            project.get("machine_name"),
            project.get("status"),
            project.get("effective_enabled"),
            project.get("inventory_enabled"),
            project.get("inventory_converged"),
            project.get("mount_available"),
        )
        for project in projects
    ]
    lines = [
        f"Registry revision: {result.get('registry_revision')}; inventory revision: {result.get('inventory_revision')}",
        _table(
            (
                "Project ID",
                "Shared root",
                "Machine",
                "Status",
                "Effective enabled",
                "Inventory enabled",
                "Inventory converged",
                "Mount",
            ),
            rows,
            empty_message="No enrolled Projects.",
        ),
    ]
    if any(not project.get("inventory_converged", True) for project in projects):
        lines.append("Warning: registry enablement is authoritative; inventory mirror repair is incomplete or blocked.")
    return "\n".join(lines)


def _render_context(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    action = result.get("action")
    if action == "selected":
        state = "selected" if result.get("changed", True) else "already selected"
        return _details((("Context", state), ("Project", result.get("shared_root"))))
    if action == "cleared":
        state = "cleared" if result.get("changed", False) else "already clear"
        return _details((("Context", state),))
    return f"shared_root: {result.get('shared_root') or '<not set>'}"


def _validate_machine_init(result: Any) -> None:
    value = _mapping(result, "machine-init payload")
    if _required(value, "action", "machine-init payload") not in {"initialized", "reinitialized"}:
        raise ValueError("machine-init payload.action is invalid")
    for key in (
        "machine_runtime_root",
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
    action = _required(value, "action", "project-operation payload")
    if "project_id" in value:
        _required(value, "shared_root", "project-operation payload")
    if action in {"project_enabled", "project_disabled"}:
        for key in (
            "status",
            "requested_enabled",
            "effective_enabled",
            "registry_revision",
            "inventory_revision",
            "inventory_converged",
            "reason",
        ):
            _required(value, key, "project-operation payload")
        if value["status"] not in {"committed", "partially_committed", "not_committed", "outcome_unknown"}:
            raise ValueError("project-operation payload.status is invalid")
        _required_bool(value, "requested_enabled", "project-operation payload")
        if value["effective_enabled"] is not None and type(value["effective_enabled"]) is not bool:
            raise ValueError("project-operation payload.effective_enabled must be a bool or null")
        for key in ("registry_revision", "inventory_revision"):
            revision = value[key]
            if revision is not None and (type(revision) is not int or revision < 0):
                raise ValueError(f"project-operation payload.{key} must be a nonnegative integer or null")
        _required_bool(value, "inventory_converged", "project-operation payload")
        if value["reason"] is not None and not isinstance(value["reason"], str):
            raise ValueError("project-operation payload.reason must be a string or null")


def _validate_project_registration_replacement(value: Mapping[str, Any]) -> None:
    label = "project-registration-replacement payload"
    if _required(value, "action", label) != "project_registration_replacement":
        raise ValueError(f"{label}.action is invalid")
    for key in (
        "status",
        "project_id",
        "shared_root",
        "runtime_id",
        "runtime_root",
        "old_machine_name",
        "old_generation",
        "requested_machine_name",
        "target_generation",
        "phase",
        "blockers",
        "next_actions",
        "follow_up_actions",
    ):
        _required(value, key, label)
    if value["status"] not in {
        "replacement_pending",
        "replacement_completed",
        "replacement_cancelled",
        "unchanged",
    }:
        raise ValueError(f"{label}.status is invalid")
    _required_bool(value, "cancellable", label)
    for key in ("blockers", "next_actions", "follow_up_actions"):
        _required_sequence(value, key, label)
    if value["status"] == "replacement_completed":
        _required(value, "registration_generation", label)


def _validate_project_register(result: Any) -> None:
    value = _mapping(result, "project-register payload")
    if value.get("action") == "project_registration_replacement":
        _validate_project_registration_replacement(value)
        return
    _required_int(value, "revision", "project-register payload")
    projects = _required_sequence(value, "projects", "project-register payload")
    for index, item in enumerate(projects):
        project = _mapping(item, f"project-register payload.projects[{index}]")
        for key in ("project_id", "shared_root", "enabled", "name_source", "status"):
            _required(project, key, f"project-register payload.projects[{index}]")


def _validate_takeover_strings(value: Mapping[str, Any], label: str, *keys: str) -> None:
    for key in keys:
        field = _required(value, key, label)
        if not isinstance(field, str) or not field:
            raise ValueError(f"{label}.{key} must be a non-empty string")


def _validate_takeover_group_changes(changes: Sequence[Any], label: str) -> None:
    for index, item in enumerate(changes):
        change_label = f"{label}[{index}]"
        change = _mapping(item, change_label)
        if _required(change, "action", change_label) != "add_primary_worker":
            raise ValueError(f"{change_label}.action is invalid")
        for key in ("group_name", "machine_name"):
            name = _required(change, key, change_label)
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
                raise ValueError(f"{change_label}.{key} must be an identifier")
        for key, minimum in (("group_revision", 0), ("worker_set_epoch", 0), ("affected_task_count", 1)):
            if _required_int(change, key, change_label) < minimum:
                raise ValueError(f"{change_label}.{key} must be at least {minimum}")
        digest = _required(change, "affected_tasks_digest", change_label)
        if not isinstance(digest, str) or not TAKEOVER_PLAN_ID.fullmatch(digest):
            raise ValueError(f"{change_label}.affected_tasks_digest must be 64 lowercase hexadecimal characters")


def _validate_takeover_operation(value: Mapping[str, Any], label: str) -> None:
    _validate_takeover_strings(
        value, label, "plan_id", "phase", "project_id", "shared_root", "machine_runtime_root", "target_machine"
    )
    if not TAKEOVER_PLAN_ID.fullmatch(value["plan_id"]):
        raise ValueError(f"{label}.plan_id must be 64 lowercase hexadecimal characters")
    if value["phase"] not in TAKEOVER_PHASES:
        raise ValueError(f"{label}.phase is invalid")
    _required_bool(value, "is_cancel_requested", label)
    for key in ("departing_machine", "agent_start_error", "operation_issue"):
        field = _required(value, key, label)
        if field is not None and (not isinstance(field, str) or not field):
            raise ValueError(f"{label}.{key} must be a non-empty string or null")
    group_label = f"{label}.group_approvals"
    approvals = _required_sequence(value, "group_approvals", label)
    _validate_takeover_group_changes(approvals, group_label)
    for index, approval in enumerate(approvals):
        approval_label = f"{group_label}[{index}]"
        if _required(approval, "outcome", approval_label) not in (None, "applied", "stale"):
            raise ValueError(f"{approval_label}.outcome is invalid")


def _validate_project_takeover(result: Any) -> None:
    label = "project-takeover payload"
    value = _mapping(result, label)
    action = _required(value, "action", label)
    if action == "project_takeover_operation":
        _validate_takeover_operation(value, label)
        return
    if action != "project_takeover_preview":
        raise ValueError(f"{label}.action is invalid")
    _validate_takeover_strings(
        value, label, "status", "project_id", "shared_root", "runtime_id", "machine_runtime_root"
    )
    blockers = _required_sequence(value, "blockers", label)
    if value["status"] != ("blocked" if blockers else "preview"):
        raise ValueError(f"{label}.status must match its blockers")
    plan_id = _required(value, "plan_id", label)
    changes = _required_sequence(value, "selected_group_changes", label)
    if blockers:
        if plan_id is not None or changes:
            raise ValueError(f"{label} must have a null plan_id and no selected_group_changes when blocked")
    elif not isinstance(plan_id, str) or not TAKEOVER_PLAN_ID.fullmatch(plan_id):
        raise ValueError(f"{label}.plan_id must be 64 lowercase hexadecimal characters")
    _validate_takeover_group_changes(changes, f"{label}.selected_group_changes")
    for index, item in enumerate(blockers):
        blocker_label = f"{label}.blockers[{index}]"
        _validate_takeover_strings(_mapping(item, blocker_label), blocker_label, "code", "message")

    binding = _required(value, "current_binding", label)
    if binding is not None:
        binding_label = f"{label}.current_binding"
        binding = _mapping(binding, binding_label)
        _validate_takeover_strings(binding, binding_label, "machine_name", "registration_generation")
        _required_bool(binding, "enabled", binding_label)

    target_label = f"{label}.target_registration"
    target = _required_mapping(value, "target_registration", label)
    _validate_takeover_strings(target, target_label, "machine_name", "state")
    fields = ("generation", "runtime_id", "runtime_root", "eligibility_expires_at")
    if target["state"] == "missing":
        for key in fields:
            if _required(target, key, target_label) is not None:
                raise ValueError(f"{target_label}.{key} must be null for a missing registration")
    elif target["state"] in {"eligible", "expired", "superseded"}:
        _validate_takeover_strings(target, target_label, *fields)
    else:
        raise ValueError(f"{target_label}.state is invalid")
    recovery = _required(value, "departing_recovery", label)
    if recovery is not None:
        recovery_label = f"{label}.departing_recovery"
        recovery = _mapping(recovery, recovery_label)
        state = _required(recovery, "status", recovery_label)
        count = _required(recovery, "count", recovery_label)
        cursor = _required(recovery, "next_cursor", recovery_label)
        records = _required_sequence(recovery, "records", recovery_label)
        if state == "observed":
            if type(count) is not int or not 0 <= len(records) <= count:
                raise ValueError(f"{recovery_label}.count is invalid")
            if cursor is not None and (not isinstance(cursor, str) or not cursor):
                raise ValueError(f"{recovery_label}.next_cursor is invalid")
        elif (
            state not in {"absent", "unavailable"}
            or count != (0 if state == "absent" else None)
            or records
            or cursor is not None
        ):
            raise ValueError(f"{recovery_label}.status and contents are inconsistent")
        for index, record in enumerate(records):
            record_label = f"{recovery_label}.records[{index}]"
            record = _mapping(record, record_label)
            _validate_takeover_strings(record, record_label, "identity", "kind")
            _required(record, "task_id", record_label)
            _required(record, "attempt_number", record_label)
            _required_bool(record, "cleanup_receipt_recorded", record_label)
    elif binding is not None:
        raise ValueError(f"{label}.departing_recovery is missing for the local binding")

    impact = _required(value, "task_impact", label)
    if impact is not None:
        _validate_takeover_tasks(impact)


def _validate_takeover_tasks(value: Any) -> None:
    label = "project-takeover payload.task_impact"
    impact = _mapping(value, label)
    status = _required(impact, "status", label)
    count = _required(impact, "count", label)
    tasks = _required_sequence(impact, "tasks", label)
    after = _required(impact, "next_task", label)
    if status == "unavailable" and count is None and not tasks and after is None:
        return
    if status != "observed" or type(count) is not int or not 0 <= len(tasks) <= count:
        raise ValueError(f"{label}.status and contents are inconsistent")
    if after is not None and (not isinstance(after, str) or not after):
        raise ValueError(f"{label}.next_task is invalid")
    for index, record in enumerate(tasks):
        task_label = f"{label}.tasks[{index}]"
        task = _mapping(record, task_label)
        _validate_takeover_strings(task, task_label, "task_id", "phase", "home_machine", "disposition")
        for key in ("group_name", "claim_machine", "cancellation", "attempt", "placement"):
            _required(task, key, task_label)


def _validate_project_list(result: Any) -> None:
    value = _mapping(result, "project-list payload")
    _required_int(value, "revision", "project-list payload")
    _required_int(value, "registry_revision", "project-list payload")
    _required_int(value, "inventory_revision", "project-list payload")
    projects = _required_sequence(value, "projects", "project-list payload")
    for index, item in enumerate(projects):
        project = _mapping(item, f"project-list payload.projects[{index}]")
        for key in ("project_id", "shared_root", "enabled", "name_source", "status", "mount_available"):
            _required(project, key, f"project-list payload.projects[{index}]")
        _required_bool(project, "enabled", f"project-list payload.projects[{index}]")
        _required_bool(project, "mount_available", f"project-list payload.projects[{index}]")
        if project.get("effective_enabled") is not None and type(project["effective_enabled"]) is not bool:
            raise ValueError(f"project-list payload.projects[{index}].effective_enabled must be a bool or null")
        _required_bool(project, "inventory_enabled", f"project-list payload.projects[{index}]")
        _required_bool(project, "inventory_converged", f"project-list payload.projects[{index}]")
        for key in ("registry_revision", "inventory_revision"):
            revision = project.get(key)
            if type(revision) is not int or revision < 0:
                raise ValueError(f"project-list payload.projects[{index}].{key} must be a nonnegative integer")


def _validate_context(result: Any) -> None:
    value = _mapping(result, "context payload")
    _required(value, "shared_root", "context payload")


CONTRACTS = {
    OutputKind.MACHINE_INIT: OutputContract(_validate_machine_init, _render_machine_init),
    OutputKind.PROJECT_OPERATION: OutputContract(_validate_project_operation, _render_project_operation),
    OutputKind.PROJECT_REGISTER: OutputContract(_validate_project_register, _render_project_register),
    OutputKind.PROJECT_TAKEOVER: OutputContract(_validate_project_takeover, _render_project_takeover),
    OutputKind.PROJECT_LIST: OutputContract(_validate_project_list, _render_project_list),
    OutputKind.CONTEXT: OutputContract(_validate_context, _render_context),
}

__all__ = ["CONTRACTS"]
