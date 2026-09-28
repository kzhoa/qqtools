"""Pure machine setup, Project enrollment, and saved-context contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def _render_machine_init(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    action = str(result.get("action") or "completed").replace("_", " ")
    return _operation(
        action,
        "completed",
        (
            ("MachineRuntime root", result.get("machine_runtime_root")),
            ("Agent name", result.get("agent_name")),
            ("Agent mode", result.get("agent_mode")),
            ("Runtime ID", result.get("new_runtime_id")),
            ("Old runtime ID", result.get("old_runtime_id")),
            ("Detached", result.get("detached")),
            ("Archive", result.get("archive_path")),
            ("Obligations", result.get("obligations")),
        ),
    )


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


def _render_project_register(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
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
    OutputKind.PROJECT_LIST: OutputContract(_validate_project_list, _render_project_list),
    OutputKind.CONTEXT: OutputContract(_validate_context, _render_context),
}

__all__ = ["CONTRACTS"]
