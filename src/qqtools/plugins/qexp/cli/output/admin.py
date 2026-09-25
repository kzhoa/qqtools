"""Pure administrative and upgrade output contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    aggregate_state = result.get("aggregate_state", "pending" if result.get("projects") else "complete")
    summary = _details(
        (
            ("Action", result.get("action")),
            ("Outcome", result.get("outcome")),
            ("Reason", result.get("reason")),
            ("Next action", result.get("next_action")),
        )
    )
    lines = [item for item in (summary, f"Upgrade state: {aggregate_state}") if item]
    inaccessible = result.get("inaccessible_projects", ())
    if inaccessible:
        lines.append("Inaccessible Projects:")
        for item in inaccessible:
            upgrade = item.get("upgrade", item)
            reason = item.get("reason") or ", ".join(upgrade.get("blockers", ()))
            lines.append(f"- {item.get('project_id')}: {reason}")
    projects = result["projects"]
    if not projects:
        lines.append("No registered Projects." if not inaccessible else "No accessible Projects.")
        return "\n".join(lines)
    lines.append(
        _table(
            ("Project", "Phase", "State", "Pending", "Admission blocked", "Blockers"),
            _upgrade_rows(projects, nested=True),
        )
    )
    return "\n".join(lines)


def _render_upgrade_advance(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    aggregate_state = result.get("aggregate_state", "pending" if result.get("projects") else "complete")
    summary = _details(
        (
            ("Action", result.get("action")),
            ("Outcome", result.get("outcome")),
            ("Reason", result.get("reason")),
            ("Slices", result.get("slices")),
            ("Remaining Projects", result.get("pending_project_ids")),
            ("Worker", result.get("worker_state")),
            ("Next action", result.get("next_action")),
        )
    )
    lines = [item for item in (summary, f"Upgrade state: {aggregate_state}") if item]
    inaccessible = result.get("inaccessible_projects", ())
    for item in inaccessible:
        lines.append(f"Inaccessible: {item.get('project_id')}: {item.get('error') or item.get('reason')}")
    if not result["projects"]:
        lines.append(
            "All registered Projects are complete." if result.get("all_roots_complete") else "No Projects advanced."
        )
        return "\n".join(lines)
    lines.append(
        _table(
            ("Project", "Phase", "State", "Pending", "Admission blocked", "Blockers"),
            _upgrade_rows(result["projects"], nested=False),
        )
    )
    return "\n".join(lines)


def _render_upgrade_project(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action", "upgrade"),
        result.get("outcome", result.get("state", result.get("aggregate_state", "completed"))),
        (
            ("Project", result.get("project_id")),
            ("Phase", result.get("phase")),
            ("Pending", result.get("pending")),
            ("Admission blocked", result.get("admission_blocked")),
            ("Blockers", result.get("blockers")),
            ("Reason", result.get("reason")),
            ("Next action", result.get("next_action")),
        ),
    )


def _render_upgrade_repair(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    return _operation(
        result.get("action", presentation.get("action", "upgrade-repair")),
        result.get("outcome", result["state"]),
        (
            ("Repair ID", result["repair_id"]),
            ("Target", result["target"]),
            ("Reason", result.get("error")),
            ("Next action", result.get("next_action")),
        ),
    )


def _render_schema6(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    preferred = (
        ("Action", result.get("action")),
        ("Outcome", result.get("outcome")),
        ("Phase", result.get("phase")),
        ("Project", result.get("project")),
        ("Source schema", result.get("source_schema", result.get("from_schema"))),
        ("Target schema", result.get("target_schema", result.get("to_schema"))),
        ("Activation ID", result.get("activation_id")),
        ("Capabilities", result.get("capabilities")),
        ("Blockers", result.get("blockers")),
        ("Destructive boundary", result.get("destructive_boundary")),
        ("Reason", result.get("reason")),
        ("Next action", result.get("next_action")),
        ("Error", result.get("error")),
    )
    rendered = _details(preferred)
    if rendered:
        return rendered
    return _details(tuple((str(label).replace("_", " ").capitalize(), value) for label, value in result.items()))


def _render_doctor(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    if "repaired" in result or "blocked" in result:
        scope = result.get("scope", {})
        budget = result.get("budget", {})
        return _details(
            (
                ("Outcome", result.get("outcome")),
                ("Complete", result.get("complete")),
                ("Scope", scope.get("kind") if isinstance(scope, Mapping) else None),
                ("Capture ID", scope.get("capture_id") if isinstance(scope, Mapping) else None),
                ("Work generation", scope.get("work_generation") if isinstance(scope, Mapping) else None),
                ("Phase", result.get("phase")),
                ("Cursor", result.get("cursor")),
                ("Semantic items", budget.get("semantic_items_consumed") if isinstance(budget, Mapping) else None),
                ("Operations", budget.get("operations_consumed") if isinstance(budget, Mapping) else None),
                ("Elapsed ms", budget.get("elapsed_ms") if isinstance(budget, Mapping) else None),
                (
                    "Semantic items remaining",
                    budget.get("semantic_items_remaining") if isinstance(budget, Mapping) else None,
                ),
                ("Operations remaining", budget.get("operations_remaining") if isinstance(budget, Mapping) else None),
                ("Exhaustion reason", budget.get("exhaustion_reason") if isinstance(budget, Mapping) else None),
                ("Deferred phases", result.get("deferred_phases")),
                ("Deferred phase count", result.get("deferred_phase_count")),
                ("Remaining work", result.get("remaining_work")),
                ("Repaired", result.get("repaired_count", len(result.get("repaired", ())))),
                ("Blocked", result.get("blocked_count", len(result.get("blocked", ())))),
                ("Rerun required", result.get("rerun_required")),
                ("Next action", result.get("next_action")),
                ("Message", result.get("message")),
            )
        )
    if "healthy" in result or "complete" in result:
        issues = result.get("issues", ())
        issue_rows = []
        if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes, bytearray)):
            for issue in issues[:20]:
                if isinstance(issue, Mapping):
                    issue_rows.append(
                        (
                            issue.get("severity"),
                            issue.get("code"),
                            issue.get("path"),
                            issue.get("message"),
                        )
                    )
        summary = _details(
            (
                ("Outcome", result.get("outcome", "healthy" if result.get("healthy") else "partial")),
                ("Complete", result.get("complete")),
                ("Healthy", result.get("healthy")),
                ("Tasks checked", result.get("tasks_checked")),
                ("Issue counts", result.get("issue_counts")),
                ("Incomplete reasons", result.get("incomplete_reasons")),
                ("Repair command", result.get("repair_command")),
            )
        )
        if not issue_rows:
            return summary
        return "\n\n".join(
            (
                summary,
                _table(("Severity", "Issue", "Path", "Message"), issue_rows, empty_message="No issues."),
            )
        )
    return _render_named_operation(result, "doctor")


def _render_clean(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    lines = [
        _details(
            (
                ("Outcome", result.get("outcome")),
                ("Deletion", "none (dry run)" if result.get("dry_run") else result.get("deletion", "requested")),
                ("Deletion performed", result.get("deletion_performed")),
                ("Candidates", result.get("candidate_count", len(result.get("candidates", ())))),
                ("Removed", result.get("removed_count", 0)),
                ("Pending", result.get("pending_count", 0)),
                ("Skipped", result.get("skipped_count", len(result.get("skipped", {})))),
                ("Blocked", result.get("blocked_count", 0)),
            )
        )
    ]
    if result.get("candidates"):
        lines.append(_details((("Candidate Task IDs", result.get("candidates")),)))
    if result.get("removed_task_ids"):
        lines.append(_details((("Removed Task IDs", result.get("removed_task_ids")),)))
    if result.get("pending_task_ids"):
        lines.append(_details((("Pending Task IDs", result.get("pending_task_ids")),)))
    if result.get("skipped_task_ids"):
        lines.append(_details((("Skipped Task IDs", result.get("skipped_task_ids")),)))
    operations = result.get("operations")
    if isinstance(operations, Mapping):
        operation_rows = []
        for task_id, operation in operations.items():
            if not isinstance(operation, Mapping):
                continue
            operation_rows.append(
                (
                    task_id,
                    operation.get("state"),
                    operation.get("operation_reference"),
                    operation.get("follow_up_command"),
                )
            )
        if operation_rows:
            lines.append(
                _table(
                    ("Task", "State", "Operation", "Follow-up"),
                    operation_rows,
                    empty_message="No cleanup operations.",
                )
            )
    return "\n\n".join(item for item in lines if item)


def _render_operation(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    return _details(
        (
            ("Reference", result.get("reference")),
            ("Kind", result.get("kind")),
            ("Operation ID", result.get("operation_id")),
            ("Target", result.get("target")),
            ("State", result.get("state")),
            ("Outcome", result.get("lifecycle_outcome", result.get("outcome"))),
            ("Progress", result.get("progress")),
            ("Blockers", result.get("blockers")),
            ("Pending machines", result.get("pending_machines")),
            ("Created", result.get("created_at")),
            ("Updated", result.get("updated_at")),
            ("Completed", result.get("completed_at")),
            ("Next action", result.get("next_action")),
            ("Error", result.get("error")),
        )
    )


def _validate_upgrade_registry(result: Any) -> None:
    value = _mapping(result, "upgrade-registry-status payload")
    projects = _required_sequence(value, "projects", "upgrade-registry-status payload")
    if "inaccessible_projects" in value:
        _sequence(value["inaccessible_projects"], "upgrade-registry-status payload.inaccessible_projects")
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
    if "inaccessible_projects" in value:
        _sequence(value["inaccessible_projects"], "upgrade-advance payload.inaccessible_projects")
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


def _validate_doctor_verify(result: Any) -> None:
    value = _mapping(result, "doctor-verify payload")
    for key in ("schema_version", "tasks_checked", "issues", "complete", "healthy"):
        _required(value, key, "doctor-verify payload")
    _sequence(value["issues"], "doctor-verify payload.issues")


def _validate_doctor_repair(result: Any) -> None:
    value = _mapping(result, "doctor-repair payload")
    # Older in-process callers may still construct the pre-resumable payload.
    # The CLI producer always emits the additive contract below.
    if "complete" not in value:
        for key in ("repaired", "blocked", "message"):
            _required(value, key, "doctor-repair payload")
        _sequence(value["repaired"], "doctor-repair payload.repaired")
        _sequence(value["blocked"], "doctor-repair payload.blocked")
        return
    for key in (
        "repaired",
        "blocked",
        "message",
        "complete",
        "scope",
        "budget",
        "phase",
        "cursor",
        "deferred_phases",
        "deferred_phase_count",
        "next_due_at",
        "remaining_work",
    ):
        _required(value, key, "doctor-repair payload")
    _sequence(value["repaired"], "doctor-repair payload.repaired")
    _sequence(value["blocked"], "doctor-repair payload.blocked")
    _required_bool(value, "complete", "doctor-repair payload")
    scope = _required_mapping(value, "scope", "doctor-repair payload")
    for key in ("kind", "capture_id", "work_generation"):
        _required(scope, key, "doctor-repair payload.scope")
    if scope["kind"] != "full_audit":
        raise ValueError("doctor-repair payload.scope.kind must be 'full_audit'")
    for key in ("capture_id", "work_generation"):
        if scope[key] is not None and (not isinstance(scope[key], str) or not scope[key]):
            raise TypeError(f"doctor-repair payload.scope.{key} must be a non-empty string or null")
    budget = _required_mapping(value, "budget", "doctor-repair payload")
    for key in (
        "semantic_items_consumed",
        "operations_consumed",
        "elapsed_ms",
        "semantic_items_remaining",
        "operations_remaining",
    ):
        _required_int(budget, key, "doctor-repair payload.budget")
    exhaustion_reason = _required(budget, "exhaustion_reason", "doctor-repair payload.budget")
    if exhaustion_reason not in {None, "semantic_items", "operations", "deadline"}:
        raise ValueError("doctor-repair payload.budget.exhaustion_reason is invalid")
    if not isinstance(value["phase"], str) or not value["phase"]:
        raise TypeError("doctor-repair payload.phase must be a non-empty string")
    _required_mapping(value, "cursor", "doctor-repair payload")
    deferred = _required_sequence(value, "deferred_phases", "doctor-repair payload")
    if not all(isinstance(phase, str) and phase for phase in deferred):
        raise TypeError("doctor-repair payload.deferred_phases must contain non-empty strings")
    deferred_count = _required_int(value, "deferred_phase_count", "doctor-repair payload")
    if deferred_count < len(deferred):
        raise ValueError("doctor-repair payload.deferred_phase_count is smaller than its detail list")
    if value["next_due_at"] is not None and not isinstance(value["next_due_at"], str):
        raise TypeError("doctor-repair payload.next_due_at must be a string or null")


def _validate_clean(result: Any) -> None:
    value = _mapping(result, "clean payload")
    for key in ("dry_run", "candidates", "removed", "skipped"):
        _required(value, key, "clean payload")
    _required_bool(value, "dry_run", "clean payload")
    _sequence(value["candidates"], "clean payload.candidates")
    _sequence(value["removed"], "clean payload.removed")
    _mapping(value["skipped"], "clean payload.skipped")


def _validate_operation(result: Any) -> None:
    value = _mapping(result, "operation payload")
    for key in ("reference", "kind", "operation_id", "outcome", "error"):
        _required(value, key, "operation payload")


CONTRACTS = {
    OutputKind.UPGRADE_REGISTRY_STATUS: OutputContract(_validate_upgrade_registry, _render_upgrade_registry),
    OutputKind.UPGRADE_ADVANCE: OutputContract(_validate_upgrade_advance, _render_upgrade_advance),
    OutputKind.UPGRADE_PROJECT: OutputContract(_validate_upgrade_project, _render_upgrade_project),
    OutputKind.UPGRADE_REPAIR: OutputContract(_validate_upgrade_repair, _render_upgrade_repair),
    OutputKind.SCHEMA6_UPGRADE: OutputContract(_validate_schema6_upgrade, _render_schema6),
    OutputKind.DOCTOR_VERIFY: OutputContract(_validate_doctor_verify, _render_doctor),
    OutputKind.DOCTOR_REPAIR: OutputContract(_validate_doctor_repair, _render_doctor),
    OutputKind.CLEAN: OutputContract(_validate_clean, _render_clean),
    OutputKind.OPERATION: OutputContract(_validate_operation, _render_operation),
}

__all__ = ["CONTRACTS", "_validate_upgrade_registry"]
