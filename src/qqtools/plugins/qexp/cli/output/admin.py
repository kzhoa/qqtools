"""Pure administrative and upgrade output contracts."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from ...upgrade_progress import validate_progress
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
    _value,
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


def _upgrade_status(project: Mapping[str, Any], *, nested: bool) -> Mapping[str, Any]:
    status = project.get("upgrade", {}) if nested else project
    return status if isinstance(status, Mapping) else {}


def _validated_progress(status: Mapping[str, Any]) -> Mapping[str, Any] | None:
    progress = status.get("progress")
    if progress is None:
        return None
    try:
        validated = validate_progress(progress)
    except (KeyError, TypeError, ValueError):
        return None
    return validated if isinstance(validated, Mapping) else None


def _upgrade_words(value: object, *, title: bool = False) -> str:
    text = str(value).replace("_", " ").strip()
    return text.title() if title else text


def _singular_word(value: object) -> str:
    text = _upgrade_words(value)
    if len(text) > 1 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def _truncate_upgrade_text(value: object, limit: int) -> str:
    text = _value(value)
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return f"{text[: limit - 1]}…"


def _wrap_upgrade_detail(text: str) -> list[str]:
    return textwrap.wrap(
        text,
        width=120,
        break_long_words=True,
        break_on_hyphens=False,
        subsequent_indent="  ",
    ) or [""]


def _progress_summary(status: Mapping[str, Any]) -> str:
    if status.get("state") in {"complete", "completed"}:
        return "complete"
    progress = _validated_progress(status)
    if progress is None:
        return "unavailable"
    stage = _upgrade_words(progress.get("stage"), title=False)
    stage_state = progress.get("stage_state")
    completed = progress.get("completed_units")
    total = progress.get("total_units")
    total_kind = progress.get("total_kind")
    if stage_state == "complete" and (completed is None or total_kind != "snapshot_exact"):
        return f"{stage} stage complete"
    if stage_state == "inventory":
        counted = progress.get("inventoried_units")
        counted_text = str(counted) if type(counted) is int else "unavailable"
        return f"{stage} inventory, {counted_text} entries counted; total pending"
    if stage_state in {"restarted", "recounting"} or completed is None:
        return f"{stage} scan restarted; completed count unavailable; recounting"
    if total_kind == "snapshot_exact" and type(total) is int:
        if total > 0 and type(completed) is int:
            percent = completed * 100 / total
            return f"{stage} {completed}/{total} ({percent:.1f}%)"
        if type(completed) is int:
            return f"{stage} {completed}/{total}"
    epoch = progress.get("scan_epoch")
    return f"{stage} {completed} processed in scan {epoch}; total pending"


def _progress_detail_lines(project_id: object, status: Mapping[str, Any]) -> list[str]:
    progress = _validated_progress(status)
    if progress is None:
        return []
    details: list[str] = []
    current_item = progress.get("current_item")
    if isinstance(current_item, Mapping):
        kind = _upgrade_words(current_item.get("kind"), title=True)
        item_id = current_item.get("id")
        item_text = f"{kind} ID, {item_id}"
        completed_bytes = current_item.get("completed_bytes")
        total_bytes = current_item.get("total_bytes")
        if type(completed_bytes) is int and type(total_bytes) is int:
            item_text += f", {completed_bytes}/{total_bytes} bytes"
        if current_item.get("restarted") is True:
            item_text += ", restarted"
        details.append(f"{project_id}: {item_text}")

    remaining = progress.get("remaining_stages")
    if isinstance(remaining, Sequence) and not isinstance(remaining, (str, bytes, bytearray)) and remaining:
        details.append(f"remaining stages: {', '.join(str(item) for item in remaining)}")
    details.append("scope: shared Project.")
    last_progress_at = progress.get("last_progress_at")
    if last_progress_at is not None:
        details.append(f"last committed progress: {last_progress_at}")
    invocation = status.get("invocation")
    if isinstance(invocation, Mapping) and invocation.get("next_probe_at") is not None:
        details.append(f"next probe at: {invocation.get('next_probe_at')}")
    elif status.get("next_probe_at") is not None:
        details.append(f"next probe at: {status.get('next_probe_at')}")
    return [line for detail in details for line in _wrap_upgrade_detail(detail)]


def _delta_detail(invocation: Mapping[str, Any]) -> str | None:
    if invocation.get("slice_committed") is not True:
        return None
    delta = invocation.get("delta")
    if not isinstance(delta, Mapping):
        return None
    kind = delta.get("kind")
    before = delta.get("before")
    after = delta.get("after")
    if type(before) is not int or type(after) is not int:
        return None
    if kind == "bytes":
        item_label = _upgrade_words(delta.get("item_kind") or delta.get("stage"), title=True)
        unit_label = _singular_word(delta.get("unit") or "source")
        return f"advanced {item_label} {unit_label} bytes {before} -> {after}"
    if kind == "units":
        label = _singular_word(delta.get("stage") or delta.get("unit"))
        total = delta.get("total")
        if type(total) is int:
            return f"completed {label.title()} {after} of {total}"
        return f"processed {before}->{after}"
    return None


def _invocation_detail_lines(result: Mapping[str, Any]) -> list[str]:
    invocation = result.get("invocation")
    if not isinstance(invocation, Mapping):
        return []
    lines: list[str] = []
    lock = invocation.get("contended_lock")
    if lock in {"upgrade", "schema"}:
        lock_text = f"waiting for Project {lock} lock"
        if invocation.get("observed_progress") is True:
            lock_text += "; another coordinator advanced shared Project work"
        else:
            lock_text += "; progress unconfirmed"
        lines.append(lock_text)
    delta = _delta_detail(invocation)
    if delta is not None:
        lines.append(delta)
    if invocation.get("next_probe_at") is not None:
        lines.append(f"next probe at: {invocation.get('next_probe_at')}")
    return [line for detail in lines for line in _wrap_upgrade_detail(detail)]


def _upgrade_table(projects: Sequence[Mapping[str, Any]], *, nested: bool) -> str:
    headers = ("Project", "Phase", "State", "Progress", "Pending", "Admission blocked", "Blockers")
    limits = (20, 14, 16, 36, 7, 17, 24)
    rows = _upgrade_rows(projects, nested=nested)
    text_rows = [[_truncate_upgrade_text(value, limits[index]) for index, value in enumerate(row)] for row in rows]
    widths = [len(header) for header in headers]
    for index, limit in enumerate(limits):
        widths[index] = (
            min(limit, max(widths[index], *(len(row[index]) for row in text_rows))) if text_rows else widths[index]
        )
    render_row = lambda row: "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
    return "\n".join([render_row(headers), render_row(["-" * width for width in widths]), *map(render_row, text_rows)])


def _upgrade_detail_lines(projects: Sequence[Mapping[str, Any]], *, nested: bool) -> list[str]:
    lines: list[str] = []
    for project in projects:
        status = _upgrade_status(project, nested=nested)
        project_id = project.get("project_id")
        progress = _validated_progress(status)
        if progress is not None:
            lines.extend(_progress_detail_lines(project_id, status))
            summary = _progress_summary(status)
            if len(summary) > 36:
                lines.extend(_wrap_upgrade_detail(f"{project_id}: progress: {summary}"))
        blockers = status.get("blockers")
        if blockers and len(_value(blockers)) > 24:
            lines.extend(_wrap_upgrade_detail(f"{project_id}: blockers: {_value(blockers)}"))
        lines.extend(_invocation_detail_lines(status))
    return lines


def _upgrade_rows(projects: Sequence[Mapping[str, Any]], *, nested: bool) -> list[tuple[Any, ...]]:
    rows = []
    for project in projects:
        status = _upgrade_status(project, nested=nested)
        rows.append(
            (
                project.get("project_id"),
                status.get("phase"),
                status.get("state"),
                _progress_summary(status),
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
    lines.extend(_invocation_detail_lines(result))
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
    lines.append(_upgrade_table(projects, nested=True))
    lines.extend(_upgrade_detail_lines(projects, nested=True))
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
    lines.extend(_invocation_detail_lines(result))
    inaccessible = result.get("inaccessible_projects", ())
    for item in inaccessible:
        lines.append(f"Inaccessible: {item.get('project_id')}: {item.get('error') or item.get('reason')}")
    if not result["projects"]:
        lines.append(
            "All registered Projects are complete." if result.get("all_roots_complete") else "No Projects advanced."
        )
        return "\n".join(lines)
    lines.append(_upgrade_table(result["projects"], nested=False))
    lines.extend(_upgrade_detail_lines(result["projects"], nested=False))
    return "\n".join(lines)


def _validate_bounded_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256 or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a non-empty single-line string of at most 256 characters")


def _validate_invocation(value: Any, label: str) -> None:
    invocation = _mapping(value, label)
    for key in ("slice_committed", "contended_lock", "next_probe_at", "observed_progress", "delta"):
        _required(invocation, key, label)
    _required_bool(invocation, "slice_committed", label)
    _required_bool(invocation, "observed_progress", label)
    lock = invocation["contended_lock"]
    if lock not in {None, "upgrade", "schema"}:
        raise ValueError(f"{label}.contended_lock must be null, 'upgrade', or 'schema'")
    if invocation["next_probe_at"] is not None and not isinstance(invocation["next_probe_at"], str):
        raise TypeError(f"{label}.next_probe_at must be a string or null")
    delta = invocation["delta"]
    if delta is None:
        return
    if invocation["slice_committed"] is not True:
        raise ValueError(f"{label}.delta is only valid when slice_committed is true")
    delta = _mapping(delta, f"{label}.delta")
    for key in ("kind", "stage", "unit", "before", "after", "total"):
        _required(delta, key, f"{label}.delta")
    if delta["kind"] not in {"units", "bytes"}:
        raise ValueError(f"{label}.delta.kind must be 'units' or 'bytes'")
    _validate_bounded_text(delta["stage"], f"{label}.delta.stage")
    _validate_bounded_text(delta["unit"], f"{label}.delta.unit")
    for key in ("before", "after"):
        number = _required_int(delta, key, f"{label}.delta")
        if number < 0:
            raise ValueError(f"{label}.delta.{key} must be non-negative")
    total = delta["total"]
    if total is not None:
        if type(total) is not int or total < 0:
            raise ValueError(f"{label}.delta.total must be a non-negative integer or null")
    for key in ("item_kind", "item_id"):
        if key in delta and delta[key] is not None:
            _validate_bounded_text(delta[key], f"{label}.delta.{key}")


def _validate_optional_upgrade_fields(value: Mapping[str, Any], label: str) -> None:
    progress = value.get("progress")
    if progress is not None:
        try:
            validate_progress(progress)
        except (KeyError, TypeError, ValueError) as exc:
            raise type(exc)(f"{label}.progress: {exc}") from exc
    if "invocation" in value and value["invocation"] is not None:
        _validate_invocation(value["invocation"], f"{label}.invocation")


def _render_upgrade_project(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    rendered = _operation(
        result.get("action", "upgrade"),
        result.get("outcome", result.get("state", result.get("aggregate_state", "completed"))),
        (
            ("Project", result.get("project_id")),
            ("State", result.get("state")),
            ("Phase", result.get("phase")),
            ("Progress", _progress_summary(result)),
            ("Pending", result.get("pending")),
            ("Admission blocked", result.get("admission_blocked")),
            ("Blockers", result.get("blockers")),
            ("Reason", result.get("reason")),
            ("Next action", result.get("next_action")),
        ),
    )
    detail_lines = _progress_detail_lines(result.get("project_id"), result)
    detail_lines.extend(_invocation_detail_lines(result))
    return "\n".join([item for item in (rendered, *detail_lines) if item])


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
        _validate_optional_upgrade_fields(status, f"upgrade-registry-status payload.projects[{index}].upgrade")
    for index, project in enumerate(value.get("inaccessible_projects", ())):
        if not isinstance(project, Mapping):
            continue
        status = project.get("upgrade", project)
        if isinstance(status, Mapping):
            _validate_optional_upgrade_fields(status, f"upgrade-registry-status payload.inaccessible_projects[{index}]")
    if "invocation" in value and value["invocation"] is not None:
        _validate_invocation(value["invocation"], "upgrade-registry-status payload.invocation")


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
        _validate_optional_upgrade_fields(item, f"upgrade-advance payload.projects[{index}]")
    if "invocation" in value and value["invocation"] is not None:
        _validate_invocation(value["invocation"], "upgrade-advance payload.invocation")


def _validate_upgrade_project(result: Any) -> None:
    value = _mapping(result, "upgrade-project payload")
    for key in ("project_id", "phase", "state", "pending", "admission_blocked", "blockers"):
        _required(value, key, "upgrade-project payload")
    _required_bool(value, "pending", "upgrade-project payload")
    _required_bool(value, "admission_blocked", "upgrade-project payload")
    _sequence(value["blockers"], "upgrade-project payload.blockers")
    _validate_optional_upgrade_fields(value, "upgrade-project payload")


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
