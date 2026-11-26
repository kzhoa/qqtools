"""Pure submission output contract."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import _details, _mapping, _required, _required_mapping, _required_sequence, _sequence, _value

_DIAGNOSTIC_FIELDS = {
    "version",
    "component",
    "operation",
    "stage",
    "check_id",
    "reason_code",
    "facts",
    "exception_type",
    "task_id",
    "generation",
    "group_name",
    "input_index",
    "errno",
    "json_line",
    "json_column",
}
_DIAGNOSTIC_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")
_DIAGNOSTIC_FACT_CHOICES = {
    "record_type": {
        "member_page",
        "member_catalog",
        "member_locator",
        "member_header",
        "member_directory",
        "member_writable_index",
        "member_global_state",
    },
    "field": {
        "task_id",
        "queue_scope",
        "home_machine",
        "partition",
        "marker_name",
        "submission_operation_id",
    },
    "queue_field": {"head", "tail", "free", "count"},
}
_DIAGNOSTIC_INTEGER_FACTS = {
    "actual_bytes",
    "limit_bytes",
    "expected_revision",
    "observed_revision",
    "member_count",
    "page",
}


def _failure_diagnostic(value: object) -> Mapping[str, Any]:
    diagnostic = _mapping(value, "submission payload.error.diagnostic")
    if set(diagnostic) != _DIAGNOSTIC_FIELDS:
        raise ValueError("submission payload.error.diagnostic fields are invalid")
    if (
        diagnostic["version"] != 1
        or diagnostic["component"] != "group_ready_members"
        or diagnostic["operation"] != "publish"
    ):
        raise ValueError("submission payload.error.diagnostic identity is invalid")
    for key in ("stage", "check_id", "reason_code", "exception_type", "task_id", "group_name"):
        if not isinstance(diagnostic[key], str) or not _DIAGNOSTIC_TOKEN.fullmatch(diagnostic[key]):
            raise ValueError(f"submission payload.error.diagnostic.{key} must be a non-empty string")
    if type(diagnostic["generation"]) is not int or diagnostic["generation"] < 0:
        raise ValueError("submission payload.error.diagnostic.generation is invalid")
    for key in ("input_index", "errno", "json_line", "json_column"):
        item = diagnostic[key]
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError(f"submission payload.error.diagnostic.{key} is invalid")
    facts = _mapping(diagnostic["facts"], "submission payload.error.diagnostic.facts")
    if len(facts) > 8:
        raise ValueError("submission payload.error.diagnostic.facts is invalid")
    for key, item in facts.items():
        if key in _DIAGNOSTIC_INTEGER_FACTS:
            if type(item) is not int or item < 0:
                raise ValueError("submission payload.error.diagnostic.facts is invalid")
        elif key not in _DIAGNOSTIC_FACT_CHOICES or item not in _DIAGNOSTIC_FACT_CHOICES[key]:
            raise ValueError("submission payload.error.diagnostic.facts is invalid")
    return diagnostic


def _format_failure_diagnostic(value: object) -> str:
    diagnostic = _failure_diagnostic(value)
    fields = [
        f"component={diagnostic['component']}",
        f"operation={diagnostic['operation']}",
        f"stage={diagnostic['stage']}",
        f"check={diagnostic['check_id']}",
        f"reason={diagnostic['reason_code']}",
        f"exception={diagnostic['exception_type']}",
        f"task={diagnostic['task_id']}",
        f"generation={diagnostic['generation']}",
    ]
    for key in ("input_index", "errno", "json_line", "json_column"):
        if diagnostic[key] is not None:
            fields.append(f"{key}={diagnostic[key]}")
    facts = diagnostic["facts"]
    if facts:
        fields.append("facts=" + ",".join(f"{key}={facts[key]}" for key in sorted(facts)))
    return "Diagnostic: " + " ".join(fields)


def _render_submission(result: Mapping[str, Any], presentation: Mapping[str, object]) -> str:
    """Render the submission result without implying a Batch resource."""
    outcome = result["outcome"]
    mode = result.get("mode")
    project = result.get("project") or {}
    project_path = project.get("path") if isinstance(project, Mapping) else None
    group = result.get("group") or {}
    group_name = group.get("name") if isinstance(group, Mapping) else None
    if outcome == "preview":
        preview = result.get("preview") or {}
        tasks = preview.get("tasks", ()) if isinstance(preview, Mapping) else ()
        lines = [f"Preview ({len(tasks)} task{'s' if len(tasks) != 1 else ''})"]
        if project_path:
            lines.append(f"Project: {project_path}")
        lines.append(
            f"Group action: {preview.get('group_action', 'unknown') if isinstance(preview, Mapping) else 'unknown'}"
        )
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                continue
            lines.append(
                f"Task {index}: name={_value(task.get('name'))} command={_value(task.get('command'))} "
                f"GPUs={_value(task.get('requested_gpus'))} CPUs={_value(task.get('requested_cpus'))} "
                f"home={_value(task.get('home_machine'))} cwd={_value(task.get('working_directory'))}"
            )
            sources = task.get("sources")
            if sources:
                lines.append(f"  Sources: {_value(sources)}")
        gaps = preview.get("evidence_gaps", ()) if isinstance(preview, Mapping) else ()
        if gaps:
            lines.append(f"Evidence gaps: {_value(gaps)}")
        return "\n".join(lines)
    if outcome == "committed":
        task_ids = list(result.get("task_ids", ()))
        activation = result.get("activation")
        activation = activation if isinstance(activation, Mapping) else {}
        activation_outcome = activation.get("outcome")
        lines = [f"Committed {len(task_ids)} task{'s' if len(task_ids) != 1 else ''}."]
        if activation_outcome == "failed":
            error = result.get("error")
            error = error if isinstance(error, Mapping) else {}
            lines.extend(
                (
                    "Agent activation failed.",
                    f"Task IDs: {_value(task_ids)}",
                    f"Error: {_value(error.get('message'))}",
                    f"Next: {_value(activation.get('follow_up_command'))}",
                )
            )
        elif isinstance(activation_outcome, str):
            lines.append(f"Agent activation: {activation_outcome.replace('_', ' ')}")
        if project_path:
            lines.append(f"Project: {project_path}")
        if activation_outcome != "failed":
            if mode == "command":
                task = presentation.get("task", {})
                task_mapping = task if isinstance(task, Mapping) else {}
                task_id = task_mapping.get("task_id") or (task_ids[0] if task_ids else None)
                name = task_mapping.get("name")
                lines.extend((f"Task ID: {_value(task_id)}", f"Name: {_value(name)}", f"Group: {_value(group_name)}"))
                if project_path and task_id:
                    locator = shlex.join(["qexp", "--project", str(project_path), "task", "show", str(task_id)])
                    logs = shlex.join(["qexp", "--project", str(project_path), "task", "logs", str(task_id)])
                    lines.extend([f"Show: {locator}", f"Logs: {logs}"])
            elif group_name:
                lines.append(f"Group: {group_name}")
                if project_path:
                    lines.append(
                        f"Inspect: {shlex.join(['qexp', '--project', str(project_path), 'group', 'show', str(group_name)])}"
                    )
            else:
                lines.append(f"Task IDs: {_value(task_ids)}")
        recovery = result.get("diagnostic_recovery")
        if isinstance(recovery, Mapping) and recovery.get("state") == "blocked":
            from ..project_presentation import encode_display_text

            lines.append("")
            lines.extend(
                (
                    f"Machine scheduler diagnostic recovery blocked ({_value(recovery.get('reason'))}).",
                    f"Machine runtime: {encode_display_text(recovery['machine_runtime_root'])}",
                    f"Read-only preview: {recovery['next_action']}",
                )
            )
        return "\n".join(lines)
    error = result.get("error") or {}
    message = error.get("message") if isinstance(error, Mapping) else None
    lines = [f"Submission {outcome}: {_value(message)}"]
    diagnostic = error.get("diagnostic") if isinstance(error, Mapping) else None
    if diagnostic is not None:
        lines.append(_format_failure_diagnostic(diagnostic))
    return "\n".join(lines)


def _validate_submission(result: Any) -> None:
    value = _mapping(result, "submission payload")
    required = (
        "schema_version",
        "mode",
        "outcome",
        "project",
        "group",
        "operation",
        "idempotency_key",
        "task_ids",
        "preview",
        "error",
        "activation",
    )
    for key in required:
        _required(value, key, "submission payload")
    unknown = sorted(set(value) - {*required, "diagnostic_recovery"})
    if unknown:
        raise ValueError(f"submission payload has unknown fields: {', '.join(unknown)}")
    if value["schema_version"] != 1:
        raise ValueError("submission payload.schema_version must be 1")
    if value["mode"] not in {"command", "file", None}:
        raise ValueError("submission payload.mode is invalid")
    if value["outcome"] not in {"committed", "rejected", "pending", "unknown", "preview"}:
        raise ValueError("submission payload.outcome is invalid")
    project = value["project"]
    if project is not None:
        project = _mapping(project, "submission payload.project")
        if set(project) != {"path", "source"}:
            raise ValueError("submission payload.project fields are invalid")
        if not isinstance(project["path"], str) or project["source"] not in {
            "cli",
            "environment",
            "manifest_ancestor",
            "cwd_ancestor",
            "saved",
        }:
            raise ValueError("submission payload.project is invalid")
    group = _required_mapping(value, "group", "submission payload")
    if set(group) != {"name", "source", "disposition"}:
        raise ValueError("submission payload.group fields are invalid")
    if group["name"] is not None and not isinstance(group["name"], str):
        raise ValueError("submission payload.group.name must be a string or null")
    if group["source"] not in {"cli", "manifest", "none", None}:
        raise ValueError("submission payload.group.source is invalid")
    if group["disposition"] not in {"created", "reused", "none", None}:
        raise ValueError("submission payload.group.disposition is invalid")
    operation = value["operation"]
    if operation is not None:
        operation = _mapping(operation, "submission payload.operation")
        if set(operation) != {"id", "state"}:
            raise ValueError("submission payload.operation fields are invalid")
        if not isinstance(operation["id"], str) or operation["state"] not in {
            "preparing",
            "committing",
            "committed",
            "aborted",
            "blocked",
            None,
        }:
            raise ValueError("submission payload.operation is invalid")
    task_ids = _required_sequence(value, "task_ids", "submission payload")
    if any(not isinstance(item, str) for item in task_ids):
        raise ValueError("submission payload.task_ids must contain strings")
    preview = value["preview"]
    if preview is not None:
        preview = _mapping(preview, "submission payload.preview")
        for key in ("tasks", "group_action", "worker_additions", "evidence_gaps"):
            _required(preview, key, "submission payload.preview")
        _sequence(preview["tasks"], "submission payload.preview.tasks")
        if preview["group_action"] not in {"create", "reuse", "none", "unknown"}:
            raise ValueError("submission payload.preview.group_action is invalid")
        worker_additions = preview["worker_additions"]
        if isinstance(worker_additions, Mapping):
            _mapping(worker_additions, "submission payload.preview.worker_additions")
        else:
            _sequence(worker_additions, "submission payload.preview.worker_additions")
        _sequence(preview["evidence_gaps"], "submission payload.preview.evidence_gaps")
    error = value["error"]
    if error is not None:
        error = _mapping(error, "submission payload.error")
        if set(error) not in ({"code", "message"}, {"code", "message", "diagnostic"}):
            raise ValueError("submission payload.error fields are invalid")
        if not isinstance(error["code"], str) or not isinstance(error["message"], str):
            raise ValueError("submission payload.error must contain string code and message")
        if "diagnostic" in error:
            _failure_diagnostic(error["diagnostic"])
    activation = value["activation"]
    if activation is not None:
        activation = _mapping(activation, "submission payload.activation")
        if set(activation) != {"outcome", "follow_up_command"}:
            raise ValueError("submission payload.activation fields are invalid")
        if activation["outcome"] not in {"started", "already_running", "requested", "not_requested", "failed"}:
            raise ValueError("submission payload.activation.outcome is invalid")
        follow_up = activation["follow_up_command"]
        if not isinstance(follow_up, str) and (follow_up is not None or activation["outcome"] == "failed"):
            raise ValueError("submission payload.activation.follow_up_command is invalid")
    if "diagnostic_recovery" in value:
        label = "submission payload.diagnostic_recovery"
        recovery = _required_mapping(value, "diagnostic_recovery", "submission payload")
        if (
            value["outcome"] != "committed"
            or set(recovery) != {"scope", "machine_runtime_root", "state", "reason", "next_action"}
            or recovery["scope"] != "machine_runtime"
            or recovery["state"] != "blocked"
        ):
            raise ValueError(f"{label} fields are invalid")
        for key in ("machine_runtime_root", "next_action"):
            if not isinstance(recovery[key], str):
                raise ValueError(f"{label}.{key} must be a string")
        if recovery["reason"] is not None and not isinstance(recovery["reason"], str):
            raise ValueError(f"{label}.reason must be a string or null")


CONTRACTS = {
    OutputKind.SUBMISSION: OutputContract(_validate_submission, _render_submission),
}

__all__ = ["CONTRACTS"]
