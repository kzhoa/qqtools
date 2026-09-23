"""Submission-specific CLI adaptation and output handling."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from ..activation import AgentActivationError, ensure_local_agent_active
from ..agent.context import ExecutionContext
from ..commands import task as task_commands
from ..layout import load_context
from ..manifest import UNSET, normalize_command_submission, parse_submission_manifest
from ..project_resolution import resolve_submission_project
from ..runtime.group_namespace import GroupNotPublished
from ..runtime.paths import idempotency_path, submission_path
from ..runtime.store import read_json
from ..runtime.submission import (
    IdempotencyConflict,
    SubmissionFinalizationError,
    SubmissionPending,
    SubmissionRejected,
    SubmissionUnknown,
)
from ..runtime.submission_plan import SubmissionTargetInvalid, semantic_digest
from ..submission_contracts import SubmissionRequest, submission_result_payload
from .outcome import CommandOutcome
from .output import CliOutput, OutputKind


class SubmissionCommandError(Exception):
    """A recognized submission workflow failure requiring its fixed result schema."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


class SubmissionInputError(ValueError):
    """A recognized submission invocation or manifest error."""


def _submission_group_file(request: SubmissionRequest) -> Path | None:
    if request.group_name is None:
        return None
    # Submission's public group tree is schema-stable for this layer.  Keep
    # this read-only probe separate from runtime publication decisions.
    return request.project.control_root / "groups" / f"{request.group_name}.json"


def _submission_preview(request: SubmissionRequest, value: object) -> dict[str, object]:
    raw = value.to_dict() if callable(getattr(value, "to_dict", None)) else value
    preview = dict(raw) if isinstance(raw, dict) else {}
    tasks = preview.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    normalized_tasks = []
    for index, spec in enumerate(request.normalized_specs):
        runtime_task = dict(tasks[index]) if index < len(tasks) and isinstance(tasks[index], dict) else {}
        runtime_task.update(spec)
        runtime_task["input_index"] = index
        runtime_task["task_id"] = spec.get("task_id")
        runtime_task["sources"] = dict(request.field_sources[index])
        normalized_tasks.append(runtime_task)
    group_file = _submission_group_file(request)
    group_action = preview.get("group_action")
    if group_action not in {"create", "reuse", "none", "unknown"}:
        group_action = (
            "none" if request.group_name is None else "reuse" if group_file and group_file.exists() else "create"
        )
    return {
        "tasks": normalized_tasks,
        "group_action": group_action,
        "worker_additions": preview.get("worker_additions", request.worker_set),
        "evidence_gaps": preview.get("evidence_gaps", []),
    }


def _submission_result_payload(request: SubmissionRequest, value: object) -> dict[str, object]:
    if request.dry_run:
        return submission_result_payload(
            mode=request.mode,
            outcome="preview",
            project={"path": str(request.project.path), "source": request.project.source},
            group={"name": request.group_name, "source": request.group_source, "disposition": None},
            operation=None,
            idempotency_key=request.idempotency_key,
            task_ids=[],
            preview=_submission_preview(request, value),
            error=None,
        )
    raw = value.to_dict() if callable(getattr(value, "to_dict", None)) else value
    raw = raw if isinstance(raw, dict) else {}
    state = getattr(value, "state", raw.get("state"))
    operation_id = getattr(value, "operation_id", raw.get("operation_id"))
    idempotency_key = getattr(value, "idempotency_key", raw.get("idempotency_key"))
    target_group = getattr(value, "target_group", raw.get("target_group", request.group_name))
    committed = state == "committed"
    outcome = "committed" if committed else "pending" if state in {"preparing", "committing", "blocked"} else "unknown"
    disposition = (
        "none"
        if target_group is None
        else "reused"
        if request.group_existed is True
        else "created"
        if request.group_existed is False
        else "reused"
    )
    if target_group is not None and operation_id:
        try:
            persisted = read_json(submission_path(request.project.control_root, operation_id))
            resolved_context = persisted.get("submission", {}).get("resolved_context", {})
            if isinstance(resolved_context, dict) and isinstance(resolved_context.get("create_group"), bool):
                disposition = "created" if resolved_context["create_group"] else "reused"
        except (OSError, KeyError, TypeError, ValueError):
            pass
    return submission_result_payload(
        mode=request.mode,
        outcome=outcome,
        project={"path": str(request.project.path), "source": request.project.source},
        group={"name": target_group, "source": request.group_source, "disposition": disposition},
        operation={"id": operation_id, "state": state} if operation_id else None,
        idempotency_key=idempotency_key,
        task_ids=[task.task_id for task in value] if committed and hasattr(value, "__iter__") else [],
        preview=None,
        error=None,
    )


def _known_submission_state(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    """Recover bounded operation facts disclosed before a failed return."""
    request = getattr(args, "_submission_request", None)
    if request is None:
        return {}
    key = (
        getattr(exc, "idempotency_key", None)
        or getattr(args, "_submission_prepared_key", None)
        or request.idempotency_key
    )
    operation_id = getattr(exc, "operation_id", None) or getattr(args, "_submission_prepared_operation_id", None)
    if operation_id is None and key:
        try:
            mapping = read_json(
                idempotency_path(
                    request.project.control_root,
                    semantic_digest({"project": str(request.project.control_root), "key": key}),
                )
            )
            operation_id = mapping.get("operation_id") if isinstance(mapping, dict) else None
        except (OSError, KeyError, TypeError, ValueError):
            pass
    if not isinstance(operation_id, str):
        return {"idempotency_key": key}
    result: dict[str, Any] = {"operation_id": operation_id, "state": None, "idempotency_key": key}
    try:
        operation = read_json(submission_path(request.project.control_root, operation_id))["submission"]
        result.update(
            state=operation.get("state"),
            idempotency_key=operation.get("idempotency_key", key),
            target_group=operation.get("target_group", request.group_name),
        )
        context = operation.get("resolved_context")
        if isinstance(context, dict):
            result["create_group"] = context.get("create_group")
            task_ids = context.get("task_ids")
            if isinstance(task_ids, list) and all(isinstance(item, str) for item in task_ids):
                result["task_ids"] = task_ids
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return result


def _submission_error_payload(args: argparse.Namespace, exc: BaseException) -> tuple[dict[str, object], int]:
    request = getattr(args, "_submission_request", None)
    mode = getattr(args, "_submission_mode", None) or _submission_error_mode(args)
    project = None
    group_name = getattr(args, "group", None)
    group_source = "cli" if group_name is not None else "none"
    known = _known_submission_state(args, exc)
    operation_id = known.get("operation_id")
    state = known.get("state")
    operation = {"id": operation_id, "state": state} if operation_id else None
    key = getattr(args, "idempotency_key", None)
    if request is not None:
        project = {"path": str(request.project.path), "source": request.project.source}
        group_name = request.group_name
        group_source = request.group_source
        key = known.get("idempotency_key", request.idempotency_key)
        group_name = known.get("target_group", group_name)
    name = type(exc).__name__
    if name == "IdempotencyConflict" or "idempotency key" in str(exc).lower():
        code, outcome, exit_code = "idempotency_conflict", "rejected", 1
    elif state == "committed":
        code, outcome, exit_code = "finalization_pending", "committed", 1
    elif state in {"preparing", "committing", "blocked"}:
        code = "submission_blocked" if state == "blocked" else "submission_pending"
        outcome, exit_code = "pending", 1
    elif state == "aborted":
        code, outcome, exit_code = "submission_aborted", "rejected", 1
    elif request is None and isinstance(exc, (ValueError, OSError)):
        code, exit_code = (
            ("invalid_input", 2)
            if not getattr(args, "_submission_resolution_started", False)
            or getattr(args, "_submission_project_resolved", False)
            else ("context_error", 1)
        )
        outcome = "rejected"
    elif request is not None and not getattr(args, "_submission_cfg_resolved", False):
        code, outcome, exit_code = "context_error", "rejected", 2
    elif isinstance(exc, ValueError):
        code, outcome, exit_code = "invalid_input", "rejected", 2
    elif "blocked" in str(exc).lower():
        code, outcome, exit_code = "submission_blocked", "pending", 1
    elif "pending" in str(exc).lower():
        code, outcome, exit_code = "submission_pending", "pending", 1
    elif operation_id is not None:
        code, outcome, exit_code = "commit_unknown", "unknown", 1
    else:
        code, outcome, exit_code = "submission_aborted", "rejected", 1
    disposition = None
    if outcome == "committed":
        disposition = "none" if group_name is None else "created" if known.get("create_group") is True else "reused"
    payload = submission_result_payload(
        mode=mode,
        outcome=outcome,
        project=project,
        group={"name": group_name, "source": group_source, "disposition": disposition},
        operation=operation,
        idempotency_key=key,
        task_ids=known.get("task_ids", []) if outcome == "committed" else [],
        preview=None,
        error={"code": code, "message": str(exc)},
    )
    return payload, exit_code


def _command(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SubmissionInputError("submit requires a command after '--'.")
    return argv


def _submission_raw_prefix(raw_argv: list[str]) -> list[str]:
    """Return submit options before the literal command separator."""
    try:
        submit_index = raw_argv.index("submit")
    except ValueError:
        return []
    prefix = raw_argv[submit_index + 1 :]
    try:
        return prefix[: prefix.index("--")]
    except ValueError:
        return prefix


def _submission_option_supplied(raw_argv: list[str], *names: str) -> bool:
    prefix = _submission_raw_prefix(raw_argv)
    return any(item in names or any(item.startswith(f"{name}=") for name in names) for item in prefix)


def _submission_raw_option(raw_argv: list[str], name: str) -> str | None:
    prefix = _submission_raw_prefix(raw_argv)
    for index, item in enumerate(prefix):
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
        if item == name and index + 1 < len(prefix):
            return prefix[index + 1]
    return None


def _submission_parse_failure_payload(raw_argv: list[str], message: str) -> dict[str, Any]:
    """Build the complete result schema when argparse cannot create a Namespace."""
    has_file = _submission_option_supplied(raw_argv, "--file", "-f")
    try:
        tail = raw_argv[raw_argv.index("submit") + 1 :]
    except ValueError:
        tail = []
    has_command_separator = "--" in tail
    mode = (
        "file"
        if has_file and not has_command_separator
        else "command"
        if has_command_separator and not has_file
        else None
    )
    group_name = _submission_raw_option(raw_argv, "--group")
    return submission_result_payload(
        mode=mode,
        outcome="rejected",
        project=None,
        group={"name": group_name, "source": "cli" if group_name is not None else None, "disposition": None},
        operation=None,
        idempotency_key=_submission_raw_option(raw_argv, "--idempotency-key"),
        task_ids=[],
        preview=None,
        error={"code": "invalid_input", "message": message},
    )


def _submission_mode_from_args(args: argparse.Namespace) -> tuple[str, list[str]]:
    argv = _command(list(getattr(args, "argv", ()))) if getattr(args, "argv", None) else []
    has_file = getattr(args, "manifest_file", None) is not None
    has_command = bool(argv)
    if has_file and has_command:
        raise SubmissionInputError("submit requires exactly one input mode: --file MANIFEST or -- COMMAND....")
    if not has_file and not has_command:
        raise SubmissionInputError("submit requires exactly one input mode: --file MANIFEST or -- COMMAND....")
    return ("file", []) if has_file else ("command", argv)


def _submission_error_mode(args: argparse.Namespace) -> str | None:
    """Infer a presentation mode without resolving a Project or reading state."""
    try:
        mode, _ = _submission_mode_from_args(args)
    except ValueError:
        mode = None
    return mode


def _prepare_submission_request(
    args: argparse.Namespace, raw_argv: list[str], *, invocation_cwd: Path | None = None
) -> SubmissionRequest:
    """Validate modes/options and normalize a submission before cfg resolution."""
    if args.command_spec.handler != "submit":
        raise RuntimeError("submission request preparation requires submit command.")
    invocation_cwd = Path(invocation_cwd or Path.cwd()).expanduser().resolve()
    try:
        submission_tail = raw_argv[raw_argv.index("submit") + 1 :]
    except ValueError:
        submission_tail = []
    if getattr(args, "argv", None) and "--" not in submission_tail:
        raise SubmissionInputError("command mode requires the literal '--' separator before COMMAND.")
    mode, command = _submission_mode_from_args(args)
    args._submission_mode = mode
    command_only = {
        "--task-id": "--task-id is only valid in command mode.",
        "--name": "--name is only valid in command mode.",
        "--depends-on": "--depends-on is only valid in command mode.",
        "--sharing": "--sharing is only valid in command mode.",
        "--offer-after-seconds": "--offer-after-seconds is only valid in command mode.",
    }
    if mode == "file":
        for option, message in command_only.items():
            if _submission_option_supplied(raw_argv, option):
                raise SubmissionInputError(message)
    elif _submission_option_supplied(raw_argv, "--file", "-f"):
        raise SubmissionInputError("--file is only valid in file mode.")
    if args.quiet and args.format == "json":
        raise SubmissionInputError("--quiet cannot be combined with --format json.")
    if args.quiet and args.dry_run:
        raise SubmissionInputError("--quiet cannot be combined with --dry-run.")

    manifest_path = None
    if mode == "file":
        manifest_path = Path(args.manifest_file).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = invocation_cwd / manifest_path
        manifest_path = manifest_path.resolve()
    explicit_project = args.project if args.project is not None else getattr(args, "shared_root", None)
    environment_value = os.environ.get("QEXP_SHARED_ROOT")
    args._submission_resolution_started = True
    saved_context = load_context() if explicit_project is None and environment_value is None else None
    try:
        selection = resolve_submission_project(
            explicit_project=explicit_project,
            manifest_path=manifest_path,
            invocation_cwd=invocation_cwd,
            environment_value=environment_value,
            saved_context=saved_context,
        )
    except ValueError as exc:
        raise SubmissionInputError(str(exc)) from exc
    args._submission_project_resolved = True
    if mode == "command":
        try:
            item, field_sources = normalize_command_submission(
                command,
                requested_gpus=1 if args.gpus is None else args.gpus,
                requested_cpus=args.cpus,
                task_id=args.task_id,
                name=args.name,
                group=args.group,
                working_directory=args.cwd,
                home_machine="current" if args.home_machine is None else args.home_machine,
                sharing_mode="private" if args.sharing is None else args.sharing,
                offer_after_seconds=args.offer_after_seconds,
                depends_on_task_ids=[] if args.depends_on is None else args.depends_on,
                tmux_override=args.tmux_override,
                invocation_cwd=invocation_cwd,
                project_directory=selection.path,
            )
        except ValueError as exc:
            raise SubmissionInputError(str(exc)) from exc
        group_name = args.group
        group_source = "cli" if group_name is not None else "none"
        workers: dict[str, dict[str, Any]] = {}
        workers_declared = False
        specs = (item,)
        command_sources = dict(field_sources)
        if args.gpus is None:
            command_sources["requested_gpus"] = "builtin"
        if args.home_machine is None:
            command_sources["home_machine"] = "builtin"
        if args.sharing is None:
            command_sources["sharing_mode"] = "builtin"
        field_sources = (command_sources,)
    else:
        try:
            result = parse_submission_manifest(
                manifest_path,
                group_name=args.group if args.group is not None else UNSET,
                tmux_override=(
                    args.tmux_override if _submission_option_supplied(raw_argv, "--tmux", "--no-tmux") else UNSET
                ),
                requested_gpus=args.gpus if _submission_option_supplied(raw_argv, "--gpus") else UNSET,
                requested_cpus=args.cpus if _submission_option_supplied(raw_argv, "--cpus") else UNSET,
                home_machine=args.home_machine if _submission_option_supplied(raw_argv, "--home-machine") else UNSET,
                working_directory=args.cwd if _submission_option_supplied(raw_argv, "--cwd") else UNSET,
                project_directory=selection.path,
                invocation_cwd=invocation_cwd,
            )
        except (OSError, ValueError) as exc:
            raise SubmissionInputError(str(exc)) from exc
        group_name = result.group_name
        group_source = result.group_source
        workers = result.workers
        workers_declared = result.workers_declared
        specs = result.specs
        field_sources = result.field_sources
        args.manifest_file = str(result.manifest_path)
    request = SubmissionRequest(
        mode=mode,
        specs=tuple(specs),
        group_name=group_name,
        group_source=group_source,
        workers=workers,
        workers_declared=workers_declared,
        project=selection,
        invocation_cwd=invocation_cwd,
        manifest_path=manifest_path,
        idempotency_key=args.idempotency_key,
        no_activate=args.no_activate,
        dry_run=args.dry_run,
        output_format=args.format,
        quiet=args.quiet,
        field_sources=tuple(field_sources),
        group_existed=(
            (selection.control_root / "groups" / f"{group_name}.json").exists() if group_name is not None else None
        ),
    )
    args._submission_project = selection.control_root
    args._submission_mode = mode
    args._submission_request = request
    return request


def dispatch_submission(
    args: argparse.Namespace,
    cfg: object,
    execution_context: ExecutionContext,
) -> CommandOutcome:
    """Execute a prepared submission and preserve its output/activation policy."""

    def print_prepared(operation_id: str, idempotency_key: str) -> None:
        args._submission_prepared_operation_id = operation_id
        args._submission_prepared_key = idempotency_key
        print(
            f"qexp: prepared operation_id={operation_id} idempotency_key={idempotency_key}",
            file=sys.stderr,
            flush=True,
        )

    request = args._submission_request
    try:
        values = task_commands.submit_request(cfg, request, on_prepared=print_prepared)
    except (
        GroupNotPublished,
        IdempotencyConflict,
        SubmissionFinalizationError,
        SubmissionPending,
        SubmissionRejected,
        SubmissionTargetInvalid,
        SubmissionUnknown,
    ) as exc:
        raise SubmissionCommandError(exc) from exc
    # Activation is a post-commit follow-up.  It must never relabel a
    # verified commit or erase the IDs from a quiet/JSON response.
    payload = _submission_result_payload(request, values)
    if not request.dry_run and payload["outcome"] == "committed" and not request.no_activate:
        try:
            ensure_local_agent_active(
                cfg,
                reason="submit",
                machine_runtime=execution_context.machine_runtime,
            )
        except AgentActivationError as activation_error:
            payload["error"] = {"code": "activation_failed", "message": str(activation_error)}
            follow_up = activation_error.next_action or shlex.join(
                [
                    "qexp",
                    "--machine-runtime-root",
                    str(execution_context.machine_runtime.root),
                    "agent",
                    "start",
                ]
            )
            payload["activation"] = {"outcome": "failed", "follow_up_command": follow_up}
            if request.quiet:
                for task_id in payload["task_ids"]:
                    print(task_id)
            else:
                return CommandOutcome(
                    1,
                    CliOutput(
                        OutputKind.SUBMISSION,
                        payload,
                        {
                            "task": {
                                "task_id": request.normalized_specs[0].get("task_id"),
                                "name": request.normalized_specs[0].get("name"),
                            }
                        },
                    ),
                )
            return CommandOutcome(1)
    if request.quiet:
        if payload["outcome"] == "committed":
            for task_id in payload["task_ids"]:
                print(task_id)
    else:
        presentation = {
            "task": {
                "task_id": request.normalized_specs[0].get("task_id"),
                "name": request.normalized_specs[0].get("name"),
            }
        }
        return CommandOutcome(0, CliOutput(OutputKind.SUBMISSION, payload, presentation))
    return CommandOutcome(0)
