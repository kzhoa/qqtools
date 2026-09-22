"""qexp command line routing for the schema-6 product contract."""

from __future__ import annotations

import argparse
import contextvars
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import observer
from .activation import (
    ensure_local_agent_active,
    managed_project_agent_status,
    restart_local_agent,
    run_local_agent_foreground,
    start_local_agent,
    stop_local_agent,
)
from .agent.config import agent_config_payload, set_agent_config
from .agent.context import ExecutionContext, MachineRuntime
from .agent.lifecycle import (
    ensure_machine_agent_started,
    get_machine_agent_status,
    restart_machine_agent,
    stop_machine_agent,
)
from .agent.project_admin import (
    _enable_command,
    adoption_warning_generation,
    enable_project,
    migrate_project,
    register_project,
    set_project_enabled,
    unregister_project,
)
from .agent.readiness import capture_readiness_snapshot, evaluate_readiness, wait_for_readiness
from .agent.setup import (
    get_agent_config,
    initialize_machine,
    initialize_project,
    list_projects,
    machine_init_facts,
    register_projects,
    remove_project,
    set_project_enablement,
)
from .commands import cleanup
from .commands import configuration as configuration_commands
from .commands import context as context_commands
from .commands import group as group_commands
from .commands import logs as log_commands
from .commands import operation as operation_commands
from .commands import status as status_commands
from .commands import task as task_commands
from .commands import wait as wait_commands
from .commands import watch as watch_commands
from .commands.registry import CommandSpec, bind_command
from .config_types import RootConfig
from .doctor import repair_metadata, resolve_verify_exit_code, verify_integrity
from .formatter import CliOutput, OutputKind, render
from .gpu_policy import parse_gpu_id_list, reset_gpu_policy, set_gpu_policy, show_gpu_policy
from .launch_policy import (
    set_launch_handoff_policy,
    show_launch_handoff_policy,
    validate_launch_handoff_timeout_seconds,
)
from .layout import clear_context, load_context, load_root_config, migrate_schema5_to_schema6, save_context
from .lease import LeasePolicy, load_lease_policy, save_lease_policy
from .legacy_agent import get_agent_status
from .manifest import UNSET, normalize_command_submission, parse_submission_manifest
from .notification_config import (
    DEFAULT_WEBHOOK_ENV,
    load_notifications,
    update_notifications,
    write_shared_feishu_webhook,
)
from .progress_policy import set_progress_policy, show_progress_policy, validate_interval_seconds
from .project_resolution import resolve_submission_project
from .runtime.observation.api import ObservationError
from .runtime.paths import idempotency_path, shared_paths, submission_path
from .runtime.resources.cpu_lane import get_cpu_lane_policy, initialize_cpu_lane_capacity, set_cpu_lane_capacity
from .runtime.store import iter_json, read_json
from .runtime.submission_plan import semantic_digest
from .runtime.upgrade.framework import UpgradeCoordinator
from .runtime.upgrade.machine import advance_registered_upgrades, inspect_registered_upgrades
from .schema6_upgrade import (
    attest_schema6_upgrade,
    check_schema6_upgrade,
    resume_schema6_upgrade,
    schema6_upgrade_status,
    start_schema6_upgrade,
)
from .submission_contracts import SubmissionRequest, submission_result_payload
from .tmux_policy import set_tmux_policy, show_tmux_policy


class _PaginationParseError(RuntimeError):
    """An argparse error that belongs to paginated JSON output."""


class _SubmissionParseError(RuntimeError):
    """An argparse error that belongs to structured submission JSON output."""


class _JsonParseError(RuntimeError):
    """An argparse error that belongs to another finite JSON invocation."""


_PAGINATION_JSON_PARSE_MODE = False
_SUBMISSION_JSON_PARSE_MODE = False
_JSON_PARSE_MODE = False
_ACTIVE_COMMAND_SPEC: contextvars.ContextVar[CommandSpec | None] = contextvars.ContextVar(
    "qexp_active_command_spec", default=None
)

_DYNAMIC_OUTPUT_KINDS: dict[str, frozenset[OutputKind]] = {
    "agent-operation": frozenset({OutputKind.AGENT_OPERATION, OutputKind.AGENT_READINESS}),
    "doctor-check": frozenset({OutputKind.DOCTOR_VERIFY}),
    "group-operation": frozenset({OutputKind.GROUP_OPERATION, OutputKind.GROUP_MACHINES}),
    "task-list": frozenset({OutputKind.TASK_LIST, OutputKind.TASK_PAGE}),
    "task-show": frozenset({OutputKind.TASK_SHOW, OutputKind.TASK_WATCH}),
    "upgrade": frozenset(
        {
            OutputKind.UPGRADE_REGISTRY_STATUS,
            OutputKind.UPGRADE_ADVANCE,
            OutputKind.UPGRADE_PROJECT,
        }
    ),
}


class _QexpArgumentParser(argparse.ArgumentParser):
    """Use structured parse errors only for the narrowly scoped page mode."""

    def error(self, message: str) -> None:
        if _PAGINATION_JSON_PARSE_MODE:
            raise _PaginationParseError(message)
        if _SUBMISSION_JSON_PARSE_MODE:
            raise _SubmissionParseError(message)
        if _JSON_PARSE_MODE:
            raise _JsonParseError(message)
        super().error(message)

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None):
        raw = list(sys.argv[1:] if args is None else args)
        conflict = _common_option_conflict(raw)
        if conflict is not None:
            self.error(conflict)
        return super().parse_args(raw, namespace)


_COMMON_OPTIONS = frozenset({"--project", "--machine", "--runtime-root", "--machine-runtime-root", "--format"})


def _normalized_common_value(name: str, value: str) -> str:
    if name == "--project":
        return str(context_commands.normalize_project_path(value))
    if name in {"--runtime-root", "--machine-runtime-root"}:
        return str(Path(value).expanduser().resolve())
    return value


def _common_option_conflict(argv: list[str]) -> str | None:
    """Reject conflicting duplicate common options before argparse dispatches."""
    values: dict[str, str] = {}
    stop = len(argv)
    try:
        submit_index = argv.index("submit")
    except ValueError:
        submit_index = -1
    if submit_index >= 0:
        try:
            separator = argv.index("--", submit_index + 1)
        except ValueError:
            separator = len(argv)
        stop = separator
    index = 0
    while index < stop:
        token = argv[index]
        name = token.split("=", 1)[0]
        if name not in _COMMON_OPTIONS:
            index += 1
            continue
        if "=" in token:
            value = token.split("=", 1)[1]
        elif index + 1 < stop:
            value = argv[index + 1]
            index += 1
        else:
            index += 1
            continue
        normalized = _normalized_common_value(name, value)
        previous = values.get(name)
        if previous is not None and previous != normalized:
            return f"conflicting values for {name}: {previous!r} and {value!r}."
        values[name] = normalized
        index += 1
    return None


def _is_paginated_json_argv(argv: list[str]) -> bool:
    """Return whether raw arguments select JSON task-list pagination."""
    task_index = next(
        (index for index in range(len(argv) - 1) if argv[index : index + 2] == ["task", "list"]),
        None,
    )
    if task_index is None:
        return False
    has_pagination_flag = any(
        item == "--page-size" or item.startswith("--page-size=") or item == "--cursor" or item.startswith("--cursor=")
        for item in argv[task_index + 2 :]
    )
    if not has_pagination_flag:
        return False
    return any(
        item == "--format=json" or (item == "--format" and index + 1 < len(argv) and argv[index + 1] == "json")
        for index, item in enumerate(argv)
    )


def _is_submission_json_argv(argv: list[str]) -> bool:
    prefix = _submission_raw_prefix(argv)
    return bool(prefix) and any(
        item == "--format=json" or (item == "--format" and index + 1 < len(prefix) and prefix[index + 1] == "json")
        for index, item in enumerate(prefix)
    )


def _is_json_argv(argv: list[str]) -> bool:
    """Return whether a pre-payload option requests JSON diagnostics."""
    stop = len(argv)
    try:
        submit_index = argv.index("submit")
        stop = argv.index("--", submit_index + 1)
    except (ValueError, IndexError):
        pass
    return any(
        item == "--format=json" or (item == "--format" and index + 1 < stop and argv[index + 1] == "json")
        for index, item in enumerate(argv[:stop])
    )


def _parse_page_size(value: str | None) -> int:
    """Parse a CLI page size so malformed values become ObservationErrors."""
    if value is None:
        return 50
    try:
        return int(value, 10)
    except (TypeError, ValueError) as exc:
        raise ObservationError("invalid_argument", "page_size must be an integer from 1 through 1000.", 2) from exc


def _emit_observation_error(error: ObservationError, output_format: str) -> int:
    """Render one stable observation error at the CLI boundary."""
    if output_format == "json":
        print(json.dumps({"error": {"code": error.code, "message": error.message}}))
    else:
        print(f"qexp: {error.message}", file=sys.stderr)
    return error.exit_code


def _is_wait_json_argv(argv: list[str]) -> bool:
    """Return whether raw argv selects JSON ``task wait`` diagnostics."""
    return any(argv[index : index + 2] == ["task", "wait"] for index in range(len(argv) - 1)) and _is_json_argv(argv)


def _raw_wait_task_id(argv: list[str]) -> str | None:
    """Recover the wait Task ID for parser errors without running argparse again."""
    option_values = {
        "--project",
        "--machine",
        "--runtime-root",
        "--machine-runtime-root",
        "--format",
        "--timeout",
    }
    for index in range(len(argv) - 1):
        if argv[index : index + 2] != ["task", "wait"]:
            continue
        cursor = index + 2
        while cursor < len(argv):
            token = argv[cursor]
            if token == "--":
                return None
            if token in option_values:
                cursor += 2
                continue
            if any(token.startswith(f"{option}=") for option in option_values):
                cursor += 1
                continue
            if token.startswith("-"):
                cursor += 1
                continue
            return token
    return None


def _emit_wait_error(
    *,
    task_id: str | None,
    project: object | None,
    outcome: str,
    reason: str,
    code: str,
    message: str,
    exit_code: int,
) -> int:
    """Emit the fixed task-wait schema for pre-resolution CLI failures."""
    result = wait_commands.make_wait_result(
        project=project,
        task_id=task_id,
        outcome=outcome,
        reason=reason,
        error={"code": code, "message": message},
    )
    _emit(CliOutput(OutputKind.TASK_WAIT, result), "json")
    return exit_code


def _emit_task_page(cfg: RootConfig, args: argparse.Namespace, page: dict[str, object], page_size: int) -> None:
    """Render paginated task results and a shell-safe continuation command."""
    presentation: dict[str, object] = {}
    next_cursor = page.get("next_cursor")
    if args.format == "human" and next_cursor is not None:
        command = [
            "qexp",
            "--project",
            str(cfg.shared_root),
            "task",
            "list",
        ]
        if args.phase:
            command.extend(("--phase", args.phase))
        if args.group:
            command.extend(("--group", args.group))
        if getattr(args, "name", None) is not None:
            command.extend(("--name", args.name))
        command.extend(("--page-size", str(page_size), "--cursor", str(next_cursor)))
        presentation["continuation_command"] = shlex.join(command)
    _emit(CliOutput(OutputKind.TASK_PAGE, page, presentation), args.format)


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    """Add a leaf format option without overwriting a root-level value."""
    if "--format" not in parser._option_string_actions:
        parser.add_argument("--format", choices=("human", "json"), default=argparse.SUPPRESS)


def _add_common_options(parser: argparse.ArgumentParser, *, include_format: bool = True) -> None:
    """Allow common context options at every ordinary command level."""
    options = (
        ("--project", {"help": "Project directory or its .qexp control directory."}),
        ("--machine", {"help": "Assert the local logical machine identity."}),
        ("--runtime-root", {"help": "Override the project-local runtime root."}),
        ("--machine-runtime-root", {"help": "Override the machine-global runtime root."}),
    )
    for option, kwargs in options:
        if option not in parser._option_string_actions:
            parser.add_argument(option, default=argparse.SUPPRESS, **kwargs)
    if include_format and "--format" not in parser._option_string_actions:
        parser.add_argument("--format", choices=("human", "json"), default=argparse.SUPPRESS)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _add_common_options(child, include_format=include_format)


def _emit(output: CliOutput[object], output_format: str, *, flush: bool = False) -> None:
    command_spec = _ACTIVE_COMMAND_SPEC.get()
    if command_spec is not None:
        allowed = _DYNAMIC_OUTPUT_KINDS.get(command_spec.output)
        if allowed is None:
            try:
                allowed = frozenset({OutputKind(command_spec.output)})
            except ValueError as exc:
                raise RuntimeError(
                    f"command handler {command_spec.handler!r} has no structured output contract "
                    f"for {command_spec.output!r}."
                ) from exc
        if output.kind not in allowed:
            raise RuntimeError(
                f"command handler {command_spec.handler!r} emitted {output.kind.value!r}; "
                f"registered output is {command_spec.output!r}."
            )
    print(render(output, output_format), flush=flush)


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


def _machine_assertion(args: argparse.Namespace) -> str | None:
    """Return the caller's identity assertion after checking duplicate inputs."""
    flag_value = getattr(args, "machine", None)
    environment_value = os.environ.get("QEXP_MACHINE")
    if flag_value is not None and environment_value is not None and flag_value != environment_value:
        raise ValueError(f"--machine {flag_value!r} conflicts with QEXP_MACHINE {environment_value!r}.")
    return flag_value if flag_value is not None else environment_value


def _requires_verified_binding(args: argparse.Namespace) -> bool:
    """Use the leaf's registered context policy as the sole write classification."""
    if args.command_spec.handler in {"config_set", "config_reset"}:
        return getattr(args, "section", None) != "agent"
    return args.command_spec.context == "project-write"


def _resolve_cfg(args: argparse.Namespace, *, require_binding: bool) -> tuple[object, ExecutionContext, str]:
    submission_request = getattr(args, "_submission_request", None)
    if args.command_spec.handler == "submit" and submission_request is not None:
        selection = context_commands.ProjectSelection(
            submission_request.project.control_root,
            submission_request.project.source,
        )
    else:
        selection = context_commands.resolve_project(getattr(args, "project", None))
    assertion = _machine_assertion(args)
    machine_runtime = MachineRuntime(getattr(args, "machine_runtime_root", None))

    if require_binding:
        try:
            execution_context = machine_runtime.verified_execution_context(selection.shared_root)
        except ValueError as exc:
            if str(exc).startswith("no local project binding exists"):
                raise ValueError(
                    f"{exc} To join this machine for the first time, run 'qexp project register <PATH>'; "
                    "to restore a missing current-generation binding, run "
                    "'qexp project register <PATH>'."
                ) from exc
            raise
        verified_machine = execution_context.cfg.machine_name
        if assertion is not None and assertion != verified_machine:
            raise ValueError(
                f"Local project binding is {verified_machine!r}, but --machine asserted {assertion!r}.\n"
                f"Use '--home-machine {assertion}' to select Task placement."
            )
        return execution_context.cfg, execution_context, selection.source

    # Read-only project commands should use the binding-owned local runtime when one is
    # available, while remaining usable for observation before local registration.
    try:
        execution_context = machine_runtime.verified_execution_context(selection.shared_root)
    except (ValueError, RuntimeError):
        execution_context = None
    if execution_context is not None:
        verified_machine = execution_context.cfg.machine_name
        if assertion is not None and assertion != verified_machine:
            raise ValueError(f"Local project binding is {verified_machine!r}, but --machine asserted {assertion!r}.")
        return execution_context.cfg, execution_context, selection.source

    # Read-only project observation must remain possible without a local binding. The sentinel is
    # never used as an authority source because this branch is not used for mutations.
    machine = assertion or "unbound"
    runtime = getattr(args, "runtime_root", None) or os.environ.get("QEXP_RUNTIME_ROOT")
    cfg = context_commands.config_for_selection(
        selection,
        machine_name=machine,
        runtime_root=runtime,
        require_initialized=True,
    )
    return cfg, ExecutionContext(cfg, machine_runtime), selection.source


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical qexp resource/action command tree."""
    parser = _QexpArgumentParser(
        prog="qexp",
        description=(
            "qexp schema-6 experiment queue. Project commands use --project; --machine asserts local identity, "
            "while --home-machine selects Task placement. Selecting placement does not remotely start an agent."
        ),
        epilog=(
            "Machine setup: qexp init --machine NAME. Create shared truth with qexp project init PATH, "
            "then enroll it with qexp project register PATH. qexp use only saves local CLI fallback context."
        ),
    )
    parser.add_argument("--project", help="Project directory or its .qexp control directory.")
    # Parse the former root-position spelling only far enough to provide the
    # bounded QQTOOLS-COMPAT-0014 diagnostic.  It is intentionally hidden from
    # help and never participates in ordinary Project selection.
    parser.add_argument("--shared-root", dest="compat_shared_root", help=argparse.SUPPRESS)
    parser.add_argument("--machine", help="Assert the local logical machine identity.")
    parser.add_argument("--runtime-root", help="Override the project-local runtime root.")
    parser.add_argument("--machine-runtime-root", help="Override the machine-global runtime root.")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init",
        help="Initialize or replace the machine runtime.",
        description=(
            "Initialize the local machine identity and global agent policy. Project creation and enrollment are "
            "separate; use qexp project register PATH after creating shared truth."
        ),
    )
    init.add_argument("--shared-root", dest="init_shared_root")
    init.add_argument("--machine", dest="init_machine")
    init.add_argument("--agent-mode", choices=("on_demand", "daemon"))
    init.add_argument("--detach-old-runtime", action="store_true")
    init.add_argument("--yes", action="store_true")
    init.add_argument("--cpu-lane-capacity", type=int)
    _add_output_format(init)
    bind_command(init, handler="init", context="machine", output="machine-init")

    project = commands.add_parser(
        "project",
        help="Create shared Projects and manage local enrollment.",
        description="Project setup owns shared truth and machine enrollment.",
    )
    project_sub = project.add_subparsers(dest="project_action", required=True)
    project_init = project_sub.add_parser("init", help="Create shared Project truth without enrollment.")
    project_init.add_argument("path", nargs="?")
    _add_output_format(project_init)
    bind_command(project_init, handler="project_init", context="setup", output="project-operation")
    project_register = project_sub.add_parser("register", help="Enroll explicit Projects or saved inventory entries.")
    project_register.add_argument("paths", nargs="*")
    project_register.add_argument("--from-pool", action="store_true")
    project_register.add_argument("--machine", dest="project_machine")
    project_register.add_argument("--name-source", choices=("default", "explicit"))
    _add_output_format(project_register)
    bind_command(project_register, handler="project_register", context="machine", output="project-register")
    project_list = project_sub.add_parser("list", help="List local Project registrations.")
    _add_output_format(project_list)
    bind_command(project_list, handler="project_list", context="machine", output="project-list")
    for project_action in ("enable", "disable", "remove"):
        selector = project_sub.add_parser(project_action, help=f"{project_action.capitalize()} a Project registration.")
        selector.add_argument("selector")
        _add_output_format(selector)
        bind_command(selector, handler=f"project_{project_action}", context="machine", output="project-operation")

    use = commands.add_parser(
        "use",
        help="Save or inspect fallback Project context without initialization.",
        description=(
            "Use --project to save a fallback locator, or --show/--clear to inspect or clear it. "
            "This does not initialize a shared root or register the project with the local machine agent."
        ),
    )
    use.add_argument("--project", dest="use_project", help="Project directory or .qexp control directory to save.")
    use.add_argument("--show", action="store_true", help="Show the saved fallback Project.")
    use.add_argument("--clear", action="store_true", help="Clear the saved fallback Project.")
    _add_output_format(use)
    bind_command(use, handler="use", context="none", output="context")

    submit = commands.add_parser(
        "submit",
        help="Submit one Task using the verified local identity.",
        description=(
            "Submit one command or a manifest using the verified local identity. "
            "qexp does not remotely start the target agent."
        ),
    )
    _add_output_format(submit)
    submit.add_argument("--project", dest="project", default=argparse.SUPPRESS)
    submit.add_argument("-f", "--file", dest="manifest_file")
    submit.add_argument("--task-id")
    submit.add_argument("--name")
    submit.add_argument("--group")
    submit.add_argument("--gpus", type=int, default=None)
    submit.add_argument("--cpus", type=int, default=None)
    submit.add_argument("--cwd")
    submit.add_argument("--home-machine", default=None)
    submit.add_argument("--sharing", choices=["private", "spillover"], default=None)
    submit.add_argument("--offer-after-seconds", type=int, default=None)
    submit.add_argument("--idempotency-key")
    submit.add_argument("--depends-on", action="append", default=None)
    submit.add_argument("--no-activate", action="store_true", help="Submit without activating the local agent.")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--quiet", action="store_true")
    tmux_values = submit.add_mutually_exclusive_group()
    tmux_values.add_argument("--tmux", dest="tmux_override", action="store_true")
    tmux_values.add_argument("--no-tmux", dest="tmux_override", action="store_false")
    submit.set_defaults(tmux_override=None)
    submit.add_argument("argv", nargs=argparse.REMAINDER)
    bind_command(submit, handler="submit", context="project-write", output="submission")

    status = commands.add_parser("status", help="Show a bounded Project overview and next actions.")
    _add_output_format(status)
    bind_command(status, handler="status", context="project-read", output="status")

    task = commands.add_parser("task", help="Inspect and operate on Tasks in the selected Project.")
    task_sub = task.add_subparsers(dest="task_action", required=True)
    cancel = task_sub.add_parser("cancel", help="Cancel one Task and preserve its lifecycle identity.")
    cancel.add_argument("task_id")
    _add_output_format(cancel)
    bind_command(cancel, handler="task_cancel", context="project-write", output="task-operation")
    retry = task_sub.add_parser("retry", help="Queue the next Attempt for one failed Task.")
    retry.add_argument("task_id")
    retry.add_argument("--quiet", action="store_true", help="Print only the retained Task ID.")
    _add_output_format(retry)
    bind_command(retry, handler="task_retry", context="project-write", output="task-operation")
    share = task_sub.add_parser("share", help="Enable immediate or delayed Task sharing.")
    share.add_argument("task_id")
    share.add_argument("--after")
    share.add_argument("--with", dest="helper_machines", action="append")
    _add_output_format(share)
    bind_command(share, handler="task_share", context="project-write", output="availability")
    unshare = task_sub.add_parser("unshare", help="Return a queued Task to private home placement.")
    unshare.add_argument("task_id")
    _add_output_format(unshare)
    bind_command(unshare, handler="task_unshare", context="project-write", output="availability")
    offer = task_sub.add_parser("offer", help="Offer eligible spillover work immediately.")
    offer.add_argument("task_id")
    _add_output_format(offer)
    bind_command(offer, handler="task_offer", context="project-write", output="availability")
    listing = task_sub.add_parser("list", help="List Tasks with bounded filters and pagination.")
    listing.add_argument("--phase")
    listing.add_argument("--group")
    listing.add_argument("--name", help="Exact, case-sensitive Task name filter.")
    listing.add_argument("--limit", type=int, default=None)
    listing.add_argument("--page-size", default=None)
    listing.add_argument("--cursor", default=None)
    _add_output_format(listing)
    bind_command(listing, handler="task_list", context="project-read", output="task-list")
    show = task_sub.add_parser(
        "show",
        help="Show one Task and its bounded Attempt history.",
        description=(
            "Show one Task. --watch performs screen refreshes; it does not change progress reporting. "
            "--follow-retries continues onto a later retry."
        ),
    )
    show.add_argument("task_id")
    show.add_argument("--watch", action="store_true")
    show.add_argument("--interval-seconds", default=None)
    show.add_argument("--follow-retries", action="store_true")
    _add_output_format(show)
    bind_command(show, handler="task_show", context="project-read", output="task-show")
    logs = task_sub.add_parser(
        "logs",
        help="Read Task logs or explicitly follow a live byte stream.",
        description=(
            "Read finite logs or follow each Attempt and file generation. --follow-retries continues onto a later "
            "retry."
        ),
    )
    logs.add_argument("task_id")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--tail", default=None)
    logs.add_argument("--interval-seconds", default=None)
    logs.add_argument("--follow-retries", action="store_true")
    _add_output_format(logs)
    bind_command(logs, handler="task_logs", context="project-read", output="raw-logs")
    wait = task_sub.add_parser("wait", help="Wait for one selected Task lifecycle without mutating it.")
    wait.add_argument("task_id")
    wait.add_argument("--timeout", default=None)
    _add_output_format(wait)
    bind_command(wait, handler="task_wait", context="project-read", output="task-wait")
    dependencies = task_sub.add_parser("dependencies", help="Inspect or atomically edit Task dependencies.")
    dependencies_sub = dependencies.add_subparsers(dest="dependencies_action", required=True)
    for name in ("show", "replace", "add", "remove"):
        action = dependencies_sub.add_parser(name, help=f"{name.capitalize()} Task dependencies.")
        action.add_argument("task_id")
        if name != "show":
            action.add_argument("--depends-on", action="append", default=[])
        _add_output_format(action)
        bind_command(
            action,
            handler=f"task_dependencies_{name}",
            context="project-write" if name != "show" else "project-read",
            output="dependencies",
        )

    group = commands.add_parser("group", help="Inspect Groups and control admission, dispatch, and workers.")
    group_sub = group.add_subparsers(dest="group_action", required=True)
    create = group_sub.add_parser("create", help="Create a Group and its initial Worker Set.")
    create.add_argument("name")
    create.add_argument("--workers", nargs="*", default=None)
    _add_output_format(create)
    bind_command(create, handler="group_create", context="project-write", output="group-operation")
    group_list = group_sub.add_parser("list", help="List Groups in the selected Project.")
    _add_output_format(group_list)
    bind_command(group_list, handler="group_list", context="project-read", output="group-list")
    show_group = group_sub.add_parser("show", help="Show one Group and its Worker Set.")
    show_group.add_argument("name")
    _add_output_format(show_group)
    bind_command(show_group, handler="group_show", context="project-read", output="group-show")
    for name in ("seal", "reopen", "pause", "resume", "cancel", "retry"):
        action = group_sub.add_parser(name, help=f"{name.capitalize()} Group control state.")
        action.add_argument("name")
        if name == "cancel":
            action.add_argument("--all", action="store_true", help="Include launch-authorized and running Tasks.")
        _add_output_format(action)
        bind_command(action, handler=f"group_{name}", context="project-write", output="group-operation")
    worker = group_sub.add_parser("worker", help="Manage Group Worker Set membership and lifecycle.")
    worker_sub = worker.add_subparsers(dest="worker_action", required=True)
    for name in ("list", "add", "set", "drain", "resume", "remove"):
        action = worker_sub.add_parser(name, help=f"{name.capitalize()} a Group worker.")
        action.add_argument("group_name")
        if name != "list":
            action.add_argument("worker_machine")
        if name in {"add", "set"}:
            action.add_argument("--role", choices=("primary", "borrow"))
            action.add_argument("--gpu-limit-gpus", type=_gpu_limit_gpus)
        if name == "remove":
            action.add_argument("--all", action="store_true", help="Terminate running work while removing.")
        _add_output_format(action)
        bind_command(
            action,
            handler=f"group_worker_{name}",
            context="project-write" if name != "list" else "project-read",
            output="group-operation",
        )

    machine = commands.add_parser("machine", help="Inspect declared Project machines.")
    machine_sub = machine.add_subparsers(dest="machine_action", required=True)
    machine_list = machine_sub.add_parser("list", help="List bounded machine observations.")
    _add_output_format(machine_list)
    bind_command(machine_list, handler="machine_list", context="project-read", output="machines")
    machine_show = machine_sub.add_parser("show", help="Show one machine declaration and observations.")
    machine_show.add_argument("name")
    _add_output_format(machine_show)
    bind_command(machine_show, handler="machine_show", context="project-read", output="machine-show")

    agent = commands.add_parser("agent", help="Manage the local machine agent and its global resources.")
    agent_sub = agent.add_subparsers(dest="agent_action", required=True)
    for name in ("start", "run", "restart", "status", "stop"):
        action = agent_sub.add_parser(name, help=f"{name.capitalize()} the machine-wide local agent.")
        _add_output_format(action)
        output = "agent-status" if name == "status" else "agent-operation"
        bind_command(action, handler=f"agent_{name}", context="machine", output=output)
    agent_sub.choices["start"].add_argument("--timeout", type=float, default=30.0)
    agent_name = agent_sub.add_parser("name", help="Show or change the machine-global agent name.")
    agent_name.add_argument("--set-to", dest="set_to")
    _add_output_format(agent_name)
    bind_command(agent_name, handler="agent_name", context="machine", output="agent-config")
    for legacy_name in (
        "add-project",
        "list-projects",
        "enable-project",
        "disable-project",
        "remove-project",
        "migrate-project",
    ):
        legacy = agent_sub.add_parser(
            legacy_name,
            help="Retired non-executing compatibility diagnostic for Project enrollment commands.",
        )
        legacy.add_argument("legacy_selector", nargs="?")
        if legacy_name == "add-project":
            legacy.add_argument("--adopt-existing", action="store_true")
        bind_command(legacy, handler=f"retired_{legacy_name}", context="none", output="diagnostic")
    agent_config = agent_sub.add_parser("config", help="Inspect or set machine-wide agent resources.")
    agent_config_sub = agent_config.add_subparsers(dest="agent_config_resource", required=True)
    gpus = agent_config_sub.add_parser("gpus", help="Inspect or change GPU admission policy.")
    gpus_sub = gpus.add_subparsers(dest="gpu_action", required=True)
    gpu_show = gpus_sub.add_parser("show", help="Show GPU admission policy.")
    _add_output_format(gpu_show)
    bind_command(gpu_show, handler="agent_config_gpus_show", context="machine", output="gpu-policy")
    gpu_set = gpus_sub.add_parser("set", help="Set visible GPU IDs or disable GPU admission.")
    gpu_set_values = gpu_set.add_mutually_exclusive_group(required=True)
    gpu_set_values.add_argument("--visible")
    gpu_set_values.add_argument("--none", action="store_true")
    gpu_set.add_argument("--expected-revision", type=int)
    _add_output_format(gpu_set)
    bind_command(gpu_set, handler="agent_config_gpus_set", context="machine", output="gpu-policy")
    gpu_reset = gpus_sub.add_parser("reset", help="Reset GPU admission policy.")
    gpu_reset.add_argument("--expected-revision", type=int)
    _add_output_format(gpu_reset)
    bind_command(gpu_reset, handler="agent_config_gpus_reset", context="machine", output="gpu-policy")
    cpu = agent_config_sub.add_parser("cpu", help="Inspect or set machine-wide CPU lane capacity.")
    cpu_sub = cpu.add_subparsers(dest="cpu_action", required=True)
    cpu_show = cpu_sub.add_parser("show", help="Show CPU lane capacity.")
    _add_output_format(cpu_show)
    bind_command(cpu_show, handler="agent_config_cpu_show", context="machine", output="cpu-lane")
    cpu_set = cpu_sub.add_parser("set", help="Set CPU lane capacity.")
    cpu_set.add_argument("--capacity", type=int, required=True)
    _add_output_format(cpu_set)
    bind_command(cpu_set, handler="agent_config_cpu_set", context="machine", output="cpu-lane")

    config = commands.add_parser("config", help="Inspect or set typed Project and global agent configuration.")
    config_sub = config.add_subparsers(dest="config_action", required=True)
    config_show = config_sub.add_parser("show", help="Show all Project sections or one typed section.")
    config_show.add_argument("section", nargs="?", choices=configuration_commands.CONFIG_SECTIONS)
    _add_output_format(config_show)
    bind_command(config_show, handler="config_show", context="section", output="config")
    config_set = config_sub.add_parser("set", help="Set typed options in one configuration section.")
    config_set.add_argument("section", choices=configuration_commands.CONFIG_SECTIONS)
    config_set.add_argument("--provider")
    config_set.add_argument("--name")
    config_set.add_argument("--agent-mode", choices=("daemon", "on_demand"))
    config_boolean = config_set.add_mutually_exclusive_group()
    config_boolean.add_argument("--enabled", action="store_true", default=None)
    config_boolean.add_argument("--disabled", action="store_true", default=None)
    config_set.add_argument("--interval-seconds", type=float)
    config_set.add_argument("--timeout-seconds", type=float)
    config_set.add_argument("--ttl-seconds", type=int)
    config_set.add_argument("--renew-interval-seconds", type=float)
    config_set.add_argument("--max-clock-skew-seconds", type=float)
    config_set.add_argument("--clock-observation-max-age-seconds", type=float)
    config_set.add_argument("--clock-provider-margin-seconds", type=float)
    config_set.add_argument("--clock-provider-priority")
    config_set.add_argument("--renewal-commit-margin-seconds", type=float)
    config_set.add_argument("--webhook-env")
    config_set.add_argument("--credential-source", choices=("env", "shared_file"))
    config_set.add_argument("--webhook-stdin", action="store_true")
    config_set.add_argument("--acknowledge-shared-secret-risk", action="store_true")
    config_set.add_argument("--secret-env")
    config_set.add_argument("--unset-secret-env", action="store_true")
    config_set.add_argument("--shared-webhook")
    _add_output_format(config_set)
    bind_command(config_set, handler="config_set", context="section", output="config")
    config_reset = config_sub.add_parser("reset", help="Reset one typed configuration override.")
    config_reset.add_argument("section", choices=configuration_commands.CONFIG_SECTIONS)
    config_reset.add_argument("--provider")
    _add_output_format(config_reset)
    bind_command(config_reset, handler="config_reset", context="section", output="config")

    admin = commands.add_parser("admin", help="Run bounded Project maintenance and machine upgrade operations.")
    admin_sub = admin.add_subparsers(dest="admin_action", required=True)
    for name in ("check", "repair"):
        action = admin_sub.add_parser(name, help=f"Run bounded metadata {name} checks.")
        action.add_argument("--strict", action="store_true")
        action.add_argument("--max-work-items", type=int, default=64)
        _add_output_format(action)
        bind_command(
            action,
            handler=f"admin_{name}",
            context="project-write" if name == "repair" else "project-read",
            output=f"doctor-{name}",
        )
    clean = admin_sub.add_parser(
        "clean",
        help="Clean terminal Task metadata under bounded retention rules.",
        description="Clean retained qexp metadata while preserving experiment work directories.",
    )
    clean_scope = clean.add_mutually_exclusive_group()
    clean_scope.add_argument("--task-id")
    clean_scope.add_argument("--group")
    clean.add_argument("--older-than-days", type=int, default=30)
    clean.add_argument("--limit", type=int, default=100)
    clean.add_argument("--max-work-items", type=int, default=64)
    clean.add_argument("--dry-run", action="store_true")
    _add_output_format(clean)
    bind_command(clean, handler="admin_clean", context="project-write", output="clean")
    operation = admin_sub.add_parser("operation", help="Inspect one durable asynchronous operation reference.")
    operation_sub = operation.add_subparsers(dest="operation_action", required=True)
    operation_show = operation_sub.add_parser("show", help="Inspect a Group, worker, or cleanup operation reference.")
    operation_show.add_argument("reference")
    _add_output_format(operation_show)
    bind_command(operation_show, handler="admin_operation_show", context="project-read", output="operation")
    admin_upgrade = admin_sub.add_parser("upgrade", help="Inspect or advance machine-global Project upgrades.")
    admin_upgrade_sub = admin_upgrade.add_subparsers(dest="admin_upgrade_action", required=True)
    for name in ("status", "advance"):
        action = admin_upgrade_sub.add_parser(name, help=f"{name.capitalize()} registered Project upgrades.")
        _add_output_format(action)
        bind_command(action, handler=f"admin_upgrade_{name}", context="machine", output="upgrade")
    pause = admin_upgrade_sub.add_parser("pause", help="Pause a registered Project upgrade.")
    pause.add_argument("--reason", required=True)
    _add_output_format(pause)
    bind_command(pause, handler="admin_upgrade_pause", context="registered-project", output="upgrade")
    plan = admin_upgrade_sub.add_parser("plan", help="Plan a registered Project upgrade repair.")
    plan.add_argument("--target", required=True)
    _add_output_format(plan)
    bind_command(plan, handler="admin_upgrade_plan", context="registered-project", output="upgrade-repair")
    for name in ("apply", "validate"):
        action = admin_upgrade_sub.add_parser(name, help=f"{name.capitalize()} a registered Project upgrade repair.")
        action.add_argument("--repair-id", required=True)
        _add_output_format(action)
        bind_command(action, handler=f"admin_upgrade_{name}", context="registered-project", output="upgrade-repair")
    resume = admin_upgrade_sub.add_parser("resume", help="Resume a registered Project upgrade.")
    _add_output_format(resume)
    bind_command(resume, handler="admin_upgrade_resume", context="registered-project", output="upgrade")
    migrate = admin_sub.add_parser("migrate", help="Run explicit schema and legacy-agent migrations.")
    migrate_sub = migrate.add_subparsers(dest="migrate_action", required=True)
    schema = migrate_sub.add_parser("schema", help="Convert an explicit Project to schema 6.")
    schema.add_argument("--to-schema", type=int, required=True)
    _add_output_format(schema)
    bind_command(schema, handler="admin_migrate_schema", context="explicit-project", output="schema6-upgrade")
    schema6 = migrate_sub.add_parser("schema6", help="Run drained schema-6 capability activation.")
    schema6_sub = schema6.add_subparsers(dest="schema6_upgrade_action", required=True)
    for name in ("check", "start", "status"):
        action = schema6_sub.add_parser(name, help=f"Schema-6 upgrade {name}.")
        if name != "status":
            action.add_argument(
                "--capability", action="append", dest="capabilities", choices=("cpu-lane-v1", "task-dependencies-v1")
            )
        _add_output_format(action)
        bind_command(
            action, handler=f"admin_migrate_schema6_{name}", context="explicit-project", output="schema6-upgrade"
        )
    attest = schema6_sub.add_parser("attest", help="Attest stopped clients for schema-6 activation.")
    attest.add_argument("--activation-id", required=True)
    attest.add_argument("--confirm-clients-stopped", action="store_true")
    _add_output_format(attest)
    bind_command(attest, handler="admin_migrate_schema6_attest", context="explicit-project", output="schema6-upgrade")
    resume_schema6 = schema6_sub.add_parser("resume", help="Resume schema-6 activation.")
    resume_schema6.add_argument("--activation-id", required=True)
    _add_output_format(resume_schema6)
    bind_command(
        resume_schema6, handler="admin_migrate_schema6_resume", context="explicit-project", output="schema6-upgrade"
    )
    legacy = migrate_sub.add_parser("agent", help="Migrate legacy Project agent metadata.")
    _add_output_format(legacy)
    bind_command(legacy, handler="admin_migrate_agent", context="explicit-project", output="agent-operation")

    # Add common options to every ordinary parser level so options can appear
    # before or after a command path.  Setup parsers retain their feature-owned
    # explicit options; ``use`` has its own --project selector.
    ordinary = {"status", "task", "group", "machine", "agent", "config", "admin", "submit"}
    for name in ordinary:
        _add_common_options(commands.choices[name], include_format=True)
    _add_common_options(parser, include_format=False)
    return parser


def _command(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ValueError("submit requires a command after '--'.")
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
        raise ValueError("submit requires exactly one input mode: --file MANIFEST or -- COMMAND....")
    if not has_file and not has_command:
        raise ValueError("submit requires exactly one input mode: --file MANIFEST or -- COMMAND....")
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
        raise ValueError("submission request preparation requires submit command.")
    invocation_cwd = Path(invocation_cwd or Path.cwd()).expanduser().resolve()
    try:
        submission_tail = raw_argv[raw_argv.index("submit") + 1 :]
    except ValueError:
        submission_tail = []
    if getattr(args, "argv", None) and "--" not in submission_tail:
        raise ValueError("command mode requires the literal '--' separator before COMMAND.")
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
                raise ValueError(message)
    elif _submission_option_supplied(raw_argv, "--file", "-f"):
        raise ValueError("--file is only valid in file mode.")
    if args.quiet and args.format == "json":
        raise ValueError("--quiet cannot be combined with --format json.")
    if args.quiet and args.dry_run:
        raise ValueError("--quiet cannot be combined with --dry-run.")

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
    selection = resolve_submission_project(
        explicit_project=explicit_project,
        manifest_path=manifest_path,
        invocation_cwd=invocation_cwd,
        environment_value=environment_value,
        saved_context=saved_context,
    )
    args._submission_project_resolved = True
    if mode == "command":
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
        result = parse_submission_manifest(
            manifest_path,
            group_name=args.group if args.group is not None else UNSET,
            tmux_override=args.tmux_override if _submission_option_supplied(raw_argv, "--tmux", "--no-tmux") else UNSET,
            requested_gpus=args.gpus if _submission_option_supplied(raw_argv, "--gpus") else UNSET,
            requested_cpus=args.cpus if _submission_option_supplied(raw_argv, "--cpus") else UNSET,
            home_machine=args.home_machine if _submission_option_supplied(raw_argv, "--home-machine") else UNSET,
            working_directory=args.cwd if _submission_option_supplied(raw_argv, "--cwd") else UNSET,
            project_directory=selection.path,
            invocation_cwd=invocation_cwd,
        )
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


def _duration_seconds(value: str) -> int:
    matched = re.fullmatch(r"([0-9]+)([smh])", value)
    if not matched:
        raise ValueError("duration must use an explicit s, m, or h unit, for example 10m.")
    amount = int(matched.group(1))
    return amount * {"s": 1, "m": 60, "h": 3600}[matched.group(2)]


def _parse_progress_interval_argument(value: str) -> int | float:
    """Parse the policy value while retaining integer JSON/text output."""
    text = value.strip()
    if not text:
        raise ValueError("--interval-seconds requires a finite number of at least 1 second.")
    try:
        parsed: int | float = int(text, 10)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError("--interval-seconds must be a finite number of at least 1 second.") from exc
    return validate_interval_seconds(parsed)


def _parse_continuous_interval_argument(value: str) -> int | float:
    """Parse a continuous viewer interval before any observation I/O starts."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--interval-seconds must be a finite number of at least 1 second.")
    text = value.strip()
    try:
        parsed: int | float = int(text, 10)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError("--interval-seconds must be a finite number of at least 1 second.") from exc
    try:
        return validate_interval_seconds(parsed)
    except ValueError as exc:
        raise ValueError("--interval-seconds must be a finite number of at least 1 second.") from exc


def _parse_follow_tail_argument(value: str) -> int:
    """Parse the nonnegative base-10 line count for log following."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value.strip()) is None:
        raise ValueError("--tail must be a non-negative base-10 integer.")
    return int(value.strip(), 10)


def _is_continuous_task(args: argparse.Namespace) -> bool:
    return (args.command_spec.handler == "task_show" and bool(getattr(args, "watch", False))) or (
        args.command_spec.handler == "task_logs" and bool(getattr(args, "follow", False))
    )


def _validate_continuous_options(args: argparse.Namespace) -> None:
    """Validate continuous-only flags before resolving a project or reading Task truth."""
    handler = args.command_spec.handler
    if handler == "task_show":
        watch = bool(getattr(args, "watch", False))
        interval = getattr(args, "interval_seconds", None)
        follow_retries = bool(getattr(args, "follow_retries", False))
        if not watch:
            if interval is not None:
                raise ValueError("--interval-seconds requires --watch.")
            if follow_retries:
                raise ValueError("--follow-retries requires --watch.")
            return
        if getattr(args, "format", "human") == "json":
            raise ValueError("--watch cannot be combined with --format json.")
        args.interval_seconds = 2 if interval is None else _parse_continuous_interval_argument(interval)
        isatty = getattr(sys.stdout, "isatty", None)
        if not callable(isatty) or not isatty():
            raise ValueError("--watch requires terminal stdout.")
        return
    if handler != "task_logs":
        return
    follow = bool(getattr(args, "follow", False))
    tail = getattr(args, "tail", None)
    interval = getattr(args, "interval_seconds", None)
    follow_retries = bool(getattr(args, "follow_retries", False))
    if not follow:
        if interval is not None:
            raise ValueError("--interval-seconds requires --follow.")
        if follow_retries:
            raise ValueError("--follow-retries requires --follow.")
        if tail is not None:
            args.tail = _parse_follow_tail_argument(tail)
        return
    args.tail = 100 if tail is None else _parse_follow_tail_argument(tail)
    args.interval_seconds = 2 if interval is None else _parse_continuous_interval_argument(interval)


def _restore_watch_terminal() -> None:
    """Leave one readable line after an interrupted in-place watch."""
    try:
        sys.stdout.write(chr(27) + "[0m\n")
        sys.stdout.flush()
    except BrokenPipeError:
        pass


def _parse_launch_handoff_timeout_argument(value: str) -> int | float:
    """Parse a launch timeout so malformed values become a specific ValueError."""
    text = value.strip()
    if not text:
        raise ValueError("--timeout-seconds requires a finite number from 1 through 300 seconds.")
    try:
        parsed: int | float = int(text, 10)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError("--timeout-seconds must be a finite number from 1 through 300 seconds.") from exc
    return validate_launch_handoff_timeout_seconds(parsed)


def _gpu_limit_gpus(value: str) -> int | str:
    if value == "unlimited":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gpu-limit-gpus must be a positive integer or 'unlimited'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("gpu-limit-gpus must be a positive integer or 'unlimited'.")
    return parsed


def _split_machine_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    machines = [machine.strip() for value in values for machine in value.split(",")]
    if not all(machines):
        raise ValueError("--with machine names must be comma-separated non-empty values.")
    return machines


def _try_save_context(shared_root: str) -> None:
    try:
        save_context(shared_root)
    except OSError as exc:
        print(
            "qexp: initialized successfully, but failed to save CLI context "
            f"at {exc.filename or '~/.qqtools/qexp-context.json'}: {exc}",
            file=sys.stderr,
        )


def _upgrade_project_config(runtime: MachineRuntime, identifier: str):
    """Resolve an explicit registered project for a repair or project-scoped retry."""
    candidate = str(identifier)
    canonical = Path(candidate).expanduser().resolve() if candidate.endswith(".qexp") or "/" in candidate else None
    _revision, bindings = runtime.load_registry()
    matches = [
        binding
        for binding in bindings
        if binding.project_id == candidate or (canonical is not None and binding.shared_root == canonical)
    ]
    if len(matches) != 1:
        raise ValueError(f"machine registry must identify exactly one project for {identifier!r}.")
    return matches[0].root_config()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    global _JSON_PARSE_MODE, _PAGINATION_JSON_PARSE_MODE, _SUBMISSION_JSON_PARSE_MODE
    _PAGINATION_JSON_PARSE_MODE = _is_paginated_json_argv(raw_argv)
    _SUBMISSION_JSON_PARSE_MODE = _is_submission_json_argv(raw_argv)
    _JSON_PARSE_MODE = _is_json_argv(raw_argv)
    try:
        args = build_parser().parse_args(raw_argv)
    except _PaginationParseError as exc:
        print(json.dumps({"error": {"code": "invalid_argument", "message": str(exc)}}))
        return 2
    except _SubmissionParseError as exc:
        payload = _submission_parse_failure_payload(raw_argv, str(exc))
        print(json.dumps(payload))
        print(f"qexp: {exc}", file=sys.stderr)
        return 2
    except _JsonParseError as exc:
        if _is_wait_json_argv(raw_argv):
            return _emit_wait_error(
                task_id=_raw_wait_task_id(raw_argv),
                project=None,
                outcome="invalid_input",
                reason="invalid_input",
                code="invalid_input",
                message=str(exc),
                exit_code=2,
            )
        print(json.dumps({"error": {"code": "invalid_argument", "message": str(exc)}}))
        return 2
    finally:
        _PAGINATION_JSON_PARSE_MODE = False
        _SUBMISSION_JSON_PARSE_MODE = False
        _JSON_PARSE_MODE = False
    resolved_cfg = None
    selection_source = None
    handler = args.command_spec.handler
    command_spec_token = _ACTIVE_COMMAND_SPEC.set(args.command_spec)
    try:
        _validate_continuous_options(args)
        if handler == "task_wait":
            # Invocation validity is independent of Project discovery.  Check
            # it first so a bad duration cannot be mislabeled as an
            # observation failure merely because context is also unavailable.
            wait_commands.parse_wait_timeout(args.timeout)
        if getattr(args, "compat_shared_root", None) is not None:
            # Parse the former root-position locator only far enough to give a
            # bounded migration diagnostic.  It never participates in normal
            # Project selection.
            print(
                "qexp: QQTOOLS-COMPAT-0014: --shared-root is retired; use "
                f"'--project {shlex.quote(str(args.compat_shared_root))}'.",
                file=sys.stderr,
            )
            return 2
        if handler == "init":
            # QQTOOLS-COMPAT-0014: the former project-bound init spelling is
            # retained only as a bounded diagnostic during the transition.
            if args.init_shared_root or args.runtime_root or args.cpu_lane_capacity is not None:
                print(
                    "qexp: QQTOOLS-COMPAT-0014: qexp init is machine-only. "
                    "Use 'qexp project init PATH' to create shared Project truth, then "
                    "'qexp project register PATH --machine NAME' to enroll it.",
                    file=sys.stderr,
                )
                return 2
            target_name = args.init_machine or args.machine
            if not target_name:
                raise ValueError("init requires explicit --machine NAME.")
            runtime = MachineRuntime(args.machine_runtime_root)
            facts = machine_init_facts(runtime)
            confirmed = bool(args.yes)
            expected_old_runtime_id = None
            if args.detach_old_runtime and facts.get("old_runtime_id"):
                print(
                    "WARNING: --detach-old-runtime preserves old local evidence in an isolated archive "
                    "without completing recovery or changing shared Tasks, claims, or registrations.",
                    file=sys.stderr,
                )
            if facts.get("old_runtime_id") and not confirmed:
                if args.format == "json":
                    raise RuntimeError("machine identity replacement requires --yes when --format=json is selected.")
                isatty = getattr(sys.stdin, "isatty", None)
                if not callable(isatty) or not isatty():
                    raise RuntimeError("machine identity replacement requires --yes in noninteractive input.")
                print("Machine identity replacement will:", file=sys.stderr)
                print(f"  old runtime ID: {facts.get('old_runtime_id')}", file=sys.stderr)
                print(f"  current name: {facts.get('current_name') or '-'}", file=sys.stderr)
                print(f"  requested name: {target_name}", file=sys.stderr)
                print(
                    f"  requested mode: {args.agent_mode or facts.get('current_agent_mode') or 'daemon'}",
                    file=sys.stderr,
                )
                print(f"  obligations: {facts.get('obligations') or 'none'}", file=sys.stderr)
                print("  this creates a new runtime ID and replaces local authority.", file=sys.stderr)
                print(
                    f"  shell-safe command: qexp init --machine {shlex.quote(target_name)} --yes",
                    file=sys.stderr,
                )
                answer = input("Continue? [y/N] ")
                if answer.strip().lower() not in {"y", "yes"}:
                    raise RuntimeError("machine identity replacement cancelled; no state was changed.")
                confirmed = True
                expected_old_runtime_id = facts["old_runtime_id"]
            result = initialize_machine(
                runtime,
                target_name,
                agent_mode=args.agent_mode,
                detach_old_runtime=args.detach_old_runtime,
                confirmed=confirmed,
                expected_old_runtime_id=expected_old_runtime_id,
            )
            _emit(CliOutput(OutputKind.MACHINE_INIT, result), args.format)
            return 0
        if handler.startswith("project_"):
            runtime = MachineRuntime(args.machine_runtime_root)
            if handler == "project_init":
                result = initialize_project(args.path)
                _emit(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
                return 0
            if handler == "project_register":
                project_machine = args.project_machine
                if project_machine is not None and args.machine is not None and project_machine != args.machine:
                    raise ValueError("project register --machine conflicts with the global --machine assertion.")
                if project_machine is None:
                    project_machine = args.machine
                result = register_projects(
                    runtime,
                    args.paths,
                    from_pool=args.from_pool,
                    machine_name=project_machine,
                    name_source=args.name_source,
                )
                _emit(CliOutput(OutputKind.PROJECT_REGISTER, result), args.format)
                statuses = [item.get("status") for item in result.get("projects", ())]
                return 0 if not statuses or all(item in {"registered", "disabled"} for item in statuses) else 2
            if handler == "project_list":
                _emit(CliOutput(OutputKind.PROJECT_LIST, list_projects(runtime)), args.format)
                return 0
            if handler == "project_remove":
                result = remove_project(runtime, args.selector)
            else:
                result = set_project_enablement(runtime, args.selector, handler == "project_enable")
            _emit(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
            return 0
        if handler in {"config_show", "config_set", "config_reset"} and getattr(args, "section", None) == "agent":
            runtime = MachineRuntime(args.machine_runtime_root)
            if handler == "config_show":
                result = configuration_commands.show_config("agent", cfg=None, runtime=runtime)
            elif handler == "config_set":
                values = {
                    key: value
                    for key, value in {"name": args.name, "agent_mode": args.agent_mode}.items()
                    if value is not None
                }
                result = configuration_commands.set_config(
                    "agent", cfg=None, runtime=runtime, provider=args.provider, values=values
                )
            else:
                # Agent configuration is machine-global and has no reset
                # default.  Reject it without resolving a Project context.
                result = configuration_commands.reset_config("agent", cfg=None, runtime=runtime)
            _emit(CliOutput(OutputKind.CONFIG, result), args.format)
            return 0
        if handler == "agent_name":
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.set_to is not None:
                set_agent_config(runtime, name=args.set_to)
            result = agent_config_payload(runtime)
            _emit(CliOutput(OutputKind.AGENT_CONFIG, result), args.format)
            return 0
        if handler.startswith("retired_"):
            # QQTOOLS-COMPAT-0014: retired executing routes fail before any
            # runtime/configuration resolution or state creation.
            replacement = {
                "add-project": "qexp project register PATH",
                "list-projects": "qexp project list",
                "enable-project": "qexp project enable SELECTOR",
                "disable-project": "qexp project disable SELECTOR",
                "remove-project": "qexp project remove SELECTOR",
                "migrate-project": "qexp admin migrate agent --project PATH --machine NAME",
            }[handler.removeprefix("retired_")]
            print(
                f"qexp: QQTOOLS-COMPAT-0014: retired command; use '{replacement}'.",
                file=sys.stderr,
            )
            return 2
        if handler in {"agent_start", "agent_run"}:
            runtime = MachineRuntime(args.machine_runtime_root)
            if handler == "agent_start" and (
                isinstance(args.timeout, bool) or not math.isfinite(args.timeout) or args.timeout <= 0
            ):
                raise ValueError("agent start timeout must be a positive finite number of seconds.")
            snapshot = capture_readiness_snapshot(runtime)
            _registry_revision, current_bindings = runtime.load_registry()
            registered_ids = {binding.project_id for binding in current_bindings if binding.enabled}
            if not snapshot["project_ids"] or not registered_ids.intersection(snapshot["project_ids"]):
                result = evaluate_readiness(runtime, snapshot)
                _emit(CliOutput(OutputKind.AGENT_READINESS, result), args.format)
                return 2
            if handler == "agent_run":
                from .agent.lifecycle import run_machine_agent_loop

                run_machine_agent_loop(runtime)
                return 0
            ready = evaluate_readiness(runtime, snapshot)
            if not ready["ready"]:
                ensure_machine_agent_started(runtime)
                ready = wait_for_readiness(runtime, snapshot, timeout_seconds=args.timeout)
            _emit(CliOutput(OutputKind.AGENT_READINESS, ready), args.format)
            return 0 if ready["ready"] else 2
        if handler == "use":
            use_project = args.use_project if args.use_project is not None else getattr(args, "project", None)
            is_selecting = use_project is not None
            if sum((is_selecting, args.show, args.clear)) != 1:
                raise ValueError("use requires exactly one of --project, --show, or --clear.")
            if args.machine is not None or args.runtime_root is not None or args.machine_runtime_root is not None:
                raise ValueError("qexp use accepts only --project, --show, or --clear.")
            if args.clear:
                clear_context()
                return 0
            if args.show:
                context = load_context()
                _emit(
                    CliOutput(
                        OutputKind.CONTEXT,
                        {"shared_root": context["shared_root"] if context else None},
                    ),
                    args.format or "human",
                )
                return 0
            if not use_project:
                raise ValueError("use requires a non-empty --project.")
            save_context(context_commands.normalize_project_path(use_project))
            return 0
        if handler in {"agent_status", "agent_stop", "agent_restart"}:
            runtime = MachineRuntime(args.machine_runtime_root)
            if handler == "agent_status":
                _emit(
                    CliOutput(OutputKind.AGENT_STATUS, {"action": "status", **get_machine_agent_status(runtime)}),
                    args.format,
                )
            elif handler == "agent_stop":
                stopped = stop_machine_agent(runtime)
                _emit(
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {"action": "stopped" if stopped else "already_stopped", **get_machine_agent_status(runtime)},
                    ),
                    args.format,
                )
            else:
                process = restart_machine_agent(runtime)
                _emit(
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {
                            "action": "restarted",
                            **get_machine_agent_status(runtime),
                            "pid": process.pid,
                            "previous_pid": getattr(process, "previous_pid", None),
                        },
                    ),
                    args.format,
                )
            return 0
        if handler.startswith("agent_config_"):
            runtime = MachineRuntime(args.machine_runtime_root)
            if handler.startswith("agent_config_gpus_"):
                runtime.ensure_layout()
                if handler == "agent_config_gpus_show":
                    result = show_gpu_policy(runtime)
                elif handler == "agent_config_gpus_reset":
                    result = reset_gpu_policy(runtime, expected_revision=args.expected_revision)
                else:
                    configured = () if args.none else parse_gpu_id_list(args.visible)
                    result = set_gpu_policy(runtime, configured, expected_revision=args.expected_revision)
                result["machine_runtime_root"] = str(runtime.root)
                _emit(CliOutput(OutputKind.GPU_POLICY, result), args.format)
                return 0
            runtime.ensure_layout()
            policy = (
                set_cpu_lane_capacity(runtime.root, capacity=args.capacity)
                if handler == "agent_config_cpu_set"
                else get_cpu_lane_policy(runtime.root)
            )
            _emit(
                CliOutput(
                    OutputKind.CPU_LANE,
                    {"machine_runtime_root": str(runtime.root), "cpu_lane": policy.to_dict},
                ),
                args.format,
            )
            return 0

        if handler.startswith("admin_upgrade_"):
            runtime = MachineRuntime(args.machine_runtime_root)
            action = handler.removeprefix("admin_upgrade_")
            project = getattr(args, "project", None)
            if action in {"status", "advance"}:
                if project is None:
                    result = (
                        inspect_registered_upgrades(runtime)
                        if action == "status"
                        else advance_registered_upgrades(runtime, force_discovery=True)
                    )
                    kind = OutputKind.UPGRADE_REGISTRY_STATUS if action == "status" else OutputKind.UPGRADE_ADVANCE
                else:
                    selected = _upgrade_project_config(runtime, project)
                    coordinator = UpgradeCoordinator(selected)
                    result = coordinator.status() if action == "status" else coordinator.advance(force_retry=True)
                    kind = OutputKind.UPGRADE_PROJECT
                _emit(CliOutput(kind, result), args.format)
                return 0
            if project is None:
                raise ValueError(f"admin upgrade {action} requires explicit --project PATH.")
            selected = _upgrade_project_config(runtime, project)
            coordinator = UpgradeCoordinator(selected)
            if action == "pause":
                result = coordinator.request_pause(args.reason)
            elif action == "plan":
                result = coordinator.inspect_repair(args.target)
            elif action == "apply":
                result = coordinator.apply_repair(args.repair_id)
            elif action == "validate":
                result = coordinator.validate_repair(args.repair_id)
            else:
                result = coordinator.resume()
            kind = OutputKind.UPGRADE_REPAIR if action in {"plan", "apply", "validate"} else OutputKind.UPGRADE_PROJECT
            _emit(CliOutput(kind, result, {"action": action}), args.format)
            return 0

        if handler.startswith("admin_migrate_"):
            if args.project is None:
                raise ValueError("admin migrate requires explicit --project PATH.")
            selection_path = context_commands.normalize_project_path(args.project)
            machine = args.machine or "upgrade-coordinator"
            cfg = load_root_config(
                selection_path,
                machine,
                getattr(args, "runtime_root", None),
                require_initialized=False,
            )
            if handler == "admin_migrate_schema":
                if args.to_schema != 6:
                    raise ValueError("only --to-schema 6 is supported.")
                migrate_schema5_to_schema6(cfg)
                _emit(
                    CliOutput(OutputKind.SCHEMA6_UPGRADE, {"phase": "completed", "project": str(cfg.project_root)}),
                    args.format,
                )
                return 0
            if handler == "admin_migrate_agent":
                if args.machine is None:
                    raise ValueError("admin migrate agent requires --machine NAME.")
                runtime = MachineRuntime(args.machine_runtime_root)
                binding = migrate_project(runtime, cfg)
                _process, agent_status = ensure_machine_agent_started(runtime)
                _emit(
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {
                            "action": "project_migrated",
                            **binding.to_dict(),
                            **agent_status,
                            "migration_candidates": [],
                        },
                    ),
                    args.format,
                )
                return 0
            action = handler.removeprefix("admin_migrate_schema6_")
            if action == "check":
                result = check_schema6_upgrade(
                    cfg,
                    capabilities=args.capabilities,
                    machine_runtime_root=args.machine_runtime_root,
                )
            elif action == "status":
                result = schema6_upgrade_status(cfg)
            elif action == "start":
                result = start_schema6_upgrade(
                    cfg,
                    capabilities=args.capabilities,
                    machine_runtime_root=args.machine_runtime_root,
                )
            elif action == "resume":
                result = resume_schema6_upgrade(
                    cfg,
                    activation_id=args.activation_id,
                    machine_runtime_root=args.machine_runtime_root,
                )
            else:
                if not args.confirm_clients_stopped or args.machine is None:
                    raise ValueError("attest requires --machine and --confirm-clients-stopped.")
                result = attest_schema6_upgrade(
                    cfg,
                    activation_id=args.activation_id,
                    machine_name=args.machine,
                    machine_runtime_root=args.machine_runtime_root,
                )
            _emit(CliOutput(OutputKind.SCHEMA6_UPGRADE, result), args.format)
            return 0
        if handler in {"admin_check", "admin_repair", "admin_clean"} and getattr(args, "project", None) is None:
            raise ValueError(f"admin {handler.removeprefix('admin_')} requires explicit --project PATH.")
        if handler == "submit":
            _prepare_submission_request(args, raw_argv, invocation_cwd=Path.cwd())
        cfg, execution_context, selection_source = _resolve_cfg(args, require_binding=_requires_verified_binding(args))
        resolved_cfg = cfg
        if handler == "submit":
            args._submission_cfg_resolved = True

        def get_execution_context() -> ExecutionContext:
            return execution_context

        def get_lifecycle_kwargs() -> dict[str, MachineRuntime]:
            return {"machine_runtime": execution_context.machine_runtime}

        if handler.startswith("config_"):
            section = getattr(args, "section", None)
            if handler == "config_show":
                result = configuration_commands.show_config(section, cfg=cfg, runtime=execution_context.machine_runtime)
            elif handler == "config_set":
                if args.enabled and args.disabled:
                    raise ValueError("--enabled and --disabled are mutually exclusive")
                values = {
                    key: value
                    for key, value in {
                        "name": args.name,
                        "agent_mode": args.agent_mode,
                        "enabled": True if args.enabled else False if args.disabled else None,
                        "interval_seconds": args.interval_seconds,
                        "timeout_seconds": args.timeout_seconds,
                        "ttl_seconds": args.ttl_seconds,
                        "renew_interval_seconds": args.renew_interval_seconds,
                        "max_clock_skew_seconds": args.max_clock_skew_seconds,
                        "clock_observation_max_age_seconds": args.clock_observation_max_age_seconds,
                        "clock_provider_margin_seconds": args.clock_provider_margin_seconds,
                        "clock_provider_priority": args.clock_provider_priority,
                        "renewal_commit_margin_seconds": args.renewal_commit_margin_seconds,
                        "webhook_env": args.webhook_env,
                        "credential_source": args.credential_source,
                        "secret_env": None if args.unset_secret_env else args.secret_env,
                        "acknowledge_shared_secret_risk": args.acknowledge_shared_secret_risk or None,
                        "shared_webhook": args.shared_webhook,
                    }.items()
                    if value is not None
                }
                if args.webhook_stdin:
                    if args.credential_source != "shared_file":
                        raise ValueError("--webhook-stdin requires --credential-source shared_file")
                    values["shared_webhook"] = sys.stdin.readline().rstrip("\r\n")
                    if not values["shared_webhook"]:
                        raise ValueError("--webhook-stdin requires a non-empty first input line")
                result = configuration_commands.set_config(
                    section,
                    cfg=cfg,
                    runtime=execution_context.machine_runtime,
                    provider=args.provider,
                    values=values,
                )
            else:
                result = configuration_commands.reset_config(
                    section,
                    cfg=cfg,
                    runtime=execution_context.machine_runtime,
                    provider=args.provider,
                )
            _emit(CliOutput(OutputKind.CONFIG, result), args.format)
            if handler == "config_show" and section is None and result.get("complete") is False:
                return 1
            return 0
        if handler == "submit":

            def print_prepared(operation_id: str, idempotency_key: str) -> None:
                args._submission_prepared_operation_id = operation_id
                args._submission_prepared_key = idempotency_key
                print(
                    f"qexp: prepared operation_id={operation_id} idempotency_key={idempotency_key}",
                    file=sys.stderr,
                    flush=True,
                )

            request = args._submission_request
            values = task_commands.submit_request(cfg, request, on_prepared=print_prepared)
            # Activation is a post-commit follow-up.  It must never relabel a
            # verified commit or erase the IDs from a quiet/JSON response.
            payload = _submission_result_payload(request, values)
            if not request.dry_run and payload["outcome"] == "committed" and not request.no_activate:
                try:
                    ensure_local_agent_active(cfg, reason="submit", **get_lifecycle_kwargs())
                except Exception as activation_error:
                    payload["error"] = {"code": "activation_failed", "message": str(activation_error)}
                    if request.quiet:
                        for task_id in payload["task_ids"]:
                            print(task_id)
                    else:
                        _emit(
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
                            args.format,
                        )
                    return 1
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
                _emit(CliOutput(OutputKind.SUBMISSION, payload, presentation), args.format)
            return 0
        if handler.startswith("task_"):
            if handler.startswith("task_dependencies_"):
                dependency_action = handler.removeprefix("task_dependencies_")
                if handler == "task_dependencies_show":
                    task_value = task_commands.load_task(cfg, args.task_id)
                    _emit(
                        CliOutput(
                            OutputKind.DEPENDENCIES,
                            {"task_id": task_value.task_id, "depends_on_task_ids": task_value.depends_on_task_ids},
                        ),
                        args.format,
                    )
                else:
                    task_value = task_commands.edit_dependencies(
                        cfg, args.task_id, args.depends_on, action=dependency_action
                    )
                    _emit(
                        CliOutput(
                            OutputKind.DEPENDENCIES,
                            {"task_id": task_value.task_id, "depends_on_task_ids": task_value.depends_on_task_ids},
                        ),
                        args.format,
                    )
                return 0
            if handler == "task_cancel":
                context = get_execution_context()
                task_value = task_commands.cancel(cfg, args.task_id, reservation_runtime_root=context.reservation_root)
                claim = task_value.claim_control.get("active_claim") or {}
                is_pending = bool(
                    task_value.state["projection"] == "running"
                    and task_value.control.get("terminate_running")
                    and not task_value.control.get("termination_acknowledged_at")
                )
                _emit(
                    CliOutput(
                        OutputKind.TASK_OPERATION,
                        {
                            "task_id": task_value.task_id,
                            "task_state": task_value.state["projection"],
                            "owning_machine": claim.get("machine_name") or task_value.placement_policy["home_machine"],
                            "operation_state": "waiting_ack" if is_pending else "completed",
                            "pending_acknowledgement": is_pending,
                            "termination_acknowledged_at": task_value.control.get("termination_acknowledged_at"),
                            "follow_up_command": f"qexp task show {shlex.quote(task_value.task_id)} --project {shlex.quote(str(cfg.project_root))}",
                        },
                    ),
                    args.format,
                )
            elif handler == "task_retry":
                if args.quiet and args.format == "json":
                    raise ValueError("--quiet cannot be combined with --format json.")
                ensure_local_agent_active(cfg, reason="task-retry", **get_lifecycle_kwargs())
                task_value = task_commands.retry(cfg, args.task_id)
                if args.quiet or args.format == "human":
                    print(task_value.task_id)
                else:
                    _emit(
                        CliOutput(
                            OutputKind.TASK_OPERATION,
                            {
                                "action": "retry",
                                "task_id": task_value.task_id,
                                "task_state": task_value.state["projection"],
                                "operation_state": "accepted",
                            },
                        ),
                        args.format,
                    )
            elif handler == "task_offer":
                ensure_local_agent_active(cfg, reason="task-offer", **get_lifecycle_kwargs())
                result = task_commands.offer(cfg, args.task_id)
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif handler == "task_share":
                after_seconds = _duration_seconds(args.after) if args.after is not None else None
                ensure_local_agent_active(cfg, reason="task-share", **get_lifecycle_kwargs())
                result = task_commands.share(
                    cfg,
                    args.task_id,
                    after_seconds=after_seconds,
                    helper_machines=_split_machine_list(args.helper_machines),
                )
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif handler == "task_unshare":
                ensure_local_agent_active(cfg, reason="task-unshare", **get_lifecycle_kwargs())
                result = task_commands.keep_local(cfg, args.task_id)
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif handler == "task_list":
                # Exact-name lookup is index-backed even when callers do not opt into
                # explicit pagination.  This keeps the daily lookup path bounded and
                # gives it the same cursor/filter integrity guarantees as a page query.
                is_paginated = args.page_size is not None or args.cursor is not None or args.name is not None
                if is_paginated:
                    if args.limit is not None:
                        raise ObservationError(
                            "invalid_argument",
                            "--limit conflicts with --page-size and --cursor.",
                            2,
                        )
                    page_size = _parse_page_size(args.page_size)
                    page = observer.list_tasks_page(
                        cfg,
                        phase=args.phase,
                        group=args.group,
                        name=args.name,
                        page_size=page_size,
                        cursor=args.cursor,
                    )
                    _emit_task_page(cfg, args, page, page_size)
                else:
                    limit = 50 if args.limit is None else args.limit
                    _emit(
                        CliOutput(
                            OutputKind.TASK_LIST,
                            [
                                item
                                for item in observer.list_tasks(cfg, phase=args.phase, group=args.group, limit=limit)
                                if args.name is None or item.get("name") == args.name
                            ],
                        ),
                        args.format,
                    )
            elif handler == "task_show":
                if args.watch:
                    try:
                        return watch_commands.watch_task(
                            cfg,
                            args.task_id,
                            interval_seconds=args.interval_seconds,
                            follow_retries=args.follow_retries,
                        )
                    except KeyboardInterrupt:
                        _restore_watch_terminal()
                        return 130
                    except BrokenPipeError:
                        return 0
                _emit(CliOutput(OutputKind.TASK_SHOW, observer.inspect_task(cfg, args.task_id)), args.format)
            elif handler == "task_logs":
                if args.follow:
                    try:
                        return log_commands.follow_logs(
                            cfg,
                            args.task_id,
                            tail_lines=args.tail,
                            interval_seconds=args.interval_seconds,
                            follow_retries=args.follow_retries,
                        )
                    except KeyboardInterrupt:
                        return 130
                    except BrokenPipeError:
                        return 0
                print(log_commands.read_logs(cfg, args.task_id, tail_lines=args.tail), end="")
            elif handler == "task_wait":
                timeout = wait_commands.parse_wait_timeout(args.timeout)
                result, wait_exit = wait_commands.wait_for_task(cfg, args.task_id, timeout_seconds=timeout)
                _emit(CliOutput(OutputKind.TASK_WAIT, result), args.format)
                return wait_exit
            return 0
        if handler.startswith("group_"):
            presentation = None
            if handler == "group_create":
                result = group_commands.create_group(cfg, args.name, args.workers)
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": "create"}
            elif handler == "group_list":
                result = observer.list_groups(cfg)
                kind = OutputKind.GROUP_LIST
            elif handler == "group_show":
                result = group_commands.show_group(cfg, args.name)
                kind = OutputKind.GROUP_SHOW
            elif handler == "group_retry":
                ensure_local_agent_active(cfg, reason="group-retry", **get_lifecycle_kwargs())
                result = group_commands.group_retry_failed(cfg, args.name)
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": "retry", "name": args.name, "status": "completed"}
            elif handler.startswith("group_worker_"):
                worker_action = handler.removeprefix("group_worker_")
                if handler == "group_worker_list":
                    result = observer.list_group_machines(
                        cfg,
                        args.group_name,
                        reservation_runtime_root=execution_context.reservation_root,
                    )
                    _emit(CliOutput(OutputKind.GROUP_MACHINES, result), args.format)
                    return 0
                gpu_limit_gpus = getattr(args, "gpu_limit_gpus", None)
                result = group_commands.change_worker(
                    cfg,
                    args.group_name,
                    args.worker_machine,
                    worker_action,
                    terminate_running=getattr(args, "all", False),
                    role=getattr(args, "role", None),
                    gpu_limit_gpus=None if gpu_limit_gpus in {None, "unlimited"} else gpu_limit_gpus,
                    has_gpu_limit=gpu_limit_gpus is not None,
                )
                kind = OutputKind.GROUP_OPERATION
                presentation = {
                    "action": worker_action,
                    "worker_machine": args.worker_machine,
                }
                worker_control = result.get("worker_control") if isinstance(result, dict) else None
                if (
                    handler == "group_worker_remove"
                    and isinstance(worker_control, dict)
                    and isinstance(worker_control.get("discovery"), dict)
                    and worker_control.get("state") in {"preparing", "converging", "waiting_ack", "blocked"}
                ):
                    ensure_local_agent_active(cfg, reason="group-worker-remove", **get_lifecycle_kwargs())
                    presentation.update(
                        {
                            "status": worker_control.get("state"),
                            "reason": worker_control.get("blocked_reason"),
                        }
                    )
                worker_control = result.get("worker_control") if isinstance(result, dict) else None
                worker_operation_id = worker_control.get("operation_id") if isinstance(worker_control, dict) else None
                if handler == "group_worker_remove" and worker_operation_id:
                    reference = operation_commands.create_operation_reference(
                        cfg, "worker_remove", worker_operation_id, worker_operation_id
                    )
                    result["operation_reference"] = reference
                    result["follow_up_command"] = (
                        f"qexp admin operation show {reference} --project {shlex.quote(str(cfg.project_root))}"
                    )
            else:
                group_action = handler.removeprefix("group_")
                context = get_execution_context()
                if handler == "group_resume":
                    ensure_local_agent_active(cfg, reason="group-resume", **get_lifecycle_kwargs())
                result = group_commands.group_control(
                    cfg,
                    args.name,
                    group_action,
                    terminate_running=getattr(args, "all", False),
                    reservation_runtime_root=context.reservation_root,
                )
                if handler == "group_cancel":
                    control = result.get("cancellation_operation", {})
                    if (
                        isinstance(control, dict)
                        and isinstance(control.get("discovery"), dict)
                        and control.get("state") in {"preparing", "converging", "waiting_ack", "blocked"}
                    ):
                        ensure_local_agent_active(cfg, reason="group-cancel", **get_lifecycle_kwargs())
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": group_action}
                if handler == "group_cancel":
                    control = result.get("cancellation_operation", {})
                    pending_machines = control.get("pending_machine_acknowledgements", {})
                    presentation.update(
                        {
                            "status": control.get("state"),
                            "pending_machines": list(pending_machines.keys())
                            if isinstance(pending_machines, dict)
                            else [],
                            "reason": control.get("blocked_reason"),
                        }
                    )
                    operation_id = control.get("operation_id") if isinstance(control, dict) else None
                    if operation_id:
                        reference = operation_commands.create_operation_reference(
                            cfg, "group_cancel", operation_id, operation_id
                        )
                        result["operation_reference"] = reference
                        result["follow_up_command"] = (
                            f"qexp admin operation show {reference} --project {shlex.quote(str(cfg.project_root))}"
                        )
            _emit(
                CliOutput(kind, result, presentation or {}),
                args.format,
            )
            return 0
        if handler == "status":
            result = status_commands.project_status(
                cfg,
                selection_source=selection_source,
                machine_runtime=execution_context.machine_runtime,
            )
            _emit(CliOutput(OutputKind.STATUS, result), args.format)
            return 0
        if handler in {"machine_list", "machine_show"}:
            if handler == "machine_list":
                result = observer.list_machines(cfg)
                _emit(CliOutput(OutputKind.MACHINES, result), args.format)
            else:
                result = status_commands.machine_detail(cfg, args.name)
                _emit(CliOutput(OutputKind.MACHINE_SHOW, result), args.format)
            return 0
        if handler in {"admin_check", "admin_repair"}:
            context = get_execution_context()
            result = (
                verify_integrity(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                    project_id=context.project_id,
                    max_work_items=args.max_work_items,
                )
                if handler == "admin_check"
                else repair_metadata(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                    max_work_items=args.max_work_items,
                )
            )
            output_kind = OutputKind.DOCTOR_VERIFY if handler == "admin_check" else OutputKind.DOCTOR_REPAIR
            _emit(CliOutput(output_kind, result), args.format)
            return resolve_verify_exit_code(result, strict=args.strict)
        if handler == "admin_operation_show":
            if args.project is None:
                raise ValueError("admin operation show requires explicit --project PATH.")
            result, operation_exit = operation_commands.inspect_operation(cfg, args.reference)
            _emit(CliOutput(OutputKind.OPERATION, result), args.format)
            return operation_exit
        if handler == "admin_clean":
            context = get_execution_context()
            result = cleanup.clean(
                context.local_cfg,
                task_id=args.task_id,
                group=args.group,
                older_than_days=args.older_than_days,
                limit=args.limit,
                dry_run=args.dry_run,
                max_work_items=args.max_work_items,
                reservation_runtime_root=context.reservation_root,
            )
            operations = result.get("operations")
            if isinstance(operations, dict):
                for task_id, operation in operations.items():
                    if not isinstance(operation, dict):
                        continue
                    operation_id = operation.get("operation_id")
                    if isinstance(operation_id, str):
                        reference = operation_commands.create_operation_reference(
                            cfg, "cleanup", str(task_id), operation_id
                        )
                        operation["operation_reference"] = reference
                        operation["follow_up_command"] = (
                            f"qexp admin operation show {reference} --project {shlex.quote(str(cfg.project_root))}"
                        )
            _emit(CliOutput(OutputKind.CLEAN, result), args.format)
            return 0
    except ObservationError as exc:
        if handler == "task_wait" and getattr(args, "format", "human") == "json":
            return _emit_wait_error(
                task_id=getattr(args, "task_id", None),
                project=resolved_cfg.shared_root if resolved_cfg is not None else None,
                outcome="observation_failed",
                reason="observation_failed",
                code=exc.code,
                message=exc.message,
                exit_code=6,
            )
        return _emit_observation_error(exc, getattr(args, "format", "human"))
    except KeyboardInterrupt as exc:
        if handler == "submit":
            payload, _exit_code = _submission_error_payload(args, exc)
            payload["error"] = {"code": "interrupted", "message": "submission interrupted; retry with the same key."}
            if getattr(args, "quiet", False):
                for task_id in payload["task_ids"]:
                    print(task_id)
                print("qexp: submission interrupted; retry with the same key.", file=sys.stderr)
            elif getattr(args, "format", "human") == "json":
                print(json.dumps(payload))
                print(f"qexp: {payload['error']['message']}", file=sys.stderr)
            else:
                print(f"qexp: {payload['error']['message']}", file=sys.stderr)
            return 130
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        if handler == "submit":
            payload, exit_code = _submission_error_payload(args, exc)
            if getattr(args, "quiet", False):
                print(f"qexp: {exc}", file=sys.stderr)
            elif getattr(args, "format", "human") == "json":
                print(json.dumps(payload))
                print(f"qexp: {exc}", file=sys.stderr)
            else:
                print(f"qexp: {exc}", file=sys.stderr)
            return exit_code
        if handler == "task_wait" and getattr(args, "format", "human") == "json":
            invalid_timeout = isinstance(exc, ValueError) and "timeout" in str(exc).lower()
            return _emit_wait_error(
                task_id=getattr(args, "task_id", None),
                project=resolved_cfg.shared_root if resolved_cfg is not None else None,
                outcome="invalid_input" if invalid_timeout else "observation_failed",
                reason="invalid_input" if invalid_timeout else "project_context",
                code="invalid_input" if invalid_timeout else "project_context",
                message=str(exc),
                exit_code=2 if invalid_timeout else 6,
            )
        if handler == "task_list" and (args.page_size is not None or args.cursor is not None):
            code = "invalid_argument" if isinstance(exc, ValueError) else "index_unavailable"
            return _emit_observation_error(ObservationError(code, str(exc)), args.format)
        if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError) and not _is_continuous_task(args):
            if getattr(args, "format", "human") != "json":
                raise
        if getattr(args, "format", "human") == "json" and not _is_continuous_task(args):
            operational = isinstance(exc, (RuntimeError, OSError))
            return _emit_observation_error(
                ObservationError(
                    "operational_failure" if operational else "invalid_argument",
                    str(exc),
                    1 if operational else 2,
                ),
                "json",
            )
        print(f"qexp: {exc}", file=sys.stderr)
        return 2
    finally:
        _ACTIVE_COMMAND_SPEC.reset(command_spec_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
