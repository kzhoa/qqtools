"""qexp command line routing for the schema-6 product contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
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
from .commands import group as group_commands
from .commands import logs as log_commands
from .commands import task as task_commands
from .commands import watch as watch_commands
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


_PAGINATION_JSON_PARSE_MODE = False
_SUBMISSION_JSON_PARSE_MODE = False


class _QexpArgumentParser(argparse.ArgumentParser):
    """Use structured parse errors only for the narrowly scoped page mode."""

    def error(self, message: str) -> None:
        if _PAGINATION_JSON_PARSE_MODE:
            raise _PaginationParseError(message)
        if _SUBMISSION_JSON_PARSE_MODE:
            raise _SubmissionParseError(message)
        super().error(message)


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


def _emit_task_page(cfg: RootConfig, args: argparse.Namespace, page: dict[str, object], page_size: int) -> None:
    """Render paginated task results and a shell-safe continuation command."""
    presentation: dict[str, object] = {}
    next_cursor = page.get("next_cursor")
    if args.format == "human" and next_cursor is not None:
        command = [
            "qexp",
            "--shared-root",
            str(cfg.shared_root),
            "--machine",
            cfg.machine_name,
            "task",
            "list",
        ]
        if args.phase:
            command.extend(("--phase", args.phase))
        if args.group:
            command.extend(("--group", args.group))
        command.extend(("--page-size", str(page_size), "--cursor", str(next_cursor)))
        presentation["continuation_command"] = shlex.join(command)
    _emit(CliOutput(OutputKind.TASK_PAGE, page, presentation), args.format)


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("human", "json"), default="human")


def _emit(output: CliOutput[object], output_format: str, *, flush: bool = False) -> None:
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


def _shared_root_input(args: argparse.Namespace) -> tuple[str | None, dict | None]:
    """Resolve only the shared-root locator; saved identity fields are intentionally ignored."""
    submission_project = getattr(args, "_submission_project", None)
    if args.command == "submit" and submission_project is not None:
        return str(submission_project), None
    flag_value = getattr(args, "shared_root", None)
    environment_value = os.environ.get("QEXP_SHARED_ROOT")
    context = load_context() if flag_value is None and environment_value is None else None
    shared = flag_value or environment_value or (context or {}).get("shared_root")
    return shared, context


def _requires_verified_binding(args: argparse.Namespace) -> bool:
    """Classify commands that can create or change project-owned state."""
    if args.command in {"submit", "clean"}:
        return True
    if args.command == "config":
        return (
            getattr(args, "notifications_action", None) == "set"
            or getattr(args, "provider_action", None) == "set"
            or getattr(args, "progress_action", None) == "set"
            or getattr(args, "launch_handoff_action", None) == "set"
            or getattr(args, "tmux_action", None) == "set"
        )
    if args.command == "lease-policy":
        return args.lease_policy_action == "set"
    if args.command == "task":
        return not (
            args.task_action in {"list", "show", "logs"}
            or (args.task_action == "dependencies" and args.dependencies_action == "show")
        )
    if args.command == "group":
        return args.group_action not in {"list", "show"}
    if args.command == "agent":
        return args.agent_action in {"start", "run"}
    return args.command == "doctor" and args.action == "repair"


def _resolve_cfg(args: argparse.Namespace, *, require_binding: bool) -> tuple[object, ExecutionContext]:
    shared, _saved_context = _shared_root_input(args)
    if not shared:
        raise ValueError("--shared-root is required or must be saved with qexp use.")
    assertion = _machine_assertion(args)
    machine_runtime = MachineRuntime(getattr(args, "machine_runtime_root", None))

    if require_binding:
        try:
            execution_context = machine_runtime.verified_execution_context(shared)
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
        return execution_context.cfg, execution_context

    # Read-only project commands should use the binding-owned local runtime when one is
    # available, while remaining usable for observation before local registration.
    if not (args.command == "agent" and args.agent_action in {"add-project", "migrate-project"}):
        try:
            execution_context = machine_runtime.verified_execution_context(shared)
        except (ValueError, RuntimeError):
            execution_context = None
        if execution_context is not None:
            verified_machine = execution_context.cfg.machine_name
            if assertion is not None and assertion != verified_machine:
                raise ValueError(
                    f"Local project binding is {verified_machine!r}, but --machine asserted {assertion!r}."
                )
            return execution_context.cfg, execution_context

    # Read-only project observation must remain possible without a local binding. The sentinel is
    # never used as an authority source because this branch is not used for mutations.
    if args.command == "agent" and args.agent_action in {"add-project", "migrate-project"}:
        if assertion is None:
            raise ValueError(
                "agent add-project and migrate-project require --machine or QEXP_MACHINE "
                "when no local project binding exists."
            )
    machine = assertion or "unbound"
    runtime = None
    if args.command == "agent" and args.agent_action in {"add-project", "migrate-project"}:
        runtime = getattr(args, "runtime_root", None) or os.environ.get("QEXP_RUNTIME_ROOT")
    cfg = load_root_config(shared, machine, runtime, require_initialized=True)
    return cfg, ExecutionContext(cfg, machine_runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = _QexpArgumentParser(
        description=(
            "qexp schema-6 experiment queue; --machine is local identity, --home-machine is Task "
            "placement, and Attempt machine is selected later by claim. qexp does not remotely "
            "start a target agent."
        ),
        epilog=(
            "Machine setup is: qexp init --machine NAME. Shared Project creation is: "
            "qexp project init PATH; local enrollment is: qexp project register PATH. "
            "qexp use only saves local CLI context."
        ),
    )
    parser.add_argument("--shared-root", help="Locate the shared project control root.")
    parser.add_argument(
        "--machine",
        help="Assert the local logical machine identity; this is not the Task target machine.",
    )
    parser.add_argument("--runtime-root")
    parser.add_argument("--machine-runtime-root")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init",
        help="Initialize or replace this machine runtime.",
        description=(
            "Initialize or replace the local machine identity and global agent policy. "
            "Project creation and enrollment are separate: use 'qexp project init' and "
            "'qexp project register'."
        ),
    )
    init.add_argument("--shared-root", dest="init_shared_root")
    init.add_argument("--machine", dest="init_machine")
    init.add_argument("--agent-mode", choices=["on_demand", "daemon"])
    init.add_argument("--detach-old-runtime", action="store_true")
    init.add_argument("--yes", action="store_true")
    _add_output_format(init)
    init.add_argument("--cpu-lane-capacity", type=int)

    project = commands.add_parser(
        "project",
        help="Create shared Projects and manage local machine enrollment.",
        description="Project init creates shared truth; project register/list/enable/disable/remove manage local enrollment.",
    )
    project_sub = project.add_subparsers(dest="project_action", required=True)
    project_init = project_sub.add_parser("init", help="Create shared Project truth without enrollment.")
    _add_output_format(project_init)
    project_init.add_argument("path", nargs="?")
    project_register = project_sub.add_parser("register", help="Enroll explicit Projects or saved inventory entries.")
    _add_output_format(project_register)
    project_register.add_argument("paths", nargs="*")
    project_register.add_argument("--from-pool", action="store_true")
    project_register.add_argument("--machine", dest="project_machine")
    project_register.add_argument("--name-source", choices=("default", "explicit"))
    for project_action in ("list",):
        project_list_parser = project_sub.add_parser(project_action)
        _add_output_format(project_list_parser)
    for project_action in ("enable", "disable", "remove"):
        project_selector = project_sub.add_parser(project_action)
        _add_output_format(project_selector)
        project_selector.add_argument("selector")
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--to-schema", type=int, required=True)
    upgrade = commands.add_parser("upgrade")
    upgrade_sub = upgrade.add_subparsers(dest="upgrade_feature", required=True)
    coordinator_upgrade = upgrade_sub.add_parser(
        "coordinator",
        help="Run the exceptional machine-level coordinator over registered projects.",
    )
    _add_output_format(coordinator_upgrade)
    coordinator_upgrade.add_argument("--project", dest="upgrade_project")
    schema6_upgrade = upgrade_sub.add_parser("schema6")
    schema6_upgrade_sub = schema6_upgrade.add_subparsers(dest="schema6_upgrade_action", required=True)
    for name in ("check", "start", "status"):
        action = schema6_upgrade_sub.add_parser(name)
        _add_output_format(action)
        if name != "status":
            action.add_argument(
                "--capability",
                action="append",
                dest="capabilities",
                choices=("cpu-lane-v1", "task-dependencies-v1"),
            )
    schema6_attest = schema6_upgrade_sub.add_parser("attest")
    _add_output_format(schema6_attest)
    schema6_attest.add_argument("--activation-id", required=True)
    schema6_attest.add_argument("--confirm-clients-stopped", action="store_true")
    schema6_resume = schema6_upgrade_sub.add_parser("resume")
    _add_output_format(schema6_resume)
    schema6_resume.add_argument("--activation-id", required=True)
    lease_policy = commands.add_parser("lease-policy", help=argparse.SUPPRESS)
    lease_policy_sub = lease_policy.add_subparsers(dest="lease_policy_action", required=True)
    lease_show = lease_policy_sub.add_parser("show")
    _add_output_format(lease_show)
    policy_set = lease_policy_sub.add_parser("set")
    _add_output_format(policy_set)
    policy_set.add_argument("--ttl-seconds", type=int)
    policy_set.add_argument("--renew-interval-seconds", type=float)
    policy_set.add_argument("--max-clock-skew-seconds", type=float)
    policy_set.add_argument("--clock-observation-max-age-seconds", type=float)
    policy_set.add_argument("--clock-provider-margin-seconds", type=float)
    policy_set.add_argument("--clock-provider-priority")
    policy_set.add_argument("--renewal-commit-margin-seconds", type=float)
    config = commands.add_parser("config")
    config_sub = config.add_subparsers(dest="config_action", required=True)
    config_lease = config_sub.add_parser("lease", help="Configure lease policy.")
    config_lease_sub = config_lease.add_subparsers(dest="lease_policy_action", required=True)
    for _name in ("show", "set"):
        _lp = config_lease_sub.add_parser(_name)
        _add_output_format(_lp)
        if _name == "set":
            for _arg, _typ in (
                ("ttl-seconds", int),
                ("renew-interval-seconds", float),
                ("max-clock-skew-seconds", float),
                ("clock-observation-max-age-seconds", float),
                ("clock-provider-margin-seconds", float),
                ("renewal-commit-margin-seconds", float),
            ):
                _lp.add_argument("--" + _arg, type=_typ)
            _lp.add_argument("--clock-provider-priority")
    config_agent = config_sub.add_parser("agent", help="Configure the machine-global agent.")
    config_agent_sub = config_agent.add_subparsers(dest="agent_config_action", required=True)
    config_agent_show = config_agent_sub.add_parser("show")
    _add_output_format(config_agent_show)
    config_agent_set = config_agent_sub.add_parser("set")
    _add_output_format(config_agent_set)
    config_agent_set.add_argument("--name")
    config_agent_set.add_argument("--agent-mode", choices=("daemon", "on_demand"))
    # Machine-global configuration also has the public verb-first spelling.
    config_show = config_sub.add_parser("show")
    _add_output_format(config_show)
    config_show.add_argument("config_target", choices=("agent",))
    config_set = config_sub.add_parser("set")
    _add_output_format(config_set)
    config_set.add_argument("config_target", choices=("agent",))
    config_set.add_argument("--name")
    config_set.add_argument("--agent-mode", choices=("daemon", "on_demand"))
    notifications = config_sub.add_parser("notifications")
    notifications_sub = notifications.add_subparsers(dest="notifications_action", required=True)
    notifications_show = notifications_sub.add_parser("show")
    _add_output_format(notifications_show)
    notifications_set = notifications_sub.add_parser("set")
    _add_output_format(notifications_set)
    notifications_set.add_argument("--enabled", action="store_true")
    notifications_set.add_argument("--disabled", action="store_true")
    progress_config = config_sub.add_parser("progress", help="Configure advisory progress reporting frequency.")
    progress_config_sub = progress_config.add_subparsers(dest="progress_action", required=True)
    progress_show = progress_config_sub.add_parser("show")
    _add_output_format(progress_show)
    progress_set = progress_config_sub.add_parser("set")
    _add_output_format(progress_set)
    progress_set.add_argument("--interval-seconds", required=True)
    launch_handoff_config = config_sub.add_parser(
        "launch-handoff",
        help="Configure the runner launch-handoff timeout.",
    )
    launch_handoff_sub = launch_handoff_config.add_subparsers(dest="launch_handoff_action", required=True)
    launch_handoff_show = launch_handoff_sub.add_parser("show")
    _add_output_format(launch_handoff_show)
    launch_handoff_set = launch_handoff_sub.add_parser("set")
    _add_output_format(launch_handoff_set)
    launch_handoff_set.add_argument("--timeout-seconds", required=True)
    tmux_config = config_sub.add_parser("tmux", help="Configure project tmux log observer windows.")
    tmux_config_sub = tmux_config.add_subparsers(dest="tmux_action", required=True)
    tmux_show = tmux_config_sub.add_parser("show")
    _add_output_format(tmux_show)
    tmux_set = tmux_config_sub.add_parser("set")
    _add_output_format(tmux_set)
    tmux_values = tmux_set.add_mutually_exclusive_group(required=True)
    tmux_values.add_argument("--enabled", action="store_true")
    tmux_values.add_argument("--disabled", action="store_true")
    provider = notifications_sub.add_parser("provider")
    provider_sub = provider.add_subparsers(dest="provider_action", required=True)
    provider_set = provider_sub.add_parser("set")
    _add_output_format(provider_set)
    provider_set.add_argument("provider")
    provider_set.add_argument("--enabled", action="store_true")
    provider_set.add_argument("--disabled", action="store_true")
    provider_set.add_argument("--webhook-env")
    provider_set.add_argument("--credential-source", choices=["env", "shared_file"])
    provider_set.add_argument("--webhook-stdin", action="store_true")
    provider_set.add_argument("--acknowledge-shared-secret-risk", action="store_true")
    provider_set.add_argument("--secret-env")
    provider_set.add_argument("--unset-secret-env", action="store_true")
    provider_set.add_argument("--timeout-seconds", type=float)
    submit = commands.add_parser(
        "submit",
        description=(
            "Submit one command or a manifest using the verified local identity. "
            "qexp does not remotely start the target agent."
        ),
    )
    _add_output_format(submit)
    submit.add_argument("--project", dest="project")
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
    task = commands.add_parser("task")
    task_sub = task.add_subparsers(dest="task_action", required=True)
    cancel = task_sub.add_parser("cancel")
    _add_output_format(cancel)
    cancel.add_argument("task_id")
    retry = task_sub.add_parser("retry")
    retry.add_argument("task_id")
    retry.add_argument(
        "--acknowledge-duplicate-risk",
        action="store_true",
        help="Deprecated compatibility option; retained as a no-op.",
    )
    offer = task_sub.add_parser("offer")
    _add_output_format(offer)
    offer.add_argument("task_id")
    share = task_sub.add_parser("share")
    _add_output_format(share)
    share.add_argument("task_id")
    share.add_argument("--after")
    share.add_argument(
        "--with",
        dest="helper_machines",
        action="append",
        help="Helper machine; repeat the option or use comma-separated machine names.",
    )
    keep_local = task_sub.add_parser("keep-local")
    _add_output_format(keep_local)
    keep_local.add_argument("task_id")
    listing = task_sub.add_parser("list")
    _add_output_format(listing)
    listing.add_argument("--phase")
    listing.add_argument("--group")
    listing.add_argument("--limit", type=int, default=None)
    listing.add_argument("--page-size", default=None)
    listing.add_argument("--cursor", default=None)
    show = task_sub.add_parser("show")
    _add_output_format(show)
    show.add_argument("task_id")
    show.add_argument(
        "--watch",
        action="store_true",
        help="Refresh a compact human Task view; interval controls reads and screen refreshes.",
    )
    show.add_argument(
        "--interval-seconds",
        default=None,
        help="Watch refresh/read cadence in seconds (default: 2); it does not change progress reporting.",
    )
    show.add_argument(
        "--follow-retries",
        action="store_true",
        help="Keep watching after a terminal result until a later retry appears.",
    )
    logs = task_sub.add_parser("logs")
    logs.add_argument("task_id")
    logs.add_argument(
        "--follow",
        action="store_true",
        help="Follow application bytes; --tail applies on each Attempt and file generation.",
    )
    logs.add_argument(
        "--tail",
        default=None,
        help="Initial lines per Attempt or replacement generation (default: 100; zero means new bytes only).",
    )
    logs.add_argument(
        "--interval-seconds",
        default=None,
        help="Follow polling/read cadence in seconds (default: 2).",
    )
    logs.add_argument(
        "--follow-retries",
        action="store_true",
        help="Keep following at terminal EOF until a later retry appears.",
    )
    dependencies = task_sub.add_parser("dependencies")
    dependencies_sub = dependencies.add_subparsers(dest="dependencies_action", required=True)
    for name in ("show", "replace", "add", "remove"):
        action = dependencies_sub.add_parser(name)
        _add_output_format(action)
        action.add_argument("task_id")
        if name != "show":
            action.add_argument("--depends-on", action="append", default=[])
    group = commands.add_parser("group")
    group_sub = group.add_subparsers(dest="group_action", required=True)
    create = group_sub.add_parser("create")
    _add_output_format(create)
    create.add_argument("name")
    create.add_argument("--workers", nargs="*", default=None)
    group_list = group_sub.add_parser("list")
    _add_output_format(group_list)
    show_group = group_sub.add_parser("show")
    _add_output_format(show_group)
    show_group.add_argument("name")
    for name in ("seal", "reopen", "pause", "resume", "cancel", "retry-failed"):
        action = group_sub.add_parser(name)
        _add_output_format(action)
        action.add_argument("name")
        if name == "cancel":
            action.add_argument("--terminate-running", action="store_true")
    machines = group_sub.add_parser("machines")
    machines_sub = machines.add_subparsers(dest="machine_action", required=True)
    for name in ("add", "set", "drain", "remove", "list"):
        action = machines_sub.add_parser(name)
        _add_output_format(action)
        action.add_argument("group_name")
        if name != "list":
            action.add_argument("worker_machine")
        if name in {"add", "set"}:
            action.add_argument("--role", choices=("primary", "borrow"))
            gpu_limit = action.add_mutually_exclusive_group()
            gpu_limit.add_argument("--gpu-limit-gpus", type=_gpu_limit_gpus)
        if name == "remove":
            action.add_argument("--terminate-running", action="store_true")
    agent = commands.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_action", required=True)
    for name in (
        "start",
        "run",
        "restart",
        "status",
        "stop",
        "add-project",
        "list-projects",
        "enable-project",
        "disable-project",
        "remove-project",
        "migrate-project",
    ):
        action = agent_sub.add_parser(name)
        _add_output_format(action)
    agent_sub.choices["start"].add_argument("--timeout", type=float, default=30.0)
    agent_sub.choices["name"] = agent_sub.add_parser("name", help="Show or change the machine-global agent name.")
    _add_output_format(agent_sub.choices["name"])
    agent_sub.choices["name"].add_argument("--set-to", dest="set_to")
    cpu_lane = agent_sub.add_parser("cpu-lane")
    cpu_lane_sub = cpu_lane.add_subparsers(dest="cpu_lane_action", required=True)
    cpu_lane_show = cpu_lane_sub.add_parser("show")
    _add_output_format(cpu_lane_show)
    cpu_lane_set = cpu_lane_sub.add_parser("set")
    _add_output_format(cpu_lane_set)
    cpu_lane_set.add_argument("--capacity", type=int, required=True)
    gpus = agent_sub.add_parser("gpus", help="Inspect or change the machine GPU admission policy.")
    gpus_sub = gpus.add_subparsers(dest="gpu_action", required=True)
    gpu_show = gpus_sub.add_parser("show")
    _add_output_format(gpu_show)
    gpu_set = gpus_sub.add_parser("set")
    _add_output_format(gpu_set)
    gpu_set_values = gpu_set.add_mutually_exclusive_group(required=True)
    gpu_set_values.add_argument("--visible")
    gpu_set_values.add_argument("--none", action="store_true")
    gpu_set.add_argument("--expected-revision", type=int)
    gpu_reset = gpus_sub.add_parser("reset")
    _add_output_format(gpu_reset)
    gpu_reset.add_argument("--expected-revision", type=int)
    agent_sub.choices["add-project"].add_argument(
        "--adopt-existing",
        action="store_true",
        help="Explicitly reuse a logically owned name after its old write eligibility has expired.",
    )
    for name in ("disable-project", "enable-project", "remove-project"):
        agent_sub.choices[name].add_argument("project")
    agent_upgrade = agent_sub.add_parser(
        "upgrade",
        help="Inspect and advance registered-project rolling upgrades.",
    )
    agent_upgrade_sub = agent_upgrade.add_subparsers(dest="agent_upgrade_action", required=True)
    for name in ("status", "retry", "coordinate"):
        action = agent_upgrade_sub.add_parser(name)
        _add_output_format(action)
        action.add_argument("--project", dest="upgrade_project")
    pause = agent_upgrade_sub.add_parser("pause")
    _add_output_format(pause)
    pause.add_argument("--project", dest="upgrade_project", required=True)
    pause.add_argument("--reason", required=True)
    inspect = agent_upgrade_sub.add_parser("inspect")
    _add_output_format(inspect)
    inspect.add_argument("--project", dest="upgrade_project", required=True)
    plan = agent_upgrade_sub.add_parser("plan")
    _add_output_format(plan)
    plan.add_argument("--project", dest="upgrade_project", required=True)
    plan.add_argument("--target", required=True)
    for name in ("apply", "validate"):
        action = agent_upgrade_sub.add_parser(name)
        _add_output_format(action)
        action.add_argument("--project", dest="upgrade_project", required=True)
        action.add_argument("--repair-id", required=True)
    resume = agent_upgrade_sub.add_parser("resume")
    _add_output_format(resume)
    resume.add_argument("--project", dest="upgrade_project", required=True)
    top = commands.add_parser("top")
    _add_output_format(top)
    machine_list = commands.add_parser("machines")
    _add_output_format(machine_list)
    doctor = commands.add_parser("doctor")
    _add_output_format(doctor)
    doctor.add_argument("action", choices=["verify", "repair"], default="verify", nargs="?")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--max-work-items", type=int, default=64)
    clean = commands.add_parser(
        "clean",
        help="Remove terminal qexp metadata while preserving experiment work directories.",
        description="Remove terminal qexp metadata while preserving experiment work directories.",
    )
    _add_output_format(clean)
    clean_scope = clean.add_mutually_exclusive_group()
    clean_scope.add_argument("--task-id", help="Clean one terminal task, regardless of its age.")
    clean_scope.add_argument("--group", help="Clean terminal tasks in one group subject to retention and limit.")
    clean.add_argument(
        "--older-than-days", type=int, default=30, help="Minimum task age for bulk cleanup (default: 30)."
    )
    clean.add_argument(
        "--limit", type=int, default=100, help="Maximum number of bulk-cleanup candidates (default: 100)."
    )
    clean.add_argument(
        "--max-work-items", type=int, default=64, help="Maximum group-member archive entries per cleanup slice."
    )
    clean.add_argument("--dry-run", action="store_true", help="Show candidates without cleaning them.")
    use = commands.add_parser(
        "use",
        help="Save local CLI context without initializing or registering a project.",
        description=(
            "Save the local default shared root without validating it. "
            "This command does not initialize a shared root, create a Project machine record, "
            "or register the project with the local machine agent."
        ),
    )
    use.add_argument("--shared-root", dest="use_shared_root", help="Shared project control root to save.")
    use.add_argument("--show", action="store_true")
    use.add_argument("--clear", action="store_true")
    use.add_argument("--format", choices=("human", "json"))
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
    if args.command != "submit":
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
    return args.command == "task" and (
        (args.task_action == "show" and bool(getattr(args, "watch", False)))
        or (args.task_action == "logs" and bool(getattr(args, "follow", False)))
    )


def _validate_continuous_options(args: argparse.Namespace) -> None:
    """Validate continuous-only flags before resolving a project or reading Task truth."""
    if args.command != "task":
        return
    if args.task_action == "show":
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
    if args.task_action != "logs":
        return
    follow = bool(getattr(args, "follow", False))
    tail = getattr(args, "tail", None)
    interval = getattr(args, "interval_seconds", None)
    follow_retries = bool(getattr(args, "follow_retries", False))
    if not follow:
        if tail is not None:
            raise ValueError("--tail requires --follow.")
        if interval is not None:
            raise ValueError("--interval-seconds requires --follow.")
        if follow_retries:
            raise ValueError("--follow-retries requires --follow.")
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
    global _PAGINATION_JSON_PARSE_MODE, _SUBMISSION_JSON_PARSE_MODE
    _PAGINATION_JSON_PARSE_MODE = _is_paginated_json_argv(raw_argv)
    _SUBMISSION_JSON_PARSE_MODE = _is_submission_json_argv(raw_argv)
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
    finally:
        _PAGINATION_JSON_PARSE_MODE = False
        _SUBMISSION_JSON_PARSE_MODE = False
    try:
        _validate_continuous_options(args)
        if args.command == "init":
            # QQTOOLS-COMPAT-0014: the former project-bound init spelling is
            # retained only as a bounded diagnostic during the transition.
            if args.init_shared_root or args.shared_root or args.runtime_root or args.cpu_lane_capacity is not None:
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
        if args.command == "project":
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.project_action == "init":
                result = initialize_project(args.path)
                _emit(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
                return 0
            if args.project_action == "register":
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
            if args.project_action == "list":
                _emit(CliOutput(OutputKind.PROJECT_LIST, list_projects(runtime)), args.format)
                return 0
            if args.project_action == "remove":
                result = remove_project(runtime, args.selector)
            else:
                result = set_project_enablement(runtime, args.selector, args.project_action == "enable")
            _emit(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
            return 0
        if args.command == "config" and (
            args.config_action == "agent" or (args.config_action in {"show", "set"} and args.config_target == "agent")
        ):
            runtime = MachineRuntime(args.machine_runtime_root)
            config_action = args.agent_config_action if args.config_action == "agent" else args.config_action
            if config_action == "show":
                result = get_agent_config(runtime)
            else:
                if args.name is None and args.agent_mode is None:
                    raise ValueError("config set agent requires --name or --agent-mode.")
                set_agent_config(runtime, name=args.name, agent_mode=args.agent_mode)
                result = get_agent_config(runtime)
            _emit(CliOutput(OutputKind.AGENT_CONFIG, result), args.format)
            return 0
        if args.command == "agent" and args.agent_action == "name":
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.set_to is not None:
                set_agent_config(runtime, name=args.set_to)
            result = agent_config_payload(runtime)
            _emit(CliOutput(OutputKind.AGENT_CONFIG, result), args.format)
            return 0
        if args.command == "agent" and args.agent_action in {
            "add-project",
            "list-projects",
            "enable-project",
            "disable-project",
            "remove-project",
        }:
            # QQTOOLS-COMPAT-0014: retired executing routes fail before any
            # runtime/configuration resolution or state creation.
            replacement = {
                "add-project": "qexp project register PATH",
                "list-projects": "qexp project list",
                "enable-project": "qexp project enable SELECTOR",
                "disable-project": "qexp project disable SELECTOR",
                "remove-project": "qexp project remove SELECTOR",
            }[args.agent_action]
            print(
                f"qexp: QQTOOLS-COMPAT-0014: retired command; use '{replacement}'.",
                file=sys.stderr,
            )
            return 2
        if args.command == "agent" and args.agent_action in {"start", "run"}:
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.agent_action == "start" and (
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
            if args.agent_action == "run":
                from .agent.lifecycle import run_machine_agent_loop

                run_machine_agent_loop(runtime)
                return 0
            ready = evaluate_readiness(runtime, snapshot)
            if not ready["ready"]:
                ensure_machine_agent_started(runtime)
                ready = wait_for_readiness(runtime, snapshot, timeout_seconds=args.timeout)
            _emit(CliOutput(OutputKind.AGENT_READINESS, ready), args.format)
            return 0 if ready["ready"] else 2
        if args.command == "migrate":
            if not args.shared_root or not args.machine:
                raise ValueError("migrate requires --shared-root and --machine.")
            if args.to_schema != 6:
                raise ValueError("only --to-schema 6 is supported.")
            root = Path(args.shared_root).expanduser().resolve()
            runtime = Path(args.runtime_root) if args.runtime_root else root.parent / ".qexp-runtime" / args.machine
            migrate_schema5_to_schema6(RootConfig(root, root.parent, args.machine, runtime))
            print(root)
            return 0
        if args.command == "upgrade":
            if args.upgrade_feature == "coordinator":
                runtime = MachineRuntime(args.machine_runtime_root)
                project = getattr(args, "upgrade_project", None)
                if project is None:
                    result = advance_registered_upgrades(runtime, force_discovery=True)
                else:
                    result = UpgradeCoordinator(_upgrade_project_config(runtime, project)).advance(force_retry=True)
                _emit(
                    CliOutput(
                        OutputKind.UPGRADE_PROJECT if project is not None else OutputKind.UPGRADE_ADVANCE,
                        result,
                    ),
                    args.format,
                )
                return 0
            if not args.shared_root:
                raise ValueError("schema-6 upgrade requires an explicit --shared-root.")
            machine = args.machine or "upgrade-coordinator"
            runtime = Path(args.runtime_root) if args.runtime_root else None
            cfg = load_root_config(args.shared_root, machine, runtime, require_initialized=False)
            if args.upgrade_feature == "schema6":
                if args.schema6_upgrade_action == "check":
                    _emit(
                        CliOutput(
                            OutputKind.SCHEMA6_UPGRADE,
                            check_schema6_upgrade(
                                cfg,
                                capabilities=args.capabilities,
                                machine_runtime_root=args.machine_runtime_root,
                            ),
                        ),
                        args.format,
                    )
                    return 0
                if args.schema6_upgrade_action == "status":
                    _emit(CliOutput(OutputKind.SCHEMA6_UPGRADE, schema6_upgrade_status(cfg)), args.format)
                    return 0
                if args.schema6_upgrade_action == "start":
                    _emit(
                        CliOutput(
                            OutputKind.SCHEMA6_UPGRADE,
                            start_schema6_upgrade(
                                cfg,
                                capabilities=args.capabilities,
                                machine_runtime_root=args.machine_runtime_root,
                            ),
                        ),
                        args.format,
                    )
                    return 0
                if args.schema6_upgrade_action == "resume":
                    _emit(
                        CliOutput(
                            OutputKind.SCHEMA6_UPGRADE,
                            resume_schema6_upgrade(
                                cfg,
                                activation_id=args.activation_id,
                                machine_runtime_root=args.machine_runtime_root,
                            ),
                        ),
                        args.format,
                    )
                    return 0
                if not args.confirm_clients_stopped or not args.machine:
                    raise ValueError("attest requires --machine and --confirm-clients-stopped.")
                _emit(
                    CliOutput(
                        OutputKind.SCHEMA6_UPGRADE,
                        attest_schema6_upgrade(
                            cfg,
                            activation_id=args.activation_id,
                            machine_name=args.machine,
                            machine_runtime_root=args.machine_runtime_root,
                        ),
                    ),
                    args.format,
                )
                return 0
        if args.command == "use":
            is_selecting = args.use_shared_root is not None
            if sum((is_selecting, args.show, args.clear)) != 1:
                raise ValueError("use requires exactly one of --shared-root, --show, or --clear.")
            if args.machine is not None or args.runtime_root is not None:
                raise ValueError("qexp use accepts only --shared-root, --show, or --clear.")
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
            if args.format:
                raise ValueError("--format requires --show.")
            if not args.use_shared_root:
                raise ValueError("use requires a non-empty --shared-root.")
            save_context(args.use_shared_root)
            return 0
        if args.command == "agent" and args.agent_action in {
            "status",
            "list-projects",
            "disable-project",
            "enable-project",
            "remove-project",
            "stop",
            "restart",
        }:
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.agent_action == "status":
                _emit(
                    CliOutput(OutputKind.AGENT_STATUS, {"action": "status", **get_machine_agent_status(runtime)}),
                    args.format,
                )
            elif args.agent_action == "list-projects":
                _emit(
                    CliOutput(
                        OutputKind.AGENT_PROJECT_LIST,
                        {"action": "project_list", "projects": get_machine_agent_status(runtime)["projects"]},
                    ),
                    args.format,
                )
            elif args.agent_action == "disable-project":
                binding = set_project_enabled(runtime, args.project, False)
                _emit(
                    CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_disabled", **binding.to_dict()}),
                    args.format,
                )
            elif args.agent_action == "enable-project":
                binding = enable_project(runtime, args.project)
                _emit(
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {
                            "action": "project_enabled",
                            **binding.to_dict(),
                            "message": "Project registration checks passed; new task admission is enabled.",
                        },
                    ),
                    args.format,
                )
            elif args.agent_action == "remove-project":
                binding = unregister_project(runtime, args.project)
                _emit(
                    CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_removed", **binding.to_dict()}),
                    args.format,
                )
            elif args.agent_action == "stop":
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
        if args.command == "agent" and args.agent_action == "upgrade":
            runtime = MachineRuntime(args.machine_runtime_root)
            action = args.agent_upgrade_action
            project = getattr(args, "upgrade_project", None)
            if action == "status":
                if project is None:
                    result = inspect_registered_upgrades(runtime)
                else:
                    result = UpgradeCoordinator(_upgrade_project_config(runtime, project)).status()
                _emit(
                    CliOutput(
                        OutputKind.UPGRADE_REGISTRY_STATUS if project is None else OutputKind.UPGRADE_PROJECT,
                        result,
                    ),
                    args.format,
                )
                return 0
            if action in {"retry", "coordinate"}:
                if project is None:
                    result = advance_registered_upgrades(runtime, force_discovery=True)
                else:
                    result = UpgradeCoordinator(_upgrade_project_config(runtime, project)).advance(force_retry=True)
                _emit(
                    CliOutput(OutputKind.UPGRADE_ADVANCE if project is None else OutputKind.UPGRADE_PROJECT, result),
                    args.format,
                )
                return 0
            cfg = _upgrade_project_config(runtime, project)
            coordinator = UpgradeCoordinator(cfg)
            if action == "pause":
                result = coordinator.request_pause(args.reason)
            elif action == "inspect":
                result = coordinator.status()
            elif action == "plan":
                result = coordinator.inspect_repair(args.target)
            elif action == "apply":
                result = coordinator.apply_repair(args.repair_id)
            elif action == "validate":
                result = coordinator.validate_repair(args.repair_id)
            else:
                result = coordinator.resume()
            output_kind = (
                OutputKind.UPGRADE_REPAIR if action in {"plan", "apply", "validate"} else OutputKind.UPGRADE_PROJECT
            )
            _emit(CliOutput(output_kind, result, {"action": action}), args.format)
            return 0
        if args.command == "agent" and args.agent_action == "cpu-lane":
            runtime = MachineRuntime(args.machine_runtime_root)
            runtime.ensure_layout()
            if args.cpu_lane_action == "set":
                policy = set_cpu_lane_capacity(runtime.root, capacity=args.capacity)
            else:
                policy = get_cpu_lane_policy(runtime.root)
            _emit(CliOutput(OutputKind.CPU_LANE, {"cpu_lane": policy.to_dict}), args.format)
            return 0
        if args.command == "agent" and args.agent_action == "gpus":
            runtime = MachineRuntime(args.machine_runtime_root)
            runtime.ensure_layout()
            if args.gpu_action == "show":
                result = show_gpu_policy(runtime)
            elif args.gpu_action == "reset":
                result = reset_gpu_policy(runtime, expected_revision=args.expected_revision)
            else:
                configured = () if args.none else parse_gpu_id_list(args.visible)
                result = set_gpu_policy(
                    runtime,
                    configured,
                    expected_revision=args.expected_revision,
                )
            _emit(CliOutput(OutputKind.GPU_POLICY, result), args.format)
            return 0
        if args.command == "submit":
            _prepare_submission_request(args, raw_argv, invocation_cwd=Path.cwd())
        cfg, execution_context = _resolve_cfg(args, require_binding=_requires_verified_binding(args))
        if args.command == "submit":
            args._submission_cfg_resolved = True

        def get_execution_context() -> ExecutionContext:
            return execution_context

        def get_lifecycle_kwargs() -> dict[str, MachineRuntime]:
            return {"machine_runtime": execution_context.machine_runtime}

        if args.command == "config":
            if args.config_action == "tmux":
                if args.tmux_action == "show":
                    _emit(CliOutput(OutputKind.TMUX_POLICY, show_tmux_policy(cfg)), args.format)
                    return 0
                _emit(
                    CliOutput(
                        OutputKind.TMUX_POLICY,
                        set_tmux_policy(cfg, args.enabled),
                    ),
                    args.format,
                )
                return 0
            if args.config_action == "progress":
                if args.progress_action == "show":
                    _emit(CliOutput(OutputKind.PROGRESS_POLICY, show_progress_policy(cfg)), args.format)
                    return 0
                _emit(
                    CliOutput(
                        OutputKind.PROGRESS_POLICY,
                        set_progress_policy(cfg, _parse_progress_interval_argument(args.interval_seconds)),
                    ),
                    args.format,
                )
                return 0
            if args.config_action == "launch-handoff":
                if args.launch_handoff_action == "show":
                    _emit(CliOutput(OutputKind.LAUNCH_HANDOFF_POLICY, show_launch_handoff_policy(cfg)), args.format)
                    return 0
                _emit(
                    CliOutput(
                        OutputKind.LAUNCH_HANDOFF_POLICY,
                        set_launch_handoff_policy(
                            cfg,
                            _parse_launch_handoff_timeout_argument(args.timeout_seconds),
                        ),
                    ),
                    args.format,
                )
                return 0
            if args.config_action != "notifications":
                raise ValueError("unknown config action")
            if args.notifications_action == "show":
                _emit(CliOutput(OutputKind.NOTIFICATIONS, load_notifications(cfg)), args.format)
                return 0
            if args.notifications_action == "set":
                if args.enabled and args.disabled:
                    raise ValueError("--enabled and --disabled are mutually exclusive")
                if not args.enabled and not args.disabled:
                    raise ValueError("one of --enabled or --disabled is required")
                value = update_notifications(
                    cfg,
                    lambda current: {
                        **current,
                        "enabled": args.enabled,
                    },
                )
                _emit(CliOutput(OutputKind.NOTIFICATIONS, value), args.format)
                return 0
            if args.provider_action == "set":
                if args.provider != "feishu":
                    raise ValueError(f"unknown notification provider {args.provider!r}")
                if args.enabled and args.disabled:
                    raise ValueError("--enabled and --disabled are mutually exclusive")
                if args.secret_env and args.unset_secret_env:
                    raise ValueError("--secret-env and --unset-secret-env are mutually exclusive")
                if args.webhook_stdin and args.credential_source != "shared_file":
                    raise ValueError("--webhook-stdin requires --credential-source shared_file")
                if args.credential_source == "shared_file" and not args.acknowledge_shared_secret_risk:
                    raise ValueError("--credential-source shared_file requires --acknowledge-shared-secret-risk")
                shared_webhook = None
                if args.webhook_stdin:
                    shared_webhook = sys.stdin.readline().rstrip("\r\n")
                    if not shared_webhook:
                        raise ValueError("--webhook-stdin requires a non-empty first input line")

                def update_provider(current):
                    providers = dict(current["providers"])
                    provider_value = dict(
                        providers.get(
                            "feishu",
                            {
                                "enabled": False,
                                "webhook_env": DEFAULT_WEBHOOK_ENV,
                                "secret_env": None,
                                "timeout_seconds": 5,
                                "credential_source": "env",
                            },
                        )
                    )
                    if args.enabled:
                        provider_value["enabled"] = True
                    if args.disabled:
                        provider_value["enabled"] = False
                    if args.credential_source is not None:
                        provider_value["credential_source"] = args.credential_source
                    if args.webhook_env is not None:
                        provider_value["webhook_env"] = args.webhook_env
                    if args.secret_env is not None:
                        provider_value["secret_env"] = args.secret_env
                    if args.unset_secret_env:
                        provider_value["secret_env"] = None
                    if args.timeout_seconds is not None:
                        provider_value["timeout_seconds"] = args.timeout_seconds
                    providers["feishu"] = provider_value
                    return {**current, "providers": providers}

                value = update_notifications(cfg, update_provider)
                if shared_webhook is not None:
                    write_shared_feishu_webhook(cfg, shared_webhook)
                _emit(CliOutput(OutputKind.NOTIFICATIONS, value), args.format)
                return 0
        if args.command == "lease-policy" or (args.command == "config" and args.config_action == "lease"):
            current = load_lease_policy(cfg)
            if args.lease_policy_action == "show":
                _emit(CliOutput(OutputKind.LEASE_POLICY, {"lease_policy": asdict(current)}), args.format)
                return 0
            has_active_claim = any(
                bool(data.get("task", {}).get("claim_control", {}).get("active_claim"))
                for path in iter_json(shared_paths(cfg.shared_root)["tasks"])
                for data in [read_json(path)]
            )
            if has_active_claim:
                raise RuntimeError("lease policy cannot change while an active claim exists.")
            values = asdict(current)
            for field, value in {
                "ttl_seconds": args.ttl_seconds,
                "renew_interval_seconds": args.renew_interval_seconds,
                "max_clock_skew_seconds": args.max_clock_skew_seconds,
                "clock_observation_max_age_seconds": args.clock_observation_max_age_seconds,
                "clock_provider_margin_seconds": args.clock_provider_margin_seconds,
                "renewal_commit_margin_seconds": args.renewal_commit_margin_seconds,
            }.items():
                if value is not None:
                    values[field] = value
            if args.clock_provider_priority is not None:
                values["clock_provider_priority"] = tuple(
                    item.strip() for item in args.clock_provider_priority.split(",") if item.strip()
                )
            updated = LeasePolicy(**values)
            save_lease_policy(cfg, updated)
            _emit(CliOutput(OutputKind.LEASE_POLICY, {"lease_policy": values}), args.format)
            return 0
        if args.command == "submit":

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
        if args.command == "task":
            if args.task_action == "dependencies":
                if args.dependencies_action == "show":
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
                        cfg, args.task_id, args.depends_on, action=args.dependencies_action
                    )
                    _emit(
                        CliOutput(
                            OutputKind.DEPENDENCIES,
                            {"task_id": task_value.task_id, "depends_on_task_ids": task_value.depends_on_task_ids},
                        ),
                        args.format,
                    )
                return 0
            if args.task_action == "cancel":
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
                        },
                    ),
                    args.format,
                )
            elif args.task_action == "retry":
                ensure_local_agent_active(cfg, reason="task-retry", **get_lifecycle_kwargs())
                task_value = task_commands.retry(
                    cfg,
                    args.task_id,
                    acknowledge_duplicate_risk=args.acknowledge_duplicate_risk,
                )
                print(task_value.task_id)
            elif args.task_action == "offer":
                ensure_local_agent_active(cfg, reason="task-offer", **get_lifecycle_kwargs())
                result = task_commands.offer(cfg, args.task_id)
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif args.task_action == "share":
                after_seconds = _duration_seconds(args.after) if args.after is not None else None
                ensure_local_agent_active(cfg, reason="task-share", **get_lifecycle_kwargs())
                result = task_commands.share(
                    cfg,
                    args.task_id,
                    after_seconds=after_seconds,
                    helper_machines=_split_machine_list(args.helper_machines),
                )
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif args.task_action == "keep-local":
                ensure_local_agent_active(cfg, reason="task-keep-local", **get_lifecycle_kwargs())
                result = task_commands.keep_local(cfg, args.task_id)
                _emit(CliOutput(OutputKind.AVAILABILITY, result.to_dict()), args.format)
            elif args.task_action == "list":
                is_paginated = args.page_size is not None or args.cursor is not None
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
                        page_size=page_size,
                        cursor=args.cursor,
                    )
                    _emit_task_page(cfg, args, page, page_size)
                else:
                    limit = 50 if args.limit is None else args.limit
                    _emit(
                        CliOutput(
                            OutputKind.TASK_LIST,
                            observer.list_tasks(cfg, phase=args.phase, group=args.group, limit=limit),
                        ),
                        args.format,
                    )
            elif args.task_action == "show":
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
            elif args.task_action == "logs":
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
                print(log_commands.read_logs(cfg, args.task_id), end="")
            return 0
        if args.command == "group":
            presentation = None
            if args.group_action == "create":
                result = group_commands.create_group(cfg, args.name, args.workers)
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": "create"}
            elif args.group_action == "list":
                result = observer.list_groups(cfg)
                kind = OutputKind.GROUP_LIST
            elif args.group_action == "show":
                result = group_commands.show_group(cfg, args.name)
                kind = OutputKind.GROUP_SHOW
            elif args.group_action == "retry-failed":
                ensure_local_agent_active(cfg, reason="group-retry-failed", **get_lifecycle_kwargs())
                result = {
                    "task_ids": [task_value.task_id for task_value in group_commands.group_retry_failed(cfg, args.name)]
                }
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": "retry-failed", "name": args.name, "status": "completed"}
            elif args.group_action == "machines":
                if args.machine_action == "list":
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
                    args.machine_action,
                    terminate_running=getattr(args, "terminate_running", False),
                    role=getattr(args, "role", None),
                    gpu_limit_gpus=None if gpu_limit_gpus in {None, "unlimited"} else gpu_limit_gpus,
                    has_gpu_limit=gpu_limit_gpus is not None,
                )
                kind = OutputKind.GROUP_OPERATION
                presentation = {
                    "action": args.machine_action,
                    "worker_machine": args.worker_machine,
                }
                worker_control = result.get("worker_control") if isinstance(result, dict) else None
                if (
                    args.machine_action == "remove"
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
            else:
                context = get_execution_context()
                if args.group_action == "resume":
                    ensure_local_agent_active(cfg, reason="group-resume", **get_lifecycle_kwargs())
                result = group_commands.group_control(
                    cfg,
                    args.name,
                    args.group_action,
                    terminate_running=getattr(args, "terminate_running", False),
                    reservation_runtime_root=context.reservation_root,
                )
                if args.group_action == "cancel":
                    control = result.get("cancellation_operation", {})
                    if (
                        isinstance(control, dict)
                        and isinstance(control.get("discovery"), dict)
                        and control.get("state") in {"preparing", "converging", "waiting_ack", "blocked"}
                    ):
                        ensure_local_agent_active(cfg, reason="group-cancel", **get_lifecycle_kwargs())
                kind = OutputKind.GROUP_OPERATION
                presentation = {"action": args.group_action}
                if args.group_action == "cancel":
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
            _emit(
                CliOutput(kind, result, presentation or {}),
                args.format,
            )
            return 0
        if args.command == "agent":
            runtime = get_execution_context().machine_runtime
            if args.agent_action == "add-project":
                warned_generation = None
                if args.adopt_existing:
                    warned_generation = adoption_warning_generation(runtime, cfg)
                    if warned_generation is not None:
                        print(
                            f"Warning: registration generation {warned_generation!r} will be replaced; "
                            "the previous environment will lose automatic access to this name on its next start.",
                            file=sys.stderr,
                        )
                registration = register_project(
                    runtime,
                    cfg.shared_root,
                    cfg.machine_name,
                    adopt_existing=args.adopt_existing,
                )
                action = "project_added" if registration.is_added else "project_already_registered"
                if registration.is_adopted and warned_generation is None:
                    print(
                        f"Warning: registration generation {registration.binding.registration_generation!r} "
                        "replaced the previous logical-machine ownership; the previous environment will lose "
                        "automatic access to this name on its next start.",
                        file=sys.stderr,
                    )
                result = {
                    "action": action,
                    **registration.binding.to_dict(),
                    "message": registration.message,
                }
                if not registration.binding.enabled:
                    result["enable_command"] = _enable_command(runtime, registration.binding)
                _emit(CliOutput(OutputKind.AGENT_OPERATION, result), args.format)
                return 0
            if args.agent_action == "migrate-project":
                binding = migrate_project(runtime, cfg)
                _process, status = ensure_machine_agent_started(runtime)
                siblings = [
                    str(path)
                    for path in sorted(cfg.shared_root.parent.parent.glob("*/.qexp"))
                    if path.resolve() != cfg.shared_root
                ]
                _emit(
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {"action": "project_migrated", **binding.to_dict(), **status, "migration_candidates": siblings},
                    ),
                    args.format,
                )
                return 0
            if args.agent_action == "list-projects":
                _emit(
                    CliOutput(
                        OutputKind.AGENT_PROJECT_LIST,
                        {"action": "project_list", "projects": get_machine_agent_status(runtime)["projects"]},
                    ),
                    args.format,
                )
                return 0
            if args.agent_action == "disable-project":
                binding = set_project_enabled(runtime, args.project, False)
                _emit(
                    CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_disabled", **binding.to_dict()}),
                    args.format,
                )
                return 0
            if args.agent_action == "remove-project":
                binding = unregister_project(runtime, args.project)
                _emit(
                    CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_removed", **binding.to_dict()}),
                    args.format,
                )
                return 0
            if args.agent_action == "status":
                _emit(
                    CliOutput(OutputKind.AGENT_STATUS, {"action": "status", **get_machine_agent_status(runtime)}),
                    args.format,
                )
                return 0
            if args.agent_action == "start":
                action, status = start_local_agent(
                    cfg, reason="manual_start", require_eligible_work=False, machine_runtime=runtime
                )
                _emit(CliOutput(OutputKind.AGENT_OPERATION, {"action": action, **status}), args.format)
            elif args.agent_action == "run":
                run_local_agent_foreground(
                    cfg,
                    reason="manual_run",
                    on_started=lambda status: _emit(
                        CliOutput(OutputKind.AGENT_OPERATION, {"action": "running", **status}),
                        args.format,
                        flush=True,
                    ),
                    machine_runtime=runtime,
                )
            elif args.agent_action == "restart":
                action, status = restart_local_agent(cfg, machine_runtime=runtime)
                _emit(CliOutput(OutputKind.AGENT_OPERATION, {"action": action, **status}), args.format)
            elif args.agent_action == "stop":
                action, status = stop_local_agent(cfg, machine_runtime=runtime)
                _emit(CliOutput(OutputKind.AGENT_OPERATION, {"action": action, **status}), args.format)
            return 0
        if args.command == "top":
            result = observer.top_view(cfg, all_machines=True)
            _emit(CliOutput(OutputKind.TOP, result), args.format)
            return 0
        if args.command == "machines":
            result = observer.list_machines(cfg)
            _emit(CliOutput(OutputKind.MACHINES, result), args.format)
            return 0
        if args.command == "doctor":
            context = get_execution_context()
            result = (
                verify_integrity(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                    project_id=context.project_id,
                    max_work_items=args.max_work_items,
                )
                if args.action == "verify"
                else repair_metadata(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                    max_work_items=args.max_work_items,
                )
            )
            output_kind = OutputKind.DOCTOR_VERIFY if args.action == "verify" else OutputKind.DOCTOR_REPAIR
            _emit(CliOutput(output_kind, result), args.format)
            return resolve_verify_exit_code(result, strict=args.strict)
        if args.command == "clean":
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
            _emit(CliOutput(OutputKind.CLEAN, result), args.format)
            return 0
    except ObservationError as exc:
        return _emit_observation_error(exc, getattr(args, "format", "human"))
    except KeyboardInterrupt as exc:
        if args.command == "submit":
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
        if args.command == "submit":
            payload, exit_code = _submission_error_payload(args, exc)
            if getattr(args, "quiet", False):
                print(f"qexp: {exc}", file=sys.stderr)
            elif getattr(args, "format", "human") == "json":
                print(json.dumps(payload))
                print(f"qexp: {exc}", file=sys.stderr)
            else:
                print(f"qexp: {exc}", file=sys.stderr)
            return exit_code
        if (
            args.command == "task"
            and args.task_action == "list"
            and (args.page_size is not None or args.cursor is not None)
        ):
            code = "invalid_argument" if isinstance(exc, ValueError) else "index_unavailable"
            return _emit_observation_error(ObservationError(code, str(exc)), args.format)
        if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError) and not _is_continuous_task(args):
            raise
        print(f"qexp: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
