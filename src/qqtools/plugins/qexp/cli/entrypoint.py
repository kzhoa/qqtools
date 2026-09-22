"""qexp CLI entry point and cross-command orchestration."""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from ..agent.context import ExecutionContext, MachineRuntime
from ..commands import context as context_commands
from ..commands import wait as wait_commands
from ..commands.registry import CommandSpec
from ..config_types import RootConfig
from ..formatter import CliOutput, OutputKind, render
from ..runtime.observation.api import ObservationError
from .local_handlers import LOCAL_HANDLERS, dispatch_local
from .parser import (
    _is_json_argv,
    _JsonParseError,
    _PaginationParseError,
    _set_parse_modes,
    _SubmissionParseError,
    build_parser,
)
from .project_handlers import _is_continuous_task, _validate_continuous_options, dispatch_project
from .submission import (
    _prepare_submission_request,
    _submission_error_payload,
    _submission_parse_failure_payload,
    dispatch_submission,
)

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


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _set_parse_modes(raw_argv)
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
        _set_parse_modes(None)
    resolved_cfg: RootConfig | None = None
    selection_source: str | None = None
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
        is_local = handler in LOCAL_HANDLERS and (
            not handler.startswith("config_") or getattr(args, "section", None) == "agent"
        )
        if is_local:
            return dispatch_local(handler, args, emitter=_emit)
        if handler in {"admin_check", "admin_repair", "admin_clean"} and getattr(args, "project", None) is None:
            raise ValueError(f"admin {handler.removeprefix('admin_')} requires explicit --project PATH.")
        if handler == "submit":
            _prepare_submission_request(args, raw_argv, invocation_cwd=Path.cwd())
        cfg, execution_context, selection_source = _resolve_cfg(args, require_binding=_requires_verified_binding(args))
        resolved_cfg = cfg
        if handler == "submit":
            args._submission_cfg_resolved = True
            return dispatch_submission(args, cfg, execution_context, emitter=_emit)
        return dispatch_project(args, cfg, execution_context, selection_source, emitter=_emit)
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
        if (
            isinstance(exc, OSError)
            and not isinstance(exc, FileNotFoundError)
            and not _is_continuous_task(args)
            and getattr(args, "format", "human") != "json"
        ):
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
