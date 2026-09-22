"""Resolved-Project command handlers for qexp CLI."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

from .. import observer
from ..activation import ensure_local_agent_active
from ..agent.context import ExecutionContext
from ..commands import cleanup
from ..commands import configuration as configuration_commands
from ..commands import group as group_commands
from ..commands import logs as log_commands
from ..commands import operation as operation_commands
from ..commands import status as status_commands
from ..commands import task as task_commands
from ..commands import wait as wait_commands
from ..commands import watch as watch_commands
from ..config_types import RootConfig
from ..doctor import repair_metadata, resolve_verify_exit_code, verify_integrity
from ..progress_policy import validate_interval_seconds
from ..runtime.observation.api import ObservationError
from .errors import CliUsageError
from .outcome import CommandOutcome
from .output import CliOutput, OutputKind


def _parse_page_size(value: str | None) -> int:
    """Parse a CLI page size so malformed values become ObservationErrors."""
    if value is None:
        return 50
    try:
        return int(value, 10)
    except (TypeError, ValueError) as exc:
        raise ObservationError("invalid_argument", "page_size must be an integer from 1 through 1000.", 2) from exc


def _task_page_output(
    cfg: RootConfig,
    args: argparse.Namespace,
    page: dict[str, object],
    page_size: int,
) -> CliOutput[object]:
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
    return CliOutput(OutputKind.TASK_PAGE, page, presentation)


def _duration_seconds(value: str) -> int:
    matched = re.fullmatch(r"([0-9]+)([smh])", value)
    if not matched:
        raise CliUsageError("duration must use an explicit s, m, or h unit, for example 10m.")
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


def _split_machine_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    machines = [machine.strip() for value in values for machine in value.split(",")]
    if not all(machines):
        raise CliUsageError("--with machine names must be comma-separated non-empty values.")
    return machines


def dispatch_project(
    args: argparse.Namespace,
    cfg: RootConfig,
    execution_context: ExecutionContext,
    selection_source: str,
) -> CommandOutcome:
    """Dispatch a handler after the entry point resolved Project authority."""
    handler = args.command_spec.handler

    def get_execution_context() -> ExecutionContext:
        return execution_context

    def get_lifecycle_kwargs() -> dict[str, object]:
        return {"machine_runtime": execution_context.machine_runtime}

    if handler.startswith("config_"):
        section = getattr(args, "section", None)
        if handler == "config_show":
            result = configuration_commands.show_config(section, cfg=cfg, runtime=execution_context.machine_runtime)
        elif handler == "config_set":
            if args.enabled and args.disabled:
                raise CliUsageError("--enabled and --disabled are mutually exclusive")
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
                    raise CliUsageError("--webhook-stdin requires --credential-source shared_file")
                values["shared_webhook"] = sys.stdin.readline().rstrip("\r\n")
                if not values["shared_webhook"]:
                    raise CliUsageError("--webhook-stdin requires a non-empty first input line")
            try:
                result = configuration_commands.set_config(
                    section,
                    cfg=cfg,
                    runtime=execution_context.machine_runtime,
                    provider=args.provider,
                    values=values,
                )
            except ValueError as exc:
                raise CliUsageError(str(exc)) from exc
        else:
            try:
                result = configuration_commands.reset_config(
                    section,
                    cfg=cfg,
                    runtime=execution_context.machine_runtime,
                    provider=args.provider,
                )
            except ValueError as exc:
                raise CliUsageError(str(exc)) from exc
        if handler == "config_show" and section is None and result.get("complete") is False:
            return CommandOutcome(1, CliOutput(OutputKind.CONFIG, result))
        return CommandOutcome(0, CliOutput(OutputKind.CONFIG, result))

    if handler.startswith("task_"):
        if handler.startswith("task_dependencies_"):
            dependency_action = handler.removeprefix("task_dependencies_")
            if handler == "task_dependencies_show":
                task_value = task_commands.load_task(cfg, args.task_id)
                output = CliOutput(
                    OutputKind.DEPENDENCIES,
                    {"task_id": task_value.task_id, "depends_on_task_ids": task_value.depends_on_task_ids},
                )
            else:
                task_value = task_commands.edit_dependencies(
                    cfg, args.task_id, args.depends_on, action=dependency_action
                )
                output = CliOutput(
                    OutputKind.DEPENDENCIES,
                    {"task_id": task_value.task_id, "depends_on_task_ids": task_value.depends_on_task_ids},
                )
            return CommandOutcome(0, output)
        if handler == "task_cancel":
            context = get_execution_context()
            task_value = task_commands.cancel(cfg, args.task_id, reservation_runtime_root=context.reservation_root)
            claim = task_value.claim_control.get("active_claim") or {}
            is_pending = bool(
                task_value.state["projection"] == "running"
                and task_value.control.get("terminate_running")
                and not task_value.control.get("termination_acknowledged_at")
            )
            is_blocked = task_value.state["projection"] == "blocked"
            output = CliOutput(
                OutputKind.TASK_CANCEL,
                {
                    "action": "cancel",
                    "outcome": "waiting_ack" if is_pending else "blocked" if is_blocked else "completed",
                    "task_id": task_value.task_id,
                    "task_state": task_value.state["projection"],
                    "owning_machine": claim.get("machine_name") or task_value.placement_policy["home_machine"],
                    "operation_state": "waiting_ack" if is_pending else "blocked" if is_blocked else "completed",
                    "pending_acknowledgement": is_pending,
                    "termination_acknowledged_at": task_value.control.get("termination_acknowledged_at"),
                    "reason": task_value.state.get("reason"),
                    "follow_up_command": f"qexp task show {shlex.quote(task_value.task_id)} --project {shlex.quote(str(cfg.project_root))}",
                },
            )
            return CommandOutcome(1 if is_blocked else 0, output)
        elif handler == "task_retry":
            if args.quiet and args.format == "json":
                raise CliUsageError("--quiet cannot be combined with --format json.")
            ensure_local_agent_active(cfg, reason="task-retry", **get_lifecycle_kwargs())
            task_value = task_commands.retry(cfg, args.task_id)
            if args.quiet:
                print(task_value.task_id)
            else:
                return CommandOutcome(
                    0,
                    CliOutput(
                        OutputKind.TASK_RETRY,
                        {
                            "action": "retry",
                            "outcome": "accepted",
                            "task_id": task_value.task_id,
                            "task_state": task_value.state["projection"],
                            "operation_state": "accepted",
                            "follow_up_command": (
                                f"qexp task show {shlex.quote(task_value.task_id)} "
                                f"--project {shlex.quote(str(cfg.project_root))}"
                            ),
                        },
                    ),
                )
            return CommandOutcome(0)
        elif handler == "task_offer":
            ensure_local_agent_active(cfg, reason="task-offer", **get_lifecycle_kwargs())
            result = task_commands.offer(cfg, args.task_id)
            payload = result.to_dict()
            payload["outcome"] = "no_change" if payload["idempotent"] else "completed"
            return CommandOutcome(0, CliOutput(OutputKind.AVAILABILITY, payload))
        elif handler == "task_share":
            after_seconds = _duration_seconds(args.after) if args.after is not None else None
            ensure_local_agent_active(cfg, reason="task-share", **get_lifecycle_kwargs())
            result = task_commands.share(
                cfg,
                args.task_id,
                after_seconds=after_seconds,
                helper_machines=_split_machine_list(args.helper_machines),
            )
            payload = result.to_dict()
            payload["outcome"] = "no_change" if payload["idempotent"] else "completed"
            return CommandOutcome(0, CliOutput(OutputKind.AVAILABILITY, payload))
        elif handler == "task_unshare":
            ensure_local_agent_active(cfg, reason="task-unshare", **get_lifecycle_kwargs())
            result = task_commands.keep_local(cfg, args.task_id)
            payload = result.to_dict()
            payload["outcome"] = "no_change" if payload["idempotent"] else "completed"
            return CommandOutcome(0, CliOutput(OutputKind.AVAILABILITY, payload))
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
                return CommandOutcome(0, _task_page_output(cfg, args, page, page_size))
            else:
                limit = 50 if args.limit is None else args.limit
                return CommandOutcome(
                    0,
                    CliOutput(
                        OutputKind.TASK_LIST,
                        [
                            item
                            for item in observer.list_tasks(cfg, phase=args.phase, group=args.group, limit=limit)
                            if args.name is None or item.get("name") == args.name
                        ],
                    ),
                )
        elif handler == "task_show":
            if args.watch:
                try:
                    return CommandOutcome(
                        watch_commands.watch_task(
                            cfg,
                            args.task_id,
                            interval_seconds=args.interval_seconds,
                            follow_retries=args.follow_retries,
                        )
                    )
                except KeyboardInterrupt:
                    _restore_watch_terminal()
                    return CommandOutcome(130)
                except BrokenPipeError:
                    return CommandOutcome(0)
            return CommandOutcome(0, CliOutput(OutputKind.TASK_SHOW, observer.inspect_task(cfg, args.task_id)))
        elif handler == "task_logs":
            if args.follow:
                try:
                    return CommandOutcome(
                        log_commands.follow_logs(
                            cfg,
                            args.task_id,
                            tail_lines=args.tail,
                            interval_seconds=args.interval_seconds,
                            follow_retries=args.follow_retries,
                        )
                    )
                except KeyboardInterrupt:
                    return CommandOutcome(130)
                except BrokenPipeError:
                    return CommandOutcome(0)
            log_commands.write_logs(cfg, args.task_id, tail_lines=args.tail)
            return CommandOutcome(0)
        elif handler == "task_wait":
            timeout = wait_commands.parse_wait_timeout(args.timeout)
            result, wait_exit = wait_commands.wait_for_task(cfg, args.task_id, timeout_seconds=timeout)
            return CommandOutcome(wait_exit, CliOutput(OutputKind.TASK_WAIT, result))
        return CommandOutcome(0)
    if handler.startswith("group_"):
        if handler == "group_create":
            result = group_commands.create_group(cfg, args.name, args.workers)
            kind = OutputKind.GROUP_STATE_CHANGE
            result = {"action": "create", "outcome": "completed", **result}
        elif handler == "group_list":
            result = observer.list_groups(cfg)
            kind = OutputKind.GROUP_LIST
        elif handler == "group_show":
            result = group_commands.show_group(cfg, args.name)
            kind = OutputKind.GROUP_SHOW
        elif handler == "group_retry":
            ensure_local_agent_active(cfg, reason="group-retry", **get_lifecycle_kwargs())
            result = group_commands.group_retry_failed(cfg, args.name)
            kind = OutputKind.GROUP_RETRY
            result = {
                "action": "retry",
                "outcome": "completed" if result["retried_count"] else "no_change",
                **result,
                "follow_up_command": f"qexp group show {shlex.quote(args.name)} --project {shlex.quote(str(cfg.project_root))}",
            }
        elif handler.startswith("group_worker_"):
            worker_action = handler.removeprefix("group_worker_")
            if handler == "group_worker_list":
                result = observer.list_group_machines(
                    cfg,
                    args.group_name,
                    reservation_runtime_root=execution_context.reservation_root,
                )
                return CommandOutcome(0, CliOutput(OutputKind.GROUP_MACHINES, result))
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
            kind = OutputKind.GROUP_WORKER_CHANGE
            result = {
                "action": worker_action,
                "outcome": "completed",
                "worker_machine": args.worker_machine,
                **result,
            }
            worker_control = result.get("worker_control") if isinstance(result, dict) else None
            group_record = result.get("group") if isinstance(result.get("group"), dict) else {}
            worker_set = group_record.get("worker_set") if isinstance(group_record.get("worker_set"), dict) else {}
            worker = worker_set.get(args.worker_machine)
            worker_is_present = isinstance(worker, dict)
            if not isinstance(worker, dict) and isinstance(worker_control, dict):
                worker = worker_control.get("worker_before")
            if isinstance(worker, dict):
                result.update(
                    {
                        "worker_state": (
                            "removed"
                            if not worker_is_present and worker_control.get("state") == "completed"
                            else worker.get("state")
                        ),
                        "scheduling_role": worker.get("scheduling_role"),
                        "gpu_limit_gpus": worker.get("gpu_limit_gpus"),
                    }
                )
            if (
                handler == "group_worker_remove"
                and isinstance(worker_control, dict)
                and isinstance(worker_control.get("discovery"), dict)
                and worker_control.get("state") in {"preparing", "converging", "waiting_ack", "blocked"}
            ):
                ensure_local_agent_active(cfg, reason="group-worker-remove", **get_lifecycle_kwargs())
                result["outcome"] = "blocked" if worker_control.get("state") == "blocked" else "waiting_ack"
                result["status"] = worker_control.get("state")
                result["reason"] = worker_control.get("blocked_reason")
            worker_control = result.get("worker_control") if isinstance(result, dict) else None
            if isinstance(worker_control, dict):
                result["blockers"] = list(worker_control.get("blockers") or [])
                pending = worker_control.get("pending_machine_acknowledgements")
                result["pending_machines"] = list(pending) if isinstance(pending, dict) else []
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
            kind = OutputKind.GROUP_CANCEL if handler == "group_cancel" else OutputKind.GROUP_STATE_CHANGE
            result = {"action": group_action, "outcome": "completed", **result}
            if handler == "group_cancel":
                control = result.get("cancellation_operation", {})
                pending_machines = control.get("pending_machine_acknowledgements", {})
                result.update(
                    {
                        "outcome": "blocked"
                        if control.get("state") == "blocked"
                        else "waiting_ack"
                        if control.get("state") in {"preparing", "converging", "waiting_ack"}
                        else "completed",
                        "status": control.get("state"),
                        "pending_machines": list(pending_machines.keys()) if isinstance(pending_machines, dict) else [],
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
        return CommandOutcome(0, CliOutput(kind, result))
    if handler == "status":
        result = status_commands.project_status(
            cfg,
            selection_source=selection_source,
            machine_runtime=execution_context.machine_runtime,
        )
        return CommandOutcome(0, CliOutput(OutputKind.STATUS, result))
    if handler in {"machine_list", "machine_show"}:
        if handler == "machine_list":
            result = observer.list_machines(cfg)
            return CommandOutcome(0, CliOutput(OutputKind.MACHINES, result))
        else:
            result = status_commands.machine_detail(cfg, args.name)
            return CommandOutcome(0, CliOutput(OutputKind.MACHINE_SHOW, result))
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
        return CommandOutcome(resolve_verify_exit_code(result, strict=args.strict), CliOutput(output_kind, result))
    if handler == "admin_operation_show":
        if args.project is None:
            raise CliUsageError("admin operation show requires explicit --project PATH.")
        result, operation_exit = operation_commands.inspect_operation(cfg, args.reference)
        return CommandOutcome(operation_exit, CliOutput(OutputKind.OPERATION, result))
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
        return CommandOutcome(0, CliOutput(OutputKind.CLEAN, result))

    return CommandOutcome(0)
