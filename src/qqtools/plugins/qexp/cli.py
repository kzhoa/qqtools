"""qexp command line routing for the schema-6 product contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import re
from dataclasses import asdict
from pathlib import Path

from .activation import (
    ensure_local_agent_active,
    managed_project_agent_status,
    restart_local_agent,
    run_local_agent_foreground,
    start_local_agent,
    stop_local_agent,
)
from .agent import get_agent_status
from .commands import cleanup, group as group_commands, logs as log_commands, task as task_commands
from .doctor import repair_metadata, resolve_verify_exit_code, verify_integrity
from .layout import clear_context, load_context, load_root_config, migrate_schema5_to_schema6, save_context
from .config_types import RootConfig
from .lease import LeasePolicy, load_lease_policy, save_lease_policy
from .runtime.paths import shared_paths
from .runtime.store import iter_json, read_json
from .machine_agent import (
    ensure_machine_agent_started,
    get_machine_agent_status,
    migrate_project,
    register_project,
    restart_machine_agent,
    set_project_enabled,
    stop_machine_agent,
    unregister_project,
)
from .machine_config import init_shared_root, load_machine_policy
from .machine_runtime import ExecutionContext, MachineRuntime
from .notification_config import (
    DEFAULT_WEBHOOK_ENV,
    load_notifications,
    update_notifications,
    write_shared_feishu_webhook,
)
from . import observer
from .formatter import render


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("human", "json"), default="human")


def _emit(
    kind: str, result: object, output_format: str, *, tasks: list[dict] | None = None, flush: bool = False
) -> None:
    print(render(kind, result, output_format, tasks=tasks or ()), flush=flush)


def _machine_assertion(args: argparse.Namespace) -> str | None:
    """Return the caller's identity assertion after checking duplicate inputs."""
    flag_value = getattr(args, "machine", None)
    environment_value = os.environ.get("QEXP_MACHINE")
    if flag_value is not None and environment_value is not None and flag_value != environment_value:
        raise ValueError(
            f"--machine {flag_value!r} conflicts with QEXP_MACHINE {environment_value!r}."
        )
    return flag_value if flag_value is not None else environment_value


def _shared_root_input(args: argparse.Namespace) -> tuple[str | None, dict | None]:
    """Resolve only the shared-root locator; saved identity fields are intentionally ignored."""
    flag_value = getattr(args, "shared_root", None)
    environment_value = os.environ.get("QEXP_SHARED_ROOT")
    context = load_context() if flag_value is None and environment_value is None else None
    shared = flag_value or environment_value or (context or {}).get("shared_root")
    return shared, context


def _requires_verified_binding(args: argparse.Namespace) -> bool:
    """Classify commands that can create or change project-owned state."""
    if args.command in {"submit", "batch-submit", "clean"}:
        return True
    if args.command == "config":
        return (
            getattr(args, "notifications_action", None) == "set"
            or getattr(args, "provider_action", None) == "set"
        )
    if args.command == "lease-policy":
        return args.lease_policy_action == "set"
    if args.command == "task":
        return args.task_action not in {"list", "show", "logs"}
    if args.command == "group":
        return args.group_action not in {"list", "show"}
    if args.command == "agent":
        return args.agent_action in {"start", "run"}
    return args.command == "doctor" and args.action == "repair"


def _resolve_cfg(
    args: argparse.Namespace, *, require_binding: bool
) -> tuple[object, ExecutionContext]:
    shared, saved_context = _shared_root_input(args)
    if not shared:
        raise ValueError("--shared-root is required or must be saved with qexp use.")
    assertion = _machine_assertion(args)
    machine_runtime = MachineRuntime(getattr(args, "machine_runtime_root", None))

    if require_binding:
        execution_context = machine_runtime.verified_execution_context(shared)
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
    legacy_machine = assertion
    if args.command == "agent" and args.agent_action in {"add-project", "migrate-project"}:
        legacy_machine = legacy_machine or (saved_context or {}).get("machine")
    machine = legacy_machine or "unbound"
    runtime = None
    if args.command == "agent" and args.agent_action == "migrate-project":
        runtime = getattr(args, "runtime_root", None) or os.environ.get("QEXP_RUNTIME_ROOT")
        runtime = runtime or (saved_context or {}).get("runtime_root")
    cfg = load_root_config(shared, machine, runtime, require_initialized=True)
    return cfg, ExecutionContext(cfg, machine_runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "qexp schema-6 experiment queue; --machine is local identity, --home-machine is Task "
            "placement, and Attempt machine is selected later by claim. qexp does not remotely "
            "start a target agent."
        ),
        epilog=(
            "To join a new machine to an existing project, run: qexp init --shared-root "
            "<project/.qexp> --machine <local-machine>. qexp use only saves local CLI context; "
            "it does not initialize or register a project."
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
        help="Initialize a project or join this machine to an existing project.",
        description=(
            "Initialize a new shared project or join this machine to an existing project. "
            "It creates or updates this machine's Project record, registers the project with "
            "the local machine agent, and saves CLI context. It does not start the agent."
        ),
    )
    init.add_argument("--agent-mode", choices=["on_demand", "daemon"], default="on_demand")
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--to-schema", type=int, required=True)
    lease_policy = commands.add_parser("lease-policy")
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
    notifications = config_sub.add_parser("notifications")
    notifications_sub = notifications.add_subparsers(dest="notifications_action", required=True)
    notifications_show = notifications_sub.add_parser("show")
    _add_output_format(notifications_show)
    notifications_set = notifications_sub.add_parser("set")
    _add_output_format(notifications_set)
    notifications_set.add_argument("--enabled", action="store_true")
    notifications_set.add_argument("--disabled", action="store_true")
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
            "Submit one Task using the verified local identity. --home-machine selects placement; "
            "private Tasks are executable only by that home machine. qexp does not remotely start "
            "the target agent."
        ),
    )
    for action in (submit,):
        action.add_argument("--task-id")
        action.add_argument("--name")
        action.add_argument("--group")
        action.add_argument("--gpus", type=int, default=1)
        action.add_argument("--cwd")
        action.add_argument(
            "--home-machine",
            default="current",
            help="Task home placement (default: verified current machine); does not activate a remote agent.",
        )
        action.add_argument("--sharing", choices=["private", "spillover"], default="private")
        action.add_argument("--offer-after-seconds", type=int)
        action.add_argument("--idempotency-key")
        action.add_argument("--no-activate", action="store_true", help="Submit without activating the local agent.")
        action.add_argument("argv", nargs=argparse.REMAINDER)
    bulk = commands.add_parser("batch-submit")
    _add_output_format(bulk)
    bulk.add_argument("--file", required=True, dest="manifest_file")
    bulk.add_argument("--group")
    bulk.add_argument("--idempotency-key")
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
    listing.add_argument("--limit", type=int, default=50)
    show = task_sub.add_parser("show")
    _add_output_format(show)
    show.add_argument("task_id")
    logs = task_sub.add_parser("logs")
    logs.add_argument("task_id")
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
            action.add_argument("--gpu-limit-gpus", type=_gpu_limit_gpus)
        if name == "remove":
            action.add_argument("--terminate-running", action="store_true")
    agent = commands.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_action", required=True)
    for name in (
        "start", "run", "restart", "status", "stop", "add-project", "list-projects",
        "disable-project", "remove-project", "migrate-project",
    ):
        action = agent_sub.add_parser(name)
        _add_output_format(action)
    for name in ("disable-project", "remove-project"):
        agent_sub.choices[name].add_argument("project")
    top = commands.add_parser("top")
    _add_output_format(top)
    machine_list = commands.add_parser("machines")
    _add_output_format(machine_list)
    logs_top = commands.add_parser("logs")
    logs_top.add_argument("task_id")
    doctor = commands.add_parser("doctor")
    _add_output_format(doctor)
    doctor.add_argument("action", choices=["verify", "repair"], default="verify", nargs="?")
    doctor.add_argument("--strict", action="store_true")
    clean = commands.add_parser(
        "clean",
        help="Remove terminal qexp metadata while preserving experiment work directories.",
        description="Remove terminal qexp metadata while preserving experiment work directories.",
    )
    _add_output_format(clean)
    clean_scope = clean.add_mutually_exclusive_group()
    clean_scope.add_argument("--task-id", help="Clean one terminal task, regardless of its age.")
    clean_scope.add_argument("--group", help="Clean terminal tasks in one group subject to retention and limit.")
    clean.add_argument("--older-than-days", type=int, default=30,
                       help="Minimum task age for bulk cleanup (default: 30).")
    clean.add_argument("--limit", type=int, default=100,
                       help="Maximum number of bulk-cleanup candidates (default: 100).")
    clean.add_argument("--dry-run", action="store_true", help="Show candidates without cleaning them.")
    use = commands.add_parser(
        "use",
        help="Save local CLI context without initializing or registering a project.",
        description=(
            "Save local default CLI context for an existing shared root and machine name. "
            "This command does not initialize a shared root, create a Project machine record, "
            "or register the project with the local machine agent."
        ),
    )
    use.add_argument(
        "--shared-root", dest="use_shared_root", help="Shared project control root to save."
    )
    use.add_argument(
        "--machine", dest="use_machine", help="Local machine name to save as context."
    )
    use.add_argument(
        "--runtime-root", dest="use_runtime_root", help="Legacy runtime root to save."
    )
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


def _duration_seconds(value: str) -> int:
    matched = re.fullmatch(r"([0-9]+)([smh])", value)
    if not matched:
        raise ValueError("duration must use an explicit s, m, or h unit, for example 10m.")
    amount = int(matched.group(1))
    return amount * {"s": 1, "m": 60, "h": 3600}[matched.group(2)]


def _gpu_limit_gpus(value: str) -> int | str:
    if value == "unlimited":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpu-limit-gpus must be a positive integer or 'unlimited'."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "gpu-limit-gpus must be a positive integer or 'unlimited'."
        )
    return parsed


def _split_machine_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    machines = [machine.strip() for value in values for machine in value.split(",")]
    if not all(machines):
        raise ValueError("--with machine names must be comma-separated non-empty values.")
    return machines


def _try_save_context(shared_root: str, machine: str, runtime_root: str | None) -> None:
    try:
        save_context(shared_root, machine, runtime_root)
    except OSError as exc:
        print(
            "qexp: initialized successfully, but failed to save CLI context "
            f"at {exc.filename or '~/.qqtools/qexp-context.json'}: {exc}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            if not args.shared_root or not args.machine:
                raise ValueError("init requires --shared-root and --machine.")
            cfg = init_shared_root(
                Path(args.shared_root),
                args.machine,
                agent_mode=args.agent_mode,
                runtime_root=Path(args.runtime_root) if args.runtime_root else None,
            )
            try:
                register_project(MachineRuntime(args.machine_runtime_root), cfg.shared_root, cfg.machine_name)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "qexp project initialized but was not registered with the machine agent; "
                    f"resolve the registration error and run 'qexp agent add-project': {exc}"
                ) from exc
            _try_save_context(str(cfg.shared_root), cfg.machine_name, str(cfg.runtime_root))
            print(cfg.shared_root)
            return 0
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
        if args.command == "use":
            if args.clear:
                clear_context()
                return 0
            if args.show:
                _emit("context", load_context() or {}, args.format or "human")
                return 0
            if args.format:
                raise ValueError("--format requires --show.")
            if not args.use_shared_root or not args.use_machine:
                raise ValueError("use requires --shared-root and --machine.")
            save_context(args.use_shared_root, args.use_machine, args.use_runtime_root)
            return 0
        if args.command == "agent" and args.agent_action in {
            "status", "list-projects", "disable-project", "remove-project", "stop", "restart"
        }:
            runtime = MachineRuntime(args.machine_runtime_root)
            if args.agent_action == "status":
                _emit("agent", {"action": "status", **get_machine_agent_status(runtime)}, args.format)
            elif args.agent_action == "list-projects":
                _emit("agent", {"action": "project_list", "projects": get_machine_agent_status(runtime)["projects"]}, args.format)
            elif args.agent_action == "disable-project":
                binding = set_project_enabled(runtime, args.project, False)
                _emit("agent", {"action": "project_disabled", **binding.to_dict()}, args.format)
            elif args.agent_action == "remove-project":
                binding = unregister_project(runtime, args.project)
                _emit("agent", {"action": "project_removed", **binding.to_dict()}, args.format)
            elif args.agent_action == "stop":
                stopped = stop_machine_agent(runtime)
                _emit("agent", {"action": "stopped" if stopped else "already_stopped", **get_machine_agent_status(runtime)}, args.format)
            else:
                process = restart_machine_agent(runtime)
                _emit(
                    "agent",
                    {
                        "action": "restarted",
                        **get_machine_agent_status(runtime),
                        "pid": process.pid,
                        "previous_pid": getattr(process, "previous_pid", None),
                    },
                    args.format,
                )
            return 0
        cfg, execution_context = _resolve_cfg(
            args, require_binding=_requires_verified_binding(args)
        )

        def get_execution_context() -> ExecutionContext:
            return execution_context

        def get_lifecycle_kwargs() -> dict[str, MachineRuntime]:
            return {"machine_runtime": execution_context.machine_runtime}

        if args.command == "config":
            if args.config_action != "notifications":
                raise ValueError("unknown config action")
            if args.notifications_action == "show":
                _emit("notifications", load_notifications(cfg), args.format)
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
                _emit("notifications", value, args.format)
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
                _emit("notifications", value, args.format)
                return 0
        if args.command == "lease-policy":
            current = load_lease_policy(cfg)
            if args.lease_policy_action == "show":
                _emit("lease-policy", {"lease_policy": asdict(current)}, args.format)
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
            _emit("lease-policy", {"lease_policy": values}, args.format)
            return 0
        if args.command == "submit":
            if not args.no_activate:
                ensure_local_agent_active(cfg, reason="submit", **get_lifecycle_kwargs())
            task_value = task_commands.submit(
                cfg,
                _command(args.argv),
                requested_gpus=args.gpus,
                task_id=args.task_id,
                name=args.name,
                group=args.group,
                working_dir=args.cwd,
                home_machine=args.home_machine,
                sharing_mode=args.sharing,
                offer_after_seconds=args.offer_after_seconds,
                idempotency_key=args.idempotency_key,
            )
            print(task_value.task_id)
            return 0
        if args.command == "batch-submit":

            ensure_local_agent_active(cfg, reason="batch-submit", **get_lifecycle_kwargs())

            def print_prepared(operation_id: str, idempotency_key: str) -> None:
                print(
                    f"qexp: prepared operation_id={operation_id} idempotency_key={idempotency_key}",
                    file=sys.stderr,
                    flush=True,
                )

            values = task_commands.batch_submit(
                cfg,
                Path(args.manifest_file),
                group=args.group,
                idempotency_key=args.idempotency_key,
                on_prepared=print_prepared,
            )
            _emit("batch-submit", values.to_dict(), args.format)
            return 0
        if args.command == "task":
            if args.task_action == "cancel":
                context = get_execution_context()
                task_value = task_commands.cancel(
                    cfg, args.task_id, reservation_runtime_root=context.reservation_root
                )
                claim = task_value.claim_control.get("active_claim") or {}
                is_pending = bool(
                    task_value.state["projection"] == "running"
                    and task_value.control.get("terminate_running")
                    and not task_value.control.get("termination_acknowledged_at")
                )
                _emit(
                    "task-operation",
                    {
                        "task_id": task_value.task_id,
                        "task_state": task_value.state["projection"],
                        "owning_machine": claim.get("machine_name") or task_value.placement_policy["home_machine"],
                        "operation_state": "waiting_ack" if is_pending else "completed",
                        "pending_acknowledgement": is_pending,
                        "termination_acknowledged_at": task_value.control.get("termination_acknowledged_at"),
                    },
                    args.format,
                )
            elif args.task_action == "retry":
                ensure_local_agent_active(
                    cfg, reason="task-retry", **get_lifecycle_kwargs()
                )
                task_value = task_commands.retry(
                    cfg,
                    args.task_id,
                    acknowledge_duplicate_risk=args.acknowledge_duplicate_risk,
                )
                print(task_value.task_id)
            elif args.task_action == "offer":
                ensure_local_agent_active(
                    cfg, reason="task-offer", **get_lifecycle_kwargs()
                )
                result = task_commands.offer(cfg, args.task_id)
                _emit("availability", result.to_dict(), args.format)
            elif args.task_action == "share":
                after_seconds = _duration_seconds(args.after) if args.after is not None else None
                ensure_local_agent_active(
                    cfg, reason="task-share", **get_lifecycle_kwargs()
                )
                result = task_commands.share(
                    cfg,
                    args.task_id,
                    after_seconds=after_seconds,
                    helper_machines=_split_machine_list(args.helper_machines),
                )
                _emit("availability", result.to_dict(), args.format)
            elif args.task_action == "keep-local":
                ensure_local_agent_active(
                    cfg, reason="task-keep-local", **get_lifecycle_kwargs()
                )
                result = task_commands.keep_local(cfg, args.task_id)
                _emit("availability", result.to_dict(), args.format)
            elif args.task_action == "list":
                _emit(
                    "task-list",
                    observer.list_tasks(cfg, phase=args.phase, group=args.group, limit=args.limit),
                    args.format,
                )
            elif args.task_action == "show":
                _emit("task-show", observer.inspect_task(cfg, args.task_id), args.format)
            elif args.task_action == "logs":
                print(log_commands.read_logs(cfg, args.task_id), end="")
            return 0
        if args.command == "group":
            presentation = None
            if args.group_action == "create":
                result = group_commands.create_group(cfg, args.name, args.workers)
                kind = "group-operation"
                presentation = {**result, "action": "create"}
            elif args.group_action == "list":
                result = observer.list_groups(cfg)
                kind = "group-list"
            elif args.group_action == "show":
                result = group_commands.show_group(cfg, args.name)
                kind = "group-show"
            elif args.group_action == "retry-failed":
                ensure_local_agent_active(
                    cfg, reason="group-retry-failed", **get_lifecycle_kwargs()
                )
                result = {
                    "task_ids": [task_value.task_id for task_value in group_commands.group_retry_failed(cfg, args.name)]
                }
                kind = "group-operation"
                presentation = {**result, "action": "retry-failed", "name": args.name, "status": "completed"}
            elif args.group_action == "machines":
                if args.machine_action == "list":
                    result = observer.list_group_machines(
                        cfg,
                        args.group_name,
                        reservation_runtime_root=execution_context.reservation_root,
                    )
                    _emit("group-machines", result, args.format)
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
                kind = "group-operation"
                presentation = {
                    **result,
                    "action": args.machine_action,
                    "worker_machine": args.worker_machine,
                }
            else:
                context = get_execution_context()
                if args.group_action == "resume":
                    ensure_local_agent_active(
                        cfg, reason="group-resume", **get_lifecycle_kwargs()
                    )
                result = group_commands.group_control(
                    cfg,
                    args.name,
                    args.group_action,
                    terminate_running=getattr(args, "terminate_running", False),
                    reservation_runtime_root=context.reservation_root,
                )
                kind = "group-operation"
                presentation = {**result, "action": args.group_action}
            task_views = observer.list_tasks(cfg, limit=10**9) if args.format == "human" else None
            _emit(
                kind,
                presentation if args.format == "human" and presentation is not None else result,
                args.format,
                tasks=task_views,
            )
            return 0
        if args.command == "agent":
            runtime = get_execution_context().machine_runtime
            if args.agent_action == "add-project":
                registration = register_project(runtime, cfg.shared_root, cfg.machine_name)
                action = "project_added" if registration.is_added else "project_already_registered"
                _emit("agent", {"action": action, **registration.binding.to_dict()}, args.format)
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
                    "agent",
                    {"action": "project_migrated", **binding.to_dict(), **status, "migration_candidates": siblings},
                    args.format,
                )
                return 0
            if args.agent_action == "list-projects":
                _emit("agent", {"action": "project_list", "projects": get_machine_agent_status(runtime)["projects"]}, args.format)
                return 0
            if args.agent_action == "disable-project":
                binding = set_project_enabled(runtime, args.project, False)
                _emit("agent", {"action": "project_disabled", **binding.to_dict()}, args.format)
                return 0
            if args.agent_action == "remove-project":
                binding = unregister_project(runtime, args.project)
                _emit("agent", {"action": "project_removed", **binding.to_dict()}, args.format)
                return 0
            if args.agent_action == "status":
                _emit("agent", {"action": "status", **get_machine_agent_status(runtime)}, args.format)
                return 0
            if args.agent_action == "start":
                action, status = start_local_agent(cfg, reason="manual_start", require_eligible_work=False, machine_runtime=runtime)
                _emit("agent", {"action": action, **status}, args.format)
            elif args.agent_action == "run":
                run_local_agent_foreground(
                    cfg,
                    reason="manual_run",
                    on_started=lambda status: _emit("agent", {"action": "running", **status}, args.format, flush=True),
                    machine_runtime=runtime,
                )
            elif args.agent_action == "restart":
                action, status = restart_local_agent(cfg, machine_runtime=runtime)
                _emit("agent", {"action": action, **status}, args.format)
            elif args.agent_action == "stop":
                action, status = stop_local_agent(cfg, machine_runtime=runtime)
                _emit("agent", {"action": action, **status}, args.format)
            return 0
        if args.command == "top":
            result = observer.top_view(cfg, all_machines=True)
            _emit("top", result, args.format, tasks=result["tasks"] if args.format == "human" else None)
            return 0
        if args.command == "machines":
            result = observer.list_machines(cfg)
            task_views = observer.list_tasks(cfg, limit=10**9) if args.format == "human" else None
            _emit("machines", result, args.format, tasks=task_views)
            return 0
        if args.command == "logs":
            print(log_commands.read_logs(cfg, args.task_id), end="")
            return 0
        if args.command == "doctor":
            context = get_execution_context()
            result = (
                verify_integrity(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                    project_id=context.project_id,
                )
                if args.action == "verify"
                else repair_metadata(
                    context.local_cfg,
                    reservation_runtime_root=context.reservation_root,
                )
            )
            _emit("doctor", result, args.format)
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
                reservation_runtime_root=context.reservation_root,
            )
            _emit("clean", result, args.format)
            return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"qexp: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
