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

from .activation import (ensure_local_agent_active, restart_local_agent, run_local_agent_foreground,
                         start_local_agent, stop_local_agent)
from .agent import get_agent_status
from .commands import cleanup, group as group_commands, logs as log_commands, task as task_commands
from .doctor import repair_metadata, resolve_verify_exit_code, verify_integrity
from .layout import clear_context, load_context, load_root_config, migrate_schema5_to_schema6, save_context
from .config_types import RootConfig
from .lease import LeasePolicy, load_lease_policy, save_lease_policy
from .runtime.paths import shared_paths
from .runtime.store import iter_json, read_json
from .machine_config import init_shared_root, load_machine_policy
from .notification_config import (DEFAULT_WEBHOOK_ENV, load_notifications,
                                  update_notifications, write_shared_feishu_webhook)
from . import observer


def _resolve_cfg(args: argparse.Namespace):
    shared = getattr(args, "shared_root", None) or os.environ.get("QEXP_SHARED_ROOT")
    machine = getattr(args, "machine", None) or os.environ.get("QEXP_MACHINE")
    runtime = getattr(args, "runtime_root", None) or os.environ.get("QEXP_RUNTIME_ROOT")
    context = load_context() if not shared or not machine else None
    if context:
        shared = shared or context.get("shared_root")
        machine = machine or context.get("machine")
        runtime = runtime or context.get("runtime_root")
    if not shared or not machine:
        raise ValueError("--shared-root and --machine are required or must be saved with qexp use.")
    return load_root_config(shared, machine, runtime, require_initialized=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp schema-6 experiment queue")
    parser.add_argument("--shared-root")
    parser.add_argument("--machine")
    parser.add_argument("--runtime-root")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--agent-mode", choices=["on_demand", "daemon"], default="on_demand")
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--to-schema", type=int, required=True)
    lease_policy = commands.add_parser("lease-policy")
    lease_policy_sub = lease_policy.add_subparsers(dest="lease_policy_action", required=True)
    lease_policy_sub.add_parser("show")
    policy_set = lease_policy_sub.add_parser("set")
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
    notifications_sub.add_parser("show")
    notifications_set = notifications_sub.add_parser("set")
    notifications_set.add_argument("--enabled", action="store_true")
    notifications_set.add_argument("--disabled", action="store_true")
    provider = notifications_sub.add_parser("provider")
    provider_sub = provider.add_subparsers(dest="provider_action", required=True)
    provider_set = provider_sub.add_parser("set")
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
    submit = commands.add_parser("submit")
    for action in (submit,):
        action.add_argument("--task-id"); action.add_argument("--name"); action.add_argument("--group")
        action.add_argument("--gpus", type=int, default=1); action.add_argument("--cwd")
        action.add_argument("--sharing", choices=["private", "spillover"], default="private")
        action.add_argument("--offer-after-seconds", type=int); action.add_argument("--idempotency-key")
        action.add_argument("argv", nargs=argparse.REMAINDER)
    bulk = commands.add_parser("batch-submit")
    bulk.add_argument("--file", required=True, dest="manifest_file"); bulk.add_argument("--group")
    bulk.add_argument("--idempotency-key")
    task = commands.add_parser("task")
    task_sub = task.add_subparsers(dest="task_action", required=True)
    cancel = task_sub.add_parser("cancel"); cancel.add_argument("task_id")
    retry = task_sub.add_parser("retry"); retry.add_argument("task_id")
    retry.add_argument("--acknowledge-duplicate-risk", action="store_true")
    offer = task_sub.add_parser("offer"); offer.add_argument("task_id")
    offer.add_argument("--format", choices=["text", "json"], default="text")
    share = task_sub.add_parser("share"); share.add_argument("task_id")
    share.add_argument("--after"); share.add_argument("--with", dest="helper_machines", action="append")
    share.add_argument("--format", choices=["text", "json"], default="text")
    keep_local = task_sub.add_parser("keep-local"); keep_local.add_argument("task_id")
    keep_local.add_argument("--format", choices=["text", "json"], default="text")
    listing = task_sub.add_parser("list"); listing.add_argument("--phase"); listing.add_argument("--group"); listing.add_argument("--limit", type=int, default=50)
    show = task_sub.add_parser("show"); show.add_argument("task_id")
    logs = task_sub.add_parser("logs"); logs.add_argument("task_id")
    group = commands.add_parser("group")
    group_sub = group.add_subparsers(dest="group_action", required=True)
    create = group_sub.add_parser("create"); create.add_argument("name"); create.add_argument("--workers", nargs="*", default=[])
    for action in (group_sub.add_parser("list"),):
        pass
    show_group = group_sub.add_parser("show"); show_group.add_argument("name")
    for name in ("seal", "reopen", "pause", "resume", "cancel", "retry-failed"):
        action = group_sub.add_parser(name); action.add_argument("name")
        if name == "cancel": action.add_argument("--terminate-running", action="store_true")
    machines = group_sub.add_parser("machines")
    machines_sub = machines.add_subparsers(dest="machine_action", required=True)
    for name in ("add", "drain", "remove"):
        action = machines_sub.add_parser(name); action.add_argument("group_name"); action.add_argument("worker_machine")
        if name == "remove": action.add_argument("--terminate-running", action="store_true")
    agent = commands.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_action", required=True)
    agent_sub.add_parser("start"); agent_sub.add_parser("run")
    agent_sub.add_parser("restart"); agent_sub.add_parser("status"); agent_sub.add_parser("stop")
    commands.add_parser("top")
    commands.add_parser("machines")
    logs_top = commands.add_parser("logs"); logs_top.add_argument("task_id")
    doctor = commands.add_parser("doctor"); doctor.add_argument("action", choices=["verify", "repair"], default="verify", nargs="?"); doctor.add_argument("--strict", action="store_true")
    clean = commands.add_parser("clean")
    clean.add_argument("--task-id")
    clean.add_argument("--older-than-days", type=int, default=30)
    clean.add_argument("--limit", type=int, default=100)
    clean.add_argument("--dry-run", action="store_true")
    use = commands.add_parser("use"); use.add_argument("--shared-root", dest="use_shared_root"); use.add_argument("--machine", dest="use_machine"); use.add_argument("--runtime-root", dest="use_runtime_root"); use.add_argument("--show", action="store_true"); use.add_argument("--clear", action="store_true")
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


def _try_save_context(shared_root: str, machine: str, runtime_root: str | None) -> None:
    try:
        save_context(shared_root, machine, runtime_root)
    except OSError as exc:
        print(
            "qexp: initialized successfully, but failed to save CLI context "
            f"at {exc.filename or '~/.qqtools/qexp-context.json'}: {exc}",
            file=sys.stderr,
        )


def _print_availability_result(result: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.message)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            if not args.shared_root or not args.machine:
                raise ValueError("init requires --shared-root and --machine.")
            cfg = init_shared_root(Path(args.shared_root), args.machine, agent_mode=args.agent_mode,
                                   runtime_root=Path(args.runtime_root) if args.runtime_root else None)
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
            if args.clear: clear_context(); return 0
            if args.show: print(json.dumps(load_context() or {}, indent=2)); return 0
            if not args.use_shared_root or not args.use_machine: raise ValueError("use requires --shared-root and --machine.")
            save_context(args.use_shared_root, args.use_machine, args.use_runtime_root); return 0
        cfg = _resolve_cfg(args)
        if args.command == "config":
            if args.config_action != "notifications":
                raise ValueError("unknown config action")
            if args.notifications_action == "show":
                print(json.dumps(load_notifications(cfg), indent=2, sort_keys=True))
                return 0
            if args.notifications_action == "set":
                if args.enabled and args.disabled:
                    raise ValueError("--enabled and --disabled are mutually exclusive")
                if not args.enabled and not args.disabled:
                    raise ValueError("one of --enabled or --disabled is required")
                value = update_notifications(cfg, lambda current: {
                    **current, "enabled": args.enabled,
                })
                print(json.dumps(value, indent=2, sort_keys=True)); return 0
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
                    raise ValueError(
                        "--credential-source shared_file requires --acknowledge-shared-secret-risk"
                    )
                shared_webhook = None
                if args.webhook_stdin:
                    shared_webhook = sys.stdin.readline().rstrip("\r\n")
                    if not shared_webhook:
                        raise ValueError("--webhook-stdin requires a non-empty first input line")
                def update_provider(current):
                    providers = dict(current["providers"])
                    provider_value = dict(providers.get("feishu", {
                        "enabled": False, "webhook_env": DEFAULT_WEBHOOK_ENV,
                        "secret_env": None, "timeout_seconds": 5, "credential_source": "env",
                    }))
                    if args.enabled: provider_value["enabled"] = True
                    if args.disabled: provider_value["enabled"] = False
                    if args.credential_source is not None:
                        provider_value["credential_source"] = args.credential_source
                    if args.webhook_env is not None: provider_value["webhook_env"] = args.webhook_env
                    if args.secret_env is not None: provider_value["secret_env"] = args.secret_env
                    if args.unset_secret_env: provider_value["secret_env"] = None
                    if args.timeout_seconds is not None: provider_value["timeout_seconds"] = args.timeout_seconds
                    providers["feishu"] = provider_value
                    return {**current, "providers": providers}
                value = update_notifications(cfg, update_provider)
                if shared_webhook is not None:
                    write_shared_feishu_webhook(cfg, shared_webhook)
                print(json.dumps(value, indent=2, sort_keys=True)); return 0
        if args.command == "lease-policy":
            current = load_lease_policy(cfg)
            if args.lease_policy_action == "show":
                print(json.dumps({"lease_policy": asdict(current)}, indent=2))
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
            print(json.dumps({"lease_policy": values}, indent=2))
            return 0
        if args.command == "submit":
            task_value = task_commands.submit(cfg, _command(args.argv), requested_gpus=args.gpus, task_id=args.task_id, name=args.name,
                                              group=args.group, working_dir=args.cwd, sharing_mode=args.sharing,
                                              offer_after_seconds=args.offer_after_seconds, idempotency_key=args.idempotency_key)
            ensure_local_agent_active(cfg, reason="submit")
            print(task_value.task_id); return 0
        if args.command == "batch-submit":
            def print_prepared(operation_id: str, idempotency_key: str) -> None:
                print(json.dumps({"operation_id": operation_id, "idempotency_key": idempotency_key,
                                  "state": "prepared"}), flush=True)

            values = task_commands.batch_submit(cfg, Path(args.manifest_file), group=args.group,
                                                idempotency_key=args.idempotency_key, on_prepared=print_prepared)
            ensure_local_agent_active(cfg, reason="batch-submit")
            print(json.dumps({"task_ids": [value.task_id for value in values], "state": "committed"})); return 0
        if args.command == "task":
            if args.task_action == "cancel":
                task_value = task_commands.cancel(cfg, args.task_id)
                claim = task_value.claim_control.get("active_claim") or {}
                is_pending = bool(task_value.state["projection"] == "running"
                                  and task_value.control.get("terminate_running")
                                  and not task_value.control.get("termination_acknowledged_at"))
                print(json.dumps({"task_id": task_value.task_id,
                                  "task_state": task_value.state["projection"],
                                  "owning_machine": claim.get("machine_name")
                                  or task_value.placement_policy["home_machine"],
                                  "operation_state": "waiting_ack" if is_pending else "completed",
                                  "pending_acknowledgement": is_pending,
                                  "termination_acknowledged_at":
                                      task_value.control.get("termination_acknowledged_at")}))
            elif args.task_action == "retry":
                task_value = task_commands.retry(
                    cfg,
                    args.task_id,
                    acknowledge_duplicate_risk=args.acknowledge_duplicate_risk,
                )
                ensure_local_agent_active(cfg, reason="task-retry")
                print(task_value.task_id)
            elif args.task_action == "offer":
                result = task_commands.offer(cfg, args.task_id)
                ensure_local_agent_active(cfg, reason="task-offer")
                _print_availability_result(result, args.format)
            elif args.task_action == "share":
                after_seconds = _duration_seconds(args.after) if args.after is not None else None
                result = task_commands.share(
                    cfg, args.task_id, after_seconds=after_seconds, helper_machines=args.helper_machines
                )
                ensure_local_agent_active(cfg, reason="task-share")
                _print_availability_result(result, args.format)
            elif args.task_action == "keep-local":
                result = task_commands.keep_local(cfg, args.task_id)
                ensure_local_agent_active(cfg, reason="task-keep-local")
                _print_availability_result(result, args.format)
            elif args.task_action == "list": print(json.dumps(observer.list_tasks(cfg, phase=args.phase, group=args.group, limit=args.limit)))
            elif args.task_action == "show": print(json.dumps(observer.inspect_task(cfg, args.task_id), indent=2))
            elif args.task_action == "logs": print(log_commands.read_logs(cfg, args.task_id), end="")
            return 0
        if args.command == "group":
            if args.group_action == "create": result = group_commands.create_group(cfg, args.name, args.workers)
            elif args.group_action == "list": result = observer.list_groups(cfg)
            elif args.group_action == "show": result = group_commands.show_group(cfg, args.name)
            elif args.group_action == "retry-failed":
                result = {"task_ids": [task_value.task_id for task_value in group_commands.group_retry_failed(cfg, args.name)]}
                ensure_local_agent_active(cfg, reason="group-retry-failed")
            elif args.group_action == "machines": result = group_commands.change_worker(
                cfg, args.group_name, args.worker_machine, args.machine_action,
                terminate_running=getattr(args, "terminate_running", False))
            else:
                result = group_commands.group_control(cfg, args.name, args.group_action,
                                                      terminate_running=getattr(args, "terminate_running", False))
                if args.group_action == "resume":
                    ensure_local_agent_active(cfg, reason="group-resume")
            print(json.dumps(result)); return 0
        if args.command == "agent":
            policy = load_machine_policy(cfg)
            if args.agent_action == "status":
                print(json.dumps({"action": "status", "agent_mode": policy.agent_mode, **get_agent_status(cfg)}))
            elif args.agent_action == "start":
                action, status = start_local_agent(cfg, reason="manual_start", require_eligible_work=False)
                print(json.dumps({"action": action, "agent_mode": policy.agent_mode, **status}))
            elif args.agent_action == "run":
                run_local_agent_foreground(
                    cfg,
                    reason="manual_run",
                    on_started=lambda status: print(json.dumps({
                        "action": "running", "agent_mode": policy.agent_mode, **status,
                    }), flush=True),
                )
            elif args.agent_action == "restart":
                action, status = restart_local_agent(cfg)
                print(json.dumps({"action": action, "agent_mode": policy.agent_mode, **status}))
            elif args.agent_action == "stop":
                action, status = stop_local_agent(cfg)
                print(json.dumps({"action": action, "agent_mode": policy.agent_mode, **status}))
            return 0
        if args.command == "top": print(json.dumps(observer.top_view(cfg, all_machines=True))); return 0
        if args.command == "machines": print(json.dumps(observer.list_machines(cfg))); return 0
        if args.command == "logs": print(log_commands.read_logs(cfg, args.task_id), end=""); return 0
        if args.command == "doctor":
            result = verify_integrity(cfg) if args.action == "verify" else repair_metadata(cfg)
            print(json.dumps(result, indent=2)); return resolve_verify_exit_code(result, strict=args.strict)
        if args.command == "clean":
            result = cleanup.clean(cfg, task_id=args.task_id,
                                   older_than_days=args.older_than_days,
                                   limit=args.limit, dry_run=args.dry_run)
            print(json.dumps(result, indent=2)); return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"qexp: {exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
