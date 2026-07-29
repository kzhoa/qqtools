"""qexp command line routing for the schema-5 product contract."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .activation import (ensure_local_agent_active, restart_local_agent, run_local_agent_foreground,
                         start_local_agent, stop_local_agent)
from .agent import get_agent_status
from .commands import cleanup, group as group_commands, logs as log_commands, task as task_commands
from .doctor import repair_metadata, resolve_verify_exit_code, verify_integrity
from .layout import clear_context, load_context, load_root_config, save_context
from .machine_config import init_shared_root, load_machine_policy
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
    parser = argparse.ArgumentParser(description="qexp schema-5 experiment queue")
    parser.add_argument("--shared-root")
    parser.add_argument("--machine")
    parser.add_argument("--runtime-root")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--agent-mode", choices=["on_demand", "daemon"], default="on_demand")
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
            cfg = init_shared_root(Path(args.shared_root), args.machine, agent_mode=args.agent_mode,
                                   runtime_root=Path(args.runtime_root) if args.runtime_root else None)
            _try_save_context(str(cfg.shared_root), cfg.machine_name, str(cfg.runtime_root))
            print(cfg.shared_root)
            return 0
        if args.command == "use":
            if args.clear: clear_context(); return 0
            if args.show: print(json.dumps(load_context() or {}, indent=2)); return 0
            if not args.use_shared_root or not args.use_machine: raise ValueError("use requires --shared-root and --machine.")
            save_context(args.use_shared_root, args.use_machine, args.use_runtime_root); return 0
        cfg = _resolve_cfg(args)
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
                task_value = task_commands.offer(cfg, args.task_id)
                ensure_local_agent_active(cfg, reason="task-offer")
                print(task_value.task_id)
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
