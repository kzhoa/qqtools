from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import api, observer
from .agent import get_agent_status, run_agent_loop
from .doctor import (
    build_verify_jsonl_records,
    cleanup_stale_locks,
    normalize_verify_severity,
    repair_metadata,
    rebuild_indexes,
    repair_orphans,
    resolve_verify_exit_code,
    verify_integrity,
)
from .layout import (
    clear_context,
    init_shared_root,
    load_context,
    load_root_config,
    save_context,
)

TOP_REFRESH_INTERVAL_SECONDS = 1.0


def _resolve_cfg(args: argparse.Namespace):
    shared_root = getattr(args, "shared_root", None)
    machine = getattr(args, "machine", None)
    runtime_root = getattr(args, "runtime_root", None)

    if shared_root and machine:
        return load_root_config(
            shared_root,
            machine,
            runtime_root,
            require_initialized=True,
        )

    # Try to load from environment or defaults
    sr = shared_root or os.environ.get("QEXP_SHARED_ROOT")
    mn = machine or os.environ.get("QEXP_MACHINE")
    rr = runtime_root or os.environ.get("QEXP_RUNTIME_ROOT")

    # Fall back to saved context file
    if not sr or not mn:
        ctx = load_context()
        if ctx:
            sr = sr or ctx.get("shared_root")
            mn = mn or ctx.get("machine")
            rr = rr or ctx.get("runtime_root")

    if not sr or not mn:
        print(
            "Error: --shared-root and --machine are required "
            "(or set QEXP_SHARED_ROOT / QEXP_MACHINE, "
            "or run 'qexp use' to save defaults). "
            "shared_root must point to the project control root '.qexp'.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return load_root_config(sr, mn, rr, require_initialized=True)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "qexp — project-root control-plane experiment queue.\n\n"
            "Mental model:\n"
            "  - task: one concrete run\n"
            "  - group: long-lived grouping key; runtime projects it to one tmux session\n"
            "  - batch: one bulk submit operation; not the same thing as group\n\n"
            "High-frequency path:\n"
            "  qexp submit -- python train.py --config configs/a.yaml\n"
            "  qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4\n"
            "  qexp batch-submit --file runs.yaml"
        )
    )
    parser.add_argument("--shared-root", type=str, default=None)
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--runtime-root", type=str, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    # init
    init_p = sub.add_parser(
        "init",
        help="initialize or refresh machine registration and save CLI context",
        description=(
            "Initialize qexp for one machine. Safe to run multiple times: "
            "it ensures the layout exists, refreshes machine.json, and saves "
            "the current CLI context. shared_root must be the project "
            "control directory named '.qexp'."
        ),
    )
    init_p.add_argument("--agent-mode", choices=["on_demand", "persistent"], default="on_demand")

    # submit
    submit_p = sub.add_parser(
        "submit",
        help="submit one new task on the current machine",
        description=(
            "Submit one new task on the current machine.\n\n"
            "Use submit for the high-frequency path: one command, one task.\n"
            "Do not write YAML for a single task.\n\n"
            "About --group:\n"
            "  - group is a long-lived grouping key inside the project\n"
            "  - runtime projects group directly to one tmux session\n"
            "  - group is not batch\n"
            "  - group is not a scientific truth object; it is the tool-layer key you use\n"
            "    to keep related runs in one working context\n\n"
            "Examples:\n"
            "  qexp submit -- python train.py --config configs/a.yaml\n"
            "  qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4\n"
            "  qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6\n"
            "The last two commands create two different tasks that share one long-lived group\n"
            "and therefore land in the same tmux session."
        ),
    )
    submit_p.add_argument("--task-id", type=str, default=None)
    submit_p.add_argument("--name", type=str, default=None)
    submit_p.add_argument("--group", type=str, default=None)
    submit_p.add_argument("--gpus", type=int, default=1)
    submit_p.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Override the task working directory. Defaults to the current directory at submit time.",
    )
    submit_p.add_argument("argv", nargs=argparse.REMAINDER)

    # cancel
    cancel_p = sub.add_parser("cancel", help="cancel a task")
    cancel_p.add_argument("task_id", type=str)

    # retry
    retry_p = sub.add_parser("retry", help="retry a terminal task")
    retry_p.add_argument("task_id", type=str)
    retry_p.add_argument("--group", type=str, default=None)

    # resubmit
    resubmit_p = sub.add_parser("resubmit", help="replace a terminal task in place")
    resubmit_p.add_argument("task_id", type=str)
    resubmit_p.add_argument("--name", type=str, default=None)
    resubmit_p.add_argument("--group", type=str, default=None)
    resubmit_p.add_argument("--gpus", type=int, default=None)
    resubmit_p.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Override the task working directory. Defaults to the current directory at resubmit time.",
    )
    resubmit_p.add_argument("argv", nargs=argparse.REMAINDER)

    # batch-submit
    batch_p = sub.add_parser(
        "batch-submit",
        help="submit one batch manifest on the current machine",
        description=(
            "Submit a manifest that creates one batch and all of its member tasks.\n\n"
            "Use batch-submit only when you are submitting a set of tasks together.\n\n"
            "About batch vs group:\n"
            "  - batch = one bulk submit operation\n"
            "  - batch.group = the default long-lived group for tasks in this batch\n"
            "  - group still means long-lived working context and runtime projects it to a\n"
            "    tmux session\n"
            "  - if you submit another related set tomorrow, usually create a new batch and\n"
            "    reuse the same group\n\n"
            "Example manifest shape:\n"
            "  batch:\n"
            "    name: contract-compare-round1\n"
            "    group: contract_n_4and6\n"
            "  tasks:\n"
            "    - name: n4\n"
            "      command: [\"python\", \"train.py\", \"--n\", \"4\"]\n"
            "    - name: n6\n"
            "      command: [\"python\", \"train.py\", \"--n\", \"6\"]"
        ),
    )
    batch_p.add_argument("--file", type=str, required=True, dest="manifest_file")

    # batch-retry-failed
    brf_p = sub.add_parser("batch-retry-failed", help="retry failed tasks in batch")
    brf_p.add_argument("batch_id", type=str)

    # batch-retry-cancelled
    brc_p = sub.add_parser("batch-retry-cancelled", help="retry cancelled tasks in batch")
    brc_p.add_argument("batch_id", type=str)

    # list
    list_p = sub.add_parser("list", help="list tasks")
    list_p.add_argument("--phase", type=str, default=None)
    list_p.add_argument("--batch", type=str, default=None, dest="batch_id")
    list_p.add_argument("--group", type=str, default=None)
    list_p.add_argument("--limit", type=int, default=50)

    # inspect
    inspect_p = sub.add_parser("inspect", help="inspect a single task")
    inspect_p.add_argument("task_id", type=str)

    # top
    top_p = sub.add_parser("top", help="live queue overview")
    top_p.add_argument("--all", action="store_true", dest="all_machines")

    # batches
    sub.add_parser("batches", help="list batches")

    # batch (inspect)
    batch_i = sub.add_parser("batch", help="inspect a batch")
    batch_i.add_argument("batch_id", type=str)

    # machines
    sub.add_parser("machines", help="list registered machines")

    # logs
    logs_p = sub.add_parser("logs", help="show task log output")
    logs_p.add_argument("task_id", type=str)
    logs_p.add_argument("-f", "--follow", action="store_true")

    # clean
    clean_p = sub.add_parser("clean", help="clean old terminal task records")
    clean_p.add_argument("--dry-run", action="store_true")
    clean_p.add_argument("--task-id", type=str, default=None)
    clean_p.add_argument("--include-failed", action="store_true")
    clean_p.add_argument(
        "--older-than-seconds",
        type=int,
        default=None,
    )

    # agent
    agent_p = sub.add_parser("agent", help="agent management")
    agent_sub = agent_p.add_subparsers(dest="agent_command", required=True)
    agent_start = agent_sub.add_parser("start", help="start agent")
    agent_start.add_argument("--persistent", action="store_true")
    agent_start.add_argument("--background", action="store_true", help="start in tmux background")
    agent_sub.add_parser("stop", help="stop agent")
    agent_sub.add_parser("status", help="show agent status")

    # use (context management)
    use_p = sub.add_parser(
        "use",
        help="set, show, or clear saved CLI context",
        description=(
            "Manage the saved qexp CLI context used as a fallback after "
            "explicit flags and environment variables."
        ),
    )
    use_p.add_argument("--shared-root", type=str, default=None, dest="use_shared_root")
    use_p.add_argument("--machine", type=str, default=None, dest="use_machine")
    use_p.add_argument("--runtime-root", type=str, default=None, dest="use_runtime_root")
    use_p.add_argument("--show", action="store_true", help="show current context")
    use_p.add_argument("--clear", action="store_true", help="clear saved context")

    # doctor
    doctor_p = sub.add_parser("doctor", help="diagnostics and repair")
    doctor_sub = doctor_p.add_subparsers(dest="doctor_command")
    doctor_sub.add_parser("repair", help="converge unfinished metadata repair operations")
    doctor_sub.add_parser("rebuild-index", help="rebuild all derived indexes from truth")
    doctor_sub.add_parser("repair-orphans", help="repair orphaned tasks via task/agent/claim truth")
    doctor_sub.add_parser("cleanup-locks", help="clean stale locks")
    doctor_verify = doctor_sub.add_parser("verify", help="non-destructive integrity check")
    doctor_verify.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when verify finds any governed issue",
    )
    doctor_verify.add_argument(
        "--fail-on",
        choices=["low", "medium", "high"],
        default=None,
        help="exit non-zero when severity is at or above the threshold",
    )
    doctor_verify.add_argument(
        "--jsonl",
        action="store_true",
        help="emit machine-readable JSONL records instead of one formatted JSON document",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _normalize_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("submit requires a command after '--'.")
    return normalized


def handle_init(args: argparse.Namespace) -> int:
    sr = args.shared_root
    mn = args.machine
    if not sr or not mn:
        print(
            "Error: --shared-root and --machine are required for init. "
            "shared_root must be the project control root '.qexp'.",
            file=sys.stderr,
        )
        return 1
    cfg = init_shared_root(
        Path(sr), mn,
        agent_mode=args.agent_mode,
        runtime_root=Path(args.runtime_root) if args.runtime_root else None,
    )
    # Only persist runtime_root when explicitly passed via --runtime-root;
    # the default path is derivable at runtime and should not be hardcoded.
    save_context(
        str(cfg.shared_root),
        cfg.machine_name,
        str(cfg.runtime_root) if args.runtime_root else None,
    )
    print(
        f"Initialized project control root: shared_root={cfg.shared_root} "
        f"project_root={cfg.project_root} machine={cfg.machine_name} (context saved)"
    )
    return 0


def handle_use(args: argparse.Namespace) -> int:
    if args.clear:
        if clear_context():
            print("Context cleared.")
        else:
            print("No context to clear.")
        return 0
    if args.show:
        ctx = load_context()
        if ctx:
            print(json.dumps(ctx, indent=2, sort_keys=True))
        else:
            print("No context saved. Run 'qexp use --shared-root ... --machine ...' to set.")
        return 0
    # Accept flags from both 'qexp --shared-root X use' and 'qexp use --shared-root X'
    sr = getattr(args, "use_shared_root", None) or args.shared_root
    mn = getattr(args, "use_machine", None) or args.machine
    rr = getattr(args, "use_runtime_root", None) or args.runtime_root
    if not sr or not mn:
        print(
            "Error: --shared-root and --machine are required for 'qexp use'. "
            "shared_root must be the project control root '.qexp'.",
            file=sys.stderr,
        )
        return 1
    try:
        load_root_config(sr, mn, rr, require_initialized=True)
    except Exception as exc:
        print(f"Error: cannot save qexp context: {exc}", file=sys.stderr)
        return 1
    path = save_context(sr, mn, rr)
    print(f"Context saved to {path}: shared_root={sr} machine={mn}")
    return 0


def handle_submit(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    task = api.submit(
        cfg,
        command=_normalize_argv(args.argv),
        requested_gpus=args.gpus,
        task_id=args.task_id,
        name=args.name,
        group=args.group,
        working_dir=args.cwd,
    )
    print(task.task_id)
    return 0


def handle_cancel(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    task = api.cancel(cfg, args.task_id)
    print(f"{task.task_id} {task.status.phase}")
    return 0


def handle_retry(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    new_task = api.retry(cfg, args.task_id, group=args.group)
    print(f"{new_task.task_id} (retry of {new_task.lineage.retry_of})")
    return 0


def handle_resubmit(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    task = api.resubmit(
        cfg,
        args.task_id,
        command=_normalize_argv(args.argv),
        requested_gpus=args.gpus,
        name=args.name,
        group=args.group,
        working_dir=args.cwd,
    )
    print(task.task_id)
    return 0


def handle_batch_submit(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    batch = api.batch_submit(cfg, Path(args.manifest_file))
    print(f"batch={batch.batch_id} tasks={len(batch.task_ids)}")
    return 0


def handle_batch_retry_failed(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    new_tasks = api.batch_retry_failed(cfg, args.batch_id)
    print(f"Retried {len(new_tasks)} failed tasks.")
    return 0


def handle_batch_retry_cancelled(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    new_tasks = api.batch_retry_cancelled(cfg, args.batch_id)
    print(f"Retried {len(new_tasks)} cancelled tasks.")
    return 0


def handle_list(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    tasks = observer.list_tasks(
        cfg,
        phase=args.phase,
        batch_id=args.batch_id,
        group=args.group,
        limit=args.limit,
    )
    if not tasks:
        print("No tasks found.")
        return 0
    for t in tasks:
        print(
            f"{t['phase']:12s} {t['task_id']:14s} "
            f"{(t.get('group') or '-'):20s} "
            f"{(t.get('name') or '-'):20s} gpus={t['gpus']}"
        )
    return 0


def handle_inspect(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    data = observer.inspect_task(cfg, args.task_id)
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def handle_top(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)

    def _fmt_count(value):
        return "?" if value is None else str(value)

    try:
        while True:
            view = observer.top_view(cfg, all_machines=args.all_machines)
            sys.stdout.write("\033[2J\033[H")  # clear screen
            print(f"=== qexp top ({cfg.machine_name}) ===\n")
            counts = view["counts"]
            print(
                f"queued={counts.get('queued', 0)}  running={counts.get('running', 0)}  "
                f"succeeded={counts.get('succeeded', 0)}  failed={counts.get('failed', 0)}"
            )
            print()
            for m in view.get("machines", []):
                phase_counts = m.get("counts_by_phase", {})
                print(
                    "  "
                    f"machine={m['machine_name']}  "
                    f"agent={m['agent_state']}  "
                    f"gpu_status={m['gpu_status']}  "
                    f"visible={_fmt_count(m['gpu_visible_count'])}  "
                    f"reserved={_fmt_count(m['gpu_reserved_count'])}  "
                    f"free={_fmt_count(m['gpu_free_count'])}  "
                    f"queued={phase_counts.get('queued', 0)}  "
                    f"dispatching={phase_counts.get('dispatching', 0)}  "
                    f"starting={phase_counts.get('starting', 0)}  "
                    f"running={phase_counts.get('running', 0)}"
                )
            print()
            for e in view.get("recent_events", [])[:5]:
                print(f"  {e.get('timestamp', '')}  {e.get('event_type', '')}  {e.get('task_id', '')}")
            sys.stdout.flush()
            time.sleep(TOP_REFRESH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        return 0


def handle_batches(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    batches = observer.list_batches(cfg)
    if not batches:
        print("No batches found.")
        return 0
    for b in batches:
        print(
            f"{b['batch_id']:14s} {(b.get('group') or '-'):20s} "
            f"{b.get('name') or '-':20s} total={b['total']} failed={b['failed']}"
        )
    return 0


def handle_batch_inspect(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    data = observer.inspect_batch(cfg, args.batch_id)
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def handle_machines(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    machines = observer.list_machines(cfg)
    if not machines:
        print("No machines registered.")
        return 0

    def _fmt_count(value):
        return "?" if value is None else str(value)

    for m in machines:
        phase_counts = m.get("counts_by_phase", {})
        print(
            f"{m['machine_name']:14s} host={m.get('hostname') or '-':20s} "
            f"mode={m['agent_mode']}  state={m['agent_state']}  gpu_status={m['gpu_status']}  "
            f"visible={_fmt_count(m['gpu_visible_count'])}  "
            f"reserved={_fmt_count(m['gpu_reserved_count'])}  "
            f"free={_fmt_count(m['gpu_free_count'])}  "
            f"queued={phase_counts.get('queued', 0)}  "
            f"dispatching={phase_counts.get('dispatching', 0)}  "
            f"starting={phase_counts.get('starting', 0)}  "
            f"running={phase_counts.get('running', 0)}"
        )
    return 0


def handle_logs(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    if args.follow:
        try:
            api.tail_log(cfg, args.task_id)
        except KeyboardInterrupt:
            pass
        return 0
    sys.stdout.write(api.read_logs(cfg, args.task_id))
    return 0


def handle_clean(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    if args.task_id and args.include_failed:
        print(
            "Error: --task-id cannot be combined with --include-failed.",
            file=sys.stderr,
        )
        return 1
    if args.task_id and args.older_than_seconds is not None:
        print(
            "Error: --task-id cannot be combined with --older-than-seconds.",
            file=sys.stderr,
        )
        return 1
    result = api.clean(
        cfg,
        dry_run=args.dry_run,
        task_id=args.task_id,
        include_failed=args.include_failed,
        older_than_seconds=args.older_than_seconds,
    )
    mode = "Dry run" if result["dry_run"] else "Cleaned"
    task_count = (
        result["planned_task_count"] if result["dry_run"] else result["deleted_task_count"]
    )
    print(
        f"{mode}: mode={result['mode']} tasks={task_count} "
        f"logs_deleted={result['deleted_log_count']} "
        f"batches_repaired={len(result['repaired_batches'])}"
    )
    for tid in result["task_ids"]:
        print(f"  {tid}")
    for log_result in result.get("log_results", []):
        status = log_result.get("status")
        path = log_result.get("path") or "-"
        print(f"  log[{status}] {log_result['task_id']} {path}")
    return 0


def handle_agent(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    # 'daemon' without subcommand means foreground start (v1 compat)
    if not getattr(args, "agent_command", None):
        return run_agent_loop(cfg, persistent=getattr(args, "persistent", False))
    if args.agent_command == "start":
        if args.background:
            from .agent import wake_agent_if_needed
            ok = wake_agent_if_needed(cfg)
            if ok:
                print("Agent started in background.")
                return 0
            else:
                print("Failed to start agent in background.", file=sys.stderr)
                return 1
        return run_agent_loop(cfg, persistent=args.persistent)
    elif args.agent_command == "stop":
        from .agent import stop_agent_record, is_agent_running, read_agent_state
        state = read_agent_state(cfg)
        if state and state.get("pid") and is_agent_running(cfg):
            import signal as _sig
            try:
                os.kill(state["pid"], _sig.SIGTERM)
            except OSError:
                pass
        stop_agent_record(cfg, reason="manual_stop")
        print("Agent stop requested.")
        return 0
    elif args.agent_command == "status":
        status = get_agent_status(cfg)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    return 1


def handle_doctor(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args)
    cmd = args.doctor_command

    if cmd == "rebuild-index":
        stats = rebuild_indexes(cfg)
        print(json.dumps(stats, indent=2))
        return 0
    elif cmd == "repair":
        result = repair_metadata(cfg)
        print(json.dumps(result, indent=2))
        return 0
    elif cmd == "repair-orphans":
        orphaned = repair_orphans(cfg)
        print(f"Repaired {len(orphaned)} orphaned tasks.")
        for tid in orphaned:
            print(f"  {tid}")
        return 0
    elif cmd == "cleanup-locks":
        cleaned = cleanup_stale_locks(cfg)
        print(f"Cleaned {len(cleaned)} stale locks.")
        return 0
    elif cmd == "verify":
        result = verify_integrity(cfg)
        fail_on = normalize_verify_severity(args.fail_on) if args.fail_on else None
        if args.jsonl:
            for record in build_verify_jsonl_records(
                result,
                strict=args.strict,
                fail_on=fail_on,
            ):
                print(json.dumps(record, sort_keys=True))
        else:
            print(json.dumps(result, indent=2))
        return resolve_verify_exit_code(
            result,
            strict=args.strict,
            fail_on=fail_on,
        )
    else:
        # Default: run verify
        result = verify_integrity(cfg)
        print(json.dumps(result, indent=2))
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HANDLERS = {
    "init": handle_init,
    "use": handle_use,
    "submit": handle_submit,
    "cancel": handle_cancel,
    "retry": handle_retry,
    "resubmit": handle_resubmit,
    "batch-submit": handle_batch_submit,
    "batch-retry-failed": handle_batch_retry_failed,
    "batch-retry-cancelled": handle_batch_retry_cancelled,
    "list": handle_list,
    "inspect": handle_inspect,
    "top": handle_top,
    "batches": handle_batches,
    "batch": handle_batch_inspect,
    "machines": handle_machines,
    "logs": handle_logs,
    "clean": handle_clean,
    "agent": handle_agent,
    "doctor": handle_doctor,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
