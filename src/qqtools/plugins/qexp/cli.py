from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import api, fsqueue, manager

TOP_REFRESH_INTERVAL_SECONDS = 1.0


def _normalize_root(root: str | None) -> Path | None:
    if root is None:
        return None
    return Path(root).expanduser().resolve()


def _parse_extra_env(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None

    extra_env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --env entry {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("Invalid --env entry with empty key.")
        extra_env[key] = value
    return extra_env


def _build_env_payload(args: argparse.Namespace) -> dict | None:
    extra_env = _parse_extra_env(args.env)
    if args.conda_name or args.conda_activate_script:
        if not args.conda_name or not args.conda_activate_script:
            raise ValueError("Conda mode requires both --conda-name and --conda-activate-script.")
        payload = {
            "kind": "conda",
            "name": args.conda_name,
            "activate_script": args.conda_activate_script,
        }
    elif args.venv_path:
        payload = {
            "kind": "venv",
            "path": args.venv_path,
        }
    elif extra_env:
        payload = {"kind": "none"}
    else:
        return None

    if extra_env:
        payload["extra_env"] = extra_env
    return payload


def _normalize_submit_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("qexp submit requires a task argv after '--'.")
    return normalized


def _wait_for_daemon_start(root: Path | None) -> None:
    wait_seconds = manager.DEFAULT_STARTUP_WAIT_SECONDS
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    if not manager.is_daemon_active(root):
        raise RuntimeError(
            "qexp daemon background startup failed. Run `qexp daemon` for debugging."
        )


def _tail_log_forever(log_path: Path) -> int:
    with log_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                continue
            time.sleep(0.5)


def _render_status_text(snapshot: dict) -> str:
    lines = [
        f"Daemon: {snapshot['daemon']['state']}",
        (
            "Pending: {pending}  Running: {running}  Done: {done}  Failed: {failed}  "
            "Cancelled: {cancelled}"
        ).format(**snapshot["counts"]),
        "",
    ]

    columns = [
        ("STATE", "state"),
        ("TASK_ID", "task_id"),
        ("NAME", "name"),
        ("GPUS", "gpus"),
        ("ASSIGNED", "assigned"),
        ("CREATED_AT", "created_at"),
        ("EXIT_REASON", "exit_reason"),
    ]
    rows = snapshot["tasks"]
    widths: list[int] = []
    for header, key in columns:
        cell_width = max((len(str(row[key])) for row in rows), default=0)
        widths.append(max(len(header), cell_width))

    header_line = "  ".join(header.ljust(width) for (header, _key), width in zip(columns, widths))
    lines.append(header_line)
    for row in rows:
        lines.append(
            "  ".join(str(row[key]).ljust(width) for (_header, key), width in zip(columns, widths))
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_gpu_slot(slot: dict) -> str:
    return "\n".join(
        [
            f"GPU {slot['gpu_id']}",
            slot["state"],
            slot["task_id"],
        ]
    )


def _create_top_console():
    from rich.console import Console

    return Console()


def _render_top(snapshot: dict, *, console=None) -> None:
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table

    if console is None:
        console = _create_top_console()
    daemon = snapshot["daemon"]
    daemon_table = Table.grid(padding=(0, 1))
    daemon_table.add_row("Daemon", daemon["state"])
    daemon_table.add_row("Host", snapshot["host"]["hostname"] or "-")
    daemon_table.add_row("Platform", snapshot["host"]["platform"] or "-")
    daemon_table.add_row("GPU Backend", snapshot["gpus"]["backend"] or daemon["gpu_backend"] or "-")
    daemon_table.add_row("Visible GPUs", str(len(snapshot["gpus"]["visible_gpu_ids"])))

    gpu_panels = []
    slots = list(snapshot["gpus"]["slots"])
    visible_count = max(8, len(slots))
    for index in range(visible_count):
        slot = slots[index] if index < len(slots) else {"gpu_id": "-", "state": "-", "task_id": "-"}
        style = "red" if slot["state"] == "Reserved" else "green"
        if slot["state"] == "-":
            style = "dim"
        gpu_panels.append(Panel(_render_gpu_slot(slot), border_style=style))

    pending_table = Table(show_header=True)
    pending_table.add_column("TASK_ID")
    pending_table.add_column("NAME")
    pending_table.add_column("GPUS")
    pending_table.add_column("CREATED_AT")
    for row in snapshot["pending_preview"]:
        pending_table.add_row(row["task_id"], row["name"], str(row["gpus"]), row["created_at"])

    event_table = Table(show_header=True)
    event_table.add_column("TIMESTAMP")
    event_table.add_column("TASK_ID")
    event_table.add_column("STATE")
    event_table.add_column("EXIT_REASON")
    for event in snapshot["events"]:
        event_table.add_row(
            event["timestamp"],
            event["task_id"],
            event["state"],
            event["exit_reason"],
        )

    console.print(Panel(daemon_table, title="Daemon And Host"))
    console.print(Panel(Columns(gpu_panels, equal=True, expand=True), title="GPU Occupancy"))
    console.print(Panel(pending_table, title="Pending Preview"))
    console.print(Panel(event_table, title="Recent Events"))


def _run_top_monitor(*, root: Path | None, refresh_interval: float = TOP_REFRESH_INTERVAL_SECONDS) -> int:
    console = _create_top_console()
    try:
        while True:
            console.clear()
            _render_top(api.get_status_snapshot(root=root), console=console)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        return 0


def _render_clean_summary(result: dict) -> str:
    mode = "Dry run" if result["dry_run"] else "Deleted"
    lines = [
        f"{mode}: {len(result['task_ids'])} task files, {len(result['deleted_log_files'])} log files",
    ]
    for path in result["deleted_task_files"]:
        lines.append(path)
    for path in result["deleted_log_files"]:
        lines.append(path)
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp local experiment scheduler")
    parser.add_argument("--root", type=str, default=None, help="override qexp state root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="queue a task")
    submit_parser.add_argument("--num-gpus", type=int, required=True)
    submit_parser.add_argument("--job-id", type=str, default=None)
    submit_parser.add_argument("--job-name", type=str, default=None)
    submit_parser.add_argument("--workdir", type=str, default=None)
    submit_parser.add_argument("--conda-name", type=str, default=None)
    submit_parser.add_argument("--conda-activate-script", type=str, default=None)
    submit_parser.add_argument("--venv-path", type=str, default=None)
    submit_parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra environment variable to inject into task runtime",
    )
    submit_parser.add_argument("argv", nargs=argparse.REMAINDER)

    daemon_parser = subparsers.add_parser("daemon", help="run or start the qexp daemon")
    daemon_parser.add_argument("--background", action="store_true")
    daemon_parser.add_argument("--foreground", action="store_true")

    cancel_parser = subparsers.add_parser("cancel", help="cancel a queued or running task")
    cancel_parser.add_argument("task_id", type=str)

    logs_parser = subparsers.add_parser("logs", help="show persisted logs for a task")
    logs_parser.add_argument("task_id", type=str)
    logs_parser.add_argument("-f", "--follow", action="store_true")

    status_parser = subparsers.add_parser("status", help="show queue and daemon status")
    status_parser.add_argument("--json", action="store_true", dest="json_output")

    top_parser = subparsers.add_parser("top", help="show live operator dashboard")
    top_parser.add_argument("--once", action="store_true", help="print one snapshot and exit")

    clean_parser = subparsers.add_parser("clean", help="clean completed qexp records")
    clean_parser.add_argument("--dry-run", action="store_true")
    clean_parser.add_argument("--include-failed", action="store_true")
    clean_parser.add_argument(
        "--older-than-seconds",
        type=int,
        default=api.DEFAULT_CLEAN_OLDER_THAN_SECONDS,
    )

    return parser


def handle_submit(args: argparse.Namespace) -> int:
    task = api.submit(
        argv=_normalize_submit_argv(args.argv),
        num_gpus=args.num_gpus,
        job_id=args.job_id,
        job_name=args.job_name,
        workdir=args.workdir,
        env=_build_env_payload(args),
        root=_normalize_root(args.root),
    )
    print(task.task_id)
    return 0


def handle_daemon(args: argparse.Namespace) -> int:
    root = _normalize_root(args.root)
    if args.background and args.foreground:
        raise ValueError("Choose either --background or --foreground, not both.")

    if args.background:
        manager.run_preflight_checks()
        manager.start_daemon_background(root)
        _wait_for_daemon_start(root)
        print("qexp daemon started in background")
        return 0

    return manager.run_daemon_foreground(root=root)


def handle_cancel(args: argparse.Namespace) -> int:
    task = api.cancel(args.task_id, root=_normalize_root(args.root))
    print(f"{task.task_id} {task.status}")
    return 0


def handle_logs(args: argparse.Namespace) -> int:
    root = _normalize_root(args.root)
    log_path = fsqueue.get_log_path(args.task_id, root=root)
    if not log_path.exists():
        raise FileNotFoundError(f"qexp log for task '{args.task_id}' does not exist.")

    if args.follow:
        return _tail_log_forever(log_path)

    sys.stdout.write(api.read_logs(args.task_id, root=root))
    return 0


def handle_status(args: argparse.Namespace) -> int:
    snapshot = api.get_status_snapshot(root=_normalize_root(args.root))
    if args.json_output:
        sys.stdout.write(json.dumps(snapshot, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(_render_status_text(snapshot))
    return 0


def handle_top(args: argparse.Namespace) -> int:
    root = _normalize_root(args.root)
    if args.once:
        _render_top(api.get_status_snapshot(root=root))
        return 0
    return _run_top_monitor(root=root)


def handle_clean(args: argparse.Namespace) -> int:
    result = api.clean(
        root=_normalize_root(args.root),
        dry_run=args.dry_run,
        include_failed=args.include_failed,
        older_than_seconds=args.older_than_seconds,
    )
    sys.stdout.write(_render_clean_summary(result))
    return 0


def _should_use_v1(argv: list[str] | None) -> bool:
    """Check if the user explicitly requested v1 via --v1 flag or QEXP_VERSION=1."""
    import os
    if os.environ.get("QEXP_VERSION") == "1":
        return True
    if argv:
        try:
            sep = argv.index("--")
            prefix = argv[:sep]
        except ValueError:
            prefix = argv
        if "--v1" in prefix:
            return True
    return False


def _strip_v1_flag(argv: list[str]) -> list[str]:
    """Remove the --v1 flag only from the portion before '--'."""
    try:
        sep = argv.index("--")
        prefix = [a for a in argv[:sep] if a != "--v1"]
        return prefix + argv[sep:]
    except ValueError:
        return [a for a in argv if a != "--v1"]


# Keep old names for backward compatibility during transition
def _should_use_v2(argv: list[str] | None) -> bool:
    return not _should_use_v1(argv)


def main(argv: list[str] | None = None) -> int:
    # v2 is the default since 1.2.7. Use --v1 or QEXP_VERSION=1 to
    # fall back to the legacy single-machine engine.
    # v1 will be removed in 1.3.0.
    if not _should_use_v1(argv):
        from .v2.cli import main as v2_main
        return v2_main(argv)

    cleaned = _strip_v1_flag(argv or sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(cleaned)

    if args.command == "submit":
        return handle_submit(args)
    if args.command == "daemon":
        return handle_daemon(args)
    if args.command == "cancel":
        return handle_cancel(args)
    if args.command == "logs":
        return handle_logs(args)
    if args.command == "status":
        return handle_status(args)
    if args.command == "top":
        return handle_top(args)
    if args.command == "clean":
        return handle_clean(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
