from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import api, fsqueue, manager


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "submit":
        return handle_submit(args)
    if args.command == "daemon":
        return handle_daemon(args)
    if args.command == "cancel":
        return handle_cancel(args)
    if args.command == "logs":
        return handle_logs(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
