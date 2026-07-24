"""Background qexp agent process entrypoint."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .agent import run_agent_loop
from .layout import load_root_config, runtime_pid_path
from .machine_config import load_machine_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp agent process")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runtime-root")
    return parser


def spawn_agent_process(cfg, *, stdin=None, stdout=None, stderr=None):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qqtools.plugins.qexp.agent_process",
            "--shared-root",
            str(cfg.shared_root),
            "--machine",
            cfg.machine_name,
            "--runtime-root",
            str(cfg.runtime_root),
        ],
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL if stdin is None else stdin,
        stdout=subprocess.DEVNULL if stdout is None else stdout,
        stderr=subprocess.DEVNULL if stderr is None else stderr,
        start_new_session=True,
    )
    pid_path = runtime_pid_path(cfg)
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(process.pid), encoding="utf-8")
    except OSError:
        process.terminate()
        raise
    return process


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_root_config(
        args.shared_root,
        args.machine,
        args.runtime_root,
        require_initialized=True,
    )
    policy = load_machine_policy(cfg)
    run_agent_loop(cfg, exit_when_idle=policy.exit_when_idle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
