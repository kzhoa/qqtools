"""Background entrypoint for the qexp machine agent."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .machine_agent import run_machine_agent_loop
from .machine_runtime import MachineRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp machine agent process")
    parser.add_argument("--machine-runtime-root", required=True)
    parser.add_argument("--loop-interval", type=float, default=5.0)
    return parser


def spawn_machine_agent_process(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    stdin=None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.ensure_layout()
    startup_stderr = subprocess.PIPE if stderr is None else stderr
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qqtools.plugins.qexp.machine_agent_process",
            "--machine-runtime-root",
            str(machine_runtime.root),
        ],
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL if stdin is None else stdin,
        stdout=subprocess.DEVNULL if stdout is None else stdout,
        stderr=startup_stderr,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status_path = machine_runtime.paths["agent"] / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8")).get("machine_agent", {})
            except (OSError, ValueError):
                status = {}
            if status.get("state") == "active" and status.get("pid") == process.pid:
                return process
        exit_code = process.poll()
        if exit_code is not None:
            details = ""
            if process.stderr is not None:
                details = process.stderr.read().strip()
            message = f"machine agent exited during startup with exit code {exit_code}."
            raise RuntimeError(f"{message} {details}".strip())
        time.sleep(0.02)
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise RuntimeError("machine agent did not acquire scheduler authority within 5 seconds.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_machine_agent_loop(args.machine_runtime_root, loop_interval=args.loop_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
