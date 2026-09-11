"""Background entrypoint for the qexp machine agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .lifecycle import run_machine_agent_loop
from .context import MachineRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp machine agent process")
    parser.add_argument("--machine-runtime-root", required=True)
    parser.add_argument("--loop-interval", type=float, default=5.0)
    parser.add_argument("--available-gpus", default=None)
    return parser


def spawn_machine_agent_process(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.ensure_layout()
    startup_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8") if stderr is None else None
    command = [
        sys.executable,
        "-m",
        "qqtools.plugins.qexp.agent.process",
        "--machine-runtime-root",
        str(machine_runtime.root),
    ]
    if available_gpus is not None:
        command.extend(("--available-gpus", ",".join(str(item) for item in available_gpus)))
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[4])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL if stdin is None else stdin,
        stdout=subprocess.DEVNULL if stdout is None else stdout,
        stderr=startup_log if startup_log is not None else stderr,
        start_new_session=True,
    )
    try:
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
                if startup_log is not None:
                    startup_log.seek(0)
                    lines = startup_log.read().strip().splitlines()
                    details = f": {lines[-1]}" if lines else ""
                raise RuntimeError(f"machine agent exited during startup with exit code {exit_code}{details}.")
            time.sleep(0.02)
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise RuntimeError("machine agent did not acquire scheduler authority within 5 seconds.")
    finally:
        if startup_log is not None:
            startup_log.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available_gpus = None
    if args.available_gpus is not None:
        try:
            available_gpus = [int(item) for item in args.available_gpus.split(",") if item.strip()]
        except ValueError as exc:
            raise RuntimeError("--available-gpus must be a comma-separated list of integers.") from exc
    run_machine_agent_loop(
        args.machine_runtime_root,
        loop_interval=args.loop_interval,
        available_gpus=available_gpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
