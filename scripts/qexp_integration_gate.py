"""Run the two qexp integration phases under one bounded execution budget."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUDGET_SECONDS = 600.0
REPORT_ENV = "QEXP_GATE_REPORT_DIR"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args, extra = parser.parse_known_args(argv)
    args.pytest_args = [*args.pytest_args, *extra]
    return args


def _report_dir(value: Path | None) -> Path:
    directory = value or Path(os.environ.get(REPORT_ENV, "qexp-gate-reports"))
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "ordinary-junit.xml",
        "ordinary-stdout.log",
        "ordinary-stderr.log",
        "ordinary-collection.txt",
        "ordinary-timing.json",
        "lifecycle-junit.xml",
        "lifecycle-stdout.log",
        "lifecycle-stderr.log",
        "lifecycle-collection.txt",
        "lifecycle-timing.json",
        "summary.json",
    ):
        path = directory / name
        if path.is_file():
            path.unlink()
    return directory


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_commands(extra: list[str], report_dir: Path) -> list[tuple[str, list[str]]]:
    ordinary = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/qexp",
        "-q",
        "--ignore=tests/integration/qexp/test_agent_lifecycle_independence.py",
        "-n",
        "4",
        "--dist",
        "load",
        f"--junitxml={report_dir / 'ordinary-junit.xml'}",
        f"--qexp-collection-manifest={report_dir / 'ordinary-collection.txt'}",
        f"--qexp-timing-json={report_dir / 'ordinary-timing.json'}",
        *extra,
    ]
    lifecycle = [
        sys.executable,
        "-m",
        "pytest",
        "--lifecycle-gate=full",
        "tests/integration/qexp/test_agent_lifecycle_independence.py",
        "-n",
        "4",
        "--dist",
        "load",
        "-q",
        f"--junitxml={report_dir / 'lifecycle-junit.xml'}",
        f"--qexp-collection-manifest={report_dir / 'lifecycle-collection.txt'}",
        f"--qexp-timing-json={report_dir / 'lifecycle-timing.json'}",
        *extra,
    ]
    return [("ordinary", ordinary), ("lifecycle", lifecycle)]


def _run_phase(
    name: str,
    command: list[str],
    report_dir: Path,
    deadline: float,
) -> dict[str, object]:
    started = time.monotonic()
    stdout_path = report_dir / f"{name}-stdout.log"
    stderr_path = report_dir / f"{name}-stderr.log"
    result: dict[str, object] = {"name": name, "command": command, "started_at": started}
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                text=True,
            )
            try:
                return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result["exit_code"] = 124
                result["status"] = "timeout"
                return result
        result["exit_code"] = return_code
        result["status"] = "passed" if return_code == 0 else "failed"
        return result
    except OSError as error:
        result["exit_code"] = 127
        result["status"] = "spawn_failed"
        result["error"] = str(error)
        return result
    finally:
        result["finished_at"] = time.monotonic()
        result["duration_seconds"] = result["finished_at"] - started


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.budget_seconds <= 0:
        raise SystemExit("--budget-seconds must be positive")
    report_dir = _report_dir(args.report_dir)
    started = time.monotonic()
    phases: list[dict[str, object]] = []
    overall_status = "passed"
    for name, command in _phase_commands(args.pytest_args, report_dir):
        phase = _run_phase(name, command, report_dir, started + args.budget_seconds)
        phases.append(phase)
        if phase.get("status") != "passed":
            overall_status = "timeout" if phase.get("status") == "timeout" else "failed"
            if phase.get("status") == "timeout":
                break
        if time.monotonic() >= started + args.budget_seconds:
            break
    finished = time.monotonic()
    if finished > started + args.budget_seconds:
        overall_status = "timeout"
    summary = {
        "status": overall_status,
        "budget_seconds": args.budget_seconds,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": finished - started,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "phases": phases,
    }
    _write_json(report_dir / "summary.json", summary)
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
