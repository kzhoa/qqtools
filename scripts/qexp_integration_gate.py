"""Run both qexp integration phases with a soft budget and hard timeout."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUDGET_SECONDS = 600.0
DEFAULT_HARD_TIMEOUT_SECONDS = 1200.0
REPORT_ENV = "QEXP_GATE_REPORT_DIR"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--hard-timeout-seconds", type=float, default=DEFAULT_HARD_TIMEOUT_SECONDS)
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


def _phase_evidence_errors(name: str, report_dir: Path) -> list[str]:
    errors: list[str] = []
    junit_path = report_dir / f"{name}-junit.xml"
    collection_path = report_dir / f"{name}-collection.txt"
    timing_path = report_dir / f"{name}-timing.json"
    try:
        junit_root = ET.parse(junit_path).getroot()
        root_tag = junit_root.tag.rpartition("}")[2]
        has_testcase = any(element.tag.rpartition("}")[2] == "testcase" for element in junit_root.iter())
        if root_tag not in {"testsuite", "testsuites"} or not has_testcase:
            errors.append(f"incomplete JUnit report {junit_path}")
    except (ET.ParseError, OSError) as exc:
        errors.append(f"invalid JUnit report {junit_path}: {exc}")
    try:
        if not collection_path.read_text(encoding="utf-8").strip():
            errors.append(f"empty collection manifest {collection_path}")
    except OSError as exc:
        errors.append(f"invalid collection manifest {collection_path}: {exc}")
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if not isinstance(timing, dict) or timing.get("exit_code") != 0 or not timing.get("reports"):
            errors.append(f"incomplete timing report {timing_path}")
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"invalid timing report {timing_path}: {exc}")
    return errors


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait()


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
    hard_deadline: float,
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
                return_code = process.wait(timeout=max(0.01, hard_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                result["exit_code"] = 124
                result["status"] = "hard_timeout"
                return result
        result["exit_code"] = return_code
        evidence_errors = _phase_evidence_errors(name, report_dir)
        if evidence_errors:
            result["evidence_errors"] = evidence_errors
        if return_code != 0:
            result["status"] = "failed"
        elif evidence_errors:
            result["status"] = "evidence_failed"
        else:
            result["status"] = "passed"
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
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        raise SystemExit("--budget-seconds must be finite and positive")
    if not math.isfinite(args.hard_timeout_seconds) or args.hard_timeout_seconds <= args.budget_seconds:
        raise SystemExit("--hard-timeout-seconds must be greater than --budget-seconds")
    report_dir = _report_dir(args.report_dir)
    started = time.monotonic()
    hard_deadline = started + args.hard_timeout_seconds
    phases: list[dict[str, object]] = []
    phase_commands = _phase_commands(args.pytest_args, report_dir)
    for name, command in phase_commands:
        phase = _run_phase(name, command, report_dir, hard_deadline)
        phases.append(phase)
        if phase.get("status") == "hard_timeout":
            break
    finished = time.monotonic()
    duration_seconds = finished - started
    statuses = {phase.get("status") for phase in phases}
    if "hard_timeout" in statuses or len(phases) != len(phase_commands):
        overall_status = "hard_timeout"
    elif "spawn_failed" in statuses:
        overall_status = "spawn_failed"
    elif statuses != {"passed"}:
        overall_status = "test_failed"
    elif duration_seconds > args.budget_seconds:
        overall_status = "budget_exceeded"
    else:
        overall_status = "passed"
    summary = {
        "status": overall_status,
        "budget_seconds": args.budget_seconds,
        "hard_timeout_seconds": args.hard_timeout_seconds,
        "budget_overrun_seconds": max(0.0, duration_seconds - args.budget_seconds),
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": duration_seconds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "phases": phases,
    }
    _write_json(report_dir / "summary.json", summary)
    if overall_status != "passed":
        print(
            f"qexp integration gate {overall_status}: {duration_seconds:.2f}s elapsed; "
            f"soft budget {args.budget_seconds:.2f}s; hard timeout {args.hard_timeout_seconds:.2f}s; "
            f"reports: {report_dir}",
            file=sys.stderr,
        )
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
