"""Run and reconcile the six release qexp shards for one exact source commit."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from scripts.qexp_integration_gate import _phase_evidence_errors, _print_failure_log_tails, _run_phase, _write_json

PHASE_COUNTS = {"ordinary": 4, "lifecycle": 2}
LIFECYCLE_PATH = "tests/integration/qexp/test_agent_lifecycle_independence.py"
BUDGET_SECONDS = 600


def _read_manifest(path: Path) -> list[str]:
    nodeids = path.read_text(encoding="utf-8").splitlines()
    if not nodeids or any(not nodeid for nodeid in nodeids) or len(nodeids) != len(set(nodeids)):
        raise ValueError(f"empty or duplicate collection: {path}")
    return sorted(nodeids)


def _source_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run_shard(phase: str, index: int, root: Path, sha: str) -> int:
    count = PHASE_COUNTS[phase]
    if not 0 <= index < count:
        raise ValueError(f"{phase} shard index must be in [0, {count})")
    if _source_sha() != sha:
        raise ValueError("checkout does not match expected source SHA")
    directory = root / f"{phase}-{index}"
    # A stale result must never survive a retry or partial subprocess failure.
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "pytest", "-q"]
    if phase == "ordinary":
        command += ["tests/integration/qexp", f"--ignore={LIFECYCLE_PATH}", "-n", "2", "--dist", "load"]
    else:
        command += [LIFECYCLE_PATH, "--lifecycle-gate=full", "-n", "0"]
    command += [
        f"--qexp-shard={index}/{count}",
        f"--qexp-shard-directory={directory}",
        f"--qexp-collection-manifest={directory / (phase + '-collection.txt')}",
        f"--qexp-timing-json={directory / (phase + '-timing.json')}",
        f"--junitxml={directory / (phase + '-junit.xml')}",
    ]
    started = time.time()
    result = _run_phase(phase, command, directory, time.monotonic() + 1200)
    finished = time.time()
    _write_json(
        directory / "shard.json",
        {
            "phase": phase,
            "index": index,
            "count": count,
            "git_sha": sha,
            "started_epoch": started,
            "finished_epoch": finished,
            "cpu_count": os.cpu_count(),
            "result": result,
        },
    )
    if result["status"] != "passed":
        _print_failure_log_tails([result], directory)
        return 1
    return 0


def _validate_shard(directory: Path, phase: str, index: int, sha: str) -> tuple[list[str], float, float]:
    payload = json.loads((directory / "shard.json").read_text(encoding="utf-8"))
    count = PHASE_COUNTS[phase]
    for field, expected in (("phase", phase), ("index", index), ("count", count), ("git_sha", sha)):
        if payload.get(field) != expected:
            raise ValueError(f"{directory}: wrong {field}")
    result = payload["result"]
    if result.get("status") != "passed" or result.get("exit_code") != 0:
        raise ValueError(f"{directory}: shard did not pass")
    start, finish = payload["started_epoch"], payload["finished_epoch"]
    duration = result["duration_seconds"]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (start, finish, duration)):
        raise ValueError(f"{directory}: invalid timing")
    if start <= 0 or finish < start or duration < 0 or abs(finish - start - duration) > 5:
        raise ValueError(f"{directory}: inconsistent wall/monotonic timing")
    errors = _phase_evidence_errors(phase, directory)
    if errors:
        raise ValueError("; ".join(errors))
    full_paths = sorted(directory.glob("full-collection-*.txt"))
    expected_workers = {"gw0", "gw1"} if phase == "ordinary" else {"main"}
    if {path.stem.removeprefix("full-collection-") for path in full_paths} != expected_workers:
        raise ValueError(f"{directory}: missing or extra collector manifests")
    full = _read_manifest(full_paths[0])
    for path in full_paths[1:]:
        if _read_manifest(path) != full:
            raise ValueError(f"{directory}: collectors disagree")
    for nodeid in full:
        path = nodeid.split("::", 1)[0]
        if not path.startswith("tests/integration/qexp/") or (path == LIFECYCLE_PATH) != (phase == "lifecycle"):
            raise ValueError(f"{directory}: node outside phase: {nodeid}")
    selected = _read_manifest(directory / f"{phase}-collection.txt")
    if selected != full[index::count]:
        raise ValueError(f"{directory}: selected collection does not match complete partition")
    timing = json.loads((directory / f"{phase}-timing.json").read_text(encoding="utf-8"))
    expected_reports = Counter(
        (nodeid, when, "passed") for nodeid in selected for when in ("setup", "call", "teardown")
    )
    actual_reports = Counter((report["nodeid"], report["when"], report["outcome"]) for report in timing["reports"])
    if timing.get("has_skipped") is not False or actual_reports != expected_reports:
        raise ValueError(f"{directory}: missing, duplicate, skipped, or failed test reports")
    return full, start, finish


def aggregate(root: Path, sha: str) -> dict[str, object]:
    """Reject incomplete coverage, failed execution, stale SHA, or over-budget evidence."""
    expected = {f"{phase}-{index}" for phase, count in PHASE_COUNTS.items() for index in range(count)}
    actual = {path.parent.name for path in root.glob("*/shard.json")}
    if actual != expected:
        raise ValueError(f"missing/extra shards: expected {sorted(expected)}, got {sorted(actual)}")
    starts, finishes, totals = [], [], {}
    for phase, count in PHASE_COUNTS.items():
        collection = None
        for index in range(count):
            full, start, finish = _validate_shard(root / f"{phase}-{index}", phase, index, sha)
            if collection is not None and full != collection:
                raise ValueError(f"{phase}: shards disagree on full collection")
            collection = full
            starts.append(start)
            finishes.append(finish)
        totals[phase] = len(collection)
    duration = max(finishes) - min(starts)
    if duration > BUDGET_SECONDS:
        raise ValueError(f"parallel qexp wall time {duration:.2f}s exceeds {BUDGET_SECONDS}s")
    return {"status": "passed", "git_sha": sha, "duration_seconds": duration, "case_counts": totals}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "aggregate"))
    parser.add_argument("--phase", choices=tuple(PHASE_COUNTS))
    parser.add_argument("--index", type=int)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("qexp-shard-reports"))
    args = parser.parse_args(argv)
    try:
        if args.operation == "run":
            if args.phase is None or args.index is None:
                parser.error("run requires --phase and --index")
            return run_shard(args.phase, args.index, args.report_dir, args.sha)
        summary = aggregate(args.report_dir, args.sha)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        summary = {"status": "failed", "error": str(exc), "git_sha": args.sha}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
