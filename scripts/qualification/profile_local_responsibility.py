"""Measure real durable I/O for the isolated ledger; never opens qexp roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import tests.helpers.qexp.local_responsibility as ledger_adapter
from qqtools.plugins.qexp.runtime import responsibility_store
from tests.helpers.qexp.local_responsibility import BUCKETS, DurableIO, Ledger, identity_key


def names_in_bucket(count: int) -> list[str]:
    result = []
    index = 0
    while len(result) < count:
        identity = f"project/attempt-{index}"
        if identity_key(identity)[0] == "0":
            result.append(identity)
        index += 1
    return result


def measure(ledger: Ledger, operation: str, callback, rows: list[dict], **context):
    ledger.io = DurableIO()
    started = time.perf_counter()
    result = callback()
    rows.append(
        {
            "operation": operation,
            **context,
            "seconds": time.perf_counter() - started,
            "counts": dict(ledger.io.counts),
            "timing_seconds": dict(ledger.io.seconds),
        }
    )
    return result


def traverse(ledger: Ledger, stage: str | None = None) -> int:
    seen = set()
    for bucket in range(BUCKETS):
        cursor = None
        while True:
            records, cursor = ledger.service_page(bucket, cursor, stage=stage)
            for entry in records:
                assert entry["identity"] not in seen
                seen.add(entry["identity"])
            if cursor is None:
                break
    return len(seen)


def run(output: Path, cycles: int, samples: int, live_counts: list[int]) -> dict:
    output.mkdir()  # Refuse to overwrite prior evidence or reuse arbitrary roots.
    sources = [Path(__file__), Path(ledger_adapter.__file__), Path(responsibility_store.__file__)]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    rows = []
    names = names_in_bucket(max(cycles + 4, max(live_counts) + samples))
    for live in live_counts:
        ledger = Ledger.create(output / f"live-{live}")
        for identity in names[:live]:
            ledger.publish(identity, {"task_id": "fixture-task", "attempt_number": 1})
        for sample in range(samples):
            assert measure(ledger, "traverse", lambda: traverse(ledger), rows, live=live) == live
            identity = names[live + sample]
            generation = measure(ledger, "publish", lambda: ledger.publish(identity, {}), rows, live=live)
            assert measure(ledger, "publish_retry", lambda: ledger.publish(identity, {}), rows, live=live) == generation
            generation = measure(ledger, "handoff", lambda: ledger.handoff(identity, generation), rows, live=live)
            assert measure(ledger, "retire_tail", lambda: ledger.retire(identity, generation), rows, live=live)

        # Move the all-members tail locator across pages.
        first = names[0]
        generation = ledger.handoff(first, ledger.lookup(first)["generation"])
        measure(ledger, "retire_first", lambda: ledger.retire(first, generation), rows, live=live)
        assert traverse(ledger) == live - 1
        print(f"completed live={live}", flush=True)

    ledger = Ledger.create(output / "stage-compaction")
    stage_names = names_in_bucket(68)
    for identity in stage_names:
        ledger.publish(identity, {})
    for identity in stage_names[:66]:
        generation = ledger.lookup(identity)["generation"]
        measure(ledger, "stage_handoff", lambda: ledger.handoff(identity, generation), rows)
    assert measure(ledger, "traverse_active", lambda: traverse(ledger, "active"), rows) == 2
    assert measure(ledger, "traverse_maintenance", lambda: traverse(ledger, "maintenance"), rows) == 66
    first = stage_names[0]
    generation = ledger.lookup(first)["generation"]
    assert measure(ledger, "retire_dual", lambda: ledger.retire(first, generation), rows)
    assert traverse(ledger) == 67

    ledger = Ledger.create(output / "churn")
    for identity in names[:4]:
        ledger.publish(identity, {})
    for index, identity in enumerate(names[4 : cycles + 4]):
        generation = measure(ledger, "churn_publish", lambda: ledger.publish(identity, {}), rows, history=index)
        generation = measure(ledger, "churn_handoff", lambda: ledger.handoff(identity, generation), rows, history=index)
        measure(ledger, "churn_retire", lambda: ledger.retire(identity, generation), rows, history=index)
        if index in (0, cycles // 2, cycles - 1):
            for _ in range(samples):
                assert measure(ledger, "churn_traverse", lambda: traverse(ledger), rows, history=index + 1) == 4
    files = sorted(str(path.relative_to(ledger.root)) for path in ledger.root.rglob("*") if path.is_file())
    assert len(files) == 39  # 33 fixed files, four locators, base and active pages.
    # One-record fsync baseline, not a semantically equivalent recovery protocol.
    baseline = output / "atomic-baseline"
    baseline.mkdir()
    ledger.io = DurableIO()
    for sample in range(samples):
        measure(
            ledger, "single_atomic_replace", lambda: ledger.io.replace(baseline / "record", {"revision": sample}), rows
        )
    summary = {}
    for operation in sorted({row["operation"] for row in rows}):
        selected = [row for row in rows if row["operation"] == operation]
        times = sorted(row["seconds"] for row in selected)
        summary[operation] = {
            "samples": len(times),
            "median_ms": statistics.median(times) * 1000,
            "p95_ms": times[math.ceil(0.95 * len(times)) - 1] * 1000,
            "max_ms": max(times) * 1000,
            "max_counts": {
                key: max(row["counts"].get(key, 0) for row in selected)
                for key in sorted({key for row in selected for key in row["counts"]})
            },
        }
    result = {
        "source_sha256": source_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "root": str(output.resolve()),
        "cycles": cycles,
        "live_counts": live_counts,
        "samples_per_case": samples,
        "cache": "uncontrolled, immediately after writes; not cold",
        "durability": "real file and directory fsync; no power-loss emulation",
        "churn_remaining_files": files,
        "summary": summary,
        "rows": rows,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New experiment directory under an existing parent")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--live", type=int, nargs="+", default=[4, 64, 65, 256])
    args = parser.parse_args()
    if not 1 <= args.cycles <= 100000 or not 1 <= args.samples <= 100:
        parser.error("cycles must be 1..100000 and samples must be 1..100")
    if (
        not args.live
        or any(value not in (4, 64, 65, 256) for value in args.live)
        or len(set(args.live)) != len(args.live)
    ):
        parser.error("live counts must be distinct values from 4, 64, 65, 256")
    result = run(args.output, args.cycles, args.samples, args.live)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
