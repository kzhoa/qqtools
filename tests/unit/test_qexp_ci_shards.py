from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ci import qexp_shards, run_preflight

ROOT = Path(__file__).resolve().parents[2]


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def reports(tmp_path):
    for phase, count in qexp_shards.PHASE_COUNTS.items():
        path = qexp_shards.LIFECYCLE_PATH if phase == "lifecycle" else "tests/integration/qexp/test_example.py"
        full = [f"{path}::test_case_{index}" for index in range(8)]
        for index in range(count):
            directory = tmp_path / f"{phase}-{index}"
            directory.mkdir()
            workers = ("gw0", "gw1") if phase == "ordinary" else ("main",)
            for worker in workers:
                (directory / f"full-collection-{worker}.txt").write_text("\n".join(full) + "\n")
            selected = full[index::count]
            (directory / f"{phase}-collection.txt").write_text("\n".join(selected) + "\n")
            (directory / f"{phase}-junit.xml").write_text("<testsuite><testcase/></testsuite>")
            _write_json(
                directory / "shard.json",
                {
                    "phase": phase,
                    "index": index,
                    "count": count,
                    "git_sha": "source",
                    "started_epoch": 1000 + index,
                    "finished_epoch": 1100 + index,
                    "result": {"status": "passed", "exit_code": 0, "duration_seconds": 100},
                },
            )
            _write_json(
                directory / f"{phase}-timing.json",
                {
                    "exit_code": 0,
                    "has_skipped": False,
                    "reports": [
                        {"nodeid": nodeid, "when": when, "outcome": "passed"}
                        for nodeid in selected
                        for when in ("setup", "call", "teardown")
                    ],
                },
            )
    return tmp_path


def test_full_partition_passes_with_parallel_wall_time(reports):
    summary = qexp_shards.aggregate(reports, "source")
    assert summary["case_counts"] == {"ordinary": 8, "lifecycle": 8}
    assert summary["duration_seconds"] == 103


@pytest.mark.parametrize("damage", ["missing", "duplicate", "skipped", "failed", "extra"])
def test_aggregate_rejects_incomplete_execution(reports, damage):
    path = reports / "ordinary-0/ordinary-timing.json"
    timing = json.loads(path.read_text())
    if damage == "missing":
        timing["reports"].pop()
    elif damage == "duplicate":
        timing["reports"].append(timing["reports"][0])
    elif damage in {"skipped", "failed"}:
        timing["reports"][0]["outcome"] = damage
    else:
        timing["reports"].append({"nodeid": "uncollected", "when": "call", "outcome": "passed"})
    _write_json(path, timing)
    with pytest.raises(ValueError, match="test reports"):
        qexp_shards.aggregate(reports, "source")


@pytest.mark.parametrize(
    "damage",
    [
        "missing-shard",
        "wrong-sha",
        "wrong-index",
        "collector",
        "partition",
        "full-collection",
        "nan",
        "clock",
        "budget",
    ],
)
def test_aggregate_rejects_invalid_evidence(reports, damage):
    directory = reports / "ordinary-0"
    path = directory / "shard.json"
    payload = json.loads(path.read_text())
    if damage == "missing-shard":
        path.unlink()
    elif damage == "wrong-sha":
        payload["git_sha"] = "other"
    elif damage == "wrong-index":
        payload["index"] = 1
    elif damage == "collector":
        (directory / "full-collection-gw1.txt").unlink()
    elif damage == "partition":
        (directory / "ordinary-collection.txt").write_text("tests/integration/qexp/test_example.py::test_case_1\n")
    elif damage == "full-collection":
        for manifest in directory.glob("full-collection-*"):
            manifest.write_text(manifest.read_text() + "tests/integration/qexp/test_example.py::test_case_z\n")
    elif damage == "nan":
        payload["finished_epoch"] = float("nan")
    elif damage == "clock":
        payload["finished_epoch"] = 1001
    else:
        payload["started_epoch"] += 1000
        payload["finished_epoch"] += 1000
    if damage != "missing-shard":
        _write_json(path, payload)
    with pytest.raises(ValueError):
        qexp_shards.aggregate(reports, "source")


def test_release_source_is_explicitly_partial_and_complete_local_gate_remains():
    partial = run_preflight._commands("release-source", "base", "head", "kzhoa")
    complete = run_preflight._commands("release", "base", "head", "kzhoa")
    assert complete[:-1] == partial
    assert "scripts/qexp_integration_gate.py" in complete[-1]


def test_reusable_release_evidence_requires_source_shards_and_aggregation():
    jobs = yaml.safe_load((ROOT / ".github/workflows/dev-preflight.yml").read_text())["jobs"]
    gate = jobs["release-preflight"]
    assert gate["name"] == "Release preflight (Python 3.13)"
    assert set(gate["needs"]) >= {"preflight", "qexp-shards", "select-evidence"}
    assert "always()" in gate["if"]
    assert "release-preflight" in jobs["gate-result"]["needs"]
    commands = "\n".join(step.get("run", "") for step in gate["steps"])
    assert 'qexp_shards aggregate --sha "$SOURCE_SHA"' in commands
    for result in ("SOURCE_RESULT", "SHARD_RESULT", "EVIDENCE_RESULT"):
        assert f'[[ "${result}" == "success" ]]' in commands
    shards = jobs["qexp-shards"]
    assert shards["strategy"]["fail-fast"] is False
    assert {(item["phase"], item["index"]) for item in shards["strategy"]["matrix"]["include"]} == {
        (phase, index) for phase, count in qexp_shards.PHASE_COUNTS.items() for index in range(count)
    }
