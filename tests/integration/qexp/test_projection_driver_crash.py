"""Process-death coverage for the cooperative projection driver prototype."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOOKS = (
    "spool_write",
    "spool_fsync",
    "checkpoint_write",
    "checkpoint_fsync",
    "checkpoint_replace",
    "directory_fsync",
)
_CRASH_EXIT = 73
_SOURCE_CHANGED_EXIT = 17
_CHILD_TIMEOUT = 90
_CHECKPOINT_KEYS = {
    "version",
    "source_path",
    "source_revision",
    "expected_operation_id",
    "expected_group",
    "spool",
    "scanner",
    "projection",
}


def _session_class() -> type:
    from tests.helpers.qexp.qexp_projection_session import ProjectionSession

    return ProjectionSession


def _fixture_payload() -> tuple[bytes, dict[tuple[str, int], bytes]]:
    plain_id = b"task-" + (b"a" * 133_120)
    escaped_id = b"task-" + (b"\\u0062" * 85_000)
    decoded_escaped_id = b"task-" + (b"b" * 85_000)
    sequence = b"1" + (b"7" * 131_072)
    second_sequence = b"42"
    payload = (
        b'{"meta":{"schema_version":6},"submission":{"operation_id":"op-1","target_group":"exp",'
        b'"state":"committed","resolved_context":{"task_ids":["'
        + plain_id
        + b'","'
        + escaped_id
        + b'"]},"commit_plan":{"group_membership_sequences":['
        + sequence
        + b","
        + second_sequence
        + b"]}}}"
    )
    return payload, {
        ("task_id", 0): plain_id,
        ("task_id", 1): decoded_escaped_id,
        ("sequence", 0): sequence,
        ("sequence", 1): second_sequence,
    }


def _summary_fields(summary: object) -> dict[str, object]:
    return {name: getattr(summary, name) for name in ("state", "matches_group", "task_count", "sequence_count")}


def _drive(session: object) -> None:
    while not session.is_complete:
        session.step(max_bytes=65_536, max_fragments=2)
        session.checkpoint()
    session.checkpoint()


def _baseline(tmp_path: Path, payload: bytes) -> tuple[bytes, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "baseline-source.json"
    scratch = tmp_path / "baseline-scratch"
    source.write_bytes(payload)
    session = _session_class().create(source, scratch, "op-1", "exp")
    try:
        _drive(session)
        summary = _summary_fields(session.summary)
    finally:
        session.close()
    return (scratch / "events.jsonl").read_bytes(), summary


def _assert_events(event_bytes: bytes, expected_fields: dict[tuple[str, int], bytes]) -> None:
    events = [json.loads(line) for line in event_bytes.splitlines()]
    assert events
    chunk_events = [event for event in events if event["type"] == "chunk"]
    end_events = [event for event in events if event["type"] == "end"]
    assert all(set(event) == {"type", "kind", "ordinal", "data_b64", "is_final"} for event in chunk_events)
    assert all(
        set(event) == {"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"} for event in end_events
    )

    values: dict[tuple[str, int], bytearray] = {}
    finals: dict[tuple[str, int], int] = {}
    for event in chunk_events:
        key = (event["kind"], event["ordinal"])
        data = base64.b64decode(event["data_b64"], validate=True)
        values.setdefault(key, bytearray()).extend(data)
        finals[key] = finals.get(key, 0) + int(event["is_final"])
    assert {key: bytes(value) for key, value in values.items()} == expected_fields
    assert finals == {key: 1 for key in expected_fields}

    end_counts: dict[str, int] = {}
    for event in end_events:
        end_counts[event["kind"]] = end_counts.get(event["kind"], 0) + 1
        assert event["decoded_size"] == len(values[(event["kind"], event["ordinal"])])
        assert event["start"] < event["end"]
    assert end_counts == {"task_id": 2, "sequence": 2}
    assert len(end_events) == 4


def _checkpoint(scratch: Path) -> dict[str, Any]:
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    assert set(checkpoint) == _CHECKPOINT_KEYS
    assert checkpoint["version"] == 1
    assert set(checkpoint["spool"]) == {"device", "inode", "size", "events"}
    return checkpoint


def _run_child(source: Path, scratch: Path, mode: str, hook_name: str = "", saved_offset: int = -1):
    code = r"""
import json
import os
import sys
from pathlib import Path

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import YIELD, SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.source_revision import SourceChangedError

source = Path(sys.argv[1])
scratch = Path(sys.argv[2])
mode = sys.argv[3]
hook_name = sys.argv[4]
saved_offset = int(sys.argv[5])
driver_ref = {"value": None}
hook_state = {"post_initial": 0, "initial": 0}
source_tracker = {
    "fd": None,
    "position": 0,
    "lseeks": [],
    "reads": [],
}


class TrackingSliceIO(SliceIO):
    def open(self, path, flags, mode=0o600):
        result = super().open(path, flags, mode)
        if result is not YIELD and Path(path) == source:
            source_tracker["fd"] = result
            source_tracker["position"] = 0
        return result

    def lseek(self, fd, offset, whence):
        result = super().lseek(fd, offset, whence)
        if result is not YIELD and fd == source_tracker["fd"]:
            source_tracker["position"] = result
            source_tracker["lseeks"].append(result)
        return result

    def read(self, fd, size):
        result = super().read(fd, size)
        if result is not YIELD and fd == source_tracker["fd"]:
            source_tracker["reads"].append(
                {"start": source_tracker["position"], "bytes": len(result)}
            )
            source_tracker["position"] += len(result)
        return result


def hook(event):
    driver = driver_ref["value"]
    if event != hook_name or driver is None:
        return
    if mode == "crash_initial":
        if driver.checkpoint_generation == 0:
            hook_state["initial"] += 1
            if hook_state["initial"] == 1:
                os._exit(73)
    elif mode == "crash_after_initial" and driver.checkpoint_generation >= 1:
        hook_state["post_initial"] += 1
        if hook_state["post_initial"] == 2:
            os._exit(73)


def advance_to_completion(driver):
    last_generation = driver.checkpoint_generation
    checkpoint_outstanding = False
    last_requested_offset = driver.processed_offset
    for iteration in range(100000):
        if driver.is_complete:
            break
        io = TrackingSliceIO()
        step = driver.advance(io, max_processed_bytes=65536)
        if driver.checkpoint_generation != last_generation:
            last_generation = driver.checkpoint_generation
            checkpoint_outstanding = False
        if (
            driver.checkpoint_generation >= 1
            and not checkpoint_outstanding
            and driver.processed_offset - last_requested_offset >= 32768
        ):
            driver.request_checkpoint()
            checkpoint_outstanding = True
            last_requested_offset = driver.processed_offset
    else:
        raise RuntimeError(f"advance loop exceeded phase={step.phase!r}")


driver = None
try:
    hook_arg = hook if mode in {"crash_initial", "crash_after_initial"} else None
    driver = ProjectionDriver(source, scratch, "op-1", "exp", hook=hook_arg)
    driver_ref["value"] = driver
    advance_to_completion(driver)
    if not driver.is_complete or driver.summary is None:
        raise RuntimeError("driver did not expose a completed summary")
    summary = {
        "state": driver.summary.state,
        "matches_group": driver.summary.matches_group,
        "task_count": driver.summary.task_count,
        "sequence_count": driver.summary.sequence_count,
    }
    driver.request_close()
    for iteration in range(100000):
        if driver.is_closed:
            break
        step = driver.advance(SliceIO())
    else:
        raise RuntimeError(f"cleanup loop exceeded phase={step.phase!r}")
    if mode == "expect_source_changed":
        raise AssertionError("source replacement was accepted")
    print(
        json.dumps(
            {
                "summary": summary,
                "source_reads": source_tracker["reads"],
                "source_lseeks": source_tracker["lseeks"],
                "saved_offset": saved_offset,
            },
            sort_keys=True,
        )
    )
except SourceChangedError as exc:
    if mode != "expect_source_changed":
        raise
    print(f"SourceChangedError: {exc}", file=sys.stderr)
    raise SystemExit(17)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(source), str(scratch), mode, hook_name, str(saved_offset)],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_CHILD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(f"child timed out and was killed: stdout={stdout!r} stderr={stderr!r}") from exc
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _assert_crashed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == _CRASH_EXIT, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert result.stdout == ""


def _assert_recovered(
    result: subprocess.CompletedProcess[str],
    scratch: Path,
    baseline_events: bytes,
    baseline_summary: dict[str, object],
    expected_fields: dict[tuple[str, int], bytes],
    saved_offset: int,
    *,
    check_source_offset: bool = True,
) -> None:
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["summary"] == baseline_summary
    if check_source_offset:
        assert report["source_reads"]
        assert report["source_reads"][0]["start"] >= saved_offset
        assert report["source_lseeks"]
    recovered_events = (scratch / "events.jsonl").read_bytes()
    assert recovered_events == baseline_events
    _assert_events(recovered_events, expected_fields)


@pytest.mark.parametrize("hook_name", _HOOKS)
def test_projection_driver_recovers_each_post_initial_durability_hook(tmp_path: Path, hook_name: str):
    payload, expected_fields = _fixture_payload()
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    _assert_events(baseline_events, expected_fields)

    case = tmp_path / f"hook-{hook_name}"
    source = case / "source.json"
    scratch = case / "scratch"
    case.mkdir()
    source.write_bytes(payload)
    crashed = _run_child(source, scratch, "crash_after_initial", hook_name)
    _assert_crashed(crashed)
    checkpoint = _checkpoint(scratch)
    saved_offset = checkpoint["scanner"]["offset"]

    recovered = _run_child(source, scratch, "recover", saved_offset=saved_offset)
    _assert_recovered(recovered, scratch, baseline_events, baseline_summary, expected_fields, saved_offset)


def test_projection_driver_recovers_repeated_checkpoint_write_and_replace_deaths(tmp_path: Path):
    payload, expected_fields = _fixture_payload()
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    case = tmp_path / "repeated"
    source = case / "source.json"
    scratch = case / "scratch"
    case.mkdir()
    source.write_bytes(payload)

    first = _run_child(source, scratch, "crash_after_initial", "checkpoint_write")
    _assert_crashed(first)
    _checkpoint(scratch)
    second = _run_child(source, scratch, "crash_after_initial", "checkpoint_replace")
    _assert_crashed(second)
    checkpoint = _checkpoint(scratch)
    saved_offset = checkpoint["scanner"]["offset"]

    recovered = _run_child(source, scratch, "recover", saved_offset=saved_offset)
    _assert_recovered(recovered, scratch, baseline_events, baseline_summary, expected_fields, saved_offset)


def test_projection_driver_rejects_same_byte_source_replacement_before_spool_mutation(tmp_path: Path):
    payload, _expected_fields = _fixture_payload()
    case = tmp_path / "replacement"
    source = case / "source.json"
    scratch = case / "scratch"
    case.mkdir()
    source.write_bytes(payload)

    crashed = _run_child(source, scratch, "crash_after_initial", "checkpoint_write")
    _assert_crashed(crashed)
    checkpoint_before = (scratch / "checkpoint.json").read_bytes()
    events_before = (scratch / "events.jsonl").read_bytes()
    saved_offset = _checkpoint(scratch)["scanner"]["offset"]

    replacement = case / "replacement-source.tmp"
    replacement.write_bytes(payload)
    original_inode = source.stat().st_ino
    os.replace(replacement, source)
    assert source.stat().st_ino != original_inode

    rejected = _run_child(source, scratch, "expect_source_changed", saved_offset=saved_offset)
    assert rejected.returncode == _SOURCE_CHANGED_EXIT
    assert rejected.stdout == ""
    assert "SourceChangedError" in rejected.stderr
    assert (scratch / "checkpoint.json").read_bytes() == checkpoint_before
    assert (scratch / "events.jsonl").read_bytes() == events_before


def test_projection_driver_recovers_interrupted_initial_checkpoint_and_preserves_unknown_entries(tmp_path: Path):
    payload, expected_fields = _fixture_payload()
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    case = tmp_path / "initial"
    source = case / "source.json"
    scratch = case / "scratch"
    case.mkdir()
    source.write_bytes(payload)

    crashed = _run_child(source, scratch, "crash_initial", "checkpoint_write")
    _assert_crashed(crashed)
    assert scratch.is_dir()
    assert not (scratch / "checkpoint.json").exists()
    unknown = scratch / "unknown-entry"
    unknown.write_bytes(b"leave this entry alone")

    recovered = _run_child(source, scratch, "recover")
    _assert_recovered(
        recovered,
        scratch,
        baseline_events,
        baseline_summary,
        expected_fields,
        0,
        check_source_offset=False,
    )
    assert unknown.read_bytes() == b"leave this entry alone"
