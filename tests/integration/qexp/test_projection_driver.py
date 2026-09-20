"""Integration coverage for the cooperative projection driver."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.source_revision import SourceChangedError
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

pytestmark = pytest.mark.integration


def _payload(*, schema_version: int = 6, large: bool = False) -> bytes:
    task_id = "a" * 200_000 if large else "task-a"
    sequence = "9" * 100_000 if large else "1"
    value = {
        "meta": {"schema_version": schema_version},
        "submission": {
            "operation_id": "op-1",
            "target_group": "exp",
            "state": "committed",
            "resolved_context": {"task_ids": [task_id]},
            "commit_plan": {"group_membership_sequences": [int(sequence) if not large else sequence]},
        },
    }
    if large:
        # Keep the large sequence as a JSON number without converting it to an
        # integer, which is the shape accepted by the streaming projection.
        return (
            b'{"meta":{"schema_version":6},"submission":{"operation_id":"op-1","target_group":"exp",'
            b'"state":"committed","resolved_context":{"task_ids":["'
            + task_id.encode()
            + b'"]},"commit_plan":{"group_membership_sequences":['
            + sequence.encode()
            + b"]}}}"
        )
    return json.dumps(value, separators=(",", ":")).encode()


def _slice() -> SliceIO:
    return SliceIO(max_io_bytes=262_144, max_operations=32)


def _drive(driver: ProjectionDriver, *, max_processed_bytes: int = 65_536) -> None:
    for _ in range(2_000):
        result = driver.advance(_slice(), max_processed_bytes=max_processed_bytes)
        if result.state == "complete":
            return
    raise AssertionError("projection driver did not complete")


def _close(driver: ProjectionDriver) -> None:
    driver.request_close()
    for _ in range(200):
        if driver.advance(_slice()).state == "closed":
            return
    raise AssertionError("projection driver did not close")


def test_driver_matches_baseline_session_and_publishes_completion(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload())

    baseline_scratch = tmp_path / "baseline"
    baseline = ProjectionSession.create(source, baseline_scratch, "op-1", "exp")
    try:
        while not baseline.is_complete:
            baseline.step(65_536, max_fragments=2)
        baseline_events = (baseline_scratch / "events.jsonl").read_bytes()
    finally:
        baseline.close()

    scratch = tmp_path / "driver"
    driver = ProjectionDriver(source, scratch, "op-1", "exp")
    assert driver.processed_offset == 0
    assert driver.checkpoint_generation == 0
    _drive(driver)
    assert driver.is_complete
    assert driver.summary is not None
    assert driver.checkpoint_generation >= 2
    assert (scratch / "events.jsonl").read_bytes() == baseline_events
    _close(driver)


def test_driver_checkpoint_close_and_resume_preserve_event_prefix(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload(large=True))
    baseline_scratch = tmp_path / "baseline"
    baseline = ProjectionDriver(source, baseline_scratch, "op-1", "exp")
    _drive(baseline)
    expected_events = (baseline_scratch / "events.jsonl").read_bytes()
    _close(baseline)

    scratch = tmp_path / "scratch"
    driver = ProjectionDriver(source, scratch, "op-1", "exp")
    for _ in range(200):
        driver.advance(_slice(), max_processed_bytes=1_024)
        if driver.processed_offset >= 1_024 and driver.checkpoint_generation == 1:
            break
    saved_offset = driver.processed_offset
    driver.request_checkpoint()
    for _ in range(200):
        driver.advance(_slice(), max_processed_bytes=1_024)
        if driver.checkpoint_generation >= 2:
            break
    assert driver.processed_offset == saved_offset
    saved_events = (scratch / "events.jsonl").read_bytes()
    assert expected_events.startswith(saved_events)
    _close(driver)

    resumed = ProjectionDriver(source, scratch, "op-1", "exp")
    _drive(resumed, max_processed_bytes=1_024)
    assert resumed.processed_offset == len(source.read_bytes())
    assert (scratch / "events.jsonl").read_bytes() == expected_events
    _close(resumed)


def test_driver_requires_pinned_schema_before_completion(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload(schema_version=5))
    driver = ProjectionDriver(source, tmp_path / "scratch", "op-1", "exp")
    with pytest.raises(ValueError, match="schema version"):
        _drive(driver)
    assert driver.is_closed


def test_driver_busy_lock_waits_without_mutating_scratch(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    lock_path = scratch.with_name(scratch.name + ".lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        driver = ProjectionDriver(source, scratch, "op-1", "exp")
        result = driver.advance(_slice())
        assert result.state == "waiting"
        assert result.reason == "scratch_busy"
        assert driver.checkpoint_generation == 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    _drive(driver)
    _close(driver)


def test_driver_changed_source_resume_keeps_speculative_suffix(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload())
    scratch = tmp_path / "scratch"
    driver = ProjectionDriver(source, scratch, "op-1", "exp")
    while driver.checkpoint_generation < 1:
        driver.advance(_slice(), max_processed_bytes=256)
    driver.request_checkpoint()
    while driver.checkpoint_generation < 2:
        driver.advance(_slice(), max_processed_bytes=256)
    _close(driver)
    events = scratch / "events.jsonl"
    events.open("ab").write(b"speculative suffix")
    source.write_bytes(source.read_bytes() + b" ")

    resumed = ProjectionDriver(source, scratch, "op-1", "exp")
    with pytest.raises(SourceChangedError):
        resumed.advance(_slice())
    assert events.read_bytes().endswith(b"speculative suffix")


def test_invalid_completed_resume_preserves_spool_suffix(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload())
    scratch = tmp_path / "scratch"
    original = ProjectionDriver(source, scratch, "op-1", "exp")
    _drive(original)
    _close(original)
    checkpoint_path = scratch / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_bytes())
    checkpoint["projection"]["state"]["source_schema_version"] = None
    checkpoint_path.write_text(json.dumps(checkpoint))
    events = scratch / "events.jsonl"
    expected = events.read_bytes() + b"speculative suffix"
    events.write_bytes(expected)
    resumed = ProjectionDriver(source, scratch, "op-1", "exp")
    try:
        with pytest.raises(ValueError, match="schema"):
            _drive(resumed)
        assert resumed.is_closed
        assert events.read_bytes() == expected
    finally:
        if not resumed.is_closed:
            _close(resumed)


def test_close_after_replace_before_directory_fsync_remains_resumable(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_payload())
    scratch = tmp_path / "scratch"
    replacements = []

    def hook(name: str) -> None:
        if name == "checkpoint_replace":
            replacements.append(name)

    driver = ProjectionDriver(source, scratch, "op-1", "exp", hook=hook)
    try:
        for _ in range(5000):
            driver.advance(SliceIO(max_io_bytes=262_144, max_operations=1))
            if len(replacements) == 2:
                break
        else:
            pytest.fail("final checkpoint was not replaced")
        assert driver.checkpoint_generation == 1
        assert not driver.is_complete
        checkpoint_bytes = (scratch / "checkpoint.json").read_bytes()
        saved_size = json.loads(checkpoint_bytes)["spool"]["size"]
        expected = (scratch / "events.jsonl").read_bytes()
        assert saved_size == len(expected) > 0
    finally:
        _close(driver)
    assert (scratch / "checkpoint.json").read_bytes() == checkpoint_bytes
    assert (scratch / "events.jsonl").read_bytes() == expected
    resumed = ProjectionDriver(source, scratch, "op-1", "exp")
    try:
        _drive(resumed)
        assert resumed.is_complete
        assert (scratch / "events.jsonl").read_bytes() == expected
    finally:
        _close(resumed)
