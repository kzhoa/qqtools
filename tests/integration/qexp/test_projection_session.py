"""Integration coverage for the durable provisional projection session."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from tests.helpers.qexp.qexp_projection_session import ProjectionSession, ProjectionSummary

pytestmark = pytest.mark.integration


def _payload(*, large: bool = False, truncated: bool = False) -> bytes:
    task_a = "a" * 70_000 if large else "task-a"
    task_b = "\\u0062" * 70_000 if large else "task-b"
    sequence_b = "7" * 70_000 if large else "2"
    payload = (
        b'{"meta":{"schema_version":6},"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["'
        + task_a.encode()
        + b'","'
        + task_b.encode()
        + b'"]},"commit_plan":{"group_membership_sequences":[1,'
        + sequence_b.encode()
        + b"]}}}"
    )
    return payload[:-1] if truncated else payload


def _run_to_completion(source: Path, scratch: Path, *, max_fragments: int = 64) -> tuple[ProjectionSummary, bytes]:
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        while not session.is_complete:
            session.step(65_536, max_fragments=max_fragments)
        summary = session.summary
        assert summary is not None
        session.checkpoint()
        events = (scratch / "events.jsonl").read_bytes()
        return summary, events
    finally:
        session.close()


def _event_lines(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.splitlines()]


def test_create_resume_empty_prefix_and_checkpoint_schema(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())

    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    assert not session.is_complete
    assert session.summary is None
    assert (scratch / "events.jsonl").read_bytes() == b""
    checkpoint = json.loads((scratch / "checkpoint.json").read_bytes())
    assert set(checkpoint) == {
        "version",
        "source_path",
        "source_revision",
        "expected_operation_id",
        "expected_group",
        "spool",
        "scanner",
        "projection",
    }
    session.close()

    resumed = ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert not resumed.is_complete
    assert resumed.summary is None
    resumed.close()


def test_large_plain_and_escaped_fields_match_uninterrupted_event_baseline(tmp_path: Path) -> None:
    payload = _payload(large=True)
    baseline_source = tmp_path / "baseline.json"
    resumed_source = tmp_path / "resumed.json"
    baseline_source.write_bytes(payload)
    resumed_source.write_bytes(payload)

    baseline_summary, baseline_events = _run_to_completion(
        baseline_source, tmp_path / "baseline-scratch", max_fragments=64
    )

    scratch = tmp_path / "resumed-scratch"
    session = ProjectionSession.create(resumed_source, scratch, "op-1", "exp")
    try:
        for _ in range(4):
            if session.is_complete:
                break
            session.step(65_536, max_fragments=2)
            session.checkpoint()
            session.close()
            session = ProjectionSession.resume(resumed_source, scratch, "op-1", "exp")
        while not session.is_complete:
            session.step(65_536, max_fragments=2)
        assert session.summary == baseline_summary
        session.checkpoint()
        resumed_events = (scratch / "events.jsonl").read_bytes()
    finally:
        session.close()

    assert resumed_events == baseline_events
    events = _event_lines(resumed_events)
    assert events
    for event in events:
        if event["type"] == "chunk":
            assert set(event) == {"type", "kind", "ordinal", "data_b64", "is_final"}
            assert len(base64.b64decode(event["data_b64"])) <= 65_536
        else:
            assert set(event) == {"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"}


def test_resume_truncates_arbitrary_trailing_spool_bytes(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    session.step(65_536, max_fragments=2)
    session.checkpoint()
    session.close()

    events_path = scratch / "events.jsonl"
    saved = events_path.read_bytes()
    events_path.open("ab").write(b"trailing bytes")

    resumed = ProjectionSession.resume(source, scratch, "op-1", "exp")
    try:
        assert events_path.read_bytes() == saved
    finally:
        resumed.close()


@pytest.mark.parametrize("mutation", ["short", "replace", "symlink"])
def test_invalid_spool_rejects_without_mutating_source(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / f"{mutation}.json"
    scratch = tmp_path / f"{mutation}-scratch"
    source_bytes = _payload()
    source.write_bytes(source_bytes)
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    session.step(65_536)
    session.checkpoint()
    session.close()
    events_path = scratch / "events.jsonl"
    original_events = events_path.read_bytes()
    if mutation == "short":
        os.truncate(events_path, len(original_events) - 1)
    elif mutation == "replace":
        replacement = tmp_path / f"{mutation}-replacement.jsonl"
        replacement.write_bytes(original_events)
        os.replace(replacement, events_path)
    else:
        target = tmp_path / f"{mutation}-target.jsonl"
        target.write_bytes(original_events)
        events_path.unlink()
        events_path.symlink_to(target)

    with pytest.raises((OSError, ValueError)):
        ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert source.read_bytes() == source_bytes


def test_replaced_source_rejects_before_spool_truncation(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source_bytes = _payload()
    source.write_bytes(source_bytes)
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    session.step(65_536)
    session.checkpoint()
    session.close()
    events_path = scratch / "events.jsonl"
    events_path.open("ab").write(b"uncheckpointed suffix")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(source_bytes)
    os.replace(replacement, source)
    with pytest.raises(ValueError):
        ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert events_path.read_bytes().endswith(b"uncheckpointed suffix")


@pytest.mark.parametrize("mutation", ["malformed", "context", "offset", "parser"])
def test_invalid_checkpoint_rejects_without_fallback_or_spool_mutation(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / f"{mutation}.json"
    scratch = tmp_path / f"{mutation}-scratch"
    source.write_bytes(_payload())
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    session.checkpoint()
    session.close()
    checkpoint_path = scratch / "checkpoint.json"
    original_checkpoint = checkpoint_path.read_bytes()
    original_events = (scratch / "events.jsonl").read_bytes()
    if mutation == "malformed":
        checkpoint_path.write_bytes(b'{"version":1,"version":1}')
    else:
        checkpoint = json.loads(original_checkpoint)
        if mutation == "context":
            checkpoint["expected_group"] = "other"
        elif mutation == "offset":
            checkpoint["scanner"]["offset"] = source.stat().st_size + 1
        else:
            checkpoint["projection"]["state"]["last_token_id"] = 0
        checkpoint_path.write_text(json.dumps(checkpoint, separators=(",", ":")))

    with pytest.raises(ValueError):
        ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert (scratch / "events.jsonl").read_bytes() == original_events


def test_hook_failure_poison_closes_session_and_prior_checkpoint_resumes(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    calls: list[str] = []

    def fail_on_spool_write(name: str) -> None:
        calls.append(name)
        if name == "spool_write":
            raise RuntimeError("injected spool failure")

    session = ProjectionSession.create(source, scratch, "op-1", "exp", hook=fail_on_spool_write)
    with pytest.raises(RuntimeError, match="injected spool failure"):
        session.step(65_536)
    with pytest.raises(ValueError):
        session.step(1)
    assert "spool_write" in calls

    resumed = ProjectionSession.resume(source, scratch, "op-1", "exp")
    try:
        while not resumed.is_complete:
            resumed.step(65_536, max_fragments=2)
        assert resumed.summary == ProjectionSummary("committed", True, 2, 2)
    finally:
        resumed.close()


def test_hook_reentry_is_rejected_and_poisoned(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    holder: dict[str, ProjectionSession] = {}

    def reenter(name: str) -> None:
        if name == "spool_write":
            holder["session"].checkpoint()

    session = ProjectionSession.create(source, scratch, "op-1", "exp", hook=reenter)
    holder["session"] = session
    with pytest.raises(RuntimeError, match="busy"):
        session.step(65_536)
    with pytest.raises(ValueError):
        session.checkpoint()


def test_incomplete_json_never_certifies(tmp_path: Path) -> None:
    source = tmp_path / "truncated.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload(truncated=True))
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    with pytest.raises(ValueError):
        while not session.is_complete:
            session.step(65_536, max_fragments=2)
    assert not session.is_complete
    assert session.summary is None


def test_finished_checkpoint_is_idempotent_without_duplicate_events(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        while not session.is_complete:
            session.step(65_536, max_fragments=2)
        before = (scratch / "events.jsonl").read_bytes()
        session.checkpoint()
        once = (scratch / "events.jsonl").read_bytes()
        session.checkpoint()
        twice = (scratch / "events.jsonl").read_bytes()
    finally:
        session.close()
    assert once == before == twice


def test_stale_temp_hardlink_is_unlinked_without_modifying_sentinel(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    sentinel = tmp_path / "sentinel"
    sentinel_bytes = b"sentinel bytes remain"
    sentinel.write_bytes(sentinel_bytes)
    temp = scratch / "checkpoint.tmp"
    os.link(sentinel, temp)
    sentinel_stat = sentinel.stat()
    try:
        session.checkpoint()
    finally:
        session.close()
    assert sentinel.read_bytes() == sentinel_bytes
    assert sentinel.stat().st_ino == sentinel_stat.st_ino
    assert not temp.exists()


def test_relative_source_and_scratch_survive_cwd_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "submission.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    with ProjectionSession.create(Path("submission.json"), Path("scratch"), "op-1", "exp") as session:
        monkeypatch.chdir(other)
        session.step(13)
        session.checkpoint()
    assert not (other / "scratch").exists()
    with ProjectionSession.resume(source, scratch, "op-1", "exp") as resumed:
        while not resumed.is_complete:
            resumed.step(65_536)
        resumed.checkpoint()
        assert resumed.summary == ProjectionSummary("committed", True, 2, 2)


def test_finished_checkpoint_requires_full_source_offset_before_truncation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    with ProjectionSession.create(source, scratch, "op-1", "exp") as session:
        while not session.is_complete:
            session.step()
        session.checkpoint()
    checkpoint_path = scratch / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["scanner"]["is_complete"] is True
    assert checkpoint["scanner"]["offset"] == checkpoint["source_revision"]["size"]
    checkpoint["scanner"]["offset"] -= 1
    checkpoint_path.write_text(json.dumps(checkpoint))
    events_path = scratch / "events.jsonl"
    with events_path.open("ab") as output:
        output.write(b"unconfirmed suffix")
    before = events_path.read_bytes()

    with pytest.raises(ValueError, match="completed scanner offset"):
        ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert events_path.read_bytes() == before
    assert source.read_bytes() == _payload()
