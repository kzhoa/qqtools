"""Crash-boundary integration coverage for the provisional projection session."""

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
        b'{"meta":{"schema_version":6},"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["'
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


def _prepare_prefix(tmp_path: Path, payload: bytes, name: str) -> tuple[Path, Path, bytes, int]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / f"{name}-source.json"
    scratch = tmp_path / f"{name}-scratch"
    source.write_bytes(payload)
    session = _session_class().create(source, scratch, "op-1", "exp")
    try:
        for _ in range(100):
            session.step(max_bytes=65_536, max_fragments=2)
            session.checkpoint()
            checkpoint = json.loads((scratch / "checkpoint.json").read_text())
            active = checkpoint["projection"].get("active")
            if checkpoint["scanner"]["offset"] > 0 and isinstance(active, dict) and active.get("handler") == "task_id":
                checkpoint_bytes = (scratch / "checkpoint.json").read_bytes()
                saved_offset = checkpoint["scanner"]["offset"]
                break
        else:
            raise AssertionError("failed to establish a durable partial task_id checkpoint")
    finally:
        session.close()
    assert (scratch / "checkpoint.json").read_bytes() == checkpoint_bytes
    return source, scratch, checkpoint_bytes, saved_offset


def _checkpoint(scratch: Path) -> dict[str, Any]:
    value = json.loads((scratch / "checkpoint.json").read_text())
    assert set(value) == _CHECKPOINT_KEYS
    assert value["version"] == 1
    assert set(value["spool"]) == {"device", "inode", "size", "events"}
    assert isinstance(value["scanner"], dict)
    assert isinstance(value["projection"], dict)
    return value


def _crash_child(source: Path, scratch: Path, hook_name: str) -> subprocess.CompletedProcess[str]:
    code = r"""
import json
import os
import sys
from pathlib import Path

from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

source = Path(sys.argv[1])
scratch = Path(sys.argv[2])
hook_name = sys.argv[3]
checkpoint = json.loads((scratch / "checkpoint.json").read_text())
saved_offset = checkpoint["scanner"]["offset"]
original_read = BoundSource.read

def checked_read(bound_source, size):
    if bound_source.tell() < saved_offset:
        raise AssertionError(f"source reread before checkpoint offset {saved_offset}")
    return original_read(bound_source, size)

BoundSource.read = checked_read

def hook(event):
    if event == hook_name:
        os._exit(73)

session = ProjectionSession.resume(source, scratch, "op-1", "exp", hook=hook)
while not session.is_complete:
    session.step(max_bytes=65_536, max_fragments=2)
    session.checkpoint()
session.checkpoint()
session.close()
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(source), str(scratch), hook_name],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _replacement_child(source: Path, scratch: Path) -> subprocess.CompletedProcess[str]:
    code = r"""
import sys
from pathlib import Path

from qqtools.plugins.qexp.runtime.group_discovery.source_revision import SourceChangedError
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

try:
    ProjectionSession.resume(Path(sys.argv[1]), Path(sys.argv[2]), "op-1", "exp")
except SourceChangedError:
    print("SourceChangedError")
    raise SystemExit(0)
raise AssertionError("source replacement was accepted")
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(source), str(scratch)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_crashed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 73, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"


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


def _finish_recovery(source: Path, scratch: Path, saved_offset: int) -> dict[str, object]:
    code = r"""
import json
import sys
from pathlib import Path

from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

source = Path(sys.argv[1])
scratch = Path(sys.argv[2])
checkpoint = json.loads((scratch / "checkpoint.json").read_text())
saved_offset = checkpoint["scanner"]["offset"]
if saved_offset != int(sys.argv[3]):
    raise AssertionError("checkpoint offset changed before recovery")
original_read = BoundSource.read

def checked_read(bound_source, size):
    if bound_source.tell() < saved_offset:
        raise AssertionError(f"source reread before checkpoint offset {saved_offset}")
    return original_read(bound_source, size)

BoundSource.read = checked_read
session = ProjectionSession.resume(source, scratch, "op-1", "exp")
try:
    while not session.is_complete:
        session.step(max_bytes=65_536, max_fragments=2)
        session.checkpoint()
    session.checkpoint()
    summary = session.summary
    print(json.dumps({
        "state": summary.state,
        "matches_group": summary.matches_group,
        "task_count": summary.task_count,
        "sequence_count": summary.sequence_count,
    }, sort_keys=True))
finally:
    session.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source), str(scratch), str(saved_offset)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    return json.loads(result.stdout)


@pytest.mark.parametrize("hook_name", _HOOKS)
def test_projection_session_recovers_each_durability_hook(tmp_path: Path, hook_name: str):
    payload, expected_fields = _fixture_payload()
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    _assert_events(baseline_events, expected_fields)

    source, scratch, _checkpoint_bytes, _saved_offset = _prepare_prefix(tmp_path / hook_name, payload, hook_name)
    _assert_crashed(_crash_child(source, scratch, hook_name))
    checkpoint = _checkpoint(scratch)
    recovered_summary = _finish_recovery(source, scratch, checkpoint["scanner"]["offset"])
    assert recovered_summary == baseline_summary
    recovered_events = (scratch / "events.jsonl").read_bytes()
    assert recovered_events == baseline_events
    _assert_events(recovered_events, expected_fields)
    completed = _checkpoint(scratch)
    assert completed["scanner"]["is_complete"] is True
    assert completed["projection"]["summary"] is not None


def test_projection_session_repeated_crash_does_not_accumulate_speculative_suffix(tmp_path: Path):
    payload, expected_fields = _fixture_payload()
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    source, scratch, checkpoint_bytes, _saved_offset = _prepare_prefix(tmp_path / "repeat", payload, "repeat")

    for _ in range(2):
        _assert_crashed(_crash_child(source, scratch, "spool_write"))
        checkpoint = _checkpoint(scratch)
        assert (scratch / "checkpoint.json").read_bytes() == checkpoint_bytes
        assert checkpoint["scanner"]["offset"] > 0

    assert _finish_recovery(source, scratch, checkpoint["scanner"]["offset"]) == baseline_summary
    events = (scratch / "events.jsonl").read_bytes()
    assert events == baseline_events
    _assert_events(events, expected_fields)


def test_projection_session_rejects_identical_byte_source_replacement(tmp_path: Path):
    payload, _expected_fields = _fixture_payload()
    source, scratch, checkpoint_bytes, _saved_offset = _prepare_prefix(tmp_path, payload, "replacement")
    replacement = tmp_path / "replacement-source.tmp"
    original_inode = source.stat().st_ino
    replacement.write_bytes(source.read_bytes())
    os.replace(replacement, source)
    assert source.stat().st_ino != original_inode

    result = _replacement_child(source, scratch)
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert result.stdout.strip() == "SourceChangedError"
    assert (scratch / "checkpoint.json").read_bytes() == checkpoint_bytes
