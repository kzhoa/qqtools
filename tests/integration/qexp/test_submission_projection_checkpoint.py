"""Integration coverage for SubmissionProjection JSON checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner, Span
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import (
    FieldChunk,
    FieldEnd,
    ProjectionSummary,
    SubmissionProjection,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _payload(
    *,
    state: str = "committed",
    group: str | None = "exp",
    task_ids: list[str] | None = None,
    sequences: list[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> bytes:
    submission: dict[str, Any] = {
        "operation_id": "op-1",
        "target_group": group,
        "state": state,
        "resolved_context": {"task_ids": task_ids if task_ids is not None else ["task-a", "task-b"]},
        "commit_plan": {
            "group_membership_sequences": sequences if sequences is not None else [1, 2],
        },
    }
    if extra:
        submission.update(extra)
    return json.dumps({"submission": submission}, separators=(",", ":"), ensure_ascii=True).encode()


def _run(
    tmp_path: Path,
    payload: bytes,
    *,
    restart_each_step: bool = False,
    chunk_bytes: int = 7,
    budget: int = 13,
    max_fragments: int | None = None,
    checkpoints: list[dict[str, object]] | None = None,
) -> tuple[ProjectionSummary, list[FieldChunk], list[FieldEnd], SubmissionProjection]:
    path = tmp_path / "submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    source = path.open("rb")
    try:
        scanner = Scanner(source, lambda _span: None, chunk_bytes=chunk_bytes, emit_bytes=projection.feed)
        while True:
            result = scanner.step(budget, max_fragments=max_fragments)
            if result.is_complete:
                break
            if restart_each_step:
                scanner_snapshot = json.loads(json.dumps(scanner.snapshot()))
                projection_snapshot = json.loads(json.dumps(projection.snapshot()))
                if checkpoints is not None:
                    checkpoints.append(projection_snapshot)
                source.seek(scanner_snapshot["offset"])
                projection = SubmissionProjection.from_snapshot(
                    "op-1", "exp", chunks.append, ends.append, projection_snapshot
                )
                scanner = Scanner.from_snapshot(
                    source, lambda _span: None, scanner_snapshot, emit_bytes=projection.feed
                )
        summary = projection.finish()
        return summary, chunks, ends, projection
    finally:
        source.close()


def _fields(chunks: list[FieldChunk]) -> dict[tuple[str, int], bytes]:
    values: dict[tuple[str, int], bytearray] = {}
    finals: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.kind, chunk.ordinal)
        values.setdefault(key, bytearray()).extend(chunk.data)
        if chunk.is_final:
            finals.add(key)
    assert set(values) == finals
    return {key: bytes(value) for key, value in values.items()}


def _fingerprint(data: bytes) -> str:
    from qqtools.plugins.qexp.runtime.group_discovery.fingerprint import ChainedDigest

    digest = ChainedDigest()
    digest.update(data)
    return digest.hexdigest()


def test_checkpoint_resume_matches_uninterrupted_stream(tmp_path: Path) -> None:
    payload = _payload(
        extra={
            "unicode_noise": {"λ": "值", "nested": [{"submission": {"operation_id": "fake"}}]},
            "fake_selected": {"resolved_context": {"task_ids": ["fake"]}},
        }
    )
    expected = _run(tmp_path / "baseline", payload, chunk_bytes=3, budget=5)
    resumed = _run(tmp_path / "resumed", payload, restart_each_step=True, chunk_bytes=3, budget=5)

    assert resumed[0] == expected[0]
    assert resumed[1] == expected[1]
    assert resumed[2] == expected[2]


def test_checkpoint_can_restart_after_every_source_byte(tmp_path: Path) -> None:
    payload = _payload(extra={"ignored": {"nested": [[{"value": "x"}]]}})
    expected = _run(tmp_path / "baseline", payload, chunk_bytes=1, budget=1)
    resumed = _run(tmp_path / "resumed", payload, restart_each_step=True, chunk_bytes=1, budget=1)

    assert resumed[0] == expected[0]
    assert resumed[1] == expected[1]
    assert resumed[2] == expected[2]


def test_fresh_checkpoint_roundtrips_without_callbacks() -> None:
    called = {"chunk": 0, "end": 0}
    projection = SubmissionProjection(
        "op-1",
        "exp",
        lambda _chunk: called.__setitem__("chunk", called["chunk"] + 1),
        lambda _end: called.__setitem__("end", called["end"] + 1),
    )
    snapshot = projection.snapshot()
    restored = SubmissionProjection.from_snapshot("op-1", "exp", lambda _chunk: None, lambda _end: None, snapshot)

    assert restored.snapshot() == snapshot
    assert called == {"chunk": 0, "end": 0}


def test_checkpoint_roundtrip_preserves_escaped_selected_fields(tmp_path: Path) -> None:
    payload = (
        b'{"sub\\u006d\\u0069ssion":{"state":"comm\\u0069tted",'
        b'"target\\u005fgroup":"e\\u0078p","operation\\u005fid":"op\\u002d1",'
        b'"resolved\\u005fcontext":{"task\\u005fids":["task\\u0061"]},'
        b'"commit\\u005fplan":{"group_\\u006dembership\\u005fsequences":[123456789012345678901234567890]}}}'
    )
    summary, chunks, ends, _projection = _run(tmp_path, payload, restart_each_step=True, chunk_bytes=1, budget=7)

    assert summary == ProjectionSummary("committed", True, 1, 1)
    assert _fields(chunks) == {("task_id", 0): b"taska", ("sequence", 0): b"123456789012345678901234567890"}
    assert [end.kind for end in ends] == ["task_id", "sequence"]


@pytest.mark.parametrize(
    ("state", "group"),
    [("preparing", None), ("aborted", "foreign"), ("blocked", "exp")],
)
def test_checkpoint_preserves_noncandidate_metadata_and_null_plan(
    tmp_path: Path, state: str, group: str | None
) -> None:
    payload = json.dumps(
        {
            "submission": {
                "commit_plan": None,
                "state": state,
                "target_group": group,
                "operation_id": "op-1",
            }
        },
        separators=(",", ":"),
    ).encode()
    summary, chunks, ends, projection = _run(tmp_path, payload, restart_each_step=True, chunk_bytes=2, budget=3)

    assert summary == ProjectionSummary(state, group == "exp", 0, 0)
    assert chunks == []
    assert ends == []
    restored = SubmissionProjection.from_snapshot(
        "op-1", "exp", lambda _chunk: None, lambda _end: None, projection.snapshot()
    )
    assert restored.finish() == summary


def test_large_selected_id_and_sequence_restore_without_materializing_values(tmp_path: Path) -> None:
    task_id = "a" * 70_000
    sequence = "1" + ("7" * 70_000)
    payload = (
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["' + task_id.encode() + b'"]},'
        b'"commit_plan":{"group_membership_sequences":[' + sequence.encode() + b"]}}}"
    )
    checkpoints: list[dict[str, object]] = []
    summary, chunks, ends, projection = _run(
        tmp_path,
        payload,
        restart_each_step=True,
        chunk_bytes=65_536,
        budget=65_536,
        max_fragments=2,
        checkpoints=checkpoints,
    )

    assert summary == ProjectionSummary("committed", True, 1, 1)
    active_sizes = [len(json.dumps(checkpoint)) for checkpoint in checkpoints if checkpoint["active"] is not None]
    assert active_sizes and max(active_sizes) < 200_000
    assert sum(len(chunk.data) for chunk in chunks if chunk.kind == "task_id") == len(task_id)
    assert sum(len(chunk.data) for chunk in chunks if chunk.kind == "sequence") == len(sequence)
    assert [end.decoded_size for end in ends] == [len(task_id), len(sequence)]
    assert [end.digest for end in ends] == [_fingerprint(task_id.encode()), _fingerprint(sequence.encode())]


def test_escaped_large_id_has_same_checkpoint_fingerprint(tmp_path: Path) -> None:
    decoded = b"a" * 80_000
    plain = _payload(task_ids=[decoded.decode()], sequences=[1])
    escaped = (
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["' + b"\\u0061" * len(decoded) + b'"]},'
        b'"commit_plan":{"group_membership_sequences":[1]}}}'
    )
    plain_result = _run(
        tmp_path / "plain", plain, restart_each_step=True, chunk_bytes=65_536, budget=65_536, max_fragments=2
    )
    checkpoints: list[dict[str, object]] = []
    escaped_result = _run(
        tmp_path / "escaped",
        escaped,
        restart_each_step=True,
        chunk_bytes=65_536,
        budget=65_536,
        max_fragments=2,
        checkpoints=checkpoints,
    )

    assert _fields(plain_result[1]) == _fields(escaped_result[1])
    assert plain_result[2][0].digest == escaped_result[2][0].digest == _fingerprint(decoded)
    assert plain_result[2][0].start != escaped_result[2][0].start or plain_result[2][0].end != escaped_result[2][0].end
    active = [checkpoint["active"] for checkpoint in checkpoints if checkpoint["active"] is not None]
    assert any(token["decoder"]["state"] == "unicode" for token in active)
    assert any(token["decoder"]["decoded_size"] >= 65_536 for token in active)


def test_deep_ignored_nesting_restores_iterative_stack(tmp_path: Path) -> None:
    ignored: Any = {"leaf": "value"}
    for _ in range(80):
        ignored = {"object": [ignored]}
    payload = _payload(extra={"ignored": ignored})
    summary, chunks, ends, _projection = _run(
        tmp_path, payload, restart_each_step=True, chunk_bytes=3, budget=5, max_fragments=2
    )

    assert summary == ProjectionSummary("committed", True, 2, 2)
    assert _fields(chunks)[("task_id", 0)] == b"task-a"
    assert len(ends) == 4


def test_active_capture_shape_and_digest_mismatches_are_rejected(tmp_path: Path) -> None:
    task_id = b"a" * 70_000
    payload = (
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["' + task_id + b'"]},'
        b'"commit_plan":{"group_membership_sequences":[1]}}}'
    )
    path = tmp_path / "active.json"
    path.write_bytes(payload)
    projection = SubmissionProjection("op-1", "exp", lambda _chunk: None, lambda _end: None)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, chunk_bytes=65_536, emit_bytes=projection.feed)
        active_snapshot: dict[str, object] | None = None
        while not scanner.step(65_536, max_fragments=2).is_complete:
            candidate = projection.snapshot()
            if candidate["active"] is not None:
                active_snapshot = candidate
                break
    assert active_snapshot is not None

    bad_digest = copy.deepcopy(active_snapshot)
    bad_digest["active"]["decoder"]["decoded_size"] += 1  # type: ignore[index]
    wrong_handler = copy.deepcopy(active_snapshot)
    wrong_handler["active"]["handler"] = "sequence"  # type: ignore[index]
    bad_offsets = copy.deepcopy(active_snapshot)
    bad_offsets["active"]["end"] = bad_offsets["active"]["start"]  # type: ignore[index]
    for mutation in (bad_digest, wrong_handler, bad_offsets):
        with pytest.raises(ValueError):
            SubmissionProjection.from_snapshot("op-1", "exp", lambda _chunk: None, lambda _end: None, mutation)


def test_finished_checkpoint_is_idempotent_and_rejects_feed(tmp_path: Path) -> None:
    summary, _chunks, _ends, projection = _run(tmp_path, _payload())
    snapshot = projection.snapshot()
    restored = SubmissionProjection.from_snapshot("op-1", "exp", lambda _chunk: None, lambda _end: None, snapshot)

    assert restored.finish() == summary
    assert restored.finish() is restored.finish()
    with pytest.raises(ValueError, match="already finished"):
        restored.feed(Span(0, "{", 0, 1, True), b"{")


def test_snapshot_is_defensive_and_rejects_context_version_and_shape_mutations(tmp_path: Path) -> None:
    _summary, _chunks, _ends, projection = _run(tmp_path, _payload())
    original = projection.snapshot()
    changed = copy.deepcopy(original)
    changed["state"]["task_count"] = 999  # type: ignore[index]
    changed["stack"].append({})  # type: ignore[union-attr]
    assert projection.snapshot() == original

    mutations = [
        {**original, "version": 3},
        {**original, "fingerprint_version": 2},
        {**original, "expected_group": "other"},
        {**original, "unknown": True},
        {**original, "state": {**original["state"], "task_count": True}},  # type: ignore[index]
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            SubmissionProjection.from_snapshot("op-1", "exp", lambda _chunk: None, lambda _end: None, mutation)


def test_incomplete_checkpoint_restores_and_finish_rejects(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_bytes(b"{")
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=projection.feed)
        assert not scanner.step(1).is_complete
        assert scanner.step(1).is_complete
    snapshot = projection.snapshot()
    restored = SubmissionProjection.from_snapshot("op-1", "exp", chunks.append, ends.append, snapshot)
    with pytest.raises(ValueError):
        restored.finish()
    with pytest.raises(ValueError):
        restored.snapshot()


@pytest.mark.parametrize(
    "payload",
    [
        _payload(task_ids=["bad/id"], sequences=[1]),
        _payload(task_ids=["task-a", "task-b"], sequences=[1]),
        b'{"submission":{"operation_id":"op-1","operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1,]}}}',
        _payload(state="preparing", group="qqtools_internal"),
    ],
)
def test_delayed_errors_survive_checkpoint_resume(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    source = path.open("rb")
    try:
        scanner = Scanner(source, lambda _span: None, chunk_bytes=2, emit_bytes=projection.feed)
        try:
            while not scanner.step(3).is_complete:
                snapshot = json.loads(json.dumps(projection.snapshot()))
                scanner_snapshot = json.loads(json.dumps(scanner.snapshot()))
                source.seek(scanner_snapshot["offset"])
                projection = SubmissionProjection.from_snapshot("op-1", "exp", chunks.append, ends.append, snapshot)
                scanner = Scanner.from_snapshot(
                    source, lambda _span: None, scanner_snapshot, emit_bytes=projection.feed
                )
        except ValueError:
            pass
        with pytest.raises(ValueError):
            projection.finish()
    finally:
        source.close()


def test_reentrant_callback_and_callback_failure_poison_projection(tmp_path: Path) -> None:
    payload = _payload(task_ids=["task-a"], sequences=[1])
    path = tmp_path / "callback.json"
    path.write_bytes(payload)
    holder: dict[str, SubmissionProjection] = {}

    def reentrant(_chunk: FieldChunk) -> None:
        holder["projection"].snapshot()

    projection = SubmissionProjection("op-1", "exp", reentrant, lambda _end: None)
    holder["projection"] = projection
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=projection.feed)
        with pytest.raises(RuntimeError, match="busy"):
            while not scanner.step(65_536).is_complete:
                pass
    with pytest.raises(ValueError, match="invalid"):
        projection.finish()
    with pytest.raises(ValueError, match="invalid"):
        projection.snapshot()

    def fail(_chunk: FieldChunk) -> None:
        raise RuntimeError("callback failure")

    failing = SubmissionProjection("op-1", "exp", fail, lambda _end: None)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=failing.feed)
        with pytest.raises(RuntimeError, match="callback failure"):
            while not scanner.step(65_536).is_complete:
                pass
    with pytest.raises(ValueError, match="invalid"):
        failing.finish()


@pytest.mark.parametrize("operation", ["feed", "finish"])
def test_reentrant_feed_and_finish_callbacks_are_rejected(tmp_path: Path, operation: str) -> None:
    payload = _payload(task_ids=["task-a"], sequences=[1])
    path = tmp_path / f"reentrant-{operation}.json"
    path.write_bytes(payload)
    holder: dict[str, SubmissionProjection] = {}

    def reentrant(_chunk: FieldChunk) -> None:
        if operation == "feed":
            holder["projection"].feed(Span(0, "{", 0, 1, True), b"{")
        else:
            holder["projection"].finish()

    projection = SubmissionProjection("op-1", "exp", reentrant, lambda _end: None)
    holder["projection"] = projection
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=projection.feed)
        with pytest.raises(RuntimeError, match="busy"):
            while not scanner.step(65_536).is_complete:
                pass
    with pytest.raises(ValueError, match="invalid"):
        projection.finish()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_seen", False),
        ("operation_matches", False),
        ("target_group_seen", False),
        ("metadata_error", "invalid retained metadata"),
        ("shape_error", "invalid retained selected shape"),
        ("resolved_context_seen", False),
        ("task_ids_array_seen", False),
        ("commit_plan_seen", False),
        ("sequences_array_seen", False),
    ],
)
def test_finished_snapshot_cannot_bypass_final_validation(tmp_path: Path, field: str, value: object) -> None:
    _summary, _chunks, _ends, projection = _run(tmp_path, _payload())
    snapshot = projection.snapshot()
    snapshot["state"][field] = value
    with pytest.raises(ValueError):
        SubmissionProjection.from_snapshot("op-1", "exp", lambda _chunk: None, lambda _end: None, snapshot)


def test_invalid_long_identifier_remains_invalid_after_active_checkpoint(tmp_path: Path) -> None:
    checkpoints: list[dict[str, object]] = []
    payload = _payload(task_ids=["a" * 65_540 + "/" + "a" * 70_000], sequences=[1])
    with pytest.raises(ValueError, match="nonempty ASCII identifiers"):
        _run(
            tmp_path,
            payload,
            restart_each_step=True,
            chunk_bytes=65_536,
            budget=65_536,
            max_fragments=2,
            checkpoints=checkpoints,
        )
    assert any(
        checkpoint["active"] is not None
        and checkpoint["active"]["handler"] == "task_id"
        and not checkpoint["active"]["data_valid"]
        for checkpoint in checkpoints
    )
