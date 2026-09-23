"""Integration coverage for the one-pass Submission projection prototype."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.group_discovery.fingerprint import ChainedDigest
from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner, Span
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import (
    FieldChunk,
    FieldEnd,
    ProjectionSummary,
    SubmissionProjection,
)
from qqtools.plugins.qexp.runtime.paths import submission_path

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _stream(
    tmp_path: Path,
    payload: bytes,
    *,
    operation_id: str = "op-1",
    group: str = "exp",
    chunk_bytes: int = 7,
    budget: int = 13,
    max_fragments: int | None = None,
    emit_chunk: Callable[[FieldChunk], None] | None = None,
    emit_end: Callable[[FieldEnd], None] | None = None,
) -> tuple[ProjectionSummary, list[FieldChunk], list[FieldEnd]]:
    path = tmp_path / "submission.json"
    path.write_bytes(payload)
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection(operation_id, group, emit_chunk or chunks.append, emit_end or ends.append)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, chunk_bytes=chunk_bytes, emit_bytes=projection.feed)
        while not scanner.step(budget, max_fragments=max_fragments).is_complete:
            pass
    return projection.finish(), chunks, ends


def _decoded_fields(chunks: list[FieldChunk]) -> dict[tuple[str, int], bytes]:
    values: dict[tuple[str, int], bytearray] = {}
    finals: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.kind, chunk.ordinal)
        values.setdefault(key, bytearray()).extend(chunk.data)
        if chunk.is_final:
            assert key not in finals
            finals.add(key)
    assert set(values) == finals
    return {key: bytes(value) for key, value in values.items()}


def _fingerprint(data: bytes) -> str:
    """Compute the independent ChainedDigest v1 oracle for normalized bytes."""
    digest = ChainedDigest()
    digest.update(data)
    return digest.hexdigest()


def _fixture_submission(
    *,
    state: str = "committed",
    group: str | None = "exp",
    task_ids: list[str] | None = None,
    sequences: list[int] | None = None,
    include_selected: bool = True,
) -> dict[str, Any]:
    submission: dict[str, Any] = {
        "operation_id": "op-1",
        "target_group": group,
        "state": state,
    }
    if include_selected:
        submission["resolved_context"] = {"task_ids": task_ids or ["task-a", "task-b"]}
        submission["commit_plan"] = {"group_membership_sequences": sequences or [1, 2]}
    elif state != "committed":
        submission["commit_plan"] = None
    return {"submission": submission}


@pytest.mark.parametrize("chunk_bytes", [1, 7, 65_536])
@pytest.mark.parametrize("budget", [13, 65_536])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_projection_streams_selected_fields_and_matches_json_oracle(
    tmp_path: Path, chunk_bytes: int, budget: int, ensure_ascii: bool
) -> None:
    payload_object = _fixture_submission()
    payload_object["unrelated"] = {
        "submission": {
            "operation_id": "fake",
            "resolved_context": {"task_ids": ["fake-id"]},
        },
        "deep": [{"commit_plan": {"group_membership_sequences": [999]}}],
    }
    payload_object["unicode_noise"] = {"λ": "值", "array": [{"task_ids": ["fake"]}]}
    payload = json.dumps(payload_object, ensure_ascii=ensure_ascii, separators=(",", ":")).encode()

    summary, chunks, ends = _stream(tmp_path, payload, chunk_bytes=chunk_bytes, budget=budget)

    assert summary == ProjectionSummary("committed", True, 2, 2)
    assert _decoded_fields(chunks) == {
        ("task_id", 0): b"task-a",
        ("task_id", 1): b"task-b",
        ("sequence", 0): b"1",
        ("sequence", 1): b"2",
    }
    assert [end.ordinal for end in ends] == [0, 1, 0, 1]
    assert [(end.kind, end.decoded_size) for end in ends] == [
        ("task_id", 6),
        ("task_id", 6),
        ("sequence", 1),
        ("sequence", 1),
    ]
    assert all(end.end > end.start and end.end - end.start <= 65_536 for end in ends)
    for end in ends:
        source_scalar = payload[end.start : end.end]
        assert json.loads(source_scalar) in {"task-a", "task-b", 1, 2}
        assert end.digest == _fingerprint(str(json.loads(source_scalar)).encode())


def test_projection_accepts_unicode_escaped_known_keys_and_ascii_values(tmp_path: Path) -> None:
    payload = (
        b'{"sub\\u006d\\u0069ssion":{"commit\\u005fplan":{"group_\\u006dembership\\u005fsequences":[1]},'
        b'"resolved\\u005fcontext":{"task\\u005fids":["task\\u0031"]},'
        b'"state":"comm\\u0069tted","target\\u005fgroup":"e\\u0078p",'
        b'"operation\\u005fid":"op\\u002d1"}}'
    )

    summary, chunks, _ends = _stream(tmp_path, payload)

    assert summary == ProjectionSummary("committed", True, 1, 1)
    assert _decoded_fields(chunks) == {("task_id", 0): b"task1", ("sequence", 0): b"1"}


@pytest.mark.parametrize("state", ["preparing", "aborted", "blocked"])
def test_noncommitted_states_do_not_require_selected_arrays(tmp_path: Path, state: str) -> None:
    payload = json.dumps(_fixture_submission(state=state, include_selected=False), separators=(",", ":")).encode()

    summary, chunks, ends = _stream(tmp_path, payload)

    assert summary == ProjectionSummary(state, True, 0, 0)
    assert chunks == []
    assert ends == []


@pytest.mark.parametrize("group", ["foreign", None])
def test_committed_foreign_or_null_group_is_a_nonmatching_candidate(tmp_path: Path, group: str | None) -> None:
    payload = json.dumps(_fixture_submission(group=group), separators=(",", ":")).encode()

    summary, chunks, ends = _stream(tmp_path, payload)

    assert summary == ProjectionSummary("committed", False, 2, 2)
    assert _decoded_fields(chunks)[("task_id", 0)] == b"task-a"
    assert len(ends) == 4


@pytest.mark.parametrize("group", [".foreign", "-foreign", "experiments", "qqtools_internal"])
def test_projection_rejects_invalid_foreign_group_metadata(tmp_path: Path, group: str) -> None:
    payload = json.dumps(_fixture_submission(state="preparing", group=group), separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="target_group"):
        _stream(tmp_path, payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a",]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]}}} {"x":1}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[-1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1.5]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["bad/id"]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[0]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":[1]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":["1"]}}}',
    ],
)
def test_projection_rejects_structural_and_selected_shape_errors(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(ValueError):
        _stream(tmp_path, payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"submission":{"operation_id":"op-1","operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","state":"blocked","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]}}}',
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed","resolved_context":{"task_ids":["a"]},"commit_plan":{"group_membership_sequences":[1]},"commit_plan":{"group_membership_sequences":[1]}}}',
    ],
)
def test_projection_rejects_ambiguous_relevant_duplicates(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        _stream(tmp_path, payload)


def test_projection_rejects_missing_arrays_and_mismatched_operation(tmp_path: Path) -> None:
    missing = json.dumps(_fixture_submission(include_selected=False), separators=(",", ":")).encode()
    mismatch_object = _fixture_submission()
    mismatch_object["submission"]["operation_id"] = "other"
    mismatch = json.dumps(mismatch_object, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="missing resolved_context"):
        _stream(tmp_path, missing)
    with pytest.raises(ValueError, match="does not match"):
        _stream(tmp_path, mismatch)


def test_projection_defers_provisional_outputs_until_truncated_finish_fails(tmp_path: Path) -> None:
    payload = json.dumps(_fixture_submission(), separators=(",", ":")).encode()
    path = tmp_path / "truncated.json"
    path.write_bytes(payload[:-2])
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=projection.feed)
        while not scanner.step(13).is_complete:
            pass
    assert chunks
    with pytest.raises(ValueError):
        projection.finish()
    with pytest.raises(ValueError):
        projection.feed(Span(0, "{", 0, 1, True), b"{")


@pytest.mark.parametrize("length", [65_535, 65_536, 131_073])
def test_large_selected_id_is_decoded_incrementally(tmp_path: Path, length: int) -> None:
    task_id = "a" * length
    payload = json.dumps(_fixture_submission(task_ids=[task_id], sequences=[1]), separators=(",", ":")).encode()
    counters = {"chunks": 0, "final": 0, "bytes": 0, "max": 0}
    final_by_kind = {"task_id": 0, "sequence": 0}
    empty_final_marker = {"seen": False}
    ends: list[FieldEnd] = []

    def on_chunk(chunk: FieldChunk) -> None:
        counters["chunks"] += 1
        counters["final"] += chunk.is_final
        final_by_kind[chunk.kind] += chunk.is_final
        empty_final_marker["seen"] |= chunk.kind == "task_id" and chunk.is_final and not chunk.data
        counters["bytes"] += len(chunk.data) if chunk.kind == "task_id" else 0
        counters["max"] = max(counters["max"], len(chunk.data))

    summary, _ignored_chunks, _ignored_ends = _stream(
        tmp_path,
        payload,
        chunk_bytes=65_536,
        budget=65_536,
        emit_chunk=on_chunk,
        emit_end=ends.append,
    )

    assert summary == ProjectionSummary("committed", True, 1, 1)
    assert counters["bytes"] == length
    assert final_by_kind == {"task_id": 1, "sequence": 1}
    assert counters["max"] <= 65_536
    assert [end.decoded_size for end in ends if end.kind == "task_id"] == [length]
    if length == 65_535:
        assert empty_final_marker["seen"]


def test_task_id_fingerprint_uses_decoded_bytes_across_source_encodings(tmp_path: Path) -> None:
    decoded_id = b"a" * 70_000
    escaped_id = b"\\u0061" * len(decoded_id)
    prefix = (
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":['
    )
    suffix = b']},"commit_plan":{"group_membership_sequences":[1]}}}'
    plain_payload = prefix + b'"' + decoded_id + b'"' + suffix
    escaped_payload = prefix + b'"' + escaped_id + b'"' + suffix

    plain_root = tmp_path / "plain"
    escaped_root = tmp_path / "escaped"
    plain_root.mkdir()
    escaped_root.mkdir()
    plain_summary, plain_chunks, plain_ends = _stream(
        plain_root,
        plain_payload,
        chunk_bytes=7,
        budget=13,
        max_fragments=2,
    )
    escaped_summary, escaped_chunks, escaped_ends = _stream(
        escaped_root,
        escaped_payload,
        chunk_bytes=65_536,
        budget=65_536,
        max_fragments=64,
    )

    assert plain_summary == escaped_summary == ProjectionSummary("committed", True, 1, 1)
    assert (
        _decoded_fields(plain_chunks)
        == _decoded_fields(escaped_chunks)
        == {
            ("task_id", 0): decoded_id,
            ("sequence", 0): b"1",
        }
    )
    assert [(chunk.kind, chunk.ordinal, chunk.is_final) for chunk in plain_chunks if chunk.is_final] == [
        ("task_id", 0, True),
        ("sequence", 0, True),
    ]
    assert [(chunk.kind, chunk.ordinal, chunk.is_final) for chunk in escaped_chunks if chunk.is_final] == [
        ("task_id", 0, True),
        ("sequence", 0, True),
    ]
    plain_task_end = next(end for end in plain_ends if end.kind == "task_id")
    escaped_task_end = next(end for end in escaped_ends if end.kind == "task_id")
    assert plain_task_end.digest == escaped_task_end.digest == _fingerprint(decoded_id)
    assert plain_task_end.decoded_size == escaped_task_end.decoded_size == len(decoded_id)
    assert plain_task_end.start != escaped_task_end.start or plain_task_end.end != escaped_task_end.end
    assert (plain_root / "submission.json").read_bytes()[
        plain_task_end.start : plain_task_end.end
    ] == b'"' + decoded_id + b'"'
    assert (escaped_root / "submission.json").read_bytes()[
        escaped_task_end.start : escaped_task_end.end
    ] == b'"' + escaped_id + b'"'
    for ends, payload in ((plain_ends, plain_payload), (escaped_ends, escaped_payload)):
        sequence_end = next(end for end in ends if end.kind == "sequence")
        assert payload[sequence_end.start : sequence_end.end] == b"1"
    plain_sequence_end = next(end for end in plain_ends if end.kind == "sequence")
    escaped_sequence_end = next(end for end in escaped_ends if end.kind == "sequence")
    assert plain_sequence_end.digest == escaped_sequence_end.digest == _fingerprint(b"1")


@pytest.mark.parametrize("count", [128, pytest.param(100_000, marks=pytest.mark.stress)])
def test_large_selected_arrays_do_not_require_array_retention(tmp_path: Path, count: int) -> None:
    payload_object = _fixture_submission(
        task_ids=[f"task-{index}" for index in range(count)],
        sequences=list(range(1, count + 1)),
    )
    payload_object["command_payload"] = [
        {"resolved_context": {"task_ids": ["fake"]}, "commit_plan": {"group_membership_sequences": [99]}}
        for _ in range(8)
    ]
    payload = json.dumps(payload_object, separators=(",", ":")).encode()
    counters = {"chunks": 0, "ends": 0, "task_bytes": 0, "sequence_bytes": 0}

    def on_chunk(chunk: FieldChunk) -> None:
        counters["chunks"] += 1
        counters["task_bytes"] += len(chunk.data) if chunk.kind == "task_id" else 0
        counters["sequence_bytes"] += len(chunk.data) if chunk.kind == "sequence" else 0

    def on_end(_end: FieldEnd) -> None:
        counters["ends"] += 1

    tracemalloc.start()
    tracemalloc.clear_traces()
    try:
        summary, _chunks, _ends = _stream(
            tmp_path,
            payload,
            chunk_bytes=65_536,
            budget=65_536,
            max_fragments=64,
            emit_chunk=on_chunk,
            emit_end=on_end,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert summary == ProjectionSummary("committed", True, count, count)
    assert counters["chunks"] == count * 2
    assert counters["ends"] == count * 2
    assert counters["task_bytes"] == sum(len(f"task-{index}") for index in range(count))
    assert counters["sequence_bytes"] == sum(len(str(index)) for index in range(1, count + 1))
    assert peak < 2 * 1024 * 1024


def test_long_decimal_sequence_is_streamed_without_integer_conversion(tmp_path: Path) -> None:
    digits = b"1" + (b"7" * 131_072)
    payload = (
        b'{"submission":{"operation_id":"op-1","target_group":"exp","state":"committed",'
        b'"resolved_context":{"task_ids":["task-a"]},"commit_plan":{"group_membership_sequences":[' + digits + b"]}}}"
    )

    summary, chunks, ends = _stream(tmp_path, payload, chunk_bytes=65_536, budget=65_536)

    fields = _decoded_fields(chunks)
    assert summary == ProjectionSummary("committed", True, 1, 1)
    assert fields[("sequence", 0)] == digits
    sequence_end = next(end for end in ends if end.kind == "sequence")
    assert sequence_end.decoded_size == len(digits)
    assert sequence_end.digest == _fingerprint(digits)


def test_projection_finish_is_idempotent_and_feed_after_finish_rejects(tmp_path: Path) -> None:
    payload = json.dumps(_fixture_submission(), separators=(",", ":")).encode()
    path = tmp_path / "idempotent.json"
    path.write_bytes(payload)
    projection = SubmissionProjection("op-1", "exp", lambda _chunk: None, lambda _end: None)
    with path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, emit_bytes=projection.feed)
        while not scanner.step(13).is_complete:
            pass

    first = projection.finish()
    assert projection.finish() is first
    with pytest.raises(ValueError, match="already finished"):
        projection.feed(Span(0, "{", 0, 1, True), b"{")


def test_projection_rejects_invalid_expected_context() -> None:
    with pytest.raises(ValueError):
        SubmissionProjection("", "exp", lambda _chunk: None, lambda _end: None)
    with pytest.raises(ValueError):
        SubmissionProjection("op-1", "", lambda _chunk: None, lambda _end: None)


def test_projection_accepts_current_batch_submit_writer_record(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    manifest = tmp_path / "explicit.yaml"
    manifest.write_text(
        "tasks:\n  - task_id: explicit-a\n    command: [echo, a]\n  - task_id: explicit-b\n    command: [echo, b]\n",
        encoding="utf-8",
    )
    tasks = batch_submit(cfg, manifest, group="exp")
    operation_id = tasks[0].submission_operation_id
    assert operation_id is not None
    expected_ids = [task.task_id for task in tasks]
    expected_sequences = [task.group_membership_sequence for task in tasks]
    source_path = submission_path(cfg.shared_root, operation_id)
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection(operation_id, "exp", chunks.append, ends.append)

    with source_path.open("rb") as source:
        scanner = Scanner(source, lambda _span: None, chunk_bytes=7, emit_bytes=projection.feed)
        while not scanner.step(13).is_complete:
            pass
    summary = projection.finish()

    fields = _decoded_fields(chunks)
    assert summary == ProjectionSummary("committed", True, 2, 2)
    assert [fields["task_id", index].decode() for index in range(2)] == expected_ids
    assert [fields["sequence", index].decode() for index in range(2)] == [str(value) for value in expected_sequences]
    assert {end.kind for end in ends} == {"task_id", "sequence"}
