"""Integration coverage for source schema qualification in qexp projection."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

import tests.helpers.qexp.qexp_projection_session as projection_session_module
from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import (
    SUPPORTED_SOURCE_SCHEMA_VERSION,
    FieldChunk,
    FieldEnd,
    ProjectionSummary,
    SubmissionProjection,
)
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

_EXPECTED_SUMMARY = ProjectionSummary("committed", True, 2, 2)
_INVALID_VERSION_LITERALS = (
    b"7",
    b"5",
    b"0",
    b"-1",
    b"true",
    b"null",
    b'"6"',
    b"6.0",
    b"6e0",
)
_INVALID_CONTEXTS = (
    ("committed", "exp", True),
    ("committed", "foreign", True),
    ("aborted", "exp", False),
)


def _submission_bytes(*, state: str, group: str | None, selected: bool) -> bytes:
    group_value = b"null" if group is None else json.dumps(group, separators=(",", ":")).encode("ascii")
    fields = [
        b'"operation_id":"op-1"',
        b'"target_group":' + group_value,
        b'"state":' + json.dumps(state, separators=(",", ":")).encode("ascii"),
    ]
    if selected:
        fields.extend(
            [
                b'"resolved_context":{"task_ids":["task-a","task-b"]}',
                b'"commit_plan":{"group_membership_sequences":[1,2]}',
            ]
        )
    return b"{" + b",".join(fields) + b"}"


def _root_payload(
    *,
    state: str = "committed",
    group: str | None = "exp",
    selected: bool = True,
    meta_present: bool = False,
    schema_literal: bytes | None = b"6",
    meta_raw: bytes | None = None,
    meta_first: bool = False,
    escaped_meta_keys: bool = False,
    duplicate_meta: bool = False,
    duplicate_schema: bool = False,
    extra_root: bytes | None = None,
) -> bytes:
    submission_entry = b'"submission":' + _submission_bytes(state=state, group=group, selected=selected)
    entries = [submission_entry]
    if extra_root is not None:
        entries.append(extra_root)
    if meta_present:
        meta_key = b'"me\\u0074a"' if escaped_meta_keys else b'"meta"'
        schema_key = b'"schema_\\u0076ersion"' if escaped_meta_keys else b'"schema_version"'
        if meta_raw is not None:
            meta_value = meta_raw
        else:
            schema_fields = []
            if schema_literal is not None:
                schema_fields.append(schema_key + b":" + schema_literal)
                if duplicate_schema:
                    schema_fields.append(schema_key + b":" + schema_literal)
            meta_value = b"{" + b",".join(schema_fields) + b"}"
        meta_entry = meta_key + b":" + meta_value
        if meta_first:
            entries.insert(0, meta_entry)
        else:
            entries.append(meta_entry)
        if duplicate_meta:
            entries.append(meta_entry)
    return b"{" + b",".join(entries) + b"}"


def _run_projection(
    payload: bytes,
    *,
    chunk_bytes: int = 7,
    budget: int = 13,
    max_fragments: int | None = None,
    restart_each_step: bool = False,
) -> tuple[ProjectionSummary, list[FieldChunk], list[FieldEnd], SubmissionProjection]:
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    source = io.BytesIO(payload)
    scanner = Scanner(source, lambda _span: None, chunk_bytes=chunk_bytes, emit_bytes=projection.feed)
    while not scanner.step(budget, max_fragments=max_fragments).is_complete:
        if restart_each_step:
            scanner_state = json.loads(json.dumps(scanner.snapshot()))
            projection_state = json.loads(json.dumps(projection.snapshot()))
            source.seek(scanner_state["offset"])
            projection = SubmissionProjection.from_snapshot("op-1", "exp", chunks.append, ends.append, projection_state)
            scanner = Scanner.from_snapshot(source, lambda _span: None, scanner_state, emit_bytes=projection.feed)
    return projection.finish(), chunks, ends, projection


def _session_paths(tmp_path: Path, payload: bytes, name: str = "source") -> tuple[Path, Path]:
    source = tmp_path / f"{name}.json"
    scratch = tmp_path / f"{name}-scratch"
    source.write_bytes(payload)
    return source, scratch


def _complete_session(source: Path, scratch: Path) -> tuple[ProjectionSummary, dict[str, Any]]:
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        while not session.is_complete:
            session.step(13, max_fragments=2)
        summary = session.summary
        assert summary is not None
        session.checkpoint()
    finally:
        session.close()
    return summary, json.loads((scratch / "checkpoint.json").read_text())


def _assert_session_rejects(source: Path, scratch: Path) -> None:
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        with pytest.raises(ValueError):
            while not session.is_complete:
                session.step(13, max_fragments=2)
        assert not session.is_complete
        assert session.summary is None
    finally:
        session.close()


def test_supported_source_schema_version_is_pinned() -> None:
    assert SUPPORTED_SOURCE_SCHEMA_VERSION == 6


@pytest.mark.parametrize("meta_first", [True, False])
@pytest.mark.parametrize("escaped_meta_keys", [True, False])
def test_valid_root_meta_first_or_last_and_escaped_keys(meta_first: bool, escaped_meta_keys: bool) -> None:
    payload = _root_payload(meta_present=True, meta_first=meta_first, escaped_meta_keys=escaped_meta_keys)
    baseline = _run_projection(payload, chunk_bytes=1, budget=1)
    summary, chunks, ends, projection = _run_projection(payload, chunk_bytes=1, budget=1, restart_each_step=True)

    assert (summary, chunks, ends) == baseline[:3]
    assert summary == _EXPECTED_SUMMARY
    assert projection.source_schema_version == SUPPORTED_SOURCE_SCHEMA_VERSION
    with pytest.raises(AttributeError):
        projection.source_schema_version = 7  # type: ignore[misc]


def test_schema_version_is_unavailable_before_successful_finish() -> None:
    projection = SubmissionProjection("op-1", "exp", lambda _chunk: None, lambda _end: None)

    with pytest.raises(ValueError, match="unavailable"):
        _ = projection.source_schema_version


def test_metadata_free_structural_projection_returns_none() -> None:
    summary, chunks, ends, projection = _run_projection(
        _root_payload(state="aborted", group="foreign", selected=False),
    )

    assert summary == ProjectionSummary("aborted", False, 0, 0)
    assert chunks == []
    assert ends == []
    assert projection.source_schema_version is None


@pytest.mark.parametrize(
    "meta_raw",
    [
        b"{}",
        b'{"other":6}',
        b"[]",
        b"null",
        b"6",
    ],
)
def test_session_rejects_missing_schema_and_wrong_meta_shape(tmp_path: Path, meta_raw: bytes) -> None:
    payload = _root_payload(meta_present=True, meta_raw=meta_raw)
    if meta_raw.startswith(b"{"):
        summary, _chunks, _ends, projection = _run_projection(payload)
        assert summary == _EXPECTED_SUMMARY
        assert projection.source_schema_version is None
    else:
        with pytest.raises(ValueError):
            _run_projection(payload)
    source, scratch = _session_paths(tmp_path, payload)
    _assert_session_rejects(source, scratch)


@pytest.mark.parametrize(("state", "group", "selected"), _INVALID_CONTEXTS)
@pytest.mark.parametrize("schema_literal", _INVALID_VERSION_LITERALS)
def test_wrong_schema_literal_is_rejected_for_every_submission_context(
    state: str, group: str, selected: bool, schema_literal: bytes
) -> None:
    with pytest.raises(ValueError):
        _run_projection(
            _root_payload(
                state=state,
                group=group,
                selected=selected,
                meta_present=True,
                schema_literal=schema_literal,
            )
        )


@pytest.mark.parametrize(("duplicate_meta", "duplicate_schema"), [(True, False), (False, True)])
@pytest.mark.parametrize(("state", "group", "selected"), _INVALID_CONTEXTS)
def test_duplicate_meta_or_schema_keys_are_rejected_regardless_of_context(
    duplicate_meta: bool, duplicate_schema: bool, state: str, group: str, selected: bool
) -> None:
    with pytest.raises(ValueError):
        _run_projection(
            _root_payload(
                state=state,
                group=group,
                selected=selected,
                meta_present=True,
                duplicate_meta=duplicate_meta,
                duplicate_schema=duplicate_schema,
            )
        )


def test_nested_fake_meta_does_not_qualify_the_root() -> None:
    summary, _chunks, _ends, projection = _run_projection(
        _root_payload(extra_root=b'"nested":{"meta":{"schema_version":6}}')
    )

    assert summary == _EXPECTED_SUMMARY
    assert projection.source_schema_version is None


@pytest.mark.parametrize(
    ("state", "group", "selected"),
    [
        ("committed", "exp", True),
        ("committed", "foreign", True),
        ("aborted", "exp", False),
    ],
)
def test_projection_session_requires_schema_six_even_for_nonmatching_records(
    tmp_path: Path, state: str, group: str, selected: bool
) -> None:
    source, scratch = _session_paths(
        tmp_path,
        _root_payload(state=state, group=group, selected=selected),
    )

    _assert_session_rejects(source, scratch)


def test_session_resumes_checkpoint_before_meta_last_and_then_qualifies(tmp_path: Path) -> None:
    source, scratch = _session_paths(tmp_path, _root_payload(meta_present=True), "meta-last")
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        for _ in range(100):
            session.step(13, max_fragments=2)
            session.checkpoint()
            checkpoint = json.loads((scratch / "checkpoint.json").read_text())
            if (
                checkpoint["projection"]["state"]["task_count"] == 2
                and not checkpoint["projection"]["state"]["meta_seen"]
            ):
                break
        else:
            raise AssertionError("failed to checkpoint before the root meta object")
    finally:
        session.close()

    resumed = ProjectionSession.resume(source, scratch, "op-1", "exp")
    try:
        while not resumed.is_complete:
            resumed.step(13, max_fragments=2)
        assert resumed.summary == _EXPECTED_SUMMARY
        resumed.checkpoint()
    finally:
        resumed.close()
    completed = json.loads((scratch / "checkpoint.json").read_text())
    assert completed["version"] == 1
    assert completed["projection"]["state"]["source_schema_version"] == 6


def test_selected_output_before_invalid_meta_never_qualifies(tmp_path: Path) -> None:
    payload = _root_payload(meta_present=True, schema_literal=b"7")
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    source = io.BytesIO(payload)
    scanner = Scanner(source, lambda _span: None, chunk_bytes=7, emit_bytes=projection.feed)

    while not scanner.step(13, max_fragments=2).is_complete:
        pass
    assert chunks
    assert ends
    with pytest.raises(ValueError):
        projection.finish()
    with pytest.raises(ValueError):
        _ = projection.source_schema_version

    session_source, scratch = _session_paths(tmp_path, payload, "invalid-meta")
    _assert_session_rejects(session_source, scratch)
    events = (scratch / "events.jsonl").read_bytes()
    assert any(json.loads(line)["type"] == "chunk" for line in events.splitlines())
    checkpoint = json.loads((scratch / "checkpoint.json").read_text())
    assert checkpoint["scanner"]["is_complete"] is False
    assert checkpoint["projection"]["summary"] is None


def test_giant_invalid_schema_literal_restores_from_active_checkpoint() -> None:
    giant_literal = b"6" + (b"0" * 65_536)
    payload = _root_payload(meta_present=True, schema_literal=giant_literal)
    chunks: list[FieldChunk] = []
    ends: list[FieldEnd] = []
    projection = SubmissionProjection("op-1", "exp", chunks.append, ends.append)
    source = io.BytesIO(payload)
    scanner = Scanner(source, lambda _span: None, chunk_bytes=65_536, emit_bytes=projection.feed)

    while True:
        result = scanner.step(65_536, max_fragments=2)
        projection_snapshot = json.loads(json.dumps(projection.snapshot()))
        active = projection_snapshot["active"]
        if (
            isinstance(active, dict)
            and active["handler"] == "schema_version"
            and active["literal"]["position"] >= 65_536
        ):
            assert active["literal"]["valid"] is False
            break
        assert not result.is_complete

    assert projection_snapshot["version"] == 2
    scanner_snapshot = json.loads(json.dumps(scanner.snapshot()))
    source.seek(scanner_snapshot["offset"])
    restored_projection = SubmissionProjection.from_snapshot(
        "op-1",
        "exp",
        chunks.append,
        ends.append,
        projection_snapshot,
    )
    restored_scanner = Scanner.from_snapshot(
        source,
        lambda _span: None,
        scanner_snapshot,
        emit_bytes=restored_projection.feed,
    )
    while not restored_scanner.step(65_536, max_fragments=2).is_complete:
        pass
    with pytest.raises(ValueError):
        restored_projection.finish()


def test_projection_snapshot_version_one_is_rejected() -> None:
    _summary, _chunks, _ends, projection = _run_projection(_root_payload(meta_present=True))
    snapshot = projection.snapshot()
    snapshot["version"] = 1

    with pytest.raises(ValueError):
        SubmissionProjection.from_snapshot(
            "op-1",
            "exp",
            lambda _chunk: None,
            lambda _end: None,
            snapshot,
        )


@pytest.mark.parametrize("mutation", ["schema_marker", "projection_version", "boolean_version", "boolean_schema"])
def test_invalid_finished_checkpoint_rejects_before_spool_open_or_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source, scratch = _session_paths(tmp_path, _root_payload(meta_present=True), "finished")
    _summary, checkpoint = _complete_session(source, scratch)
    events_path = scratch / "events.jsonl"
    suffix = b"unconfirmed spool suffix"
    with events_path.open("ab") as events:
        events.write(suffix)

    if mutation == "schema_marker":
        state = checkpoint["projection"]["state"]
        state["meta_seen"] = True
        state["schema_version_seen"] = True
        state["source_schema_version"] = None
    elif mutation == "projection_version":
        checkpoint["projection"]["version"] = 1
    elif mutation == "boolean_version":
        checkpoint["projection"]["version"] = True
    else:
        checkpoint["projection"]["state"]["source_schema_version"] = True
    (scratch / "checkpoint.json").write_text(json.dumps(checkpoint, separators=(",", ":")))

    def fail_if_spool_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid parser checkpoint opened the spool")

    monkeypatch.setattr(projection_session_module, "_open_spool_for_resume", fail_if_spool_open)
    with pytest.raises(ValueError):
        ProjectionSession.resume(source, scratch, "op-1", "exp")
    assert events_path.read_bytes().endswith(suffix)
