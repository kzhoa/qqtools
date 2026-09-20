"""Integration coverage for the bounded JSON lexical stream prototype."""

from __future__ import annotations

import hashlib
import json
import struct
import tracemalloc
from pathlib import Path
from typing import BinaryIO, Callable

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner, Span

pytestmark = pytest.mark.integration


class TracingBinaryIO:
    """Record read sizes while retaining a real file handle as the source."""

    def __init__(self, handle: BinaryIO, max_return: int | None = None):
        self._handle = handle
        self._max_return = max_return
        self.requests: list[int] = []
        self.returned: list[int] = []

    def tell(self) -> int:
        return self._handle.tell()

    def read(self, size: int) -> bytes:
        self.requests.append(size)
        if self._max_return is not None:
            size = min(size, self._max_return)
        data = self._handle.read(size)
        self.returned.append(len(data))
        return data


def _scan_file(
    path: Path,
    chunk_bytes: int,
    budgets: tuple[int, ...],
    emit: list[Span],
    *,
    max_return: int | None = None,
    emit_bytes: Callable[[Span, bytes], None] | None = None,
) -> tuple[TracingBinaryIO, int]:
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle, max_return=max_return)
        scanner = Scanner(source, emit.append, chunk_bytes=chunk_bytes, emit_bytes=emit_bytes)
        total_bytes = 0
        index = 0
        while True:
            budget = budgets[index % len(budgets)]
            before = len(source.requests)
            result = scanner.step(budget)
            requests = source.requests[before:]
            assert sum(requests) <= budget
            assert all(1 <= request <= chunk_bytes for request in requests)
            assert result.bytes_read <= budget
            assert result.bytes_processed <= budget
            total_bytes += result.bytes_read
            if result.is_complete:
                break
            index += 1
        assert not handle.closed
        return source, total_bytes


def _assert_spans_cover_tokens(payload: bytes, spans: list[Span]) -> list[bytes]:
    assert spans
    assert [span.token_id for span in spans] == sorted(span.token_id for span in spans)
    assert all(span.end > span.start for span in spans)
    assert all(span.end - span.start <= 65_536 for span in spans)
    assert all(0 <= span.start < span.end <= len(payload) for span in spans)

    tokens: list[bytes] = []
    current_id = -1
    current_end = -1
    saw_final = False
    for span in spans:
        if span.token_id != current_id:
            assert span.token_id == current_id + 1
            assert not saw_final or current_id >= 0
            current_id = span.token_id
            current_end = span.start
            saw_final = False
            tokens.append(b"")
        assert span.start == current_end
        assert not saw_final
        tokens[-1] += payload[span.start : span.end]
        current_end = span.end
        if span.is_final:
            saw_final = True
    assert saw_final
    assert all(
        sum(span.is_final for span in spans if span.token_id == token_id) == 1 for token_id in range(current_id + 1)
    )
    assert all(
        spans[index].is_final or spans[index + 1].token_id == spans[index].token_id for index in range(len(spans) - 1)
    )
    return tokens


@pytest.mark.parametrize("chunk_bytes", [1, 2, 7, 65_536])
@pytest.mark.parametrize("budget", [1, 13, 65_536])
def test_valid_payload_streams_across_chunks_and_budgets(tmp_path: Path, chunk_bytes: int, budget: int):
    payload = (
        '{"text":"hé \\"quoted\\" and \\\\slash", '
        '"escaped":"\\u03bb\\ud800", "numbers":[0,-12,3.5,6E+2], '
        '"values":[true,null,NaN,-Infinity], "nested":{"ok":false}}'
    ).encode()
    path = tmp_path / "payload.json"
    path.write_bytes(payload)
    spans: list[Span] = []

    source, total_bytes = _scan_file(path, chunk_bytes, (budget,), spans)

    assert total_bytes == len(payload)
    tokens = _assert_spans_cover_tokens(payload, spans)
    assert json.loads(b"".join(tokens)) == json.loads(payload)
    assert [span.kind for span in spans if span.is_final] == [
        "{",
        "string",
        ":",
        "string",
        ",",
        "string",
        ":",
        "string",
        ",",
        "string",
        ":",
        "[",
        "number",
        ",",
        "number",
        ",",
        "number",
        ",",
        "number",
        "]",
        ",",
        "string",
        ":",
        "[",
        "literal",
        ",",
        "literal",
        ",",
        "literal",
        ",",
        "literal",
        "]",
        ",",
        "string",
        ":",
        "{",
        "string",
        ":",
        "literal",
        "}",
        "}",
    ]
    assert max(source.requests) <= chunk_bytes


@pytest.mark.parametrize("chunk_bytes", [1, 7, 65_536])
def test_raw_fragment_bridge_preserves_source_bytes_and_span_order(tmp_path: Path, chunk_bytes: int):
    long_value = b"a" * 131_071
    escaped_utf8_value = (b"\xc3\xa9\\u03bb" * 8_192) + b'\\"'
    payload = b'{"long":"' + long_value + b'", "escaped":"' + escaped_utf8_value + b'", "special":-Infinity}'
    path = tmp_path / "raw-fragments.json"
    path.write_bytes(payload)
    spans: list[Span] = []
    raw_events: list[tuple[Span, bytes]] = []

    source, total_bytes = _scan_file(
        path,
        chunk_bytes,
        (1, 13, 65_536),
        spans,
        emit_bytes=lambda span, raw: raw_events.append((span, raw)),
    )

    assert total_bytes == len(payload)
    assert sum(source.returned) == len(payload)
    assert len(raw_events) == len(spans)
    assert [span for span, _ in raw_events] == spans
    assert all(raw for _, raw in raw_events)
    assert all(len(raw) <= 65_536 for _, raw in raw_events)
    assert all(raw == payload[span.start : span.end] for span, raw in raw_events)

    tokens = _assert_spans_cover_tokens(payload, spans)
    raw_tokens: list[bytes] = []
    current_token_id = -1
    for span, raw in raw_events:
        if span.token_id != current_token_id:
            raw_tokens.append(raw)
        else:
            raw_tokens[-1] += raw
        current_token_id = span.token_id
    assert raw_tokens == tokens
    assert sum(span.is_final for span in spans) == len({span.token_id for span in spans})
    string_lengths: dict[int, int] = {}
    for span, raw in raw_events:
        if span.kind == "string":
            string_lengths[span.token_id] = string_lengths.get(span.token_id, 0) + len(raw)
    assert sorted(string_lengths.values())[-2:] == [65_540, 131_073]
    assert any(span.kind == "literal" and raw == b"-Infinity" for span, raw in raw_events)


def _scan_with_fragment_limit(
    path: Path,
    *,
    chunk_bytes: int,
    byte_budget: int,
    max_fragments: int,
    max_return: int | None = None,
) -> tuple[TracingBinaryIO, list[Span], bool]:
    spans: list[Span] = []
    saw_buffered_progress = False
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle, max_return=max_return)
        scanner = Scanner(source, spans.append, chunk_bytes=chunk_bytes)
        while True:
            request_start = len(source.requests)
            returned_start = len(source.returned)
            span_start = len(spans)
            result = scanner.step(byte_budget, max_fragments=max_fragments)
            requests = source.requests[request_start:]
            returned = source.returned[returned_start:]
            emitted = len(spans) - span_start
            assert emitted <= max_fragments
            assert result.bytes_read == sum(returned)
            assert result.bytes_read <= byte_budget
            assert result.bytes_processed <= byte_budget
            assert sum(requests) <= byte_budget
            assert all(1 <= request <= chunk_bytes for request in requests)
            if result.bytes_read == 0 and result.bytes_processed > 0:
                saw_buffered_progress = True
            if result.is_complete:
                break
        assert not handle.closed
        assert sum(source.returned) == path.stat().st_size
        return source, spans, saw_buffered_progress


@pytest.mark.parametrize("max_fragments", [2, 3, 64])
def test_fragment_pause_preserves_dense_array_tokens(tmp_path: Path, max_fragments: int):
    payload = b"[" + b",".join(str(value).encode() for value in range(128)) + b"]"
    path = tmp_path / "dense-array.json"
    path.write_bytes(payload)

    _, spans, _ = _scan_with_fragment_limit(
        path,
        chunk_bytes=7,
        byte_budget=13,
        max_fragments=max_fragments,
    )

    assert b"".join(_assert_spans_cover_tokens(payload, spans)) == payload


@pytest.mark.parametrize("max_fragments", [2, 3, 64])
def test_fragment_pause_preserves_long_string_with_short_reads(tmp_path: Path, max_fragments: int):
    payload = b'"' + (b"a" * 131_071) + b'"'
    path = tmp_path / "long-string.json"
    path.write_bytes(payload)

    _, spans, _ = _scan_with_fragment_limit(
        path,
        chunk_bytes=65_536,
        byte_budget=65_536,
        max_fragments=max_fragments,
        max_return=13,
    )

    assert b"".join(_assert_spans_cover_tokens(payload, spans)) == payload


def test_fragment_pause_reports_buffered_byte_progress(tmp_path: Path):
    payload = b'"' + (b"a" * 65_536) + b'"'
    path = tmp_path / "buffered-progress.json"
    path.write_bytes(payload)

    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        spans: list[Span] = []
        scanner = Scanner(source, spans.append, chunk_bytes=65_536)
        first = scanner.step(65_536, max_fragments=2)
        second = scanner.step(65_536, max_fragments=2)
        buffered = scanner.step(1, max_fragments=2)

    assert first.bytes_read == 65_536
    assert first.bytes_processed == 65_536
    assert second.bytes_read == 2
    assert second.bytes_processed == 1
    assert buffered.bytes_read == 0
    assert buffered.bytes_processed == 1
    assert not buffered.is_complete
    assert len(spans) == 2


def test_exact_maximum_span_is_final_without_a_zero_length_fragment(tmp_path: Path):
    payload = b'"' + (b"x" * (65_536 - 2)) + b'"'
    path = tmp_path / "exact.json"
    path.write_bytes(payload)
    spans: list[Span] = []

    _scan_file(path, 7, (13,), spans)

    assert spans == [Span(0, "string", 0, 65_536, True)]


def test_short_reads_and_utf8_split_keep_offsets_and_completion(tmp_path: Path):
    payload = b'{"value":"\xc3\xa9", "number":-Infinity}'
    path = tmp_path / "short.json"
    path.write_bytes(payload)
    spans: list[Span] = []

    source, total_bytes = _scan_file(path, 7, (13, 1), spans, max_return=2)

    assert total_bytes == len(payload)
    assert b"".join(payload[span.start : span.end] for span in spans) == payload.replace(b" ", b"")
    assert source.returned
    assert all(returned <= requested for returned, requested in zip(source.returned, source.requests))


@pytest.mark.parametrize(
    "payload",
    [
        b'"unterminated',
        b'"bad\\q"',
        b'"\\u12"',
        b'"\\uD8G0"',
        b'"line\n"',
        b'"\xc3("',
        b"01",
        b"1.",
        b"1e+",
        b"tru",
        b"unknown",
    ],
)
@pytest.mark.parametrize("chunk_bytes", [1, 2, 7, 65_536])
def test_invalid_lexical_tokens_fail_at_every_boundary(tmp_path: Path, payload: bytes, chunk_bytes: int):
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    spans: list[Span] = []

    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(source, spans.append, chunk_bytes=chunk_bytes)
        with pytest.raises(ValueError):
            while True:
                result = scanner.step(13)
                if result.is_complete:
                    pytest.fail("invalid lexical input was accepted")


def test_structural_validation_is_outside_the_lexical_layer(tmp_path: Path):
    payload = b"[1,]"
    path = tmp_path / "structurally-invalid.json"
    path.write_bytes(payload)
    spans: list[Span] = []

    _scan_file(path, 1, (1,), spans)

    assert b"".join(payload[span.start : span.end] for span in spans) == payload


@pytest.mark.parametrize("raw_mode", [False, True])
def test_giant_string_has_bounded_memory_and_no_event_retention(tmp_path: Path, raw_mode: bool):
    path = tmp_path / "giant-string.json"
    with path.open("wb") as handle:
        handle.write(b'"')
        block = b"a" * 65_536
        for _ in range(128):
            handle.write(block)
        handle.write(b'"')
    expected_size = 8 * 1024 * 1024 + 2
    counters = {"spans": 0, "final": 0, "covered": 0}
    raw_counters = {"fragments": 0, "covered": 0}
    digest = hashlib.blake2b(digest_size=16)

    def emit(span: Span) -> None:
        counters["spans"] += 1
        counters["final"] += span.is_final
        counters["covered"] += span.end - span.start
        digest.update(struct.pack("!QII?", span.token_id, span.start, span.end, span.is_final))

    def emit_bytes(span: Span, raw: bytes) -> None:
        assert len(raw) == span.end - span.start
        raw_counters["fragments"] += 1
        raw_counters["covered"] += len(raw)

    tracemalloc.start()
    tracemalloc.clear_traces()
    try:
        with path.open("rb") as handle:
            source = TracingBinaryIO(handle)
            scanner = Scanner(source, emit, chunk_bytes=65_536, emit_bytes=emit_bytes if raw_mode else None)
            while not scanner.step(1 << 20).is_complete:
                pass
            peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert expected_size == path.stat().st_size
    assert counters == {"spans": 129, "final": 1, "covered": expected_size}
    assert raw_counters == (
        {"fragments": 129, "covered": expected_size} if raw_mode else {"fragments": 0, "covered": 0}
    )
    assert digest.digest()
    assert max(source.requests) <= 65_536
    assert peak < 2 * 1024 * 1024


def test_giant_number_streams_without_numeric_conversion(tmp_path: Path):
    path = tmp_path / "giant-number.json"
    number_size = 2 * 1024 * 1024 + 1
    with path.open("wb") as handle:
        handle.write(b"1")
        block = b"7" * 65_536
        remaining = number_size - 1
        while remaining:
            take = min(remaining, len(block))
            handle.write(block[:take])
            remaining -= take
    spans: list[Span] = []

    source, total_bytes = _scan_file(path, 65_536, (1 << 20,), spans)

    assert total_bytes == number_size
    assert sum(span.end - span.start for span in spans) == number_size
    assert sum(span.is_final for span in spans) == 1
    assert max(source.requests) <= 65_536
