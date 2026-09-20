"""Integration coverage for JSON lexical scanner checkpoints."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import BinaryIO, Callable

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner, Span, StepResult

pytestmark = pytest.mark.integration


class TracingBinaryIO:
    """Record read starts and sizes while retaining a real file handle."""

    def __init__(self, handle: BinaryIO, max_return: int | None = None):
        self._handle = handle
        self._max_return = max_return
        self.read_starts: list[int] = []
        self.requests: list[int] = []
        self.returned: list[int] = []

    def tell(self) -> int:
        return self._handle.tell()

    def read(self, size: int) -> bytes:
        self.read_starts.append(self._handle.tell())
        self.requests.append(size)
        if self._max_return is not None:
            size = min(size, self._max_return)
        data = self._handle.read(size)
        self.returned.append(len(data))
        return data


def _drive(
    scanner: Scanner,
    budgets: tuple[int, ...],
    *,
    max_fragments: int | None = None,
) -> None:
    index = 0
    while True:
        result = scanner.step(budgets[index % len(budgets)], max_fragments=max_fragments)
        if result.is_complete:
            return
        index += 1
        assert index < 1_000_000


def _scan_payload(
    path: Path,
    *,
    chunk_bytes: int,
    budgets: tuple[int, ...],
    raw_mode: bool,
    max_fragments: int | None = None,
) -> tuple[list[Span], list[tuple[Span, bytes]], TracingBinaryIO]:
    spans: list[Span] = []
    raw_events: list[tuple[Span, bytes]] = []
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(
            source,
            spans.append,
            chunk_bytes=chunk_bytes,
            emit_bytes=(lambda span, raw: raw_events.append((span, raw))) if raw_mode else None,
        )
        _drive(scanner, budgets, max_fragments=max_fragments)
        return spans, raw_events, source


def _mixed_payload() -> bytes:
    return (
        b'{"utf8":"\xf0\x9f\x98\x80", "escape":"quoted \\" text", '
        b'"unicode":"\\u03bb", "number":-12.3e+4, "literal":-Infinity, '
        b'"dense":[true,null,NaN,false,0,1,2,3,4,5]} '
    )


@pytest.mark.parametrize("raw_mode", [False, True])
@pytest.mark.parametrize(
    ("budget", "max_fragments"),
    [(1, None), (13, 2), (65_536, 64)],
)
def test_snapshot_json_roundtrip_resumes_with_identical_events(
    tmp_path: Path,
    raw_mode: bool,
    budget: int,
    max_fragments: int | None,
):
    payload = _mixed_payload()
    path = tmp_path / "mixed.json"
    path.write_bytes(payload)

    baseline_spans, baseline_raw, _ = _scan_payload(
        path,
        chunk_bytes=7,
        budgets=(1, 13, 65_536),
        raw_mode=raw_mode,
        max_fragments=max_fragments,
    )

    prefix_spans: list[Span] = []
    prefix_raw: list[tuple[Span, bytes]] = []
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(
            source,
            prefix_spans.append,
            chunk_bytes=7,
            emit_bytes=(lambda span, raw: prefix_raw.append((span, raw))) if raw_mode else None,
        )
        scanner.step(budget, max_fragments=max_fragments)
        before_snapshot = scanner.snapshot()
        requests_before_snapshot = source.requests.copy()
        checkpoint = json.loads(json.dumps(before_snapshot))
        assert checkpoint == before_snapshot
        assert scanner.snapshot() == before_snapshot
        assert source.requests == requests_before_snapshot
        offset = checkpoint["offset"]

        with path.open("rb") as restored_handle:
            restored_handle.seek(offset)
            restored_source = TracingBinaryIO(restored_handle)
            suffix_spans: list[Span] = []
            suffix_raw: list[tuple[Span, bytes]] = []
            restored = Scanner.from_snapshot(
                restored_source,
                suffix_spans.append,
                checkpoint,
                emit_bytes=(lambda span, raw: suffix_raw.append((span, raw))) if raw_mode else None,
            )
            _drive(restored, (1, 13, 65_536), max_fragments=max_fragments)

        assert prefix_spans + suffix_spans == baseline_spans
        assert prefix_raw + suffix_raw == baseline_raw
        if not checkpoint["is_complete"]:
            assert restored_source.read_starts
            assert restored_source.read_starts[0] == offset


@pytest.mark.parametrize(
    ("payload", "steps", "expected"),
    [
        (b'"\xf0\x9f\x98\x80"', 2, {"string_state": "normal", "utf8_remaining": 3, "utf8_first": True}),
        (b'"\\""', 2, {"string_state": "escape", "utf8_remaining": 0}),
        (b'"\\u03bb"', 4, {"string_state": "unicode", "unicode_digits": 1}),
        (b"1.2e+4", 5, {"number_state": "exp_sign", "token_kind": "number"}),
        (b"true", 2, {"literal_target": "true", "literal_progress": 2, "token_kind": "literal"}),
        (b"-Infinity", 2, {"literal_target": "-Infinity", "literal_progress": 2, "token_kind": "literal"}),
    ],
)
def test_snapshot_restores_partial_lexical_states(
    tmp_path: Path,
    payload: bytes,
    steps: int,
    expected: dict[str, object],
):
    path = tmp_path / "partial.json"
    path.write_bytes(payload)
    baseline_spans, _, _ = _scan_payload(path, chunk_bytes=1, budgets=(1,), raw_mode=False)

    prefix: list[Span] = []
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(source, prefix.append, chunk_bytes=1)
        scanner.step(steps)
        checkpoint = scanner.snapshot()
        assert all(checkpoint[name] == value for name, value in expected.items())
        with path.open("rb") as restored_handle:
            restored_handle.seek(checkpoint["offset"])
            suffix: list[Span] = []
            restored = Scanner.from_snapshot(TracingBinaryIO(restored_handle), suffix.append, checkpoint)
            _drive(restored, (1,))
    assert prefix + suffix == baseline_spans


@pytest.mark.parametrize("raw_mode", [False, True])
@pytest.mark.parametrize("payload", [b'"' + (b"a" * 131_071) + b'"', b"7" * 131_073], ids=["string", "number"])
def test_snapshot_discards_prefetch_and_resumes_at_processed_offset(tmp_path: Path, raw_mode: bool, payload: bytes):
    path = tmp_path / "prefetched.json"
    path.write_bytes(payload)
    all_spans, all_raw, _ = _scan_payload(path, chunk_bytes=65_536, budgets=(65_536,), raw_mode=raw_mode)

    prefix: list[Span] = []
    prefix_raw: list[tuple[Span, bytes]] = []
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(
            source,
            prefix.append,
            chunk_bytes=65_536,
            emit_bytes=(lambda span, raw: prefix_raw.append((span, raw))) if raw_mode else None,
        )
        scanner.step(65_536, max_fragments=2)
        scanner.step(65_536, max_fragments=2)
        checkpoint = scanner.snapshot()
        source_reads = source.read_starts.copy()
        assert source.tell() > checkpoint["offset"]
        assert scanner.snapshot() == checkpoint
        assert source.read_starts == source_reads

        with path.open("rb") as restored_handle:
            restored_handle.seek(checkpoint["offset"])
            restored_source = TracingBinaryIO(restored_handle)
            suffix: list[Span] = []
            suffix_raw: list[tuple[Span, bytes]] = []
            restored = Scanner.from_snapshot(
                restored_source,
                suffix.append,
                checkpoint,
                emit_bytes=(lambda span, raw: suffix_raw.append((span, raw))) if raw_mode else None,
            )
            _drive(restored, (1, 13, 65_536), max_fragments=2)

    assert prefix + suffix == all_spans
    assert prefix_raw + suffix_raw == all_raw
    assert restored_source.read_starts[0] == checkpoint["offset"]
    assert all(start >= checkpoint["offset"] for start in restored_source.read_starts)


def test_repeated_fresh_scanners_resume_every_short_read_step(tmp_path: Path):
    path = tmp_path / "repeated.json"
    path.write_bytes(_mixed_payload())
    expected_spans, expected_raw, _ = _scan_payload(path, chunk_bytes=7, budgets=(13,), raw_mode=True, max_fragments=3)
    spans: list[Span] = []
    raw_events: list[tuple[Span, bytes]] = []
    checkpoint = None

    def emit_bytes(span: Span, raw: bytes) -> None:
        raw_events.append((span, raw))

    for _ in range(1000):
        with path.open("rb") as handle:
            if checkpoint is not None:
                handle.seek(checkpoint["offset"])
            source = TracingBinaryIO(handle, max_return=2)
            scanner = (
                Scanner(source, spans.append, chunk_bytes=7, emit_bytes=emit_bytes)
                if checkpoint is None
                else Scanner.from_snapshot(source, spans.append, checkpoint, emit_bytes=emit_bytes)
            )
            result = scanner.step(13, max_fragments=3)
            checkpoint = json.loads(json.dumps(scanner.snapshot()))
        if result.is_complete:
            break
        assert result.bytes_processed > 0
    else:
        pytest.fail("fresh scanners did not make bounded forward progress")
    assert spans == expected_spans
    assert raw_events == expected_raw


def test_complete_snapshot_restores_without_input_or_output(tmp_path: Path):
    path = tmp_path / "complete.json"
    path.write_bytes(b"[1,true,-Infinity]")
    with path.open("rb") as handle:
        original_source = TracingBinaryIO(handle)
        original_spans: list[Span] = []
        original = Scanner(original_source, original_spans.append, chunk_bytes=7)
        _drive(original, (13,))
        snapshot = original.snapshot()
    with path.open("rb") as handle:
        handle.seek(len(path.read_bytes()))
        restored_source = TracingBinaryIO(handle)
        restored_spans: list[Span] = []
        restored = Scanner.from_snapshot(restored_source, restored_spans.append, snapshot)
        result = restored.step(1)
    assert result == StepResult(0, True, 0, 0)
    assert restored_spans == []
    assert restored_source.read_starts == []


def _active_checkpoint(tmp_path: Path, *, raw_mode: bool) -> tuple[Path, dict[str, object]]:
    path = tmp_path / ("active-raw.json" if raw_mode else "active-span.json")
    path.write_bytes(b'"checkpoint"')
    with path.open("rb") as handle:
        source = TracingBinaryIO(handle)
        scanner = Scanner(
            source,
            lambda span: None,
            chunk_bytes=7,
            emit_bytes=(lambda span, raw: None) if raw_mode else None,
        )
        scanner.step(4)
        return path, scanner.snapshot()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(version=2),
        lambda state: state.update(chunk_bytes=True),
        lambda state: state.update(offset=-1),
        lambda state: state.update(token_kind="bogus"),
        lambda state: state.update(token_id=None),
        lambda state: state.update(fragment_b64="not-base64"),
        lambda state: state.pop("offset"),
        lambda state: state.update(unicode_digits=4),
    ],
)
def test_invalid_snapshot_shape_is_rejected(tmp_path: Path, mutation: Callable[[dict[str, object]], None]):
    path, checkpoint = _active_checkpoint(tmp_path, raw_mode=True)
    invalid = copy.deepcopy(checkpoint)
    mutation(invalid)
    with path.open("rb") as handle:
        handle.seek(checkpoint["offset"])
        with pytest.raises((TypeError, ValueError)):
            Scanner.from_snapshot(
                TracingBinaryIO(handle), lambda span: None, invalid, emit_bytes=lambda span, raw: None
            )


def test_snapshot_source_offset_and_callback_mode_are_required(tmp_path: Path):
    path, span_checkpoint = _active_checkpoint(tmp_path, raw_mode=False)
    with path.open("rb") as handle:
        handle.seek(span_checkpoint["offset"])
        with pytest.raises(ValueError):
            Scanner.from_snapshot(
                TracingBinaryIO(handle), lambda span: None, span_checkpoint, emit_bytes=lambda s, b: None
            )
    raw_path, raw_checkpoint = _active_checkpoint(tmp_path, raw_mode=True)
    with raw_path.open("rb") as handle:
        handle.seek(raw_checkpoint["offset"])
        with pytest.raises(ValueError):
            Scanner.from_snapshot(TracingBinaryIO(handle), lambda span: None, raw_checkpoint)
    with raw_path.open("rb") as handle:
        handle.seek(raw_checkpoint["offset"] + 1)
        source = TracingBinaryIO(handle)
        with pytest.raises(ValueError):
            Scanner.from_snapshot(source, lambda span: None, raw_checkpoint, emit_bytes=lambda s, b: None)
        assert source.read_starts == []


def test_snapshot_size_is_bounded_by_current_fragment(tmp_path: Path):
    path = tmp_path / "bounded.json"
    path.write_bytes(b'"' + (b"a" * 131_071) + b'"')
    with path.open("rb") as handle:
        scanner = Scanner(TracingBinaryIO(handle), lambda span: None, chunk_bytes=65_536, emit_bytes=lambda s, b: None)
        scanner.step(65_536)
        checkpoint = scanner.snapshot()
    encoded = json.dumps(checkpoint, separators=(",", ":"))
    assert len(encoded) < 100_000
    assert len(checkpoint["fragment_b64"]) <= 90_000


def test_oversized_fragment_checkpoint_is_rejected_before_decode(tmp_path: Path):
    path, checkpoint = _active_checkpoint(tmp_path, raw_mode=True)
    checkpoint["fragment_b64"] = "A" * (4 * ((65_536 + 2) // 3) + 4)
    with path.open("rb") as handle:
        handle.seek(checkpoint["offset"])
        with pytest.raises(ValueError, match="exceeds the fragment limit"):
            Scanner.from_snapshot(TracingBinaryIO(handle), lambda span: None, checkpoint, emit_bytes=lambda s, b: None)


@pytest.mark.parametrize("failure_payload", [b"@", b'"unterminated'])
def test_step_failure_poison_scanner_for_checkpoint_and_resume(failure_payload: bytes):
    scanner = Scanner(io.BytesIO(failure_payload), lambda span: None)
    with pytest.raises(ValueError):
        while not scanner.step(13).is_complete:
            pass
    with pytest.raises(RuntimeError):
        scanner.step(1)
    with pytest.raises(RuntimeError):
        scanner.snapshot()


def test_callback_failure_poison_scanner():
    def emit(span: Span) -> None:
        raise LookupError("callback failed")

    scanner = Scanner(io.BytesIO(b"["), emit)
    with pytest.raises(LookupError):
        scanner.step(1)
    with pytest.raises(RuntimeError):
        scanner.snapshot()


def test_reentrant_step_and_snapshot_are_rejected():
    holder: dict[str, Scanner] = {}

    def emit(span: Span) -> None:
        holder["scanner"].step(1)

    scanner = Scanner(io.BytesIO(b"["), emit)
    holder["scanner"] = scanner
    with pytest.raises(RuntimeError):
        scanner.step(1)
    with pytest.raises(RuntimeError):
        scanner.step(1)

    holder.clear()

    def snapshotting_emit(span: Span) -> None:
        holder["scanner"].snapshot()

    scanner = Scanner(io.BytesIO(b"["), snapshotting_emit)
    holder["scanner"] = scanner
    with pytest.raises(RuntimeError):
        scanner.step(1)
    with pytest.raises(RuntimeError):
        scanner.snapshot()


@pytest.mark.parametrize(
    ("payload", "budget", "changes"),
    [
        (b'"\xf0\x80\x80\x80"', 2, {"utf8_first": False}),
        (b'"\xc2\xa2"', 2, {"utf8_first_min": 0x90}),
        (b"t ", 1, {"literal_progress": 4}),
        (b"[", 1, {"next_token_id": 2}),
    ],
)
def test_impossible_checkpoint_progress_is_rejected(payload: bytes, budget: int, changes: dict[str, object]):
    source = io.BytesIO(payload)
    scanner = Scanner(source, lambda span: None)
    scanner.step(budget)
    checkpoint = scanner.snapshot()
    checkpoint.update(changes)
    source.seek(checkpoint["offset"])
    with pytest.raises(ValueError):
        Scanner.from_snapshot(source, lambda span: None, checkpoint)


@pytest.mark.parametrize("raw_mode", [False, True])
@pytest.mark.parametrize(
    "encoded",
    [
        b"\xc2\xa2",
        b"\xe0\xa0\x80",
        b"\xe1\x80\x80",
        b"\xed\x9f\xbf",
        b"\xee\xb0\x80",
        b"\xf0\x90\x80\x80",
        b"\xf1\x80\x80\x80",
        b"\xf4\x8f\xbf\xbf",
    ],
)
def test_every_utf8_prefix_restores_without_changing_output(tmp_path: Path, encoded: bytes, raw_mode: bool):
    path = tmp_path / "utf8.json"
    payload = b'"' + encoded + b'"'
    path.write_bytes(payload)
    expected_spans, expected_raw, _ = _scan_payload(path, chunk_bytes=7, budgets=(13,), raw_mode=raw_mode)
    for split in range(1, len(payload)):
        spans: list[Span] = []
        raw_events: list[tuple[Span, bytes]] = []
        emit_bytes = (lambda span, raw: raw_events.append((span, raw))) if raw_mode else None
        with path.open("rb") as source:
            scanner = Scanner(source, spans.append, chunk_bytes=7, emit_bytes=emit_bytes)
            scanner.step(split)
            checkpoint = json.loads(json.dumps(scanner.snapshot()))
        with path.open("rb") as source:
            source.seek(checkpoint["offset"])
            restored = Scanner.from_snapshot(source, spans.append, checkpoint, emit_bytes=emit_bytes)
            _drive(restored, (1,))
        assert spans == expected_spans
        assert raw_events == expected_raw
