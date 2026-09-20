"""Bounded validation for the projection driver's JSONL event spool."""

from __future__ import annotations

import base64
import binascii
import json
from collections import deque
from dataclasses import dataclass
from typing import Final

from .fingerprint import ChainedDigest
from .submission_projection import FieldChunk, FieldEnd

_MAX_INPUT = 65_536
_MAX_LINE = 90_000
_MAX_CHUNK = 65_536
_CHUNK_KEYS: Final = frozenset({"type", "kind", "ordinal", "data_b64", "is_final"})
_END_KEYS: Final = frozenset({"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"})
_KINDS: Final = frozenset({"task_id", "sequence"})
_IDENTIFIER_BYTES: Final = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


@dataclass(frozen=True, slots=True)
class DecodedEvent:
    """A validated spool event and its bounded source-line locations."""

    event: FieldChunk | FieldEnd
    start: int
    end: int
    field_start: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("event object contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"event contains non-JSON constant {value}")


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)!r}")
    return value


def _exact_string(value: object, expected: str, label: str) -> None:
    if type(value) is not str or value != expected:
        raise ValueError(f"{label} must be {expected!r}")


def _strict_base64(value: object) -> bytes:
    if type(value) is not str:
        raise ValueError("chunk data_b64 must be a string")
    try:
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("chunk data_b64 must be strict canonical base64") from exc
    if base64.b64encode(decoded) != encoded:
        raise ValueError("chunk data_b64 must use canonical base64 encoding")
    if len(decoded) > _MAX_CHUNK:
        raise ValueError("decoded chunk exceeds 65536 bytes")
    return decoded


def _strict_digest(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("end digest must be exactly 64 lowercase hexadecimal characters")
    return value


class EventDecoder:
    """Incrementally decode and validate bounded projection events.

    The decoder accepts at most 65536 bytes per call.  Callers must drain all
    events from one call with :meth:`pop` before feeding the next input chunk.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._buffer_start = 0
        self._queue: deque[DecodedEvent] = deque()
        self._failed = False
        self._finished = False

        self._event_count = 0
        self._field_counts = {"task_id": 0, "sequence": 0}
        self._next_ordinals = {"task_id": 0, "sequence": 0}
        self._seen_kinds: set[str] = set()
        self._last_kind: str | None = None
        self._last_source_end: int | None = None

        self._active_kind: str | None = None
        self._active_ordinal: int | None = None
        self._active_digest: ChainedDigest | None = None
        self._active_decoded_size = 0
        self._active_saw_chunk = False
        self._active_is_final = False
        self._active_field_start: int | None = None

    def feed(self, data: bytes) -> None:
        """Consume one input fragment, raising when the decoder is invalid."""

        self._require_usable()
        if self._queue:
            raise RuntimeError("pop all decoded events before feeding more input")
        try:
            if type(data) is not bytes:
                raise TypeError("event decoder input must be bytes")
            if len(data) > _MAX_INPUT:
                raise ValueError("event decoder input cannot exceed 65536 bytes")
            if data:
                self._buffer.extend(data)
                self._drain_lines()
        except Exception as exc:
            self._poison()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("event input is invalid") from exc

    def pop(self) -> DecodedEvent | None:
        """Return the next validated event, if one is queued."""

        if self._failed:
            raise ValueError("event decoder is poisoned")
        if self._queue:
            return self._queue.popleft()
        return None

    def finish(self, *, expected_events: int, task_count: int, sequence_count: int) -> None:
        """Require a complete spool with exactly the supplied event counts."""

        self._require_usable()
        try:
            expected_events = _nonnegative_int(expected_events, "expected_events")
            task_count = _nonnegative_int(task_count, "task_count")
            sequence_count = _nonnegative_int(sequence_count, "sequence_count")
            if self._queue:
                raise ValueError("decoded events must be drained before finish")
            if self._buffer:
                raise ValueError("event spool ended with a partial JSONL line")
            if self._active_kind is not None:
                raise ValueError("event spool ended in the middle of a scalar")
            if self._event_count != expected_events:
                raise ValueError("event count does not match expected_events")
            if self._field_counts["task_id"] != task_count:
                raise ValueError("task field count does not match task_count")
            if self._field_counts["sequence"] != sequence_count:
                raise ValueError("sequence field count does not match sequence_count")
            self._finished = True
        except Exception as exc:
            self._poison()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("event decoder could not finish") from exc

    def _require_usable(self) -> None:
        if self._failed:
            raise ValueError("event decoder is poisoned")
        if self._finished:
            raise ValueError("event decoder has already finished")

    def _poison(self) -> None:
        self._failed = True
        self._buffer.clear()
        self._queue.clear()
        self._active_kind = None
        self._active_ordinal = None
        self._active_digest = None
        self._active_field_start = None

    def _drain_lines(self) -> None:
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) >= _MAX_LINE:
                    raise ValueError("JSONL event line exceeds 90000 bytes")
                return
            line_length = newline + 1
            if line_length > _MAX_LINE:
                raise ValueError("JSONL event line exceeds 90000 bytes")
            line_start = self._buffer_start
            line = bytes(self._buffer[:line_length])
            del self._buffer[:line_length]
            self._buffer_start += line_length
            self._decode_line(line, line_start, line_start + line_length)

    def _decode_line(self, line: bytes, line_start: int, line_end: int) -> None:
        try:
            value = json.loads(
                line[:-1],
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (RecursionError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("event line is not strict UTF-8 JSON") from exc
        if type(value) is not dict:
            raise ValueError("event line must contain a JSON object")
        event_type = value.get("type")
        if event_type == "chunk":
            self._decode_chunk(value, line_start, line_end)
        elif event_type == "end":
            self._decode_end(value, line_start, line_end)
        else:
            raise ValueError("event type must be 'chunk' or 'end'")

    def _decode_chunk(self, value: dict[str, object], line_start: int, line_end: int) -> None:
        event = _exact_keys(value, _CHUNK_KEYS, "chunk event")
        _exact_string(event["type"], "chunk", "chunk type")
        kind = self._event_kind(event["kind"])
        ordinal = _nonnegative_int(event["ordinal"], "chunk ordinal")
        data = _strict_base64(event["data_b64"])
        if type(event["is_final"]) is not bool:
            raise ValueError("chunk is_final must be boolean")
        is_final = event["is_final"]
        if not data and not is_final:
            raise ValueError("empty chunk is allowed only when final")

        if self._active_kind is None:
            self._begin_scalar(kind, ordinal, line_start)
        elif kind != self._active_kind or ordinal != self._active_ordinal:
            raise ValueError("chunk changed the active scalar")
        if self._active_is_final:
            raise ValueError("chunk followed a final chunk")

        self._validate_chunk_data(kind, data)
        digest = self._active_digest
        field_start = self._active_field_start
        if digest is None or field_start is None:
            raise ValueError("active scalar has no digest")
        digest.update(data)
        self._active_decoded_size += len(data)
        self._active_saw_chunk = True
        self._active_is_final = is_final
        self._queue_event(
            FieldChunk(kind, ordinal, data, is_final),
            line_start,
            line_end,
            field_start=field_start,
        )

    def _decode_end(self, value: dict[str, object], line_start: int, line_end: int) -> None:
        event = _exact_keys(value, _END_KEYS, "end event")
        _exact_string(event["type"], "end", "end type")
        kind = self._event_kind(event["kind"])
        ordinal = _nonnegative_int(event["ordinal"], "end ordinal")
        digest_value = _strict_digest(event["digest"])
        decoded_size = _nonnegative_int(event["decoded_size"], "decoded_size")
        if decoded_size == 0:
            raise ValueError("decoded_size must be positive")
        source_start = _nonnegative_int(event["start"], "source start")
        source_end = _nonnegative_int(event["end"], "source end")
        if source_end <= source_start:
            raise ValueError("source end must be greater than source start")

        if self._active_kind is None or kind != self._active_kind or ordinal != self._active_ordinal:
            raise ValueError("end did not close the active scalar")
        if not self._active_saw_chunk or not self._active_is_final:
            raise ValueError("scalar must end after a final chunk")
        digest = self._active_digest
        field_start = self._active_field_start
        if digest is None or field_start is None:
            raise ValueError("active scalar has incomplete state")
        if decoded_size != self._active_decoded_size or decoded_size != digest.size:
            raise ValueError("decoded_size does not match chunk data")
        if digest.hexdigest() != digest_value:
            raise ValueError("end digest does not match chunk data")
        if self._last_source_end is not None and source_start < self._last_source_end:
            raise ValueError("source spans overlap or move backwards")

        self._validate_scalar_end(kind)
        self._last_source_end = source_end
        self._field_counts[kind] += 1
        self._next_ordinals[kind] += 1
        self._queue_event(
            FieldEnd(kind, ordinal, digest_value, decoded_size, source_start, source_end),
            line_start,
            line_end,
            field_start=field_start,
        )
        self._active_kind = None
        self._active_ordinal = None
        self._active_digest = None
        self._active_decoded_size = 0
        self._active_saw_chunk = False
        self._active_is_final = False
        self._active_field_start = None

    def _event_kind(self, value: object) -> str:
        if type(value) is not str or value not in _KINDS:
            raise ValueError("event kind must be 'task_id' or 'sequence'")
        return value

    def _begin_scalar(self, kind: str, ordinal: int, field_start: int) -> None:
        if ordinal != self._next_ordinals[kind]:
            raise ValueError(f"{kind} ordinals must start at zero and advance contiguously")
        if self._last_kind != kind:
            if kind in self._seen_kinds:
                raise ValueError("event kind returned after another kind began")
            self._seen_kinds.add(kind)
            self._last_kind = kind
        self._active_kind = kind
        self._active_ordinal = ordinal
        self._active_digest = ChainedDigest()
        self._active_decoded_size = 0
        self._active_saw_chunk = False
        self._active_is_final = False
        self._active_field_start = field_start

    def _validate_chunk_data(self, kind: str, data: bytes) -> None:
        if kind == "task_id":
            for byte in data:
                if byte not in _IDENTIFIER_BYTES:
                    raise ValueError("task_id chunks must contain ASCII identifier bytes")
            return

        offset = self._active_decoded_size
        for index, byte in enumerate(data):
            if (offset + index == 0 and not 49 <= byte <= 57) or (offset + index > 0 and not 48 <= byte <= 57):
                raise ValueError("sequence chunks must contain a canonical positive decimal")

    def _validate_scalar_end(self, kind: str) -> None:
        if kind == "task_id":
            if self._active_decoded_size == 0:
                raise ValueError("task_id must be nonempty")
            return
        if self._active_decoded_size == 0:
            raise ValueError("sequence must be nonempty")

    def _queue_event(
        self,
        event: FieldChunk | FieldEnd,
        line_start: int,
        line_end: int,
        *,
        field_start: int | None = None,
    ) -> None:
        self._queue.append(
            DecodedEvent(
                event,
                line_start,
                line_end,
                line_start if field_start is None else field_start,
            )
        )
        self._event_count += 1


__all__ = ["DecodedEvent", "EventDecoder"]
