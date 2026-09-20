"""Internal runtime lexical token streaming for JSON-like qexp payloads.

This extraction deliberately performs lexical validation only.  It does not
validate JSON structure, object schemas, projections, or uniqueness rules.
It does not certify Group membership or publish runtime authority.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import BinaryIO, Callable

_MAX_SPAN = 65_536
_MAX_FRAGMENT_B64_LENGTH = 4 * ((_MAX_SPAN + 2) // 3)
_WHITESPACE = b" \t\r\n"
_PUNCTUATION = b"{}[]:,"
_SIMPLE_ESCAPES = b'"\\/bfnrt'
_HEX_DIGITS = b"0123456789abcdefABCDEF"
_TOKEN_KINDS = {"string", "number", "literal"}
_NUMBER_STATES = {"minus", "zero", "int", "frac_dot", "frac_digits", "exp_marker", "exp_sign", "exp_digits"}
_LITERAL_TARGETS = {"true", "false", "null", "NaN", "Infinity", "-Infinity"}
_STRING_STATES = {"normal", "escape", "unicode"}
_UTF8_BOUNDS = {(0x80, 0xBF), (0xA0, 0xBF), (0x80, 0x9F), (0x90, 0xBF), (0x80, 0x8F)}
_UTF8_PENDING_BOUNDS = {
    (1, True): {(0x80, 0xBF)},
    (2, True): {(0xA0, 0xBF), (0x80, 0xBF), (0x80, 0x9F)},
    (3, True): {(0x90, 0xBF), (0x80, 0xBF), (0x80, 0x8F)},
    (2, False): {(0x90, 0xBF), (0x80, 0xBF), (0x80, 0x8F)},
    (1, False): _UTF8_BOUNDS,
}
_SNAPSHOT_KEYS = frozenset(
    {
        "version",
        "chunk_bytes",
        "raw_mode",
        "offset",
        "is_complete",
        "next_token_id",
        "token_kind",
        "token_id",
        "fragment_start",
        "fragment_b64",
        "number_state",
        "literal_target",
        "literal_progress",
        "string_state",
        "unicode_digits",
        "utf8_remaining",
        "utf8_first_min",
        "utf8_first_max",
        "utf8_first",
    }
)


@dataclass(frozen=True, slots=True)
class Span:
    """A bounded source-byte span belonging to one lexical token."""

    token_id: int
    kind: str
    start: int
    end: int
    is_final: bool


@dataclass(frozen=True, slots=True)
class StepResult:
    """Progress returned by one scanner step.

    ``bytes_read`` counts bytes fetched from the source during the step, while
    ``bytes_processed`` counts bytes consumed from the fetched input. Each is
    bounded by the step's byte budget; a fragment pause can leave buffered
    bytes for a later step, so the two values may differ.
    """

    bytes_read: int
    is_complete: bool
    tokens_finished: int
    bytes_processed: int = 0


class Scanner:
    """Incrementally lex a binary JSON stream without retaining token values.

    Args:
        source: A binary file-like object positioned at offset zero.
        emit: Callback receiving each nonempty token fragment.
        chunk_bytes: Maximum source read size, up to 65536. Token fragments
            have a separate fixed limit of 65536 bytes.
        emit_bytes: Optional callback receiving the same span and its raw
            source bytes, when supplied.

    Raises:
        ValueError: If the source position, chunk size, budget, or lexical
            input is invalid.
    """

    def __init__(
        self,
        source: BinaryIO,
        emit: Callable[[Span], None],
        chunk_bytes: int = _MAX_SPAN,
        *,
        emit_bytes: Callable[[Span, bytes], None] | None = None,
    ):
        if not callable(emit):
            raise TypeError("emit must be callable")
        if emit_bytes is not None and not callable(emit_bytes):
            raise TypeError("emit_bytes must be callable")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
            raise TypeError("chunk_bytes must be an integer")
        if not 1 <= chunk_bytes <= _MAX_SPAN:
            raise ValueError("chunk_bytes must be between 1 and 65536")
        try:
            position = source.tell()
        except (AttributeError, OSError, io.UnsupportedOperation) as exc:
            raise ValueError("source must expose an initial file offset of zero") from exc
        if position != 0:
            raise ValueError("source must be positioned at offset zero")

        self._source = source
        self._emit = emit
        self._emit_bytes = emit_bytes
        self._chunk_bytes = chunk_bytes
        self._chunk: bytes | bytearray | memoryview | None = None
        self._chunk_pos = 0
        self._offset = 0
        self._eof_seen = False
        self._is_complete = False
        self._next_token_id = 0
        self._tokens_finished = 0
        self._step_fragments_emitted = 0
        self._stepping = False
        self._failed = False

        self._token_kind: str | None = None
        self._token_id: int | None = None
        self._fragment_start: int | None = None
        self._fragment_bytes = bytearray() if emit_bytes is not None else None

        self._number_state: str | None = None
        self._literal_target: bytes | None = None
        self._literal_progress = 0
        self._string_state: str | None = None
        self._unicode_digits = 0
        self._utf8_remaining = 0
        self._utf8_first_min = 0
        self._utf8_first_max = 0
        self._utf8_first = False

    def step(self, byte_budget: int, *, max_fragments: int | None = None) -> StepResult:
        """Fetch and process input within a byte budget and optional fragment limit."""
        self._validate_step_arguments(byte_budget, max_fragments)
        if self._stepping:
            raise RuntimeError("step cannot be called reentrantly")
        if self._failed:
            raise RuntimeError("scanner is invalid after a previous step failure")
        self._stepping = True
        try:
            return self._step_impl(byte_budget, max_fragments)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._stepping = False

    @staticmethod
    def _validate_step_arguments(byte_budget: int, max_fragments: int | None) -> None:
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int):
            raise TypeError("byte_budget must be an integer")
        if byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if max_fragments is not None:
            if isinstance(max_fragments, bool) or not isinstance(max_fragments, int):
                raise TypeError("max_fragments must be an integer or None")
            if max_fragments < 2:
                raise ValueError("max_fragments must be at least 2")

    def _step_impl(self, byte_budget: int, max_fragments: int | None) -> StepResult:
        if self._is_complete:
            return StepResult(0, True, 0)

        request_budget = byte_budget
        bytes_processed = 0
        bytes_read = 0
        self._tokens_finished = 0
        self._step_fragments_emitted = 0
        while bytes_processed < byte_budget:
            if self._chunk is None:
                if max_fragments is not None and self._step_fragments_emitted >= max_fragments:
                    break
                if self._eof_seen:
                    self._finish_eof()
                    self._is_complete = True
                    break
                if request_budget <= 0:
                    break
                request_size = min(self._chunk_bytes, request_budget, byte_budget - bytes_processed)
                data = self._source.read(request_size)
                request_budget -= request_size
                if data is None:
                    raise TypeError("source.read() must return bytes")
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    raise TypeError("source.read() must return bytes")
                if len(data) > request_size:
                    raise ValueError("source.read() returned more bytes than requested")
                bytes_read += len(data)
                if not data:
                    self._eof_seen = True
                    if max_fragments is None or self._step_fragments_emitted < max_fragments:
                        self._finish_eof()
                        self._is_complete = True
                    break
                self._chunk = data
                self._chunk_pos = 0

            if max_fragments is not None and max_fragments - self._step_fragments_emitted < 2:
                break
            remaining = len(self._chunk) - self._chunk_pos
            take = min(remaining, byte_budget - bytes_processed)
            chunk = self._chunk
            index = self._chunk_pos
            chunk_end = self._chunk_pos + take
            while index < chunk_end:
                if max_fragments is not None and max_fragments - self._step_fragments_emitted < 2:
                    break
                byte = chunk[index]
                self._process_byte(byte, self._offset)
                self._offset += 1
                bytes_processed += 1
                index += 1
            self._chunk_pos = index
            if self._chunk_pos == len(chunk):
                self._chunk = None
                self._chunk_pos = 0
            if index < chunk_end:
                break

        return StepResult(bytes_read, self._is_complete, self._tokens_finished, bytes_processed)

    def snapshot(self) -> dict[str, object]:
        """Return the lexical state as a JSON-native checkpoint."""
        if self._stepping:
            raise RuntimeError("cannot snapshot while a step is running")
        if self._failed:
            raise RuntimeError("scanner cannot be snapshotted after a step failure")

        raw_mode = self._emit_bytes is not None
        if raw_mode:
            if self._fragment_bytes is None:
                raise RuntimeError("raw fragment buffer is unavailable")
            fragment_b64 = base64.b64encode(bytes(self._fragment_bytes)).decode("ascii")
        else:
            fragment_b64 = None
        if self._literal_target is None:
            literal_target = None
        else:
            literal_target = self._literal_target.decode("ascii")
        utf8_first_min = self._utf8_first_min if self._utf8_remaining else 0
        utf8_first_max = self._utf8_first_max if self._utf8_remaining else 0
        snapshot: dict[str, object] = {
            "version": 1,
            "chunk_bytes": self._chunk_bytes,
            "raw_mode": raw_mode,
            "offset": self._offset,
            "is_complete": self._is_complete,
            "next_token_id": self._next_token_id,
            "token_kind": self._token_kind,
            "token_id": self._token_id,
            "fragment_start": self._fragment_start,
            "fragment_b64": fragment_b64,
            "number_state": self._number_state,
            "literal_target": literal_target,
            "literal_progress": self._literal_progress,
            "string_state": self._string_state,
            "unicode_digits": self._unicode_digits,
            "utf8_remaining": self._utf8_remaining,
            "utf8_first_min": utf8_first_min,
            "utf8_first_max": utf8_first_max,
            "utf8_first": self._utf8_first if self._utf8_remaining else False,
        }
        self._validate_snapshot(snapshot)
        return snapshot

    @classmethod
    def from_snapshot(
        cls,
        source: BinaryIO,
        emit: Callable[[Span], None],
        snapshot: object,
        *,
        emit_bytes: Callable[[Span, bytes], None] | None = None,
    ) -> "Scanner":
        """Restore a scanner from a trusted, shape-validated checkpoint."""
        if not callable(emit):
            raise TypeError("emit must be callable")
        if emit_bytes is not None and not callable(emit_bytes):
            raise TypeError("emit_bytes must be callable")
        state = cls._validate_snapshot(snapshot)
        raw_mode = state["raw_mode"]
        if raw_mode != (emit_bytes is not None):
            raise ValueError("emit_bytes mode does not match snapshot raw_mode")

        offset = state["offset"]
        try:
            position = source.tell()
        except (AttributeError, OSError, io.UnsupportedOperation) as exc:
            raise ValueError("source must be positioned at the snapshot offset") from exc
        if type(position) is not int or position != offset:
            raise ValueError("source must be positioned at the snapshot offset")

        scanner = cls.__new__(cls)
        scanner._source = source
        scanner._emit = emit
        scanner._emit_bytes = emit_bytes
        scanner._chunk_bytes = state["chunk_bytes"]
        scanner._chunk = None
        scanner._chunk_pos = 0
        scanner._offset = offset
        scanner._eof_seen = state["is_complete"]
        scanner._is_complete = state["is_complete"]
        scanner._next_token_id = state["next_token_id"]
        scanner._tokens_finished = 0
        scanner._step_fragments_emitted = 0
        scanner._stepping = False
        scanner._failed = False

        scanner._token_kind = state["token_kind"]
        scanner._token_id = state["token_id"]
        scanner._fragment_start = state["fragment_start"]
        if raw_mode:
            fragment_b64 = state["fragment_b64"]
            scanner._fragment_bytes = bytearray(base64.b64decode(fragment_b64.encode("ascii"), validate=True))
        else:
            scanner._fragment_bytes = None

        scanner._number_state = state["number_state"]
        literal_target = state["literal_target"]
        scanner._literal_target = None if literal_target is None else literal_target.encode("ascii")
        scanner._literal_progress = state["literal_progress"]
        scanner._string_state = state["string_state"]
        scanner._unicode_digits = state["unicode_digits"]
        scanner._utf8_remaining = state["utf8_remaining"]
        scanner._utf8_first_min = state["utf8_first_min"]
        scanner._utf8_first_max = state["utf8_first_max"]
        scanner._utf8_first = state["utf8_first"]
        return scanner

    @classmethod
    def _validate_snapshot(cls, snapshot: object) -> dict[str, object]:
        if type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_KEYS:
            raise ValueError("snapshot must contain exactly the version-1 scanner fields")

        def value(name: str) -> object:
            return snapshot[name]

        def integer(name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
            current = value(name)
            if type(current) is not int:
                raise ValueError(f"snapshot field {name!r} must be an integer")
            if minimum is not None and current < minimum:
                raise ValueError(f"snapshot field {name!r} is below its minimum")
            if maximum is not None and current > maximum:
                raise ValueError(f"snapshot field {name!r} is above its maximum")
            return current

        def optional_integer(name: str) -> int | None:
            current = value(name)
            if current is not None and type(current) is not int:
                raise ValueError(f"snapshot field {name!r} must be an integer or null")
            return current

        def boolean(name: str) -> bool:
            current = value(name)
            if type(current) is not bool:
                raise ValueError(f"snapshot field {name!r} must be a boolean")
            return current

        def optional_string(name: str) -> str | None:
            current = value(name)
            if current is not None and type(current) is not str:
                raise ValueError(f"snapshot field {name!r} must be a string or null")
            return current

        version = integer("version")
        if version != 1:
            raise ValueError("unsupported scanner snapshot version")
        chunk_bytes = integer("chunk_bytes", minimum=1, maximum=_MAX_SPAN)
        raw_mode = boolean("raw_mode")
        offset = integer("offset", minimum=0)
        is_complete = boolean("is_complete")
        next_token_id = integer("next_token_id", minimum=0)
        if next_token_id > offset:
            raise ValueError("token count exceeds processed bytes")
        token_kind = optional_string("token_kind")
        if token_kind is not None and token_kind not in _TOKEN_KINDS:
            raise ValueError("snapshot token_kind is invalid")
        token_id = optional_integer("token_id")
        fragment_start = optional_integer("fragment_start")
        number_state = optional_string("number_state")
        if number_state is not None and number_state not in _NUMBER_STATES:
            raise ValueError("snapshot number_state is invalid")
        literal_target = optional_string("literal_target")
        if literal_target is not None:
            try:
                literal_target.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("snapshot literal_target must be ASCII") from exc
            if literal_target not in _LITERAL_TARGETS:
                raise ValueError("snapshot literal_target is invalid")
        literal_progress = integer("literal_progress", minimum=0)
        string_state = optional_string("string_state")
        if string_state is not None and string_state not in _STRING_STATES:
            raise ValueError("snapshot string_state is invalid")
        unicode_digits = integer("unicode_digits", minimum=0)
        utf8_remaining = integer("utf8_remaining", minimum=0, maximum=3)
        utf8_first_min = integer("utf8_first_min", minimum=0, maximum=255)
        utf8_first_max = integer("utf8_first_max", minimum=0, maximum=255)
        utf8_first = boolean("utf8_first")
        fragment_b64 = optional_string("fragment_b64")

        if raw_mode:
            if fragment_b64 is None:
                raise ValueError("raw snapshots require fragment_b64")
            if len(fragment_b64) > _MAX_FRAGMENT_B64_LENGTH:
                raise ValueError("snapshot fragment_b64 exceeds the fragment limit")
            try:
                encoded = fragment_b64.encode("ascii")
                fragment = base64.b64decode(encoded, validate=True)
            except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
                raise ValueError("snapshot fragment_b64 is not canonical base64") from exc
            if base64.b64encode(fragment).decode("ascii") != fragment_b64:
                raise ValueError("snapshot fragment_b64 is not canonical base64")
        else:
            if fragment_b64 is not None:
                raise ValueError("span-only snapshots cannot contain fragment_b64")
            fragment = b""

        active = token_kind is not None
        if not active:
            if (
                token_id is not None
                or fragment_start is not None
                or number_state is not None
                or literal_target is not None
                or literal_progress != 0
                or string_state is not None
                or unicode_digits != 0
                or utf8_remaining != 0
                or utf8_first_min != 0
                or utf8_first_max != 0
                or utf8_first
            ):
                raise ValueError("inactive scanner state contains active-token fields")
            if fragment:
                raise ValueError("inactive raw scanner state contains fragment bytes")
        else:
            if token_id is None or fragment_start is None:
                raise ValueError("active scanner state requires token identity and fragment start")
            if token_id < 0 or next_token_id == 0 or token_id != next_token_id - 1:
                raise ValueError("active scanner token identity is inconsistent")
            fragment_length = offset - fragment_start
            if fragment_start < 0 or not 1 <= fragment_length <= _MAX_SPAN:
                raise ValueError("active scanner fragment bounds are invalid")
            if raw_mode and len(fragment) != fragment_length:
                raise ValueError("raw fragment length does not match active span")
            if token_kind == "number":
                if number_state is None:
                    raise ValueError("active number token requires number_state")
                if literal_target is not None or literal_progress != 0 or string_state is not None:
                    raise ValueError("number token contains unrelated lexical state")
                if (
                    unicode_digits != 0
                    or utf8_remaining != 0
                    or utf8_first_min != 0
                    or utf8_first_max != 0
                    or utf8_first
                ):
                    raise ValueError("number token contains string state")
            elif token_kind == "literal":
                if number_state is not None or literal_target is None:
                    raise ValueError("active literal token state is incomplete")
                if not 1 <= literal_progress <= len(literal_target.encode("ascii")):
                    raise ValueError("literal progress is out of range")
                if literal_progress != fragment_length:
                    raise ValueError("literal progress does not match active span")
                if literal_target == "-Infinity" and literal_progress < 2:
                    raise ValueError("negative Infinity literal progress is out of range")
                if string_state is not None or unicode_digits != 0:
                    raise ValueError("literal token contains string state")
                if utf8_remaining != 0 or utf8_first_min != 0 or utf8_first_max != 0 or utf8_first:
                    raise ValueError("literal token contains UTF-8 state")
            else:
                if number_state is not None or literal_target is not None or literal_progress != 0:
                    raise ValueError("string token contains unrelated lexical state")
                if string_state is None:
                    raise ValueError("active string token requires string_state")
                if string_state == "unicode":
                    if not 0 <= unicode_digits <= 3:
                        raise ValueError("unicode escape progress is out of range")
                elif unicode_digits != 0:
                    raise ValueError("unicode_digits is only valid in unicode string state")
                if utf8_remaining:
                    legal_bounds = _UTF8_PENDING_BOUNDS.get((utf8_remaining, utf8_first), set())
                    if string_state != "normal" or (utf8_first_min, utf8_first_max) not in legal_bounds:
                        raise ValueError("active UTF-8 state is invalid")
                elif utf8_first or utf8_first_min != 0 or utf8_first_max != 0:
                    raise ValueError("inactive UTF-8 state is not normalized")

        if is_complete and active:
            raise ValueError("completed scanner state cannot have an active token")
        return {
            "version": version,
            "chunk_bytes": chunk_bytes,
            "raw_mode": raw_mode,
            "offset": offset,
            "is_complete": is_complete,
            "next_token_id": next_token_id,
            "token_kind": token_kind,
            "token_id": token_id,
            "fragment_start": fragment_start,
            "fragment_b64": fragment_b64,
            "number_state": number_state,
            "literal_target": literal_target,
            "literal_progress": literal_progress,
            "string_state": string_state,
            "unicode_digits": unicode_digits,
            "utf8_remaining": utf8_remaining,
            "utf8_first_min": utf8_first_min,
            "utf8_first_max": utf8_first_max,
            "utf8_first": utf8_first,
        }

    def _process_byte(self, byte: int, offset: int) -> None:
        if self._token_kind == "string":
            self._process_string_byte(byte, offset)
            return
        if self._token_kind == "number":
            self._process_number_byte(byte, offset)
            return
        if self._token_kind == "literal":
            self._process_literal_byte(byte, offset)
            return

        if byte in _WHITESPACE:
            return
        if byte in _PUNCTUATION:
            self._begin_token(chr(byte), offset)
            self._append_byte(offset, byte)
            self._finish(offset + 1)
            return
        if byte == ord('"'):
            self._begin_token("string", offset)
            self._append_byte(offset, byte)
            self._string_state = "normal"
            return
        if byte == ord("-"):
            self._begin_token("number", offset)
            self._append_byte(offset, byte)
            self._number_state = "minus"
            return
        if byte == ord("0"):
            self._begin_token("number", offset)
            self._append_byte(offset, byte)
            self._number_state = "zero"
            return
        if ord("1") <= byte <= ord("9"):
            self._begin_token("number", offset)
            self._append_byte(offset, byte)
            self._number_state = "int"
            return
        literal = {
            ord("t"): b"true",
            ord("f"): b"false",
            ord("n"): b"null",
            ord("N"): b"NaN",
            ord("I"): b"Infinity",
        }.get(byte)
        if literal is not None:
            self._begin_token("literal", offset)
            self._append_byte(offset, byte)
            self._literal_target = literal
            self._literal_progress = 1
            return
        raise ValueError(f"invalid JSON byte 0x{byte:02x} at offset {offset}")

    def _begin_token(self, kind: str, offset: int) -> None:
        if self._token_kind is not None:
            raise RuntimeError("cannot begin a token while another token is active")
        self._token_kind = kind
        self._token_id = self._next_token_id
        self._next_token_id += 1
        self._fragment_start = offset

    def _append_byte(self, offset: int, byte: int) -> None:
        fragment_start = self._fragment_start
        if fragment_start is None or self._token_id is None or self._token_kind is None:
            raise RuntimeError("no active token")
        if offset - fragment_start >= _MAX_SPAN:
            self._emit_fragment(Span(self._token_id, self._token_kind, fragment_start, offset, False))
            self._fragment_start = offset
        if self._fragment_bytes is not None:
            self._fragment_bytes.append(byte)

    def _emit_fragment(self, span: Span) -> None:
        self._emit(span)
        if self._emit_bytes is not None:
            fragment_bytes = self._fragment_bytes
            if fragment_bytes is None:
                raise RuntimeError("raw fragment buffer is unavailable")
            self._emit_bytes(span, bytes(fragment_bytes))
            fragment_bytes.clear()
        self._step_fragments_emitted += 1

    def _finish(self, end: int) -> None:
        if self._fragment_start is None or self._token_id is None or self._token_kind is None:
            raise RuntimeError("no active token")
        if end <= self._fragment_start or end - self._fragment_start > _MAX_SPAN:
            raise RuntimeError("invalid token fragment bounds")
        self._emit_fragment(Span(self._token_id, self._token_kind, self._fragment_start, end, True))
        self._tokens_finished += 1
        self._token_kind = None
        self._token_id = None
        self._fragment_start = None
        self._number_state = None
        self._literal_target = None
        self._literal_progress = 0
        self._string_state = None
        self._unicode_digits = 0
        self._utf8_remaining = 0
        self._utf8_first = False

    def _process_string_byte(self, byte: int, offset: int) -> None:
        self._append_byte(offset, byte)
        state = self._string_state
        if state == "escape":
            if byte in _SIMPLE_ESCAPES:
                self._string_state = "normal"
            elif byte == ord("u"):
                self._string_state = "unicode"
                self._unicode_digits = 0
            else:
                raise ValueError(f"invalid string escape at offset {offset}")
            return
        if state == "unicode":
            if byte not in _HEX_DIGITS:
                raise ValueError(f"invalid unicode escape at offset {offset}")
            self._unicode_digits += 1
            if self._unicode_digits == 4:
                self._string_state = "normal"
                self._unicode_digits = 0
            return
        if self._utf8_remaining:
            if not self._utf8_first:
                if not 0x80 <= byte <= 0xBF:
                    raise ValueError(f"invalid UTF-8 continuation at offset {offset}")
            elif not self._utf8_first_min <= byte <= self._utf8_first_max:
                raise ValueError(f"invalid UTF-8 continuation at offset {offset}")
            self._utf8_first = False
            self._utf8_remaining -= 1
            return
        if byte == ord('"'):
            self._finish(offset + 1)
            return
        if byte == ord("\\"):
            self._string_state = "escape"
            return
        if byte < 0x20:
            raise ValueError(f"unescaped control byte at offset {offset}")
        if byte < 0x80:
            return
        if 0xC2 <= byte <= 0xDF:
            self._utf8_remaining = 1
            self._utf8_first_min, self._utf8_first_max = 0x80, 0xBF
            self._utf8_first = True
        elif byte == 0xE0:
            self._utf8_remaining = 2
            self._utf8_first_min, self._utf8_first_max = 0xA0, 0xBF
            self._utf8_first = True
        elif 0xE1 <= byte <= 0xEC or 0xEE <= byte <= 0xEF:
            self._utf8_remaining = 2
            self._utf8_first_min, self._utf8_first_max = 0x80, 0xBF
            self._utf8_first = True
        elif byte == 0xED:
            self._utf8_remaining = 2
            self._utf8_first_min, self._utf8_first_max = 0x80, 0x9F
            self._utf8_first = True
        elif byte == 0xF0:
            self._utf8_remaining = 3
            self._utf8_first_min, self._utf8_first_max = 0x90, 0xBF
            self._utf8_first = True
        elif 0xF1 <= byte <= 0xF3:
            self._utf8_remaining = 3
            self._utf8_first_min, self._utf8_first_max = 0x80, 0xBF
            self._utf8_first = True
        elif byte == 0xF4:
            self._utf8_remaining = 3
            self._utf8_first_min, self._utf8_first_max = 0x80, 0x8F
            self._utf8_first = True
        else:
            raise ValueError(f"invalid UTF-8 lead byte at offset {offset}")

    def _process_number_byte(self, byte: int, offset: int) -> None:
        state = self._number_state
        if state in {"zero", "int", "frac_digits", "exp_digits"}:
            is_continuation = (
                (state == "zero" and byte in (ord("."), ord("e"), ord("E")))
                or (state == "int" and (ord("0") <= byte <= ord("9") or byte in (ord("."), ord("e"), ord("E"))))
                or (state == "frac_digits" and (ord("0") <= byte <= ord("9") or byte in (ord("e"), ord("E"))))
                or (state == "exp_digits" and ord("0") <= byte <= ord("9"))
            )
            if not is_continuation:
                if state == "zero" and ord("0") <= byte <= ord("9"):
                    self._append_byte(offset, byte)
                    raise ValueError(f"leading zero in number at offset {offset}")
                self._finish(offset)
                self._process_byte(byte, offset)
                return

        self._append_byte(offset, byte)
        if state == "minus":
            if byte == ord("I"):
                self._token_kind = "literal"
                self._number_state = None
                self._literal_target = b"-Infinity"
                self._literal_progress = 2
            elif byte == ord("0"):
                self._number_state = "zero"
            elif ord("1") <= byte <= ord("9"):
                self._number_state = "int"
            else:
                raise ValueError(f"invalid number after '-' at offset {offset}")
        elif state == "zero":
            if byte == ord("."):
                self._number_state = "frac_dot"
            elif byte in (ord("e"), ord("E")):
                self._number_state = "exp_marker"
            else:
                raise ValueError(f"invalid number continuation at offset {offset}")
        elif state == "int":
            if ord("0") <= byte <= ord("9"):
                return
            if byte == ord("."):
                self._number_state = "frac_dot"
            elif byte in (ord("e"), ord("E")):
                self._number_state = "exp_marker"
            else:
                raise ValueError(f"invalid number continuation at offset {offset}")
        elif state == "frac_dot":
            if ord("0") <= byte <= ord("9"):
                self._number_state = "frac_digits"
            else:
                raise ValueError(f"fraction requires a digit at offset {offset}")
        elif state == "frac_digits":
            if ord("0") <= byte <= ord("9"):
                return
            if byte in (ord("e"), ord("E")):
                self._number_state = "exp_marker"
            else:
                raise ValueError(f"invalid fraction continuation at offset {offset}")
        elif state == "exp_marker":
            if byte in (ord("+"), ord("-")):
                self._number_state = "exp_sign"
            elif ord("0") <= byte <= ord("9"):
                self._number_state = "exp_digits"
            else:
                raise ValueError(f"exponent requires a digit at offset {offset}")
        elif state == "exp_sign":
            if ord("0") <= byte <= ord("9"):
                self._number_state = "exp_digits"
            else:
                raise ValueError(f"exponent requires a digit at offset {offset}")
        elif state == "exp_digits":
            if not ord("0") <= byte <= ord("9"):
                raise ValueError(f"invalid exponent continuation at offset {offset}")
        else:
            raise RuntimeError("unknown number state")

    def _process_literal_byte(self, byte: int, offset: int) -> None:
        target = self._literal_target
        if target is None:
            raise RuntimeError("literal has no target")
        if self._literal_progress < len(target):
            self._append_byte(offset, byte)
            if byte != target[self._literal_progress]:
                raise ValueError(f"invalid literal at offset {offset}")
            self._literal_progress += 1
            return
        self._finish(offset)
        self._process_byte(byte, offset)

    def _finish_eof(self) -> None:
        if self._token_kind is None:
            return
        if self._token_kind == "string":
            if self._utf8_remaining:
                raise ValueError("incomplete UTF-8 sequence at end of string")
            raise ValueError("unterminated string at end of input")
        if self._token_kind == "literal":
            if self._literal_target is not None and self._literal_progress == len(self._literal_target):
                self._finish(self._offset)
                return
            raise ValueError("incomplete literal at end of input")
        if self._number_state in {"zero", "int", "frac_digits", "exp_digits"}:
            self._finish(self._offset)
            return
        raise ValueError("incomplete number at end of input")
