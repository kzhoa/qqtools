"""Internal runtime structural validation and bounded Submission projection.

This is a source-level extraction. It validates one JSON value while it is
fed lexical spans by :class:`json_stream.Scanner`, and projects
selected task IDs and membership sequence numbers without rereading the source
or retaining arrays or identifiers. Chunks and field ends are provisional:
the caller must discard them when ``feed`` or ``finish`` raises. Duplicate
task/sequence checks, source restart proof, persistence, and certification are
deliberately outside this structural prototype. Session-level source-version
qualification consumes the pinned marker recorded here. Field fingerprints use
the versioned ``ChainedDigest`` v1 over normalized decoded bytes; they are not
authenticating hashes and do not by themselves enable parser resumption.

The lexer remains responsible for UTF-8 and JSON-token lexical validation.
This module validates JSON structure and incrementally decodes only keys,
selected metadata, and selected task IDs. A root ``meta.schema_version`` of
``6`` is recognized as the pinned source qualification marker; unrelated meta
fields remain outside this extraction's boundary. It does not certify Group
membership or publish runtime authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from ..records import validate_group_name, validate_identifier
from .fingerprint import ChainedDigest
from .json_stream import Span

_MAX_FRAGMENT = 65_536
_GROUP_MAX_LENGTH = 64
_IDENTIFIER_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_SIMPLE_ESCAPES = {
    ord('"'): ord('"'),
    ord("\\"): ord("\\"),
    ord("/"): ord("/"),
    ord("b"): 8,
    ord("f"): 12,
    ord("n"): 10,
    ord("r"): 13,
    ord("t"): 9,
}
_KNOWN_STATES = frozenset({"preparing", "committing", "committed", "aborted", "blocked"})
_MAX_STATE_LENGTH = max(map(len, _KNOWN_STATES))

_CHECKPOINT_VERSION = 2
_FINGERPRINT_VERSION = 1
SUPPORTED_SOURCE_SCHEMA_VERSION = 6
_CHECKPOINT_TOP_KEYS = frozenset(
    {
        "version",
        "fingerprint_version",
        "expected_operation_id",
        "expected_group",
        "stack",
        "state",
        "active",
        "summary",
    }
)
_CHECKPOINT_STATE_KEYS = frozenset(
    {
        "root_done",
        "last_token_id",
        "shape_error",
        "metadata_error",
        "operation_seen",
        "operation_matches",
        "target_group_seen",
        "target_group_matches",
        "target_group_valid",
        "state",
        "resolved_context_seen",
        "task_ids_array_seen",
        "task_count",
        "commit_plan_seen",
        "sequences_array_seen",
        "sequence_count",
        "meta_seen",
        "schema_version_seen",
        "source_schema_version",
    }
)
_CHECKPOINT_FRAME_KEYS = frozenset({"role", "kind", "state", "seen", "pending_key", "count"})
_CHECKPOINT_ACTIVE_KEYS = frozenset(
    {"id", "kind", "start", "end", "handler", "ordinal", "data_valid", "decoder", "sequence", "literal"}
)
_CHECKPOINT_DECODER_KEYS = frozenset(
    {
        "purpose",
        "state",
        "unicode_value",
        "unicode_digits",
        "closed",
        "non_ascii",
        "too_long",
        "prefix",
        "identifier_prefix",
        "length",
        "matches",
        "valid_identifier",
        "decoded_size",
        "digest",
    }
)
_CHECKPOINT_SEQUENCE_KEYS = frozenset({"ordinal", "valid", "seen", "decoded_size", "digest"})
_CHECKPOINT_LITERAL_KEYS = frozenset({"position", "valid"})
_CHECKPOINT_SUMMARY_KEYS = frozenset({"state", "matches_group", "task_count", "sequence_count"})
_STRING_PURPOSES = frozenset({"key", "operation_id", "target_group", "state", "task_id"})
_FRAME_ROLES = frozenset({"root", "submission", "resolved", "plan", "meta", "task_ids", "sequences", "ignore"})
_FRAME_STATES = frozenset({"key_or_end", "key", "colon", "value", "child", "comma_or_end", "value_or_end"})

_ROOT_KEYS = {"submission": 1, "meta": 2}
_SUBMISSION_KEYS = {
    "operation_id": 1,
    "target_group": 2,
    "state": 4,
    "resolved_context": 8,
    "commit_plan": 16,
}
_RESOLVED_KEYS = {"task_ids": 1}
_PLAN_KEYS = {"group_membership_sequences": 1}
_META_KEYS = {"schema_version": 1}

_FrameRole = Literal["root", "submission", "resolved", "plan", "meta", "task_ids", "sequences", "ignore"]
_FrameKind = Literal["object", "array"]
_TokenHandler = Literal[
    "key",
    "operation_id",
    "target_group",
    "state",
    "task_id",
    "sequence",
    "schema_version",
    "null_target_group",
    "null_commit_plan",
    "ignored",
]

_ACTIVE_HANDLERS = frozenset(
    {
        "key",
        "operation_id",
        "target_group",
        "state",
        "task_id",
        "sequence",
        "schema_version",
        "null_target_group",
        "null_commit_plan",
        "ignored",
    }
)


@dataclass(frozen=True, slots=True)
class FieldChunk:
    """A provisional bounded fragment of one selected scalar field."""

    kind: str
    ordinal: int
    data: bytes
    is_final: bool


@dataclass(frozen=True, slots=True)
class FieldEnd:
    """A provisional ChainedDigest v1 fingerprint and source span.

    The fingerprint is a comparison aid, not an authenticating hash or parser
    resume state.
    """

    kind: str
    ordinal: int
    digest: str
    decoded_size: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ProjectionSummary:
    """The bounded metadata result returned after a complete valid projection."""

    state: str
    matches_group: bool
    task_count: int
    sequence_count: int


@dataclass(slots=True)
class _Frame:
    role: _FrameRole
    kind: _FrameKind
    state: str
    seen: int = 0
    pending_key: str | None = None
    count: int = 0


class _StringDecoder:
    """Decode a JSON string incrementally, retaining one fragment at most."""

    def __init__(self, purpose: str, projection: "SubmissionProjection") -> None:
        self.purpose = purpose
        self.projection = projection
        self.state = "opening"
        self.unicode_value = 0
        self.unicode_digits = 0
        self.closed = False
        self.non_ascii = False
        self.too_long = False
        self.prefix = bytearray()
        self.identifier_prefix = bytearray()
        self.fragment_output = bytearray()
        self.fragment_invalid = False
        self.length = 0
        self.matches = True
        self.valid_identifier = True
        self.expected = projection._expected_operation_id if purpose == "operation_id" else None
        self.digest = ChainedDigest() if purpose == "task_id" else None
        self.decoded_size = 0

    def feed(self, raw: bytes) -> bytes:
        """Consume one source fragment and return decoded task bytes, if any."""
        self.fragment_output.clear()
        self.fragment_invalid = False
        for byte in raw:
            self._feed_byte(byte)
        decoded = bytes(self.fragment_output)
        if self.purpose == "task_id" and self.digest is not None:
            self.digest.update(decoded)
        return decoded

    def _feed_byte(self, byte: int) -> None:
        if self.closed:
            self.projection._fail("bytes followed a completed JSON string token")
        if self.state == "opening":
            if byte != ord('"'):
                self.projection._fail("JSON string token did not start with a quote")
            self.state = "normal"
            return
        if self.state == "escape":
            simple = _SIMPLE_ESCAPES.get(byte)
            if simple is not None:
                self._decoded(simple)
                self.state = "normal"
                return
            if byte == ord("u"):
                self.state = "unicode"
                self.unicode_value = 0
                self.unicode_digits = 0
                return
            self.projection._fail(f"invalid JSON string escape at source byte 0x{byte:02x}")
        if self.state == "unicode":
            if byte not in b"0123456789abcdefABCDEF":
                self.projection._fail("invalid JSON unicode escape")
            self.unicode_value = self.unicode_value * 16 + int(chr(byte), 16)
            self.unicode_digits += 1
            if self.unicode_digits == 4:
                self._decoded(self.unicode_value)
                self.state = "normal"
                self.unicode_value = 0
                self.unicode_digits = 0
            return
        if byte == ord('"'):
            self.closed = True
            self.state = "closed"
            return
        if byte == ord("\\"):
            self.state = "escape"
            return
        if byte < 0x20:
            self.projection._fail("unescaped control byte in JSON string")
        self._decoded(byte)

    def _decoded(self, value: int) -> None:
        self.length += 1
        if value > 0x7F:
            self.non_ascii = True
            self.fragment_invalid = True
        if self.purpose == "key":
            if value > 0x7F or self.non_ascii:
                return
            if len(self.prefix) >= self.projection._max_key_length:
                self.too_long = True
            elif not self.too_long:
                self.prefix.append(value)
            return
        if self.purpose == "operation_id":
            expected = self.expected
            if (
                value > 0x7F
                or expected is None
                or self.length > len(expected)
                or expected[self.length - 1] != chr(value)
            ):
                self.matches = False
            return
        if self.purpose == "state":
            if value > 0x7F or self.non_ascii:
                return
            if self.length > _MAX_STATE_LENGTH:
                self.too_long = True
            elif not self.too_long:
                self.prefix.append(value)
            return
        if self.purpose == "target_group":
            if value <= 0x7F and len(self.identifier_prefix) < _GROUP_MAX_LENGTH:
                self.identifier_prefix.append(value)
            if value > 0x7F or value not in _IDENTIFIER_BYTES:
                self.valid_identifier = False
                self.fragment_invalid = True
            if self.length > _GROUP_MAX_LENGTH:
                self.valid_identifier = False
                self.too_long = True
            if self.length <= len(self.projection._expected_group):
                self.matches = self.matches and self.projection._expected_group[self.length - 1] == chr(value)
            else:
                self.matches = False
            return
        if self.purpose == "task_id":
            if value not in _IDENTIFIER_BYTES:
                self.fragment_invalid = True
                return
            self.fragment_output.append(value)
            self.decoded_size += 1
            return
        self.projection._fail(f"unknown string decoder purpose {self.purpose!r}")

    def finish(self) -> None:
        """Require a complete quoted string."""
        if not self.closed or self.state != "closed":
            self.projection._fail("incomplete JSON string token")


class _SequenceCapture:
    """Stream and validate one positive decimal sequence scalar."""

    def __init__(self, projection: "SubmissionProjection", ordinal: int) -> None:
        self.projection = projection
        self.ordinal = ordinal
        self.digest = ChainedDigest()
        self.decoded_size = 0
        self.valid = True
        self.seen = 0

    def feed(self, raw: bytes, is_final: bool) -> None:
        fragment_valid = True
        for byte in raw:
            if self.seen == 0:
                valid = ord("1") <= byte <= ord("9")
            else:
                valid = ord("0") <= byte <= ord("9")
            if not valid:
                self.valid = False
                fragment_valid = False
            else:
                self.seen += 1
        if fragment_valid and self.valid:
            self.digest.update(raw)
            self.decoded_size += len(raw)
            self.projection._emit_chunk_data("sequence", self.ordinal, raw, is_final)


class _LiteralCapture:
    """Bounded capture for the only literal shape accepted by metadata."""

    def __init__(self, expected: bytes) -> None:
        self.expected = expected
        self.position = 0
        self.valid = True

    def feed(self, raw: bytes) -> None:
        for byte in raw:
            if self.position >= len(self.expected) or byte != self.expected[self.position]:
                self.valid = False
            self.position += 1

    def finish(self) -> bool:
        return self.valid and self.position == len(self.expected)


def _checkpoint_dict(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{label} must contain exactly the version-1 fields")
    return value


def _checkpoint_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be a non-boolean integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    return value


def _checkpoint_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _checkpoint_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _checkpoint_optional_str(value: object, label: str) -> str | None:
    if value is not None and type(value) is not str:
        raise ValueError(f"{label} must be a string or null")
    return value


def _checkpoint_ascii(value: object, label: str) -> str:
    string = _checkpoint_str(value, label)
    try:
        string.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain only ASCII characters") from exc
    return string


def _checkpoint_digest(value: object, label: str) -> ChainedDigest:
    if type(value) is not dict:
        raise ValueError(f"{label} must contain a ChainedDigest snapshot")
    try:
        return ChainedDigest.from_snapshot(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid ChainedDigest snapshot") from exc


class SubmissionProjection:
    """Validate and project one streamed JSON Submission document.

    Args:
        expected_operation_id: Existing ASCII operation identifier to match.
        expected_group: Existing non-null group identifier to match.
        emit_chunk: Callback for provisional selected scalar fragments.
        emit_end: Callback for provisional valid selected scalar digests.

    ``feed`` accepts the ``Span`` and raw bytes delivered by the lexer's
    ``emit_bytes`` callback. ``finish`` must be called only after lexical EOF.
    """

    def __init__(
        self,
        expected_operation_id: str,
        expected_group: str,
        emit_chunk: Callable[[FieldChunk], None],
        emit_end: Callable[[FieldEnd], None],
    ) -> None:
        validate_identifier(expected_operation_id, "expected_operation_id")
        validated_group = validate_group_name(expected_group)
        if validated_group is None:
            raise ValueError("expected_group must be a non-null identifier")
        if not callable(emit_chunk):
            raise TypeError("emit_chunk must be callable")
        if not callable(emit_end):
            raise TypeError("emit_end must be callable")
        self._expected_operation_id = expected_operation_id
        self._expected_group = validated_group
        self._emit_chunk_fn = emit_chunk
        self._emit_end_fn = emit_end
        self._stack: list[_Frame] = []
        self._root_done = False
        self._active_id: int | None = None
        self._active_kind: str | None = None
        self._active_start: int | None = None
        self._active_end: int | None = None
        self._active_handler: _TokenHandler | None = None
        self._active_decoder: _StringDecoder | None = None
        self._active_sequence: _SequenceCapture | None = None
        self._active_literal: _LiteralCapture | None = None
        self._active_ordinal = 0
        self._active_data_valid = True
        self._last_token_id = -1
        self._invalid = False
        self._summary: ProjectionSummary | None = None
        self._busy = False
        self._shape_error: str | None = None
        self._metadata_error: str | None = None
        self._operation_seen = False
        self._operation_matches = False
        self._target_group_seen = False
        self._target_group_matches = False
        self._target_group_valid = True
        self._state: str | None = None
        self._resolved_context_seen = False
        self._task_ids_array_seen = False
        self._task_count = 0
        self._commit_plan_seen = False
        self._sequences_array_seen = False
        self._sequence_count = 0
        self._meta_seen = False
        self._schema_version_seen = False
        self._source_schema_version: int | None = None

    @property
    def _max_key_length(self) -> int:
        return max(
            len(key) for keys in (_ROOT_KEYS, _SUBMISSION_KEYS, _RESOLVED_KEYS, _PLAN_KEYS, _META_KEYS) for key in keys
        )

    @property
    def source_schema_version(self) -> int | None:
        """Return the supported source schema marker after structural completion."""
        if self._summary is None:
            raise ValueError("source schema version is unavailable before projection completion")
        return self._source_schema_version

    def feed(self, span: Span, raw: bytes) -> None:
        """Consume one lexer span and its exact source bytes."""
        if self._busy:
            raise RuntimeError("submission projection is busy")
        if self._invalid:
            raise ValueError("submission projection is invalid")
        if self._summary is not None:
            raise ValueError("submission projection has already finished")
        self._busy = True
        try:
            self._validate_span(span, raw)
            if self._active_id is None:
                self._begin_token(span)
            else:
                self._continue_token(span)
            self._consume_fragment(raw, span.is_final)
            if span.is_final:
                self._finish_token()
        except Exception:
            self._invalidate()
            raise
        finally:
            self._busy = False

    def finish(self) -> ProjectionSummary:
        """Complete validation and return an immutable summary."""
        if self._busy:
            raise RuntimeError("submission projection is busy")
        if self._summary is not None:
            return self._summary
        if self._invalid:
            raise ValueError("submission projection is invalid")
        self._busy = True
        try:
            if self._active_id is not None:
                self._fail("input ended in the middle of a lexical token")
            if not self._root_done or self._stack:
                self._fail("input did not contain exactly one complete root object")
            missing = []
            if not self._operation_seen:
                missing.append("operation_id")
            if not self._target_group_seen:
                missing.append("target_group")
            if self._state is None:
                missing.append("state")
            if missing:
                self._fail(f"submission is missing required metadata: {', '.join(missing)}")
            if self._metadata_error is not None:
                self._fail(self._metadata_error)
            if not self._operation_matches:
                self._fail("submission operation_id does not match the expected operation")
            if self._shape_error is not None and self._state == "committed" and self._target_group_matches:
                self._fail(self._shape_error)
            candidate = self._state == "committed" and self._target_group_matches
            if candidate:
                if not self._resolved_context_seen or not self._task_ids_array_seen:
                    self._fail("committed matching submission is missing resolved_context.task_ids")
                if not self._commit_plan_seen or not self._sequences_array_seen:
                    self._fail("committed matching submission is missing commit_plan.group_membership_sequences")
                if self._task_count != self._sequence_count:
                    self._fail("selected task and sequence arrays have different lengths")
            state = self._state
            if state is None:
                self._fail("submission state was not captured")
            self._summary = ProjectionSummary(state, self._target_group_matches, self._task_count, self._sequence_count)
            return self._summary
        except Exception:
            self._invalidate()
            raise
        finally:
            self._busy = False

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-native checkpoint of the parser state.

        The checkpoint describes state after a completed :meth:`feed` call. It
        intentionally omits transient fragment buffers and has no source or
        authenticity binding; callers must provide those separately when they
        later resume a source.
        """
        if self._busy:
            raise RuntimeError("submission projection is busy")
        if self._invalid:
            raise ValueError("cannot snapshot an invalid submission projection")
        self._busy = True
        try:
            return {
                "version": _CHECKPOINT_VERSION,
                "fingerprint_version": _FINGERPRINT_VERSION,
                "expected_operation_id": self._expected_operation_id,
                "expected_group": self._expected_group,
                "stack": [self._snapshot_frame(frame) for frame in self._stack],
                "state": self._snapshot_state(),
                "active": self._snapshot_active(),
                "summary": self._snapshot_summary(),
            }
        finally:
            self._busy = False

    @classmethod
    def from_snapshot(
        cls,
        expected_operation_id: str,
        expected_group: str,
        emit_chunk: Callable[[FieldChunk], None],
        emit_end: Callable[[FieldEnd], None],
        snapshot: object,
    ) -> "SubmissionProjection":
        """Create a fresh projection from a trusted JSON-native checkpoint."""
        projection = cls(expected_operation_id, expected_group, emit_chunk, emit_end)
        projection._restore_snapshot(snapshot)
        return projection

    def _snapshot_state(self) -> dict[str, object]:
        return {
            "root_done": self._root_done,
            "last_token_id": self._last_token_id,
            "shape_error": self._shape_error,
            "metadata_error": self._metadata_error,
            "operation_seen": self._operation_seen,
            "operation_matches": self._operation_matches,
            "target_group_seen": self._target_group_seen,
            "target_group_matches": self._target_group_matches,
            "target_group_valid": self._target_group_valid,
            "state": self._state,
            "resolved_context_seen": self._resolved_context_seen,
            "task_ids_array_seen": self._task_ids_array_seen,
            "task_count": self._task_count,
            "commit_plan_seen": self._commit_plan_seen,
            "sequences_array_seen": self._sequences_array_seen,
            "sequence_count": self._sequence_count,
            "meta_seen": self._meta_seen,
            "schema_version_seen": self._schema_version_seen,
            "source_schema_version": self._source_schema_version,
        }

    def _snapshot_frame(self, frame: _Frame) -> dict[str, object]:
        return {
            "role": frame.role,
            "kind": frame.kind,
            "state": frame.state,
            "seen": frame.seen,
            "pending_key": frame.pending_key,
            "count": frame.count,
        }

    def _snapshot_active(self) -> dict[str, object] | None:
        if self._active_id is None:
            return None
        decoder = self._active_decoder
        sequence = self._active_sequence
        literal = self._active_literal
        start = self._active_start
        end = self._active_end
        kind = self._active_kind
        handler = self._active_handler
        if start is None or end is None or kind is None or handler is None:
            raise ValueError("active projection token is incomplete")
        return {
            "id": self._active_id,
            "kind": kind,
            "start": start,
            "end": end,
            "handler": handler,
            "ordinal": self._active_ordinal,
            "data_valid": self._active_data_valid,
            "decoder": self._snapshot_decoder(decoder),
            "sequence": self._snapshot_sequence(sequence),
            "literal": self._snapshot_literal(literal),
        }

    def _snapshot_decoder(self, decoder: _StringDecoder | None) -> dict[str, object] | None:
        if decoder is None:
            return None
        digest = decoder.digest.snapshot() if decoder.digest is not None else None
        return {
            "purpose": decoder.purpose,
            "state": decoder.state,
            "unicode_value": decoder.unicode_value,
            "unicode_digits": decoder.unicode_digits,
            "closed": decoder.closed,
            "non_ascii": decoder.non_ascii,
            "too_long": decoder.too_long,
            "prefix": bytes(decoder.prefix).decode("ascii"),
            "identifier_prefix": bytes(decoder.identifier_prefix).decode("ascii"),
            "length": decoder.length,
            "matches": decoder.matches,
            "valid_identifier": decoder.valid_identifier,
            "decoded_size": decoder.decoded_size,
            "digest": digest,
        }

    def _snapshot_sequence(self, sequence: _SequenceCapture | None) -> dict[str, object] | None:
        if sequence is None:
            return None
        return {
            "ordinal": sequence.ordinal,
            "valid": sequence.valid,
            "seen": sequence.seen,
            "decoded_size": sequence.decoded_size,
            "digest": sequence.digest.snapshot(),
        }

    def _snapshot_literal(self, literal: _LiteralCapture | None) -> dict[str, object] | None:
        if literal is None:
            return None
        return {"position": literal.position, "valid": literal.valid}

    def _snapshot_summary(self) -> dict[str, object] | None:
        if self._summary is None:
            return None
        return {
            "state": self._summary.state,
            "matches_group": self._summary.matches_group,
            "task_count": self._summary.task_count,
            "sequence_count": self._summary.sequence_count,
        }

    def _restore_snapshot(self, snapshot: object) -> None:
        top = _checkpoint_dict(snapshot, _CHECKPOINT_TOP_KEYS, "projection snapshot")
        version = _checkpoint_int(top["version"], "snapshot version")
        if version != _CHECKPOINT_VERSION:
            raise ValueError("unsupported projection snapshot version")
        fingerprint_version = _checkpoint_int(top["fingerprint_version"], "snapshot fingerprint_version")
        if fingerprint_version != _FINGERPRINT_VERSION:
            raise ValueError("unsupported projection fingerprint version")
        snapshot_operation = _checkpoint_str(top["expected_operation_id"], "snapshot expected_operation_id")
        snapshot_group = _checkpoint_str(top["expected_group"], "snapshot expected_group")
        if snapshot_operation != self._expected_operation_id or snapshot_group != self._expected_group:
            raise ValueError("projection snapshot context does not match the constructor context")

        stack_value = top["stack"]
        if type(stack_value) is not list:
            raise ValueError("snapshot stack must be a list")
        stack = [self._restore_frame(value, index) for index, value in enumerate(stack_value)]
        self._validate_stack_relationships(stack)

        state = self._restore_state(top["state"])
        active = self._restore_active(top["active"], state["last_token_id"], stack)
        summary = self._restore_summary(top["summary"], state, stack, active)
        if state["root_done"] and (stack or active is not None):
            raise ValueError("root_done snapshot must have an empty stack and no active token")
        if summary is not None and not state["root_done"]:
            raise ValueError("finished summary requires root_done")

        self._stack = stack
        self._root_done = state["root_done"]
        self._last_token_id = state["last_token_id"]
        self._shape_error = state["shape_error"]
        self._metadata_error = state["metadata_error"]
        self._operation_seen = state["operation_seen"]
        self._operation_matches = state["operation_matches"]
        self._target_group_seen = state["target_group_seen"]
        self._target_group_matches = state["target_group_matches"]
        self._target_group_valid = state["target_group_valid"]
        self._state = state["state"]
        self._resolved_context_seen = state["resolved_context_seen"]
        self._task_ids_array_seen = state["task_ids_array_seen"]
        self._task_count = state["task_count"]
        self._commit_plan_seen = state["commit_plan_seen"]
        self._sequences_array_seen = state["sequences_array_seen"]
        self._sequence_count = state["sequence_count"]
        self._meta_seen = state["meta_seen"]
        self._schema_version_seen = state["schema_version_seen"]
        self._source_schema_version = state["source_schema_version"]
        if active is None:
            self._active_id = None
            self._active_kind = None
            self._active_start = None
            self._active_end = None
            self._active_handler = None
            self._active_decoder = None
            self._active_sequence = None
            self._active_literal = None
            self._active_ordinal = 0
            self._active_data_valid = True
        else:
            (
                self._active_id,
                self._active_kind,
                self._active_start,
                self._active_end,
                self._active_handler,
                self._active_ordinal,
                self._active_data_valid,
                self._active_decoder,
                self._active_sequence,
                self._active_literal,
            ) = active
        self._summary = None
        self._invalid = False
        self._busy = False
        if summary is not None:
            # A saved summary cannot bypass the same final checks as live input.
            self.finish()

    def _restore_state(self, value: object) -> dict[str, object]:
        state = _checkpoint_dict(value, _CHECKPOINT_STATE_KEYS, "snapshot state")
        result: dict[str, object] = {}
        result["root_done"] = _checkpoint_bool(state["root_done"], "state.root_done")
        result["last_token_id"] = _checkpoint_int(state["last_token_id"], "state.last_token_id", minimum=-1)
        result["shape_error"] = _checkpoint_optional_str(state["shape_error"], "state.shape_error")
        result["metadata_error"] = _checkpoint_optional_str(state["metadata_error"], "state.metadata_error")
        for name in (
            "operation_seen",
            "operation_matches",
            "target_group_seen",
            "target_group_matches",
            "target_group_valid",
            "resolved_context_seen",
            "task_ids_array_seen",
            "commit_plan_seen",
            "sequences_array_seen",
        ):
            result[name] = _checkpoint_bool(state[name], f"state.{name}")
        captured_state = _checkpoint_optional_str(state["state"], "state.state")
        if captured_state is not None and captured_state not in _KNOWN_STATES:
            raise ValueError("state.state is not a recognized submission state")
        result["state"] = captured_state
        result["task_count"] = _checkpoint_int(state["task_count"], "state.task_count", minimum=0)
        result["sequence_count"] = _checkpoint_int(state["sequence_count"], "state.sequence_count", minimum=0)
        result["meta_seen"] = _checkpoint_bool(state["meta_seen"], "state.meta_seen")
        result["schema_version_seen"] = _checkpoint_bool(state["schema_version_seen"], "state.schema_version_seen")
        source_schema_version = state["source_schema_version"]
        if source_schema_version is not None:
            source_schema_version = _checkpoint_int(source_schema_version, "state.source_schema_version", minimum=0)
            if source_schema_version != SUPPORTED_SOURCE_SCHEMA_VERSION:
                raise ValueError("state.source_schema_version is unsupported")
        result["source_schema_version"] = source_schema_version
        if result["schema_version_seen"] and not result["meta_seen"]:
            raise ValueError("schema_version_seen requires meta_seen")
        if source_schema_version is not None and (not result["schema_version_seen"] or not result["meta_seen"]):
            raise ValueError("source_schema_version requires its metadata markers")
        return result

    def _restore_frame(self, value: object, index: int) -> _Frame:
        frame = _checkpoint_dict(value, _CHECKPOINT_FRAME_KEYS, f"snapshot stack frame {index}")
        role = _checkpoint_str(frame["role"], f"stack[{index}].role")
        kind = _checkpoint_str(frame["kind"], f"stack[{index}].kind")
        state = _checkpoint_str(frame["state"], f"stack[{index}].state")
        seen = _checkpoint_int(frame["seen"], f"stack[{index}].seen", minimum=0)
        count = _checkpoint_int(frame["count"], f"stack[{index}].count", minimum=0)
        pending_key = _checkpoint_optional_str(frame["pending_key"], f"stack[{index}].pending_key")
        if role not in _FRAME_ROLES:
            raise ValueError(f"stack[{index}].role is invalid")
        if kind not in {"object", "array"}:
            raise ValueError(f"stack[{index}].kind is invalid")
        allowed_states = (
            {"key_or_end", "key", "colon", "value", "child", "comma_or_end"}
            if kind == "object"
            else {"value_or_end", "value", "child", "comma_or_end"}
        )
        if state not in allowed_states:
            raise ValueError(f"stack[{index}].state is invalid for its kind")
        expected_kind = {
            "root": "object",
            "submission": "object",
            "resolved": "object",
            "plan": "object",
            "meta": "object",
            "task_ids": "array",
            "sequences": "array",
        }.get(role)
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"stack[{index}] has an invalid role/kind pair")
        if role == "root" and seen & ~sum(_ROOT_KEYS.values()):
            raise ValueError(f"stack[{index}].seen contains unknown root bits")
        if role == "submission" and seen & ~sum(_SUBMISSION_KEYS.values()):
            raise ValueError(f"stack[{index}].seen contains unknown submission bits")
        if role == "resolved" and seen & ~sum(_RESOLVED_KEYS.values()):
            raise ValueError(f"stack[{index}].seen contains unknown resolved bits")
        if role == "plan" and seen & ~sum(_PLAN_KEYS.values()):
            raise ValueError(f"stack[{index}].seen contains unknown plan bits")
        if role == "meta" and seen & ~sum(_META_KEYS.values()):
            raise ValueError(f"stack[{index}].seen contains unknown meta bits")
        if role in {"task_ids", "sequences", "ignore"} and seen != 0:
            raise ValueError(f"stack[{index}].seen must be zero for this role")
        if kind == "object" and count != 0:
            raise ValueError(f"stack[{index}].count must be zero for an object")
        if kind == "array" and pending_key is not None:
            raise ValueError(f"stack[{index}].pending_key must be null for an array")
        if pending_key is not None:
            keys = {
                "root": _ROOT_KEYS,
                "submission": _SUBMISSION_KEYS,
                "resolved": _RESOLVED_KEYS,
                "plan": _PLAN_KEYS,
                "meta": _META_KEYS,
            }.get(role)
            if keys is None or pending_key not in keys or state not in {"colon", "value"}:
                raise ValueError(f"stack[{index}].pending_key is inconsistent with its frame")
            if not seen & keys[pending_key]:
                raise ValueError(f"stack[{index}].pending_key has not been recorded in seen")
        elif state in {"colon", "value"} and role in {"root", "submission", "resolved", "plan"}:
            # Unknown object keys legitimately have no pending recognized key.
            pass
        return _Frame(role, kind, state, seen, pending_key, count)  # type: ignore[arg-type]

    def _validate_stack_relationships(self, stack: list[_Frame]) -> None:
        if stack and stack[0].role != "root":
            raise ValueError("snapshot stack must begin with the root frame")
        if any(frame.role == "root" for frame in stack[1:]):
            raise ValueError("snapshot stack cannot contain a second root frame")
        for index in range(1, len(stack)):
            if stack[index - 1].state != "child":
                raise ValueError("snapshot child frame does not occupy a parent child state")

    def _restore_active(
        self, value: object, last_token_id: int, stack: list[_Frame]
    ) -> (
        tuple[
            int,
            str,
            int,
            int,
            _TokenHandler,
            int,
            bool,
            _StringDecoder | None,
            _SequenceCapture | None,
            _LiteralCapture | None,
        ]
        | None
    ):
        if value is None:
            return None
        active = _checkpoint_dict(value, _CHECKPOINT_ACTIVE_KEYS, "snapshot active token")
        active_id = _checkpoint_int(active["id"], "active.id", minimum=0)
        if active_id != last_token_id + 1:
            raise ValueError("active.id is inconsistent with state.last_token_id")
        kind = _checkpoint_str(active["kind"], "active.kind")
        if kind not in {"string", "number", "literal"}:
            raise ValueError("active.kind is invalid")
        start = _checkpoint_int(active["start"], "active.start", minimum=0)
        end = _checkpoint_int(active["end"], "active.end", minimum=0)
        if end <= start:
            raise ValueError("active source offsets are invalid")
        handler_value = _checkpoint_str(active["handler"], "active.handler")
        if handler_value not in _ACTIVE_HANDLERS:
            raise ValueError("active.handler is invalid")
        handler = handler_value  # type: ignore[assignment]
        ordinal = _checkpoint_int(active["ordinal"], "active.ordinal", minimum=0)
        data_valid = _checkpoint_bool(active["data_valid"], "active.data_valid")
        decoder = self._restore_decoder(active["decoder"], handler, kind)
        sequence = self._restore_sequence(active["sequence"], handler, kind, ordinal)
        literal = self._restore_literal(active["literal"], handler, kind)
        if handler == "schema_version":
            if literal is None or literal.position != end - start:
                raise ValueError("schema_version literal position does not match its source span")
            if literal.valid and literal.position > 1:
                raise ValueError("valid schema_version literal must contain one byte")
        captures = sum(capture is not None for capture in (decoder, sequence, literal))
        if handler == "ignored":
            if captures:
                raise ValueError("ignored active token cannot contain a capture")
        elif captures != 1:
            raise ValueError("active selected token must contain exactly one capture")
        if handler == "key" and (not stack or stack[-1].kind != "object"):
            raise ValueError("active key token has no relevant object frame")
        if handler != "key" and (not stack or stack[-1].state != "child"):
            raise ValueError("active value token does not occupy a child frame state")
        return (
            active_id,
            kind,
            start,
            end,
            handler,
            ordinal,
            data_valid,
            decoder,
            sequence,
            literal,
        )

    def _restore_decoder(self, value: object, handler: str, kind: str) -> _StringDecoder | None:
        if value is None:
            if handler in {"key", "operation_id", "target_group", "state", "task_id"}:
                raise ValueError("selected string token requires a decoder")
            return None
        if kind != "string" or handler not in {"key", "operation_id", "target_group", "state", "task_id"}:
            raise ValueError("decoder is only valid for selected string tokens")
        decoder_snapshot = _checkpoint_dict(value, _CHECKPOINT_DECODER_KEYS, "snapshot decoder")
        purpose = _checkpoint_str(decoder_snapshot["purpose"], "decoder.purpose")
        if purpose not in _STRING_PURPOSES or purpose != handler:
            raise ValueError("decoder.purpose does not match active handler")
        state = _checkpoint_str(decoder_snapshot["state"], "decoder.state")
        if state not in {"normal", "escape", "unicode"}:
            raise ValueError("decoder.state must be a non-opening active state")
        unicode_value = _checkpoint_int(decoder_snapshot["unicode_value"], "decoder.unicode_value", minimum=0)
        unicode_digits = _checkpoint_int(decoder_snapshot["unicode_digits"], "decoder.unicode_digits", minimum=0)
        if state == "unicode":
            if unicode_digits > 3 or unicode_value >= 16**unicode_digits:
                raise ValueError("decoder unicode escape progress is invalid")
        elif unicode_value != 0 or unicode_digits != 0:
            raise ValueError("decoder unicode fields are only valid in unicode state")
        closed = _checkpoint_bool(decoder_snapshot["closed"], "decoder.closed")
        if closed:
            raise ValueError("active decoder cannot be closed")
        non_ascii = _checkpoint_bool(decoder_snapshot["non_ascii"], "decoder.non_ascii")
        too_long = _checkpoint_bool(decoder_snapshot["too_long"], "decoder.too_long")
        prefix = _checkpoint_ascii(decoder_snapshot["prefix"], "decoder.prefix")
        identifier_prefix = _checkpoint_ascii(decoder_snapshot["identifier_prefix"], "decoder.identifier_prefix")
        if purpose == "key" and len(prefix) > self._max_key_length:
            raise ValueError("decoder key prefix exceeds its bound")
        if purpose == "state" and len(prefix) > _MAX_STATE_LENGTH:
            raise ValueError("decoder state prefix exceeds its bound")
        if purpose not in {"key", "state"} and prefix:
            raise ValueError("decoder prefix is only valid for key and state purposes")
        if purpose == "target_group" and len(identifier_prefix) > _GROUP_MAX_LENGTH:
            raise ValueError("decoder group prefix exceeds its bound")
        if purpose != "target_group" and identifier_prefix:
            raise ValueError("decoder identifier_prefix is only valid for target_group")
        length = _checkpoint_int(decoder_snapshot["length"], "decoder.length", minimum=0)
        matches = _checkpoint_bool(decoder_snapshot["matches"], "decoder.matches")
        valid_identifier = _checkpoint_bool(decoder_snapshot["valid_identifier"], "decoder.valid_identifier")
        decoded_size = _checkpoint_int(decoder_snapshot["decoded_size"], "decoder.decoded_size", minimum=0)
        if decoded_size > length:
            raise ValueError("decoder decoded_size exceeds decoder length")
        digest_value = decoder_snapshot["digest"]
        if purpose == "task_id":
            digest = _checkpoint_digest(digest_value, "decoder.digest")
            if digest.size != decoded_size:
                raise ValueError("decoder digest size does not match decoded_size")
        else:
            if digest_value is not None or decoded_size != 0:
                raise ValueError("non-task decoder cannot contain a digest or decoded bytes")
            digest = None
        decoder = _StringDecoder(purpose, self)
        decoder.state = state
        decoder.unicode_value = unicode_value
        decoder.unicode_digits = unicode_digits
        decoder.closed = closed
        decoder.non_ascii = non_ascii
        decoder.too_long = too_long
        decoder.prefix = bytearray(prefix.encode("ascii"))
        decoder.identifier_prefix = bytearray(identifier_prefix.encode("ascii"))
        decoder.length = length
        decoder.matches = matches
        decoder.valid_identifier = valid_identifier
        decoder.decoded_size = decoded_size
        decoder.digest = digest
        decoder.fragment_output.clear()
        decoder.fragment_invalid = False
        return decoder

    def _restore_sequence(self, value: object, handler: str, kind: str, ordinal: int) -> _SequenceCapture | None:
        if value is None:
            if handler == "sequence":
                raise ValueError("active sequence token requires a sequence capture")
            return None
        if handler != "sequence" or kind != "number":
            raise ValueError("sequence capture is only valid for an active number sequence")
        sequence_snapshot = _checkpoint_dict(value, _CHECKPOINT_SEQUENCE_KEYS, "snapshot sequence capture")
        sequence_ordinal = _checkpoint_int(sequence_snapshot["ordinal"], "sequence.ordinal", minimum=0)
        if sequence_ordinal != ordinal:
            raise ValueError("sequence.ordinal does not match active.ordinal")
        valid = _checkpoint_bool(sequence_snapshot["valid"], "sequence.valid")
        seen = _checkpoint_int(sequence_snapshot["seen"], "sequence.seen", minimum=0)
        decoded_size = _checkpoint_int(sequence_snapshot["decoded_size"], "sequence.decoded_size", minimum=0)
        digest = _checkpoint_digest(sequence_snapshot["digest"], "sequence.digest")
        if digest.size != decoded_size:
            raise ValueError("sequence digest size does not match decoded_size")
        if valid and seen != decoded_size:
            raise ValueError("valid sequence capture has inconsistent byte counts")
        sequence = _SequenceCapture(self, ordinal)
        sequence.valid = valid
        sequence.seen = seen
        sequence.decoded_size = decoded_size
        sequence.digest = digest
        return sequence

    def _restore_literal(self, value: object, handler: str, kind: str) -> _LiteralCapture | None:
        if value is None:
            if handler in {"null_target_group", "null_commit_plan", "schema_version"}:
                raise ValueError("active null metadata token requires a literal capture")
            return None
        if handler in {"null_target_group", "null_commit_plan"}:
            expected = b"null"
            maximum = 4
            if kind != "literal":
                raise ValueError("literal capture is only valid for null metadata")
        elif handler == "schema_version":
            expected = b"6"
            maximum = None
            if kind != "number":
                raise ValueError("literal capture is only valid for schema_version numbers")
        else:
            raise ValueError("literal capture is only valid for selected metadata literals")
        literal_snapshot = _checkpoint_dict(value, _CHECKPOINT_LITERAL_KEYS, "snapshot literal capture")
        position = _checkpoint_int(literal_snapshot["position"], "literal.position", minimum=1)
        if maximum is not None and position > maximum:
            raise ValueError("literal.position is out of range")
        valid = _checkpoint_bool(literal_snapshot["valid"], "literal.valid")
        literal = _LiteralCapture(expected)
        literal.position = position
        literal.valid = valid
        return literal

    def _restore_summary(
        self, value: object, state: dict[str, object], stack: list[_Frame], active: object
    ) -> ProjectionSummary | None:
        if value is None:
            return None
        summary_snapshot = _checkpoint_dict(value, _CHECKPOINT_SUMMARY_KEYS, "snapshot summary")
        summary_state = _checkpoint_str(summary_snapshot["state"], "summary.state")
        if summary_state not in _KNOWN_STATES:
            raise ValueError("summary.state is not a recognized submission state")
        matches_group = _checkpoint_bool(summary_snapshot["matches_group"], "summary.matches_group")
        task_count = _checkpoint_int(summary_snapshot["task_count"], "summary.task_count", minimum=0)
        sequence_count = _checkpoint_int(summary_snapshot["sequence_count"], "summary.sequence_count", minimum=0)
        if stack or active is not None or not state["root_done"]:
            raise ValueError("summary requires a complete inactive root")
        if (
            not state["operation_seen"]
            or not state["operation_matches"]
            or not state["target_group_seen"]
            or not state["target_group_valid"]
            or state["metadata_error"] is not None
        ):
            raise ValueError("summary is inconsistent with required metadata")
        if state["state"] == "committed" and state["target_group_matches"] and state["shape_error"] is not None:
            raise ValueError("summary is inconsistent with selected field shape")
        if summary_state != state["state"]:
            raise ValueError("summary.state does not match parser state")
        if matches_group != state["target_group_matches"]:
            raise ValueError("summary.matches_group does not match parser state")
        if task_count != state["task_count"] or sequence_count != state["sequence_count"]:
            raise ValueError("summary counts do not match parser state")
        return ProjectionSummary(summary_state, matches_group, task_count, sequence_count)

    def _validate_span(self, span: Span, raw: bytes) -> None:
        if not isinstance(span, Span):
            self._fail("feed requires a Span instance")
        if not isinstance(raw, bytes):
            self._fail("feed requires raw source bytes")
        if not isinstance(span.is_final, bool):
            self._fail("Span.is_final must be boolean")
        if span.end <= span.start or span.end - span.start > _MAX_FRAGMENT:
            self._fail("span must be a nonempty range no larger than 65536 bytes")
        if len(raw) != span.end - span.start:
            self._fail("raw bytes must equal the Span source range length")
        if span.start < 0 or span.token_id < 0:
            self._fail("span offsets and token_id must be nonnegative")

    def _begin_token(self, span: Span) -> None:
        if self._root_done:
            self._fail("multiple root values are not allowed")
        if span.token_id != self._last_token_id + 1:
            self._fail("lexer token IDs must be contiguous")
        self._active_id = span.token_id
        self._active_kind = span.kind
        self._active_start = span.start
        self._active_end = span.end
        self._active_handler = None
        self._active_decoder = None
        self._active_sequence = None
        self._active_literal = None
        self._active_data_valid = True
        if not self._stack:
            if span.kind != "{" or not span.is_final:
                self._fail("root value must be one complete object")
            self._stack.append(_Frame("root", "object", "key_or_end"))
            return
        frame = self._stack[-1]
        if frame.kind == "object":
            self._begin_object_token(frame, span)
        else:
            self._begin_array_token(frame, span)

    def _continue_token(self, span: Span) -> None:
        if span.token_id != self._active_id or span.kind != self._active_kind:
            self._fail("lexical token fragments changed token identity")
        if self._active_end != span.start:
            self._fail("lexical token fragments are not contiguous")
        self._active_end = span.end

    def _begin_object_token(self, frame: _Frame, span: Span) -> None:
        state = frame.state
        if state in {"key_or_end", "key"}:
            if span.kind == "}" and state == "key_or_end":
                self._finish_container_punctuation(frame, span.kind)
                return
            if span.kind != "string":
                self._fail("object keys must be strings and trailing commas are not allowed")
            self._active_handler = "key"
            self._active_decoder = _StringDecoder("key", self)
            return
        if state == "colon":
            if span.kind != ":":
                self._fail("object key must be followed by a colon")
            frame.state = "value"
            return
        if state == "value":
            self._begin_value(frame, span)
            return
        if state == "comma_or_end":
            if span.kind == ",":
                frame.state = "key"
                return
            if span.kind == "}":
                self._finish_container_punctuation(frame, span.kind)
                return
            self._fail("object value must be followed by a comma or closing brace")
        if state == "child":
            self._fail("object received a token while its child value was open")
        self._fail(f"unknown object parser state {state!r}")

    def _begin_array_token(self, frame: _Frame, span: Span) -> None:
        state = frame.state
        if state in {"value_or_end", "value"}:
            if span.kind == "]" and state == "value_or_end":
                self._finish_container_punctuation(frame, span.kind)
                return
            if span.kind in {",", "}", "]", ":"}:
                self._fail("array requires a value and does not allow trailing commas")
            self._begin_value(frame, span)
            return
        if state == "comma_or_end":
            if span.kind == ",":
                frame.state = "value"
                return
            if span.kind == "]":
                self._finish_container_punctuation(frame, span.kind)
                return
            self._fail("array value must be followed by a comma or closing bracket")
        if state == "child":
            self._fail("array received a token while its child value was open")
        self._fail(f"unknown array parser state {state!r}")

    def _begin_value(self, frame: _Frame, span: Span) -> None:
        ordinal = frame.count
        if frame.kind == "array":
            frame.count += 1
            key = None
        else:
            key = frame.pending_key
            frame.pending_key = None
        frame.state = "child"
        expected = self._expected_value(frame.role, key)
        if expected == "object:meta":
            self._meta_seen = True
            if span.kind == "{":
                self._stack.append(_Frame("meta", "object", "key_or_end"))
            else:
                self._record_metadata_error("meta must be an object")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "schema_version":
            self._schema_version_seen = True
            if span.kind == "number":
                self._active_handler = "schema_version"
                self._active_literal = _LiteralCapture(b"6")
            else:
                self._record_metadata_error("unsupported source schema version")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "object:submission":
            if span.kind == "{":
                self._stack.append(_Frame("submission", "object", "key_or_end"))
            else:
                self._record_shape("submission must be an object")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "object:resolved":
            self._resolved_context_seen = True
            if span.kind == "{":
                self._stack.append(_Frame("resolved", "object", "key_or_end"))
            else:
                self._record_shape("resolved_context must be an object")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "object:plan":
            self._commit_plan_seen = True
            if span.kind == "{":
                self._stack.append(_Frame("plan", "object", "key_or_end"))
            elif span.kind == "literal":
                self._active_handler = "null_commit_plan"
                self._active_literal = _LiteralCapture(b"null")
            else:
                self._record_shape("commit_plan must be an object or null")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "array:task_ids":
            self._task_ids_array_seen = True
            if span.kind == "[":
                self._stack.append(_Frame("task_ids", "array", "value_or_end"))
            else:
                self._record_shape("resolved_context.task_ids must be an array")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "array:sequences":
            self._sequences_array_seen = True
            if span.kind == "[":
                self._stack.append(_Frame("sequences", "array", "value_or_end"))
            else:
                self._record_shape("commit_plan.group_membership_sequences must be an array")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "string:operation_id":
            self._operation_seen = True
            if span.kind == "string":
                self._active_handler = "operation_id"
                self._active_decoder = _StringDecoder("operation_id", self)
            else:
                self._record_shape("operation_id must be a string")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "string:target_group":
            self._target_group_seen = True
            if span.kind == "string":
                self._active_handler = "target_group"
                self._active_decoder = _StringDecoder("target_group", self)
            elif span.kind == "literal":
                self._active_handler = "null_target_group"
                self._active_literal = _LiteralCapture(b"null")
            else:
                self._record_metadata_error("target_group must be a string or null")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "string:state":
            if span.kind == "string":
                self._active_handler = "state"
                self._active_decoder = _StringDecoder("state", self)
            else:
                self._record_shape("state must be a string")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "task_id":
            if span.kind == "string":
                self._active_handler = "task_id"
                self._active_ordinal = ordinal
                self._active_decoder = _StringDecoder("task_id", self)
            else:
                self._record_shape("task_ids entries must be strings")
                self._start_wrong_container_or_scalar(span)
            return
        if expected == "sequence":
            if span.kind == "number":
                self._active_handler = "sequence"
                self._active_ordinal = ordinal
                self._active_sequence = _SequenceCapture(self, ordinal)
            else:
                self._record_shape("group membership sequences must be positive decimal integers")
                self._start_wrong_container_or_scalar(span)
            return
        if span.kind == "{":
            self._stack.append(_Frame("ignore", "object", "key_or_end"))
        elif span.kind == "[":
            self._stack.append(_Frame("ignore", "array", "value_or_end"))
        self._active_handler = "ignored"

    def _expected_value(self, role: _FrameRole, key: str | None) -> str | None:
        if role == "root":
            return {"submission": "object:submission", "meta": "object:meta"}.get(key)
        if role == "submission":
            return {
                "operation_id": "string:operation_id",
                "target_group": "string:target_group",
                "state": "string:state",
                "resolved_context": "object:resolved",
                "commit_plan": "object:plan",
            }.get(key)
        if role == "resolved":
            return "array:task_ids" if key == "task_ids" else None
        if role == "plan":
            return "array:sequences" if key == "group_membership_sequences" else None
        if role == "meta":
            return "schema_version" if key == "schema_version" else None
        if role == "task_ids":
            return "task_id"
        if role == "sequences":
            return "sequence"
        return None

    def _start_wrong_container_or_scalar(self, span: Span) -> None:
        if span.kind == "{":
            self._stack.append(_Frame("ignore", "object", "key_or_end"))
        elif span.kind == "[":
            self._stack.append(_Frame("ignore", "array", "value_or_end"))
        else:
            self._active_handler = "ignored"

    def _consume_fragment(self, raw: bytes, is_final: bool) -> None:
        kind = self._active_kind
        if kind in {"{", "}", "[", "]", ":", ","}:
            if len(raw) != 1 or raw != kind.encode() or not is_final:
                self._fail("punctuation span must contain exactly one final punctuation byte")
            return
        decoder = self._active_decoder
        if decoder is not None:
            decoded = decoder.feed(raw)
            if self._active_handler == "task_id":
                if decoded and not decoder.fragment_invalid and self._active_data_valid:
                    self._emit_chunk_data("task_id", self._active_ordinal, decoded, is_final)
                elif decoder.fragment_invalid:
                    self._active_data_valid = False
            return
        if self._active_sequence is not None:
            self._active_sequence.feed(raw, is_final)
            if not self._active_sequence.valid:
                self._active_data_valid = False
            return
        if self._active_literal is not None:
            self._active_literal.feed(raw)

    def _finish_token(self) -> None:
        kind = self._active_kind
        if kind in {"{", "}", "[", "]", ":", ","}:
            self._clear_active_token()
            return
        handler = self._active_handler
        decoder = self._active_decoder
        if decoder is not None:
            decoder.finish()
            if handler == "key":
                self._finish_key(decoder)
            elif handler == "operation_id":
                self._operation_matches = (
                    not decoder.non_ascii and decoder.length == len(self._expected_operation_id) and decoder.matches
                )
            elif handler == "target_group":
                valid = (
                    decoder.valid_identifier and not decoder.non_ascii and not decoder.too_long and decoder.length > 0
                )
                if valid:
                    try:
                        validate_group_name(bytes(decoder.identifier_prefix).decode("ascii"))
                    except ValueError:
                        valid = False
                self._target_group_matches = valid and decoder.matches and decoder.length == len(self._expected_group)
                if not valid:
                    self._target_group_valid = False
                    self._record_metadata_error("target_group is not a valid ASCII group identifier")
            elif handler == "state":
                state = bytes(decoder.prefix).decode("ascii", errors="ignore") if not decoder.too_long else ""
                if decoder.non_ascii or decoder.too_long or state not in _KNOWN_STATES:
                    self._record_metadata_error("state is not a recognized submission state")
                else:
                    self._state = state
            elif handler == "task_id":
                if decoder.fragment_invalid:
                    self._active_data_valid = False
                if self._active_data_valid and not decoder.fragment_output:
                    self._emit_chunk_data("task_id", self._active_ordinal, b"", True)
                if not self._active_data_valid or decoder.decoded_size == 0:
                    self._record_shape("task_ids entries must be nonempty ASCII identifiers")
                elif decoder.digest is not None:
                    self._emit_end("task_id", self._active_ordinal, decoder.digest, decoder.decoded_size)
        elif handler == "schema_version":
            if self._active_literal is None or not self._active_literal.finish():
                self._record_metadata_error("unsupported source schema version")
            else:
                self._source_schema_version = SUPPORTED_SOURCE_SCHEMA_VERSION
        elif handler == "sequence":
            sequence = self._active_sequence
            if sequence is None or not sequence.valid or sequence.seen == 0:
                self._record_shape("group membership sequences must be positive decimal integers")
            elif sequence.decoded_size == 0:
                self._record_shape("group membership sequences must be positive decimal integers")
            else:
                self._emit_end("sequence", self._active_ordinal, sequence.digest, sequence.decoded_size)
        elif handler == "null_target_group":
            if self._active_literal is None or not self._active_literal.finish():
                self._record_metadata_error("target_group must be a string or null")
            else:
                self._target_group_matches = False
        elif handler == "null_commit_plan":
            if self._active_literal is None or not self._active_literal.finish():
                self._record_shape("commit_plan must be an object or null")
        if handler != "key":
            self._complete_scalar_value()
        self._clear_active_token()

    def _finish_key(self, decoder: _StringDecoder) -> None:
        frame = self._stack[-1]
        raw_key = bytes(decoder.prefix).decode("ascii", errors="ignore") if not decoder.non_ascii else ""
        key = self._classify_key(frame.role, raw_key, decoder)
        if key is not None:
            bit = self._key_bit(frame.role, key)
            if frame.seen & bit:
                self._fail(f"ambiguous duplicate recognized key {key!r} at {frame.role}")
            frame.seen |= bit
        frame.pending_key = key
        frame.state = "colon"

    def _classify_key(self, role: _FrameRole, key: str, decoder: _StringDecoder) -> str | None:
        if decoder.non_ascii or decoder.too_long:
            return None
        keys = {
            "root": _ROOT_KEYS,
            "submission": _SUBMISSION_KEYS,
            "resolved": _RESOLVED_KEYS,
            "plan": _PLAN_KEYS,
            "meta": _META_KEYS,
        }.get(role)
        if keys is None:
            return None
        return key if key in keys else None

    def _key_bit(self, role: _FrameRole, key: str) -> int:
        return {
            "root": _ROOT_KEYS,
            "submission": _SUBMISSION_KEYS,
            "resolved": _RESOLVED_KEYS,
            "plan": _PLAN_KEYS,
            "meta": _META_KEYS,
        }.get(role, {}).get(key, 0)

    def _finish_container_punctuation(self, frame: _Frame, closer: str) -> None:
        expected = "}" if frame.kind == "object" else "]"
        if closer != expected:
            self._fail(f"mismatched container closer {closer!r}; expected {expected!r}")
        if frame.kind == "object" and frame.state not in {"key_or_end", "comma_or_end"}:
            self._fail("object ended before its key/value grammar was complete")
        if frame.kind == "array" and frame.state not in {"value_or_end", "comma_or_end"}:
            self._fail("array ended before its value grammar was complete")
        self._stack.pop()
        if frame.role == "task_ids":
            self._task_count = frame.count
        elif frame.role == "sequences":
            self._sequence_count = frame.count
        if not self._stack:
            self._root_done = True
            return
        parent = self._stack[-1]
        if parent.state != "child":
            self._fail("container close did not complete a parent value")
        parent.state = "comma_or_end"

    def _complete_scalar_value(self) -> None:
        if not self._stack:
            self._fail("scalar value appeared outside the root object")
        frame = self._stack[-1]
        if frame.state != "child":
            self._fail("scalar token did not occupy an expected value slot")
        frame.state = "comma_or_end"

    def _emit_chunk_data(self, kind: str, ordinal: int, data: bytes, is_final: bool) -> None:
        if len(data) > _MAX_FRAGMENT:
            self._fail("projected field chunk exceeded 65536 bytes")
        if not data and not is_final:
            return
        if kind not in {"task_id", "sequence"}:
            self._fail("unknown projected field kind")
        self._emit_chunk_fn(FieldChunk(kind, ordinal, bytes(data), is_final))

    def _emit_end(self, kind: str, ordinal: int, digest: ChainedDigest, decoded_size: int) -> None:
        start = self._active_start
        end = self._active_end
        if start is None or end is None:
            self._fail("selected field has no source span")
        self._emit_end_fn(FieldEnd(kind, ordinal, digest.hexdigest(), decoded_size, start, end))

    def _record_shape(self, message: str) -> None:
        if self._shape_error is None:
            self._shape_error = message

    def _record_metadata_error(self, message: str) -> None:
        if self._metadata_error is None:
            self._metadata_error = message

    def _clear_active_token(self) -> None:
        self._last_token_id = self._active_id if self._active_id is not None else self._last_token_id
        self._active_id = None
        self._active_kind = None
        self._active_start = None
        self._active_end = None
        self._active_handler = None
        self._active_decoder = None
        self._active_sequence = None
        self._active_literal = None

    def _invalidate(self) -> None:
        self._invalid = True
        self._stack.clear()
        self._active_decoder = None
        self._active_sequence = None
        self._active_literal = None

    def _fail(self, message: str) -> None:
        self._invalid = True
        raise ValueError(message)


__all__ = [
    "FieldChunk",
    "FieldEnd",
    "ProjectionSummary",
    "SUPPORTED_SOURCE_SCHEMA_VERSION",
    "SubmissionProjection",
]
