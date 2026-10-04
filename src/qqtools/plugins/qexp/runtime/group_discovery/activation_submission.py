"""Bounded, restartable projection of Submission metadata for activation bootstrap."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..records import validate_group_name, validate_identifier
from ..store import json_encoded_size
from .json_stream import Scanner, Span

if TYPE_CHECKING:
    from ..upgrade.contracts import RegularFileRead, RegularFileRevision, UpgradeStorage

STREAM_CHUNK_BYTES = 16 * 1024
MAX_DIRECT_BYTES = 64 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024
MAX_NESTING = 64
MAX_KEY_BYTES = 256
MAX_OPERATION_ID_BYTES = 256
MAX_STATE_BYTES = 32
MAX_GROUP_BYTES = 64

_VALID_STATES = frozenset({"prepared", "preparing", "committing", "committed", "aborted", "blocked"})
_LOCATOR_STATES = frozenset({"prepared", "committing", "blocked"})
_REVISION_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_CHECKPOINT_KEYS = frozenset(
    {
        "version",
        "namespace",
        "operation_id",
        "source_path",
        "source_revision",
        "scanner",
        "projector",
        "checksum",
    }
)
_PROJECTOR_KEYS = frozenset({"version", "root_started", "root_done", "stack", "required", "active"})
_FRAME_KEYS = frozenset({"role", "kind", "state", "pending_key", "seen"})
_ACTIVE_KEYS = frozenset({"kind", "handler", "collector", "literal"})
_COLLECTOR_KEYS = frozenset({"state", "unicode_value", "unicode_digits", "closed", "decoded_b64"})
_FRAME_ROLES = frozenset({"root", "submission", "ignored"})
_FRAME_KINDS = frozenset({"object", "array"})
_FRAME_STATES = frozenset({"key_or_end", "key", "colon", "value", "comma_or_end", "value_or_end", "array_value"})
_HANDLERS = frozenset({"key", "ignored", "operation_id", "state", "target_group", "target_group_null"})
_TOKEN_KINDS = frozenset({"string", "number", "literal"})
_PUNCTUATION = frozenset({"{", "}", "[", "]", ":", ","})
_JSON_LITERALS = frozenset({b"true", b"false", b"null"})
_SIMPLE_ESCAPES = {
    ord('"'): b'"',
    ord("\\"): b"\\",
    ord("/"): b"/",
    ord("b"): b"\b",
    ord("f"): b"\f",
    ord("n"): b"\n",
    ord("r"): b"\r",
    ord("t"): b"\t",
}


@dataclass(frozen=True, slots=True)
class ActivationSubmissionStep:
    """The bounded outcome of one activation-source projection step."""

    state: Literal["progressed", "complete"]
    submission_state: str | None = None
    target_group: str | None = None
    completed_bytes: int | None = None
    total_bytes: int | None = None
    source_revision: dict[str, int] | None = None
    restarted: bool = False
    previous_completed_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.state not in {"progressed", "complete"}:
            raise ValueError("activation Submission step has an invalid state")
        if self.state == "progressed" and (self.submission_state is not None or self.target_group is not None):
            raise ValueError("an incomplete activation Submission step cannot contain completed metadata")
        if self.state == "complete" and self.submission_state not in _VALID_STATES:
            raise ValueError("a complete activation Submission step has an invalid state value")
        if type(self.restarted) is not bool:
            raise TypeError("activation Submission restart marker must be a boolean")
        if self.state == "complete":
            if self.completed_bytes is not None or self.total_bytes is not None or self.source_revision is not None:
                raise ValueError("a complete activation Submission step cannot contain byte progress")
            if self.restarted:
                raise ValueError("a complete activation Submission step cannot be marked restarted")
            return
        if type(self.completed_bytes) is not int or self.completed_bytes < 0:
            raise ValueError("activation Submission completed_bytes must be nonnegative")
        if type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("activation Submission total_bytes must be nonnegative")
        if self.completed_bytes > self.total_bytes:
            raise ValueError("activation Submission completed_bytes exceeds total_bytes")
        if type(self.source_revision) is not dict or frozenset(self.source_revision) != _REVISION_KEYS:
            raise ValueError("activation Submission source revision is invalid")
        if any(type(value) is not int or value < 0 for value in self.source_revision.values()):
            raise ValueError("activation Submission source revision values are invalid")
        if self.source_revision["size"] != self.total_bytes:
            raise ValueError("activation Submission total_bytes does not match source revision")


@dataclass(slots=True)
class _Frame:
    role: str
    kind: str
    state: str
    pending_key: str | None = None
    seen: set[str] = field(default_factory=set)


class _ChunkReader:
    """A short-lived source window positioned at an arbitrary source offset."""

    __slots__ = ("_data", "_position", "_start")

    def __init__(self, start: int, data: bytes) -> None:
        self._data = data
        self._start = start
        self._position = 0

    def tell(self) -> int:
        return self._start + self._position

    def read(self, size: int = -1) -> bytes:
        if type(size) is not int:
            raise TypeError("source read size must be an integer")
        if size < 0:
            size = len(self._data) - self._position
        end = min(len(self._data), self._position + size)
        result = self._data[self._position : end]
        self._position = end
        return result


class _StringCollector:
    """Decode one bounded JSON string without retaining an ignored value."""

    __slots__ = ("max_bytes", "state", "unicode_value", "unicode_digits", "closed", "decoded")

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.state = "opening"
        self.unicode_value = 0
        self.unicode_digits = 0
        self.closed = False
        self.decoded = bytearray()

    def feed(self, raw: bytes) -> None:
        for byte in raw:
            if self.closed:
                raise ValueError("JSON string contains bytes after its closing quote")
            if self.state == "opening":
                if byte != ord('"'):
                    raise ValueError("JSON string token did not start with a quote")
                self.state = "normal"
            elif self.state == "escape":
                escaped = _SIMPLE_ESCAPES.get(byte)
                if escaped is not None:
                    self._append(escaped)
                    self.state = "normal"
                elif byte == ord("u"):
                    self.state = "unicode"
                    self.unicode_value = 0
                    self.unicode_digits = 0
                else:
                    raise ValueError("JSON string contains an invalid escape")
            elif self.state == "unicode":
                if byte not in b"0123456789abcdefABCDEF":
                    raise ValueError("JSON string contains an invalid unicode escape")
                self.unicode_value = self.unicode_value * 16 + int(chr(byte), 16)
                self.unicode_digits += 1
                if self.unicode_digits == 4:
                    try:
                        value = chr(self.unicode_value).encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise ValueError("JSON string contains an invalid unicode scalar") from exc
                    self._append(value)
                    self.state = "normal"
                    self.unicode_value = 0
                    self.unicode_digits = 0
            else:
                if byte == ord('"'):
                    self.closed = True
                    self.state = "closed"
                elif byte == ord("\\"):
                    self.state = "escape"
                elif byte < 0x20:
                    raise ValueError("JSON string contains an unescaped control byte")
                else:
                    self._append(bytes((byte,)))

    def _append(self, value: bytes) -> None:
        if len(self.decoded) + len(value) > self.max_bytes:
            raise ValueError("selected Submission string exceeds its bounded scalar limit")
        self.decoded.extend(value)

    def finish(self) -> str:
        if not self.closed or self.state != "closed":
            raise ValueError("selected JSON string is incomplete")
        try:
            return bytes(self.decoded).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("selected JSON string is not valid UTF-8") from exc

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "unicode_value": self.unicode_value,
            "unicode_digits": self.unicode_digits,
            "closed": self.closed,
            "decoded_b64": base64.b64encode(bytes(self.decoded)).decode("ascii"),
        }

    @classmethod
    def from_snapshot(cls, value: object, max_bytes: int) -> "_StringCollector":
        if type(value) is not dict or frozenset(value) != _COLLECTOR_KEYS:
            raise ValueError("activation Submission collector checkpoint has an invalid shape")
        state = value["state"]
        if state not in {"opening", "normal", "escape", "unicode", "closed"}:
            raise ValueError("activation Submission collector state is invalid")
        unicode_value = _exact_int(value["unicode_value"], "collector unicode_value", minimum=0)
        unicode_digits = _exact_int(value["unicode_digits"], "collector unicode_digits", minimum=0, maximum=3)
        closed = _exact_bool(value["closed"], "collector closed")
        encoded = _exact_str(value["decoded_b64"], "collector decoded_b64")
        try:
            decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("activation Submission collector bytes are not canonical base64") from exc
        if base64.b64encode(decoded).decode("ascii") != encoded or len(decoded) > max_bytes:
            raise ValueError("activation Submission collector bytes exceed their bound")
        if state == "unicode" and unicode_digits > 3:
            raise ValueError("activation Submission collector unicode progress is invalid")
        if state != "unicode" and unicode_digits != 0:
            raise ValueError("activation Submission collector unicode progress is stale")
        if state == "closed" and not closed:
            raise ValueError("activation Submission collector closed state is inconsistent")
        collector = cls(max_bytes)
        collector.state = state
        collector.unicode_value = unicode_value
        collector.unicode_digits = unicode_digits
        collector.closed = closed
        collector.decoded = bytearray(decoded)
        return collector


class _ActivationSubmissionProjector:
    """Validate one JSON root and project only the selected Submission fields."""

    __slots__ = (
        "_operation_id",
        "_stack",
        "_root_started",
        "_root_done",
        "_operation_id_value",
        "_state_value",
        "_target_group_value",
        "_operation_seen",
        "_state_seen",
        "_target_group_seen",
        "_active_kind",
        "_active_handler",
        "_collector",
        "_literal",
    )

    def __init__(self, operation_id: str) -> None:
        validate_identifier(operation_id, "submission operation_id")
        self._operation_id = operation_id
        self._stack: list[_Frame] = []
        self._root_started = False
        self._root_done = False
        self._operation_id_value: str | None = None
        self._state_value: str | None = None
        self._target_group_value: str | None = None
        self._operation_seen = False
        self._state_seen = False
        self._target_group_seen = False
        self._active_kind: str | None = None
        self._active_handler: str | None = None
        self._collector: _StringCollector | None = None
        self._literal = bytearray()

    @property
    def active_handler(self) -> str | None:
        return self._active_handler

    def feed(self, span: Span, raw: bytes) -> None:
        if not isinstance(span, Span) or type(raw) is not bytes:
            raise TypeError("activation Submission projector received an invalid token span")
        if span.end <= span.start or len(raw) > MAX_DIRECT_BYTES:
            raise ValueError("activation Submission token span is invalid")
        if span.kind in _PUNCTUATION:
            if not span.is_final or raw != span.kind.encode("ascii"):
                raise ValueError("activation Submission punctuation span is invalid")
            self._feed_punctuation(span.kind)
            return
        if span.kind not in _TOKEN_KINDS:
            raise ValueError(f"activation Submission token kind is invalid: {span.kind!r}")
        if self._active_kind is None:
            self._start_token(span.kind)
        elif self._active_kind != span.kind:
            raise ValueError("activation Submission token fragments changed kind")
        if self._collector is not None:
            self._collector.feed(raw)
        elif self._active_kind == "literal":
            self._literal.extend(raw)
            if len(self._literal) > 9:
                raise ValueError("JSON literal exceeds its bounded scalar limit")
        if span.is_final:
            self._finish_token(span.kind)

    def snapshot(self) -> dict[str, object]:
        active: dict[str, object] | None
        if self._active_kind is None or self._active_handler is None:
            active = None
        else:
            active = {
                "kind": self._active_kind,
                "handler": self._active_handler,
                "collector": self._collector.snapshot() if self._collector is not None else None,
                "literal": base64.b64encode(bytes(self._literal)).decode("ascii"),
            }
        return {
            "version": 1,
            "root_started": self._root_started,
            "root_done": self._root_done,
            "stack": [
                {
                    "role": frame.role,
                    "kind": frame.kind,
                    "state": frame.state,
                    "pending_key": frame.pending_key,
                    "seen": sorted(frame.seen),
                }
                for frame in self._stack
            ],
            "required": {
                "operation_id": self._operation_id_value,
                "state": self._state_value,
                "target_group": self._target_group_value,
                "operation_seen": self._operation_seen,
                "state_seen": self._state_seen,
                "target_group_seen": self._target_group_seen,
            },
            "active": active,
        }

    @classmethod
    def from_snapshot(cls, operation_id: str, value: object) -> "_ActivationSubmissionProjector":
        if type(value) is not dict or frozenset(value) != _PROJECTOR_KEYS or value.get("version") != 1:
            raise ValueError("activation Submission projector checkpoint has an invalid shape")
        projector = cls(operation_id)
        projector._root_started = _exact_bool(value["root_started"], "projector root_started")
        projector._root_done = _exact_bool(value["root_done"], "projector root_done")
        raw_stack = value["stack"]
        if type(raw_stack) is not list or len(raw_stack) > MAX_NESTING:
            raise ValueError("activation Submission projector nesting is invalid")
        for raw_frame in raw_stack:
            if type(raw_frame) is not dict or frozenset(raw_frame) != _FRAME_KEYS:
                raise ValueError("activation Submission frame checkpoint has an invalid shape")
            role = _exact_str(raw_frame["role"], "projector frame role")
            kind = _exact_str(raw_frame["kind"], "projector frame kind")
            state = _exact_str(raw_frame["state"], "projector frame state")
            if role not in _FRAME_ROLES or kind not in _FRAME_KINDS or state not in _FRAME_STATES:
                raise ValueError("activation Submission frame checkpoint contains an invalid value")
            pending_key = raw_frame["pending_key"]
            if pending_key is not None:
                pending_key = _exact_str(pending_key, "projector pending key")
                if len(pending_key.encode("utf-8")) > MAX_KEY_BYTES:
                    raise ValueError("activation Submission pending key is too long")
            seen = raw_frame["seen"]
            if type(seen) is not list or any(type(item) is not str for item in seen) or len(set(seen)) != len(seen):
                raise ValueError("activation Submission frame seen keys are invalid")
            if any(len(item.encode("utf-8")) > MAX_KEY_BYTES for item in seen):
                raise ValueError("activation Submission frame key is too long")
            projector._stack.append(_Frame(role, kind, state, pending_key, set(seen)))
        required = value["required"]
        if type(required) is not dict or set(required) != {
            "operation_id",
            "state",
            "target_group",
            "operation_seen",
            "state_seen",
            "target_group_seen",
        }:
            raise ValueError("activation Submission required checkpoint has an invalid shape")
        projector._operation_id_value = _optional_str(required["operation_id"], "projector operation_id")
        projector._state_value = _optional_str(required["state"], "projector state")
        projector._target_group_value = _optional_str(required["target_group"], "projector target_group")
        projector._operation_seen = _exact_bool(required["operation_seen"], "projector operation_seen")
        projector._state_seen = _exact_bool(required["state_seen"], "projector state_seen")
        projector._target_group_seen = _exact_bool(required["target_group_seen"], "projector target_group_seen")
        active = value["active"]
        if active is not None:
            if type(active) is not dict or frozenset(active) != _ACTIVE_KEYS:
                raise ValueError("activation Submission active token checkpoint has an invalid shape")
            kind = _exact_str(active["kind"], "projector active kind")
            handler = _exact_str(active["handler"], "projector active handler")
            if kind not in _TOKEN_KINDS or handler not in _HANDLERS:
                raise ValueError("activation Submission active token checkpoint has an invalid value")
            projector._active_kind = kind
            projector._active_handler = handler
            collector = active["collector"]
            if collector is not None:
                if kind != "string" or handler not in {"key", "operation_id", "state", "target_group"}:
                    raise ValueError("activation Submission collector is attached to the wrong token")
                limit = {
                    "key": MAX_KEY_BYTES,
                    "operation_id": MAX_OPERATION_ID_BYTES,
                    "state": MAX_STATE_BYTES,
                    "target_group": MAX_GROUP_BYTES,
                }[handler]
                projector._collector = _StringCollector.from_snapshot(collector, limit)
            literal = _exact_str(active["literal"], "projector active literal")
            try:
                decoded_literal = base64.b64decode(literal.encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as exc:
                raise ValueError("activation Submission literal checkpoint is not canonical base64") from exc
            if base64.b64encode(decoded_literal).decode("ascii") != literal or len(decoded_literal) > 9:
                raise ValueError("activation Submission literal checkpoint is invalid")
            projector._literal = bytearray(decoded_literal)
        if projector._root_done and projector._stack:
            raise ValueError("completed activation Submission projector retains a stack")
        if not projector._root_started and (projector._stack or projector._root_done):
            raise ValueError("activation Submission projector root state is inconsistent")
        return projector

    def finish(self) -> tuple[str, str | None]:
        if self._active_kind is not None or self._stack or not self._root_started or not self._root_done:
            raise ValueError("Submission JSON ended before its structure was complete")
        if not self._operation_seen or not self._state_seen or not self._target_group_seen:
            raise ValueError("Submission JSON is missing one of the required metadata fields")
        if self._operation_id_value != self._operation_id:
            raise ValueError("Submission operation_id does not match its filename")
        if self._state_value not in _VALID_STATES:
            raise ValueError("Submission state is invalid")
        try:
            validate_group_name(self._target_group_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Submission target_group is invalid") from exc
        return self._state_value, self._target_group_value

    def _feed_punctuation(self, kind: str) -> None:
        if kind == "{":
            self._begin_container("object")
        elif kind == "[":
            self._begin_container("array")
        elif kind in {"}", "]"}:
            self._end_container(kind)
        elif kind == ":":
            frame = self._require_frame("object")
            if frame.state != "colon":
                raise ValueError("JSON colon appeared outside an object key")
            frame.state = "value"
        elif kind == ",":
            frame = self._require_frame()
            if frame.state != "comma_or_end":
                raise ValueError("JSON comma appeared without a preceding value")
            frame.state = "key" if frame.kind == "object" else "array_value"

    def _start_token(self, kind: str) -> None:
        if kind == "string":
            handler = self._prepare_scalar(kind)
            limit = {
                "key": MAX_KEY_BYTES,
                "operation_id": MAX_OPERATION_ID_BYTES,
                "state": MAX_STATE_BYTES,
                "target_group": MAX_GROUP_BYTES,
            }.get(handler)
            self._collector = _StringCollector(limit) if limit is not None else None
            self._active_handler = handler
        elif kind in {"number", "literal"}:
            self._active_handler = self._prepare_scalar(kind)
            self._collector = None
            self._literal.clear()
        else:
            raise ValueError("unsupported JSON scalar token")
        self._active_kind = kind

    def _finish_token(self, kind: str) -> None:
        handler = self._active_handler
        if handler is None:
            raise RuntimeError("activation Submission token handler is missing")
        if kind == "string":
            if self._collector is None and handler != "ignored":
                raise RuntimeError("activation Submission string collector is missing")
            text = self._collector.finish() if self._collector is not None else None
            if handler == "key":
                assert text is not None
                self._consume_key(text)
            elif handler == "operation_id":
                assert text is not None
                try:
                    validate_identifier(text, "submission operation_id")
                except ValueError as exc:
                    raise ValueError("Submission operation_id is invalid") from exc
                if text != self._operation_id:
                    raise ValueError("Submission operation_id does not match its filename")
                self._operation_id_value = text
                self._operation_seen = True
            elif handler == "state":
                assert text is not None
                if text not in _VALID_STATES:
                    raise ValueError("Submission state is invalid")
                self._state_value = text
                self._state_seen = True
            elif handler == "target_group":
                assert text is not None
                try:
                    self._target_group_value = validate_group_name(text)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Submission target_group is invalid") from exc
                self._target_group_seen = True
        elif kind == "literal":
            literal = bytes(self._literal)
            if literal not in _JSON_LITERALS:
                raise ValueError("JSON contains a non-standard literal")
            if handler == "target_group_null":
                if literal != b"null":
                    raise ValueError("Submission target_group must be a Group name or null")
                self._target_group_value = None
                self._target_group_seen = True
        self._active_kind = None
        self._active_handler = None
        self._collector = None
        self._literal.clear()

    def _consume_key(self, key: str) -> None:
        if not self._stack or self._stack[-1].kind != "object":
            raise ValueError("JSON string appeared where an object key was required")
        frame = self._stack[-1]
        if frame.state not in {"key_or_end", "key"}:
            raise ValueError("JSON object key appeared in an invalid position")
        if frame.role == "root" and key == "submission":
            if key in frame.seen:
                raise ValueError("Submission root contains duplicate submission fields")
            frame.seen.add(key)
        elif frame.role == "submission" and key in {"operation_id", "state", "target_group"}:
            if key in frame.seen:
                raise ValueError(f"Submission contains duplicate {key} fields")
            frame.seen.add(key)
        frame.pending_key = key
        frame.state = "colon"

    def _prepare_scalar(self, kind: str) -> str:
        if not self._stack:
            if not self._root_started:
                if kind != "string":
                    raise ValueError("Submission root must be an object")
                raise ValueError("Submission root must be an object")
            raise ValueError("JSON contains more than one root value")
        frame = self._stack[-1]
        if frame.kind == "object" and frame.state in {"key_or_end", "key"}:
            if kind != "string":
                raise ValueError("JSON object key must be a string")
            return "key"
        key = self._prepare_value(frame)
        if frame.role == "root" and key == "submission":
            raise ValueError("root submission field must be an object")
        if frame.role != "submission" or key not in {"operation_id", "state", "target_group"}:
            return "ignored"
        if key in {"operation_id", "state"} and kind != "string":
            raise ValueError(f"Submission {key} must be a string")
        if key == "target_group":
            if kind == "string":
                return "target_group"
            if kind == "literal":
                return "target_group_null"
            raise ValueError("Submission target_group must be a Group name or null")
        return key

    def _prepare_value(self, frame: _Frame) -> str | None:
        if frame.kind == "object":
            if frame.state != "value" or frame.pending_key is None:
                raise ValueError("JSON object value appeared without its key")
            key = frame.pending_key
            frame.pending_key = None
            frame.state = "comma_or_end"
            return key
        if frame.state not in {"value_or_end", "array_value"}:
            raise ValueError("JSON array value appeared in an invalid position")
        frame.state = "comma_or_end"
        return None

    def _begin_value_for_container(self, kind: str) -> str:
        if not self._stack:
            if self._root_started:
                raise ValueError("JSON contains more than one root value")
            if kind != "object":
                raise ValueError("Submission root must be an object")
            self._root_started = True
            self._stack.append(_Frame("root", "object", "key_or_end"))
            return "root"
        parent = self._stack[-1]
        key = self._prepare_value(parent)
        if len(self._stack) >= MAX_NESTING:
            raise ValueError("Submission JSON nesting exceeds its bound")
        if parent.role == "root" and key == "submission":
            if kind != "object":
                raise ValueError("root submission field must be an object")
            role = "submission"
        elif parent.role == "submission" and key in {"operation_id", "state", "target_group"}:
            raise ValueError(f"Submission {key} must be a scalar")
        else:
            role = "ignored"
        self._stack.append(_Frame(role, kind, "key_or_end" if kind == "object" else "value_or_end"))
        return role

    def _begin_container(self, kind: str) -> None:
        self._begin_value_for_container(kind)

    def _end_container(self, kind: str) -> None:
        expected_kind = "object" if kind == "}" else "array"
        frame = self._require_frame(expected_kind)
        if frame.state not in {"key_or_end", "comma_or_end", "value_or_end"}:
            raise ValueError("JSON container ended before its value was complete")
        self._stack.pop()
        if frame.role == "root":
            self._root_done = True

    def _require_frame(self, kind: str | None = None) -> _Frame:
        if not self._stack:
            raise ValueError("JSON punctuation appeared outside a container")
        frame = self._stack[-1]
        if kind is not None and frame.kind != kind:
            raise ValueError("JSON punctuation does not match its container")
        return frame


def advance_activation_submission_source(
    storage: UpgradeStorage,
    source_path: os.PathLike[str] | str,
    operation_id: str,
    checkpoint_path: os.PathLike[str] | str,
    checkpoint_namespace: str,
) -> ActivationSubmissionStep:
    """Advance one bounded, revision-bound activation projection step."""
    from ..upgrade.contracts import UpgradeStorage

    if not isinstance(storage, UpgradeStorage):
        raise TypeError("activation Submission storage must be UpgradeStorage")
    source = _canonical_path(source_path, "source_path")
    checkpoint = _canonical_path(checkpoint_path, "checkpoint_path")
    validate_identifier(operation_id, "submission operation_id")
    if checkpoint_namespace not in {"submissions", "submission_control"}:
        raise ValueError("activation Submission checkpoint namespace is invalid")

    envelope = _read_checkpoint(storage, checkpoint)
    if envelope is not None and (
        envelope["namespace"] != checkpoint_namespace
        or envelope["operation_id"] != operation_id
        or envelope["source_path"] != str(source)
    ):
        # A single coordinator-owned checkpoint is reused between directory entries.
        # Its identity binding makes replacing it safe and restartable.
        storage.unlink(checkpoint)
        envelope = None

    offset = 0
    projector = _ActivationSubmissionProjector(operation_id)
    scanner_snapshot: dict[str, object] | None = None
    expected_revision: RegularFileRevision | None = None
    if envelope is not None:
        expected_revision = _revision_from_value(envelope["source_revision"])
        scanner_snapshot = dict(envelope["scanner"])
        offset = _snapshot_offset(scanner_snapshot)
        if offset > expected_revision.size:
            storage.unlink(checkpoint)
            expected_revision = None
            scanner_snapshot = None
            offset = 0
        else:
            projector = _ActivationSubmissionProjector.from_snapshot(operation_id, envelope["projector"])

    read = storage.read_regular_bytes(source, offset, STREAM_CHUNK_BYTES)
    source_revision = _coerce_revision(read)
    if expected_revision is not None and source_revision != expected_revision:
        # The bytes read at a stale nonzero offset cannot be reused for a fresh
        # parse. Persist a zero-offset checkpoint and let the next bounded call
        # read the replacement's first chunk.
        storage.unlink(checkpoint)
        offset = 0
        projector = _ActivationSubmissionProjector(operation_id)
        scanner_snapshot = None
        _persist_checkpoint(
            storage,
            checkpoint,
            _checkpoint_value(
                checkpoint_namespace,
                operation_id,
                source,
                source_revision,
                _initial_scanner_snapshot(),
                projector.snapshot(),
            ),
        )
        return ActivationSubmissionStep(
            "progressed",
            completed_bytes=0,
            total_bytes=source_revision.size,
            source_revision=source_revision.to_dict(),
            restarted=True,
        )
    else:
        data = read.data
        source_eof = read.eof

    reader = _ChunkReader(offset, data)

    def emit(_span: Span) -> None:
        return None

    def emit_bytes(span: Span, raw: bytes) -> None:
        projector.feed(span, raw)

    if scanner_snapshot is None:
        scanner = Scanner(reader, emit, chunk_bytes=STREAM_CHUNK_BYTES, emit_bytes=emit_bytes)
    else:
        if scanner_snapshot.get("raw_mode") is not True:
            raise ValueError("activation Submission checkpoint is missing raw scanner state")
        scanner = Scanner.from_snapshot(reader, emit, scanner_snapshot, emit_bytes=emit_bytes)
    result = scanner.step(max(1, len(data)))
    if source_eof and not result.is_complete:
        # The reader is an in-memory window and this empty read does not fetch a
        # second source chunk. It lets Scanner distinguish a short final slice
        # from a parser that merely ran out of input.
        result = scanner.step(1)
    if not result.is_complete:
        scanner.flush_fragment()
        scanner_snapshot = scanner.snapshot()
        _persist_checkpoint(
            storage,
            checkpoint,
            _checkpoint_value(
                checkpoint_namespace,
                operation_id,
                source,
                source_revision,
                scanner_snapshot,
                projector.snapshot(),
            ),
        )
        return ActivationSubmissionStep(
            "progressed",
            completed_bytes=_snapshot_offset(scanner_snapshot),
            total_bytes=source_revision.size,
            source_revision=source_revision.to_dict(),
            previous_completed_bytes=offset,
        )

    submission_state, target_group = projector.finish()
    storage.unlink(checkpoint)
    return ActivationSubmissionStep("complete", submission_state, target_group)


def _checkpoint_value(
    namespace: str,
    operation_id: str,
    source: Path,
    revision: RegularFileRevision,
    scanner: dict[str, object],
    projector: dict[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "namespace": namespace,
        "operation_id": operation_id,
        "source_path": str(source),
        "source_revision": revision.to_dict(),
        "scanner": scanner,
        "projector": projector,
    }
    value["checksum"] = _checkpoint_checksum(value)
    return value


def _initial_scanner_snapshot() -> dict[str, object]:
    return Scanner(
        _ChunkReader(0, b""),
        lambda _span: None,
        chunk_bytes=STREAM_CHUNK_BYTES,
        emit_bytes=lambda _span, _raw: None,
    ).snapshot()


def _read_checkpoint(storage: UpgradeStorage, path: os.PathLike[str] | str) -> dict[str, object] | None:
    checkpoint = _canonical_path(path, "checkpoint_path")
    try:
        value = storage.read_regular_bytes(checkpoint, 0, MAX_CHECKPOINT_BYTES)
    except FileNotFoundError:
        return None
    if not value.eof:
        raise ValueError(f"activation Submission checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes: {checkpoint}")
    try:
        parsed = json.loads(value.data, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"activation Submission checkpoint is malformed: {checkpoint}") from exc
    if type(parsed) is not dict or set(parsed) != _CHECKPOINT_KEYS or parsed.get("version") != 1:
        raise ValueError(f"activation Submission checkpoint has an invalid shape: {checkpoint}")
    checksum = parsed["checksum"]
    if type(checksum) is not str or checksum != _checkpoint_checksum(parsed):
        raise ValueError(f"activation Submission checkpoint failed its integrity check: {checkpoint}")
    if parsed["namespace"] not in {"submissions", "submission_control"}:
        raise ValueError("activation Submission checkpoint namespace is invalid")
    if type(parsed["operation_id"]) is not str or type(parsed["source_path"]) is not str:
        raise ValueError("activation Submission checkpoint identity is invalid")
    _revision_from_value(parsed["source_revision"])
    Scanner._validate_snapshot(parsed["scanner"])
    if type(parsed["projector"]) is not dict:
        raise ValueError("activation Submission projector checkpoint is invalid")
    return parsed


def _persist_checkpoint(storage: UpgradeStorage, path: os.PathLike[str] | str, value: dict[str, object]) -> None:
    if json_encoded_size(value) > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"activation Submission checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes")
    storage.atomic_replace(path, value)


def _checkpoint_checksum(value: dict[str, object]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "checksum"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _revision_from_value(value: object) -> RegularFileRevision:
    from ..upgrade.contracts import RegularFileRevision

    if type(value) is not dict or frozenset(value) != _REVISION_KEYS:
        raise ValueError("activation Submission source revision has an invalid shape")
    values = {key: _exact_int(value[key], f"source revision {key}", minimum=0) for key in _REVISION_KEYS}
    return RegularFileRevision(**values)


def _coerce_revision(value: "RegularFileRead") -> "RegularFileRevision":
    from ..upgrade.contracts import RegularFileRead

    if not isinstance(value, RegularFileRead):
        raise TypeError("regular-file read returned an invalid result")
    return value.revision


def _snapshot_offset(snapshot: object) -> int:
    if type(snapshot) is not dict:
        raise ValueError("activation Submission scanner checkpoint is not an object")
    offset = snapshot.get("offset")
    if type(offset) is not int or offset < 0:
        raise ValueError("activation Submission scanner offset is invalid")
    return offset


def _canonical_path(value: os.PathLike[str] | str, label: str) -> os.PathLike[str]:
    path = os.fspath(value)
    if isinstance(path, bytes):
        raise TypeError(f"{label} must be a text path")
    candidate = os.path.abspath(os.path.normpath(path))
    if not os.path.isabs(path) or path != candidate:
        raise ValueError(f"{label} must be an absolute canonical path")
    from pathlib import Path

    return Path(path)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint field: {key}")
        result[key] = value
    return result


def _exact_int(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} is above its maximum")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _optional_str(value: object, label: str) -> str | None:
    if value is not None and type(value) is not str:
        raise ValueError(f"{label} must be a string or null")
    return value


__all__ = ["ActivationSubmissionStep", "advance_activation_submission_source"]
