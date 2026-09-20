"""Durable, identity-bound consecutive Group membership coverage.

This module stores only committed membership locators derived from a whole-source
confirmation.  It does not authorize Task effects, launch, cancellation, or
resource release.  Membership publication performs fixed per-row filesystem
I/O and is deliberately outside the cooperative :class:`SliceIO` syscall
accounting used by source projection and qualification.
"""

from __future__ import annotations

import base64
import binascii
import errno
import json
import os
import stat
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..group_namespace import group_authority_identity
from ..locks import exclusive
from ..paths import group_path, submission_path
from ..records import validate_group_name, validate_identifier
from ..store import atomic_replace, read_json_limited
from .fingerprint import ChainedDigest
from .recovery import _decode_receipt
from .source_revision import SourceRevision

_ROW_BYTES = 32
_MAX_LINE_BYTES = 90_000
_MAX_RECORD_BYTES = 65_536
_MAX_RECEIPT_BYTES = 16_384
_MAX_MEMBERS = 64
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_U64_MAX = (1 << 64) - 1
_EVENT_CHUNK_KEYS = frozenset({"type", "kind", "ordinal", "data_b64", "is_final"})
_EVENT_END_KEYS = frozenset({"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"})
_IDENTIFIER_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

_STATE_KEYS = frozenset({"version", "identity", "prefix", "blocked_reason"})
_CURSOR_KEYS = frozenset({"version", "identity", "operation_id", "source_revision", "task_count", "next_ordinal"})
_SLOT_KEYS = frozenset(
    {
        "version",
        "identity",
        "sequence",
        "operation_id",
        "ordinal",
        "source_revision",
        "spool",
        "task_ref",
    }
)
_TASK_REF_KEYS = frozenset({"start", "end", "size"})
_REVISION_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})


@dataclass(frozen=True, slots=True)
class CoverageStatus:
    """The bounded contiguous membership prefix and current Group tail."""

    prefix: int
    tail: int
    is_complete: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class PublicationStep:
    """The bounded result of one source membership publication call."""

    state: str
    published: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class MemberIdentity:
    """A certified Group member locator returned by direct coverage lookup."""

    sequence: int
    task_id: str
    operation_id: str


@dataclass(frozen=True, slots=True)
class _SourceFiles:
    spool: Path
    task_references: Path
    sequence_references: Path
    source_root: Path


@dataclass(frozen=True, slots=True)
class _Row:
    ordinal: int
    task_start: int
    task_end: int
    task_size: int
    sequence: int


class _PendingGroupFinalization(Exception):
    """The source's next sequence is beyond the currently committed tail."""


class _MalformedRecord(Exception):
    """A bounded coverage record failed its exact schema or provenance check."""


class GroupCoverage:
    """Persist durable locators for one Group authority identity.

    Construction is pure.  The first operation that needs filesystem state
    reads the canonical Group authority identity and binds this instance to
    that project, directory identity, and Group name.  A later authority
    replacement raises instead of silently selecting a new namespace.
    """

    def __init__(self, root: Path, group: str):
        if not isinstance(root, (Path, str)):
            raise TypeError("root must be a path")
        if not isinstance(group, str):
            raise TypeError("group must be a string")
        validate_group_name(group)
        self._root = Path(os.path.abspath(os.fspath(root)))
        self._group = group
        self._identity: dict[str, Any] | None = None
        self._directory: Path | None = None

    @property
    def directory(self) -> Path:
        """Return the identity-keyed durable coverage directory."""

        self._bind_identity()
        if self._directory is None:
            raise RuntimeError("coverage identity was not initialized")
        return self._directory

    def source_scratch(self, operation_id: str, *, revision: SourceRevision | None = None) -> Path:
        """Return the scratch path for one operation and optional source revision.

        The path is only derived; this method does not create directories or
        inspect source files.  A revision child keeps a new committed source
        for a reused operation separate from stale confirmation scratch.
        """

        operation_id = _validate_operation_id(operation_id)
        base = self.directory / "sources" / _digest_text(operation_id)
        if revision is None:
            return base
        return base / _digest_revision(revision)

    def status(self) -> CoverageStatus:
        """Read coverage metadata under a short nonblocking publication lock."""

        directory = self.directory
        with exclusive(directory / ".lock", blocking=False) as acquired:
            if not acquired:
                return self._busy_status()
            self._initialize_layout()
            return self._status_locked()

    def read_member(self, sequence: int) -> MemberIdentity:
        """Read one certified member after checking its complete source provenance.

        The caller must reread the authoritative Task while holding the Group and
        Task fences before applying any effect.  This method performs direct
        locator I/O only: memory is O(the single identifier), and I/O is
        O(the encoded identifier), outside machine ``SliceIO`` accounting.

        Args:
            sequence: The positive membership sequence to resolve.

        Returns:
            The requested sequence, task identifier, and source operation ID.

        Raises:
            BlockingIOError: If the coverage lock is busy.
            ValueError: If coverage metadata or source provenance is invalid.
            OSError: If a required coverage artifact is unavailable.
        """

        sequence = _validate_member_sequence(sequence)
        directory = self.directory
        with exclusive(directory / ".lock", blocking=False) as acquired:
            if not acquired:
                raise BlockingIOError(errno.EWOULDBLOCK, "Group coverage is busy")
            self._initialize_layout()
            status = self._status_locked()
            if status.reason not in {None, "pending_submission"}:
                raise ValueError(f"Group coverage is unavailable: {status.reason}")
            if sequence > status.tail or sequence > status.prefix:
                raise ValueError("requested membership sequence is not certified")

            identity = self._require_bound_identity()
            slot_path = directory / "members" / f"{sequence}.json"
            try:
                slot = self._read_optional_record(slot_path)
                if slot is None:
                    raise ValueError(f"certified member slot is missing: {slot_path}")
                self._validate_slot(slot, identity, expected_sequence=sequence)
                member = self._read_member_locked(slot)
            except _MalformedRecord as exc:
                raise ValueError("certified member slot is malformed") from exc

            self._require_bound_identity()
            return member

    def publish(self, source: Any, *, max_members: int = 1) -> PublicationStep:
        """Publish at most ``max_members`` locators from one confirmed source."""

        max_members = _validate_max_members(max_members)
        directory = self.directory
        with exclusive(directory / ".lock", blocking=False) as acquired:
            if not acquired:
                return PublicationStep("waiting", 0, "busy")
            self._initialize_layout()
            return self._publish_locked(source, max_members)

    def advance(self, *, max_members: int = _MAX_MEMBERS) -> CoverageStatus:
        """Advance the durable contiguous prefix through at most ``max_members`` slots."""

        max_members = _validate_max_members(max_members)
        directory = self.directory
        with exclusive(directory / ".lock", blocking=False) as acquired:
            if not acquired:
                return self._busy_status()
            self._initialize_layout()
            return self._advance_locked(max_members)

    def _initialize_layout(self) -> None:
        directory = self.directory
        if (directory / "state.json").exists():
            return
        # The initial state is also the commit point for this fixed layout.
        # Retrying after a mkdir-before-fsync crash resynchronizes every link.
        for name in ("members", "publications", "sources"):
            child = directory / name
            child.mkdir(exist_ok=True)
            _sync_directory(child)
        for parent in (directory, directory.parent, directory.parent.parent):
            _sync_directory(parent)
        self._write_state(self._require_bound_identity(), 0, None)

    def _bind_identity(self) -> dict[str, Any]:
        current = _owner_identity(group_authority_identity(self._root), self._group)
        if self._identity is None:
            self._identity = current
            digest = _digest_json(current)
            self._directory = self._root / "indexes" / "group-discovery" / digest
            return current
        if current != self._identity:
            raise RuntimeError("Group authority identity changed while coverage was bound")
        return self._identity

    def _busy_status(self) -> CoverageStatus:
        try:
            tail, pending = self._group_state()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return CoverageStatus(0, 0, False, "busy")
        return CoverageStatus(0, tail, False, "busy" if not pending else "pending_submission")

    def _status_locked(self) -> CoverageStatus:
        identity = self._require_bound_identity()
        try:
            tail, pending = self._group_state()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return CoverageStatus(0, 0, False, "invalid_group")
        try:
            prefix, blocked_reason = self._read_state(identity)
        except _MalformedRecord:
            return CoverageStatus(0, tail, False, "invalid_state")
        if prefix > tail:
            return CoverageStatus(prefix, tail, False, "invalid_state")
        reason = blocked_reason or ("pending_submission" if pending else None)
        complete = prefix >= tail and reason is None
        return CoverageStatus(prefix, tail, complete, reason)

    def _advance_locked(self, max_members: int) -> CoverageStatus:
        identity = self._require_bound_identity()
        try:
            tail, pending = self._group_state()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return CoverageStatus(0, 0, False, "invalid_group")
        try:
            prefix, blocked_reason = self._read_state(identity)
        except _MalformedRecord:
            return CoverageStatus(0, tail, False, "invalid_state")
        if prefix > tail:
            return CoverageStatus(prefix, tail, False, "invalid_state")
        if blocked_reason is not None:
            return CoverageStatus(prefix, tail, False, blocked_reason)

        candidate = prefix + 1
        checked = 0
        while candidate <= tail and checked < max_members:
            slot_path = self.directory / "members" / f"{candidate}.json"
            try:
                slot = self._read_optional_record(slot_path)
            except _MalformedRecord:
                return self._block_locked(identity, prefix, "invalid_member_slot", tail, pending)
            if slot is None:
                break
            try:
                self._validate_slot(slot, identity, expected_sequence=candidate)
            except _MalformedRecord:
                return self._block_locked(identity, prefix, "invalid_member_slot", tail, pending)
            prefix = candidate
            candidate += 1
            checked += 1

        original_prefix = self._read_state(identity)[0]
        if prefix != original_prefix:
            self._write_state(identity, prefix, None)
        reason = "pending_submission" if pending else None
        return CoverageStatus(prefix, tail, prefix >= tail and reason is None, reason)

    def _publish_locked(self, source: Any, max_members: int) -> PublicationStep:
        identity = self._require_bound_identity()
        try:
            prefix, blocked_reason = self._read_state(identity)
        except _MalformedRecord:
            return self._block_locked_step(identity, 0, "invalid_state")
        if blocked_reason is not None:
            return PublicationStep("blocked", 0, blocked_reason)

        try:
            operation_id, source_revision, task_count, files = self._validate_source(source, identity)
        except _PendingPublication as exc:
            return PublicationStep("waiting", 0, str(exc))
        cursor_path = self.directory / "publications" / f"{_digest_text(operation_id)}.json"
        cursor = self._read_optional_record(cursor_path)
        if cursor is not None:
            try:
                next_ordinal = self._validate_cursor(
                    cursor,
                    identity,
                    operation_id,
                    source_revision,
                    task_count,
                )
            except _MalformedRecord as exc:
                reason = "source_replaced" if str(exc) == "source_replaced" else "invalid_publication_cursor"
                return self._block_locked_step(identity, prefix, reason)
        else:
            next_ordinal = 0

        if next_ordinal == task_count:
            if cursor is None:
                self._write_cursor(cursor_path, identity, operation_id, source_revision, task_count, next_ordinal)
            return PublicationStep("complete", 0, None)

        refs_task_fd, refs_sequence_fd, spool_stream = self._open_source_files(files, task_count)
        published = 0
        processed = 0
        try:
            while next_ordinal < task_count and processed < max_members:
                processed += 1
                try:
                    row = self._read_row(
                        refs_task_fd,
                        refs_sequence_fd,
                        spool_stream,
                        next_ordinal,
                        source_revision,
                        files.spool,
                        self._group_tail_value(),
                    )
                except _PendingGroupFinalization:
                    return PublicationStep("waiting", published, "pending_group_finalization")
                slot_path = self.directory / "members" / f"{row.sequence}.json"
                existing = self._read_optional_record(slot_path)
                expected_slot = self._slot_record(
                    identity,
                    operation_id,
                    source_revision,
                    files,
                    row,
                )
                if existing is not None:
                    try:
                        self._validate_slot(existing, identity, expected_sequence=row.sequence)
                    except _MalformedRecord:
                        return self._block_locked_step(identity, prefix, "member_conflict")
                    if existing != expected_slot:
                        return self._block_locked_step(identity, prefix, "member_conflict")
                else:
                    atomic_replace(slot_path, expected_slot)
                    published += 1

                next_ordinal += 1
                self._write_cursor(
                    cursor_path,
                    identity,
                    operation_id,
                    source_revision,
                    task_count,
                    next_ordinal,
                )
        finally:
            os.close(refs_task_fd)
            os.close(refs_sequence_fd)
            spool_stream.close()

        state = "complete" if next_ordinal == task_count else "progressed"
        return PublicationStep(state, published, None)

    def _validate_source(self, source: Any, identity: dict[str, Any]) -> tuple[str, SourceRevision, int, _SourceFiles]:
        if source is None:
            raise TypeError("source must be a ConfirmedSource")
        if getattr(source, "status", None) != "qualified":
            return _raise_publication_wait("source_not_qualified")
        operation_id = _validate_operation_id(getattr(source, "operation_id", None))
        if getattr(source, "group", None) != self._group:
            return _raise_publication_wait("wrong_group")
        source_revision = getattr(source, "source_revision", None)
        if not isinstance(source_revision, SourceRevision):
            raise ValueError("confirmed source has an invalid source revision")
        task_count = getattr(source, "task_count", None)
        if type(task_count) is not int or task_count < 0:
            raise ValueError("confirmed source task_count must be a nonnegative integer")

        source_path = _as_path(getattr(source, "source", None), "source")
        _verify_source_revision(source_path, source_revision)
        spool = _as_path(getattr(source, "spool", None), "spool")
        task_references = _as_path(getattr(source, "task_references", None), "task_references")
        sequence_references = _as_path(getattr(source, "sequence_references", None), "sequence_references")
        source_root = _as_path(spool.parent.parent, "source scratch")

        for path, label in (
            (spool, "spool"),
            (task_references, "task references"),
            (sequence_references, "sequence references"),
        ):
            _validate_locator(path, self.directory, operation_id, label)
        _validate_directory_locator(source_root, self.directory, operation_id)
        receipt = source_root / "receipt.json"
        _require_regular_path(receipt, "source receipt")

        if source_root != spool.parent.parent:
            raise ValueError("confirmed source spool does not identify its scratch root")
        # Keep the owner argument in the validation boundary: callers cannot
        # smuggle a source from another Group into this namespace.
        if identity.get("group") != self._group:
            raise RuntimeError("coverage identity does not match its Group")
        return (
            operation_id,
            source_revision,
            task_count,
            _SourceFiles(
                spool=spool,
                task_references=task_references,
                sequence_references=sequence_references,
                source_root=source_root,
            ),
        )

    def _open_source_files(self, files: _SourceFiles, task_count: int) -> tuple[int, int, BinaryIO]:
        task_fd = _open_regular(files.task_references, "task references")
        sequence_fd: int | None = None
        spool_fd: int | None = None
        try:
            task_size = os.fstat(task_fd).st_size
            expected_size = task_count * _ROW_BYTES
            if task_size != expected_size:
                raise ValueError("task reference size does not match confirmed task_count")
            sequence_fd = _open_regular(files.sequence_references, "sequence references")
            if os.fstat(sequence_fd).st_size != expected_size:
                raise ValueError("sequence reference size does not match confirmed task_count")
            spool_fd = _open_regular(files.spool, "spool")
            stream = os.fdopen(spool_fd, "rb", buffering=0)
            spool_fd = None
            return task_fd, sequence_fd, stream
        except BaseException:
            os.close(task_fd)
            if sequence_fd is not None:
                os.close(sequence_fd)
            if spool_fd is not None:
                os.close(spool_fd)
            raise

    def _read_row(
        self,
        task_fd: int,
        sequence_fd: int,
        spool: BinaryIO,
        ordinal: int,
        source_revision: SourceRevision,
        spool_path: Path,
        tail: int,
    ) -> _Row:
        task_record = _read_reference(task_fd, ordinal, "task")
        sequence_record = _read_reference(sequence_fd, ordinal, "sequence")
        task_start, task_end, task_size = _validate_reference(task_record, spool_path, "task")
        sequence_start, sequence_end, sequence_size = _validate_reference(sequence_record, spool_path, "sequence")
        sequence = _decode_sequence(
            spool,
            sequence_start,
            sequence_end,
            sequence_size,
            ordinal,
            tail,
            source_revision,
        )
        return _Row(ordinal, task_start, task_end, task_size, sequence)

    def _slot_record(
        self,
        identity: dict[str, Any],
        operation_id: str,
        source_revision: SourceRevision,
        files: _SourceFiles,
        row: _Row,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "identity": identity,
            "sequence": row.sequence,
            "operation_id": operation_id,
            "ordinal": row.ordinal,
            "source_revision": _revision_dict(source_revision),
            "spool": _relative_locator(files.spool, self.directory),
            "task_ref": {"start": row.task_start, "end": row.task_end, "size": row.task_size},
        }

    def _write_cursor(
        self,
        path: Path,
        identity: dict[str, Any],
        operation_id: str,
        source_revision: SourceRevision,
        task_count: int,
        next_ordinal: int,
    ) -> None:
        atomic_replace(
            path,
            {
                "version": 1,
                "identity": identity,
                "operation_id": operation_id,
                "source_revision": _revision_dict(source_revision),
                "task_count": task_count,
                "next_ordinal": next_ordinal,
            },
        )

    def _validate_cursor(
        self,
        cursor: dict[str, Any],
        identity: dict[str, Any],
        operation_id: str,
        source_revision: SourceRevision,
        task_count: int,
    ) -> int:
        if frozenset(cursor) != _CURSOR_KEYS or cursor.get("version") != 1:
            raise _MalformedRecord("invalid cursor")
        if cursor.get("identity") != identity or cursor.get("operation_id") != operation_id:
            raise _MalformedRecord("invalid cursor identity")
        try:
            stored_revision = _revision_from_dict(cursor.get("source_revision"))
        except (TypeError, ValueError) as exc:
            raise _MalformedRecord("invalid cursor revision") from exc
        if stored_revision != source_revision:
            raise _MalformedRecord("source_replaced")
        if cursor.get("task_count") != task_count:
            raise _MalformedRecord("source task count changed")
        next_ordinal = cursor.get("next_ordinal")
        if type(next_ordinal) is not int or not 0 <= next_ordinal <= task_count:
            raise _MalformedRecord("invalid cursor ordinal")
        return next_ordinal

    def _validate_slot(
        self,
        slot: dict[str, Any],
        identity: dict[str, Any],
        *,
        expected_sequence: int | None = None,
    ) -> None:
        if frozenset(slot) != _SLOT_KEYS or slot.get("version") != 1:
            raise _MalformedRecord("invalid slot")
        if slot.get("identity") != identity:
            raise _MalformedRecord("slot identity mismatch")
        sequence = slot.get("sequence")
        if type(sequence) is not int or not 1 <= sequence <= _U64_MAX:
            raise _MalformedRecord("invalid slot sequence")
        if expected_sequence is not None and sequence != expected_sequence:
            raise _MalformedRecord("slot sequence mismatch")
        operation_id = slot.get("operation_id")
        try:
            _validate_operation_id(operation_id)
        except (TypeError, ValueError) as exc:
            raise _MalformedRecord("invalid slot operation") from exc
        ordinal = slot.get("ordinal")
        if type(ordinal) is not int or ordinal < 0:
            raise _MalformedRecord("invalid slot ordinal")
        try:
            _revision_from_dict(slot.get("source_revision"))
        except (TypeError, ValueError) as exc:
            raise _MalformedRecord("invalid slot revision") from exc
        spool = slot.get("spool")
        if type(spool) is not str:
            raise _MalformedRecord("invalid slot spool locator")
        try:
            _validate_relative_locator(spool, self.directory, operation_id)
        except (TypeError, ValueError, OSError) as exc:
            raise _MalformedRecord("invalid slot spool locator") from exc
        task_ref = slot.get("task_ref")
        if type(task_ref) is not dict or frozenset(task_ref) != _TASK_REF_KEYS:
            raise _MalformedRecord("invalid slot task reference")
        start, end, size = task_ref["start"], task_ref["end"], task_ref["size"]
        if (
            type(start) is not int
            or type(end) is not int
            or type(size) is not int
            or not 0 <= start < end
            or not 0 <= size <= _U64_MAX
        ):
            raise _MalformedRecord("invalid slot task reference")

    def _read_member_locked(self, slot: dict[str, Any]) -> MemberIdentity:
        operation_id = slot["operation_id"]
        source_revision = _revision_from_dict(slot["source_revision"])
        spool_path = _validate_relative_locator(slot["spool"], self.directory, operation_id)
        if spool_path.parts[-2:] != ("projection", "events.jsonl"):
            raise ValueError("member slot spool must be projection/events.jsonl")
        scratch = spool_path.parent.parent

        receipt_path = scratch / "receipt.json"
        receipt_revision = SourceRevision.from_stat(_require_regular_path(receipt_path, "source receipt"))
        receipt_data = _read_bounded_bytes(receipt_path, _MAX_RECEIPT_BYTES, "source receipt")
        receipt = _decode_receipt(
            receipt_data,
            submission_path(self._root, operation_id),
            operation_id,
            self._group,
            scratch,
        )
        if receipt.get("status") != "qualified":
            raise ValueError("member source receipt is not qualified")
        receipt_source_revision = receipt.get("source_revision")
        if not isinstance(receipt_source_revision, SourceRevision) or receipt_source_revision != source_revision:
            raise ValueError("member source revision does not match its receipt")

        ordinal = slot["ordinal"]
        task_count = receipt.get("task_count")
        if type(ordinal) is not int or type(task_count) is not int or ordinal >= task_count:
            raise ValueError("member ordinal is outside the confirmed task list")

        spool_record = receipt.get("spool")
        if type(spool_record) is not dict or spool_record.get("path") != "projection/events.jsonl":
            raise ValueError("member receipt spool path is invalid")
        spool_revision = _receipt_file_revision(spool_record, "spool")
        if scratch / spool_record["path"] != spool_path:
            raise ValueError("member slot spool does not match its receipt")

        task_record = receipt.get("task_references")
        if type(task_record) is not dict or task_record.get("path") != "qualification/task.refs":
            raise ValueError("qualified member receipt is missing task references")
        task_revision = _receipt_file_revision(task_record, "task references")
        task_path = scratch / task_record["path"]
        _validate_locator(task_path, self.directory, operation_id, "task references")
        sequence_record = receipt.get("sequence_references")
        if type(sequence_record) is not dict or sequence_record.get("path") != "qualification/sequence.refs":
            raise ValueError("qualified member receipt is missing sequence references")
        sequence_revision = _receipt_file_revision(sequence_record, "sequence references")
        sequence_path = scratch / sequence_record["path"]
        _validate_locator(sequence_path, self.directory, operation_id, "sequence references")

        task_ref = slot["task_ref"]
        task_start = task_ref["start"]
        task_end = task_ref["end"]
        task_size = task_ref["size"]
        if (
            type(task_start) is not int
            or type(task_end) is not int
            or type(task_size) is not int
            or task_size < 1
            or not 0 <= task_start < task_end <= spool_revision.size
        ):
            raise ValueError("member task reference span is invalid")

        source_path = submission_path(self._root, operation_id)
        with ExitStack() as resources:
            source_fd = _open_expected_revision(source_path, receipt_source_revision, "source")
            resources.callback(os.close, source_fd)
            task_fd = _open_expected_revision(task_path, task_revision, "task references")
            resources.callback(os.close, task_fd)
            sequence_fd = _open_expected_revision(sequence_path, sequence_revision, "sequence references")
            resources.callback(os.close, sequence_fd)
            spool_fd = _open_expected_revision(spool_path, spool_revision, "spool")
            try:
                spool = os.fdopen(spool_fd, "rb", buffering=65_536)
            except BaseException:
                os.close(spool_fd)
                raise
            resources.callback(spool.close)

            _verify_descriptor_revision(source_fd, source_path, receipt_source_revision, "source")
            _verify_descriptor_revision(task_fd, task_path, task_revision, "task references")
            _verify_descriptor_revision(spool.fileno(), spool_path, spool_revision, "spool")
            try:
                task_reference = _read_reference(task_fd, ordinal, "task")
            except (OverflowError, ValueError, OSError) as exc:
                raise ValueError("could not read member task reference") from exc
            if task_reference != (ordinal, task_start, task_end, task_size):
                raise ValueError("member task reference does not match its slot")
            _verify_descriptor_revision(task_fd, task_path, task_revision, "task references")

            sequence_reference = _read_reference(sequence_fd, ordinal, "sequence")
            sequence_start, sequence_end, sequence_size = _validate_reference(
                sequence_reference, spool_path, "sequence"
            )
            try:
                sequence = _decode_sequence(
                    spool,
                    sequence_start,
                    sequence_end,
                    sequence_size,
                    ordinal,
                    slot["sequence"],
                    receipt_source_revision,
                )
            except _PendingGroupFinalization as exc:
                raise ValueError("member sequence does not match its qualified reference") from exc
            if sequence != slot["sequence"]:
                raise ValueError("member sequence does not match its qualified reference")

            task_id = _decode_task_id(
                spool,
                task_start,
                task_end,
                task_size,
                ordinal,
                receipt_source_revision,
            )
            _verify_descriptor_revision(spool.fileno(), spool_path, spool_revision, "spool")
            _verify_descriptor_revision(source_fd, source_path, receipt_source_revision, "source")
            _verify_descriptor_revision(task_fd, task_path, task_revision, "task references")
            _verify_descriptor_revision(sequence_fd, sequence_path, sequence_revision, "sequence references")
            _verify_source_revision(receipt_path, receipt_revision)

        return MemberIdentity(slot["sequence"], task_id, operation_id)

    def _read_state(self, identity: dict[str, Any]) -> tuple[int, str | None]:
        path = self.directory / "state.json"
        try:
            value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        except FileNotFoundError:
            return 0, None
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise _MalformedRecord("invalid state") from exc
        if frozenset(value) != _STATE_KEYS or value.get("version") != 1:
            raise _MalformedRecord("invalid state")
        if value.get("identity") != identity:
            raise _MalformedRecord("state identity mismatch")
        prefix = value.get("prefix")
        if type(prefix) is not int or prefix < 0:
            raise _MalformedRecord("invalid state prefix")
        blocked_reason = value.get("blocked_reason")
        if blocked_reason is not None and (type(blocked_reason) is not str or not blocked_reason):
            raise _MalformedRecord("invalid state blocked reason")
        return prefix, blocked_reason

    def _write_state(self, identity: dict[str, Any], prefix: int, blocked_reason: str | None) -> None:
        atomic_replace(
            self.directory / "state.json",
            {"version": 1, "identity": identity, "prefix": prefix, "blocked_reason": blocked_reason},
        )

    def _read_optional_record(self, path: Path) -> dict[str, Any] | None:
        try:
            _require_regular_path(path, "coverage record")
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise _MalformedRecord("invalid coverage record") from exc
        try:
            return read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise _MalformedRecord("invalid coverage record") from exc

    def _group_state(self) -> tuple[int, bool]:
        value = read_json_limited(group_path(self._root, self._group), max_bytes=_MAX_RECORD_BYTES)
        group = value.get("group")
        if type(group) is not dict:
            raise ValueError("Group envelope is malformed")
        next_sequence = group.get("next_membership_sequence")
        if type(next_sequence) is not int or next_sequence < 1:
            raise ValueError("Group membership tail is malformed")
        return next_sequence - 1, group.get("pending_submission_commit") is not None

    def _group_tail_value(self) -> int:
        return self._group_state()[0]

    def _require_bound_identity(self) -> dict[str, Any]:
        self._bind_identity()
        if self._identity is None:
            raise RuntimeError("coverage identity was not initialized")
        return self._identity

    def _block_locked(
        self,
        identity: dict[str, Any],
        prefix: int,
        reason: str,
        tail: int,
        pending: bool,
    ) -> CoverageStatus:
        self._write_state(identity, prefix, reason)
        return CoverageStatus(prefix, tail, False, reason or ("pending_submission" if pending else None))

    def _block_locked_step(self, identity: dict[str, Any], prefix: int, reason: str) -> PublicationStep:
        self._write_state(identity, prefix, reason)
        return PublicationStep("blocked", 0, reason)


def _raise_publication_wait(reason: str) -> Any:
    raise _PendingPublication(reason)


class _PendingPublication(Exception):
    """Internal nonqualified-source publication result."""


def _validate_max_members(value: int) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_MEMBERS:
        raise ValueError("max_members must be an integer from 1 through 64")
    return value


def _validate_member_sequence(value: object) -> int:
    if type(value) is not int:
        raise TypeError("sequence must be an exact integer")
    if value < 1:
        raise ValueError("sequence must be positive")
    return value


def _validate_operation_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("operation_id must be a string")
    validate_identifier(value, "operation_id")
    return value


def _as_path(value: Any, label: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise ValueError(f"confirmed source {label} must be a path")
    return Path(os.path.abspath(os.fspath(value)))


def _owner_identity(value: Any, group: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"project_id", "directory_identity"}:
        raise RuntimeError("Group authority identity is malformed")
    project_id = value.get("project_id")
    directory_identity = value.get("directory_identity")
    if type(project_id) is not str or not project_id or type(directory_identity) is not dict:
        raise RuntimeError("Group authority identity is malformed")
    return {"project_id": project_id, "directory_identity": dict(directory_identity), "group": group}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_json(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _revision_dict(revision: SourceRevision) -> dict[str, int]:
    if not isinstance(revision, SourceRevision):
        raise TypeError("source revision must be a SourceRevision")
    return {
        "device": revision.device,
        "inode": revision.inode,
        "size": revision.size,
        "mtime_ns": revision.mtime_ns,
        "ctime_ns": revision.ctime_ns,
    }


def _digest_revision(revision: SourceRevision) -> str:
    return _digest_json(_revision_dict(revision))


def _revision_from_dict(value: Any) -> SourceRevision:
    if type(value) is not dict or frozenset(value) != _REVISION_KEYS:
        raise ValueError("source revision record is malformed")
    kwargs = {key: value[key] for key in _REVISION_KEYS}
    return SourceRevision(**kwargs)


def _verify_source_revision(path: Path, expected: SourceRevision) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"confirmed source is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"confirmed source is not a regular file: {path}")
    actual = SourceRevision.from_stat(info)
    if actual != expected:
        raise ValueError(f"confirmed source revision changed: {path}")


def _require_regular_path(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return info


def _open_regular(path: Path, label: str) -> int:
    before = _require_regular_path(path, label)
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except OSError as exc:
        raise ValueError(f"could not open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening: {path}")
        after = _require_regular_path(path, label)
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"{label} changed while opening: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_expected_revision(path: Path, expected: SourceRevision, label: str) -> int:
    descriptor = _open_regular(path, label)
    try:
        _verify_descriptor_revision(descriptor, path, expected, label)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_descriptor_revision(
    descriptor: int,
    path: Path,
    expected: SourceRevision | None,
    label: str,
) -> SourceRevision:
    descriptor_info = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_info.st_mode):
        raise ValueError(f"{label} descriptor is not a regular file: {path}")
    descriptor_revision = SourceRevision.from_stat(descriptor_info)
    named_info = _require_regular_path(path, label)
    named_revision = SourceRevision.from_stat(named_info)
    if descriptor_revision != named_revision:
        raise ValueError(f"{label} path was replaced while reading: {path}")
    if expected is not None and descriptor_revision != expected:
        raise ValueError(f"{label} revision changed: {path}")
    return descriptor_revision


def _read_bounded_bytes(path: Path, max_bytes: int, label: str) -> bytes:
    """Read one regular file through a stable descriptor under a byte limit."""

    descriptor = _open_regular(path, label)
    try:
        revision = _verify_descriptor_revision(descriptor, path, None, label)
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, max_bytes + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise ValueError(f"{label} exceeds its {max_bytes}-byte limit: {path}")
        _verify_descriptor_revision(descriptor, path, revision, label)
        return bytes(payload)
    finally:
        os.close(descriptor)


def _receipt_file_revision(record: dict[str, object], label: str) -> SourceRevision:
    revision = record.get("revision")
    if not isinstance(revision, SourceRevision):
        raise ValueError(f"receipt {label} revision is invalid")
    return revision


def _validate_locator(path: Path, directory: Path, operation_id: str, label: str) -> None:
    _validate_relative_locator(_relative_locator(path, directory), directory, operation_id)
    _require_regular_path(path, label)


def _validate_directory_locator(path: Path, directory: Path, operation_id: str) -> None:
    _validate_relative_locator(_relative_locator(path, directory), directory, operation_id)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"source scratch directory is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"source scratch path is not a real directory: {path}")


def _relative_locator(path: Path, directory: Path) -> str:
    try:
        relative = path.relative_to(directory)
    except ValueError as exc:
        raise ValueError("coverage locator escapes its identity directory") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("coverage locator is not canonical")
    return relative.as_posix()


def _validate_relative_locator(value: str, directory: Path, operation_id: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("coverage locator must be a relative POSIX path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("coverage locator is not a safe descendant")
    expected_prefix = Path("sources") / _digest_text(operation_id)
    if relative.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
        raise ValueError("coverage locator belongs to another source")
    path = directory / relative
    current = directory
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("coverage locator traverses a symlink")
    return path


def _read_reference(descriptor: int, ordinal: int, label: str) -> tuple[int, int, int, int]:
    offset = ordinal * _ROW_BYTES
    try:
        payload = os.pread(descriptor, _ROW_BYTES, offset)
    except OSError as exc:
        raise ValueError(f"could not read {label} reference {ordinal}") from exc
    if len(payload) != _ROW_BYTES:
        raise ValueError(f"{label} reference {ordinal} is truncated")
    values = struct.unpack(">QQQQ", payload)
    if values[0] != ordinal:
        raise ValueError(f"{label} reference ordinal is not exact")
    return values


def _decode_task_id(
    spool: BinaryIO,
    start: int,
    end: int,
    expected_size: int,
    ordinal: int,
    source_revision: SourceRevision,
) -> str:
    position = start
    decoded_size = 0
    identifier = bytearray()
    digest = ChainedDigest()
    final_seen = False
    end_seen = False

    spool.seek(start, os.SEEK_SET)
    while position < end:
        line = spool.readline(_MAX_LINE_BYTES)
        if not line or not line.endswith(b"\n") or len(line) > _MAX_LINE_BYTES:
            raise ValueError("task ID spool line is incomplete or oversized")
        line_start = position
        position += len(line)
        if position > end:
            raise ValueError("task ID spool read exceeded its reference")
        try:
            event = _decode_event_line(line)
        except RecursionError as exc:
            raise ValueError("task ID spool line is too deeply nested") from exc
        event_type = event.get("type")
        if event_type == "chunk":
            if frozenset(event) != _EVENT_CHUNK_KEYS:
                raise ValueError("task ID chunk event shape is invalid")
            if event["kind"] != "task_id" or type(event["ordinal"]) is not int or event["ordinal"] != ordinal:
                raise ValueError("task ID chunk event identity is invalid")
            if type(event["is_final"]) is not bool or final_seen or end_seen:
                raise ValueError("task ID chunk final marker is invalid")
            data = _decode_chunk(event["data_b64"])
            if not data and not event["is_final"]:
                raise ValueError("empty task ID chunk is not final")
            if any(byte not in _IDENTIFIER_BYTES for byte in data):
                raise ValueError("task ID chunk is not an ASCII identifier")
            identifier.extend(data)
            digest.update(data)
            decoded_size += len(data)
            final_seen = event["is_final"]
        elif event_type == "end":
            if frozenset(event) != _EVENT_END_KEYS:
                raise ValueError("task ID end event shape is invalid")
            if event["kind"] != "task_id" or type(event["ordinal"]) is not int or event["ordinal"] != ordinal:
                raise ValueError("task ID end event identity is invalid")
            if type(event["decoded_size"]) is not int or event["decoded_size"] != expected_size:
                raise ValueError("task ID decoded size does not match its reference")
            digest_value = event["digest"]
            if (
                type(digest_value) is not str
                or len(digest_value) != 64
                or any(character not in "0123456789abcdef" for character in digest_value)
            ):
                raise ValueError("task ID end digest is invalid")
            source_start, source_end = event["start"], event["end"]
            if (
                type(source_start) is not int
                or type(source_end) is not int
                or not 0 <= source_start < source_end <= source_revision.size
            ):
                raise ValueError("task ID source span is invalid")
            if not final_seen or end_seen or line_start < start or position != end:
                raise ValueError("task ID end does not close its exact reference")
            if digest.size != expected_size or digest.hexdigest() != digest_value:
                raise ValueError("task ID end digest does not match chunk data")
            end_seen = True
        else:
            raise ValueError("unknown task ID spool event")

    if position != end or not end_seen or decoded_size != expected_size or decoded_size == 0:
        raise ValueError("task ID spool reference is incomplete")
    try:
        return bytes(identifier).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("task ID is not ASCII") from exc


def _validate_reference(values: tuple[int, int, int, int], spool_path: Path, label: str) -> tuple[int, int, int]:
    _, start, end, size = values
    try:
        spool_size = _require_regular_path(spool_path, "spool").st_size
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError("spool is unavailable") from exc
    if not 0 <= start < end <= spool_size or type(size) is not int or size < 1:
        raise ValueError(f"{label} reference span is invalid")
    return start, end, size


def _decode_sequence(
    spool: BinaryIO,
    start: int,
    end: int,
    expected_size: int,
    ordinal: int,
    tail: int,
    source_revision: SourceRevision,
) -> int:
    tail_digits = _decimal_text(tail)
    position = start
    decoded_size = 0
    value = 0
    comparison = 0
    final_seen = False
    end_seen = False
    first_chunk = True

    spool.seek(start, os.SEEK_SET)
    while position < end:
        line = spool.readline(_MAX_LINE_BYTES)
        if not line or not line.endswith(b"\n") or len(line) > _MAX_LINE_BYTES:
            raise ValueError("sequence spool line is incomplete or oversized")
        line_start = position
        position += len(line)
        if position > end:
            raise ValueError("sequence spool read exceeded its reference")
        event = _decode_event_line(line)
        event_type = event.get("type")
        if event_type == "chunk":
            if frozenset(event) != frozenset({"type", "kind", "ordinal", "data_b64", "is_final"}):
                raise ValueError("sequence chunk event shape is invalid")
            if event["kind"] != "sequence" or event["ordinal"] != ordinal:
                raise ValueError("sequence chunk event identity is invalid")
            if type(event["is_final"]) is not bool or final_seen or end_seen:
                raise ValueError("sequence chunk final marker is invalid")
            data = _decode_chunk(event["data_b64"])
            if not data and not event["is_final"]:
                raise ValueError("empty sequence chunk is not final")
            for byte in data:
                if (first_chunk and byte not in b"123456789") or (not first_chunk and byte not in b"0123456789"):
                    raise ValueError("sequence chunk is not canonical decimal")
                if decoded_size < len(tail_digits):
                    digit_position = decoded_size
                    digit = byte - 48
                    value = value * 10 + digit
                    if comparison == 0 and digit_position < len(tail_digits):
                        if byte > ord(tail_digits[digit_position]):
                            comparison = 1
                        elif byte < ord(tail_digits[digit_position]):
                            comparison = -1
                decoded_size += 1
                first_chunk = False
            final_seen = event["is_final"]
        elif event_type == "end":
            if frozenset(event) != frozenset({"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"}):
                raise ValueError("sequence end event shape is invalid")
            if event["kind"] != "sequence" or event["ordinal"] != ordinal:
                raise ValueError("sequence end event identity is invalid")
            if type(event["decoded_size"]) is not int or event["decoded_size"] != expected_size:
                raise ValueError("sequence decoded size does not match its reference")
            source_start, source_end = event["start"], event["end"]
            if (
                type(source_start) is not int
                or type(source_end) is not int
                or not 0 <= source_start < source_end <= source_revision.size
            ):
                raise ValueError("sequence source span is invalid")
            if not final_seen or end_seen or line_start < start or position != end:
                raise ValueError("sequence end does not close its exact reference")
            end_seen = True
        else:
            raise ValueError("unknown sequence spool event")

    if position != end or not end_seen or decoded_size != expected_size:
        raise ValueError("sequence spool reference is incomplete")
    if decoded_size > len(tail_digits) or (decoded_size == len(tail_digits) and comparison > 0):
        raise _PendingGroupFinalization
    return value


def _decode_event_line(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            line[:-1].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("sequence spool line is not strict JSON") from exc
    if type(value) is not dict:
        raise ValueError("sequence spool event is not an object")
    return value


def _decode_chunk(value: Any) -> bytes:
    if type(value) is not str:
        raise ValueError("sequence chunk data is not text")
    try:
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("sequence chunk data is not strict base64") from exc
    if base64.b64encode(decoded) != encoded or len(decoded) > 65_536:
        raise ValueError("sequence chunk data is not canonical")
    return decoded


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-JSON constant {value}")


def _decimal_text(value: int) -> str:
    try:
        return str(value)
    except ValueError as exc:
        raise ValueError("Group membership tail cannot be represented safely") from exc


__all__ = ["CoverageStatus", "GroupCoverage", "MemberIdentity", "PublicationStep"]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
