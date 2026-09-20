"""Durable, identity-bound consecutive Group membership coverage.

This module stores only committed membership locators derived from a whole-source
confirmation.  It does not authorize Task effects, launch, cancellation, or
resource release.  Membership publication performs fixed per-row filesystem
I/O and is deliberately outside the cooperative :class:`SliceIO` syscall
accounting used by source projection and qualification.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..group_namespace import group_authority_identity
from ..locks import exclusive
from ..paths import group_path
from ..records import validate_group_name, validate_identifier
from ..store import atomic_replace, read_json_limited
from .member_reader import (
    MemberIdentity,
    MemberRow,
    SequenceBeyondLimit,
    _ReaderWaiting,
    open_publication_reader,
    read_published_member,
)
from .source_revision import SourceRevision

_MAX_RECORD_BYTES = 65_536
_MAX_MEMBERS = 64
_U64_MAX = (1 << 64) - 1

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
                member = read_published_member(
                    self._root,
                    coverage_directory=directory,
                    group=self._group,
                    slot=slot,
                )
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
            with open_publication_reader(source, group=self._group, coverage_directory=self.directory) as reader:
                operation_id = reader.operation_id
                source_revision = reader.source_revision
                task_count = reader.task_count
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
                        self._write_cursor(
                            cursor_path, identity, operation_id, source_revision, task_count, next_ordinal
                        )
                    return PublicationStep("complete", 0, None)

                published = 0
                processed = 0
                while next_ordinal < task_count and processed < max_members:
                    processed += 1
                    try:
                        row = reader.read_row(next_ordinal, sequence_limit=self._group_tail_value())
                    except SequenceBeyondLimit:
                        return PublicationStep("waiting", published, "pending_group_finalization")
                    slot_path = self.directory / "members" / f"{row.sequence}.json"
                    existing = self._read_optional_record(slot_path)
                    expected_slot = self._slot_record(
                        identity,
                        operation_id,
                        source_revision,
                        reader.spool,
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

                state = "complete" if next_ordinal == task_count else "progressed"
                return PublicationStep(state, published, None)
        except _ReaderWaiting as exc:
            return PublicationStep("waiting", 0, str(exc))

    def _slot_record(
        self,
        identity: dict[str, Any],
        operation_id: str,
        source_revision: SourceRevision,
        spool: Path,
        row: MemberRow,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "identity": identity,
            "sequence": row.sequence,
            "operation_id": operation_id,
            "ordinal": row.ordinal,
            "source_revision": _revision_dict(source_revision),
            "spool": _relative_locator(spool, self.directory),
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


def _require_regular_path(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return info


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


__all__ = ["CoverageStatus", "GroupCoverage", "MemberIdentity", "PublicationStep"]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
