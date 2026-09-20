"""Interpret confirmed Group member artifacts outside membership policy."""

from __future__ import annotations

import base64
import binascii
import json
import os
import stat
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from ..paths import submission_path
from ..records import validate_identifier
from .fingerprint import ChainedDigest
from .receipt import decode_receipt
from .source_revision import SourceRevision

_ROW_BYTES = 32
_MAX_LINE_BYTES = 90_000
_MAX_RECEIPT_BYTES = 16_384
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_U64_MAX = (1 << 64) - 1
_EVENT_CHUNK_KEYS = frozenset({"type", "kind", "ordinal", "data_b64", "is_final"})
_EVENT_END_KEYS = frozenset({"type", "kind", "ordinal", "digest", "decoded_size", "start", "end"})
_IDENTIFIER_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_REVISION_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})


@dataclass(frozen=True, slots=True)
class MemberRow:
    """One decoded task/sequence reference row."""

    ordinal: int
    task_start: int
    task_end: int
    task_size: int
    sequence: int


@dataclass(frozen=True, slots=True)
class MemberIdentity:
    """A certified Group member locator returned by direct lookup."""

    sequence: int
    task_id: str
    operation_id: str


class SequenceBeyondLimit(ValueError):
    """The canonical sequence is numerically beyond the supplied bound."""


class _ReaderWaiting(Exception):
    """A confirmed-source condition that coverage reports as waiting."""


@dataclass(frozen=True, slots=True)
class _SourceFiles:
    spool: Path
    task_references: Path
    sequence_references: Path
    source_root: Path


class _ResourceOwner:
    """Attempt each registered cleanup callback exactly once."""

    __slots__ = ("_callbacks",)

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], object]] = []

    def add(self, callback: Callable[[], object]) -> None:
        self._callbacks.append(callback)

    def cleanup(self) -> list[BaseException]:
        callbacks = self._callbacks
        self._callbacks = []
        failures: list[BaseException] = []
        for callback in callbacks:
            try:
                callback()
            except BaseException as exc:
                failures.append(exc)
        return failures


class PublicationReader:
    """Read bounded rows from one validated confirmed source."""

    __slots__ = (
        "_operation_id",
        "_source_revision",
        "_task_count",
        "_spool_path",
        "_task_fd",
        "_sequence_fd",
        "_spool",
        "_spool_revision",
        "_closed",
    )

    def __init__(
        self,
        operation_id: str,
        source_revision: SourceRevision,
        task_count: int,
        spool_path: Path,
        task_fd: int,
        sequence_fd: int,
        spool: BinaryIO,
        spool_revision: SourceRevision,
    ) -> None:
        self._operation_id = operation_id
        self._source_revision = source_revision
        self._task_count = task_count
        self._spool_path = spool_path
        self._task_fd = task_fd
        self._sequence_fd = sequence_fd
        self._spool = spool
        self._spool_revision = spool_revision
        self._closed = False

    @property
    def operation_id(self) -> str:
        """Return the confirmed source operation identifier."""

        return self._operation_id

    @property
    def source_revision(self) -> SourceRevision:
        """Return the confirmed source revision."""

        return self._source_revision

    @property
    def task_count(self) -> int:
        """Return the confirmed task count."""

        return self._task_count

    @property
    def spool(self) -> Path:
        """Return the validated event-spool path."""

        return self._spool_path

    def _mark_closed(self) -> None:
        self._closed = True

    def read_row(self, ordinal: int, *, sequence_limit: int) -> MemberRow:
        """Read and decode one reference row under a Group sequence bound."""

        if self._closed:
            raise ValueError("publication reader is closed")
        _validate_ordinal(ordinal, self._task_count)
        _validate_sequence_limit(sequence_limit)
        spool_revision = _verify_descriptor_revision(
            self._spool.fileno(), self._spool_path, self._spool_revision, "spool"
        )
        task_record = _read_reference(self._task_fd, ordinal, "task")
        sequence_record = _read_reference(self._sequence_fd, ordinal, "sequence")
        task_start, task_end, task_size = _validate_reference(task_record, spool_revision.size, "task")
        sequence_start, sequence_end, sequence_size = _validate_reference(
            sequence_record, spool_revision.size, "sequence"
        )
        sequence = _decode_sequence(
            self._spool,
            sequence_start,
            sequence_end,
            sequence_size,
            ordinal,
            sequence_limit,
            self._source_revision,
        )
        _verify_descriptor_revision(self._spool.fileno(), self._spool_path, spool_revision, "spool")
        return MemberRow(ordinal, task_start, task_end, task_size, sequence)


@contextmanager
def open_publication_reader(
    source: Any,
    *,
    group: str,
    coverage_directory: Path,
) -> Iterator[PublicationReader]:
    """Validate and open one confirmed source's publication artifacts."""

    owner = _ResourceOwner()
    primary: BaseException | None = None
    try:
        reader = _open_publication_reader(source, group, coverage_directory, owner)
        yield reader
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _finish_cleanup(owner.cleanup(), primary, "publication reader cleanup failed")


def read_published_member(
    shared_root: Path,
    *,
    coverage_directory: Path,
    group: str,
    slot: dict[str, Any],
) -> MemberIdentity:
    """Verify one stored slot and return its task identity."""

    owner = _ResourceOwner()
    primary: BaseException | None = None
    try:
        result = _read_published_member(shared_root, coverage_directory, group, slot, owner)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _finish_cleanup(owner.cleanup(), primary, "publication reader cleanup failed")
    return result


def _open_publication_reader(
    source: Any,
    group: str,
    coverage_directory: Path,
    owner: _ResourceOwner,
) -> PublicationReader:
    if source is None:
        raise TypeError("source must be a ConfirmedSource")
    if getattr(source, "status", None) != "qualified":
        raise _ReaderWaiting("source_not_qualified")
    operation_id = _validate_operation_id(getattr(source, "operation_id", None))
    if getattr(source, "group", None) != group:
        raise _ReaderWaiting("wrong_group")
    source_revision = getattr(source, "source_revision", None)
    if not isinstance(source_revision, SourceRevision):
        raise ValueError("confirmed source has an invalid source revision")
    task_count = getattr(source, "task_count", None)
    if type(task_count) is not int or task_count < 0:
        raise ValueError("confirmed source task_count must be a nonnegative integer")

    directory = _normalized_absolute(coverage_directory, "coverage directory")
    source_path = _as_path(getattr(source, "source", None), "source")
    _verify_source_revision(source_path, source_revision)
    spool = _as_path(getattr(source, "spool", None), "spool")
    task_references = _as_path(getattr(source, "task_references", None), "task_references")
    sequence_references = _as_path(getattr(source, "sequence_references", None), "sequence_references")
    source_root = _as_path(spool.parent.parent, "source scratch")
    files = _SourceFiles(spool, task_references, sequence_references, source_root)

    for path, label in (
        (files.spool, "spool"),
        (files.task_references, "task references"),
        (files.sequence_references, "sequence references"),
    ):
        _validate_locator(path, directory, operation_id, label)
    _validate_directory_locator(source_root, directory, operation_id)
    receipt = source_root / "receipt.json"
    _require_regular_path(receipt, "source receipt")
    if source_root != spool.parent.parent:
        raise ValueError("confirmed source spool does not identify its scratch root")

    expected_size = task_count * _ROW_BYTES
    task_fd = _open_regular(files.task_references, "task references")
    owner.add(lambda descriptor=task_fd: os.close(descriptor))
    if os.fstat(task_fd).st_size != expected_size:
        raise ValueError("task reference size does not match confirmed task_count")
    sequence_fd = _open_regular(files.sequence_references, "sequence references")
    owner.add(lambda descriptor=sequence_fd: os.close(descriptor))
    if os.fstat(sequence_fd).st_size != expected_size:
        raise ValueError("sequence reference size does not match confirmed task_count")
    spool_fd = _open_regular(files.spool, "spool")
    try:
        spool = os.fdopen(spool_fd, "rb", buffering=0)
    except BaseException as exc:
        _close_after_failure(spool_fd, exc, "spool")
        raise
    owner.add(spool.close)
    spool_revision = _verify_descriptor_revision(spool.fileno(), files.spool, None, "spool")
    reader = PublicationReader(
        operation_id,
        source_revision,
        task_count,
        files.spool,
        task_fd,
        sequence_fd,
        spool,
        spool_revision,
    )
    owner.add(reader._mark_closed)
    return reader


def _read_published_member(
    shared_root: Path,
    coverage_directory: Path,
    group: str,
    slot: dict[str, Any],
    owner: _ResourceOwner,
) -> MemberIdentity:
    root = _as_path(shared_root, "shared root")
    directory = _normalized_absolute(coverage_directory, "coverage directory")
    if type(slot) is not dict:
        raise ValueError("member slot is malformed")
    if (
        frozenset(slot)
        != frozenset(
            {"version", "identity", "sequence", "operation_id", "ordinal", "source_revision", "spool", "task_ref"}
        )
        or slot.get("version") != 1
    ):
        raise ValueError("member slot is malformed")
    try:
        operation_id = _validate_operation_id(slot["operation_id"])
        source_revision = _revision_from_dict(slot["source_revision"])
        spool_locator = slot["spool"]
        ordinal = slot["ordinal"]
        slot_sequence = slot["sequence"]
        task_ref = slot["task_ref"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("member slot is malformed") from exc
    if type(slot_sequence) is not int or not 1 <= slot_sequence <= _U64_MAX:
        raise ValueError("member slot sequence is invalid")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("member slot ordinal is invalid")
    if type(task_ref) is not dict or frozenset(task_ref) != frozenset({"start", "end", "size"}):
        raise ValueError("member slot task reference is invalid")
    task_start, task_end, task_size = task_ref["start"], task_ref["end"], task_ref["size"]
    if (
        type(task_start) is not int
        or type(task_end) is not int
        or type(task_size) is not int
        or not 0 <= task_start < task_end
        or not 0 <= task_size <= _U64_MAX
    ):
        raise ValueError("member slot task reference is invalid")

    spool_path = _validate_relative_locator(spool_locator, directory, operation_id)
    if spool_path.parts[-2:] != ("projection", "events.jsonl"):
        raise ValueError("member slot spool must be projection/events.jsonl")
    scratch = spool_path.parent.parent

    receipt_path = scratch / "receipt.json"
    receipt_info = _require_regular_path(receipt_path, "source receipt")
    receipt_revision = SourceRevision.from_stat(receipt_info)
    receipt_data, receipt_fd = _read_bounded_bytes_owned(
        receipt_path,
        _MAX_RECEIPT_BYTES,
        "source receipt",
        owner,
        receipt_revision,
    )
    receipt = decode_receipt(receipt_data, submission_path(root, operation_id), operation_id, group, scratch)
    if receipt.get("status") != "qualified":
        raise ValueError("member source receipt is not qualified")
    receipt_source_revision = receipt.get("source_revision")
    if not isinstance(receipt_source_revision, SourceRevision) or receipt_source_revision != source_revision:
        raise ValueError("member source revision does not match its receipt")

    task_count = receipt.get("task_count")
    if type(task_count) is not int or ordinal >= task_count:
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
    _validate_locator(task_path, directory, operation_id, "task references")
    sequence_record = receipt.get("sequence_references")
    if type(sequence_record) is not dict or sequence_record.get("path") != "qualification/sequence.refs":
        raise ValueError("qualified member receipt is missing sequence references")
    sequence_revision = _receipt_file_revision(sequence_record, "sequence references")
    sequence_path = scratch / sequence_record["path"]
    _validate_locator(sequence_path, directory, operation_id, "sequence references")

    if task_size < 1 or not 0 <= task_start < task_end <= spool_revision.size:
        raise ValueError("member task reference span is invalid")

    source_path = submission_path(root, operation_id)
    source_fd = _open_expected_revision(source_path, receipt_source_revision, "source")
    owner.add(lambda descriptor=source_fd: os.close(descriptor))
    task_fd = _open_expected_revision(task_path, task_revision, "task references")
    owner.add(lambda descriptor=task_fd: os.close(descriptor))
    sequence_fd = _open_expected_revision(sequence_path, sequence_revision, "sequence references")
    owner.add(lambda descriptor=sequence_fd: os.close(descriptor))
    spool_fd = _open_expected_revision(spool_path, spool_revision, "spool")
    try:
        spool = os.fdopen(spool_fd, "rb", buffering=65_536)
    except BaseException as exc:
        _close_after_failure(spool_fd, exc, "spool")
        raise
    owner.add(spool.close)

    _verify_descriptor_revision(source_fd, source_path, receipt_source_revision, "source")
    _verify_descriptor_revision(task_fd, task_path, task_revision, "task references")
    _verify_descriptor_revision(sequence_fd, sequence_path, sequence_revision, "sequence references")
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
        sequence_reference, spool_revision.size, "sequence"
    )
    try:
        sequence = _decode_sequence(
            spool,
            sequence_start,
            sequence_end,
            sequence_size,
            ordinal,
            slot_sequence,
            receipt_source_revision,
        )
    except SequenceBeyondLimit as exc:
        raise ValueError("member sequence does not match its qualified reference") from exc
    if sequence != slot_sequence:
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
    _verify_descriptor_revision(receipt_fd, receipt_path, receipt_revision, "source receipt")
    _verify_source_revision(receipt_path, receipt_revision)
    return MemberIdentity(slot_sequence, task_id, operation_id)


def _finish_cleanup(
    failures: list[BaseException],
    primary: BaseException | None,
    label: str,
) -> None:
    if not failures:
        return
    if primary is not None:
        primary.add_note(_cleanup_note(label, failures))
        return
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(label, failures)


def _cleanup_note(label: str, failures: list[BaseException]) -> str:
    details = "; ".join(f"{type(error).__name__}: {error}" for error in failures)
    return f"{label}: {details}"


def _close_after_failure(descriptor: int, primary: BaseException, label: str) -> None:
    try:
        os.close(descriptor)
    except BaseException as exc:
        primary.add_note(f"{label} cleanup failed: {type(exc).__name__}: {exc}")


def _validate_ordinal(value: object, task_count: int) -> int:
    if type(value) is not int or value < 0 or value >= task_count:
        raise ValueError("ordinal must be a nonnegative integer within task_count")
    return value


def _validate_sequence_limit(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("sequence_limit must be a nonnegative exact integer")
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


def _normalized_absolute(value: Path, label: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise TypeError(f"{label} must be a path")
    raw = os.fspath(value)
    if isinstance(raw, bytes):
        raise TypeError(f"{label} must be a text path")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    normalized = Path(os.path.normpath(str(path)))
    if path != normalized:
        raise ValueError(f"{label} must be normalized")
    return path


def _revision_from_dict(value: Any) -> SourceRevision:
    if type(value) is not dict or frozenset(value) != _REVISION_KEYS:
        raise ValueError("source revision record is malformed")
    return SourceRevision(**{key: value[key] for key in _REVISION_KEYS})


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
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _READ_FLAGS)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening: {path}")
        after = _require_regular_path(path, label)
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"{label} changed while opening: {path}")
        return descriptor
    except BaseException as exc:
        if descriptor is not None:
            _close_after_failure(descriptor, exc, label)
        raise


def _open_expected_revision(path: Path, expected: SourceRevision, label: str) -> int:
    descriptor = _open_regular(path, label)
    try:
        _verify_descriptor_revision(descriptor, path, expected, label)
        return descriptor
    except BaseException as exc:
        _close_after_failure(descriptor, exc, label)
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


def _read_bounded_bytes_owned(
    path: Path,
    max_bytes: int,
    label: str,
    owner: _ResourceOwner,
    expected: SourceRevision | None,
) -> tuple[bytes, int]:
    descriptor = _open_regular(path, label)
    owner.add(lambda descriptor=descriptor: os.close(descriptor))
    revision = _verify_descriptor_revision(descriptor, path, expected, label)
    payload = bytearray()
    while len(payload) <= max_bytes:
        chunk = os.read(descriptor, max_bytes + 1 - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise ValueError(f"{label} exceeds its {max_bytes}-byte limit: {path}")
    _verify_descriptor_revision(descriptor, path, revision, label)
    return bytes(payload), descriptor


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
    except (OSError, OverflowError) as exc:
        raise ValueError(f"could not read {label} reference {ordinal}") from exc
    if len(payload) != _ROW_BYTES:
        raise ValueError(f"{label} reference {ordinal} is truncated")
    values = struct.unpack(">QQQQ", payload)
    if values[0] != ordinal:
        raise ValueError(f"{label} reference ordinal is not exact")
    return values


def _validate_reference(values: tuple[int, int, int, int], spool_size: int, label: str) -> tuple[int, int, int]:
    _, start, end, size = values
    if not 0 <= start < end <= spool_size or type(size) is not int or size < 1:
        raise ValueError(f"{label} reference span is invalid")
    return start, end, size


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


def _decode_sequence(
    spool: BinaryIO,
    start: int,
    end: int,
    expected_size: int,
    ordinal: int,
    sequence_limit: int,
    source_revision: SourceRevision,
) -> int:
    limit_digits = _decimal_text(sequence_limit)
    position = start
    decoded_size = 0
    comparison = 0
    bounded_digits = bytearray()
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
        try:
            event = _decode_event_line(line)
        except RecursionError as exc:
            raise ValueError("sequence spool line is too deeply nested") from exc
        event_type = event.get("type")
        if event_type == "chunk":
            if frozenset(event) != _EVENT_CHUNK_KEYS:
                raise ValueError("sequence chunk event shape is invalid")
            if event["kind"] != "sequence" or type(event["ordinal"]) is not int or event["ordinal"] != ordinal:
                raise ValueError("sequence chunk event identity is invalid")
            if type(event["is_final"]) is not bool or final_seen or end_seen:
                raise ValueError("sequence chunk final marker is invalid")
            data = _decode_chunk(event["data_b64"])
            if not data and not event["is_final"]:
                raise ValueError("empty sequence chunk is not final")
            for byte in data:
                if (first_chunk and byte not in b"123456789") or (not first_chunk and byte not in b"0123456789"):
                    raise ValueError("sequence chunk is not canonical decimal")
                digit_position = decoded_size
                if digit_position < len(limit_digits):
                    if comparison == 0:
                        if byte > ord(limit_digits[digit_position]):
                            comparison = 1
                        elif byte < ord(limit_digits[digit_position]):
                            comparison = -1
                    bounded_digits.append(byte)
                decoded_size += 1
                first_chunk = False
            final_seen = event["is_final"]
        elif event_type == "end":
            if frozenset(event) != _EVENT_END_KEYS:
                raise ValueError("sequence end event shape is invalid")
            if event["kind"] != "sequence" or type(event["ordinal"]) is not int or event["ordinal"] != ordinal:
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
    if decoded_size > len(limit_digits) or (decoded_size == len(limit_digits) and comparison > 0):
        raise SequenceBeyondLimit
    value = 0
    for byte in bounded_digits:
        value = value * 10 + byte - 48
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


def _digest_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MemberIdentity",
    "MemberRow",
    "PublicationReader",
    "SequenceBeyondLimit",
    "open_publication_reader",
    "read_published_member",
]
