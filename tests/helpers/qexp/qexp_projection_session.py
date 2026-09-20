"""Durable, provisional qexp source scanning and projection checkpoints.

This Linux-qualified test oracle composes the bounded source binding, lexical
scanner, and structural submission projection into one restartable session.
Its JSONL events remain provisional output and are not a membership
certificate.  It does not provide uniqueness proof, complete shared-slice
accounting, authority publication, or cross-host/power-loss guarantees.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from typing import BinaryIO, Callable

from qqtools.plugins.qexp.runtime.group_discovery.checkpoint import (
    _CHECKPOINT_VERSION,
    _encode_json,
    _projection_context,
    _reject_duplicate_pairs,
    _reject_json_constant,
    _revision_dict,
    _snapshot_offset,
    _validate_cross_consistency,
    _validate_envelope,
)
from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner, StepResult
from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource, SourceChangedError, SourceRevision
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import (
    SUPPORTED_SOURCE_SCHEMA_VERSION,
    FieldChunk,
    FieldEnd,
    ProjectionSummary,
    SubmissionProjection,
)

_MAX_IO_CHUNK = 65_536
_SPOOL_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_SPOOL_RESUME_FLAGS = os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_CHECKPOINT_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_CHECKPOINT_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


class ProjectionSession:
    """Run a bounded source projection with an append-only durable checkpoint.

    Use :meth:`create` for a new scratch directory or :meth:`resume` for a
    previously checkpointed directory.  Events and summaries are provisional;
    a successful session verifies its source and pinned schema marker again
    before exposing completion.
    """

    __slots__ = (
        "_source",
        "_source_path_name",
        "_spool",
        "_spool_path",
        "_spool_device_value",
        "_spool_inode_value",
        "_spool_size",
        "_spool_events",
        "_directory_fd",
        "_scratch",
        "_checkpoint_path",
        "_checkpoint_temp_path",
        "_scanner",
        "_projection",
        "_hook",
        "_summary",
        "_is_complete",
        "_closed",
        "_busy",
    )

    @classmethod
    def create(
        cls,
        source: Path,
        scratch: Path,
        expected_operation_id: str,
        expected_group: str,
        *,
        hook: Callable[[str], None] | None = None,
    ) -> "ProjectionSession":
        """Create a new session and establish its initial durable checkpoint."""

        _validate_hook(hook)
        source_path = _lexical_absolute(source)
        scratch_path = _lexical_absolute(scratch)
        bound_source: BoundSource | None = None
        spool: BinaryIO | None = None
        directory_fd: int | None = None
        try:
            bound_source = BoundSource.open(source_path)
            scratch_path.mkdir()
            spool, spool_stat = _open_spool_for_create(scratch_path / "events.jsonl")
            directory_fd = _open_directory(scratch_path)
            os.fsync(directory_fd)

            session = cls.__new__(cls)
            session._initialize_fields(
                bound_source,
                spool,
                scratch_path / "events.jsonl",
                source_path,
                spool_stat.st_dev,
                spool_stat.st_ino,
                0,
                0,
                directory_fd,
                scratch_path,
                hook,
            )
            projection = SubmissionProjection(
                expected_operation_id,
                expected_group,
                session._emit_chunk,
                session._emit_end,
            )
            session._projection = projection
            session._scanner = Scanner(
                bound_source,
                lambda _span: None,
                chunk_bytes=_MAX_IO_CHUNK,
                emit_bytes=projection.feed,
            )
            session.checkpoint()
            bound_source = None
            spool = None
            directory_fd = None
            return session
        except BaseException:
            if spool is not None:
                _close_handle(spool)
            if directory_fd is not None:
                _close_fd(directory_fd)
            if bound_source is not None:
                bound_source.close()
            raise

    @classmethod
    def resume(
        cls,
        source: Path,
        scratch: Path,
        expected_operation_id: str,
        expected_group: str,
        *,
        hook: Callable[[str], None] | None = None,
    ) -> "ProjectionSession":
        """Resume exactly the durable prefix recorded in ``scratch``."""

        _validate_hook(hook)
        source_path = _lexical_absolute(source)
        scratch_path = _lexical_absolute(scratch)
        envelope = _read_checkpoint(scratch_path / "checkpoint.json")
        checkpoint = _validate_envelope(envelope, source_path, expected_operation_id, expected_group)
        source_revision = checkpoint["source_revision"]
        bound_source: BoundSource | None = None
        spool: BinaryIO | None = None
        directory_fd: int | None = None
        try:
            bound_source = BoundSource.open(source_path, expected_revision=source_revision)
            scanner_snapshot = checkpoint["scanner"]
            projection_snapshot = checkpoint["projection"]
            scanner_offset = _snapshot_offset(scanner_snapshot)
            bound_source.seek(scanner_offset)

            session = cls.__new__(cls)
            projection = SubmissionProjection.from_snapshot(
                expected_operation_id,
                expected_group,
                session._emit_chunk,
                session._emit_end,
                projection_snapshot,
            )
            if projection_snapshot["summary"] is not None and (
                projection.source_schema_version != SUPPORTED_SOURCE_SCHEMA_VERSION
            ):
                raise ValueError("unsupported or missing source schema version")

            directory_fd = _open_directory(scratch_path)
            spool_path = scratch_path / "events.jsonl"
            spool, spool_stat = _open_spool_for_resume(spool_path, checkpoint["spool"])
            saved_size = checkpoint["spool"]["size"]
            session._initialize_fields(
                bound_source,
                spool,
                spool_path,
                source_path,
                spool_stat.st_dev,
                spool_stat.st_ino,
                saved_size,
                checkpoint["spool"]["events"],
                directory_fd,
                scratch_path,
                hook,
            )
            scanner = Scanner.from_snapshot(
                bound_source,
                lambda _span: None,
                scanner_snapshot,
                emit_bytes=projection.feed,
            )
            scanner_snapshot = scanner.snapshot()
            projection_snapshot = projection.snapshot()
            _validate_cross_consistency(scanner_snapshot, projection_snapshot, bound_source.revision)
            summary = projection.finish() if projection_snapshot["summary"] is not None else None
            os.ftruncate(spool.fileno(), saved_size)
            spool.seek(saved_size, os.SEEK_SET)
            session._summary = summary
            session._is_complete = summary is not None
            session._scanner = scanner
            session._projection = projection
            bound_source = None
            spool = None
            directory_fd = None
            return session
        except BaseException:
            if spool is not None:
                _close_handle(spool)
            if directory_fd is not None:
                _close_fd(directory_fd)
            if bound_source is not None:
                bound_source.close()
            raise

    def _initialize_fields(
        self,
        source: BoundSource,
        spool: BinaryIO,
        spool_path: Path,
        source_path: Path,
        spool_device: int,
        spool_inode: int,
        spool_size: int,
        spool_events: int,
        directory_fd: int,
        scratch: Path,
        hook: Callable[[str], None] | None,
    ) -> None:
        self._source = source
        self._source_path_name = source_path
        self._spool = spool
        self._spool_path = spool_path
        self._spool_device_value = spool_device
        self._spool_inode_value = spool_inode
        self._spool_size = spool_size
        self._spool_events = spool_events
        self._directory_fd = directory_fd
        self._scratch = scratch
        self._checkpoint_path = scratch / "checkpoint.json"
        self._checkpoint_temp_path = scratch / "checkpoint.tmp"
        self._hook = hook
        self._summary = None
        self._is_complete = False
        self._closed = False
        self._busy = False

    @property
    def is_complete(self) -> bool:
        """Whether lexical EOF, structural validation, and final source verify passed."""

        return self._is_complete

    @property
    def summary(self) -> ProjectionSummary | None:
        """Return the provisional summary after successful completion, if any."""

        return self._summary

    def __enter__(self) -> "ProjectionSession":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close owned descriptors without creating an implicit checkpoint."""

        if self._closed:
            return
        try:
            _close_handle(self._spool)
            _close_fd(self._directory_fd)
            self._source.close()
        finally:
            self._closed = True

    def step(self, max_bytes: int = _MAX_IO_CHUNK, *, max_fragments: int = 64) -> StepResult:
        """Advance lexical and structural processing within the requested byte budget."""

        self._begin_operation()
        try:
            result = self._scanner.step(max_bytes, max_fragments=max_fragments)
            if result.is_complete and not self._is_complete:
                summary = self._projection.finish()
                if self._projection.source_schema_version != SUPPORTED_SOURCE_SCHEMA_VERSION:
                    raise ValueError("unsupported or missing source schema version")
                self._source.verify()
                self._summary = summary
                self._is_complete = True
            return result
        except BaseException:
            self._poison()
            raise
        finally:
            self._busy = False

    def checkpoint(self) -> None:
        """Durably publish a coherent source, parser, and spool prefix checkpoint."""

        self._begin_operation()
        try:
            self._source.verify()
            spool_stat = self._verify_spool(self._spool_size, exact_size=True)
            os.fsync(self._spool.fileno())
            self._call_hook("spool_fsync")

            scanner_snapshot = self._scanner.snapshot()
            projection_snapshot = self._projection.snapshot()
            _validate_cross_consistency(scanner_snapshot, projection_snapshot, self._source.revision)
            envelope = {
                "version": _CHECKPOINT_VERSION,
                "source_path": str(self._source_path_name),
                "source_revision": _revision_dict(self._source.revision),
                "expected_operation_id": _projection_context(projection_snapshot, "expected_operation_id"),
                "expected_group": _projection_context(projection_snapshot, "expected_group"),
                "spool": {
                    "device": spool_stat.st_dev,
                    "inode": spool_stat.st_ino,
                    "size": self._spool_size,
                    "events": self._spool_events,
                },
                "scanner": scanner_snapshot,
                "projection": projection_snapshot,
            }
            checkpoint_bytes = _encode_json(envelope)
            self._write_checkpoint_temp(checkpoint_bytes)
            self._source.verify()
            os.replace(self._checkpoint_temp_path, self._checkpoint_path)
            self._call_hook("checkpoint_replace")
            os.fsync(self._directory_fd)
            self._call_hook("directory_fsync")
        except BaseException:
            self._poison()
            raise
        finally:
            self._busy = False

    def _begin_operation(self) -> None:
        self._require_open()
        if self._busy:
            raise RuntimeError("projection session operation is busy")
        self._busy = True

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("projection session is closed")

    def _poison(self) -> None:
        self._summary = None
        self._is_complete = False
        self.close()

    def _emit_chunk(self, chunk: FieldChunk) -> None:
        event = {
            "type": "chunk",
            "kind": chunk.kind,
            "ordinal": chunk.ordinal,
            "data_b64": base64.b64encode(chunk.data).decode("ascii"),
            "is_final": chunk.is_final,
        }
        self._write_event(event)

    def _emit_end(self, end: FieldEnd) -> None:
        event = {
            "type": "end",
            "kind": end.kind,
            "ordinal": end.ordinal,
            "digest": end.digest,
            "decoded_size": end.decoded_size,
            "start": end.start,
            "end": end.end,
        }
        self._write_event(event)

    def _write_event(self, event: dict[str, object]) -> None:
        encoded = _encode_json(event) + b"\n"
        _write_bounded(self._spool, encoded, self._call_hook, "spool_write")
        self._spool_size += len(encoded)
        self._spool_events += 1

    def _call_hook(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)

    def _verify_spool(self, expected_size: int, *, exact_size: bool) -> os.stat_result:
        first = os.fstat(self._spool.fileno())
        _require_regular(first.st_mode, self._spool_path, "spool descriptor")
        named = os.lstat(self._spool_path)
        _require_regular(named.st_mode, self._spool_path, "spool path")
        if (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("spool path was replaced")
        if (first.st_dev, first.st_ino) != (self._spool_device_value, self._spool_inode_value):
            raise ValueError("spool identity does not match its checkpoint")
        if exact_size:
            if first.st_size != expected_size:
                raise ValueError("spool size changed before checkpoint")
        elif first.st_size < expected_size:
            raise ValueError("spool is shorter than its checkpoint")
        second = os.fstat(self._spool.fileno())
        _require_regular(second.st_mode, self._spool_path, "spool descriptor")
        if (second.st_dev, second.st_ino) != (first.st_dev, first.st_ino):
            raise ValueError("spool descriptor changed")
        if exact_size and second.st_size != expected_size:
            raise ValueError("spool size changed before checkpoint")
        if not exact_size and second.st_size < expected_size:
            raise ValueError("spool is shorter than its checkpoint")
        return second

    def _write_checkpoint_temp(self, payload: bytes) -> None:
        _remove_stale_temp(self._checkpoint_temp_path)
        descriptor: int | None = None
        handle: BinaryIO | None = None
        try:
            descriptor = os.open(self._checkpoint_temp_path, _CHECKPOINT_TEMP_FLAGS, 0o600)
            first = os.fstat(descriptor)
            _require_regular(first.st_mode, self._checkpoint_temp_path, "checkpoint temp")
            named = os.lstat(self._checkpoint_temp_path)
            _require_regular(named.st_mode, self._checkpoint_temp_path, "checkpoint temp")
            if (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
                raise ValueError("checkpoint temp was replaced")
            handle = os.fdopen(descriptor, "wb", buffering=0)
            descriptor = None
            _write_bounded(handle, payload, self._call_hook, "checkpoint_write")
            os.fsync(handle.fileno())
            self._call_hook("checkpoint_fsync")
        finally:
            if handle is not None:
                _close_handle(handle)
            if descriptor is not None:
                _close_fd(descriptor)


def _lexical_absolute(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _validate_hook(hook: Callable[[str], None] | None) -> None:
    if hook is not None and not callable(hook):
        raise TypeError("hook must be callable or None")


def _open_directory(path: Path) -> int:
    descriptor = os.open(path, _DIRECTORY_FLAGS)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"scratch is not a directory: {path}")
        return descriptor
    except BaseException:
        _close_fd(descriptor)
        raise


def _open_spool_for_create(path: Path) -> tuple[BinaryIO, os.stat_result]:
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(path, _SPOOL_CREATE_FLAGS, 0o600)
        info = os.fstat(descriptor)
        _require_regular(info.st_mode, path, "spool descriptor")
        handle = os.fdopen(descriptor, "wb", buffering=0)
        descriptor = None
        return handle, info
    except BaseException:
        if handle is not None:
            _close_handle(handle)
        if descriptor is not None:
            _close_fd(descriptor)
        raise


def _open_spool_for_resume(path: Path, checkpoint: dict[str, object]) -> tuple[BinaryIO, os.stat_result]:
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(path, _SPOOL_RESUME_FLAGS)
        first = os.fstat(descriptor)
        _require_regular(first.st_mode, path, "spool descriptor")
        if (first.st_dev, first.st_ino) != (checkpoint["device"], checkpoint["inode"]):
            raise ValueError("spool identity does not match its checkpoint")
        named = os.lstat(path)
        _require_regular(named.st_mode, path, "spool path")
        if (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("spool path was replaced")
        if first.st_size < checkpoint["size"]:
            raise ValueError("spool is shorter than its checkpoint")
        second = os.fstat(descriptor)
        if (second.st_dev, second.st_ino) != (first.st_dev, first.st_ino) or second.st_size < checkpoint["size"]:
            raise ValueError("spool changed while opening")
        handle = os.fdopen(descriptor, "wb", buffering=0)
        descriptor = None
        return handle, second
    except BaseException:
        if handle is not None:
            _close_handle(handle)
        if descriptor is not None:
            _close_fd(descriptor)
        raise


def _read_checkpoint(path: Path) -> dict[str, object]:
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(path, _CHECKPOINT_READ_FLAGS)
        first = os.fstat(descriptor)
        _require_regular(first.st_mode, path, "checkpoint")
        named = os.lstat(path)
        _require_regular(named.st_mode, path, "checkpoint")
        if (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("checkpoint was replaced")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None
        data = bytearray()
        while True:
            chunk = handle.read(_MAX_IO_CHUNK)
            if chunk is None:
                raise OSError("checkpoint read returned no data")
            if not chunk:
                break
            data.extend(chunk)
        return json.loads(
            bytes(data),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("checkpoint is not valid UTF-8 JSON") from exc
    finally:
        if handle is not None:
            _close_handle(handle)
        if descriptor is not None:
            _close_fd(descriptor)


def _write_bounded(
    handle: BinaryIO,
    payload: bytes,
    hook: Callable[[str], None],
    hook_name: str,
) -> None:
    offset = 0
    while offset < len(payload):
        end = min(offset + _MAX_IO_CHUNK, len(payload))
        written = handle.write(payload[offset:end])
        if type(written) is not int or written <= 0 or written > end - offset:
            raise OSError("short or invalid bounded write")
        offset += written
        hook(hook_name)


def _remove_stale_temp(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        raise ValueError("checkpoint temp path is a directory")
    os.unlink(path)


def _require_regular(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _close_handle(handle: BinaryIO) -> None:
    try:
        handle.close()
    except OSError:
        pass


def _close_fd(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


__all__ = ["ProjectionSession", "ProjectionSummary", "StepResult", "SourceChangedError", "SourceRevision"]
