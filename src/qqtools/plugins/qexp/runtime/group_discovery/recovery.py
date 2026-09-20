"""Restartable whole-source confirmation for Group discovery.

ProjectionDriver owns parser checkpoints and SourceQualification owns the
provisional qualification pass.  This module is the durable boundary between
those two pieces: a completed receipt is the only thing that makes a
confirmed source reusable after a process restart.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

from .driver import ProjectionDriver
from .receipt import decode_receipt
from .slice_io import YIELD, SliceIO
from .source_qualification import QualifiedSource, SourceQualification
from .source_revision import SourceChangedError, SourceRevision
from .submission_projection import SubmissionProjection

_MAX_IO_CHUNK = 65_536
_MAX_RECEIPT_BYTES = 16_384
_CHECKPOINT_INTERVAL = 1 << 20
_MAX_EVENTS = 64

_LOCK_FLAGS = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK

_STAT_ORDER = ("device", "inode", "size", "mtime_ns", "ctime_ns")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class RecoveryStep:
    """The bounded result of one cooperative confirmation advance."""

    state: Literal["progressed", "waiting", "complete", "closed"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class ConfirmedSource:
    """An immutable, receipt-backed whole-source confirmation."""

    status: str
    source: Path
    source_revision: SourceRevision
    spool: Path
    task_references: Path | None
    sequence_references: Path | None
    operation_id: str
    group: str
    task_count: int


@dataclass(frozen=True, slots=True)
class _IORequest:
    method: str
    args: tuple[object, ...]
    owner_attr: str | None = None


@dataclass(frozen=True, slots=True)
class _ChildRequest:
    kind: Literal["projection", "qualification"]


@dataclass(frozen=True, slots=True)
class _Boundary:
    state: Literal["progressed", "waiting", "complete"]
    reason: str | None = None


@dataclass(slots=True)
class _CleanFrame:
    path: Path
    iterator: os.ScandirIterator | None = None


class _StopRequested(Exception):
    pass


class RecoverableSource:
    """Confirm one source through bounded, restartable filesystem work.

    Construction is pure. Projection and qualification directories, the
    receipt, and all descriptors are created only from advance.
    """

    def __init__(self, source: Path, scratch: Path, operation_id: str, group: str) -> None:
        self._source = _normalized_absolute(source, "source")
        self._scratch = _normalized_absolute(scratch, "scratch")
        try:
            self._source.relative_to(self._scratch)
        except ValueError:
            pass
        else:
            raise ValueError("source must not be inside scratch")

        # Reuse the established projection context validation without doing
        # any filesystem work.
        context = SubmissionProjection(operation_id, group, lambda _value: None, lambda _value: None)
        self._operation_id = context._expected_operation_id
        self._group = context._expected_group

        self._projection_path = self._scratch / "projection"
        self._qualification_path = self._scratch / "qualification"
        self._receipt_path = self._scratch / "receipt.json"
        self._receipt_temp_path = self._scratch / "receipt.tmp"
        self._outer_lock_path = self._scratch.with_name(self._scratch.name + ".lock")

        self._generator: Generator[object, object, None] | None = None
        self._pending: object | None = None
        self._boundary_fresh = False
        self._started = False
        self._stop_injected = False
        self._close_requested = False
        self._busy = False
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._error_reported = False
        self._cleanup_error: BaseException | None = None
        self._current_io: SliceIO | None = None
        self._soft_deadline: float | None = None
        self._advance_quantum = _MAX_IO_CHUNK
        self._phase = "initial"

        self._outer_lock_fd: int | None = None
        self._scratch_fd: int | None = None
        self._receipt_fd: int | None = None
        self._receipt_temp_fd: int | None = None
        self._receipt_temp_exists = False
        self._clean_stack: list[_CleanFrame] = []

        self._projection: ProjectionDriver | None = None
        self._qualification: SourceQualification | None = None
        self._qualified: QualifiedSource | None = None
        self._receipt_record: dict[str, object] | None = None
        self._session_started = False
        self._checkpoint_requested = False
        self._checkpoint_inflight = False
        self._checkpoint_generation = 0
        self._next_checkpoint_offset = _CHECKPOINT_INTERVAL
        self._published = False
        self._result: ConfirmedSource | None = None
        self._is_complete = False

    @property
    def is_complete(self) -> bool:
        """Whether a durable receipt has been published and validated."""

        return self._is_complete

    @property
    def is_closed(self) -> bool:
        """Whether all child and wrapper resources have been released."""

        return self._closed

    @property
    def result(self) -> ConfirmedSource | None:
        """Return the result only after durable receipt publication."""

        return self._result

    def request_checkpoint(self) -> None:
        """Request a durable projection checkpoint at the next safe boundary."""

        if self._closed:
            raise ValueError("source confirmation is closed")
        if self._close_requested:
            raise ValueError("source confirmation is closing")
        self._checkpoint_requested = True

    def request_close(self) -> None:
        """Request cooperative cleanup without performing filesystem I/O."""

        if self._closed or self._close_requested:
            return
        self._close_requested = True
        if self._generator is None:
            self._closed = True

    def advance(
        self,
        io: SliceIO,
        *,
        max_processed_bytes: int = _MAX_IO_CHUNK,
        soft_deadline: float | None = None,
    ) -> RecoveryStep:
        """Perform bounded projection, qualification, or publication work."""

        if not isinstance(io, SliceIO):
            raise TypeError("io must be a SliceIO")
        if type(max_processed_bytes) is not int:
            raise TypeError("max_processed_bytes must be an exact int")
        if max_processed_bytes <= 0:
            raise ValueError("max_processed_bytes must be positive")
        if soft_deadline is not None:
            if isinstance(soft_deadline, bool) or not isinstance(soft_deadline, (int, float)):
                raise TypeError("soft_deadline must be a real number or None")
        if self._closed:
            if self._fatal_error is not None and not self._error_reported:
                self._error_reported = True
                raise self._fatal_error
            return RecoveryStep("closed", None)
        if self._busy:
            raise RuntimeError("source confirmation advance is busy")

        self._busy = True
        self._current_io = io
        self._soft_deadline = float(soft_deadline) if soft_deadline is not None else None
        self._advance_quantum = max_processed_bytes
        try:
            if self._generator is None:
                self._generator = self._run()
            return self._drive()
        finally:
            self._current_io = None
            self._soft_deadline = None
            self._busy = False

    def _drive(self) -> RecoveryStep:
        while True:
            if self._closed:
                if self._fatal_error is not None and not self._error_reported:
                    self._error_reported = True
                    raise self._fatal_error
                return RecoveryStep("closed", None)

            if self._pending is None:
                self._receive(None, start=not self._started)
                if self._closed:
                    continue

            if self._close_requested and not self._stop_injected:
                self._inject_stop()
                continue

            pending = self._pending
            if isinstance(pending, _Boundary):
                if self._boundary_fresh:
                    self._boundary_fresh = False
                    if pending.state == "complete" and self._is_complete:
                        return RecoveryStep("complete", pending.reason)
                    if pending.state == "waiting":
                        return RecoveryStep("waiting", pending.reason)
                    return RecoveryStep("progressed", pending.reason)
                self._pending = None
                self._receive(None)
                continue

            if isinstance(pending, _IORequest):
                if (
                    self._soft_deadline is not None
                    and time.monotonic() >= self._soft_deadline
                    and self._phase != "cleanup"
                ):
                    return RecoveryStep("waiting", "deadline")
                current_io = self._current_io
                if current_io is None:
                    raise RuntimeError("source confirmation has no active SliceIO")
                if pending.owner_attr is not None:
                    setattr(self, pending.owner_attr, None)
                try:
                    value = getattr(current_io, pending.method)(*pending.args)
                except BaseException as exc:
                    self._pending = None
                    self._throw(exc)
                    continue
                if value is YIELD:
                    if pending.owner_attr is not None:
                        setattr(self, pending.owner_attr, pending.args[0])
                    return RecoveryStep("waiting", "budget")
                self._pending = None
                self._receive(value)
                continue

            if isinstance(pending, _ChildRequest):
                current_io = self._current_io
                if current_io is None:
                    raise RuntimeError("source confirmation has no active SliceIO")
                child = self._child(pending.kind)
                if child is None:
                    raise RuntimeError(f"{pending.kind} child is unavailable")
                try:
                    if pending.kind == "projection":
                        value = child.advance(
                            current_io,
                            max_processed_bytes=self._advance_quantum,
                            soft_deadline=self._soft_deadline,
                        )
                    else:
                        value = child.advance(
                            current_io,
                            max_events=_MAX_EVENTS,
                            soft_deadline=self._soft_deadline,
                        )
                except BaseException as exc:
                    self._pending = None
                    self._throw(exc)
                    continue
                self._pending = None
                self._receive(value)
                continue

            if isinstance(pending, _Hold):
                return RecoveryStep("complete", None)
            raise RuntimeError("source confirmation yielded an unknown request")

    def _receive(self, value: object, *, start: bool = False) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("source confirmation executor is unavailable")
        try:
            self._started = self._started or start
            yielded = next(generator) if start else generator.send(value)
        except StopIteration:
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            if self._fatal_error is None and self._cleanup_error is not None:
                self._fatal_error = self._cleanup_error
            return
        except BaseException as exc:
            if not isinstance(exc, _StopRequested) and self._fatal_error is None:
                self._fatal_error = exc
            self._result = None
            self._is_complete = False
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            return
        self._pending = yielded
        self._boundary_fresh = isinstance(yielded, _Boundary)

    def _throw(self, error: BaseException) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("source confirmation executor is unavailable")
        try:
            yielded = generator.throw(error)
        except StopIteration:
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            if self._fatal_error is None and self._cleanup_error is not None:
                self._fatal_error = self._cleanup_error
            return
        except BaseException as exc:
            if self._fatal_error is None and not isinstance(exc, _StopRequested):
                self._fatal_error = exc
            self._result = None
            self._is_complete = False
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            return
        self._pending = yielded
        self._boundary_fresh = isinstance(yielded, _Boundary)

    def _inject_stop(self) -> None:
        if self._stop_injected:
            return
        if not self._started or self._generator is None:
            self._closed = True
            self._pending = None
            return
        self._stop_injected = True
        self._pending = None
        try:
            yielded = self._generator.throw(_StopRequested())
        except StopIteration:
            self._closed = True
            if self._cleanup_error is not None and self._fatal_error is None:
                self._fatal_error = self._cleanup_error
            return
        except BaseException as exc:
            if self._fatal_error is None and not isinstance(exc, _StopRequested):
                self._fatal_error = exc
            self._closed = True
            return
        self._pending = yielded
        self._boundary_fresh = isinstance(yielded, _Boundary)

    def _run(self) -> Generator[object, object, None]:
        try:
            while True:
                if self._is_complete:
                    yield _Hold()
                    continue
                if not self._session_started:
                    receipt = yield from self._read_receipt()
                    if receipt is not _MISSING:
                        yield from self._verify_receipt(receipt)
                        self._publish_cached(receipt)
                        self._session_started = True
                        yield _Boundary("complete")
                        continue

                    acquired = yield from self._acquire_outer_lock()
                    if not acquired:
                        yield _Boundary("waiting", "source_busy")
                        continue

                    # A peer may have published while this session waited for
                    # the outer lock. Receipt publication is immutable, so
                    # reread it before touching qualification scratch.
                    receipt = yield from self._read_receipt()
                    if receipt is not _MISSING:
                        yield from self._verify_receipt(receipt)
                        self._publish_cached(receipt)
                        self._session_started = True
                        yield _Boundary("complete")
                        continue

                    yield from self._ensure_scratch_directory()
                    yield from self._clean_qualification_scratch()
                    self._projection = ProjectionDriver(
                        self._source,
                        self._projection_path,
                        self._operation_id,
                        self._group,
                    )
                    self._session_started = True
                    yield _Boundary("progressed")
                    continue

                projection = self._projection
                if projection is None:
                    raise RuntimeError("source confirmation projection is unavailable")
                if self._qualified is None and not projection.is_complete:
                    yield from self._advance_projection()
                    continue

                if self._qualification is None:
                    self._qualification = SourceQualification(projection, self._qualification_path)
                    yield _Boundary("progressed")
                    continue

                qualification = self._qualification
                if self._qualified is None and not qualification.is_complete:
                    yield from self._advance_qualification()
                    continue

                if self._qualified is None:
                    self._qualified = qualification.result
                    if self._qualified is None:
                        raise RuntimeError("qualification completed without a result")

                if not qualification.is_closed:
                    yield from self._close_child("qualification")
                    continue
                if not projection.is_closed:
                    yield from self._close_child("projection")
                    continue

                if not self._published:
                    yield from self._publish_receipt(self._qualified)
                    self._published = True
                    self._publish_result(self._qualified)
                    yield _Boundary("complete")
                    continue

                yield _Hold()
        except _StopRequested:
            pass
        except BaseException as exc:
            if self._fatal_error is None:
                self._fatal_error = exc
            self._result = None
            self._is_complete = False
        finally:
            try:
                yield from self._cleanup()
            except BaseException as exc:
                self._record_cleanup_error(exc)

    def _advance_projection(self) -> Generator[object, object, None]:
        projection = self._projection
        if projection is None:
            raise RuntimeError("projection is unavailable")
        if self._checkpoint_requested and not self._checkpoint_inflight:
            projection.request_checkpoint()
            self._checkpoint_requested = False
            self._checkpoint_inflight = True
            self._checkpoint_generation = projection.checkpoint_generation
        elif not self._checkpoint_inflight and projection.processed_offset >= self._next_checkpoint_offset:
            projection.request_checkpoint()
            self._checkpoint_inflight = True
            self._checkpoint_generation = projection.checkpoint_generation

        step = yield _ChildRequest("projection")
        if not hasattr(step, "state"):
            raise RuntimeError("projection returned an invalid step")
        if step.state == "waiting":
            reason = "source_busy" if step.reason == "scratch_busy" else step.reason
            yield _Boundary("waiting", reason)
            return
        if step.state == "closed" and not projection.is_closed:
            raise RuntimeError("projection reported closed without releasing resources")
        if self._checkpoint_inflight and projection.checkpoint_generation > self._checkpoint_generation:
            self._checkpoint_inflight = False
            while self._next_checkpoint_offset <= projection.processed_offset:
                self._next_checkpoint_offset += _CHECKPOINT_INTERVAL
        if step.state == "progressed":
            yield _Boundary("progressed")

    def _advance_qualification(self) -> Generator[object, object, None]:
        qualification = self._qualification
        if qualification is None:
            raise RuntimeError("qualification is unavailable")
        step = yield _ChildRequest("qualification")
        if not hasattr(step, "state"):
            raise RuntimeError("qualification returned an invalid step")
        if step.state == "waiting":
            yield _Boundary("waiting", step.reason)
            return
        if step.state == "closed" and not qualification.is_closed:
            raise RuntimeError("qualification reported closed without releasing resources")
        if step.state == "progressed":
            yield _Boundary("progressed")

    def _close_child(self, kind: Literal["projection", "qualification"]) -> Generator[object, object, None]:
        child = self._child(kind)
        if child is None:
            return
        try:
            child.request_close()
        except BaseException as exc:
            self._record_cleanup_error(exc)
            return
        while not child.is_closed:
            try:
                step = yield _ChildRequest(kind)
            except BaseException as exc:
                self._record_cleanup_error(exc)
                return
            if getattr(step, "state", None) == "waiting":
                yield _Boundary("waiting", getattr(step, "reason", None))
            elif not child.is_closed:
                yield _Boundary("progressed")

    def _child(self, kind: Literal["projection", "qualification"]) -> object | None:
        return self._projection if kind == "projection" else self._qualification

    def _read_receipt(self) -> Generator[object, object, object]:
        self._phase = "receipt_read"
        try:
            descriptor = yield from self._call("open", self._receipt_path, _READ_FLAGS, 0o600)
        except FileNotFoundError:
            return _MISSING
        if type(descriptor) is not int:
            raise TypeError("receipt open returned an invalid descriptor")
        self._receipt_fd = descriptor
        try:
            first = yield from self._call("fstat", descriptor)
            _require_regular(first.st_mode, self._receipt_path, "receipt")
            if first.st_size > _MAX_RECEIPT_BYTES:
                raise ValueError("receipt exceeds the bounded record size")
            named = yield from self._call("lstat", self._receipt_path)
            _require_regular(named.st_mode, self._receipt_path, "receipt")
            _require_same_identity(first, named, self._receipt_path, "receipt")
            data = bytearray()
            remaining = first.st_size
            while remaining:
                chunk = yield from self._call("read", descriptor, min(remaining, _MAX_RECEIPT_BYTES))
                if type(chunk) is not bytes or not chunk or len(chunk) > remaining:
                    raise ValueError("receipt read exceeded its recorded size")
                data.extend(chunk)
                remaining -= len(chunk)
            second = yield from self._call("fstat", descriptor)
            _require_regular(second.st_mode, self._receipt_path, "receipt")
            if second.st_size != first.st_size:
                raise ValueError("receipt changed while reading")
            named_second = yield from self._call("lstat", self._receipt_path)
            _require_regular(named_second.st_mode, self._receipt_path, "receipt")
            _require_same_identity(second, named_second, self._receipt_path, "receipt")
            return decode_receipt(bytes(data), self._source, self._operation_id, self._group, self._scratch)
        finally:
            descriptor = self._receipt_fd
            if descriptor is not None:
                yield from self._close_owned("_receipt_fd", descriptor)

    def _acquire_outer_lock(self) -> Generator[object, object, bool]:
        if self._outer_lock_fd is not None:
            return True
        self._phase = "source_lock"
        yield from self._ensure_parents(self._scratch.parent)
        descriptor = yield from self._call("open", self._outer_lock_path, _LOCK_FLAGS, 0o600)
        if type(descriptor) is not int:
            raise TypeError("source lock open returned an invalid descriptor")
        self._outer_lock_fd = descriptor
        info = yield from self._call("fstat", descriptor)
        _require_regular(info.st_mode, self._outer_lock_path, "source lock")
        try:
            yield from self._call("flock", descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield from self._close_owned("_outer_lock_fd", descriptor)
            return False
        return True

    def _ensure_parents(self, directory: Path) -> Generator[object, object, None]:
        missing = []
        candidate = directory
        while True:
            try:
                info = yield from self._call("lstat", candidate)
            except FileNotFoundError:
                missing.append(candidate)
                candidate = candidate.parent
                continue
            _require_directory(info.st_mode, candidate, "scratch ancestor")
            break
        for candidate in reversed(missing):
            try:
                yield from self._call("mkdir", candidate, 0o700)
            except FileExistsError:
                pass
            yield from self._sync_directory(candidate)
            yield from self._sync_directory(candidate.parent)

    def _ensure_scratch_directory(self) -> Generator[object, object, None]:
        self._phase = "scratch"
        try:
            yield from self._call("mkdir", self._scratch, 0o700)
        except FileExistsError:
            pass
        descriptor = yield from self._call("open", self._scratch, _DIRECTORY_FLAGS, 0o700)
        if type(descriptor) is not int:
            raise TypeError("scratch open returned an invalid descriptor")
        self._scratch_fd = descriptor
        first = yield from self._call("fstat", descriptor)
        _require_directory(first.st_mode, self._scratch, "scratch")
        named = yield from self._call("lstat", self._scratch)
        _require_directory(named.st_mode, self._scratch, "scratch")
        _require_same_identity(first, named, self._scratch, "scratch")
        yield from self._sync_directory(self._scratch.parent)

    def _clean_qualification_scratch(self) -> Generator[object, object, None]:
        self._phase = "qualification_cleanup"
        root = self._qualification_path
        try:
            info = yield from self._call("lstat", root)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            try:
                yield from self._call("unlink", root)
            except FileNotFoundError:
                pass
            return

        self._clean_stack = [_CleanFrame(root)]
        while self._clean_stack:
            frame = self._clean_stack[-1]
            if frame.iterator is None:
                try:
                    iterator = yield from self._call("scandir", frame.path)
                except FileNotFoundError:
                    self._clean_stack.pop()
                    continue
                frame.iterator = iterator
            try:
                entry = yield from self._call("next_entry", frame.iterator)
            except FileNotFoundError:
                entry = None
            if entry is None:
                iterator = frame.iterator
                frame.iterator = None
                if iterator is not None:
                    yield from self._call("close_directory", iterator)
                self._clean_stack.pop()
                try:
                    yield from self._call("rmdir", frame.path)
                except FileNotFoundError:
                    pass
                yield _Boundary("progressed")
                continue

            name = entry.name
            if not isinstance(name, str) or name in {".", ".."} or "/" in name:
                raise ValueError("qualification scratch contains an invalid directory entry")
            child = frame.path / name
            try:
                child_info = yield from self._call("lstat", child)
            except FileNotFoundError:
                yield _Boundary("progressed")
                continue
            if stat.S_ISDIR(child_info.st_mode):
                self._clean_stack.append(_CleanFrame(child))
            else:
                try:
                    yield from self._call("unlink", child)
                except FileNotFoundError:
                    pass
            yield _Boundary("progressed")
        self._clean_stack.clear()

    def _publish_receipt(self, qualified: QualifiedSource) -> Generator[object, object, None]:
        self._phase = "receipt_publish"
        manifest = qualified.projection
        yield from self._verify_source(manifest.source, manifest.source_revision)
        spool_revision = yield from self._sync_file(manifest.spool, "event spool")
        if spool_revision.device != manifest.spool_device or spool_revision.inode != manifest.spool_inode:
            raise ValueError("event spool does not match the completed projection")
        if spool_revision.size != manifest.spool_size:
            raise ValueError("event spool size does not match the completed projection")

        task_record = None
        sequence_record = None
        if qualified.status == "qualified":
            if qualified.task_references is None or qualified.sequence_references is None:
                raise ValueError("qualified source is missing reference files")
            task_revision = yield from self._sync_file(qualified.task_references, "task references")
            sequence_revision = yield from self._sync_file(qualified.sequence_references, "sequence references")
            task_record = _file_record(self._relative(qualified.task_references), task_revision)
            sequence_record = _file_record(self._relative(qualified.sequence_references), sequence_revision)

        yield from self._sync_directory_if_exists(self._qualification_path)
        yield from self._sync_directory_if_exists(self._qualification_path / "task-audit")
        yield from self._sync_directory_if_exists(self._qualification_path / "sequence-audit")
        yield from self._sync_directory(self._projection_path)

        task_count = manifest.summary.task_count
        if type(task_count) is not int or task_count < 0:
            raise ValueError("projection task count is invalid")
        record = {
            "version": 1,
            "source_path": str(self._source),
            "source_revision": _revision_dict(manifest.source_revision),
            "operation_id": self._operation_id,
            "group": self._group,
            "status": qualified.status,
            "spool": _file_record(self._relative(manifest.spool), spool_revision),
            "task_references": task_record,
            "sequence_references": sequence_record,
            "task_count": task_count,
        }
        payload = _encode_json(record)
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds the bounded record size")

        try:
            yield from self._call("unlink", self._receipt_temp_path)
        except FileNotFoundError:
            pass
        descriptor = yield from self._call("open", self._receipt_temp_path, _TEMP_FLAGS, 0o600)
        if type(descriptor) is not int:
            raise TypeError("receipt temp open returned an invalid descriptor")
        self._receipt_temp_fd = descriptor
        self._receipt_temp_exists = True
        temp_info = yield from self._call("fstat", descriptor)
        _require_regular(temp_info.st_mode, self._receipt_temp_path, "receipt temp")
        temp_named = yield from self._call("lstat", self._receipt_temp_path)
        _require_regular(temp_named.st_mode, self._receipt_temp_path, "receipt temp")
        _require_same_identity(temp_info, temp_named, self._receipt_temp_path, "receipt temp")
        offset = 0
        while offset < len(payload):
            written = yield from self._call("write", descriptor, payload[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise OSError("short or invalid receipt write")
            offset += written
        yield from self._call("fsync", descriptor)
        yield from self._close_owned("_receipt_temp_fd", descriptor)
        self._receipt_temp_exists = False
        yield from self._call("replace", self._receipt_temp_path, self._receipt_path)
        yield from self._sync_scratch_directory()
        self._receipt_record = record

    def _publish_result(self, qualified: QualifiedSource) -> None:
        manifest = qualified.projection
        self._result = ConfirmedSource(
            status=qualified.status,
            source=manifest.source,
            source_revision=manifest.source_revision,
            spool=manifest.spool,
            task_references=qualified.task_references,
            sequence_references=qualified.sequence_references,
            operation_id=self._operation_id,
            group=self._group,
            task_count=manifest.summary.task_count,
        )
        self._is_complete = True
        self._phase = "complete"

    def _publish_cached(self, record: dict[str, object]) -> None:
        source_revision = record["source_revision"]
        spool = record["spool"]
        if not isinstance(source_revision, SourceRevision) or type(spool) is not dict:
            raise ValueError("receipt cache is invalid")
        task = record["task_references"]
        sequence = record["sequence_references"]
        task_path = self._scratch / task["path"] if isinstance(task, dict) else None
        sequence_path = self._scratch / sequence["path"] if isinstance(sequence, dict) else None
        self._result = ConfirmedSource(
            status=record["status"],
            source=self._source,
            source_revision=source_revision,
            spool=self._scratch / spool["path"],
            task_references=task_path,
            sequence_references=sequence_path,
            operation_id=self._operation_id,
            group=self._group,
            task_count=record["task_count"],
        )
        self._is_complete = True
        self._phase = "complete"

    def _verify_receipt(self, record: dict[str, object]) -> Generator[object, object, None]:
        source_revision = record["source_revision"]
        spool = record["spool"]
        if not isinstance(source_revision, SourceRevision) or type(spool) is not dict:
            raise ValueError("receipt cache is invalid")
        yield from self._verify_source(self._source, source_revision)
        yield from self._verify_file_record(spool)
        for value in (record["task_references"], record["sequence_references"]):
            if value is not None:
                if type(value) is not dict:
                    raise ValueError("receipt reference record is invalid")
                yield from self._verify_file_record(value)

    def _verify_file_record(self, record: dict[str, object]) -> Generator[object, object, None]:
        path = self._scratch / record["path"]
        expected_revision = record["revision"]
        expected_size = record["size"]
        if not isinstance(expected_revision, SourceRevision) or type(expected_size) is not int:
            raise ValueError("receipt file record is invalid")
        actual = yield from self._verify_file(path, expected_revision, expected_size, "confirmed output")
        if actual != expected_revision:
            raise ValueError("confirmed output revision changed")

    def _verify_source(self, path: Path, expected: SourceRevision) -> Generator[object, object, None]:
        try:
            yield from self._verify_file(path, expected, expected.size, "source")
        except SourceChangedError:
            raise
        except (OSError, ValueError) as exc:
            raise SourceChangedError(f"source revision changed for {path}") from exc

    def _verify_file(
        self,
        path: Path,
        expected: SourceRevision,
        expected_size: int,
        label: str,
    ) -> Generator[object, object, SourceRevision]:
        descriptor = yield from self._call("open", path, _READ_FLAGS, 0o600)
        if type(descriptor) is not int:
            raise TypeError(f"{label} open returned an invalid descriptor")
        self._receipt_fd = descriptor
        try:
            first = yield from self._call("fstat", descriptor)
            actual = _stat_revision(first, path, label)
            if actual != expected or actual.size != expected_size:
                raise ValueError(f"{label} revision does not match its receipt")
            named = yield from self._call("lstat", path)
            named_revision = _stat_revision(named, path, label)
            if named_revision != actual:
                raise ValueError(f"{label} path was replaced")
            second = yield from self._call("fstat", descriptor)
            if _stat_revision(second, path, label) != actual:
                raise ValueError(f"{label} changed while opening")
            return actual
        finally:
            descriptor = self._receipt_fd
            if descriptor is not None:
                yield from self._close_owned("_receipt_fd", descriptor)

    def _sync_file(self, path: Path, label: str) -> Generator[object, object, SourceRevision]:
        descriptor = yield from self._call("open", path, _READ_FLAGS, 0o600)
        if type(descriptor) is not int:
            raise TypeError(f"{label} open returned an invalid descriptor")
        self._receipt_fd = descriptor
        try:
            first = yield from self._call("fstat", descriptor)
            actual = _stat_revision(first, path, label)
            named = yield from self._call("lstat", path)
            if _stat_revision(named, path, label) != actual:
                raise ValueError(f"{label} path was replaced")
            yield from self._call("fsync", descriptor)
            second = yield from self._call("fstat", descriptor)
            if _stat_revision(second, path, label) != actual:
                raise ValueError(f"{label} changed while syncing")
            return actual
        finally:
            descriptor = self._receipt_fd
            if descriptor is not None:
                yield from self._close_owned("_receipt_fd", descriptor)

    def _sync_directory(self, path: Path) -> Generator[object, object, None]:
        descriptor = yield from self._call("open", path, _DIRECTORY_FLAGS, 0o700)
        if type(descriptor) is not int:
            raise TypeError("directory open returned an invalid descriptor")
        self._receipt_fd = descriptor
        try:
            first = yield from self._call("fstat", descriptor)
            _require_directory(first.st_mode, path, "directory")
            named = yield from self._call("lstat", path)
            _require_directory(named.st_mode, path, "directory")
            _require_same_identity(first, named, path, "directory")
            yield from self._call("fsync", descriptor)
        finally:
            descriptor = self._receipt_fd
            if descriptor is not None:
                yield from self._close_owned("_receipt_fd", descriptor)

    def _sync_directory_if_exists(self, path: Path) -> Generator[object, object, None]:
        try:
            yield from self._sync_directory(path)
        except FileNotFoundError:
            return

    def _sync_scratch_directory(self) -> Generator[object, object, None]:
        descriptor = self._scratch_fd
        if descriptor is None:
            raise RuntimeError("scratch directory is unavailable")
        first = yield from self._call("fstat", descriptor)
        _require_directory(first.st_mode, self._scratch, "scratch")
        named = yield from self._call("lstat", self._scratch)
        _require_directory(named.st_mode, self._scratch, "scratch")
        _require_same_identity(first, named, self._scratch, "scratch")
        yield from self._call("fsync", descriptor)

    def _cleanup(self) -> Generator[object, object, None]:
        self._phase = "cleanup"
        qualification = self._qualification
        if qualification is not None and not qualification.is_closed:
            yield from self._close_child("qualification")
        projection = self._projection
        if projection is not None and not projection.is_closed:
            yield from self._close_child("projection")

        for frame in reversed(self._clean_stack):
            iterator = frame.iterator
            frame.iterator = None
            if iterator is not None:
                try:
                    yield from self._cleanup_call("close_directory", iterator)
                except BaseException as exc:
                    self._record_cleanup_error(exc)
        self._clean_stack.clear()

        if self._receipt_temp_fd is not None:
            descriptor = self._receipt_temp_fd
            self._receipt_temp_fd = None
            yield from self._cleanup_call("close", descriptor)
        if self._receipt_temp_exists:
            self._receipt_temp_exists = False
            try:
                yield from self._cleanup_call("unlink", self._receipt_temp_path)
            except BaseException as exc:
                if not isinstance(exc, FileNotFoundError):
                    self._record_cleanup_error(exc)
        if self._receipt_fd is not None:
            descriptor = self._receipt_fd
            self._receipt_fd = None
            yield from self._cleanup_call("close", descriptor)
        if self._scratch_fd is not None:
            descriptor = self._scratch_fd
            self._scratch_fd = None
            yield from self._cleanup_call("close", descriptor)
        if self._outer_lock_fd is not None:
            descriptor = self._outer_lock_fd
            self._outer_lock_fd = None
            yield from self._cleanup_call("flock", descriptor, fcntl.LOCK_UN)
            yield from self._cleanup_call("close", descriptor)

    def _cleanup_call(self, method: str, *args: object) -> Generator[object, object, object]:
        try:
            return_value = yield _IORequest(method, tuple(args))
            return return_value
        except BaseException as exc:
            self._record_cleanup_error(exc)
            return None

    def _close_owned(self, attribute: str, descriptor: int) -> Generator[object, object, None]:
        # Detach before admission and restore only after SliceIO returns YIELD.
        yield _IORequest("close", (descriptor,), attribute)

    def _call(self, method: str, *args: object) -> Generator[object, object, object]:
        value = yield _IORequest(method, tuple(args))
        return value

    def _relative(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._scratch)
        except ValueError as exc:
            raise ValueError(f"path is outside scratch: {path}") from exc
        return str(relative)

    def _record_cleanup_error(self, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if self._cleanup_error is None:
            self._cleanup_error = error
        self._result = None
        self._is_complete = False


class _Hold:
    __slots__ = ()


def _normalized_absolute(value: Path, name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a path") from exc
    if isinstance(raw, bytes):
        raise TypeError(f"{name} must be a text path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be absolute")
    normalized = Path(os.path.normpath(str(candidate)))
    if candidate != normalized:
        raise ValueError(f"{name} must be normalized")
    return candidate


def _require_regular(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _require_directory(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is not a directory: {path}")


def _require_same_identity(first: os.stat_result, second: os.stat_result, path: Path, label: str) -> None:
    if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
        raise ValueError(f"{label} path was replaced: {path}")


def _stat_revision(info: os.stat_result, path: Path, label: str) -> SourceRevision:
    _require_regular(info.st_mode, path, label)
    return SourceRevision.from_stat(info)


def _revision_dict(revision: SourceRevision) -> dict[str, int]:
    return {name: getattr(revision, name) for name in _STAT_ORDER}


def _file_record(path: str, revision: SourceRevision) -> dict[str, object]:
    return {"path": path, "size": revision.size, "revision": _revision_dict(revision)}


def _encode_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("receipt cannot be encoded as deterministic JSON") from exc


__all__ = ["ConfirmedSource", "RecoverableSource", "RecoveryStep"]
