"""Bounded qualification of a completed qexp source projection.

The qualification pass validates the completed projection's event spool,
materializes positional references, and audits task and sequence fingerprints
for uniqueness.  A successful result is source qualification only; it does
not establish Group coverage, writer admission, consumer authorization, or
runtime ownership.  Qualification scratch is deliberately fresh and has no
restart or persistence promise: callers must start a new qualification after
an interruption or process restart.
"""

from __future__ import annotations

import os
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .digest_audit import DigestAuditDriver
from .driver import CompletedProjection, ProjectionDriver
from .event_decoder import DecodedEvent, EventDecoder
from .slice_io import YIELD, SliceIO
from .source_revision import SourceRevision
from .submission_projection import FieldChunk, FieldEnd

_MAX_IO_CHUNK = 65_536
_MAX_EVENTS = 64
_U64_MAX = (1 << 64) - 1
_SOURCE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_OUTPUT_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


@dataclass(frozen=True, slots=True)
class QualificationStep:
    """Bounded progress returned by one qualification advance."""

    state: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class QualifiedSource:
    """The provisional source qualification result."""

    status: str
    projection: CompletedProjection
    task_references: Path | None
    sequence_references: Path | None


@dataclass(frozen=True, slots=True)
class _IORequest:
    method: str
    args: tuple[object, ...]
    owner_attr: str | None = None


@dataclass(frozen=True, slots=True)
class _AuditRequest:
    kind: str


@dataclass(frozen=True, slots=True)
class _Boundary:
    kind: str
    reason: str | None = None


class _StopRequested(Exception):
    pass


class SourceQualification:
    """Validate a completed projection and audit its positional field output.

    Construction performs no filesystem I/O.  The caller owns the projection
    driver and must keep it complete, open, and unadvanced until qualification
    finishes or is closed.  The qualification scratch path must be new; an
    interrupted qualification is discarded and restarted with a fresh path.
    """

    def __init__(self, projection: ProjectionDriver, scratch: Path):
        if not isinstance(projection, ProjectionDriver):
            raise TypeError("projection must be a ProjectionDriver")
        manifest = projection.completed_projection
        if manifest is None:
            raise ValueError("projection must be complete and open")

        self._projection = projection
        self._manifest = manifest
        self._scratch = _normalized_absolute(scratch, "scratch")

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
        self._phase = "initial"
        self._current_io: SliceIO | None = None
        self._soft_deadline: float | None = None
        self._max_events = _MAX_EVENTS
        self._events_used = 0
        self._audit_called = False

        self._initialized = False
        self._classification: str | None = None
        self._decoder: EventDecoder | None = None
        self._spool_offset = 0
        self._spool_eof = False
        self._event_count = 0
        self._task_count = 0
        self._sequence_count = 0

        self._source_fd: int | None = None
        self._spool_fd: int | None = None
        self._source_revision: SourceRevision | None = None
        self._spool_revision: SourceRevision | None = None

        self._task_digest_fd: int | None = None
        self._task_reference_fd: int | None = None
        self._sequence_digest_fd: int | None = None
        self._sequence_reference_fd: int | None = None
        self._task_digest_path = self._scratch / "task.digests"
        self._task_reference_path = self._scratch / "task.refs"
        self._sequence_digest_path = self._scratch / "sequence.digests"
        self._sequence_reference_path = self._scratch / "sequence.refs"

        self._task_audit: DigestAuditDriver | None = None
        self._sequence_audit: DigestAuditDriver | None = None
        self._audit_kind: str | None = None

        self._is_complete = False
        self._result: QualifiedSource | None = None

    @property
    def is_complete(self) -> bool:
        """Whether qualification has reached an irreversible terminal result."""

        return self._is_complete

    @property
    def is_closed(self) -> bool:
        """Whether qualification has released all owned resources."""

        return self._closed

    @property
    def result(self) -> QualifiedSource | None:
        """Return the terminal result, if qualification has completed."""

        return self._result

    def request_close(self) -> None:
        """Request cooperative cleanup without touching the caller's driver."""

        if self._closed:
            return
        if self._close_requested:
            return
        self._close_requested = True
        if self._generator is None:
            self._closed = True

    def advance(
        self,
        io: SliceIO,
        *,
        max_events: int = _MAX_EVENTS,
        soft_deadline: float | None = None,
    ) -> QualificationStep:
        """Perform bounded spool, reference, and audit work."""

        if not isinstance(io, SliceIO):
            raise TypeError("io must be a SliceIO")
        max_events = _validate_max_events(max_events)
        if soft_deadline is not None:
            if isinstance(soft_deadline, bool) or not isinstance(soft_deadline, (int, float)):
                raise TypeError("soft_deadline must be a real number or None")
        if self._closed:
            return QualificationStep("closed", None)
        if self._busy:
            raise RuntimeError("source qualification advance is busy")

        self._busy = True
        self._current_io = io
        self._soft_deadline = float(soft_deadline) if soft_deadline is not None else None
        self._max_events = max_events
        self._events_used = 0
        self._audit_called = False
        try:
            if self._generator is None:
                self._generator = self._run()
            return self._drive()
        finally:
            self._current_io = None
            self._soft_deadline = None
            self._busy = False

    def _drive(self) -> QualificationStep:
        while True:
            if self._closed:
                if self._fatal_error is not None and not self._error_reported:
                    self._error_reported = True
                    raise self._fatal_error
                return QualificationStep("closed", None)
            if self._is_complete and not self._close_requested:
                return QualificationStep("complete", None)

            if self._pending is None:
                self._receive(None, start=not self._started)
                if self._closed:
                    continue

            pending = self._pending
            if isinstance(pending, _Boundary):
                if self._close_requested and not self._stop_injected:
                    self._inject_stop()
                    continue
                if self._boundary_fresh:
                    self._boundary_fresh = False
                    if pending.reason is not None:
                        return QualificationStep("waiting", pending.reason)
                    if pending.kind == "complete" and self._is_complete:
                        return QualificationStep("complete", None)
                    return QualificationStep("progressed", None)
                self._pending = None
                self._receive(None)
                continue

            if isinstance(pending, _AuditRequest):
                if self._close_requested and not self._stop_injected:
                    self._inject_stop()
                    continue
                if self._audit_called:
                    return QualificationStep("progressed", None)
                if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                    return QualificationStep("waiting", "deadline")
                if self._phase != "cleanup":
                    self._assert_active_projection()
                audit = self._audit_for_kind(pending.kind)
                if audit is None:
                    raise RuntimeError("qualification audit is unavailable")
                self._audit_called = True
                try:
                    audit_step = audit.advance(
                        self._current_io,
                        max_records=_MAX_EVENTS,
                        soft_deadline=self._soft_deadline,
                    )
                except BaseException as exc:
                    self._throw(exc)
                    continue
                self._pending = None
                self._receive(None)
                if audit_step.state == "waiting":
                    return QualificationStep("waiting", audit_step.reason)
                return QualificationStep("progressed", None)

            if isinstance(pending, _Hold):
                if self._close_requested and not self._stop_injected:
                    self._inject_stop()
                    continue
                return QualificationStep("complete", None)
            if not isinstance(pending, _IORequest):
                raise RuntimeError("qualification executor yielded an unknown request")
            if self._close_requested and not self._stop_injected and self._phase != "cleanup":
                self._inject_stop()
                continue
            if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                return QualificationStep("waiting", "deadline")
            if self._phase != "cleanup":
                self._assert_active_projection()
            current_io = self._current_io
            if current_io is None:
                raise RuntimeError("qualification has no active SliceIO")
            # A Linux close may have already released this integer descriptor
            # even when os.close reports an error.  Detach ownership before
            # attempting close, restoring it only when SliceIO admits no
            # syscall at all.
            if pending.owner_attr is not None:
                setattr(self, pending.owner_attr, None)
            try:
                result = getattr(current_io, pending.method)(*pending.args)
            except BaseException as exc:
                self._pending = None
                self._throw(exc)
                continue
            if result is YIELD:
                if pending.owner_attr is not None:
                    setattr(self, pending.owner_attr, pending.args[0])
                return QualificationStep("waiting", "budget")
            self._pending = None
            self._receive(result)

    def _receive(self, value: object, *, start: bool = False) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("qualification executor is unavailable")
        try:
            self._started = self._started or start
            result = next(generator) if start else generator.send(value)
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
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _throw(self, error: BaseException) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("qualification executor is unavailable")
        try:
            result = generator.throw(error)
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
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _inject_stop(self) -> None:
        if self._stop_injected:
            return
        if not self._started:
            self._closed = True
            self._pending = None
            return
        generator = self._generator
        if generator is None:
            self._closed = True
            return
        self._stop_injected = True
        self._pending = None
        self._phase = "cleanup"
        try:
            result = generator.throw(_StopRequested())
        except StopIteration:
            self._closed = True
            if self._cleanup_error is not None:
                self._fatal_error = self._cleanup_error
            return
        except BaseException as exc:
            if self._fatal_error is None and not isinstance(exc, _StopRequested):
                self._fatal_error = exc
            self._closed = True
            return
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _run(self) -> Generator[object, object, None]:
        try:
            while True:
                if not self._initialized:
                    yield from self._initialize()
                    self._initialized = True
                    yield _Boundary("initialized")
                    continue

                if self._close_requested:
                    raise _StopRequested()
                self._assert_active_projection()

                if self._is_complete:
                    yield _Hold()
                    continue

                if self._classification in {"irrelevant", "ambiguous"}:
                    yield from self._verify_primary()
                    yield from self._close_primary()
                    self._publish(self._classification)
                    yield _Boundary("complete")
                    continue

                if self._audit_kind is not None:
                    audit = self._audit_for_kind(self._audit_kind)
                    if audit is None:
                        raise RuntimeError("qualification audit is unavailable")
                    if not audit.is_complete:
                        self._phase = f"{self._audit_kind}_audit"
                        yield _AuditRequest(self._audit_kind)
                        continue
                    if audit.has_repeated_digest:
                        self._classification = "ambiguous"
                        self._audit_kind = None
                        yield from self._verify_primary()
                        yield from self._close_primary()
                        self._publish("ambiguous")
                        yield _Boundary("complete")
                        continue
                    if self._audit_kind == "task":
                        self._audit_kind = "sequence"
                        continue
                    yield from self._verify_primary()
                    yield from self._close_primary()
                    self._publish("qualified")
                    yield _Boundary("complete")
                    continue

                if self._decoder is None:
                    raise RuntimeError("qualification decoder is unavailable")
                if self._events_used >= self._max_events:
                    yield _Boundary("events", "budget")
                    continue

                decoded = self._decoder.pop()
                if decoded is not None:
                    yield from self._process_event(decoded)
                    self._events_used += 1
                    self._event_count += 1
                    continue

                if not self._spool_eof:
                    if self._spool_offset >= self._manifest.spool_size:
                        self._spool_eof = True
                        continue
                    read_size = min(_MAX_IO_CHUNK, self._manifest.spool_size - self._spool_offset)
                    self._phase = "spool_read"
                    data = yield from self._call("read", self._spool_fd, read_size)
                    if type(data) is not bytes or not data:
                        raise ValueError("event spool ended before its manifest size")
                    if len(data) > read_size:
                        raise ValueError("event spool read exceeded its bounded request")
                    self._spool_offset += len(data)
                    self._decoder.feed(data)
                    continue

                self._decoder.finish(
                    expected_events=self._manifest.spool_events,
                    task_count=self._manifest.summary.task_count,
                    sequence_count=self._manifest.summary.sequence_count,
                )
                yield from self._sync_and_close_outputs()
                yield from self._verify_primary()
                self._task_audit = DigestAuditDriver(
                    self._task_digest_path,
                    self._scratch / "task-audit",
                    expected_count=self._manifest.summary.task_count,
                    run_records=_MAX_EVENTS,
                )
                self._sequence_audit = DigestAuditDriver(
                    self._sequence_digest_path,
                    self._scratch / "sequence-audit",
                    expected_count=self._manifest.summary.sequence_count,
                    run_records=_MAX_EVENTS,
                )
                self._audit_kind = "task"
                yield _Boundary("audits")
                continue
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

    def _initialize(self) -> Generator[object, object, None]:
        self._phase = "scratch"
        yield from self._call("mkdir", self._scratch, 0o700)
        yield from self._open_primary()

        summary = self._manifest.summary
        _validate_u64(summary.task_count, "summary task count")
        _validate_u64(summary.sequence_count, "summary sequence count")
        _validate_u64(self._manifest.spool_events, "manifest spool events")
        if summary.state != "committed" or not summary.matches_group:
            self._classification = "irrelevant"
            return
        if summary.task_count != summary.sequence_count:
            self._classification = "ambiguous"
            return

        self._classification = "matching"
        self._decoder = EventDecoder()
        yield from self._open_output_files()

    def _open_primary(self) -> Generator[object, object, None]:
        manifest = self._manifest
        self._phase = "source_open"
        source_fd = yield from self._call("open", manifest.source, _SOURCE_FLAGS, 0o600)
        if type(source_fd) is not int:
            raise TypeError("source open returned an invalid descriptor")
        self._source_fd = source_fd
        source_first = yield from self._call("fstat", source_fd)
        source_revision = _stat_revision(source_first, manifest.source, "source descriptor")
        if source_revision != manifest.source_revision:
            raise ValueError("source revision does not match the completed projection")
        source_named = yield from self._call("lstat", manifest.source)
        source_named_revision = _stat_revision(source_named, manifest.source, "source path")
        if source_named_revision != source_revision:
            raise ValueError("source path changed while opening qualification")
        source_second = yield from self._call("fstat", source_fd)
        if _stat_revision(source_second, manifest.source, "source descriptor") != source_revision:
            raise ValueError("source descriptor changed while opening qualification")
        self._source_revision = source_revision

        self._phase = "spool_open"
        spool_fd = yield from self._call("open", manifest.spool, _SOURCE_FLAGS, 0o600)
        if type(spool_fd) is not int:
            raise TypeError("spool open returned an invalid descriptor")
        self._spool_fd = spool_fd
        spool_first = yield from self._call("fstat", spool_fd)
        spool_revision = _stat_revision(spool_first, manifest.spool, "spool descriptor")
        _validate_spool_manifest(spool_revision, manifest)
        spool_named = yield from self._call("lstat", manifest.spool)
        if _stat_revision(spool_named, manifest.spool, "spool path") != spool_revision:
            raise ValueError("spool path changed while opening qualification")
        spool_second = yield from self._call("fstat", spool_fd)
        if _stat_revision(spool_second, manifest.spool, "spool descriptor") != spool_revision:
            raise ValueError("spool descriptor changed while opening qualification")
        self._spool_revision = spool_revision

    def _open_output_files(self) -> Generator[object, object, None]:
        self._phase = "outputs"
        for attribute, path in (
            ("_task_digest_fd", self._task_digest_path),
            ("_task_reference_fd", self._task_reference_path),
            ("_sequence_digest_fd", self._sequence_digest_path),
            ("_sequence_reference_fd", self._sequence_reference_path),
        ):
            yield from self._open_output(path, attribute)

    def _open_output(self, path: Path, attribute: str) -> Generator[object, object, None]:
        descriptor = yield from self._call("open", path, _OUTPUT_FLAGS, 0o600)
        if type(descriptor) is not int:
            raise TypeError(f"output open returned an invalid descriptor: {path}")
        # The next validation can yield or fail. Cleanup must own this handle
        # as soon as open returns, before either event can interrupt the helper.
        setattr(self, attribute, descriptor)
        info = yield from self._call("fstat", descriptor)
        _require_regular(info.st_mode, path, "qualification output")

    def _process_event(self, decoded: DecodedEvent) -> Generator[object, object, None]:
        _validate_u64(decoded.start, "event spool start")
        _validate_u64(decoded.end, "event spool end")
        _validate_u64(decoded.field_start, "event field start")
        if decoded.end > self._manifest.spool_size or decoded.field_start > decoded.end:
            raise ValueError("decoded event positions exceed the event spool")

        event = decoded.event
        if isinstance(event, FieldChunk):
            return
        if not isinstance(event, FieldEnd):
            raise ValueError("event decoder returned an unknown event")

        _validate_u64(event.ordinal, "event ordinal")
        _validate_u64(event.decoded_size, "decoded field size")
        _validate_u64(event.start, "source field start")
        _validate_u64(event.end, "source field end")
        if event.end > self._manifest.source_revision.size or event.start >= event.end:
            raise ValueError("decoded source field span exceeds the source revision")
        try:
            digest = bytes.fromhex(event.digest)
        except (TypeError, ValueError) as exc:
            raise ValueError("decoded field digest is invalid") from exc
        if len(digest) != 32:
            raise ValueError("decoded field digest must contain 32 bytes")
        reference = struct.pack(
            ">QQQQ",
            event.ordinal,
            decoded.field_start,
            decoded.end,
            event.decoded_size,
        )
        if event.kind == "task_id":
            digest_fd = self._task_digest_fd
            reference_fd = self._task_reference_fd
            self._task_count += 1
        elif event.kind == "sequence":
            digest_fd = self._sequence_digest_fd
            reference_fd = self._sequence_reference_fd
            self._sequence_count += 1
        else:
            raise ValueError("decoded field kind is invalid")
        if digest_fd is None or reference_fd is None:
            raise RuntimeError("qualification output is closed")
        yield from self._write_all(digest_fd, digest)
        yield from self._write_all(reference_fd, reference)

    def _write_all(self, descriptor: int, payload: bytes) -> Generator[object, object, None]:
        offset = 0
        while offset < len(payload):
            self._phase = "output_write"
            written = yield from self._call("write", descriptor, payload[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise OSError("short or invalid qualification output write")
            offset += written

    def _sync_and_close_outputs(self) -> Generator[object, object, None]:
        self._phase = "output_fsync"
        for attribute in (
            "_task_digest_fd",
            "_task_reference_fd",
            "_sequence_digest_fd",
            "_sequence_reference_fd",
        ):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            yield from self._call("fsync", descriptor)
            yield from self._close_owned(attribute, descriptor)

    def _verify_primary(self) -> Generator[object, object, None]:
        source_fd = self._source_fd
        spool_fd = self._spool_fd
        source_revision = self._source_revision
        spool_revision = self._spool_revision
        if source_fd is None or spool_fd is None or source_revision is None or spool_revision is None:
            raise RuntimeError("qualification primary files are unavailable")

        self._phase = "source_verify"
        source_first = yield from self._call("fstat", source_fd)
        if _stat_revision(source_first, self._manifest.source, "source descriptor") != source_revision:
            raise ValueError("source changed during qualification")
        source_named = yield from self._call("lstat", self._manifest.source)
        if _stat_revision(source_named, self._manifest.source, "source path") != source_revision:
            raise ValueError("source path changed during qualification")
        source_second = yield from self._call("fstat", source_fd)
        if _stat_revision(source_second, self._manifest.source, "source descriptor") != source_revision:
            raise ValueError("source changed during qualification")

        self._phase = "spool_verify"
        spool_first = yield from self._call("fstat", spool_fd)
        if _stat_revision(spool_first, self._manifest.spool, "spool descriptor") != spool_revision:
            raise ValueError("event spool changed during qualification")
        _validate_spool_manifest(spool_revision, self._manifest)
        spool_named = yield from self._call("lstat", self._manifest.spool)
        if _stat_revision(spool_named, self._manifest.spool, "spool path") != spool_revision:
            raise ValueError("event spool path changed during qualification")
        spool_second = yield from self._call("fstat", spool_fd)
        if _stat_revision(spool_second, self._manifest.spool, "spool descriptor") != spool_revision:
            raise ValueError("event spool changed during qualification")

    def _close_primary(self) -> Generator[object, object, None]:
        self._phase = "primary_close"
        for attribute in ("_source_fd", "_spool_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            yield from self._close_owned(attribute, descriptor)

    def _publish(self, status: str) -> None:
        if status == "qualified":
            task_references: Path | None = self._task_reference_path
            sequence_references: Path | None = self._sequence_reference_path
        else:
            task_references = None
            sequence_references = None
        self._result = QualifiedSource(status, self._manifest, task_references, sequence_references)
        self._is_complete = True
        self._phase = "complete"

    def _audit_for_kind(self, kind: str) -> DigestAuditDriver | None:
        if kind == "task":
            return self._task_audit
        if kind == "sequence":
            return self._sequence_audit
        return None

    def _assert_active_projection(self) -> None:
        if self._close_requested or self._is_complete:
            return
        current = self._projection.completed_projection
        if current is None or current != self._manifest:
            raise ValueError("projection changed or closed during source qualification")

    def _cleanup(self) -> Generator[object, object, None]:
        self._phase = "cleanup"
        for kind in ("task", "sequence"):
            audit = self._audit_for_kind(kind)
            if audit is None or audit.is_closed:
                continue
            try:
                audit.request_close()
            except BaseException as exc:
                self._record_cleanup_error(exc)
                continue
            while not audit.is_closed:
                try:
                    yield _AuditRequest(kind)
                except BaseException as exc:
                    self._record_cleanup_error(exc)
                    break
                yield _Boundary("cleanup_audit")

        for attribute in (
            "_task_digest_fd",
            "_task_reference_fd",
            "_sequence_digest_fd",
            "_sequence_reference_fd",
            "_source_fd",
            "_spool_fd",
        ):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            try:
                yield from self._close_owned(attribute, descriptor)
            except BaseException as exc:
                self._record_cleanup_error(exc)

    def _close_owned(self, attribute: str, descriptor: int) -> Generator[object, object, None]:
        yield _IORequest("close", (descriptor,), attribute)

    def _call(self, method: str, *args: object) -> Generator[object, object, object]:
        return_value = yield _IORequest(method, tuple(args))
        return return_value

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
    return Path(os.path.abspath(os.path.normpath(raw)))


def _validate_max_events(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_events must be an exact integer")
    if not 1 <= value <= _MAX_EVENTS:
        raise ValueError("max_events must be between 1 and 64")
    return value


def _validate_u64(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _U64_MAX:
        raise ValueError(f"{label} must fit an unsigned 64-bit integer")
    return value


def _stat_revision(info: os.stat_result, path: Path, label: str) -> SourceRevision:
    _require_regular(info.st_mode, path, label)
    revision = SourceRevision.from_stat(info)
    _validate_u64(revision.device, f"{label}.device")
    _validate_u64(revision.inode, f"{label}.inode")
    _validate_u64(revision.size, f"{label}.size")
    return revision


def _validate_spool_manifest(revision: SourceRevision, manifest: CompletedProjection) -> None:
    _validate_u64(manifest.spool_device, "manifest spool device")
    _validate_u64(manifest.spool_inode, "manifest spool inode")
    _validate_u64(manifest.spool_size, "manifest spool size")
    _validate_u64(manifest.spool_events, "manifest spool events")
    if (
        revision.device != manifest.spool_device
        or revision.inode != manifest.spool_inode
        or revision.size != manifest.spool_size
    ):
        raise ValueError("event spool does not match the completed projection")


def _require_regular(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


__all__ = ["QualifiedSource", "QualificationStep", "SourceQualification"]
