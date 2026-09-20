"""Internal runtime cooperative, restartable source projection driver.

This extraction is the I/O boundary around the pure submission projection.  A
driver owns one source descriptor, a stable scratch lock, the provisional
JSONL event spool, and the version-one checkpoint envelope.  It performs no
I/O in its constructor: callers advance it with a :class:`SliceIO` budget.

The private executor is a generator of one-syscall requests.  The generator
is deliberately kept alive between calls so ``YIELD`` leaves the same request
pending.  Event data, summaries, and schema checks remain provisional source
extraction; this module does not provide uniqueness, authority, or publication
certification.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import stat
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Literal

from .checkpoint import (
    _CHECKPOINT_VERSION,
    _encode_json,
    _reject_duplicate_pairs,
    _reject_json_constant,
    _revision_dict,
    _snapshot_offset,
    _validate_cross_consistency,
    _validate_envelope,
)
from .json_stream import Scanner, StepResult
from .slice_io import YIELD, SliceIO
from .source_revision import SourceChangedError, SourceRevision
from .submission_projection import (
    SUPPORTED_SOURCE_SCHEMA_VERSION,
    FieldChunk,
    FieldEnd,
    ProjectionSummary,
    SubmissionProjection,
)

_MAX_IO_CHUNK = 65_536
_MAX_EVENT_QUEUE = 4
_LOCK_FLAGS = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_SOURCE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_SPOOL_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_SPOOL_RESUME_FLAGS = os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_CHECKPOINT_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_CHECKPOINT_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


@dataclass(frozen=True, slots=True)
class DriverStep:
    """The bounded result of one cooperative driver advance."""

    state: Literal["progressed", "waiting", "complete", "closed"]
    phase: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class CompletedProjection:
    """Read-only manifest for a completed, still-open source projection."""

    source: Path
    spool: Path
    source_revision: SourceRevision
    spool_device: int
    spool_inode: int
    spool_size: int
    spool_events: int
    operation_id: str
    group: str
    summary: ProjectionSummary


@dataclass(frozen=True, slots=True)
class _IORequest:
    method: str
    args: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _Boundary:
    kind: str
    reason: str | None = None


class _Hold:
    __slots__ = ()


class _StopRequested(Exception):
    pass


class _InputStarved(Exception):
    pass


class _InputAdapter:
    """Expose delivered source bytes to ``Scanner`` without doing I/O."""

    __slots__ = (
        "expected_size",
        "physical_offset",
        "delivered_offset",
        "processed_offset",
        "pending",
        "eof",
    )

    def __init__(self, offset: int, expected_size: int) -> None:
        self.expected_size = expected_size
        self.physical_offset = offset
        self.delivered_offset = offset
        self.processed_offset = offset
        self.pending = bytearray()
        self.eof = False

    def tell(self) -> int:
        return self.processed_offset

    def read(self, size: int) -> bytes:
        if type(size) is not int or size <= 0:
            raise ValueError("scanner requested an invalid read size")
        if self.pending:
            take = min(size, len(self.pending))
            data = bytes(self.pending[:take])
            del self.pending[:take]
            self.delivered_offset += take
            return data
        if self.eof:
            return b""
        raise _InputStarved("driver did not admit a physical source read")

    def add_physical(self, data: bytes) -> None:
        if type(data) is not bytes:
            raise TypeError("source read must return bytes")
        if not data:
            if self.physical_offset != self.expected_size:
                raise SourceChangedError("source ended before its bound size")
            self.eof = True
            return
        end = self.physical_offset + len(data)
        if end > self.expected_size:
            raise SourceChangedError("source returned bytes beyond its bound size")
        self.physical_offset = end
        self.pending.extend(data)

    def set_processed(self, offset: int) -> None:
        if type(offset) is not int or offset < self.processed_offset or offset > self.delivered_offset:
            raise ValueError("scanner processed offset is inconsistent")
        self.processed_offset = offset

    def known_available(self) -> int:
        return self.delivered_offset - self.processed_offset + len(self.pending)


class ProjectionDriver:
    """Advance a bounded source projection using caller-owned ``SliceIO``."""

    def __init__(
        self,
        source: Path,
        scratch: Path,
        expected_operation_id: str,
        expected_group: str,
        *,
        hook: Callable[[str], None] | None = None,
    ) -> None:
        self._source_path = _normalized_absolute(source, "source")
        self._scratch_path = _normalized_absolute(scratch, "scratch")
        self._lock_path = self._scratch_path.with_name(self._scratch_path.name + ".lock")
        self._checkpoint_path = self._scratch_path / "checkpoint.json"
        self._checkpoint_temp_path = self._scratch_path / "checkpoint.tmp"
        try:
            self._source_path.relative_to(self._scratch_path)
        except ValueError:
            pass
        else:
            raise ValueError("source must not be inside scratch")
        if self._source_path == self._lock_path:
            raise ValueError("source must not equal the scratch lock path")
        if hook is not None and not callable(hook):
            raise TypeError("hook must be callable or None")

        # Construction must validate the same caller context as the pure
        # projection while keeping callbacks side-effect free.
        context = SubmissionProjection(expected_operation_id, expected_group, lambda _value: None, lambda _value: None)
        self._expected_operation_id = context._expected_operation_id
        self._expected_group = context._expected_group
        self._hook = hook

        self._generator: Generator[object, object, None] | None = None
        self._pending: object | None = None
        self._boundary_fresh = False
        self._started = False
        self._stop_injected = False
        self._close_requested = False
        self._checkpoint_requested = False
        self._busy = False
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._phase = "initial"
        self._current_io: SliceIO | None = None
        self._soft_deadline: float | None = None
        self._advance_progress = False
        self._advance_quantum = 0
        self._quantum_used = 0

        self._lock_fd: int | None = None
        self._directory_fd: int | None = None
        self._source_fd: int | None = None
        self._spool_fd: int | None = None
        self._checkpoint_fd: int | None = None
        self._temp_fd: int | None = None
        self._temp_path_exists = False

        self._source_revision: SourceRevision | None = None
        self._spool_device = 0
        self._spool_inode = 0
        self._spool_size = 0
        self._spool_events = 0
        self._published_spool_size = 0
        self._published_spool_events = 0
        self._checkpoint_generation = 0

        self._projection: SubmissionProjection | None = None
        self._scanner: Scanner | None = None
        self._input: _InputAdapter | None = None
        self._processed_offset = 0
        self._initialized = False
        self._scanner_complete = False
        self._pending_summary: ProjectionSummary | None = None
        self._summary: ProjectionSummary | None = None
        self._is_complete = False
        self._event_queue: deque[bytes] = deque()
        self._current_event: bytes | None = None
        self._current_event_offset = 0
        self._step_event_count = 0

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def completed_projection(self) -> CompletedProjection | None:
        """Return the completed projection manifest while the driver is open."""

        if self._closed or not self._is_complete:
            return None
        revision = self._source_revision
        summary = self._summary
        if revision is None or summary is None:
            return None
        return CompletedProjection(
            source=self._source_path,
            spool=self._scratch_path / "events.jsonl",
            source_revision=revision,
            spool_device=self._spool_device,
            spool_inode=self._spool_inode,
            spool_size=self._spool_size,
            spool_events=self._spool_events,
            operation_id=self._expected_operation_id,
            group=self._expected_group,
            summary=summary,
        )

    @property
    def summary(self) -> ProjectionSummary | None:
        return self._summary

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def processed_offset(self) -> int:
        return self._processed_offset

    @property
    def checkpoint_generation(self) -> int:
        return self._checkpoint_generation

    def request_checkpoint(self) -> None:
        """Request one durable checkpoint without performing I/O."""

        self._require_open()
        if self._close_requested:
            raise ValueError("driver is closing")
        self._checkpoint_requested = True

    def request_close(self) -> None:
        """Request accounted cleanup without publishing speculative state."""

        if self._closed:
            return
        if self._close_requested:
            return
        self._close_requested = True
        self._checkpoint_requested = False
        self._summary = None
        self._pending_summary = None
        self._is_complete = False
        if not self._started:
            self._closed = True

    def advance(
        self,
        io: SliceIO,
        *,
        max_processed_bytes: int = _MAX_IO_CHUNK,
        soft_deadline: float | None = None,
    ) -> DriverStep:
        """Perform bounded parser and one-syscall progress."""

        self._require_open()
        if self._busy:
            raise RuntimeError("projection driver advance is busy")
        if not isinstance(io, SliceIO):
            raise TypeError("io must be a SliceIO")
        if type(max_processed_bytes) is not int:
            raise TypeError("max_processed_bytes must be an exact int")
        if max_processed_bytes <= 0:
            raise ValueError("max_processed_bytes must be positive")
        if soft_deadline is not None:
            if isinstance(soft_deadline, bool) or not isinstance(soft_deadline, (int, float)):
                raise TypeError("soft_deadline must be a real number or None")

        if self._close_requested and not self._started:
            self._closed = True
            return DriverStep("closed", "closed", None)

        self._busy = True
        self._current_io = io
        self._soft_deadline = float(soft_deadline) if soft_deadline is not None else None
        self._advance_quantum = max_processed_bytes
        self._quantum_used = 0
        self._advance_progress = False
        try:
            if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                return DriverStep("waiting", self._phase, "deadline")
            if self._generator is None:
                self._generator = self._run()
            return self._drive()
        finally:
            self._current_io = None
            self._soft_deadline = None
            self._busy = False

    def _drive(self) -> DriverStep:
        while True:
            if self._closed:
                if self._fatal_error is not None:
                    error = self._fatal_error
                    self._fatal_error = None
                    raise error
                return DriverStep("closed", self._phase, None)

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
                        return DriverStep("waiting", self._phase, pending.reason)
                    if pending.kind == "hold" and self._is_complete:
                        return DriverStep("complete", self._phase, None)
                    return DriverStep("progressed", self._phase, None)
                self._pending = None
                self._receive(None)
                continue

            if isinstance(pending, _Hold):
                if self._close_requested and not self._stop_injected:
                    self._inject_stop()
                    continue
                if self._checkpoint_requested:
                    self._pending = None
                    self._receive(None)
                    continue
                return DriverStep("complete" if self._is_complete else "progressed", self._phase, None)

            if not isinstance(pending, _IORequest):
                raise RuntimeError("driver executor yielded an unknown request")
            if self._close_requested and not self._stop_injected and self._phase != "cleanup":
                self._inject_stop()
                continue
            if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                return DriverStep("waiting", self._phase, "deadline")
            current_io = self._current_io
            if current_io is None:
                raise RuntimeError("driver has no active SliceIO")
            try:
                result = getattr(current_io, pending.method)(*pending.args)
            except BaseException as exc:
                self._pending = None
                self._throw(exc)
                continue
            if result is YIELD:
                return DriverStep("waiting", self._phase, "budget")
            self._advance_progress = True
            self._pending = None
            self._receive(result)

    def _receive(self, value: object, *, start: bool = False) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("driver executor is unavailable")
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
            self._summary = None
            self._pending_summary = None
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
            raise RuntimeError("driver executor is unavailable")
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
            self._summary = None
            self._pending_summary = None
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
                    initialized = yield from self._initialize_attempt()
                    if not initialized:
                        self._phase = "lock"
                        yield _Boundary("wait", "scratch_busy")
                        continue
                    self._initialized = True
                    yield _Boundary("initialized")
                    continue

                if self._is_complete:
                    if self._checkpoint_requested:
                        yield from self._checkpoint()
                        yield _Boundary("checkpoint")
                        continue
                    yield _Hold()
                    continue

                if self._event_queue or self._current_event is not None:
                    yield from self._drain_events()
                    yield _Boundary("events")
                    continue

                if self._checkpoint_requested:
                    yield from self._checkpoint()
                    if self._pending_summary is not None and not self._close_requested:
                        self._summary = self._pending_summary
                        self._pending_summary = None
                        self._is_complete = True
                    yield _Boundary("checkpoint")
                    continue

                if self._quantum_used >= self._advance_quantum:
                    yield _Boundary("quantum", "budget")
                    continue

                yield from self._parse_one()
                reason = "budget" if self._quantum_used >= self._advance_quantum else None
                yield _Boundary("parse", reason)
        except _StopRequested:
            pass
        finally:
            yield from self._cleanup()

    def _initialize_attempt(self) -> Generator[object, object, bool]:
        self._phase = "lock"
        lock = yield from self._call("open", self._lock_path, _LOCK_FLAGS, 0o600)
        if type(lock) is not int:
            raise TypeError("lock open returned an invalid descriptor")
        self._lock_fd = lock
        lock_stat = yield from self._call("fstat", lock)
        _require_regular(lock_stat.st_mode, self._lock_path, "scratch lock")
        try:
            yield from self._call("flock", lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._phase = "lock"
            self._lock_fd = None
            yield from self._cleanup_call("close", lock)
            return False

        self._phase = "scratch"
        try:
            yield from self._call("mkdir", self._scratch_path, 0o700)
        except FileExistsError:
            pass
        directory = yield from self._call("open", self._scratch_path, _DIRECTORY_FLAGS, 0o700)
        if type(directory) is not int:
            raise TypeError("scratch open returned an invalid descriptor")
        self._directory_fd = directory
        directory_stat = yield from self._call("fstat", directory)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(f"scratch is not a directory: {self._scratch_path}")

        self._phase = "checkpoint_read"
        checkpoint_fd: int | None = None
        try:
            checkpoint_fd = yield from self._call("open", self._checkpoint_path, _CHECKPOINT_READ_FLAGS, 0o600)
        except FileNotFoundError:
            return (yield from self._initialize_new())
        if type(checkpoint_fd) is not int:
            raise TypeError("checkpoint open returned an invalid descriptor")
        self._checkpoint_fd = checkpoint_fd
        checkpoint = yield from self._read_checkpoint(checkpoint_fd)
        self._checkpoint_fd = None
        yield from self._cleanup_call("close", checkpoint_fd)
        self._phase = "resume"
        return (yield from self._initialize_resume(checkpoint))

    def _initialize_new(self) -> Generator[object, object, bool]:
        self._phase = "initialize"
        for path in (self._scratch_path / "events.jsonl", self._scratch_path / "checkpoint.tmp"):
            try:
                yield from self._call("unlink", path)
            except FileNotFoundError:
                pass

        spool = yield from self._call("open", self._scratch_path / "events.jsonl", _SPOOL_CREATE_FLAGS, 0o600)
        if type(spool) is not int:
            raise TypeError("spool open returned an invalid descriptor")
        self._spool_fd = spool
        spool_stat = yield from self._call("fstat", spool)
        _require_regular(spool_stat.st_mode, self._scratch_path / "events.jsonl", "spool")
        self._spool_device = spool_stat.st_dev
        self._spool_inode = spool_stat.st_ino
        self._spool_size = 0
        self._spool_events = 0
        self._published_spool_size = 0
        self._published_spool_events = 0

        self._source_revision = yield from self._bind_source(None)
        self._input = _InputAdapter(0, self._source_revision.size)
        self._projection = self._new_projection()
        self._scanner = Scanner(
            self._input,
            lambda _span: None,
            chunk_bytes=_MAX_IO_CHUNK,
            emit_bytes=self._projection.feed,
        )
        self._processed_offset = 0
        self._scanner_complete = False
        yield from self._checkpoint()
        self._checkpoint_generation = 1
        return True

    def _initialize_resume(self, envelope: dict[str, object]) -> Generator[object, object, bool]:
        source_revision = envelope["source_revision"]
        if not isinstance(source_revision, SourceRevision):
            raise ValueError("checkpoint source revision is invalid")
        scanner_snapshot = envelope["scanner"]
        projection_snapshot = envelope["projection"]
        if type(scanner_snapshot) is not dict or type(projection_snapshot) is not dict:
            raise ValueError("checkpoint parser snapshots must be objects")
        scanner_offset = _snapshot_offset(scanner_snapshot)
        self._source_revision = yield from self._bind_source(source_revision)
        yield from self._call("lseek", self._source_fd, scanner_offset, os.SEEK_SET)

        self._input = _InputAdapter(scanner_offset, source_revision.size)
        self._projection = SubmissionProjection.from_snapshot(
            self._expected_operation_id,
            self._expected_group,
            self._emit_chunk,
            self._emit_end,
            projection_snapshot,
        )
        self._scanner = Scanner.from_snapshot(
            self._input,
            lambda _span: None,
            scanner_snapshot,
            emit_bytes=self._projection.feed,
        )
        _validate_cross_consistency(scanner_snapshot, projection_snapshot, source_revision)
        self._processed_offset = scanner_offset
        self._scanner_complete = bool(scanner_snapshot["is_complete"])

        restored_summary = None
        if projection_snapshot.get("summary") is not None:
            if self._projection.source_schema_version != SUPPORTED_SOURCE_SCHEMA_VERSION:
                raise ValueError("unsupported or missing source schema version")
            restored_summary = self._projection.finish()

        spool_checkpoint = envelope["spool"]
        if type(spool_checkpoint) is not dict:
            raise ValueError("checkpoint spool is invalid")
        spool = yield from self._call("open", self._scratch_path / "events.jsonl", _SPOOL_RESUME_FLAGS, 0o600)
        if type(spool) is not int:
            raise TypeError("spool open returned an invalid descriptor")
        self._spool_fd = spool
        first = yield from self._call("fstat", spool)
        _require_regular(first.st_mode, self._scratch_path / "events.jsonl", "spool descriptor")
        if (first.st_dev, first.st_ino) != (spool_checkpoint["device"], spool_checkpoint["inode"]):
            raise ValueError("spool identity does not match its checkpoint")
        named = yield from self._call("lstat", self._scratch_path / "events.jsonl")
        _require_regular(named.st_mode, self._scratch_path / "events.jsonl", "spool path")
        if (named.st_dev, named.st_ino) != (first.st_dev, first.st_ino):
            raise ValueError("spool path was replaced")
        saved_size = spool_checkpoint["size"]
        saved_events = spool_checkpoint["events"]
        if type(saved_size) is not int or type(saved_events) is not int or saved_size < 0 or saved_events < 0:
            raise ValueError("checkpoint spool counters are invalid")
        if first.st_size < saved_size:
            raise ValueError("spool is shorter than its checkpoint")
        second = yield from self._call("fstat", spool)
        if (second.st_dev, second.st_ino) != (first.st_dev, first.st_ino) or second.st_size < saved_size:
            raise ValueError("spool changed while opening")
        self._spool_device = second.st_dev
        self._spool_inode = second.st_ino
        self._spool_size = saved_size
        self._spool_events = saved_events
        self._published_spool_size = saved_size
        self._published_spool_events = saved_events
        yield from self._call("ftruncate", spool, saved_size)
        yield from self._call("lseek", spool, saved_size, os.SEEK_SET)

        if restored_summary is not None:
            self._summary = restored_summary
            self._is_complete = True
        self._checkpoint_generation = 1
        return True

    def _read_checkpoint(self, descriptor: int) -> Generator[object, object, dict[str, object]]:
        data = bytearray()
        while True:
            chunk = yield from self._call("read", descriptor, _MAX_IO_CHUNK)
            if type(chunk) is not bytes:
                raise TypeError("checkpoint read returned an invalid result")
            if not chunk:
                break
            data.extend(chunk)
        first = yield from self._call("fstat", descriptor)
        _require_regular(first.st_mode, self._checkpoint_path, "checkpoint descriptor")
        named = yield from self._call("lstat", self._checkpoint_path)
        _require_regular(named.st_mode, self._checkpoint_path, "checkpoint path")
        if (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("checkpoint was replaced")
        second = yield from self._call("fstat", descriptor)
        if (second.st_dev, second.st_ino) != (first.st_dev, first.st_ino) or second.st_size != first.st_size:
            raise ValueError("checkpoint changed while reading")
        try:
            value = json.loads(
                bytes(data),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("checkpoint is not valid UTF-8 JSON") from exc
        return _validate_envelope(value, self._source_path, self._expected_operation_id, self._expected_group)

    def _bind_source(self, expected: SourceRevision | None) -> Generator[object, object, SourceRevision]:
        try:
            descriptor = yield from self._call("open", self._source_path, _SOURCE_FLAGS, 0o600)
        except OSError as exc:
            if expected is not None:
                raise SourceChangedError(f"expected source is unavailable: {self._source_path}") from exc
            raise
        if type(descriptor) is not int:
            raise TypeError("source open returned an invalid descriptor")
        self._source_fd = descriptor
        first = yield from self._call("fstat", descriptor)
        _require_regular(first.st_mode, self._source_path, "source descriptor")
        revision = SourceRevision.from_stat(first)
        if expected is not None and revision != expected:
            raise SourceChangedError(f"source revision changed for {self._source_path}")
        try:
            named = yield from self._call("lstat", self._source_path)
        except OSError as exc:
            raise SourceChangedError(f"source changed while opening {self._source_path}") from exc
        _require_regular(named.st_mode, self._source_path, "source path")
        named_revision = SourceRevision.from_stat(named)
        if named_revision != revision:
            raise SourceChangedError(f"source changed while opening {self._source_path}")
        second = yield from self._call("fstat", descriptor)
        _require_regular(second.st_mode, self._source_path, "source descriptor")
        second_revision = SourceRevision.from_stat(second)
        if second_revision != revision:
            raise SourceChangedError(f"source changed while opening {self._source_path}")
        return revision

    def _verify_source(self) -> Generator[object, object, None]:
        descriptor = self._source_fd
        expected = self._source_revision
        if descriptor is None or expected is None:
            raise ValueError("source is not bound")
        first = yield from self._call("fstat", descriptor)
        _require_regular(first.st_mode, self._source_path, "source descriptor")
        if SourceRevision.from_stat(first) != expected:
            raise SourceChangedError(f"source revision changed for {self._source_path}")
        try:
            named = yield from self._call("lstat", self._source_path)
        except OSError as exc:
            raise SourceChangedError(f"source revision changed for {self._source_path}") from exc
        _require_regular(named.st_mode, self._source_path, "source path")
        if SourceRevision.from_stat(named) != expected:
            raise SourceChangedError(f"source revision changed for {self._source_path}")
        second = yield from self._call("fstat", descriptor)
        _require_regular(second.st_mode, self._source_path, "source descriptor")
        if SourceRevision.from_stat(second) != expected:
            raise SourceChangedError(f"source revision changed for {self._source_path}")

    def _new_projection(self) -> SubmissionProjection:
        return SubmissionProjection(
            self._expected_operation_id,
            self._expected_group,
            self._emit_chunk,
            self._emit_end,
        )

    def _parse_one(self) -> Generator[object, object, None]:
        scanner = self._scanner
        adapter = self._input
        projection = self._projection
        if scanner is None or adapter is None or projection is None:
            raise RuntimeError("parser is not initialized")
        if adapter.known_available() == 0 and not adapter.eof:
            request_size = _MAX_IO_CHUNK
            if adapter.physical_offset < adapter.expected_size:
                request_size = min(request_size, adapter.expected_size - adapter.physical_offset)
            else:
                request_size = 1
            self._phase = "source_read"
            data = yield from self._call("read", self._source_fd, request_size)
            adapter.add_physical(data)
            return

        remaining = self._advance_quantum - self._quantum_used
        if remaining <= 0:
            return
        known = adapter.known_available()
        if known == 0 and adapter.eof:
            step_budget = 1
        else:
            step_budget = min(_MAX_IO_CHUNK, remaining, known)
            if step_budget <= 0:
                return
        self._phase = "parse"
        self._step_event_count = 0
        result: StepResult = scanner.step(step_budget, max_fragments=2)
        if (
            type(result.bytes_processed) is not int
            or result.bytes_processed < 0
            or result.bytes_processed > step_budget
        ):
            raise ValueError("scanner returned an invalid processed-byte count")
        self._processed_offset += result.bytes_processed
        adapter.set_processed(self._processed_offset)
        self._quantum_used += result.bytes_processed
        if self._step_event_count > _MAX_EVENT_QUEUE:
            raise RuntimeError("scanner emitted too many provisional events in one step")
        self._scanner_complete = result.is_complete
        if result.is_complete:
            summary = projection.finish()
            if projection.source_schema_version != SUPPORTED_SOURCE_SCHEMA_VERSION:
                raise ValueError("unsupported or missing source schema version")
            yield from self._verify_source()
            self._pending_summary = summary
            self._checkpoint_requested = True

    def _drain_events(self) -> Generator[object, object, None]:
        descriptor = self._spool_fd
        if descriptor is None:
            raise RuntimeError("spool is not open")
        while self._event_queue or self._current_event is not None:
            if self._current_event is None:
                self._current_event = self._event_queue.popleft()
                self._current_event_offset = 0
            event = self._current_event
            if event is None:
                continue
            start = self._current_event_offset
            end = min(len(event), start + _MAX_IO_CHUNK)
            self._phase = "spool_write"
            written = yield from self._call("write", descriptor, event[start:end])
            if type(written) is not int or written <= 0 or written > end - start:
                raise OSError("short or invalid spool write")
            self._spool_size += written
            self._current_event_offset += written
            self._call_hook("spool_write")
            if self._current_event_offset == len(event):
                self._current_event = None
                self._current_event_offset = 0
                self._spool_events += 1

    def _checkpoint(self) -> Generator[object, object, None]:
        self._checkpoint_requested = False
        self._phase = "checkpoint"
        if self._event_queue or self._current_event is not None:
            yield from self._drain_events()
        yield from self._verify_source()
        yield from self._verify_spool(exact_size=True)
        yield from self._call("fsync", self._spool_fd)
        self._call_hook("spool_fsync")

        scanner = self._scanner
        projection = self._projection
        revision = self._source_revision
        if scanner is None or projection is None or revision is None:
            raise RuntimeError("checkpoint parser state is incomplete")
        scanner_snapshot = scanner.snapshot()
        projection_snapshot = projection.snapshot()
        _validate_cross_consistency(scanner_snapshot, projection_snapshot, revision)
        envelope = {
            "version": _CHECKPOINT_VERSION,
            "source_path": str(self._source_path),
            "source_revision": _revision_dict(revision),
            "expected_operation_id": self._expected_operation_id,
            "expected_group": self._expected_group,
            "spool": {
                "device": self._spool_device,
                "inode": self._spool_inode,
                "size": self._spool_size,
                "events": self._spool_events,
            },
            "scanner": scanner_snapshot,
            "projection": projection_snapshot,
        }
        payload = _encode_json(envelope)
        try:
            yield from self._call("unlink", self._scratch_path / "checkpoint.tmp")
        except FileNotFoundError:
            pass
        temp = yield from self._call("open", self._scratch_path / "checkpoint.tmp", _CHECKPOINT_TEMP_FLAGS, 0o600)
        if type(temp) is not int:
            raise TypeError("checkpoint temp open returned an invalid descriptor")
        self._temp_fd = temp
        self._temp_path_exists = True
        temp_stat = yield from self._call("fstat", temp)
        _require_regular(temp_stat.st_mode, self._scratch_path / "checkpoint.tmp", "checkpoint temp")
        temp_named = yield from self._call("lstat", self._scratch_path / "checkpoint.tmp")
        _require_regular(temp_named.st_mode, self._scratch_path / "checkpoint.tmp", "checkpoint temp")
        if (temp_stat.st_dev, temp_stat.st_ino) != (temp_named.st_dev, temp_named.st_ino):
            raise ValueError("checkpoint temp was replaced")
        offset = 0
        while offset < len(payload):
            self._phase = "checkpoint_write"
            written = yield from self._call("write", temp, payload[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise OSError("short or invalid checkpoint write")
            offset += written
            self._call_hook("checkpoint_write")
        self._phase = "checkpoint_fsync"
        yield from self._call("fsync", temp)
        self._call_hook("checkpoint_fsync")
        self._temp_fd = None
        yield from self._cleanup_call("close", temp)
        self._phase = "checkpoint"
        yield from self._verify_source()
        self._phase = "checkpoint_replace"
        yield from self._call("replace", self._scratch_path / "checkpoint.tmp", self._scratch_path / "checkpoint.json")
        self._temp_path_exists = False
        # Replacement publishes a checkpoint that already references file-synced
        # events. Close must retain that prefix even before directory fsync:
        # either the old or new checkpoint can then recover after interruption.
        self._published_spool_size = self._spool_size
        self._published_spool_events = self._spool_events
        self._call_hook("checkpoint_replace")
        self._phase = "directory_fsync"
        yield from self._call("fsync", self._directory_fd)
        self._call_hook("directory_fsync")
        self._checkpoint_generation += 1

    def _verify_spool(self, *, exact_size: bool) -> Generator[object, object, None]:
        descriptor = self._spool_fd
        if descriptor is None:
            raise RuntimeError("spool is not open")
        path = self._scratch_path / "events.jsonl"
        first = yield from self._call("fstat", descriptor)
        _require_regular(first.st_mode, path, "spool descriptor")
        if (first.st_dev, first.st_ino) != (self._spool_device, self._spool_inode):
            raise ValueError("spool identity changed")
        named = yield from self._call("lstat", path)
        _require_regular(named.st_mode, path, "spool path")
        if (named.st_dev, named.st_ino) != (first.st_dev, first.st_ino):
            raise ValueError("spool path was replaced")
        if exact_size and first.st_size != self._spool_size:
            raise ValueError("spool size changed before checkpoint")
        if not exact_size and first.st_size < self._spool_size:
            raise ValueError("spool is shorter than its checkpoint")
        second = yield from self._call("fstat", descriptor)
        _require_regular(second.st_mode, path, "spool descriptor")
        if (second.st_dev, second.st_ino) != (first.st_dev, first.st_ino):
            raise ValueError("spool descriptor changed")
        if exact_size and second.st_size != self._spool_size:
            raise ValueError("spool size changed before checkpoint")
        if not exact_size and second.st_size < self._spool_size:
            raise ValueError("spool is shorter than its checkpoint")

    def _cleanup(self) -> Generator[object, object, None]:
        self._phase = "cleanup"
        self._event_queue.clear()
        self._current_event = None
        if self._close_requested and self._spool_fd is not None and self._spool_size > self._published_spool_size:
            descriptor = self._spool_fd
            try:
                yield from self._cleanup_call("ftruncate", descriptor, self._published_spool_size)
                yield from self._cleanup_call("lseek", descriptor, self._published_spool_size, os.SEEK_SET)
            except BaseException as exc:
                self._record_cleanup_error(exc)
        if self._temp_fd is not None:
            descriptor = self._temp_fd
            self._temp_fd = None
            yield from self._cleanup_call("close", descriptor)
        if self._temp_path_exists:
            self._temp_path_exists = False
            try:
                yield from self._cleanup_call("unlink", self._scratch_path / "checkpoint.tmp")
            except BaseException as exc:
                if not isinstance(exc, FileNotFoundError):
                    self._record_cleanup_error(exc)
        for attr in ("_spool_fd", "_source_fd", "_checkpoint_fd", "_directory_fd"):
            descriptor = getattr(self, attr)
            if descriptor is not None:
                setattr(self, attr, None)
                yield from self._cleanup_call("close", descriptor)
        lock = self._lock_fd
        self._lock_fd = None
        if lock is not None:
            yield from self._cleanup_call("flock", lock, fcntl.LOCK_UN)
            yield from self._cleanup_call("close", lock)

    def _cleanup_call(self, method: str, *args: object) -> Generator[object, object, None]:
        self._phase = "cleanup"
        try:
            yield from self._call(method, *args)
        except BaseException as exc:
            self._record_cleanup_error(exc)

    def _call(self, method: str, *args: object) -> Generator[object, object, object]:
        return_value = yield _IORequest(method, tuple(args))
        return return_value

    def _record_cleanup_error(self, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if self._cleanup_error is None:
            self._cleanup_error = error

    def _call_hook(self, name: str) -> None:
        if self._hook is not None:
            self._hook(name)

    def _emit_chunk(self, chunk: FieldChunk) -> None:
        event = {
            "type": "chunk",
            "kind": chunk.kind,
            "ordinal": chunk.ordinal,
            "data_b64": base64.b64encode(chunk.data).decode("ascii"),
            "is_final": chunk.is_final,
        }
        self._queue_event(event)

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
        self._queue_event(event)

    def _queue_event(self, event: dict[str, object]) -> None:
        self._step_event_count += 1
        if self._step_event_count > _MAX_EVENT_QUEUE:
            raise RuntimeError("scanner emitted more than four provisional events")
        self._event_queue.append(_encode_json(event) + b"\n")

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("projection driver is closed")


def _normalized_absolute(value: Path, name: str) -> Path:
    try:
        candidate = Path(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a path") from exc
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be absolute")
    normalized = Path(os.path.normpath(str(candidate)))
    if candidate != normalized:
        raise ValueError(f"{name} must be normalized")
    return candidate


def _require_regular(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


__all__ = ["CompletedProjection", "DriverStep", "ProjectionDriver"]
