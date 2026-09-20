"""Cooperative bounded-memory uniqueness audit for fixed-width digests.

The audit consumes a caller-owned file of concatenated 32-byte digest values.
It creates sorted runs in an exclusively created scratch directory and merges
those runs pairwise.  Equal values are ambiguous: an audit never publishes a
unique result after observing an equality.  Scratch files are provisional and
non-resumable; callers own disposal of the retained scratch directory.
"""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

from .slice_io import YIELD, SliceIO
from .source_revision import SourceChangedError, SourceRevision

_DIGEST_BYTES = 32
_MAX_RUN_RECORDS = 64
_SOURCE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_RUN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_EOF = object()


@dataclass(frozen=True, slots=True)
class AuditStep:
    """The bounded result of one cooperative audit advance."""

    state: Literal["progressed", "waiting", "complete", "closed"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class _IORequest:
    method: str
    args: tuple[object, ...]
    owner_attr: str | None = None


@dataclass(frozen=True, slots=True)
class _Boundary:
    state: Literal["progressed", "waiting"]
    reason: str | None = None


class _StopRequested(Exception):
    pass


class DigestAuditDriver:
    """Incrementally audit uniqueness of a fixed-width digest source.

    Construction only validates arguments and normalizes paths.  Source and
    scratch filesystem access starts with :meth:`advance`, and every such
    operation is admitted through the caller's :class:`SliceIO`.
    """

    def __init__(
        self,
        source: Path,
        scratch: Path,
        *,
        expected_count: int,
        run_records: int = _MAX_RUN_RECORDS,
    ) -> None:
        self._source_path = _normalize_absolute(source, "source")
        self._scratch_path = _normalize_absolute(scratch, "scratch")
        self._expected_count = _validate_nonnegative_int(expected_count, "expected_count")
        self._expected_size = self._expected_count * _DIGEST_BYTES
        self._run_records = _validate_run_records(run_records)

        self._generator: Generator[object, object, None] | None = None
        self._pending: object | None = None
        self._boundary_fresh = False
        self._started = False
        self._stop_injected = False
        self._close_requested = False
        self._busy = False
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._phase = "initial"
        self._current_io: SliceIO | None = None
        self._soft_deadline: float | None = None
        self._advance_max_records = _MAX_RUN_RECORDS
        self._records_read = 0
        self._records_written = 0

        self._source_fd: int | None = None
        self._source_revision: SourceRevision | None = None
        self._source_records_read = 0
        self._source_partial = bytearray()

        self._forming_buffer: list[bytes] = []
        self._forming_output_fd: int | None = None
        self._forming_output_path: Path | None = None
        self._forming_write_index = 0
        self._run_count = 0

        self._merge_pass = 0
        self._pair_index = 0
        self._left_fd: int | None = None
        self._right_fd: int | None = None
        self._merge_output_fd: int | None = None
        self._merge_output_path: Path | None = None
        self._left_partial = bytearray()
        self._right_partial = bytearray()
        self._left_eof = False
        self._right_eof = False
        self._left_head: bytes | None = None
        self._right_head: bytes | None = None
        self._merge_previous: bytes | None = None
        self._pending_merge_record: bytes | None = None
        self._pending_merge_offset = 0

        self._terminal_kind: Literal["unique", "ambiguous"] | None = None
        self._result_candidate: Path | None = None
        self._result_path: Path | None = None
        self._is_complete = False
        self._has_repeated_digest = False

    @property
    def is_complete(self) -> bool:
        """Whether the audit reached a published unique or ambiguous result."""

        return self._is_complete

    @property
    def has_repeated_digest(self) -> bool:
        """Whether the completed audit found an equal digest."""

        return self._has_repeated_digest

    @property
    def result_path(self) -> Path | None:
        """The sorted result path for a completed unique audit."""

        return self._result_path

    @property
    def is_closed(self) -> bool:
        """Whether all descriptors have been released and the driver stopped."""

        return self._closed

    def request_close(self) -> None:
        """Request cooperative release of active descriptors without I/O."""

        if self._closed or self._is_complete:
            return
        self._close_requested = True
        self._result_candidate = None
        self._result_path = None
        self._has_repeated_digest = False
        self._is_complete = False
        if not self._started:
            self._closed = True

    def advance(
        self,
        io: SliceIO,
        *,
        max_records: int = _MAX_RUN_RECORDS,
        soft_deadline: float | None = None,
    ) -> AuditStep:
        """Perform bounded read, write, metadata, and cleanup work."""

        if not isinstance(io, SliceIO):
            raise TypeError("io must be a SliceIO")
        max_records = _validate_positive_int(max_records, "max_records")
        if soft_deadline is not None:
            if isinstance(soft_deadline, bool) or not isinstance(soft_deadline, (int, float)):
                raise TypeError("soft_deadline must be a real number or None")

        if self._is_complete:
            return AuditStep("complete", None)
        if self._fatal_error is not None and self._closed:
            error = self._fatal_error
            self._fatal_error = None
            raise error
        if self._closed:
            return AuditStep("closed", None)
        if self._busy:
            raise RuntimeError("audit advance is busy")
        if self._close_requested and not self._started:
            self._closed = True
            return AuditStep("closed", None)

        self._busy = True
        self._current_io = io
        self._soft_deadline = float(soft_deadline) if soft_deadline is not None else None
        self._advance_max_records = max_records
        self._records_read = 0
        self._records_written = 0
        try:
            if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                return AuditStep("waiting", "deadline")
            if self._generator is None:
                self._generator = self._run()
            return self._drive()
        finally:
            self._current_io = None
            self._soft_deadline = None
            self._busy = False

    def _drive(self) -> AuditStep:
        while True:
            if self._is_complete:
                return AuditStep("complete", None)
            if self._closed:
                if self._fatal_error is not None:
                    error = self._fatal_error
                    self._fatal_error = None
                    raise error
                return AuditStep("closed", None)

            if self._pending is None:
                self._receive(None, start=not self._started)
                if self._closed:
                    continue

            pending = self._pending
            if isinstance(pending, _Boundary):
                if self._close_requested and not self._stop_injected and self._phase != "cleanup":
                    self._inject_stop()
                    continue
                if self._boundary_fresh:
                    self._boundary_fresh = False
                    return AuditStep(pending.state, pending.reason)
                self._pending = None
                self._receive(None)
                continue

            if not isinstance(pending, _IORequest):
                raise RuntimeError("audit executor yielded an unknown request")
            if self._close_requested and not self._stop_injected and self._phase != "cleanup":
                self._inject_stop()
                continue
            if self._soft_deadline is not None and time.monotonic() >= self._soft_deadline:
                return AuditStep("waiting", "deadline")
            current_io = self._current_io
            if current_io is None:
                raise RuntimeError("audit has no active SliceIO")
            # A failed Linux close must not retry an integer that another owner
            # may already have reused. Restore ownership only on admission yield,
            # which guarantees no syscall occurred.
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
                return AuditStep("waiting", "budget")
            self._pending = None
            self._receive(result)

    def _receive(self, value: object, *, start: bool = False) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("audit executor is unavailable")
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
            self._record_failure(exc)
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            return
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _throw(self, error: BaseException) -> None:
        generator = self._generator
        if generator is None:
            raise RuntimeError("audit executor is unavailable")
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
            self._record_failure(exc)
            self._pending = None
            self._boundary_fresh = False
            self._closed = True
            return
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _inject_stop(self) -> None:
        if self._stop_injected:
            return
        self._stop_injected = True
        self._pending = None
        self._phase = "cleanup"
        generator = self._generator
        if generator is None:
            self._closed = True
            return
        try:
            result = generator.throw(_StopRequested())
        except StopIteration:
            self._closed = True
            if self._cleanup_error is not None and self._fatal_error is None:
                self._fatal_error = self._cleanup_error
            return
        except BaseException as exc:
            self._record_failure(exc)
            self._closed = True
            return
        self._pending = result
        self._boundary_fresh = isinstance(result, _Boundary)

    def _run(self) -> Generator[object, object, None]:
        try:
            try:
                yield from self._work()
            except _StopRequested:
                self._cancel_publication()
            except BaseException as exc:
                self._record_failure(exc)
                self._cancel_publication()

            if self._close_requested or self._fatal_error is not None or self._stop_injected:
                self._cancel_publication()
            try:
                yield from self._cleanup()
            except BaseException as exc:
                self._record_cleanup_error(exc)

            if self._fatal_error is not None or self._cleanup_error is not None:
                self._cancel_publication()
                if self._fatal_error is None:
                    self._fatal_error = self._cleanup_error
                return
            if self._close_requested or self._stop_injected:
                self._cancel_publication()
                return
            if self._terminal_kind == "ambiguous":
                self._has_repeated_digest = True
                self._is_complete = True
                self._phase = "complete"
            elif self._terminal_kind == "unique":
                self._has_repeated_digest = False
                self._result_path = self._result_candidate
                self._is_complete = True
                self._phase = "complete"
            else:
                return
        finally:
            if self._is_complete:
                self._closed = True

    def _work(self) -> Generator[object, object, None]:
        self._phase = "scratch"
        yield from self._call("mkdir", self._scratch_path, 0o700)
        yield from self._bind_source()
        self._phase = "forming"

        while self._terminal_kind is None:
            yield from self._pause_for_record_budget()
            if self._forming_output_fd is not None:
                yield from self._flush_forming_run()
                yield _Boundary("progressed")
                continue

            if self._source_records_read < self._expected_count:
                record = yield from self._read_source_record()
                if record is _EOF:
                    raise ValueError("source ended before the expected digest count")
                self._forming_buffer.append(record)
                if len(self._forming_buffer) < self._run_records and self._source_records_read < self._expected_count:
                    continue

            if self._forming_buffer:
                self._forming_buffer.sort()
                if _has_adjacent_equal(self._forming_buffer):
                    self._terminal_kind = "ambiguous"
                    break
                self._forming_output_path = self._run_path(0, self._run_count)
                self._forming_output_fd = yield from self._open_output(self._forming_output_path)
                self._forming_write_index = 0
                continue

            if self._source_records_read == self._expected_count:
                break

        if self._terminal_kind == "ambiguous":
            return

        if self._expected_count == 0:
            self._phase = "result"
            result_path = self._scratch_path / "result.bin"
            result_fd = yield from self._open_output(result_path)
            self._merge_output_fd = result_fd
            yield from self._sync_close_fd("_merge_output_fd", result_fd)
            self._result_candidate = result_path
            yield from self._verify_source()
            self._terminal_kind = "unique"
            return

        self._phase = "merging"
        self._merge_pass = 0
        self._pair_index = 0
        while self._run_count > 1 and self._terminal_kind is None:
            yield from self._pause_for_record_budget()
            pair_count = self._run_count // 2
            if self._pair_index < pair_count:
                yield from self._merge_pair()
                if self._terminal_kind == "ambiguous":
                    return
                yield _Boundary("progressed")
                continue
            yield from self._advance_merge_pass()

        if self._terminal_kind == "ambiguous":
            return
        if self._run_count == 1:
            self._result_candidate = self._run_path(self._merge_pass, 0)
        else:
            raise RuntimeError("audit produced no result run")
        yield from self._verify_source()
        self._terminal_kind = "unique"

    def _bind_source(self) -> Generator[object, object, None]:
        descriptor = yield from self._call("open", self._source_path, _SOURCE_FLAGS, 0o600)
        _require_descriptor(descriptor, "source open")
        self._source_fd = descriptor
        first = yield from self._call("fstat", descriptor)
        _require_regular(first.st_mode, self._source_path, "source descriptor")
        revision = SourceRevision.from_stat(first)
        if first.st_size != self._expected_size or first.st_size % _DIGEST_BYTES:
            raise ValueError("source size does not match expected digest count")

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
        self._source_revision = revision

    def _verify_source(self) -> Generator[object, object, None]:
        descriptor = self._source_fd
        expected = self._source_revision
        if descriptor is None or expected is None:
            raise RuntimeError("source is not bound")
        yield from self._pause_for_record_budget()
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

    def _read_source_record(self) -> Generator[object, object, bytes | object]:
        descriptor = self._source_fd
        if descriptor is None:
            raise RuntimeError("source descriptor is unavailable")
        record = yield from self._read_record(descriptor, self._source_partial, "source")
        if record is not _EOF:
            self._source_records_read += 1
        return record

    def _read_run_record(
        self,
        descriptor: int,
        partial: bytearray,
        label: str,
    ) -> Generator[object, object, bytes | object]:
        return (yield from self._read_record(descriptor, partial, label))

    def _read_record(
        self,
        descriptor: int,
        partial: bytearray,
        label: str,
    ) -> Generator[object, object, bytes | object]:
        while len(partial) < _DIGEST_BYTES:
            if self._records_read >= self._advance_max_records:
                yield _Boundary("waiting", "budget")
                continue
            chunk = yield from self._call("read", descriptor, _DIGEST_BYTES - len(partial))
            if type(chunk) is not bytes:
                raise TypeError(f"{label} read returned an invalid result")
            if not chunk:
                if partial:
                    raise ValueError(f"{label} ended with a truncated digest")
                return _EOF
            partial.extend(chunk)
            if len(partial) > _DIGEST_BYTES:
                raise ValueError(f"{label} returned more than one digest")
        record = bytes(partial)
        partial.clear()
        self._records_read += 1
        return record

    def _flush_forming_run(self) -> Generator[object, object, None]:
        fd = self._forming_output_fd
        run = self._forming_buffer
        if fd is None or self._forming_output_path is None:
            raise RuntimeError("formation output is unavailable")
        while self._forming_write_index < len(run):
            yield from self._write_record(fd, run[self._forming_write_index], "formation output")
            self._forming_write_index += 1
        yield from self._pause_for_record_budget()
        yield from self._sync_close_fd("_forming_output_fd", fd)
        self._forming_output_path = None
        self._forming_buffer = []
        self._forming_write_index = 0
        self._run_count += 1

    def _merge_pair(self) -> Generator[object, object, None]:
        left_number = self._pair_index * 2
        left_path = self._run_path(self._merge_pass, left_number)
        right_path = self._run_path(self._merge_pass, left_number + 1)
        output_path = self._run_path(self._merge_pass + 1, self._pair_index)
        self._phase = "merge_open"
        self._left_fd = yield from self._open_input(left_path, "left merge run")
        self._right_fd = yield from self._open_input(right_path, "right merge run")
        self._merge_output_path = output_path
        self._merge_output_fd = yield from self._open_output(output_path)
        self._left_partial.clear()
        self._right_partial.clear()
        self._left_eof = False
        self._right_eof = False
        self._left_head = None
        self._right_head = None
        self._merge_previous = None
        self._pending_merge_record = None
        self._pending_merge_offset = 0

        while True:
            if self._pending_merge_record is not None:
                fd = self._merge_output_fd
                if fd is None:
                    raise RuntimeError("merge output is unavailable")
                yield from self._write_record(
                    fd,
                    self._pending_merge_record,
                    "merge output",
                    offset_attr="_pending_merge_offset",
                )
                self._merge_previous = self._pending_merge_record
                self._pending_merge_record = None
                self._pending_merge_offset = 0
                continue

            if self._left_head is None and not self._left_eof:
                record = yield from self._read_run_record(self._left_fd, self._left_partial, "left merge run")
                if record is _EOF:
                    self._left_eof = True
                else:
                    self._left_head = record
                continue
            if self._right_head is None and not self._right_eof:
                record = yield from self._read_run_record(self._right_fd, self._right_partial, "right merge run")
                if record is _EOF:
                    self._right_eof = True
                else:
                    self._right_head = record
                continue

            if self._left_head is None and self._right_head is None:
                break
            if self._left_head is not None and self._right_head is not None:
                if self._left_head == self._right_head:
                    self._terminal_kind = "ambiguous"
                    return
                if self._left_head < self._right_head:
                    chosen = self._left_head
                    self._left_head = None
                else:
                    chosen = self._right_head
                    self._right_head = None
            elif self._left_head is not None:
                chosen = self._left_head
                self._left_head = None
            else:
                chosen = self._right_head
                self._right_head = None
            if self._merge_previous == chosen:
                self._terminal_kind = "ambiguous"
                return
            self._pending_merge_record = chosen

        yield from self._pause_for_record_budget()
        yield from self._sync_close_fd("_merge_output_fd", self._merge_output_fd)
        yield from self._pause_for_record_budget()
        yield from self._close_fd("_left_fd")
        yield from self._pause_for_record_budget()
        yield from self._close_fd("_right_fd")
        self._merge_output_path = None
        yield from self._pause_for_record_budget()
        yield from self._call("unlink", left_path)
        yield from self._pause_for_record_budget()
        yield from self._call("unlink", right_path)
        self._left_partial.clear()
        self._right_partial.clear()
        self._left_head = None
        self._right_head = None
        self._merge_previous = None
        self._pair_index += 1

    def _advance_merge_pass(self) -> Generator[object, object, None]:
        old_pass = self._merge_pass
        old_count = self._run_count
        if old_count % 2:
            odd_path = self._run_path(old_pass, old_count - 1)
            odd_destination = self._run_path(old_pass + 1, old_count // 2)
            yield from self._pause_for_record_budget()
            yield from self._call("replace", odd_path, odd_destination)
        self._merge_pass += 1
        self._run_count = (old_count + 1) // 2
        self._pair_index = 0
        yield _Boundary("progressed")

    def _open_input(self, path: Path, label: str) -> Generator[object, object, int]:
        descriptor = yield from self._call("open", path, _SOURCE_FLAGS, 0o600)
        _require_descriptor(descriptor, label)
        return descriptor

    def _open_output(self, path: Path) -> Generator[object, object, int]:
        descriptor = yield from self._call("open", path, _RUN_FLAGS, 0o600)
        _require_descriptor(descriptor, "run output")
        return descriptor

    def _write_record(
        self,
        descriptor: int,
        record: bytes,
        label: str,
        *,
        offset_attr: str | None = None,
    ) -> Generator[object, object, None]:
        offset = getattr(self, offset_attr) if offset_attr is not None else 0
        while offset < _DIGEST_BYTES:
            if self._records_written >= self._advance_max_records:
                if offset_attr is not None:
                    setattr(self, offset_attr, offset)
                yield _Boundary("waiting", "budget")
                offset = getattr(self, offset_attr) if offset_attr is not None else offset
                continue
            written = yield from self._call("write", descriptor, record[offset:])
            if type(written) is not int or written < 0 or written > _DIGEST_BYTES - offset:
                raise OSError(f"{label} returned an invalid short write")
            if written == 0:
                if offset_attr is not None:
                    setattr(self, offset_attr, offset)
                yield _Boundary("waiting", "write")
                offset = getattr(self, offset_attr) if offset_attr is not None else offset
                continue
            offset += written
            if offset_attr is not None:
                setattr(self, offset_attr, offset)
        self._records_written += 1
        if offset_attr is not None:
            setattr(self, offset_attr, offset)

    def _sync_close_fd(self, attr: str, descriptor: int) -> Generator[object, object, None]:
        yield from self._pause_for_record_budget()
        yield from self._call("fsync", descriptor)
        yield from self._pause_for_record_budget()
        yield from self._close_owned(attr, descriptor)

    def _close_fd(self, attr: str) -> Generator[object, object, None]:
        descriptor = getattr(self, attr)
        if descriptor is None:
            return
        yield from self._pause_for_record_budget()
        yield from self._close_owned(attr, descriptor)

    def _cleanup(self) -> Generator[object, object, None]:
        self._phase = "cleanup"
        for attr in ("_forming_output_fd", "_left_fd", "_right_fd", "_merge_output_fd", "_source_fd"):
            descriptor = getattr(self, attr)
            if descriptor is None:
                continue
            try:
                yield from self._pause_for_record_budget()
                yield from self._close_owned(attr, descriptor)
            except BaseException as exc:
                self._record_cleanup_error(exc)

    def _close_owned(self, attr: str, descriptor: int) -> Generator[object, object, None]:
        yield _IORequest("close", (descriptor,), attr)

    def _call(self, method: str, *args: object) -> Generator[object, object, object]:
        value = yield _IORequest(method, tuple(args))
        return value

    def _pause_for_record_budget(self) -> Generator[object, object, None]:
        if self._records_read >= self._advance_max_records and self._records_written >= self._advance_max_records:
            yield _Boundary("waiting", "budget")

    def _record_failure(self, error: BaseException) -> None:
        if isinstance(error, _StopRequested):
            return
        if self._fatal_error is None:
            self._fatal_error = error
        self._cancel_publication()

    def _record_cleanup_error(self, error: BaseException) -> None:
        if self._cleanup_error is None:
            self._cleanup_error = error

    def _cancel_publication(self) -> None:
        self._terminal_kind = None
        self._result_candidate = None
        self._result_path = None
        self._is_complete = False
        self._has_repeated_digest = False

    def _run_path(self, pass_number: int, run_number: int) -> Path:
        return self._scratch_path / f"pass-{pass_number}-{run_number}.bin"


def _normalize_absolute(value: Path, name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a path") from exc
    if isinstance(raw, bytes):
        raise TypeError(f"{name} must be a text path")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _validate_nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_run_records(value: int) -> int:
    value = _validate_positive_int(value, "run_records")
    if value > _MAX_RUN_RECORDS:
        raise ValueError(f"run_records must be at most {_MAX_RUN_RECORDS}")
    return value


def _require_descriptor(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} returned an invalid descriptor")
    return value


def _require_regular(mode: int, path: Path, label: str) -> None:
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _has_adjacent_equal(records: list[bytes]) -> bool:
    return any(records[index] == records[index - 1] for index in range(1, len(records)))


__all__ = ["AuditStep", "DigestAuditDriver"]
