"""Independent syscall and budget coverage for the cooperative projection driver."""

from __future__ import annotations

import builtins
import fcntl
import io as io_module
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from tests.helpers.qexp.qexp_projection_session import ProjectionSession

pytestmark = pytest.mark.integration

_IO_BYTES = 262_144
_OPERATIONS = 32
_READ_WRITE_LIMIT = 65_536


class _SyscallRecorder:
    """Check that every active driver syscall passes through the shared SliceIO."""

    _COUNTED_OS_CALLS = (
        "open",
        "close",
        "read",
        "write",
        "fstat",
        "lstat",
        "lseek",
        "ftruncate",
        "unlink",
        "replace",
        "fsync",
        "mkdir",
        "rmdir",
    )
    _FORBIDDEN_CALLS = ("stat", "scandir", "listdir", "rename", "fdopen")

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        source_path: Path | None = None,
        short_read: int | None = None,
        short_write: int | None = None,
    ) -> None:
        self._active_io: SliceIO | None = None
        self._forbid = False
        self._label = ""
        self._base_operations = 0
        self._call_count = 0
        self._expected_charged = 0
        self._active_actual = 0
        self._active_start_actual = 0
        self._total_actual: dict[SliceIO, int] = {}
        self._source_path = source_path
        self._short_read = short_read
        self._short_write = short_write
        self.opened_fds: set[int] = set()
        self.closed_fds: set[int] = set()
        self.live_fds: set[int] = set()
        self.unlocked_fds: set[int] = set()
        self.source_fds: set[int] = set()
        self.fd_paths: dict[int, Path] = {}
        self.fd_positions: dict[int, int] = {}
        self.source_read_positions: list[int] = []
        self.calls: list[tuple[str, str]] = []

        for name in self._COUNTED_OS_CALLS:
            original = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap_os_call(name, original))
        original_flock = fcntl.flock
        monkeypatch.setattr(fcntl, "flock", self._wrap_flock(original_flock))
        for name in self._FORBIDDEN_CALLS:
            original = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap_forbidden(f"os.{name}", original))
        monkeypatch.setattr(builtins, "open", self._wrap_forbidden("builtins.open", builtins.open))
        monkeypatch.setattr(io_module, "open", self._wrap_forbidden("io.open", io_module.open))

    @contextmanager
    def observe(self, io: SliceIO, label: str) -> Iterator[None]:
        if self._active_io is not None or self._forbid:
            raise AssertionError("nested syscall recorder activation")
        self._active_io = io
        self._label = label
        self._base_operations = io.operations_used
        self._call_count = 0
        self._expected_charged = io.io_bytes_used
        self._active_start_actual = self._total_actual.get(io, 0)
        self._active_actual = 0
        try:
            yield
        finally:
            try:
                actual = self._total_actual.get(io, 0) - self._active_start_actual
                assert io.io_bytes_used == self._expected_charged, (
                    self._label,
                    io.io_bytes_used,
                    self._expected_charged,
                )
                assert actual <= io.io_bytes_used <= _IO_BYTES, (
                    self._label,
                    actual,
                    io.io_bytes_used,
                )
                assert io.operations_used <= _OPERATIONS, (self._label, io.operations_used)
            finally:
                self._active_io = None
                self._label = ""

    @contextmanager
    def forbid_io(self, label: str) -> Iterator[None]:
        if self._active_io is not None or self._forbid:
            raise AssertionError("nested syscall recorder activation")
        self._forbid = True
        self._label = label
        try:
            yield
        finally:
            self._forbid = False
            self._label = ""

    def assert_closed(self) -> None:
        assert not self.live_fds, self.live_fds
        if self.opened_fds:
            assert self.unlocked_fds, "driver never released its scratch lock"

    def _before(self, name: str, *, request: int | None = None) -> SliceIO | None:
        if self._forbid:
            raise AssertionError(f"filesystem syscall {name} during no-I/O phase {self._label}")
        io = self._active_io
        if io is None:
            return None
        expected = self._base_operations + self._call_count + 1
        assert io.operations_used == expected, (
            self._label,
            name,
            io.operations_used,
            expected,
        )
        if request is not None:
            assert 0 < request <= _READ_WRITE_LIMIT, (self._label, name, request)
            assert io.io_bytes_used == self._expected_charged + request, (
                self._label,
                name,
                request,
                io.io_bytes_used,
                self._expected_charged,
            )
        else:
            assert io.io_bytes_used == self._expected_charged, (
                self._label,
                name,
                io.io_bytes_used,
                self._expected_charged,
            )
        self._call_count += 1
        self.calls.append((self._label, name))
        return io

    def _after(self, io: SliceIO | None) -> None:
        if io is None:
            return
        assert io.operations_used == self._base_operations + self._call_count
        assert io.io_bytes_used <= _IO_BYTES

    def _record_actual(self, io: SliceIO | None, amount: int) -> None:
        if io is None:
            return
        assert amount >= 0
        self._active_actual += amount
        self._total_actual[io] = self._total_actual.get(io, 0) + amount
        self._expected_charged += amount

    def _record_error_charge(self, io: SliceIO | None, request: int | None) -> None:
        if io is not None and request is not None:
            self._expected_charged += request

    def _wrap_os_call(self, name: str, original):
        def wrapped(*args, **kwargs):
            request = None
            if name == "read":
                request = args[1]
            elif name == "write":
                data = args[1]
                request = data.nbytes if isinstance(data, memoryview) else len(data)
            io = self._before(name, request=request)
            try:
                if name == "read" and io is not None and self._short_read is not None:
                    result = original(args[0], min(args[1], self._short_read), **kwargs)
                elif name == "write" and io is not None and self._short_write is not None:
                    data = args[1]
                    size = data.nbytes if isinstance(data, memoryview) else len(data)
                    short = min(size, self._short_write)
                    result = original(args[0], data[:short], *args[2:], **kwargs)
                else:
                    result = original(*args, **kwargs)
            except BaseException:
                self._record_error_charge(io, request)
                self._after(io)
                raise
            if name == "read":
                assert type(result) is bytes
                self._record_actual(io, len(result))
                fd = args[0]
                if fd in self.source_fds:
                    self.source_read_positions.append(self.fd_positions[fd])
                    self.fd_positions[fd] += len(result)
            elif name == "write":
                assert type(result) is int
                self._record_actual(io, result)
            elif name == "lseek" and io is not None:
                self.fd_positions[args[0]] = result
            elif name == "open" and io is not None and type(result) is int:
                self.opened_fds.add(result)
                self.live_fds.add(result)
                path = Path(os.fspath(args[0]))
                self.fd_paths[result] = path
                self.fd_positions[result] = 0
                if self._source_path is not None and path == self._source_path:
                    self.source_fds.add(result)
            elif name == "close" and io is not None:
                self.closed_fds.add(args[0])
                self.live_fds.discard(args[0])
                self.source_fds.discard(args[0])
            self._after(io)
            return result

        return wrapped

    def _wrap_flock(self, original):
        def wrapped(fd: int, operation: int):
            io = self._before("flock")
            try:
                result = original(fd, operation)
            except BaseException:
                self._after(io)
                raise
            if io is not None and operation & fcntl.LOCK_UN:
                self.unlocked_fds.add(fd)
            self._after(io)
            return result

        return wrapped

    def _wrap_forbidden(self, name: str, original):
        def wrapped(*args, **kwargs):
            if self._active_io is not None or self._forbid:
                raise AssertionError(f"unadmitted filesystem call {name} during {self._label}")
            return original(*args, **kwargs)

        return wrapped


def _payload(
    *,
    plain_length: int = 10,
    escaped_repeats: int = 4,
    sequence_digits: int = 5,
    ignored_depth: int = 0,
) -> bytes:
    plain_id = b"task-" + (b"a" * plain_length)
    escaped_id = b"task-" + (b"\\u0062" * escaped_repeats)
    sequence = b"1" + (b"7" * sequence_digits)
    ignored = b""
    if ignored_depth:
        ignored = b',"ignored":' + (b"[" * ignored_depth) + b"0" + (b"]" * ignored_depth)
    return (
        b'{"meta":{"schema_version":6},"submission":{"operation_id":"op-1",'
        b'"target_group":"exp","state":"committed","resolved_context":{"task_ids":["'
        + plain_id
        + b'","'
        + escaped_id
        + b'"]},"commit_plan":{"group_membership_sequences":['
        + sequence
        + b",2]}}"
        + ignored
        + b"}"
    )


def _baseline(tmp_path: Path, payload: bytes, name: str = "baseline") -> tuple[bytes, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / f"{name}.json"
    scratch = tmp_path / f"{name}-scratch"
    source.write_bytes(payload)
    session = ProjectionSession.create(source, scratch, "op-1", "exp")
    try:
        while not session.is_complete:
            session.step(65_536, max_fragments=2)
        summary = session.summary
        assert summary is not None
        session.checkpoint()
    finally:
        session.close()
    return (scratch / "events.jsonl").read_bytes(), summary


def _advance(
    driver: ProjectionDriver,
    recorder: _SyscallRecorder,
    *,
    max_processed_bytes: int = 65_536,
    max_io_bytes: int = _IO_BYTES,
    max_operations: int = _OPERATIONS,
    soft_deadline: float | None = None,
    label: str = "advance",
):
    io = SliceIO(max_io_bytes=max_io_bytes, max_operations=max_operations)
    with recorder.observe(io, label):
        result = driver.advance(
            io,
            max_processed_bytes=max_processed_bytes,
            soft_deadline=soft_deadline,
        )
    return result, io


def _close_driver(driver: ProjectionDriver, recorder: _SyscallRecorder) -> None:
    if driver.is_closed:
        return
    with recorder.forbid_io("request_close"):
        driver.request_close()
    for _ in range(10_000):
        if driver.is_closed:
            break
        _advance(driver, recorder, label="cleanup")
    assert driver.is_closed
    recorder.assert_closed()


def _new_driver(
    source: Path,
    scratch: Path,
    recorder: _SyscallRecorder,
    *,
    hook=None,
) -> ProjectionDriver:
    with recorder.forbid_io("construct"):
        driver = ProjectionDriver(source, scratch, "op-1", "exp", hook=hook)
    assert driver.processed_offset == 0
    assert driver.checkpoint_generation == 0
    assert not driver.is_complete
    return driver


def _drive_to_completion(
    driver: ProjectionDriver,
    recorder: _SyscallRecorder,
    *,
    request_checkpoints: bool = False,
    max_processed_bytes: int = 65_536,
    limit: int = 20_000,
) -> None:
    previous_offset = driver.processed_offset
    for index in range(limit):
        if driver.is_complete:
            return
        result, _io = _advance(
            driver,
            recorder,
            max_processed_bytes=max_processed_bytes,
        )
        assert result.state in {"progressed", "waiting", "complete"}
        assert driver.processed_offset - previous_offset <= max_processed_bytes
        previous_offset = driver.processed_offset
        if request_checkpoints and index % 7 == 3 and not driver.is_complete:
            with recorder.forbid_io("request_checkpoint"):
                driver.request_checkpoint()
    raise AssertionError(f"driver did not complete after {limit} advances; offset={driver.processed_offset}")


def test_driver_budget_and_exact_baseline_for_large_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload(plain_length=200_000, escaped_repeats=90_000, sequence_digits=140_000)
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    source = tmp_path / "driver" / "source.json"
    scratch = tmp_path / "driver" / "scratch"
    source.parent.mkdir()
    source.write_bytes(payload)

    recorder = _SyscallRecorder(monkeypatch, source_path=source)
    driver = _new_driver(source, scratch, recorder)
    try:
        with recorder.forbid_io("request_checkpoint"):
            driver.request_checkpoint()
        _drive_to_completion(driver, recorder, request_checkpoints=True)
        assert driver.is_complete
        assert driver.summary == baseline_summary
        assert driver.processed_offset == len(payload)
        assert driver.checkpoint_generation >= 1
    finally:
        _close_driver(driver, recorder)

    assert (scratch / "events.jsonl").read_bytes() == baseline_events


def test_large_checkpoint_writes_across_slices_and_cold_resume_starts_at_saved_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(ignored_depth=6_000)
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    source = tmp_path / "driver" / "source.json"
    scratch = tmp_path / "driver" / "scratch"
    source.parent.mkdir()
    source.write_bytes(payload)

    recorder = _SyscallRecorder(monkeypatch, source_path=source)
    driver = _new_driver(source, scratch, recorder)
    saved_offset = 0
    try:
        for _ in range(20_000):
            if driver.processed_offset >= 3_500 and not driver.is_complete:
                break
            _advance(driver, recorder, max_processed_bytes=4_000, label="prefix")
        assert driver.processed_offset >= 3_500
        assert not driver.is_complete
        saved_offset = driver.processed_offset
        with recorder.forbid_io("request_checkpoint"):
            driver.request_checkpoint()
        slices = 0
        while driver.checkpoint_generation < 2:
            _advance(driver, recorder, max_processed_bytes=4_000, label="large-checkpoint")
            slices += 1
            assert slices < 20_000
        assert slices > 1
        assert (scratch / "checkpoint.json").stat().st_size > _IO_BYTES
    finally:
        _close_driver(driver, recorder)

    recorder.source_read_positions.clear()
    resumed = _new_driver(source, scratch, recorder)
    try:
        for _ in range(20_000):
            if resumed.checkpoint_generation:
                break
            _advance(resumed, recorder, max_processed_bytes=4_000, label="cold-resume")
        else:
            pytest.fail("cold resume did not finish within the slice limit")
        assert resumed.checkpoint_generation == 1
        assert resumed.processed_offset >= saved_offset
        _drive_to_completion(resumed, recorder, max_processed_bytes=4_000)
        assert resumed.summary == baseline_summary
    finally:
        _close_driver(resumed, recorder)

    assert recorder.source_read_positions
    assert min(recorder.source_read_positions) >= saved_offset
    assert (scratch / "events.jsonl").read_bytes() == baseline_events


@pytest.mark.parametrize("short_size", [1, 7, 13])
def test_driver_handles_short_reads_and_writes_without_false_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_size: int
) -> None:
    payload = _payload(plain_length=2_000, escaped_repeats=2_000, sequence_digits=3_000)
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    source = tmp_path / "driver" / "source.json"
    scratch = tmp_path / "driver" / "scratch"
    source.parent.mkdir()
    source.write_bytes(payload)

    recorder = _SyscallRecorder(
        monkeypatch,
        source_path=source,
        short_read=short_size,
        short_write=short_size,
    )
    driver = _new_driver(source, scratch, recorder)
    try:
        # One-byte writes also fragment the larger JSONL event representation.
        _drive_to_completion(driver, recorder, limit=200_000)
        assert driver.summary == baseline_summary
    finally:
        _close_driver(driver, recorder)
    assert (scratch / "events.jsonl").read_bytes() == baseline_events


def test_checkpoint_fsync_error_propagates_after_budgeted_cleanup_and_old_checkpoint_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(plain_length=40, escaped_repeats=20, sequence_digits=40)
    baseline_events, baseline_summary = _baseline(tmp_path / "baseline", payload)
    source = tmp_path / "driver" / "source.json"
    scratch = tmp_path / "driver" / "scratch"
    source.parent.mkdir()
    source.write_bytes(payload)
    holder: dict[str, ProjectionDriver] = {}
    injected = {"armed": False, "done": False}

    def hook(name: str) -> None:
        driver = holder["driver"]
        if name == "checkpoint_fsync" and injected["armed"]:
            assert driver.checkpoint_generation >= 1
        if name == "checkpoint_fsync" and injected["armed"] and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("checkpoint fsync fault")

    recorder = _SyscallRecorder(monkeypatch, source_path=source)
    driver = _new_driver(source, scratch, recorder, hook=hook)
    holder["driver"] = driver
    try:
        for _ in range(1000):
            _advance(driver, recorder, max_processed_bytes=1, label="initial-checkpoint")
            if driver.checkpoint_generation >= 1:
                break
        assert driver.checkpoint_generation >= 1
        assert not driver.is_complete
        injected["armed"] = True
        failure: RuntimeError | None = None
        for _ in range(20_000):
            if driver.is_closed:
                break
            try:
                _advance(driver, recorder, max_operations=1, label="fault-cleanup")
            except RuntimeError as exc:
                failure = exc
                break
        assert failure is not None
        assert str(failure) == "checkpoint fsync fault"
        assert injected["done"]
        assert driver.is_closed
        recorder.assert_closed()
    finally:
        if not driver.is_closed:
            _close_driver(driver, recorder)

    successor = _new_driver(source, scratch, recorder)
    try:
        _drive_to_completion(successor, recorder)
        assert successor.summary == baseline_summary
    finally:
        _close_driver(successor, recorder)
    assert (scratch / "events.jsonl").read_bytes() == baseline_events


def test_zero_capacity_and_expired_deadline_perform_no_filesystem_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    source = tmp_path / "source.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(payload)
    recorder = _SyscallRecorder(monkeypatch, source_path=source)
    driver = _new_driver(source, scratch, recorder)
    try:
        zero_io = SliceIO(max_io_bytes=0, max_operations=0)
        with recorder.observe(zero_io, "zero-capacity"):
            waiting = driver.advance(zero_io, max_processed_bytes=65_536)
        assert waiting.state == "waiting"
        assert waiting.reason in {"budget", "deadline"}
        assert zero_io.operations_used == 0
        assert zero_io.io_bytes_used == 0

        expired = SliceIO(max_io_bytes=_IO_BYTES, max_operations=_OPERATIONS)
        with recorder.observe(expired, "expired-deadline"):
            deadline_result = driver.advance(expired, soft_deadline=0.0)
        assert deadline_result.state == "waiting"
        assert deadline_result.reason == "deadline"
        assert expired.operations_used == 0
        assert expired.io_bytes_used == 0
    finally:
        _close_driver(driver, recorder)


def test_invalid_processing_budget_is_rejected_without_filesystem_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    scratch = tmp_path / "scratch"
    source.write_bytes(_payload())
    recorder = _SyscallRecorder(monkeypatch, source_path=source)
    driver = _new_driver(source, scratch, recorder)
    try:
        io = SliceIO(max_io_bytes=_IO_BYTES, max_operations=_OPERATIONS)
        with recorder.observe(io, "invalid-budget"):
            with pytest.raises((TypeError, ValueError)):
                driver.advance(io, max_processed_bytes=0)
        assert io.operations_used == 0
        assert io.io_bytes_used == 0
    finally:
        _close_driver(driver, recorder)
