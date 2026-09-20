"""Integration coverage for the narrow cooperative SliceIO facade."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path

import pytest

import qqtools.plugins.qexp.runtime.group_discovery.slice_io as slice_io
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import YIELD, SliceIO, YieldRequired

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", ["max_io_bytes", "max_operations"])
@pytest.mark.parametrize("value", [True, False, -1, 1.0])
def test_constructor_rejects_non_exact_or_negative_maxima(name: str, value: object):
    with pytest.raises((TypeError, ValueError)):
        SliceIO(**{name: value})


def test_yield_is_an_explicit_singleton_and_zero_capacities_do_not_call_os(monkeypatch: pytest.MonkeyPatch):
    assert isinstance(YIELD, YieldRequired)
    assert repr(YIELD) == "YIELD"
    io = SliceIO(max_io_bytes=0, max_operations=0)

    def fail(*_args, **_kwargs):
        raise AssertionError("yielded calls must not reach the OS")

    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "read", fail)
        patch.setattr(slice_io.os, "write", fail)
        patch.setattr(slice_io.os, "open", fail)
        assert io.read(1, 1) is YIELD
        assert io.write(1, b"x") is YIELD
        assert io.open(Path("source"), os.O_RDONLY) is YIELD
        assert io.io_bytes_used == 0
        assert io.operations_used == 0


def test_zero_io_capacity_still_allows_metadata_calls(tmp_path: Path):
    io = SliceIO(max_io_bytes=0, max_operations=4)
    directory = tmp_path / "directory"
    assert io.mkdir(directory) is None
    result = io.lstat(directory)
    assert isinstance(result, os.stat_result)
    assert io.io_bytes_used == 0
    assert io.operations_used == 2
    assert io.rmdir(directory) is None


def test_read_and_write_share_an_aggregate_io_cap_and_operation_cap(tmp_path: Path):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefghij")
    io = SliceIO(max_io_bytes=10, max_operations=3)
    fd = os.open(path, os.O_RDWR)
    try:
        assert io.read(fd, 4) == b"abcd"
        assert io.write(fd, b"1234567890") == 6
        assert io.io_bytes_used == 10
        assert io.operations_used == 2
        assert io.read(fd, 1) is YIELD
        assert io.write(fd, b"x") is YIELD
        assert io.io_bytes_used == 10
        assert io.operations_used == 2
        assert io.close(fd) is None
        fd = None
        assert io.operations_used == 3
    finally:
        if fd is not None:
            os.close(fd)


def test_read_eof_refunds_all_requested_bytes_but_keeps_operation_charge(tmp_path: Path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    io = SliceIO(max_io_bytes=100, max_operations=2)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert io.read(fd, 100) == b""
        assert io.io_bytes_used == 0
        assert io.operations_used == 1
    finally:
        os.close(fd)


def test_short_real_read_refunds_actual_bytes(tmp_path: Path):
    path = tmp_path / "short"
    path.write_bytes(b"abc")
    io = SliceIO(max_io_bytes=65_536, max_operations=2)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert io.read(fd, 65_536) == b"abc"
        assert io.io_bytes_used == 3
        assert io.operations_used == 1
    finally:
        os.close(fd)


def test_short_write_refunds_without_retrying(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def short_write(_fd: int, data: memoryview) -> int:
        calls.append(len(data))
        return min(3, len(data))

    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "write", short_write)
        io = SliceIO(max_io_bytes=20, max_operations=2)
        assert io.write(17, b"abcdef") == 3
        assert calls == [6]
        assert io.io_bytes_used == 3
        assert io.operations_used == 1
        assert io.write(17, b"xy") == 2
        assert calls == [6, 2]
        assert io.io_bytes_used == 5
        assert io.operations_used == 2


def test_requests_are_bounded_to_65536_and_refund_fake_short_results(monkeypatch: pytest.MonkeyPatch):
    read_requests: list[int] = []
    write_requests: list[int] = []

    def short_read(_fd: int, size: int) -> bytes:
        read_requests.append(size)
        return b"r" * 5

    def short_write(_fd: int, data: memoryview) -> int:
        write_requests.append(len(data))
        return 7

    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "read", short_read)
        patch.setattr(slice_io.os, "write", short_write)
        io = SliceIO(max_io_bytes=131_072, max_operations=2)
        assert io.read(1, 100_000) == b"r" * 5
        assert io.write(1, b"w" * 100_000) == 7
        assert read_requests == [65_536]
        assert write_requests == [65_536]
        assert io.io_bytes_used == 12
        assert io.operations_used == 2


def test_failed_syscalls_keep_conservative_requested_byte_charge(tmp_path: Path):
    io = SliceIO(max_io_bytes=20, max_operations=3)
    with pytest.raises(OSError):
        io.read(-1, 4)
    assert io.io_bytes_used == 4
    assert io.operations_used == 1
    with pytest.raises(OSError):
        io.write(-1, b"abcd")
    assert io.io_bytes_used == 8
    assert io.operations_used == 2
    with pytest.raises(FileNotFoundError):
        io.open(tmp_path / "missing", os.O_RDONLY)
    assert io.operations_used == 3


def test_invalid_read_write_inputs_are_rejected_before_admission(monkeypatch: pytest.MonkeyPatch):
    io = SliceIO(max_io_bytes=100, max_operations=100)

    def fail(*_args, **_kwargs):
        raise AssertionError("invalid input must not call the OS")

    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "read", fail)
        patch.setattr(slice_io.os, "write", fail)
        for value in (True, False, 0, -1, 1.0):
            with pytest.raises((TypeError, ValueError)):
                io.read(1, value)
        for value in (b"", bytearray(), memoryview(b""), "text"):
            with pytest.raises((TypeError, ValueError)):
                io.write(1, value)
        with pytest.raises(ValueError):
            io.write(1, memoryview(bytearray(b"abc"))[::2])
        assert io.io_bytes_used == 0
        assert io.operations_used == 0


def test_metadata_failures_charge_one_operation(tmp_path: Path):
    io = SliceIO(max_operations=2)
    with pytest.raises(FileNotFoundError):
        io.unlink(tmp_path / "missing")
    assert io.operations_used == 1
    with pytest.raises(OSError):
        io.fstat(-1)
    assert io.operations_used == 2


def test_real_filesystem_lifecycle_and_independent_flock_attempts(tmp_path: Path):
    io = SliceIO(max_io_bytes=128, max_operations=32)
    root = tmp_path / "root"
    source = root / "source"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    assert io.mkdir(root) is None
    fd = io.open(source, os.O_RDWR | os.O_CREAT, 0o600)
    assert isinstance(fd, int)
    fd2 = None
    try:
        assert io.write(fd, b"abcdef") == 6
        assert io.lseek(fd, 0, os.SEEK_SET) == 0
        assert io.read(fd, 3) == b"abc"
        assert io.ftruncate(fd, 4) is None
        assert io.fsync(fd) is None
        assert isinstance(io.fstat(fd), os.stat_result)
        assert isinstance(io.lstat(source), os.stat_result)
        fd2 = io.open(source, os.O_RDWR)
        assert isinstance(fd2, int)
        assert io.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB) is None
        with pytest.raises(BlockingIOError) as blocked:
            io.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert blocked.value.errno in {errno.EACCES, errno.EAGAIN}
        assert io.flock(fd, fcntl.LOCK_UN) is None
        assert io.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB) is None
        assert io.flock(fd2, fcntl.LOCK_UN) is None
        assert io.close(fd2) is None
        fd2 = None
        assert io.close(fd) is None
        fd = None
        assert io.replace(replacement, source) is None
        assert io.unlink(source) is None
        assert io.rmdir(root) is None
        assert io.operations_used == 20
        assert io.io_bytes_used == 9
    finally:
        if fd2 is not None:
            os.close(fd2)
        if fd is not None:
            os.close(fd)


def test_read_write_yield_without_syscall_when_capacity_is_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    io = SliceIO(max_io_bytes=1, max_operations=1)
    path = tmp_path / "empty"
    path.write_bytes(b"")
    fd = os.open(path, os.O_RDONLY)
    try:
        assert io.read(fd, 1) == b""
    finally:
        os.close(fd)

    def fail(*_args, **_kwargs):
        raise AssertionError("yielded calls must not reach the OS")

    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "read", fail)
        patch.setattr(slice_io.os, "write", fail)
        assert io.read(1, 1) is YIELD
        assert io.write(1, b"x") is YIELD


def test_zero_write_is_charged_once_without_retry(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def no_progress(fd, data):
        calls.append((fd, len(data)))
        return 0

    budget = SliceIO(max_io_bytes=8, max_operations=1)
    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "write", no_progress)
        assert budget.write(17, b"abc") == 0
        assert budget.write(17, b"abc") is YIELD
    assert calls == [(17, 3)]
    assert budget.operations_used == 1
    assert budget.io_bytes_used == 0


@pytest.mark.parametrize("result", [bytearray(b"a"), None, b"abcd"])
def test_invalid_read_result_retains_full_precharge(monkeypatch: pytest.MonkeyPatch, result):
    budget = SliceIO(max_io_bytes=3, max_operations=1)
    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "read", lambda _fd, _size: result)
        with pytest.raises(TypeError):
            budget.read(17, 10)
    assert budget.operations_used == 1
    assert budget.io_bytes_used == 3


@pytest.mark.parametrize("result", [True, None, -1, 4])
def test_invalid_write_result_retains_full_precharge(monkeypatch: pytest.MonkeyPatch, result):
    budget = SliceIO(max_io_bytes=3, max_operations=1)
    with monkeypatch.context() as patch:
        patch.setattr(slice_io.os, "write", lambda _fd, _data: result)
        with pytest.raises(ValueError):
            budget.write(17, b"abcde")
    assert budget.operations_used == 1
    assert budget.io_bytes_used == 3
