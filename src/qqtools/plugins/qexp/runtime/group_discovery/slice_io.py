"""Cooperative bounded OS calls for qexp runtime source extraction.

``SliceIO`` is only an admission and accounting facade.  It does not qualify
the projection session's combined budget, descriptor ownership, checkpoint
durability, storage protocol, or Group authority.  Callers choose all OS
flags and own every descriptor returned by this module.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Callable

_MAX_SINGLE_IO = 65_536


class YieldRequired:
    """Sentinel type returned when the next syscall does not fit this slice."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "YIELD"


YIELD = YieldRequired()


class SliceIO:
    """Admit at most one bounded syscall at a time and account for its cost.

    Read and write calls reserve their requested bytes before the OS call and
    refund bytes that the OS reports as not transferred.  Metadata calls cost
    one operation and no I/O bytes.  A yielded call performs no syscall and
    leaves both counters unchanged.
    """

    __slots__ = ("_max_io_bytes", "_max_operations", "_io_bytes_used", "_operations_used")

    def __init__(self, *, max_io_bytes: int = 262_144, max_operations: int = 32):
        self._max_io_bytes = _validate_maximum(max_io_bytes, "max_io_bytes")
        self._max_operations = _validate_maximum(max_operations, "max_operations")
        self._io_bytes_used = 0
        self._operations_used = 0

    @property
    def io_bytes_used(self) -> int:
        """Return I/O bytes charged in this slice."""

        return self._io_bytes_used

    @property
    def operations_used(self) -> int:
        """Return syscall operations charged in this slice."""

        return self._operations_used

    def open(self, path: Path, flags: int, mode: int = 0o600) -> int | YieldRequired:
        """Call :func:`os.open` once when one operation remains."""

        return self._metadata(os.open, path, flags, mode)

    def mkdir(self, path: Path, mode: int = 0o700) -> None | YieldRequired:
        """Call :func:`os.mkdir` once when one operation remains."""

        return self._metadata(os.mkdir, path, mode)

    def rmdir(self, path: Path) -> None | YieldRequired:
        """Call :func:`os.rmdir` once when one operation remains."""

        return self._metadata(os.rmdir, path)

    def scandir(self, path: Path) -> os.ScandirIterator | YieldRequired:
        """Open one directory iterator when one operation remains."""

        return self._metadata(os.scandir, path)

    def next_entry(self, iterator: os.ScandirIterator) -> os.DirEntry | None | YieldRequired:
        """Advance one directory iterator, charging EOF and failures as metadata."""

        return self._metadata(next, iterator, None)

    def close_directory(self, iterator: os.ScandirIterator) -> None | YieldRequired:
        """Close one directory iterator when one operation remains."""

        return self._metadata(iterator.close)

    def lstat(self, path: Path) -> os.stat_result | YieldRequired:
        """Call :func:`os.lstat` once when one operation remains."""

        return self._metadata(os.lstat, path)

    def fstat(self, fd: int) -> os.stat_result | YieldRequired:
        """Call :func:`os.fstat` once when one operation remains."""

        return self._metadata(os.fstat, fd)

    def lseek(self, fd: int, offset: int, whence: int) -> int | YieldRequired:
        """Call :func:`os.lseek` once when one operation remains."""

        return self._metadata(os.lseek, fd, offset, whence)

    def ftruncate(self, fd: int, length: int) -> None | YieldRequired:
        """Call :func:`os.ftruncate` once when one operation remains."""

        return self._metadata(os.ftruncate, fd, length)

    def unlink(self, path: Path) -> None | YieldRequired:
        """Call :func:`os.unlink` once when one operation remains."""

        return self._metadata(os.unlink, path)

    def replace(self, source: Path, destination: Path) -> None | YieldRequired:
        """Call :func:`os.replace` once when one operation remains."""

        return self._metadata(os.replace, source, destination)

    def fsync(self, fd: int) -> None | YieldRequired:
        """Call :func:`os.fsync` once when one operation remains."""

        return self._metadata(os.fsync, fd)

    def flock(self, fd: int, operation: int) -> None | YieldRequired:
        """Call :func:`fcntl.flock` once when one operation remains."""

        return self._metadata(fcntl.flock, fd, operation)

    def close(self, fd: int) -> None | YieldRequired:
        """Call :func:`os.close` once when one operation remains."""

        return self._metadata(os.close, fd)

    def read(self, fd: int, size: int) -> bytes | YieldRequired:
        """Read one bounded request, refunding bytes not returned by the OS."""

        _validate_positive_int(size, "size")
        request = self._admit_io(min(size, _MAX_SINGLE_IO))
        if request is None:
            return YIELD
        try:
            result = os.read(fd, request)
        except BaseException:
            raise
        if type(result) is not bytes or len(result) > request:
            raise TypeError("os.read returned an invalid result")
        self._io_bytes_used -= request - len(result)
        return result

    def write(self, fd: int, data: bytes | bytearray | memoryview) -> int | YieldRequired:
        """Write one bounded contiguous view without retrying short writes."""

        view = _contiguous_bytes(data)
        request = self._admit_io(min(view.nbytes, _MAX_SINGLE_IO))
        if request is None:
            return YIELD
        bounded = view[:request]
        try:
            result = os.write(fd, bounded)
        except BaseException:
            raise
        if type(result) is not int or not 0 <= result <= request:
            raise ValueError("os.write returned an invalid byte count")
        self._io_bytes_used -= request - result
        return result

    def _metadata(self, operation: Callable[..., object], *args: object) -> object | YieldRequired:
        if self._operations_used >= self._max_operations:
            return YIELD
        self._operations_used += 1
        return operation(*args)

    def _admit_io(self, requested: int) -> int | None:
        if self._operations_used >= self._max_operations:
            return None
        available = self._max_io_bytes - self._io_bytes_used
        if available <= 0:
            return None
        request = min(requested, available)
        self._operations_used += 1
        self._io_bytes_used += request
        return request


def _validate_maximum(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_positive_int(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _contiguous_bytes(data: bytes | bytearray | memoryview) -> memoryview:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes, bytearray, or memoryview")
    try:
        view = memoryview(data)
        if view.nbytes == 0:
            raise ValueError("data must be nonempty")
        return view.cast("B")
    except TypeError as exc:
        raise ValueError("data must be a contiguous byte view") from exc


__all__ = ["SliceIO", "YIELD", "YieldRequired"]
