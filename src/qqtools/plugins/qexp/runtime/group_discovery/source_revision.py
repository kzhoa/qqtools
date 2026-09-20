"""Bind a read-only qexp runtime source to a stable filesystem revision.

This Linux-qualified extraction provides a bounded random-read handle for a
regular file.  It never scans or hashes the source, and it does not provide a
parser checkpoint or durable restart protocol.  ``expected_revision`` checks
the source reopened for a new binding; it does not restore JSON grammar state,
and it deliberately rejects a replaced file even when the replacement has
identical bytes.  Callers must keep parser outputs provisional and call
``verify`` before using them.  Physical fsync, cross-host coordination, and
Group membership certification are outside this extraction's scope.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_MAX_READ_SIZE = 65_536
_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """The complete scalar filesystem stamp used to bind a source."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        for name in ("device", "inode", "size", "mtime_ns", "ctime_ns"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact int")
        for name in ("device", "inode", "size"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def from_stat(cls, stat_result: os.stat_result) -> SourceRevision:
        """Construct a revision from an ``os.stat`` or ``os.fstat`` result."""

        return cls(
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            ctime_ns=stat_result.st_ctime_ns,
        )


class SourceChangedError(ValueError):
    """The named source or bound descriptor no longer has its original stamp."""


class BoundSource:
    """Read a regular file through an unbuffered descriptor with a fixed stamp.

    ``open`` owns the descriptor and checks the descriptor, named path, and
    descriptor again before returning.  A caller can use ``seek`` for bounded
    scalar reads, but this class does not restore parser state.  ``verify``
    checks the same three points again and closes the source if any check
    fails.
    """

    __slots__ = ("_handle", "_path", "_revision", "_closed")

    def __init__(self, handle: BinaryIO, path: Path, revision: SourceRevision):
        self._handle = handle
        self._path = path
        self._revision = revision
        self._closed = False

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        expected_revision: SourceRevision | None = None,
    ) -> BoundSource:
        """Open and bind a regular file, optionally requiring its exact stamp.

        Args:
            path: Named path to the regular file.
            expected_revision: Exact revision required for this reopened source.

        Raises:
            SourceChangedError: If an expected source is missing, replaced,
                symlinked, nonregular, or has a different revision.
            ValueError: If an initial source is symlinked or nonregular.
        """

        if expected_revision is not None and not isinstance(expected_revision, SourceRevision):
            raise TypeError("expected_revision must be a SourceRevision or None")

        source_path = Path(path)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        descriptor: int | None = None
        handle: BinaryIO | None = None
        try:
            try:
                descriptor = os.open(source_path, _OPEN_FLAGS)
            except OSError as exc:
                cls._raise_open_error(source_path, expected_revision, exc)

            first_stat = os.fstat(descriptor)
            first_revision = cls._revision_for_stat(
                first_stat,
                source_path,
                expected_revision,
                "descriptor",
            )
            cls._require_regular(first_stat.st_mode, source_path, expected_revision, "descriptor")

            try:
                named_stat = os.lstat(source_path)
            except FileNotFoundError as exc:
                if expected_revision is not None:
                    raise SourceChangedError(f"expected source is missing: {source_path}") from exc
                raise
            named_revision = cls._revision_for_stat(
                named_stat,
                source_path,
                expected_revision,
                "named path",
            )
            cls._require_regular(named_stat.st_mode, source_path, expected_revision, "named path")
            cls._require_same_revision(first_revision, named_revision, source_path, expected_revision)

            second_stat = os.fstat(descriptor)
            second_revision = cls._revision_for_stat(
                second_stat,
                source_path,
                expected_revision,
                "descriptor",
            )
            cls._require_regular(second_stat.st_mode, source_path, expected_revision, "descriptor")
            cls._require_same_revision(first_revision, second_revision, source_path, expected_revision)
            if expected_revision is not None and first_revision != expected_revision:
                raise SourceChangedError(f"source revision changed while opening {source_path}")

            handle = os.fdopen(descriptor, "rb", buffering=0)
            descriptor = None
            return cls(handle, source_path, first_revision)
        except BaseException:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise

    @property
    def revision(self) -> SourceRevision:
        """Return the immutable revision captured at binding time."""

        return self._revision

    @property
    def closed(self) -> bool:
        """Whether this source has been closed or invalidated."""

        return self._closed

    def __enter__(self) -> BoundSource:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close the owned descriptor; repeated calls are harmless."""

        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            self._closed = True

    def read(self, size: int) -> bytes:
        """Read at most the requested bounded size without source verification."""

        self._require_open()
        _require_exact_int(size, "size")
        if not 1 <= size <= _MAX_READ_SIZE:
            raise ValueError("size must be between 1 and 65536")
        return self._handle.read(size)

    def tell(self) -> int:
        """Return the current descriptor offset."""

        self._require_open()
        return self._handle.tell()

    def seek(self, offset: int) -> int:
        """Seek to an absolute offset within the original source size."""

        self._require_open()
        _require_exact_int(offset, "offset")
        if not 0 <= offset <= self._revision.size:
            raise ValueError("offset must be within the original source size")
        return self._handle.seek(offset, os.SEEK_SET)

    def verify(self) -> None:
        """Recheck the descriptor and named path, closing on every failure."""

        self._require_open()
        try:
            first_stat = os.fstat(self._handle.fileno())
            first_revision = self._revision_for_stat(
                first_stat,
                self._path,
                self._revision,
                "descriptor",
            )
            self._require_regular(first_stat.st_mode, self._path, self._revision, "descriptor")
            self._require_original(first_revision, self._path, "descriptor")

            named_stat = os.lstat(self._path)
            named_revision = self._revision_for_stat(
                named_stat,
                self._path,
                self._revision,
                "named path",
            )
            self._require_regular(named_stat.st_mode, self._path, self._revision, "named path")
            self._require_original(named_revision, self._path, "named path")

            second_stat = os.fstat(self._handle.fileno())
            second_revision = self._revision_for_stat(
                second_stat,
                self._path,
                self._revision,
                "descriptor",
            )
            self._require_regular(second_stat.st_mode, self._path, self._revision, "descriptor")
            self._require_original(second_revision, self._path, "descriptor")
        except BaseException as exc:
            self.close()
            if isinstance(exc, SourceChangedError):
                raise
            if isinstance(exc, OSError):
                raise SourceChangedError(f"could not verify source {self._path}") from exc
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("source is closed")

    @staticmethod
    def _raise_open_error(path: Path, expected_revision: SourceRevision | None, exc: OSError) -> None:
        try:
            named_stat = os.lstat(path)
        except FileNotFoundError:
            if expected_revision is not None:
                raise SourceChangedError(f"expected source is missing: {path}") from exc
            raise
        except OSError:
            raise exc

        if stat.S_ISLNK(named_stat.st_mode):
            if expected_revision is not None:
                raise SourceChangedError(f"expected source is a symlink: {path}") from exc
            raise ValueError(f"source must not be a symlink: {path}") from exc
        if not stat.S_ISREG(named_stat.st_mode):
            if expected_revision is not None:
                raise SourceChangedError(f"expected source is not a regular file: {path}") from exc
            raise ValueError(f"source must be a regular file: {path}") from exc
        raise exc

    @staticmethod
    def _revision_for_stat(
        stat_result: os.stat_result,
        path: Path,
        expected_revision: SourceRevision | None,
        source_label: str,
    ) -> SourceRevision:
        try:
            return SourceRevision.from_stat(stat_result)
        except (TypeError, ValueError) as exc:
            if expected_revision is not None:
                raise SourceChangedError(f"invalid {source_label} revision for {path}") from exc
            raise

    @staticmethod
    def _require_regular(
        mode: int,
        path: Path,
        expected_revision: SourceRevision | None,
        source_label: str,
    ) -> None:
        if stat.S_ISREG(mode):
            return
        if expected_revision is not None:
            raise SourceChangedError(f"expected {source_label} is not a regular file: {path}")
        raise ValueError(f"source {source_label} must be a regular file: {path}")

    @staticmethod
    def _require_same_revision(
        actual: SourceRevision,
        other: SourceRevision,
        path: Path,
        expected_revision: SourceRevision | None,
    ) -> None:
        if actual == other:
            return
        if expected_revision is not None:
            raise SourceChangedError(f"source changed while opening {path}")
        raise ValueError(f"source changed while opening {path}")

    def _require_original(self, actual: SourceRevision, path: Path, source_label: str) -> None:
        if actual != self._revision:
            raise SourceChangedError(f"{source_label} revision changed for {path}")


def _require_exact_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
