"""Cooperative, provisional discovery of Submission source paths.

This module only visits a bounded number of directory entries per advance and
returns source-path hints.  It does not validate source contents, file types,
symlinks, Group membership, or coverage.  Callers own those checks and the
iterator lifetime.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..records import validate_identifier
from .slice_io import YIELD, SliceIO

_MAX_PAGE_SIZE = 64


@dataclass(frozen=True, slots=True)
class SweepStep:
    """Result of one bounded source-directory advance."""

    state: str
    candidates: tuple[Path, ...] = ()
    reason: str | None = None


class SubmissionSourceSweep:
    """Visit Submission source names once through a caller-owned I/O slice.

    The iterator is retained between advances.  A full encounter-order page is
    returned as ``"page"``; only directory EOF followed by a successful close
    returns ``"complete"``.  Neither result is Group coverage evidence.
    The terminal result is stable on repeated calls. Callers consume its final
    page once when transitioning their sweep to complete, not once per retry.
    """

    __slots__ = (
        "_directory",
        "_page_size",
        "_iterator",
        "_candidates",
        "_at_eof",
        "_state",
        "_close_requested",
        "_explicit_close",
        "_failure",
        "_failure_reason",
        "_failure_exposed",
        "_completed_step",
    )

    def __init__(self, directory: Path, *, page_size: int = _MAX_PAGE_SIZE) -> None:
        try:
            directory_value = os.fspath(directory)
        except TypeError as exc:
            raise TypeError("directory must be a path") from exc
        if isinstance(directory_value, bytes):
            raise TypeError("directory must be a text path")
        if type(page_size) is not int:
            raise TypeError("page_size must be an exact int")
        if not 1 <= page_size <= _MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")

        # ``abspath`` normalizes lexically without resolving symlinks.  The
        # constructor deliberately performs no filesystem operation.
        self._directory = Path(os.path.abspath(directory_value))
        self._page_size = page_size
        self._iterator: os.ScandirIterator | None = None
        self._candidates: list[Path] = []
        self._at_eof = False
        self._state = "initial"
        self._close_requested = False
        self._explicit_close = False
        self._failure: BaseException | None = None
        self._failure_reason: str | None = None
        self._failure_exposed = False
        self._completed_step: SweepStep | None = None

    @property
    def is_closed(self) -> bool:
        """Return whether this sweep has released its directory ownership."""

        return self._state == "closed"

    def request_close(self) -> None:
        """Request accounted iterator cleanup without performing I/O."""

        if self.is_closed:
            return
        self._close_requested = True
        self._explicit_close = True
        self._candidates.clear()
        if self._iterator is None:
            self._completed_step = None
            self._state = "closed"

    def advance(
        self,
        io: SliceIO,
        *,
        max_entries: int = _MAX_PAGE_SIZE,
        soft_deadline: float | None = None,
    ) -> SweepStep:
        """Perform bounded directory discovery using ``io`` for every syscall."""

        if not isinstance(io, SliceIO):
            raise TypeError("io must be a SliceIO")
        if type(max_entries) is not int:
            raise TypeError("max_entries must be an exact int")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        deadline = _validate_deadline(soft_deadline)

        if self._state == "closed":
            return SweepStep("closed")
        if self._completed_step is not None:
            return self._completed_step
        if self._state == "failed" and self._iterator is None:
            return self._failed_step()

        if self._close_requested:
            return self._advance_close(io, deadline)

        if self._iterator is None:
            if _deadline_reached(deadline):
                return SweepStep("waiting", reason="deadline")
            try:
                iterator = io.scandir(self._directory)
            except BaseException as exc:
                return self._source_error(exc)
            if iterator is YIELD:
                return SweepStep("waiting", reason="budget")
            self._iterator = iterator
            self._state = "scanning"

        if self._at_eof:
            return self._advance_close(io, deadline)

        entries_used = 0
        while entries_used < max_entries:
            if _deadline_reached(deadline):
                return SweepStep("waiting", reason="deadline")
            iterator = self._iterator
            if iterator is None:
                raise RuntimeError("source sweep lost its directory iterator")
            try:
                entry = io.next_entry(iterator)
            except BaseException as exc:
                return self._source_error(exc)
            if entry is YIELD:
                return SweepStep("waiting", reason="budget")
            entries_used += 1
            if entry is None:
                self._at_eof = True
                return self._advance_close(io, deadline)
            try:
                candidate = self._candidate_path(entry)
            except BaseException as exc:
                return self._source_error(exc)
            if candidate is None:
                continue
            self._candidates.append(candidate)
            if len(self._candidates) == self._page_size:
                page = tuple(self._candidates)
                self._candidates.clear()
                return SweepStep("page", page)

        return SweepStep("progressed")

    def _advance_close(self, io: SliceIO, deadline: float | None) -> SweepStep:
        if _deadline_reached(deadline):
            return SweepStep("waiting", reason="deadline")
        iterator = self._iterator
        if iterator is None:
            if self._explicit_close:
                self._state = "closed"
                self._completed_step = None
                self._candidates.clear()
                return SweepStep("closed")
            if self._failure is not None:
                self._state = "failed"
                return self._failed_step()
            if self._at_eof:
                return self._complete()
            self._state = "closed"
            return SweepStep("closed")
        try:
            result = io.close_directory(iterator)
        except BaseException:
            # Keep the iterator and the close phase intact.  The caller may
            # retry with a later slice; claiming closure here would leak the
            # ownership contract whenever the close call was interrupted.
            raise
        if result is YIELD:
            return SweepStep("waiting", reason="budget")

        self._iterator = None
        if self._explicit_close:
            self._state = "closed"
            self._completed_step = None
            self._candidates.clear()
            return SweepStep("closed")
        if self._failure is not None:
            self._state = "failed"
            self._candidates.clear()
            return self._failed_step()
        return self._complete()

    def _complete(self) -> SweepStep:
        candidates = tuple(self._candidates)
        self._candidates.clear()
        self._state = "complete"
        self._completed_step = SweepStep("complete", candidates)
        return self._completed_step

    def _candidate_path(self, entry: os.DirEntry) -> Path | None:
        name = entry.name
        if not isinstance(name, str) or not name.endswith(".json"):
            return None
        operation_id = name[:-5]
        if not operation_id:
            return None
        try:
            validate_identifier(operation_id, "submission operation id")
        except (TypeError, ValueError):
            return None
        return self._directory / name

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            self._failure_reason = f"{type(error).__name__}: {error}"
        self._state = "failed"
        self._close_requested = True
        self._candidates.clear()

    def _source_error(self, error: BaseException) -> SweepStep:
        self._record_failure(error)
        if not self._failure_exposed:
            self._failure_exposed = True
            raise error
        return self._failed_step()

    def _failed_step(self) -> SweepStep:
        return SweepStep("failed", reason=self._failure_reason)


def _validate_deadline(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("soft_deadline must be a real number or None")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("soft_deadline must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("soft_deadline must be finite")
    return result


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


__all__ = ["SubmissionSourceSweep", "SweepStep"]
