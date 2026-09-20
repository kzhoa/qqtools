"""Bounded-memory provisional uniqueness audit for fixed-width SHA-256 digests.

The audit consumes a caller-owned immutable file containing concatenated
32-byte digest values.  It creates sorted runs in a newly created scratch
directory and merges those runs incrementally.  A repeated digest is reported
as ambiguous: the audit cannot distinguish a duplicate from a valid hash
collision, so it never certifies uniqueness after an equality.

Source stability and restart safety are caller responsibilities.  A caller
must discard provisional results if the source changes or the audit process
stops before completion.  This test oracle makes no latency, durability, or
crash-recovery promise for scratch files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_DIGEST_BYTES = 32
_MAX_RUN_RECORDS = 64
_BUDGET_EXHAUSTED = object()
_END_OF_FILE = object()


@dataclass(frozen=True, slots=True)
class AuditStep:
    """Counters and terminal state returned by :meth:`Audit.step`."""

    records_read: int
    records_written: int
    is_complete: bool
    has_repeated_digest: bool


@dataclass(slots=True)
class _StepBudget:
    max_records: int
    records_read: int = 0
    records_written: int = 0


class Audit:
    """Incrementally audit uniqueness of a fixed-width digest file.

    Args:
        source: Existing file containing concatenated 32-byte digest values.
        scratch: Path that must not exist; the audit creates and owns it.
        run_records: Maximum number of digests retained while forming one run.

    Raises:
        FileExistsError: If ``scratch`` already exists.
        TypeError: If a numeric argument is not an integer, or is boolean.
        ValueError: If a numeric argument is not positive or exceeds the
            maximum run size.

    ``step`` returns a terminal ambiguous result when an equal digest is
    found.  ``is_complete`` therefore means that the audit has finished, not
    that uniqueness was certified; a unique result has
    ``is_complete=True`` and ``has_repeated_digest=False``.
    """

    def __init__(self, source: Path, scratch: Path, *, run_records: int = _MAX_RUN_RECORDS):
        self._run_records = _validate_run_records(run_records)
        self.source_path = Path(source)
        self.scratch_path = Path(scratch)

        self._source: BinaryIO | None = None
        self._formation_output: BinaryIO | None = None
        self._left_input: BinaryIO | None = None
        self._right_input: BinaryIO | None = None
        self._merge_output: BinaryIO | None = None

        self._phase = "forming"
        self._source_eof = False
        self._forming_buffer: list[bytes] = []
        self._pending_run: list[bytes] | None = None
        self._pending_index = 0
        self._run_count = 0

        self._merge_pass = 0
        self._pair_index = 0
        self._left_next: bytes | None = None
        self._right_next: bytes | None = None
        self._left_eof = False
        self._right_eof = False
        self._merge_previous: bytes | None = None

        self._is_terminal = False
        self._has_repeated_digest = False
        self._result_path: Path | None = None
        self._closed = False
        self._failed = False

        try:
            self._source = self.source_path.open("rb")
            self.scratch_path.mkdir()
        except Exception:
            if self._source is not None:
                self._source.close()
                self._source = None
            raise

    @property
    def result_path(self) -> Path | None:
        """Sorted digest file after a unique audit, otherwise ``None``."""

        return self._result_path

    @property
    def is_complete(self) -> bool:
        """Whether the audit has reached a unique or ambiguous terminal state."""

        return self._is_terminal

    @property
    def has_repeated_digest(self) -> bool:
        """Whether the terminal audit found an equal digest."""

        return self._has_repeated_digest

    def __enter__(self) -> Audit:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close all handles owned by this audit without deleting scratch files."""

        if self._closed:
            return
        try:
            self._close_all_handles()
        finally:
            self._closed = True

    def step(self, max_records: int = _MAX_RUN_RECORDS) -> AuditStep:
        """Perform bounded read/write work and return exact per-step counters.

        At most ``max_records`` complete records are read and at most the same
        number are written during a call.  Every file read requests exactly
        32 bytes; an incomplete nonempty record raises ``ValueError``.
        """

        max_records = _validate_positive_int(max_records, "max_records")
        if self._is_terminal:
            return AuditStep(0, 0, True, self._has_repeated_digest)
        if self._closed:
            raise RuntimeError("audit is closed")
        if self._failed:
            raise RuntimeError("audit is unusable after a failed step")

        budget = _StepBudget(max_records)
        try:
            if self._phase == "forming":
                self._step_forming(budget)
            elif self._phase == "merging":
                self._step_merging(budget)
            else:
                raise RuntimeError(f"unknown audit phase: {self._phase}")
        except Exception:
            self._failed = True
            self._close_all_handles()
            raise

        return AuditStep(
            budget.records_read,
            budget.records_written,
            self._is_terminal,
            self._has_repeated_digest,
        )

    def _step_forming(self, budget: _StepBudget) -> None:
        while not self._is_terminal:
            if self._pending_run is not None:
                if not self._flush_pending_run(budget):
                    return
                continue

            if self._source_eof:
                if self._forming_buffer:
                    self._prepare_formation_run()
                    continue
                self._begin_merge()
                return

            if len(self._forming_buffer) >= self._run_records:
                self._prepare_formation_run()
                continue

            record = self._read_source_record(budget)
            if record is _BUDGET_EXHAUSTED:
                return
            if record is _END_OF_FILE:
                self._source_eof = True
                continue
            self._forming_buffer.append(record)

    def _prepare_formation_run(self) -> None:
        run = self._forming_buffer
        run.sort()
        if _has_adjacent_equal(run):
            self._finish_terminal(has_repeated_digest=True)
            return

        self._forming_buffer = []
        self._pending_run = run
        self._pending_index = 0
        path = self._run_path(0, self._run_count)
        self._formation_output = path.open("wb")

    def _flush_pending_run(self, budget: _StepBudget) -> bool:
        run = self._pending_run
        output = self._formation_output
        if run is None or output is None:
            raise RuntimeError("formation run is missing its output handle")

        while self._pending_index < len(run):
            if not _write_record(output, run[self._pending_index], budget):
                return False
            self._pending_index += 1

        output.close()
        self._formation_output = None
        self._pending_run = None
        self._pending_index = 0
        self._run_count += 1
        return True

    def _begin_merge(self) -> None:
        if self._run_count == 0:
            path = self.scratch_path / "result.bin"
            with path.open("wb"):
                pass
            self._result_path = path
            self._finish_terminal(has_repeated_digest=False)
            return
        if self._run_count == 1:
            self._result_path = self._run_path(0, 0)
            self._finish_terminal(has_repeated_digest=False)
            return

        self._phase = "merging"
        self._merge_pass = 0
        self._pair_index = 0

    def _step_merging(self, budget: _StepBudget) -> None:
        while not self._is_terminal:
            if self._run_count <= 1:
                self._result_path = self._run_path(self._merge_pass, 0)
                self._finish_terminal(has_repeated_digest=False)
                return

            pair_count = self._run_count // 2
            if self._pair_index < pair_count:
                if self._left_input is None:
                    self._open_merge_pair()
                if not self._merge_pair(budget):
                    return
                self._finish_merge_pair()
                continue

            self._advance_merge_pass()

    def _open_merge_pair(self) -> None:
        left_number = self._pair_index * 2
        left_path = self._run_path(self._merge_pass, left_number)
        right_path = self._run_path(self._merge_pass, left_number + 1)
        output_path = self._run_path(self._merge_pass + 1, self._pair_index)
        self._left_input = left_path.open("rb")
        try:
            self._right_input = right_path.open("rb")
            self._merge_output = output_path.open("wb")
        except Exception:
            if self._right_input is not None:
                self._right_input.close()
                self._right_input = None
            self._left_input.close()
            self._left_input = None
            raise
        self._left_next = None
        self._right_next = None
        self._left_eof = False
        self._right_eof = False
        self._merge_previous = None

    def _merge_pair(self, budget: _StepBudget) -> bool:
        left_input = self._left_input
        right_input = self._right_input
        output = self._merge_output
        if left_input is None or right_input is None or output is None:
            raise RuntimeError("merge pair is missing an input or output handle")

        while True:
            if self._left_next is None and not self._left_eof:
                record = _read_run_record(left_input, budget, "left merge run")
                if record is _BUDGET_EXHAUSTED:
                    return False
                if record is _END_OF_FILE:
                    self._left_eof = True
                else:
                    self._left_next = record
                continue

            if self._right_next is None and not self._right_eof:
                record = _read_run_record(right_input, budget, "right merge run")
                if record is _BUDGET_EXHAUSTED:
                    return False
                if record is _END_OF_FILE:
                    self._right_eof = True
                else:
                    self._right_next = record
                continue

            if self._left_next is None and self._right_next is None:
                return True

            if self._left_next is not None and self._right_next is not None:
                if self._left_next == self._right_next:
                    self._finish_terminal(has_repeated_digest=True)
                    return False
                if self._left_next < self._right_next:
                    chosen = self._left_next
                    self._left_next = None
                    chosen_from_left = True
                else:
                    chosen = self._right_next
                    self._right_next = None
                    chosen_from_left = False
            elif self._left_next is not None:
                chosen = self._left_next
                self._left_next = None
                chosen_from_left = True
            else:
                chosen = self._right_next
                self._right_next = None
                chosen_from_left = False

            if self._merge_previous == chosen:
                self._finish_terminal(has_repeated_digest=True)
                return False
            if not _write_record(output, chosen, budget):
                if chosen_from_left:
                    self._left_next = chosen
                else:
                    self._right_next = chosen
                return False
            self._merge_previous = chosen

    def _finish_merge_pair(self) -> None:
        left_number = self._pair_index * 2
        left_path = self._run_path(self._merge_pass, left_number)
        right_path = self._run_path(self._merge_pass, left_number + 1)

        if self._merge_output is not None:
            self._merge_output.close()
            self._merge_output = None
        if self._left_input is not None:
            self._left_input.close()
            self._left_input = None
        if self._right_input is not None:
            self._right_input.close()
            self._right_input = None
        left_path.unlink()
        right_path.unlink()

        self._left_next = None
        self._right_next = None
        self._left_eof = False
        self._right_eof = False
        self._merge_previous = None
        self._pair_index += 1

    def _advance_merge_pass(self) -> None:
        old_pass = self._merge_pass
        old_count = self._run_count
        if old_count % 2:
            odd_path = self._run_path(old_pass, old_count - 1)
            odd_destination = self._run_path(old_pass + 1, old_count // 2)
            odd_path.rename(odd_destination)

        self._merge_pass += 1
        self._run_count = (old_count + 1) // 2
        self._pair_index = 0

    def _read_source_record(self, budget: _StepBudget) -> object:
        source = self._source
        if source is None:
            raise RuntimeError("source handle is unavailable")
        return _read_record(source, budget, "source")

    def _run_path(self, pass_number: int, run_number: int) -> Path:
        return self.scratch_path / f"pass-{pass_number}-{run_number:08d}.bin"

    def _finish_terminal(self, *, has_repeated_digest: bool) -> None:
        self._has_repeated_digest = has_repeated_digest
        self._is_terminal = True
        self._phase = "terminal"
        self._close_all_handles()

    def _close_all_handles(self) -> None:
        handles = (
            ("_source", self._source),
            ("_formation_output", self._formation_output),
            ("_left_input", self._left_input),
            ("_right_input", self._right_input),
            ("_merge_output", self._merge_output),
        )
        for name, handle in handles:
            if handle is not None:
                handle.close()
            setattr(self, name, None)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_run_records(value: int) -> int:
    value = _validate_positive_int(value, "run_records")
    if value > _MAX_RUN_RECORDS:
        raise ValueError(f"run_records must be at most {_MAX_RUN_RECORDS}")
    return value


def _has_adjacent_equal(records: list[bytes]) -> bool:
    return any(records[index] == records[index - 1] for index in range(1, len(records)))


def _read_record(handle: BinaryIO, budget: _StepBudget, label: str) -> object:
    if budget.records_read >= budget.max_records:
        return _BUDGET_EXHAUSTED
    data = handle.read(_DIGEST_BYTES)
    if data is None or not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} read() must return bytes")
    record = bytes(data)
    if not record:
        return _END_OF_FILE
    if len(record) != _DIGEST_BYTES:
        raise ValueError(f"{label} ended with a truncated digest of {len(record)} bytes")
    budget.records_read += 1
    return record


def _read_run_record(handle: BinaryIO, budget: _StepBudget, label: str) -> object:
    return _read_record(handle, budget, label)


def _write_record(handle: BinaryIO, record: bytes, budget: _StepBudget) -> bool:
    if budget.records_written >= budget.max_records:
        return False
    written = handle.write(record)
    if written != _DIGEST_BYTES:
        raise OSError(f"digest write returned {written!r}, expected {_DIGEST_BYTES}")
    budget.records_written += 1
    return True
