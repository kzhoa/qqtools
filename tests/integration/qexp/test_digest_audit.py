"""Integration coverage for the bounded qexp digest uniqueness audit."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import BinaryIO

import pytest

from tests.helpers.qexp.qexp_digest_audit import Audit, AuditStep

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


class _TraceFile:
    def __init__(self, path: Path, handle: BinaryIO, trace: "_IOTrace"):
        self.path = path
        self._handle = handle
        self._trace = trace
        trace.handles.append(self)

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def read(self, size: int = -1) -> bytes:
        self._trace.read_requests.append((self.path, size))
        data = self._handle.read(size)
        self._trace.read_returns.append((self.path, len(data)))
        return data

    def write(self, data: bytes) -> int:
        self._trace.write_requests.append((self.path, len(data)))
        written = self._handle.write(data)
        self._trace.write_returns.append((self.path, written))
        return written

    def close(self) -> None:
        self._handle.close()

    def flush(self) -> None:
        self._handle.flush()

    def __enter__(self) -> "_TraceFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


class _IOTrace:
    def __init__(self) -> None:
        self.read_requests: list[tuple[Path, int]] = []
        self.read_returns: list[tuple[Path, int]] = []
        self.write_requests: list[tuple[Path, int]] = []
        self.write_returns: list[tuple[Path, int]] = []
        self.handles: list[_TraceFile] = []


def _install_io_trace(monkeypatch: pytest.MonkeyPatch) -> _IOTrace:
    trace = _IOTrace()
    original_open = Path.open

    def traced_open(path: Path, *args, **kwargs) -> _TraceFile:
        return _TraceFile(path, original_open(path, *args, **kwargs), trace)

    monkeypatch.setattr(Path, "open", traced_open)
    return trace


def _digests(count: int) -> list[bytes]:
    return [hashlib.sha256(f"qexp-digest-{index:08d}".encode()).digest() for index in range(count)]


def _write_source(path: Path, records: list[bytes]) -> bytes:
    payload = b"".join(records)
    path.write_bytes(payload)
    return payload


def _run_audit(
    source: Path,
    scratch: Path,
    *,
    run_records: int,
    max_records: int,
    trace: _IOTrace | None = None,
) -> tuple[Audit, list[AuditStep]]:
    audit = Audit(source, scratch, run_records=run_records)
    steps: list[AuditStep] = []
    for _ in range(100_000):
        before_state = (
            audit._phase,
            audit._source_eof,
            audit._run_count,
            audit._pair_index,
            audit._pending_index,
        )
        before_reads = len(trace.read_returns) if trace is not None else 0
        before_writes = len(trace.write_returns) if trace is not None else 0
        result = audit.step(max_records)
        steps.append(result)
        if trace is not None:
            read_events = trace.read_returns[before_reads:]
            write_events = trace.write_returns[before_writes:]
            actual_reads = sum(size == 32 for _, size in read_events)
            actual_writes = sum(size == 32 for _, size in write_events)
            assert actual_reads == result.records_read
            assert actual_writes == result.records_written
            assert all(size == 32 for _, size in trace.read_requests[before_reads:])
            assert all(size == 32 for _, size in trace.write_requests[before_writes:])
        assert result.records_read <= max_records
        assert result.records_written <= max_records
        after_state = (
            audit._phase,
            audit._source_eof,
            audit._run_count,
            audit._pair_index,
            audit._pending_index,
        )
        if not result.records_read and not result.records_written and not result.is_complete:
            assert before_state != after_state
        if result.is_complete:
            return audit, steps
    raise AssertionError("audit did not complete within the bounded step limit")


@pytest.mark.parametrize("run_records", [1, 7, 64])
@pytest.mark.parametrize("max_records", [1, 3, 64])
def test_unique_digest_audit_sorts_without_changing_source(
    tmp_path: Path, run_records: int, max_records: int, monkeypatch: pytest.MonkeyPatch
):
    records = _digests(257)
    records.reverse()
    source = tmp_path / "source.bin"
    original = _write_source(source, records)
    trace = _install_io_trace(monkeypatch)

    audit, steps = _run_audit(
        source,
        tmp_path / "scratch",
        run_records=run_records,
        max_records=max_records,
        trace=trace,
    )

    assert steps[-1] == AuditStep(steps[-1].records_read, steps[-1].records_written, True, False)
    assert all(not step.has_repeated_digest for step in steps)
    assert audit.result_path is not None
    assert audit.result_path.read_bytes() == b"".join(sorted(records))
    assert source.read_bytes() == original
    assert sum(step.records_read for step in steps) >= len(records)
    assert sum(step.records_written for step in steps) >= len(records)


def test_empty_and_single_digest_are_unique(tmp_path: Path):
    for name, records in (("empty", []), ("one", [b"x" * 32])):
        source = tmp_path / f"{name}.bin"
        _write_source(source, records)
        with Audit(source, tmp_path / f"{name}-scratch", run_records=1) as audit:
            result = _run_until_complete(audit, 1)
            assert result.is_complete
            assert not result.has_repeated_digest
            assert audit.result_path is not None
            assert audit.result_path.read_bytes() == b"".join(sorted(records))


@pytest.mark.parametrize(
    ("records", "run_records"),
    [
        ([b"a" * 32, b"b" * 32, b"a" * 32], 1),
        ([b"a" * 32, b"a" * 32], 7),
        ([b"z" * 32] * 100, 7),
    ],
)
def test_equal_digests_finish_as_ambiguous(tmp_path: Path, records: list[bytes], run_records: int):
    source = tmp_path / "duplicate.bin"
    _write_source(source, records)
    scratch = tmp_path / "scratch"
    with Audit(source, scratch, run_records=run_records) as audit:
        result = _run_until_complete(audit, 3)
        assert result.is_complete
        assert result.has_repeated_digest
        assert audit.result_path is None
        before = (len(list(scratch.iterdir())), result)
        assert audit.step(3) == AuditStep(0, 0, True, True)
        assert len(list(scratch.iterdir())) == before[0]


def test_partial_digest_is_rejected(tmp_path: Path):
    source = tmp_path / "partial.bin"
    source.write_bytes(b"x" * 31)
    with Audit(source, tmp_path / "scratch") as audit:
        with pytest.raises(ValueError, match="truncated digest"):
            audit.step(1)


def test_scratch_must_not_exist_and_numeric_inventory_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.bin"
    _write_source(source, _digests(100))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(FileExistsError):
        Audit(source, scratch)
    scratch.rmdir()

    audit = Audit(source, scratch, run_records=7)

    def reject_inventory(*args, **kwargs):
        raise AssertionError("directory inventory is outside the audit contract")

    with monkeypatch.context() as context:
        context.setattr(os, "scandir", reject_inventory)
        context.setattr(Path, "glob", reject_inventory)
        result = _run_until_complete(audit, 1)
        assert result.is_complete and not result.has_repeated_digest
    audit.close()


@pytest.mark.parametrize("value", [0, -1, 65, True, False, 1.5])
def test_invalid_run_size_is_rejected_before_scratch_creation(tmp_path: Path, value):
    source = tmp_path / "source.bin"
    original = _write_source(source, _digests(2))
    scratch = tmp_path / "scratch"
    expected_error = TypeError if isinstance(value, bool) or not isinstance(value, int) else ValueError
    with pytest.raises(expected_error):
        Audit(source, scratch, run_records=value)
    assert not scratch.exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5])
def test_invalid_step_budget_performs_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value):
    source = tmp_path / "source.bin"
    _write_source(source, _digests(2))
    trace = _install_io_trace(monkeypatch)
    with Audit(source, tmp_path / "scratch") as audit:
        expected_error = TypeError if isinstance(value, bool) or not isinstance(value, int) else ValueError
        with pytest.raises(expected_error):
            audit.step(value)
        assert trace.read_requests == []
        assert trace.write_requests == []
        assert not _run_until_complete(audit, 1).has_repeated_digest


@pytest.mark.parametrize("open_handles", [1, 2, 4], ids=["source", "formation", "merge"])
def test_close_closes_owned_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_handles: int):
    source = tmp_path / "source.bin"
    _write_source(source, _digests(10))
    trace = _install_io_trace(monkeypatch)
    audit = Audit(source, tmp_path / "scratch", run_records=7)
    for _ in range(1000):
        audit.step(1)
        if sum(not handle.closed for handle in trace.handles) == open_handles:
            break
    else:
        pytest.fail("audit never reached the requested open-handle boundary")
    audit.close()
    audit.close()
    assert trace.read_returns
    assert all(handle.closed for handle in trace.handles)


def test_large_audit_keeps_digest_working_set_bounded(tmp_path: Path, checkout_subprocess_env):
    # tracemalloc observes the entire process, including unrelated suite threads
    # and instrumentation. Measure the audit in its own process at the same bound.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from tests.integration.qexp.test_digest_audit import _assert_large_audit_working_set; "
            "_assert_large_audit_working_set(Path(sys.argv[1]))",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=checkout_subprocess_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_large_audit_working_set(tmp_path: Path):
    records = _digests(100_000)
    source = tmp_path / "large.bin"
    original = _write_source(source, records)
    audit = Audit(source, tmp_path / "scratch", run_records=64)
    total_read = 0
    tracemalloc.start()
    try:
        for _ in range(100_000):
            result = audit.step(64)
            total_read += result.records_read
            if result.is_complete:
                break
        else:
            raise AssertionError("large audit did not complete")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert audit.is_complete
    assert not audit.has_repeated_digest
    assert peak < 2 * 1024 * 1024, peak
    assert source.read_bytes() == original
    assert total_read >= len(records)


def _run_until_complete(audit: Audit, max_records: int) -> AuditStep:
    for _ in range(100_000):
        result = audit.step(max_records)
        if result.is_complete:
            return result
    raise AssertionError("audit did not complete")
