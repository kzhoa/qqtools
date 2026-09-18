"""Scoped authority measurements preserve real storage and lock behavior."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.locks import exclusive, shared
from qqtools.plugins.qexp.runtime.store import (
    CASConflict,
    atomic_replace,
    create_if_absent,
    iter_json,
    read_json,
    read_json_limited,
)
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

pytestmark = pytest.mark.integration


def test_storage_measurements_count_operations_without_changing_durability(tmp_path: Path) -> None:
    diagnostics = RuntimeDiagnostics()
    records = tmp_path / "records"
    record = records / "record.json"
    with activate_diagnostics(diagnostics):
        create_if_absent(record, {"revision": 1})
        atomic_replace(record, {"revision": 2})
        assert read_json(record) == {"revision": 2}
        assert read_json_limited(record, max_bytes=100) == {"revision": 2}
        (records / "ignored.txt").touch()
        assert iter_json(records) == [record]
    counters = diagnostics.snapshot()["counters"]
    assert counters == {
        "store.atomic_replace.calls": 1,
        "store.create_if_absent.calls": 1,
        "store.fsync.calls": 3,
        "store.iter_json.calls": 1,
        "store.inventory_entries": 2,
        "store.read_json.calls": 1,
        "store.read_json_limited.calls": 1,
    }
    assert read_json(record) == {"revision": 2}
    assert diagnostics.snapshot()["counters"] == counters


def test_failed_io_and_lock_contention_are_measured_and_propagated(tmp_path: Path) -> None:
    diagnostics = RuntimeDiagnostics()
    record = tmp_path / "record.json"
    lock = tmp_path / "record.lock"
    with activate_diagnostics(diagnostics):
        with pytest.raises(FileNotFoundError):
            read_json(record)
        create_if_absent(record, {})
        with pytest.raises(CASConflict):
            create_if_absent(record, {})
        with exclusive(lock) as acquired:
            assert acquired
            with exclusive(lock, blocking=False) as acquired_again:
                assert not acquired_again
            with shared(lock, blocking=False) as acquired_shared:
                assert not acquired_shared
        with shared(lock) as acquired_shared:
            assert acquired_shared
    snapshot = diagnostics.snapshot()
    assert snapshot["counters"]["locks.acquire.calls"] == 4
    assert snapshot["counters"]["store.read_json.calls"] == 1
    assert snapshot["counters"]["store.create_if_absent.calls"] == 2
    assert snapshot["counters"]["store.fsync.calls"] == 1
    assert snapshot["timings"]["locks.acquire"]["total_ns"] > 0
    assert snapshot["timings"]["store.read_json"]["total_ns"] > 0
