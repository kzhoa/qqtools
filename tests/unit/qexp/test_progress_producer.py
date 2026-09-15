import os
import threading
import time

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol import read_advisory_snapshot


@pytest.fixture(autouse=True)
def clean_reporter(monkeypatch):
    progress.flush(timeout=0)
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "QEXP_PROGRESS_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(progress, "_reporter", None)
    yield
    progress.flush(timeout=1)


def test_noop_outside_qexp():
    assert progress.update(stage="train", current=1) is False
    assert progress._reporter is None


def test_final_flush_and_full_replacement(tmp_path):
    path = tmp_path / "progress.json"
    reporter = progress._Reporter(path)
    reporter.update(stage="train", current=9, total=10, unit="step")
    reporter.update(stage="validation")
    reporter.close(timeout=1)
    assert read_advisory_snapshot(path)["stage"] == "validation"
    assert read_advisory_snapshot(path)["current"] is None


def test_global_helper_and_flush_reset(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(first))
    assert progress.update(stage="train", current=4, total=5)
    progress.flush(timeout=1)
    assert read_advisory_snapshot(first)["current"] == 4
    assert progress._reporter is None

    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(second))
    assert progress.update(stage="train", current=7, total=8)
    progress.flush(timeout=1)
    assert read_advisory_snapshot(second)["current"] == 7


def test_timed_out_flush_keeps_old_writer_as_singleton(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(path, value, **kwargs):
        entered.set()
        release.wait(3)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", blocked)
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(tmp_path / "progress.json"))
    assert progress.update(stage="train", current=1)
    assert entered.wait(1)
    old = progress._reporter
    progress.flush(timeout=0.01)
    assert progress._reporter is old
    assert progress.update(stage="train", current=2) is False
    release.set()
    deadline = time.monotonic() + 1
    while not old.replaceable() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert old.replaceable()
    assert progress.update(stage="train", current=3)
    assert progress._reporter is not old


def test_rank_nonzero_and_fork_inheritance_are_noops(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    reporter = progress._Reporter(path)
    monkeypatch.setenv("RANK", "1")
    assert not reporter.update(stage="train")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(reporter, "_pid", os.getpid() + 1)
    assert not reporter.update(stage="train")
    assert not path.exists()


def test_invalid_and_missing_channel_do_not_raise(tmp_path):
    reporter = progress._Reporter(tmp_path / "absent" / "progress.json")
    assert not reporter.update(stage="train", current=-1)
    reporter.update(stage="train", current=1)
    reporter.close(timeout=1)


def test_broken_storage_does_not_escape_to_caller(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(progress, "replace_advisory_snapshot", fail)
    reporter = progress._Reporter(tmp_path / "progress.json")
    reporter.update(stage="train", current=1)
    reporter.close(timeout=1)


def test_blocked_storage_keeps_a_bounded_latest_slot(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    writes = []

    def blocked(path, value, **kwargs):
        entered.set()
        release.wait(3)
        writes.append(value["current"])

    monkeypatch.setattr(progress, "replace_advisory_snapshot", blocked)
    reporter = progress._Reporter(tmp_path / "progress.json")
    assert reporter.update(stage="train", current=0)
    assert entered.wait(1)
    for i in range(1, 1001):
        reporter.update(stage="train", current=i)
    assert reporter._pending["current"] == 1000
    start = time.monotonic()
    reporter.close(timeout=0.01)
    assert time.monotonic() - start < 0.5
    release.set()
    reporter.close(timeout=1)
    assert writes == [0, 1000]


def test_close_does_not_wait_unbounded_for_state_lock(tmp_path):
    reporter = progress._Reporter(tmp_path / "progress.json")
    assert reporter._lock.acquire(blocking=False)
    try:
        start = time.monotonic()
        reporter.close(timeout=0.02)
        assert time.monotonic() - start < 0.2
    finally:
        reporter._lock.release()


def test_normal_updates_are_coalesced(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()

    def capture(path, value, **kwargs):
        writes.append(value["current"])
        first.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=10)
    reporter.update(stage="train", current=0)
    assert first.wait(1)
    for i in range(1, 101):
        reporter.update(stage="train", current=i)
    reporter.close(timeout=1)
    assert writes == [0, 100]
