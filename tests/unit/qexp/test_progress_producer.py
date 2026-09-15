import os
import threading
import time

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol import read_advisory_snapshot


@pytest.fixture(autouse=True)
def clean_reporter(monkeypatch):
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "QEXP_PROGRESS_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(progress, "_reporter", None)
    yield
    progress.flush()


def test_noop_outside_qexp():
    assert progress.update(stage="train", current=1) is False
    assert progress._reporter is None


def test_final_flush_and_full_replacement(tmp_path):
    path = tmp_path / "progress.json"
    reporter = progress.Reporter(path)
    reporter.update(stage="train", current=9, total=10, unit="step")
    reporter.update(stage="validation")
    reporter.close(timeout=1)
    assert read_advisory_snapshot(path)["stage"] == "validation"
    assert read_advisory_snapshot(path)["current"] is None


def test_global_helper(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(path))
    assert progress.update(stage="train", current=4, total=5)
    progress.flush(timeout=1)
    assert read_advisory_snapshot(path)["current"] == 4


def test_rank_nonzero_and_fork_inheritance_are_noops(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    reporter = progress.Reporter(path)
    monkeypatch.setenv("RANK", "1")
    assert not reporter.update(stage="train")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(reporter, "_pid", os.getpid() + 1)
    assert not reporter.update(stage="train")
    assert not path.exists()


def test_invalid_and_missing_channel_do_not_raise(tmp_path):
    reporter = progress.Reporter(tmp_path / "absent" / "progress.json")
    assert not reporter.update(stage="train", current=-1)
    reporter.update(stage="train", current=1)
    reporter.close(timeout=1)


def test_broken_storage_does_not_escape_to_caller(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(progress, "replace_advisory_snapshot", fail)
    reporter = progress.Reporter(tmp_path / "progress.json")
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
    reporter = progress.Reporter(tmp_path / "progress.json")
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


def test_normal_updates_are_coalesced(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()
    def capture(path, value, **kwargs):
        writes.append(value["current"])
        first.set()
    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress.Reporter(tmp_path / "progress.json", interval_seconds=10)
    reporter.update(stage="train", current=0)
    assert first.wait(1)
    for i in range(1, 101):
        reporter.update(stage="train", current=i)
    reporter.close(timeout=1)
    assert writes == [0, 100]
