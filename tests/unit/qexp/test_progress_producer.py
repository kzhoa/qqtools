import math
import os
import threading
import time
from collections import Counter
from enum import StrEnum

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol import read_advisory_snapshot


def _render_managed_test_message(parts):
    return str(parts[0])


@pytest.fixture(autouse=True)
def clean_reporter(monkeypatch):
    progress.flush(timeout=0)
    for name in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "QEXP_PROGRESS_PATH",
        "QEXP_PROGRESS_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(progress, "_reporter", None)
    yield
    progress.flush(timeout=1)


def test_noop_outside_qexp():
    assert progress.update(stage="train", current=1) is False
    assert progress._reporter is None


@pytest.mark.parametrize("value", [None, "", "0", "0.5", "nan", "inf", "-inf", "garbage"])
def test_missing_or_malformed_environment_interval_uses_safe_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("QEXP_PROGRESS_INTERVAL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("QEXP_PROGRESS_INTERVAL_SECONDS", value)
    assert progress._environment_interval() == 30.0


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


def test_timed_out_flush_fences_next_attempt_path_until_old_writer_exits(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(path, value, **kwargs):
        entered.set()
        release.wait(3)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", blocked)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(first))
    assert progress.update(stage="train", current=1)
    assert entered.wait(1)
    old = progress._reporter
    progress.flush(timeout=0.01)
    assert progress._reporter is old

    # A later Attempt must never reuse the still-running writer for the old path.
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(second))
    assert progress.update(stage="train", current=2) is False
    assert progress._reporter is old

    release.set()
    deadline = time.monotonic() + 1
    while not old.replaceable() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert old.replaceable()
    assert progress.update(stage="train", current=3)
    assert progress._reporter is not old
    assert progress._reporter.path == second


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
    reporter._failure_delay_seconds = 0.01
    assert not reporter.update(stage="train", current=-1)
    reporter.update(stage="train", current=1)
    reporter.close(timeout=1)


def test_broken_storage_does_not_escape_to_caller(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(progress, "replace_advisory_snapshot", fail)
    reporter = progress._Reporter(tmp_path / "progress.json")
    reporter._failure_delay_seconds = 0.01
    reporter.update(stage="train", current=1)
    reporter.close(timeout=1)
    assert reporter._thread is not None
    reporter._thread.join(timeout=5)
    assert not reporter._thread.is_alive()


def test_failed_storage_backs_off_and_keeps_latest_update(tmp_path, monkeypatch):
    attempts = []
    first_failure = threading.Event()
    release_failure = threading.Event()
    success = threading.Event()

    def write(path, value, **kwargs):
        attempts.append((time.monotonic(), value["current"]))
        if len(attempts) == 1:
            first_failure.set()
            assert release_failure.wait(2)
            raise OSError("unavailable")
        success.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", write)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=30)
    reporter._failure_delay_seconds = 0.05
    assert reporter.update(stage="train", current=1)
    assert first_failure.wait(1)
    assert reporter.update(stage="train", current=2)
    release_failure.set()
    assert success.wait(2)
    assert [value for _, value in attempts] == [1, 2]
    assert attempts[1][0] - attempts[0][0] >= 0.045
    reporter.close(timeout=1)


def test_failed_storage_backoff_escalates_caps_and_resets_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "_INITIAL_RETRY_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(progress, "_MAX_RETRY_DELAY_SECONDS", 0.06)
    attempted_at = []
    succeeded = threading.Event()

    def write(_path, _value, **_kwargs):
        attempted_at.append(time.monotonic())
        if len(attempted_at) <= 7:
            raise OSError("unavailable")
        succeeded.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", write)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=30)
    reporter._failure_delay_seconds = 0.01
    assert reporter.update(stage="train", current=1)
    assert succeeded.wait(2)
    intervals = [later - earlier for earlier, later in zip(attempted_at, attempted_at[1:])]
    assert len(intervals) == 7
    assert all(
        actual >= expected * 0.8 for actual, expected in zip(intervals, [0.01, 0.02, 0.04, 0.06, 0.06, 0.06, 0.06])
    )
    deadline = time.monotonic() + 1
    while reporter._failure_delay_seconds != 0.01 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reporter._failure_delay_seconds == 0.01
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


def test_identical_update_during_slow_initial_write_is_not_rewritten(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    writes = []

    def blocked(path, value, **kwargs):
        writes.append(value["current"])
        entered.set()
        release.wait(2)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", blocked)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=1)
    assert reporter.update(stage="train", current=1)
    assert entered.wait(1)
    for _ in range(100):
        assert reporter.update(stage="train", current=1)
    release.set()
    reporter.close(timeout=1)
    assert writes == [1]


def test_slow_ordinary_write_preserves_phase_without_shortening_minimum_spacing(tmp_path, monkeypatch):
    clock = [0.0]
    completions = []
    written = threading.Event()

    def slow_write(path, value, **kwargs):
        clock[0] += 5
        completions.append(clock[0])
        written.set()

    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(progress, "replace_advisory_snapshot", slow_write)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=30)
    assert reporter.update(stage="train", current=1)
    assert written.wait(1)
    while reporter._last_written_key is None:
        time.sleep(0.001)

    written.clear()
    clock[0] = reporter._next_due
    assert reporter.update(stage="train", current=2)
    assert written.wait(1)
    while reporter._last_written_key[-4] != 2:
        time.sleep(0.001)

    assert reporter._next_due >= completions[-1] + 30
    phase_remainder = (reporter._next_due - (completions[0] + 30 + progress._path_offset(reporter.path, 30))) % 30
    assert math.isclose(phase_remainder, 0.0, abs_tol=1e-9) or math.isclose(phase_remainder, 30.0, abs_tol=1e-9)
    reporter.close(timeout=1)


def test_large_finite_interval_keeps_writer_alive_and_final_flushes(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()

    def capture(path, value, **kwargs):
        writes.append(value["current"])
        first.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=1e308)
    assert reporter.update(stage="train", current=1)
    assert first.wait(1)
    assert reporter.update(stage="train", current=2)
    time.sleep(0.02)
    assert reporter._thread is not None and reporter._thread.is_alive()
    reporter.close(timeout=1)
    assert writes == [1, 2]


@pytest.mark.parametrize("interval", [30.0, 60.0])
def test_producer_path_phases_bound_synthetic_scale_bursts(tmp_path, interval):
    bursts = Counter()
    for index in range(1000):
        path = tmp_path / f"attempt-{index}" / "latest.json"
        deadline = interval + progress._path_offset(path, interval)
        writes = 0
        while deadline < 11 * interval:
            bursts[int(deadline)] += 1
            writes += 1
            deadline = progress._next_deadline(deadline, deadline, interval)
        assert writes == 10
    assert max(bursts.values()) <= 2 * math.ceil(1000 / interval)


def test_close_does_not_wait_unbounded_for_state_lock(tmp_path):
    reporter = progress._Reporter(tmp_path / "progress.json")
    assert reporter._lock.acquire(blocking=False)
    try:
        start = time.monotonic()
        reporter.close(timeout=0.02)
        assert time.monotonic() - start < 0.2
    finally:
        reporter._lock.release()


def test_managed_close_does_not_wait_for_blocked_writer(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        release.wait(3)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", blocked)
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(tmp_path / "progress.json"))
    assert progress._offer_managed_progress(
        stage="train", current=1, message_parts=("ready",), render_message=_render_managed_test_message
    )
    assert entered.wait(1)
    reporter = progress._reporter

    start = time.monotonic()
    progress._close_managed_progress()
    assert time.monotonic() - start < 0.1
    assert progress._reporter is reporter
    assert progress.update(stage="train", current=2) is False

    release.set()
    progress.flush(timeout=1)


def test_managed_close_does_not_wait_for_reporter_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(tmp_path / "progress.json"))
    assert progress._offer_managed_progress(
        stage="train", current=1, message_parts=("ready",), render_message=_render_managed_test_message
    )
    reporter = progress._reporter
    assert progress._reporter_lock.acquire(blocking=False)
    try:
        start = time.monotonic()
        progress._close_managed_progress()
        assert time.monotonic() - start < 0.1
        assert reporter._close_requested.is_set()
    finally:
        progress._reporter_lock.release()
    progress.flush(timeout=1)


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


def test_stage_changes_wait_for_interval_and_close_flushes_once(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()

    def capture(path, value, **kwargs):
        writes.append((value["stage"], value["current"]))
        first.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=1)
    reporter.update(stage="train", current=1)
    assert first.wait(1)
    for index in range(100):
        reporter.update(stage=f"stage-{index}", current=index)
    time.sleep(0.2)
    assert writes == [("train", 1)]

    reporter.close(timeout=1)
    reporter.close(timeout=1)
    assert writes == [("train", 1), ("stage-99", 99)]


def test_identical_updates_do_not_create_ordinary_rewrites(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()

    def capture(path, value, **kwargs):
        writes.append(value)
        first.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=1)
    assert reporter.update(stage="train", current=1, total=10, unit="step", message="same")
    assert first.wait(1)
    for _ in range(100):
        assert reporter.update(stage="train", current=1, total=10, unit="step", message="same")
    time.sleep(1.1)
    reporter.close(timeout=1)

    assert len(writes) == 1


def test_latest_accepted_value_replaces_a_pending_change_even_if_previously_written(tmp_path, monkeypatch):
    writes = []
    first = threading.Event()

    def capture(path, value, **kwargs):
        writes.append(value["current"])
        first.set()

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=30)
    assert reporter.update(stage="train", current=1)
    assert first.wait(1)
    deadline = time.monotonic() + 1
    while reporter._last_written_key is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reporter._last_written_key is not None
    assert reporter.update(stage="train", current=2)
    assert reporter.update(stage="train", current=1)
    reporter.close(timeout=1)

    assert writes[0] == 1
    assert writes[-1] == 1
    assert 2 not in writes


def test_latest_accepted_value_survives_inflight_different_write(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    writes = []

    def capture(path, value, **kwargs):
        writes.append(value["current"])
        if value["current"] == 2:
            entered.set()
            release.wait(3)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", capture)
    reporter = progress._Reporter(tmp_path / "progress.json", interval_seconds=30)
    assert reporter.update(stage="train", current=1)
    deadline = time.monotonic() + 1
    while reporter._last_written_key is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reporter._last_written_key is not None

    with reporter._lock:
        reporter._next_due = float("-inf")
    assert reporter.update(stage="train", current=2)
    assert entered.wait(1)
    assert reporter.update(stage="train", current=1)
    release.set()
    reporter.close(timeout=1)

    assert writes[:2] == [1, 2]
    assert writes[-1] == 1


def test_public_update_accepts_string_subclasses_without_executing_overrides(tmp_path, monkeypatch):
    class Phase(StrEnum):
        TRAIN = "train"

    class Text(str):
        def __str__(self):
            raise AssertionError("must not execute user conversion")

        def encode(self, *args, **kwargs):
            raise AssertionError("must not execute user encoding")

        def __eq__(self, other):
            raise AssertionError("must not retain user comparison")

    path = tmp_path / "progress.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(path))
    assert progress.update(stage=Phase.TRAIN, current=1, total=2, unit=Text("step"), message=Text("ready"))
    progress.flush(timeout=1)
    payload = read_advisory_snapshot(path)
    assert (payload["stage"], payload["unit"], payload["message"]) == ("train", "step", "ready")


def test_managed_update_rejects_nonprimitive_text(tmp_path):
    class Phase(StrEnum):
        TRAIN = "train"

    reporter = progress._Reporter(tmp_path / "progress.json")
    assert not reporter.update(
        stage=Phase.TRAIN, _message_parts=("ready",), _render_message=_render_managed_test_message
    )
    assert reporter._thread is None
