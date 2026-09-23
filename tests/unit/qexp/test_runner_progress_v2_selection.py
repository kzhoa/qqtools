"""Optional v2 selection never stalls an authorized child launch."""

import threading
import time
import warnings

from qqtools.plugins.qexp import runner


def test_stalled_selection_read_times_out_without_join(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def stalled_read(_cfg, _task_id, *, warn):
        assert warn is False
        entered.set()
        release.wait(2)
        return True

    monkeypatch.setattr(runner, "read_task_live_progress", stalled_read)
    original_filters = warnings.filters[:]
    started = time.monotonic()
    try:
        assert runner._read_live_progress_bounded(object(), "task") is False
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert warnings.filters == original_filters
    finally:
        release.set()
