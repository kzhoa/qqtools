"""Worker shutdown must retain unfinished cleanup ownership."""

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.service import MachineGroupDiscoveryWorker


@pytest.mark.parametrize("method", ["_close_service", "_close_sweep"])
def test_cleanup_keeps_owner_past_previous_attempt_limit(monkeypatch, method):
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.group_discovery.service.time.sleep", lambda _: None)

    class InterruptedCleanup:
        attempts = 0
        is_closed = False

        def request_close(self):
            pass

        def advance(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts <= 20000:
                raise OSError("cleanup temporarily unavailable")
            self.is_closed = True

    owner = InterruptedCleanup()
    getattr(MachineGroupDiscoveryWorker, method)(owner)
    assert owner.is_closed
