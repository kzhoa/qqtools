import pytest
from datetime import datetime, timedelta, timezone
import time

from qqtools.plugins.qexp.lease import ClockCapability, ClockObservation


@pytest.fixture(autouse=True)
def _qexp_healthy_clock(monkeypatch):
    """Provide the bounded-clock deployment prerequisite to qexp unit tests."""
    observation = ClockObservation("unit-observation", "chrony", "2026-08-06T00:00:00Z",
                                   time.monotonic(), "unit-boot", -0.001, 0.001, 0.0, 0.0)
    capability = ClockCapability("healthy", "healthy", observation, ("chrony",))
    monkeypatch.setattr("qqtools.plugins.qexp.lease.clock_capability", lambda *_args: capability)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.clock_capability", lambda *_args: capability)
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.recovery.clock_capability", lambda *_args: capability)
    # Legacy state-machine tests exercise the post-gate expiry transition.  Clock proof math is
    # covered independently by test_clock_capability_local_safe.
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.reclaim_allowed_at",
        lambda *_args: datetime.now(timezone.utc) - timedelta(seconds=1),
    )
