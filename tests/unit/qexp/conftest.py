import pytest


@pytest.fixture(autouse=True)
def _qexp_healthy_clock(monkeypatch):
    """Provide the bounded-clock deployment prerequisite to qexp unit tests."""
    monkeypatch.setattr(
        "qqtools.plugins.qexp.lease.chrony_health", lambda _policy: (True, "healthy")
    )
