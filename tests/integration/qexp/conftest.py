import os

import pytest


@pytest.fixture(autouse=True)
def _qexp_integration_prerequisites(qexp_healthy_clock, monkeypatch, request):
    """Use deterministic clock proof and honor the explicit fast-I/O marker."""
    if request.node.get_closest_marker("qexp_fast_io") is not None:
        monkeypatch.setattr(os, "fsync", lambda _descriptor: None)
