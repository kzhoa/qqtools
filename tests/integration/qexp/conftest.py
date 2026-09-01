import os

import pytest
@pytest.fixture(autouse=True)
def _qexp_integration_prerequisites(
    qexp_healthy_clock,
    qexp_resource_scope,
    monkeypatch,
    request,
):
    """Use deterministic clock proof and honor the explicit fast-I/O marker."""
    environment = qexp_resource_scope.child_environment()
    for name in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "TMUX_TMPDIR",
        "QEXP_MACHINE_RUNTIME_ROOT",
    ):
        monkeypatch.setenv(name, environment[name])
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_runtime.tempfile.gettempdir",
        lambda: str(qexp_resource_scope.local_temp_root),
    )
    if request.node.get_closest_marker("qexp_fast_io") is not None:
        monkeypatch.setattr(os, "fsync", lambda _descriptor: None)
