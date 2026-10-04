import os
from pathlib import Path

import pytest


@pytest.fixture
def qexp_subprocess_bounded_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The ordinary fixture patches only this pytest process. Provide the same
    # bounded-clock prerequisite to the real agent and peer through their provider
    # boundary; wall time, TTL and expiry/claim decisions remain production code.
    clock_bin = tmp_path / "clock-bin"
    clock_bin.mkdir()
    chronyc = clock_bin / "chronyc"
    chronyc.write_text(
        "#!/bin/sh\ncat <<'CLOCK'\n"
        "System time : 0.000001 seconds slow of NTP time\n"
        "Root delay : 0.000002 seconds\n"
        "Root dispersion : 0.000001 seconds\n"
        "Skew : 0.001 ppm\n"
        "Leap status : Normal\nCLOCK\n"
    )
    chronyc.chmod(0o755)
    monkeypatch.setenv("PATH", str(clock_bin) + os.pathsep + os.environ["PATH"])


@pytest.fixture(autouse=True)
def _qexp_integration_prerequisites(
    qexp_healthy_clock,
    qexp_subprocess_bounded_clock,
    qexp_resource_scope,
    monkeypatch,
    request,
):
    """Use deterministic clock proof and honor the explicit fast-I/O marker."""
    environment = qexp_resource_scope.child_environment()
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("TMUX_PANE", raising=False)
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
        "qqtools.plugins.qexp.agent.context.tempfile.gettempdir",
        lambda: str(qexp_resource_scope.local_temp_root),
    )
    if request.node.get_closest_marker("qexp_fast_io") is not None:
        monkeypatch.setattr(os, "fsync", lambda _descriptor: None)
