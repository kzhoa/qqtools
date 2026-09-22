"""Cross-plugin qPipeline progress integration through a real qexp attempt."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.runner import run_attempt
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.progress import ProgressProjector
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _qexp_integration_prerequisites(qexp_healthy_clock, qexp_resource_scope, monkeypatch):
    """Retain qexp's process and filesystem isolation outside its test subtree."""
    del qexp_healthy_clock
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


def test_unchanged_qpipeline_command_reaches_task_show_progress(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("qexp process guardian requires Linux")
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    source_root = str(Path(__file__).resolve().parents[4] / "src")
    monkeypatch.setenv("PYTHONPATH", source_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    fixture = Path(__file__).resolve().parents[3] / "fixtures" / "qexp_progress_qpipeline.py"
    task = submit(cfg, [sys.executable, str(fixture)])
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]

    class BoundedChild:
        def __init__(self, *args, **kwargs):
            self.child = subprocess.Popen(*args, **kwargs)
            self.pid = self.child.pid

        def wait(self):
            try:
                return self.child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(self.pid, 9)
                self.child.wait(timeout=5)
                raise

    assert (
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            launch_id,
            popen_factory=BoundedChild,
        )
        == 0
    )
    AuthoritySupervisor(cfg).tick()
    projector = ProgressProjector(cfg, registration_generation="test-generation")
    projector.tick()
    projector.close()
    view = inspect_task(cfg, task.task_id)

    assert view["progress"]["observation_state"] == "available"
    assert view["progress"]["progress"]["stage"] == "train"
    assert "Epoch 1" in view["progress"]["progress"]["message"]
