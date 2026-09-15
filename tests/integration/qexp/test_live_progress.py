"""Repository-level lifecycle tests for the optional live-progress sidecar."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, read_logs, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.runner import run_attempt
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.progress import ProgressProjector, local_progress_path, shared_progress_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task
from qqtools.qexp._progress_protocol import replace_advisory_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")


def launch(cfg, task, monkeypatch, *, code=0, payload=None):
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]["authorization"]["launch_id"]
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks", lambda pid: pid + 100)
    envs = []

    class Child:
        pid = 4321

        def wait(self):
            return code

    def popen(*args, **kwargs):
        envs.append(kwargs["env"])
        assert kwargs["env"]["QEXP_PROGRESS_PATH"] == str(local_progress_path(cfg.runtime_root, attempt.attempt_id))
        assert "QEXP_PROGRESS_FD" not in kwargs["env"]
        if payload is not None:
            replace_advisory_snapshot(Path(kwargs["env"]["QEXP_PROGRESS_PATH"]), payload)
        kwargs["stdout"].write(b"application event\n")
        return Child()

    result = run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                         launch_id, popen_factory=popen)
    assert result == code
    AuthoritySupervisor(cfg).tick()
    return attempt, envs[0]


def report(current=3, update_id="update-1"):
    return dict(protocol_version=1, update_id=update_id, stage="train",
                current=current, total=10, unit="step", message="epoch 0")


def test_short_command_final_report_survives_normal_evidence_cleanup(cfg, monkeypatch):
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/must-not-be-inherited")
    monkeypatch.setenv("QEXP_PROGRESS_FD", "99")
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    assert not (cfg.runtime_root / "process-registrations" / f"{attempt.attempt_id}.json").exists()
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    view = inspect_task(cfg, task.task_id)
    assert view["task"]["state"]["projection"] == "succeeded"
    assert view["progress"]["progress"]["current"] == 3
    assert read_logs(cfg, task.task_id) == "application event\n"
    assert not local_progress_path(cfg.runtime_root, attempt.attempt_id).exists()


def test_retry_never_displays_previous_attempt_as_current(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    old, _ = launch(cfg, task, monkeypatch, code=1, payload=report(7))
    # Keep old context/mailbox, as if the progress collector was offline.
    retry(cfg, task.task_id)
    assert inspect_task(cfg, task.task_id)["progress"]["status"] == "unavailable"
    new, _ = launch(cfg, task, monkeypatch, payload=report(2, "new-update"))
    assert old.attempt_id != new.attempt_id
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    view = inspect_task(cfg, task.task_id)["progress"]
    assert view["attempt_id"] == new.attempt_id
    assert view["progress"]["current"] == 2
    assert not shared_progress_path(cfg.shared_root, task.task_id, old.attempt_id).exists()


def test_inspector_rejects_stale_token(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    path = shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    value = read_json(path)
    value["fencing_token"] += 1
    replace_advisory_snapshot(path, value)
    assert inspect_task(cfg, task.task_id)["progress"]["status"] == "unavailable"


@pytest.mark.parametrize("payload", [None, {"protocol_version": 1000}, {"metrics": {"loss": 1.0}}])
def test_missing_or_malformed_progress_does_not_fail_task(cfg, monkeypatch, payload):
    task = submit(cfg, ["echo", "ok"])
    launch(cfg, task, monkeypatch, payload=payload)
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    view = inspect_task(cfg, task.task_id)
    assert view["task"]["state"]["projection"] == "succeeded"
    assert view["progress"]["status"] == "unavailable"


def test_cleanup_removes_progress_sidecars(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)
    assert not (cfg.shared_root / "progress" / task.task_id).exists()
    for directory in ("progress", "progress-contexts", "progress-observed", "progress-diagnostics"):
        assert not (cfg.runtime_root / directory / f"{attempt.attempt_id}.json").exists()


def test_real_guardian_inherits_channel_and_runs_public_api(cfg, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("qexp process guardian requires Linux")
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    monkeypatch.setenv("PYTHONPATH", source_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(key, raising=False)
    code = "from qqtools.qexp import progress; progress.update(stage='download',current=3,total=10,unit='file'); progress.flush(timeout=1); print('done')"
    task = submit(cfg, [sys.executable, "-c", code])
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]["authorization"]["launch_id"]

    class BoundedChild:
        def __init__(self, *args, **kwargs):
            self.child = subprocess.Popen(*args, **kwargs)
            self.pid = self.child.pid

        def wait(self):
            try:
                return self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.pid, 9)
                self.child.wait(timeout=5)
                raise

    assert run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                       launch_id, popen_factory=BoundedChild) == 0
    AuthoritySupervisor(cfg).tick()
    projector = ProgressProjector(cfg)
    projector.tick()
    projector.close()
    assert inspect_task(cfg, task.task_id)["progress"]["progress"]["current"] == 3
    assert read_logs(cfg, task.task_id).strip() == "done"


def test_progress_loop_is_separate_and_stop_is_bounded(monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    from qqtools.plugins.qexp.agent import progress_loop

    entered, release = threading.Event(), threading.Event()
    binding = SimpleNamespace(project_id="p", registration_generation="g", enabled=True)

    class Runtime:
        def load_registry(self):
            return 1, [binding]

        def binding_write_eligible(self, selected, *, renew=False):
            assert selected is binding
            assert renew is False  # Observability must not renew identity authority.
            return True

    class SlowProjector:
        def __init__(self, cfg, **kwargs):
            pass

        def tick(self):
            entered.set()
            release.wait(5)

        def close(self):
            pass

    monkeypatch.setattr(progress_loop, "ProgressProjector", SlowProjector)
    monkeypatch.setattr(progress_loop.helpers, "_binding_config", lambda *args: object())
    loop = progress_loop.ProgressObservationLoop(Runtime())
    try:
        loop.start()
        assert entered.wait(2)
        assert loop._thread.name == "qexp-machine-progress"
        assert loop._thread.daemon
        before = time.monotonic()
        loop.stop()
        assert time.monotonic() - before < 2
    finally:
        release.set()
        loop.stop()
