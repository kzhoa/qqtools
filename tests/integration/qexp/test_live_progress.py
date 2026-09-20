"""Repository-level lifecycle tests for the optional live-progress sidecar."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, read_logs, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.formatter import CliOutput, OutputKind, render
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.progress_policy import set_progress_policy
from qqtools.plugins.qexp.runner import run_attempt
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.progress import ProgressProjector, local_progress_path, shared_progress_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task
from qqtools.qexp._progress_protocol import replace_advisory_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

_TEST_GENERATION = "test-generation"


@pytest.fixture
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")


def projector(cfg):
    return ProgressProjector(cfg, registration_generation=_TEST_GENERATION)


def launch(cfg, task, monkeypatch, *, code=0, payload=None, expected_interval="30"):
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks", lambda pid: pid + 100)
    envs = []

    class Child:
        pid = 4321

        def wait(self):
            return code

    def popen(*args, **kwargs):
        envs.append(kwargs["env"])
        assert kwargs["env"]["QEXP_PROGRESS_PATH"] == str(local_progress_path(cfg.runtime_root, attempt.attempt_id))
        assert kwargs["env"]["QEXP_PROGRESS_INTERVAL_SECONDS"] == expected_interval
        assert "QEXP_PROGRESS_FD" not in kwargs["env"]
        if payload is not None:
            replace_advisory_snapshot(Path(kwargs["env"]["QEXP_PROGRESS_PATH"]), payload)
        kwargs["stdout"].write(b"application event\n")
        return Child()

    result = run_attempt(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        launch_id,
        popen_factory=popen,
    )
    assert result == code
    AuthoritySupervisor(cfg).tick()
    return attempt, envs[0]


def report(current=3, update_id="update-1"):
    return dict(
        protocol_version=1,
        update_id=update_id,
        stage="train",
        current=current,
        total=10,
        unit="step",
        message="epoch 0",
    )


def test_short_command_final_report_survives_normal_evidence_cleanup(cfg, monkeypatch):
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/must-not-be-inherited")
    monkeypatch.setenv("QEXP_PROGRESS_FD", "99")
    monkeypatch.setenv("QEXP_PROGRESS_INTERVAL_SECONDS", "1")
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    assert not (cfg.runtime_root / "process-registrations" / f"{attempt.attempt_id}.json").exists()
    p = projector(cfg)
    p.tick()
    p.close()
    view = inspect_task(cfg, task.task_id)
    assert view["task"]["state"]["projection"] == "succeeded"
    assert view["progress"]["progress"]["current"] == 3
    assert read_logs(cfg, task.task_id) == "application event\n"
    assert not local_progress_path(cfg.runtime_root, attempt.attempt_id).exists()


def test_retry_never_displays_previous_attempt_as_current(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    old, _ = launch(cfg, task, monkeypatch, code=1, payload=report(7))
    retry(cfg, task.task_id)
    pending_view = inspect_task(cfg, task.task_id)
    pending = pending_view["progress"]
    assert pending == {
        "status": "unavailable",
        "observation_state": "pending",
        "reason": "not_started",
    }
    assert json.loads(render(CliOutput(OutputKind.TASK_SHOW, pending_view), "json")) == pending_view
    assert set(pending) == {"status", "observation_state", "reason"}
    new, _ = launch(cfg, task, monkeypatch, payload=report(2, "new-update"))
    assert old.attempt_id != new.attempt_id
    p = projector(cfg)
    p.tick()
    p.close()
    view = inspect_task(cfg, task.task_id)["progress"]
    assert view["attempt_id"] == new.attempt_id
    assert view["progress"]["current"] == 2
    assert not shared_progress_path(cfg.shared_root, task.task_id, old.attempt_id).exists()


def test_retry_resolves_fresh_project_progress_policy(cfg, monkeypatch):
    set_progress_policy(cfg, 60)
    task = submit(cfg, ["echo", "ok"])
    first, _ = launch(cfg, task, monkeypatch, code=1, expected_interval="60")
    first_context = read_json(cfg.runtime_root / "progress-contexts" / f"{first.attempt_id}.json")
    assert first_context["interval_seconds"] == 60

    retry(cfg, task.task_id)
    set_progress_policy(cfg, 90)
    second, _ = launch(cfg, task, monkeypatch, expected_interval="90")
    second_context = read_json(cfg.runtime_root / "progress-contexts" / f"{second.attempt_id}.json")
    assert second_context["interval_seconds"] == 90


def test_inspector_rejects_stale_token(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    p = projector(cfg)
    p.tick()
    p.close()
    path = shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    value = read_json(path)
    value["fencing_token"] += 1
    replace_advisory_snapshot(path, value)
    rejected = inspect_task(cfg, task.task_id)["progress"]
    assert rejected["status"] == "unavailable"
    assert rejected["observation_state"] == "unavailable"
    assert rejected["reason"] == "identity_mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"protocol_version": 1000},
        {"metrics": {"loss": 1.0}},
    ],
)
def test_missing_or_malformed_progress_does_not_fail_task(cfg, monkeypatch, payload):
    task = submit(cfg, ["echo", "ok"])
    launch(cfg, task, monkeypatch, payload=payload)
    p = projector(cfg)
    p.tick()
    p.close()
    view = inspect_task(cfg, task.task_id)
    assert view["task"]["state"]["projection"] == "succeeded"
    assert view["progress"]["status"] == "unavailable"
    assert view["progress"]["observation_state"] == "no_report"
    assert view["progress"]["reason"] == "no_snapshot"


def test_reader_classifies_malformed_shared_snapshot(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    p = projector(cfg)
    p.tick()
    p.close()
    path = shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    replace_advisory_snapshot(path, {"protocol_version": 1000})

    observation = inspect_task(cfg, task.task_id)["progress"]

    assert observation["status"] == "unavailable"
    assert observation["observation_state"] == "unavailable"
    assert observation["reason"] == "invalid_snapshot"


def test_terminal_inspection_is_read_only_and_preserves_observed_timestamp(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    p = projector(cfg)
    p.tick()
    p.close()

    shared = shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    before = (shared.stat().st_ino, shared.stat().st_size, shared.stat().st_mtime_ns)
    first = inspect_task(cfg, task.task_id)["progress"]
    second = inspect_task(cfg, task.task_id)["progress"]
    after = (shared.stat().st_ino, shared.stat().st_size, shared.stat().st_mtime_ns)

    assert first == second
    assert first["observation_state"] == "available"
    assert first["progress"]["current"] == 3
    assert first["reported_at"] == second["reported_at"]
    assert before == after


def test_query_normalizes_optional_payload_fields_and_formatter_keeps_json_canonical(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    launch(
        cfg,
        task,
        monkeypatch,
        payload={"protocol_version": 1, "update_id": "minimal", "stage": "train"},
    )
    p = projector(cfg)
    p.tick()
    p.close()
    view = inspect_task(cfg, task.task_id)
    observation = view["progress"]

    assert set(observation) == {
        "status",
        "observation_state",
        "reason",
        "protocol_version",
        "task_id",
        "attempt_id",
        "attempt_number",
        "machine_name",
        "launch_id",
        "wrapper_pid",
        "wrapper_start_time_ticks",
        "registration_generation",
        "fencing_token",
        "source_update_id",
        "sequence",
        "reported_at",
        "advanced_at",
        "progress",
    }
    assert observation["progress"] == {
        "stage": "train",
        "current": None,
        "total": None,
        "unit": None,
        "message": None,
    }
    assert "Progress status: available" in render(CliOutput(OutputKind.TASK_SHOW, view), "human")

    def fail_if_formatted(*_args, **_kwargs):
        raise AssertionError("JSON rendering invoked human progress formatting")

    monkeypatch.setattr("qqtools.plugins.qexp.formatter.format_progress_details", fail_if_formatted)
    assert json.loads(render(CliOutput(OutputKind.TASK_SHOW, view), "json")) == view


def test_running_query_rejects_registration_replacement_and_keeps_nullable_identity(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    attempt_record = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
    snapshot = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "machine_name": attempt.machine_name,
        "launch_id": attempt_record["authorization"]["launch_id"],
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "registration_generation": "generation-1",
        "fencing_token": attempt.current_fencing_token,
        "source_update_id": "update-1",
        "sequence": 1,
        "reported_at": "2026-09-20T00:00:00Z",
        "advanced_at": "2026-09-20T00:00:00Z",
        "progress": {"stage": "train", "current": 1},
    }
    replace_advisory_snapshot(shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id), snapshot)
    generations = iter(("generation-1", "generation-2"))
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.progress._running_registration_generation",
        lambda *_args: next(generations),
    )

    observation = inspect_task(cfg, task.task_id)["progress"]

    assert observation == {
        "status": "unavailable",
        "observation_state": "unavailable",
        "reason": "identity_mismatch",
    }


def test_terminal_query_does_not_require_current_registration_generation(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    launch(cfg, task, monkeypatch, payload=report())
    p = projector(cfg)
    p.tick()
    p.close()
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.progress._running_registration_generation",
        lambda *_args: pytest.fail("terminal progress queried the current registration"),
    )

    observation = inspect_task(cfg, task.task_id)["progress"]

    assert observation["observation_state"] == "available"


def test_cleanup_removes_progress_sidecars(cfg, monkeypatch):
    task = submit(cfg, ["echo", "ok"])
    attempt, _ = launch(cfg, task, monkeypatch, payload=report())
    p = projector(cfg)
    p.tick()
    p.close()
    clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)
    assert not (cfg.shared_root / "progress" / task.task_id).exists()
    assert not local_progress_path(cfg.runtime_root, attempt.attempt_id).parent.exists()
    for directory in ("progress-contexts", "progress-observed", "progress-diagnostics"):
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
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]

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
    p = projector(cfg)
    p.tick()
    p.close()
    assert inspect_task(cfg, task.task_id)["progress"]["progress"]["current"] == 3
    assert read_logs(cfg, task.task_id).strip() == "done"


def test_unchanged_qpipeline_command_reaches_task_show_progress(cfg, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("qexp process guardian requires Linux")
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    monkeypatch.setenv("PYTHONPATH", source_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "qexp_progress_qpipeline.py"
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
                return self.child.wait(timeout=30)
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
    p = projector(cfg)
    p.tick()
    p.close()
    view = inspect_task(cfg, task.task_id)

    assert view["progress"]["observation_state"] == "available"
    assert view["progress"]["progress"]["stage"] == "train"
    assert "Epoch 1" in view["progress"]["progress"]["message"]


def test_progress_loop_is_separate_and_stop_is_bounded(monkeypatch, tmp_path):
    import threading
    import time
    from types import SimpleNamespace

    from qqtools.plugins.qexp.agent import progress_loop

    entered, release = threading.Event(), threading.Event()
    binding = SimpleNamespace(project_id="p", registration_generation="g", enabled=True)
    runtime_root = tmp_path / "project-runtime"
    mailbox = runtime_root / "progress" / "attempt" / "latest.json"
    mailbox.parent.mkdir(parents=True)
    mailbox.write_text("{}")

    class Runtime:
        def load_registry(self):
            return 1, [binding]

        def project_paths(self, project_id):
            assert project_id == "p"
            return {"root": runtime_root}

        def binding_state(self, selected):
            return "enabled"

        def binding_write_eligible(self, selected, *, renew=False):
            assert selected is binding
            assert renew is False
            return True

    class SlowProjector:
        def __init__(self, cfg, **kwargs):
            assert kwargs["registration_generation"] == "g"

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


def test_idle_progress_loop_does_not_poll_shared_registration(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from qqtools.plugins.qexp.agent import progress_loop

    binding = SimpleNamespace(project_id="p", registration_generation="g", enabled=True)
    runtime_root = tmp_path / "project-runtime"
    runtime_root.mkdir()

    class Runtime:
        def load_registry(self):
            return 1, [binding]

        def project_paths(self, project_id):
            return {"root": runtime_root}

        def binding_state(self, selected):
            return "enabled"

        def binding_write_eligible(self, *args, **kwargs):
            pytest.fail("idle progress loop touched shared registration")

    loop = progress_loop.ProgressObservationLoop(Runtime())
    loop.cycle()
    assert not loop._projectors
