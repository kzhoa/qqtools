import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, load_root_config, read_logs, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.runner import _publish_launch_intent, run_attempt
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.reservations import active_reservations
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, cancel_task, claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

@pytest.fixture(autouse=True)
def _disable_process_group_guardian(monkeypatch):
    monkeypatch.setattr("qqtools.plugins.qexp.runner._configure_process_group_guardian", lambda: None)


def _launch_id(cfg, attempt):
    return read_json(attempt_path(cfg.shared_root, attempt.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]


class FakeChild:
    pid = 4321
    returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def test_runner_publishes_registration_and_exit_observation_only(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks", lambda pid: pid + 100)
    launched: dict[str, object] = {}

    def popen_factory(*args, **kwargs):
        launched["command"] = args[0]
        launched.update(kwargs)
        return FakeChild()

    result = run_attempt(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        _launch_id(cfg, attempt),
        popen_factory=popen_factory,
    )
    assert result == 0
    assert launched["start_new_session"] is True
    assert launched["command"][-3:] == ["--", "echo", "ok"]
    assert "--parent-pid" in launched["command"]
    registration_path = cfg.runtime_root / "process-registrations" / f"{attempt.attempt_id}.json"
    registration = json.loads(registration_path.read_text())["process_registration"]
    assert registration["wrapper_start_time_ticks"] is not None
    assert registration["process_group_start_time_ticks"] == FakeChild.pid + 100
    observation_path = cfg.runtime_root / "process-observations" / f"{attempt.attempt_id}.json"
    assert json.loads(observation_path.read_text())["exit_observation"]["observed_exit_code"] == 0
    assert not (cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json").exists()


def test_read_logs_uses_attempt_log_written_by_runner(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks", lambda pid: pid + 100)

    def popen_factory(*args, **kwargs):
        kwargs["stdout"].write(b"attempt output\n")
        kwargs["stdout"].flush()
        return FakeChild()

    assert (
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            _launch_id(cfg, attempt),
            popen_factory=popen_factory,
        )
        == 0
    )

    remote_cfg = load_root_config(cfg.shared_root, "gpu-2", tmp_path / "remote-runtime", require_initialized=True)
    log_path = cfg.shared_root / "logs" / task.task_id / f"{attempt.attempt_id}.log"
    assert log_path.exists()
    assert read_logs(cfg, task.task_id) == "attempt output\n"
    assert read_logs(remote_cfg, task.task_id) == "attempt output\n"
    assert not (cfg.runtime_root / "logs").exists()

    assert not (cfg.shared_root / "claims" / "archive" / task.task_id / "1.json").exists()
    AuthoritySupervisor(cfg).tick()
    archive_path = cfg.shared_root / "claims" / "archive" / task.task_id / "1.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))["claim_archive"]
    assert archive["reason"] == "completed"
    assert archive["claim"]["attempt_id"] == attempt.attempt_id
    local = local_paths(cfg.runtime_root)
    assert not (local["processes"] / f"{attempt.attempt_id}.json").exists()
    assert not (local["registrations"] / f"{attempt.attempt_id}.json").exists()
    assert not (local["observations"] / f"{attempt.attempt_id}.json").exists()
    assert not (local["launch_intents"] / f"{attempt.attempt_id}.json").exists()

    orphan_observation = local["observations"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        orphan_observation,
        {"exit_observation": {"attempt_id": attempt.attempt_id, "observed_exit_code": 0}},
    )
    AuthoritySupervisor(cfg).tick()
    assert not orphan_observation.exists()


def test_read_logs_prefers_persisted_attempt_log_reference(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    referenced_log = tmp_path / "shared-visible.log"
    referenced_log.write_text("persisted reference\n", encoding="utf-8")
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    data = read_json(path)
    data["attempt"]["process"]["log_references"] = [str(referenced_log)]
    atomic_replace(path, data)

    assert read_logs(cfg, task.task_id) == "persisted reference\n"


def test_observed_success_after_cancel_remains_succeeded(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks", lambda pid: pid + 100)
    assert (
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            _launch_id(cfg, attempt),
            popen_factory=lambda *args, **kwargs: FakeChild(),
        )
        == 0
    )
    cancel_task(cfg, task.task_id, terminate_running=True)

    AuthoritySupervisor(cfg).tick()

    stored_task = read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")["task"]
    stored_attempt = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
    assert stored_task["state"] == {"projection": "succeeded", "reason": "completed"}
    assert stored_attempt["result"]["exit_code"] == 0
    assert stored_attempt["result"]["reason"] == "completed"
    assert stored_task["control"]["termination_result"] == "already_exited"
    assert stored_task["control"]["termination_acknowledged_at"] is not None
    assert stored_attempt["termination"]["result"] == "already_exited"


def test_missing_registration_after_dead_wrapper_is_isolated_without_releasing_gpu(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    _publish_launch_intent(cfg, attempt, task)
    monkeypatch.setattr(AuthoritySupervisor, "_wrapper_matches", staticmethod(lambda _intent: False))

    AuthoritySupervisor(cfg).tick()

    manifest = read_json(local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json")["process"]
    diagnostic = read_json(local_paths(cfg.runtime_root)["authority_diagnostics"] / f"{attempt.attempt_id}.json")[
        "authority_diagnostic"
    ]
    assert manifest["observed_state"] == "launch_unverifiable"
    assert manifest["authority_state"] == "isolated"
    assert diagnostic["reason"] == "launch_unverifiable"
    assert active_reservations(cfg.runtime_root)[0]["attempt_id"] == attempt.attempt_id


def test_unavailable_shared_policy_isolated_by_cached_deadline(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    expiry = (datetime.now(timezone.utc) - timedelta(seconds=1)).replace(microsecond=0)
    atomic_replace(
        local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json",
        {
            "process": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "lease_expires_at": expiry.isoformat().replace("+00:00", "Z"),
            }
        },
    )
    supervisor = AuthoritySupervisor(cfg)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.authority.load_lease_policy",
        lambda _cfg: (_ for _ in ()).throw(OSError("shared root down")),
    )

    supervisor.tick()

    manifest = read_json(local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json")["process"]
    diagnostic = read_json(local_paths(cfg.runtime_root)["authority_diagnostics"] / f"{attempt.attempt_id}.json")[
        "authority_diagnostic"
    ]
    assert manifest["authority_state"] == "suspect"
    assert diagnostic["reason"] == "lease_policy_unavailable"


def test_runner_parent_death_kills_training_process_group_descendants(tmp_path: Path):
    pid_file = tmp_path / "pids.json"
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH")]))
    child_code = (
        "import json, os, subprocess, sys, time; "
        "descendant = subprocess.Popen(['sleep', '60']); "
        "json.dump({'child': os.getpid(), 'descendant': descendant.pid}, open(sys.argv[1], 'w')); "
        "time.sleep(60)"
    )
    guardian_code = (
        "import os, subprocess, sys; "
        "from qqtools.plugins.qexp.runner import _configure_process_group_guardian; "
        "os.setsid(); "
        "_configure_process_group_guardian(os.getppid()); "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "child.wait()"
    )
    launcher_code = """
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]])
time.sleep(60)
"""
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            launcher_code,
            guardian_code,
            child_code,
            str(pid_file),
        ],
        env=environment,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        pids = json.loads(pid_file.read_text(encoding="utf-8"))

        os.kill(launcher.pid, signal.SIGKILL)
        launcher.wait(timeout=5)

        deadline = time.monotonic() + 5
        while any(_is_process_running(pid) for pid in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(_is_process_running(pid) for pid in pids.values())
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=5)


def _is_process_running(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False
