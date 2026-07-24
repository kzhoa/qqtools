import json
import os
from pathlib import Path

from qqtools.plugins.qexp import init_shared_root, read_logs, submit
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runner import run_attempt
from qqtools.plugins.qexp.scheduler import claim_task


class FakeChild:
    pid = 4321
    returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def test_runner_publishes_manifest_terminal_truth_and_releases_gpu(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks",
                        lambda pid: pid + 100)
    result = run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                         popen_factory=lambda *args, **kwargs: FakeChild())
    assert result == 0
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    manifest = json.loads(manifest_path.read_text())["process"]
    assert manifest["wrapper_start_time_ticks"] == os.getpid() + 100
    assert manifest["process_group_start_time_ticks"] == FakeChild.pid + 100


def test_read_logs_uses_attempt_log_written_by_runner(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr("qqtools.plugins.qexp.runner._process_start_time_ticks",
                        lambda pid: pid + 100)

    def popen_factory(*args, **kwargs):
        kwargs["stdout"].write(b"attempt output\n")
        kwargs["stdout"].flush()
        return FakeChild()

    assert run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                       popen_factory=popen_factory) == 0

    assert read_logs(cfg, task.task_id) == "attempt output\n"
    assert not (cfg.runtime_root / "logs" / f"{task.task_id}-current.log").exists()


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


class HangingChild:
    pid = 5432
    returncode = None

    def __init__(self):
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.returncode = -15
        return self.returncode


def test_runner_kills_child_when_lease_renewal_raises(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    child = HangingChild()

    def raise_renewal(*args, **kwargs):
        raise OSError("shared filesystem unavailable")

    monkeypatch.setattr("qqtools.plugins.qexp.runner.renew_attempt_lease", raise_renewal)
    monkeypatch.setattr("qqtools.plugins.qexp.runner.os.killpg", lambda *args: None)
    assert run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                       popen_factory=lambda *args, **kwargs: child, poll_interval=0) == -15
    assert child.wait_calls == 1
