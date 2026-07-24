import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.layout import load_root_config, runtime_pid_path
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (authorize_launch, claim_task, expire_claim,
                                             fail_attempt)


def _base_args(cfg) -> list[str]:
    return ["--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name,
            "--runtime-root", str(cfg.runtime_root)]


def test_task_cancel_reports_pending_acknowledgement(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert main([*_base_args(cfg), "task", "cancel", task.task_id]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["owning_machine"] == "gpu-1"
    assert output["operation_state"] == "waiting_ack"
    assert output["pending_acknowledgement"] is True


def test_prelaunch_cancel_reports_completed_without_pending_acknowledgement(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert main([*_base_args(cfg), "task", "cancel", task.task_id]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["task_state"] == "cancelled"
    assert output["operation_state"] == "completed"
    assert output["pending_acknowledgement"] is False


def test_task_retry_accepts_explicit_duplicate_risk_acknowledgement(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert main([*_base_args(cfg), "task", "retry", task.task_id,
                 "--acknowledge-duplicate-risk"]) == 0
    assert capsys.readouterr().out.strip() == task.task_id
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_clean_cli_reports_dry_run_candidates(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure"
    )
    assert main([*_base_args(cfg), "clean", "--task-id", task.task_id, "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["candidates"] == [task.task_id]
    assert output["removed"] == []


def test_init_succeeds_when_context_save_fails(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / ".qexp"
    runtime_root = tmp_path / "rt"

    def fail_save_context(*args, **kwargs):
        raise OSError(30, "Read-only file system", "/readonly/.qqtools/qexp-context.json")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.save_context", fail_save_context)

    assert main([
        "--shared-root", str(root), "--machine", "gpu-1", "--runtime-root", str(runtime_root), "init"
    ]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == str(root.resolve())
    assert "initialized successfully, but failed to save CLI context" in captured.err
    cfg = load_root_config(root, "gpu-1", runtime_root, require_initialized=True)
    assert cfg.shared_root == root.resolve()


def test_use_still_fails_when_context_save_fails(tmp_path: Path, monkeypatch):
    def fail_save_context(*args, **kwargs):
        raise OSError(30, "Read-only file system", "/readonly/.qqtools/qexp-context.json")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.save_context", fail_save_context)

    with pytest.raises(OSError, match="Read-only file system"):
        main([
            "use", "--shared-root", str(tmp_path / ".qexp"), "--machine", "gpu-1", "--runtime-root",
            str(tmp_path / "rt")
        ])


def test_background_agent_start_records_parent_visible_pid(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr("qqtools.plugins.qexp.cli.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    assert main([*_base_args(cfg), "agent", "start", "--persistent", "--background"]) == 0
    assert runtime_pid_path(cfg).read_text(encoding="utf-8").strip() == "4321"


def test_agent_stop_clears_stale_pid_file(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.cli.os.kill",
                        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))

    assert main([*_base_args(cfg), "agent", "stop"]) == 0
    assert not pid_path.exists()


def test_agent_stop_clears_pid_file_after_process_exits(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.cli.os.kill", lambda pid, sig: None)
    probe = {"count": 0}

    def fake_exists(path: str) -> bool:
        if path != "/proc/4321":
            return True
        probe["count"] += 1
        return probe["count"] == 1

    monkeypatch.setattr("qqtools.plugins.qexp.cli.os.path.exists", fake_exists)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.time.sleep", lambda _: None)

    assert main([*_base_args(cfg), "agent", "stop"]) == 0
    assert not pid_path.exists()
