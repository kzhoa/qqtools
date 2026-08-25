import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import AGENT_MODE_DAEMON, init_shared_root, submit
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.agent import get_agent_status
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
    assert main([*_base_args(cfg), "task", "cancel", task.task_id, "--format=json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["owning_machine"] == "gpu-1"
    assert output["operation_state"] == "waiting_ack"
    assert output["pending_acknowledgement"] is True


def test_prelaunch_cancel_reports_completed_without_pending_acknowledgement(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert main([*_base_args(cfg), "task", "cancel", task.task_id, "--format=json"]) == 0
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
    assert main([*_base_args(cfg), "clean", "--task-id", task.task_id, "--dry-run", "--format=json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["candidates"] == [task.task_id]
    assert output["removed"] == []


def test_clean_help_documents_group_scope_and_work_directory_boundary(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["clean", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--group GROUP" in output
    assert "preserving experiment work directories" in output


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


def test_agent_start_records_parent_visible_pid(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(
        tmp_path / ".qexp",
        "gpu-1",
        agent_mode=AGENT_MODE_DAEMON,
        runtime_root=tmp_path / "rt",
    )

    class FakeProcess:
        pid = 4321

    def spawn(cfg):
        pid_path = runtime_pid_path(cfg)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(FakeProcess.pid), encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.activation.spawn_agent_process", spawn)

    assert main([*_base_args(cfg), "agent", "start", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "started"
    assert runtime_pid_path(cfg).read_text(encoding="utf-8").strip() == "4321"


def test_submit_tolerates_a_malformed_runtime_pid_file(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("not-a-pid", encoding="utf-8")

    assert get_agent_status(cfg)["is_running"] is False
    assert main([*_base_args(cfg), "submit", "--", "echo", "ok"]) == 0
    assert capsys.readouterr().out.strip()


def test_agent_start_rejects_removed_background_flag(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "agent", "start", "--background"])


def test_agent_run_reports_foreground_start(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    received = []

    def run_foreground(cfg, *, reason, on_started):
        received.append(reason)
        on_started({"machine_name": cfg.machine_name, "agent_state": "active", "pid": 123, "is_running": True})

    monkeypatch.setattr("qqtools.plugins.qexp.cli.run_local_agent_foreground", run_foreground)

    assert main([*_base_args(cfg), "agent", "run", "--format=json"]) == 0
    assert received == ["manual_run"]
    assert json.loads(capsys.readouterr().out)["action"] == "running"


def test_agent_start_rejects_legacy_persistent_flag(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "agent", "start", "--persistent"])


def test_submit_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "submit", "--", "echo", "ok"]) == 0
    assert reasons == ["submit"]


def test_submit_without_activation_persists_task_and_skips_local_agent(
        tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "submit", "--no-activate", "--", "echo", "ok"]) == 0
    task_id = capsys.readouterr().out.strip()

    assert reasons == []
    assert load_task(cfg, task_id).state["projection"] == "queued"


def test_submit_without_activation_does_not_start_local_agent(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert get_agent_status(cfg)["is_running"] is False
    assert main([*_base_args(cfg), "submit", "--no-activate", "--", "echo", "ok"]) == 0
    assert get_agent_status(cfg)["is_running"] is False
    assert not runtime_pid_path(cfg).exists()


def test_batch_submit_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: ['echo', 'ok']\n", encoding="utf-8")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "batch-submit", "--file", str(manifest), "--group", "demo"]) == 0
    assert reasons == ["batch-submit"]


def test_retry_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(
        tmp_path / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "rt",
    )
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id, "--acknowledge-duplicate-risk"]) == 0
    assert reasons == ["task-retry"]


def test_offer_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="demo", sharing_mode="spillover")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "task", "offer", task.task_id]) == 0
    assert reasons == ["task-offer"]


def test_group_resume_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    main([*_base_args(cfg), "group", "create", "demo"])
    main([*_base_args(cfg), "group", "pause", "demo"])
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "group", "resume", "demo"]) == 0
    assert reasons == ["group-resume"]


def test_group_retry_failed_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="demo")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure"
    )
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "group", "retry-failed", "demo"]) == 0
    assert reasons == ["group-retry-failed"]


def test_agent_stop_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.stop_local_agent",
        lambda cfg: ("already_stopped", {"machine_name": cfg.machine_name, "agent_state": "stopped", "pid": None, "is_running": False}),
    )

    assert main([*_base_args(cfg), "agent", "stop", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already_stopped"


def test_agent_restart_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.restart_local_agent",
        lambda cfg: ("restarted", {"machine_name": cfg.machine_name, "agent_state": "active", "pid": 987, "is_running": True, "previous_pid": 432}),
    )

    assert main([*_base_args(cfg), "agent", "restart", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["previous_pid"] == 432
