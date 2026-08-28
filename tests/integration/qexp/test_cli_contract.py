import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import AGENT_MODE_DAEMON, init_shared_root, submit
from qqtools.plugins.qexp.agent import get_agent_status
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.layout import load_root_config, runtime_pid_path
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, expire_claim, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

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


def test_task_retry_accepts_blocked_orphan_without_acknowledgement(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active", lambda *_args, **_kwargs: False
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id]) == 0

    assert capsys.readouterr().out.strip() == task.task_id
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_task_retry_accepts_deprecated_duplicate_risk_flag_as_noop(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active", lambda *_args, **_kwargs: False
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id,
                 "--acknowledge-duplicate-risk"]) == 0

    assert capsys.readouterr().out.strip() == task.task_id
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "queued"
    assert "duplicate_risk_attempt_id" not in stored.control


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


def test_init_registers_project_with_machine_agent(tmp_path: Path, capsys):
    root = tmp_path / ".qexp"
    runtime_root = tmp_path / "rt"
    machine_runtime_root = tmp_path / "machine-runtime"
    args = [
        "--shared-root", str(root), "--machine", "gpu-1", "--runtime-root", str(runtime_root),
        "--machine-runtime-root", str(machine_runtime_root), "init",
    ]

    assert main(args) == 0
    assert main(args) == 0

    cfg = load_root_config(root, "gpu-1", runtime_root, require_initialized=True)
    _, bindings = MachineRuntime(machine_runtime_root).load_registry()
    assert len(bindings) == 1
    assert bindings[0].shared_root == cfg.shared_root
    assert bindings[0].machine_name == "gpu-1"


def test_init_rejects_project_legacy_metadata_without_overwriting_it(tmp_path: Path, capsys):
    root = tmp_path / ".qexp"
    runtime_root = tmp_path / "legacy-runtime"
    cfg = init_shared_root(root, "gpu-1", runtime_root=runtime_root)
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["machine"].pop("agent_runtime")
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert main([
        "--shared-root", str(root), "--machine", "gpu-2", "--runtime-root", str(runtime_root), "init",
    ]) == 2

    assert "qexp agent migrate-project" in capsys.readouterr().err
    assert json.loads(record_path.read_text(encoding="utf-8")) == record
    assert not (cfg.shared_root / "machines" / "gpu-2").exists()


def test_init_allows_project_local_machine_names_in_one_runtime(tmp_path: Path):
    machine_runtime_root = tmp_path / "machine-runtime"
    for project, machine_name in (("first", "a"), ("second", "b")):
        assert main([
            "--shared-root", str(tmp_path / project / ".qexp"), "--machine", machine_name,
            "--machine-runtime-root", str(machine_runtime_root), "init",
        ]) == 0

    _, bindings = MachineRuntime(machine_runtime_root).load_registry()
    assert {(binding.shared_root.parent.name, binding.machine_name) for binding in bindings} == {
        ("first", "a"), ("second", "b"),
    }


def test_init_reports_recoverable_registration_failure(tmp_path: Path, monkeypatch, capsys):
    saved_contexts = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.register_project",
        lambda *_args: (_ for _ in ()).throw(OSError("machine runtime unavailable")),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli._try_save_context", lambda *args: saved_contexts.append(args),
    )

    assert main([
        "--shared-root", str(tmp_path / ".qexp"), "--machine", "gpu-1", "init",
    ]) == 2

    assert saved_contexts == []
    assert "initialized but was not registered" in capsys.readouterr().err


def test_use_still_fails_when_context_save_fails(tmp_path: Path, monkeypatch):
    def fail_save_context(*args, **kwargs):
        raise OSError(30, "Read-only file system", "/readonly/.qqtools/qexp-context.json")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.save_context", fail_save_context)

    with pytest.raises(OSError, match="Read-only file system"):
        main([
            "use", "--shared-root", str(tmp_path / ".qexp"), "--machine", "gpu-1", "--runtime-root",
            str(tmp_path / "rt")
        ])


def test_agent_start_starts_the_registered_global_agent(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(
        tmp_path / ".qexp",
        "gpu-1",
        agent_mode=AGENT_MODE_DAEMON,
        runtime_root=tmp_path / "rt",
    )

    from qqtools.plugins.qexp.machine_runtime import MachineRuntime

    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.start_local_agent",
        lambda *_args, **_kwargs: ("started", {"agent_state": "active", "pid": 4321}),
    )

    assert main([*_base_args(cfg), "--machine-runtime-root", str(runtime.root), "agent", "start", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "started"


def test_submit_requires_explicit_project_registration(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("not-a-pid", encoding="utf-8")

    assert main([*_base_args(cfg), "submit", "--", "echo", "ok"]) == 2
    assert "qexp agent add-project" in capsys.readouterr().err


def test_agent_start_rejects_removed_background_flag(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "agent", "start", "--background"])


def test_agent_run_reports_foreground_start(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    from qqtools.plugins.qexp.machine_runtime import MachineRuntime

    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    received = []

    def run_foreground(cfg, *, reason, on_started, machine_runtime=None):
        assert machine_runtime.root == runtime.root
        received.append(reason)
        on_started({"machine_name": cfg.machine_name, "agent_state": "active", "pid": 123, "is_running": True})

    monkeypatch.setattr("qqtools.plugins.qexp.cli.run_local_agent_foreground", run_foreground)

    assert main([*_base_args(cfg), "--machine-runtime-root", str(runtime.root), "agent", "run", "--format=json"]) == 0
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


def test_group_retry_failed_skips_blocked_orphans_and_requests_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    failed_task = submit(cfg, ["echo", "failed"], group="demo")
    failed_attempt = claim_task(cfg, failed_task.task_id, [0])
    assert failed_attempt is not None
    assert fail_attempt(
        cfg, failed_task.task_id, failed_attempt.attempt_id, failed_attempt.current_fencing_token, "test_failure"
    )
    blocked_task = submit(cfg, ["echo", "blocked"], group="demo")
    orphaned_attempt = claim_task(cfg, blocked_task.task_id, [0])
    assert orphaned_attempt is not None
    assert authorize_launch(
        cfg, blocked_task.task_id, orphaned_attempt.attempt_id, orphaned_attempt.current_fencing_token
    )
    assert expire_claim(
        cfg, blocked_task.task_id, orphaned_attempt.attempt_id, orphaned_attempt.current_fencing_token
    )
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "group", "retry-failed", "demo"]) == 0
    assert reasons == ["group-retry-failed"]
    assert load_task(cfg, failed_task.task_id).state["projection"] == "queued"
    assert load_task(cfg, blocked_task.task_id).state["projection"] == "blocked"


def test_agent_stop_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    runtime_root = tmp_path / "machine-runtime"
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.stop_machine_agent", lambda _runtime: False,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.get_machine_agent_status",
        lambda _runtime: {"agent_state": "stopped", "pid": None, "is_running": False, "projects": []},
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "stop", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already_stopped"


def test_agent_restart_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    runtime_root = tmp_path / "machine-runtime"
    class FakeProcess:
        pid = 987

    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.restart_machine_agent", lambda _runtime: FakeProcess(),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.get_machine_agent_status",
        lambda _runtime: {"agent_state": "active", "pid": 432, "is_running": True, "projects": []},
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "restart", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["pid"] == 987


def test_agent_add_project_is_explicit_and_machine_agent_prefix_is_not_public(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime_root = tmp_path / "machine-runtime"
    args = [
        "--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name,
        "--runtime-root", str(cfg.runtime_root), "--machine-runtime-root", str(runtime_root),
    ]

    assert main([*args, "agent", "add-project", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "project_added"
    assert main([*args, "agent", "add-project", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "project_already_registered"
    with pytest.raises(SystemExit):
        main(["machine-agent", "status"])


def test_machine_agent_status_branches_before_project_configuration(tmp_path: Path, capsys) -> None:
    runtime_root = tmp_path / "machine-runtime"

    assert main([
        "--machine-runtime-root", str(runtime_root), "agent", "status", "--format=json"
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["machine_runtime_root"] == str(runtime_root.resolve())
    assert output["projects"] == []


def test_agent_project_registry_commands(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime_root = tmp_path / "machine-runtime"
    base = [*_base_args(cfg), "--machine-runtime-root", str(runtime_root), "agent"]

    assert main([*base, "add-project", "--format=json"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["project_id"]

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "list-projects", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["projects"][0]["state"] == "enabled"

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "disable-project", added["project_id"], "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "remove-project", added["project_id"], "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "project_removed"


def test_agent_project_add_can_register_while_scheduler_is_running(
        tmp_path: Path, capsys) -> None:
    from qqtools.plugins.qexp.machine_runtime import MachineRuntime

    cfg = init_shared_root(
        tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime_root = tmp_path / "machine-runtime"
    runtime = MachineRuntime(runtime_root)

    with runtime.scheduler_authority(blocking=True):
        result = main([
            *_base_args(cfg), "--machine-runtime-root", str(runtime_root), "agent", "add-project",
        ])

    assert result == 0
    assert runtime.load_registry()[1]


def test_read_only_task_list_does_not_initialize_machine_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from qqtools.plugins.qexp.machine_runtime import MACHINE_RUNTIME_ENV

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    machine_runtime_root = tmp_path / "unused-machine-runtime"
    monkeypatch.setenv(MACHINE_RUNTIME_ENV, str(machine_runtime_root))

    assert main([*_base_args(cfg), "task", "list", "--format=json"]) == 0

    assert json.loads(capsys.readouterr().out) == []
    assert not machine_runtime_root.exists()


def test_legacy_project_requires_explicit_migration(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime_root = tmp_path / "machine-runtime"
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["machine"].pop("agent_runtime")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    base = [*_base_args(cfg), "--machine-runtime-root", str(runtime_root), "agent"]

    assert main([*base, "start"]) == 2
    assert "qexp agent migrate-project" in capsys.readouterr().err


def test_global_agent_status_and_stop_do_not_require_project_context(
        tmp_path: Path, monkeypatch, capsys) -> None:
    runtime_root = tmp_path / "machine-runtime"
    machine_status = {"agent_state": "active", "is_running": True, "pid": 4321, "projects": []}
    monkeypatch.setattr("qqtools.plugins.qexp.cli.get_machine_agent_status", lambda _runtime: machine_status)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.stop_machine_agent", lambda _runtime: True)

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "status", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["agent_state"] == "active"

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "stop", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "stopped"


def test_explicit_machine_runtime_root_submits_through_global_activation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from qqtools.plugins.qexp.machine_runtime import MachineRuntime

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda _cfg, *, reason, machine_runtime: machine_runtime.root == runtime.root and reason == "submit",
    )

    assert main([
        "--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name,
        "--runtime-root", str(cfg.runtime_root),
        "--machine-runtime-root", str(runtime.root), "submit", "--", "echo", "ok",
    ]) == 0
    assert capsys.readouterr().out.strip()


def test_managed_doctor_reads_project_local_process_evidence(
        tmp_path: Path, capsys) -> None:
    from qqtools.plugins.qexp.machine_runtime import MachineRuntime
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    cfg = init_shared_root(
        tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    process_path = runtime.project_paths(binding.project_id)["processes"] / "missing.json"
    atomic_replace(process_path, {"process": {"task_id": "missing-task"}})

    result = main([
        *_base_args(cfg),
        "--machine-runtime-root", str(runtime.root),
        "doctor", "verify", "--format=json",
    ])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert any(
        issue["code"] == "process_manifest_task_missing" for issue in output["issues"]
    )
