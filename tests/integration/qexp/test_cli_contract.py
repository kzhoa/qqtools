import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import MachineAgentStartError, MachineAgentStopError
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.layout import load_context, runtime_pid_path
from qqtools.plugins.qexp.legacy_agent import get_agent_status
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, expire_claim, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _base_args(cfg) -> list[str]:
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


def test_explicit_cli_validation_errors_are_structured(tmp_path: Path, capsys) -> None:
    assert main(["init", "--format=json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"

    assert (
        main(
            [
                "--machine-runtime-root",
                str(tmp_path / "machine-runtime"),
                "agent",
                "start",
                "--timeout",
                "0",
                "--format=json",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    assert main([*_base_args(cfg), "task", "retry", task.task_id, "--quiet", "--format=json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"


def test_task_cancel_reports_pending_acknowledgement(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert main([*_base_args(cfg), "task", "cancel", task.task_id, "--format=json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["owning_machine"] == "gpu-1"
    assert output["outcome"] == "waiting_ack"
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


def test_blocked_orphan_cancel_reports_blocked_recovery_state(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    assert main([*_base_args(cfg), "task", "cancel", task.task_id, "--format=json"]) == 1

    output = json.loads(capsys.readouterr().out)
    assert output["outcome"] == "blocked"
    assert output["operation_state"] == "blocked"
    assert output["reason"]
    assert output["follow_up_command"]


def test_task_retry_accepts_blocked_orphan_without_acknowledgement(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active", lambda *_args, **_kwargs: False
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id]) == 0

    output = capsys.readouterr().out
    assert task.task_id in output
    assert "accepted" in output.lower()
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_task_retry_quiet_preserves_task_id_only_output(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active", lambda *_args, **_kwargs: False
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id, "--quiet"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{task.task_id}\n"
    assert captured.err == ""


def test_task_retry_rejects_retired_duplicate_risk_flag_without_mutation(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active", lambda *_args, **_kwargs: False
    )

    with pytest.raises(SystemExit) as exc_info:
        main([*_base_args(cfg), "task", "retry", task.task_id, "--acknowledge-duplicate-risk"])

    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "blocked"
    assert "duplicate_risk_attempt_id" not in stored.control


def test_clean_cli_reports_dry_run_candidates(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    assert main([*_base_args(cfg), "admin", "clean", "--task-id", task.task_id, "--dry-run", "--format=json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["candidates"] == [task.task_id]
    assert output["removed"] == []
    assert output["outcome"] == "preview"
    assert output["deletion"] == "none"
    assert output["deletion_performed"] is False
    assert output["candidate_count"] == 1
    assert output["removed_count"] == 0


def test_clean_help_documents_group_scope_and_work_directory_boundary(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["admin", "clean", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--group GROUP" in output
    assert "preserving experiment work directories" in output


def test_agent_migration_reports_partial_success_when_start_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    _revision, bindings = runtime.load_registry()
    binding = bindings[0]

    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.migrate_project", lambda *_args: binding)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.project_migration_state",
        lambda *_args: {"state": "active", "updated_at": "2026-09-22T10:00:00Z"},
    )

    def fail_start(*_args, **_kwargs):
        raise MachineAgentStartError("test start failure")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.ensure_machine_agent_started", fail_start)

    assert main([*base, "admin", "migrate", "agent", "--format=json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "partial"
    assert result["migration_state"] == "active"
    assert result["error"]["code"] == "agent_start_failed"
    assert result["follow_up_command"] == "qexp agent start"


def test_schema_migration_reports_already_current_without_synthetic_completion(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert main([*_base_args(cfg), "admin", "migrate", "schema", "--to-schema", "6", "--format=json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "already_current"
    assert result["outcome"] == "no_change"
    assert result["source_schema"] == 6
    assert result["target_schema"] == 6
    assert result["destructive_boundary_reached"] is False


def test_committed_schema_migration_retry_reports_no_change(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    journal_path = cfg.shared_root.parent / f".{cfg.shared_root.name}.schema6-migration.json"
    atomic_replace(
        journal_path,
        {
            "migration": {
                "from_schema": 5,
                "to_schema": 6,
                "source_root": str(cfg.shared_root),
                "stage_root": str(cfg.shared_root.parent / ".qexp.schema6-stage-token" / cfg.shared_root.name),
                "backup_root": str(cfg.shared_root.parent / ".qexp.schema5-backup-token"),
                "phase": "committed",
            }
        },
    )

    assert main([*_base_args(cfg), "admin", "migrate", "schema", "--to-schema", "6", "--format=json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "already_current"
    assert result["outcome"] == "no_change"
    assert result["migration_journal_phase"] == "committed"
    assert result["destructive_boundary_reached"] is False


def test_help_explains_existing_project_machine_join_and_context_only_use(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    top_level_output = " ".join(capsys.readouterr().out.split())
    assert "qexp init --machine NAME" in top_level_output
    assert "qexp project register PATH" in top_level_output
    assert "qexp use only saves local CLI" in top_level_output

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "--help"])

    assert exc_info.value.code == 0
    init_output = " ".join(capsys.readouterr().out.split())
    assert "local machine identity and global agent policy" in init_output
    assert "qexp project register" in init_output

    with pytest.raises(SystemExit) as exc_info:
        main(["use", "--help"])

    assert exc_info.value.code == 0
    use_output = " ".join(capsys.readouterr().out.split())
    assert "does not initialize a shared root" in use_output
    assert "or register the project with the local machine agent" in use_output


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_use_context_save_failure_is_bounded_and_format_independent(
    tmp_path: Path, monkeypatch, capsys, output_format: str
) -> None:
    def fail_save_context(*args, **kwargs):
        raise OSError(30, "Read-only file system", "/readonly/.qqtools/qexp-context.json")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.save_context", fail_save_context)

    assert main(["use", "--project", str(tmp_path / ".qexp"), "--format", output_format]) == 1

    captured = capsys.readouterr()
    if output_format == "json":
        error = json.loads(captured.out)["error"]
        assert error["code"] == "context_write_failed"
        assert "Read-only file system" in error["message"]
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert "Read-only file system" in captured.err


def test_use_saves_only_a_stable_shared_root_and_show_has_a_fixed_contract(tmp_path: Path, monkeypatch, capsys) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    project = tmp_path / "project" / ".qexp"
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.chdir(tmp_path)

    assert main(["use", "--project", "project/.qexp"]) == 0
    assert str(project) in capsys.readouterr().out
    assert json.loads(context_path.read_text(encoding="utf-8")) == {"shared_root": str(project)}
    monkeypatch.chdir(other_directory)
    assert main(["use", "--show", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"shared_root": str(project)}
    assert load_context() == {"shared_root": str(project)}


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_unexpected_handler_runtime_error_is_not_downgraded(tmp_path: Path, monkeypatch, output_format: str) -> None:
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.get_machine_agent_status",
        lambda _runtime: (_ for _ in ()).throw(RuntimeError("programming defect")),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        main(
            [
                "--machine-runtime-root",
                str(tmp_path / "machine-runtime"),
                "agent",
                "status",
                "--format",
                output_format,
            ]
        )


def test_use_rejects_removed_global_identity_flag_without_writing_context(tmp_path: Path, monkeypatch, capsys) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    root = tmp_path / "project" / ".qexp"

    assert main(["--machine", "gpu-1", "use", "--project", str(root)]) == 2
    assert "accepts only" in capsys.readouterr().err
    assert not context_path.exists()


def test_use_rejects_removed_subcommand_identity_flags_without_writing_context(tmp_path: Path, monkeypatch) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    root = tmp_path / "project" / ".qexp"

    assert main(["use", "--project", str(root), "--machine", "gpu-1"]) == 2
    assert not context_path.exists()


def test_use_mode_conflicts_and_malformed_context_fail_without_rewriting(tmp_path: Path, monkeypatch, capsys) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    context_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        main(["use", "--show"])
    assert main(["use", "--project", str(tmp_path / ".qexp"), "--show"]) == 2
    assert "exactly one" in capsys.readouterr().err
    assert main(["use", "--clear", "--machine", "gpu-1"]) == 2
    assert "accepts only" in capsys.readouterr().err
    assert context_path.read_text(encoding="utf-8") == "[]"


def test_use_clear_is_idempotent_and_empty_show_is_explicit(tmp_path: Path, monkeypatch, capsys) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)

    assert main(["use", "--clear"]) == 0
    assert "clear" in capsys.readouterr().out.lower()
    assert main(["use", "--show"]) == 0
    assert capsys.readouterr().out == "shared_root: <not set>\n"


def test_unbound_context_allows_reads_but_mutations_explain_join_and_recovery(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main(["use", "--project", str(cfg.shared_root)]) == 0

    assert main(["task", "list", "--format=json"]) == 0
    assert main(["submit", "--no-activate", "--", "echo", "blocked"]) == 2
    error = capsys.readouterr().err
    assert "qexp project register" in error
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


def test_submit_requires_explicit_project_registration(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("not-a-pid", encoding="utf-8")

    args = [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(tmp_path / "machine-runtime"),
        "submit",
        "--",
        "echo",
        "ok",
    ]
    assert main(args) == 2
    assert "qexp project register" in capsys.readouterr().err


def test_agent_start_rejects_removed_background_flag(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "agent", "start", "--background"])


def test_agent_run_reports_foreground_start(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    from qqtools.plugins.qexp.agent.context import MachineRuntime

    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    received = []

    def run_foreground(machine_runtime):
        assert machine_runtime.root == runtime.root
        received.append("manual_run")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle.run_machine_agent_loop", run_foreground)

    assert main([*_base_args(cfg), "--machine-runtime-root", str(runtime.root), "agent", "run"]) == 0
    assert received == ["manual_run"]
    assert capsys.readouterr().out == ""


def test_agent_run_already_running_is_operational(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle.run_machine_agent_loop",
        lambda _runtime: (_ for _ in ()).throw(MachineAgentStartError("already running")),
    )

    assert main([*_base_args(cfg), "--machine-runtime-root", str(runtime.root), "agent", "run"]) == 1
    assert "already running" in capsys.readouterr().err


def test_agent_run_has_no_finite_readiness_record(tmp_path: Path, monkeypatch, capsys):
    runtime_root = tmp_path / "machine-runtime"
    runtime = MachineRuntime(runtime_root)
    initialize_machine(runtime, "gpu-1")
    entered: list[Path] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle.run_machine_agent_loop",
        lambda selected: entered.append(selected.root),
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "run"]) == 0
    assert entered == [runtime_root]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["agent", "run", "--format=json"],
        ["--format=json", "agent", "run"],
    ],
)
def test_agent_run_rejects_format_before_runtime_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.capture_readiness_snapshot",
        lambda _runtime: pytest.fail("format rejection performed a readiness read"),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle.run_machine_agent_loop",
        lambda _runtime: pytest.fail("format rejection entered the agent loop"),
    )

    assert main(["--machine-runtime-root", str(tmp_path / "machine-runtime"), *argv]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["task", "logs", "task-1", "--format=json"],
        ["--format=human", "task", "logs", "task-1"],
        ["task", "show", "task-1", "--watch", "--format=human"],
        ["--format=json", "task", "show", "task-1", "--watch"],
    ],
)
def test_raw_and_continuous_task_commands_reject_format_before_project_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint._resolve_cfg",
        lambda *_args, **_kwargs: pytest.fail("format rejection resolved Project state"),
    )

    assert main(["--machine-runtime-root", str(tmp_path / "machine-runtime"), *argv]) == 2


def test_init_legacy_diagnostic_does_not_emit_a_contract_error(tmp_path: Path, capsys):
    assert main(["init", "--machine", "gpu-1", "--runtime-root", str(tmp_path / "runtime")]) == 2

    captured = capsys.readouterr()
    assert "QQTOOLS-COMPAT-0014" in captured.err
    assert "returned no structured output" not in captured.err
    assert captured.out == ""


def test_agent_start_rejects_legacy_persistent_flag(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "agent", "start", "--persistent"])


def test_submit_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "submit", "--", "echo", "ok"]) == 0
    assert reasons == ["submit"]


def test_submit_without_activation_persists_task_and_skips_local_agent(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "submit", "--no-activate", "--quiet", "--", "echo", "ok"]) == 0
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
    create_group(cfg, "demo")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: ['echo', 'ok']\n", encoding="utf-8")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "submit", "--file", str(manifest), "--group", "demo"]) == 0
    assert reasons == ["submit"]


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
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "task", "retry", task.task_id]) == 0
    assert reasons == ["task-retry"]


def test_offer_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "demo")
    task = submit(cfg, ["echo", "ok"], group="demo", sharing_mode="spillover")
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "task", "offer", task.task_id]) == 0
    assert reasons == ["task-offer"]


def test_group_resume_requests_local_agent_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    main([*_base_args(cfg), "group", "create", "demo"])
    main([*_base_args(cfg), "group", "pause", "demo"])
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "group", "resume", "demo"]) == 0
    assert reasons == ["group-resume"]


def test_group_retry_failed_skips_blocked_orphans_and_requests_activation(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "demo")
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
    assert expire_claim(cfg, blocked_task.task_id, orphaned_attempt.attempt_id, orphaned_attempt.current_fencing_token)
    reasons: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: reasons.append(reason) or True,
    )

    assert main([*_base_args(cfg), "group", "retry", "demo"]) == 0
    assert reasons == ["group-retry"]
    assert load_task(cfg, failed_task.task_id).state["projection"] == "queued"
    assert load_task(cfg, blocked_task.task_id).state["projection"] == "blocked"
    output = capsys.readouterr().out
    assert "Retried: 1" in output
    assert failed_task.task_id in output
    assert "Orphaned: 1" in output


def test_agent_stop_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    runtime_root = tmp_path / "machine-runtime"
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.stop_machine_agent",
        lambda _runtime: False,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.get_machine_agent_status",
        lambda runtime: {
            "machine_runtime_root": str(runtime.root),
            "agent_state": "stopped",
            "pid": None,
            "is_running": False,
            "projects": [],
        },
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "stop", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already_stopped"


def test_agent_start_and_stop_named_process_failures_are_operational(tmp_path: Path, monkeypatch, capsys) -> None:
    runtime_root = tmp_path / "machine-runtime"
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.capture_readiness_snapshot",
        lambda _runtime: {"project_ids": ["project-1"]},
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.MachineRuntime.load_registry",
        lambda _runtime: (1, [SimpleNamespace(project_id="project-1", enabled=True)]),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.evaluate_readiness",
        lambda *_args: {"ready": False},
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.ensure_machine_agent_started",
        lambda _runtime: (_ for _ in ()).throw(MachineAgentStartError("spawn denied")),
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "start", "--format=json"]) == 1
    assert "spawn denied" in json.loads(capsys.readouterr().out)["error"]["message"]

    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.stop_machine_agent",
        lambda _runtime: (_ for _ in ()).throw(MachineAgentStopError("stop timed out")),
    )
    assert main(["--machine-runtime-root", str(runtime_root), "agent", "stop", "--format=json"]) == 1
    assert "stop timed out" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_unexpected_setup_runtime_error_is_not_downgraded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.initialize_project",
        lambda _path: (_ for _ in ()).throw(RuntimeError("programming defect")),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        main(["project", "init", str(tmp_path / "project")])


def test_project_register_uninitialized_machine_is_operational(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "--machine-runtime-root",
                str(tmp_path / "machine-runtime"),
                "project",
                "register",
                str(tmp_path / "project"),
                "--format=json",
            ]
        )
        == 1
    )

    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "operational_failure"
    assert "qexp init --machine NAME" in error["message"]


def test_agent_restart_returns_structured_status(tmp_path: Path, monkeypatch, capsys):
    runtime_root = tmp_path / "machine-runtime"

    class FakeProcess:
        pid = 987
        previous_pid = 432

    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.restart_machine_agent",
        lambda _runtime: FakeProcess(),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.get_machine_agent_status",
        lambda runtime: {
            "machine_runtime_root": str(runtime.root),
            "agent_state": "active",
            "pid": 432,
            "is_running": True,
            "projects": [],
        },
    )

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "restart", "--format=json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pid"] == 987
    assert payload["previous_pid"] == 432
    assert payload["machine_runtime_root"] == str(runtime_root)


def test_machine_agent_status_rejects_an_uninitialized_runtime(tmp_path: Path, capsys) -> None:
    runtime_root = tmp_path / "machine-runtime"

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "status", "--format=json"]) == 1
    output = capsys.readouterr()
    assert not output.err
    error = json.loads(output.out)["error"]
    assert error["code"] == "operational_failure"
    assert "qexp init --machine NAME" in error["message"]


def test_read_only_task_list_does_not_initialize_machine_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from qqtools.plugins.qexp.agent.context import MACHINE_RUNTIME_ENV

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    machine_runtime_root = tmp_path / "unused-machine-runtime"
    monkeypatch.setenv(MACHINE_RUNTIME_ENV, str(machine_runtime_root))

    assert main([*_base_args(cfg), "task", "list", "--format=json"]) == 0

    assert json.loads(capsys.readouterr().out) == []
    assert not machine_runtime_root.exists()


def test_legacy_project_requires_explicit_migration(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime_root = tmp_path / "machine-runtime"
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["machine"].pop("agent_runtime")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    base = [*_base_args(cfg), "--machine-runtime-root", str(runtime_root), "agent"]
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.ensure_machine_agent_started",
        lambda _runtime: (None, {}),
    )

    assert main([*base, "start", "--timeout", "0.1"]) == 2
    assert "legacy_project_requires_migration" in capsys.readouterr().out


def test_global_agent_status_and_stop_do_not_require_project_context(tmp_path: Path, monkeypatch, capsys) -> None:
    runtime_root = tmp_path / "machine-runtime"
    machine_status = {
        "machine_runtime_root": str(runtime_root),
        "agent_state": "active",
        "is_running": True,
        "pid": 4321,
        "registry_revision": 1,
        "projects": [],
        "upgrade": {"projects": []},
    }
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.get_machine_agent_status", lambda _runtime: machine_status
    )
    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.stop_machine_agent", lambda _runtime: True)

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "status", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["agent_state"] == "active"

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "stop", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "stopped"


def test_explicit_machine_runtime_root_submits_through_global_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda _cfg, *, reason, machine_runtime: machine_runtime.root == runtime.root and reason == "submit",
    )

    assert (
        main(
            [
                "--project",
                str(cfg.shared_root),
                "--machine",
                cfg.machine_name,
                "--runtime-root",
                str(cfg.runtime_root),
                "--machine-runtime-root",
                str(runtime.root),
                "submit",
                "--",
                "echo",
                "ok",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip()


def test_managed_doctor_reads_project_local_process_evidence(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    process_path = runtime.project_paths(binding.project_id)["processes"] / "missing.json"
    atomic_replace(process_path, {"process": {"task_id": "missing-task"}})

    result = main(
        [
            *_base_args(cfg),
            "--machine-runtime-root",
            str(runtime.root),
            "admin",
            "check",
            "--format=json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert any(issue["code"] == "process_manifest_task_missing" for issue in output["issues"])
