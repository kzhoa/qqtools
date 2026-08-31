import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.machine_runtime import MachineRuntime

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _base_args(cfg) -> list[str]:
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--shared-root",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


def test_task_list_defaults_to_human_and_json_is_explicit(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])

    assert main([*_base_args(cfg), "task", "list"]) == 0
    human = capsys.readouterr().out
    assert human.startswith("Task ID")
    assert task.task_id in human
    assert not human.lstrip().startswith("[")

    assert main([*_base_args(cfg), "task", "list", "--format=json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == [
        {
            "task_id": task.task_id,
            "name": None,
            "group": None,
            "phase": "queued",
            "reason": None,
            "gpus": 1,
            "home_machine": "gpu-1",
            "queue_scope": "home",
            "current_attempt_id": None,
            "claim_machine": None,
        }
    ]


def test_empty_task_list_uses_fixed_message(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert main([*_base_args(cfg), "task", "list"]) == 0
    assert capsys.readouterr().out == "No results.\n"


@pytest.mark.parametrize("value", ["text", "xml"])
def test_invalid_format_is_rejected_before_task_action(tmp_path: Path, monkeypatch, value: str):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("workflow must not run")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.task_commands.offer", fail_if_called)
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "task", "offer", "task_x", f"--format={value}"])
    assert called is False


def test_submit_preserves_training_format_argument(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    received: list[str] = []

    class Result:
        task_id = "task_x"

    def fake_submit(cfg, command, **kwargs):
        received.extend(command)
        return Result()

    monkeypatch.setattr("qqtools.plugins.qexp.cli.task_commands.submit", fake_submit)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.ensure_local_agent_active", lambda *args, **kwargs: True)
    assert main([*_base_args(cfg), "submit", "--", "python", "train.py", "--format=json"]) == 0
    assert received == ["python", "train.py", "--format=json"]
    assert capsys.readouterr().out == "task_x\n"


def test_group_human_outputs_project_summary_and_operation_context(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert main([*_base_args(cfg), "group", "create", "demo"]) == 0
    create = capsys.readouterr().out
    assert "Action: create" in create
    assert "Group: demo" in create

    assert main([*_base_args(cfg), "group", "list"]) == 0
    listing = capsys.readouterr().out
    assert listing.splitlines()[0].startswith("Group")
    assert "Group, demo" not in listing
    assert "demo" in listing
    assert "open" in listing

    assert main([*_base_args(cfg), "group", "show", "demo"]) == 0
    show = capsys.readouterr().out
    assert "Workers: gpu-1=active" in show
    assert "added_by_operation" not in show

    assert main([*_base_args(cfg), "group", "seal", "demo"]) == 0
    seal = capsys.readouterr().out
    assert "Action: seal" in seal
    assert "Group: demo" in seal


def test_group_json_remains_raw_workflow_result(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["group"]["name"] == "demo"
    assert "action" not in created


def test_group_machines_cli_exposes_normalized_role_and_limit(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    capsys.readouterr()

    assert main([
        *_base_args(cfg), "group", "machines", "add", "demo", "gpu-2",
        "--role", "borrow", "--gpu-limit-gpus", "2", "--format=json",
    ]) == 0
    capsys.readouterr()
    assert main([
        *_base_args(cfg), "group", "machines", "list", "demo", "--format=json"
    ]) == 0
    machines = json.loads(capsys.readouterr().out)["machines"]
    assert machines[-1]["scheduling_role"] == "borrow"
    assert machines[-1]["gpu_limit_gpus"] == 2

    assert main([
        *_base_args(cfg), "group", "machines", "set", "demo", "gpu-2",
        "--gpu-limit-gpus", "unlimited", "--format=json",
    ]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["group"]["worker_set"]["gpu-2"]["gpu_limit_gpus"] is None

    assert main([
        *_base_args(cfg), "group", "machines", "add", "demo", "gpu-3",
        "--role", "primary", "--gpu-limit-gpus", "1", "--format=json",
    ]) == 0
    primary = json.loads(capsys.readouterr().out)
    assert primary["group"]["worker_set"]["gpu-3"]["gpu_limit_gpus"] == 1


def test_group_machines_cli_keeps_max_gpus_alias_compatible(tmp_path: Path, capsys):
    """QQTOOLS-COMPAT-0003: Keep --max-gpus through the N release."""
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    capsys.readouterr()

    assert main([
        *_base_args(cfg), "group", "machines", "add", "demo", "gpu-2",
        "--role", "borrow", "--max-gpus", "2", "--format=json",
    ]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["group"]["worker_set"]["gpu-2"]["gpu_limit_gpus"] == 2

    with pytest.raises(SystemExit) as exc_info:
        main([
            *_base_args(cfg), "group", "machines", "set", "demo", "gpu-2",
            "--gpu-limit-gpus", "1", "--max-gpus", "1",
        ])
    assert exc_info.value.code == 2
    assert "not allowed with argument --gpu-limit-gpus" in capsys.readouterr().err


def test_batch_json_is_one_document_and_idempotent_retry_is_silent_on_stderr(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, ok]\n", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.cli.ensure_local_agent_active", lambda *args, **kwargs: True)

    assert main([*_base_args(cfg), "batch-submit", "--file", str(manifest), "--format=json"]) == 0
    first_capture = capsys.readouterr()
    first = json.loads(first_capture.out)
    assert first["state"] == "committed"
    assert first["operation_id"]
    assert first["idempotency_key"]
    assert first_capture.err.startswith("qexp: prepared operation_id=")

    assert (
        main(
            [
                *_base_args(cfg),
                "batch-submit",
                "--file",
                str(manifest),
                "--idempotency-key",
                first["idempotency_key"],
                "--format=json",
            ]
        )
        == 0
    )
    second_capture = capsys.readouterr()
    assert json.loads(second_capture.out) == first
    assert second_capture.err == ""


def test_batch_prepared_notice_precedes_task_staging(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, ok]\n", encoding="utf-8")
    observed: list[tuple[str, str]] = []

    from qqtools.plugins.qexp.commands.task import batch_submit

    def on_prepared(operation_id: str, key: str) -> None:
        observed.append((operation_id, key))
        assert not list((cfg.shared_root / "tasks").glob("*.json"))

    result = batch_submit(cfg, manifest, on_prepared=on_prepared)
    assert observed == [(result.operation_id, result.idempotency_key)]
