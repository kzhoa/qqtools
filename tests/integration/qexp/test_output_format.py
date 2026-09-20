import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.commands.group import create_group

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
            "depends_on_task_ids": [],
            "dependency_state": "ready",
            "dependency_reasons": [],
        }
    ]


def test_progress_policy_cli_reports_default_and_configured_value(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)

    assert main([*base, "config", "progress", "show", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "interval_seconds": 30,
        "source": "default",
        "applies_to": "new_launches",
    }

    assert main([*base, "config", "progress", "set", "--interval-seconds", "60", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "interval_seconds": 60,
        "source": "configured",
        "applies_to": "new_launches",
    }

    assert main([*base, "config", "progress", "show"]) == 0
    human = capsys.readouterr().out
    assert "Interval seconds: 60" in human
    assert "Source: configured" in human
    assert "Applies to: new_launches" in human
    assert "filesystem" in human.lower()


def test_progress_policy_cli_accepts_large_finite_interval(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert (
        main(
            [
                *_base_args(cfg),
                "config",
                "progress",
                "set",
                "--interval-seconds=1e308",
                "--format=json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["interval_seconds"] == 1e308


@pytest.mark.parametrize("value", ["0", "0.5", "nan", "inf", "-inf"])
def test_progress_policy_cli_rejects_invalid_interval(tmp_path: Path, capsys, value: str):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert (
        main(
            [
                *_base_args(cfg),
                "config",
                "progress",
                "set",
                f"--interval-seconds={value}",
            ]
        )
        == 2
    )
    assert "interval" in capsys.readouterr().err.lower()


def test_task_list_json_reports_dependency_gate(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "demo", ["gpu-1"])
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="demo")
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="demo",
        depends_on_task_ids=[parent.task_id],
    )

    assert main([*_base_args(cfg), "task", "list", "--format=json"]) == 0
    tasks = {item["task_id"]: item for item in json.loads(capsys.readouterr().out)}

    assert tasks[parent.task_id]["depends_on_task_ids"] == []
    assert tasks[parent.task_id]["dependency_state"] == "ready"
    assert tasks[parent.task_id]["dependency_reasons"] == []
    assert tasks[child.task_id]["depends_on_task_ids"] == [parent.task_id]
    assert tasks[child.task_id]["dependency_state"] == "waiting"
    assert tasks[child.task_id]["dependency_reasons"] == [{"task_id": parent.task_id, "reason": "queued"}]


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

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "machines",
                "add",
                "demo",
                "gpu-2",
                "--role",
                "borrow",
                "--gpu-limit-gpus",
                "2",
                "--format=json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main([*_base_args(cfg), "group", "machines", "list", "demo", "--format=json"]) == 0
    machines = json.loads(capsys.readouterr().out)["machines"]
    assert machines[-1]["scheduling_role"] == "borrow"
    assert machines[-1]["gpu_limit_gpus"] == 2

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "machines",
                "set",
                "demo",
                "gpu-2",
                "--gpu-limit-gpus",
                "unlimited",
                "--format=json",
            ]
        )
        == 0
    )
    updated = json.loads(capsys.readouterr().out)
    assert updated["group"]["worker_set"]["gpu-2"]["gpu_limit_gpus"] is None

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "machines",
                "add",
                "demo",
                "gpu-3",
                "--role",
                "primary",
                "--gpu-limit-gpus",
                "1",
                "--format=json",
            ]
        )
        == 0
    )
    primary = json.loads(capsys.readouterr().out)
    assert primary["group"]["worker_set"]["gpu-3"]["gpu_limit_gpus"] == 1


def test_group_machines_cli_rejects_removed_max_gpus_alias(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                *_base_args(cfg),
                "group",
                "machines",
                "set",
                "demo",
                "gpu-2",
                "--max-gpus",
                "1",
            ]
        )
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --max-gpus" in capsys.readouterr().err


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


def test_agent_project_list_human_and_json_share_the_registry_result(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime_root = tmp_path / "machine-runtime"
    runtime = MachineRuntime(runtime_root)
    binding, _created = runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    base = ["--machine-runtime-root", str(runtime_root), "agent", "list-projects"]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert human.splitlines()[0].startswith("Project ID")
    assert binding.project_id in human
    assert str(cfg.shared_root) in human
    assert cfg.machine_name in human

    assert main([*base, "--format=json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "project_list"
    assert payload["projects"][0]["project_id"] == binding.project_id
    assert "kind" not in payload
    assert "payload" not in payload
    assert "presentation" not in payload


def test_empty_agent_project_list_human_output_uses_fixed_message(tmp_path: Path, capsys):
    runtime_root = tmp_path / "machine-runtime"

    assert main(["--machine-runtime-root", str(runtime_root), "agent", "list-projects"]) == 0
    assert capsys.readouterr().out == "No results.\n"


def test_cpu_lane_human_and_json_share_the_policy_result(tmp_path: Path, capsys):
    runtime_root = tmp_path / "machine-runtime"
    base = ["--machine-runtime-root", str(runtime_root), "agent", "cpu-lane"]

    assert main([*base, "set", "--capacity", "4"]) == 0
    human = capsys.readouterr().out
    assert "Capacity: 4" in human
    assert "Revision: 1" in human

    assert main([*base, "show", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"cpu_lane": {"capacity": 4, "revision": 1}}


@pytest.mark.parametrize("action", ["coordinate", "retry"])
def test_machine_upgrade_advance_human_reads_flat_projects_and_json_is_unchanged(
    tmp_path: Path, monkeypatch, capsys, action: str
):
    result = {
        "projects": [
            {
                "project_id": "flat-project",
                "phase": "audit",
                "state": "runnable",
                "pending": True,
                "admission_blocked": False,
                "blockers": ["waiting"],
            }
        ],
        "slices": 1,
        "pending_project_ids": ["flat-project"],
        "worker_state": "runnable",
        "discovery_source": "machine_registry",
    }
    calls = 0

    def fake_advance(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr("qqtools.plugins.qexp.cli.advance_registered_upgrades", fake_advance)
    base = ["--machine-runtime-root", str(tmp_path / "machine-runtime"), "agent", "upgrade", action]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert "flat-project" in human
    assert "audit" in human
    assert "runnable" in human
    assert "waiting" in human

    assert main([*base, "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert calls == 2


def test_machine_upgrade_status_human_reads_nested_projects_and_json_is_unchanged(tmp_path: Path, monkeypatch, capsys):
    result = {
        "projects": [
            {
                "project_id": "nested-project",
                "upgrade": {
                    "phase": "backfill",
                    "state": "pending",
                    "pending": True,
                    "admission_blocked": False,
                    "blockers": [],
                },
            }
        ],
        "inaccessible_projects": [],
        "aggregate_state": "pending",
        "pending_project_ids": ["nested-project"],
        "all_roots_complete": False,
        "discovery_source": "machine_registry",
        "discovery_boundary": "locally_registered_bindings",
    }
    monkeypatch.setattr("qqtools.plugins.qexp.cli.inspect_registered_upgrades", lambda _runtime: result)
    base = ["--machine-runtime-root", str(tmp_path / "machine-runtime"), "agent", "upgrade", "status"]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert "nested-project" in human
    assert "backfill" in human
    assert "pending" in human

    assert main([*base, "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_group_and_machine_human_output_do_not_query_task_history_for_presentation(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "demo", ["gpu-1"])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("human presentation must not perform a Task history query")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.observer.list_tasks", fail_if_called)
    base = _base_args(cfg)

    assert main([*base, "group", "list"]) == 0
    group_list = capsys.readouterr().out
    assert "demo" in group_list
    assert main([*base, "group", "show", "demo"]) == 0
    group_show = capsys.readouterr().out
    assert "Task summary: -" in group_show
    assert "Queue summary: -" in group_show
    assert main([*base, "machines"]) == 0
    machines = capsys.readouterr().out
    assert machines == "No results.\n" or "Task summary" in machines.splitlines()[0]
