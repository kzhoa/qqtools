import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.tasks import load_task

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


def test_human_and_json_each_execute_once_and_emit_once(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "ok"])
    from qqtools.plugins.qexp.cli import entrypoint, project_handlers

    workflow_calls: list[str] = []
    render_calls: list[str] = []
    original_list_tasks = project_handlers.observer.list_tasks
    original_render = entrypoint.render

    def counted_list_tasks(*args, **kwargs):
        workflow_calls.append("list")
        return original_list_tasks(*args, **kwargs)

    def counted_render(output, output_format):
        render_calls.append(output_format)
        return original_render(output, output_format)

    monkeypatch.setattr(project_handlers.observer, "list_tasks", counted_list_tasks)
    monkeypatch.setattr(entrypoint, "render", counted_render)

    assert main([*_base_args(cfg), "task", "list"]) == 0
    human = capsys.readouterr()
    assert main([*_base_args(cfg), "task", "list", "--format=json"]) == 0
    structured = capsys.readouterr()

    assert workflow_calls == ["list", "list"]
    assert render_calls == ["human", "json"]
    assert human.out.count("Task ID") == 1
    assert human.err == ""
    assert len(json.loads(structured.out)) == 1
    assert structured.err == ""


def test_progress_policy_cli_reports_default_and_configured_value(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)

    assert main([*base, "config", "show", "progress", "--format=json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == {
        "action": "show",
        "section": "progress",
        "scope": "project",
        "source": "default",
        "applies_to": "new_launches",
        "values": {"interval_seconds": 30, "source": "default", "applies_to": "new_launches"},
        "effective_values": {"interval_seconds": 30, "source": "default", "applies_to": "new_launches"},
    }

    assert main([*base, "config", "set", "progress", "--interval-seconds", "60", "--format=json"]) == 0
    configured = json.loads(capsys.readouterr().out)
    assert configured["values"] == {
        "interval_seconds": 60,
        "source": "configured",
        "applies_to": "new_launches",
    }

    assert main([*base, "config", "show", "progress"]) == 0
    human = capsys.readouterr().out
    assert "interval_seconds=60" in human
    assert "source=configured" in human
    assert "applies_to=new_launches" in human


def test_tmux_policy_cli_reports_project_default_and_configured_value_without_activation(
    tmp_path: Path, monkeypatch, capsys
):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)
    activations = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda *_args, **_kwargs: activations.append(True),
    )

    assert main([*base, "config", "show", "tmux", "--format=json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["values"] == {
        "enabled": False,
        "source": "default",
        "applies_to": "new_observer_decisions",
    }
    assert not (cfg.shared_root / "tmux-policy.json").exists()

    assert main([*base, "config", "set", "tmux", "--enabled", "--format=json"]) == 0
    configured = json.loads(capsys.readouterr().out)
    assert configured["values"] == {
        "enabled": True,
        "source": "configured",
        "applies_to": "new_observer_decisions",
    }
    assert main([*base, "config", "show", "tmux"]) == 0
    human = capsys.readouterr().out.lower()
    assert "enabled" in human
    assert "configured" in human
    assert "project" in human
    assert "new_observer_decisions" in human
    assert activations == []


def test_progress_policy_cli_accepts_large_finite_interval(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert (
        main(
            [
                *_base_args(cfg),
                "config",
                "set",
                "progress",
                "--interval-seconds=1e308",
                "--format=json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["values"]["interval_seconds"] == 1e308


def test_launch_handoff_policy_cli_reports_default_and_configured_value(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)

    assert main([*base, "config", "show", "launch-handoff", "--format=json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["values"] == {
        "timeout_seconds": 10,
        "source": "default",
        "applies_to": "new_launches",
    }

    assert (
        main(
            [
                *base,
                "config",
                "set",
                "launch-handoff",
                "--timeout-seconds",
                "12.5",
                "--format=json",
            ]
        )
        == 0
    )
    configured = json.loads(capsys.readouterr().out)
    assert configured["values"] == {
        "timeout_seconds": 12.5,
        "source": "configured",
        "applies_to": "new_launches",
    }

    assert main([*base, "config", "show", "launch-handoff"]) == 0
    human = capsys.readouterr().out
    assert "timeout_seconds=12.5" in human
    assert "source=configured" in human
    assert "applies_to=new_launches" in human


@pytest.mark.parametrize("value", ["0", "0.5", "301", "nan", "inf", "-inf"])
def test_launch_handoff_policy_cli_rejects_invalid_timeout(tmp_path: Path, capsys, value: str):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert (
        main(
            [
                *_base_args(cfg),
                "config",
                "set",
                "launch-handoff",
                f"--timeout-seconds={value}",
                "--format=json",
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    error = json.loads(output.out)["error"]
    assert not output.err
    assert error["code"] == "invalid_argument"
    assert "timeout" in error["message"].lower()


@pytest.mark.parametrize("value", ["0", "0.5", "nan", "inf", "-inf"])
def test_progress_policy_cli_rejects_invalid_interval(tmp_path: Path, capsys, value: str):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert (
        main(
            [
                *_base_args(cfg),
                "config",
                "set",
                "progress",
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
    assert capsys.readouterr().out == "No Tasks.\n"


@pytest.mark.parametrize("value", ["text", "xml"])
def test_invalid_format_is_rejected_before_task_action(tmp_path: Path, monkeypatch, value: str):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("workflow must not run")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.project_handlers.task_commands.offer", fail_if_called)
    with pytest.raises(SystemExit):
        main([*_base_args(cfg), "task", "offer", "task_x", f"--format={value}"])
    assert called is False


def test_submit_preserves_training_format_argument(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    received: list[str] = []

    monkeypatch.setattr("qqtools.plugins.qexp.cli.submission.ensure_local_agent_active", lambda *args, **kwargs: True)
    assert main([*_base_args(cfg), "submit", "--quiet", "--", "python", "train.py", "--format=json"]) == 0
    task_id = capsys.readouterr().out.strip()
    task = load_task(cfg, task_id)
    received.extend(task.spec.command)
    assert received == ["python", "train.py", "--format=json"]


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
    assert created["action"] == "create"
    assert created["outcome"] == "completed"


def test_group_workers_cli_exposes_normalized_role_and_limit(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "worker",
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
    assert main([*_base_args(cfg), "group", "worker", "list", "demo", "--format=json"]) == 0
    machines = json.loads(capsys.readouterr().out)["machines"]
    assert machines[-1]["scheduling_role"] == "borrow"
    assert machines[-1]["gpu_limit_gpus"] == 2

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "worker",
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
    assert updated["action"] == "set"
    assert updated["outcome"] == "completed"
    assert updated["worker_state"] == "active"
    assert updated["scheduling_role"] == "borrow"
    assert updated["gpu_limit_gpus"] is None

    assert (
        main(
            [
                *_base_args(cfg),
                "group",
                "worker",
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


def test_group_workers_cli_rejects_removed_max_gpus_alias(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    assert main([*_base_args(cfg), "group", "create", "demo", "--format=json"]) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                *_base_args(cfg),
                "group",
                "worker",
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
    monkeypatch.setattr("qqtools.plugins.qexp.cli.submission.ensure_local_agent_active", lambda *args, **kwargs: True)

    assert main([*_base_args(cfg), "submit", "--file", str(manifest), "--format=json"]) == 0
    first_capture = capsys.readouterr()
    first = json.loads(first_capture.out)
    assert first["outcome"] == "committed"
    assert first["operation"]["state"] == "committed"
    assert first["operation"]["id"]
    assert first["idempotency_key"]
    assert first["preview"] is None
    assert first["error"] is None
    assert first_capture.err.startswith("qexp: prepared operation_id=")

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
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


def test_project_list_human_and_json_share_the_registry_result(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime_root = tmp_path / "machine-runtime"
    prefix = ["--machine-runtime-root", str(runtime_root)]
    assert main([*prefix, "init", "--machine", "gpu-1", "--format=json"]) == 0
    capsys.readouterr()
    assert main([*prefix, "project", "register", str(cfg.shared_root), "--format=json"]) == 0
    binding = json.loads(capsys.readouterr().out)["projects"][0]
    base = [*prefix, "project", "list"]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert human.splitlines()[0].startswith("Project ID")
    assert binding["project_id"] in human
    assert str(cfg.shared_root) in human
    assert cfg.machine_name in human

    assert main([*base, "--format=json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["projects"][0]["project_id"] == binding["project_id"]
    assert "kind" not in payload
    assert "payload" not in payload
    assert "presentation" not in payload


def test_empty_project_list_human_output_uses_fixed_message(tmp_path: Path, capsys):
    runtime_root = tmp_path / "machine-runtime"
    prefix = ["--machine-runtime-root", str(runtime_root)]
    assert main([*prefix, "init", "--machine", "gpu-1", "--format=json"]) == 0
    capsys.readouterr()

    assert main([*prefix, "project", "list"]) == 0
    assert capsys.readouterr().out == "No enrolled Projects.\nNext: qexp project register PATH\n"


def test_cpu_lane_human_and_json_share_the_policy_result(tmp_path: Path, capsys):
    runtime_root = tmp_path / "machine-runtime"
    base = ["--machine-runtime-root", str(runtime_root), "agent", "config", "cpu"]

    assert main([*base, "set", "--capacity", "4"]) == 0
    human = capsys.readouterr().out
    assert "Capacity: 4" in human
    assert "Revision: 1" in human
    assert f"MachineRuntime root: {runtime_root}" in human

    assert main([*base, "show", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "machine_runtime_root": str(runtime_root),
        "cpu_lane": {"capacity": 4, "revision": 1},
        "action": "shown",
    }


def test_machine_upgrade_advance_human_reads_flat_projects_and_json_is_unchanged(tmp_path: Path, monkeypatch, capsys):
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

    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.advance_registered_upgrades", fake_advance)
    base = ["--machine-runtime-root", str(tmp_path / "machine-runtime"), "admin", "upgrade", "advance"]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert "flat-project" in human
    assert "audit" in human
    assert "runnable" in human
    assert "waiting" in human

    assert main([*base, "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        **result,
        "action": "advance",
        "outcome": "waiting",
        "reason": "upgrade work remains pending",
        "next_action": "qexp admin upgrade status",
    }
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
    monkeypatch.setattr("qqtools.plugins.qexp.cli.local_handlers.inspect_registered_upgrades", lambda _runtime: result)
    base = ["--machine-runtime-root", str(tmp_path / "machine-runtime"), "admin", "upgrade", "status"]

    assert main(base) == 0
    human = capsys.readouterr().out
    assert "nested-project" in human
    assert "backfill" in human
    assert "pending" in human

    assert main([*base, "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        **result,
        "action": "status",
        "outcome": "waiting",
        "reason": "upgrade work remains pending",
        "next_action": "qexp admin upgrade status",
    }


def test_group_and_machine_human_output_do_not_query_task_history_for_presentation(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "demo", ["gpu-1"])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("human presentation must not perform a Task history query")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.project_handlers.observer.list_tasks", fail_if_called)
    base = _base_args(cfg)

    assert main([*base, "group", "list"]) == 0
    group_list = capsys.readouterr().out
    assert "demo" in group_list
    assert main([*base, "group", "show", "demo"]) == 0
    group_show = capsys.readouterr().out
    assert "Task summary: -" in group_show
    assert "Queue summary: -" in group_show
    assert main([*base, "machine", "list"]) == 0
    machines = capsys.readouterr().out
    assert machines == "No results.\n" or "Task summary" in machines.splitlines()[0]


@pytest.mark.parametrize(
    ("argv", "expected_code", "stderr_expected"),
    [
        (["task", "list", "--page-size", "10", "--format=json", "--unknown"], "invalid_argument", False),
        (["status", "--format=json", "--unknown"], "invalid_argument", False),
        (["submit", "--format=json", "--unknown"], "invalid_input", True),
    ],
)
def test_raw_argv_parse_failures_retain_their_structured_routing(
    argv: list[str], expected_code: str, stderr_expected: bool, capsys
) -> None:
    assert main(argv) == 2

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    error = result["error"]
    assert error["code"] == expected_code
    assert bool(captured.err) is stderr_expected
