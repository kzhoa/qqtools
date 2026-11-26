from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

# QQTOOLS-COMPAT-0015: verify the provisional Group publication writer fence and recovery path.


def _project(tmp_path: Path, directory: str = "project"):
    project = tmp_path / directory
    cfg = init_shared_root(project / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    machine_runtime = MachineRuntime(tmp_path / "machine-runtime")
    machine_runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    return project, cfg, machine_runtime


def _submit_prefix(project: Path, cfg, machine_runtime: MachineRuntime) -> list[str]:
    return [
        "--machine",
        cfg.machine_name,
        "--machine-runtime-root",
        str(machine_runtime.root),
        "submit",
        "--project",
        str(project),
    ]


def _disable_activation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda *_args, **kwargs: calls.append(str(kwargs.get("reason"))),
    )
    return calls


def test_command_mode_json_uses_complete_submission_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    activation = _disable_activation(monkeypatch)

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--format", "json", "--", "echo", "ok"]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result == {
        "schema_version": 1,
        "mode": "command",
        "outcome": "committed",
        "project": {"path": str(project.resolve()), "source": "cli"},
        "group": {"name": None, "source": "none", "disposition": "none"},
        "operation": {"id": result["operation"]["id"], "state": "committed"},
        "idempotency_key": result["idempotency_key"],
        "task_ids": [result["task_ids"][0]],
        "preview": None,
        "error": None,
        "activation": {"outcome": "requested", "follow_up_command": None},
    }
    assert captured.err.startswith("qexp: prepared operation_id=")
    assert f"idempotency_key={result['idempotency_key']}" in captured.err
    assert activation == ["submit"]
    task_id = result["task_ids"][0]
    task = read_json(cfg.shared_root / "tasks" / f"{task_id}.json")["task"]
    assert task["spec"]["command"] == ["echo", "ok"]
    assert task["spec"]["working_directory"] == str(Path.cwd().resolve())


def test_command_mode_creates_missing_group_with_current_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    _disable_activation(monkeypatch)

    assert (
        main([*_submit_prefix(project, cfg, machine_runtime), "--group", "new", "--format", "json", "--", "echo", "ok"])
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["group"] == {"name": "new", "source": "cli", "disposition": "created"}
    group = read_json(group_path(cfg.shared_root, "new"))["group"]
    assert list(group["worker_set"]) == [cfg.machine_name]


def test_file_mode_group_identity_overrides_and_manifest_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    _disable_activation(monkeypatch)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "runs.yaml"
    manifest.write_text(
        """
group:
  name: from-file
  workers: [gpu-1]
defaults:
  requested_gpus: 2
  working_directory: work
  tmux: true
tasks:
  - name: one
    command: [python, train.py]
    requested_gpus: 1
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                *_submit_prefix(project, cfg, machine_runtime),
                "--file",
                str(manifest),
                "--group",
                "from-cli",
                "--gpus",
                "0",
                "--cpus",
                "3",
                "--no-tmux",
                "--format",
                "json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["group"] == {"name": "from-cli", "source": "cli", "disposition": "created"}
    task = read_json(cfg.shared_root / "tasks" / f"{result['task_ids'][0]}.json")["task"]
    assert task["group_name"] == "from-cli"
    assert task["spec"]["lane"] == "cpu"
    assert task["spec"]["requested_cpus"] == 3
    assert task["spec"]["working_directory"] == str((manifests / "work").resolve())
    operation = read_json(submission_path(cfg.shared_root, result["operation"]["id"]))
    assert operation["task_observation"]["tasks"][0]["tmux_override"] is False
    group = read_json(group_path(cfg.shared_root, "from-cli"))["group"]
    assert group["creation_operation_id"] == result["operation"]["id"]


def test_file_mode_absent_working_directory_uses_selected_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    _disable_activation(monkeypatch)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, ok]\n", encoding="utf-8")

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--file", str(manifest), "--quiet"]) == 0

    task_id = capsys.readouterr().out.strip()
    task = read_json(cfg.shared_root / "tasks" / f"{task_id}.json")["task"]
    assert task["spec"]["working_directory"] == str(project.resolve())


@pytest.mark.parametrize("live_progress", [None, True, False])
def test_dry_run_is_read_only_and_reports_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], live_progress: bool | None
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    activation = _disable_activation(monkeypatch)
    manifest = project / "runs.yaml"
    manifest.write_text(
        """
group:
  name: preview-group
  workers: [gpu-1]
defaults:
  requested_gpus: 2
tasks:
  - command: [echo, ok]
    requested_gpus: 1
""",
        encoding="utf-8",
    )
    if live_progress is not None:
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(f"    live_progress: {str(live_progress).lower()}\n")
    before = sorted(path.relative_to(cfg.shared_root) for path in cfg.shared_root.rglob("*.json"))

    assert (
        main(
            [
                *_submit_prefix(project, cfg, machine_runtime),
                "--file",
                str(manifest),
                "--gpus",
                "3",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "preview"
    assert result["operation"] is None
    assert result["task_ids"] == []
    assert result["preview"]["tasks"][0]["task_id"] is None
    assert result["preview"]["tasks"][0]["requested_gpus"] == 3
    assert result["preview"]["tasks"][0]["sources"]["requested_gpus"] == "cli"
    assert result["preview"]["group_action"] == "create"
    assert activation == []
    after = sorted(path.relative_to(cfg.shared_root) for path in cfg.shared_root.rglob("*.json"))
    assert after == before


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "exactly one"),
        (["--file", "runs.yaml", "--", "echo", "ok"], "exactly one"),
        (["--file", "runs.yaml", "--task-id", "x"], "command mode"),
        (["--quiet", "--format", "json", "--", "echo", "ok"], "--quiet"),
        (["--quiet", "--dry-run", "--", "echo", "ok"], "--quiet"),
    ],
)
def test_invalid_mode_combinations_fail_before_resolution_or_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    activation = _disable_activation(monkeypatch)

    assert main(["submit", *arguments]) == 2

    captured = capsys.readouterr()
    assert message in captured.err
    assert activation == []


def test_json_parser_failure_uses_complete_result_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["submit", "--format", "json", "--gpus", "not-an-int", "--", "echo", "ok"]) == 2

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["mode"] == "command"
    assert result["outcome"] == "rejected"
    assert result["project"] is None
    assert result["operation"] is None
    assert result["task_ids"] == []
    assert result["error"]["code"] == "invalid_input"
    assert "invalid" in captured.err


def test_command_mode_requires_literal_separator(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["submit", "--format", "json", "echo", "ok"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "rejected"
    assert result["error"]["code"] == "invalid_input"
    assert "separator" in result["error"]["message"]


def test_manifest_ancestry_precedes_cwd_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cwd_project, _cwd_cfg, _cwd_runtime = _project(tmp_path / "cwd")
    manifest_project, cfg, machine_runtime = _project(tmp_path / "manifest")
    manifest = manifest_project / "inputs" / "runs.yaml"
    manifest.parent.mkdir()
    manifest.write_text("tasks:\n  - command: [echo, ok]\n", encoding="utf-8")
    monkeypatch.chdir(cwd_project)
    monkeypatch.setenv("QEXP_MACHINE_RUNTIME_ROOT", str(machine_runtime.root))
    monkeypatch.setenv("QEXP_MACHINE", cfg.machine_name)
    _disable_activation(monkeypatch)

    assert main(["submit", "--file", str(manifest), "--format", "json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["project"] == {"path": str(manifest_project.resolve()), "source": "manifest_ancestor"}


def test_same_key_changed_override_is_a_structured_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    _disable_activation(monkeypatch)
    prefix = [*_submit_prefix(project, cfg, machine_runtime), "--idempotency-key", "same", "--format", "json"]
    assert main([*prefix, "--gpus", "1", "--", "echo", "ok"]) == 0
    capsys.readouterr()

    assert main([*prefix, "--gpus", "2", "--", "echo", "ok"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "rejected"
    assert result["error"]["code"] == "idempotency_conflict"
    assert result["operation"] is not None
    assert result["task_ids"] == []


def test_activation_failure_preserves_committed_ids_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)

    from qqtools.plugins.qexp.activation import AgentActivationError

    def fail_activation(*_args, **_kwargs):
        raise AgentActivationError("agent unavailable")

    monkeypatch.setattr("qqtools.plugins.qexp.cli.submission.ensure_local_agent_active", fail_activation)

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--format", "json", "--", "echo", "ok"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "committed"
    assert result["task_ids"]
    assert result["operation"]["state"] == "committed"
    assert result["error"]["code"] == "activation_failed"
    assert result["activation"]["outcome"] == "failed"
    follow_up = shlex.split(result["activation"]["follow_up_command"])
    assert follow_up == ["qexp", "--machine-runtime-root", str(machine_runtime.root), "agent", "start"]


def test_activation_failure_human_states_commit_failure_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from qqtools.plugins.qexp.activation import AgentActivationError

    project, cfg, machine_runtime = _project(tmp_path, "training project")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AgentActivationError("agent unavailable")),
    )

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--", "echo", "ok"]) == 1

    output = capsys.readouterr().out
    assert "submitted" in output.splitlines()[0].lower()
    assert "activation failed" in output.splitlines()[0].lower()
    assert "Do not submit again." in output
    assert "agent unavailable" in output
    assert "agent start" in output
    assert str(machine_runtime.root) in output
    task_id = next(line.removeprefix("Task ID: ") for line in output.splitlines() if line.startswith("Task ID: "))
    assert read_json(cfg.shared_root / "tasks" / f"{task_id}.json")["task"]["task_id"] == task_id
    for action in ("show", "logs"):
        command = next(line[line.index("qexp ") :] for line in output.splitlines() if f"task {action}" in line)
        assert shlex.split(command) == [
            "qexp",
            "--machine-runtime-root",
            str(machine_runtime.root),
            "--project",
            str(project),
            "task",
            action,
            task_id,
        ]


@pytest.mark.parametrize("outcome", ["preview", "not_requested", "started", "failed"])
def test_manifest_human_separates_submission_from_activation(tmp_path, monkeypatch, capsys, outcome):
    from qqtools.plugins.qexp.activation import AgentActivationError

    project, cfg, runtime = _project(tmp_path, "training project")
    task_ids = ["train-" + "x" * 84, "train-b"]
    manifest = tmp_path / "training manifest.yaml"
    manifest.write_text(
        "group:\n  name: study\n  workers: [gpu-1]\ntasks:\n"
        + "".join(
            f"  - task_id: {task_id}\n    command: [echo, ok]\n    requested_gpus: 0\n    requested_cpus: 1\n"
            for task_id in task_ids
        ),
        encoding="utf-8",
    )
    calls = []

    def activate(*args, **kwargs):
        calls.append(kwargs["machine_runtime"])
        if outcome == "failed":
            raise AgentActivationError("agent unavailable")
        return True

    monkeypatch.setattr("qqtools.plugins.qexp.cli.submission.ensure_local_agent_active", activate)
    options = {"preview": ["--dry-run"], "not_requested": ["--no-activate"]}.get(outcome, [])
    assert main([*_submit_prefix(project, cfg, runtime), "--file", str(manifest), *options]) == (outcome == "failed")

    output = capsys.readouterr().out
    if outcome == "preview":
        assert "no tasks were submitted" in output.splitlines()[0].lower()
        assert not calls
        assert all(not (cfg.shared_root / "tasks" / f"{task_id}.json").exists() for task_id in task_ids)
        return

    assert "submitted" in output.splitlines()[0].lower()
    assert all(task_id in output for task_id in task_ids)
    command = next(line[line.index("qexp ") :] for line in output.splitlines() if "group show" in line)
    assert shlex.split(command) == [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "--project",
        str(project),
        "group",
        "show",
        "study",
    ]
    assert "tasks are running" not in output.lower()
    if outcome == "failed":
        assert "activation failed" in output.lower() and "Do not submit again." in output
    elif outcome == "not_requested":
        assert "local agent activation was not requested" in output.lower()
    else:
        assert "local agent started" in output.lower()
    assert [call.root for call in calls] == ([] if outcome == "not_requested" else [runtime.root])


@pytest.mark.parametrize("output_format", ["human", "json", "quiet"])
def test_activation_read_failure_keeps_committed_ids(tmp_path, monkeypatch, capsys, output_format):
    project, cfg, runtime = _project(tmp_path, "training project")

    def fail_binding_read(*args, **kwargs):
        raise PermissionError("cannot reread local registration")

    monkeypatch.setattr(MachineRuntime, "matching_binding", fail_binding_read)
    options = ["--quiet"] if output_format == "quiet" else ["--format", output_format]
    task_id = "already-submitted"
    assert main([*_submit_prefix(project, cfg, runtime), *options, "--task-id", task_id, "--", "echo", "ok"]) == 1
    captured = capsys.readouterr()
    if output_format == "quiet":
        assert captured.out.splitlines() == [task_id]
    else:
        if output_format == "json":
            result = json.loads(captured.out)
            assert result["outcome"] == "committed"
            assert result["task_ids"] == [task_id]
            assert result["error"]["code"] == "activation_failed"
            follow_up = result["activation"]["follow_up_command"]
        else:
            assert "Do not submit again." in captured.out
            assert "Task ID: already-submitted" in captured.out
            follow_up = next(
                line.removeprefix("Next: ") for line in captured.out.splitlines() if line.startswith("Next: ")
            )
        assert shlex.split(follow_up) == [
            "qexp",
            "--machine-runtime-root",
            str(runtime.root),
            "--project",
            str(project),
            "status",
        ]
    assert (cfg.shared_root / "tasks" / f"{task_id}.json").exists()


@pytest.mark.parametrize("malformed", [False, True])
def test_activation_handles_registry_loss_after_commit(tmp_path, monkeypatch, capsys, malformed):
    from qqtools.plugins.qexp.activation import ensure_local_agent_active
    from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects

    project = tmp_path / "training project"
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "gpu-1")
    initialize_project(project)
    assert register_projects(runtime, [project])["projects"][0]["status"] == "registered"
    cfg = runtime.verified_execution_context(project / ".qexp").cfg
    assert runtime.paths["inventory"].is_file()
    task_id = "already-submitted"

    def remove_registry_before_activation(*args, **kwargs):
        assert (cfg.shared_root / "tasks" / f"{task_id}.json").is_file()
        if malformed:
            runtime.paths["registry"].write_text('{"registry": {}}', encoding="utf-8")
        else:
            runtime.paths["registry"].unlink()
        return ensure_local_agent_active(*args, **kwargs)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active", remove_registry_before_activation
    )
    assert main([*_submit_prefix(project, cfg, runtime), "--task-id", task_id, "--", "echo", "ok"]) == 1
    captured = capsys.readouterr()
    assert "Do not submit again." in captured.out
    assert "Task ID: already-submitted" in captured.out
    assert "machine registry" in captured.out
    assert "task show" in captured.out
    assert not runtime.paths["pid"].exists()


def test_unexpected_activation_exception_is_not_downgraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.ensure_local_agent_active",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("programming defect")),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        main([*_submit_prefix(project, cfg, machine_runtime), "--", "echo", "ok"])


@pytest.mark.parametrize(
    "error", [ValueError("programming defect"), RuntimeError("programming defect"), OSError("programming defect")]
)
def test_unexpected_submission_workflow_exception_is_not_downgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.submission.task_commands.submit_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error), match="programming defect"):
        main([*_submit_prefix(project, cfg, machine_runtime), "--", "echo", "ok"])


@pytest.mark.parametrize("output_format", ["human", "json", "quiet"])
def test_finalization_failure_preserves_committed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    from qqtools.plugins.qexp.runtime import submission as submission_runtime

    project, cfg, machine_runtime = _project(tmp_path)
    monkeypatch.setattr(
        submission_runtime,
        "finalize_submission_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("finalizer unavailable")),
    )

    options = ["--quiet"] if output_format == "quiet" else ["--format", output_format]
    assert main([*_submit_prefix(project, cfg, machine_runtime), *options, "--", "echo", "ok"]) == 1

    captured = capsys.readouterr()
    if output_format == "human":
        assert not captured.out
        assert "Submitted 1 Task. A post-submission step failed." in captured.err
        assert "Do not submit again." in captured.err
        assert "finalizer unavailable" in captured.err
        task_id = next(
            line.removeprefix("Task ID: ") for line in captured.err.splitlines() if line.startswith("Task ID: ")
        )
        assert (cfg.shared_root / "tasks" / f"{task_id}.json").exists()
        assert "task show" in captured.err
        return
    if output_format == "quiet":
        (task_id,) = captured.out.splitlines()
        assert (cfg.shared_root / "tasks" / f"{task_id}.json").exists()
        return
    result = json.loads(captured.out)
    assert result["outcome"] == "committed"
    assert result["operation"]["state"] == "committed"
    assert result["task_ids"]
    assert result["error"]["code"] == "finalization_pending"


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_interruption_after_commit_preserves_verified_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    from qqtools.plugins.qexp.runtime import submission as submission_runtime

    project, cfg, machine_runtime = _project(tmp_path)
    monkeypatch.setattr(
        submission_runtime,
        "finalize_submission_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--format", output_format, "--", "echo", "ok"]) == 130

    captured = capsys.readouterr()
    assert "Do not submit again." in captured.err
    if output_format == "human":
        assert "Submitted 1 Task." in captured.err and "interrupted" in captured.err
        task_id = next(
            line.removeprefix("Task ID: ") for line in captured.err.splitlines() if line.startswith("Task ID: ")
        )
        assert (cfg.shared_root / "tasks" / f"{task_id}.json").exists()
        assert "task show" in captured.err
        return
    result = json.loads(captured.out)
    assert result["outcome"] == "committed"
    assert result["operation"]["state"] == "committed"
    assert result["task_ids"]
    assert result["error"]["code"] == "interrupted"


def test_recovery_identifiers_are_disclosed_before_provisional_group_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from qqtools.plugins.qexp.runtime import submission as submission_runtime
    from qqtools.plugins.qexp.runtime.submission import SubmissionPending

    project, cfg, machine_runtime = _project(tmp_path)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  name: staged\n  workers: [gpu-1]\ntasks:\n  - command: [echo, ok]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        submission_runtime,
        "sync_primary_ready_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SubmissionPending("projection failed")),
    )

    assert (
        main(
            [
                *_submit_prefix(project, cfg, machine_runtime),
                "--file",
                str(manifest),
                "--format",
                "json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["outcome"] == "pending"
    assert result["operation"]["id"]
    assert result["idempotency_key"]
    assert f"operation_id={result['operation']['id']}" in captured.err
    assert f"idempotency_key={result['idempotency_key']}" in captured.err


def test_existing_group_worker_addition_requires_publication_capability(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.commands.task import batch_submit
    from qqtools.plugins.qexp.runtime.protocol_compatibility import SUBMISSION_GROUP_PUBLICATION_CAPABILITY

    _project_root, cfg, _machine_runtime = _project(tmp_path)
    init_shared_root(tmp_path / "project" / ".qexp", "gpu-2", runtime_root=tmp_path / "runtime-2")
    create_group(cfg, "existing", ["gpu-1"])
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove(SUBMISSION_GROUP_PUBLICATION_CAPABILITY)
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  name: existing\n  workers: [gpu-2]\ntasks:\n  - command: [echo, ok]\n",
        encoding="utf-8",
    )

    submissions = cfg.shared_root / "operations" / "submissions"
    before = set(submissions.glob("*.json"))
    with pytest.raises(RuntimeError, match=SUBMISSION_GROUP_PUBLICATION_CAPABILITY):
        batch_submit(cfg, manifest)

    assert set(submissions.glob("*.json")) == before
    assert list(read_json(group_path(cfg.shared_root, "existing"))["group"]["worker_set"]) == ["gpu-1"]


def test_human_follow_up_commands_use_parseable_project_locator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, cfg, machine_runtime = _project(tmp_path)
    _disable_activation(monkeypatch)

    assert main([*_submit_prefix(project, cfg, machine_runtime), "--", "echo", "ok"]) == 0

    lines = capsys.readouterr().out.splitlines()
    for label in ("Show: ", "Logs: "):
        command = next(line.removeprefix(label) for line in lines if line.startswith(label))
        parsed = build_parser().parse_args(shlex.split(command)[1:])
        assert parsed.project == str(project)


@pytest.mark.parametrize("abort", [False, True])
def test_process_death_after_provisional_group_publication_hides_group_and_allows_same_key_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], abort: bool
) -> None:
    from qqtools.plugins.qexp.commands.task import batch_submit
    from qqtools.plugins.qexp.observer import list_groups
    from qqtools.plugins.qexp.runtime import submission as submission_runtime
    from qqtools.plugins.qexp.runtime.group_namespace import GroupNotPublished, read_group

    project, cfg, machine_runtime = _project(tmp_path)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  name: interrupted\n  workers: [gpu-1]\ntasks:\n  - command: [echo, ok]\n",
        encoding="utf-8",
    )
    original_save = submission_runtime.save_task
    interrupted = False

    def die_after_group_publication(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_save(*args, **kwargs)

    monkeypatch.setattr(submission_runtime, "save_task", die_after_group_publication)
    with pytest.raises(KeyboardInterrupt):
        batch_submit(cfg, manifest, idempotency_key="recover")

    raw_group = read_json(group_path(cfg.shared_root, "interrupted"))["group"]
    operation_id = raw_group["creation_operation_id"]
    operation = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    assert operation["state"] in {"preparing", "committing"}
    with pytest.raises(GroupNotPublished, match=operation_id):
        read_group(cfg.shared_root, "interrupted")
    assert all(item["name"] != "interrupted" for item in list_groups(cfg))
    from qqtools.plugins.qexp.observer import list_groups_page

    for history in (False, True):
        assert list_groups_page(cfg, all_groups=history)["items"] == []

    assert (
        main(
            [
                *_submit_prefix(project, cfg, machine_runtime),
                "--file",
                str(manifest),
                "--idempotency-key",
                "different",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 1
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["outcome"] == "rejected"
    assert "not published" in preview["error"]["message"]

    monkeypatch.setattr(submission_runtime, "save_task", original_save)
    if abort:
        assert submission_runtime.reconcile_submission(cfg, operation_id, abort_incomplete=True) == "aborted"
        for history in (False, True):
            assert list_groups_page(cfg, all_groups=history)["items"] == []
        return
    recovered = batch_submit(cfg, manifest, idempotency_key="recover")
    assert recovered.operation_id == operation_id
    assert read_group(cfg.shared_root, "interrupted")["group"]["name"] == "interrupted"


def test_committed_group_is_visible_even_when_finalization_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qqtools.plugins.qexp.commands.task import batch_submit
    from qqtools.plugins.qexp.runtime import submission as submission_runtime
    from qqtools.plugins.qexp.runtime.group_namespace import read_group

    _project_root, cfg, _machine_runtime = _project(tmp_path)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  name: committed\n  workers: [gpu-1]\ntasks:\n  - command: [echo, ok]\n",
        encoding="utf-8",
    )
    original_finalize = submission_runtime.finalize_submission_group

    def interrupt_finalization(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(submission_runtime, "finalize_submission_group", interrupt_finalization)
    with pytest.raises(KeyboardInterrupt):
        batch_submit(cfg, manifest, idempotency_key="finish")

    raw_group = read_json(group_path(cfg.shared_root, "committed"))["group"]
    operation = read_json(submission_path(cfg.shared_root, raw_group["creation_operation_id"]))["submission"]
    assert operation["state"] == "committed"
    assert read_group(cfg.shared_root, "committed")["group"]["name"] == "committed"

    from qqtools.plugins.qexp.observer import list_groups_page

    for history in (False, True):
        assert [item["group"]["name"] for item in list_groups_page(cfg, all_groups=history)["items"]] == ["committed"]

    monkeypatch.setattr(submission_runtime, "finalize_submission_group", original_finalize)
    recovered = batch_submit(cfg, manifest, idempotency_key="finish")
    assert recovered.state == "committed"
    assert read_group(cfg.shared_root, "committed")["group"]["pending_submission_commit"] is None
