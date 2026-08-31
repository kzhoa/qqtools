from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.commands.task import submit as submit_task
from qqtools.plugins.qexp.layout import load_context
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _setup_project(
    tmp_path: Path, machines: tuple[str, ...] = ("g3", "g4")
) -> tuple[Path, MachineRuntime]:
    shared_root = tmp_path / ".qexp"
    for machine in machines:
        init_shared_root(shared_root, machine, runtime_root=tmp_path / f"{machine}-legacy")
    runtime = MachineRuntime(tmp_path / "g3-machine-runtime")
    runtime.add_binding(shared_root, "g3")
    return shared_root, runtime


def _args(shared_root: Path, runtime: MachineRuntime, *values: str) -> list[str]:
    return [
        "--shared-root",
        str(shared_root),
        "--machine-runtime-root",
        str(runtime.root),
        *values,
    ]


def _submit_remote_task(
    shared_root: Path, runtime: MachineRuntime, capsys: pytest.CaptureFixture[str]
) -> str:
    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--home-machine",
            "g4",
            "--",
            "echo",
            "ok",
        )
    ) == 0
    return capsys.readouterr().out.strip()


def test_submit_resolves_local_identity_and_supports_remote_private_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)

    task_id = _submit_remote_task(shared_root, runtime, capsys)
    task = load_task(runtime.verified_execution_context(shared_root).cfg, task_id)
    operation = read_json(submission_path(shared_root, task.submission_operation_id))["submission"]

    assert task.group_name is None
    assert task.placement_policy["home_machine"] == "g4"
    assert task.placement_policy["sharing_mode"] == "private"
    assert operation["original_submitting_machine"] == "g3"
    ready = next((shared_root / "indexes" / "ready" / "home").rglob(f"{task_id}.*.json"))
    assert read_json(ready)["ready_marker"]["home_machine"] == "g4"


def test_remote_private_home_is_not_claimable_by_origin_but_is_claimable_by_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, g3_runtime = _setup_project(tmp_path)
    task_id = _submit_remote_task(shared_root, g3_runtime, capsys)
    g3_context = g3_runtime.verified_execution_context(shared_root)

    assert claim_task(
        g3_context.cfg,
        task_id,
        [0],
        reservation_runtime_root=g3_runtime.root,
        project_id=g3_context.project_id,
    ) is None

    g4_runtime = MachineRuntime(tmp_path / "g4-machine-runtime")
    g4_runtime.add_binding(shared_root, "g4")
    g4_context = g4_runtime.verified_execution_context(shared_root)
    attempt = claim_task(
        g4_context.cfg,
        task_id,
        [0],
        reservation_runtime_root=g4_runtime.root,
        project_id=g4_context.project_id,
    )
    assert attempt is not None
    shown = inspect_task(g3_context.cfg, task_id)
    assert shown["submission"]["original_submitting_machine"] == "g3"
    assert shown["task"]["placement_policy"]["home_machine"] == "g4"
    assert shown["attempts"][0]["attempt"]["machine_name"] == "g4"


def test_remote_home_requires_a_current_generation_machine_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path, machines=("g3",))

    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--home-machine",
            "g4",
            "--",
            "echo",
            "missing",
        )
    ) == 2
    assert "no current-generation Project machine record" in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))


def test_empty_home_machine_is_rejected_instead_of_defaulting_to_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)

    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--home-machine",
            "",
            "--",
            "echo",
            "invalid",
        )
    ) == 2
    assert "home_machine must contain only" in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))


@pytest.mark.parametrize("field", ["project_id", "shared_root", "agent_runtime"])
def test_remote_home_rejects_inconsistent_machine_record(
    tmp_path: Path, field: str, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    record_path = shared_root / "machines" / "g4" / "machine.json"
    record = read_json(record_path)
    if field == "project_id":
        record["machine"][field] = "wrong-project"
    elif field == "shared_root":
        record["machine"][field] = str(tmp_path / "other" / ".qexp")
    else:
        record["machine"].pop(field)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--home-machine",
            "g4",
            "--",
            "echo",
            "invalid",
        )
    ) == 2
    assert "current-generation Project machine record" in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))


def test_remote_home_does_not_activate_the_target_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    calls: list[tuple[str, Path]] = []

    def activate(cfg, *, reason, machine_runtime):
        calls.append((cfg.machine_name, machine_runtime.root))
        return True

    monkeypatch.setattr("qqtools.plugins.qexp.cli.ensure_local_agent_active", activate)
    assert main(
        _args(shared_root, runtime, "submit", "--home-machine", "g4", "--", "echo", "ok")
    ) == 0
    capsys.readouterr()
    assert calls == [("g3", runtime.root)]
    assert not (tmp_path / "g4-machine-runtime").exists()


def test_environment_machine_assertion_must_match_local_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    monkeypatch.setenv("QEXP_MACHINE", "g4")

    assert main(_args(shared_root, runtime, "submit", "--no-activate", "--", "echo", "bad")) == 2
    assert "Local project binding is 'g3'" in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["submit", "--help"]],
)
def test_help_separates_identity_and_home_placement(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "local identity" in output
    assert "home-machine" in output
    assert "remotely start" in output


def test_machine_assertion_conflict_fails_before_mutation_even_without_activation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)

    assert main(
        _args(
            shared_root,
            runtime,
            "--machine",
            "g4",
            "submit",
            "--no-activate",
            "--",
            "echo",
            "bad",
        )
    ) == 2
    assert "Local project binding is 'g3', but --machine asserted 'g4'." in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))
    assert not list((shared_root / "operations" / "submissions").glob("*.json"))


def test_flag_and_environment_machine_assertions_cannot_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    monkeypatch.setenv("QEXP_MACHINE", "g4")

    assert main(
        _args(
            shared_root,
            runtime,
            "--machine",
            "g3",
            "submit",
            "--no-activate",
            "--",
            "echo",
            "bad",
        )
    ) == 2
    assert "conflicts with QEXP_MACHINE" in capsys.readouterr().err
    assert not list((shared_root / "tasks").glob("*.json"))


def test_saved_machine_and_legacy_runtime_do_not_override_verified_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    context_path = tmp_path / "saved-context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    context_path.write_text(
        json.dumps(
            {
                "shared_root": str(shared_root),
                "machine": "g4",
                "runtime_root": str(tmp_path / "wrong-runtime"),
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--machine-runtime-root",
            str(runtime.root),
            "--runtime-root",
            str(tmp_path / "another-wrong-runtime"),
            "submit",
            "--no-activate",
            "--",
            "echo",
            "ok",
        ]
    ) == 0
    task_id = capsys.readouterr().out.strip()
    context = runtime.verified_execution_context(shared_root)
    assert load_task(context.cfg, task_id).placement_policy["home_machine"] == "g3"
    assert context.local_cfg.runtime_root == runtime.project_paths(context.project_id)["root"]
    assert not (tmp_path / "wrong-runtime").exists()
    assert load_context() == {
        "shared_root": str(shared_root),
        "machine": "g4",
        "runtime_root": str(tmp_path / "wrong-runtime"),
    }


def test_single_submit_does_not_create_missing_group_or_add_origin_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)

    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--group",
            "missing",
            "--home-machine",
            "g4",
            "--",
            "echo",
            "bad",
        )
    ) == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (shared_root / "groups" / "missing.json").exists()
    assert not list((shared_root / "tasks").glob("*.json"))

    assert main(_args(shared_root, runtime, "group", "create", "exp", "--workers", "g4")) == 0
    capsys.readouterr()
    assert main(
        _args(
            shared_root,
            runtime,
            "submit",
            "--no-activate",
            "--group",
            "exp",
            "--home-machine",
            "g4",
            "--sharing",
            "spillover",
            "--",
            "echo",
            "ok",
        )
    ) == 0
    capsys.readouterr()
    assert set(read_json(group_path(shared_root, "exp"))["group"]["worker_set"]) == {"g4"}


def test_group_create_without_workers_defaults_current_and_explicit_workers_are_exact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    assert main(_args(shared_root, runtime, "group", "create", "current-workers")) == 0
    capsys.readouterr()
    assert set(read_json(group_path(shared_root, "current-workers"))["group"]["worker_set"]) == {"g3"}

    assert main(_args(shared_root, runtime, "group", "create", "remote-workers", "--workers", "g4")) == 0
    capsys.readouterr()
    assert set(read_json(group_path(shared_root, "remote-workers"))["group"]["worker_set"]) == {"g4"}


def test_batch_manifest_can_atomically_create_exact_worker_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, runtime = _setup_project(tmp_path)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  workers: [g4]\ntasks:\n  - placement:\n      home_machine: g4\n    command: [echo, ok]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("qqtools.plugins.qexp.cli.ensure_local_agent_active", lambda *args, **kwargs: True)

    assert main(
        _args(shared_root, runtime, "batch-submit", "--file", str(manifest), "--group", "exp", "--format=json")
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "committed"
    assert set(read_json(group_path(shared_root, "exp"))["group"]["worker_set"]) == {"g4"}


def test_cross_machine_retry_reuses_first_verified_submission_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared_root, g3_runtime = _setup_project(tmp_path)
    first_id = _submit_remote_task(shared_root, g3_runtime, capsys)
    first_task = load_task(g3_runtime.verified_execution_context(shared_root).cfg, first_id)
    operation_id = first_task.submission_operation_id

    init_shared_root(shared_root, "g2", runtime_root=tmp_path / "g2-legacy")
    g2_runtime = MachineRuntime(tmp_path / "g2-machine-runtime")
    g2_runtime.add_binding(shared_root, "g2")
    second = submit_task(
        g2_runtime.verified_execution_context(shared_root).cfg,
        ["echo", "ok"],
        home_machine="g4",
        idempotency_key=read_json(submission_path(shared_root, operation_id))["submission"]["idempotency_key"],
    )
    assert second.task_id == first_id
    operation = read_json(submission_path(shared_root, operation_id))["submission"]
    assert operation["original_submitting_machine"] == "g3"
    assert second.placement_policy["home_machine"] == "g4"
