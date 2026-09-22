from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.cli.entrypoint import main

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _status(args: list[str], capsys) -> dict:
    assert main([*args, "--format=json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_explicit_project_normalizes_directory_and_common_options_work_after_path(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    cfg = init_shared_root(project / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    result = _status(["status", "--project", str(project)], capsys)

    assert result["project"]["path"] == str(project)
    assert result["project"]["selection_source"] == "explicit"
    assert cfg.shared_root == project / ".qexp"


def test_project_selection_precedence_is_explicit_environment_cwd_then_saved(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    explicit = init_shared_root(tmp_path / "explicit" / ".qexp", "gpu-1", runtime_root=tmp_path / "r1")
    environment = init_shared_root(tmp_path / "environment" / ".qexp", "gpu-1", runtime_root=tmp_path / "r2")
    cwd = init_shared_root(tmp_path / "cwd" / ".qexp", "gpu-1", runtime_root=tmp_path / "r3")
    saved = init_shared_root(tmp_path / "saved" / ".qexp", "gpu-1", runtime_root=tmp_path / "r4")
    context_path = tmp_path / "context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    assert main(["use", "--project", str(saved.shared_root)]) == 0
    capsys.readouterr()
    nested = cwd.project_root / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(environment.project_root))

    selected = _status(["--project", str(explicit.project_root), "status"], capsys)
    assert selected["project"]["path"] == str(explicit.project_root)
    assert selected["project"]["selection_source"] == "explicit"
    environment_selected = _status(["status"], capsys)
    assert environment_selected["project"]["path"] == str(environment.project_root)
    assert environment_selected["project"]["selection_source"] == "environment"
    monkeypatch.delenv("QEXP_SHARED_ROOT")
    cwd_selected = _status(["status"], capsys)
    assert cwd_selected["project"]["path"] == str(cwd.project_root)
    assert cwd_selected["project"]["selection_source"] == "cwd"
    monkeypatch.chdir(tmp_path)
    saved_selected = _status(["status"], capsys)
    assert saved_selected["project"]["path"] == str(saved.project_root)
    assert saved_selected["project"]["selection_source"] == "saved"


def test_malformed_nearest_project_does_not_fall_through_to_ancestor(tmp_path: Path, monkeypatch, capsys) -> None:
    init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    child = tmp_path / "child"
    malformed = child / ".qexp"
    malformed.mkdir(parents=True)
    monkeypatch.chdir(child)

    assert main(["status", "--format=json"]) != 0
    output = capsys.readouterr()
    error = json.loads(output.out)["error"]
    assert not output.err
    assert error["code"] == "invalid_argument"
    assert str(malformed) in error["message"]
    assert "uninitialized" in error["message"]


def test_equal_common_duplicates_normalize_and_conflicting_duplicates_fail(tmp_path: Path, capsys) -> None:
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "r1")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "r2")

    assert (
        main(
            [
                "--project",
                str(first.project_root),
                "status",
                "--project",
                str(first.shared_root),
                "--format=json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--project",
                str(first.project_root),
                "status",
                "--project",
                str(second.project_root),
                "--format=json",
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert "conflict" in (output.out + output.err).lower()


@pytest.mark.parametrize(
    ("section", "options"),
    [
        ("lease", ["--ttl-seconds", "90"]),
        ("notifications", ["--enabled"]),
        ("progress", ["--interval-seconds", "45"]),
        ("tmux", ["--enabled"]),
        ("launch-handoff", ["--timeout-seconds", "20"]),
    ],
)
@pytest.mark.parametrize("action", ["set", "reset"])
def test_project_config_mutations_require_verified_binding(
    tmp_path: Path, capsys, section: str, options: list[str], action: str
) -> None:
    cfg = init_shared_root(tmp_path / section / ".qexp", "gpu-1", runtime_root=tmp_path / section / "runtime")
    machine_runtime = tmp_path / section / f"unbound-{action}"
    argv = [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--machine-runtime-root",
        str(machine_runtime),
        "config",
        action,
        section,
        "--format=json",
    ]
    if action == "set":
        argv.extend(options)

    assert main(argv) == 2
    output = capsys.readouterr()
    assert not output.err
    error = json.loads(output.out)["error"]
    assert error["code"] == "invalid_argument"
    assert "qexp project register" in error["message"]


def test_agent_config_reset_rejects_without_resolving_a_project(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["config", "reset", "agent", "--format=json"]) == 2

    output = capsys.readouterr()
    assert not output.err
    error = json.loads(output.out)["error"]
    assert error["code"] == "invalid_argument"
    assert "cannot be reset" in error["message"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--machine-runtime-root", "/tmp/ignored", "use", "--show"],
        ["use", "--show", "--machine-runtime-root", "/tmp/ignored"],
    ],
)
def test_use_rejects_machine_runtime_scope_options(argv: list[str], capsys) -> None:
    assert main(argv) == 2
    assert "accepts only --project, --show, or --clear" in capsys.readouterr().err


@pytest.mark.parametrize("action", ["check", "repair", "clean"])
@pytest.mark.parametrize("source", ["environment", "cwd", "saved"])
def test_admin_maintenance_requires_explicit_project_even_with_fallback_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    action: str,
    source: str,
) -> None:
    cfg = init_shared_root(tmp_path / source / ".qexp", "gpu-1", runtime_root=tmp_path / source / "runtime")
    if source == "environment":
        monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
    elif source == "cwd":
        monkeypatch.chdir(cfg.project_root)
    else:
        context_path = tmp_path / "context.json"
        monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
        assert main(["use", "--project", str(cfg.shared_root)]) == 0
        capsys.readouterr()

    assert main(["admin", action, "--format=json"]) == 2
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out)["error"]["message"] == f"admin {action} requires explicit --project PATH."


@pytest.mark.parametrize(
    "argv", [["task", "wait", "--format=json"], ["task", "wait", "task-1", "--timeout=bad", "--format=json"]]
)
def test_task_wait_json_input_errors_keep_fixed_schema(argv: list[str], capsys) -> None:
    assert main(argv) == 2

    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    assert set(result) == {
        "schema_version",
        "task_id",
        "project",
        "selected_attempt_number",
        "selected_attempt_id",
        "outcome",
        "reason",
        "task_exit_code",
        "error",
    }
    assert result["outcome"] == "invalid_input"
    assert result["error"]["code"] == "invalid_input"
