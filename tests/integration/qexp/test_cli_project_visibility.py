from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _runtime_args(tmp_path: Path) -> list[str]:
    return ["--machine-runtime-root", str(tmp_path / "machine-runtime")]


_ORDINARY_PROJECT_LEAVES = [
    ["submit", "--format=json", "--no-activate", "--", "echo", "ok"],
    ["status", "--format=json"],
    ["task", "attach", "task-1"],
    ["task", "cancel", "task-1", "--format=json"],
    ["task", "retry", "task-1", "--format=json"],
    ["task", "share", "task-1", "--format=json"],
    ["task", "unshare", "task-1", "--format=json"],
    ["task", "offer", "task-1", "--format=json"],
    ["task", "list", "--format=json"],
    ["task", "show", "task-1", "--format=json"],
    ["task", "logs", "task-1"],
    ["task", "wait", "task-1", "--format=json"],
    ["task", "dependencies", "show", "task-1", "--format=json"],
    ["task", "dependencies", "replace", "task-1", "--depends-on", "task-0", "--format=json"],
    ["task", "dependencies", "add", "task-1", "--depends-on", "task-0", "--format=json"],
    ["task", "dependencies", "remove", "task-1", "--depends-on", "task-0", "--format=json"],
    ["group", "create", "demo", "--format=json"],
    ["group", "list", "--format=json"],
    ["group", "show", "demo", "--format=json"],
    ["group", "seal", "demo", "--format=json"],
    ["group", "reopen", "demo", "--format=json"],
    ["group", "pause", "demo", "--format=json"],
    ["group", "resume", "demo", "--format=json"],
    ["group", "cancel", "demo", "--format=json"],
    ["group", "retry", "demo", "--format=json"],
    ["group", "worker", "list", "demo", "--format=json"],
    ["group", "worker", "add", "demo", "gpu-2", "--format=json"],
    ["group", "worker", "set", "demo", "gpu-2", "--format=json"],
    ["group", "worker", "drain", "demo", "gpu-2", "--format=json"],
    ["group", "worker", "resume", "demo", "gpu-2", "--format=json"],
    ["group", "worker", "remove", "demo", "gpu-2", "--format=json"],
    ["group", "config", "show", "demo", "progress", "--format=json"],
    ["group", "config", "set", "demo", "progress", "--live-progress", "--format=json"],
    ["machine", "list", "--format=json"],
    ["machine", "show", "gpu-1", "--format=json"],
]


_CONDITIONAL_PROJECT_LEAVES = [
    ["config", "show", "progress", "--format=json"],
    ["config", "set", "progress", "--interval-seconds", "30", "--format=json"],
    ["config", "reset", "progress", "--format=json"],
    ["notifications", "setup", "--scope", "project", "--format=json"],
    ["notifications", "show", "--scope", "project", "--format=json"],
    ["notifications", "test", "--scope", "project", "--format=json"],
    ["notifications", "set", "--scope", "project", "--enabled", "--format=json"],
    ["notifications", "reset", "--scope", "project", "--format=json"],
    ["notifications", "resolve", "--scope", "project", "--prefer", "canonical", "--format=json"],
]


@pytest.mark.parametrize("argv", [*_ORDINARY_PROJECT_LEAVES, *_CONDITIONAL_PROJECT_LEAVES])
def test_every_implicit_single_project_dispatch_leaf_routes_one_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
    outcome = SimpleNamespace(exit_code=0)
    observed_before_dispatch: list[str] = []

    def dispatch(*_args, **_kwargs):
        observed_before_dispatch.append(capsys.readouterr().err)
        return outcome

    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_project", dispatch)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_submission", dispatch)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_notifications", dispatch)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint._finalize_outcome",
        lambda _args, selected_outcome, **_kwargs: selected_outcome.exit_code,
    )

    assert main([*_runtime_args(tmp_path), *argv]) == 0

    assert observed_before_dispatch == [f"Project: {cfg.project_root} (from $QEXP_SHARED_ROOT)\n"]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["notifications", "show", "--scope", "global", "--format=json"],
        ["notifications", "set", "--scope", "global", "--enabled", "--format=json"],
        ["config", "show", "agent", "--format=json"],
        ["config", "set", "agent", "--agent-mode", "daemon", "--format=json"],
        ["project", "list", "--format=json"],
        ["agent", "status", "--format=json"],
    ],
)
def test_global_and_machine_dispatch_alternatives_do_not_present_a_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
    outcome = SimpleNamespace(exit_code=0)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_notifications", lambda *_args, **_kwargs: outcome)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint.dispatch_config_notifications", lambda *_args, **_kwargs: outcome
    )
    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_local", lambda *_args, **_kwargs: outcome)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint._finalize_outcome",
        lambda _args, selected_outcome: selected_outcome.exit_code,
    )

    assert main([*_runtime_args(tmp_path), *argv]) == 0

    assert "Project:" not in capsys.readouterr().err


def test_implicit_json_read_keeps_stdout_structured_and_reports_environment_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd_cfg = init_shared_root(tmp_path / "cwd" / ".qexp", "gpu-1", runtime_root=tmp_path / "cwd-rt")
    env_cfg = init_shared_root(tmp_path / "environment" / ".qexp", "gpu-1", runtime_root=tmp_path / "env-rt")
    monkeypatch.chdir(cwd_cfg.project_root)
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(env_cfg.shared_root))

    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0

    output = capsys.readouterr()
    assert json.loads(output.out) == []
    assert output.err == f"Project: {env_cfg.project_root} (from $QEXP_SHARED_ROOT)\n"


def test_cwd_and_parent_discovery_have_distinct_notices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.chdir(cfg.project_root)
    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0
    assert capsys.readouterr().err == f"Project: {cfg.project_root}\n"

    nested = cfg.project_root / "runs" / "one"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0
    assert capsys.readouterr().err == f"Project: {cfg.project_root} (from parent directory)\n"


def test_saved_context_notice_uses_the_actual_context_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    context_path = Path.home() / ".qqtools" / "visibility-test-context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps({"shared_root": str(cfg.shared_root)}), encoding="utf-8")
    try:
        assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0
    finally:
        context_path.unlink(missing_ok=True)

    assert capsys.readouterr().err == (f"Project: {cfg.project_root} (from ~/.qqtools/visibility-test-context.json)\n")


def test_explicit_project_suppresses_the_default_notice(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    assert main([*_runtime_args(tmp_path), "--project", str(cfg.project_root), "task", "list"]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert "No Tasks" in output.out


def test_human_status_integrates_implicit_source_while_json_uses_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    nested = cfg.project_root / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)

    assert main([*_runtime_args(tmp_path), "status"]) == 0
    human = capsys.readouterr()
    assert human.err == ""
    assert f"Project: {cfg.project_root} (from parent directory)" in human.out
    assert "Selection source:" not in human.out

    assert main([*_runtime_args(tmp_path), "status", "--format=json"]) == 0
    structured = capsys.readouterr()
    assert json.loads(structured.out)["project"]["path"] == str(cfg.project_root)
    assert structured.err == f"Project: {cfg.project_root} (from parent directory)\n"


@pytest.mark.parametrize("explicit", [False, True])
def test_human_status_failure_after_selection_uses_one_stderr_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit: bool,
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
    argv = [*_runtime_args(tmp_path)]
    if explicit:
        argv.extend(("--project", str(cfg.project_root)))
    argv.append("status")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint.dispatch_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("status render failed")),
    )

    with pytest.raises(RuntimeError, match="status render failed"):
        main(argv)

    expected = f"Project: {cfg.project_root}"
    if not explicit:
        expected += " (from $QEXP_SHARED_ROOT)"
    assert capsys.readouterr().err == f"{expected}\n"


def test_submission_reports_selection_before_later_input_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.chdir(cfg.project_root)

    assert main([*_runtime_args(tmp_path), "submit", "--gpus", "-1", "--", "echo", "no"]) != 0

    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"Project: {cfg.project_root}"
    assert any("gpu" in line.lower() for line in lines[1:])


def test_file_submission_reports_manifest_ancestor_instead_of_invocation_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd_cfg = init_shared_root(tmp_path / "cwd" / ".qexp", "gpu-1", runtime_root=tmp_path / "cwd-rt")
    manifest_cfg = init_shared_root(
        tmp_path / "manifest-project" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "manifest-rt",
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cwd_cfg.shared_root, cwd_cfg.machine_name)
    runtime.ensure_binding(manifest_cfg.shared_root, manifest_cfg.machine_name)
    manifest = manifest_cfg.project_root / "inputs" / "runs.yaml"
    manifest.parent.mkdir()
    manifest.write_text("tasks:\n  - command: [echo, ok]\n", encoding="utf-8")
    context_path = tmp_path / "missing-context.json"
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    monkeypatch.chdir(cwd_cfg.project_root)
    outcome = SimpleNamespace(exit_code=0)
    observed: list[str] = []

    def dispatch(*_args, **_kwargs):
        observed.append(capsys.readouterr().err)
        return outcome

    monkeypatch.setattr("qqtools.plugins.qexp.cli.entrypoint.dispatch_submission", dispatch)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.entrypoint._finalize_outcome",
        lambda _args, selected_outcome: selected_outcome.exit_code,
    )

    assert main([*_runtime_args(tmp_path), "submit", "--file", str(manifest), "--no-activate"]) == 0

    assert observed == [f"Project: {manifest_cfg.project_root} (from manifest directory)\n"]


def test_corrupt_saved_context_still_preempts_submission_discovery_but_not_ordinary_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    context_path = tmp_path / "broken-context.json"
    context_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.layout._CONTEXT_PATH", context_path)
    monkeypatch.chdir(cfg.project_root)

    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0
    ordinary = capsys.readouterr()
    assert ordinary.err == f"Project: {cfg.project_root}\n"

    assert main([*_runtime_args(tmp_path), "submit", "--no-activate", "--", "echo", "no"]) != 0
    submission = capsys.readouterr()
    assert "Project:" not in submission.err
    assert str(context_path) in submission.err


def test_empty_environment_retains_distinct_ordinary_and_submission_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.chdir(cfg.project_root)
    monkeypatch.setenv("QEXP_SHARED_ROOT", "")

    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0
    assert capsys.readouterr().err == f"Project: {cfg.project_root}\n"

    assert main([*_runtime_args(tmp_path), "submit", "--no-activate", "--quiet", "--", "echo", "ok"]) == 0
    submitted = capsys.readouterr()
    assert submitted.out.strip()
    assert submitted.out.count("\n") == 1
    assert submitted.err.splitlines()[0] == f"Project: {cfg.project_root} (from $QEXP_SHARED_ROOT)"
    assert submitted.err.count(f"Project: {cfg.project_root}") == 1


def test_project_notice_encodes_controls_without_adding_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project\\with\ncontrols\t\u200b"
    cfg = init_shared_root(project / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.project_root))

    assert main([*_runtime_args(tmp_path), "task", "list", "--format=json"]) == 0

    notice = capsys.readouterr().err
    encoded_project = (
        str(cfg.project_root)
        .replace("\\", "\\\\")
        .replace("\n", r"\n")
        .replace("\t", r"\t")
        .replace("\u200b", r"\u200b")
    )
    assert notice.count("\n") == 1
    assert notice == f"Project: {encoded_project} (from $QEXP_SHARED_ROOT)\n"
