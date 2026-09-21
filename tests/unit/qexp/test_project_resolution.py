from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp.project_resolution import resolve_submission_project


def _initialized_project(path: Path) -> Path:
    control = path / ".qexp"
    (control / "schema").mkdir(parents=True)
    (control / "schema" / "version.json").write_text("{}", encoding="utf-8")
    return path


def test_command_resolution_prefers_environment_before_cwd_and_saved(tmp_path: Path) -> None:
    environment = _initialized_project(tmp_path / "environment")
    cwd_project = _initialized_project(tmp_path / "cwd")
    saved = _initialized_project(tmp_path / "saved")

    selected = resolve_submission_project(
        explicit_project=None,
        manifest_path=None,
        invocation_cwd=cwd_project / "nested",
        environment_value=str(environment / ".qexp"),
        saved_context={"shared_root": str(saved / ".qexp")},
    )

    assert selected.path == environment.resolve()
    assert selected.control_root == (environment / ".qexp").resolve()
    assert selected.source == "environment"


def test_file_resolution_prefers_manifest_ancestry_before_cwd(tmp_path: Path) -> None:
    manifest_project = _initialized_project(tmp_path / "manifest")
    cwd_project = _initialized_project(tmp_path / "cwd")
    manifest = manifest_project / "inputs" / "runs.yaml"
    manifest.parent.mkdir()
    manifest.write_text("tasks: []\n", encoding="utf-8")

    selected = resolve_submission_project(
        explicit_project=None,
        manifest_path=manifest,
        invocation_cwd=cwd_project,
        environment_value=None,
        saved_context=None,
    )

    assert selected.path == manifest_project.resolve()
    assert selected.source == "manifest_ancestor"


def test_explicit_malformed_project_does_not_fall_through(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed"
    (malformed / ".qexp").mkdir(parents=True)
    cwd_project = _initialized_project(tmp_path / "cwd")

    with pytest.raises(ValueError, match="initialized"):
        resolve_submission_project(
            explicit_project=malformed,
            manifest_path=None,
            invocation_cwd=cwd_project,
            environment_value=None,
            saved_context=None,
        )


def test_resolution_is_read_only_when_no_candidate_exists(tmp_path: Path) -> None:
    cwd = tmp_path / "empty" / "nested"
    cwd.mkdir(parents=True)

    with pytest.raises(ValueError, match="Project"):
        resolve_submission_project(
            explicit_project=None,
            manifest_path=None,
            invocation_cwd=cwd,
            environment_value=None,
            saved_context=None,
        )

    assert not list(tmp_path.rglob(".qexp"))
