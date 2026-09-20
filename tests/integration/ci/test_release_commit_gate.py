from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.checks import check_release_commit


def test_release_checker_supports_documented_direct_execution():
    result = subprocess.run(
        [sys.executable, "scripts/checks/check_release_commit.py", "--help"],
        cwd=check_release_commit.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_release_files(root: Path, version: str, notes: str) -> None:
    version_path = root / "src/qqtools/version.py"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        f"# History\n\n## Unreleased\n\n## v{version}\n\n- {notes}\n",
        encoding="utf-8",
    )


def _release_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "dev")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _write_release_files(root, "1.3.18", "previous release")
    for relative in (
        "src/qqtools/__init__.py",
        "src/qqtools/__init__.pyi",
        "src/qqtools/plugins/qpipeline/__init__.py",
        "src/qqtools/plugins/qpipeline/__init__.pyi",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    registry = root / "docs/spec/compatibility-registry.toml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("next_id = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    return root, _git(root, "rev-parse", "HEAD")


def _classify(monkeypatch: pytest.MonkeyPatch, root: Path, base: str, head: str) -> dict[str, str]:
    output = root / "github-output"
    monkeypatch.setattr(check_release_commit, "REPO_ROOT", root)
    assert (
        check_release_commit.main(
            [
                "classify",
                "--base-ref",
                base,
                "--head-ref",
                head,
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    return dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())


def test_valid_version_bump_classifies_and_validates_as_release(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    _write_release_files(root, "1.3.19", "new fixes")
    _git(root, "add", "CHANGELOG.md", "src/qqtools/version.py")
    _git(root, "commit", "-m", "release: prepare v1.3.19")
    head = _git(root, "rev-parse", "HEAD")

    assert _classify(monkeypatch, root, base, head) == {"profile": "release", "base_ref": base}
    assert check_release_commit.main(["validate", "--base-ref", base, "--head-ref", head, "--actor", "kzhoa"]) == 0


def test_metadata_only_malformed_candidate_cannot_fall_back_to_feature(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    changelog = root / "CHANGELOG.md"
    changelog.write_text(changelog.read_text(encoding="utf-8") + "\nextra\n", encoding="utf-8")
    _git(root, "add", "CHANGELOG.md")
    _git(root, "commit", "-m", "edit changelog")
    head = _git(root, "rev-parse", "HEAD")

    assert _classify(monkeypatch, root, base, head)["profile"] == "release"
    with pytest.raises((RuntimeError, ValueError, SystemExit)):
        check_release_commit.main(["validate", "--base-ref", base, "--head-ref", head, "--actor", "kzhoa"])


def test_later_valid_bump_cannot_hide_malformed_commit_in_batched_push(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    changelog = root / "CHANGELOG.md"
    changelog.write_text(changelog.read_text(encoding="utf-8") + "\nintermediate edit\n", encoding="utf-8")
    _git(root, "add", "CHANGELOG.md")
    _git(root, "commit", "-m", "malformed release metadata")

    _write_release_files(root, "1.3.19", "new fixes")
    _git(root, "add", "CHANGELOG.md", "src/qqtools/version.py")
    _git(root, "commit", "-m", "release: prepare v1.3.19")
    head = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(check_release_commit, "REPO_ROOT", root)

    assert _classify(monkeypatch, root, base, head)["profile"] == "release"
    with pytest.raises(check_release_commit.ReleaseCommitError, match="expected a correction"):
        check_release_commit.main(["validate", "--base-ref", base, "--head-ref", head, "--actor", "kzhoa"])


def test_release_validation_rejects_empty_changelog_section(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    (root / "src/qqtools/version.py").write_text('__version__ = "1.3.19"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        "# History\n\n## Unreleased\n\n## v1.3.19\n\n## v1.3.18\n\n- previous release\n",
        encoding="utf-8",
    )
    _git(root, "add", "CHANGELOG.md", "src/qqtools/version.py")
    _git(root, "commit", "-m", "release: prepare v1.3.19")
    head = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(check_release_commit, "REPO_ROOT", root)

    with pytest.raises(check_release_commit.ReleaseCommitError, match="must contain release notes"):
        check_release_commit.main(["validate", "--base-ref", base, "--head-ref", head, "--actor", "kzhoa"])


def test_commit_touching_non_release_path_uses_feature_profile(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    _write_release_files(root, "1.3.19", "new fixes")
    extra = root / "src/qqtools/extra.py"
    extra.write_text("value = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "mixed release and code")
    head = _git(root, "rev-parse", "HEAD")

    assert _classify(monkeypatch, root, base, head)["profile"] == "feature"


def test_metadata_correction_keeps_release_profile(monkeypatch, tmp_path):
    root, original = _release_repo(tmp_path)
    _write_release_files(root, "1.3.19", "new fixes")
    _git(root, "add", "CHANGELOG.md", "src/qqtools/version.py")
    _git(root, "commit", "-m", "release: prepare v1.3.19")
    release = _git(root, "rev-parse", "HEAD")
    changelog = root / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text(encoding="utf-8").replace("new fixes", "corrected fixes"), encoding="utf-8"
    )
    _git(root, "add", "CHANGELOG.md")
    _git(root, "commit", "-m", "release: correct v1.3.19 notes")
    correction = _git(root, "rev-parse", "HEAD")

    assert _classify(monkeypatch, root, release, correction)["profile"] == "release"
    assert (
        check_release_commit.main(["validate", "--base-ref", release, "--head-ref", correction, "--actor", "kzhoa"])
        == 0
    )
    assert _git(root, "rev-parse", f"{release}^") == original


def test_release_validation_requires_owner(monkeypatch, tmp_path):
    root, base = _release_repo(tmp_path)
    _write_release_files(root, "1.3.19", "new fixes")
    _git(root, "add", "CHANGELOG.md", "src/qqtools/version.py")
    _git(root, "commit", "-m", "release: prepare v1.3.19")
    head = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(check_release_commit, "REPO_ROOT", root)

    with pytest.raises((RuntimeError, ValueError, SystemExit)):
        check_release_commit.main(["validate", "--base-ref", base, "--head-ref", head, "--actor", "other"])
