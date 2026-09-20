from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.ci import promotion_provenance

SOURCE_SHA = "a" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _repository(tmp_path: Path, trailers: str | None = None) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "dev")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "commit", "--allow-empty", "-m", "base")
    parent = _git(repo, "rev-parse", "HEAD")
    message = (
        "promoted change\n\n"
        "Promoted-from: feat/example\n"
        f"Promotion-source: {SOURCE_SHA}\n"
        "Promotion-run: 123\n"
        "Promotion-attempt: 2\n"
        "Promotion-provenance: github-attestation-v1"
    )
    _git(repo, "commit", "--allow-empty", "-m", trailers or message)
    return repo, parent, _git(repo, "rev-parse", "HEAD")


def test_manifest_binds_exact_commit_tree_parent_and_workflow_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, parent, commit = _repository(tmp_path)
    monkeypatch.chdir(repo)

    provenance = promotion_provenance.provenance_for_commit(commit, "kzhoa/qqtools", parent)

    assert provenance.target_commit == commit
    assert provenance.target_tree == _git(repo, "rev-parse", f"{commit}^{{tree}}")
    assert provenance.target_parent == parent
    assert provenance.source_commit == SOURCE_SHA
    assert provenance.source_ref == "refs/heads/feat/example"
    assert provenance.workflow_run == 123
    assert provenance.workflow_attempt == 2
    serialized = json.dumps(asdict(provenance), sort_keys=True, separators=(",", ":")) + "\n"
    assert json.loads(serialized)["schema"] == promotion_provenance.SCHEMA


@pytest.mark.parametrize(
    ("trailers", "error"),
    [
        (
            "promoted change\n\n"
            f"Promotion-source: {SOURCE_SHA}\n"
            "Promotion-run: 123\n"
            "Promotion-attempt: 1\n"
            "Promotion-provenance: github-attestation-v1",
            "Promoted-from",
        ),
        (
            "promoted change\n\n"
            "Promoted-from: bug/example\n"
            f"Promotion-source: {SOURCE_SHA}\n"
            "Promotion-run: 123\n"
            "Promotion-attempt: 1\n"
            "Promotion-provenance: github-attestation-v1",
            r"feat/\* or feature/\*",
        ),
        (
            "promoted change\n\n"
            "Promoted-from: feat/example\n"
            "Promoted-from: feat/duplicate\n"
            f"Promotion-source: {SOURCE_SHA}\n"
            "Promotion-run: 123\n"
            "Promotion-attempt: 1\n"
            "Promotion-provenance: github-attestation-v1",
            "exactly one Promoted-from",
        ),
    ],
)
def test_invalid_or_ambiguous_trailers_cannot_claim_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trailers: str, error: str
) -> None:
    repo, parent, commit = _repository(tmp_path, trailers)
    monkeypatch.chdir(repo)

    with pytest.raises(ValueError, match=error):
        promotion_provenance.provenance_for_commit(commit, "kzhoa/qqtools", parent)


def test_previous_dev_must_be_the_promoted_commits_only_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _, commit = _repository(tmp_path)
    monkeypatch.chdir(repo)

    with pytest.raises(ValueError, match="previous dev"):
        promotion_provenance.provenance_for_commit(commit, "kzhoa/qqtools", "b" * 40)
