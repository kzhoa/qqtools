"""Execute promotion credential guards without network access or real secrets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("job_name", ["promote-feature", "promote-dev-to-main"])
@pytest.mark.parametrize(
    ("actor", "rerunner", "token", "login", "api_status", "expected_success", "calls_api"),
    [
        ("kzhoa", "kzhoa", "test-only", "kzhoa", 0, True, True),
        ("other", "kzhoa", "test-only", "kzhoa", 0, False, False),
        ("kzhoa", "other", "test-only", "kzhoa", 0, False, False),
        ("kzhoa", "kzhoa", "", "kzhoa", 0, False, False),
        ("kzhoa", "kzhoa", "test-only", "other", 0, False, True),
        ("kzhoa", "kzhoa", "test-only", "", 1, False, True),
        ("kzhoa", "kzhoa", "test-only", "kzhoa", 1, False, True),
    ],
)
def test_promotion_guard(
    tmp_path: Path,
    job_name: str,
    actor: str,
    rerunner: str,
    token: str,
    login: str,
    api_status: int,
    expected_success: bool,
    calls_api: bool,
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/repository-governance.yml").read_text())["jobs"]
    guard = jobs[job_name]["steps"][0]["run"]
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        '[[ "$*" == "api user --jq .login" ]] || exit 99\n'
        'touch "$API_MARKER"\n'
        'printf "%s\\n" "$API_LOGIN"\n'
        'exit "$API_STATUS"\n'
    )
    gh.chmod(0o755)
    marker = tmp_path / "api-called"
    result = subprocess.run(
        ["bash", "-c", guard],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GH_TOKEN": token,
            "REQUEST_ACTOR": actor,
            "TRIGGERING_ACTOR": rerunner,
            "API_LOGIN": login,
            "API_STATUS": str(api_status),
            "API_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is expected_success, result.stdout + result.stderr
    assert marker.exists() is calls_api
    assert "test-only" not in result.stdout + result.stderr


@pytest.mark.parametrize("state", ["ready", "advanced", "diverged", "already-promoted"])
def test_release_push_preserves_validated_sha_and_ancestry(tmp_path: Path, state: str) -> None:
    """Run the actual release shell against a local bare remote, including workflow edits."""
    jobs = yaml.safe_load((ROOT / ".github/workflows/repository-governance.yml").read_text())["jobs"]
    script = jobs["promote-dev-to-main"]["steps"][-1]["run"]
    remote = tmp_path / "remote.git"
    repo = tmp_path / "checkout"

    def git(*args: str, cwd: Path = tmp_path) -> str:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--bare", str(remote))
    git("init", "-b", "main", str(repo))
    git("config", "user.name", "Test", cwd=repo)
    git("config", "user.email", "test@example.invalid", cwd=repo)
    git("commit", "--allow-empty", "-m", "base", cwd=repo)
    base = git("rev-parse", "HEAD", cwd=repo)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "origin", "main", cwd=repo)
    git("switch", "-c", "dev", cwd=repo)
    workflow = repo / ".github/workflows/example.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: updated\n")
    git("add", ".", cwd=repo)
    git("commit", "-m", "approved workflow change", cwd=repo)
    validated = git("rev-parse", "HEAD", cwd=repo)
    if state == "advanced":
        git("commit", "--allow-empty", "-m", "new dev change", cwd=repo)
    git("push", "origin", "dev", cwd=repo)
    if state == "diverged":
        git("switch", "main", cwd=repo)
        git("commit", "--allow-empty", "-m", "divergent main", cwd=repo)
        git("push", "origin", "main", cwd=repo)
    elif state == "already-promoted":
        git("push", "origin", "dev:main", cwd=repo)
    before = git("rev-parse", "refs/heads/main", cwd=remote)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        env={**os.environ, "EXPECTED_DEV_SHA": validated},
        capture_output=True,
        text=True,
        check=False,
    )
    after = git("rev-parse", "refs/heads/main", cwd=remote)
    if state in {"ready", "already-promoted"}:
        assert result.returncode == 0, result.stdout + result.stderr
        assert after == validated
        assert after != base
    else:
        assert result.returncode != 0
        assert after == before


def test_prepare_promotion_builds_attested_subject_for_exact_squash_commit(tmp_path: Path) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/repository-governance.yml").read_text())["jobs"]
    prepare = next(step for step in jobs["promote-feature"]["steps"] if step["name"] == "Prepare promotion")
    remote = tmp_path / "remote.git"
    repo = tmp_path / "checkout"

    def git(*args: str, cwd: Path = tmp_path) -> str:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--bare", str(remote))
    git("init", "-b", "dev", str(repo))
    git("config", "user.name", "Test", cwd=repo)
    git("config", "user.email", "test@example.invalid", cwd=repo)
    script = repo / "scripts/ci/promotion_provenance.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/ci/promotion_provenance.py", script)
    (repo / "base.txt").write_text("base\n")
    git("add", ".", cwd=repo)
    git("commit", "-m", "base", cwd=repo)
    parent = git("rev-parse", "HEAD", cwd=repo)
    git("remote", "add", "origin", str(remote), cwd=repo)
    git("push", "origin", "dev", cwd=repo)
    git("switch", "-c", "feat/example", cwd=repo)
    (repo / "feature.txt").write_text("feature\n")
    git("add", ".", cwd=repo)
    git("commit", "-m", "promote: feature", cwd=repo)
    source = git("rev-parse", "HEAD", cwd=repo)
    output = tmp_path / "github-output"
    provenance = tmp_path / "qqtools-promotion-provenance.json"

    result = subprocess.run(
        ["bash", "-c", prepare["run"]],
        cwd=repo,
        env={
            **os.environ,
            "BRANCH": "feat/example",
            "ACTOR": "kzhoa",
            "HEAD_MESSAGE": "promote: feature",
            "SOURCE_SHA": source,
            "PROMOTION_RUN": "123",
            "PROMOTION_ATTEMPT": "1",
            "REPOSITORY": "kzhoa/qqtools",
            "PROVENANCE_PATH": str(provenance),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    promoted = values["commit_sha"]
    subject = json.loads(provenance.read_text())
    assert subject["target_commit"] == promoted
    assert subject["target_parent"] == parent
    assert subject["target_tree"] == git("rev-parse", f"{promoted}^{{tree}}", cwd=repo)
    assert subject["source_commit"] == source
    assert subject["source_ref"] == "refs/heads/feat/example"


@pytest.mark.parametrize("workflow", ["dev-preflight.yml", "ci.yml"])
@pytest.mark.parametrize(
    ("evidence", "reused", "fresh", "success"),
    [
        ("success", "true", "skipped skipped", True),
        ("success", "false", "success success", True),
        ("failure", "true", "skipped skipped", False),
        ("cancelled", "false", "success success", False),
        ("success", "false", "success skipped", False),
        ("success", "false", "success failure", False),
        ("success", "false", "", False),
    ],
)
def test_gate_result_requires_evidence_or_all_fresh_jobs(workflow, evidence, reused, fresh, success):
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    result = subprocess.run(
        ["bash", "-c", jobs["gate-result"]["steps"][0]["run"]],
        env={
            **os.environ,
            "EVIDENCE_RESULT": evidence,
            "PROVENANCE_RESULT": "success",
            "REUSED": reused,
            "ATTESTED": "false",
            "FRESH_RESULTS": fresh,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success


def test_attested_dev_promotion_does_not_require_fresh_preflight() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/dev-preflight.yml").read_text())["jobs"]
    result = subprocess.run(
        ["bash", "-c", jobs["gate-result"]["steps"][0]["run"]],
        env={
            **os.environ,
            "EVIDENCE_RESULT": "success",
            "PROVENANCE_RESULT": "success",
            "REUSED": "false",
            "ATTESTED": "true",
            "FRESH_RESULTS": "skipped",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
