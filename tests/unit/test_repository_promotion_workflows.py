from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_feature_promotion_preflights_stripped_candidate_before_advancing_dev() -> None:
    governance = _workflow("repository-governance.yml")
    preflight = _workflow("dev-preflight.yml")

    assert "workflow_call:" in preflight
    assert "strip_dev_state:" in preflight
    assert "if: inputs.strip_dev_state" in preflight
    assert "run: rm -rf .dev" in preflight

    preflight_job = governance.index("preflight-feature-promotion:")
    advance_step = governance.index("- name: Advance dev with squash commit")
    assert preflight_job < advance_step
    assert "uses: ./.github/workflows/dev-preflight.yml" in governance[preflight_job:advance_step]
    assert "strip_dev_state: true" in governance[preflight_job:advance_step]
    assert "gh workflow run dev-preflight.yml" not in governance
    assert "GitHub's workflow token cannot update workflow files." in governance


def test_dev_release_runs_both_gates_and_fast_forwards_exact_validated_sha() -> None:
    governance = _workflow("repository-governance.yml")
    ordinary_ci = _workflow("ci.yml")

    assert "workflow_call:" in ordinary_ci
    assert "preflight-dev-release:" in governance
    assert "artifact-e2e-dev-release:" in governance
    assert "uses: ./.github/workflows/dev-preflight.yml" in governance
    assert "uses: ./.github/workflows/ci.yml" in governance
    assert 'if [[ "$dev_sha" != "$EXPECTED_DEV_SHA" ]]' in governance
    assert 'git merge-base --is-ancestor "$main_sha" "$dev_sha"' in governance
    assert 'git push origin "$dev_sha:refs/heads/main"' in governance
    assert "GitHub's workflow token cannot promote workflow-file changes to main." in governance
    assert "--force" not in governance


def test_publish_rejects_tags_that_are_not_reachable_from_main() -> None:
    publish = _workflow("publish.yml")

    assert "fetch-depth: 0" in publish
    assert 'tag_commit="$(git rev-parse "${GITHUB_SHA}^{commit}")"' in publish
    assert 'git merge-base --is-ancestor "$tag_commit" origin/main' in publish
