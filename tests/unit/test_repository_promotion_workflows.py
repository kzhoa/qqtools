from __future__ import annotations

from pathlib import Path

import yaml

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


def test_promoted_commit_is_attested_and_dev_verifies_exact_provenance() -> None:
    governance = _workflow("repository-governance.yml")
    preflight = _workflow("dev-preflight.yml")

    assert "id-token: write" in governance
    assert "attestations: write" in governance
    assert "uses: actions/attest@v4" in governance
    assert "Promotion-source: %s" in governance
    assert "Promotion-run: %s" in governance
    assert "Promotion-attempt: %s" in governance
    assert "Promotion-provenance: github-attestation-v1" in governance
    assert governance.index("- name: Attest exact promotion commit") < governance.index(
        "- name: Advance dev with squash commit"
    )

    assert "Trusted promotion provenance" in preflight
    assert "github.ref == 'refs/heads/dev'" in preflight
    assert 'signer_workflow="$GITHUB_REPOSITORY/.github/workflows/repository-governance.yml"' in preflight
    for option in (
        "--signer-workflow",
        "--signer-digest",
        "--source-digest",
        "--source-ref",
        "--deny-self-hosted-runners",
    ):
        assert option in preflight
    assert "needs.promotion-provenance.outputs.attested != 'true'" in preflight

    jobs = yaml.safe_load(governance)["jobs"]
    assert jobs["preflight-feature-promotion"]["permissions"]["attestations"] == "read"
    assert jobs["preflight-dev-release"]["permissions"]["attestations"] == "read"


def test_promotion_provenance_helper_is_owner_protected() -> None:
    governance = _workflow("repository-governance.yml")
    codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")

    assert "scripts/ci/promotion_provenance\\.py" in governance
    assert "path === 'scripts/ci/promotion_provenance.py'" in governance
    assert "/scripts/ci/promotion_provenance.py @kzhoa" in codeowners


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
    assert "--force" not in governance


def test_publish_rejects_tags_that_are_not_reachable_from_main() -> None:
    publish = _workflow("publish.yml")

    assert "fetch-depth: 0" in publish
    assert 'tag_commit="$(git rev-parse "${GITHUB_SHA}^{commit}")"' in publish
    assert 'git merge-base --is-ancestor "$tag_commit" origin/main' in publish


def test_owner_credential_is_confined_to_gated_promotion_jobs() -> None:
    jobs = yaml.safe_load(_workflow("repository-governance.yml"))["jobs"]
    expected_gates = {
        "promote-feature": {"validate-push", "preflight-feature-promotion"},
        "promote-dev-to-main": {"validate-release-request", "preflight-dev-release", "artifact-e2e-dev-release"},
    }
    for name, job in jobs.items():
        if name not in expected_gates:
            assert "OWNER_PROMOTION_TOKEN" not in str(job)
            continue
        assert set(job["needs"]) == expected_gates[name]
        assert job["permissions"]["contents"] == "read"
        steps = job["steps"]
        verify, checkout = steps[:2]
        assert verify["env"]["GH_TOKEN"] == "${{ secrets.OWNER_PROMOTION_TOKEN }}"
        assert "gh api user --jq .login" in verify["run"]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"]["token"] == "${{ secrets.OWNER_PROMOTION_TOKEN }}"
        assert "workflow token cannot" not in str(job)
