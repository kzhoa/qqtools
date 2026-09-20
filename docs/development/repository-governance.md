# Repository governance

The root [AGENTS.md](../../AGENTS.md) defines mandatory execution constraints.
This guide explains protected changes and publication boundaries. Use
[development workflow](development-workflow.md) for promotion procedures.

## Approval, local editing, and publication

The protected surface consists of root `AGENTS.md`, `.github/CODEOWNERS`, every
file under `.github/workflows/`, and
`scripts/checks/check_repository_governance.py`. The promotion provenance helper
`scripts/ci/promotion_provenance.py` is also protected because it executes inside
the credentialed promotion job and reconstructs the subject trusted by Dev
Preflight. Owner `kzhoa` retains authority over these files.

An explicit owner request to change a protected file authorizes an agent to
prepare those local edits on the owner's behalf. Approval is scoped to that
change; it does not authorize unrelated governance edits, workflow changes, or
publication. A local agent need not possess GitHub credentials to prepare an
authorized change. Local Git author configuration is not proof of GitHub identity
or owner authorization.

Without authorization, pause the protected portion and request the owner's
decision. Continue independent authorized work. Do not request the same approval
again when its scope has already been granted. Publication of protected changes
remains owner-controlled; local edit approval does not allow another GitHub actor
to publish them.

Workflow edits require explicit owner approval for the workflow change in the
current task, even when owner credentials are available. Approval to change a
guide or application code does not implicitly authorize workflow edits.

## Stable workflow set

The allowlist is:

- `.github/workflows/ci.yml`
- `.github/workflows/dev-preflight.yml`
- `.github/workflows/publish.yml`
- `.github/workflows/repository-governance.yml`

An approved workflow-set change must update the allowlist in
[the governance checker](../../scripts/checks/check_repository_governance.py)
and this guide while preserving ownership protections. Never use workflows as
an ad-hoc remote shell or temporary editing/recovery mechanism. Prefer existing
stable workflows, repository scripts, or normal local Git/file operations.
Public contributor PRs may use stable CI workflows; owner-driven work does not
require a PR.

## Workflow publication credentials

The owner may authorize an [administrator manual release](../../.github/publish.md)
when normal promotion is unsuitable. That procedure permits a different release
route, not skipped gates, unauthorized workflow edits, or rewritten public
history. Authorization to maintain this guide is not authorization to publish
a particular release.

The existing promotion jobs use the repository Actions secret
`OWNER_PROMOTION_TOKEN` for Git authentication. Create it as owner `kzhoa` using a
fine-grained personal access token limited to this repository with **Contents:
read and write** and **Workflows: read and write**. Metadata read access is
implicit. Actions write permission is not required for Git pushes. Set an expiry
and replace the secret before the token expires; never put the token in Git or
workflow logs. The separate publishing secret `ACCESS_TOKEN` is unchanged.

Only owner `kzhoa` may request or rerun automatic promotion. Before authenticated
checkout, each final promotion job checks both actors and verifies the token's
identity through GitHub's user API. A missing, expired, or non-owner token stops
promotion; insufficient write permissions cause the push to fail. There is no
fallback to `GITHUB_TOKEN`. The owner token is exposed only to the final promotion
job, after validation, and checkout removes its persisted Git credential during
job cleanup. Validation jobs retain their read-only built-in tokens.

This credential permits approved workflow-file changes and allows dev/main pushes
to trigger their normal post-promotion workflows. GitHub's built-in token cannot
perform workflow-file updates, and its pushes do not trigger push workflows.
Credential possession does not replace explicit owner approval for workflow edits.

Feature promotion also requests GitHub's short-lived OIDC identity to create a
Sigstore attestation for a deterministic manifest binding the final squash
commit, tree, parent, source feature SHA/ref, and workflow run. This does not add
another repository secret. The `dev` workflow trusts the provenance only when
GitHub verifies the expected repository, signer workflow and digest, source
digest/ref, and GitHub-hosted runner. Commit messages and the owner PAT alone are
not accepted as provenance. Verification failure runs complete preflight.

Retain normal candidate validation, `.dev/**` stripping, ancestry checks, and
equality between the validated candidate tree and published tree. Follow the
development workflow for each promotion stage. Configure the secret before the
first `promote:` feature push using this workflow. Do not invent a temporary
workflow to bypass the credential boundary.

## Validation and enforcement

Preserve root governance markers and run:

```bash
python scripts/checks/check_repository_governance.py
```

The checker validates required markers, sections, selected policy statements,
the workflow allowlist, and `.dev/**` exclusion on public branches. Passing it
does not prove owner authorization or replace the full integration gate.

CI detection cannot revoke an already-authorized direct push. Effective remote
enforcement also requires owner-managed branch/ruleset protection that makes
governance checks required on `dev` and `main` and restricts direct pushes.
Do not assume those remote settings are verified by the local checker.
