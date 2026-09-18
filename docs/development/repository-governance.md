# Repository governance

The root [AGENTS.md](../../AGENTS.md) defines mandatory execution constraints.
This guide explains protected changes and publication boundaries. Use
[development workflow](development-workflow.md) for promotion procedures.

## Approval, local editing, and publication

The protected surface consists of root `AGENTS.md`, `.github/CODEOWNERS`, every
file under `.github/workflows/`, and
`scripts/checks/check_repository_governance.py`. Owner `kzhoa` retains authority
over these files.

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

GitHub's built-in workflow token cannot create or update workflow files. For an
approved feature containing workflow changes, perform the final feature-to-dev
promotion and the first dev-to-main fast-forward containing those changes with
owner credentials. This is a credential boundary, not a gate exemption.

Retain normal candidate validation, `.dev/**` stripping, ancestry checks, and
equality between the validated candidate tree and published tree. Follow the
development workflow for each promotion stage. Do not invent a temporary
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
