<!-- qqtools-governance:branch-model=v1 -->
<!-- qqtools-governance:strip-dot-dev=v1 -->
<!-- qqtools-governance:agents-owner=kzhoa -->
<!-- qqtools-governance:no-pr-required=v1 -->
<!-- qqtools-governance:workflow-policy=v1 -->
<!-- qqtools-governance:release-model=v1 -->

# AGENTS.md

This file is the stable execution contract for coding agents and repository automation working on qqtools. It intentionally contains repository-level rules, not private design exploration or implementation journals.

## Authority

- These rules apply to human-assisted and autonomous coding agents working in this repository.
- Public repository correctness must not depend on ignored, local-only, or private planning documents.
- Private planning material may inform an owner decision, but once implementation starts, the feature branch itself must contain enough non-sensitive execution context for an agent to continue the work.

## Branch model

The canonical development flow is:

```text
feat/* or feature/*
        |
        | squash promotion
        v
       dev
        |
        | validated fast-forward promotion
        v
       main
```

- `main` is the stable branch.
- `dev` is the persistent integration branch.
- Substantial changes must be developed on disposable `feat/*` or `feature/*` branches based on the latest `dev`.
- Pull requests are optional and are not required for owner-driven development.
- Do not promote a feature branch if it is behind `dev`; rebase or otherwise rebuild it on the current `dev` first.
- Do not merge a feature branch directly into `main`.
- `main` must remain an ancestor of `dev`. Release promotion advances `main` to an already validated `dev` commit without creating a second squash commit.

## Feature-local agent state

A feature branch may contain `.dev/**` for temporary execution state, including `.dev/work-item.md`.

Recommended `.dev/work-item.md` sections are:

- Goal
- Scope
- Invariants
- Acceptance criteria
- Current state
- Known blockers
- Next actions

Rules:

- `.dev/**` is disposable and may be visible while the public feature branch exists. Never place secrets, private pitch content, unpublished roadmap material, credentials, or other sensitive information there.
- `.dev/**` must never enter `dev` or `main`.
- Promotion must automatically strip `.dev/**` before constructing the squash commit.
- `dev` and `main` CI must fail if `.dev/**` is present.

## Feature promotion

Promotion is a repository operation, not an ordinary merge.

A conforming promotion must:

1. verify the feature branch contains the current `dev` head;
2. validate repository-governance invariants;
3. build the feature tree with `.dev/**` removed;
4. run the configured preflight against that promotion candidate before changing `dev`;
5. create one squash commit whose parent is the current `dev` head;
6. advance `dev` to that commit;
7. delete the disposable feature branch only after the successful push; and
8. allow the `dev` push to trigger `Dev Preflight` again as post-promotion verification.

The repository governance workflow may use a final feature-branch commit whose subject starts with `promote:` as the promotion request. Do not use that convention until the feature is ready to integrate.

## Dev release promotion

Promotion from `dev` to `main` publishes an already validated commit; it does not create another squash commit.

A conforming release promotion must:

1. be explicitly dispatched from the current `dev` ref;
2. run repository preflight and installed-artifact E2E against that exact `dev` commit;
3. abort if `dev` advances while those gates run;
4. verify the current `main` is an ancestor of the validated `dev` commit;
5. update `main` by fast-forward only; and
6. rely on the `main` push workflows for post-promotion verification.

The standard owner-driven command is:

```bash
gh workflow run repository-governance.yml --ref dev -f operation=promote-dev-to-main
```

Do not squash `dev` into `main`, merge `main` back into `dev` after an ordinary release, or force-push either public branch as part of this release flow. Release tags must point to commits reachable from `main`.

GitHub's built-in workflow token cannot create or update `.github/workflows/**`. An explicitly approved owner workflow change must still pass the normal candidate gates, but its final feature-to-`dev` promotion and the first `dev`-to-`main` fast-forward containing that change must be performed with owner credentials. This is a credential boundary, not permission to bypass ancestry, validation, `.dev/**` stripping, or tree-equality checks.

## Validation

Before code is treated as integrated:

- repository governance checks must pass;
- `ruff check src tests scripts` must pass;
- `ruff format --check src tests scripts` must pass;
- the configured `preflight` lane must pass, except for a separately identified pre-existing baseline blocker that is explicitly being repaired; and
- feature-specific regression tests must exist for defects discovered during implementation.

Feature preflight is a gate before `dev` changes. Dev release preflight and installed-artifact E2E are gates before `main` changes. Post-push workflows are verification and must not be the first point at which an integration candidate is tested.

Do not weaken tests, skip lifecycle coverage, or create a superficial fast lane merely to make a feature pass.

## Workflow governance

GitHub Actions workflows are long-lived repository infrastructure, not an ad-hoc remote shell or a temporary editing mechanism.

The stable workflow allowlist is:

- `.github/workflows/ci.yml`
- `.github/workflows/dev-preflight.yml`
- `.github/workflows/publish.yml`
- `.github/workflows/repository-governance.yml`

Rules:

- Agents must not create, delete, rename, or modify any file under `.github/workflows/**` unless the repository owner has explicitly approved that workflow change in the current task.
- If the existing workflows cannot support a task, stop that part of the implementation and ask the owner for approval before changing workflow infrastructure.
- Do not create one-shot, temporary, recovery, patching, or code-editing workflows. Use normal repository file/Git operations for code changes.
- Do not add a new workflow merely to run a command once. Prefer an existing stable workflow, a repository script, or a local/connector execution path.
- Any approved workflow-set change must also update the repository-governance allowlist and keep workflow ownership protections intact.
- Public contributor PRs may still use the stable CI workflows; owner-driven development does not require PRs.

## Compatibility governance

- `docs/spec/compatibility-registry.toml` is a lifecycle ledger, not a design-history database.
- Compatibility IDs such as `QQTOOLS-COMPAT-NNNN` are stable correlation identifiers across registry entries, temporary code markers, tests, commits, and feature work items.
- Public compatibility validation must not require ignored or private pitch/ADR files.
- Temporary compatibility behavior must remain discoverable through its compatibility marker and verification tests.
- Private planning records are optional context and must never be required to build, test, release, or safely advance a compatibility lifecycle.
- Any intentionally temporary compatibility shim that must survive until a later release must be registered before integration, including temporary parser/checker tolerance in developer tooling. Do not rely on TODOs, private notes, or human memory for future cleanup. If the shim can be removed immediately, remove it instead of registering a fake completed item.

## Protected governance surface

The following files define or enforce repository governance and are owner-controlled:

- `AGENTS.md`
- `.github/CODEOWNERS`
- every file under `.github/workflows/**`
- `scripts/checks/check_repository_governance.py`

Only GitHub actor `kzhoa` may intentionally modify this protected governance surface.

Agents running under any other actor must not edit these files. If a requested change requires modifying them, stop that part of the change and ask the repository owner to perform or authorize it.

Even when an agent is operating through credentials that appear as GitHub actor `kzhoa`, the agent must still follow the workflow-governance approval rule above. Owner credentials are not implicit approval to change workflow infrastructure.

Owner changes must preserve the machine-readable governance markers at the top of this file and pass the repository-governance checker. This requirement is intended to prevent accidental policy erosion as well as unauthorized edits.

CI detection alone cannot revoke an already-authorized direct push. The strongest enforcement therefore also requires GitHub branch/ruleset protection that makes the governance check required on `dev` and `main` and restricts direct pushes according to the repository owner's policy.
