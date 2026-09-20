<!-- qqtools-governance:branch-model=v1 -->
<!-- qqtools-governance:strip-dot-dev=v1 -->
<!-- qqtools-governance:agents-owner=kzhoa -->
<!-- qqtools-governance:no-pr-required=v1 -->
<!-- qqtools-governance:workflow-policy=v1 -->
<!-- qqtools-governance:release-model=v1 -->

# AGENTS.md

This file defines the shared execution contract for agents working on qqtools.
Linked guides provide scenario-specific procedures. Private configuration may
supplement collaboration preferences but must not override public contracts or
gates. Public development, validation, and release must not depend on ignored
or private documents.

Read `.codex/AGENTS.md` for supplemental personal preferences if it exists.

## Engineering boundaries

- Inspect relevant code, tests, and specifications before editing. Preserve
  unrelated work, keep changes scoped, and report material assumptions.
- Keep affected public contracts and documentation aligned with behavior.
  Private planning templates are not required public deliverables.
- Use standard tools to manage project environments and caches. Do not modify
  shared/system environments, dependency implementations, or unrelated external
  files unless the task explicitly requires it.

## Scenario navigation

Read the applicable guide; do not load every reference for every task.

| Scenario | Reference |
| --- | --- |
| Implementation and code review | [Code style](docs/development/code-style.md) |
| Commits, feature state, promotion, release | [Development workflow](docs/development/development-workflow.md) |
| Environment setup, dependencies, tool maintenance | [Developer tooling](docs/development/developer-tooling.md) |
| Writing or moving documentation | [Documentation guide](docs/development/documentation-guide.md) |
| Test selection, isolation, execution, evidence | [Test governance](docs/development/test-governance.md) |
| Temporary compatibility, persisted formats, upgrades | [Compatibility governance](docs/development/compatibility-governance.md) |
| Protected files, workflow changes, owner credentials | [Repository governance](docs/development/repository-governance.md) |

## Branch model

The normal route uses `feat/*` or `feature/*` based on current `dev` for substantial changes.
Promote features to `dev` by squash, then `dev` to `main` by validated
fast-forward. `main` must remain an ancestor of `dev`. Never merge a feature
directly into `main` or force-push either public branch as part of promotion.
Pull requests are optional for owner-driven development.

After all release contents have reached `dev`, the repository owner may commit
version-release metadata directly to `dev` under the standard administrator
release procedure. Each such commit is limited to
`src/qqtools/version.py` and `CHANGELOG.md`; it does not authorize direct code,
configuration, test, or workflow changes. The exact pushed SHA must pass complete
Dev Preflight before release promotion begins.

The owner may explicitly choose an [administrator manual release](.github/publish.md)
instead of the normal promotion route below. This exception permits a directly
prepared release and dev reconciliation, but preserves validation gates,
public-history integrity, `.dev/**` exclusion, and `main` ancestry in `dev`.

## Feature-local agent state

Optional `.dev/**` holds disposable, non-sensitive feature execution context.
Never put secrets or private planning content there. Promotion must strip it;
CI must reject its presence on `dev` and `main`. The feature must remain
understandable without private documents.

## Feature promotion

Except for the owner-only version-release metadata commit defined above, before
changing `dev` the feature must contain current `dev`, and the candidate tree
with `.dev/**` removed must pass governance and complete preflight in the
configured promotion workflow before `dev` advances.
Create one squash commit with current `dev` as parent, attest that exact commit
through the trusted promotion workflow, and delete the feature only after a
successful push. Dev Preflight verifies promotion provenance without repeating
the successful feature gate. Before release, the exact `dev` commit must obtain
complete preflight evidence under the development workflow rules.
Reserve `promote:` commit subjects for ready-to-integrate promotion requests.

## Dev release promotion

Dispatch from current `dev`; validate preflight and installed-artifact E2E
against that exact commit. Abort if `dev` advances. Verify `main` ancestry and
fast-forward it to the validated commit without another squash or merge-back.
Main push workflows verify exact-commit evidence from the release gates, or run
the full artifact gate when reusable evidence is unavailable. Release tags must
point to commits reachable from `main`. Follow the development workflow for
dispatch commands and owner-credential promotion of workflow changes.

## Validation

Use `./scripts/dev test [pytest arguments]` for selected tests (Unit by default),
`./scripts/dev preflight` for the complete gate, and `./scripts/dev env` for the
optional IDE environment. Standard tooling uses Python 3.13; full preflight
requires Linux and system tmux. Missing prerequisites must fail, not skip coverage.

During development, run checks appropriate to the changed behavior and report
actual results. Add or update effective regression coverage for executable
behavior defects fixed in the task; use applicable static checks for documents.
Record unrelated defects without expanding scope automatically.

Before requesting integration, run governance, `ruff check src tests scripts`,
`ruff format --check src tests scripts`, and relevant focused tests locally.
Running `./scripts/dev preflight` locally is recommended, and may be required by
an explicit task or risk-specific policy, but it is not a universal local
prerequisite for feature promotion. The configured promotion workflow must run
complete preflight against the exact candidate and pass before `dev` advances.
Release additionally requires the configured installed-artifact gate. A known
baseline failure may be investigated during development but does not waive a
mandatory gate. Post-push checks never replace candidate validation.
The documented owner-only release metadata exception deliberately validates its
exact `dev` SHA after push and blocks `dev`-to-`main` promotion until that run
succeeds.
Do not weaken assertions, hide failures, or skip required lifecycle coverage.

## Workflow governance

Agents must not create, delete, rename, or modify any file under `.github/workflows/**` unless the repository owner has explicitly approved that workflow change in the current task.

Workflows are stable infrastructure. Do not create temporary, one-shot, recovery,
or code-editing workflows. If existing infrastructure cannot support the task,
pause that part and seek owner approval. Approved changes must preserve ownership
protections and update the allowlist as described in the governance guide.
Owner credentials alone are not workflow-change approval.

## Compatibility governance

Register intentionally temporary compatibility behavior before integration in
`docs/spec/compatibility-registry.toml`; keep stable compatibility IDs, code
markers, and verification tests discoverable. Remove shims immediately when
possible. Private notes and TODOs cannot replace lifecycle registration.

Before changing qexp CLI, initialization, registration, lifecycle, migration,
or ownership, read the protected workflows in the
[product spec](docs/spec/qexp_product_spec.md) and invariants in the
[runtime spec](docs/spec/qexp_runtime_spec.md). Breaking protected behavior
requires an explicit compatibility decision and approval before implementation.
Record the affected workflow, impact, and decision in public delivery context.
Do not rewrite protected tests or specs to accommodate a regression.

## Protected governance surface

`AGENTS.md`, `.github/CODEOWNERS`, `.github/workflows/**`,
`scripts/checks/check_release_commit.py`,
`scripts/checks/check_repository_governance.py`, and
`scripts/ci/promotion_provenance.py` are owner-controlled.

Only GitHub actor `kzhoa` may intentionally modify this protected governance surface.

Owner-authorized agents may prepare the explicitly approved local edits on the
owner's behalf; this does not grant another GitHub actor publication rights.
Without owner authorization, pause the affected edits and request it. Preserve
governance markers and pass the governance checker. See the governance guide
for approval scope, publication credentials, and enforcement limits.
