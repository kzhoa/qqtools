# Development workflow

The root [AGENTS.md](../../AGENTS.md) defines the repository execution contract.
This guide describes the development and delivery sequence for maintainers and
coding agents. Read [developer tooling](developer-tooling.md) for environment setup
and [test governance](test-governance.md) for validation selection.
Use the [code style guide](code-style.md) when implementing or reviewing code.
For protected-file approval and publication credentials, read
[repository governance](repository-governance.md).

## Start and implement a change

1. Inspect the worktree and relevant code, tests, and specifications. Preserve
   unrelated local changes. Substantial work belongs on a disposable `feat/*` or
   `feature/*` branch based on the latest `dev`.
2. Identify the goal, scope, public contracts, and acceptance criteria. Follow
   [documentation guidance](documentation-guide.md) when behavior or contracts
   change. Read [compatibility governance](compatibility-governance.md) before
   changing temporary compatibility, persisted formats, or upgrade/recovery paths.
3. Keep non-sensitive implementation state in `.dev/work-item.md` when needed.
   Public implementation must be understandable without private planning files.
4. Implement bounded changes and run the smallest tests that can disprove each
   changed behavior. Record actual results and any limitations. Update affected
   public documentation alongside the implementation.

## Optional feature execution state

Use `.dev/work-item.md` when a change needs persistent execution context.
Useful sections are Goal, Scope, Invariants, Acceptance criteria, Current state,
Known blockers, and Next actions. This is a suggested structure, not a required
template for every task. Never include secrets, credentials, private pitch
content, or unpublished roadmap material. Promotion strips all `.dev/**` before
constructing the squash commit; public-branch CI rejects its presence.

## Prepare a commit and feature push

Review the diff for scope, accidental files, stale references, and missing tests.
For packaging, entry points, or optional dependencies, review `pyproject.toml`,
source entry points, and documentation together. Commit messages should describe
the concrete change; pull requests are optional for owner-driven development.

Before synchronizing branches, account for every local change. Fetch current
remote refs and update the feature branch against current `dev` when necessary.
Do not apply a generic rebase operation to `dev` or `main`. Re-run checks whose
evidence was invalidated by conflict resolution or subsequent edits.

A normal feature push does not integrate the change. Reserve a commit subject
starting with `promote:` for a feature that is ready for integration. Do not use
that prefix merely to save work or request a review.

## Promote a feature to dev

Promotion is a governed repository operation. The feature must contain the
current `dev` head. The promotion candidate excludes `.dev/**`; governance and
the complete shared preflight must pass against that candidate before `dev`
changes. `./scripts/dev preflight` is the standard local gate; the configured
promotion workflow also validates the candidate before advancing `dev`.

The successful operation creates one squash commit whose parent is current
`dev`, pushes it to `dev`, and only then deletes the disposable feature branch.
The `dev` push triggers post-promotion verification. That verification does not
replace the candidate gate. Never merge a feature directly into `main`.

Protected governance changes require the owner under the root contract. Workflow
changes additionally require explicit approval for the current task. When an
approved feature changes workflow files, the owner must perform the final
promotion using owner credentials after the normal candidate gates; the workflow
token cannot perform that update. Preserve the validated candidate tree and
ancestry when doing so.

## Promote dev to main

Dispatch the existing governance workflow from current `dev`:

```bash
gh workflow run repository-governance.yml --ref dev -f operation=promote-dev-to-main
```

The exact `dev` commit must pass repository preflight and installed-artifact E2E.
If `dev` advances during validation, the operation must abort and be dispatched
again. Current `main` must be an ancestor of the validated commit; promotion
fast-forwards `main` to that same commit. Do not create another squash commit,
force-push public branches, or merge `main` back into `dev` after a normal release.

The first such promotion containing approved workflow changes also requires
owner credentials for the final fast-forward. All gates, ancestry checks, and
validated tree equality still apply. Main push workflows provide post-promotion
verification.

## Prepare a versioned release

Inspect the target release's compatibility obligations and resolve due actions
using [compatibility governance](compatibility-governance.md#commands). Run
`scripts/release_preflight.py --target-version X.Y.Z` with Python 3.13 from a
clean, committed candidate before creating the release version commit; the
target must be later than the current source version.

Release changes normally follow the same feature/dev/main flow. The owner may
instead authorize the [administrator manual release procedure](../../.github/publish.md),
which retains validation gates and public-history invariants while allowing a
directly prepared release candidate and reconciliation into dev. Tags must reference
commits reachable from `main`. Tagged publishing validates the exact selected
wheel through `release-e2e`; a source test result does not replace that gate.
