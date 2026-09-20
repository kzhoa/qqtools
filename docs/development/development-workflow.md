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
changes. The configured promotion workflow is authoritative for that complete
candidate gate and must not advance `dev` when it fails. Before requesting
promotion, run governance, `ruff check src tests scripts`,
`ruff format --check src tests scripts`, and relevant focused tests locally.
`./scripts/dev preflight` remains the standard way to run the complete gate
locally and is recommended, but is required locally only when an explicit task
or risk-specific policy says so.

The successful operation creates one squash commit whose parent is current
`dev`, pushes it to `dev`, and only then deletes the disposable feature branch.
The `dev` push triggers post-promotion verification. That verification does not
replace the candidate gate. Never merge a feature directly into `main`.

Once the promotion workflow run is accepted, the local terminal or agent session
does not need to remain open. Retain the run URL and candidate SHA for inspection.
An accepted run means validation was submitted, not that promotion succeeded;
gate failure must leave `dev` unchanged.

Protected governance changes require the owner under the root contract. Workflow
changes additionally require explicit approval for the current task. Automatic
promotion requires owner `kzhoa` as both the requesting and rerunning actor, and
the configured `OWNER_PROMOTION_TOKEN`; see
[credential setup](repository-governance.md#workflow-publication-credentials).
The final job uses that owner credential for approved workflow-file updates and
normal pushes, preserving the validated candidate tree and ancestry.

## Promote dev to main

Dispatch the existing governance workflow from current `dev`:

```bash
gh workflow run repository-governance.yml --ref dev -f operation=promote-dev-to-main
```

The exact `dev` commit must pass repository preflight and installed-artifact E2E.
The release preflight reuses the recent Dev Preflight push run for that SHA,
waiting for it to finish if necessary; it does not cancel and restart that run.
If `dev` advances during validation, the operation must abort and be dispatched
again. Current `main` must be an ancestor of the validated commit; promotion
fast-forwards `main` to that same commit. Do not create another squash commit,
force-push public branches, or merge `main` back into `dev` after a normal release.

The final fast-forward uses `OWNER_PROMOTION_TOKEN`, including when the candidate
contains approved workflow changes. All gates, ancestry checks, and validated
tree equality still apply. Main push workflows provide post-promotion verification.

After dispatch is accepted, execution belongs to GitHub Actions and does not
require a local terminal or agent session to remain open. Keep the run URL and
candidate SHA to inspect the result. Dispatch success means submitted, not
promoted; a failed gate or credential check leaves main unchanged. Enable Actions
notifications in GitHub notification settings for completion/failure updates.
Post-promotion workflows run separately and have their own results. Main's
artifact workflow verifies successful release-gate jobs for the exact SHA instead
of rebuilding and retesting the same commit.

## Reusing gate evidence

Reuse accepts only `kzhoa`-requested and `kzhoa`-rerun executions on `dev`, created
within the last 24 hours, for the identical commit SHA. Preflight evidence comes
from `dev-preflight.yml` push runs; artifact evidence comes from
`repository-governance.yml` release dispatches and must include all three Python
smoke jobs and the Python 3.13 installed E2E job. Evidence jobs must finish
successfully; skipped jobs are not evidence. The run attempt is fixed while
waiting, and the newest eligible run takes precedence over older successes.

An in-progress source run is awaited for up to 20 minutes. A failed/cancelled
source gate, changed attempt, or timeout fails the consuming gate; it never
silently retries a known failure. Missing or expired evidence, or an unavailable
GitHub API, selects full execution. Each selector writes its evidence run/attempt
URL or fresh-execution decision to the Actions summary. Gate-result jobs require
successful evidence or every fresh job, preventing skipped dependencies from
turning a release green.

This is commit-result reuse within a bounded time window, not a dependency lock:
external package indexes and runner images may change during that window. Code,
workflow, tests, or dependency declaration edits change the SHA and invalidate
reuse. Manually dispatch Dev Preflight for a fresh source run; for fresh artifact
verification use PR CI or the release dispatch, which always executes artifact
tests. Feature and squash commits remain separate candidates and are not equated
by tree or commit message. Version-tag publishing keeps its own exact-wheel gate.

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
