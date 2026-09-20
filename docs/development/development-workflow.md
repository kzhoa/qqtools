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
`dev`, generates GitHub OIDC/Sigstore provenance for a deterministic manifest
binding that commit, tree, parent, source feature SHA/ref, and promotion run,
then pushes it to `dev` and deletes the disposable feature branch. The `dev`
push reconstructs the manifest and verifies its repository, signer workflow,
workflow/source digest, source ref, and GitHub-hosted runner identity. Valid
provenance avoids repeating the complete feature gate. Missing, malformed, or
unverifiable provenance falls back to complete preflight. Never merge a feature
directly into `main`.

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
The release preflight reuses a recent complete Dev Preflight push run for that
SHA, waiting for it to finish if necessary; a provenance-only promotion run is
not complete preflight evidence, so release executes the full source gate.
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
from `dev-preflight.yml` push runs whose CPU preflight job actually completed;
an attested promotion whose duplicate job was skipped deliberately selects fresh
release execution. Artifact evidence comes from
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

Promotion provenance is not test evidence. It proves that the trusted workflow
created and attested the exact squash result after its feature candidate gate;
it cannot satisfy the later exact-`dev` release gate.

This is commit-result reuse within a bounded time window, not a dependency lock:
external package indexes and runner images may change during that window. Code,
workflow, tests, or dependency declaration edits change the SHA and invalidate
reuse. Manually dispatch Dev Preflight for a fresh source run; for fresh artifact
verification use PR CI or the release dispatch, which always executes artifact
tests. Feature and squash commits remain separate candidates and are not equated
by tree or commit message. Version-tag publishing keeps its own exact-wheel gate.

## Standard administrator version release

Use this procedure after all features intended for a release have been promoted
to `dev`. Patch, minor, and major releases use the same delivery path. Select an
unused `X.Y.Z` version later than the current source version according to the
release's compatibility scope. The repository owner adds the version metadata
directly to `dev`, waits for complete preflight evidence for that exact commit,
promotes it to `main`, and publishes it by tag. This narrow release exception
does not permit other direct changes to `dev`.

1. **Freeze and inspect the release contents.** Fetch current `main`, `dev`, and
   tags. Confirm that every intended feature is present on `dev`, unrelated work
   is excluded, and current `main` is an ancestor of `dev`. Review the commits
   and user-visible changes since the previous release. Do not start publication
   while another change is being promoted to `dev`.
2. **Validate before changing the version.** Use a clean, committed checkout that
   exactly matches current remote `dev`. Record that commit SHA, inspect and
   resolve the target release's compatibility
   obligations using [compatibility governance](compatibility-governance.md#commands),
   then run with Python 3.13:

   ```bash
   python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z
   python scripts/release_preflight.py --target-version X.Y.Z
   ```

   `release_preflight.py` requires the target to be later than the current source
   version, so run it before applying the version bump. A failure blocks the
   release; resolve it and repeat the check against the updated committed
   candidate.
3. **Commit the release metadata directly to `dev`.** Update
   `src/qqtools/version.py` to `X.Y.Z`, add a
   nonempty `## vX.Y.Z` section below `## Unreleased` in `CHANGELOG.md`, and move
   the applicable unreleased notes into it. Review the release notes against all
   commits being published. This owner-only exception permits exactly those two
   files in the commit. Fetch remote `dev` again before committing; if it no
   longer matches the validated SHA, rebuild and revalidate from the new head.
   Commit and push with owner credentials, without a `promote:` subject:

   ```bash
   release_base=VALIDATED_DEV_SHA
   git fetch --no-tags origin main dev
   test "$(git rev-parse HEAD)" = "$release_base"
   test "$(git rev-parse origin/dev)" = "$release_base"
   git add src/qqtools/version.py CHANGELOG.md
   test "$(git diff --cached --name-only | sort)" = \
     "$(printf '%s\n' CHANGELOG.md src/qqtools/version.py)"
   git commit -m "release: prepare vX.Y.Z"
   git push origin HEAD:dev
   ```

   Do not include code, configuration, tests, workflows, `.dev/**`, or any other
   file. Such changes require the normal feature-promotion path.
4. **Wait for complete Dev Preflight.** Record the pushed release commit SHA and
   wait for its `dev-preflight.yml` push run. Because a direct release commit has
   no feature-promotion provenance, the workflow runs complete preflight for that
   exact SHA. Do not begin `dev`-to-`main` promotion until the run succeeds. If it
   fails, keep the release blocked; code fixes must use feature promotion, and any
   permitted metadata correction creates a new SHA that must pass again.
5. **Promote the release commit to `main`.** Dispatch the governed release from
   that current `dev` commit with owner credentials:

   ```bash
   gh workflow run repository-governance.yml --ref dev -f operation=promote-dev-to-main
   ```

   This workflow validates the exact `dev` commit with repository preflight and
   installed-artifact E2E, verifies that `main` is its ancestor, and fast-forwards
   `main` to it. It aborts if `dev` advances during validation. Confirm the run
   succeeded and that remote `main` points to the recorded release commit before
   creating a tag; otherwise investigate or dispatch again from the new intended
   `dev` head.
6. **Tag the validated commit.** Create an annotated `vX.Y.Z` tag on the recorded
   release commit and push only that tag:

   ```bash
   release_tag=vX.Y.Z
   release_commit=VALIDATED_COMMIT_SHA
   git fetch origin main dev
   test -z "$(git ls-remote --tags origin "refs/tags/$release_tag")"
   test "$(git rev-parse origin/main)" = "$release_commit"
   git merge-base --is-ancestor "$release_commit" origin/dev
   git tag -a "$release_tag" "$release_commit" -m "Release $release_tag"
   git push origin "refs/tags/$release_tag"
   ```

   Stop if the tag already exists, `main` differs from the validated commit, or
   that commit is not an ancestor of `dev`. Never move a published tag.
7. **Verify publication.** The tag starts `publish.yml`, which checks that the
   tagged commit is reachable from `main` and that the tag, package version, and
   changelog agree. It builds the distributions, runs `release-e2e` against the
   exact wheel, creates the GitHub Release, and publishes to PyPI. Confirm the
   workflow, release assets, release notes, and both PyPI distributions before
   declaring the release complete. If publication fails, inspect which outputs
   already exist and follow the
   [same-version retry rules](../../.github/publish.md#retry-a-failed-release-with-the-same-version);
   never reuse a version accepted by PyPI.

The repository owner may explicitly choose the
[administrator manual release procedure](../../.github/publish.md) when the
normal feature-to-`dev`-to-`main` path is unsuitable. That is an exception for a
particular release, not the standard version-release procedure.
