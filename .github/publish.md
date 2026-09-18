# Administrator manual release

Use this procedure when the repository owner chooses a manual release instead
of the normal feature-to-dev-to-main promotion. The administrator may prepare a
release candidate directly from `main`, select the intended fixes, and publish
with owner credentials without a feature squash or governance-workflow dispatch.
Agents must have explicit owner authorization for the particular manual release;
this guide alone is not permission to publish.

This exception changes the delivery route, not the validation requirements.
Keep public history intact, exclude `.dev/**`, and ensure the released `main`
remains an ancestor of `dev`. Workflow edits still require separate explicit
approval. See [repository governance](../docs/development/repository-governance.md)
for protected files and credentials.

## Manual release checklist

1. **Prepare a clean candidate.** Fetch current `main`, `dev`, and tags. Record
   why a manual release is needed and which committed changes are included.
   Use an isolated checkout or local release branch based on `main` when needed;
   do not include unrelated working-tree changes. Select an unused `X.Y.Z`
   version later than the candidate's current version.
2. **Validate before the version bump.** In the Python 3.13 development
   environment, inspect and resolve compatibility obligations, then run:

   ```bash
   python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z
   python scripts/release_preflight.py --target-version X.Y.Z
   ```

   The release script requires a clean committed worktree, checks compatibility
   and export stubs, and runs Unit, general Integration, and complete qexp
   Integration. Fix failures and repeat before creating the release commit.
3. **Create and validate the release commit.** Update
   `src/qqtools/version.py` and finalize the nonempty `## vX.Y.Z` section in
   `CHANGELOG.md`. Commit these together. Remove any `.dev/**` from the candidate
   before validation. Run the complete source gate and installed-artifact gate
   on this final committed candidate:

   ```bash
   ./scripts/dev preflight
   tox run -e artifact-e2e
   ```

   Use the configured development environment for tox. Record the candidate SHA
   and successful results. Any subsequent candidate changes invalidate affected
   evidence. Do not rerun `release_preflight.py` with an already-applied target:
   that script intentionally requires the target to exceed the current version.
4. **Publish the validated history with owner credentials.** Re-fetch and check
   that the remote refs have not changed since preparation; if they have, rebuild
   the affected candidate and repeat its gates. Update `main` by fast-forward to
   the validated release commit. Ensure `dev` contains that commit before or
   atomically with the `main` update. If `dev` has unreleased work, retain it by
   incorporating the release into a separate dev candidate and validating that
   candidate with complete preflight before updating `dev`. Never reset `dev`
   to discard that work or force-push public history. This explicit manual path
   permits release reconciliation into `dev`; normal releases need no merge-back.
5. **Tag and verify publication.** After confirming the release commit is on
   remote `main`, create `vX.Y.Z` on that exact commit and push that tag explicitly.
   Do not tag an earlier feature commit or push unrelated tags. Monitor the
   publishing workflow and confirm both GitHub Release assets and PyPI publication.

## What the tag workflow does

The existing [publish workflow](workflows/publish.yml) checks that the tagged
commit is reachable from `main` and that tag, package version, and changelog
agree. It builds the sdist and wheel, runs `release-e2e` against that exact wheel,
then creates the GitHub Release using the changelog section and publishes to PyPI.
It does not rerun the local source-level release or compatibility gates.

If installed-wheel E2E fails, inspect `release-e2e-diagnostics` for the JUnit
report and retained runtime evidence (seven-day retention). Later publishing
steps can fail after a GitHub Release has already been created; inspect which
outputs exist before retrying. Follow the same-version retry procedure below
when PyPI has not accepted any distribution for the version.

## Retry a failed release with the same version

If `vX.Y.Z` fails and the version has not been used on PyPI, retry `X.Y.Z`;
do not bump the version merely because the workflow failed. Delete the remote
tag and push it again to trigger a fresh publishing run.

1. **Confirm eligibility.** Ensure the previous publishing run has finished or
   been cancelled and no other release attempt is active. Check both the upload
   logs and PyPI: neither a wheel nor an sdist may have been accepted for
   `X.Y.Z`. A failed workflow alone does not prove this. If publication is partial
   or its state is uncertain, do not delete or move the tag under this procedure.
   A previously published version is not eligible even if its files were later
   removed.
2. **Prepare the retry.** Fix the cause and keep the package version and changelog
   at `X.Y.Z`. For an infrastructure-only failure, reuse the validated commit.
   If code changes, commit the fixes without rewriting public branch history,
   recheck compatibility for `X.Y.Z`, rerun the source release checks (Unit,
   Integration, complete qexp Integration, and export-stub validation), complete
   preflight, and installed-artifact gate. The release script rejects a target
   already present in the source version; run its checks individually when
   validating such a retry. Update `dev` and `main` as in step 4 above, and record
   the exact validated commit reachable from remote `main`.
3. **Remove stale release output and recreate the tag.** Preserve useful failure
   diagnostics. If the failed attempt created a GitHub Release, delete that
   release and its assets before retrying so they cannot describe an older
   candidate. Then delete the remote tag and recreate the local tag on the
   validated commit. With `X.Y.Z` and `VALIDATED_COMMIT_SHA` replaced:

   ```bash
   release_tag=vX.Y.Z
   release_commit=VALIDATED_COMMIT_SHA
   git fetch origin main dev
   git merge-base --is-ancestor "$release_commit" origin/main
   git merge-base --is-ancestor "$release_commit" origin/dev
   # Only if a GitHub Release exists for the failed attempt:
   gh release delete "$release_tag" --yes
   git push origin --delete "$release_tag"
   # Only if the local tag exists:
   git tag -d "$release_tag"
   git tag -a "$release_tag" "$release_commit" -m "Release $release_tag"
   git push origin "refs/tags/$release_tag"
   ```

   Run these steps individually and stop on any failed check or unexpected
   deletion failure. Do not force-push `main` or `dev`. Recreating this failed,
   unpublished release tag is the specific exception to tag immutability.
4. **Verify the new run.** Confirm the tag push starts a fresh publishing run for
   the intended commit. Check the GitHub Release assets and PyPI upload results.
   Once any distribution has been published to PyPI, do not use tag recreation
   to replace that version with changed contents.
