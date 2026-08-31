Publish Pipeline

1. Commit all feature and fix work.
2. Choose the exact `X.Y.Z` release target and inspect its compatibility obligations with
   `python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z`.
3. From that clean candidate commit, run
   `python scripts/release_preflight.py --target-version X.Y.Z`. It first enforces the compatibility
   lifecycle for that target, then runs the complete qexp source-level regression suite, rebuilds a
   wheel, and runs installed-wheel E2E tests.
4. If preflight fails, add or amend the feature/fix commit, then repeat preflight. Do not create a
   release commit yet.
5. After preflight passes, update `qqtools/version.py` and finalize the matching `## vX.X.X` section
   in `CHANGELOG.md`. The version controls the build filename, and the changelog section becomes the
   GitHub Release body.
6. Create one separate release commit for `vX.X.X`.
7. Add the `vX.X.X` tag to that release commit, then push the main branch and tag.
8. The tag workflow trusts the completed local preflight. It validates the tag/version/changelog
   relationship, rebuilds the exact tagged wheel, runs installed-wheel E2E tests, and only then
   creates the GitHub Release and publishes to PyPI; it does not repeat the compatibility lifecycle
   gate.

Test responsibilities:

- Pull-request CI validates unit, non-qexp integration, and CPU e2e behavior. During qexp
  development, contributors run the integration files that protect the changed behavior.
- Release validation runs the complete qexp unit and integration suites, then validates the built,
  installed wheel. Installed-wheel E2E does not replace the source-level release gate.
