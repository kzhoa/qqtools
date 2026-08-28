Publish Pipeline

1. Commit all feature and fix work.
2. From that clean candidate commit, run `python scripts/release_preflight.py`. It runs the complete qexp source-level regression suite, rebuilds a wheel, and runs installed-wheel E2E tests.
3. If preflight fails, add or amend the feature/fix commit, then repeat preflight. Do not create a release commit yet.
4. After preflight passes, update `qqtools/version.py` and finalize the matching `## vX.X.X` section in `CHANGELOG.md`. The version controls the build filename, and the changelog section becomes the GitHub Release body.
5. Create one separate release commit for `vX.X.X`.
6. Add the `vX.X.X` tag to that release commit, then push the main branch and tag.
7. The tag workflow validates the tag/version/changelog relationship, rebuilds the exact tagged wheel, runs installed-wheel E2E tests, and only then creates the GitHub Release and publishes to PyPI.

Test responsibilities:

- Pull-request CI validates unit, non-qexp integration, and CPU e2e behavior. During qexp development, contributors run the integration files that protect the changed behavior.
- Release validation runs the complete qexp unit and integration suites, then validates the built, installed wheel. Installed-wheel E2E does not replace the source-level release gate.
