Publish Pipeline

- commit workspace
- change version number in `qqtools/version.py`, this controls the build filename `qqtools-x.x.x-py3-none-any.whl`;
- finalize the matching `## vX.X.X` section in `CHANGELOG.md`; the GitHub Release body is extracted from this section exactly;
- run `tox -e smoke`; this builds and installs a wheel in an isolated environment before release (the release workflow tests the wheel in `dist/` directly);
- create a separate release commit for vX.X.X before creating the tag;
- add a new tag `vX.X.X`;
- push the main branch and the version tag;
