---
doc_type: adr
adr_id: ADR-0003
status: accepted
updated_at: 2026-09-20
archived_at:
supersedes: [ADR-0001, ADR-0002]
superseded_by:
---

# ADR-0003: Validate Releases Through an Exact-SHA CI Profile

## Context

The earlier release procedure ran a local release preflight before changing the
source version, then ran Dev Preflight after committing the version and changelog.
Both gates repeated Unit and general Integration coverage, but the local result
could not serve as trusted evidence for `dev`-to-`main` promotion. It also tested
a different commit from the one ultimately published.

qqtools now develops through feature branches, `dev`, and a validated
fast-forward to `main`. A standard version release adds only
`src/qqtools/version.py` and `CHANGELOG.md` directly to `dev` under owner control.

## Decision

Compatibility `plan` remains the pre-bump planning command. It reports due work
but is not release evidence. After the owner pushes the version and changelog
commit, Dev Preflight recognizes the restricted metadata diff and selects a
release source profile for that exact SHA.

The release profile validates the owner and release-commit shape, version and
changelog agreement, compatibility lifecycle, export stubs, static and repository
governance, Unit, general Integration, and complete qexp Integration. Complete
qexp replaces the representative qexp selection used by the ordinary feature
profile; the release profile does not run both.

Feature-profile and release-profile evidence have distinct job identities.
`dev`-to-`main` release promotion accepts only recent, owner-triggered,
release-profile evidence for the identical SHA. Missing or expired evidence
causes the release profile to run again. The promotion separately validates the
installed artifact before fast-forwarding `main`.

The tag workflow continues to build and run `release-e2e` against the exact wheel
selected for publication. Source evidence and installed-wheel evidence remain
separate because they protect different delivery boundaries.

## Consequences

- Unit and general Integration run once for the final release source commit
  instead of once before and once after the version bump.
- Compatibility enforcement is attached to the committed release version and
  produces trusted GitHub evidence.
- A failed release profile leaves the metadata candidate on `dev`, but cannot
  advance `main` or create a valid release tag.
- Local compatibility planning gives earlier feedback without requiring a local
  test environment; mandatory release tests run in GitHub Actions.
- Ordinary feature promotion keeps its faster representative qexp profile.
- Manual administrator releases retain explicit source and artifact checks
  because they intentionally bypass the standard `dev` release route.

## Rejected Alternatives

- **Keep both complete source gates.** Repeating the same Unit and Integration
  suites adds cost without producing stronger exact-commit evidence.
- **Reuse feature-profile evidence for release.** Representative qexp coverage
  does not satisfy the complete release lifecycle requirement.
- **Move compatibility enforcement to the tag workflow.** A failure would occur
  after the release tag exists and after `main` has already been selected.
- **Drop exact-wheel validation.** Source tests do not prove that the built and
  installed distribution behaves correctly.
