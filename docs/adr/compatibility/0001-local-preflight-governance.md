---
doc_type: adr
status: active
updated_at: 2026-08-31
archived_at:
---

# Govern Compatibility Through the Local Release Preflight

## Context

qqtools development merges locally into `main` and publishes by following `.github/publish.md`.
It does not use a pull-request merge gate. Compatibility windows may overlap across components and
releases, so pitches and relative N/N+1/N+2 labels cannot serve as a durable release authority.

## Decision

qqtools records every temporary compatibility contract in the tracked
`docs/spec/compatibility-registry.toml`. Each item has an independent ID, exact release versions,
an explicit lifecycle state, verification paths, and a source/test marker. A standard-library
checker validates the registry and calculates the state required for a target release.

The registry contains only unfinished compatibility items. Completion is represented by deleting
the item at `transition_purged_in`, not by retaining a permanent `completed` record. A monotonic
top-level `next_id` prevents identifier reuse. During release planning and checking, the checker
compares the candidate with the registry stored at the exact previous release tag; this rejects
premature deletion, remaining markers, `next_id` rollback, and reuse of retired IDs. Tags that
predate the registry form an explicit bootstrap boundary. Git history, release tags, and the
CHANGELOG retain completed history.

Planned items also carry one or more local `pitch_refs` so later agents can find their deferred
implementation requirements. These references are limited to existing Markdown files under
`docs/pitch/`, are not required to be Git-tracked, and are removed with the registry item. They are
implementation navigation only; tracked `decision_refs` remain the formal behavior authority.

The only hard compatibility gate is the local command:

```bash
python scripts/release_preflight.py --target-version X.Y.Z
```

The preflight requires a clean committed candidate, validates that the target is later than the
current source version, enforces all compatibility obligations for that target, and only then runs
the expensive regression, build, and installed-wheel checks.

The tag-triggered publish workflow deliberately does not repeat the compatibility gate. It trusts
that the operator followed the local release procedure. Local `.codex` guidance may help register,
review, and retire compatibility items, but it is ignored by Git and is not an authority source.

## Consequences

- Compatibility windows from different releases can overlap without sharing one global N.
- The active registry remains bounded by unfinished work rather than growing with project history.
- Ordinary compatibility work uses the registry; only high-risk migrations need a dedicated pitch
  or ADR.
- Skipping local preflight can bypass compatibility governance. This is an accepted consequence of
  the repository's trusted local-publish model.
- Registry and checker changes are tracked and reviewable even though local Codex instructions and
  active pitch documents are not.
- A fresh clone without the local pitch set cannot pass registry validation or release preflight
  for an item that references those pitches. This is accepted within the trusted local-publish
  model.
- Deadline extensions require an append-only decision record rather than silently changing the
  original release targets.

## Rejected Alternatives

- **Pitch-only governance.** Active pitches are Git-ignored and cannot enforce overlapping release
  obligations.
- **Codex-only governance.** Local instructions are advisory and unavailable to deterministic
  release execution.
- **Pull-request compatibility CI.** The project does not publish through a PR merge workflow.
- **Tag-time compatibility enforcement.** It conflicts with the accepted local-publish trust
  boundary and would duplicate the preflight decision after the release tag is created.
- **Permanent completed registry entries.** They make an operational register grow as historical
  storage and duplicate the immutable evidence already retained by release tags and Git history.
