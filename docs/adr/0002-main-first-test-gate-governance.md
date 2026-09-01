---
doc_type: adr
adr_id: ADR-0002
status: accepted
updated_at: 2026-09-01
archived_at:
supersedes: []
superseded_by:
---

# ADR-0002: Main-first Test Gate Governance

## Context

The repository develops directly on main. Repeating source Unit and Integration tests in ordinary
post-push CI delays feedback without adding a distinct delivery boundary.

## Decision

Local `preflight` is the normal source gate and runs Unit plus selected Integration evidence.
Ordinary main and PR CI build a non-editable wheel and run only installed-artifact E2E or smoke
evidence. Release validation remains an explicit, broader upgrade. The lane checker enforces the
directory, marker, tox, and workflow boundaries.

## Consequences

Developers must run preflight before push. CI detects installed-product regressions rather than
repeating source evidence. New tests are placed by verification boundary, not CI job name.
