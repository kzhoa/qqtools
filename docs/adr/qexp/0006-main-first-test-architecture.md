---
doc_type: adr
adr_id: ADR-QEXP-0006
status: accepted
updated_at: 2026-09-01
archived_at:
supersedes: [ADR-QEXP-0005]
superseded_by:
---

# ADR-QEXP-0006: Main-first qexp Test Architecture

## Context

qexp needs isolated local-process evidence without colliding with a developer's real agent. The
former three execution profiles conflated test boundary and scheduling lane.

## Decision

All qexp Integration tests receive isolated per-test resources and ledger-based cleanup checks.
`machine_lab` remains an Integration capability. The isolated authority algorithm is Integration;
the production default authority contract is a serial `host_exclusive` installed-wheel E2E.

## Consequences

Leaked registered resources fail their owning test. No Integration test uses production default
authority. Installed E2E is intentionally small and validates public CLI behavior.
