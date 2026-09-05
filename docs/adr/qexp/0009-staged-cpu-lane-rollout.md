---
doc_type: adr
status: accepted
updated_at: 2026-09-05
archived_at:
adr_id: ADR-QEXP-0009
supersedes: [ADR-QEXP-0008]
superseded_by:
---

# ADR-QEXP-0009: Activate Existing Roots with the 1.3.15 CPU Lane Release

## Context

CPU-only scheduling is only useful if existing schema-6 roots can activate it in the release that
introduces the feature. The activation must still be explicit, drained and recoverable.

## Decision

1.3.15 introduces `qexp upgrade cpu-lane` for drained legacy schema-6 roots. The operator starts
one durable activation session, obtains machine attestations, and resumes canonicalization; normal
clients are blocked while the session prepares. Fresh roots remain canonical from initialization.

The compatibility window spans 1.3.15, 1.3.16 and 1.3.17. Ordinary legacy parsing is removed in
1.3.16; the restricted upgrader and its transition code are removed in 1.3.17. Schema 6 remains
unchanged, and CPU capacity remains machine-local policy rather than Project truth.

## Consequences

- Existing projects can use CPU-only Tasks as soon as they complete the explicit 1.3.15 activation.
- The runtime temporarily carries both legacy GPU and canonical lane codecs, governed by
  `QQTOOLS-COMPAT-0005`.
- The compatibility code has a bounded three-release lifetime.

## References

- [ADR-QEXP-0007](0007-cpu-only-lane-isolation.md)
- [CPU-only Task Lane pitch](../../pitch/qexp-cpu-only-task-scheduling.md)
