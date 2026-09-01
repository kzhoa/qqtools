---
doc_type: adr
status: accepted
updated_at: 2026-09-01
archived_at:
adr_id: ADR-QEXP-0005
supersedes: []
superseded_by:
---

# ADR-QEXP-0005: Single-Host qexp Test Architecture

## Context

qexp coordinates shared Task truth and machine-local process authority through a filesystem,
locks, runtime roots, clocks, and child processes. Tests that combine all of those concerns make
business decisions slow to enumerate and can accidentally depend on an operator's running agent.

The project develops and validates qexp on one Linux host. It needs repeatable evidence for
deterministic decisions, shared-root process races, and the production one-agent-per-host lock
without claiming that those tests certify a particular network filesystem.

## Decision

qexp test evidence is divided into three execution profiles:

- `hermetic` tests receive an isolated authority namespace and may run in the fast gate. Pure
  decision reducers, reference-model scenarios, simulated protocol interleavings, and ordinary
  qexp integration tests belong here when they do not require the production global lock.
- `machine_lab` tests run independent Python participants. Each participant receives a frozen
  `TMPDIR`, HOME, XDG root, runtime root, and tmux root before importing qexp; participants share
  only the test Project's `.qexp` root. These tests prove single-host shared-root protocol races.
- `host_exclusive` tests intentionally use the production default temporary-directory authority
  path. They run serially in the merge gate. Local runs skip when another qexp agent holds that
  lock; CI fails instead, because a designated runner must be free of external agents.

The test-only architecture exposes serializable trace envelopes, a reference model, deterministic
protocol dispositions, and a narrow participant control plane. It does not add a production
configuration switch that weakens scheduler authority.

`qexp-fast` runs deterministic unit and hermetic integration evidence. The merge gate runs the
machine-lab and host-exclusive profiles. The verification matrix links runtime crash windows to
decision and local-protocol evidence.

## Consequences

Business rules can be exhausted and replayed without file or process timing, while real
filesystem, process, and authority behavior remains covered at the boundary where it matters.

Machine-lab participants are more expensive than in-process mocks and require explicit cleanup,
timeouts, and trace diagnostics. Host-exclusive evidence cannot be reliably run on a developer
machine with a live agent, so local skips are expected and CI runner hygiene is required.

This evidence proves qexp's protocol among independent local processes on one POSIX filesystem.
It does not certify NFS, Lustre, or another cross-host filesystem's cache, lock-service, or
failure semantics.

## References

- [qexp Runtime Spec](../../spec/qexp_runtime_spec.md)
- [qexp runtime verification matrix](../../../tests/CONTRACT_MATRIX.md)
- [ADR-QEXP-0001: Shared Filesystem Coordination](0001-shared-filesystem-coordination.md)
