---
doc_type: adr
adr_id: ADR-QEXP-0010
status: accepted
updated_at: 2026-09-09
archived_at:
supersedes: []
superseded_by:
---

# ADR-QEXP-0010: Machine-Rolling Upgrade Coordinator

## Context

qexp projects are shared by several machine agents, while a machine may supervise several
registered projects. A persisted protocol change therefore cannot depend on a project-by-project
operator loop or on draining running Attempts. Agent restart must restore supervision immediately;
background conversion may make progress only when its protocol-specific safety proof permits it.

The existing schema6 upgrade is intentionally a drained transition and is not an online migration
implementation. The new coordinator is a permanent runtime capability for future persisted fields,
projections, capabilities, and protocols.

## Decision

1. Each protocol migration is a `MigrationPlugin` under
   `src/qqtools/plugins/qexp/runtime/upgrade/`. Its `MigrationSpec` declares source and target
   protocols, compatible readers and writers, applicable expand/readiness/backfill/audit/activation/
   contraction phases, dependencies, conflicts, work budget, interruption budget, operational
   level, and cleanup version. The plugin, not the coordinator, owns record semantics and writer
   safety proofs.
2. Each project stores one versioned journal below
   `.qexp/operations/upgrades/journal.json` and serializes progress with the project upgrade lock.
   Journal updates are atomic. A phase slice is idempotent, bounded, and resumable after a process
   crash. Activation additionally acquires the existing schema lock; ordinary backfill does not
   hold that lock and therefore does not turn migration I/O into a writer-quiescence barrier.
3. The machine agent discovers only registered roots and maintains a fair, bounded cursor. It
   performs protocol/journal metadata checks before admission and starts a worker only for a
   pending runnable migration. Waiting, paused, repair-required, and completed migrations release
   the worker; no history scan or migration timer is retained for an idle project.
4. Machine status reports the local discovery boundary, inaccessible roots, per-project phase and
   blockers, package/agent state, and migration state separately. `qexp agent restart` is the
   normal L1 trigger. `qexp agent upgrade coordinate` is an exceptional all-registered-root
   recovery surface; it does not claim roots absent from the local registry.
5. A migration may block admission only when its journal explicitly records that the current write
   path is unsafe or indeterminate. A failed migration with a provably safe old path keeps
   compatible service available. The coordinator never fabricates terminal truth, releases a
   reservation, or changes an Attempt identity.
6. Pause and repair are explicit project-scoped operations. A repair plan must identify observed
   revisions, evidence, intended changes, snapshot proof, and validation. Apply, validation, and
   resume are durable and idempotent; stale plans and unsupported authoritative repairs are
   rejected. The exceptional coordinator uses these same rules.

The lock order for coordinator work is machine scheduling/registry lock, project upgrade lock,
then schema lock only for activation or an explicitly declared repair. Cross-project locks are not
used as migration authority.

## Consequences

The normal rolling command sequence can deploy a new agent without waiting for full backfill or
running Tasks to finish. Multiple capable agents converge on one project journal, and multiple
projects share a bounded machine budget with fair progress. Operators gain a machine-level status,
retry, pause, repair, validation, resume, and recovery surface.

The framework does not make an arbitrary conversion safe. A production migration must still
declare its released source matrix, inventory every relevant writer, prove concurrent-update
handling, test active training and terminal evidence, and establish numerical shared-storage
budgets. Cross-host filesystem behavior and the first real persisted reader/writer migration remain
deployment evidence requirements. The built-in `upgrade-journal-v1` migration installs
coordinator-owned metadata only and deliberately leaves `schema/version.json` unchanged so 1.3.16
strict readers remain compatible. It validates the framework path but does not satisfy participant
readiness or writer-fencing evidence for an incompatible shared protocol activation. The existing
drained schema6 transition remains outside this coordinator.

## References

- [Machine rolling upgrade coordinator pitch](../../pitch/qexp-machine-rolling-upgrade-coordinator.md)
- [Compatibility governance](../../spec/compatibility-governance.md)
- [Upgrade and ready-index recovery guide](../../spec/qexp_upgrade_guide.md)
