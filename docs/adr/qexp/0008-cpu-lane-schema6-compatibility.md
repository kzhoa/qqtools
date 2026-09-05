---
doc_type: adr
status: superseded
updated_at: 2026-09-05
archived_at:
adr_id: ADR-QEXP-0008
supersedes: []
superseded_by: ADR-QEXP-0009
---

# ADR-QEXP-0008: Introduce CPU Lanes Through a Bounded Schema-6 Transition

> Superseded by [ADR-QEXP-0009](0009-staged-cpu-lane-rollout.md). The original single-release
> activation schedule is retained below as historical context.

> **Schedule clarification (2026-09-05):** The historical 1.3.17 retirement date below is not
> the effective release plan. The effective three-version path is legacy roots before 1.3.15,
> compatibility activation in 1.3.15, and retirement in 1.3.16. The compatibility registry is
> authoritative for the effective deadline.

## Context

ADR-QEXP-0007 defines independent GPU and CPU-only lanes. CPU support changes Task, Attempt and
ready contracts, but need not change the global schema number. qqtools already governs temporary
readers and migrators through its compatibility registry. Existing installations need a release
window to normalize their persisted GPU data.

Existing schema-envelope validation rejects unknown fields, but the writer-capability list check
does not reject every unknown capability. Appending to that list alone cannot exclude old agents.

## Decision

Retain schema 6 and introduce canonical lane records under a permanent root required-capability
gate. Register the temporary legacy reader and upgrader as QQTOOLS-COMPAT-0005: introduce support
in 1.3.15, remove ordinary legacy parsing in 1.3.16, and plan remaining transition removal for
1.3.17. Exact deadlines and approved extensions belong to the compatibility registry.

1.3.15 reads legacy and canonical records but writes only canonical records after root activation.
1.3.16 reads legacy records only inside the upgrade boundary. 1.3.17 requires a canonical or fresh
root; older installations must pass through a transition release. All three releases use the same
canonical wire contract. The canonical protocol and its required-capability gate remain permanent.

Activation uses a durable feature-transition marker and resumable normalization, not schema-7
migration. Before canonical publication, exclude old writers and drain the target project's active
execution. Normalize affected durable history as well as queued work and rebuild ready state.
Never remove the gate automatically after canonical writes have begun.

This decision complements ADR-QEXP-0007 without replacing its isolation decision or introducing
CPU requests for GPU Tasks. Detailed release and activation behavior belongs to the linked spec.

## Consequences

- Old GPU data has a bounded upgrade path without replacing the entire schema-6 control root.
- New readers accept old data during the window; pre-1.3.15 binaries cannot operate on an activated
  root. Data compatibility does not authorize arbitrary mixed-version execution.
- Only temporary legacy parsing, upgrade machinery, warnings and fixtures are retired. Canonical
  validation and the capability gate remain after the registry item is removed.
- Skipping the window requires using 1.3.15 or 1.3.16 before upgrading to 1.3.17. Postponement uses
  append-only registry extensions and a recorded decision.
- CPU recovery, old-process exclusion, historical normalization and interrupted activation still
  require implementation and verification; registry checks alone do not establish runtime safety.

## References

- [ADR-QEXP-0007](0007-cpu-only-lane-isolation.md)
- [CPU lane compatibility contract](../../spec/qexp_cpu_lane_compatibility.md)
- [Compatibility governance](../../spec/compatibility-governance.md)
- [Compatibility registry](../../spec/compatibility-registry.toml)
- [CPU-only Task Lane pitch](../../pitch/arxiv/053-qexp-cpu-only-task-scheduling.md)
