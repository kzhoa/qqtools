---
doc_type: adr
status: active
updated_at: 2026-08-29
archived_at:
---

# Shared Filesystem Coordination

## Decision

Schema 6 separates filesystem serialization from time-based authority. Cross-host locks,
atomic replacement, Task CAS, and fencing serialize an initial claim even when a host has no
qualified wall clock. Lease expiry and Recovery additionally require independently bounded UTC
clock observations from holder and reclaimer. A holder-bound claim has no time-based remote
takeover path; availability is sacrificed rather than risking duplicate training. Target
filesystems require real multi-host flock, rename, and fsync qualification before use for
cross-machine authority.

The schema-5 qexp runtime uses atomic JSON replacement for shared records, exclusive
creation for idempotency mappings, and short-lived `fcntl.flock` locks for local process
serialization. Authoritative Task updates carry a monotonically increasing revision and
must be fenced by the active claim token.

The first implementation qualifies these primitives only for one host or a shared
filesystem that preserves POSIX lock and atomic-replace semantics. Cross-host dispatch
must be enabled only after the deployment operator runs a two-host qualification probe.
The two participants must have distinct identities and exercise the same mounted control root.
The probe must prove exclusive-lock contention, whole-value atomic replacement, file and directory
`fsync` visibility, and safe process-failure cleanup. Missing or failed evidence is not qualified.
Results describe only that deployment and never tune repository-wide scheduling defaults.

The most recent decision and its evidence are persisted atomically at
`.qexp/project/filesystem-qualification.json`, bound to the project identity and canonical shared
root. A failed requalification replaces and revokes an earlier success. Missing, malformed,
mismatched, or failed evidence is unqualified. Initialization and project
registration reject an invalid record when one is present. A same-machine claim does not require
this record; every claim whose execution machine differs from the Task home machine revalidates it
before reserving local GPUs and again while holding Task authority. Launch authorization and
`starting` recovery also revalidate it, so failed requalification prevents an unlaunched remote
Attempt from starting. There is no boolean override for cross-machine dispatch.

## Consequences

- Partial JSON is never published because writes use a temporary file and `os.replace`.
- Lock age is never used as proof of ownership or as a stale-lock repair decision.
- A shared filesystem with delayed lock visibility, broken `flock`, or non-atomic replace is
  unsupported for cross-machine claims.
- Claim and fencing tests can inject a clock and use the same record store without a
  central coordinator.
- A single-host development run can test probe judgment logic but cannot certify cross-host
  filesystem behavior.
- Ready-index liveness, budgeting, and cutover decisions are recorded in
  [0002-ready-index-and-portable-work-budget.md](0002-ready-index-and-portable-work-budget.md).
