---
doc_type: adr
status: accepted
updated_at: 2026-08-06
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

## Consequences

- Partial JSON is never published because writes use a temporary file and `os.replace`.
- Lock age is never used as proof of ownership or as a stale-lock repair decision.
- A shared filesystem with delayed lock visibility, broken `flock`, or non-atomic replace is
  unsupported for cross-machine claims.
- Claim and fencing tests can inject a clock and use the same record store without a
  central coordinator.
