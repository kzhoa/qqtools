---
doc_type: adr
status: active
updated_at: 2026-08-29
archived_at:
---

# Remove Filesystem Qualification Gate

## Decision

qexp no longer requires `.qexp/project/filesystem-qualification.json` before a machine can
claim, authorize, or resume a cross-machine Attempt. Project initialization and machine-agent
registration also ignore malformed or absent legacy records. The qualification probe APIs and their
persisted record format are removed.

## Rationale

qexp is a lightweight experiment queue. Requiring deployment-specific qualification evidence made
ordinary multi-machine use depend on an additional stateful operational procedure. Existing shared
record locking, atomic replacement, Task revision checks, claim fencing, and local GPU reservations
remain the concurrency protections for normal operation.

## Consequences

- Stale qualification files have no effect on qexp scheduling and may be removed by operators.
- qexp does not certify filesystem semantics before cross-machine use; operators remain responsible
  for choosing a shared filesystem compatible with qexp's documented coordination primitives.
- The prior qualification decision is superseded where it made cross-machine execution conditional
  on recorded evidence.
