---
doc_type: spec
status: active
updated_at: 2026-09-06
archived_at:
---

# qexp Submission Schema-Lock Compatibility Transition

`QQTOOLS-COMPAT-0009` tracks the temporary wide schema-lock path while roots adopt the
group-ready-members protocol. This specification is the behavior authority for the transition;
the compatibility registry is authoritative for lifecycle dates.

## Lock protocol

Legacy roots use the existing exclusive schema lock for the whole submission and cleanup identity
workflow. A root may select the narrow protocol only when it requires
`group-ready-members-v1` and `indexes/ready/group-members/state.json` contains
`group_ready_members.state: active`.
Missing, malformed, or unrecognized capability/state evidence does not select the narrow path and
must not leak a storage-shape exception: the writer retains the legacy exclusive schema fence.

The narrow protocol holds a shared schema lock for each authoritative writer section. Schema and
capability mutations continue to take the exclusive side of that same lock, so they cannot
interleave with an in-flight Task or Group writer. Submission additionally holds its keyed
idempotency lock, then its Group lock when applicable, then all Task locks in sorted Task-ID
order. Scheduler lifecycle, availability, Group control, dependency edits, and cleanup use shared
schema, Group, and Task locks in the same order.
Ready-index build and repair use `Schema -> ready-index state -> Group -> Task`; a state-lock
holder never waits to acquire the schema fence. Doctor's terminal pending-commit repair rereads
and validates Group truth inside its `Schema -> Group` writer section before clearing it.

The idempotency mapping is the winner record for an idempotency key; Task truth is the winner
record for a Task identity; cleanup operation truth is the permanent tombstone; and Group pending
submission commit is the winner record for sequence reservations and inactive Worker additions.
Every loser reads and verifies the applicable record. A failed operation removes only its own
uncommitted staged Tasks and inactive additions.

## Release contract

| Release | Ordinary root behavior | Temporary behavior |
| --- | --- | --- |
| 1.3.16 | Active member-projection roots use the narrow protocol | Legacy roots retain the wide lock |
| 1.3.17 | Only narrow protocol permits ordinary mutation | Restricted upgrader/repair may activate legacy roots |
| 1.3.18 | Narrow protocol only | Remove the marker, legacy path, and fixtures |

`QQTOOLS-COMPAT-0008` and `QQTOOLS-COMPAT-0009` are jointly active in 1.3.16. New roots create
the canonical member projection immediately. Existing roots use `qexp upgrade group-ready-members`
to build and audit it under the wide schema fence; the atomic `active` state transition is also the
only point at which the narrow writer protocol becomes selectable.

## Verification boundary

The integration matrix exercises the production capability, build, audit, and active-state gate.
It verifies that legacy roots retain the exclusive fence during construction, and that the first
active member projection selects the shared schema writer fence without permitting mixed writers.
