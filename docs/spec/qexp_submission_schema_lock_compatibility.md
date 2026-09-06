---
doc_type: spec
status: drafting
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

This change is intentionally unshipped while `QQTOOLS-COMPAT-0008` remains planned. The two
transitions must be released together; no current root can activate this branch by itself.

## Verification boundary

The integration matrix writes the future capability and active-state evidence and exercises the
real gate, while a test-only reader capability shim stands in for the unfinished 0008 reader.
It verifies the 0009 locking contract but is not evidence that a production root may activate.
Release activation, including the compatibility-registry lifecycle transition, remains blocked on
the 0008 projection writer, audit, and joint activation commit.
