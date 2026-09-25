# qexp resumable maintenance

qexp separates routine recovery, explicitly targeted repair, and full-history
audit. Routine service consumes durable active obligations and affected
projection regions. It does not enumerate retained Task or operation history.
A targeted repair certifies only its declared object or partition. A full audit
may read all retained authoritative truth, but its capture and audit cursors are
durable and every invocation is bounded.

## `admin repair` compatibility contract

`qexp --project PATH admin repair` remains the explicit whole-Project repair
operation. The resumable implementation does not silently narrow an existing
successful repair to active records. A new invocation either creates a durable
full-audit request or attaches to the current unfinished request. Completion
certifies every phase in that request's captured scope; writes after its capture
watermark are successor work.

`--max-work-items` retains its integer range of 1 through 64. It limits semantic
items across the complete invocation, rather than granting that allowance to
each nested phase. A source record costs one item before it is examined,
including skipped, invalid, terminal, and replayed records. A durable capture,
empty-phase transition, descriptor initialization, or completion proof also
costs one item when it advances persisted work.

A partial non-strict invocation retains exit status 0 for compatibility and
sets `complete=false`, `outcome=partial`, and `rerun_required=true`. With
`--strict`, the same incomplete result exits 1. A genuine blocked repair exits
1 in either mode. Human output never describes a partial slice as a completed
repair.

JSON results retain the established repair fields and add:

- `scope.kind`, `scope.capture_id`, and `scope.work_generation`;
- semantic-item, filesystem-operation, elapsed-time, remaining-budget, and
  exhaustion evidence under `budget`;
- the current `phase` and phase-specific `cursor`;
- a bounded `deferred_phases` page and its total count;
- `next_due_at`; and
- `remaining_work`, which is `null` when no captured inventory supports an
  estimate.

Remaining budget is not an estimate of remaining work.

## Invocation ledger

One ledger starts before descriptor resolution and covers setup, capture,
operation reconciliation, projection validation and rebuild, and final status.
The default limits are the requested semantic-item count, 256 initiated
filesystem operations, and a 50 ms cooperative admission deadline. Context
resolution, bounded descriptor loading, lock handling, one admitted checkpoint
commit, and bounded rendering cost no semantic items, but still consume the
operation and elapsed-time budgets.

Before a semantic step starts, it reserves the maximum operations required for
its safe checkpoint. Unused reservations are released. After the deadline, no
new semantic step starts; an admitted step may finish its reserved commit and
report an overrun. This is not a hard syscall timeout. Indefinitely blocked I/O
requires separate execution isolation.

An N=1 invocation can advance every phase over repeated calls. It may initialize
one descriptor, examine one source record and checkpoint its cursor, advance one
empty phase, or commit one completion proof. No phase runs an unmetered
transition loop, and ready-index reconstruction no longer loops to completion
inside one repair invocation.

## Descriptor identity and persistence

The shared descriptor identity is
`(project_id, kind, target_id, work_generation)`. Machine-local work additionally
includes runtime and registration generation. There is no maintenance authority
epoch. The Project activation epoch and sequence identify wake transport only.

Operation-backed work uses its immutable operation ID as both `target_id` and
`work_generation`. Repeatable projection repair and audit requests allocate one
maintenance-only request token with the existing identifier allocator. Tokens
have no ordering meaning. A current-target pointer and durable retirement or
supersession evidence fence older work.

Descriptors live under `operations/maintenance-v1/` and contain their identity,
source capture or revision, applicable projection/build generation, phase,
cursor, monotonic progress revision, last meaningful progress, due time, retry
count, sanitized failure evidence, and lifecycle state. Descriptor records are
bounded to 64 KiB. Missing or untrusted identity is unknown work requiring
scoped reconstruction; it never means complete.

The JSON identity index, active or retired descriptor, and immutable queue-slot
record are authoritative recovery evidence. `queue-index.sqlite3` is a derived,
ordered index of active queue slots only. It is protected by the maintenance
outbox lock, uses full synchronous commits, and may be reconstructed in bounded
steps from JSON queue slots and active descriptors. Corrupt SQLite state is an
explicit recovery condition; it does not authorize a history scan or an empty
queue result. Queue metadata persists rotation, the remaining entries in the
current cycle, and the earliest delayed due time, so retired history does not
increase selection cost.

Descriptor preparation allocates its queue position, writes immutable JSON
reconstruction evidence, registers the derived active slot, writes the active
descriptor, and publishes its identity index before the authoritative mutation.
A prepared descriptor is not executable until the matching truth mutation is
provable. After truth commits, activation publishes a recoverable Project wake
transaction before advancing the descriptor revision. Its bounded reason is
`mw:<descriptor-identity-sha256>:<progress-revision>`. A crash before explicit
activation is recovered from kind-specific truth evidence; a descriptor with
no such proof remains prepared and retries rather than applying an uncommitted
intent.

Agent and CLI advancement use the existing authority locks plus current-work
and projection/build fences. Effects and replay evidence commit before cursor
advancement. If an indexed active descriptor loses its latest JSON progress,
the immutable initial queue slot supports fail-closed intervention; recovery
never resets it to the initial cursor. Terminal retirement writes the retired
descriptor first, flips the identity index second, then removes active queue
evidence. Recovery adopts a terminal descriptor left before the index flip.

Completed or superseded evidence remains until every relevant activation
consumer permits compaction, or an authoritative checkpoint retains outstanding
descriptors and retirement fences. A stale notification cannot reopen a retired
generation.

## Retry and service rules

Routine resident service gives maintenance a turn independently of free CPU or
GPU capacity. It rotates due Projects, then due descriptors within each Project;
new arrivals and retries join the tail and cannot overtake already due work. One
turn advances at most one descriptor with the same ledger primitive used by the
CLI. A complete pass over active descriptors with none due returns the persisted
earliest `next_due_at` instead of keeping the Project hot.

Transient failure k uses a capped exponential base
`min(60 seconds, 1 second * 2^k)` and persists one jittered due time in
`[0.8 * base, base]`. Restart does not redraw an already persisted retry.
Meaningful committed progress resets k. Malformed authority, incompatible
schema, ambiguous ownership, and unrecoverable storage evidence enter
intervention instead of retrying forever.

The full phase set covers Submission, cleanup, availability, Group cancellation,
deadline indexes, orphan recovery, ready audit/build, group-ready members, Task
observation, Submission control, and final verification. Phase rotation and
per-phase cursors persist across CLI calls and agent restarts.
