# qexp binding working set

The machine agent keeps a process-local working set for each registered binding.
Residency is keyed by `(runtime_id, project_id, registration_generation)`, so a
Project can be resident on one machine and dormant on another. Dormancy releases
hot service state. It does not disable, unregister, drain, move, or revoke the
binding.

## Work and residency

Runtime work has three obligation classes:

- **active**: running or starting execution, actionable candidates, terminal
  collection, and immediate control operations;
- **waiting**: dependency, capacity, Group limit, deadline, retry, and recovery
  conditions represented by durable indexes or compact retry state; and
- **history**: terminal work whose execution, accounting, and cleanup fences are
  complete.

A resident binding participates in high-frequency service. A dormant binding
retains its identity, durable consumer progress, activation checkpoint, and
recoverable waiting records, but does not retain a Project supervisor or Group
source session. Running processes, reservations, launch handoffs, incomplete
source traversals, local progress mailboxes, unresolved recovery, or unreadable
state keep the binding resident.

## Recoverable activation publication

Project activation state is stored under
`operations/project-activation-v1/`. The checkpoint contains a random nonzero
epoch and a sequence that increases within that epoch. Every committed sequence
has an event in `events/<epoch>/`. Supported current writers cover ready-route
changes, Group locator changes, offer/retry/recheck deadline changes, observation
rebuild requests, and the submission and dependency transitions that update
those indexes.

A writer holds `locks/project-activation-v1.lock` and performs this transaction:

1. durably write `pending.json` and its exact event;
2. commit the covered authoritative index or request;
3. replace `checkpoint.json` with the same event identity; and
4. remove the pending record.

Recovery completes any prepared event before another writer advances the
sequence. A missing event in an existing journal is corruption and leaves demand
unknown. A root created before the journal format may rotate once into a new
reconstruction epoch only when the complete event directory is absent. Activation
selects work for reconciliation; claims still validate authoritative Task,
registration, ownership, and eligibility state.

Older writers that do not publish a compatible activation remain covered by the
bounded periodic cold reconciliation described below. That fallback has a longer
latency envelope and cannot authorize borrow admission while the binding is
omitted.

## Per-consumer handoff

Each exact consumer stores fenced progress at
`consumers/<runtime-id>/<registration-generation>.json`. A process restart
re-fences the same generation and preserves only same-epoch progress. A new
registration generation starts without an acknowledgement. Retired generations
cannot register or acknowledge again.

Five service lanes participate in the handoff: `scheduler`, `authority`, `group`,
`observation`, and `submission`. A lane acknowledges quiescence only after its
bounded reconciliation has persisted any active or waiting obligation. A binding
becomes dormant only after all five lanes acknowledge the same checkpoint and
the shared consumer cursor is durably advanced. A checkpoint change during the
final handoff rejects dormancy.

Local handoff state is stored in
`<machine-runtime>/projects/<project-id>/working-set-v1.json`. It is disposable
coordination metadata. Restart always begins resident and repeats reconciliation;
missing or corrupt local progress leaves work unknown.

Binding removal first writes a machine-local retirement intent, commits registry
removal, and then retires the exact shared consumer generation. Recovery rechecks
registry absence under the registry lock before retirement. Re-registration
performs a direct recovery barrier for the target generation, so it cannot reuse
a generation whose retirement is pending. Shared-root disappearance preserves
the intent for retry.

## Snapshot compaction and reconstruction

`snapshot.json` declares a retention floor, the checkpoint sequence captured for
that pass, the activation epoch, the consumer-membership revision, and coverage
of all five authoritative active/waiting index lanes. `membership.json` advances
when an exact consumer is created or retired. Consumer membership mutation and
snapshot creation use the same activation lock.

Compaction advances by at most 256 events. It writes and syncs the new snapshot
before deleting the covered prefix, records the previous floor until deletion is
complete, and resumes an interrupted deletion idempotently. Publication starts a
bounded compaction pass when the retained suffix reaches 256 events. A corrupt or
mismatched snapshot fails closed.

A consumer below the floor cannot acknowledge the current tail directly. After
all five lanes durably reconcile the authoritative indexes, it reconstructs at
the exact snapshot floor, replays the retained suffix in batches of at most 256,
and then acknowledges. This applies to offline, new, and locally damaged
consumers. Repeated compaction does not retire an offline consumer by age or
timeout.

The durable maintenance descriptor protocol is owned by the separate resumable
index-maintenance design. No descriptor producer is introduced by this feature.
When such producers are added, their prepare, identity, progress revision,
retirement, and successor-generation records must be committed through this same
activation transaction and remain covered by the snapshot's active/waiting index
reconstruction. An activation acknowledgement transfers discovery responsibility;
it does not complete or retire maintenance work.

## Running, startup, and stopped guarantees

On initial start and readiness, the agent validates every captured enabled entry
and current registration authority. This O(N) cold pass is preserved by design.
It restores obligations from durable indexes without scanning retained Task
history.

During steady state, an immutable registry tuple is reused while the file witness
and revision are unchanged. Working-set reconciliation then returns without
walking registered bindings. Resident loops enumerate resident bindings only.
The scheduler cold-polls at most four enabled dormant bindings per cycle using a
fair rotating roster. At that budget, the lost-notification revisit envelope is
`ceil(dormant_enabled_bindings / 4)` scheduler cycles plus filesystem latency.
An unchanged dormant binding is also reconciled after 60 seconds when selected,
which covers compatible older writers and lost activation.

A stopped on-demand agent has no remote wake promise. Activation, snapshots,
consumer progress, and authoritative indexes survive shutdown. The next explicit
start performs the full readiness and initial reconciliation contract before it
can report ready or return to dormancy.

## Admission and resource bounds

Dormancy cannot prove absence of primary demand. If any enabled binding is
dormant, or any activation/index state is unreadable, a new borrow claim is
denied. Existing borrow Attempts continue under their normal ownership fences.
The dormant-enabled summary is maintained with the roster, so this guard does
not reread every dormant Project.

The recurring budgets are four dormant checkpoint probes per scheduler cycle,
64 dormant registration renewals per heartbeat, scheduled early enough to
cover the bounded roster revisit horizon, 16 machine-local progress
mailbox probes per progress cycle, 16 authority bindings per authority slice,
256 activation replay events per acknowledgement, and 256 events per compaction
pass. Group discovery retains its existing caps of 64 resident entries and 256
descriptors and closes binding-owned sessions when the resident set changes.
Compact binding state and the registry remain O(N) in registered bindings;
shared Project truth reads, supervisors, open descriptors, and hot service scale
with resident work plus the fixed cold budgets.

Measured scale and replay evidence is recorded in the
[working-set acceptance report](qexp_working_set_acceptance.md).
