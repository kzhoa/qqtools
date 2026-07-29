---
doc_type: spec
status: active
updated_at: 2026-07-24
archived_at:
---

# qexp Runtime Spec

## 1. Purpose and Authority

This document defines the target runtime contract for `qexp` after the breaking transition
to Experiment Group, flat Task, and Attempt truth.

It owns:

- shared and machine-local storage boundaries
- authoritative record shapes
- lock ordering and atomic transition requirements
- Submission Operation commit and idempotency mechanics
- home-first offer, claim, lease, fencing, and launch protocols
- local GPU reservation compensation
- crash recovery and `doctor` convergence
- schema validation and destructive cutover behavior

User-facing concepts, command names, and product behavior belong in
[qexp_product_spec.md](qexp_product_spec.md). This runtime spec may reference canonical
commands to identify a transition, but it must not define a competing CLI surface.

The rationale and delivery sequence are recorded in
[qexp-experiment-group-task-scheduling.md](../pitch/qexp-experiment-group-task-scheduling.md).

Where documents overlap:

- the Product Spec is authoritative for user-visible behavior
- this Runtime Spec is authoritative for persistence, concurrency, and recovery mechanics
- the pitch is non-normative rationale and delivery tracking

## 2. Status and Cutover

This is the target runtime contract. The installed implementation may still use the old
Batch and fixed-machine layout until the new schema is implemented.

The transition is intentionally destructive:

- no Batch-era truth is read or migrated
- no old Task, Batch, retry-lineage, index, claim, or operation record is imported
- no mixed-schema runtime is supported
- no legacy reader or compatibility repair path is provided
- an unsupported `.qexp` schema fails before any agent or mutating command starts
- users initialize a fresh control root for the new runtime

Loss of access to old scheduling metadata is a known and accepted product risk.

## 3. Deployment Model

### 3.1 Project Shared Root

One project has one shared control root:

```text
<project_root>/.qexp
```

The same root is visible to every participating machine. Group is a logical object inside
that root, not a separate filesystem control plane.

Unsupported truth layouts include:

```text
<project_root>/.qexp/<group>/
<project_root>/.qexp/groups/<group>/shared/
<project_root>/<experiment>/.qexp/
```

### 3.2 Shared Project Filesystem

The primary deployment assumes:

- project code and configurations are visible on all machines at the same absolute path
- `.qexp` truth is visible through the same shared filesystem
- machine identities are registered explicitly and are administratively trusted
- processes, PIDs, local launch backend state, GPU reservations, and wrapper state remain local

The runtime is primarily designed for machines where `tmux` is available and can serve as
the interactive launch backend. When `tmux` is unavailable, the runtime may launch local
detached processes directly as a degraded compatibility path. The runtime contract does not
guarantee equal observability, ergonomics, or performance for that non-`tmux` path.

qexp does not snapshot source code. A Task executes the files visible on its execution
machine at launch time.

### 3.3 Machine-Local Runtime Root

Each machine has a private runtime root, recommended as:

```text
~/.qqtools/qexp-runtime/<project-id>/<machine-name>/
```

It contains:

- agent PID and local lifecycle state
- provisional and active GPU reservations
- local Attempt process manifests
- wrapper control files
- optional tmux and log references
- machine-local recovery evidence

Local runtime files are never global scheduling authority.

## 4. Runtime Invariants

The implementation must preserve these invariants:

1. One project uses one schema version and one shared control plane.
2. Batch does not exist in new runtime truth.
3. Group, Task, Attempt, Machine, and Submission Operation have distinct records.
4. Task truth is authoritative for queue scope, current claim, fencing epoch, and current
   Attempt reference.
5. At most one claim is authoritative for a Task at a time.
6. At most one Attempt may be active for a Task at a time.
7. No process launches before the fenced `claimed -> starting` authorization commits.
8. A stale fencing token cannot update Task or Attempt truth.
9. A queued Task is claimable only when its Submission Operation is committed.
10. A private Task never becomes shared through an agent decision.
11. A running Attempt is never migrated by clearing its claim.
12. Group pause and cancellation linearize against launch authorization.
13. GPU reservation failures converge without permanently leaking qexp capacity.
14. Lease expiry after launch authorization never creates an automatic replacement.
15. Indexes, summaries, due-time indexes, and event projections are rebuildable data.

## 5. Coordination Primitives

### 5.1 Required Abstract Operations

The runtime requires a filesystem-backed implementation of:

```text
create_if_absent(path, value)
compare_and_swap(path, expected_revision, new_value)
atomic_replace(path, value)
acquire_exclusive(lock_path, owner, timeout)
release_exclusive(lock_path, owner)
```

Required semantics:

- exclusive creation has one winner across participating hosts
- a successful compare-and-swap validates and replaces one authoritative record
- readers never observe partial JSON
- lock ownership is not inferred only from wall-clock age
- all truth writes carry a monotonic object revision
- durable-write behavior is validated on every supported filesystem profile

Transition locks are short-lived serialization tools, not execution ownership. The
selected primitive must either release automatically when its holder process disappears
or provide a separately fenced recovery protocol. File existence alone must never be
treated as proof that a lock is still owned, and agents must not delete an apparently old
lock solely from wall-clock age.

### 5.2 Filesystem Qualification Gate

**Assumption / Unverified**:

- the target shared filesystem provides a validated cross-host exclusive primitive
- atomic replacement is visible consistently enough for the documented protocols
- stale-client, partition, and delayed-visibility behavior is bounded and tested
- participating machines maintain bounded clock skew through time synchronization

Cross-machine dispatch remains disabled until a dedicated ADR selects the primitive and
the target filesystem qualification suite passes. A process-local mock or single-host
test is not sufficient evidence.

### 5.3 Lock Order

All implementations use this global lock order:

```text
project schema lock
  -> Group control lock
       -> Task lock, ordered lexicographically by task_id
```

Rules:

- a grouped Task control operation acquires Group then Task
- an ungrouped Task operation acquires only the Task lock
- operations that create, permanently tombstone, or delete Task identity acquire the project
  schema lock before any Group or Task lock
- a multi-Task operation acquires Task locks in sorted `task_id` order
- no code path may acquire a Group lock while holding a Task lock
- machine-local reservation locks are outside the shared lock order and must not be held
  while waiting indefinitely for a shared lock

The Group lock linearizes dispatch barriers, membership sequence allocation, Worker Set
changes, and grouped launch authorization. The Task lock linearizes claim, offer, retry,
cancel, lease, and current-Attempt transitions.

## 6. Shared Directory Layout

The target layout is:

```text
<shared-root>/
  schema/
    version.json
  project/
    identity.json
  groups/
    <group-name>.json
  tasks/
    <task-id>.json
  attempts/
    <task-id>/
      <attempt-number>.json
  operations/
    submissions/
      <operation-id>.json
    group-control/
      <operation-id>.json
    cleanup/
      <task-id>.json
  idempotency/
    submissions/
      <key-digest>.json
  claims/
    archive/
      <task-id>/
        <fencing-token>.json
  machines/
    <machine-name>/
      machine.json
      state/
        agent.json
        gpu.json
        summary.json
      events/
        <date>/
          <event-id>.json
  locks/
    schema.lock
    groups/
      <group-name>.lock
    tasks/
      <task-id>.lock
  events/
    <date>/
      <event-id>.json
  indexes/
    tasks-by-state/
    tasks-by-group/
    offer-deadlines/
```

Layout rules:

- object truth is organized by object type, not by experiment directory
- active claim truth is embedded in the Task record, not split across machine-private
  claim directories
- archived claim records are immutable audit history
- machine state is writable only by that machine, except repair metadata explicitly
  written by `doctor`
- indexes and summaries may be deleted and rebuilt without losing truth

## 7. Common Record Envelope

Every mutable shared truth record uses:

```yaml
meta:
  schema_version: int
  revision: int
  created_at: str
  updated_at: str
  updated_by:
    actor_type: cli | agent | doctor
    machine_name: str
    process_id: str
```

Rules:

- `revision` starts at one and increases on every accepted mutation
- the object identifier inside the record matches its path
- timestamps use UTC RFC 3339 form
- timestamps support observation and soft deadlines; revisions and locks decide ordering
- unknown required schema fields fail validation
- writers must not silently discard unknown fields from a newer schema

## 8. Authoritative Object Contracts

### 8.1 Schema and Project Identity

`schema/version.json` contains:

```yaml
schema:
  name: qexp-runtime
  version: int
  minimum_reader_version: int
  created_at: str
```

`project/identity.json` contains a stable project ID and canonical shared-root identity.
Every local runtime root is namespaced by that project ID so two projects using the same
machine name cannot share reservations or process manifests.

### 8.2 Group Truth

Group truth lives at:

```text
groups/<group-name>.json
```

Required logical shape:

```yaml
group:
  name: str
  admission_state: open | sealed
  dispatch_state: active | paused
  dispatch_epoch: int
  worker_set_epoch: int
  next_membership_sequence: int
  pending_submission_commit: null | object
  worker_set:
    <machine-name>:
      state: active | draining | removing
      state_epoch: int
      added_at: str
      added_by_operation: str | null
      drain_requested_at: str | null
      remove_requested_at: str | null
      terminate_running: bool
  cancellation_barriers:
    - operation_id: str
      membership_high_watermark: int
      terminate_running: bool
      created_at: str
```

Group rules:

- Group membership is derived from Task truth
- each grouped Task receives one immutable `group_membership_sequence`
- membership sequences are allocated only during commit under the Group lock
- `pending_submission_commit` reserves one operation's sequence range and Worker Set
  changes while its cross-file commit is being completed
- every Group-lock operation must finish or abort an existing pending submission commit
  before performing unrelated Group control
- cancellation captures the current membership high watermark
- Tasks committed later have a higher sequence and are outside that cancellation snapshot
- pause increments `dispatch_epoch`
- every Worker Set state change increments `worker_set_epoch` and records the resulting
  value as that worker's `state_epoch`
- Worker Set state is authoritative eligibility input, not a display-only summary
- removed machines remain auditable through events or tombstone history

### 8.3 Task Truth

Task truth lives at:

```text
tasks/<task-id>.json
```

Required logical shape:

```yaml
task:
  task_id: str
  group_name: str | null
  group_membership_sequence: int | null
  submission_operation_id: str | null
  name: str | null
  spec:
    command: list[str]
    working_directory: str
    requested_gpus: int
  placement_policy:
    home_machine: str
    sharing_mode: private | spillover
    fallback_constraint: group | list[str]
    offer_after_seconds: int | null
  placement_runtime:
    queue_scope: home | shared
    queued_home_at: str | null
    offer_eligible_at: str | null
    offered_at: str | null
    offer_reason: manual | elapsed | null
    offered_by: str | null
  state:
    projection: queued | running | succeeded | failed | cancelled | blocked
    reason: str | null
  control:
    cancellation_requested_at: str | null
    cancellation_operation_id: str | null
    terminate_running: bool
    requested_by: str | null
    termination_acknowledged_at: str | null
    termination_result: terminated | already_exited | failed | null
    cleanup_operation_id: str | null
    cleanup_state: preparing | waiting_ack | null
  attempt_control:
    next_attempt_number: int
    current_attempt_id: str | null
    current_attempt_number: int | null
  claim_control:
    fencing_epoch: int
    active_claim: null | object
```

`active_claim` contains:

```yaml
claim_id: str
attempt_id: str
attempt_number: int
machine_name: str
reservation_id: str
queue_origin: home | shared
fencing_token: int
claimed_at: str
lease_expires_at: str
launch_state: claimed | starting | running
launch_authorized_at: str | null
group_dispatch_epoch: int | null
group_worker_set_epoch: int | null
```

Task rules:

- Task has no `batch_id`
- Task identity and command specification survive retries
- placement policy is user authorization; placement runtime is scheduler state
- `private + shared` is invalid
- an ungrouped Task must be private; spillover requires a Group Worker Set
- `fencing_epoch` increases whenever new ownership authority is issued
- `active_claim.fencing_token` equals the current `fencing_epoch`
- absence of `active_claim` means no machine owns execution authority
- user-facing Task projection is derived from Task and current Attempt truth
- Task truth wins over every queue index or machine summary

### 8.4 Attempt Truth

Attempt truth lives at:

```text
attempts/<task-id>/<attempt-number>.json
```

Required logical shape:

```yaml
attempt:
  attempt_id: str
  task_id: str
  attempt_number: int
  phase: claimed | starting | running | succeeded | failed | cancelled | orphaned
  machine_name: str
  assigned_gpus: list[int]
  reservation_id: str
  current_fencing_token: int
  token_history: list[int]
  lease:
    claimed_at: str
    renewed_at: str
    expires_at: str
  authorization:
    group_name: str | null
    group_dispatch_epoch: int | null
    group_worker_set_epoch: int | null
  process:
    wrapper_pid: int | null
    wrapper_start_time_ticks: int | null
    process_group_id: int | null
    process_group_start_time_ticks: int | null
    tmux_reference: str | null
    local_process_manifest: str
    log_references: list[str]
  termination:
    requested_by_operation_id: str | null
    requested_at: str | null
    acknowledged_at: str | null
    result: terminated | already_exited | failed | null
  timestamps:
    launch_authorized_at: str | null
    process_created_at: str | null
    running_at: str | null
    orphaned_at: str | null
    recovered_at: str | null
    finished_at: str | null
  result:
    exit_code: int | null
    signal: int | null
    category: str | null
    reason: str | null
```

Attempt rules:

- Attempt is materialized idempotently when a claim allocates its attempt number
- `claimed` is pre-launch and revocable
- `starting` means the shared launch gate committed
- `running` means local process creation was recorded
- a terminal Attempt never returns to an active phase
- `orphaned` records ambiguous process ownership after lease expiry
- a recovery CAS may issue a new token for the same orphaned Attempt
- every older token remains in immutable history and loses write authority

### 8.5 Submission Operation Truth

Submission Operation truth lives at:

```text
operations/submissions/<operation-id>.json
```

It is an internal diagnostic object, not a daily CLI resource.

Required logical shape:

```yaml
submission:
  operation_id: str
  kind: single | bulk
  idempotency_key: str
  raw_request_digest: str
  resolved_context_digest: str
  original_submitting_machine: str
  target_group: str | null
  state: preparing | committing | committed | aborted | blocked
  resolved_context:
    task_ids: list[str]
    task_specs: list[object]
    create_group: bool
    worker_set_additions: list[str]
    group_revision_precondition: int | null
  commit_plan:
    group_membership_sequences: list[int] | null
    pending_group_revision: int | null
  staged_task_count: int
  committed_at: str | null
  failure_reason: str | null
```

The idempotency mapping lives at:

```text
idempotency/submissions/<sha256(project-id + key)>.json
```

It maps one project-scoped key to one operation ID and is created exclusively.

Submission rules:

- the canonical raw request covers normalized semantic manifest content and
  submission-affecting CLI arguments
- YAML formatting, key order, and manifest path are not semantic input
- `--group` is the sole source of Group identity
- manifest `group.name` is invalid
- the first operation resolves `home_machine: current`, generated Task IDs, placement
  defaults, original submitting machine, and Worker Set additions exactly once
- retries load the existing operation before resolving machine-relative values
- the same key and raw request reuse the stored resolved context across machines
- the same key with different raw input fails with an idempotency conflict
- retries never reinterpret `current` or recompute Worker Set changes

### 8.6 Group Control Operation Truth

Group cancellation operation truth lives at:

```text
operations/group-control/<operation-id>.json
```

Required logical shape:

```yaml
group_control:
  operation_id: str
  operation_type: cancel
  group_name: str
  state: preparing | converging | waiting_ack | completed | blocked
  group_revision_at_start: int
  dispatch_epoch_at_start: int
  membership_high_watermark: int
  terminate_running: bool
  progress:
    target_tasks: int
    already_terminal: int
    queued_cancelled: int
    prelaunch_cancelled: int
    running_allowed: int
    termination_pending: int
    termination_acknowledged: int
    blocked: int
  pending_machine_acknowledgements:
    <machine-name>: list[str]
  created_at: str
  updated_at: str
  completed_at: str | null
  blocked_reason: str | null
```

Rules:

- Group truth stores the cancellation barrier and operation ID
- Task and Attempt truth store authoritative per-Task cancellation and termination result
- operation progress and pending-machine maps are durable, revisioned projections that can
  be recomputed from Task and Attempt truth
- default cancellation completes after every in-snapshot Task is classified as already
  terminal, cancelled, or allowed to continue running
- terminating cancellation completes only after every targeted active Attempt has a
  terminal acknowledgement
- unreachable or ambiguous active processes keep the operation in `waiting_ack` or
  `blocked`; they are never counted as acknowledged
- completed operations remain immutable audit history

### 8.7 Machine Truth

Machine declaration lives at:

```text
machines/<machine-name>/machine.json
```

Required fields include:

- stable `machine_name`
- hostname as descriptive metadata
- project and shared-root identity
- local runtime root
- agent mode
- GPU inventory or discovery configuration
- optional administrative tags

In shared mode, machine identity is explicit:

```bash
qexp init --shared-root /path/to/project/.qexp --machine gpu2a
```

Hostname is never the primary machine key.

### 8.8 Agent and Snapshot Truth

`machines/<machine>/state/agent.json` is the machine-owned shared lifecycle record.

It contains:

- configured mode: `on_demand | daemon`
- observed state: `stopped | starting | active | draining | idle | stale | failed`
- agent instance ID and PID
- started, heartbeat, idle, and transition timestamps
- current active Attempt IDs
- stop or drain reason

`state/gpu.json` and `state/summary.json` are advisory snapshots. They must not authorize a
claim, prove physical GPU idleness, or override local reservation truth.

### 8.9 Cross-Record Ordering

Task truth is the ownership authority. Attempt truth is durable execution history and
process evidence.

Required write order:

- claim, launch authorization, and lease renewal update Task authority first, then
  reconcile Attempt truth idempotently
- terminal process reconciliation validates the Task token, writes the Attempt terminal
  result, then clears Task claim and publishes the Task terminal projection
- Submission Operation state decides staged Task visibility
- Group truth decides dispatch, membership ordering, and Worker Set eligibility

If a crash leaves only the first write complete, no reader may infer new execution
authority from the second record. `doctor` reconciles the lagging record from the
authoritative token and immutable evidence.

## 9. Machine-Local Runtime Contracts

### 9.1 Local Directory Layout

```text
<local-runtime-root>/
  agent/
    state.json
    agent.pid
  reservations/
    provisional/
      <reservation-id>.json
    active/
      <reservation-id>.json
    released/
      <reservation-id>.json
  processes/
    <attempt-id>.json
  wrappers/
    <attempt-id>/
  locks/
    gpu-reservations.lock
```

### 9.2 GPU Reservation Record

Required shape:

```yaml
reservation:
  reservation_id: str
  acquisition_id: str
  task_id: str
  attempt_id: str | null
  fencing_token: int | null
  gpu_ids: list[int]
  state: provisional | active | released
  created_at: str
  expires_at: str
  released_at: str | null
  release_reason: str | null
```

Rules:

- allocation is serialized by the local GPU reservation lock
- provisional reservations have a short TTL
- attaching to an authoritative Attempt records Attempt ID and fencing token
- active reservations remain until terminal process reconciliation or verified cleanup
- releasing an already released reservation succeeds idempotently
- `unreserved` means visible GPU capacity not reserved by qexp, not physically idle GPUs

Terminal Task cleanup uses `operations/cleanup/<task-id>.json` as authoritative coordination
truth. The Task record also carries `control.cleanup_operation_id` and
`control.cleanup_state`; submission, retry, claim, launch authorization, cancel, and offer must
reject any Task with cleanup intent or a cleanup tombstone. Task IDs are permanently reserved
by cleanup tombstones and cannot be submitted again.

The cleanup operation freezes `required_machines` to the Task home machine, all historical
Attempt machines, and the machine that prepared cleanup. It progresses from `preparing` to
`waiting_ack` and records a per-machine acknowledgement only after that machine has removed
matching active or provisional reservations, exited process manifests, and logs. Shared Task
and Attempt truth is deleted only after all acknowledgements exist; the operation then becomes
`completed`. Agent and doctor reconciliation are idempotent and resume interrupted local
cleanup or shared finalization. A live or unverifiable local process prevents acknowledgement.
Acknowledgement updates acquire the Task lock only. Cleanup startup acquires
`Schema -> Group -> Task` for grouped Tasks or `Schema -> Task` for ungrouped Tasks. Final
shared deletion uses the same grouped/ungrouped lock split. These schema-lock sections ensure
cleanup tombstone creation and Task truth deletion cannot interleave with submission staging,
while the Group lock prevents grouped Task deletion from racing Group control enumeration.
All writes to `operations/cleanup/<task-id>.json` are serialized by the Task lock for the same
`task_id`; the operation revision is not an independent compare-and-swap authority.

### 9.3 Local Process Manifest

Before or immediately after process creation, the owning agent persists:

```yaml
process:
  attempt_id: str
  task_id: str
  fencing_token: int
  wrapper_pid: int
  wrapper_start_time_ticks: int
  process_group_id: int
  process_group_start_time_ticks: int
  gpu_ids: list[int]
  command: list[str]
  working_directory: str
  created_at: str
  observed_state: starting | running | exited | missing | quarantined
  supervisor: runner | agent
  exit_code: int | null
  signal: int | null
```

This manifest is machine-local recovery evidence. PID identity requires both the PID and
the Linux `/proc/<pid>/stat` start-time ticks to match. It does not supersede the shared
Task fencing token.

Reconciliation distinguishes three outcomes:

- matching identity with a live process permits Recovery CAS
- matching historical identity with a confirmed absent process permits recovery-finalize
- mismatched or unverifiable identity remains blocked and retains its reservation

## 10. Submission Protocol

### 10.1 Common Submission Pipeline

`qexp submit` and each member of `qexp batch-submit` normalize into the same `TaskSpec`.
Both commands use Submission Operation commit; a single submit is an operation containing
one Task. This prevents a separate single-Task crash protocol from drifting away from
bulk correctness.

Single submission keeps its lightweight CLI behavior. Its generated operation identity is
internal and is exposed only through JSON diagnostics or `doctor` after interruption.
Bulk submission additionally exposes the explicit idempotency-key retry contract.

An ungrouped Task has no membership sequence and must use private placement because no
Group Worker Set exists to authorize remote execution.

### 10.2 Prepare

Submission executes:

1. parse and normalize the complete raw request
2. derive the raw-request digest
3. exclusively create or load the idempotency mapping
4. if existing, validate the raw digest and reuse its stored resolved context
5. if new, acquire the Group lock when grouped
6. resolve whether the Group must be created, plus Task IDs, homes, placement, and Worker
   Set additions
7. persist the Submission Operation in `preparing`
8. release the Group lock

The operation exists before Task staging, so `doctor` can recover a terminal interruption.

### 10.3 Stage and Commit

The runtime then:

1. writes every Task idempotently with `submission_operation_id`
2. leaves those Tasks non-claimable while the operation is `preparing`
3. verifies that all staged Tasks exactly match the resolved context
4. reacquires the Group lock when grouped
5. resolves any earlier `pending_submission_commit`
6. creates the Group when the stored plan requires it, or validates the existing Group,
   admission state, and Worker Set preconditions
7. reserves membership sequences and Worker Set changes in Group
   `pending_submission_commit`
8. persists the same assignments in the operation `commit_plan` and changes it to
   `committing`
9. writes the reserved membership sequence into every staged Task
10. records Worker Set additions tagged with the operation ID
11. changes the Submission Operation to `committed`
12. advances Group `next_membership_sequence` and clears `pending_submission_commit`
13. releases the Group lock

An existing Task ID is reusable only when it already belongs to the same Submission
Operation and exactly matches the stored resolved `TaskSpec`. Any other collision fails;
submission never overwrites Task truth.

Readers treat operation-tagged Worker Set additions as effective only when their
Submission Operation is committed. This provides atomic scheduler visibility without
requiring a cross-file filesystem transaction.

Every operation that later acquires the Group lock first reconciles
`pending_submission_commit`:

- committed operation: finalize Group counters and clear the pending pointer
- complete, exact staged truth: finish the commit idempotently
- incomplete uncommitted truth: abort and remove inactive additions
- mismatched truth: mark blocked and refuse unrelated Group mutation

Crash behavior:

- before operation creation: no durable submission exists
- after operation creation but before complete staging: no staged Task is claimable
- after complete staging but before commit: retry or `doctor` may commit idempotently
- during `committing`: the Group pending pointer forces completion or abort before another
  Group mutation can overtake it
- incompatible later Group changes: operation becomes `blocked` with an explicit reason
- an abort removes only uncommitted staged truth and inactive Worker Set additions
- a committed operation is never rolled back into `preparing`

## 11. Group Control Protocol

### 11.1 Admission and Dispatch

`qexp group seal`, `reopen`, `pause`, and `resume` acquire the Group lock and update one
Group revision.

Pause behavior:

- sets `dispatch_state = paused`
- increments `dispatch_epoch`
- prevents new claims and new launch authorizations
- does not terminate Attempts whose launch authorization already won
- does not prevent an open Group from receiving queued Tasks

### 11.2 Group Cancellation

`qexp group cancel` acquires the Group lock and creates a durable Group control operation
with:

- the current membership high watermark
- whether running termination was requested
- the Group revision and dispatch epoch
- per-Task convergence counters

Creation order under the Group lock is:

1. create the Group control operation in `preparing`
2. append its cancellation barrier and operation ID to Group truth
3. change the operation to `converging`
4. release the Group lock and begin asynchronous Task convergence

For each Task at or below the captured membership sequence:

- queued Task becomes cancelled
- a pre-launch `claimed` Attempt loses its launch gate, becomes cancelled, and releases
  claim and reservation
- an Attempt whose `starting` gate already committed follows running semantics
- default cancellation lets starting/running work finish
- `--terminate-running` publishes machine-local termination intent

Tasks appended after the high watermark are not affected.

Each Task convergence transition records `cancellation_operation_id`. For terminating
cancellation, the owning agent records termination acknowledgement and result in Task and
Attempt truth. The operation periodically rebuilds its counters and pending-machine map
from those authoritative records.

Completion rules:

- default cancellation: complete after every targeted Task is already terminal, cancelled,
  or durably classified as already authorized running work
- terminating cancellation: use `waiting_ack` while any targeted starting/running Attempt
  lacks terminal acknowledgement
- terminal acknowledgement includes confirmed termination or a process that had already
  exited and was reconciled terminally
- unknown process state is not acknowledgement and may move the operation to `blocked`

After CLI or agent restart, `qexp group show`, the cancellation command, or `doctor`
reloads the operation and resumes convergence idempotently. CLI pending-machine output is
derived from `pending_machine_acknowledgements`, never from an in-memory command session.

### 11.3 Worker Set Changes

Add, drain, and remove operations acquire the Group lock.

Drain:

- changes the worker state to `draining`
- blocks new home and shared claims on that machine
- allows already authorized Attempts to finish

Final removal requires:

- no active Attempt owned by the machine for the Group
- no private queued Task whose home is the machine
- no `queued_home` spillover Task whose home is the machine
- every remaining `queued_shared` Task has another eligible fallback worker

If blockers exist, the operation reports Task IDs and remains incomplete. The user must
rehome, change queued placement policy, offer, or cancel those Tasks explicitly.

Forced removal publishes termination intent but does not bypass queued-work or ambiguous
process safety checks.

### 11.4 Single Task Cancellation

`qexp task cancel <task-id>` acquires Group then Task lock for a grouped Task, or only the
Task lock for an ungrouped Task.

Behavior:

- queued and unclaimed: change Task projection to cancelled
- claimed before launch authorization: install cancellation intent, cancel the Attempt,
  and release claim and reservation
- starting or running: persist `terminate_running = true`; only the owning machine agent
  may signal and reconcile its local process
- terminal Task: return the existing terminal result idempotently
- blocked orphan: do not claim termination succeeded; require recovery or explicit local
  confirmation

The CLI reports pending owning-machine acknowledgement. It never sends a signal directly
to a remote PID.

### 11.5 Queued Placement Changes

A user-authorized placement change is accepted only while the Task is queued and has no
active claim. It acquires Group then Task lock and validates:

- the new home is an active non-draining Group worker
- fallback scope does not exceed the Group Worker Set
- changing private to spillover is explicit user intent
- an agent-initiated transition changes queue scope only and never broadens policy

Rehoming resets queue scope to home and recomputes `queued_home_at` and
`offer_eligible_at`. An active Attempt rejects placement mutation; any future-Attempt
override requires a separate product contract.

## 12. Offer Protocol

The first release supports exactly:

- `qexp task offer <task-id>`
- persisted `after_seconds` eligibility

The manifest does not accept `on_overload`, `min_local_wait_seconds`,
`max_offer_per_cycle`, or `cooldown_seconds` in the first release.

`after_seconds` is either `null` or a non-negative integer. `null` disables elapsed-time
offering while retaining manual offer capability; zero permits immediate shared-pool
eligibility after commit.

Task commit persists:

```text
queued_home_at
offer_eligible_at = queued_home_at + after_seconds
```

Any active Group worker may scan due-time indexes, but the index is advisory. Offering
acquires the Task lock and validates authoritative Task truth:

- Task is committed
- Task projection is queued
- queue scope is home
- no active claim exists
- sharing mode is spillover
- manual request is authorized or current time is at/after `offer_eligible_at`

The winning transition sets queue scope to shared and records reason, actor, and time.
Repeated offers succeed idempotently. A concurrent claim changes Task revision and causes
the offer CAS to fail.

Heartbeat staleness does not bypass `after_seconds` in the first release. Automatic
overload and heartbeat-based early offering require separate future contracts.

## 13. Claim and Launch Protocol

### 13.1 Eligibility

Before reserving capacity, an agent verifies advisory eligibility. Before claiming, it
must revalidate under authoritative locks:

- Submission Operation is committed
- Task is queued and unclaimed
- Group dispatch is active, if grouped
- machine is an active non-draining Group worker
- home queue is claimed only by home machine
- shared queue claimant is allowed by Worker Set and fallback constraint
- Task requires no more locally reservable qexp GPUs than available

### 13.2 Provisional Reservation

The agent creates a TTL-bound local provisional reservation keyed by an acquisition ID.
It must not reserve more Tasks than it can promptly launch.

If shared-lock acquisition is delayed beyond the provisional TTL, the agent releases and
restarts acquisition rather than extending capacity indefinitely without ownership.

### 13.3 Global Claim

For a grouped Task, the agent acquires Group then Task lock. It validates eligibility and
updates Task truth in one revision:

1. allocate `attempt_number = next_attempt_number`
2. derive a deterministic Attempt ID from Task ID and attempt number
3. increment `next_attempt_number`
4. set the current Attempt ID and number
5. increment `fencing_epoch`
6. create `active_claim` with `launch_state = claimed`
7. record queue origin, machine, reservation ID, token, and lease
8. project Task as running for user-facing active-work counts

The agent then creates the Attempt record idempotently in `claimed` and attaches the local
reservation to the Attempt and fencing token.

If Attempt creation fails, the agent or `doctor` materializes a cancelled pre-launch
Attempt when storage permits, then releases the claim and reservation through the same
token. An allocated attempt number is never reused. No process may launch while Attempt
truth is missing.

### 13.4 Final Launch Gate

Immediately before local process creation, the agent acquires Group then Task lock and
validates:

- Task still references the same claim and token
- lease remains valid
- Attempt exists in `claimed`
- Group dispatch is active
- no applicable cancellation barrier won
- machine remains an active Worker Set member

The winning fenced Task transition changes:

```text
active_claim.launch_state: claimed -> starting
active_claim.launch_authorized_at: null -> timestamp
```

It records the current Group dispatch epoch and Worker Set epoch. Attempt truth is then
reconciled to `starting` under the same token.

This transition is the linearization point:

- if pause, cancellation, expiry, or Worker Set control commits first, launch fails
- if `starting` commits first, pause allows the Attempt to proceed
- later default cancellation treats it as running work
- later terminating cancellation publishes termination intent

A plain read followed by process creation is invalid.

### 13.5 Process Creation

After launch authorization, the agent:

1. creates the local wrapper/process under the assigned GPUs
2. persists the local process manifest with the fencing token
3. updates Task active claim to `running`
4. updates Attempt to `running` with process references
5. begins lease renewal and process reconciliation

If local process creation fails, the Attempt becomes failed and claim and reservation are
released idempotently.

## 14. Compensation and Reservation Convergence

Every acquisition step has a repeatable compensation:

| Failure window | Required convergence |
| :--- | :--- |
| Provisional reservation succeeds, claim loses | Release provisional reservation |
| Claim succeeds, Attempt write fails | Release claim with winning token; release reservation |
| Attempt exists, launch gate loses | Cancel pre-launch Attempt; release claim and reservation |
| Launch authorized, local spawn fails | Fail Attempt; release claim and reservation |
| Process exits normally or terminally | Publish result; close claim; release reservation |
| Agent crashes during compensation | Next agent start or `doctor` repeats the same steps |

Compensation rules:

- releasing an absent or already released resource is success
- only the current fencing token may close an active claim
- provisional reservation expiry does not delete a reservation attached to an
  authoritative active Attempt
- local reconciliation checks claim, Attempt, and process identity together
- an unattached expired reservation is deleted only after no matching local process exists
- ambiguous process ownership remains blocked instead of being counted as free capacity

## 15. Lease and Fencing Protocol

### 15.1 Renewal

The owning agent renews before expiry by presenting:

- Task ID and Attempt ID
- machine and agent instance identity
- current fencing token
- observed local process state

Renewal updates Task active claim and Attempt lease idempotently. A mismatched token,
Attempt, machine, or agent authority is rejected.

Machine heartbeat and Attempt lease are distinct:

- heartbeat describes agent reachability
- lease describes current execution write authority
- stale heartbeat alone does not revoke a valid lease

### 15.2 Pre-Launch Expiry

If `launch_state = claimed` expires, the protocol proves that no process was authorized to
start. Reconciliation may therefore:

- archive the claim and token
- cancel the incomplete Attempt with a pre-launch expiry reason
- clear Task active claim
- restore its previous home or shared queue scope
- release the local reservation when the owning machine is reachable, or mark it for local
  reservation cleanup

### 15.3 Post-Authorization Expiry

If `launch_state = starting | running` expires:

- archive the claim with its token preserved
- change Attempt to `orphaned`
- change Task projection to `blocked`
- clear active execution authority without returning Task to a queue
- preserve machine, GPU, process, log, and token history
- set `orphaned_at` without setting terminal `finished_at`
- record the expiry reason in the append-only event stream rather than terminal result
- forbid automatic replacement Attempt creation

`orphaned` means process state is unknown, not that the process stopped.

### 15.4 Recovery CAS

A returning machine with a verified live process cannot perform ordinary renewal after
expiry. A grouped Task acquires Group then Task lock; an ungrouped Task acquires only Task
lock.

For a grouped Task, recovery first reconciles any pending Group submission commit and
verifies:

- Group pause does not block recovery because the process was already launch-authorized
- the machine is still `active`, or is `draining` and the Attempt's recorded launch
  `group_worker_set_epoch` predates the worker's draining `state_epoch`
- the machine is not `removing` or removed
- no applicable Group cancellation barrier requests running termination
- no forced-removal termination intent applies to the Attempt

The Task-lock CAS then verifies:

- Task is still blocked
- current Attempt is the same orphaned Attempt
- no successor Attempt exists
- no newer fencing token exists
- the presented expired token matches that Attempt's archived token
- local process identity matches the Attempt evidence
- no Task-level termination intent remains applicable

Success:

1. increments Task `fencing_epoch`
2. creates a new active claim for the same Attempt
3. appends the new token to Attempt token history
4. issues a new lease
5. clears terminal result fields and `finished_at`
6. sets `recovered_at` and reconciles Attempt and Task to running

Failure means the local process is obsolete or no longer permitted to resume authority.
The agent must terminate or quarantine it and must not publish scheduler truth.

### 15.5 Other Return Cases

- finished process: for a grouped Task acquire Group then Task lock, use a
  recovery-finalize CAS with the same no-successor and token preconditions, publish the
  terminal result without returning to running, and satisfy any applicable termination
  acknowledgement
- missing process: for a grouped Task acquire Group then Task lock, confirm machine-local
  cleanup, use a recovery-resolution CAS to fail the orphaned Attempt, and satisfy any
  applicable termination acknowledgement
- newer Attempt or token: reject stale writes and terminate or quarantine the old process

Fencing protects qexp metadata. It cannot undo external side effects already produced by
an obsolete process.

## 16. Retry Protocol

`qexp task retry <task-id>` is accepted only when the Task's current Attempt is failed.

Under the Task lock it:

1. validates no active claim exists
2. validates Task is not blocked, queued, running, succeeded, or cancelled
3. changes Task projection to queued
4. preserves command, Group, home, sharing, and fallback policy
5. restores queue scope according to the preserved placement runtime contract
6. leaves historical Attempt truth immutable

The next Attempt number is materialized when a later claim wins. Task ID and Group Task
count do not change.

`qexp group retry-failed <group>` acquires the Group lock to establish a stable selection
revision, then applies the same Task retry transition only to Tasks whose current
projection and current Attempt are failed. A historical failure behind a newer Attempt is
never selected.

Blocked or orphaned work requires recovery, not ordinary retry. A duplicate-risk override
must be explicit.

`qexp task retry <task-id> --acknowledge-duplicate-risk` may override an unresolved
orphan only when the Task is blocked and the current Attempt is orphaned. Under the Task
lock it:

1. records the operator acknowledgement and reason in an audit event
2. preserves the orphaned Attempt and expired token history
3. increments the Task fencing epoch so the next claim receives newer authority
4. queues the same Task for its next Attempt

This override does not stop the old process or undo external side effects. It is never
used by automatic retry or Group retry-failed.

## 17. Agent Lifecycle

### 17.1 On-Demand Mode

Default behavior:

- local submission starts the current machine's agent when it has eligible local work
- the agent polls or watches eligible local/shared work
- it exits only after true idleness
- provisional reservations, active processes, pending termination, and repair work prevent
  idle exit

### 17.2 Daemon Mode

Daemon mode remains active until explicit stop or unrecoverable local failure.

```bash
qexp agent start
qexp agent run
```

`agent start` always starts a detached agent, regardless of configured mode. `agent run` keeps
the agent in the current terminal for debugging. Both modes auto-start after local submission;
they differ only in whether true idleness ends the agent process.

### 17.3 Agent Startup Reconciliation

Before claiming new work, an agent:

1. acquires its local lifecycle lock
2. reconciles local process manifests
3. reconciles provisional and active reservations
4. compares every local token with shared Task truth
5. terminates or quarantines obsolete local processes
6. publishes fresh agent and GPU snapshots
7. starts normal claim scanning

## 18. Events and Derived Data

Project events include:

- Group admission, dispatch, cancellation, and Worker Set transitions
- Task submission, offer, claim, retry, cancel, and projection changes
- Attempt launch authorization, start, finish, failure, orphan, and recovery
- Submission Operation prepare, commit, abort, block, and repair
- claim expiry and stale-token rejection

Event requirements:

- events are append-only audit records
- events do not override object truth
- duplicate event emission is tolerated through stable event IDs
- training metrics, checkpoints, and scientific progress are not scheduler events

Derived indexes and summaries:

- may lag truth
- are never used as the final claim or control authorization
- are rebuilt from Group, Task, Attempt, operation, and machine truth
- must not recreate intentionally deleted Task truth

## 19. Doctor Contract

`qexp doctor` diagnoses shared truth and coordinates machine-local repair where evidence is
available.

It must distinguish:

- unsupported or mixed schema
- orphan idempotency mapping
- uncommitted Submission Operation with partial staged Tasks
- committed operation with missing or mismatched staged Task
- inactive Worker Set addition from an interrupted operation
- missing, corrupt, or stalled Group control operation referenced by a cancellation barrier
- Group cancellation progress or pending-machine acknowledgement drift
- queued Task referencing an uncommitted operation
- claim without Attempt truth
- expired pre-launch claim
- expired post-authorization claim
- unattached or expired local reservation
- active reservation with no authoritative claim
- local process with obsolete token
- orphaned Attempt with live, finished, missing, or unknown process
- unacknowledged Task or Group termination intent
- stale or corrupt derived index

Safe automatic repairs include:

- rebuilding indexes and summaries
- completing an idempotent operation whose staged truth exactly matches resolved context
- rebuilding Group cancellation progress from Task and Attempt truth
- resuming idempotent cancellation convergence after CLI or agent restart
- aborting an uncommitted operation before any Task became visible
- releasing an expired pre-launch claim
- repeating an already authorized compensation
- removing an unattached expired reservation after proving no process exists

Manual confirmation is required when:

- a post-authorization process may still be alive on an unreachable machine
- local process identity cannot be proven
- Group changes conflict with a stored Submission Operation plan
- retry could produce duplicate external side effects
- filesystem behavior invalidates the selected coordination assumptions

Every repair is revisioned and emits an audit event. `doctor` must return a specific
unresolved action rather than reporting generic inconsistency.

## 20. Crash-Window Matrix

| ID | Crash window | Required state after convergence |
| :--- | :--- | :--- |
| CW-01 | Before Submission Operation creation | No operation or Task exists |
| CW-02 | Operation exists before Task staging | Operation remains preparing or aborts; no Task is claimable |
| CW-03 | Some Tasks staged | Retry or `doctor` completes or aborts without partial visibility |
| CW-04 | Worker addition written before operation commit | Addition is inactive until commit and removable on abort |
| CW-05 | Provisional GPU reservation before claim | Reservation expires or is released; Task remains queued |
| CW-06 | Claim wins before Attempt write | No launch occurs; claim and reservation are recoverably released |
| CW-07 | Attempt written before launch gate | Pause/cancel may win; no process exists without authorization |
| CW-08 | Launch gate wins before process creation | Starting Attempt is recovered conservatively; spawn failure becomes failed |
| CW-09 | Process starts before running metadata | Local manifest and token reconcile the same Attempt |
| CW-10 | Process exits before terminal publish | Owning agent republishes terminal result idempotently |
| CW-11 | Lease expires before launch authorization | Claim returns safely to its previous queue scope |
| CW-12 | Lease expires after launch authorization | Attempt becomes orphaned and Task blocked; no automatic replacement |
| CW-13 | Old machine returns with current orphan token | Recovery CAS may issue a new token for the same Attempt |
| CW-14 | Old machine returns after successor token | Old process is terminated or quarantined; writes are rejected |
| CW-15 | Worker removal races with submission or claim | Group lock yields one order; removal cannot strand queued work |
| CW-16 | Pause/cancel races with launch | Exactly one Group-lock order wins and defines running semantics |
| CW-17 | Recovery races with drain/remove or terminating cancellation | Group-then-Task lock order yields one result; forbidden recovery terminates or quarantines the process |
| CW-18 | CLI restarts while Group cancellation waits for acknowledgements | Durable Group control operation resumes with the same pending-machine set |

## 21. Verification Requirements

The runtime implementation is not releasable until tests demonstrate:

- exclusive Task claims across separate processes and target hosts
- home-versus-fallback claim races produce one active Attempt
- pause and cancellation linearize against the final launch gate
- stale fencing tokens cannot mutate Task or Attempt truth
- every GPU acquisition failure window converges without reservation leakage
- cross-machine idempotent retry reuses the first resolved submission context
- manifest `group.name` is rejected and `--group` remains the only Group identity source
- uncommitted bulk Tasks are never claimable
- Worker Set drain and removal cannot strand private or home-only queued Tasks
- manual and `after_seconds` offering are idempotent under repeated scanners
- heartbeat staleness does not trigger an unsupported early offer
- pre-launch expiry requeues safely
- post-authorization expiry blocks instead of automatically retrying
- recovery CAS issues a new token only for the same orphaned Attempt
- grouped recovery obeys Group-then-Task lock order, Worker Set state, and termination
  barriers while allowing pause semantics
- duplicate-risk retry requires explicit acknowledgement and is excluded from Group retry
- Group cancellation progress and pending acknowledgements survive process restart
- on-demand idle exit does not abandon processes, reservations, or repair work
- daemon background startup preserves daemon mode
- unsupported Batch-era schema fails before mutation
- indexes and summaries rebuild without changing truth

Cross-machine correctness tests must run against every supported shared-filesystem profile.
Mock-only tests are insufficient for the coordination release gate.

## 22. Explicit Non-Goals

- backward compatibility with Batch-era `.qexp` data
- a public Batch truth object
- daily CLI resource trees for Attempt or Submission Operation
- central scheduler or coordinator service
- remote SSH or remote agent wake-up
- physical GPU utilization scheduling outside qexp reservations
- automatic overload offering in the first release
- heartbeat-based early offering in the first release
- automatic replacement after ambiguous machine loss
- source snapshotting or Git revision enforcement
- workflow DAGs, priorities, quotas, or preemption
- hostile multi-tenant isolation
- scientific metric, artifact, or checkpoint management
