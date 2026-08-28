---
doc_type: spec
status: active
updated_at: 2026-08-28
archived_at:
---

# qexp Runtime Spec

## Schema 6 clock-capability authority contract

This section supersedes prior chrony-only, single-skew lease, and bare wall-clock offer rules.
Each authority operation obtains a provider observation with a conservative UTC error interval,
monotonic observation age, drift bound, provider margin, and immutable ID. Chrony and Linux
`adjtimex(2)` are the supported providers; configured priority selects one only after all
qualifying provider intervals overlap. Missing, stale, over-limit, or conflicting evidence is
fail-closed for cross-machine time authority.

Active claims and Attempts explicitly record `authority_mode`. `bounded_lease` persists holder
clock evidence and permits reclaim only at `expiry + holder_bound + reclaimer_bound`.
`holder_bound` has null lease evidence, can only be supervised or restarted by its holder
machine with matching process identity, and never expires, enters remote Recovery, or receives
an automatic successor. Timed offers persist the complete creator observation and the creator
monotonic instant corresponding to their deadline; the home agent projects that evidence through
its drift bound, then commits only when its current UTC lower bound reaches the resulting deadline
upper bound. The proof and transition are revalidated under the Task authority lock. User-driven
share and offer are not elapsed-time authority operations.

Schema-5 cutover is fully drained, backed up and fsynced before final schema-6 configuration is
staged; the schema version replacement is the commit point. Old roots are not runtime-readable
after cutover.

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

The Group/Task/Attempt model rationale and delivery history are recorded in
[021-qexp-experiment-group-task-scheduling.md](../pitch/arxiv/021-qexp-experiment-group-task-scheduling.md).
The schema-6 authority ownership delivery history is recorded in
[027-qexp-agent-owned-authority.md](../pitch/arxiv/027-qexp-agent-owned-authority.md).

Where documents overlap:

- the Product Spec is authoritative for user-visible behavior
- this Runtime Spec is authoritative for persistence, concurrency, and recovery mechanics
- the pitch is non-normative rationale and delivery tracking

## 2. Status and Cutover

This is the installed schema-6 runtime contract. The Batch-era transition was intentionally
destructive:

- no Batch-era truth is read or migrated
- no Batch-era Task, retry-lineage, index, claim, or operation record is imported
- no mixed-schema runtime is supported
- no legacy reader or compatibility repair path is provided
- an unsupported `.qexp` schema fails before any agent or mutating command starts
- a drained schema-5 root may be upgraded with `qexp migrate --to-schema 6`; migration
  rejects any root with an active claim or running Attempt

Loss of Batch-era scheduling metadata is a known and accepted product risk. The schema-5 to
schema-6 migration is a one-way control-root upgrade, not a legacy reader or repair path.

## Compatibility-sensitive runtime invariants

This section protects only runtime semantics whose regression would break a protected workflow or
misassign process or GPU ownership. It does not promote general implementation details into public
API.

1. **Legacy classification**: if any project machine record carries legacy metadata, `init`,
   registration, and agent startup must not implicitly reclassify or overwrite it; only
   `qexp agent migrate-project` performs that conversion.
2. **Agent ownership**: one shared `MachineRuntime` root has one global agent and one unified GPU
   reservation set. Projects join it only through controlled bindings. A logical `--machine` name
   is not physical-host identity; projects share this arbitration boundary only when they use the
   same `MachineRuntime`.
3. **Running task ownership**: initialization, registration, global-agent lifecycle operations,
   and migration must not take over, terminate, or reinterpret running work outside the target
   binding. The global agent continues its established supervision only for registered projects.
4. **GPU reservation ownership**: reservations retain stable project ID and MachineRuntime
   ownership. Another project or an incomplete migration must not release, overwrite, or treat a
   valid reservation as available capacity.
5. **Migration transition**: verified legacy-agent stop, reservation and local-evidence import,
   and binding enablement are ordered and recoverable. A failure leaves a diagnosable incomplete
   state.

## Lifecycle terminal boundary and notifications

Every attempt-backed terminal transition (`succeeded`, `failed`, or `cancelled`) is committed
through one locked terminal-transition primitive. The primitive validates task/attempt identity,
fencing and source phases, writes Attempt and Task truth, archives an active claim, and returns an
immutable lifecycle event. Reservation release, local process-manifest updates, and lifecycle hook
dispatch occur after Group/Task locks are released. Hook and notifier failures are isolated from
authoritative state.

The process that successfully commits the transition dispatches using its own machine configuration;
therefore a controller cancelling a remote pre-launch Attempt sends the notification. Events retain
both execution and dispatching machine names. Machine notification configuration is additive and
default-off. Feishu uses the environment by default; an explicitly acknowledged `shared_file`
webhook source may persist a URL in that machine's shared control-root secrets directory, while a
signing secret remains environment-only. Notification records use an atomic claimed key and provide
at-most-one HTTP send attempt, without retry or exactly-once delivery guarantees.

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

### 3.3 Project-Local Runtime Root

Standalone operation retains its existing project-local `RootConfig.runtime_root` for its agent
PID, reservations, process manifests, wrapper controls, and recovery evidence. It remains local
and never becomes shared project authority.

### 3.4 MachineRuntime Root and Project Registry

Machine-agent operation adds one disposable, user-local `MachineRuntime` per qexp Machine. Its
root is `QEXP_MACHINE_RUNTIME_ROOT` when set, otherwise:

```text
~/.qqtools/qexp-machine/
```

The resolver canonicalizes the path without creating it. It rejects an existing non-directory or
a project `.qexp` root. Scheduling and registry mutations create the layout and fail before their
first write if it is not writable. Read-only project commands do not initialize this runtime.
Deployment must provide user-local rather than shared control storage. It has no fallback to a
project's runtime root.
Every qexp agent process uses this resolver to acquire the same lifetime-held machine scheduler
lock.

`MachineRuntime` owns the registry, scheduler and registry locks, one unified GPU reservation
set, global-agent PID/status, round-robin cursor, and project-ID-partitioned local process,
launch, observation, termination, and recovery records. It does not own or duplicate Task,
Group, Attempt, claim, lease, log, or terminal truth.

A registry binding contains `project_id`, canonical `shared_root`, project-local `machine_name`,
and `enabled`. `project_id` is the stable value in the project's `project/identity.json`; paths
are only location and diagnostic data. Registry changes are serialized and revisioned. Duplicate
stable IDs and duplicate canonical roots are rejected. Runtime state is derived as `enabled`,
`draining` (disabled with local blockers), or `disabled` (disabled with no blockers). A binding
may be removed only when disabled and all associated reservations, process/launch/observation
records, termination decisions, and pending local convergence evidence are absent.
After shared terminal truth commits, the agent consumes the corresponding process,
registration, observation, launch-intent, and completed termination records. Successful removal
deletes the binding's disposable project runtime partition before removing the registry entry.
`qexp init` registers new-generation projects with the global agent before reporting success.
`qexp agent add-project` may restore an absent current-generation binding while the global agent is
running. A project whose machine metadata predates the global runtime must use
`qexp agent migrate-project`; the migration creates a disabled binding, stops only a verified old
agent, imports local reservations and evidence, then enables the binding after the durable handoff.

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

Cross-machine dispatch is fail-closed until two distinct deployment hosts produce one successful
qualification result for the same mounted control root. The probe contract requires both hosts
to participate in each test and records their stable operator-supplied identities. It verifies:

1. an exclusive lock has exactly one holder while the peer observes contention;
2. readers observe either the old or complete new payload across atomic replacement, never a
   partial payload;
3. a file and its parent-directory replacement become visible to the peer after file and
   directory `fsync` complete;
4. process exit releases the lock, while an incomplete temporary write is never interpreted as
   committed truth.

Missing evidence, identical host identities, timeout, malformed payload, inconsistent observation,
or any failed property yields `not_qualified` with property-specific reasons. A process-local mock
or single-host test can verify the judge but cannot qualify a deployment. Probe output is deployment
evidence only: it never mutates scheduler budgets, partition counts, persisted Task truth, or global
defaults. Single-host scheduling does not require a fabricated qualification result.

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
    availability/
      active/<operation-id>.json
      <operation-id>.json
    group-control/
      active/<operation-id>.json
      <operation-id>.json
    cleanup/
      active/<task-id>.json
      <task-id>.json
  idempotency/
    submissions/
      <key-digest>.json
  claims/
    archive/
      <task-id>/
        <fencing-token>.json
    pending/
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
      active/<home-machine>/<utc-hour>/<task-id>.json
    ready/
      state.json
      allocators/<route>.json
      catalogs/<route>/<page>.json
      reservations/<task-id>.<generation>.json
      home/<machine>/<partition>/partition.json
      home/<machine>/<partition>/<task-id>.<generation>.json
      shared/<partition>/partition.json
      shared/<partition>/<task-id>.<generation>.json
      cursors/<project-id>.<machine>.<scope>.json
      builds/<build-id>/watermark/<page>.json
      builds/<build-id>/replaced-projection/
      locks/state.lock
      locks/<route>.lock
  logs/
    <task-id>/
      <attempt-id>.log
```

Layout rules:

- object truth is organized by object type, not by experiment directory
- active claim truth is embedded in the Task record, not split across machine-private
  claim directories
- archived claim records are immutable audit history
- pending claim records retain an archive retry payload; they must be reconciled before Task
  cleanup deletes the corresponding Attempt truth
- machine state is writable only by that machine, except repair metadata explicitly
  written by `doctor`
- indexes and summaries may be deleted and rebuilt without losing truth
- `ready` is a rebuildable liveness projection; Task and Submission truth remain authoritative
- unfinished availability, Group control, and cleanup truth lives below its type-specific
  `active/` directory; completed truth is published at the stable history path before active truth
  is removed
- a stable operation or deadline path may be a compatibility symlink while the record is active;
  the symlink is a locator, not a second authority record
- captured Attempt stdout/stderr logs are shared observability artifacts; process evidence,
  PIDs, reservations, wrappers, and locks remain machine-local

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
  writer_capabilities: [ready-v1]  # present after ready-index building starts
```

Installing `writer_capabilities` is the ready-index writer gate. Schema-6 readers predating the
capability field reject the extended schema envelope before entering a protected mutation workflow.
Current writers also validate `ready-v1` immediately before every Task create, update, or delete.
The field and the transition to `building` are committed while holding the schema lock, after any
in-flight Submission writer exits. The field is never removed on activation or repair.

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
  ready_generation: int
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
- `ready_generation` is non-negative and monotonic; additive schema-6 records without the field
  read as generation zero until a ready-producing transition or the ready builder assigns one

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
- `running` means the owning Agent fenced and published immutable local process-creation evidence into shared Attempt truth
- `launch_authorized_at` is written once by the Scheduler when the shared launch gate commits
- `process_created_at` is written once by the passive Runner immediately after guardian process creation, in its immutable local registration
- `running_at` is written once by the owning Agent when it accepts that registration and commits shared `running` truth
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
    group_precondition:
      exists: bool
      revision: int | null
      worker_set_epoch: int | null
    planned_worker_set: list[str]
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
- manifest Worker Set changes come only from `group.workers`; root `workers` and
  `defaults.placement.workers` are invalid
- the first operation resolves `home_machine: current`, generated Task IDs, placement
  defaults, original submitting machine, planned Worker Set, Group revision/worker-set epoch, and
  Worker Set additions exactly once
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
- project and shared-root identity
- local runtime root
- agent mode
- GPU inventory or discovery configuration
- optional administrative tags

In shared mode, machine identity is explicit:

```bash
qexp init --shared-root /path/to/project/.qexp --machine gpu2a
```

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
claim, prove physical GPU idleness, or override local reservation truth. The agent refreshes
the snapshots with its heartbeat; observers may mark a machine stale without treating that as
proof that a local process stopped.

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

Before `Popen`, the runner writes an immutable `launch-intents/<attempt-id>.json` containing
the Attempt/token, wrapper PID/start-time, GPU assignment, command and lease expiry. On Linux,
the runner starts a private-session guardian, which owns the training process group and receives
a parent-death notification if the runner dies. The guardian then sends `SIGKILL` to its entire
process group, covering the training leader and descendants that remain in that group. The immutable
`process-registrations/<attempt-id>.json` is still written immediately after creation. If an
agent recovers an intent whose wrapper identity is absent but which has no registration, it
materializes only a `launch_unverifiable` manifest, retains the reservation, and records a
local diagnostic. It must not automatically release, recover, retry, or signal that Attempt.
Programs that explicitly leave the guardian process group (`setsid()` / `setpgid()`) are outside
this guarantee and require cgroup-level containment to govern.

Reconciliation distinguishes three outcomes:

- matching identity with a live process permits Recovery CAS
- matching historical identity with a confirmed absent process permits recovery-finalize
- mismatched or unverifiable identity remains blocked and retains its reservation

### 9.4 Machine-Managed Local Identity and Authority

For a machine-managed binding, the reservation and every local execution artifact are namespaced
by stable project ID. Its authoritative local association is the composite
`(project_id, task_id, attempt_id)` and also records canonical shared root and project-local
machine name for validation and diagnostics. Project-local `task_id` or `attempt_id` alone must
never identify a machine-managed reservation, process manifest, launch intent, registration,
observation, wrapper, termination decision, tmux reference, or recovery operation.

The machine scheduler lock is independent of project ID and is held continuously from local
recovery through claim scanning, supervision, and final resource writes. Lock contention fails
before a second agent reserves a GPU or creates a process.

Disable commits only `enabled = false` under the registry lock. Subsequent global-agent cycles
exclude that binding from new candidate scans. Disabled bindings continue reconciliation and
terminal publication while blockers exist; removal remains rejected until those blockers are gone.

Before dispatching an enabled or draining binding, the machine agent performs that project's
durable maintenance: event and claim-archive convergence, cleanup and availability-operation
reconciliation, elapsed-offer evaluation, and reservation reconciliation. Because reservations
are machine-wide, reservation reconciliation filters by the binding's stable project ID before
reading project Task or Attempt truth; it must never release another registered project's
reservation.

Machine runtime state is disposable. Loss of its registry, cursor, PID, reservation, or process
records neither changes project truth nor proves an old process stopped. A replacement runtime
requires explicit re-registration and may schedule queued work, but it does not recover or write
a terminal outcome for executions evidenced only by the lost runtime. Project lease expiry and
fencing retain the existing post-authorization safety outcome: orphaned Attempt, blocked Task,
and no automatic replacement.

### 9.5 Explicit Legacy Project Migration

An existing Project without the global-agent machine-record marker is migrated only by explicit
`qexp agent migrate-project`. The operation verifies the old PID through Linux process identity
and its configured project arguments before signalling it. Process exit is checked against the
same PID start identity so PID reuse cannot delay the handoff. Reservation transfer holds both the
legacy and unified reservation locks while it re-reads, validates, writes, and removes records.
After the old agent stops, its local evidence is moved once into the machine runtime. The machine
agent subsequently drains only immutable registration, observation, and launch-intent records
that a previously launched runner may still write, deleting each legacy source after the target
write is durable. It enables the binding before recording the final `active` migration state.
Migration progress and the operator-controlled binding state are independent: retrying an
`active` migration never re-enables a disabled binding. The operation never automatically
discovers, stops, or registers sibling projects.

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
   Set additions against the planned active Worker Set
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

All queued availability changes use the same durable operation path:

1. write `operations/availability/<operation-id>.json` in `prepared` state
2. acquire Group then Task lock, or only Task lock for standalone idempotent `keep-local`
3. re-read Task truth and validate submission, projection, claim, cleanup, cancellation, Group,
   home worker, and helper worker state
4. commit at most one Task revision when policy or queue scope changes
5. write deterministic `task_availability_changed` audit event
6. update or remove `indexes/offer-deadlines/<task-id>.json`
7. mark the operation `completed`

Unfinished operation truth is stored in the type-specific `active/` directory. Completion first
publishes terminal history at the stable operation path and then removes active truth. Regular
agent maintenance streams at most 64 active availability, Group cancel, and cleanup records of each
type and never enumerates completed operation history. Existing flat unfinished schema-6 records
are split once under the project schema lock.

If Task truth is committed but audit or index writing is interrupted, the operation remains
replayable. Agent startup and `qexp doctor repair` reconcile incomplete availability operations
and rebuild advisory deadline indexes from Task truth.

Supported actions:

- `share_now`: private or spillover policy becomes spillover, helper scope is committed, and
  queue scope becomes shared.
- `share_after`: policy becomes spillover but queue scope remains home until a bounded deadline
  is proven elapsed.
- `keep_local`: policy becomes private, queue scope becomes home, and delayed offer state is
  cleared.
- `manual_offer`: queue scope becomes shared only when policy is already spillover.
- `elapsed_offer`: queue scope becomes shared only when creator/evaluator clock evidence proves
  the persisted deadline elapsed.

An active Attempt rejects placement mutation; any future-Attempt override requires a separate
product contract.

## 12. Offer Protocol

The first release supports exactly:

- `qexp task share <task-id>`
- `qexp task share <task-id> --after <duration>`
- `qexp task share <task-id> --with <machine>`
- `qexp task keep-local <task-id>`
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

The advisory deadline record lives at
`indexes/offer-deadlines/active/<home-machine>/<utc-hour>/<task-id>.json`; the flat path remains an
exact compatibility locator. A home agent consumes only its due UTC-hour buckets, with at most 64
records per maintenance slice. Removing the final record removes its empty bucket. Synchronization
compares the complete payload first and does not rewrite or fsync unchanged content.

Any active Group worker may scan due-time indexes, but the index is advisory. Offering
acquires the Task lock and validates authoritative Task truth:

- Task is committed
- Task projection is queued
- queue scope is home
- no active claim exists
- sharing mode is spillover
- manual request is authorized or current time is at/after `offer_eligible_at`

The winning transition sets queue scope to shared and records reason, actor, and time.
Repeated offers succeed idempotently without increasing Task revision. A concurrent claim,
cancel, cleanup, worker drain, share, keep-local, manual offer, or elapsed offer linearizes
through the same Task lock and only one state change can win.

Heartbeat staleness does not bypass `after_seconds` in the first release. Automatic
overload and heartbeat-based early offering require separate future contracts.

## 13. Claim and Launch Protocol

### 13.1 Machine-Agent Cross-Project Selection

A machine agent scans only enabled registry bindings. For each binding it builds a project-local
`RootConfig` and applies the existing project eligibility, lock, claim, lease, and fencing
protocol without changing their semantics. Candidate selection is advisory; only the winning
project-root claim authorizes execution.

Enabled bindings are sorted lexically by stable project ID. The persisted cursor identifies the
next starting binding. Missing or unknown cursor state starts at the first sorted binding. One
round visits each binding at most once and permits at most one successful claim per project. The
agent continues after a project has no candidate, lacks capacity in the machine-wide reservation
set, has an inaccessible working directory, or loses its project claim. If capacity and the work
budget remain after a round, the successor of the last successful binding starts another round.
After dispatch, the cursor persists that successor; after a full round with no successful claim,
it persists the successor of the starting binding. This is deterministic, unweighted round-robin;
it does not implement priorities, quotas, preemption, or project capacity reservations.

After ready-index activation, project selection uses a persistent project cursor and candidate
selection uses one persistent cursor for each project and queue scope. A marker name is
`<task-id>.<ready-generation>.json`. Writers allocate markers into monotonically numbered partition
records with 64 durable slots. Allocation reserves a slot under the ready allocator lock, releases
that lock, and performs shared Task/Submission I/O afterward; the allocator lock is never held while
waiting for a shared object lock. A partition is sealed when all slots have been allocated and is
removed after every marker and reservation has drained. Active partition IDs are stored in linked
catalog pages of at most 64 IDs. The candidate path reads one catalog page and one partition record,
then opens marker paths named by that record; it never discovers candidates with an unbounded
directory glob. The cursor record contains exactly:

```yaml
cursor:
  schema_version: 1
  project_id: str
  machine_name: str
  queue_scope: home | shared
  catalog_page: str | null
  partition: str | null
  after_name: str | null
  revision: int
```

Catalog pages and partitions rotate from the durable successor links. Within a partition, markers
sort lexically by full filename and resume strictly after `after_name`; reaching the end clears
`after_name` and advances the partition. The cursor advances for every inspected marker,
including stale, corrupt, temporarily ineligible, and claim-race records. This prevents one bad
candidate from pinning progress. Cursor writes are advisory and may repeat work after a crash;
authoritative claim fencing prevents duplicate execution.

The project index state gates discovery before any candidate scan:

- `absent` and `building` retain bounded legacy Task discovery so pre-cutover queued work remains
  live;
- `active` uses only ready catalog, partition, and marker records for ordinary new claims;
- `degraded` permits control, recovery, maintenance, terminal convergence, and repair, but rejects
  new claims for the affected project.

A permanently stale marker is classified against authoritative Task and Submission truth twice:
once during candidate handling and again immediately before exact-generation deletion. A corrupt
active marker or catalog/partition record degrades the project instead of falling back to Task
discovery. A temporarily unavailable or machine-ineligible marker remains in the index and still
advances candidate progress.

The durable index state has one revisioned record:

```yaml
ready_index:
  schema_version: 1
  state: absent | building | active | degraded
  writer_capability: ready-v1 | null
  revision: int
  build:
    build_id: str
    phase: inventory | backfill | audit | completed
    is_repair: bool
    watermark:
      page_count: int
      task_count: int
      captured_at: str | null
      is_complete: bool
    cursor: {page: int, offset: int}
    audit_cursor: {page: int, offset: int}
    processed: int
    repaired: int
    stale_removed: int
    started_at: str
    completed_at: str | null
  degraded_reasons: [str]
  updated_at: str
```

The first current-generation dispatch cycle with free GPU capacity installs the schema writer gate,
commits `building`, and streams the legacy Task-name watermark into immutable pages of at most 64 IDs.
The inventory stream has bounded memory but may perform one complete legacy directory pass; this is
the final history-sized operation and is never used after activation. Each later maintenance slice
opens at most 64 exact watermark Tasks and commits the resulting page/offset cursor at the slice
boundary. A crash may repeat at most one bounded slice safely. Writers operating after `building`
begins publish the new marker before Task truth, so Tasks created during watermark capture need not
be members of the legacy watermark.

Backfill creates a new generation for every queued Task whose current marker is missing, corrupt,
or unreachable from its reservation, partition, and catalog. Non-queued Tasks have any current
marker retired. A second bounded pass audits the same watermark. Activation requires the audit to
finish, the schema gate to remain installed, and every recently active machine agent to advertise
`ready-v1`; otherwise the project enters `degraded`. The single `state: active` replacement is the
cutover point. Ordinary active scheduling never reads build pages or falls back to Task history.

`doctor repair` first audits an active projection. On damage it commits `degraded`, moves the
advisory home/shared marker trees, catalogs, reservations, allocators, and cursors beneath the new
build's `replaced-projection/` directory, and rebuilds from Task truth. The move preserves forensic
evidence and never moves or rewrites Task, Attempt, Submission, claim, or GPU-reservation authority.
Repair uses the same 64-record cursor slices and restores `active` only after the final audit.

### 13.1.1 Portable Work Slice Contract

One scheduling work slice has environment-independent hard bounds:

- at most 64 candidate, deadline, Attempt, and durable-operation records combined are inspected;
- at most 64 authoritative Task JSON records and 64 Attempt JSON records are read;
- at most 256 counted filesystem operations are initiated;
- at most 256 reservation records are enumerated before the slice yields;
- at most 256 in-memory candidate descriptors and one scheduler diagnostic snapshot are retained.

The limits apply per slice, not per project, and are never increased by local benchmarks. Capacity
reconciliation may consume the reservation-record allowance before candidate selection. A full
machine performs no ordinary ready-candidate Task reads; it may still read exact Task/Attempt truth
referenced by active reservations and due control operations.

Each slice starts with `deadline = monotonic_ns() + 50_000_000`. Before starting each independent
record, the scheduler checks both count allowances and `monotonic_ns() < deadline`. It never starts
another record after either boundary is reached. The deadline cannot interrupt one synchronous I/O
already in progress. Heartbeat and authority renewal remain on their independent control threads.

The disposable in-process adaptive batch state starts at 4 records, has minimum 1 and maximum 64,
and is not persisted. For positive record duration `sample`, it maintains:

```text
estimate[0] = sample
estimate[n] = max(ceil((3 * estimate[n-1] + sample) / 4), sample)
target = clamp(floor(50_000_000 / estimate[n]), 1, 64)
```

If `target` is below the current batch, shrink immediately to no more than both `target` and half
the current batch. Growth requires three consecutive observations with a larger target and then
adds exactly one record. Any slowdown resets the growth streak. Delay, error, or corrupt-record
outcomes still consume the inspected-record allowance and advance the cursor. When due work remains
after yielding, the agent schedules an immediate continuation after giving control threads an
execution opportunity; it does not wait a complete agent loop interval.

Diagnostics use monotonic durations and count, at minimum, `maintain_project`, `offer_due_tasks`,
`run_dispatch_cycle`, reservation enumeration entries, and authoritative Task JSON reads. They are
machine-local, replaceable observations and never scheduling truth. The latest snapshot is persisted
at most once per 30 seconds so observation does not add a durable write to every cycle. Wall time is
diagnostic only; portable regression acceptance is based on operation counts and bounds.

### 13.1.2 Machine-Wide Resource Gate

Before ordinary Task discovery, the machine agent forms a trustworthy capacity snapshot:

1. under the machine reservation lock, expire unattached provisional records and snapshot active
   reservation identities;
2. release the machine lock and verify each active record through its stable project binding and
   exact Task and Attempt identity;
3. reacquire the machine lock for any retag or release and apply it only when `reservation_id`,
   `acquisition_id`, `project_id`, `task_id`, `attempt_id`, `fencing_token`, and GPU IDs still match
   the original snapshot;
4. form a second lock-consistent snapshot containing both active and unexpired provisional GPU
   occupancy;
5. enter ordinary Task discovery only when that final snapshot contains visible free qexp GPUs.

Shared Task or Attempt I/O is never performed while holding the machine reservation lock. A missing
or unreadable project binding, malformed identity, blocked Task, or otherwise unverifiable ownership
retains the reservation and is counted as isolated; uncertainty is never converted into free
capacity.

Machine-agent recovery of `starting` Attempts is driven only by the final active-reservation
snapshot. It opens the exact project Task and Attempt and requires the reservation ID, Attempt ID,
fencing token, machine, claim, and launch state to agree before replaying the existing launch gate.
The machine-agent path does not scan the Task directory to find `starting` Attempts. The standalone
project scheduler retains its legacy scan while the ready index is `absent` or `building`, switches
to ready-only discovery at `active`, and fails closed for new claims at `degraded`.

At zero free capacity, heartbeat and authority threads continue independently and project
maintenance still processes its current durable responsibilities, while `run_dispatch_cycle` is not
called. Phase diagnostics separate `recovery.reservation_reconciliation`,
`recovery.starting_attempts`, `maintenance.project`, and `scheduler.work`, and include trustworthy
reservation and visible/reserved/free GPU counts.

Scheduling runs immediately at machine-agent startup. A cycle observes the current binding
revision before project selection. Capacity released by recovery, maintenance, or a completed
claim is reused by the next fair round without waiting for the loop interval; a ready marker written
by local maintenance is therefore visible to the work plane in that same cycle. If no local event
continues work, the existing loop interval is the correctness-preserving idle probe for remote
markers and later binding changes. No filesystem notification facility is required.

The candidate working directory must be locally readable and searchable before claim. Failure
is a per-project diagnostic and does not prevent other bindings from dispatching. A later launch
failure follows the normal fenced terminal-compensation path.

### 13.2 Eligibility

Before reserving capacity, an agent verifies advisory eligibility. Before claiming, it
must revalidate under authoritative locks:

- Submission Operation is committed
- Task is queued and unclaimed
- Group dispatch is active, if grouped
- machine is an active non-draining Group worker
- home queue is claimed only by home machine
- shared queue claimant is allowed by Worker Set and fallback constraint
- Task requires no more locally reservable qexp GPUs than available

### 13.3 Provisional Reservation

The agent creates a TTL-bound local provisional reservation keyed by an acquisition ID.
It must not reserve more Tasks than it can promptly launch.

If shared-lock acquisition is delayed beyond the provisional TTL, the agent releases and
restarts acquisition rather than extending capacity indefinitely without ownership.

### 13.4 Global Claim

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

### 13.5 Final Launch Gate

Immediately before local process creation, the agent acquires Group then Task lock and
validates:

- Task still references the same claim and token
- lease remains valid
- Attempt exists in `claimed`
- Group dispatch is active
- no applicable cancellation barrier won
- machine remains an active Worker Set member

The winning fenced transition uses one authorization timestamp and changes:

```text
active_claim.launch_state: claimed -> starting
active_claim.launch_authorized_at: null -> timestamp
Attempt.phase: claimed -> starting
Attempt.timestamps.launch_authorized_at: null -> same timestamp
```

The Task write is the launch gate linearization point. If a crash leaves the matching Attempt in
`claimed`, a later fenced authorization replay may only reconcile it to `starting` using the
persisted Task timestamp; it must not authorize a second Runner. It records the current Group
dispatch epoch and Worker Set epoch.

This transition is the linearization point:

- if pause, cancellation, expiry, or Worker Set control commits first, launch fails
- if `starting` commits first, pause allows the Attempt to proceed
- later default cancellation treats it as running work
- later terminating cancellation publishes termination intent

A plain read followed by process creation is invalid.

### 13.6 Process Creation

After launch authorization, the agent starts a passive runner. The runner writes its immutable
launch intent, creates the guardian-owned training process under the assigned GPUs, writes one
immutable local process registration, and later writes a separate exit observation. The agent
materializes the mutable local process manifest from that registration and exclusively owns all
authority transitions. The Runner records `process_created_at` in its registration immediately
after guardian creation. The Agent validates the registration's Attempt ID, machine, fencing token
and process identity, then writes the registration identity and `process_created_at` to Attempt,
sets `Attempt.phase` and claim `launch_state` to `running`, and writes `running_at` once. Replayed
materialization may only complete a matching partial transition; it must never refresh these
first-write timestamps. Only the Agent may publish shared running or terminal truth, perform
Recovery, release the reservation, or issue a qexp signal.

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

Schema-6 renewal returns a tagged outcome rather than a boolean. `renewed`,
`retryable_error`, `authority_changed`, `orphaned_recovery_required`, and
`termination_requested` have distinct required handling. A retryable error enters local
`suspect`, retains the reservation, records a durable diagnostic event, and retries only
before the holder safe deadline. It is not evidence that the token changed.

The agent caches every successfully validated lease policy in local storage. If the shared
policy or Task root cannot be read, it continues local process supervision and writes an
`authority-diagnostics/<attempt-id>.json` event. It uses the cached policy and last known
lease expiry to move from `suspect` to `isolated`; without a cached deadline it remains
`suspect`. Shared I/O failures are non-fatal to the agent loop and never authorize a signal,
reservation release, Recovery, or replacement Attempt.

The shared lease policy is stored at `project/lease-policy.json`. The default is a 120
second TTL, 10 second normal renewal interval, bounded jittered retry, 1 second maximum
clock skew and 5 second renewal commit margin. Holders stop ordinary work at
`lease_expires_at - max_clock_skew_seconds`; reclaimers wait until
`lease_expires_at + max_clock_skew_seconds`. The authority gate requires healthy chrony
tracking. `lease_loss_action` is fixed to `isolate`. A retryable renewal failure enters
`suspect`; after the holder safe deadline it enters `isolated`, retains its GPU reservation, and
continues the training process without publishing new shared qexp state. Isolation has no
automatic kill deadline; explicit authority changes use durable termination.

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

Before any Recovery CAS, agent and doctor acquire the local per-Attempt control lock and inspect
durable termination decisions. Runner never performs Recovery. A decision with shared commitment
`committed` or `unavailable`, or local state `signal_committed` or later, rejects Recovery.

### 15.5 Durable Termination

Every qexp-initiated signal has a local durable decision under
`termination-decisions/<attempt-id>/<decision-id>.json`. Its irreversible sequence is
`pending -> signal_committed -> sigterm_sent -> sigkill_sent -> confirmed`.

Only agent or a doctor holding the local Attempt control lock may issue a qexp signal. The runner's
`wait()` result is an exit observation only; it never confirms a termination decision. `confirmed`
is written only after the recorded PGID and start-time identity is absent. For normal authority
loss, the Task lock first commits `termination_decision_id` and token;
Recovery CAS then rejects that Attempt. When authority is unavailable after the holder
safe deadline, qexp fsyncs `shared_commitment=unavailable` before `signal_committed` and
the signal. This local exception prevents an unsafe local recovery and is reconciled by
`decision_id` after shared storage returns. A process exit caused by this path records its
lease termination reason and must not become `process_exited_without_status`.

### 15.6 Other Return Cases

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

`qexp task retry <task-id>` is accepted only when no active claim exists and either:

- the Task projection and current Attempt are both `failed`; or
- the Task projection is `blocked` and the current Attempt is `orphaned`.

For a failed Task, under the Task lock it:

1. validates no active claim exists
2. validates the Task projection and current Attempt are both `failed`
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

For a blocked Task whose current Attempt is orphaned, the same manual retry command is the
explicit operator decision. Under the Task lock it atomically:

1. validates no active claim exists
2. validates the Task projection is `blocked` and the current Attempt is `orphaned`
3. increments the Task fencing epoch
4. records an `orphan_superseded_by_retry` audit event containing the old Attempt ID, old
   fencing token, operator, and timestamp
5. clears `current_attempt_id` while preserving the old Attempt number, record, and expired
   token history
6. restores queue scope according to the preserved placement runtime contract and changes
   the Task projection to `queued`

The next winning claim creates a new Attempt with a higher fencing token. The old Attempt
remains `orphaned`, and every late write carrying its obsolete authority is rejected.

Retry supersedes the old Attempt's qexp execution authority. It does not inspect the old
machine, assert physical process termination, or undo external side effects. The manual retry
command requires no additional duplicate-risk acknowledgement flag. During one compatibility
cycle, an existing `--acknowledge-duplicate-risk` argument may be accepted only as a deprecated
no-op and must not change the transition. Automatic retry and `qexp group retry-failed` never
select blocked or orphaned work.

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

1. acquires its local lifecycle lock and scans launch intents, immutable process registrations,
   and exit observations
2. materializes missing agent-owned manifests; a dead wrapper with no registration becomes
   `launch_unverifiable`, retains its reservation, and remains isolated for manual verification
3. scans unfinished termination decisions and, under each Attempt control lock, replays signals
   or confirms recorded process-group identity absence
4. validates shared authority for every remaining local Attempt, then renews, recovers,
   finalizes, or isolates it as its durable evidence permits
5. reconciles provisional and active reservations after authority work completes
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
- incomplete ready-index build, incompatible active writer, missing marker, and marker unreachable
  from its reservation, partition, or catalog

Safe automatic repairs include:

- rebuilding indexes and summaries
- resuming a ready-index cursor, rebuilding its advisory projection from Task truth, and restoring
  a degraded project only after the final consistency audit
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
- manual retry can supersede an unclaimed blocked/orphaned Attempt without an additional risk
  flag, issues a higher fencing epoch, and remains excluded from Group retry
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
