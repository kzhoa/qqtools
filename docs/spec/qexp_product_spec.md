---
doc_type: spec
status: active
updated_at: 2026-08-31
archived_at:
---

# qexp Product Spec

## Schema 6 clock capability and local-safe execution

This section supersedes prior statements that every qexp path needs chrony. Qualified clock
capability permits `bounded_lease`; without it, machines that already satisfy Task queue scope,
Worker Set, and fallback rules may win `holder_bound` claims. Locks, CAS, fencing, process
identity, and local GPU reservations still serialize initial execution, while time-based
cross-machine takeover is deliberately disabled.

`queued_home` remains home-only. In `queued_shared`, home and eligible remote workers compete
and the winning machine's clock capability decides authority mode. Users can use `qexp task
share`, `share --after 10m`, repeated `--with`, and `qexp task keep-local`; `task offer` remains
the immediate operation for an existing spillover policy. Doctor and agent status show provider,
full/local-safe capability, authority mode, and exact blocked reasons.

## 1. Purpose

This document defines the product contract of `qexp`:

1. which real user problem qexp exists to solve
2. how users are expected to operate it across one or several machines
3. which concepts belong in the public mental model
4. which scheduling and recovery behaviors are required
5. what qexp deliberately does not own

Runtime truth layout, locking, claims, leases, fencing, and repair implementation belong
in [qexp_runtime_spec.md](qexp_runtime_spec.md).

The Group/Task/Attempt model rationale and delivery history are recorded in
[021-qexp-experiment-group-task-scheduling.md](../pitch/arxiv/021-qexp-experiment-group-task-scheduling.md).
The schema-6 authority ownership delivery history is recorded in
[027-qexp-agent-owned-authority.md](../pitch/arxiv/027-qexp-agent-owned-authority.md).

## 2. Contract Status

This is the authoritative schema-6 product contract. The installed implementation uses the
Group, Task, and Attempt model, including cross-machine claims, leases, and fencing.

Where the current runtime spec conflicts with this document, this document defines the
target product behavior and the runtime spec defines the corresponding persistence and
recovery protocol.

Document authority is divided as follows:

- this product spec owns user-visible concepts, commands, state semantics, and safety
  requirements
- `qexp_runtime_spec.md` owns storage layout and the concrete concurrency, lease, fencing,
  compensation, and repair protocols once rewritten for the new schema
- the scheduling pitch records rationale and delivery tracking but is not a competing
  normative contract

Anything marked **Assumption / Unverified** is not a guarantee of the installed version.

## Compatibility policy

Project-wide temporary compatibility lifecycles are registered in
[`compatibility-registry.toml`](compatibility-registry.toml) and governed by
[`compatibility-governance.md`](compatibility-governance.md). The registry schedules removal and
purge work; this specification remains authoritative for qexp behavior and protected workflows.

Workflows listed under **Protected workflows**, and workflows explicitly marked as stable in
this specification, are stable public behavior. Experimental workflows are excluded only when
explicitly labeled experimental.

A change is backward incompatible if an existing documented successful workflow:

- becomes invalid;
- requires an additional mandatory user step;
- gains a new prerequisite;
- changes supported persisted-state interpretation; or
- implicitly changes project or agent ownership.

Such changes require an explicit compatibility decision recorded in the relevant requirement
pitch or delivery change description, with explicit approval.

## Protected workflows

- New project activation: `qexp init -> qexp agent start`
- New project submission: `qexp init -> qexp submit -- <command>`
- Legacy project migration: `qexp agent migrate-project`

`init` succeeds only after it initializes the project, writes the current machine configuration,
and registers the current-generation binding; it does not start an agent. `agent start` starts
the global agent only for an existing binding. It neither initializes nor registers a project and
must not require `add-project` in the new-project workflow.

Unless `--no-activate` is supplied, `submit` activates the local agent for an existing binding
and submitted work eventually converges through its normal scheduling lifecycle. `--no-activate`
is an explicit non-activation exception, not an additional prerequisite for the main workflow.

`migrate-project` handles only a project carrying legacy metadata. It verifies and stops the old
agent, imports that project's local evidence, and enables its binding. A failed migration must not
mark the project migrated or release or overwrite resources belonging to another project.

`add-project` is an operations command for restoring a missing or removed current-generation
binding; it is not part of new-project setup.

Recommended reading order:

- Sections 3-5 explain product position, deployment reality, and goals.
- Sections 6-10 define the durable product model and scheduling contract.
- Section 11 grounds the contract in representative workflows and failure cases.
- Sections 12-16 define commands, observation, and operating boundaries.
- Section 17 is the release acceptance gate.

## Breaking Schema Cutover

The Group/Task/Attempt model intentionally does not provide backward compatibility for
Batch-era control data.

Rules:

- Batch-era control data is not migrated
- the new implementation does not provide legacy readers or compatibility commands
- unsupported schemas fail before any agent or mutating command starts
- the implementation should fail fast on an unsupported schema instead of partially
  interpreting old truth
- a drained schema-5 root may be upgraded by `qexp migrate --to-schema 6`; it must have
  no active claim or running Attempt, and no mixed-schema runtime is supported

Loss of Batch-era qexp scheduling metadata is a known and accepted product risk. The schema-5
to schema-6 migration is a one-way control-root upgrade, not Batch-era compatibility, export,
or a legacy reader.

## 3. Product Summary

`qexp` is a lightweight, project-scoped experiment command queue for a trusted set of GPU
machines.

It is local-first:

- a Task defaults to the machine from which it is submitted
- each machine has at most one active global `qexp agent`, serving explicitly registered projects
- process, PID, local launch backend, and GPU reservation remain machine-local; qexp-captured
  Attempt stdout/stderr logs are shared for cross-machine inspection

`tmux` is the primary interactive launch and observation backend that qexp is designed to
work with. qexp also permits degraded operation when `tmux` is unavailable by launching
local detached processes directly, but feature depth and performance commitments are made
for the `tmux` path first.

It may optionally cooperate across machines:

- a Task can permit spillover beyond its home machine
- users or elapsed-time policy can offer unclaimed Tasks to a shared pool
- idle eligible agents can pull shared Tasks
- one globally exclusive claim decides who executes

qexp does not require a central scheduler service and does not act as a remote shell.

It is responsible for:

- submitting one Task
- reliably submitting multiple flat Tasks from a manifest
- organizing Tasks into a bounded Experiment Group
- recording scheduling and execution Attempt history
- running a lightweight local agent
- showing Group, Task, Attempt, machine, and queue state
- cancelling and retrying work
- coordinating optional home-first spillover
- diagnosing and repairing scheduler metadata

It is not responsible for:

- training log formats or metric schemas
- artifact or scientific result management
- source snapshots or Git revision enforcement
- remotely logging into machines or starting their agents
- physical GPU utilization scheduling outside qexp reservations
- a permanent background service by default
- a general cluster scheduler or workflow DAG engine

## 4. Why This Product Exists

### 4.1 Primary Shared-Filesystem Deployment Model

The primary qexp deployment model is several machines sharing both the project filesystem
and the project scheduling control state. This is a first-class product premise, not an
incidental compatibility scenario.

The shared filesystem carries two different categories of data:

- project code, configs, and user-owned experiment inputs under one common absolute path
- qexp coordination truth under the project-level `.qexp` root

Runtime processes remain local even though code and scheduling truth are shared.

Typical users have:

- about 10 GPU servers, sometimes more
- one project directory mounted at the same absolute path on all machines
- one shared project `.qexp` control root
- one or more registered qexp machine identities per physical server, with one identity for
  each independently scheduled GPU resource pool
- one local agent per registered machine
- machine-local GPU, PID, local launch backend state, and runtime state
- experiment sets containing tens or hundreds of independent commands

The project source is usually edited once on the shared filesystem. Machines execute the
code visible at execution time. qexp does not freeze the source tree.

### 4.2 Why Users SSH and Submit Repeatedly

Agents use on-demand mode by default because users do not want an unnecessary permanent
service on every research machine.

After a long period without qexp submissions, agents are likely to have exited. qexp does
not promise remote wake-up. The user therefore needs to SSH into each intended machine and
start or wake its agent.

Once the user already has 10 SSH sessions open, submitting one machine-specific manifest
in each session is natural and efficient:

```text
g1: qexp batch-submit --group stage-c1 --file runs-g1.yaml
g2: qexp batch-submit --group stage-c1 --file runs-g2.yaml
...
g10: qexp batch-submit --group stage-c1 --file runs-g10.yaml
```

This is not a product failure or an obsolete workflow. It expresses deliberate ownership:
each machine receives first responsibility for the Tasks submitted there.

### 4.3 Why One Submission Must Also Work

The same user may prefer to submit all 200 Tasks from g1 and then activate the remaining
agents:

```text
g1: submit all work
g2...g10: start agents later
```

This should not require 10 public Batch objects or manual reshaping of the experiment.
Tasks should begin with a home machine and enter shared spillover only when user policy
permits.

### 4.4 Why Capacity Must Be Borrowed Dynamically

Machine load is rarely balanced:

- one machine may have a long local queue
- another machine may become idle after finishing earlier work
- agent activation times may differ
- GPU counts and Task GPU requirements may differ

Users need local ownership without permanently stranding work on an overloaded machine.
The required behavior is owner-first work stealing:

1. the home machine gets first refusal
2. the Task remains unclaimed while waiting locally
3. policy may offer it to a shared pool
4. any eligible idle agent may claim it
5. the home agent may also claim it later if it becomes free

Users express this through story-level controls rather than editing placement fields:

```bash
qexp task share <task-id>
qexp task share <task-id> --after 10m
qexp task share <task-id> --with g2 --with g3
qexp task keep-local <task-id>
```

`share` expands the candidate set; it does not transfer ownership away from the home machine.
`keep-local` clears sharing policy and deadlines while preserving the Task home machine.

### 4.5 Why Failure Recovery Is Conservative

A missing heartbeat does not prove that a process stopped. The machine may be computing
while disconnected from the shared filesystem or network.

Automatically starting the same Task elsewhere can therefore create duplicate execution
and conflicting side effects. qexp must distinguish:

- an unclaimed queued Task whose home machine is unavailable
- a pre-launch claim that can be safely released
- a running Attempt whose process state is unknown

Only the first two may be automatically returned to schedulable work under defined rules.
An ambiguously running Attempt becomes orphaned and blocks automatic retry.

## 5. Product Goals

qexp must:

- keep single-Task submission lightweight
- preserve current-machine behavior by default
- let several machine-local submissions populate one Experiment Group
- support one-machine submission followed by gradual agent activation
- use home-first placement rather than immediate unrestricted global competition
- allow explicitly permitted spillover to idle machines
- keep logical Task counts stable across retries
- preserve concrete machine-specific Attempt history
- support later additions to an open Experiment Group
- make dangerous uncertainty visible rather than hiding it behind automatic failover

## 6. Core Mental Model

The public model is:

```text
Project
  -> Experiment Group
       -> Task
            -> Attempt 1
            -> Attempt 2

Machine <- exclusive claim -> active Attempt
```

### 6.1 Experiment Group

An Experiment Group is one clearly bounded logical experiment collection.

It may receive Tasks:

- through multiple commands
- from multiple machines
- at different times
- with different home or fallback placement

An open Group may accept a few later control experiments. A sealed Group rejects additions
until explicitly reopened.

Group is the primary observation and management boundary.

### 6.2 Task

A Task is one logical command that should eventually succeed, fail, be cancelled, or
become blocked pending recovery.

Task identity is stable across retry. Task owns:

- command and working directory
- requested qexp GPU count
- Group membership
- home machine
- sharing policy and fallback constraint
- current queue scope
- current Attempt reference

Task does not own a permanent PID, GPU assignment, exit code, or execution machine.

### 6.3 Attempt

An Attempt is one concrete execution of a Task.

Attempt owns:

- attempt number
- assigned machine and GPU IDs
- claim and fencing token
- lease timestamps
- PID, process group, optional tmux reference, and log references
- start and finish timestamps
- exit code, signal, and terminal reason

Attempt phases are:

```text
claimed | starting | running | succeeded | failed | cancelled | orphaned
```

`claimed` is pre-launch and revocable through the fenced control protocol. `starting` means
the final launch gate has committed and local process creation is authorized.

Retry authorizes the next Attempt and returns the same Task to a queue. The concrete
Attempt is created when a machine later claims that Task, because machine, GPU, claim, and
fencing data do not exist before claim.

### 6.4 Machine

A Machine is one registered, independently scheduled GPU resource pool with a local agent
and local runtime. It is a qexp scheduling boundary, not a physical server identity or other
physical machine entity.

One physical server may expose multiple qexp Machines. For example, a server with eight GPUs
may expose two Machines with four GPUs each. Each Machine owns only the GPUs visible and
permitted to that Machine; qexp does not infer the physical server topology, discover GPUs
assigned to other Machines, or coordinate reservations outside the current Machine boundary.

Concurrently active Machines must not expose overlapping underlying GPU resources unless an
external isolation layer guarantees exclusive access. Separate machine identities and local
runtime directories do not by themselves prevent physical GPU oversubscription across
overlapping resource pools.

It owns:

- machine identity
- local agent lifecycle
- local qexp GPU reservations
- local process namespace and optional tmux-managed interactive sessions
- machine heartbeat and snapshots

A Machine may be the home machine for one Task and a fallback worker for another Task. For a
single Task, its home machine remains eligible by definition and must not also be listed as a
helper.

### 6.5 Submission Operation

A Submission Operation is an internal transaction for safely creating one or more Tasks.
Both `submit` and `batch-submit` may use it so single and bulk creation share one atomic
runtime path.

It is not a public management object. Users do not pause, cancel, retry, or monitor work by
submission-operation ID.

It persists both:

- a canonical raw-request digest over normalized semantic manifest content and
  submission-affecting CLI arguments
- an immutable resolved-context digest covering the original submitting machine, resolved
  Task specifications and IDs, target Group, placement constraints, and Worker Set change
  plan

### 6.6 MachineRuntime and Machine Agent

`MachineRuntime` is the disposable, machine-local resource authority for one qexp Machine.
It owns the unified visible GPU reservation set, process and tmux supervision, local recovery
evidence, machine scheduler lock, project registry, and cross-project scheduling cursor. It is
not a project control plane: each bound project's `.qexp` remains authoritative for its Tasks,
Groups, Attempts, claims, leases, fencing tokens, logs, and terminal transitions.

A machine agent binds explicitly registered projects to that one resource authority. A project
binding contains the stable project ID from `project/identity.json`, canonical shared root, the
project-local machine name, and persistent `enabled` state. The registry has no Task copies,
commands, environments, credentials, or project lifecycle truth. Its observable state is:

- `enabled`: the binding may supply new candidates;
- `draining`: the binding is disabled but local Attempts, reservations, process evidence, or
  pending terminal writes still require supervision;
- `disabled`: the binding is disabled and has no such local blockers.

Disable excludes the binding from subsequent global-agent scheduling cycles while preserving
supervision and convergence. Removal requires `disabled` and no local blockers; it never
force-removes a project with active evidence. After terminal truth converges, machine-local
execution evidence is consumed rather than retained as history; successful removal deletes the
binding's disposable local runtime partition.

The default machine runtime root is `~/.qqtools/qexp-machine/`. Every global-agent path resolves
the same root from `QEXP_MACHINE_RUNTIME_ROOT` when set, otherwise that default. Path resolution
is read-only; scheduling and registry mutations create and validate
the writable layout on demand. The resolver rejects a project `.qexp` root; deployment must not
place it on a shared control filesystem. The shared scheduler lock is held for the agent lifetime,
so no second qexp agent can acquire local GPU scheduling authority.

Machine-local execution records use `(stable_project_id, task_id, attempt_id)` as their identity;
project-local IDs alone are insufficient. Within each local admission layer, the machine agent uses
deterministic unweighted round-robin over enabled bindings sorted by stable project ID. The primary
layer is considered before borrow; a project with no eligible candidate, no fitting capacity, or a
failed project-level claim does not block the remainder of its layer. Project-level eligibility and
fenced claim protocols remain the final authority.

## 7. No Public Batch Entity

`batch-submit` is retained as a familiar command name and manifest convenience. It means:

> Reliably validate and submit multiple independent Tasks.

It does not create a public Batch ID.

Rules:

- new Tasks have no public `batch_id`
- Tasks submitted by one invocation do not form a permanent user-visible subgroup
- one invocation must still be atomic from the scheduler's visibility perspective
- interrupted bulk submission must be recoverable and idempotent
- `batch-submit` must not make single-Task submission more complex

Old Batch records are not inspectable through the new product. The new implementation
must reject unsupported old schema instead of preserving a legacy Batch surface.

## 8. Experiment Group Lifecycle

Group uses separate admission and dispatch controls:

```text
admission_state: open | sealed
dispatch_state: active | paused
```

Rules:

- `open` accepts new Tasks
- `sealed` rejects new Tasks
- `active` permits new claims
- `paused` prevents new claims while running Attempts continue
- sealing does not pause work
- pausing does not prevent an open Group from receiving queued Tasks
- completion is derived from Task truth and does not permanently close the Group

Derived work labels are:

- `active`: at least one current Task is queued or running
- `settled`: every current Task is terminal and no Task is blocked
- `blocked`: at least one Task has unresolved execution safety

`settled` does not imply success; a settled Group may contain failed or cancelled Tasks.
An `open + settled` Group is valid. Adding later control experiments makes it active again.

Required operations:

```bash
qexp group seal <group>
qexp group reopen <group>
qexp group pause <group>
qexp group resume <group>
qexp group cancel <group>
qexp group cancel <group> --terminate-running
qexp group retry-failed <group>
```

Default Group cancellation cancels current queued Tasks and allows running Attempts to
finish. The operation targets the Group membership snapshot captured when cancellation
starts. It does not cancel Tasks appended later and does not permanently change Group
admission or dispatch state. Users who need a sustained stop should pause and, if needed,
seal the Group before cancelling current work.

Pause and cancellation must linearize against process launch through one shared-state
transition. For a pre-launch Attempt in `claimed`, exactly one outcome wins:

- if a pause or cancellation barrier commits first, launch authorization fails, the
  Attempt becomes cancelled, and its claim and provisional GPU reservation are released
- if `claimed -> starting` commits first, the Attempt is durably authorized to launch; a
  later pause lets it proceed, while cancellation applies the requested running policy

`starting` means durably authorized to create the local process. The owning agent must
perform this fenced transition immediately before process creation. It must not implement
the launch gate as an unfenced read followed by a local launch.

Attempt timing distinguishes authorization, local creation and shared confirmation:

- `launch_authorized_at` records the successful fenced launch gate;
- `process_created_at` records successful guardian creation by the passive Runner;
- `running_at` records the owning Agent accepting the fenced registration into shared running truth.

Human-facing execution duration is measured from `process_created_at` to `finished_at`. If that
creation time is unavailable, it may use `running_at`; it must never use `claimed_at`, which also
includes reservation and launch delay. Historical or unverified Attempts without either start
record display an unavailable duration.

## 9. Home-First Placement Protocol

### 9.1 Submission Policy

Recommended manifest shape:

```yaml
group:
  workers: [g1, g2, g3, g4]

defaults:
  placement:
    home_machine: current
    sharing:
      mode: spillover
      fallback_machines: group
      offer:
        after_seconds: 600

tasks:
  - name: exp-001
    command: ["python", "entry.py", "--config", "exp001.yaml"]

  - name: private-control
    placement:
      sharing:
        mode: private
    command: ["python", "entry.py", "--config", "private.yaml"]
```

Defaults:

- `--group` is the sole source of Group identity for both `submit` and `batch-submit`
- `group.name` in a manifest is invalid rather than a second source with precedence rules
- a manifest `group` configuration block requires `--group` and configures fields such as
  the initial or additional Worker Set
- `group.workers` is the only manifest Worker Set input; root `workers` and
  `defaults.placement.workers` are invalid
- omitted `home_machine` resolves to the verified submitting machine
- omitted `sharing.mode` means `private`
- Task `placement` overlays `defaults.placement` field-by-field rather than replacing the whole
  object
- no Task is remotely claimable unless the user explicitly permits it
- an ungrouped Task must remain private because no Group Worker Set exists to bound remote
  execution
- an ungrouped private Task may use a non-current home machine when that machine has a valid
  current-generation Project machine record; the Task then remains claimable only by that home
  machine
- legacy Task-level `sharing_mode`, `fallback_machines`, and `offer_after_seconds` are accepted
  only as deprecated aliases for the nested fields; declaring both forms for the same semantic
  field is invalid

`home_machine` expresses first refusal. It must not create an execution claim at
submission time.

Task placement must separate user authorization from scheduler state:

```text
placement_policy:
  home_machine
  sharing:
    mode: private | spillover
    fallback_constraint
    offer_policy

placement_runtime:
  queue_scope: home | shared
  offered_at
  offer_reason
  offered_by
```

Rules:

- policy is revisioned user intent
- runtime state may change only within policy
- `private + shared` is invalid
- agents may offer a spillover Task but may not convert a private Task into spillover
- policy changes while claimed or running must not revoke or alter the current Attempt

### 9.2 Sharing Policies

Required policies:

```text
private     only the home machine may claim
spillover   the Task may enter the shared pool under its offer policy
```

Agents may open queue scope only within the user-approved policy. An agent must never turn
`private` into `spillover` on its own.

Fallback machines are restricted by:

- the Group Worker Set
- optional Task-specific constraints
- current machine registration and drain state

Effective remote eligibility is the intersection of all three constraints. No layer may
broaden the one above it.

### 9.3 Queue Scope

Queued Tasks use:

```text
queued_home | queued_shared
```

`queued_home`:

- has no active claim
- is claimable only by the home machine
- may become shared if policy permits

`queued_shared`:

- has no active claim
- remains claimable by the home machine
- is also claimable by permitted fallback machines
- stays shared until claimed or otherwise controlled

The home agent does not need a separate operation to take work back. It competes for the
same shared Task when it becomes idle.

### 9.4 Sharing And Offering Work

Queued placement controls may change only committed, unclaimed queued Tasks. They do not migrate
or stop a running Attempt.

Supported controls:

- `qexp task share <task-id>` makes a grouped Task immediately available to eligible Group
  helpers while retaining home eligibility.
- `qexp task share <task-id> --after 10m` records a bounded deadline. The home agent offers it
  only after current clock evidence proves the deadline has elapsed.
- `qexp task share <task-id> --with g2 --with g3` restricts helper eligibility to active Group
  workers. The home machine must not be listed because it remains eligible by definition.
- `qexp task keep-local <task-id>` resets policy to private, queue scope to home, and clears
  delayed-offer state.
- `qexp task offer <task-id>` is retained for already-spillover Tasks. It cannot convert a
  private Task into spillover.

Successful placement controls return a human-readable result by default and a stable JSON
envelope with `--format=json`. The envelope contains action, Task, Group, home machine, eligible
helpers, effective time, resulting queue state, idempotency, operation id, and message.

These are the only first-release triggers. The first-release manifest does not accept or
advertise `on_overload`, `min_local_wait_seconds`, `max_offer_per_cycle`, or
`cooldown_seconds`. Heartbeat staleness does not bypass `after_seconds`; the user may offer
the Task manually when earlier sharing is required.

Heartbeat-based early offering is a possible target capability, not a first-release
promise. It requires a separate future product contract and acceptance criteria.

Automatic overload offering is a future capability. It must not ship until deterministic
thresholds, bounded selection, cooldown behavior, observability, and acceptance tests are
specified. Machine snapshots may be advisory input to that future policy but can never
directly allocate a Task to a remote machine.

Elapsed-time offering does not require a central coordinator. Task commit persists
`queued_home_at` and `offer_eligible_at`. Any active Group worker agent may scan due
spillover Tasks and attempt a revisioned `queued_home -> queued_shared` transition.

Rules:

- timestamps use UTC wall-clock time on hosts required to run time synchronization
- manual and elapsed-time triggers are the alternative first-release ways to authorize
  the same idempotent transition
- a concurrent claim or control transition changes the Task revision and makes a stale
  offer attempt fail
- correctness comes from authoritative Task state and CAS, not from the scanning agent or
  a derived due-time index
- bounded clock skew may slightly change the first-refusal duration but cannot broaden
  placement authorization

### 9.5 Claims

A claim is created only when an agent can promptly execute the Task.

Agents must not pre-claim a large queue merely to reserve future work.

A pre-launch claim may be released through a revisioned audited transition. A running
Attempt cannot be migrated by clearing its claim. It must terminate first; later execution
uses a new Attempt.

Claim acquisition uses a TTL-bound provisional local GPU reservation keyed by an
acquisition ID. The reservation is attached to the authoritative Attempt only after claim
and Attempt creation succeed.

Every failure path must compensate idempotently:

- claim loss releases the provisional reservation
- Attempt creation failure releases the claim and reservation
- a lost launch gate cancels the pre-launch Attempt and releases both resources
- local process creation failure marks the Attempt failed and releases both resources
- terminal process reconciliation closes the claim and releases the reservation

The global agent and `doctor` reconcile
reservations against current claims, Attempt state, and local process identity. A machine agent
does this separately for each supervised project and may reconcile only reservations carrying
that project's stable ID. They may delete an expired unattached reservation only after proving
no matching process exists. Ambiguous process ownership becomes blocked recovery work instead of
being treated as free capacity.

## 10. Group Worker Set and Runtime Elasticity

A Group Worker Set defines which machines may execute Group Tasks. For shared Tasks, it is
also the maximum fallback-machine scope.

### 10.1 Primary and Borrow Resource Pools

Each active Group Worker has one scheduling role:

- `primary`: normal Group capacity on that machine;
- `borrow`: capacity that may receive a new claim only when the local machine agent has no
  primary demand on that machine.

Worker role is a local admission ordering rule after Task placement authorization. It does not
broaden a private Task, alter a Task's home/fallback policy, reserve capacity, or preempt an
existing Attempt. A primary demand that is runnable now or waiting only for qexp GPU aggregation
blocks new borrow admission. Existing borrow Attempts continue after primary demand appears;
only later borrow claims stop.

`gpu_limit_gpus` is `null` or a positive integer for both Worker roles. `null` removes only the
Group-level GPU limit; visible free qexp GPUs, placement, primary demand, and machine-wide
reservations still apply. A finite limit is checked by GPU count, not Task count. Lowering a
limit below current usage reports `over_limit` and blocks later admission without terminating
work. Under `QQTOOLS-COMPAT-0004`, N readers temporarily accept `borrow_limit_gpus` from
existing Group records and pre-unification submission declarations, then normalize it to
`gpu_limit_gpus`; all N writes and outputs use `gpu_limit_gpus`. N+1 retains that reader only
inside its automatic Group upgrade, which rewrites every legacy field before normal scheduling;
N+2 removes the reader and upgrade path.

For compatibility, a borrow Worker is persisted as both `scheduling_role: borrow` and
`state: borrow`. Current agents treat `active` and `borrow` as claimable lifecycle states and
apply the scheduling role. Agents that only recognize `state: active` skip `state: borrow` and
therefore fail closed; they must never schedule it as primary capacity. `draining` and `removing`
remain non-claimable.

An enabled local project binding is required on every target machine. `qexp init` creates the
normal binding; `qexp agent add-project` restores a missing binding. Submitting on one machine
does not wake another machine's on-demand agent. A running target agent observes later Group
truth changes without re-registration.

Worker Set invariants:

- every grouped Task's home machine must be a claimable, non-draining Group worker when the
  Task is committed
- submission never adds the submitting machine to the Worker Set implicitly
- `group create` defaults to `{current}` when `--workers` is omitted; an explicit `--workers`
  list is the exact initial Worker Set
- a single submit may target only an existing Group; a new Group may be atomically created by
  `batch-submit` only when its manifest declares a non-empty `group.workers` list
- an explicitly configured non-current home machine must already be a claimable worker
- a manifest may add workers explicitly but must not silently remove or replace the existing
  Worker Set
- removal and drain occur only through explicit Group machine operations
- a draining machine cannot claim either home or shared Tasks from that Group

Required operations:

```bash
qexp group machines add <group> g11 --role borrow --gpu-limit-gpus 2
qexp group machines set <group> g11 --role primary --gpu-limit-gpus 2
qexp group machines drain <group> g5
qexp group machines remove <group> g5 --terminate-running
qexp group machines list <group>
```

Adding a machine:

- makes it eligible for compatible queued shared Tasks
- does not modify running or terminal Attempts
- does not remotely start the agent

Role and GPU-limit changes are Group-lock-linearized. They affect later claims but do not
revoke a successfully created claim. Drain, remove, pause, cancellation, lease, and fencing keep
their existing launch-gate meaning.

Draining a machine:

- prevents new claims
- allows running Attempts to finish
- remains in `draining` until active work reaches zero and removal would not strand queued
  work

Removal must fail and report blocking Task IDs when any queued Task would become
unclaimable. In particular:

- a private Task whose home is the draining machine must be explicitly rehomed, have its
  queued placement policy changed, or be cancelled
- a `queued_home` spillover Task must be atomically offered, explicitly rehomed, or
  cancelled
- an already `queued_shared` Task may remain when another claimable worker is still allowed
  by its fallback constraint
- forced removal does not bypass queued-work safety checks

Forced removal:

- publishes local termination intent
- requires owning-agent acknowledgement where possible
- does not make ambiguous running work immediately safe to retry

## 11. Typical User Scenarios

### 11.1 One Local Task

```bash
qexp submit -- python train.py --config configs/a.yaml
```

Expected behavior:

- no YAML or Group is required
- home machine defaults to current machine
- remote claim is disabled by default
- the current machine agent may auto-start

### 11.2 Ten Machine-Local Submissions Into One Group

The user SSHs to each machine because its on-demand agent may be asleep.

```bash
# g1
qexp batch-submit --group stage-c1 --file runs-g1.yaml

# g2
qexp batch-submit --group stage-c1 --file runs-g2.yaml
```

The same pattern continues through g10.

Expected behavior:

- all Tasks belong to `stage-c1`
- each Task defaults to the machine where it was submitted
- no public Batch objects are created
- Group observation aggregates all 10 submissions
- this workflow remains fully supported and is not treated as a compatibility fallback

### 11.3 Submit Once, Wake Other Agents Later

The user submits all Tasks from g1 with spillover enabled, then SSHs to g2 through g10 and
starts their agents.

Expected behavior:

- Tasks begin with the configured home machine
- g1 claims only work it can promptly execute
- remaining Tasks stay unclaimed
- Tasks enter the shared pool only under their offer policy
- newly activated agents pull compatible shared Tasks
- one global claim prevents duplicate execution

If balanced simultaneous start matters, the user may submit into a paused Group, start all
agents, and then resume the Group.

### 11.4 Busy Home Machine Shares Work

The home machine has no immediate qexp GPU capacity. The user either waits for the Task's
`after_seconds` deadline or explicitly offers selected Tasks:

```bash
qexp task offer <task-id>
```

Expected behavior:

- only the selected or deadline-eligible unclaimed Tasks are offered
- it does not broaden policy beyond user-approved fallback machines
- it does not release or migrate running work
- idle fallback agents may claim from the shared pool
- the home agent may later claim remaining shared Tasks itself

### 11.5 Add Control Experiments Later

The initial 200 Tasks settle. The Group remains open. The user adds three control Tasks:

```bash
qexp batch-submit --group stage-c1 --file additional-controls.yaml
```

Expected behavior:

- Group total becomes 203
- prior Task and Attempt history is unchanged
- the Group becomes active again
- new control Tasks may use different home or fallback policy
- a sealed Group rejects the addition until explicitly reopened

### 11.6 Add or Remove Machines During Execution

The user adds g11 after the Group has started:

```bash
qexp group machines add stage-c1 g11
```

After g11's agent starts, it may claim compatible shared Tasks.

The user drains g5:

```bash
qexp group machines drain stage-c1 g5
```

g5 stops taking new work and finishes current Attempts.

### 11.7 Home Machine Disappears Before Claim

For an unclaimed spillover Task:

- its `after_seconds` deadline or explicit `qexp task offer` may move it to
  `queued_shared`
- another active eligible agent may claim it
- no orphan exists because no Attempt was active

For a private Task:

- it remains home-only
- the user must wait, change policy, or restore the home machine

### 11.8 Claimed Machine Disappears

If the agent disappears before process launch, recovery may safely release the expired
claim after verifying no process started.

For a running Attempt, heartbeat staleness and lease expiry have different meanings:

- while the heartbeat is stale but the Attempt lease remains valid, keep the Attempt
  running and show an explicit stale-machine warning
- when the lease expires, archive the claim as expired with its fencing token preserved,
  mark the Attempt orphaned, and project the Task as blocked
- do not change the Task to `queued_home` or `queued_shared`

Schema-6 treats an individual renewal I/O failure as a visible `suspect` condition, not as
an immediate task failure. The running process and its GPUs remain reserved while qexp
retries within the authoritative lease. Users can inspect lease diagnostics and any pending
termination reconciliation through `qexp doctor verify`.

The shared default lease policy is 120 seconds with a 10 second renewal interval. Long
training may use a larger authoritative TTL after a maintenance-window policy change. A holder
that loses shared storage does not use a runner-local grace period or auto-terminate at TTL:
its agent retains the process and reservation in `isolated` state until shared authority is
available again. Quarantine remains disabled.

The resulting state is:

```text
Attempt: running -> orphaned
Task: running -> blocked
```

qexp must not automatically start a replacement. An ordinary `qexp task retry <task-id>`
is the explicit operator decision for an unclaimed `blocked` Task whose current Attempt is
`orphaned`; it supersedes qexp authority for the old Attempt without confirming that its process
stopped or requiring an additional duplicate-risk flag. `orphaned` remains a recoverable
uncertainty state, not proof that the process terminated.

When the old machine returns, its agent must reconcile local processes against current
fencing tokens before taking new work. Lease expiry archives the old claim, so a returning
agent cannot perform an ordinary renewal.

A live process may recover the same Attempt only through a recovery CAS requiring:

- the Task is still `blocked`
- the current Attempt is the same `orphaned` Attempt
- no successor Attempt or newer token exists
- the agent presents the expired token recorded for that Attempt

Grouped recovery also respects current Group control:

- pause allows recovery because the Attempt was already running
- claimable workers may recover
- a draining worker may recover only the same Attempt when launch authorization predates
  the drain request
- removing or removed workers cannot recover execution authority
- applicable Group or Task termination intent rejects recovery and requires the old
  process to terminate or quarantine

Success issues a new fencing token and lease for the same Attempt and reconciles it to
`running`. CAS failure makes the local process obsolete; it must be terminated or
quarantined.

Other cases:

- if the process finished, publish its recorded terminal result
- if the process is absent, confirm local cleanup and resolve the Attempt as failed
- if a newer Attempt exists, reject stale writes and terminate or quarantine the old
  process

Fencing protects qexp scheduler truth. It cannot undo side effects already produced by an
old process, which is why ambiguous automatic retry remains forbidden.

## 12. Submission Commands

### 12.1 `submit`

```bash
qexp submit --task-id qm9-seed-1 --group qm9-study -- python train.py --seed 1
qexp submit --home-machine gpu-b -- python train.py --seed 1
```

Rules:

- single-Task submission must not require YAML
- Group is optional for ad hoc Tasks
- `--machine` is the verified local identity assertion; `--home-machine` selects Task placement
  and defaults to the verified current machine
- a private Task is executable only by its home machine; a non-current private home requires a
  valid current-generation Project machine record and does not require a Group
- selecting a remote home never starts or controls that machine's agent
- a single `--group` submission requires an existing Group; it does not create or modify the
  Group Worker Set implicitly
- duplicate `task_id` fails by default
- submission never creates a public Batch
- `--no-activate` persists the Task without requesting local agent activation from that command
  invocation; an already-running eligible agent may still claim the Task

### 12.2 `batch-submit`

```bash
qexp batch-submit --group qm9-study --file runs.yaml \
  --idempotency-key qm9-study-submit-01
```

`submit` and `batch-submit` produce the same normalized `TaskSpec`. `batch-submit` adds
only list input, manifest-default inheritance, whole-input validation, and atomic
multi-Task commit. It does not create a different Task type or lifecycle.

Rules:

- manifest is used only when several Tasks are submitted together
- the command validates the complete input before commit
- Tasks are not claimable until the internal Submission Operation commits
- `--idempotency-key` is the explicit retry contract for scripts and uncertain outcomes
- when omitted, the CLI creates a random key, durably creates the Submission Operation,
  and prints the operation ID and key before Task staging
- automatic retries inside one invocation reuse the same key
- the first use of a key resolves and persists an immutable submission context before Task
  staging
- that context includes the original submitting machine, target Group identity, every
  Task after defaults and `home_machine: current` resolution, generated Task IDs,
  effective placement constraints, and the submission's Worker Set additions
- a retry loads the existing operation before resolving machine-relative values
- the same key and canonical raw request reuse the first operation's resolved context and
  converge even when retried from another machine
- retry never reinterprets `current`, regenerates Task IDs, or recomputes Worker Set additions
  against newer Group state
- the same key with a different canonical raw request fails with an idempotency conflict
- incompatible later Group changes leave completion recoverably blocked; they do not
  silently rewrite the stored submission plan
- a new invocation without the previous key is a new submission
- manifest hashes are not implicit idempotency keys
- `doctor` exposes interrupted operation IDs and keys for recovery
- success output reports Group, created Task count, home machines, and spillover summary
- success output does not report a Batch ID

## 13. Task Lifecycle and Cleanup

Required meanings:

- `qexp submit`: create one new logical Task
- `qexp batch-submit`: create several logical Tasks through the same `TaskSpec` contract
- `qexp task retry`: queue the next Attempt under one existing Task
- `qexp task cancel`: cancel one Task's current queued or active work
- `qexp group retry-failed`: retry Tasks whose current Attempt is failed
- `qexp group cancel`: cancel current Group work under explicit running-process semantics

Retry rules:

- Task ID remains stable
- Task totals do not increase
- historical Attempts remain auditable
- retry is allowed when no active claim exists and either the Task projection and current
  Attempt are both failed, or the Task is blocked and its current Attempt is orphaned
- retry reserves the next attempt number; claim materializes the concrete Attempt record
- stale historical failures are not retried after a newer Attempt exists

For blocked orphaned work, the ordinary manual retry command is itself the explicit operator
decision:

```bash
qexp task retry <task-id>
```

Under the Task lock, retry verifies that no active claim exists, increments the fencing epoch,
records an `orphan_superseded_by_retry` audit event, preserves the orphaned Attempt as historical
truth, clears the current Attempt reference, and queues the same Task. The next claim creates a
new Attempt with a higher fencing token.

Retry supersedes the old Attempt's qexp execution authority. It does not inspect the old machine,
claim that the old process stopped, or undo external side effects. No additional duplicate-risk
acknowledgement flag is required. `qexp group retry-failed` remains limited to failed Tasks and
never selects blocked or orphaned work.

Task cancellation semantics:

- queued or pre-launch work is cancelled without starting a process
- starting or running work receives durable termination intent
- only the owning machine agent signals and reconciles its local process
- the initiating CLI reports pending acknowledgement and never signals a remote PID
- an unreachable orphan is not reported as successfully terminated

`resubmit` is not guaranteed by the new model. It may return only through a separate new
contract and must not silently become an alias for retry.

`clean` must continue to support:

- exact terminal Task cleanup by Task ID
- bounded bulk cleanup under a defined retention policy
- dry-run output

Schema-5 commands are:

```bash
qexp clean --task-id <task-id> [--dry-run]
qexp clean --older-than-days <days> --limit <count> [--dry-run]
```

Bulk cleanup defaults to a 30-day retention window and a hard limit of 100 Tasks.
Only terminal Tasks without active claims, live local processes, or active control-operation
coverage are eligible. A Task is removed with its complete Attempt directory; partial
Attempt-history pruning remains out of scope.

Cleanup is a durable cross-machine control operation. It freezes the required machine set,
waits for every required machine agent to remove Task-local reservations, process manifests,
and logs, and only then removes shared Task and Attempt truth. Required machines are the Task
home machine, all historical Attempt machines, and the machine that prepared cleanup; unrelated
registered machines do not block completion. Until all acknowledgements arrive, the operation
reports `waiting_ack` with `pending_machines`; an offline required machine therefore delays
cleanup rather than allowing shared truth to be deleted underneath machine-local resources.

While cleanup is pending, the Task carries cleanup intent and cannot be retried, claimed,
cancelled, or offered. After cleanup completes, the cleanup operation remains as a permanent
tombstone: the same Task ID cannot be submitted again. `task_cleanup_started` is emitted when
the operation is prepared, while `task_cleaned` is emitted only after shared deletion succeeds.

Attempt-history retention and Group historical-total behavior require a separate cleanup
spec before implementation changes.

## 14. Observation

Everyday observation should remain Task-first and Group-aware:

```bash
qexp task list
qexp task list --group stage-c1
qexp task show task_xxx
qexp task show task_xxx --format=json
qexp group list
qexp group show stage-c1
qexp group show stage-c1 --format=json
qexp top
qexp machines
```

`show` is the only ordinary single-resource observation verb. The target CLI does not
define a parallel `inspect` spelling.

Group output should show:

- admission and dispatch controls
- `queued_home` and `queued_shared` counts
- logical Task totals
- Attempt totals and historical failures
- blocked and orphaned work
- Worker Set state
- home and fallback machine distribution

`top` currently reports project-wide counts, Tasks, and machine views. It does not define `--all` or `--group` filters.

GPU terminology must use:

```text
visible reserved unreserved
```

`unreserved` means not reserved by qexp. It does not mean physically idle.

## 15. Product Boundaries

### 15.1 One Shared Root Per Project

Official shared control root:

```text
<project_root>/.qexp
```

Do not split truth into one `.qexp` root per experiment. Groups, Tasks, Attempts, machines,
claims, events, and internal operations need one project control plane.

The shared filesystem is qexp's coordination transport. qexp does not require a central
network scheduler service, but cross-machine features depend on verified shared-filesystem
visibility, atomic-write, and exclusive-claim semantics.

**Assumption / Unverified**:
The exact supported filesystem profiles and cross-host locking primitive must be validated
and documented by the runtime specification and ADR before cross-machine dispatch ships.
Participating hosts are also assumed to maintain bounded clock skew through
operator-managed time synchronization; elapsed offering remains a soft placement-timing
decision rather than an execution-safety boundary.

### 15.2 Explicit Machine Identity

Shared mode requires an explicit machine name:

```bash
qexp init --shared-root /path/to/project/.qexp --machine gpu2a
```

The machine name identifies a project-local logical worker; it does not identify a physical server.
Projects on one physical server may use different machine names while the one global agent retains
one shared GPU resource pool.

Operational commands derive the submitting machine from the unique local `MachineRuntime` binding
for the canonical shared root and stable Project ID. `--machine` and `QEXP_MACHINE` are
compatibility assertions only; a mismatch fails before project mutation and suggests
`--home-machine` for placement intent. Saved-context machine/runtime fields and standalone
`--runtime-root` inputs do not select operational identity or local resource ownership.

`qexp submit --home-machine <name>` selects Task placement independently. `current` and omission
resolve to the verified local machine. A remote home needs valid current-generation shared Project
machine metadata, but qexp does not remotely activate its agent or transfer project files.

### 15.3 On-Demand Agent by Default

Default behavior:

- local work submission automatically starts the current machine's agent when needed, unless the
  submit invocation uses `--no-activate`
- `on_demand` agents exit after true idleness; `daemon` agents remain active
- qexp does not remotely wake other machines

Daemon mode is opt-in:

```bash
qexp init --shared-root /path/to/project/.qexp --machine gpu2a --agent-mode daemon
qexp agent start
```

`agent start` always starts a detached process. Use `qexp agent run` for foreground debugging.
`init` initializes the project, records machine configuration, and registers its binding; it does
not start a long-lived process.

### 15.4 Local Process Ownership

Only the owning machine agent may:

- start or signal its processes
- operate its local launch backend, including tmux sessions when present
- manage its local GPU reservations
- confirm local process termination

`tmux` is the primary supported operating mode for interactive execution and observation.
When `tmux` is absent, qexp may fall back to a reduced detached-process path so the machine
can still execute work, but that fallback is a compatibility path rather than the primary
feature target. qexp does not guarantee parity of operational ergonomics or performance for
non-`tmux` deployments.

Cross-machine commands write shared intent and wait for acknowledgements. They do not
directly operate remote PIDs.

### 15.5 Machine-Agent Operation and Migration

Global-agent operations are local to the qexp Machine:

```bash
qexp agent add-project
qexp agent list-projects
qexp agent disable-project <project-id-or-root>
qexp agent remove-project <project-id-or-root>
qexp agent start
qexp agent status
```

`qexp init` initializes and registers every new Project before it returns successfully. Ordinary
`qexp agent start` never initializes or registers the current directory. `qexp agent add-project`
is an idempotent operations command for restoring a removed or lost current-generation binding and
may run while the global agent is active. An existing Project without the global-agent machine-record
marker must use the one-time
`qexp agent migrate-project` command. It stops only a verified old agent process, imports local
execution evidence, registers the Project, and then starts or wakes the global agent without
terminating already running training processes. Late immutable runner evidence is drained from
the legacy runtime instead of permanently mirrored. Repeating a completed migration preserves
the binding's current operator-controlled enabled or disabled state.

`start`, `run`, `stop`, `restart`, and `status` are global-agent commands. They operate on the
machine authority and therefore affect every registered Project. Activation-triggering commands
require their Project to be registered; `submit --no-activate` may still persist work without
starting the agent.

Machine runtime loss is not project loss. A replacement machine agent starts from explicitly
registered bindings and does not infer, supervise, or declare the terminal state of processes
from a discarded runtime. Shared lease and fencing rules leave an unreachable previously running
Attempt `orphaned` and its Task `blocked`; no automatic retry follows.

### 15.6 Scheduling Logs Only

qexp records:

- submission outcome
- offer and claim transitions
- Attempt start and finish
- cancellation and retry
- failure category and recovery state

Training logs, metrics, checkpoints, and scientific artifacts remain owned by the training
stack.

## 16. CLI Surface

The command rule is:

> Create work with `submit`; operate an existing object with resource then action.

Attempt and Submission Operation are internal diagnostic objects. The daily CLI does not
provide `qexp attempt ...` or `qexp submission-operation ...` resource trees. Their facts
are exposed through Task/Group JSON, events, and `doctor` only.

### 16.1 Submission and Project Commands

- `qexp init`
- `qexp submit`
- `qexp batch-submit`
- `qexp top`
- `qexp machines`

### 16.2 Task Commands

- `qexp task list`
- `qexp task show`
- `qexp task retry`
- `qexp task cancel`
- `qexp task offer`

### 16.3 Group Commands

- `qexp group create`
- `qexp group list`
- `qexp group show`
- `qexp group seal`
- `qexp group reopen`
- `qexp group pause`
- `qexp group resume`
- `qexp group cancel`
- `qexp group retry-failed`
- `qexp group machines add`
- `qexp group machines drain`
- `qexp group machines remove`

### 16.4 Agent Commands

- `qexp agent start`
- `qexp agent run`
- `qexp agent restart`
- `qexp agent stop`
- `qexp agent status`
- `qexp agent add-project | list-projects`
- `qexp agent disable-project | remove-project <project-id-or-root>`
- `qexp agent migrate-project`
- `qexp doctor`
- `qexp clean`

Legacy Batch inspection and retry commands are removed when the new model becomes active.
The target CLI also does not promise aliases for the old flat `list`, `inspect`, `retry`,
`cancel`, or hyphenated Group command spellings.

## 17. Acceptance Checklist

- [ ] Shared mode requires explicit machine identity.
- [ ] A machine is defined as an independently scheduled GPU resource pool, not a physical
      server entity; one physical server may expose multiple non-overlapping machines.
- [ ] One project uses one shared `.qexp` control plane.
- [ ] MachineRuntime owns only machine-local resources; every project retains authority for its
      own queue and execution truth.
- [ ] Machine-agent project bindings use stable project identity, canonical roots, and explicit
      enabled/draining/disabled lifecycle semantics.
- [ ] The default and environment-overridden machine runtime root produce one global scheduler
      lock for every registered Project.
- [ ] Machine-managed local execution records use stable project-ID composite identity and one
      unified reservation set.
- [ ] Within each primary/borrow admission layer, cross-project dispatch is deterministic
      stable-ID round-robin and a blocked project does not prevent scanning later bindings.
- [ ] Explicit project migration verifies old PID identity, keeps training processes alive, and
      leaves one global agent process responsible for the migrated Project.
- [ ] Machine runtime loss cannot assert process termination or cause automatic retry.
- [ ] Single Task submission remains YAML-free.
- [ ] New submissions create no public Batch identity.
- [ ] Unsupported old schema fails fast and is not read, migrated, or partially imported.
- [ ] Multiple machine-local submissions can populate one Group.
- [ ] Omitted home machine resolves to the verified submitting machine.
- [ ] A private Task may use a non-current home when its current-generation Project machine
      record is valid, and only that home machine may claim it.
- [ ] Omitted sharing mode remains private.
- [ ] A Task is remotely claimable only when the user permits spillover.
- [ ] Placement authorization and runtime queue scope are separate validated Task domains.
- [ ] Agents cannot turn a private Task into spillover or produce `private + shared`.
- [ ] Home machine receives first refusal without receiving a premature claim.
- [ ] First release offers Tasks only through `qexp task offer` or persisted
  `after_seconds`.
- [ ] `--group` is the sole submission source for Group identity and manifest
      `group.name` is rejected.
- [ ] Submission never implicitly adds its origin machine to a Group Worker Set.
- [ ] `group create --workers` uses an exact explicit Worker Set and defaults to `{current}` only
      when the option is omitted.
- [ ] Single Group submission requires an existing Group; batch creation requires explicit
      non-empty manifest `group.workers`.
- [ ] Existing Task and Group operations use resource-first command namespaces and `show`
  is the only ordinary single-resource observation verb.
- [ ] Attempt and Submission Operation remain diagnostic internals without daily CLI
  resource trees.
- [ ] Idle agents pull shared Tasks through a globally exclusive claim.
- [ ] Home agents may later claim their own still-shared Tasks.
- [ ] Agents do not pre-claim more work than they can promptly execute.
- [ ] Worker Set expansion and drain work during active execution.
- [ ] Worker removal cannot strand private or home-only queued Tasks.
- [ ] Pause and cancellation linearize against the final launch gate.
- [ ] Every failed claim or launch path releases provisional GPU capacity idempotently.
- [ ] Bulk submission exposes and enforces an explicit idempotency-key contract.
- [ ] Cross-machine retry with the same key reuses the first operation's resolved
  submission context instead of reinterpreting `current`.
- [ ] Elapsed-time offering works without a coordinator and remains safe under repeated
  scans and claim races.
- [ ] Running work is never migrated by clearing its claim.
- [ ] Retry keeps Task count unchanged and materializes the next Attempt when a claim wins.
- [ ] Open settled Groups may accept later control experiments.
- [ ] Sealed Groups reject additions until reopened.
- [ ] Heartbeat loss alone never authorizes silent duplicate execution.
- [ ] An expired Attempt returns to running only through recovery CAS, never ordinary
  lease renewal.
- [ ] Grouped recovery respects Worker Set drain/removal and applicable termination intent;
  pause alone does not block the same Attempt from recovering.
- [ ] Terminating Group cancellation preserves pending-machine acknowledgements across CLI
  restart.
- [ ] Remote process operations are performed only by the owning agent.
- [ ] qexp records scheduling facts, not training semantics.
- [ ] Product and runtime specs identify unimplemented target behavior explicitly.

## 18. Explicit Non-Goals

- remote SSH or remote agent wake-up
- preserving one bulk-submission invocation as a public grouping
- source snapshots or Git revision enforcement
- protection against source edits while Tasks wait
- training epoch, step, loss, metric, or progress inference
- scientific result aggregation
- artifact and checkpoint management
- physical GPU utilization scheduling outside qexp reservations
- automatic failover after ambiguous machine loss
- arbitrary workflow DAGs
- Slurm-style partitions, priorities, quotas, reservations, or preemption
- mandatory cross-server log streaming
- hostile multi-tenant authorization and isolation
- backward compatibility or migration for Batch-era `.qexp` data
