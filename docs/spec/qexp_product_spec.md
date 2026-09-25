---
doc_type: spec
status: active
updated_at: 2026-09-22
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
share`, `share --after 10m`, repeated `--with`, and `qexp task unshare`; `task offer` remains
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
[`compatibility-governance.md`](../development/compatibility-governance.md). The registry schedules removal and
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

- Agent lifecycle independence: submit and launch a Task, stop or crash only the machine agent,
  allow the real runner to finish, then start the agent and observe the original Task/Attempt
  terminal outcome without manual repair or a successor launch.

- New project activation: `qexp init --machine NAME -> qexp project init -> qexp project register -> qexp agent start`
- New project submission: `qexp init --machine NAME -> qexp project init -> qexp project register -> qexp submit -- <command>`
- Legacy project migration: `qexp admin migrate agent --project PATH --machine NAME`

`init --machine NAME` initializes or deliberately replaces only the local MachineRuntime identity
and global agent configuration. `project init [PATH]` creates shared Project truth, and `project
register PATH...` enrolls existing Projects in the current runtime. None of these commands starts
the agent. `agent start` requires at least one enabled, registered Project and waits for current
generation readiness; it never initializes or enrolls a Project.

Unless `--no-activate` is supplied, `submit` activates the local agent for an existing binding
and submitted work eventually converges through its normal scheduling lifecycle. `--no-activate`
is an explicit non-activation exception, not an additional prerequisite for the main workflow.

`admin migrate agent` handles only a project carrying legacy metadata. It verifies and stops the old
agent, imports that project's local evidence, and enables its binding. If a prior registration from
another machine runtime remains, migration may replace it only after that registration's shared
write eligibility expires; local PID or runtime-path observations cannot shorten this interval. A
failed migration must not mark the project migrated or release or overwrite resources belonging to
another project.

The retired `agent add-project`, `list-projects`, `enable-project`, `disable-project`, and
`remove-project` spellings do not forward to the new operations. During their registered
compatibility window they fail with migration guidance to the corresponding `project` command.

## Machine setup, Project enrollment, and image reuse

One local MachineRuntime owns one random runtime ID, one mutable global `agent.name`, one global
residency policy, one Project inventory, and one resource authority. Project-specific registration
names do not create additional GPU or CPU pools. Names are administrative assertions rather than
host attestations.

`qexp init --machine NAME [--agent-mode daemon|on_demand]` always means fresh machine setup. The
first invocation creates an identity without prompting. Replacing an existing identity, including
with the same name, requires interactive confirmation or `--yes`, creates a new runtime ID, keeps
configuration and inventory, and isolates old bindings and runtime evidence. It never changes
shared registrations, Tasks, claims, or reservations. Verified or ambiguous live local execution
always blocks replacement. Ordinary replacement also requires all recovery and terminal-publication
obligations to be settled.

Machine identity checks distinguish a fresh runtime from unreadable or malformed identity.
Only a missing identity with no remaining runtime data receives initialization guidance.
A missing identity alongside configuration, inventory, generations, reservations or execution
records is a recovery failure; `init`, including `--yes`, must not treat it as fresh setup.
Empty layout directories and lifecycle lock files alone do not establish prior identity.
A validated pending identity-replacement transaction retains its existing resume path.
Errors identify the affected path and direct access failures to runtime selection, mount and
permission checks, or damaged identity to preservation and recovery of the original identity.
Validation must not synthesize identity or change durable runtime data on these failures.

`qexp admin repair identity --dry-run` diagnoses the selected MachineRuntime without
requiring a Project or loading Agent configuration. Select the root with
`--machine-runtime-root` or `QEXP_MACHINE_RUNTIME_ROOT`; otherwise the default
machine root is used. The result reports the selected path and source, inspected
evidence, passed and blocked checks, and one next action. JSON output uses
`--format=json`. A verified identity is `healthy` with exit status 0. Fresh,
damaged, inconsistent, pending-replacement, or host-unverified state is `blocked`
with exit status 1; an inspection failure is `failed` with exit status 1.
`healthy` covers identity, current-generation, binding, and host-continuity checks
only. It does not certify Agent configuration or readiness. The diagnosis makes
no filesystem changes. `qexp admin repair identity` without `--dry-run` is a
usage error because automatic identity restoration is unavailable. Project
metadata maintenance remains `qexp --project PATH admin repair`.

If diagnosis reports `host_continuity_unverified`, preserve the runtime and seek
machine-level evidence of its original host; matching IDs and paths alone do not
authorize repair. `host_mismatch` requires investigation on the original host.
For `replacement_pending`, resume the recorded `qexp init --machine NAME`
transaction with its original target and policy. Neither result authorizes
copying an archived identity into the current generation or replacing identity
to make the diagnostic pass.

`--detach-old-runtime` is an explicit recovery-responsibility exception for a copied environment.
It permits unresolved evidence to be preserved in an isolated archive when local execution is
known not to be live. It does not finalize work, release resources, adopt ownership, or prove that
another recovery environment exists. The warning and structured result identify the old and new
runtime IDs, archive, and unresolved obligations. A new runtime never consumes the archive.

`qexp project init [PATH]` creates only shared Project truth and preserves a valid existing Project.
`qexp project register PATH...` and `qexp project register --from-pool` enroll existing Projects
without creating them or starting an agent. The saved inventory is distinct from effective
bindings and records stable Project ID, canonical path, enablement, and whether the registration
name follows the global default, is explicit, or has unresolved legacy provenance. Existing
bindings keep their frozen effective names. A missing mount or name conflict is an entry-specific
failure and does not roll back successful entries.

`qexp project register PATH --machine NAME --adopt-existing` explicitly takes over one registration
without active write eligibility for that logical name. It never replaces an actively eligible
registration. The flag cannot be combined with `--from-pool` or multiple Project paths. Ordinary
registration keeps the existing ownership guard.

`project list` reports inventory-only, registered, disabled, and conflicting entries. `project
remove ID_OR_PATH` removes an inventory-only entry without mounting or reading the Project. A
current binding retains its existing disable, recovery, reservation, and process-safety checks.
Removal and pool enrollment serialize so a stale selection cannot recreate a removed entry.

`qexp agent name` reads `agent.name`; `qexp agent name --set-to NAME` and `qexp config set agent
--name NAME` update the same locked global value without changing the runtime ID or existing
bindings. New default-source registrations use the current name. The default residency for a new
runtime is `daemon`; `on_demand` is explicit. Legacy per-binding modes migrate once: any enabled
daemon binding selects daemon, otherwise enabled bindings select on-demand; stored bindings are the
fallback and an empty inventory selects daemon.

The running agent may make an idle registered binding dormant after every service
lane proves quiescence against the same durable Project activation checkpoint.
Dormancy only evicts hot machine-local service context. New shared work advances
the checkpoint and is found by a fair cold poll with a budget of four dormant
bindings per scheduler cycle; the fallback revisit envelope is
`ceil(dormant_bindings / 4)` cycles plus filesystem latency. A stopped agent is
not remotely awakened. New borrow claims remain blocked whenever an enabled
binding is omitted from primary-demand inspection. See the
[binding working-set contract](qexp_working_set.md).

Activation delivery is per registration generation. Each consumer advances a
contiguous shared cursor only after all local service lanes have persisted their
handoff. The Project periodically compacts at most 256 activation events into a
durable authoritative-index reconstruction snapshot. Offline and newly
registered consumers below that floor must reconcile the snapshot coverage and
replay the retained suffix before dormancy. Removing a binding retires only its
exact consumer generation through a crash-recoverable machine-local intent.

`agent start` is detached and idempotent. Its positive `--timeout` defaults to 30 seconds. Success
requires a fresh response from the current runtime generation, acknowledgement of the requested
policy revision, and authority validation plus an initial scheduler reconciliation for every
enabled inventory entry captured by the invocation. Disabled entries are intentionally inactive;
an enabled inventory-only or conflicting entry prevents readiness. Timeout is non-destructive and
reports every selected Project's reason. `agent run` remains foreground; stop and restart preserve
runner and recovery independence.

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
- a drained schema-5 root may be upgraded by `qexp admin migrate schema --project PATH --to-schema 6`; it must have
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

qexp launches each runner directly from the local agent with an explicit working directory and
the agent's current environment. A project may optionally create a non-authoritative `tmux`
Attempt-log observer after the runner accepts the durable handoff; this observation policy is
disabled by default. Execution does not depend on `tmux` or interactive-shell readiness. The
runner, shared log, `task show --watch`, and `task logs --follow` retain the same lifecycle
semantics when windows are disabled or unavailable.

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

New machines use daemon mode by default. Operators who do not want a persistent service select
`on_demand` explicitly.

After a long period without qexp submissions, agents are likely to have exited. qexp does
not promise remote wake-up. The user therefore needs to SSH into each intended machine and
start or wake its agent.

Once the user already has 10 SSH sessions open, submitting one machine-specific manifest
in each session is natural and efficient:

```text
g1: qexp submit --file runs-g1.yaml --group stage-c1
g2: qexp submit --file runs-g2.yaml --group stage-c1
...
g10: qexp submit --file runs-g10.yaml --group stage-c1
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
qexp task unshare <task-id>
```

`share` expands the candidate set; it does not transfer ownership away from the home machine.
`unshare` clears sharing policy and deadlines while preserving the Task home machine.

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
Both command and file modes of `submit` use it so single and bulk creation share one atomic
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

### 6.7 Machine GPU Admission Policy

One MachineRuntime owns one persistent GPU admission policy shared by every registered Project.
Operators inspect and change it without project context:

```bash
qexp agent config gpus show
qexp agent config gpus set --visible 0,2,3
qexp agent config gpus set --none
qexp agent config gpus reset
```

`set --visible` accepts a strict comma-separated list of unique, nonnegative integer device IDs.
`set --none` deliberately permits no GPU work. `reset` removes the explicit override and exposes
the agent's inherited `QEXP_VISIBLE_GPUS` value when nonempty, otherwise automatic discovery.
`--expected-revision` is available on `set` and `reset` for compare-and-swap automation.

The operator-facing sets are:

- **configured GPUs**: IDs in an explicit persisted or inherited allowlist;
- **discovered GPUs**: IDs returned by the local hardware inventory;
- **visible GPUs**: configured intersect discovered for an allowlist, or discovered in automatic
  mode;
- **reserved GPUs**: IDs held by active or unexpired provisional qexp reservations; and
- **draining GPUs**: reserved IDs that are no longer visible.

Removing a GPU stops later admission but does not signal, migrate, restart, or rewrite an already
authorized Attempt. Its reservation remains authoritative until ordinary terminal reconciliation
or verified cleanup releases it. Adding a discovered GPU makes it eligible on the next scheduling
cycle. `unreserved` continues to mean visible and not reserved by qexp; it does not prove physical
idleness or account for processes outside qexp.

The policy is machine-local static admission configuration, not Project truth, a physical GPU
utilization detector, or a Group Worker Set control. Every Project bound to the MachineRuntime
observes the same policy.

## 7. No Public Batch Entity

`submit --file` is the manifest input mode. It means:

> Reliably validate and submit multiple independent Tasks.

It does not create a public Batch ID.

Rules:

- new Tasks have no public `batch_id`
- Tasks submitted by one invocation do not form a permanent user-visible subgroup
- one invocation must still be atomic from the scheduler's visibility perspective
- interrupted bulk submission must be recoverable and idempotent
- file input must not make command input more complex

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
qexp group cancel <group> --all
qexp group retry <group>
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

- `--group` overrides `group.name`; omission of both means an ungrouped private submission
- a manifest `group` configuration block may identify the Group and configure its initial or
  additional Worker Set
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
- `qexp task unshare <task-id>` resets policy to private, queue scope to home, and clears
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
work. All writes and outputs use `gpu_limit_gpus`. A borrow Worker is persisted as
`scheduling_role: borrow, state: active`; `draining` and `removing` remain non-claimable.

An enabled local project binding is required on every target machine. `qexp project register`
creates or restores the binding. Submitting on one machine
does not wake another machine's on-demand agent. A running target agent observes later Group
truth changes without re-registration.

Worker Set invariants:

- every grouped Task's home machine must be a claimable, non-draining Group worker when the
  Task is committed
- submission never adds the submitting machine to an existing Group's Worker Set implicitly
- `group create` defaults to `{current}` when `--workers` is omitted; an explicit `--workers`
  list is the exact initial Worker Set
- either input mode may atomically create a missing named Group; without a worker declaration its
  initial Worker Set is the verified submitting machine, while an explicit declaration is exact
- an explicitly configured non-current home machine must already be a claimable worker
- a manifest may add workers explicitly but must not silently remove or replace the existing
  Worker Set
- removal and drain occur only through explicit Group machine operations
- a draining machine cannot claim either home or shared Tasks from that Group

Required operations:

```bash
qexp group worker add <group> g11 --role borrow --gpu-limit-gpus 2
qexp group worker set <group> g11 --role primary --gpu-limit-gpus 2
qexp group worker drain <group> g5
qexp group worker remove <group> g5 --all
qexp group worker list <group>
```

After a rolling upgrade activates Group isolation, an unfinished removal created
by an older release can report `blocked` with `legacy_worker_incarnation_unknown`.
It cannot safely identify a worker that may have been reactivated since the
request. Its Tasks, processes and resources remain untouched. If removal is still
wanted, run `qexp group worker remove <group> <machine>` again, including
`--all` only when termination is intended. New removal operations
recover automatically after interruption and cannot follow a reactivated worker.
The upgrade itself is automatic and preserves running training; it requires no
per-project activation command.


Adding a machine:

- makes it eligible for compatible queued shared Tasks
- does not modify running or terminal Attempts
- does not remotely start the agent

Role and GPU-limit changes are Group-lock-linearized. They affect later claims but do not
revoke a successfully created claim. Drain, remove, pause, cancellation, lease, and fencing keep
their existing launch-gate meaning.

Live ready membership is a derived, Group-partitioned projection. It records only current ready
Task generations and is rebuilt from Task, Group, Submission, ready-marker, and candidate truth.
Worker role changes therefore update candidates only for that Group's live references, rather than
scanning project Task history. A missing, building, damaged, or revision-changing projection makes
the primary-demand probe unresolved and blocks new borrow admission; it never means that primary
demand is absent.

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
  by its fallback constraint and its static GPU quota can fit the Task; temporary
  occupancy and Group pause do not count as structural stranding
- queued availability changes remain allowed while the home worker is draining,
  so sharing or offering work can resolve the blocker
- forced removal does not bypass queued-work safety checks

Removal can remain `converging` while background membership confirmation catches
up. The CLI starts local background advancement after saving the request. It then
reports blocking Task IDs or completes when current membership and changes have
been checked; `--all` does not bypass this confirmation.

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
qexp submit --file runs-g1.yaml --group stage-c1

# g2
qexp submit --file runs-g2.yaml --group stage-c1
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
qexp submit --file additional-controls.yaml --group stage-c1
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
qexp group worker add stage-c1 g11
```

After g11's agent starts, it may claim compatible shared Tasks.

The user drains g5:

```bash
qexp group worker drain stage-c1 g5
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
termination reconciliation through `qexp admin check --project PATH`.

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
qexp submit --file runs.yaml
qexp submit --file runs.yaml --group qm9-study
```

Rules:

- exactly one input mode is required: a nonempty command after `--` or one `--file` manifest
- single-Task command submission must not require YAML
- mode-specific options are rejected before activation or publication
- Group is optional for ad hoc Tasks
- `--machine` is the verified local identity assertion; `--home-machine` selects Task placement
  and defaults to the verified current machine
- a private Task is executable only by its home machine; a non-current private home requires a
  valid current-generation Project machine record and does not require a Group
- selecting a remote home never starts or controls that machine's agent
- either mode may create a missing named Group inside its Submission Operation
- duplicate `task_id` fails by default
- submission never creates a public Batch
- `--no-activate` persists the Task without requesting local agent activation from that command
  invocation; an already-running eligible agent may still claim the Task
- `--dry-run` performs resolution and read-only validation without registration, activation,
  Task, Group, operation, idempotency, or clock-evidence writes. Internal temporary
  Task IDs may be used for plan validation; omitted input IDs remain null in preview output.

The submission-owned option matrix is:

| Option/input | Command mode | File mode |
| --- | --- | --- |
| `-- COMMAND...` | Required, nonempty literal argv | Rejected |
| `-f, --file MANIFEST` | Rejected | Required exactly once |
| `--project PATH`, `--group NAME`, `--idempotency-key KEY` | Supported | Supported |
| `--task-id ID`, `--name NAME`, `--depends-on TASK_ID` | Supported | Rejected; use Task fields |
| `--gpus N`, `--cpus N`, `--home-machine NAME`, `--cwd PATH` | Task values | Override every Task |
| `--sharing`, `--offer-after-seconds` | Supported | Rejected; use manifest placement |
| `--tmux`, `--no-tmux`, `--no-activate`, `--dry-run` | Supported | Supported |
| `--format human|json`, `--quiet` | Supported | Supported |

`--quiet --format json` and `--quiet --dry-run` are invalid. Unknown options, unknown manifest
fields, duplicate YAML keys, empty Task lists, and file-mode rewrites of commands, Task IDs, names,
or dependencies are rejected. An omitted idempotency key creates a fresh random key; dry-run may
compare a supplied key but never reserves it.

### 12.2 File input, overrides, and idempotency

`submit --file` adds list input, manifest-default inheritance, whole-input validation, and atomic
multi-Task commit. It does not create a different Task type or lifecycle. `group.name` supplies a
manifest Group selector and explicit `--group` takes precedence. Supported file-wide overrides are
`--gpus`, `--cpus`, `--home-machine`, `--cwd`, and `--tmux|--no-tmux`; each applies to every Task.
The field precedence is explicit CLI, Task, manifest defaults, applicable Project policy, then the
built-in default. An omitted CLI value never overwrites a manifest value.

Rules:

- a one-Task manifest has the same transactional guarantees as command input
- manifest `tasks[].tmux` and `defaults.tmux` accept only booleans or null; an explicit invocation
  `--tmux`/`--no-tmux` wins every Task, then a Task boolean, then the manifest default; null or
  omission continues to the next level
- the normalized choice is stored as a durable per-Task override; it does not update Group
  defaults, existing Tasks, or later independent submissions to the same Group
- the command validates the complete input before commit
- dependency validation overlays the complete submitted Task set on current truth and follows the
  dependency graph reachable from those candidates by exact Task ID; candidate references retain
  same-Group, committed-submission, cleanup, self-dependency, and cycle checks
- unrelated retained Task history is outside ordinary submission validation and cannot make a
  fixed submission perform a Task-directory scan or fail because unrelated history is malformed;
  explicit integrity audit and repair own corruption outside the candidate-reachable graph
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
- if pre-commit publication of one grouped ready Task fails, the first failure receipt identifies
  the affected Task and generation, its zero-based resolved input position when available, and the
  stable publication stage, check, reason, and exception class; arbitrary exception text, Task
  commands, paths, and environment values are never included
- the same bounded diagnostic is retained by the aborted Submission Operation and projection
  degraded reason when those independent writes succeed; failure of either write does not erase the
  diagnostic returned by the submitting process or imply that the corresponding state is durable
- manifest hashes are not implicit idempotency keys
- `doctor` exposes interrupted operation IDs and keys for recovery
- success output reports Group, created Task count, home machines, and spillover summary
- success output does not report a Batch ID

Working-directory resolution is stable: command input without `--cwd` uses invocation cwd; file
input without a Task/default value uses the selected Project directory; a relative Task/default
value uses the manifest directory; and a relative CLI `--cwd` uses invocation cwd. Absolute values
are preserved after normalization. These paths and the Project resolution source are frozen in the
first operation context and reused on same-key retry.

Human output is the default. `--format json` returns the submission schema version 1 with required
`mode`, `outcome`, `project`, `group`, `operation`, `idempotency_key`, `task_ids`, `preview`, and
`error` fields. Outcomes are `committed`, `rejected`, `pending`, `unknown`, and `preview`.
`--quiet` prints committed Task IDs only and is incompatible with JSON and dry-run. Failed,
pending, unknown, and preview results never present provisional IDs as committed output.

`project` is null before resolution or contains canonical Project `path` and its resolution
`source`. `group` always contains nullable `name`, `source`, and committed-only `disposition`.
`operation` is null before durable identification or contains `id` and nullable verified `state`.
`task_ids` contains committed IDs in input order only for `committed`; `preview` is non-null only
for successful preview; and `error` is null on success or contains stable `code` and `message`, plus
an optional versioned `diagnostic` for a group-ready-member publication failure.
Preview contains normalized `tasks` with input indexes and per-field sources, `group_action`,
`worker_additions`, and `evidence_gaps`.

Input errors exit 2. Conflicts, operational rejection, `pending`, and `unknown` exit 1. Committed
and preview results exit 0. Stable error codes include `invalid_input`, `context_error`,
`idempotency_conflict`, `submission_aborted`, `submission_pending`, `submission_blocked`,
`commit_unknown`, and `interrupted`. Caught interruption exits 130. A post-commit activation or
finalization error retains `committed`, the verified IDs, and exits 1 with an operational error.

Manifest omission is distinct from explicit null. Null optional mappings (`group`, `defaults`,
`placement`, `placement.sharing`, and `placement.sharing.offer`) behave as omitted mappings.
`group.name`, `group.workers`, `task_id`, and `name` use their documented unspecified behavior;
`tmux` inherits; `requested_cpus` clears an inherited CPU request; and
`placement.sharing.offer.after_seconds` clears an inherited delay. Null command, GPU count,
working directory, home, sharing mode, fallback list, dependency list, or `tasks` is invalid.
An explicitly empty worker declaration is not omission and must still satisfy placement.

## 13. Task Lifecycle and Cleanup

Required meanings:

- `qexp submit`: create one new logical Task
- `qexp submit --file`: create several logical Tasks through the same `TaskSpec` contract
- `qexp task retry`: queue the next Attempt under one existing Task
- `qexp task cancel`: cancel one Task's current queued or active work
- `qexp group retry`: retry Tasks whose current Attempt is failed
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
acknowledgement flag is required. `qexp group retry` remains limited to failed Tasks and
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
qexp admin clean --project PATH --task-id <task-id> [--dry-run]
qexp admin clean --project PATH --older-than-days <days> --limit <count> [--dry-run]
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
qexp task show task_xxx --watch
qexp task logs task_xxx
qexp task logs task_xxx --follow
qexp group list
qexp group show stage-c1
qexp group show stage-c1 --format=json
qexp status
qexp machine list
```

`status` is a bounded Project overview. It reports the resolved Project path, stable identity and
selection source; exact local participation; machine-wide local-agent evidence; Task observation
index state; configured machine-global agent mode; separately observed mode when it differs; and
explicit next actions. Missing or malformed machine configuration yields a null configured mode
and a bounded warning, including while the agent is stopped. It does not enumerate Task, Attempt, Group, machine, or
operation history. After Project resolution it attempts at most 32 record reads, rejects records
larger than 256 KiB, and reads at most 2 MiB in total. Missing optional evidence produces a useful
partial result with null totals and reasons, never invented zero totals or an inferred stopped
agent. Invalid required Project identity remains an error.

When Project selection is implicit, human `status` renders the source in its existing Project row
and omits a duplicate Selection source row. JSON status retains the structured selection source and
receives the ordinary Project notice on stderr. If status fails after selection but before rendering,
the selected Project line is emitted on stderr before the error.

Initiating Group cancellation, worker removal, and Task cleanup results include an opaque versioned
`operation_reference` plus the internal operation ID. The reference binds Project identity,
operation kind, exact storage key, and operation ID without granting authority. `admin operation
show` requires an explicitly selected matching Project, accepts at most 4 KiB, validates identifiers
before I/O, and reads only the known active/archive paths with one archival-race retry. Invalid
references exit 2; missing, mismatched, or unreadable records exit 1; a valid blocked operation is
a successful read. Missing history is `not_found`, never inferred expiry. Single-Task cancellation
continues to use `task show` and does not fabricate an operation record.

`show` remains the structured single-resource snapshot verb. The target CLI does not
define a parallel `inspect` spelling. `task logs` is the specialized application-log
byte-stream command; it is not another structured resource view.

`task show` includes optional Attempt-scoped application progress. Progress is advisory:
it cannot renew a lease, change a claim, determine Task health, or publish completion.
Human output distinguishes pending execution, no accepted report, an available snapshot,
and an evidenced unavailable observation. Available timestamps show both absolute UTC time
and relative age. JSON preserves the legacy `available|unavailable` status while adding the
richer observation state and bounded reason. Retry never displays an older Attempt's report.
The same response reports the stored tmux observer override as `enabled`, `disabled`, or
`inherit`. This is a prospective choice for later observer decisions, not evidence that a window
exists; Attempt references remain the source of live window evidence.

Finite structured commands collect one canonical result before selecting JSON or human
presentation. JSON serializes that result without presentation wrappers. Human rendering may
derive labels and summaries only from the same result; it does not perform additional Task-history
queries. Optional empty rows are omitted unless absence is itself decision-relevant. Output uses
distinct operator vocabulary for `No results`, `No matches in this page`, `Unavailable`, `Partial`,
`Accepted`/`Waiting`, `Blocked`, `No change`/`Already`, and `Completed`.

`task show --watch` refreshes a compact human Task view on terminal stdout. It follows the
stable Task ID, shows authoritative Task phase and reason, and includes only a consistently
selected current or terminal Attempt and its advisory progress. Its persistent header shows the
canonical Project directory; an implicit direct watch includes its actual selection source, while
an explicitly bound viewer omits a source suffix. It rejects `--format` because no
structured event-stream contract exists. `task logs` and `task logs --follow` reject `--format`;
they emit application log bytes to stdout as they become readable and send qexp boundaries and
diagnostics to stderr; redirected stdout is supported. Both commands poll shared storage and
therefore cannot force application buffers to flush or promise a visibility deadline.

Continuous observation is read-only. Closing or interrupting a viewer does not signal a runner,
cancel work, start an agent, or change shared authority. By default a viewer observes the current
Task lifecycle, renders or drains its validated terminal result, and exits. `--follow-retries`
is the explicit long-lived mode that waits across a terminal result and follows a later retry.
Attempt changes and replaced or truncated log files are announced as stream boundaries; bytes
lost before a viewer can read them are not recoverable. `task logs --follow --tail N` applies the
tail limit to every new Attempt or file generation, while finite `task logs` remains the complete
read available after publication has settled.

Tmux observer reuse is identified by authoritative stable Project ID, Task ID, and Attempt ID.
An untagged legacy window or a window from another Project is ineligible for reuse and remains
untouched. Newly created live-progress and log-only windows retain their canonical Project directory
in a viewer-local persistent title; qexp does not change session-global tmux status settings.

A selected nonterminal Attempt may wait for its log to appear. Permission failures, malformed
references, and non-file targets stop the viewer with an observation error. Other temporary
filesystem errors are coalesced while the Task is nonterminal and produce one recovery notice
after reading resumes. Once Task truth is terminal, a missing or temporarily unreadable log gets
one scheduled retry and then fails nonzero instead of waiting indefinitely. These diagnostics and
Attempt/file boundaries use stderr; stdout remains only application log bytes.

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

### Training-safe live observation

The extended live view is opt-in observation for new Tasks. It never changes the
submitted command's `output_mode`, stdout/stderr descriptors, training control,
or cancellation authority. The application may report progress without an
observer, and exiting a viewer leaves training running. Existing progress-v1 and
Attempt logs remain available. A Task without structured reports keeps its log
view; attachment does not instrument an already running process.

`qexp submit --live-progress` and `--no-live-progress` are mutually exclusive
for command and file input. Omission inherits. A manifest accepts exact boolean
or null at `defaults.live_progress` and `tasks[].live_progress`; omission and
null inherit. Strings, numbers, and duplicate YAML keys are errors. Resolution
order is invocation, Task, manifest default, Group default, then false. The
choice is frozen with the first preparing Submission Operation and survives
replay, queuing, and retries. Explicit submission choices never mutate a Group.
Changing a Group default affects only new submissions. A missing Group created
by submission starts with false; unavailable optional policy freezes false for
inheriting Tasks with a bounded diagnostic, while explicit Task choices survive.

For an existing published Group, `qexp group config show GROUP progress` reports
the effective boolean, policy revision, source (`default` or `configured`), and
`applies_to: new_submissions` in finite human or JSON output. `qexp group config
set GROUP progress --live-progress|--no-live-progress` requires exactly one of
the two flags and returns the changed Group identity and revision. Missing,
malformed, oversized, unsupported, or unreadable configuration is an explicit
command error except that a missing policy record means the built-in false
default. Policy setters do not alter Group truth or its revision.

`qexp task show TASK` retains all established Task information and prints a
finite compact progress portion. `--details` prints all accepted current metrics
and completeness information once; `--watch` refreshes a compact human view;
`--watch --details` refreshes the detailed view. Details change presentation
only, never collection. `--watch --format` and redirected watch output remain
invalid. Finite JSON stays a single ANSI-free structured result: its existing
`progress` field keeps the v1 shape, with separate `progress_extended` and
`selected_progress_version` (`1`, `2`, or null) fields. A missing extension is
unavailable, not a fabricated empty metric set.

The viewer selects the latest whole, authorized Attempt observation by accepted
report time, preferring v2 on a tie. It never joins old metrics to newer v1
progress. The compact view shows stage, current/total/unit, message, and report
age; details show every accepted metric and omission reason. Unknown totals do
not create a percentage. An old report retains its original absolute time and
age; the 2-second viewer refresh does not imply a new report. No countdown,
interpolation, inferred stall, or forced 100% terminal progress is shown.
Terminal Task truth and final advisory progress are displayed separately.

Rich is only a renderer. With no Rich, ANSI capable terminals use an in-place
text table; terminals without supported cursor control append a timestamped
block only on new data or a Task/observation-state transition. Finite redirected
human output and JSON contain no cursor controls. Frames are bounded to the
viewport, sanitize producer text, disclose omitted rows, handle resize, and
restore terminal styling on exit. A renderer failure ends or degrades only the
viewer. `qexp task attach TASK` creates or joins a tmux viewer read-only without
detaching other clients. Multiple authorized SSH clients share one pane and do
not create extra collectors or accelerate publication. Different Unix accounts
need separately configured tmux authorization.

Acceptance requires command and manifest precedence, idempotent frozen choices,
Group set/read races, unavailable policy fallback, finite/detail/watch and JSON
parity, no-Rich and no-cursor rendering, truthful stale/terminal states, two
concurrent tmux clients, viewer kill/stop/recreation, and unchanged training
identity, outcome, and FDs under observer faults. A selected observer may lose
snapshots but must never hold training waiting for its reader or renderer.

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
qexp init --machine gpu2a
qexp project register /path/to/project
```

The machine name identifies a project-local logical worker; it does not identify a physical server.
Projects on one physical server may use different machine names while the one global agent retains
one shared GPU resource pool.

Operational commands derive the submitting machine from the unique local `MachineRuntime` binding
for the canonical shared root and stable Project ID. `--machine` and `QEXP_MACHINE` are
compatibility assertions only; a mismatch fails before project mutation and suggests
`--home-machine` for placement intent. Saved-context machine/runtime fields and standalone
`--runtime-root` inputs do not select operational identity or local resource ownership.

Ordinary Project commands resolve `--project PATH`, `QEXP_SHARED_ROOT`, the nearest initialized
Project in the current-directory ancestry, then saved context. A directory and its `.qexp` control
directory normalize to the same Project. An explicit, environmental, or discovered malformed
target fails instead of falling through; read-only discovery creates nothing. Common options may
occur before or after the command path, before a submission payload separator. Equal duplicates
normalize to one value and conflicting duplicates fail. Tokens after `submit --` remain literal.

After a valid implicit selection, a command operating on that one Project presents
`Project: <canonical-project-directory>` once before binding-dependent work. Environment, parent
directory, manifest directory, and saved-context selection append `(from $QEXP_SHARED_ROOT)`,
`(from parent directory)`, `(from manifest directory)`, or the home-abbreviated context-file path.
Discovery in invocation cwd itself has no suffix. The default channel is stderr so finite JSON,
quiet IDs, and application-log stdout retain their existing byte contracts. Human status and Task
watch use their integrated Project fields instead. Explicit selection adds no default notice;
machine/global, multi-Project, setup, inventory, saved-context management, help, syntax-error, and
pre-selection failure paths do not perform discovery merely to print one.

Displayed Project, source-file, and locator paths are one-line encoded. Backslash, LF, CR, and TAB
use `\\`, `\n`, `\r`, and `\t`; other C0/C1 controls use lowercase `\xhh`; Unicode line and
paragraph separators and format controls use lowercase four- or eight-digit Unicode escapes.
Other Unicode and spaces remain unchanged, with no enclosing quotes or re-escaping of generated
backslashes.

`qexp use --project <project/.qexp>` is a local default-project selector. It writes only a
canonical `shared_root`; it does not validate or register the Project, create a machine record, or
select machine identity. `qexp project register` is the normal enrollment path. `qexp use` rejects
machine/runtime inputs. `project register --machine NAME` accepts one explicit Project; the legacy
`admin migrate agent` workflow still requires explicit machine identity and may receive an
explicit custom legacy runtime.

`qexp submit --home-machine <name>` selects Task placement independently. `current` and omission
resolve to the verified local machine. A remote home needs valid current-generation shared Project
machine metadata, but qexp does not remotely activate its agent or transfer project files.

Submission accepts `--project` as either an initialized Project directory or its `.qexp` control
directory, never as an opaque ID. Command input resolves explicit `--project`,
`QEXP_SHARED_ROOT`, nearest initialized invocation-cwd ancestor, then saved context. File input
first resolves the manifest relative to invocation cwd and then resolves explicit Project,
environment, manifest ancestry, cwd ancestry, and saved context. A selected or discovered malformed
root fails without falling through. The result reports the canonical Project directory and source.
Discovery is read-only and never initializes or registers a Project.

### 15.3 Global Agent Residency

Default behavior:

- local work submission automatically starts the current machine's agent when needed, unless the
  submit invocation uses `--no-activate`
- `on_demand` agents exit after true idleness; `daemon` agents remain active
- one machine-global policy selects daemon or on-demand behavior for every binding; enabling or
  disabling a Project does not change it
- in on-demand mode, unresolved demand, maintenance errors, or local execution evidence prevent
  idle exit
- retained Group history and optional `group-service-v1` maintenance locators do not by themselves
  prevent idle exit; a later normal start resumes their durable work
- qexp does not remotely wake other machines

Daemon mode is the default; on-demand is explicit:

```bash
qexp init --machine gpu2a --agent-mode daemon
qexp project register /path/to/project
qexp agent start
```

`agent start` ensures one detached process and waits for captured enabled Projects to be ready.
Use `qexp agent run` for foreground debugging. It has no `--format` option or finite startup
record, and interrupting it does not signal already launched runners. Neither machine initialization nor Project
registration starts a long-lived process.

### 15.4 Local Process Ownership

Only the owning machine agent may:

- start or signal its processes
- operate its local launch backend, including tmux sessions when present
- manage its local GPU reservations
- confirm local process termination

Detached runner execution is the execution path. Project-controlled `tmux` windows are optional,
read-only log observers and are disabled by default. Disabling or closing one does not change the
runner, process group, reservation, or Task lifecycle. The continuous Task and log commands are
the ordinary observation path when no window is requested.

Cross-machine commands write shared intent and wait for acknowledgements. They do not
directly operate remote PIDs.

### 15.5 Machine-Agent Operation and Migration

Global-agent operations are local to the qexp Machine:

```bash
qexp project register /path/to/project
qexp project register --from-pool
qexp project list
qexp project enable <project-id-or-root>
qexp project disable <project-id-or-root>
qexp project remove <project-id-or-root>
qexp agent start
qexp agent status
qexp agent config gpus show
qexp agent config gpus set --visible 0,2,3
qexp agent config gpus set --none
qexp agent config gpus reset
```

`qexp project register` is idempotent for a valid current binding, preserves its enablement and
effective name, and reports the actual Project ID and shared path. Ordinary registration never
replaces another runtime's owner through recovery adoption. Runtime identity is bound to both its
local random identity and the current Linux host, so copying only the runtime directory cannot renew
authority on another host. `qexp project enable <project-id-or-root>` revalidates registration
authority before enabling new admission. An existing Project without the global-agent machine-record
marker must use the one-time `qexp admin migrate agent --project PATH --machine NAME` command. It
stops only a verified old agent process, imports local execution evidence, registers the Project,
and then starts or wakes the global agent without
terminating already running training processes. Late immutable runner evidence is drained from
the legacy runtime instead of permanently mirrored. Repeating a completed migration preserves
the binding's current operator-controlled enabled or disabled state.

`start`, `run`, `stop`, `restart`, and `status` are global-agent commands. They operate on the
machine authority and therefore affect every registered Project. Activation-triggering commands
require their Project to be registered; `submit --no-activate` may still persist work without
starting the agent.

`restart` waits for the old process to stop and its replacement to start, but does not add a
readiness wait. It reports available evidence and otherwise says `Ready: pending`; a later
`agent status` observes convergence. Lifecycle status reports configured `daemon|on_demand`
policy separately from a differing observed process mode.

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
are exposed through Task/Group JSON, events, and `admin check` only. Control operations for
Group cancellation, worker removal, and cleanup have a bounded read-only diagnostic view through
`admin operation show`; this does not make them general execution objects.

### 16.1 Submission and Project Commands

- `qexp init --machine NAME`
- `qexp project init [PATH]`
- `qexp project register PATH... | --from-pool`
- `qexp project list`
- `qexp project enable | disable | remove <project-id-or-root>`
- `qexp use --project PATH | --show | --clear`
- `qexp submit [--file MANIFEST | -- COMMAND...]`
- `qexp status`
- `qexp machine list | show NAME`
- `qexp config show [SECTION]`
- `qexp config set SECTION [--provider NAME] ...`
- `qexp config reset SECTION [--provider NAME]`

Submission results keep their version-1 envelope and existing `error.code`. A failure with portable
publication evidence adds optional `error.diagnostic`, whose fields match the persisted Submission
diagnostic. JSON mode emits that object within its one stdout value. Human mode writes one concise
`Diagnostic:` line after the existing error message; `--quiet` keeps stdout reserved for raw values
and writes the diagnostic only to stderr. Default output never includes a traceback or unrestricted
exception message, and submission diagnostics are not copied to application log streams.

The progress policy defaults to 30 seconds and accepts finite values greater than or equal
to 1. It applies to subsequent launches and retries, while an already-running Attempt keeps
the policy frozen at launch. A larger interval reduces progress-related local and shared
filesystem writes at the cost of freshness; it is not a visibility deadline or a quota on
arbitrary application I/O.

The project tmux policy defaults to disabled and applies only to future observer decisions.
`submit --tmux|--no-tmux` supplies a Task override. For file mode, the invocation switch wins
every Task, then a per-Task manifest value, then `defaults.tmux`, then project policy,
then the built-in disabled value. Explicit overrides survive retries; inherited Tasks sample the
then-current project policy for each new observer decision. Policy changes never remove existing
windows or create one for already-running work. Participating agents must be upgraded before the
policy is enforced across every owning machine; a config write alone is not fleet rollout proof.
An enabled decision still treats missing `tmux`/`libtmux` or window-creation failure as an
observation diagnostic and never as a training failure.

The launch-handoff policy defaults to 10 seconds and accepts finite values from 1 through
300 seconds. It applies to subsequent launch and retry Attempts and is frozen when the local
runner is initiated. The budget covers durable publication of the runner's launch intent; it
does not cover CUDA initialization, application startup, or first progress. A timeout initiates
locked compensation but does not prove that execution is absent. Existing local launch evidence
retains the reservation for recovery. Machine-agent scheduling remains available to other projects
while handoffs are pending; pending Attempts keep their reservations until confirmation or fenced
compensation.

Project configuration uses fixed typed sections: `lease`, `notifications`, `progress`, `tmux`,
and `launch-handoff`. Unqualified `config show` reports every Project section independently;
one malformed section makes the aggregate incomplete without substituting a plausible default.
The explicit global `agent` section never resolves a Project. Notifications have one enabled
switch and one complete Feishu destination per scope. `qexp notifications setup` enables a global
default for the selected MachineRuntime without a Project; the convenience family defaults to
global, while `config show/set/reset notifications` retains its Project default and permits
`--scope global`. A Project's sparse explicit fields override global values, but destination and
signing remain one indivisible credential bundle. An explicit disable or invalid Project override
never sends to a different global robot.

`config reset SECTION` removes the explicit override and restores inheritance or the built-in
default after validating the existing state. It is not a write of today's default value. Resetting
notifications is a revisioned tombstone and does not immediately delete separately stored
credentials; eligible private credentials are cleaned up later. `config reset agent` is invalid
because agent name and identity have no implicit reset.
Configuration writes retain verified binding, locking, active-claim, and policy-specific guards;
they never rewrite policies frozen into existing Attempts.

### 16.2 Task Commands

- `qexp task list`
- `qexp task show`
- `qexp task show --watch [--interval-seconds <seconds>] [--follow-retries]`
- `qexp task logs`
- `qexp task logs [-n|--tail <lines>]`
- `qexp task logs -f|--follow [-n|--tail <lines>] [--interval-seconds <seconds>] [--follow-retries]`
- `qexp task wait <task-id> [--timeout <duration>]`
- `qexp task retry`
- `qexp task cancel`
- `qexp task share`
- `qexp task unshare`
- `qexp task offer`

`task show --watch` is a terminal-only continuous view and rejects `--format`. `task logs` and
`task logs --follow` are raw application-byte streams: stdout is reserved for application bytes,
while qexp diagnostics and stream boundaries use stderr. They do not provide JSON wrapping.
Implicit Project selection is reported once on stderr before log reading or following begins;
Attempt changes, retry following, and file replacement do not repeat it. `task attach` reports the
caller's implicit source before tmux entry, while the shared viewer retains Project identity without
inheriting caller-specific provenance.

`qexp task list --format=json` returns stable Task summary records. In addition to identity,
placement, phase, claim, and GPU fields, each record includes `depends_on_task_ids`,
`dependency_state`, and `dependency_reasons`. The IDs are sorted; a Task with no prerequisites
has `[]`, `ready`, and `[]` respectively. `dependency_state` is `ready`, `waiting`, `blocked`,
or `invalid`; each reason is an object containing the prerequisite `task_id` and its reason.

`task list --name NAME` uses case-sensitive exact matching inside one bounded indexed candidate
page. Names do not become mutation identifiers. An empty result with a continuation cursor means
that the inspected candidate page contained no matches; only `stop_reason: exhausted` proves the
traversal has ended. Name-aware cursors bind the exact name and cannot be reused with another
filter.

`task wait` pins the current or next Attempt lifecycle when it starts and never follows a later
retry. It performs direct bounded Task, selected-Attempt, and dependency reads. Its exit codes are
0 succeeded, 1 failed/cancelled, 2 invalid input, 3 blocked, 4 timeout, 5 superseded,
6 observation failure, and 130 caught interruption. Timeout and interruption do not mutate work.

### 16.3 Group Commands

- `qexp group create`
- `qexp group list`
- `qexp group show`
- `qexp group seal`
- `qexp group reopen`
- `qexp group pause`
- `qexp group resume`
- `qexp group cancel`
- `qexp group retry`
- `qexp group worker list | add | set | drain | resume | remove`

### 16.4 Agent Commands

- `qexp agent start`
- `qexp agent run`
- `qexp agent restart`
- `qexp agent stop`
- `qexp agent status`
- `qexp agent name [--set-to NAME]`
- `qexp agent config cpu show | set`
- `qexp agent config gpus show`
- `qexp agent config gpus set --visible <ids> | --none [--expected-revision <revision>]`
- `qexp agent config gpus reset [--expected-revision <revision>]`
- `qexp admin migrate agent --project PATH --machine NAME`
- `qexp admin {check|repair} --project PATH`
- `qexp admin clean --project PATH`
- `qexp admin operation show REFERENCE --project PATH`
- `qexp admin upgrade status | advance [--project PATH]`
- `qexp admin upgrade pause | plan | apply | validate | resume --project PATH`
- `qexp admin migrate schema --project PATH --to-schema 6`
- `qexp admin migrate schema6 {check|start|status|attest|resume} --project PATH`

`agent run` is a foreground debugging stream with no `--format` or finite startup record.
`agent restart` reports process replacement and current readiness evidence without waiting for
Project convergence; use `agent status` for the subsequent configured-mode, observed-mode, and
readiness view.

The 1.3.22 CLI consolidation is an approved direct cutover. Retired `top`, `machines`, `doctor`,
`clean`, `lease-policy`, `task keep-local`, `group retry-failed`, `group machines`, flat agent
resource configuration, and previous upgrade/migration spellings do not forward to the new
operations. No temporary compatibility implementation is retained for these spellings, so they
have no compatibility-registry lifecycle. Legacy Batch inspection and retry commands remain
owned by the submission cutover. The target CLI also does not promise aliases for the old flat
`list`, `inspect`, `retry`, `cancel`, or hyphenated Group command spellings.

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
- [ ] Agent stop/crash leaves an authorized runner and guardian process group alive, and leaves
  any existing optional tmux observer untouched; restart reconciles the same Attempt and never
  launches a successor or replays an observer attachment decision.
- [ ] Offline runner exit evidence records Task and Attempt identity and converges on restart
  within the declared 15-second healthy-host budget; mismatched evidence is retained and diagnosed.
- [ ] Terminal evidence is retained until claim archival and reservation accounting are durable,
  while verified absent local capacity may be released independently of shared finalization.
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
- [ ] `--group` overrides manifest `group.name`; omission of both permits ungrouped private work.
- [ ] Submission never implicitly adds its origin machine to a Group Worker Set.
- [ ] `group create --workers` uses an exact explicit Worker Set and defaults to `{current}` only
      when the option is omitted.
- [ ] Both submission input modes may transactionally create a missing named Group without exposing
      uncommitted Group or Worker truth.
- [ ] Existing Task and Group operations use resource-first command namespaces; `show` is the
  structured single-resource observation verb, `logs` is the specialized application-log byte
  stream, and there is no parallel `inspect` spelling.
- [ ] Continuous Task and log viewers remain read-only, exit after validated terminal observation
  by default, and cross retries only when `--follow-retries` is explicit.
- [ ] Attempt and Submission Operation remain diagnostic internals without daily CLI
  resource trees.
- [ ] Every CLI leaf is classified as finite, raw, continuous, or compatibility diagnostic;
      raw and continuous leaves reject structured format selection before domain work begins.
- [ ] Finite JSON and human output use the same canonical result and preserve the same exit code.
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
- [ ] Application progress remains Attempt-scoped and advisory; its project reporting policy
  is frozen per launch and never changes execution authority or running Attempts.
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

## Indexed Task history pages

`qexp task list --page-size N [--cursor TOKEN]` opts into Task-ID-ascending live
pagination. `N` is 1–1000; a cursor without a page size uses 50. Explicit `--limit`
cannot be combined with either pagination flag. Existing invocations retain their
output, default limit, filter semantics and zero/negative-limit behavior.

The Python entry point is `list_tasks_page(cfg, phase=None, group=None,
page_size=50, cursor=None)` with keyword-only query options. Unfiltered, phase,
Group and combined phase/Group queries have separate seekable index paths.
Continuations must supply the same filters; empty filters mean unfiltered.
Page size may change between requests.

JSON pages contain `items`, `next_cursor`, `consistency: "live"`,
`index_generation`, and `stop_reason` (`page_full`, `budget_exhausted`, or
`exhausted`). Items retain the legacy Task summary and complete dependency fields.
There is no exact count. Follow a non-null cursor even after an empty page.
A full final page may require another request to discover the end.

Each candidate is checked against current Task truth. A valid cursor chain returns
a Task ID at most once; inserts or filter changes behind the cursor can be missed.
Pages and dependency reads are not snapshots. Restarting traversal can repeat IDs;
clients requiring restart deduplication retain previously observed IDs. Rebuilding
the index expires cursors; ordinary Task mutations preserve its generation.

Invalid arguments/cursors return exit 2. Expired cursors and unavailable indexes
return exit 1. Paginated JSON failures contain only an `error` object with `code`
and `message`; codes are `invalid_argument`, `invalid_cursor`, `cursor_expired`,
`index_not_ready`, and `index_unavailable`. The Python `ObservationError` exposes
these codes. Failures never masquerade as an empty successful page. Restart an
expired traversal without a cursor; allow the global agent to finish background
build/recovery for an unavailable index. Doctor exposes its lifecycle state and
can request a rebuild after underlying damaged truth is repaired.

The bounded path never falls back to scanning historical Tasks. Legacy unlimited
listing, `top` and full Attempt detail retain their existing costs. Complete
Task/dependency records can be large; index budgets do not promise a fixed latency
independent of dependency fan-out or blocking storage I/O.
