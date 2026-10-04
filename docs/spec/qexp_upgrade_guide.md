---
doc_type: spec
status: active
updated_at: 2026-10-04
archived_at:
---

# qexp Upgrade and Ready-Index Recovery Guide

This guide is for an existing qexp project rooted at `PROJECT_ROOT` (for example,
`/mnt/share/myproject/.qexp`). Run commands with its explicit Project path.

## Release 1.3.23

The supported released-source path for the new maintenance protocols is 1.3.22
to 1.3.23. Upgrade the package and restart each machine's global agent in turn:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
qexp agent status --format json
```

Running training continues. The new agent reconstructs existing Attempt and
reservation evidence before new admission; restart success does not certify that
all background Project maintenance has completed. Project I/O isolation, binding
working sets, and scheduler diagnostics use the operational procedures below.
Explicit `admin repair` now performs a bounded slice: inspect `complete`,
`rerun_required`, and `next_action` before considering a repair finished.

For every current enabled Project binding, the replacement agent renews the existing
shared registration under its original generation and records the version of the
running qqtools package in `client_version`. This happens even when the eligibility
lease is not otherwise due. Writer-floor checks therefore observe the replacement
process after `qexp agent restart`; do not run `qexp project register` merely to
refresh a stale package version. A genuinely older active machine remains a blocker
until that machine is upgraded and restarted or its existing eligibility expires.

Group-service activation reads retained Submission Operations through the
runtime's 64 KiB direct-source boundary. A 1.3.23 agent could instead apply the
16 KiB locator limit and leave a valid 16--64 KiB Submission bootstrap in
`repair_required`. After installing a release containing the correction and
restarting the agent, ordinary upgrade discovery verifies and retries that exact
failure automatically. Do not shrink or delete the Submission, edit the upgrade
journal, or create a Group-service repair plan for this case. Unrelated size,
path, type, JSON, or activation failures remain blocked for diagnosis.

Machine-level upgrade status reports `blocked` when any accessible Project is
repair-required, paused, migration-blocked, or admission-blocked. `waiting` is
reserved for healthy pending work that can run or await its next probe.

Compatibility items `QQTOOLS-COMPAT-0018`, `0019`, and `0020` first ship in 1.3.23.
Their `1.4.0-stage1` and `1.4.0-stage2` source labels refer only to earlier
unreleased development data formats; they are not prerequisite package releases
or intermediate upgrade steps. Retained adapters still recognize those formats.
The existing 1.5.0 legacy-removal and 1.6.0 transition-purge deadlines are unchanged.

Executor atomic-write temporaries are not committed request, process, result, or
resolution evidence. Read-only status ignores recognized, locally owned regular
temporaries. Reconciliation reclaims them under the same write lock used by all
executor publishers, preserving live writes and committed records. Malformed
formal evidence, unknown filenames, links, and foreign ownership still block
recovery; do not delete such records to make an executor healthy.

## Upgrade to machine setup and Project enrollment

Release 1.3.22 separates machine identity from shared Project creation and local enrollment. After
upgrading the package, restart the global agent once. The runtime imports version-1 registry
bindings into its Project pool without changing their effective names or shared registration
generations. Imported names have unresolved provenance; existing bindings continue operating, but
future re-enrollment requires one explicit decision per Project:

```bash
qexp project register /path/to/project --name-source default
qexp project register /path/to/aliased-project --name-source explicit
```

Inspect the result with `qexp project list`. Do not infer the source merely because a Project name
equals the current global agent name.

For a blank environment, create the machine identity, create or verify shared Project truth, enroll,
and start the agent as separate steps:

```bash
qexp init --machine gpu2 --agent-mode daemon
qexp project init /projects/example
qexp project register /projects/example
qexp agent start
```

For a copied image whose old runtime remains another environment's responsibility, explicitly
detach and then reconcile only the saved inventory:

```bash
qexp init --machine gpu3 --detach-old-runtime --yes
qexp project register --from-pool
qexp agent start
```

This is a fresh-start procedure, not an idempotent bootstrap: every successful `init` replaces the
runtime identity. Retry a failed register or start step without repeating init. Do not detach the
only recovery copy of unfinished work; ordinary initialization requires the original runtime to
finish terminal publication, claim archival, reservation release, and recovery cleanup first.

## Diagnose a damaged MachineRuntime identity

Run the read-only diagnostic on the machine that owns the runtime:

```bash
qexp admin repair identity --dry-run --format json
```

Use `--machine-runtime-root PATH` when diagnosing a selected nondefault root. The
result shows that path, how it was selected, each identity evidence check, and a
next action. `healthy` has exit status 0 and covers identity only; Agent
configuration and readiness are outside its scope. `blocked` or `failed` has
exit status 1. A blocked result does not authorize `qexp init` as a recovery
shortcut: `init` creates a new identity. A pending replacement must resume its
recorded target through the existing `init` procedure. Automatic identity
restoration is unavailable, so the same command without `--dry-run` is a usage
error and changes nothing.

## Machine-rolling coordinator for supported future protocols

For a release covered by the machine-rolling coordinator contract, the normal operation on each
already registered machine is:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
```

Restart success means that the old global-agent process stopped and its replacement started; it
does not wait for every Project to become ready or claim that activation or legacy cleanup has
completed. The result may say `Ready: pending`. Observe later convergence first with
`qexp agent status`, then inspect the bounded machine registry view with:

```bash
qexp admin upgrade status --format json
```

Releases with agent exit diagnostics add only optional machine-local configuration and diagnostic
records; they do not change shared Project schemas or writer admission. Upgrade the package and
restart each machine agent so the new process adopts the configured rotation threshold. Existing
configuration without `log_max_bytes` reads as the 10 MiB default, and current writers preserve the
field on later name or residency-policy updates. Older packages ignore the diagnostic namespace;
downgrading loses diagnostic visibility and rotation behavior but does not reinterpret Project,
Attempt, claim, reservation, or runner truth. Mixed package and running-agent versions are therefore
supported only for the interval before the operator's normal machine-by-machine restart: status
shows the running instance's effective value separately from the newly configured value.

An exceptional recovery pass can advance every project visible in that machine registry while
training processes continue:

```bash
qexp admin upgrade advance --format json
```

The output records the discovery boundary and lists inaccessible or omitted roots. It must not be
interpreted as fleet-wide completeness. A project filter is available for diagnosis, but project
filters are not part of the normal rolling workflow.

When a project reports `paused`, `repair_required`, or `pause_pending`, do not edit shared JSON
files directly. Use the explicit project-scoped flow:

```bash
qexp admin upgrade pause --project PROJECT_ID --reason "describe the incident"
qexp admin upgrade status --project PROJECT_ID --format json
qexp admin upgrade plan --project PROJECT_ID --target MIGRATION_OR_PHASE --format json
qexp admin upgrade apply --project PROJECT_ID --repair-id REPAIR_ID --format json
qexp admin upgrade validate --project PROJECT_ID --repair-id REPAIR_ID --format json
qexp admin upgrade resume --project PROJECT_ID --format json
```

Repair plans must preserve revisions, evidence and a durable snapshot proof. Unsupported
authoritative corrections, stale plans, failed snapshots and validation failures remain blocked;
the coordinator does not release reservations or fabricate Task/Attempt completion.

The built-in `upgrade-journal-v1` migration publishes only coordinator-owned metadata under
`operations/upgrades/`. Its target is `metadata:upgrade-journal-v1`; it does not add fields to
`schema/version.json` or activate a new shared writer protocol. This keeps schema-6 roots readable
by strict 1.3.16 readers during a rolling deployment. Audit and activation verify the manifest
version, capability, source and target metadata protocol, and schema digest. A future migration
that changes a shared reader or writer contract must supply durable participant eligibility and
enforceable writer exclusion before activation; this metadata migration is not that proof.
The terminal manifest's schema digest covers the base schema and excludes only
the independently fenced `local-recovery-v1` required capability. Its
[admission protocol](qexp_local_responsibility.md#shared-admission-fence) waits for
the coordinator's existing work to complete before publishing that capability.
Audit evidence and repair plans still fingerprint the complete schema record;
changes after audit or repair planning remain invalid, including a capability
change outside the admission protocol. Unknown capabilities and changes to other
schema fields are not excluded from manifest drift detection.
Migration implementations must use the supplied upgrade storage facade for callback-owned JSON I/O;
it rejects reads and writes that exceed the declared slice byte budget before the operation proceeds.

The existing `admin migrate schema6` flow below is a historical drained transition and is not converted
to an online coordinator migration by this guide.

### Submission-owned Group publication in 1.3.22

`submission-group-publication-v1` protects Groups created inside Submission Operations. Upgrade the
package and restart the global agent on each registered machine, one machine at a time. Running
training continues. The agent advances the existing machine enrollment and Group-authority fence;
only after the project requires the new capability may a submission publish a provisional Group
carrying `creation_operation_id`. A 1.3.21 process then fails before project mutation instead of
reading that Group as an ordinary active Group.

Inspect rollout state with `qexp admin upgrade status --format json`. If normal enrollment cannot
finish, use `qexp admin upgrade advance --format json` from the machine scope and resolve every
reported inaccessible Project or participant. Do not remove `creation_operation_id`, edit the
required-capability set, or delete a provisional Group file manually. Same-key `qexp submit` retry
and `qexp admin repair --project PATH` own recovery. Submission-operation commit evidence must be retained for as
long as any Group refers to it.

## Do not treat every qqtools upgrade as an agent restart

### Scheduler diagnostics v1

The scheduler diagnostic store is a forward-only, machine-local derived format.
Upgrade the qqtools package and restart that machine's global agent:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
qexp agent diagnostics active --format json
```

There is no Project schema migration, legacy-counter import, Task-history
backfill, dual write, or old-format adapter. Before the restarted writer produces
evidence, status and diagnostic commands report `coverage=unknown`; they do not
infer a healthy empty store. Existing `degraded_reasons` behavior remains
available through its established readers and is not copied into scheduler-v1.

Upgrade and restart machines one at a time. An older running process does not
produce scheduler-v1 evidence. Invoking an older writer, or downgrading a
MachineRuntime after the new writer has operated it, is unsupported. Rollback may
lose or stop updating the new diagnostic history, but it does not reinterpret or
change Project Task/Attempt truth, claims, reservations, scheduling authority, or
recovery authority. Do not copy `diagnostics/scheduler-v1` between machine
runtimes, edit its JSON to clear a scheduling blocker, or treat its loss as proof
that a fault resolved.

### Blocking Project I/O isolation

Project I/O isolation is an L1 machine-rolling change. Upgrade the package and restart one
Machine's global agent at a time; do not stop or drain running training:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
qexp agent status --format json
```

The restarted agent reconstructs reservations and existing-Attempt supervision before new
admission. Old and new machines may operate against the same Projects during the rollout because
the executor adds no shared Project schema or writer capability. Its permanent atomic-JSON state is
machine-local under `project-io-executor-v1/`; old packages ignore it. There is no SQLite database,
Project migration, legacy reader, dual writer, or temporary compatibility shim.

If restart finds an interrupted registration CLI journal, recovery admission
automatically restores its captured shared records and local registry under the
existing machine exclusions, durably retires the journal, and reloads the registry
before continuing enrollment. A malformed journal or an uncertain partial replay
keeps that Machine fail-closed and visible as worker failure or ambiguity. Do not
delete `registration-transaction.json` or edit the registry to bypass recovery;
preserve both for exact replay and diagnosis.

Primary-demand scan and verification continuations are disposable controller
state. Restart obtains new proofs before admitting borrow work; ordinary ready
cursors and previously created reservations are not deleted or repurposed as
absence evidence. CPU and GPU reservations preserve `admission.admitted_as_borrow`
using the existing reservation record. This adds no shared migration or CLI
compatibility mode.

Borrow proof consumption may allocate one bounded local provisional offer before
the shared claim wins its fair request grant. If the agent restarts in that gap,
it releases only an exact offer with no request/process/result evidence; existing
or unreadable evidence still requires ordinary ambiguous-claim reconciliation.
Do not delete offers to make this recovery appear complete.

Submission-control repair now uses bounded isolated slices instead of a direct
shared-I/O maintenance thread. Its traversal continuation is disposable, while
source witnesses, parser checkpoints and proof receipts remain in the shared
Project. Restart resumes those proofs without a format migration or payload
copy into MachineRuntime.

Application progress keeps its existing shared and local formats. The restarted
agent may give one captured observation/publication pair two bounded background
opportunities, then requires another background family to run before repeating
that preference. This is disposable admission state: restart neither migrates
progress records nor treats the lost preference as publication evidence.

Retained recovery keeps its existing target capture, source hold and pending
backfill records. Source-hold publication/replay and pending source-record reads
use isolated transactions; local process capture consumes only exact retained
identities. A missing worker result after source retention is unknown until the
worker's exit is positively verified, after which exact hold replay recovers it.
Do not delete holds, regenerate a responsibility ledger or discard pending
capture batches to clear this state. These transactions introduce no shared
format migration and do not by themselves certify recovery readiness or source
release.

Retained evidence discovery resumes from a bounded directory cursor in the
existing machine-local capture-backfill checkpoint. It does not require a
source-format migration or a persistent worker. Discovery paths are journaled
before source record reads; restart replays pending identities before advancing
the cursor. A replaced source directory invalidates the advisory cursor and
requires rediscovery, not manual journal removal or assumed empty history.

Re-registration preserves the existing capture-generation journal and source
holds. The agent first isolates the source's generation-pending fence, applies
the local reset, isolates source normalization, then clears the local intent.
Each phase replays after interruption without advancing the census twice or
discarding captured responsibilities. Group namespace activation and source
retirement are independent transactions: active Group truth does not mean that
legacy source responsibilities have retired. Source hold removal requires the
existing exact local release receipt; missing or ambiguous executor results are
not permission to delete recovery files manually.

Progress v1/v2 retain their producer channels, local accepted observations and
shared snapshot formats. The agent samples locally and uses isolated shared
identity/publication transactions; cadence advances only after exact result
consumption. Restart conservatively restores cadence and preserves observation
timestamps. Existing training, producers and viewers need no migration. Do not
remove progress caches or executor records to clear an ambiguous publication.

Upgrade discovery and one-phase advancement now use the same four-slot Project
I/O executor as other shared services. The existing Project upgrade journal
remains the migration and replay authority. Machine-local pending, admission,
idle and retry summaries are disposable and rebuilt on restart; missing evidence
blocks that Project until observed rather than being treated as no upgrade work.

Legacy notification reconciliation also runs as isolated Project work. Existing
private policy revisions, credential files and conflict resolution are retained;
no new credential migration or compatibility period is introduced. Executor
records contain no webhook contents. Restart safely repeats unfinished imports
against the same private policy; do not delete that policy or its credentials to
clear ambiguous executor work. Private aged-credential cleanup does not wait for
another notification transaction.

In status, inspect `project_io_isolation`. Capacity is four workers and the supported hang limit is
two. `degraded` means isolation is active but at least one request is overdue or otherwise impaired;
`exceeded` means the supported two-hang peer-service envelope has been exceeded; `unknown` means the
agent cannot establish safe executor state. Preserve the MachineRuntime and investigate the listed
blocking Project IDs and scheduler diagnostic reason. Do not delete request, result, process, epoch,
or resolved records to clear status. Those files can be the only local evidence that a shared
mutation or provisional resource offer remains ambiguous.

Isolation does not provide a load-independent 15-second response guarantee. The 15-second
qualification covers two blocked bindings and one recovered healthy peer completing primary
admission and a due renewal under the workload defined in the
[product specification](qexp_product_spec.md#blocking-project-io-isolation). More healthy demand or
startup/recovery work can increase latency without exceeding the hung-worker envelope. Inspect
worker state and existing scheduling diagnostics; do not delete executor records, disable Projects,
or interpret queued renewal as successful renewal to reduce the apparent delay. The approved
2026-09-29 latency-contract correction introduces no migration, compatibility shim, tuning option,
or change to the L1 rollout and downgrade prerequisites.

The local activation/idle fence uses an optional, atomically replaced
`agent/activation-wake.json` record in MachineRuntime. No backfill is required:
absence is the initial generation, and the first activation creates the record.
It contains only a runtime identity and fresh UUID, not experiment history or
shared scheduling authority. Malformed or foreign-runtime evidence inhibits
idle exit; do not delete it while the agent is running to bypass that check.
An unchanged generation alone is never an idle proof. Local CLI activation
participates even for an already-running agent; `--no-activate` and the rule
that remote submissions do not wake stopped machines remain unchanged.

Agent stop fences the executor epoch, gives workers two seconds to return, sends `SIGTERM`, waits
another two seconds, then sends `SIGKILL`. A signal delivery is not proof that a worker in blocked
kernel I/O exited. The controller may stop successfully while status retains a nonzero
`unreaped_worker_count`; do not reuse, move, copy, or remove that MachineRuntime until each recorded
PID plus process start time is proven absent and shared transaction/reservation evidence is
reconciled. When a container hides the host process boundary, restoring that certainty may require
the container or cluster operator to terminate the owning container/host process namespace.

After the ordinary epoch is shut down, final eligible Project snapshots run through a fresh
two-second stop-publication epoch rather than controller threads. Retained ambiguous requests keep
their slots and can prevent their own stop snapshot. Responsive peers still use remaining slots;
an incomplete publication is recorded as cleanup failure while local stopped status and final
diagnostics continue. Do not delete retained requests to make the stop snapshot appear successful.

The executor-specific downgrade precondition is a fully quiescent executor after stopping the new
agent: live and exit-unverified worker counts are zero, every request/result is resolved, and no
provisional offer or possible shared mutation remains ambiguous. This condition is necessary, not
sufficient; it does not override the stricter no-downgrade rules of scheduler diagnostics,
registry-authoritative enablement, or another MachineRuntime protocol used by the source and target
versions. If an uninterruptible worker cannot be proven absent, do not start the older agent on that
MachineRuntime. Restore process certainty at the container or host boundary first, restart the new
version, and allow reconciliation to complete. Deleting the executor directory is not a downgrade
procedure and does not prove that a worker or shared mutation ended.

### Registry-authoritative Project enablement

The Project enable/disable correction keeps the existing registry and inventory
formats. It adds no Project schema migration, compatibility reader, dual writer,
or enablement journal. Upgrade the package and restart each machine's global
agent while running Attempts continue:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
qexp project list --format json
```

After restart, a live binding's registry value is effective and its inventory
value is a repairable mirror. `project list` shows both values and their separate
revisions. An inventory-only entry still carries reusable intent. Do not edit
either JSON file or copy MachineRuntime state between machines to resolve a
disagreement.

If `project enable` or `project disable` exits 1 with
`status=partially_committed`, the reported registry value is already effective.
Retry the same command; the retry is idempotent and repairs the inventory mirror.
If it returns `status=outcome_unknown`, first inspect a fresh `project list`, then
retry the same requested operation to establish durable convergence. Never issue
the inverse operation as rollback. `registry_enablement_unknown`, a missing exact
inventory entry, or an identity/path conflict requires preserving the runtime
and repairing its local records; inventory is not authority for reconstruction.

The corrected ordering is guaranteed only after the machine-local agent restart.
Using an older qqtools writer afterward, or downgrading a MachineRuntime already
operated by the corrected version, is unsupported. There is no version
negotiation or old-writer adapter. Running and launch-authorized Attempts retain
their existing authority across the rolling restart and across later explicit
disablement; disable affects later claims and demand contribution only.

The binding working-set format is disposable machine-local coordination state.
After installing a release that introduces it, restart each machine agent through
the normal rolling procedure. Every current registration generation starts
resident, revalidates authority and service obligations, and only then may become
dormant. Do not copy `working-set-v1.json` between machine runtimes or registration
generations. A missing or corrupt record causes conservative resident replay; it
does not require a Project schema migration. The shared
`operations/project-activation-v1/checkpoint.json` is preserved with the Project
and must not be reset as a way to clear work.

Binding removal now records an `activation-consumer-retirements-v2` intent in
the owning MachineRuntime and returns without synchronously accessing the shared
Project root. Preserve that directory across ordinary agent restarts until the
isolated worker clears each exact intent. Do not copy it to another
MachineRuntime or delete it to unblock re-registration; a pending intent safely
forces a fresh consumer generation. This is disposable machine-local
coordination and introduces no shared Project schema migration or mixed-writer
compatibility mode.

Preserve the complete `operations/project-activation-v1/` directory, including
events, snapshots, membership, and consumer records. Do not delete an activation
suffix or consumer cursor to speed an upgrade. Agents from before the journal
format are handled by conservative epoch bootstrap only when the event directory
is wholly absent; a partially missing journal is damage and requires repair.
Rolling older writers remain discoverable through bounded cold reconciliation,
but current agents should be restarted promptly so supported writers publish the
recoverable activation transaction directly.

Use the target protocol's documented activation procedure. An unchanged root
protocol can use an agent restart; a new capability requires either an explicit
upgrade or a qualified rolling admission protocol. The `local-recovery-v1`
admission protocol automatically waits for every participant's prepared
registration after package installation and global-agent restart. Its released
writer qualification covers 1.3.17 and 1.3.18. Admission fencing alone does not
certify local writer capture or activate history-independent discovery.

For the historical drained transitions below, stop mixed-version agents and
clients before protocol activation.

Release 1.3.15 introduces the paired schema-6 capabilities `cpu-lane-v1` and
`task-dependencies-v1`. Existing schema-6 roots require the explicit `schema6` upgrade procedure
below. A plain restart does not activate or convert the root.

## Recover a degraded ready index

If agent status or diagnostics report `ready_index=degraded` or `marker corrupt`, qexp has found a
ready marker that disagrees with authoritative Task, Submission, Group, or dependency records. It
stops new claims intentionally rather than scheduling potentially wrong work.

1. Stop normal clients that access this root. Do not manually remove files below `.qexp/indexes`.
2. Inspect the failure:

   ```bash
   qexp --project PROJECT_ROOT admin check --format json
   ```

3. If no running Task, active claim, or unfinished control operation requires intervention, rebuild
   the derived projection from durable Task truth:

   ```bash
   qexp --project PROJECT_ROOT admin repair --format json
   ```

   A member audit can return `verification.state: building` while its projection remains active.
   Repeat the same command while `complete=false` and `rerun_required=true`, preserving the
   returned `scope.work_generation`. Stop and investigate an `outcome=blocked` result. Restart the
   machine agent only after repair is complete and reports both projections as active
   (or reports the member projection as legacy on a root where that capability is not installed).

4. Restart the machine agent:

   ```bash
   qexp agent restart
   ```

   Restart does not wait for Project readiness. If it reports `Ready: pending`, run
   `qexp agent status` until the required Project evidence is ready or an actionable blocker is
   reported.

`admin repair` regenerates the ready markers, catalog, and reservations. It does not discard
Task or Attempt truth. If repair reports `blocked`, resolve the listed operation or execution
evidence first; do not force-delete it.

If an upgraded schema-1 full-audit descriptor reports
`legacy_capture_identity_unknown`, repeating the ordinary command deliberately
remains attached to that blocked generation. Inspect the intervention evidence,
then use `qexp --project PROJECT_ROOT admin repair --retry-intervention` to
create a successor audit with a fresh source capture. This preserves the blocked
legacy descriptor as evidence and does not discard Task or Attempt truth.

## Upgrade an existing schema-6 root to qqtools 1.3.15

Perform these steps one project at a time. `MACHINE_RUNTIME_ROOT` is required only when the
machine runtime is not at qexp's default location.

1. Install the same qqtools version on every participating machine, but do not restart normal
   qexp agents or clients for this project yet.
2. Stop submissions for the project. Let Tasks finish or cancel them, then confirm there are no
   active claims, running process evidence, reservations, or unfinished operations. Stop normal
   agents, CLIs, and Python processes that have this root open. If a shared machine-global agent
   supervises other projects, account for those projects before stopping it.
3. From a coordinator machine, run the read-only preflight:

   ```bash
   qexp --project PROJECT_ROOT admin migrate schema6 check --format json
   ```

   Proceed only when `blockers` is empty.
4. Create the activation and save the returned `activation_id`:

   ```bash
   qexp --project PROJECT_ROOT admin migrate schema6 start --format json
   ```

5. On every machine listed in the returned `participants`, verify that normal clients remain
   stopped and attest with that machine's logical qexp name:

   ```bash
   qexp --project PROJECT_ROOT --machine MACHINE_NAME \
     admin migrate schema6 attest \
     --activation-id ACTIVATION_ID \
     --confirm-clients-stopped \
     --format json
   ```

6. On the coordinator, complete the conversion:

   ```bash
   qexp --project PROJECT_ROOT \
     admin migrate schema6 resume \
     --activation-id ACTIVATION_ID \
     --format json
   ```

   Success requires `phase: completed`. If interrupted, use `admin migrate schema6 status`, correct the
   reported blocker, collect fresh attestations, and rerun `resume` with the same activation ID.
7. Before returning the project to service, verify and repair its derived ready projection:

   ```bash
   qexp --project PROJECT_ROOT admin check --format json
   qexp --project PROJECT_ROOT admin repair --format json
   ```

   Repeat verify or repair while `complete=false` or
   `group_ready_members.verification.state` is `building`. Continue only after verification is
   completed and healthy; treat `blocked` or `degraded` as a blocker that requires diagnosis. Use
   `--max-work-items 1` when a deliberately small maintenance slice is required.

8. Restart agents and clients after the activation is complete and the ready index is active.

## Historical migration support matrix

The CLI consolidation changes where these operations are found; it does not retire any historical
migration protocol. The supported-version review retains all three entries because released source
and recovery obligations still require distinct prerequisites:

| Source state | Retained command | Reason and operator path |
| --- | --- | --- |
| Drained schema-5 Project | `qexp admin migrate schema --project PATH --to-schema 6` | This is the only supported one-way conversion to schema 6. Active claims or running Attempts block it. |
| Schema-6 root missing the 1.3.15 capabilities | `qexp admin migrate schema6 {check|start|status|attest|resume} --project PATH` | This remains the drained, participant-attested activation path; it is not an online coordinator migration. |
| Project with legacy per-Project agent metadata | `qexp admin migrate agent --project PATH --machine NAME` | This remains the protected ownership-preserving import path and keeps runner/terminal recovery ordering. |

No evidence currently proves that the supported source range or recovery obligations have ended.
Removing one of these commands therefore requires a later supported-version decision that updates
implementation, tests, and this guide together. The consolidated `admin upgrade` family remains the
separate machine-rolling path for protocols that explicitly qualify for it.

## Older schema-5 projects

Schema-5 roots use a different, one-way migration. They must be drained before conversion:

```bash
qexp admin migrate schema --project PROJECT_ROOT --machine MACHINE_NAME --to-schema 6
```

After the schema-5 migration, follow the schema-6 capability-upgrade procedure when targeting
qqtools 1.3.15.

## Quick decision table

| Situation | Correct action |
| --- | --- |
| Same qexp protocol, healthy ready index | Upgrade every participating machine, restart agents one at a time, then confirm readiness with `agent status`. |
| `ready_index=degraded` or `marker corrupt` | Stop normal clients, run `admin check`, then `admin repair`; restart only after the index is active, and observe post-restart readiness with `agent status`. |
| Existing schema-6 root moving to 1.3.15 | Drain all participants and run `admin migrate schema6 check/start/attest/resume`. |
| Existing schema-5 root | Drain it and run `qexp admin migrate schema --project PATH --to-schema 6` before the schema-6 capability upgrade. |

## Task history pagination

After upgrading the package and restarting each machine's global agent, registered
projects automatically prepare their Task observation indexes. Existing projects
first satisfy the canonical Group/recovery writer boundary; running training
continues. There is no mandatory per-project activation command. New projects
initialize their empty index during `qexp init`.

With Project-I/O isolation, the agent advances index builds and obsolete-generation
cleanup using bounded isolated requests instead of a separate shared-I/O thread.
Existing shared build checkpoints and pagination formats remain unchanged.
Restart resumes unfinished work automatically; do not delete the projection or
copy MachineRuntime files to recover a build.

An ordinary agent restart also resumes an expired registration when the exact
same runtime still owns its generation. No re-registration or `--adopt-existing`
is required merely because the agent was offline past its eligibility lease.
If a successor runtime has acquired the logical machine name, the old binding
remains fenced; restart never overwrites that successor or relaunches an orphaned
Attempt.

Use `qexp task list --page-size 50 --format json` for indexed pages. Preserve
`--phase` and `--group` when following `next_cursor`. `index_not_ready` means the
background build has not completed; legacy Task listing remains available with
its original scan cost. `index_unavailable` means the query projection cannot
currently establish completeness. Inspect `qexp admin check --project PATH --format json` and
its `task_observation` status. Interrupted publication is recovered automatically;
after correcting damaged source data, `qexp admin repair --project PATH` requests another build.
Do not remove required capabilities or downgrade writers to bypass the gate.
A rebuild invalidates old cursors; restart explicitly without `--cursor`.
