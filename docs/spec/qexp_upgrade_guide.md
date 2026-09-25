---
doc_type: spec
status: active
updated_at: 2026-09-25
archived_at:
---

# qexp Upgrade and Ready-Index Recovery Guide

This guide is for an existing qexp project rooted at `PROJECT_ROOT` (for example,
`/mnt/share/myproject/.qexp`). Run commands with its explicit Project path.

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

The binding working-set format is disposable machine-local coordination state.
After installing a release that introduces it, restart each machine agent through
the normal rolling procedure. Every current registration generation starts
resident, revalidates authority and service obligations, and only then may become
dormant. Do not copy `working-set-v1.json` between machine runtimes or registration
generations. A missing or corrupt record causes conservative resident replay; it
does not require a Project schema migration. The shared
`operations/project-activation-v1/checkpoint.json` is preserved with the Project
and must not be reset as a way to clear work.

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
   Repeat the same command until verification is `completed` and healthy, or repair reports a
   degraded gate. Restart the machine agent only after repair reports both projections as active
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

   Repeat verify or repair while `group_ready_members.verification.state` is `building`. Continue
   only after verification is `completed` and healthy; treat `degraded` as a blocker that requires
   diagnosis. Use `--max-work-items 1` when a deliberately small maintenance slice is required.

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

Use `qexp task list --page-size 50 --format json` for indexed pages. Preserve
`--phase` and `--group` when following `next_cursor`. `index_not_ready` means the
background build has not completed; legacy Task listing remains available with
its original scan cost. `index_unavailable` means the query projection cannot
currently establish completeness. Inspect `qexp admin check --project PATH --format json` and
its `task_observation` status. Interrupted publication is recovered automatically;
after correcting damaged source data, `qexp admin repair --project PATH` requests another build.
Do not remove required capabilities or downgrade writers to bypass the gate.
A rebuild invalidates old cursors; restart explicitly without `--cursor`.
