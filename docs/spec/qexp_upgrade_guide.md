---
doc_type: spec
status: active
updated_at: 2026-09-20
archived_at:
---

# qexp Upgrade and Ready-Index Recovery Guide

This guide is for an existing qexp project rooted at `PROJECT_ROOT` (for example,
`/mnt/share/myproject/.qexp`). Run commands with its explicit shared-root path.

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

## Machine-rolling coordinator for supported future protocols

For a release covered by the machine-rolling coordinator contract, the normal operation on each
already registered machine is:

```bash
python -m pip install --upgrade qqtools
qexp agent restart
```

Restart success means that the global agent restarted and resumed supervision; it does not claim
that project activation or legacy cleanup has completed. Inspect the bounded machine registry
view with:

```bash
qexp agent upgrade status --format json
```

An exceptional recovery pass can advance every project visible in that machine registry while
training processes continue:

```bash
qexp agent upgrade coordinate --format json
```

The output records the discovery boundary and lists inaccessible or omitted roots. It must not be
interpreted as fleet-wide completeness. A project filter is available for diagnosis, but project
filters are not part of the normal rolling workflow.

When a project reports `paused`, `repair_required`, or `pause_pending`, do not edit shared JSON
files directly. Use the explicit project-scoped flow:

```bash
qexp agent upgrade pause --project PROJECT_ID --reason "describe the incident"
qexp agent upgrade inspect --project PROJECT_ID --format json
qexp agent upgrade plan --project PROJECT_ID --target MIGRATION_OR_PHASE --format json
qexp agent upgrade apply --project PROJECT_ID --repair-id REPAIR_ID --format json
qexp agent upgrade validate --project PROJECT_ID --repair-id REPAIR_ID --format json
qexp agent upgrade resume --project PROJECT_ID --format json
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

The existing `upgrade schema6` flow below is a historical drained transition and is not converted
to an online coordinator migration by this guide.

### Submission-owned Group publication in 1.3.22

`submission-group-publication-v1` protects Groups created inside Submission Operations. Upgrade the
package and restart the global agent on each registered machine, one machine at a time. Running
training continues. The agent advances the existing machine enrollment and Group-authority fence;
only after the project requires the new capability may a submission publish a provisional Group
carrying `creation_operation_id`. A 1.3.21 process then fails before project mutation instead of
reading that Group as an ordinary active Group.

Inspect rollout state with `qexp agent upgrade status --format json`. If normal enrollment cannot
finish, use `qexp agent upgrade coordinate --format json` from the machine scope and resolve every
reported inaccessible Project or participant. Do not remove `creation_operation_id`, edit the
required-capability set, or delete a provisional Group file manually. Same-key `qexp submit` retry
and `qexp doctor repair` own recovery. Submission-operation commit evidence must be retained for as
long as any Group refers to it.

## Do not treat every qqtools upgrade as an agent restart

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
   qexp --shared-root PROJECT_ROOT doctor verify --format json
   ```

3. If no running Task, active claim, or unfinished control operation requires intervention, rebuild
   the derived projection from durable Task truth:

   ```bash
   qexp --shared-root PROJECT_ROOT doctor repair --format json
   ```

   A member audit can return `verification.state: building` while its projection remains active.
   Repeat the same command until verification is `completed` and healthy, or repair reports a
   degraded gate. Restart the machine agent only after repair reports both projections as active
   (or reports the member projection as legacy on a root where that capability is not installed).

4. Restart the machine agent:

   ```bash
   qexp agent restart
   ```

`doctor repair` regenerates the ready markers, catalog, and reservations. It does not discard
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
   qexp --shared-root PROJECT_ROOT upgrade schema6 check --format json
   ```

   Proceed only when `blockers` is empty.
4. Create the activation and save the returned `activation_id`:

   ```bash
   qexp --shared-root PROJECT_ROOT upgrade schema6 start --format json
   ```

5. On every machine listed in the returned `participants`, verify that normal clients remain
   stopped and attest with that machine's logical qexp name:

   ```bash
   qexp --shared-root PROJECT_ROOT --machine MACHINE_NAME \
     upgrade schema6 attest \
     --activation-id ACTIVATION_ID \
     --confirm-clients-stopped \
     --format json
   ```

6. On the coordinator, complete the conversion:

   ```bash
   qexp --shared-root PROJECT_ROOT \
     upgrade schema6 resume \
     --activation-id ACTIVATION_ID \
     --format json
   ```

   Success requires `phase: completed`. If interrupted, use `upgrade schema6 status`, correct the
   reported blocker, collect fresh attestations, and rerun `resume` with the same activation ID.
7. Before returning the project to service, verify and repair its derived ready projection:

   ```bash
   qexp --shared-root PROJECT_ROOT doctor verify --format json
   qexp --shared-root PROJECT_ROOT doctor repair --format json
   ```

   Repeat verify or repair while `group_ready_members.verification.state` is `building`. Continue
   only after verification is `completed` and healthy; treat `degraded` as a blocker that requires
   diagnosis. Use `--max-work-items 1` when a deliberately small maintenance slice is required.

8. Restart agents and clients after the activation is complete and the ready index is active.

## Older schema-5 projects

Schema-5 roots use a different, one-way migration. They must be drained before conversion:

```bash
qexp migrate --shared-root PROJECT_ROOT --machine MACHINE_NAME --to-schema 6
```

After the schema-5 migration, follow the schema-6 capability-upgrade procedure when targeting
qqtools 1.3.15.

## Quick decision table

| Situation | Correct action |
| --- | --- |
| Same qexp protocol, healthy ready index | Upgrade every participating machine, then restart agents one at a time. |
| `ready_index=degraded` or `marker corrupt` | Stop normal clients, run `doctor verify`, then `doctor repair`; restart only after the index is active. |
| Existing schema-6 root moving to 1.3.15 | Drain all participants and run `upgrade schema6 check/start/attest/resume`. |
| Existing schema-5 root | Drain it and run `qexp migrate --to-schema 6` before the schema-6 capability upgrade. |

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
currently establish completeness. Inspect `qexp doctor verify --format json` and
its `task_observation` status. Interrupted publication is recovered automatically;
after correcting damaged source data, `qexp doctor repair` requests another build.
Do not remove required capabilities or downgrade writers to bypass the gate.
A rebuild invalidates old cursors; restart explicitly without `--cursor`.
