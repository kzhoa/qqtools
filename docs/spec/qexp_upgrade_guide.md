---
doc_type: spec
status: active
updated_at: 2026-09-06
archived_at:
---

# qexp Upgrade and Ready-Index Recovery Guide

This guide is for an existing qexp project rooted at `PROJECT_ROOT` (for example,
`/mnt/share/myproject/.qexp`). Run commands with its explicit shared-root path.

## Do not treat every qqtools upgrade as an agent restart

`qexp agent restart` is sufficient only when the target release does not introduce a new qexp
root protocol or capability. Do not mix agent/client versions against the same project root during
a protocol activation.

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
