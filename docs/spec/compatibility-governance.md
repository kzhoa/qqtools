---
doc_type: spec
status: active
updated_at: 2026-09-08
archived_at:
---

# Compatibility Governance

## Authority and scope

This policy covers temporary compatibility behavior across qqtools. The tracked
`compatibility-registry.toml` is the machine-readable lifecycle authority. Product specifications
remain authoritative for behavior and protected workflows; pitches explain complex delivery work
but do not replace the registry.

Every temporary reader, writer, CLI alias, migration path, protocol adapter, or behavior fixture
must have one registry item. Permanent multi-format support is a product contract and must not be
misrepresented as temporary compatibility.

## Lifecycle

Each item moves through these states:

```text
planned -> compatibility_active -> legacy_removed -> removed from registry
```

- `planned`: the future contract is registered but no compatibility implementation or marker may
  exist in source, tests, or scripts.
- `compatibility_active`: the new contract is the write/default path and the temporary legacy
  boundary remains available.
- `legacy_removed`: the legacy public entry or writer has been removed; only an explicitly planned
  reader, migrator, or transition check may remain.
- Registry removal: temporary runtime support, warnings, markers, and old-behavior fixtures are
  absent, and the target release has reached `transition_purged_in`.

Registry items use exact `X.Y.Z` versions. `introduced_in` must precede `legacy_removed_in`, and
`transition_purged_in` must not precede `legacy_removed_in`. Equal removal and purge versions are
allowed when no separate migration release is needed.

## Registry contract

The top-level `next_id` is the next numeric compatibility ID to allocate. It only increases, even
after older items leave the registry. This prevents ID reuse without retaining completed records.
The registry may contain no `[[items]]` entries when no compatibility work is unfinished.

Every unfinished `[[items]]` entry contains:

- a unique `QQTOOLS-COMPAT-NNNN` ID and identical `marker`;
- `component`, `kind`, and module-level `owner`;
- lifecycle `status` and the three release versions;
- tracked or non-ignored `decision_refs` and `verification` files; the clean committed preflight
  guarantees they are tracked before release;
- `pitch_refs` when a planned item needs later implementation. Planned items require at least one
  local pitch reference; active and legacy-removed items may omit it. References must be unique,
  repository-relative `.md` files under `docs/pitch/` and must exist locally, but are deliberately
  not required to be Git-tracked;
- optional append-only `extensions` for approved deadline changes.

## Operational compatibility gate

A bounded release window is necessary but not sufficient. Every compatibility item must also
define and verify how operators deploy the release across real machines and projects. The decision
reference must state:

- the supported rollout unit: process, machine, project, or coordinated fleet;
- the exact operator command sequence for the normal path;
- whether old and new processes may access the same durable state during rollout;
- how already running workloads retain supervision and publish their terminal outcome;
- how interrupted migration resumes without repeating completed work; and
- the machine-level inspection and recovery command when one machine owns multiple projects.

### Operational migration levels

Compatibility work is classified by the most disruptive action required in its normal upgrade
path. The level measures operator burden and workload impact, not implementation complexity.

| Level | Name | Maximum required operator action | Running workloads |
| --- | --- | --- | --- |
| L0 | Hot compatible | Upgrade the package; no agent restart or migration command | Continue without supervision interruption |
| L1 | Machine rolling | Upgrade the package and restart the global agent on one machine at a time | Continue; restarted agents recover supervision and terminal publication |
| L2 | Coordinated agent stop | Stop all qexp agents, run one machine-level coordinator migration for all discoverable projects, then restart agents | Continue; training processes are not stopped or drained |
| L3 | Workload drain | Stop admission and wait for or terminate running workloads before migration | Interrupted or deliberately drained |

Every compatibility decision must declare one normal operational level and may declare a more
disruptive recovery level. Its verification files must exercise the declared normal level. A lower
level implementation may always recover through a declared higher level, but documentation must
not advertise the recovery path as the normal workflow.

Per-project manual migration is not a separate level and cannot satisfy L0, L1, or L2. When a
machine can own multiple projects, every supported level operates at machine scope or broader;
project filters may exist for diagnosis but cannot be mandatory in the normal procedure.

L0 applies only when already running old processes can continue safely without loading the new
package and no durable activation is required. L1 is the default ceiling for new qexp persisted
fields, indexes, capabilities, and runtime protocols. L2 requires a dedicated approved decision
with evidence that mixed-version rolling activation cannot be made safe. L3 describes a
nonconforming historical or proposed workflow; it is not an allowed qexp normal or recovery path.
A major version or breaking-change declaration does not waive running-workload continuity.

Items introduced before 2026-09-08 retain their recorded lifecycle, but their next lifecycle or
deadline decision must document the actual normal and recovery levels. This grandfathering does
not authorize a new persisted writer or a new drain requirement under an existing item.

For qexp persisted fields, indexes, capabilities, and runtime protocols first introduced after
2026-09-08, the normal path must be a machine-by-machine rolling upgrade. On each machine the
operator installs the new package and restarts the global agent; the new agent discovers every
registered project on that machine and advances required migrations automatically. The normal path
must not require per-project `cd`, `doctor`, activation IDs, attestations, or manual journal loops.

qexp migration must preserve already running training processes. Agent restart may temporarily
pause queue admission and scheduling, but it must not terminate training, wait for all training to
finish, discard machine-local execution evidence, or require idle GPUs. A release that cannot meet
this contract may not introduce the persisted change as ordinary qexp compatibility work.

When a protocol-specific safety proof shows that L1 rolling activation is not feasible, a dedicated
approved decision may select L2: stop all qexp agents and use one coordinator command to advance
every discoverable project. It must still preserve running training processes and permit later
agents to reconcile durable terminal evidence. The decision must explain why L1 cannot be met and
verify the machine-level all-project workflow; it cannot fall back to manual per-project
maintenance. Machine retirement must use explicit identity or a durable retirement tombstone;
heartbeat age alone cannot authorize metadata deletion.

If the permanent migration infrastructure needed by these rules does not yet exist, register the
feature as `planned` and deliver that infrastructure before introducing its new persisted writer.
Do not ship a drain-required schema first and defer operability to a later patch.

An extension names `legacy_removed_in` or `transition_purged_in`, chains from the previous effective
value to a later version, records the approval version and reason, and points to a tracked decision.
Original deadlines are not overwritten.

Temporary implementation and behavior fixtures carry their compatibility ID. The checker scans
tracked and non-ignored files under `src/`, `tests/`, and `scripts/`. Planned items must have no
marker; active and legacy-removed items must retain at least one marker.

`pitch_refs` are implementation navigation, not behavior authority: `decision_refs` remain the
formal specification and ADR basis. A pitch archive or rename must update any live reference. On
registry removal, its pitch references disappear with the item. Because pitches are intentionally
Git-ignored, a new clone without the local pitch set fails registry validation and local preflight;
this is an accepted consequence of the trusted local-publish workflow.

Completed items are not a registry state and are not retained indefinitely. At release check time,
the checker reads the registry from the exact `v<current source version>` tag. An item may disappear
only when the target reaches its effective `transition_purged_in` and its markers are absent. The
same baseline prevents `next_id` rollback and reuse of retired IDs. If that prior tag predates the
registry, the comparison bootstraps without historical items. Git history, release tags, and the
CHANGELOG are the historical record.

### Continuity and rollout completion

Operator coordination level and service continuity are separate contracts. Each decision must
specify admission/scheduling interruption scope, a measurable interruption budget, terminal
reconciliation behavior, and verification evidence. Two commands alone do not prove L1: waiting
for the fleet to upgrade or a full backfill to finish must not suspend compatible scheduling.

Within the declared supported version range, operators may upgrade machines in any order and
pause rollout between machines. Existing functionality remains available through compatible paths.
Distinguish package/agent deployment, per-project feature activation, and legacy cleanup in status
and acceptance criteria. Successful restart does not imply that background migration is complete.
Active legacy Attempts must not prevent deployment completion or restored agent service; their
evidence adapters remain available until safe terminal reconciliation, even if record conversion
is deferred. Cleanup must not turn deferred work into a workload-drain prerequisite.

Activation proofs cover every relevant writer: agents, runners, submission CLIs, recovery tools,
and unfinished durable operations. Agent readiness alone is insufficient. L2 also requires safety
against remaining runner writes. Returning incompatible processes must be prevented from unsafe
mutation by an enforceable protocol boundary, not merely by readiness metadata. Retirement changes
participation eligibility without discarding outstanding Attempt or terminal evidence.

Machine scope owns project discovery, bounded progress, and aggregate status. Each shared project
owns its migration journal and fenced activation; concurrent machines must not create independent
canonical migrations. Reports state the known discovery boundary and inaccessible registered roots;
they cannot enumerate projects absent from every available discovery source.

### Supported upgrade paths and cleanup

Each persisted-protocol decision declares supported source package/data versions, mixed-version
ranges, and the direct upgrade path to its target. Removing an old live writer and removing the
ability to migrate historical data require separate evidence. A fixed release count does not prove
that inactive projects or long-running Attempts no longer need an adapter.

The two-command promise applies to explicitly supported source versions. Supported sources must
not require manual intermediate package installation or downgrade. Unsupported sources must be
identified before destructive migration, with an actionable machine-level explanation. Automatic
chained migration may satisfy a supported path. Required adapters must be permanent contracts or
have their registered deadlines explicitly extended before removal; deadlines must not silently
override continuity requirements.

These operational requirements are decision/review obligations today. Structured registry fields
and automated operational-evidence enforcement remain planned in the coordinator pitch; the
existing registry validator does not prove these runtime guarantees.

### Migration development and runtime cost

For qexp persisted-protocol changes, the decision and verification evidence must also establish:

- **Minimum necessary transition:** explain whether conversion or a required capability is needed.
  Additive defaults and rebuildable projections must be checked against actual old readers and
  whole-record writers. Apply phase-specific safety predicates rather than a mandatory fleet
  barrier for every change.
- **Protocol-boundary ownership:** keep version interpretation, write-path selection, and historical
  Attempt adapters behind stable interfaces. A generic coordinator is not a proof of online safety.
  Declare dependencies and conflicting mutations when upgrades can chain.
- **Concurrent-write correctness:** prove that backfill preserves updates made before and after
  scanning a record. A watermark alone is not sufficient. Identify the boundary honored by source
  writers and prevent races between checking compatibility and mutating state; a newly introduced
  marker cannot exclude an old process that does not consult it. Retain safe compatibility until
  incompatible activation can be enforced.
- **Runtime isolation and cost:** distinguish lightweight admission checks from background migration.
  Migration must not make scans or blocking migration I/O prerequisites for supervision, renewal,
  terminal reconciliation, or compatible scheduling. Define machine-wide resource budgets, fair
  project progress, bounded concurrency/backlog, contention backoff, and steady-state behavior that
  does not scan historical records. Measure numerical latency and I/O budgets, including shared
  storage impact on training; record baseline, scale, environment, and slow-I/O results. A soft
  between-record deadline is not a bound on blocking I/O.
- **Real source-version evidence:** run released source packages and their live writers through the
  supported upgrade paths. Target-code tests over old data alone do not prove mixed-process safety.
  Verify deployment, activation, and cleanup separately, including long-lived legacy Attempts.

A failed migration may leave compatible service available only while that path remains provably
safe. Indeterminate authoritative state or protocol safety requires isolating unsafe project writes
while preserving training and safely collectable evidence; it must not disable unrelated projects.

Concrete execution isolation, locking, and cache invalidation belong in the migration architecture
decision, not in the always-loaded agent instructions. These requirements remain review obligations
until their planned automated evidence gates are implemented.

## Commands

```bash
python scripts/checks/check_compatibility_registry.py validate
python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z
python scripts/checks/check_compatibility_registry.py check --release-version X.Y.Z
```

`validate` checks registry structure and repository evidence. `plan` reports the state or registry
removal required at a target release. `check` fails when the recorded state does not match that
target, an item was removed early, a retired marker remains, or the ID watermark regresses.

The release operator must inspect `plan`, resolve every due action, commit the candidate, and run:

```bash
python scripts/release_preflight.py --target-version X.Y.Z
```

The local preflight is the only hard compatibility gate. The tag publish workflow trusts it and
does not repeat compatibility validation.

## When a dedicated decision document is required

A normal CLI alias or bounded local reader needs only a registry entry and ordinary feature
documentation. A dedicated pitch or ADR is required when compatibility changes shared persisted
truth, permits different versions to run concurrently, performs an irreversible rewrite, changes
ownership, or requires locks, markers, recovery, or rollback protocols.

## References

- [qexp machine-level rolling upgrade coordinator](../pitch/qexp-machine-rolling-upgrade-coordinator.md)
