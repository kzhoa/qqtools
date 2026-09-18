---
doc_type: spec
status: active
updated_at: 2026-09-16
archived_at:
---

# Compatibility Governance

## Authority and scope

This policy covers temporary compatibility behavior across qqtools. The tracked
`compatibility-registry.toml` is the machine-readable lifecycle authority for unfinished
compatibility work.

The public repository must be sufficient to build, test, release, and safely advance every
registered compatibility lifecycle. Private pitches, ADRs, local notes, and other design records may
inform owner decisions, but they are optional context and are never registry dependencies.

Every temporary reader, writer, CLI alias, migration path, protocol adapter, warning, or behavior
fixture must have one registry item. Permanent multi-format support is a product contract and must
not be represented as temporary compatibility.

## Lifecycle

Each item moves through these states:

```text
planned -> compatibility_active -> legacy_removed -> removed from registry
```

- `planned`: public implementation intent has been registered, but no temporary compatibility
  implementation or marker may yet exist in `src/`, `tests/`, or `scripts/`.
- `compatibility_active`: the new contract is the write/default path and the temporary legacy
  boundary remains available.
- `legacy_removed`: the legacy public entry or writer has been removed; only an explicitly retained
  reader, migrator, adapter, warning, or transition check may remain.
- Registry removal: temporary runtime support, warnings, markers, and old-behavior fixtures are
  absent, and the target release has reached `transition_purged_in`.

`planned` is a public implementation state, not a private roadmap state. Exploratory work that is
not ready to become public implementation intent stays outside the registry.

Registry items use exact `X.Y.Z` versions. `introduced_in` must precede `legacy_removed_in`, and
`transition_purged_in` must not precede `legacy_removed_in`. Equal removal and purge versions are
allowed when no separate migration release is needed.

## Registry format

The compatibility registry is a developer-facing lifecycle ledger, not a user-facing data format,
wire protocol, or persisted product schema. It intentionally has **no registry schema version**.
The top-level field `schema_version` is forbidden and must not be introduced in future revisions.
Changing developer-only registry metadata does not require a migration framework or a versioned
parser.

Current unfinished entries contain:

- a unique `QQTOOLS-COMPAT-NNNN` ID and identical `marker`;
- `component`, `kind`, and module-level `owner`;
- a concise public `summary` identifying the temporary compatibility boundary;
- lifecycle `status` and the three release versions;
- one or more repository-relative `verification` paths that exist in the committed checkout;
- optional append-only `extensions` for approved deadline changes; and
- structured rollout fields when the item declares an operational migration contract.

`pitch_refs`, `decision_refs`, and `action_refs` are not registry fields. Private planning and design
files are not registry dependencies. Implementation progress belongs to disposable feature-branch
state, commits, tests, and CI rather than to the compatibility ledger.

The checker validates only the registry in the current checkout. It does not parse compatibility
registries from old release tags and it does not provide backward compatibility for obsolete
developer-registry formats. Product compatibility moves forward: once a temporary compatibility
item has been intentionally cleaned up, it stays cleaned up. Git history and release tags remain
available for human archaeology, not as machine-readable registry inputs.

The top-level `next_id` is the next compatibility ID to allocate. Maintainers advance it when new
items are created and do not reuse retired IDs. This is a forward-only maintenance convention, not
a cross-release registry migration protocol.

Temporary implementation and behavior fixtures carry their compatibility ID. The checker scans
tracked and non-ignored files under `src/`, `tests`, and `scripts`. Planned items must have no marker;
active and legacy-removed items must retain at least one marker.

## Correlation and implementation history

The compatibility ID is the stable correlation key across public engineering evidence. Feature work
items, source markers, tests, and useful commit messages should include the relevant
`QQTOOLS-COMPAT-NNNN` ID when practical.

A feature branch may use `.dev/work-item.md` to carry temporary implementation context. That file is
not lifecycle authority and is stripped before promotion to `dev`. Pull requests and issues are not
required. Long-term correctness must remain recoverable from the registry, code markers,
verification tests, public product documentation when applicable, and Git history.

Completed items are not a registry state and are not retained indefinitely. Once their temporary
behavior and markers are intentionally removed, they leave the current ledger and do not return.
The checker does not reconstruct prior registry state to second-guess that forward cleanup. Git
history, release tags, and the CHANGELOG remain available for human historical inspection only.

## Operational compatibility gate

A bounded release window is necessary but not sufficient. Compatibility work that changes persisted
state, runtime protocols, or operator rollout must define enough structured registry metadata and
verification evidence to establish how operators deploy the release across real machines and
projects.

The structured contract records, when applicable:

- the normal and recovery operational levels;
- supported source package/data versions and target version;
- the mixed-version range;
- the normal operator command sequence;
- the machine-level inspection and recovery commands;
- whether running workloads must remain continuous; and
- the measurable interruption budget.

These fields are public operational commitments, not a record of internal design deliberation.
Private design records may contain deeper reasoning, alternatives, or implementation sketches, but
public validation must never require them.

### Operational migration levels

Compatibility work is classified by the most disruptive action required in its normal upgrade path.
The level measures operator burden and workload impact, not implementation complexity.

| Level | Name | Maximum required operator action | Running workloads |
| --- | --- | --- | --- |
| L0 | Hot compatible | Upgrade the package; no agent restart or migration command | Continue without supervision interruption |
| L1 | Machine rolling | Upgrade the package and restart the global agent on one machine at a time | Continue; restarted agents recover supervision and terminal publication |
| L2 | Coordinated agent stop | Stop all qexp agents, run one machine-level coordinator migration for all discoverable projects, then restart agents | Continue; training processes are not stopped or drained |
| L3 | Workload drain | Stop admission and wait for or terminate running workloads before migration | Interrupted or deliberately drained |

Every item with an operational contract declares one normal level and may declare a more disruptive
recovery level. Its verification files must exercise the declared normal level. A lower-level
implementation may recover through a declared higher level, but user-facing documentation must not
advertise the recovery path as the normal workflow.

Per-project manual migration is not a separate level and cannot satisfy L0, L1, or L2. When a
machine can own multiple projects, every supported level operates at machine scope or broader;
project filters may exist for diagnosis but cannot be mandatory in the normal procedure.

L0 applies only when already running old processes can continue safely without loading the new
package and no durable activation is required. L1 is the default ceiling for new qexp persisted
fields, indexes, capabilities, and runtime protocols. L2 is allowed only when implementation and
verification evidence establish that mixed-version rolling activation cannot be made safe. L3 is
not an allowed qexp normal or recovery path. A major-version or breaking-change declaration does
not waive running-workload continuity.

Items introduced before 2026-09-08 retain their recorded lifecycle, but their next lifecycle or
deadline change must record the actual normal and recovery levels when those fields are applicable.
This grandfathering does not authorize a new persisted writer or a new drain requirement under an
existing item.

For qexp persisted fields, indexes, capabilities, and runtime protocols first introduced after
2026-09-08, the normal path must be a machine-by-machine rolling upgrade. On each machine the
operator installs the new package and restarts the global agent; the new agent discovers every
registered project on that machine and advances required migrations automatically. The normal path
must not require per-project `cd`, `doctor`, activation IDs, attestations, or manual journal loops.

qexp migration must preserve already running training processes. Agent restart may temporarily
pause queue admission and scheduling, but it must not terminate training, wait for all training to
finish, discard machine-local execution evidence, or require idle GPUs. A release that cannot meet
this contract may not introduce the persisted change as ordinary qexp compatibility work.

When protocol-specific evidence shows that L1 rolling activation is not feasible, an owner-approved
registry change may select L2. The implementation must still preserve running training processes and
permit later agents to reconcile durable terminal evidence. Verification must cover the
machine-level all-project workflow and cannot fall back to manual per-project maintenance. Machine
retirement must use explicit identity or a durable retirement tombstone; heartbeat age alone cannot
authorize metadata deletion.

If the permanent migration infrastructure needed by these rules does not yet exist, register the
feature as `planned` only when implementation intent is public and deliver that infrastructure
before introducing its new persisted writer. Do not ship a drain-required schema first and defer
operability to a later patch.

### Continuity and rollout completion

Operator coordination level and service continuity are separate contracts. Verification must cover
admission/scheduling interruption scope, terminal reconciliation behavior, and the recorded
interruption budget. Two commands alone do not prove L1: waiting for the fleet to upgrade or a full
backfill to finish must not suspend compatible scheduling.

Within the declared supported version range, operators may upgrade machines in any order and pause
rollout between machines. Existing functionality remains available through compatible paths.
Distinguish package/agent deployment, per-project feature activation, and legacy cleanup in status
and acceptance criteria. Successful restart does not imply that background migration is complete.
Active legacy Attempts must not prevent deployment completion or restored agent service; their
evidence adapters remain available until safe terminal reconciliation, even if record conversion is
deferred. Cleanup must not turn deferred work into a workload-drain prerequisite.

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

Each persisted-protocol item with an operational contract declares supported source package/data
versions, mixed-version ranges, and the direct upgrade path to its target. Removing an old live
writer and removing the ability to migrate historical data require separate evidence. A fixed
release count does not prove that inactive projects or long-running Attempts no longer need an
adapter.

The two-command promise applies to explicitly supported source versions. Supported sources must not
require manual intermediate package installation or downgrade. Unsupported sources must be
identified before destructive migration, with an actionable machine-level explanation. Automatic
chained migration may satisfy a supported path. Required adapters must be permanent contracts or
have their registered deadlines explicitly extended before removal; deadlines must not silently
override continuity requirements.

The registry validator validates structured rollout, continuity, interruption-budget, lifecycle,
and evidence-path fields. It does not prove runtime guarantees; those require protocol-specific
tests and released-source evidence.

### Migration development and runtime cost

For qexp persisted-protocol changes, implementation and verification evidence must also establish:

- **Minimum necessary transition:** determine whether conversion or a required capability is needed.
  Additive defaults and rebuildable projections must be checked against actual old readers and
  whole-record writers. Apply phase-specific safety predicates rather than a mandatory fleet barrier
  for every change.
- **Protocol-boundary ownership:** keep version interpretation, write-path selection, and historical
  Attempt adapters behind stable interfaces. A generic coordinator is not proof of online safety.
  Declare dependencies and conflicting mutations when upgrades can chain.
- **Concurrent-write correctness:** prove that backfill preserves updates made before and after
  scanning a record. A watermark alone is insufficient. Identify the boundary honored by source
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

Private design records may document detailed locking, cache invalidation, rejected alternatives, or
implementation reasoning. Any constraint required for public correctness must still be enforced by
registry data, source behavior, tests, or stable public product documentation rather than by access
to those private records.

## Deadline extensions

An extension names `legacy_removed_in` or `transition_purged_in`, chains from the previous effective
value to a later version, records the approval version, and records a non-empty reason. Original
deadlines are not overwritten.

Extensions do not carry a document reference. `approved_in`, `reason`, the compatibility
ID, and Git history provide the public audit trail. A private planning or design record may contain
additional rationale but is not required by validation.

## Commands

```bash
python scripts/checks/check_compatibility_registry.py validate
python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z
python scripts/checks/check_compatibility_registry.py check --release-version X.Y.Z
```

`validate` checks registry structure and repository evidence. `plan` reports the state or registry
removal required for items that still exist at a target release. `check` fails when a currently
registered item is in the wrong lifecycle state for that target or remains registered past its purge
version. It intentionally does not reconstruct or validate already-cleaned historical items.

The release operator must inspect `plan`, resolve every due action, commit the candidate, and run:

```bash
python scripts/release_preflight.py --target-version X.Y.Z
```

Release validation must succeed from a clean committed checkout without private pitches, ADRs, local
notes, or other ignored planning files.

## Public versus private documentation

No dedicated public ADR or pitch is required by compatibility governance. The owner may keep design
history private. Public documentation is required only when users or contributors need it to use,
migrate, operate, or verify the product safely.

The compatibility registry is intentionally not a design-history database. It records unfinished
lifecycle obligations and executable public evidence; private reasoning stays outside that control
plane.
