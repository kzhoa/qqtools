# qexp Runtime Spec

Status: Draft

Updated: 2026-07-01

## Purpose

This document describes the runtime contract of `qexp`.

It answers these questions:

1. how the shared root is organized
2. how a machine is registered
3. how runtime truth is stored
4. how the local agent behaves in `on_demand` and `persistent` modes

Product positioning and CLI intent belong in
[qexp_product_spec.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md).

## Scope Notes

This document mixes:

- the current runtime contract
- accepted target behavior that may not be fully implemented yet

Anything marked as **Assumption / Unverified** is a design direction, not a
guarantee of the current installed version.

## Runtime Overview

`qexp` has two roots:

- a shared control root for project-wide truth
- a local runtime root for machine-local process state

### Shared control root

Official shape:

```text
<project_root>/.qexp
```

Rules:

- it must be project-level
- it must be named `.qexp`
- it must sit directly under the project root

Not supported as truth layout:

```text
<project_root>/.qexp/exp1/shared
<project_root>/.qexp/exp2/shared
<project_root>/.qexp/groups/<group>/...
```

Reason:

- one project should have one control plane
- tasks, batches, machines, indexes, and events should share one truth model

### Local runtime root

Recommended:

```text
~/.qqtools/qexp-runtime
```

It stores machine-local runtime state such as:

- agent pid
- local locks
- heartbeats
- wrappers
- temporary runtime files

## Truth Model

The shared root has a small set of truth domains.

Primary truth objects:

- `global/tasks/<task_id>.json`
- `global/batches/<batch_id>.json`
- `machines/<machine_name>/machine.json`
- `machines/<machine_name>/state/agent.json`
- `machines/<machine_name>/claims/active/<task_id>.json`

Derived or auxiliary objects:

- `global/indexes/`
- `global/events/`
- `machines/<machine_name>/state/gpu.json`
- `machines/<machine_name>/state/summary.json`
- `machines/<machine_name>/claims/released/`
- `machines/<machine_name>/events/`

Important rule:

- indexes and summaries are never the source of truth
- if a derived view conflicts with a truth object, trust the truth object

## Shared Directory Layout

```text
<shared-root>/
  global/
    schema/
      version.json
    tasks/
      <task_id>.json
    batches/
      <batch_id>.json
    indexes/
      tasks_by_state/
    locks/
      submit/
      batch/
      migrate/
    events/
      <date>/
        <event_id>.json
  machines/
    <machine_name>/
      machine.json
      claims/
      events/
      state/
```

Layout rules:

- organize truth by object type
- keep one private subdirectory per machine
- do not create truth directories per experiment or per group

## Recovery Priority

When runtime state is inconsistent, rebuild in this order:

1. trust task truth first
2. then trust batch truth
3. then trust agent lifecycle truth and active claim truth
4. rebuild indexes and summaries last

Extra rule for single-task clean:

- if a task truth file has been deleted, repair must not recreate it out of thin air

## Machine Model

Each machine gets its own shared subdirectory:

```text
<shared-root>/machines/<machine_name>/
  machine.json
  state/
    agent.json
    gpu.json
    summary.json
  claims/
    active/
      <task_id>.json
    released/
      <task_id>.json
  events/
    <date>/
      <event_id>.json
```

This keeps high-frequency writes local to one machine and reduces write
contention.

### Machine identity

In shared mode, machine identity must be explicit:

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

Rules:

- `--machine` is required in shared mode
- hostname may be stored, but it is not the primary key
- `machine_name` is the real identity

### Machine write rules

To avoid cross-machine corruption:

- a machine may write only to `machines/<its_name>/...`
- a machine must not write into another machine's private directory

## Machine-side Files

### `machine.json`

Static machine declaration.

At minimum it should contain:

- `machine_name`
- `hostname`
- `shared_root`
- `runtime_root`
- `agent_mode`
- machine tags or GPU inventory

### `state/agent.json`

This is the single source of truth for agent lifecycle state.

It should describe:

- current mode: `on_demand` or `persistent`
- current state: `stopped`, `starting`, `active`, `draining`, `idle`, `stale`, or `failed`
- pid
- heartbeat timestamps
- last transition time
- idle timeout information
- a small workset summary

It exists so other commands can answer:

- is the agent alive?
- is it active, draining, or idle?
- did it exit normally or unexpectedly?

### `state/gpu.json`

Current GPU snapshot for one machine.

Typical contents:

- visible GPU ids
- reserved GPU ids
- task-to-GPU assignments
- update time

This is mainly for display and debugging.

### `state/summary.json`

Lightweight aggregated machine summary.

Typical contents:

- `machine_name`
- counts by task phase
- `updated_at`

Important rule:

- this file is a cache for display
- it must not replace `state/agent.json` for lifecycle decisions

### `claims/active/<task_id>.json`

This says the machine currently owns execution responsibility for a task.

Typical contents:

- `task_id`
- `machine_name`
- `claimed_at`
- revision at claim time

### `claims/released/<task_id>.json`

This is lightweight audit history for a finished claim.

Typical contents:

- `task_id`
- `released_at`
- `release_reason`

### `events/<date>/<event_id>.json`

Machine-level scheduler events such as:

- `agent_started`
- `agent_stopped`
- `task_claimed`
- `task_started`
- `task_finished`
- `task_failed`
- `task_cancelled`
- `task_orphaned`

This is scheduler audit, not training logging.

## Shared Object Contracts

### Task truth

Task truth lives in:

```text
global/tasks/<task_id>.json
```

Suggested structure:

```yaml
meta:
  revision: int
  created_at: str
  updated_at: str
  updated_by_machine: str

task:
  task_id: str
  name: str | null
  group: str | null
  batch_id: str | null
  machine_name: str
  attempt: int
  spec:
    command: list[str]
    requested_gpus: int
  status:
    phase: enum[queued, dispatching, starting, running, succeeded, failed, cancelled, blocked, orphaned]
    reason: str | null
    category: str | null
  runtime:
    assigned_gpus: list[int]
    process_group_id: int | null
    wrapper_pid: int | null
  timestamps:
    created_at: str
    queued_at: str | null
    started_at: str | null
    finished_at: str | null
  result:
    exit_code: int | null
    terminal_reason: str | null
  lineage:
    retry_of: str | null
```

Rules:

- `meta.revision` must increase on every successful write
- `task.task_id` must match the filename
- `task.machine_name` is stable once written
- `task.status.phase` is the only official task state
- indexes must not override task truth
- `lineage.retry_of` is only for retry-created tasks

For `group`:

- it is a product-level grouping key
- it is not a control-plane boundary
- it must not imply a separate shared root
- it may be `null`

### Batch truth

Batch truth lives in:

```text
global/batches/<batch_id>.json
```

Suggested structure:

```yaml
meta:
  revision: int
  created_at: str
  updated_at: str
  updated_by_machine: str

batch:
  batch_id: str
  name: str | null
  group: str | null
  source_manifest: str | null
  machine_name: str
  task_ids: list[str]
  summary:
    total: int
    queued: int
    running: int
    succeeded: int
    failed: int
    cancelled: int
    blocked: int
    orphaned: int
  policy:
    allow_retry_failed: bool
    allow_retry_cancelled: bool
```

Rules:

- `meta.revision` must increase on every successful write
- `batch.batch_id` must match the filename
- `batch.task_ids` is the official member list
- batch summaries are rebuildable from task truth

Meaning:

- a batch is an organization container
- a batch is not the execution truth
- task truth is always more important than batch summary

## Agent Modes

### `on_demand`

Default mode:

- the agent may auto-start when work appears
- the agent exits after a true idle timeout

### `persistent`

Persistent mode is explicit:

- the agent stays up until the user stops it or an error happens

## Concurrency Principles

The runtime must be safe under concurrent writers.

Practical rules:

- treat task truth and batch truth as the authoritative records
- keep machine-local writes inside the machine's own directory
- treat indexes and summaries as rebuildable caches
- prefer repair and rebuild over inventing new truth

## Assumption / Unverified

The following are accepted design directions but may not be fully implemented in
the current version:

- `group` as a stable field across CLI and manifests
- derived group summaries such as `derived/groups/<group>/summary.json`

If such derived artifacts exist in the future, they must satisfy:

- deletable without data loss
- fully rebuildable
- never treated as truth
