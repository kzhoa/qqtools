# qexp Product Spec

Status: Draft

Updated: 2026-07-01

## Purpose

This document explains the product side of `qexp`:

1. What `qexp` is
2. How users are expected to use it
3. What `qexp` owns
4. What `qexp` does not own

Implementation details such as shared directory layout, locks, and repair flows
belong in
[qexp_runtime_spec.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md).

## Product Summary

`qexp` is a lightweight experiment submission queue for the current machine.

It is responsible for:

- submitting one task
- submitting a batch of tasks
- recording scheduling events
- running a lightweight local agent
- showing queue and machine state
- cancelling tasks
- retrying tasks
- resubmitting terminal tasks in place

It is not responsible for:

- training log management
- metric logging formats
- artifact management
- experiment result archiving
- cross-machine remote submission
- a permanent background service by default

## Core Mental Model

`qexp` uses four main terms:

- `machine`: a machine or container that runs tasks
- `group`: a long-lived project-level label for related tasks
- `task`: one submitted execution unit
- `batch`: a set of tasks submitted together

`job` is not a first-class concept.

### Group vs Batch

These two concepts must stay separate:

- `group` answers: "Which long-term work context does this task belong to?"
- `batch` answers: "Which tasks were submitted together this time?"

Rules:

- one group can contain many batches
- a batch must not replace a group
- users do not need a separate "experiment plan" object if a stable group name is enough

## Product Boundaries

### Current-machine first

By default, `qexp` submits tasks to the current machine only.

It does not promise:

- submit on machine A and run on machine B
- remotely wake an agent on another machine

Multiple machines may share metadata and queue state, but not remote control.

### One shared root per project

The official shared control directory is:

```text
<project_root>/.qexp
```

This is a project-level control plane, not an experiment-level folder.

Recommended:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu2a
```

Not recommended:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp/exp1/shared --machine gpu2a
qexp init --shared-root /mnt/share/myproject/.qexp/exp2/shared --machine gpu2a
```

Reason:

- tasks, batches, machines, indexes, and events should share one source of truth
- splitting by experiment would fragment the queue view

### Shared mode requires an explicit machine name

In shared mode, `--machine` is required.

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

Hostname may be stored as extra information, but it must not be the primary machine
identity.

### Agent runs on demand by default

Default behavior:

- `agent_mode = on_demand`
- the agent may auto-start when work appears
- the agent exits after 10 minutes of true idleness

Persistent mode is opt-in:

```bash
qexp init --shared-root /path/to/shared --machine gpu2a --agent-mode persistent
```

or:

```bash
qexp agent start --persistent
```

### Logging stops at scheduling events

`qexp` records scheduler-level facts only:

- whether submission succeeded
- when a task started
- when a task finished
- whether a task failed
- the failure category

Training logs remain the responsibility of the training stack.

## Main Commands

### `submit`

This is the most common entry point and must stay lightweight.

```bash
qexp submit -- python train.py --config configs/a.yaml
```

or:

```bash
qexp submit --task-id qm9_seed_1 --name "qm9 seed 1" -- python train.py --config configs/a.yaml
```

If the user already has a stable experiment context, `group` should represent it:

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

Rules:

- a single task submission must not require YAML
- a single task submission must not require a batch first
- `submit` only creates a new task
- if `task_id` already exists, `submit` must fail by default

### `batch-submit`

Use a manifest only when submitting a batch:

```bash
qexp batch-submit --file runs.yaml
```

Example:

```yaml
batch:
  name: contract-compare-round1
  group: contract_n_4and6

tasks:
  - name: n4
    command: ["python", "train.py", "--n", "4"]
  - name: n6
    command: ["python", "train.py", "--n", "6"]
```

Meaning:

- `batch.name` identifies this submission event
- `batch.group` identifies the long-term work context

Rules:

- the manifest is for batch submission only
- batch submission must not make single-task submission more complex
- by default, manifest tasks belong to the current machine

### `retry`, `resubmit`, and `clean`

Examples:

```bash
qexp cancel task_xxx
qexp retry task_xxx
qexp resubmit task_xxx -- python train.py --config configs/a.yaml
qexp batch-retry-failed batch_xxx
qexp batch-retry-cancelled batch_xxx
qexp clean
qexp clean --include-failed
qexp clean --older-than-seconds 259200
qexp clean --task-id task_xxx
qexp clean --task-id task_xxx --dry-run
```

Required behavior:

- the user should not need to rewrite the original command
- the user should not need to find the old YAML again
- the user should not need to clean first before retrying

Command meanings:

- `submit`: create a brand-new task
- `retry`: keep the old record and create a new task
- `resubmit`: delete one terminal task record and submit again with the same `task_id`

Rules:

- `submit` must not silently overwrite old tasks
- `submit` must not silently behave like `retry`
- `submit` must not silently behave like `resubmit`
- `retry` must create a new `task_id`
- `retry` must keep lineage
- `resubmit` must reuse the same `task_id`
- `resubmit` must not keep the old record visible

`resubmit` is stricter:

- it is only for terminal tasks
- by default it only allows `failed` and `cancelled`
- it must not be allowed for active states
- by default it must not be allowed for batch member tasks

`clean` must support both:

- bulk cleanup of old terminal tasks
- exact cleanup of one terminal task by `task_id`

### Observation commands

Everyday observation should stay flat and easy:

```bash
qexp list
qexp inspect task_xxx
qexp top
qexp top --all
qexp batches
qexp batch batch_xxx
qexp machines
```

## Expected User Flow

### First-time setup

Each machine must initialize itself:

```bash
qexp init --shared-root /path/to/myproject/.qexp --machine gpu2a
```

If the user wants a persistent agent:

```bash
qexp init --shared-root /path/to/myproject/.qexp --machine gpu2a --agent-mode persistent
```

### Daily flow

Recommended flow:

1. `qexp init`
2. `qexp submit` or `qexp batch-submit`
3. `qexp list`, `qexp inspect`, or `qexp top`
4. use `cancel`, `retry`, `resubmit`, or `clean` when needed

## CLI Surface

### Common commands

- `qexp init`
- `qexp submit`
- `qexp batch-submit`
- `qexp list`
- `qexp inspect`
- `qexp top`
- `qexp cancel`
- `qexp retry`
- `qexp resubmit`
- `qexp clean`
- `qexp batches`
- `qexp batch`
- `qexp machines`

### Low-frequency commands

- `qexp agent start`
- `qexp agent stop`
- `qexp agent status`
- `qexp doctor`

## Help Text Draft

```text
usage: qexp <command> [options]

Lightweight experiment submission queue for the current machine.

By default, qexp submits tasks to the current machine only.
It uses a shared control root for metadata and an on-demand local agent.

Common commands:
  init            Initialize qexp on the current machine
  submit          Submit one task on the current machine
  batch-submit    Submit a batch manifest on the current machine
  list            List tasks
  inspect         Show one task
  top             Show live queue and machine overview
  cancel          Cancel one task
  retry           Retry one task
  resubmit        Replace one terminal task with the same task_id
  clean           Clean terminal task records
  batches         List batches
  batch           Show one batch
  machines        List visible machines

Low-frequency commands:
  agent           Manage the local agent
  doctor          Diagnose or repair qexp metadata
```

## Non-goals for the Main Help Page

These commands should not appear on the first help page:

- `qexp task ...`
- `qexp clone`
- `qexp batch-export`
- `qexp machine inspect`
- `qexp agent wake`

## Acceptance Checklist

- Shared mode must require an explicit `--machine`.
- The recommended shared root must be project-level, not experiment-level.
- A single task must be submittable with `qexp submit -- ...`.
- `group` and `batch` must stay separate in meaning.
- Cross-machine remote submission is not a default feature.
- Persistent background execution is opt-in, not default.
- `qexp` records scheduling events, not full training logs.
- `clean` must support both bulk cleanup and exact single-task cleanup.
