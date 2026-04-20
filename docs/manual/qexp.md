# qexp Manual

Status: draft

Updated: 2026-04-20

## What This Manual Covers

This manual is for people who use `qexp` directly.

It focuses on the operational questions that matter in day-to-day use:

1. What `qexp` is and how it behaves by default
2. What you need before the first machine joins
3. How to submit single tasks and batch manifests
4. How to observe, cancel, retry, resubmit, and clean tasks
5. What to check first when something looks wrong

It is intentionally practical: one document for the product model, the common
command paths, and the first-line recovery rules. You should not need runtime
specs unless you are working on the internals.

## What qexp Is

`qexp` is a lightweight experiment submission queue built around a project-scoped
control plane.

At a glance:

- metadata lives in one shared project root
- machines join that shared control plane explicitly
- the CLI always operates in the context of one current machine
- the agent is on-demand by default

`shared_root` is required qexp context. That does not mean every command must
spell out `--shared-root` explicitly: after `qexp init`, saved context or
environment variables can provide it. The recommended and supported location is
`project_root/.qexp`.

`qexp` is not a full experiment platform. It is a shared queue and execution shell
for submitting tasks, observing them, and recovering from common queue-state issues.

## Core Mental Model

If users get confused, it is usually because they blur `task`, `group`, and `batch`.
Do not blur them.

- `shared root`: the shared `qexp` control directory for one project; it stores tasks,
  batches, indexes, machine metadata, and events
- `machine`: one machine or container instance that can execute tasks
- `runtime root`: machine-local writable state such as agent pid files and logs
- `task`: one concrete submission and execution unit
- `group`: a long-lived grouping key inside the project; at runtime it maps directly
  to one tmux session
- `batch`: one bulk submit operation that creates a set of tasks together
- `name`: a human-facing label for one task
- `task_id`: the unique identifier for one task

Keep these responsibilities stable:

- `task` answers: what exactly is being run?
- `group` answers: which long-lived working context does this task belong to?
- `batch` answers: which tasks were created together in one bulk submit operation?
- `name` answers: what should users see in listings?
- `task_id` answers: which exact task object do commands target?

Important non-equivalences:

- `group` is not `batch`
- `group` is not `name`
- `task_id` is not `group`
- one `group` can contain many `batch` objects
- one experiment plan can map to one stable `group` across multiple submits or batches

### About `group`

`group` is the most commonly misunderstood field. Read it literally:

- `group` is a long-lived grouping key
- runtime projects it directly to a tmux session
- `group` is not "the batch I submitted today"
- `group` is not a scientific truth object owned by your research workflow
- `group` is the tool-layer key you use to keep related runs in one durable working context

Typical mappings:

- one experiment plan -> one stable `group`
- one debugging campaign -> one stable `group`
- one long-running topic of work -> one stable `group`

If two tasks share the same `group`, they belong to the same long-lived working
context even if they were submitted on different days through different `batch`
objects.

### Task Lifecycle

Typical phase progression:

`queued -> dispatching -> starting -> running -> succeeded / failed / cancelled`

Meaning:

- `queued`: the task exists in the queue and is waiting for an agent to claim it
- `dispatching`: an agent has taken responsibility and is assigning execution resources
- `starting`: the wrapper is preparing to launch the user command
- `running`: the user command has started
- `succeeded` / `failed` / `cancelled`: terminal phases

## Environment Requirements

### Basic Requirements

- Python with `qqtools` installed
- one shared root visible to all relevant machines
- one writable local runtime directory on the current machine

### Requirements for Automatic Execution

If you expect submitted tasks to start automatically, the machine also needs:

- Linux
- `tmux`
- `libtmux`
- a usable GPU detection backend such as `pynvml` or `nvidia-smi`

If these dependencies are missing, tasks may still be written to the queue, but
they will not automatically start running.

## Quick Start

### 1. Initialize One Machine

Each machine must initialize itself once before joining the shared queue:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

This is not an arbitrary directory. The supported shape is:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

Do not treat `shared_root` as a per-experiment directory:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp/exp1/shared --machine gpu-a
qexp init --shared-root /mnt/share/myproject/.qexp/exp2/shared --machine gpu-a
```

Why this is wrong:

- `shared_root` is one project-level control plane
- tasks, batches, machines, indexes, and events are meant to share one truth set
- splitting roots by experiment fragments the queue view and the resource view
- current runtime validation rejects roots that do not follow the `project_root/.qexp` contract

If this machine should run a persistent agent:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a --agent-mode persistent
```

Default mode is `on_demand`. The agent starts when needed and exits after true
idleness. `persistent` is better for dedicated machines that should keep an agent
alive continuously.

`qexp init` also saves the current CLI context:

- `shared_root`
- `machine`
- explicitly provided `runtime_root`

So after initialization, these commands usually work without repeating those flags:

```bash
qexp list
qexp top
qexp logs <task_id>
```

You can still switch or inspect the saved context explicitly:

```bash
qexp use --shared-root /mnt/share/myproject/.qexp --machine gpu-a
qexp use --show
```

Flags still override saved context when needed:

```bash
qexp --shared-root /mnt/share/myproject/.qexp --machine gpu-a list
```

Environment variables also work. Precedence is:

1. CLI flags
2. environment variables
3. saved context

Example:

```bash
export QEXP_SHARED_ROOT=/mnt/share/myproject/.qexp
export QEXP_MACHINE=gpu-a
qexp list
```

If the machine-local runtime directory must be customized:

```bash
qexp init \
  --shared-root /mnt/share/myproject/.qexp \
  --machine gpu-a \
  --runtime-root /data/local/qexp-runtime
```

Re-running `init` is safe. It refreshes the layout, machine registration, and saved
context.

### 2. Submit the First Task

```bash
qexp submit -- python train.py --config configs/a.yaml
```

If you already know that multiple runs belong to the same long-lived working context,
declare `group` early:

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

Interpret those commands this way:

- they create two different tasks
- they share one long-lived `group`
- runtime places them into the same tmux session: `contract_n_4and6`
- that does not imply they belong to the same `batch`

### 3. Observe the Queue

```bash
qexp list
qexp top
qexp logs <task_id>
```

## Submitting Tasks

### Single-Task Submit

This is still the default path:

```bash
qexp submit -- python train.py --config configs/a.yaml
```

With an explicit task id and display name:

```bash
qexp submit \
  --task-id qm9_seed_1 \
  --name "qm9 seed 1" \
  --gpus 1 \
  -- python train.py --config configs/a.yaml
```

Notes:

- everything after `--` is passed through as the user command
- `--gpus` is the requested GPU count
- `--task-id` is optional; if omitted, `qexp` generates one

If several tasks belong in the same long-lived working context, declare `group`
explicitly:

```bash
qexp submit \
  --group contract_n_4and6 \
  --name n4 \
  --gpus 1 \
  -- python train.py --n 4
```

Interpretation:

- `group` is not a formal business term like "experiment plan"
- but you can map an experiment plan, debugging campaign, or durable work topic to one stable `group`
- tasks in the same `group` are treated as long-term related work
- runtime maps that `group` directly to one tmux session
- `group` is not the "submitted together today" concept; that is `batch`

`group` constraints:

- allowed characters: letters, digits, `.`, `_`, `-`
- case-sensitive
- reserved names are forbidden: `experiments`, `qqtools_internal`
- must not start with `.` or `-`
- maximum length: `64`
- after validation, the value maps directly to the tmux session name with no second sanitization pass

Recommended naming:

- one experiment plan -> one stable `group`
- one specific run -> one `name`
- one bulk submission -> one `batch`

Example:

- `group=contract_n_4and6`
- `name=n4`
- `name=n6`

### Batch Submit

Use a manifest when you want to submit a set of tasks together:

```bash
qexp batch-submit --file runs.yaml
```

Minimal manifest:

```yaml
batch:
  name: sweep-a
  group: contract_n_4and6

tasks:
  - command: ["python", "train.py", "--config", "configs/a.yaml"]
  - command: ["python", "train.py", "--config", "configs/b.yaml"]
```

More complete manifest:

```yaml
batch:
  name: sweep-a
  group: contract_n_4and6
  policy:
    allow_retry_failed: true
    allow_retry_cancelled: false

defaults:
  requested_gpus: 1

tasks:
  - task_id: qm9_seed_1
    name: qm9 seed 1
    group: contract_n_4and6
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "1"]

  - task_id: qm9_seed_2
    name: qm9 seed 2
    group: regrouped_debug
    requested_gpus: 2
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "2"]
```

Field semantics:

- `batch.name`: display name for this bulk submit operation
- `batch.group`: default long-lived `group` for tasks in this batch
- `batch.policy.allow_retry_failed`: whether `qexp batch-retry-failed` is allowed
- `batch.policy.allow_retry_cancelled`: whether `qexp batch-retry-cancelled` is allowed
- `defaults.requested_gpus`: default GPU count for tasks in this batch
- `tasks[].task_id`: optional; autogenerated if omitted
- `tasks[].name`: optional display label
- `tasks[].group`: optional task-level group override; higher priority than `batch.group`
- `tasks[].requested_gpus`: task-level GPU override
- `tasks[].command`: required user command

Grouping precedence:

1. `tasks[].group` has the highest priority
2. if `tasks[].group` is missing, inherit `batch.group`
3. if both are missing, task `group = null`

Interpretation:

- `batch` means "submitted together in this one bulk operation"
- `group` means "belongs to this long-lived working context"

A common pattern:

- submit batch one today
- submit batch two tomorrow
- they are different `batch` objects
- they can still reuse the same `group`

If two different bulk submissions belong to the same experiment plan, use:

- different `batch` values
- the same stable `group`

## Observe and Manage Tasks

### List Tasks

```bash
qexp list
qexp list --phase queued
qexp list --batch <batch_id>
```

### Inspect One Task

```bash
qexp inspect <task_id>
```

### Live Overview

Current machine only:

```bash
qexp top
```

All visible machines:

```bash
qexp top --all
```

### List Machines

```bash
qexp machines
```

### List and Inspect Batches

```bash
qexp batches
qexp batch <batch_id>
```

### Read Logs

Read existing log output:

```bash
qexp logs <task_id>
```

Follow the log continuously:

```bash
qexp logs -f <task_id>
```

### Cancel, Retry, and Resubmit

Cancel a task:

```bash
qexp cancel <task_id>
```

Retry one terminal task:

```bash
qexp retry <task_id>
```

Retry into another `group` explicitly:

```bash
qexp retry <task_id> --group regrouped_debug
```

Retry rules:

- `qexp retry <task_id>` inherits the original task `group` by default
- `qexp retry <task_id> --group <group>` overrides it explicitly
- retry creates a new task; it does not rewrite the old task's `group`

Replace one terminal task in place:

```bash
qexp resubmit <task_id> -- python train.py --config configs/a.yaml
```

Override the new display name and group:

```bash
qexp resubmit <task_id> --name rerun_a --group regrouped_debug -- python train.py --config configs/a.yaml
```

Resubmit rules:

- `resubmit` is only allowed for `failed` or `cancelled`
- `resubmit` is not allowed for batch-member tasks
- `resubmit` deletes the old formal task record and recreates a fresh first-attempt truth with the same `task_id`
- if replacement fails mid-flight and leaves an unfinished operation, run `qexp doctor repair`
- when the old task truth is gone but replacement has not converged yet, `qexp inspect <task_id>` shows the unfinished resubmit operation state

Retry failed tasks in a batch:

```bash
qexp batch-retry-failed <batch_id>
```

Retry cancelled tasks in a batch:

```bash
qexp batch-retry-cancelled <batch_id>
```

### Clean Old Records

Preview cleanup:

```bash
qexp clean --dry-run
```

Default cleanup behavior removes succeeded tasks older than 7 days:

```bash
qexp clean
```

Set an explicit age threshold:

```bash
qexp clean --older-than-seconds 259200
```

Include failed and cancelled terminal tasks:

```bash
qexp clean --include-failed
```

Clean one specific terminal task:

```bash
qexp clean --task-id <task_id>
```

Preview one task cleanup:

```bash
qexp clean --task-id <task_id> --dry-run
```

Cleanup notes:

- `qexp clean` is the bulk cleanup mode
- `qexp clean --task-id <task_id>` is precise single-task cleanup mode
- single-task mode cannot be combined with `--older-than-seconds` or `--include-failed`
- single-task cleanup is only allowed for terminal tasks: `succeeded`, `failed`, `cancelled`
- if the task belongs to a batch, cleanup also repairs batch membership and summary fields
- runtime log deletion is best-effort; if a log cannot be deleted, the CLI reports that explicitly

## Agent Lifecycle

### Start the Agent

Foreground:

```bash
qexp agent start
```

Persistent mode:

```bash
qexp agent start --persistent
```

Background:

```bash
qexp agent start --background
```

### Stop the Agent

```bash
qexp agent stop
```

### Inspect Agent State

```bash
qexp agent status
```

`qexp agent status` is the authoritative entrypoint for interpreting current agent
lifecycle state.

State meanings:

- `active`: this machine still owns `queued`, `dispatching`, or `starting` responsibilities
- `draining`: this machine has no remaining launch backlog but still owns `running` responsibility
- `idle`: this machine has no remaining active responsibility and is waiting for idle timeout
- `stopped` / `stale` / `failed`: respectively not running, heartbeat lost, or exited abnormally

An `on_demand` agent still exits automatically, but not just because it has not
launched a task recently. The exit path is:

1. this machine no longer owns `queued`, `dispatching`, `starting`, or `running` responsibility
2. the agent enters `idle`
3. `idle` lasts until `idle_timeout`
4. then the process exits automatically

## Repair and Recovery

Use doctor commands when you need to inspect or repair shared metadata:

```bash
qexp doctor verify
qexp doctor rebuild-index
qexp doctor repair
qexp doctor repair-orphans
qexp doctor cleanup-locks
```

Subcommand meanings:

- `qexp doctor verify`: read-only integrity check; does not modify files; mainly verifies readability, file-name to task-id consistency, and revision validity
- `qexp doctor rebuild-index`: rebuild indexes when index views disagree with formal task truth
- `qexp doctor repair`: converge unfinished metadata repair operations; currently continues interrupted `resubmit` replacements and repairs leftover batch summary corrections from single-task clean
- `qexp doctor repair-orphans`: moves tasks that lost machine heartbeat for too long while still appearing active into `orphaned`
- `qexp doctor cleanup-locks`: removes stale lock files after abnormal exits

How this relates to `clean`:

- if single-task clean or `resubmit` failed midway, run `qexp doctor repair` first
- if the main suspicion is index drift, run `qexp doctor rebuild-index`
- `cleanup-locks` and `repair-orphans` do not complete unfinished `resubmit`, batch repair, or index rebuild work

Recommended order:

1. run `qexp doctor verify`
2. if there is an interrupted `resubmit` or single-task clean, run `qexp doctor repair`
3. if index inconsistency is still suspected, run `qexp doctor rebuild-index`
4. if tasks look abandoned or mis-owned, consider `repair-orphans` and `cleanup-locks`

## Common Workflows

### Daily Single-Machine Use

```bash
export QEXP_SHARED_ROOT=/mnt/share/my_qexp
export QEXP_MACHINE=gpu-a

qexp submit -- python train.py --config configs/a.yaml
qexp list
qexp top
qexp logs <task_id>
```

### Manage Tasks by Experiment Plan Inside One Project

Assume the project directory is:

```text
/mnt/share/myproject/
```

And the current plan is to compare `n=4` and `n=6`.

Do not create a separate `shared_root` for that plan. Instead:

1. keep one project-wide `shared_root`
2. use one stable `group` for this experiment plan
3. submit each concrete run as its own task

Recommended initialization:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

Recommended single-task submit path:

```bash
qexp submit --group contract_n_4and6 --name n4 --gpus 1 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 --gpus 1 -- python train.py --n 6
```

If you want to submit a set together:

```yaml
batch:
  name: contract-compare-round1
  group: contract_n_4and6

tasks:
  - name: n4
    group: contract_n_4and6
    command: ["python", "train.py", "--n", "4"]
  - name: n6
    group: regrouped_debug
    command: ["python", "train.py", "--n", "6"]
```

Then:

```bash
qexp batch-submit --file runs.yaml
```

If you later submit new tasks for the same experiment plan, keep reusing:

- the same stable `group`

Do not solve that by creating:

- a new `shared_root`

Recommended mental model:

- use `group` for the experiment-plan or work-context identity
- use `name` to distinguish one concrete run from another
- use `batch` to represent one bulk submit operation
- use `task_id` as the unique key for one task
- keep one project-level `shared_root` so queue state and resource visibility stay coherent

### Multiple Machines Sharing One Queue

Machine A:

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-a
```

Machine B:

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-b
```

Both machines now point at the same `shared_root`, but each machine still uses:

- its own `machine` identity
- its own local runtime directory

Where a task actually runs depends on which machine agent claims and launches it.

## Troubleshooting

### `submit` succeeded, but the task stays in `queued`

Check these first:

- is the current machine's agent actually running: `qexp agent status`
- is the machine stuck in `active` or `draining`, and does the workset still show backlog
- is `tmux` installed
- is `libtmux` installed
- can the current machine detect GPUs

If automatic wake-up failed, start the agent manually:

```bash
qexp agent start
```

### `qexp` says `--shared-root` or `--machine` is missing

The current command has no machine context.

Fix options:

- pass `--shared-root` and `--machine` explicitly
- or set `QEXP_SHARED_ROOT` and `QEXP_MACHINE`
- or save context with `qexp use`

### Can I create one `shared_root` per experiment plan?

Not recommended.

Recommended model:

- one project-level `shared_root`
- different experiment plans separated by `group`

Problems with one root per experiment plan:

- the resource view becomes fragmented inside one project
- `top`, `machines`, and `list` only see partial truth
- after switching with `qexp use`, the new root does not automatically know about tasks occupying resources in the old root

Only create another `shared_root` when you explicitly want another independent
control plane.

### I used the old single-machine qexp before

The old single-machine interface is gone.

The supported model now is:

- project-root `.qexp` control directory
- explicit `shared_root`
- explicit `machine` context

### `logs` fails or task execution says a local path is not writable

That usually means the current machine's runtime root is not writable.

Preferred fixes:

- provide a writable directory for the current user
- or pass `--runtime-root`
- or set `QEXP_RUNTIME_ROOT`

Example:

```bash
export QEXP_RUNTIME_ROOT=/data/local/qexp-runtime
```

## CLI Argument Reference

### Global Arguments

These global arguments are supported:

- `--shared-root <path>`: shared control directory; all machines see the same tasks and indexes there
- `--machine <name>`: current machine identity; must be unique inside one shared root
- `--runtime-root <path>`: machine-local runtime directory for logs, pid files, and other local state; defaults to the standard local path if omitted

These arguments usually appear before the subcommand:

```bash
qexp --shared-root /mnt/share/my_qexp --machine gpu-a list
```

### `submit` Arguments

- `--task-id <id>`: optional custom task id; autogenerated if omitted
- `--name <text>`: optional display name
- `--group <text>`: optional long-lived grouping key inside the project; suitable for mapping one experiment plan or other durable work context; maps directly to the tmux session name
- `--gpus <int>`: requested GPU count; default is `1`
- `-- <your command...>`: everything after the separator is passed through as the user command

### `retry` Arguments

- `<task_id>`: required original task to retry
- `--group <text>`: optional explicit `group` for the new task; defaults to the original task's `group`

### `top` Arguments

- `--all`: show an overview of all machines; without it, only the current machine is shown

### `logs` Arguments

- `-f` / `--follow`: follow log output continuously; without it, only already-written content is printed

### `clean` Arguments

- `--dry-run`: show what would be removed without deleting it
- `--include-failed`: include failed and cancelled terminal tasks in cleanup
- `--older-than-seconds <int>`: only clean tasks older than this threshold; default is `604800` seconds, or 7 days
- `--task-id <id>`: clean one specific terminal task; mutually exclusive with `--older-than-seconds` and `--include-failed`

Cleanup result semantics:

- bulk cleanup selects objects mainly by terminal phase and age threshold
- single-task cleanup removes formal shared truth for that task and also repairs related batch truth and indexes
- after successful single-task cleanup, `qexp inspect <task_id>` and `qexp logs <task_id>` should behave as if the task no longer exists
- runtime log deletion is not a cross-machine strong guarantee; if a log remains, the CLI must report that explicitly

## Boundaries and Non-Goals

`qexp` handles lightweight experiment queueing and scheduling. It does not handle:

- training metric system design
- artifact hosting
- model version management
- data version management
- remote cross-machine command proxying

If you need a full experiment platform, `qexp` is not that kind of system. It is a
lightweight shared queue with scheduling semantics and recovery tooling.
