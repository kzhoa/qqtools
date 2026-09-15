# qexp live application progress (v1)

## Scope and ownership

This implements Phase 0 (TTY-aware automatic qPipeline rendering) and Phase 1
(attempt-scoped live progress). Progress is optional, best-effort **advisory
observation**, never execution truth. It cannot renew a lease, change a claim,
retry/cancel a Task, publish terminal truth, or release resources.

No Task/Attempt schema, required capability, CI workflow, or default `task list`
format changes. Old readers ignore the additional namespace; commands without
a producer show `Progress: unavailable`.

```
application -> machine-local mailbox -> agent observation thread
                                    -> shared advisory snapshot -> task show
stdout/stderr -------------------------------------------------> attempt log
```

The runner provisions the channel under its existing launch authorization and
passes only `QEXP_PROGRESS_PATH` through the guardian to the application. It
clears inherited progress environment variables to avoid a nested submission
using a parent's channel. Producers never select Task IDs or supply trusted
fencing metadata. This is an ownership contract, not a security sandbox against
applications with the same Unix user and unrestricted access to runtime storage.

## Python API

```python
from qqtools.qexp import progress

for completed in range(1, 101):
    do_work()
    progress.update(stage="indexing", current=completed, total=100, unit="shard")
progress.flush()  # Optional bounded final flush; also registered at process exit.
```

The public implementation is standard-library-only and does not import qexp's
scheduler or a training framework. Installing qqtools' normal base dependencies
is still necessary. With no injected path, `update` is a no-op returning False.
It returns whether a report was accepted into a latest-only in-memory slot,
**not** whether it reached disk or the viewer. `flush` closes the default process
reporter; use it at application end, not inside the work loop. Framework adapters
use a dedicated `Reporter` per run.

One daemon writer performs file I/O. Application updates never wait for disk.
There is one pending slot, not an unbounded queue. Ordinary local writes are
limited to roughly once per second; stage changes may bypass that interval but
have a 100 ms minimum spacing. Intermediate updates may be lost. Shutdown waits
at most 100 ms by default (at most one second with an explicit timeout); final
report delivery is not guaranteed after abrupt exit or blocked storage.

Only the canonical main process/global rank zero should report. The helper
recognizes RANK, SLURM_PROCID and OMPI_COMM_WORLD_RANK without importing torch.
An already-created Reporter inherited across fork is disabled in the child.
Arbitrary non-DDP multi-process applications must designate their own writer;
this API is not a multi-writer election protocol.

## Complete replacement protocol

```json
{
  "protocol_version": 1,
  "update_id": "a-new-identifier-per-publication",
  "stage": "train",
  "current": 7340,
  "total": 15000,
  "unit": "step",
  "message": "epoch 0"
}
```

Updates replace the whole state; omitted optional fields become null. Stage and
update_id are required. Counts are null or nonnegative signed-64-bit integers
(not booleans); current cannot exceed a known total. Unknown totals and 0/0 are
valid, but 0/0 has no percentage. Stage and unit have no scheduler semantics.
A stage switch, unit switch or counter reset is a semantic change, not proof of
forward work or health.

Limits: 8 KiB wire record, stage 64 UTF-8 bytes, unit 32 bytes, message 1024 bytes,
identifier 128 ASCII characters. NaN, Infinity, duplicate keys, unsupported
versions, unknown payload fields, nested metrics and terminal control characters
are rejected. Readers bound bytes and refuse symlink/FIFO/device payload files.
Malformed reports never fail the application; diagnostics contain a bounded
reason code, not the original payload, and use one replaceable file per attempt.

## Storage and lifecycle

The local project runtime holds:

```
progress/<attempt-id>.json             # application mailbox
progress-contexts/<attempt-id>.json    # runner-owned launch/process identity
progress-observed/<attempt-id>.json    # last observation accepted by the agent
progress-diagnostics/<attempt-id>.json # bounded latest diagnostic
```

Shared readers consume:

```
<shared-root>/progress/<task-id>/<attempt-id>.json
```

The context is advisory identity evidence, not an execution authorization. It
survives normal process-evidence removal so a command that exits between agent
polls can still have its final report collected. Local accepted state preserves
restart deduplication while shared writes are coalesced or unavailable.

The shared snapshot includes task/attempt/launch/process identity, current
fencing token, source_update_id, agent sequence, reported_at, advanced_at and the
normalized progress payload. Identity comes from the runner context cross-checked
against the current Task/Attempt and registration generation, never the payload.

Only the agent's separate progress thread projects snapshots. It does not acquire
Task/lease locks or call renewal APIs. The current binding is checked again before
publication; observers validate current attempt and fencing token when joining a
snapshot. A race can temporarily make progress unavailable, not change authority.
A queued retry never shows the previous attempt as its current progress.

Reads rotate across projects and bounded batches of local contexts. Shared writes
normally occur at most once per five seconds per attempt. First/stage/terminal
updates can be expedited, with a one-second minimum spacing even for stage floods.
These are soft best-effort intervals, not a hard visibility deadline on a slow
filesystem. An unavailable shared filesystem delays progress, not the agent's
independent heartbeat/authority threads. The existing runtime's own storage and
lease failure policies remain unchanged.

Advisory writes use same-directory temporary files and `os.replace`, without
file/directory fsync. Atomic visibility is not crash durability. This primitive
is separate from authoritative `runtime.store.atomic_replace` and never creates
parents on update, preventing stale writers from recreating cleaned shared task
directories. Loss of all advisory caches may lose deduplication evidence; this is
not an exactly-once/durable event protocol.

Terminal collection preserves the last shared snapshot and retires local
artifacts. Existing Task cleanup removes shared and remaining local artifacts;
there is no additional retention policy. No final observation creates a fake
"100%" or overrides exit status.

## Freshness and presentation

`qexp task show TASK_ID` adds Stage, Progress, Message, Progress reported and
Progress advanced; JSON includes a separate top-level progress object.
`reported_at` advances only for a new accepted update_id. `advanced_at` changes
only when stage/current/total/unit changes. Message-only updates do not refresh
advanced_at. Restart or token recovery alone changes neither time.

These timestamps mean agent receipt, not producer wall time or a health verdict.
The agent's UTC receipt timestamps are compared with the viewer's clock; negative
ages show `unknown (clock difference)`. Cross-host age is approximate without
clock synchronization. No stale threshold, automatic cancellation, speed, or ETA
is provided. Agent heartbeat and lease renewal are not renamed process heartbeat.

## qPipeline

A rank-zero best-effort peer observer consumes existing runner facts. Training
uses global_step (optimizer steps), max_steps when available, unit=step and an
epoch message. It never reconstructs an independent step counter or reads Rich
state. EvaluationStartedFact lacks a val/test discriminator, so its initial
stage is honestly `evaluation`; existing progress ticks refine it to validation
or test with batch counts. Evaluation completion restores the fact's train
cursor. Different loaders may reset evaluation counts; no total-run ETA is implied.

Automatic rendering probes actual streams: Rich uses stdout TTY, tqdm uses
stderr TTY, otherwise plain. Explicit rich/tqdm requests retain existing dependency
fallbacks. This fixes renderer selection only: plain batch logging remains the
existing policy and is **not** silently disabled by progress transport availability.
A broader quiet/sparse logging policy is a separate change.

## Validation

From a complete checkout with project test dependencies installed:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/unit/qexp/test_progress_protocol.py \
  tests/unit/qexp/test_progress_producer.py \
  tests/unit/qexp/test_progress_projector.py \
  tests/unit/qexp/test_progress_adapter.py \
  tests/integration/qexp/test_live_progress.py \
  tests/integration/functional/test_runner/test_progress_render_mode.py

PYTHONPATH=src python -m pytest -q tests/unit/qexp tests/integration/qexp \
  tests/integration/functional/test_runner
```

The added pure-module tests were executed in an isolated source workspace (75
passed). All modified/new Python files passed compileall. This is **not** a full
installed-package or full-repository regression run. Repository lifecycle tests,
real guardian execution, existing test gates and genuine multi-rank/GPU or shared
filesystem qualification require running the commands above in a complete
checkout. No CI or complete pytest success is claimed by this implementation.

Deferred: FD transport, watch, metrics, history, ETA, list progress columns,
multiple streams, stale policy, stdout parsers and third-party framework adapters.
