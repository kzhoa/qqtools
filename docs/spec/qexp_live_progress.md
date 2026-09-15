# qexp live application progress (v1)

## Scope and ownership

This implements Phase 0 (TTY-aware automatic qPipeline rendering) and Phase 1
(attempt-scoped live progress). Progress is optional, best-effort **advisory
observation**, never execution truth. It cannot renew a lease, change a claim,
retry/cancel a Task, publish terminal truth, or release resources.

No Task/Attempt schema, required capability, CI workflow, or default `task list`
format changes. Old readers ignore the additional namespace; commands without a
producer show `Progress: unavailable`.

```text
application -> machine-local mailbox -> agent observation thread
                                    -> shared advisory snapshot -> task show
stdout/stderr -------------------------------------------------> attempt log
```

The runner validates launch authority first, releases launch authority locks,
then provisions only machine-local progress state. It never creates or writes a
shared progress path. The application receives only `QEXP_PROGRESS_PATH`; nested
submissions have inherited progress variables removed. Producers never choose a
Task ID, Attempt ID, fencing token, or machine registration generation.

The agent observation thread is the sole shared progress writer. Shared progress
writes are fenced independently from Task/Attempt authority and cannot delay the
agent heartbeat or lease-renewal threads.

## Python API

```python
from qqtools.qexp import progress

for completed in range(1, 101):
    do_work()
    progress.update(stage="indexing", current=completed, total=100, unit="shard")
progress.flush()
```

The supported application surface is `update()` and `flush()`. The implementation
is standard-library-only and does not import qexp scheduling or a training
framework. Without `QEXP_PROGRESS_PATH`, `update()` is a no-op returning `False`.
The return value means only that a latest-value report was accepted in process;
it does not prove that the value reached disk, the agent, or a viewer.

One process singleton owns one daemon writer. qPipeline uses the same singleton
rather than creating a second writer, so framework facts and explicit user calls
cannot race as independent mailbox writers. `flush()` closes the singleton; if
the daemon is still blocked in filesystem I/O after the bounded timeout, the
closing reporter remains the singleton sentinel and new reports are rejected
until that daemon actually exits. This preserves both bounded shutdown and the
single-writer invariant.

Application updates use a non-blocking state lock and never perform mailbox file
I/O on the application thread. The first accepted update may pay normal Python
thread-startup cost. There is one pending latest-value slot, not an unbounded
queue. Ordinary local writes are limited to roughly once per second; stage
changes may bypass that interval with a 100 ms minimum spacing. Intermediate
updates may be lost.

Only canonical global rank zero should report. The helper recognizes `RANK`,
`SLURM_PROCID`, and `OMPI_COMM_WORLD_RANK` without importing torch. An inherited
reporter is disabled after fork. This is not a multi-writer election protocol.

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

Updates replace the whole state; omitted optional fields become null. `stage` and
`update_id` are required. Counts are null or nonnegative signed-64-bit integers
(not booleans); current cannot exceed a known total. Unknown totals and `0/0` are
valid, but `0/0` has no percentage.

Limits: 8 KiB wire record, stage 64 UTF-8 bytes, unit 32 bytes, message 1024
bytes, identifier 128 ASCII characters. NaN, Infinity, duplicate JSON keys,
unsupported versions, unknown payload fields, nested metrics, and control
characters are rejected. Readers bound bytes and refuse symlink/FIFO/device
payload files. Malformed reports never fail the application.

## Local storage and cleanup fencing

The local project runtime holds:

```text
progress/<attempt-id>/latest.json       # application mailbox
progress-contexts/<attempt-id>.json     # runner-owned launch/process identity
progress-observed/<attempt-id>.json     # last accepted agent observation
progress-diagnostics/<attempt-id>.json  # bounded latest diagnostic
```

The mailbox lives inside an Attempt-owned directory. Cleanup removes the context
first and then removes the entire Attempt mailbox directory. Advisory writers do
not create parent directories, so a late producer cannot recreate
`progress/<attempt-id>/latest.json` after cleanup removed its parent.

Local cleanup uses race-safe deletion (`missing_ok`/ignored missing directories)
rather than `exists() -> unlink()` sequences. Projector cache writes recheck the
context after replace and delete themselves if cleanup won the race.

## Shared projection and cleanup fencing

Shared readers consume:

```text
<shared-root>/progress/<task-id>/<attempt-id>.json
```

The runner never creates this namespace. Immediately before a shared write the
agent acquires a dedicated per-Task advisory progress lock, revalidates current
Task/Attempt authority and machine registration eligibility, creates the shared
progress directory if still valid, then atomically replaces the snapshot.

Task cleanup marks the Task as cleaning through the existing authoritative
workflow and later acquires the same progress lock before removing the shared
Task progress directory. Therefore:

- a projector that wins the progress lock first finishes before cleanup removes
  its result;
- a projector that arrives after cleanup was marked fails revalidation and does
  not recreate the directory;
- cleanup and projection never rely on a racy `exists() -> unlink()` contract.

The progress lock is not a Task/Attempt authority lock. Projectors never acquire
Task/lease locks, never renew a lease, and never mutate execution truth.

## Projection identity and fencing

The shared snapshot contains:

- runner-derived Task/Attempt/launch/process identity;
- current Attempt fencing token;
- **machine registration generation** owned by the projecting agent;
- source update ID and agent sequence;
- `reported_at` and `advanced_at`;
- normalized progress payload.

The machine registration generation is not producer-controlled. Each
`ProgressProjector` is permanently bound to the registration generation of the
agent binding that created it. Before every shared write, the agent checks that
that binding is still write-eligible.

Running-task readers require both the current Attempt fencing token and the
current shared machine registration generation. If a machine registration is
superseded in the narrow check-to-write window, an old-generation snapshot may
briefly exist on disk but a running reader rejects it. Terminal snapshots keep
registration generation as provenance but are not invalidated by a later machine
adoption, preserving historical final progress.

Retry isolation remains Attempt-scoped: queued retry never displays the previous
Attempt as current progress. Terminal Task publication deliberately clears
`current_attempt_id`, so terminal progress is joined using the preserved current
Attempt number and its AttemptRecord.

## Polling and I/O behavior

The progress observation loop is a dedicated daemon thread. Before it touches any
shared machine-registration state, it checks the project-local runtime for a real
producer mailbox (`progress/*/latest.json`). Projects that have never emitted a
progress update therefore add no progress-specific shared registration polling.

Projects are rotated and local contexts are processed with bounded work per
cycle. Shared writes normally occur at most once per five seconds per Attempt.
First/stage/terminal publications can be expedited with a one-second minimum
shared-write spacing. These are soft visibility intervals, not health deadlines.

Advisory snapshots use same-directory temporary files and `os.replace` without
file/directory fsync. This primitive is intentionally separate from authoritative
`runtime.store.atomic_replace`.

## Freshness and presentation

`qexp task show TASK_ID` adds Stage, Progress, Message, Progress reported, and
Progress advanced. JSON contains a separate top-level `progress` object.

`reported_at` advances for each newly accepted `update_id`. `advanced_at` changes
only when stage/current/total/unit changes; message-only updates do not count as
business advancement. Agent restart, fencing-token recovery, or registration
metadata alone do not fabricate progress time.

These timestamps are observations, not a health verdict. No stale threshold,
automatic cancellation, speed, or ETA is included in Phase 1.

## qPipeline

A rank-zero best-effort peer observer consumes existing runner facts. It calls the
same process-level `qqtools.qexp.progress.update()` API available to user code.
There is no qPipeline-owned Reporter and no second mailbox writer.

Training uses optimizer `global_step`, `max_steps` when available, `unit=step`,
and an epoch message. Evaluation starts as `evaluation` because
`EvaluationStartedFact` has no val/test discriminator; subsequent progress ticks
refine it to `validation` or `test` with batch counts. Evaluation completion
restores the train cursor. The adapter never reads Rich/Tqdm renderer state.

Automatic rendering probes actual streams: Rich requires stdout TTY, tqdm uses
stderr TTY, otherwise auto resolves to plain. Explicit renderer requests retain
existing dependency fallbacks. This does not redesign plain-log verbosity.

## Validation

Run the focused feature suite:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/unit/qexp/test_progress_protocol.py \
  tests/unit/qexp/test_progress_producer.py \
  tests/unit/qexp/test_progress_projector.py \
  tests/unit/qexp/test_progress_adapter.py \
  tests/integration/qexp/test_live_progress.py \
  tests/integration/functional/test_runner/test_progress_render_mode.py
```

Then run the related regression surface:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/unit/qexp \
  tests/integration/qexp \
  tests/integration/functional/test_runner
```

A pre-hardening machine run of the related regression surface reached 988 passed,
2 skipped, with one unrelated stale `qexp init` output-contract test; that test
was fixed separately on `main`. The concurrency-hardening changes in this branch
add new race/single-writer tests and require a fresh machine run before merge. No
post-hardening full-suite success is claimed here yet.

Deferred: FD transport, `task watch`, metrics, history, ETA, list progress
columns, multiple streams, stale policy, stdout parsers, and third-party framework
adapters.
