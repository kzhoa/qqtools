# qexp live application progress (v1)

## Scope and ownership

This implements Phase 0 (TTY-aware automatic qPipeline rendering) and Phase 1
(attempt-scoped live progress). Progress is optional, best-effort **advisory
observation**, never execution truth. It cannot renew a lease, change a claim,
retry/cancel a Task, publish terminal truth, or release resources.

No Task/Attempt schema, required capability, CI workflow, or default `task list`
format changes. Old readers ignore the additional namespace; commands without an
accepted report show `Progress status: no_report` with a short explanation.

```text
application -> machine-local mailbox -> agent observation thread
                                    -> shared advisory snapshot -> task show
stdout/stderr -------------------------------------------------> attempt log
```

The runner validates launch authority first, releases launch authority locks,
then provisions only machine-local progress state. It never creates or writes a
shared progress path. The application receives `QEXP_PROGRESS_PATH` and the frozen
`QEXP_PROGRESS_INTERVAL_SECONDS`; nested submissions have inherited progress
variables removed. Producers never choose a
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
queue. Ordinary local writes use the Attempt's reporting interval, including
stage and message changes. The default is 30 seconds. One initial write and one
final flush may bypass the ordinary interval; repeated flushes and unchanged
values do not create recurring writes. Intermediate updates and short stages may
be lost under this latest-value contract.

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
progress-contexts/<attempt-id>.json     # runner-owned launch/process identity and frozen policy
progress-observed/<attempt-id>.json     # last accepted agent observation
progress-diagnostics/<attempt-id>.json  # bounded latest diagnostic
```

New contexts include `reporting_policy_version: 1` and `interval_seconds`. The
runner resolves those values once before local channel provisioning and injects
both `QEXP_PROGRESS_PATH` and `QEXP_PROGRESS_INTERVAL_SECONDS` into the
application. Provisioning itself performs no shared-root read. A context without
policy fields is a permanent legacy-v1 interpretation and retains its original
agent projection cadence; restarting an agent does not resample project policy
for an existing Attempt.

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
cycle. Cheap machine-local file signatures avoid reparsing unchanged producer
mailboxes. For new policy contexts, producer mailboxes, accepted-observation
caches, and shared snapshots each apply the same per-Attempt interval to ordinary
successful replacements. A changed mailbox remains one pending latest value
until the observation-cache slot is due; the cache is persisted before the
corresponding shared publication can use its acceptance timestamp.

Stage and message changes do not bypass the interval. Each output permits one
bounded initial and one bounded final publication per continuous Attempt
lifetime. Restarted projectors restore accepted identity and timestamps, defer
ordinary publication by the interval plus a stable Attempt-derived phase, and do
not replay the initial exception. Missed slots do not cause catch-up bursts.
Diagnostics are coalesced per Attempt for at least `max(60, interval)` seconds.
These rules reduce per-Attempt load; they do not create a project-wide I/O quota
or a visibility deadline.

Advisory snapshots use same-directory temporary files and `os.replace` without
file/directory fsync. This primitive is intentionally separate from authoritative
`runtime.store.atomic_replace`.

## Freshness and presentation

`qexp task show TASK_ID` adds Progress status, Stage, Progress, Message,
Progress reported, and Progress advanced. JSON contains a separate top-level
`progress` object. Its legacy `status` remains `available|unavailable`; additive
fields are `observation_state` (`pending|no_report|available|unavailable`) and a
nullable `reason`.

The bounded reason codes are `not_started`, `no_snapshot`, `invalid_snapshot`,
`read_failed`, `identity_mismatch`, `cleanup`, and `unknown`. A queued Task with
no current execution is pending. A selected Attempt with no snapshot is
`no_report`, which does not claim that the application lacks integration. Invalid
identity, cleanup, malformed data, or an evidenced read failure is unavailable.
Only an accepted and currently authorized snapshot is available. A queued retry
never falls back to its preceding Attempt, and age alone never changes state.

`reported_at` advances when the agent accepts and checkpoints a newly observed
producer `update_id`. It is not the producer call time, query time, or an
unchanged republish time. `advanced_at` changes only when
stage/current/total/unit changes; message-only updates do not count as business
advancement. Agent restart, configuration changes, repeated reads, fencing-token
recovery, or registration metadata alone do not fabricate progress time.

Human output renders each accepted timestamp as an absolute UTC time with a
relative age. Clock differences never produce a negative age: the absolute time
is retained and the relative part becomes `unknown (clock difference)`. A
last-good shared snapshot remains available while it still passes current
identity checks, even if a later local reporting operation failed. With no valid
snapshot, the CLI fabricates neither timestamp nor freshness.

These timestamps are observations, not a health verdict. No stale threshold,
automatic cancellation, speed, or ETA is included in Phase 1.

## qPipeline

A rank-zero best-effort peer observer consumes existing runner facts. It calls the
same process-level `qqtools.qexp.progress.update()` API available to user code.
There is no qPipeline-owned Reporter and no second mailbox writer.

Training uses optimizer `global_step`, reliable `max_steps` when available, and
`unit=step`. Human messages use one-based active epoch and batch positions while
preserving the optimizer counter, for example `Epoch 4/10 · Batch 40/100`.
Completed epoch boundaries are labeled completed rather than inventing an active
batch position.

Evaluation orchestration emits internal loader/model boundary facts before each
loader starts, including empty loaders. The facts identify validation/test,
loader name or zero-based fallback index, and standard/EMA model variant. The
adapter reports counts per loader and emits no invented completion for an empty
loader. After evaluation it restores the latest known training context. The
adapter never reads Rich/Tqdm renderer state, and external producers continue to
use only the framework-neutral five-field application payload.

## Project reporting policy

The project-level commands are:

```bash
qexp config progress show
qexp config progress set --interval-seconds 30
```

The dedicated `<shared-root>/progress-policy.json` record has version 1 and one
finite numeric `interval_seconds` value greater than or equal to 1. Missing
configuration resolves to the 30-second default. `show` reports the effective
value, whether its source is `default` or `configured`, and
`applies_to: new_launches`. Explicit commands reject malformed or unreadable
configuration; launch-time consumption falls back to 30 with a bounded local
diagnostic so advisory configuration cannot prevent execution. Configuration
writes use a dedicated lock and durable atomic replacement.

The runner freezes the effective value at first successful provisioning of each
new Attempt. Changes apply to later launches, including retries, but never rewrite
running contexts. Inherited progress variables are removed before a nested
launch. Updated agents preserve legacy contexts without policy metadata, while
old agents may ignore the new advisory metadata; configured end-to-end cadence
therefore requires a new launch with both updated producer and agent code.

The interval controls write frequency, not arbitrary application I/O and not a
visibility service-level agreement. Independent producer and agent stages,
bounded scans, lock contention, and filesystem latency can delay visibility by
more than one interval. Initial/final exceptions can also create bounded startup
or shutdown bursts.

Deterministic scale acceptance uses 1,000 continuously changing synthetic
Attempts over the half-open 10T ordinary window after initialization. Each
producer, observed-cache, and shared output made exactly 10,000 ordinary
replacements (10 per Attempt). The measured one-second peaks were 43 producer
and 46 cache/shared replacements at T=30, and 26 producer and 29 cache/shared
replacements at T=60. Lifecycle exceptions are counted separately; these
controlled-clock distributions are not a filesystem IOPS guarantee.

Automatic rendering probes actual streams: Rich requires stdout TTY, tqdm uses
stderr TTY, otherwise auto resolves to plain. Explicit renderer requests retain
existing dependency fallbacks. This does not redesign plain-log verbosity.

## Validation

Run the focused feature suite:

```bash
./scripts/dev test \
  tests/unit/qexp/test_progress_policy.py \
  tests/unit/qexp/test_progress_protocol.py \
  tests/unit/qexp/test_progress_producer.py \
  tests/unit/qexp/test_progress_projector.py \
  tests/unit/qexp/test_progress_adapter.py \
  tests/unit/plugins/qpipeline/runner/test_runner_contracts.py \
  tests/integration/qexp/test_live_progress.py \
  tests/integration/qexp/test_output_format.py \
  tests/integration/functional/test_runner/test_multi_eval_loaders.py \
  tests/integration/functional/test_runner/test_progress_render_mode.py
```

Then run the related regression surface:

```bash
./scripts/dev test \
  tests/unit/qexp \
  tests/integration/qexp \
  tests/integration/functional/test_runner
```

Delivery status was reconciled on 2026-09-20 against dev commit
`3a5b39bebdf5f97958b0d15f6ab72374cb025630`, which includes the producer
concurrency hardening. All 10 live-progress Integration cases passed in the
retained full qexp run. That broader run had two unrelated test-boundary failures;
it is not reported as wholly green. Complete candidate preflight subsequently
passed (2,766 tests, two unrelated conditional skips), including Unit and
non-qexp Integration coverage. The promoted commit's
[Dev Preflight](https://github.com/kzhoa/qqtools/actions/runs/35488255866)
also passed. This closes the earlier pending post-hardening verification note;
it does not claim implementation of the deferred features below.

For the reporting-policy and usability extension, the focused suite passed 137
tests on 2026-09-20. Independent review found four cadence, large-value,
integration-fixture, and documentation defects; all were repaired, and follow-up
review reported no findings. The complete candidate preflight remains the
promotion gate rather than being claimed by this focused evidence.

Deferred: FD transport, `task watch`, metrics, history, ETA, list progress
columns, multiple streams, stale policy, stdout parsers, and third-party framework
adapters.
