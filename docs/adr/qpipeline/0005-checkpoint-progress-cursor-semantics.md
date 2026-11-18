---
doc_type: adr
adr_id: ADR-QPIPELINE-0005
status: accepted
updated_at: 2026-11-18
archived_at:
supersedes: []
superseded_by:
---

# ADR-QPIPELINE-0005: Checkpoint Progress-Cursor Semantics

- Status: Accepted
- Date: 2026-11-18
- Owners: qpipeline maintainers

## Context

The runner stores an epoch number and `batch_idx_in_epoch` in every training
checkpoint. At the end of an epoch, the final batch has already been consumed
while the runner has not yet advanced its epoch counter. A checkpoint written at
that boundary therefore records the progress made in the current epoch, not the
next epoch that a fresh run will start.

Treating that state as if it were already a next-epoch cursor causes an exhausted
loader cursor to be replayed incorrectly. Treating it as a committed next-epoch
state also makes the persisted epoch number disagree with the existing boundary
and filename conventions.

The framework needs one stable rule for checkpoint contents and recovery. The
rule must also preserve task state produced while finalizing an epoch.

## Decision

qpipeline checkpoints are **progress snapshots**, not precomputed next-run
cursors.

This ADR is authoritative when its ordering rules differ from the historical
training-boundary implementation pitch. The archived pitch remains useful for
the broader boundary-finalization rationale, but it does not override this
checkpoint cursor contract.

### Persist progress before advancing the epoch cursor

At an epoch boundary, the runner must:

1. finish epoch-level metric and task finalization needed for the snapshot;
2. save the checkpoint while `state.epoch` still identifies the epoch being
   completed and `state.batch_idx_in_epoch` records the number of consumed
   batches;
3. advance `state.epoch` and reset `state.batch_idx_in_epoch` only after the
   save succeeds.

For a seven-batch epoch, the checkpoint represents:

```text
epoch=0
batch_idx_in_epoch=7
global_step=7
```

The saved task state includes changes made during epoch finalization. A failed
checkpoint save prevents the successful epoch transition and follows the normal
training failure path.

Mid-epoch checkpoints keep their existing meaning: the cursor is the number of
batches already consumed in the current epoch, and the next run resumes from
that cursor.

### Resolve the next run position during recovery

With unchanged data and sampling configuration, after restoring a checkpoint
and before checking run limits or starting the next epoch, the runner
interprets the progress cursor against the current training loader:

- `0 <= batch_idx_in_epoch < len(train_loader)`: resume in the saved epoch and
  skip the consumed batches;
- `batch_idx_in_epoch == len(train_loader)`: the saved epoch is exhausted, so
  increment `epoch` once and reset `batch_idx_in_epoch` to zero;
- negative, non-integer, or `batch_idx_in_epoch > len(train_loader)` cursors:
  fail with a diagnostic error;
- an empty training loader is invalid for a resumable training session and must
  fail before entering the training loop.

Recovery performs only this cursor transition. For regular and completion
checkpoints, it must not replay validation, epoch-end callbacks, scheduler
updates, or checkpoint side effects that are already represented by the saved
progress snapshot. Best-checkpoint restore follows the same no-replay rule even
when task-specific lifecycle state requires task-owned reconstruction.

### Checkpoint roles

Regular and completion checkpoints are the framework's training-recovery
artifacts. They follow the complete progress-cursor contract above, including
the epoch-finalization ordering, subject to the recovery limits below.

Best checkpoints are model-selection artifacts. They are persisted immediately
when validation identifies a new best model so that a later training step or an
epoch-end callback cannot replace the model that won the comparison. A best
checkpoint restores the saved model weights and the other component state that
it contains, but it does not promise a complete replay-free reconstruction of
task lifecycle state captured after that validation boundary.

If a task has no meaningful mutable lifecycle state, a best checkpoint can also
serve as a practical training starting point after exhausted-cursor recovery.
If a task does have lifecycle state whose epoch-end mutation matters, the task
owner is responsible for supplying or reconstructing that state through its
normal `state_dict` contract or explicit setup. The framework must not replay
epoch-end callbacks automatically during restore.

Checkpoint filenames and checkpoint events continue to identify the boundary
that triggered the save; the nested runner state remains the authoritative
source for recovery.

No separate checkpoint version field or legacy-format detection is introduced.
The persisted `state` and its existing cursor fields define the contract.

### Recovery limit: explicitly shortened epochs

Exact epoch-boundary recovery is guaranteed here for natural loader exhaustion,
not for an epoch ended early through `LoopDirective(end_epoch=True)` while
`batch_idx_in_epoch < len(train_loader)`. This limitation applies to regular,
completion, and best checkpoints in both run modes.

For example, `(epoch=0, batch_idx_in_epoch=3)` with a seven-batch loader cannot
distinguish an ordinary mid-epoch save from a save at an explicitly shortened
epoch boundary. Recovery leaves this cursor unchanged and continues epoch 0
from batch index 3. It may process batches the original run intended to omit
and execute epoch finalization again later.

Do not infer early epoch completion, automatically advance such a cursor, or
introduce a format marker or special rejection path to distinguish these
cases. The checkpoint remains loadable, but exact continuation of the shortened
epoch is outside this ADR's recovery guarantee. This is distinct from early
stopping training at a partial-epoch cursor, which does not itself complete
that epoch.

## Consequences

Positive:

- Epoch-end checkpoints retain the existing completed-epoch naming and state
  convention.
- Recovery starts the next epoch exactly once instead of spinning over an
  exhausted loader.
- Task state produced during epoch finalization is present in the checkpoint,
  so recovery does not need to replay callbacks or scheduler work.
- Mid-epoch recovery remains unchanged.
- Checkpoint writing and epoch advancement have an explicit failure boundary:
  the epoch advances only after persistence succeeds.

Trade-offs:

- This cursor contract does not persist RNG state or mid-epoch metric
  accumulators and does not promise bitwise-equivalent training for arbitrary
  tasks.

- The checkpoint payload's epoch is the epoch being completed, while recovery
  may advance the in-memory epoch immediately after loading it.
- Checkpoint consumers must read `state.batch_idx_in_epoch` as a progress
  cursor, not as the first batch to process.
- Implementing the ordering requires separating epoch finalization from the
  final epoch/cursor transition in the runner.

## Non-Goals

- Do not add a second cursor field or a checkpoint-format version.
- Do not replay lifecycle callbacks as part of restore.
- Do not change `global_step` semantics.
- Do not change step-mode optimizer-step cadence or checkpoint naming policy.
- Do not make checkpoint loading a weights-only or partial-training restore.

## Implementation Rules

- Keep the cursor in `RunningState.batch_idx_in_epoch` and persist it through
  the existing nested `state` payload.
- Make epoch finalization produce the task/metric state that belongs in the
  snapshot without advancing the epoch cursor.
- Invoke the regular or completion save at the existing boundary policy point
  after that finalization and before the cursor transition.
- Settle rank-local epoch-finalization failures before checkpoint collectives.
  A failing rank must prevent all participating ranks from saving or advancing;
  already executed task mutations are not rolled back.
- Commit `epoch += 1` and `batch_idx_in_epoch = 0` only after the selected save
  has returned successfully on all participating ranks.
- Normalize an exhausted restored cursor once during session setup, before
  sampler epoch setup and epoch-start listeners.
- Keep best-model selection and persistence at validation time. Apply exhausted
  cursor normalization when a best checkpoint is used for training, but do not
  claim that its task lifecycle state is equivalent to a regular or completion
  recovery checkpoint.

## Verification

Required tests cover:

- a new natural epoch-end checkpoint containing `(epoch=0, batch=7, step=7)`
  and task state finalized for epoch 0;
- restoration of that checkpoint starting epoch 1 batch 0 without replaying
  epoch-end callbacks, validation, or scheduler updates;
- a mid-epoch checkpoint resuming at the first unconsumed batch;
- coincident periodic and completion saves producing one persisted checkpoint;
- completion checkpoints obeying the full recovery rule;
- best checkpoints preserving the validation-time model snapshot and restoring
  it without replaying lifecycle callbacks;
- invalid cursors failing diagnostically;
- rank-local epoch-finalization failure preventing regular/completion persistence
  and cursor advancement on every participating rank; and
- the save-failure path leaving the epoch transition uncommitted.

## Related Documents

- [ADR-QPIPELINE-0001](0001-runner-boundary-ownership.md)
- [ADR-QPIPELINE-0002](0002-gradient-accumulation-step-semantics.md)
- [ADR-QPIPELINE-0003](0003-scheduler-step-trigger-semantics.md)
- [Checkpoint progress and training resume](../../../src/qqtools/plugins/qpipeline/readme.md#checkpoint-progress-and-training-resume)
