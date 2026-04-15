# qpipeline Timing Manual

Status: Draft

Updated: 2026-04-15

## Goal

This manual explains how to read `qpipeline` training logs in time order.

It answers these user-facing questions:

1. What happens first during a run
2. Which logs belong to batch progress, evaluation, epoch completion, or final termination
3. How `run_mode=epoch` and `run_mode=step` change the timing of evaluation and checkpoint-related events
4. Why the same epoch can contain multiple evaluation summaries in step mode

This document is intentionally user-oriented. It explains observable behavior and timing, not internal implementation details beyond what is needed to interpret logs correctly.

## The Four Timing Layers

During training, the most important outputs belong to four different timing layers:

1. Batch layer
   Progress updates and batch-level metrics appear while batches are being processed.
2. Evaluation layer
   `[Eval Summary]` appears after a real evaluation event has completed.
3. Epoch layer
   `--- Epoch N Results ---` appears only when an epoch has actually finished.
4. Run layer
   The terminal event appears once, at the very end of the run.

If you keep these four layers separate, the log stream becomes much easier to read.

## What Each Log Means

### 1. Batch and progress logs

These logs are the closest thing to "live training progress".

They usually include information such as:

- current batch position
- current or averaged training loss
- current or averaged training metric
- learning rate
- timing information

These logs are tied to training batch completion, not to evaluation and not to epoch completion.

### 2. `[Eval Summary]`

`[Eval Summary]` means an evaluation was actually executed.

It is the primary signal for:

- validation/test results being refreshed
- best-model tracking being updated
- evaluation-time learning-rate context being recorded

If you want to know whether a new evaluation really happened, look for a new `[Eval Summary]`.

### 3. `--- Epoch N Results ---`

This is an epoch-end summary.

It is not the same thing as an evaluation event.

It summarizes:

- current train metrics at epoch end
- the currently known validation metric
- the currently known test metric

The validation/test values shown here may be:

- freshly computed at this epoch boundary
- reused from the most recent earlier evaluation
- missing

So this block should be read as an epoch closing summary, not as proof that evaluation just happened.

### 4. Terminal event

A terminal event is the final run-level status line.

Examples:

```text
Training finished: reason=max_steps
Training finished: reason=max_epochs
Training finished: reason=early_stop
Training stopped: reason=user_interrupt
Training failed: reason=oom
Training failed: reason=exception
Training failed: reason=nan_detected
```

There is exactly one terminal event per run.

It is the authoritative final status for the run.

## Shared Timing Rules

The following rules are always true, regardless of run mode:

- Batch/progress logs happen during training batches
- `[Eval Summary]` happens only after an evaluation has completed
- `--- Epoch N Results ---` happens only after an epoch has completed
- The terminal event happens only once the run is exiting
- If a periodic boundary detects `NaN` before evaluation or regular checkpoint, the run exits immediately with `Training failed: reason=nan_detected`

In other words:

- evaluation logs are not epoch logs
- epoch logs are not final-run logs
- final-run logs do not tell you how many evaluations happened

## `run_mode=epoch`

In epoch mode, the unit of `eval_interval` is epochs.

Example:

- `eval_interval=1` means evaluate every epoch
- `eval_interval=2` means evaluate every 2 epochs

### Typical timing in epoch mode

For an epoch that does not trigger evaluation:

1. Training batches run
2. Batch/progress logs appear during the epoch
3. The epoch finishes
4. `--- Epoch N Results ---` is printed

For an epoch that does trigger evaluation:

1. Training batches run
2. Batch/progress logs appear during the epoch
3. The last batch of the epoch finishes
4. If the same periodic boundary detects `NaN`, the run fails immediately
5. Otherwise evaluation runs
6. `[Eval Summary]` is printed
7. The epoch-end summary is printed as `--- Epoch N Results ---`

The important user-facing implication is:

- in epoch mode, evaluation only happens at epoch boundaries
- when evaluation happens, you will usually see `[Eval Summary]` before `--- Epoch N Results ---`

### Example

```text
Starting training (mode=epoch, eval_interval=2, ...)
... batch/progress logs for epoch 0 ...
--- Epoch 0 Results ---
[train] loss: ...
[val] metric: n/a source=missing

... batch/progress logs for epoch 1 ...
[Eval Summary] Epoch: 1, Step: S
  - Validation: ...
  - Testing: ...
[Eval Summary Table] Epoch: 1 | Step: S | LR: ...
--- Epoch 1 Results ---
[train] loss: ...
[val] metric: ... source=current_eval
[test] metric: ... source=current_eval
```

## `run_mode=step`

In step mode, the unit of `eval_interval` is completed optimizer steps.

This is important:

- it does not mean raw dataloader batches
- it means real completed `optimizer.step()` updates

This remains true even when gradient accumulation is enabled.

### Typical timing in step mode

Within a long epoch:

1. Training batches run
2. Batch/progress logs appear during the epoch
3. Whenever the completed optimizer-step count hits a periodic boundary, `qpipeline` first checks whether the latest observed training loss on `rank0` is `NaN`
4. If `NaN` is detected, that boundary skips evaluation and regular checkpoint, and the run exits as failed
5. Otherwise evaluation runs when `eval_interval` is hit
6. Each evaluation prints a new `[Eval Summary]`
7. The epoch-end summary still waits until the epoch actually finishes

The important user-facing implication is:

- one epoch can contain multiple `[Eval Summary]` blocks
- but still only one `--- Epoch N Results ---` block

### Example

```text
Starting training (mode=step, eval_interval=100, ...)
... batch/progress logs ...
[Eval Summary] Epoch: 0, Step: 100
... batch/progress logs ...
[Eval Summary] Epoch: 0, Step: 200
... batch/progress logs ...
[Eval Summary] Epoch: 0, Step: 300
... more batch/progress logs ...
--- Epoch 0 Results ---
[train] loss: ...
[val] metric: ... source=latest_eval_reuse
[test] metric: ... source=latest_eval_reuse
```

This is normal.

It does not mean epoch logging is duplicated.
It means evaluation and epoch completion are different kinds of events.

## How To Read Epoch-End Validation/Test Metrics

The `[val]` and `[test]` lines inside `--- Epoch N Results ---` include a `source=...` marker.

Read it as follows:

- `source=current_eval`
  A fresh evaluation happened at this epoch boundary.
- `source=latest_eval_reuse`
  No fresh evaluation happened at this epoch boundary, so the epoch summary reused the latest known evaluation result.
- `source=missing`
  No evaluation result is available yet.

This rule matters most in step mode.

In step mode, you may already have seen one or more `[Eval Summary]` blocks earlier in the same epoch. If the epoch ends without another evaluation exactly at the boundary, the epoch summary will reuse the latest known values instead of creating a new evaluation event.

## Special Case: Run Ends Mid-Epoch

In `run_mode=step`, the run may end in the middle of an epoch because `max_steps` is reached.

In that case:

- you may already have seen a recent `[Eval Summary]`
- you may see a terminal event immediately afterward
- you may not see `--- Epoch N Results ---` for that unfinished epoch

Example:

```text
... batch/progress logs ...
[Eval Summary] Epoch: 0, Step: 1000
Training finished: reason=max_steps
```

This is expected behavior.

The run ended correctly, but the current epoch did not finish naturally, so no epoch-end summary was produced for that partial epoch.

## NaN Detection Boundary

`qpipeline` does not perform a dedicated NaN stop check after every batch.

Instead, NaN handling is attached to the same periodic boundaries that already drive evaluation and regular checkpoint decisions.

Current behavior:

- the runner watches the latest observed training loss value
- the NaN gate is checked only when a periodic boundary is reached
- the first implementation checks `rank0` only
- if NaN is detected at that boundary, evaluation is skipped
- if NaN is detected at that boundary, the regular checkpoint is also skipped
- the final terminal event becomes `Training failed: reason=nan_detected`

This means the user may still see earlier batch/progress output before the failure line appears. The failure is reported at the next configured periodic boundary, not at the exact batch where NaN first appeared.

## Reading Order Recommendations

If you are debugging or monitoring a run, use this reading order:

1. Read the terminal event to determine the final run outcome
2. Read all `[Eval Summary]` blocks to understand when evaluation actually happened
3. Read `--- Epoch N Results ---` blocks as epoch-closing summaries
4. Read batch/progress logs for local context around spikes, stalls, or instability

This avoids the most common confusion:

- treating epoch summaries as evaluation events
- assuming every epoch summary means a fresh validation pass
- assuming missing epoch-end logs imply a broken run when the run actually ended mid-epoch by design

## Practical Mental Model

You can think of the timing model like this:

- batches produce progress
- evaluations produce `[Eval Summary]`
- epoch completion produces `--- Epoch N Results ---`
- run exit produces the terminal event

That is the correct top-level model for interpreting `qpipeline` timing.

## Related Documents

- [qpipeline Log Format](/mnt/c/Users/Administrator/proj/qqtools/docs/log_format.md)
- [ADR 0001: Runner Boundary Ownership](/mnt/c/Users/Administrator/proj/qqtools/docs/adr/qpipeline/0001-runner-boundary-ownership.md)
- [ADR 0002: Gradient Accumulation Step Semantics](/mnt/c/Users/Administrator/proj/qqtools/docs/adr/qpipeline/0002-gradient-accumulation-step-semantics.md)
- [ADR 0003: Scheduler Step Trigger Semantics](/mnt/c/Users/Administrator/proj/qqtools/docs/adr/qpipeline/0003-scheduler-step-trigger-semantics.md)
