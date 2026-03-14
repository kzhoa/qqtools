# ADR-QPIPELINE-0002: Gradient Accumulation Uses Optimizer-Step Semantics

- Status: Accepted
- Date: 2026-03-14
- Owners: qpipeline maintainers

## Context

The qpipeline runner previously assumed a one-to-one mapping between:

- one training batch
- one backward pass
- one optimizer update
- one `global_step` increment

After introducing optional `runner.accum_grad`, this assumption no longer holds.
Without a clear policy, the following fields become ambiguous in `run_mode='step'`:

- `global_step`
- `max_steps`
- `eval_interval`
- `save_interval`

## Decision

When `runner.accum_grad` is enabled, step-mode semantics are defined in units of completed optimizer updates, not micro-batches.

Policy:

- `runner.accum_grad=None`: disable accumulation.
- `runner.accum_grad` must be a positive integer when specified.
- In `run_mode='step'`, `global_step`, `max_steps`, `eval_interval`, and `save_interval` all count completed `optimizer.step()` calls.
- Batch-oriented progress display remains micro-batch oriented.
- Listener contexts tied to completed optimizer updates observe the post-step `global_step`. For example, with `eval_interval=2`, evaluation listeners see steps `2, 4, 6, ...`, not `1, 3, 5, ...`.

Implementation rule:

- `train_runner` validates `accum_grad` and passes the resolved value into `RunConfig`.
- `RunningAgent` accumulates gradients across micro-batches and increments `global_step` immediately after a real optimizer update, before emitting step-coupled listener state.
- If the final accumulation window in an epoch is incomplete, it is flushed at epoch end.

## Consequences

Positive:

- Step-mode counters now reflect actual optimization progress.
- Evaluation and checkpoint cadence stay aligned with parameter updates.
- EMA and scheduler stepping can remain coupled to real optimizer updates.
- Listener-visible step numbers now match the completed update that triggered the event.

Trade-offs:

- `global_step` no longer equals processed micro-batch count when accumulation is enabled.
- Some logs may show multiple batch events between consecutive step increments.

## Non-Goals

- This ADR does not redefine epoch-mode trigger semantics.
- This ADR does not add a new public event model for micro-step vs optimizer-step callbacks.

## Follow-ups

- Keep qConfig docs explicit about optimizer-step semantics in step mode.
- Maintain tests for validation, checkpoint, and stopping behavior under accumulation.
