# ADR-QPIPELINE-0003: Scheduler Step Trigger Semantics

- Status: Accepted
- Date: 2026-03-14
- Owners: qpipeline maintainers

## Context

The runner previously stepped the main learning-rate scheduler from `on_validation_end`.
That made non-plateau schedulers depend on evaluation cadence instead of completed optimizer
updates.

This was inconsistent with the runner's step-mode policy, where `global_step`,
`max_steps`, and `eval_interval` already count completed `optimizer.step()` calls.
It also made non-plateau schedulers behave differently depending on how often validation
ran.

At the same time, `ReduceLROnPlateau` still needs validation metrics, so it cannot be
treated the same way as schedulers like `StepLR` or `CosineAnnealingLR`.

## Decision

Scheduler trigger policy is configured in `optim.scheduler_params.step_on`.

Policy:

- `scheduler_params.step_on=None`: resolve by scheduler type.
- For `plateau` schedulers, the resolved value is always `valid_end`.
- For non-plateau schedulers, the resolved default is `optimizer_step`.
- Non-plateau schedulers may explicitly opt into `valid_end` for backward-compatible
  validation-driven stepping.

Validation rules:

- `plateau` only accepts `step_on="valid_end"`.
- Non-plateau schedulers accept `step_on="optimizer_step"` or `step_on="valid_end"`.

Implementation rules:

- `qWarmupScheduler.step_after_optimizer_update()` is the single entry point used after a
  real `optimizer.step()`.
- During warmup, optimizer updates only advance warmup state.
- After warmup, non-plateau schedulers step immediately on optimizer updates when
  `step_on="optimizer_step"`.
- Validation-end listeners step the scheduler only for:
  - all plateau schedulers
  - non-plateau schedulers explicitly configured with `step_on="valid_end"`

## Consequences

Positive:

- Default non-plateau behavior now follows real optimization progress.
- `run_mode='step'` remains aligned across `global_step`, evaluation triggers, checkpoint
  triggers, EMA updates, and default scheduler updates.
- Plateau retains the metric-driven contract required by `ReduceLROnPlateau`.
- Projects that still want validation-driven stepping for non-plateau schedulers can opt
  in explicitly.

Trade-offs:

- Existing non-plateau configs without `step_on` may decay learning rate faster than
  before if they previously relied on sparse validation.
- Scheduler hyperparameters such as `T_max`, `step_size`, and `milestones` should now be
  interpreted in optimizer-step units unless `step_on="valid_end"` is set.

## Non-Goals

- This ADR does not redefine evaluation trigger semantics.
- This ADR does not add a separate epoch-end scheduler stepping mode.
- This ADR does not change checkpoint payload format beyond storing resolved scheduler
  trigger metadata needed for resume safety.

## Follow-ups

- Keep scheduler-related examples explicit about the meaning of `step_on`.
- Maintain regression tests for optimizer-step and validation-end scheduler modes.
