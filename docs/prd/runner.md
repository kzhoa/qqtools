# Runner PRD (qpipeline plugin)

## 1. Scope

This document defines business-level field relationships for `train_runner` in qpipeline.
It focuses on:
- mutual exclusion
- field coupling
- precedence and effective-value resolution

It does not describe low-level loop mechanics in detail.

## 2. Ownership Boundary

### 2.1 Policy Owner

Business policy is owned by `train_runner`.

### 2.2 Execution Owner

`RunningAgent` is downstream and policy-agnostic.
`RunningAgent` only executes the already-resolved `RunConfig` boundaries and intervals.

### 2.3 Design Rule

Any rule like "field A suppresses field B" must be handled in `train_runner`, before creating `RunningAgent`.

## 3. Field Relationship Rules

## 3.1 Primary Mode Switch

`run_mode` is the primary switch that controls interpretation of multiple fields.
Supported values:
- `epoch`
- `step`

## 3.2 Boundary Mutual Exclusion (Business Policy)

Resolved in `train_runner`:
- If `run_mode='epoch'`:
  - keep `max_epochs`
  - ignore `max_steps`
- If `run_mode='step'`:
  - keep `max_steps`
  - ignore `max_epochs`

This policy is applied before `RunConfig` is passed to `RunningAgent`.

## 3.3 Interval Coupling

`eval_interval` and `save_interval` are coupled to `run_mode`:
- In `epoch` mode: interval units are epochs.
- In `step` mode: interval units are global steps.

## 3.4 Early Stop Coupling

`early_stop` is evaluated on evaluation events, so its effective cadence is indirectly coupled to:
- `run_mode`
- `eval_interval`

`early_stop.target` must correspond to a metric key available in evaluation results.

## 3.5 Checkpoint Coupling

`checkpoint.target/mode/min_delta` govern best-checkpoint update logic.
Regular checkpoint trigger cadence is coupled to:
- `run_mode`
- `save_interval`

## 4. Effective Value Resolution Order (train_runner)

1. Read explicit function arguments and `args.runner` fields.
2. Validate required base fields (`run_mode`, `args.runner`).
3. Apply business policy (including boundary mutual exclusion).
4. Apply default/fallback values where needed.
5. Build effective `RunConfig`.
6. Instantiate `RunningAgent` with effective config only.

## 5. Conflict Handling

## 5.1 Dual Boundary Input

If both `max_epochs` and `max_steps` are provided:
- `train_runner` keeps only the boundary compatible with current `run_mode`.
- `train_runner` logs a warning describing which boundary is ignored.

## 5.2 Missing Boundary

At least one effective boundary must remain after resolution.
Otherwise, `train_runner` raises an error.

## 6. Behavioral Examples

- Example A:
  - `run_mode='epoch'`, `max_epochs=10`, `max_steps=1000`
  - effective boundary: `max_epochs=10`, `max_steps=None`

- Example B:
  - `run_mode='step'`, `max_epochs=10`, `max_steps=1000`
  - effective boundary: `max_epochs=None`, `max_steps=1000`

- Example C:
  - `run_mode='epoch'`, `eval_interval=2`, `save_interval=3`
  - evaluate every 2 epochs, save regular checkpoint every 3 epochs

## 7. Non-Goals

- Do not push business mutual-exclusion logic into `RunningAgent`.
- Do not let log semantics diverge from effective behavior.

## 8. References

- ADR rationale and history:
  - `docs/adr/qpipeline/0001-runner-boundary-ownership.md`
