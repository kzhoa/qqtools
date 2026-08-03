---
doc_type: adr
status: active
updated_at: 2026-08-03
archived_at:
---

# ADR-QPIPELINE-0004: Task-Owned Evaluation Semantics

- Status: Accepted
- Date: 2026-08-03
- Owners: qpipeline maintainers

## Context

qpipeline historically exposes one `train_loader`, one `val_loader`, and one `test_loader` on `qTask`.
That contract is insufficient for benchmarks that evaluate one trained model against multiple named
validation or test subsets.

The evaluation sources may also change during training. A task may intentionally start with
`val_loader=None` or `test_loader=None`, then enable or replace those sources after the model reaches
a task-defined condition. Caching loader references in the Runner at initialization would make those
runtime changes ineffective.

Multiple validation loaders also create a control-policy question. Checkpoint selection, early
stopping, and validation-driven schedulers require one canonical scalar, but the framework cannot
infer whether a benchmark should select one loader result, compute a macro average, use sample-weighted
aggregation, or apply another domain-specific formula.

The architecture therefore needs a stable ownership boundary between:

- task-owned evaluation data and benchmark semantics;
- Runner-owned evaluation orchestration and control mechanics.

## Decision

qTask owns evaluation sources and evaluation semantics. Runner owns boundary-safe evaluation
orchestration and consumes only the canonical control value returned by qTask.

### Public evaluation-source contract

qTask continues to expose two independent public attributes:

```python
task.val_loader: None | DataLoader | Dict[str, DataLoader]
task.test_loader: None | DataLoader | Dict[str, DataLoader]
```

Policy:

- `None` disables the corresponding evaluation stage at the current evaluation boundary.
- A single DataLoader preserves the existing single-source behavior.
- A non-empty dict represents multiple explicitly named sources.
- Multi-source containers must be named mappings; anonymous list/tuple containers are unsupported.
- qTask does not gain parallel `val_loaders` or `test_loaders` attributes.
- validation and test remain separate task properties and are not merged into one public stage
  registry.

Runner may normalize both properties through shared internal code. An internal default source key may
represent a single unnamed loader, but that key is not part of the public metric namespace.

### Loader ownership and evaluation boundary

qTask remains the sole long-lived owner of `val_loader` and `test_loader`.

Runner must not cache a permanent evaluation-loader plan during initialization. At every evaluation
boundary it must:

1. read the current `task.val_loader` and `task.test_loader`;
2. validate and normalize them independently;
3. create one evaluation-pass snapshot of source names, order, and loader references;
4. reuse that snapshot for standard-model and EMA-model evaluation;
5. discard it after the evaluation pass;
6. read the task properties again at the next evaluation boundary.

The snapshot freezes pass topology and references only. It does not deep-copy a DataLoader or freeze
its underlying dataset. Task code may replace loader attributes between evaluation boundaries; a
replacement made after a pass has started takes effect at the next boundary.

### Evaluation execution responsibility

Runner owns the mechanical evaluation loop:

- iterate every normalized source;
- run forward/evaluation mechanics;
- collect raw metrics;
- apply DDP and EMA mechanics consistently;
- emit events, summaries, and complete per-loader results.

Task authors do not implement a second multi-loader orchestration loop inside qTask.

An evaluation boundary remains valid even when one or both stages currently have no source. Runner
skips only the corresponding model forward and metric reduction. Boundary lifecycle and result
distribution still occur. Target-based consumers decide whether to advance from the actual presence
of their target key, not merely from the occurrence of an evaluation boundary.

Target presence is determined by key membership. Consumers must not use
`results.get(target) is None` to conflate an absent key with a present key whose value is `None`.
Runner omits `test_metric` when the documented test-stage hook exception returns `None`; other
present-but-invalid target values proceed to the concrete consumer and are not treated as missing.

When a target key is absent:

- best-model tracking does not compare or request a best checkpoint;
- early stopping does not advance or reset patience;
- a plateau scheduler does not perform its metric-driven step;
- primary-target summary comparison is omitted while actual evaluation loaders are still rendered;
- the corresponding current canonical metric becomes `None` while its latest valid value is retained.

Non-target boundary consumers continue normally. This includes task callbacks, regular checkpoint
triggers, JSONL boundary events, evaluation events, and non-plateau schedulers explicitly driven
by `valid_end`.

Every target-based consumer that applies a missing-target default must emit its own DEBUG record at
the decision point. The record identifies the consumer, missing target key, and selected default
behavior, plus epoch and global step when available. Logs are not consolidated across consumers, but
the same consumer logs at most once per target and evaluation boundary. Distributed execution follows
the existing rank-aware logging policy. Missing ordinary CSV columns or non-target loader metrics do
not fall under this rule.

### Evaluation-pass result representation

Runner uses `EvaluationResult` as the canonical result for one evaluation boundary. It contains an
optional training interval, ordered model variants, ordered stages, and ordered loader records. Each
stage owns one task-derived score while every loader retains its raw metric mapping. A loader without
an explicit name uses `None`, not a reserved string.

The result does not use a composite-key mapping. Formatters and events traverse the model → stage →
loader structure, so loader identity is never reconstructed from a rendered metric key. Public event
contexts expose the result through `evaluation`; the old `context.eval_results` field and all flat
metric compatibility views are not retained.

### Canonical control metric

qTask owns the reduction from stage metrics to a canonical scalar through one task-level interface:

```python
post_metrics_to_value(result, *, stage) -> float
```

The framework supports signature evolution for this hook through initialization-time, name-based
call binding. Existing task implementations may retain `post_metrics_to_value(result) -> float`;
new implementations may explicitly declare `stage` in any non-positional-only order, including
keyword-only. Public qTask hooks that adopt this infrastructure must explicitly list every contract
parameter and may not use `*args` or `**kwargs`. Unknown named parameters are rejected during
initialization rather than silently ignored. The binder is shared by every call path and invokes the
task hook with named arguments only.

Runner passes the complete metric result for that stage:

- single source: one metric mapping;
- multiple sources: a mapping from source name to metric mapping.

Runner must not:

- select the first source as canonical;
- automatically average source metrics;
- infer weighting from loader size;
- create a canonical test metric from dict insertion order.

The canonical validation scalar is exposed as `val_metric`. It is the default control value for:

- best-checkpoint selection;
- early stopping;
- `ReduceLROnPlateau` and other validation-driven schedulers;
- primary validation summary state.

The public hook signature remains `-> float`; it is not widened to `float | None`. Product behavior
allows the test-stage call to return `None`, in which case Runner omits `test_metric`. This test-only
exception does not become part of the public type signature.

The training Runner checks the hook signature and binds its adapter once during initialization.
Standalone evaluation binds the same contract once at the start of each independent evaluation call,
because it has no long-lived training Runner instance to own the binding. Runtime evaluation must
not repeatedly inspect the signature or catch `TypeError` to guess the call shape.

Standalone evaluation receives an explicit `stage: Stage = Stage.TEST` parameter. Its structured
result records the stage directly and is never inferred from a presentation string.

Runner does not perform generic scalar-conversion, return-type, or finiteness validation on every
evaluation pass. The contract requires validation to return a float. Apart from the documented TEST
exception, invalid values are exposed by the concrete consumer operation that cannot use them rather
than by a duplicated hot-path validator.

If an evaluation boundary produces no validation source, `val_metric` is absent. Validation-dependent
controls do not advance because their target key does not exist; a test source at the same boundary
does not change that rule.

Runner state separates freshness from history:

- `current_val_metric` and `current_test_metric` represent only the current evaluation boundary and
  become `None` when their canonical key is absent;
- `latest_val_metric` and `latest_test_metric` retain the most recent actually produced canonical
  value and change only when that key is present;
- checkpoints persist and restore both current and latest fields;
- a latest value must never be presented or consumed as a current-boundary value.

A canonical `test_metric` exists only when qTask explicitly reduces the complete test result. All
named test-loader metrics remain available for reporting even when no canonical test scalar is
produced.

## Consequences

Positive:

- Existing single-loader tasks retain their current public properties.
- Advanced benchmarks can expose multiple named validation and test loaders without user-written
  orchestration loops.
- Runtime `None -> loader` and loader replacement workflows remain supported.
- Standard and EMA evaluation compare against the same source snapshot within one pass.
- Benchmark-specific aggregation stays with the task that owns its meaning.
- Checkpoint, early-stop, and scheduler mechanics consume one stable validation contract.
- validation and test keep distinct business responsibilities while sharing implementation code.

Trade-offs:

- The `val_loader` and `test_loader` annotations become union types.
- qTask implementations that override metric reduction must accept stage context and handle the
  result shape they declared.
- Runner must validate loader mappings at every evaluation boundary rather than only at startup.
- Replacing a loader during an active evaluation pass is intentionally not immediately visible.
- Complete per-loader results and canonical control metrics remain separate concepts that consumers
  must not conflate.
- Boundary occurrence, stage execution, and target-key presence become distinct state facts.
- Current canonical metrics no longer silently reuse stale values; historical access is explicit
  through latest-value state.
- Missing-target control flow is observable at DEBUG level without turning a legal dynamic-loader
  state into a warning or error.
- The stage-aware hook call shape is bound once at Runner initialization rather than resolved in the
  evaluation hot path.
- Internal evaluation and event interfaces do not carry compatibility adapters. External event and
  task-hook changes require a user-facing upgrade guide before release.

## Rejected Alternatives

### Add parallel singular and plural attributes

Rejected because `val_loader` plus `val_loaders`, and `test_loader` plus `test_loaders`, create two
sources of truth and require precedence rules for conflicting assignments.

### Merge validation and test into one stage dictionary

Rejected because validation participates in model control while test primarily reports outcomes.
Shared execution mechanics do not require shared public ownership.

### Cache loaders in Runner initialization

Rejected because task-owned runtime replacement would be invisible and would conflict with the
existing runtime-control ownership rule.

### Require qTask to implement the multi-loader loop

Rejected because it would duplicate DDP, EMA, event, progress, and error-handling mechanics across
tasks.

### Let Runner choose or aggregate a primary validation source

Rejected because source selection and aggregation are benchmark semantics, not framework mechanics.

## Non-Goals

- Define dynamic CSV schema evolution.
- Replace flat target, logging, and checkpoint keys with structured keys.
- Add multiple training loaders.
- Define implicit multi-loader inference behavior.
- Standardize every benchmark's validation aggregation formula.
- Freeze DataLoader or dataset objects beyond one pass's topology snapshot.

## Follow-ups

- Implement the single/multiple loader normalization and per-pass snapshot contract.
- Migrate `post_metrics_to_value(...)` to receive explicit stage context, bind that call shape at
  initialization, and document the external interface upgrade.
- Keep early-stop, checkpoint, scheduler, RunningState, callback, summary, and logging tests explicit
  about missing target metrics and per-consumer DEBUG records.
- Preserve all named evaluation loaders in events, summaries, snapshots, and logging.
- Publish a user-facing interface upgrade guide after implementation and before release.
- Resolve dynamic CSV persistence separately before declaring multi-loader logging complete.

## Related Documents

- `docs/pitch/qpipeline-single-multi-eval-loader-contract.md`
- `docs/pitch/qpipeline-runtime-training-control-implementation-plan-zh.md`
- `docs/pitch/qpipeline-runtime-training-control-implementation-plan.md`
- `docs/adr/qpipeline/0003-scheduler-step-trigger-semantics.md`
- `docs/pitch/qpipeline-dynamic-csv-metric-schema.md`
- `docs/pitch/qpipeline-structured-eval-metric-schema.md`
