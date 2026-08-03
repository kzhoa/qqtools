---
doc_type: spec
status: active
updated_at: 2026-08-03
archived_at:
---

# qpipeline v1.2.34 Structured Multi-Evaluation Upgrade

`qTask.val_loader` and `qTask.test_loader` now accept `None`, one `DataLoader`, or a non-empty
`dict[str, DataLoader]`. A mapping evaluates each source in insertion order. Mapping keys are loader
labels for structured output; a single unnamed loader is represented by `name=None`.

```python
self.val_loader = {"in_domain": val_loader, "ood": ood_loader}
self.test_loader = {"benchmark": benchmark_loader}
```

Every evaluation boundary now returns an `EvaluationResult` tree. Raw loader metrics remain under
their loader; each stage has one task-derived `score`. The task continues to use
`post_metrics_to_value` to define multi-loader aggregation.

```python
def post_metrics_to_value(self, result, *, stage):
    if stage is Stage.VAL:
        return (result["in_domain"]["mae"] + result["ood"]["mae"]) / 2
    if stage is Stage.TEST:
        return result["benchmark"]["mae"]
    return result["loss"]
```

## Breaking API migration

All evaluation APIs, including `RunningAgent.evaluate`, `evaluate_runner`, and
`qPipeline.evaluate_once`, return `EvaluationResult`. Flat metric dictionaries and `prefix`
arguments are removed. `return_outputs=True` attaches outputs to `LoaderEvaluation.outputs`.

Evaluation event contexts expose the same structure:

```python
evaluation = context.evaluation
val_score = evaluation.target_value("val_metric")
for model in evaluation.models:
    for stage in model.stages:
        for loader in stage.loaders:
            print(model.variant, stage.stage, loader.name, loader.metrics)
```

YAML target names remain concise and stable: `train_metric`, `val_metric`, `test_metric`,
`ema_val_metric`, and `ema_test_metric`. They resolve only to structured stage scores; raw metric
names never participate in checkpointing, early stop, or scheduling. A valid target without a score
for a boundary causes that consumer to skip the boundary.

## Evaluation logging

Evaluation logging separates control decisions from metric rows. The control summary contains the
canonical target, best-model state, NewBest delta, and a `best requested` marker when applicable.
The metric table contains one `train:interval` row for training aggregation followed by one row per
validation, test, and EMA loader. It never reconstructs loader identity from flat metric keys.

The default metrics artifact is `metrics.jsonl`; CSV logging has been removed. Each line is a
structured `train_batch`, `evaluation`, or successful `checkpoint_saved` event. Checkpoint paths
are also emitted in the separate readable `[Checkpoint Saved]` INFO record.

`qPipeline.infer()` and `evaluate_once()` remain single-loader actions. If their dataloader argument is
omitted while `task.test_loader` is a mapping, they fail explicitly; pass one selected loader instead.
