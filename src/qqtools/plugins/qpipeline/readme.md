# qpipeline

qpipeline builds training and evaluation workflows on top of the project's torch
utilities. For configuration fields and optimizer-step semantics, see the
[configuration guide](../../../../docs/qConfig_en.md). For evaluation data
structures and multi-loader migration, see the
[structured evaluation spec](../../../../docs/spec/qpipeline-multi-eval-loader-upgrade.md).

## Command-line arguments

Load a configuration with `python train.py --config /path/to/config.yml` and
`prepare_cmd_args()`. A `patch` callback can register additional parser arguments:

```python
import argparse

from qqtools.plugins.qpipeline import prepare_cmd_args


def patch(parser: argparse.ArgumentParser):
    parser.add_argument("--file", type=str)
    return parser


args = prepare_cmd_args(patch=patch)
file = args.file
```

Explicit parser arguments, including those registered by `patch`, are consumed
by argparse first. Remaining dotted options such as `--a.b.c value` or
`--a.b.c=value` override the configuration after YAML and explicit arguments are
merged. Dotted overrides have the highest precedence. Flag-only dotted options
are allowed only for boolean-safe targets. Values beginning with `-` must use
`=`, for example `--task.threshold=-0.5`.

```bash
python train.py \
  --config configs/train.yaml \
  --task.dataloader.eval_batch_size 32 \
  --task.val_split val_ood \
  --runner.fast_dev_run
```

## Pipeline initialization and caches

Pass prepared task and model instances, or subclass `qPipeline` and implement
`prepare_task(args)` and `prepare_model(args)`:

```python
from qqtools.plugins.qpipeline import qPipeline


# prepare_model and prepare_task are application-defined functions.
pipe = qPipeline(args, mode="train", task=prepare_task(args), model=prepare_model(args))


class MyPipeline(qPipeline):
    prepare_model = staticmethod(prepare_model)
    prepare_task = staticmethod(prepare_task)


pipe = MyPipeline(args, mode="train")
```

Both preparation methods receive the configuration arguments. The pipeline
places the model on the configured device and invokes `task.to(args.device)`
when the task implements that optional method.

`pipe.regist_extra_ckp_caches({"key": value})` adds run-wide information to
checkpoints. `pipe.regist_middleware(callback)` invokes a callback with the
pipeline; it also accepts a list or tuple of callbacks.

## Training logs and metrics

Run termination, evaluation, and epoch completion are distinct events. They may
refer to the same metrics without representing the same boundary.

### Structured metrics JSONL

On rank zero, enabled metrics logging writes `metrics.jsonl` under the runner's
save directory. Each line is an independent `train_batch`, `evaluation`, or
successful `checkpoint_saved` event. `log_granularity` selects evaluation and/or
batch records; evaluation logging is enabled by default. If metrics logging is
disabled, absence of a JSONL record does not prove an evaluation did not occur.

For machine consumption, use the structured `evaluation` records when enabled
rather than scraping display tables. They include `epoch`, `global_step`, and
an `evaluation` tree preserving model, stage, and loader identities and stage
scores. The [structured evaluation spec](../../../../docs/spec/qpipeline-multi-eval-loader-upgrade.md)
defines that data model and target selection. CSV metrics logging is no longer
supported.

### Readable evaluation and checkpoint output

The current display uses `[Evaluation]` for the control summary and
`[Evaluation Metrics]` for the metric table. The summary includes epoch, step,
learning rate (`n/a` when unavailable), the target, and available best-model
state. A new best includes its delta and a `Checkpoint: best requested` marker.
That marker is a request, not proof that a checkpoint was saved; successful
saves produce a separate `[Checkpoint Saved]` INFO record and, when enabled,
a `checkpoint_saved` JSONL event.

The metric table contains available `train:interval` metrics, stage scores, and
per-loader metrics, including EMA variants. Older `[Eval Summary]` and
`[Eval Summary Table]` examples do not describe the current display.

Periodic evaluation uses `eval_interval` in epochs for `run_mode=epoch` and
completed optimizer updates for `run_mode=step`. Step mode can evaluate several
times within one epoch. Completion actions can also request evaluation at the
final successful boundary; see [training completion actions](#training-completion-actions).
If the runner detects NaN training loss before a selected evaluation, it fails
with `nan_detected` instead of executing that evaluation.

### Epoch result summaries

A committed epoch emits `--- Epoch N Results ---`, a `[train]` line when training
metrics are available, and `[val]` / `[test]` lines with explicit provenance:

| Source | Meaning |
| --- | --- |
| `current_eval` | Evaluation ran at this epoch-end boundary and produced the score |
| `latest_eval_reuse` | The score is cached from an earlier evaluation |
| `missing` | No corresponding score is available; the value is `n/a` |

The distinction concerns the epoch-end boundary, not whether evaluation ran
somewhere earlier in the same epoch. A mid-epoch evaluation can therefore be
followed by an epoch summary marked `latest_eval_reuse`. An epoch summary is
not evidence of a new evaluation.

Evaluation at an epoch-end boundary precedes the epoch result summary. Reaching
`max_steps` mid-epoch can terminate training without an epoch summary. Epoch
numbers are zero-based internal counters; the committed summary identifies the
epoch just completed.

### Run terminal events

The outer `train_runner` boundary classifies managed run termination and emits
one terminal record on that path:

| Status | Reasons |
| --- | --- |
| `finished` | `max_steps`, `max_epochs`, `early_stop` |
| `stopped` | `user_interrupt` |
| `failed` | `oom`, `exception`, `nan_detected`, `logger_failure` |

Readable text has the form `Training <status>: reason=<reason>`. Normal returned
results include `terminal_event` with `status`, `reason`, `text`, `epoch`, and
`step`; when an exception is attached, it includes `exception_type`. The
`early_stopped` result is true only for `early_stop`, not interruption or other
termination reasons. Exception paths may emit the event and then re-raise,
so they do not necessarily return a result dictionary.

The terminal payload does not include the original exception message or
traceback; other error logging can include diagnostic information. A hard
process kill or failure before the managed run boundary does not guarantee a
terminal record. These records describe the training run, not qexp Task/Attempt
authority, and are not terminal events in the metrics JSONL stream.

## Training completion actions

`runner.completion` can request normal final-boundary work when a successful
training conclusion does not match the periodic intervals:

```yaml
runner:
  completion:
    eval: true
    save: true
```

Both values default to `false`. `eval` performs the normal evaluation flow,
including validation listeners and best-model tracking. `save` writes a normal
regular checkpoint, not a weights-only or best-checkpoint export. Actions run
only after a processed boundary that finishes through a limit or early stop;
they do not run after interruption, NaN detection, or an exception.

## Tasks and metric aggregation

A task connects dataset and model behavior through runner-defined interfaces.
Subclass `qTaskBase`, call its initializer, and prepare `train_loader`,
`val_loader`, and `test_loader`. Validation and test loaders may be `None`, one
DataLoader, or a non-empty mapping of names to DataLoaders, as described in the
structured evaluation spec.

Required methods are `batch_forward`, `batch_metric`, `batch_loss`, and
`post_metrics_to_value`:

```python
from collections.abc import Mapping
from typing import Any

from qqtools.plugins.qpipeline import Stage, qTaskBase


class MyTask(qTaskBase):
    def __init__(self, train_loader, val_loader=None, test_loader=None):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def batch_forward(self, model, batch_data):
        # Return the model output mapping for this application's data.
        raise NotImplementedError

    def batch_metric(self, out, batch_data):
        # Return {metric_name: (metric_value, sample_count)}.
        raise NotImplementedError

    def batch_loss(self, out, batch_data):
        # Return {"loss": (loss_value, sample_count)}.
        raise NotImplementedError

    def post_metrics_to_value(self, result: Mapping[str, Any], *, stage: Stage) -> float:
        # This example assumes one unnamed evaluation loader producing "mae".
        if stage in (Stage.VAL, Stage.TEST):
            return result["mae"]
        return result["loss"]
```

### Metric hook signature

`Stage` is the shared enum with `TRAIN`, `VAL`, and `TEST` values. The hook receives
the stage's metric mapping and returns its task-derived score, separately from
raw loader metrics. Named loaders supply nested mappings; consult the structured
evaluation spec for aggregation examples.

The framework binds `post_metrics_to_value` by parameter name once during runner
initialization, before training or evaluation. `result` is required; `stage` is
optional. A hook declared as `post_metrics_to_value(self, result)` remains
supported. Rename parameter aliases such as `metrics` to `result`.

When declared, `stage` may be positional-or-keyword or keyword-only and may appear
before or after `result`. The only accepted parameter names are `result` and
`stage`; parameters must be explicit. Unknown names, positional-only parameters,
`*args`, and `**kwargs` are rejected. Runtime calls reuse the bound resolver
without inspecting the signature again.

The public hook contract is `-> float`. For `Stage.TEST`, returning `None` omits
the stage score while retaining raw test metrics.

`evaluate_runner(...)` and `qPipeline.evaluate_once(...)` accept keyword-only
`stage: Stage = Stage.TEST` and return an `EvaluationResult` for successful
evaluation. For example, `pipeline.evaluate_once(stage=Stage.VAL)` selects
validation semantics. These are single-loader actions: pass a selected loader
explicitly when the task's test loaders are a mapping.

### Batch and task lifecycle

The runner moves batch data to the device, applies `pre_batch_forward`, calls
`batch_forward`, then applies `post_batch_forward`. It collects metrics and,
when training, loss. Aggregated metrics feed `post_metrics_to_value`; resulting
stage scores drive the configured checkpoint, scheduler, and early-stop targets.
Override the pre/post methods when the application needs input or output adaptation.

Supported task lifecycle hooks are declared on `qTaskBase`, listed in
`OPTIONAL_METHODS`, and use fixed typed contexts:

| Hook | Context | Available fields |
| --- | --- | --- |
| `on_epoch_start` | `TaskEpochStartContext` | `epoch`, `global_step`, `total_batches` |
| `on_train_batch_end` | `TaskTrainBoundaryContext` | `epoch`, `global_step`, `batch_index`, `total_batches`, `did_optimizer_step`, `lr`, `batch_metrics` |
| `on_validation_end` | `TaskValidationContext` | `epoch`, `global_step`, `evaluation`, `is_best`, `previous_best`, `lr` |
| `on_epoch_end` | `TaskEpochEndContext` | `completed_epoch`, `global_step`, `epoch_metrics` |
| `on_early_stop` | `StopCommittedFact` | `source`, `message`, `epoch`, `global_step` |

Signatures use `def on_hook(self, context: ContextType) -> None`. Contexts are
frozen typed snapshots with read-only metric mappings. They do not expose the
runner, `RunningState`, or a generic signal. The early-stop hook observes an
already-committed stop decision.

Task hooks may observe context and perform task-owned boundary work, but cannot
rewrite the main loop, request checkpoints, or register dynamic events. Methods
outside the declared lifecycle surface are not supported lifecycle hooks.

## Runner extension boundaries

Task lifecycle listeners react within uncommitted boundaries without controlling
training. Observers consume committed facts. `RunnerHooks` provides single-slot
lifecycle capabilities frozen during runner composition. These contracts are
distinct; registration, removal, or replacement is not allowed during a run.

The standard `CheckpointPlugin` occupies `after_validation`, `boundary_cursor`,
and `after_epoch_commit`. It calls `CheckpointManager`, writes `best_ckp_file`,
and projects checkpoint text/JSONL records. `RunningAgent` does not receive
checkpoint paths and has no command handler or checkpoint-saved notification.
DDP validates that all ranks have identical frozen hook plans before training.

## DDP evaluation deduplication

`runner.ddp_eval_dedup` handles DDP eval/infer padding that repeats logical samples
to align per-rank step counts. qPipeline automatically creates an execution view
for a one-step tail mismatch without changing the user's loader. Synthetic
occurrences are removed before gathering metrics and outputs.

This is not general-purpose deduplication. Intentional sampler repetition is not
guaranteed to survive as distinct outputs. With deduplication disabled, mismatched
rank step counts fail before forward. Automatic padding requires an observable
map-style sampler or batch sampler with stable sample identities; unsupported
custom loaders fail diagnostically rather than entering a collective.

## Automatic weight-decay exclusions

When `optim.optimizer_params.weight_decay > 0`, optimizer preparation traverses
the model to discover parameters excluded from weight decay. Implement one of
these methods on an `nn.Module`:

- `no_decay() -> List[str]`: local parameter names; traversal continues into children.
- `no_decay_deep() -> List[str]`: names including dotted descendant paths; traversal
  stops at this module, which owns the declaration for its entire subtree.

```python
from typing import List

from torch import nn


class MyLayerNorm(nn.LayerNorm):
    def no_decay(self) -> List[str]:
        return ["weight", "bias"]
```

For a backbone, `no_decay_deep()` might return `['cls_token', 'pos_embed',
'blocks.0.norm1.weight', 'blocks.0.norm1.bias']`; the list must cover its intended
subtree exclusions. If both methods exist, only `no_decay_deep` applies.

Discovery is a no-op when weight decay is zero. Frozen parameters are excluded
from optimizer groups, and unknown parameter names warn and are skipped. When
exclusions exist, the regular and exempt groups share other optimizer settings
but use configured weight decay and zero respectively. Without exclusions, the
optimizer retains a single group.

## Configuration and unfinished design notes

`$BASE` inherits configuration files; `log_dir` selects the logs, checkpoints,
and metrics location; `ckp_file` selects a checkpoint. See the configuration guide
for supported fields and dtype/EMA settings. Applications needing custom batch
dtype conversion can perform it in `pre_batch_forward`.

The earlier README's EMA recovery and multi-stage optimizer sections were design
notes, not complete usage contracts. Changing optimizer hyperparameters or the
optimizer itself between stages, including task-specific stage settings, still
requires a concrete application design; those notes do not define a public API.
