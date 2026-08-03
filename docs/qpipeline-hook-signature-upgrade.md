# qpipeline qTask Hook Signature Upgrade

`post_metrics_to_value` can now receive the execution stage that produced its metrics.

## Recommended implementation

```python
from collections.abc import Mapping
from typing import Any

from qqtools.plugins.qpipeline import Stage


def post_metrics_to_value(self, result: Mapping[str, Any], *, stage: Stage) -> float:
    if stage is Stage.VAL:
        return result["mae"]
    return result["loss"]
```

`Stage` is the shared qpipeline execution enum with `TRAIN`, `VAL`, and `TEST` values.

## Compatibility

Existing tasks remain valid without a signature migration:

```python
def post_metrics_to_value(self, result) -> float:
    return result["mae"]
```

### Required rename for legacy aliases

The framework now uses names to bind this hook. A legacy implementation such as
`def post_metrics_to_value(self, metrics)` must be renamed to
`def post_metrics_to_value(self, result)`. It remains a one-parameter hook; adding `stage` is
optional. The `result` value has type `Mapping[str, Any]`.

When `stage` is declared, it may be positional or keyword-only and may appear before or after
`result`. The framework binds hook arguments by name during Runner initialization, then uses the
bound resolver without repeated signature inspection. The only accepted hook parameter names are
`result` and `stage`.

Every parameter of this hook must be explicit. The framework rejects unknown parameters,
positional-only parameters, `*args`, and `**kwargs` before training or evaluation begins.

## Standalone evaluation

`evaluate_runner(...)` and `qPipeline.evaluate_once(...)` accept keyword-only
`stage: Stage = Stage.TEST`. `prefix` only names returned metrics; it never determines the stage.

```python
pipeline.evaluate_once(prefix="benchmark_a", stage=Stage.VAL)
```

The public hook contract remains `-> float`. For `Stage.TEST` only, returning `None` omits the
canonical `<prefix>_metric` while preserving the raw test metrics.
