"""JSON Lines metrics logging for qpipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from ..events import ProgressEventContext
from ..events.types import _EvalEndInternalContext
from .common import _is_periodic_trigger
from .types import RunConfig


class MetricsJsonlLogger:
    """Append one structured qpipeline event per JSON Lines record."""

    def __init__(self, file_path: Union[Path, str]) -> None:
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    def write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("MetricsJsonlLogger is closed")
        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(record), default=_json_default) + "\n")

    def close(self) -> None:
        self._closed = True

    def abort(self) -> None:
        self._closed = True


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (tuple, list)):
            value = value[0]
        item = getattr(value, "item", None)
        result[key] = item() if callable(item) else value
    return result


class MetricsJsonlListener:
    """Write qpipeline batch and evaluation events as machine-readable JSON Lines."""

    def __init__(
        self,
        logger: MetricsJsonlLogger,
        run_config: RunConfig,
        log_granularity: List[str],
    ) -> None:
        self.logger = logger
        self.config = run_config
        self.log_granularity = log_granularity

    def on_eval_end(self, context: _EvalEndInternalContext) -> None:
        state = context.runner.run_state
        self.logger.write(
            {
                "event": "evaluation",
                "epoch": state.epoch,
                "global_step": state.global_step,
                "evaluation": context.evaluation.to_dict(),
            }
        )

    def on_train_batch_end(self, context: ProgressEventContext) -> None:
        if "eval" in self.log_granularity:
            is_epoch_end = context.batch_idx == context.total_batches - 1
            if _is_periodic_trigger(
                run_mode=self.config.run_mode,
                interval=self.config.eval_interval,
                global_step=context.runner.run_state.global_step,
                epoch=context.runner.run_state.epoch,
                is_epoch_end=is_epoch_end,
            ):
                return
        state = context.runner.run_state
        self.logger.write(
            {
                "event": "train_batch",
                "epoch": state.epoch,
                "global_step": state.global_step,
                "metrics": _scalar_metrics(context.batch_metrics),
            }
        )
