"""Console formatting for structured qpipeline evaluation results."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import qqtools as qt

from ..events import ValidationEndEventContext
from .evaluation import EvaluationResult


class EvalFormatter:
    """Stateless formatter for structured evaluation summaries."""

    @staticmethod
    def _to_scalar_if_possible(value: Any) -> Any:
        try:
            return qt.ensure_scala(value)
        except Exception:
            return value

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _format_metric_value(cls, value: Any, metric_name: str = "", precision: int = 4) -> str:
        scalar = cls._to_scalar_if_possible(value)
        if cls._is_numeric(scalar):
            suffix = "s" if metric_name.endswith("time") else ""
            return f"{float(scalar):.{precision}f}{suffix}"
        return str(scalar)

    @classmethod
    def _format_learning_rate(cls, lr: Optional[float]) -> str:
        if lr is None:
            return "n/a"
        return cls._format_metric_value(lr, "lr", precision=6)

    @staticmethod
    def _format_delta(delta: Optional[float]) -> str:
        if delta is None:
            return "n/a"
        return f"{delta:+.4f} {'↑' if delta > 0 else '↓' if delta < 0 else '→'}"

    @staticmethod
    def _ordered_metric_names(metrics: Dict[str, Any]) -> List[str]:
        preferred = ["score", "metric", "mae", "mse", "loss"]
        ordered = [name for name in preferred if name in metrics]
        ordered.extend(sorted(name for name in metrics if name not in ordered))
        return ordered

    @staticmethod
    def _render_text_table(headers: List[str], rows: List[List[str]]) -> List[str]:
        widths = [len(column) for column in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        return [
            "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
            "  ".join("-" * width for width in widths),
            *(
                "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
                for row in rows
            ),
        ]

    @classmethod
    def format_evaluation(
        cls,
        evaluation: EvaluationResult,
        *,
        epoch: int,
        step: int,
        target_key: str,
        is_best: bool = False,
        previous_best: Optional[Dict[str, Any]] = None,
        best_model_tracker: Optional[Any] = None,
        lr: Optional[float] = None,
        color_new_best: bool = True,
    ) -> Tuple[List[str], bool, List[str], bool]:
        """Render control state and raw loader measurements without flat metric keys."""
        target_val = evaluation.target_value(target_key)
        tracker = best_model_tracker
        best_metric = cls._to_scalar_if_possible(getattr(tracker, "best_metric", None))
        best_epoch = getattr(tracker, "best_epoch", None)
        best_step = getattr(tracker, "best_step", None)
        lines = [f"\n[Evaluation] Epoch: {epoch} | Step: {step} | LR: {cls._format_learning_rate(lr)}"]
        if target_val is None:
            lines.append(f"  Target: {target_key} not produced; best state unchanged")
        else:
            lines.append(f"  Target: {target_key} = {cls._format_metric_value(target_val, target_key, 6)}")
        if best_metric is not None:
            lines.append(
                f"  Best: {target_key} = {cls._format_metric_value(best_metric, target_key, 6)} "
                f"@ epoch {best_epoch}, step {best_step}"
            )
        if is_best:
            previous = previous_best.get("metric") if previous_best else None
            delta = None
            if cls._is_numeric(target_val) and cls._is_numeric(previous):
                delta = float(target_val) - float(previous)
            label = "[green]NewBest[/green]" if color_new_best else "NewBest"
            lines.append(f"  {label}: previous best delta {cls._format_delta(delta)}")
            lines.append("  Checkpoint: best requested")

        rows: List[Tuple[str, Dict[str, Any]]] = []
        metric_names = set()
        if evaluation.training is not None:
            training_metrics = {**evaluation.training.metrics, "score": evaluation.training.score}
            rows.append(("train:interval", training_metrics))
            metric_names.update(training_metrics)
        for model in evaluation.models:
            variant_prefix = "" if model.variant == "standard" else f"{model.variant}_"
            for stage in model.stages:
                stage_label = f"{variant_prefix}{stage.stage.value}"
                rows.append((f"{stage_label}:score", {"score": stage.score}))
                metric_names.add("score")
                for loader in stage.loaders:
                    loader_label = stage.stage.value if loader.name is None else f"{stage.stage.value}:{loader.name}"
                    rows.append((f"{variant_prefix}{loader_label}", dict(loader.metrics)))
                    metric_names.update(loader.metrics)

        headers = ["Scope / loader", *cls._ordered_metric_names({name: None for name in metric_names})]
        table_lines = ["\n[Evaluation Metrics]"]
        if rows:
            table_rows = [
                [label, *(cls._format_metric_value(metrics[name], name) if name in metrics else "-" for name in headers[1:])]
                for label, metrics in rows
            ]
            table_lines.extend(cls._render_text_table(headers, table_rows))
        else:
            table_lines.append("(no evaluation loaders executed)")
        return lines, is_best and color_new_best, table_lines, False


class EvalSummaryListener:
    """Listener that logs formatted evaluation summary on validation end."""

    def __init__(
        self,
        logger: Any,
        target_key: str = "val_metric",
        color_new_best: bool = True,
    ) -> None:
        self.logger = logger
        self.target_key = target_key
        self.color_new_best = color_new_best

    def on_validation_end(self, context: ValidationEndEventContext) -> None:
        state = context.runner.run_state
        tracker = context.best_model_tracker or SimpleNamespace(
            best_metric=getattr(state, "best_monitored_metric", None),
            best_epoch=getattr(state, "best_epoch", None),
            best_step=getattr(state, "best_step", None),
        )
        summary_lines, summary_has_markup, table_lines, table_has_markup = EvalFormatter.format_evaluation(
            evaluation=context.evaluation,
            epoch=getattr(state, "epoch", 0),
            step=getattr(state, "global_step", 0),
            target_key=self.target_key,
            is_best=bool(context.is_best),
            previous_best=context.previous_best,
            best_model_tracker=tracker,
            lr=context.lr,
            color_new_best=self.color_new_best,
        )
        if summary_has_markup:
            self.logger.info("\n".join(summary_lines), extra={"markup": True})
        else:
            self.logger.info("\n".join(summary_lines))
        if table_has_markup:
            self.logger.info("\n".join(table_lines), extra={"markup": True})
        else:
            self.logger.info("\n".join(table_lines))
