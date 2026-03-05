from typing import Any, Dict, List, Optional, Tuple

import qqtools as qt


class EvalSummaryFormatter:
    """Stateless formatter for evaluation summaries."""

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
    def _format_metric_value(cls, value: Any, metric_name: str = "") -> str:
        scalar = cls._to_scalar_if_possible(value)
        if cls._is_numeric(scalar):
            suffix = "s" if metric_name.endswith("time") else ""
            return f"{float(scalar):.4f}{suffix}"
        return str(scalar)

    @staticmethod
    def _format_delta_with_arrow(delta: Optional[float]) -> Tuple[str, str]:
        if delta is None:
            return "n/a", "-"
        if delta > 0:
            return f"+{delta:.4f}", "\u2191"
        if delta < 0:
            return f"{delta:.4f}", "\u2193"
        return f"{delta:.4f}", "\u2192"

    @staticmethod
    def _parse_target_key(target_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not target_key:
            return None, None, None

        family = "standard"
        body = target_key
        if body.startswith("ema_"):
            family = "ema"
            body = body[len("ema_") :]

        if "_" not in body:
            return None, None, None

        stage, metric_name = body.split("_", 1)
        if stage not in {"train", "val", "test"} or not metric_name:
            return None, None, None
        return family, stage, metric_name

    @classmethod
    def _partition_eval_metrics(
        cls, eval_results: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
        stages = ("train", "val", "test")
        grouped = {
            "standard": {stage: {} for stage in stages},
            "ema": {stage: {} for stage in stages},
        }
        others = {}

        for key in sorted(eval_results.keys()):
            value = eval_results[key]
            family = "standard"
            body = key
            if body.startswith("ema_"):
                family = "ema"
                body = body[len("ema_") :]

            if "_" not in body:
                others[key] = value
                continue

            stage, metric_name = body.split("_", 1)
            if stage not in stages or not metric_name:
                others[key] = value
                continue

            grouped[family][stage][metric_name] = value

        return grouped, others

    @staticmethod
    def _ordered_metric_names(metrics: Dict[str, Any], stage: str) -> List[str]:
        preferred = (
            ["loss", "metric", "batch_time", "mae", "mse"]
            if stage == "train"
            else ["metric", "mae", "mse", "loss", "batch_time"]
        )
        ordered = [name for name in preferred if name in metrics]
        ordered.extend(sorted(name for name in metrics if name not in ordered))
        return ordered

    @classmethod
    def _format_stage_metrics(cls, metrics: Dict[str, Any], stage: str) -> str:
        ordered = cls._ordered_metric_names(metrics, stage)
        return ", ".join(f"{name}: {cls._format_metric_value(metrics[name], name)}" for name in ordered)

    @staticmethod
    def _render_text_table(headers: List[str], rows: List[List[str]]) -> List[str]:
        widths = [len(col) for col in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        lines = []
        lines.append("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
        lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
        lines.extend("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows)
        return lines

    @classmethod
    def _build_best_info(
        cls,
        target_val: Any,
        is_best: bool,
        previous_best: Optional[Dict[str, Any]],
        best_model_tracker: Optional[Any],
        target_mode: str,
    ) -> Dict[str, Any]:
        mode = target_mode
        best_metric = None
        best_epoch = None
        best_step = None
        if best_model_tracker is not None:
            mode = getattr(best_model_tracker, "mode", mode)
            best_metric = getattr(best_model_tracker, "best_metric", None)
            best_epoch = getattr(best_model_tracker, "best_epoch", None)
            best_step = getattr(best_model_tracker, "best_step", None)

        current = cls._to_scalar_if_possible(target_val) if target_val is not None else None
        best = cls._to_scalar_if_possible(best_metric) if best_metric is not None else None
        last_best = cls._to_scalar_if_possible(previous_best.get("metric")) if previous_best else None
        last_best_epoch = previous_best.get("epoch") if previous_best else None
        last_best_step = previous_best.get("step") if previous_best else None

        delta_current_vs_best = None
        if cls._is_numeric(current) and cls._is_numeric(best):
            delta_current_vs_best = float(current) - float(best)

        delta_current_vs_last = None
        if cls._is_numeric(current) and cls._is_numeric(last_best):
            delta_current_vs_last = float(current) - float(last_best)

        status = "NO_TARGET"
        if current is not None:
            if is_best:
                status = "NEW_BEST"
            elif best is None:
                status = "NO_BEST_YET"
            else:
                status = "NOT_BEST"

        return {
            "status": status,
            "mode": mode,
            "current": current,
            "best": best,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "last_best": last_best,
            "last_best_epoch": last_best_epoch,
            "last_best_step": last_best_step,
            "delta_current_vs_best": delta_current_vs_best,
            "delta_current_vs_last": delta_current_vs_last,
        }

    @classmethod
    def _build_hierarchical_summary_lines(
        cls,
        target_key: str,
        target_val: Any,
        best_info: Dict[str, Any],
        grouped: Dict[str, Dict[str, Dict[str, Any]]],
        others: Dict[str, Any],
        color_new_best: bool,
    ) -> Tuple[List[str], bool]:
        lines = [f"\n[Eval Summary] Epoch: {best_info['epoch']}, Step: {best_info['step']}"]
        has_markup = False

        if target_val is not None:
            lines.append(f"  - Primary Target: {target_key}: {cls._format_metric_value(target_val, target_key)}")
        else:
            lines.append(f"  - Primary Target: {target_key}: n/a (not found in eval_results)")

        best_display = (
            cls._format_metric_value(best_info["best"], target_key) if best_info["best"] is not None else "n/a"
        )
        best_epoch_display = best_info["best_epoch"] if best_info["best_epoch"] is not None else "-"
        best_step_display = best_info["best_step"] if best_info["best_step"] is not None else "-"
        tracker_label = f"{'Best Tracker':<12}"
        best_line = (
            f"  - {tracker_label}: epoch: {best_epoch_display}, step: {best_step_display}, {target_key}: {best_display}"
        )

        if best_info["status"] == "NEW_BEST":
            new_best_text = "[green]NewBest[/green]" if color_new_best else "NewBest"
            has_markup = color_new_best
            delta_str, arrow = cls._format_delta_with_arrow(best_info["delta_current_vs_last"])
            if delta_str == "n/a":
                best_line = f"{best_line} ({new_best_text})"
            else:
                best_line = f"{best_line} ({new_best_text}, Delta: {delta_str} {arrow})"
        lines.append(best_line)

        if best_info["status"] == "NEW_BEST" and best_info["last_best"] is not None:
            last_best = cls._format_metric_value(best_info["last_best"], target_key)
            last_indent = " " * 14
            lines.append(
                f"{last_indent}Last: epoch {best_info['last_best_epoch']}, step: {best_info['last_best_step']}, {target_key}: {last_best}"
            )

        target_family, target_stage, _ = cls._parse_target_key(target_key)
        stage_labels = {"val": "Validation", "test": "Testing", "train": "Training"}
        for stage in ("val", "test", "train"):
            standard_metrics = grouped["standard"][stage]
            ema_metrics = grouped["ema"][stage]
            if not standard_metrics and not ema_metrics:
                continue

            lines.append(f"  - {stage_labels[stage]}:")
            if standard_metrics:
                marker = "  (*)" if target_family == "standard" and target_stage == stage else ""
                lines.append(f"    - [Main]     {cls._format_stage_metrics(standard_metrics, stage)}{marker}")
            if ema_metrics:
                marker = "  (*)" if target_family == "ema" and target_stage == stage else ""
                lines.append(f"    - [EMA]      {cls._format_stage_metrics(ema_metrics, stage)}{marker}")

        if others:
            lines.append("  - Others:")
            for key in sorted(others.keys()):
                lines.append(f"    - {key}: {cls._format_metric_value(others[key], key)}")

        return lines, has_markup

    @classmethod
    def _build_table_summary_lines(
        cls,
        eval_results: Dict[str, Any],
        target_key: str,
        target_val: Any,
        best_info: Dict[str, Any],
        grouped: Dict[str, Dict[str, Dict[str, Any]]],
        color_new_best: bool,
    ) -> Tuple[List[str], bool]:
        target_str = cls._format_metric_value(target_val, target_key) if target_val is not None else "n/a"
        lines = [
            f"\n[Eval Summary Table] Epoch: {best_info['epoch']} | Step: {best_info['step']} | Target: {target_key}({target_str})"
        ]
        has_markup = False

        best_display = (
            cls._format_metric_value(best_info["best"], target_key) if best_info["best"] is not None else "n/a"
        )
        best_epoch_display = best_info["best_epoch"] if best_info["best_epoch"] is not None else "-"
        best_step_display = best_info["best_step"] if best_info["best_step"] is not None else "-"
        tracker_label = f"{'Best Tracker':<12}"
        best_line = (
            f"{tracker_label}: epoch: {best_epoch_display}, step: {best_step_display}, {target_key}: {best_display}"
        )
        if best_info["status"] == "NEW_BEST":
            new_best_text = "[green]NewBest[/green]" if color_new_best else "NewBest"
            has_markup = color_new_best
            delta_str, arrow = cls._format_delta_with_arrow(best_info["delta_current_vs_last"])
            if delta_str == "n/a":
                best_line = f"{best_line} ({new_best_text})"
            else:
                best_line = f"{best_line} ({new_best_text}, Delta: {delta_str} {arrow})"
        lines.append(best_line)

        if best_info["status"] == "NEW_BEST" and best_info["last_best"] is not None:
            last_best = cls._format_metric_value(best_info["last_best"], target_key)
            last_label = f"{'Last':<12}"
            lines.append(
                f"{last_label}: epoch {best_info['last_best_epoch']}, step: {best_info['last_best_step']}, {target_key}: {last_best}"
            )

        columns = [
            ("Train", "standard", "train"),
            ("Val", "standard", "val"),
            ("Test", "standard", "test"),
            ("EMA-Val", "ema", "val"),
            ("EMA-Test", "ema", "test"),
        ]

        metric_names = set()
        for _, family, stage in columns:
            metric_names.update(grouped[family][stage].keys())

        target_family, target_stage, target_metric = cls._parse_target_key(target_key)
        table_row_metrics = {name for name in metric_names if not name.endswith("time")}
        # Keep table compact by default, but always include target metric if it exists.
        if target_metric and target_metric in metric_names:
            table_row_metrics.add(target_metric)
        preferred = ["metric", "mae", "mse", "loss"]
        row_order = [name for name in preferred if name in table_row_metrics]
        row_order.extend(sorted(name for name in table_row_metrics if name not in row_order))

        represented_keys = set()
        has_target_marker_in_table = False
        rows = []
        for metric_name in row_order:
            row = [metric_name]
            for _, family, stage in columns:
                value = grouped[family][stage].get(metric_name)
                if value is None:
                    row.append("-")
                    continue

                full_key = f"{stage}_{metric_name}" if family == "standard" else f"ema_{stage}_{metric_name}"
                represented_keys.add(full_key)

                cell = cls._format_metric_value(value, metric_name)
                if family == target_family and stage == target_stage and metric_name == target_metric:
                    cell = f"{cell} (*)"
                    has_target_marker_in_table = True
                row.append(cell)
            rows.append(row)

        headers = ["Metric"] + [col_name for col_name, _, _ in columns]
        if rows:
            lines.extend(cls._render_text_table(headers, rows))
        else:
            lines.append("Metric  Train  Val  Test  EMA-Val  EMA-Test")
            lines.append("------  -----  ---  ----  -------  --------")
            lines.append("(no comparable metrics)")

        others_for_table = []
        has_target_marker_in_others = False
        for key in sorted(eval_results.keys()):
            if key not in represented_keys:
                item = f"{key}: {cls._format_metric_value(eval_results[key], key)}"
                if key == target_key and not has_target_marker_in_table:
                    item = f"{item} (*)"
                    has_target_marker_in_others = True
                others_for_table.append(item)

        if others_for_table:
            lines.append(f"Others: {', '.join(others_for_table)}")

        if has_target_marker_in_table:
            lines.append("(*) marks the primary target cell in table.")
        elif has_target_marker_in_others:
            lines.append("(*) marks the primary target metric in Others.")
        else:
            lines.append("(*) primary target is unavailable in current eval results.")
        return lines, has_markup

    @classmethod
    def format_all(
        cls,
        eval_results: Dict[str, Any],
        epoch: int,
        step: int,
        target_key: str,
        target_mode: str,
        is_best: bool,
        previous_best: Optional[Dict[str, Any]],
        best_model_tracker: Optional[Any],
        color_new_best: bool = True,
    ) -> Tuple[List[str], bool, List[str], bool]:
        target_val = eval_results.get(target_key)
        grouped, others = cls._partition_eval_metrics(eval_results)
        best_info = cls._build_best_info(
            target_val=target_val,
            is_best=is_best,
            previous_best=previous_best,
            best_model_tracker=best_model_tracker,
            target_mode=target_mode,
        )
        best_info["epoch"] = epoch
        best_info["step"] = step

        summary_lines, summary_has_markup = cls._build_hierarchical_summary_lines(
            target_key=target_key,
            target_val=target_val,
            best_info=best_info,
            grouped=grouped,
            others=others,
            color_new_best=color_new_best,
        )
        table_lines, table_has_markup = cls._build_table_summary_lines(
            eval_results=eval_results,
            target_key=target_key,
            target_val=target_val,
            best_info=best_info,
            grouped=grouped,
            color_new_best=color_new_best,
        )
        return summary_lines, summary_has_markup, table_lines, table_has_markup

