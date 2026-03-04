from __future__ import annotations

"""
Handle batch & epoch callbacks with live progress bar.
Supports:
1. Rich Live Displayer (if installed)
2. Tqdm Progress Bar (if installed)
3. Standard Console Logger (always available)


Features:
- auto downgrade strategy
- pbar suspend/recover during evaluation
- handly refresh rich progressbar and table if needed
"""

import sys
import time
from typing import Any, Dict, List, Literal, Optional, Protocol

from .types import EventContext

# Check for optional dependencies
try:
    import rich
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

__all__ = ["ProgressTracker", "RunnerListener"]


class ProgressStrategy(Protocol):
    """Common protocol for progress rendering strategies."""

    def on_epoch_start(self, context: EventContext) -> None: ...

    def on_progress_tick(self, context: EventContext) -> None: ...

    def on_table_update(self, context: EventContext) -> None: ...

    def on_epoch_end(self, context: EventContext) -> None: ...

    def on_run_end(self) -> None: ...

    def on_eval_start(self, context: EventContext) -> None: ...

    def on_eval_end(self, context: EventContext) -> None: ...


# Only define Rich-specific classes if rich is available
if HAS_RICH:

    class CustomBarColumn(BarColumn):
        """A custom progress bar column using emojis."""

        def render(self, task) -> Text:
            """Render the progress bar."""
            completed = int(task.completed / (task.total or 1) * self.bar_width)
            remaining = self.bar_width - completed - 1

            bar_str = (
                f"[#1BBAE9]{'\U0001f63c' * completed}"
                f"[#ff00d7]\U0001f638"
                f"[white]{'\U0001f41f' * max(0, remaining)}"
            )
            return Text.from_markup(bar_str)

    class CustomETAColumn(ProgressColumn):
        """A custom column to display Estimated Time of Arrival."""

        def render(self, task) -> Text:
            """Render the ETA."""
            if task.completed == 0:
                return Text("ETA: -", style="italic blue")

            custom_eta = task.elapsed * (task.total / task.completed - 1.0)
            return Text(f"ETA: {custom_eta:.1f}s", style="italic blue")

    class LiveDisplayer:
        """Manages the Rich Live display for training progress."""

        def __init__(self, enable: bool = True):
            self.enable = enable and HAS_RICH
            self.train_task_id = None
            self.eval_task_id = None
            self.console = Console() if self.enable else None
            self.progress = None
            self.layout = None
            self.live = None
            self.is_started = False
            self._has_table_layout = False

            if self.enable:
                self._init_live()

        def _init_live(self):
            self.progress = Progress(
                TextColumn("[blue]{task.description}", justify="right"),
                SpinnerColumn(),
                CustomBarColumn(bar_width=16),
                TextColumn("[bright white]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[green]{task.completed}/{task.total} batches"),
                TextColumn("•"),
                TextColumn("[dim]Elapsed: {task.elapsed:.1f}s"),
                CustomETAColumn(),
                console=self.console,
                transient=True,
            )
            self.layout = Layout()
            self.layout.split_column(
                Layout(self.progress, name="progress", size=6),
            )
            self._has_table_layout = False
            self.live = Live(self.layout, auto_refresh=False, transient=False)

        def _ensure_table_layout(self):
            if not self.layout or self._has_table_layout:
                return

            progress_layout = self.layout["progress"]
            self.layout.split_column(
                progress_layout,
                Layout(name="table", ratio=1),
            )
            self._has_table_layout = True

        def reset_progressbar(self, num_batches: int, epoch_idx: int, max_epochs: int):
            if not self.enable or not self.progress:
                return

            desc = f"[cyan]Epoch {epoch_idx}/{max_epochs}[/]"
            if self.train_task_id is None or self.train_task_id not in self.progress.task_ids:
                self.train_task_id = self.progress.add_task(desc, total=num_batches)
            else:
                self.progress.update(self.train_task_id, total=num_batches, description=desc, completed=0)

            if self.eval_task_id is not None and self.eval_task_id in self.progress.task_ids:
                self.progress.remove_task(self.eval_task_id)
                self.eval_task_id = None

            if self.live and self.is_started:
                self.live.refresh()

        def start_eval_progressbar(self, num_batches: int, eval_stage: str = "eval"):
            if not self.enable or not self.progress:
                return

            stage_label = eval_stage.capitalize() if eval_stage else "Eval"
            desc = f"[magenta]Eval ({stage_label})[/]"
            if self.eval_task_id is None or self.eval_task_id not in self.progress.task_ids:
                self.eval_task_id = self.progress.add_task(desc, total=num_batches)
            else:
                self.progress.update(self.eval_task_id, total=num_batches, description=desc, completed=0)

            if self.live and self.is_started:
                self.live.refresh()

        def end_eval_progressbar(self):
            if not self.enable or not self.progress:
                return

            if self.eval_task_id is not None and self.eval_task_id in self.progress.task_ids:
                self.progress.remove_task(self.eval_task_id)
                self.eval_task_id = None

            if self.live and self.is_started:
                self.live.refresh()

        def update_progress(self, completed: int, total: int):
            """Update only the progress bar position."""
            if not self.enable or not self.progress or self.train_task_id is None:
                return

            clamped_total = max(1, int(total or 1))
            clamped_completed = max(0, min(int(completed), clamped_total))
            self.progress.update(self.train_task_id, total=clamped_total, completed=clamped_completed)

            if self.live and self.is_started:
                self.live.refresh()

        def update_table(
            self,
            batch_metrics: Dict[str, Any],
            avg_bank: Dict[str, Any],
            lr: Optional[float],
        ):
            if not self.enable or not self.progress or self.train_task_id is None:
                return

            # Create metrics table
            table = Table(box=box.HORIZONTALS, show_header=True, padding=(0, 1))
            table.add_column("Metric", style="dim")
            table.add_column("Step", style="cyan")
            table.add_column("Avg", style="green")

            # Add metric rows
            for k, v in batch_metrics.items():
                if isinstance(v, (int, float)):
                    avg_val = avg_bank.get(k, "")
                    avg_str = f"{avg_val:.8f}" if isinstance(avg_val, (int, float)) else ""
                    table.add_row(k, f"{v:.8f}", avg_str)

            # Add learning rate if available
            if lr is not None:
                table.add_row("LR", f"{lr:.8f}", "")

            self._ensure_table_layout()
            self.layout["table"].update(table)
            if self.live and self.is_started:
                self.live.refresh()

        def update_eval(self, completed: int, total: int, eval_stage: str = "eval"):
            if not self.enable or not self.progress:
                return

            if self.eval_task_id is None or self.eval_task_id not in self.progress.task_ids:
                self.start_eval_progressbar(total, eval_stage)

            if self.eval_task_id is None:
                return

            stage_label = eval_stage.capitalize() if eval_stage else "Eval"
            desc = f"[magenta]Eval ({stage_label})[/]"
            clamped_total = max(1, int(total or 1))
            clamped_completed = max(0, min(int(completed), clamped_total))
            self.progress.update(
                self.eval_task_id,
                total=clamped_total,
                completed=clamped_completed,
                description=desc,
            )
            if self.live and self.is_started:
                self.live.refresh()

        def start(self):
            if self.enable and self.live and not self.is_started:
                self.live.start()
                self.is_started = True

        def stop(self):
            if self.enable and self.live and self.is_started:
                self.live.stop()
                # Ensure any buffered console output is flushed so prints during
                # evaluation are visible and not lost to live rendering.
                if self.console and hasattr(self.console, "file"):
                    self.console.file.flush()

                # Give live thread a moment to fully stop and then print a newline
                time.sleep(0.05)
                if self.console:
                    # Emit a newline using the same Console so subsequent
                    # stdout prints appear below the live-rendered area.
                    self.console.print("")
                    sys.stdout.flush()

                self.is_started = False
                self._has_table_layout = False
                self.train_task_id = None
                self.eval_task_id = None

    class RichProgress:
        """Strategy for Rich Live logging."""

        def __init__(self, print_freq: int = 10):
            self.print_freq = print_freq
            self.displayer = LiveDisplayer()
            self.current_stage = None
            self._last_batch_state = None
            self._eval_active = False

        def on_epoch_start(self, context: EventContext):
            self.current_stage = context.stage or "train"
            self._eval_active = False
            # `max_epochs` moved out of RunningState and is injected into EventContext
            self.displayer.reset_progressbar(context.total_batches, context.state.epoch, context.max_epochs)
            self.displayer.start()

        def on_progress_tick(
            self,
            context: EventContext,
        ):
            stage = (context.stage or self.current_stage or "train").lower()
            is_eval_stage = stage.startswith("eval") or stage.startswith("val") or stage.startswith("test")

            total = int(context.total_batches or 0)
            completed = min((context.batch_idx or 0) + 1, total) if total > 0 else (context.batch_idx or 0) + 1

            if is_eval_stage:
                self.displayer.update_eval(completed=completed, total=total, eval_stage=stage)
            else:
                self.displayer.update_progress(completed=completed, total=total)

        def on_table_update(self, context: EventContext):
            # Update metrics table (only for training stage to avoid table flickering during eval)
            stage = (context.stage or self.current_stage or "train").lower()
            is_eval_stage = stage.startswith("eval") or stage.startswith("val") or stage.startswith("test")

            if not is_eval_stage:
                # Save last batch state for potential recovery after evaluation
                total = int(context.total_batches or 0)
                completed = min((context.batch_idx or 0) + 1, total) if total > 0 else (context.batch_idx or 0) + 1
                self._last_batch_state = {
                    "batch_metrics": context.batch_metrics,
                    "avg_bank": context.avg_bank,
                    "lr": context.lr,
                    "completed": completed,
                    "total": total,
                }
                self.displayer.update_table(
                    context.batch_metrics,
                    context.avg_bank,
                    context.lr,
                )

        def on_epoch_end(self, context: EventContext):
            self.displayer.stop()

        def on_run_end(self):
            self.displayer.stop()

        def __del__(self):
            self.displayer.stop()

        def on_eval_start(self, context: EventContext):
            self._eval_active = True
            eval_stage = (context.stage or "eval").lower()
            self.displayer.start_eval_progressbar(context.total_batches, eval_stage)

        def on_eval_end(self, context: EventContext):
            self._eval_active = False
            self.displayer.end_eval_progressbar()

            if self._last_batch_state:
                # re-render last batch metrics without advancing the progress
                self.displayer.update_table(
                    self._last_batch_state["batch_metrics"],
                    self._last_batch_state["avg_bank"],
                    self._last_batch_state["lr"],
                )

            if self.displayer.live and self.displayer.is_started:
                self.displayer.live.refresh()


# Tqdm progress if available
if HAS_TQDM:

    class TqdmProgress:
        """Strategy for Tqdm logging."""

        def __init__(self, print_freq: int = 10):
            self.print_freq = print_freq
            self.pbar = None
            self.current_stage = None

        def on_epoch_start(self, context: EventContext):
            self.current_stage = context.stage or "train"
            if self.pbar is not None:
                self.pbar.close()
            desc = f"[{self.current_stage.capitalize()}] Epoch {context.state.epoch}"
            self.pbar = tqdm(total=context.total_batches, desc=desc, leave=False, dynamic_ncols=True)

        def on_progress_tick(
            self,
            context: EventContext,
        ):
            if not self.pbar:
                return

            target_n = min((context.batch_idx or 0) + 1, int(context.total_batches or 0))
            current_n = int(getattr(self.pbar, "n", 0))
            advance_n = max(0, target_n - current_n)
            if advance_n > 0:
                self.pbar.update(advance_n)

        def on_table_update(self, context: EventContext):
            if not self.pbar:
                return

            postfix = {k: f"{v:.4f}" for k, v in context.avg_bank.items() if isinstance(v, (int, float))}
            if context.lr is not None:
                postfix["lr"] = f"{context.lr:.6f}"
            self.pbar.set_postfix(postfix)

        def on_epoch_end(self, context: EventContext):
            if self.pbar:
                self.pbar.close()
                self.pbar = None

        def on_run_end(self):
            if self.pbar:
                self.pbar.close()
                self.pbar = None

        def on_eval_start(self, context: EventContext):
            if self.pbar:
                self._saved_state = {
                    "n": getattr(self.pbar, "n", 0),
                    "total": getattr(self.pbar, "total", None),
                    "desc": getattr(self.pbar, "desc", None),
                }
                # self._saved_state = {"n": 0, "total": None, "desc": None}
                self.pbar.close()
                self.pbar = None

        def on_eval_end(self, context: EventContext):
            state = getattr(self, "_saved_state", None)
            if state and state.get("total") is not None:
                desc = (
                    state.get("desc") or f"[{(self.current_stage or 'train').capitalize()}] Epoch {context.state.epoch}"
                )
                # Use `initial` to restore completed count
                self.pbar = tqdm(total=state["total"], desc=desc, leave=False, dynamic_ncols=True, initial=state["n"])


class PlainProgress:
    """Strategy for Plain logging."""

    def __init__(self, logger: Any, print_freq: int = 10):
        self.logger = logger
        self.print_freq = print_freq
        self.current_stage = None

    def on_epoch_start(self, context: EventContext):
        self.current_stage = context.stage or "train"

    def on_progress_tick(
        self,
        context: EventContext,
    ):
        pass

    def on_table_update(self, context: EventContext):
        stage = context.stage or self.current_stage or "training"
        msg_parts = [f"{k}: {v:.4f}" for k, v in context.batch_metrics.items() if isinstance(v, (int, float))]
        msg = f"[{stage.capitalize()}] Batch {context.batch_idx}/{context.total_batches} " + " ".join(msg_parts)
        if context.lr is not None:
            msg += f" LR: {context.lr:.6f}"
        self.logger.info(msg)

    def on_epoch_end(self, context: EventContext):
        pass

    def on_run_end(self):
        pass

    def on_eval_start(self, context: EventContext):
        pass

    def on_eval_end(self, context: EventContext):
        pass


def resolve_render_mode(requested_mode: Optional[str], has_rich: bool, has_tqdm: bool) -> tuple[str, Optional[str]]:
    """
    Resolve the final render mode based on request and availability.

    Returns:
        A tuple of (resolved_mode, mode_change_message).
        mode_change_message is only provided when input mode differs from output mode.
    """
    input_mode = requested_mode or "auto"

    def _resolve() -> str:
        # 1. Auto-detect if no specific mode requested (or explicitly requested as auto)
        if requested_mode is None or requested_mode == "auto":
            if has_rich:
                return "rich"
            if has_tqdm:
                return "tqdm"
            return "plain"

        # 2. Check if requested mode is available
        if requested_mode == "rich":
            if has_rich:
                return "rich"
            else:
                # Try downgrade to tqdm
                if has_tqdm:
                    return "tqdm"
                return "plain"

        if requested_mode == "tqdm":
            if has_tqdm:
                return "tqdm"
            else:
                return "plain"

        if requested_mode == "plain":
            return "plain"

        # Default fallback
        return "plain"

    resolved_mode = _resolve()
    message = f"Mode {input_mode} -> {resolved_mode}" if input_mode != resolved_mode else None
    return resolved_mode, message


def create_progress_strategy(render_mode: str, logger: Any, print_freq: int) -> ProgressStrategy:
    """Factory to create the appropriate progress strategy."""
    if render_mode == "rich":
        return RichProgress(print_freq)
    elif render_mode == "tqdm":
        return TqdmProgress(print_freq)
    else:
        return PlainProgress(logger, print_freq)


class ProgressTracker:
    """Listener for logging training progress via Rich, Tqdm, or Console."""

    def __init__(
        self,
        logger: Any,
        print_freq: int = 10,
        render_type: Optional[Literal["auto", "rich", "tqdm", "plain"]] = None,
    ):
        """Initialize the LogListener with automatic fallback."""

        self.logger = logger
        self.print_freq = print_freq

        # Resolve render type
        self.render_type, mode_change_message = resolve_render_mode(render_type, HAS_RICH, HAS_TQDM)

        # Create strategy
        self.strategy = create_progress_strategy(self.render_type, self.logger, self.print_freq)

        # Log selection details
        if mode_change_message:
            if render_type in (None, "auto"):
                self.logger.info(mode_change_message)
            else:
                self.logger.warning(mode_change_message)

    def on_epoch_start(self, context: EventContext):
        """Callback at the start of an epoch."""
        self.strategy.on_epoch_start(context)

    def on_progress_tick(self, context: EventContext):
        """Callback when a progress update should be rendered."""
        self.strategy.on_progress_tick(context)

    def on_table_update(self, context: EventContext):
        """Callback when the metrics table should be updated."""
        if hasattr(self.strategy, "on_table_update"):
            self.strategy.on_table_update(context)

    def on_epoch_end(self, context: EventContext):
        """Callback at the end of an epoch."""
        self.strategy.on_epoch_end(context)

        # Determine epoch number
        display_epoch = context.state.epoch

        self.logger.info(f"--- Epoch {display_epoch} Results ---")

        # Organize and log results
        if context.state.current_train_metric is not None or context.state.current_train_loss is not None:
            train_msg = []
            if context.state.current_train_loss is not None:
                train_msg.append(f"loss: {context.state.current_train_loss:.6f}")
            if context.state.current_train_metric is not None:
                train_msg.append(f"metric: {context.state.current_train_metric:.6f}")
            self.logger.info(f"[train] {' '.join(train_msg)}")

        if context.state.current_val_metric is not None:
            self.logger.info(f"[val] metric: {context.state.current_val_metric:.6f}")

        if context.state.current_test_metric is not None:
            self.logger.info(f"[test] metric: {context.state.current_test_metric:.6f}")

        # Also log any extra eval results if present in context
        if context.eval_results:
            for k, v in context.eval_results.items():
                # Avoid double logging standard metrics if possible, or just log everything in debug
                pass

    def on_eval_start(self, context: EventContext):
        """Callback before evaluation starts."""
        if hasattr(self.strategy, "on_eval_start"):
            try:
                self.strategy.on_eval_start(context)
            except Exception:
                pass

    def on_eval_end(self, context: EventContext):
        """Callback after evaluation ends."""
        if hasattr(self.strategy, "on_eval_end"):
            try:
                self.strategy.on_eval_end(context)
            except Exception:
                pass

    def on_run_end(self):
        """Callback at the end of the entire run."""
        self.strategy.on_run_end()
