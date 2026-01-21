"""
Handle batch & epoch callbacks with live progress bar.
log message formart:
    [epoch] [batch] [train/val/test] [loss] [metric] [lr] [time]
"""

import sys
import traceback
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import rich
from rich import box
from rich.console import Console
from rich.emoji import Emoji
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text


class CustomBarColumn(BarColumn):
    def render(self, task):
        completed = int(task.completed / task.total * self.bar_width)
        remaining = self.bar_width - completed - 1

        # bar = Emoji.replace(
        #     ":smirk_cat:" * completed + ":smile_cat:" + " " * remaining,
        # )
        # bar = "😼"*completed + "😸" + "🐟" * remaining
        bar = Emoji.replace(
            "[#1BBAE9]" + "\U0001f63c" * completed + "[#ff00d7]\U0001f638" + "[white]\U0001f41f" * remaining
        )
        return bar


class CustomETAColumn(ProgressColumn):
    def render(self, task) -> Text:
        custom_eta = task.elapsed * (task.total / (task.completed + 1e-8) - 1.0)
        return Text(f"ETA: {custom_eta:.1f}s", style="italic blue")


class LiveDisplayer:

    def __init__(self, enable=True):

        self.enable = enable
        if self.enable:
            self.init_live()

    def init_live(self):
        self.console = rich.get_console()
        self.progress = Progress(
            TextColumn("[blue]{task.description}", justify="right"),
            SpinnerColumn(),
            CustomBarColumn(
                bar_width=16,
                style="white",
                complete_style="#1BBAE9",
                finished_style="#1BBAE9",
                pulse_style="#1BBAE9",
            ),
            TextColumn("[bright white]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[green]{task.completed}/{task.total} batches"),
            TextColumn("•"),
            TextColumn("[dim]Elapsed: {task.elapsed:.4f}s"),
            CustomETAColumn(),
            console=self.console,
            transient=True,
        )
        self.progress_task = None

        # live
        self.layout = Layout(size=(80, 32), minimum_size=(40, 16))
        self.layout.split_column(
            Layout(self.progress, name="progress", size=3),
            Layout(name="table", ratio=1, size=18),
        )
        self.layout["table"].update("")
        self.live = Live(
            self.layout,
            # console=self.console,
            auto_refresh=False,
            refresh_per_second=0.2,
            screen=False,
            transient=True,
        )
        self.stop()

    def reset_progressbar(self, num_batches, epoch_idx, max_epochs):
        if not self.enable:
            return
        if self.progress_task in self.progress.task_ids:
            self.progress.update(
                self.progress_task,
                total=num_batches,
                description=f"[cyan]Epoch {epoch_idx}/{max_epochs}[/]",
            )
            self.progress.reset(self.progress_task)
        else:
            self.progress_task = self.progress.add_task(f"[cyan]Epoch {epoch_idx}/{max_epochs}[/]", total=num_batches)
        self.layout["progress"].update(self.progress)

    def stop_progressbar(self):
        if not self.enable:
            return
        self.progress.stop()

    def advanceProgressBar(self):
        if not self.enable:
            return
        self.progress.update(self.progress_task, advance=1)

    def updateBatchTable(self, batch_metrics, avgBank, lr):
        """
        Args:
            batch_metrics (dict[str, float]): k-v dict containing metrics for the current batch.
                Keys are metric names (str), values are corresponding values (float).
            avgBank (dict[str, float]): k-v dict containing average values across batches.
                Keys should match those in `batch_metrics`, values are running averages (float).
            lr (float): lr for the current batch
        """
        if not self.enable:
            return
        # table area
        table = Table(
            box=box.HORIZONTALS,
            show_header=True,
            padding=(0, 1),
            min_width=40,
            expand=False,
        )
        table.add_column("Metric", style="none", header_style="dim")
        table.add_column("Step", style="cyan", header_style="dim")
        table.add_column("Avg", style="green", header_style="dim")
        for k, v in batch_metrics.items():
            smooth_avg = f"{avgBank[k]:.8f}" if k in avgBank else ""
            table.add_row(f"step_{k}", f"{v:.8f}", smooth_avg)
        if lr is not None:
            table.add_row("LR", f"{lr:.8f}")
        # table.add_row("BatchTime ", f"{elapsed_time:.4f}s")
        self.layout["table"].update(table)

    def refresh(self):
        if not self.enable:
            return
        if hasattr(self, "live"):
            self.live.refresh()

    def clear_contnt(self):
        self.layout["progress"].update("")
        self.layout["table"].update("")
        self.refresh()

    def start(self):
        if not self.enable:
            return
        if hasattr(self, "live"):
            self.live.start()

    def stop(self):
        if not self.enable:
            return
        self.stop_progressbar()
        if hasattr(self, "live"):
            self.live.stop()


class LogListener:

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        logger,
        print_freq,
        epoch_state=None,
        progress_bar=True,
    ):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.logger = logger
        self.print_freq = print_freq
        self.epoch_state = epoch_state  # reference

        # Input validation
        if self.mode == "train":
            assert epoch_state is not None

        # progress manages
        self.progress_bar = progress_bar
        if self.progress_bar:
            self.displayer = LiveDisplayer()
            self.displayer.start()

    def onBatchEnd(self, batch_idx, num_batches, **batch_state):
        # try:
        if self.mode == "train":
            self.onBatchEndTrain(batch_idx, num_batches, **batch_state)
        else:
            self.onBatchEndVal(batch_idx, num_batches, **batch_state)

    def onBatchEndTrain(self, batch_idx, num_batches, **batch_state):
        epoch_idx = self.epoch_state["epoch"]
        batch_metrics = batch_state["batch_metrics"].copy()
        lr = batch_state.get("lr", None)
        avgBank = batch_state["avgBank"]

        self.displayer.advanceProgressBar()
        if batch_idx % self.print_freq == 0 or batch_idx == num_batches - 1:
            self.displayer.updateBatchTable(batch_metrics, avgBank, lr)

        self.displayer.refresh()

        # # downgrade
        # msg = f"Ep{epoch_idx} {batch_idx}/{num_batches} "
        # msg += avgBank.to_string()
        # msg += f" {elapsed_time:.2f}s"
        # self.logger.info(msg)

    def onBatchEndVal(self, batch_idx, num_batches, **batch_state):
        avgBank = batch_state["avgBank"]
        batch_metrics = batch_state["batch_metrics"].copy()
        elapsed_time = batch_metrics["BatchTime"]
        msg = f"{batch_idx}/{num_batches} "
        for k, v in batch_metrics.items():
            smooth_avg = f"{avgBank[k]:.6f}" if k in avgBank else ""
            msg += f"{k} avg:{smooth_avg} "
        msg += f" {elapsed_time:.2f}s"
        self.logger.info(msg)

    def onEpochStart(self, num_batches):
        epoch_idx = self.epoch_state["epoch"]
        max_epochs = self.epoch_state["max_epochs"]
        if self.progress_bar:
            self.displayer.reset_progressbar(num_batches, epoch_idx, max_epochs)
            self.displayer.start()

    def onEpochEnd(self, epoch_state, avg_results):
        # interrupt detect
        try:
            if self.progress_bar:
                self.displayer.stop_progressbar()
                self.displayer.clear_contnt()
                # self.displayer.stop()

            if self.mode == "train":
                self.onEpochEndTrain(epoch_state, avg_results)
            else:
                self.onEpochEndVal(epoch_state, avg_results)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt detected. Cleaning up...")
            if self.progress_bar:
                self.stop()
            self.console.print("[yellow]Training interrupted by user. Exiting gracefully...[/]")
            exit(0)
        except Exception as e:
            self.logger.info(f"Unexpected error occurred: {e}")
            raise

    def onEpochEndTrain(self, epoch_state, avg_results):
        self.logger.log(epoch_state)

        epoch = epoch_state["epoch"]
        self.logger.info(f"epoch {epoch}")

        prefixs = ["train", "val", "test"]
        if avg_results.get("ema_val"):
            prefixs.append("ema_val")
        if avg_results.get("ema_test"):
            prefixs.append("ema_test")

        for prefix in prefixs:
            results = avg_results[prefix]
            ss = ""
            for key, value in results.items():
                ss += f"{prefix}/{key}: {value:.6f} "
            self.logger.info(ss)

    def onEpochEndVal(self, epoch_state, avg_results):
        self.logger.log(epoch_state)

    def onRunEnd(self):
        """clear and exit"""
        if self.progress_bar:
            self.displayer.stop()
