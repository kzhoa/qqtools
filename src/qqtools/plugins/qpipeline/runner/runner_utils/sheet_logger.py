import csv
import json
import os
import queue
import time
import warnings
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

from .common import _is_periodic_trigger
from .types import EventContext, RunConfig


class SheetLogger:
    """
    Structured logging to CSV or JSONL format with optional async writing and file rotation.

    Useful for tracking metrics, hyperparameters, and results across epochs/steps. Supports
    automatic file rotation when size exceeds a threshold.
    """

    def __init__(
        self,
        file_path: str,
        columns: List[str],
        format: str = "csv",
        max_size: int = 10 * 1024 * 1024,
        buffer_size: int = 1,
        async_write: bool = False,
        recover: bool = True,
    ):
        """
        Initialize the SheetLogger for structured data logging.

        Parameters:
            file_path (str):
                Path where the log file will be stored (e.g., "logs/training.csv").
                Parent directory will be automatically created if it doesn't exist.
                Example: file_path="./logs/metrics.csv"

            columns (List[str]):
                List of column names for the structured data. Each write() call should contain
                values for these columns. Unknown keys in write() calls will be ignored with
                a warning. Required field.
                Example: columns=["epoch", "loss", "val_acc", "learning_rate"]

            format (str, optional):
                Output format for the log file (default: "csv"). Valid options:
                  - "csv"  : Comma-separated values (human-readable, Excel-compatible)
                  - "jsonl": JSON Lines format (one JSON object per line, machine-readable)
                When set to "json", it's automatically converted to "jsonl" for clarity.

            max_size (int, optional):
                Maximum file size in bytes before rotation (default: 10 MB).
                When the log file exceeds this size, it's renamed with a timestamp suffix
                and a fresh file is created. This prevents excessively large log files.
                Set to a very large number to effectively disable rotation.

            buffer_size (int, optional):
                Number of records to buffer before flushing to disk (default: 1).
                Only used in async mode (async_write=True). Higher values reduce I/O
                overhead but increase memory usage. In sync mode, this is ignored.
                Example: buffer_size=100 flushes every 100 writes

            async_write (bool, optional):
                Whether to write to disk asynchronously using a background thread (default: False).
                  - False: Blocking writes (safe, simpler, slower for frequent writes)
                  - True : Non-blocking writes via queue (faster, requires calling close())
                In async mode, data goes into a queue and a worker thread writes when buffer_size
                is reached or after 1 second timeout. Always call close() before program exit.

            recover (bool, optional):
                File handling mode when the log file already exists (default: True).
                  - True : Append mode. New data is added to existing file (preserves history)
                  - False: Write mode. File is cleared and reinitialized with headers
                When filepath=None, this parameter is ignored (no file operations).
        """
        self.file_path = os.path.abspath(file_path)
        self.columns = columns
        self.format = format.lower()
        if self.format == "json":
            self.format = "jsonl"

        self.max_size = max_size
        self.buffer_size = buffer_size
        self.async_write = async_write
        self._lock = Lock()

        assert columns is not None
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        if self.format == "csv":
            if not recover or not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
                with open(self.file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
        elif not recover and os.path.exists(self.file_path):
            # Clear file for jsonl if not recovering
            open(self.file_path, "w").close()

        if async_write:
            self._queue = queue.Queue()
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()

    def write(self, data: Dict[str, Any]):
        """
        Write a record to the log file.

        Args:
            data: Dictionary containing values for the columns. Unknown keys are ignored.
        """
        # Validation: allow subset of columns, but no extra keys
        extra_keys = set(data.keys()) - set(self.columns)
        if extra_keys:
            warnings.warn(f"SheetLogger: Unexpected keys found and will be ignored: {extra_keys}")
            data = {k: v for k, v in data.items() if k in self.columns}

        if self.async_write:
            self._queue.put(data)
        else:
            with self._lock:
                self._sync_write_data([data])

    def _sync_write_data(self, data_list: List[Dict[str, Any]]):
        """Write a list of data records to the log file."""
        if not data_list:
            return

        try:
            if self.format == "csv":
                with open(self.file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for data in data_list:
                        writer.writerow([data.get(col, "") for col in self.columns])
            elif self.format == "jsonl":
                with open(self.file_path, "a") as f:
                    for data in data_list:
                        f.write(json.dumps(data) + "\n")

            self._rotate_if_needed()
        except Exception as e:
            print(f"SheetLogger Error: {e}")

    def _worker(self):
        """Background worker thread for asynchronous writing."""
        buffer: List[Dict[str, Any]] = []
        while True:
            try:
                # Use a timeout to occasionally check if we should flush what we have
                data = self._queue.get(timeout=1.0)
                if data is None:  # Sentinel for closing
                    if buffer:
                        self._sync_write_data(buffer)
                    self._queue.task_done()
                    break

                buffer.append(data)
                if len(buffer) >= self.buffer_size:
                    with self._lock:
                        self._sync_write_data(buffer)
                    buffer = []
                self._queue.task_done()
            except queue.Empty:
                if buffer:
                    with self._lock:
                        self._sync_write_data(buffer)
                    buffer = []
                continue

    def _rotate_if_needed(self):
        """Rotate the log file if it exceeds the maximum size."""
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) <= self.max_size:
            return

        timestamp = time.strftime("%Y%m%d%H%M%S")
        backup_path = f"{self.file_path}.{timestamp}"

        try:
            os.rename(self.file_path, backup_path)
        except OSError:
            backup_path = f"{self.file_path}.{timestamp}.{int(time.time() * 1000) % 1000}"
            os.rename(self.file_path, backup_path)

        if self.format == "csv":
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def close(self):
        """Gracefully close and flush any pending data."""
        if self.async_write:
            self._queue.put(None)
            self._queue.join()


class SheetLoggerListener:
    """
    Listener that writes metrics to SheetLogger based on train/eval events.
    """

    def __init__(
        self,
        sheet_logger: SheetLogger,
        run_config: RunConfig,
        log_granularity: List[str],
        logger: Optional[Any] = None,
    ):
        self.sheet_logger = sheet_logger
        self.config = run_config
        self.log_granularity = log_granularity
        self.logger = logger

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)
        else:
            warnings.warn(message)

    def _prepare_data(self, context: EventContext, mode: str) -> Dict[str, Any]:
        """Prepare a flat dictionary of metrics for logging."""
        state = context.state
        data = {"epoch": state.epoch, "global_step": state.global_step}

        source_metrics: Dict[str, Any] = {}
        if mode == "eval":
            if getattr(context, "eval_results", None):
                source_metrics.update(context.eval_results)
            else:
                for key in [
                    "current_val_metric",
                    "current_test_metric",
                    "current_train_metric",
                    "current_train_loss",
                ]:
                    value = getattr(state, key, None)
                    if value is not None:
                        source_metrics[key.replace("current_", "")] = value
                if not source_metrics:
                    self._warn("No eval metrics found in context or state; writing empty row.")
        elif mode == "batch":
            if getattr(context, "batch_metrics", None):
                source_metrics.update(context.batch_metrics)
            else:
                self._warn("No batch_metrics found in context; writing empty row.")

        if self.sheet_logger.columns:
            for key in self.sheet_logger.columns:
                if key in data:
                    continue
                value = source_metrics.get(key)
                if value is None and mode == "batch" and key.startswith("train_"):
                    value = source_metrics.get(key[len("train_") :])
                data[key] = value
        else:
            data.update(source_metrics)

        return data

    def on_eval_end(self, context: EventContext):
        data = self._prepare_data(context, mode="eval")
        self.sheet_logger.write(data)

    def on_train_batch_end(self, context: EventContext):
        if "eval" in self.log_granularity:
            is_epoch_end = (
                context.batch_idx is not None
                and context.total_batches is not None
                and context.batch_idx == context.total_batches - 1
            )
            is_eval_trigger = _is_periodic_trigger(
                run_mode=self.config.run_mode,
                interval=self.config.eval_interval,
                global_step=context.state.global_step,
                epoch=context.state.epoch,
                is_epoch_end=is_epoch_end,
            )
            if is_eval_trigger:
                return

        data = self._prepare_data(context, mode="batch")
        self.sheet_logger.write(data)
