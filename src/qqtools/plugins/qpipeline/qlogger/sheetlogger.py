import csv
import json
import os
import queue
import time
import warnings
from threading import Lock, Thread
from typing import Dict, List, Optional


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

        Default Behavior:
            - Writes to CSV format (human-readable)
            - Synchronous writes (blocking, safe)
            - Appends to existing file if present
            - Rotates file when size exceeds 10 MB
            - Buffer size of 1 (write every record immediately in sync mode)
            - CSV files auto-initialize with column headers
            - Extra keys in write() calls are silently ignored with a warning
            - Thread-safe through Lock (safe for multi-threaded access in sync mode)

        Example Usage:
            # Basic usage: simple CSV logging
            logger = SheetLogger("logs/training.csv", columns=["epoch", "loss", "val_acc"])
            logger.write({"epoch": 1, "loss": 0.5, "val_acc": 0.95})
            logger.write({"epoch": 2, "loss": 0.3, "val_acc": 0.97})

            # JSONL format with async writing (faster for frequent writes)
            logger = SheetLogger(
                "logs/results.jsonl",
                columns=["step", "metric1", "metric2"],
                format="jsonl",
                async_write=True,
                buffer_size=100
            )
            for step in range(1000):
                logger.write({"step": step, "metric1": 0.1, "metric2": 0.2})
            logger.close()  # Important: flush remaining buffered data

            # Overwrite previous logs instead of appending
            logger = SheetLogger("logs/debug.csv", columns=["iteration", "debug_val"], recover=False)
            logger.write({"iteration": 0, "debug_val": 123})
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

        mode = "a" if recover else "w"

        if self.format == "csv":
            if not recover or not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
                with open(self.file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
        elif not recover and os.path.exists(self.file_path):
            # Clear file for jsonl if not recovering
            open(self.file_path, "w").close()

        if async_write:
            self._buffer = []
            self._queue = queue.Queue()
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()

    def write(self, data: Dict[str, any]):
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

    def _sync_write_data(self, data_list: List[Dict]):
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
        buffer = []
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

        # Simple rotation: rename current file to .1, .2, etc.
        # To be robust on Windows, we need to handle potential file access issues.
        timestamp = time.strftime("%Y%m%d%H%M%S")
        backup_path = f"{self.file_path}.{timestamp}"

        try:
            # On Windows, rename might fail if the file is open.
            # But since we use 'with open(...)', it should be closed here.
            os.rename(self.file_path, backup_path)
        except OSError:
            # Fallback to a more unique name if needed
            backup_path = f"{self.file_path}.{timestamp}.{int(time.time()*1000) % 1000}"
            os.rename(self.file_path, backup_path)

        # Re-initialize the file with headers if CSV
        if self.format == "csv":
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def close(self):
        """Gracefully close and flush any pending data."""
        if self.async_write:
            self._queue.put(None)  # Send sentinel
            self._queue.join()
        elif hasattr(self, "_lock"):
            # For sync mode, nothing special to do as we write immediately
            pass
