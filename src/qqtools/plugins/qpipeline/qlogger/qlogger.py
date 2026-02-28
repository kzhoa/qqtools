import os
import os.path as osp

from qqtools import qdist

from .consolelogger import ConsoleLogger
from .sheetlogger import SheetLogger


class DoNothing:
    """
    Dummy object that silently ignores all method calls.

    Used as a no-op replacement when a logger is disabled. Allows code to call
    logger methods without checking if the logger exists first (avoids None checks).
    """

    def passby(self, *args, **kwargs):
        """Ignore any arguments and do nothing."""
        pass

    def __getattr__(self, *args, **kwargs):
        """Return the passby method for any undefined attribute access."""
        return self.passby


class qLogger:
    """
    Unified logging interface combining console logs and metrics sheets.

    Provides a single entry point for all logging needs in the qpipeline training framework.
    Internally manages both human-readable console output (via ConsoleLogger) and structured
    metrics logging (via SheetLogger). Automatically handles distributed training scenarios
    by respecting process rank.
    """

    def __init__(self, log_dir, console=True, columns=None, recover=True):
        """
        Initialize the unified logger with console and sheet logging capabilities.

        Parameters:
            log_dir (str):
                Root directory where all log files will be stored. Parent directories are
                automatically created if they don't exist. Both console logs and metrics
                sheets will be written to subdirectories/files within this location.
                Example: log_dir="./logs/experiment_001"

            console (bool, optional):
                Whether to enable console logging (default: True). When True, creates a
                ConsoleLogger that writes debug logs to "debug.log" file and displays them
                on console (using Rich formatting if available). When False, console logging
                is disabled and a DoNothing placeholder is used instead.
                Useful for disabling noisy output in certain scenarios.

            columns (List[str], optional):
                List of column names for the metrics sheet logger (default: None).
                When provided, creates a SheetLogger that logs structured metrics to a CSV file
                named "metrics.csv". Must contain column names like ["epoch", "loss", "val_acc"].
                When None, metrics logging is disabled.
                Example: columns=["epoch", "loss", "train_metric", "val_metric"]

            recover (bool, optional):
                File handling mode for both console and sheet loggers (default: True).
                  - True : Append mode. New logs are added to existing files (preserves history)
                  - False: Write mode. Log files are cleared and overwritten on initialization
                Applies to both debug.log and metrics.csv files.

        Default Behavior:
            - Logs are written to console with Rich formatting (if available)
            - Debug logs are saved to "{log_dir}/debug.log"
            - Metrics are saved to "{log_dir}/metrics.csv" (only if columns provided)
            - Only rank 0 process outputs logs in distributed training (via ConsoleLogger/SheetLogger)
            - Non-console loggers are replaced with DoNothing objects to avoid None checks
            - Process rank is automatically detected via qdist.get_rank()

        Example Usage:
            # Basic setup: console logging only
            logger = qLogger("./logs/run_001")
            logger.info("Training started")
            logger.error("Failed to load checkpoint")

            # With metrics tracking
            logger = qLogger(
                "./logs/run_001",
                columns=["epoch", "loss", "val_acc", "learning_rate"]
            )
            logger.info("Epoch 1 started")
            logger.log({"epoch": 1, "loss": 0.5, "val_acc": 0.95, "learning_rate": 0.001})

            # Distributed training (DDP/DistributedDataParallel)
            logger = qLogger(
                "./logs/run_001",
                columns=["epoch", "loss"],
                recover=True  # Append to previous run
            )
            # Only rank 0 will actually output, others use DoNothing

            # No console output, metrics only
            logger = qLogger(
                "./logs/run_001",
                console=False,
                columns=["step", "metric1", "metric2"]
            )
        """
        self.log_dir = log_dir
        self.columns = columns

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        rank = qdist.get_rank()

        if console:
            # ConsoleLogger already handles rank internally, but we only want one logger per rank 0 usually
            # Actually, ConsoleLogger handles rank by setting adapter to None if rank != 0
            self.debuglogger = ConsoleLogger(osp.join(log_dir, "debug.log"), rank=rank, recover=recover)
        else:
            self.debuglogger = DoNothing()

        if columns is not None and rank == 0:
            self.sheetlogger = SheetLogger(osp.join(log_dir, "metrics.csv"), columns, recover=recover)
        else:
            self.sheetlogger = DoNothing()

    def info(self, msg, *args, **kwargs):
        """Log an informational message."""
        self.debuglogger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self.debuglogger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self.debuglogger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self.debuglogger.warning(msg, *args, **kwargs)

    def log(self, data: dict):
        """Log structured metrics data to the CSV sheet."""
        self.sheetlogger.write(data)

    def close(self):
        """Gracefully close all loggers and flush pending data."""
        self.debuglogger.close()
        self.sheetlogger.close()
