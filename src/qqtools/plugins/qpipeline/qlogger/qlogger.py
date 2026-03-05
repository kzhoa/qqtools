import os
import os.path as osp

from qqtools import qdist

from .consolelogger import ConsoleLogger


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
    Console-oriented logger facade for qpipeline.

    This class keeps the historical qLogger interface for debug/info/warning/error
    methods while structured sheet logging is now managed by runner utilities.
    """

    def __init__(self, log_dir, console=True, recover=True):
        """
        Initialize the logger.

        Parameters:
            log_dir (str):
                Root directory where debug logs are stored.
            console (bool, optional):
                Whether to enable console/debug logging (default: True).
            recover (bool, optional):
                Whether to append to existing debug log files (default: True).
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        rank = qdist.get_rank()

        if console:
            self.debuglogger = ConsoleLogger(
                osp.join(log_dir, "debug.log"),
                rank=rank,
                recover=recover,
            )
        else:
            self.debuglogger = DoNothing()

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

    def close(self):
        """Gracefully close all logger handlers."""
        self.debuglogger.close()
