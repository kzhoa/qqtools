import logging
import os
import sys

try:
    import rich
    from rich.logging import RichHandler

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class CallerInfoAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Default to unknown if frame extraction fails
        caller_file, caller_line, caller_func = "unknown", 0, "unknown"
        try:
            # We want to find the first frame outside of logging and this class
            f = sys._getframe()
            while f:
                if "logging" not in f.f_code.co_filename and "consolelogger.py" not in f.f_code.co_filename:
                    caller_file = os.path.basename(f.f_code.co_filename)
                    caller_line = f.f_lineno
                    caller_func = f.f_code.co_name
                    break
                f = f.f_back
        except Exception:
            pass

        kwargs.setdefault("extra", {}).update(
            {
                "caller_file": caller_file,
                "caller_line": caller_line,
                "caller_func": caller_func,
            }
        )
        return msg, kwargs


class ConsoleLogger:
    """
    ConsoleLogger handles logging to console (via rich if available) and optionally to a file.
    """

    def __init__(self, filepath=None, rank=0, logger_name="qq", level=logging.DEBUG, recover=True):
        """
        Initialize the ConsoleLogger with flexible output configuration.

        Parameters:
            filepath (str, optional):
                Path to a file where logs should also be written. If None (default), logs are only
                printed to console. When provided, logs are written to both console and file.
                The parent directory will be automatically created if it doesn't exist.
                Example: filepath="./logs/training.log"

            rank (int, optional):
                Process rank for distributed training (default: 0). Only rank 0 will actually output
                logs. Non-rank-0 processes will have logger=None and adapter=None, preventing any
                console/file output. This is useful for DDP setups to avoid duplicate logs.
                In single-GPU training, use the default rank=0.

            logger_name (str, optional):
                Base name for the internal logger object (default: "qq"). Combined with object id
                to create a unique logger name, preventing conflicts when multiple ConsoleLogger
                instances are created: f"{logger_name}_{id(self)}". Useful for debugging or
                organizing logs from different components.

            level (int, optional):
                Logging level threshold (default: logging.DEBUG). Controls minimum severity of
                messages that will be logged. Common levels:
                  - logging.DEBUG   (10): Detailed diagnostic information
                  - logging.INFO    (20): General informational messages
                  - logging.WARNING (30): Warning messages (default for production)
                  - logging.ERROR   (40): Error messages only
                Higher level values filter out lower-severity messages.

            recover (bool, optional):
                File write mode when filepath is provided (default: True).
                  - True : Append mode ('a'). New logs are appended to existing file content.
                  - False: Write mode ('w'). File is cleared and overwritten on each initialization.
                When filepath=None, this parameter is ignored (no file operations).

        Default Behavior:
            - Logs to console only (no file output)
            - Console uses Rich formatting if available, else plain StreamHandler to stdout
            - Only rank 0 produces output (logger is None for rank > 0)
            - Logs at DEBUG level and above
            - Caller information (file, line, function) is automatically appended to all messages
              using CallerInfoAdapter for better debugging context

        Example Usage:
            # Basic usage: console only
            logger = ConsoleLogger()
            logger.info("Training started")

            # Console + file logging
            logger = ConsoleLogger(filepath="./logs/train.log")
            logger.info("Training checkpoint saved")

            # DDP setup: only rank 0 logs
            logger = ConsoleLogger(filepath="./logs/train.log", rank=args.rank)
            logger.warning("Gradient clipping activated")

            # Overwrite log file each run instead of appending
            logger = ConsoleLogger(filepath="./logs/train.log", recover=False)
            logger.error("Training failed")
        """
        self.filepath = filepath
        self.logger_name = logger_name
        self.rank = rank
        self.recover = recover

        if rank == 0:
            self.logger = self._get_logger(level)
            self.adapter = CallerInfoAdapter(self.logger, {})
        else:
            self.logger = None
            self.adapter = None

    def _get_logger(self, level):
        logger = logging.getLogger(f"{self.logger_name}_{id(self)}")
        logger.setLevel(level)
        logger.propagate = False

        # Clear existing handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(asctime)s][%(caller_file)s:%(caller_line)d][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console output: Use Rich if available, otherwise fallback to StreamHandler
        if HAS_RICH:
            # When using RichHandler, we often don't want the custom formatter because Rich handles it
            # But the requirement says "Select RichHandler OR StreamHandler" to avoid double output.
            console_handler = RichHandler(level, rich.get_console(), show_path=False)
            # If we use RichHandler, we might want to still inject our caller info if Rich's path is hidden
            # or use Rich's built-in caller info. For consistency with the requested plan:
            console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)

        logger.addHandler(console_handler)

        # File output
        if self.filepath:
            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            mode = "a" if self.recover else "w"
            file_handler = logging.FileHandler(self.filepath, mode=mode)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def info(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.adapter:
            self.adapter.critical(msg, *args, **kwargs)

    def close(self):
        """Close all handlers and clean up resources."""
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
