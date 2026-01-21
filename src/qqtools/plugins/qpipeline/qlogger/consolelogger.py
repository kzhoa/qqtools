import logging
import os
import sys

import rich
from rich.logging import RichHandler


class DoNothing:
    def passby(self, *args, **kwargs):
        pass

    def __getattr__(self, *args):
        return self.passby


class CallerInfoAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        frame = sys._getframe(4)
        kwargs.setdefault("extra", {}).update(
            {
                "caller_file": os.path.basename(frame.f_code.co_filename),
                "caller_line": frame.f_lineno,
                "caller_func": frame.f_code.co_name,
            }
        )
        return msg, kwargs


class ConsoleLogger:
    """
    qq:
    use an adapter to proxy all logger behaviors
    """

    def __init__(self, filepath=None, rank=0, logger_name="qq", caller_info=True):
        self.filepath = filepath
        self.logger_name = logger_name
        self.caller_info = caller_info
        if rank == 0:
            to_file = self.filepath is not None
            self.logger = self.get_logger(to_file)
            self.adapter = CallerInfoAdapter(self.logger, {})

            for level in (
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
            ):
                setattr(self, level, getattr(self.adapter, level))

        else:
            self.logger = DoNothing()
            self.adapter = DoNothing()

    def get_logger(self, to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)

        # clear
        logger.handlers.clear()  # TODO: may not needed?

        # rich
        has_rich_handler = any(isinstance(handler, RichHandler) for handler in logger.handlers)
        if not has_rich_handler:
            console = rich.get_console()
            richhandle = RichHandler(logging.DEBUG, console)  # compatible with rich.live
            logger.addHandler(richhandle)

        # console
        formatter = logging.Formatter(
            "[%(asctime)s][%(caller_file)s:%(caller_line)d][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

        # file
        if to_file:
            debuglog = logging.FileHandler(self.filepath)
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(formatter)
            logger.addHandler(debuglog)

        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False
        return logger

    def log(self, *args, **kwargs):
        self.adapter.info(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.adapter.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.adapter.error(*args, **kwargs)
