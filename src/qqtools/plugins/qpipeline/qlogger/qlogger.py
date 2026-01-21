import os.path as osp

from qqtools import qdist

from .consolelogger import ConsoleLogger
from .sheetlogger import SheetLogger


class DoNothing:
    def passby(self, *args, **kwargs):
        pass

    def __getattr__(self, *args, **kwargs):
        return self.passby


class qLogger:

    def __init__(self, log_dir, console=True, columns=None, recover=True):
        # TODO qq: recover mode is not implemented
        self.log_dir = log_dir
        self.columns = columns
        assert osp.exists(log_dir), "log_dir {} does not exist".format(log_dir)

        rank = qdist.get_rank()

        if console:
            self.debuglogger = ConsoleLogger(osp.join(log_dir, "debug.log"), rank=rank)
        else:
            self.debuglogger = DoNothing()

        if columns is not None:
            self.sheetlogger = SheetLogger(osp.join(log_dir, "metrics.csv"), columns)
        else:
            self.sheetlogger = DoNothing()

    def info(self, *args, **kwargs):
        self.debuglogger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.debuglogger.error(*args, **kwargs)

    def log(self, *args, **kwargs):
        self.sheetlogger.write(*args, **kwargs)
