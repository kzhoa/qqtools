import time

import torch


class Timer:
    def __init__(self, enter_msg=None, cuda=False, logger=None, prefix=None, precision=2, verbose=True):
        assert isinstance(precision, int) and precision >= 0
        self.enter_msg = enter_msg
        self.cuda = cuda
        self.logger = None
        self.prefix = prefix + " " if prefix is not None else str()
        self.precision = precision
        self._start_time = None
        self._end_time = None
        self.duration = None
        self.verbose = verbose

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        if self.enter_msg is not None:
            msg = f"{self.prefix}{self.enter_msg}"
            if self.logger is None:
                print(msg)
            else:
                self.logger.info(msg)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        self.duration = self._end_time - self._start_time
        if not self.verbose:
            return

        msg = f">>>>>{self.prefix}Execution time: {self.duration:.{self.precision}f} seconds"
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
