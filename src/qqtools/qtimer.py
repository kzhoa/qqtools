import time

import torch


class Timer:
    def __init__(
        self, enter_msg=None, cuda=False, logger=None, prefix=None, precision=4, wrap=True, verbose=True, indent=None
    ):
        assert isinstance(precision, int) and precision >= 0
        self.enter_msg = enter_msg
        self.cuda = cuda
        self.logger = None
        self.prefix = prefix if prefix is not None else str()
        self.precision = precision
        self.verbose = verbose
        self.wrap = wrap
        self.indent = indent if indent is not None else ">>>>"
        self._start_time = None
        self._end_time = None
        self.duration = None

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        if self.enter_msg is not None:
            msg = f"{self.prefix}{self.enter_msg}"
            end = "\n" if self.wrap else str()
            self.print_message(msg, end)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        self.duration = self._end_time - self._start_time
        if self.verbose:
            self.show_duration()

    def show_duration(self):
        if self.precision is not None:
            msg = f"{self.indent}{self.prefix}Duration: {self.duration:.{self.precision}f} seconds"
        else:
            msg = f"{self.indent}{self.prefix}Duration: {self.duration} seconds"

        self.print_message(msg)

    def print_message(self, msg, end="\n"):
        if self.logger is None:
            print(msg, end=end)
        else:
            self.logger.info(msg)


if __name__ == "__main__":

    print("\n0 ", end="")
    with Timer("hello world", precision=5) as t:
        for _ in range(1000):
            a = 1000 + 2000

    print("\n1 ", end="")
    with Timer("no wrap ", precision=5, wrap=False) as t:
        for _ in range(1000):
            a = 1000 + 2000

    print("\n2 ", end="")
    with Timer("change indent ", precision=5, indent="===>", wrap=False) as t:
        for _ in range(1000):
            a = 1000 + 2000

    print("\n3 ", end="")
    with Timer("add prefix ", precision=5, prefix="[TEST]", wrap=False) as t:
        for _ in range(1000):
            a = 1000 + 2000

    print("\n4 ", end="")
    with Timer("control precision", precision=8, indent=" ", prefix="[TEST]", wrap=False) as t:
        for _ in range(1000):
            a = 1000 + 2000

    print("\nduration", t.duration)
