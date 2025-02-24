import time
import torch


class Timer:
    def __init__(self, enter_msg=None, cuda=False, logger=None, prefix=None):
        self.enter_msg = enter_msg
        self.cuda = cuda
        self.logger = None
        self.prefix = prefix + " " if prefix is not None else str()

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        msg = f">>>>>{self.prefix}Execution time: {execution_time:.2f} seconds"
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
