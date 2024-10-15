import time


class Timer:
    def __init__(self, enter_msg, logger=None, prefix=None):
        self.enter_msg = enter_msg
        self.logger = None
        self.prefix = prefix + " " if prefix is not None else str()

    def __enter__(self):
        self.start_time = time.time()
        msg = f"{self.prefix}{self.enter_msg}"
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        execution_time = end_time - self.start_time
        msg = f"{self.prefix}Execution time: {execution_time:.2f} seconds"
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
