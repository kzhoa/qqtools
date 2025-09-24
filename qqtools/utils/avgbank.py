import warnings

from ..torch import qdist


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self, ddp=False):
        if ddp:
            return self.ddp_average()
        else:
            return self.avg

    def ddp_average(self):
        ddp_sum = qdist.all_reduce(self.sum, "cpu")
        ddp_count = qdist.all_reduce(self.count, "cpu")
        ddp_avg = ddp_sum / ddp_count
        return ddp_avg


class AvgBank(object):
    """proxy avgmeters"""

    def __init__(self, sep=", ", verbose=False):
        self.sep = str(sep)
        self.verbose = verbose
        self.avgMeters = dict()
        self.key_order = None
        self._default_key_order = []

    def add(self, key, value, num=1):
        if key not in self.avgMeters:
            self.avgMeters[key] = AverageMeter()
            self._default_key_order.append(key)  # default: FCFS
        self.avgMeters[key].update(value, num)

    def keys(self):
        return list(self.avgMeters.keys())

    def set_order(self, key_order):
        """allow passing non-existing keys, which would be ignored and not shown in print"""
        if self.verbose:
            for k in key_order:
                if k not in self.avgMeters:
                    warnings.warn(f"[AvgBank] key: {k} not found in avgMeters, would be ignored upon printing.")
        self.key_order = key_order

    def gather_average(self, ddp: bool):
        result = dict()
        for k, meter in self.avgMeters.items():
            result[k] = meter.get_average(ddp)
        return result

    def __str__(self):
        ss = ""
        key_order = self.key_order if self.key_order else self._default_key_order
        for key in key_order:
            if key in self.avgMeters:
                ss += f"{key}: {self.avgMeters[key].avg:.5f}{self.sep}"
        return ss

    def to_string(self) -> str:
        return self.__str__()

    def to_dict(self, ddp) -> dict:
        return self.gather_average(ddp)
