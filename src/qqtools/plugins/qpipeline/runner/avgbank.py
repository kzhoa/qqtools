import warnings

import qqtools as qt
import torch
from qqtools import qdist


class AverageMeter:
    """Computes and stores the average value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = qt.ensure_scala(val)
        n = qt.ensure_scala(n)
        if n <= 0:
            return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self, ddp=False):
        if ddp:
            return self.ddp_average()
        else:
            return self.avg

    def ddp_average(self):
        device = qt.parse_device(qdist.get_rank())

        _sum = torch.tensor(self.sum, dtype=torch.float64, device=device)
        _cnt = torch.tensor(self.count, dtype=torch.float64, device=device)
        ddp_sum = qdist.all_reduce(_sum, device, "mean").item()
        ddp_count = qdist.all_reduce(_cnt, device, "mean").item()

        ddp_avg = ddp_sum / ddp_count if ddp_count > 0 else 0
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

    def toString(self) -> str:
        """For compatibility"""
        return self.to_string()

    def toDict(self, ddp) -> dict:
        """For compatibility"""
        return self.to_dict(ddp)
