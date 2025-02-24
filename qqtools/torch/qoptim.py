from typing import Dict, Protocol, Type

import torch
import torch.optim.lr_scheduler as lr_scheduler

# qq: the name changed after torch 2.x
# https://github.com/pytorch/pytorch/blob/v1.13.1/torch/optim/lr_scheduler.py
# https://github.com/pytorch/pytorch/blob/v2.0.0/torch/optim/lr_scheduler.py
if torch.__version__ < "2":
    LRScheduler: Type[object] = torch.optim.lr_scheduler._LRScheduler
else:
    LRScheduler: Type[object] = torch.optim.lr_scheduler.LRScheduler


class CompositeOptim(torch.optim.Optimizer):
    """qq: corresponds to ani model"""

    def __init__(self, optim_dict: Dict[str, torch.optim.Optimizer]):
        self._optim_dict = optim_dict.copy()

    def zero_grad(self):
        for key, optim in self._optim_dict.items():
            optim.zero_grad()

    def step(self):
        for key, optim in self._optim_dict.items():
            optim.step()

    @property
    def param_groups(self):
        groups = []
        for key, optim in self._optim_dict.items():
            groups += optim.param_groups
        return groups

    def __len__(self):
        return len(self.param_groups)

    def state_dict(self):
        dd = dict()
        for key, optim in self._optim_dict.items():
            dd[key] = optim.state_dict()
        return dd

    def load_state_dict(self, state_dict):
        miss_keys = set()
        for key, optim in self._optim_dict.items():
            if key in state_dict:
                optim.load_state_dict(state_dict[key])
            else:
                miss_keys.add(key)
        print(f"missing optimizer state_dict for: {miss_keys}")


class CompositeScheduler(LRScheduler):  # type: ignore
    def __init__(
        self,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler],
        last_epoch=-1,
    ):
        self._scheduler_dict = scheduler_dict.copy()

    def step(self, metrics=None, epoch=None):
        for scheduler in self._scheduler_dict.values():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics=metrics, epoch=epoch)  # type: ignore
            else:
                scheduler.step(epoch=epoch)

    def state_dict(self):
        dd = dict()
        for key, scheduler in self._scheduler_dict.items():
            dd[key] = scheduler.state_dict()
        return dd

    def load_state_dict(self, state_dict):
        miss_keys = set()
        for key, scheduler in self._scheduler_dict.items():
            if key in state_dict:
                scheduler.load_state_dict(state_dict[key])
            else:
                miss_keys.add(key)
        print(f"missing optimizer state_dict for: {miss_keys}")
