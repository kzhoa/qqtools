import torch


class CompositeOptim(torch.optim.Optimizer):
    """qq: corresponds to ani model"""

    def __init__(self, optim_dict):
        self._optim_dict = {}
        for key, optim in optim_dict.items():
            self._optim_dict[key] = optim

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
        pass
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        pass
        # self.__setstate__({'state': state, 'param_groups': param_groups})
        raise NotImplementedError()


class CompositeScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, scheduler_dict, last_epoch=-1):
        self._scheduler_dict = scheduler_dict.copy()

    def step(self, epoch=None):
        for scheduler in self._scheduler_dict.values():
            scheduler.step(epoch)

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass
