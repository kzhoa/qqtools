from typing import Dict, Protocol, Type

import qqtools as qt
import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, LRScheduler, ReduceLROnPlateau

__all__ = ["prepare_scheduler"]

CANONICAL_SCHEDULER_NAMES: Dict[str, str] = {
    "cosine": "CosineAnnealingLR",
    "step": "StepLR",
    "plateau": "ReduceLROnPlateau",
    "lambda": "LambdaLR",
    "multi_step": "MultiStepLR",
}


def get_canonical_name(name: str) -> str:
    __name = name.lower()
    if __name in CANONICAL_SCHEDULER_NAMES:
        return CANONICAL_SCHEDULER_NAMES[__name]
    else:
        raise KeyError(f"Unrecognized scheduler: {name}")


def get_cosine_lr(scheduler_params: qt.qDict, optimizer):
    scheduler_params = scheduler_params.copy()
    scheduler_params.allow_notexist = False

    T_max = scheduler_params.T_max
    eta_min = scheduler_params.eta_min

    cosLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return cosLR
    # if "warmup_params" not in scheduler_params:
    #     return cosLR

    # warmup_params = scheduler_params.warmup_params
    # warmup_params.allow_notexist = False
    # warmup_epochs = warmup_params.warmup_epochs
    # warmup_factor = warmup_params.warmup_factor

    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[
    #         torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_epochs),
    #         cosLR,
    #     ],
    #     milestones=[warmup_epochs],
    # )

    # return scheduler


def get_inner_scheduler(scheduler, scheduler_params: qt.qDict, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    # fetch & init
    try:
        scheduler = get_canonical_name(scheduler)
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)
        scheduler = scheduler_class(optimizer, **scheduler_params)
    except AttributeError as e:
        raise ValueError(f"Invalid scheduler class: {scheduler}") from e
    except TypeError as e:
        raise ValueError(f"Invalid scheduler params: {scheduler_params}") from e
    return scheduler


def prepare_scheduler(args: qt.qDict, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    # Interface Convention
    optim_args = args.optim.copy()
    lr = optim_args.optimizer_params.lr
    scheduler = optim_args.scheduler  # e.g. "CosineAnnealingLR"
    scheduler_params: qt.qDict = optim_args.scheduler_params
    warmup_params: qt.qDict = optim_args.warmup_params

    if scheduler is None or scheduler == "":
        return build_null_scheduler()

    if scheduler in ["cosine", "cosinelr", "cosine_lr", "CosineAnnealingLR"]:
        main_scheduler = get_cosine_lr(scheduler_params, optimizer)
    else:
        main_scheduler = get_inner_scheduler(scheduler, scheduler_params, optimizer)

    if warmup_params is not None:
        warmup_steps = warmup_params.warmup_steps
        warmup_factor = warmup_params.warmup_factor
    else:
        warmup_steps = -1
        warmup_factor = 0
    scheduler = qWarmupScheduler(optimizer, warmup_steps, warmup_factor, main_scheduler)
    return scheduler


def build_null_scheduler():
    print("[qPipeline]Lr Scheduler set to disabled")
    main_scheduler = qt.nn.DoNothing()
    warmup_steps = -1
    warmup_factor = 0
    optimizer = NullOptimizer()
    scheduler = qWarmupScheduler(optimizer, warmup_steps, warmup_factor, main_scheduler)
    return scheduler


class NullOptimizer(torch.optim.Optimizer, qt.nn.DoNothing):
    def __init__(self, *args, **kwargs):
        self.param_groups = []


class qWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        warmup_factor,
        main_scheduler,
        last_epoch=-1,
    ):
        """
        qq:
        warmup steps at BatchEnd.
        main steps at EpochEnd.
        usually warmup finishes at 0-th epoch.
        TODO: maybe we can add warmup_epochs to control the epoch-wise behavier.

        Args:
            optimizer:
            warmup_steps:
            warmup_factor: the warmup start factor
            main_scheduler:
        """
        if not isinstance(main_scheduler, (LRScheduler, ReduceLROnPlateau, qt.nn.DoNothing)):
            raise TypeError("main_scheduler must be a scheduler instance")

        self.warmup_factor = warmup_factor
        if warmup_steps > 0:
            self.warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.warmup_scheduler = qt.nn.DoNothing()

        self.main_scheduler: LRScheduler = main_scheduler
        self.current_step = 0
        self.warmup_steps = warmup_steps
        self._is_plateau = isinstance(main_scheduler, ReduceLROnPlateau)

        # should be called after everthing is initialized
        super().__init__(optimizer, last_epoch)

    def step(self, metrics=None, epoch=None):
        """backward compatibility"""
        self.current_step += 1

        # Warmup
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()

    def step_epoch(self, metrics=None):
        """call upon epoch end"""
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self._main_scheduler_step(metrics)

    def _main_scheduler_step(self, metrics=None):
        if self._is_plateau:
            self.main_scheduler.step(metrics)
        else:
            self.main_scheduler.step()

    def onTrainBatchEnd(self, *args, **kwargs):
        self.step()

    def get_lr(self):
        if self.current_step <= self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.main_scheduler.get_last_lr()

    def get_current_lr(self):
        return self.get_last_lr()

    def state_dict(self):
        return {
            "warmup": self.warmup_scheduler.state_dict(),
            "main": self.main_scheduler.state_dict(),
            "current_step": self.current_step,
            "_is_plateau": self._is_plateau,
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict["warmup"])
        self.main_scheduler.load_state_dict(state_dict["main"])
        self.current_step = state_dict["current_step"]
        self._is_plateau = state_dict.get("_is_plateau", isinstance(self.main_scheduler, ReduceLROnPlateau))

    def re_init(self, lr):
        _base_lrs: list[float] = self.main_scheduler.base_lrs
        self.main_scheduler.base_lrs = [lr] * len(_base_lrs)
        self.main_scheduler.last_epoch = -1
        # after all, run init
        self.main_scheduler._initial_step()
