import qqtools as qt
import torch

from .scheduler import build_null_scheduler, prepare_scheduler

__all__ = ["prepare_optim"]

CANONICAL_OPTIMIZER_NAMES = {
    "adamw": "AdamW",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adam": "Adam",
}


def getCanonicalName(name):
    __name = name.lower()
    if __name in CANONICAL_OPTIMIZER_NAMES:
        return CANONICAL_OPTIMIZER_NAMES[__name]
    else:
        raise KeyError(f"not recognized optimizer: {name}")


def prepare_optimizer(args: qt.qDict, model):
    # Interface Convention
    args = args.copy()
    args.allow_notexist = False
    optimizer = args.optim.optimizer  # "AdamW"
    optimizer_params = args.optim.optimizer_params

    # fetch & init
    optimizer = getCanonicalName(optimizer)
    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_params,
    )

    return optimizer


def prepare_optim(args: qt.qDict, model):

    optimizer = prepare_optimizer(args, model)
    if "scheduler" in args.optim and args.optim.scheduler is not None:
        scheduler = prepare_scheduler(args, optimizer)
        return optimizer, scheduler
    else:
        scheduler = build_null_scheduler()
        return optimizer, scheduler
