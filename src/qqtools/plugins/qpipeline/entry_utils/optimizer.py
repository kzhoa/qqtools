import qqtools as qt
import torch

from .no_decay import build_param_groups, collect_no_decay_params

__all__ = ["prepare_optimizer"]

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


def prepare_optimizer(args: qt.qDict, model, logger=None):
    args = args.copy()
    args.allow_notexist = False
    optimizer_name = args.optim.optimizer
    optimizer_params = dict(args.optim.optimizer_params)

    optimizer_cls_name = getCanonicalName(optimizer_name)
    optimizer_cls = getattr(torch.optim, optimizer_cls_name)

    no_decay_names = collect_no_decay_params(model)
    if no_decay_names:
        msg = f"[Optimizer] no-decay discovery: {len(no_decay_names)} parameter(s) declared via convention methods"
        if logger is not None:
            logger.info(msg)
        elif qt.qdist.get_rank() == 0:
            print(f"[qPipeline:0] {msg}")

    param_groups = build_param_groups(model, optimizer_params, no_decay_names=no_decay_names)

    if len(param_groups) > 1:
        nd_count = len(param_groups[1]["params"])
        total = sum(len(g["params"]) for g in param_groups)
        msg = (
            f"[Optimizer] {nd_count}/{total} trainable parameters "
            f"assigned to no-weight-decay group"
        )
        if logger is not None:
            logger.info(msg)
        elif qt.qdist.get_rank() == 0:
            print(f"[qPipeline:0] {msg}")

    optimizer = optimizer_cls(param_groups)
    return optimizer
