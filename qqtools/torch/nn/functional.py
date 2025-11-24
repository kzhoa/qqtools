import torch
import torch.nn.functional as F

import qqtools as qt


def l1_loss_ddp(input, target, reduction="mean", batch_size=None):

    loss = F.l1_loss(input, target)
    # world_size = qt.qdist.get_world_size()

    if reduction == "mean":
        if batch_size is not None:
            num_samples = batch_size
        else:
            assert input.dim() <= 2  # (bz, d)
            num_samples = input.shape[0]
        num_samples = qt.qdist.all_reduce(num_samples, device=input.device, reduceOp="sum")
        return loss * qt.qdist.get_world_size() / num_samples
    elif reduction == "sum":
        return loss  # * world_size?
