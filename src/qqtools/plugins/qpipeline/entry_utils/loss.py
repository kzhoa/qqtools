import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from qqtools import qdist


class DDPLoss(torch.nn.Module):
    def __init__(self, loss_fn, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ):
        """
        # assume input to be (bz*nAtoms, ...)
        """
        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            warnings.warn("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        if natoms is None:
            loss = self.loss_fn(input, target)
        else:  # atom-wise loss
            loss = self.loss_fn(input, target, natoms)
        if self.reduction == "mean":
            if batch_size is not None:
                num_samples = batch_size
            else:
                # (bz*nAtoms, ...)
                # qq: if it's (bz,nA), there would be a mistake
                assert input.dim() <= 2
                num_samples = input.shape[0]
            num_samples = qdist.all_reduce(num_samples, device=input.device, reduceOp="sum")
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * qdist.get_world_size() / num_samples
        else:
            return loss


class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def parse_loss_name(loss_name, dpp=False):
    if loss_name in ["l1", "mae"]:
        ret = torch.nn.L1Loss()
    elif loss_name == "mse":
        ret = torch.nn.MSELoss()
    elif loss_name == "l2mae":
        ret = L2MAELoss()
    elif loss_name in ["bce"]:
        ret = torch.nn.BCELoss()
    elif loss_name in ["ce", "cross_entropy"]:
        ret = torch.nn.CrossEntropyLoss()
    elif loss_name in ["focal", "focal_loss"]:
        ret = FocalLoss()
    else:
        raise NotImplementedError(f"Unknown loss function name: {loss_name}")
    if dpp:
        ret = DDPLoss(ret)
    return ret


# === composite ===


class ComboLoss(object):
    def __init__(self, loss_fns: dict, loss_weights: dict):
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights

    def __call__(
        self,
        input_dict: dict,
        target_dict: dict,
    ):
        total_loss = 0.0
        for lname, loss_fn in self.loss_fns.items():
            weight = self.loss_weights.get(lname, 1.0)
            input_tensor = input_dict[lname]
            target_tensor = target_dict[lname]
            loss = loss_fn(input_tensor, target_tensor)
            total_loss += weight * loss
        return total_loss


def parse_comboloss_params(loss_params, ddp=False):
    loss_fns = dict()
    loss_weights = dict()
    for key, v in loss_params.items():
        if isinstance(v, list) and len(v) == 2:
            loss_name, weight = v
        else:
            loss_name = v
            weight = 1.0
        loss_fns[key] = parse_loss_name(loss_name, ddp)
        loss_weights[key] = weight

    loss_fn = ComboLoss(loss_fns, loss_weights)
    return loss_fn


def prepare_loss(args):
    ddp = args.distributed
    loss = args.optim.loss
    if isinstance(loss, str):
        if loss.lower() in ["comboloss", "combo_loss", "composite", "combination"]:
            loss_params = args.optim.loss_params
            loss_fn = parse_comboloss_params(loss_params, ddp)
        else:
            loss_fn = parse_loss_name(loss, ddp)
    elif isinstance(loss, dict):
        loss_fn = {k: parse_loss_name(v, ddp) for k, v in loss.items()}
    else:
        raise TypeError(f"Unknown loss type: {type(loss)}")

    return loss_fn
