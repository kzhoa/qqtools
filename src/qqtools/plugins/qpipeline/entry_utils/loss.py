import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from qqtools import qdist


class DDPMeanReducedLoss(torch.nn.Module):
    def __init__(self, loss_fn, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = loss_fn
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
    def __init__(self, alpha=1, gamma=2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        assert reduction in ["none", "mean", "sum"]

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss.mean()


class RMSELoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return torch.sqrt(torch.mean(torch.square(input - target)))


class LossSpec(ABC):
    names: Tuple[str, ...]

    @abstractmethod
    def build_local(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_ddp(self) -> torch.nn.Module:
        raise NotImplementedError

    def build(self, dpp: bool = False) -> torch.nn.Module:
        return self.build_ddp() if dpp else self.build_local()


@dataclass(frozen=True)
class ReductionLossSpec(LossSpec):
    names: Tuple[str, ...]
    factory: Callable[..., torch.nn.Module]

    def build_local(self) -> torch.nn.Module:
        return self.factory(reduction="mean")

    def build_ddp(self) -> torch.nn.Module:
        return DDPMeanReducedLoss(self.factory(reduction="sum"))


@dataclass(frozen=True)
class RMSELossSpec(LossSpec):
    names: Tuple[str, ...] = ("rmse",)

    def build_local(self) -> torch.nn.Module:
        return RMSELoss()

    def build_ddp(self) -> torch.nn.Module:
        raise NotImplementedError(
            "rmse is not supported in DDP because strict global RMSE aggregation is not compatible with "
            "the current local-loss/DDP-gradient workflow."
        )


@dataclass(frozen=True)
class FocalLossSpec(LossSpec):
    names: Tuple[str, ...] = ("focal", "focal_loss")
    alpha: float = 1
    gamma: float = 2

    def build_local(self) -> torch.nn.Module:
        return FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="mean")

    def build_ddp(self) -> torch.nn.Module:
        return DDPMeanReducedLoss(FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="sum"))


LOSS_SPECS = (
    ReductionLossSpec(names=("l1", "mae"), factory=torch.nn.L1Loss),
    ReductionLossSpec(names=("mse",), factory=torch.nn.MSELoss),
    RMSELossSpec(),
    ReductionLossSpec(names=("l2mae",), factory=L2MAELoss),
    ReductionLossSpec(names=("bce",), factory=torch.nn.BCELoss),
    ReductionLossSpec(names=("ce", "cross_entropy"), factory=torch.nn.CrossEntropyLoss),
    FocalLossSpec(),
)

LOSS_REGISTRY: Dict[str, LossSpec] = {
    name: spec
    for spec in LOSS_SPECS
    for name in spec.names
}


def parse_loss_name(loss_name, dpp=False):
    spec = LOSS_REGISTRY.get(loss_name)
    if spec is None:
        raise NotImplementedError(f"Unknown loss function name: {loss_name}")
    return spec.build(dpp=dpp)


# === composite ===


class ComboLoss(torch.nn.Module):
    def __init__(self, loss_fns: dict, loss_weights: dict):
        super().__init__()
        assert loss_fns, "loss_fns must not be empty"
        self.loss_fns = torch.nn.ModuleDict(loss_fns)
        self.loss_weights = loss_weights

    def forward(
        self,
        input_dict: dict,
        target_dict: dict,
        **kwargs,
    ):
        total_loss = 0.0
        for lname, loss_fn in self.loss_fns.items():
            weight = self.loss_weights.get(lname, 1.0)
            input_tensor = input_dict[lname]
            target_tensor = target_dict[lname]
            loss = loss_fn(input_tensor, target_tensor, **kwargs)
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
            if loss_params is None:
                raise ValueError("optim.loss_params is required when optim.loss is 'comboloss'.")
            loss_fn = parse_comboloss_params(loss_params, ddp)
        else:
            loss_fn = parse_loss_name(loss, ddp)
    elif isinstance(loss, dict):
        loss_fn = {k: parse_loss_name(v, ddp) for k, v in loss.items()}
    else:
        raise TypeError(f"Unknown loss type: {type(loss)}")

    return loss_fn
