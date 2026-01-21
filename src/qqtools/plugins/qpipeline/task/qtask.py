from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import qqtools as qt
import torch
from torch import Tensor
from torch.utils.data import DataLoader

__all__ = ["qTaskBase", "PotentialTaskBase"]

OPTIONAL_METHODS = [
    # === functional hook ===
    "get_loss_fn",
    "extra_batch_metric",
    "epoch_metric",
    # === lifetime hook ===
    "onEpochStart",
    "onEpochEnd",
    "onBatchEnd",
    "on_better_model",
    "state_dict",
    "load_state_dict",
]

REQUIRED_METHODS = []


class qTaskBase(ABC):
    """
    Optional methods (can be implemented by users):

    - def get_loss_fn(self, args) -> Callable:
    - def extra_batch_metric
    - def epoch_metric(epoch_cache)
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    meta: Dict[str, Any]  # flow with epoch runner and saved in checkpoint

    _opt_impl = set()  # implemented optional methods
    _opt_todo = set()  # un-implemented optional methods
    _required = set(REQUIRED_METHODS)

    def __init__(self):
        """
        set 3 dataloaders:
            - self.train_loader
            - self.val_loader
            - self.test_loader
        """
        for name in OPTIONAL_METHODS:
            if qt.is_overriden(self, name, qTaskBase):
                self._opt_impl.add(name)
            else:
                self._opt_todo.add(name)

        for name in REQUIRED_METHODS:
            assert qt.is_overriden(self, name, qTaskBase)

    @abstractmethod
    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        """data format & model interface

        Returns:
            out (dict): dict-like
        """
        pass

    @abstractmethod
    def batch_loss(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'loss': (loss_value, sample_count) }`
        """
        pass

    @abstractmethod
    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'metric_name': (metric_value, sample_count) }`
        """
        pass

    @abstractmethod
    def post_metrics_to_value(self, result) -> float:
        """
        In cases where multiple metrics are available, the error metric must be prioritized
        to identify the optimal validation performance.

        Args:
            result: Dict[metric_name, metric_avg]

        Returns:
            float: The performance error value.
        """
        pass

    @staticmethod
    def pre_batch_forward(batch_data):
        """
        Args:
            batch_data (dict):

        Returns:
            dict: > dict of preprocessed data
        """
        return batch_data

    @staticmethod
    def post_batch_forward(output, batch_data):
        """
        Args:
           output :  return value of `batch_forward`

        Returns:
            dict: revised output
        """
        return output

    # other conventions:
    def get_loss_fn(self, args) -> Callable:
        raise NotImplementedError

    def extra_batch_metric(self, out, batch_data, losses, model, epoch_cache):
        """Compute extended metrics that require cross-batch information.

        Some metrics need information more than current batch
        (e.g., running statistics, cumulative performance, sequence-aware metrics).
        This hook provides access to the current training state and epoch-level
        cache to compute such cross-batch metrics.

        Note:
            Currently modifies `epoch_cache` in-place for state persistence
            across batches. TODO this is not elegant
            This implementation may be refined in future versions for better encapsulation.

        Args:
            out (dict): Model outputs
            batch_data (dict-like qData): Batch input data
            losses (dict): Computed losses
            model (nn.Module): Model instance
            epoch_cache (dict): Shared epoch state cache

        Returns:
            dict: Additional batch-level metric names and values
        """
        raise NotImplementedError

    def epoch_metric(self, epoch_cache) -> Dict:
        """
        take epoch cache and calculate epoch-wise metrics.

        `epoch_cache` can be revised through `extra_batch_metric`.

        Return:
            dict of epoch metrics
        """
        raise NotImplementedError

    def on_better_model(self, epoch_agent, current_state) -> None:
        raise NotImplementedError

    def to(self, device) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict:
        raise NotImplementedError

    def load_state_dict(self, d: Dict) -> None:
        raise NotImplementedError

    # def pipe_middle_ware(self, pipe: qPipeline):
    # pass


class PotentialTaskBase(qTaskBase):
    """

    Inherited from qTaskBase, designed for potential energy surface prediction tasks.
    Conventions:
        - implement `batch_forward`, `batch_loss`, `batch_metric`, `post_metrics_to_value` following qTaskBase;
        - implement `init_loader()` to set `train_loader`, `val_loader`, `test_loader`;
        - call `super().__init__(args)` in `__init__()`;
        - `args` should follow `qargs-convention`;
    """

    def __init__(self, args):
        self.extract_args(args)

        if hasattr(self, "init_loader"):
            self.init_loader()

    def extract_args(self, args):
        self.args = args.copy()

        batch_size = args.task.dataloader.batch_size
        eval_batch_size = args.task.dataloader.eval_batch_size or batch_size
        num_workers = args.task.dataloader.num_workers
        pin_memory = args.task.dataloader.pin_memory
        distributed = args.distributed
        standarize = args.task.get("standarize", False)
        with_force = args.task.get("with_force", False)

        loader_meta = {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "distributed": distributed,
        }

        meta = {
            "standarize": standarize,
            "with_force": with_force,
        }

        self.loader_meta = loader_meta
        self.meta = meta

        self.standarize = standarize
        self.with_force = with_force

    def batch_forward(self, model, batch_data, **meta) -> Dict[str, Tensor]:
        with_force = self.with_force
        z, pos, batch = batch_data["z"], batch_data["pos"], batch_data["batch"]
        out_force: Optional[Tensor] = None
        if with_force:
            pos.requires_grad_()
            result: Union[Tensor, Tuple[Tensor, Tensor]] = model(z, pos, batch)
            if isinstance(result, tuple):
                out_energy, out_force = result  # adaption for models that output force
            else:
                out_energy = result
                out_force = -torch.autograd.grad(
                    out_energy,
                    pos,
                    grad_outputs=torch.ones_like(out_energy),
                    create_graph=True,
                    allow_unused=True,
                )[0]

        else:
            out_energy = model(z, pos, batch)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {"energy": out_energy}
        if with_force and out_force is not None:
            if out_force.shape[-1] != 3:
                out_force = out_force.view(-1, 3)
            out["force"] = out_force

        return out

    def batch_metric(self, out, batch_data, **meta) -> Dict[str, Tuple[Tensor, int]]:
        mean, std = meta["norm_factor"]
        ret = dict()
        with torch.no_grad():
            e_out = out["energy"].detach().flatten()
            e_target = batch_data.energy.flatten()
            e_pred = e_out * std + mean
            e_mae = torch.nn.functional.l1_loss(e_pred, e_target)
            ret["e_mae"] = (e_mae, e_out.shape[0])

            if self.with_force:
                f_out = out["force"].flatten().detach()
                f_target = batch_data.force.flatten()
                f_pred = f_out * std + mean
                f_err = torch.nn.functional.l1_loss(f_pred, f_target)
                ret["f_mae"] = (f_err, f_out.numel)

        return ret

    def batch_loss(self, out, batch_data, loss_fn, **meta) -> Dict[str, Tuple[Tensor, int]]:
        mean, std = meta["norm_factor"]
        ret = dict()

        e_target = batch_data.energy.flatten()
        e_out = out["energy"] * std + mean
        e_loss: Tensor = loss_fn["energy"](e_out, (e_target - mean) / std)
        ret["e_loss"] = (e_loss, e_out.shape[0])
        if self.with_force:
            f_out = out["force"].view(-1, 3)
            f_target = batch_data.force.view(-1, 3)
            f_loss = loss_fn["force"](f_out, f_target / std)
            ret["f_loss"] = (f_loss, f_out.numel)
            ew, fw = self.ef_weight
            ret["loss"] = (ew * e_loss + fw * f_loss, e_out.shape[0])
        else:
            ret["loss"] = ret["e_loss"]
        return ret

    def post_metrics_to_value(self, result) -> float:
        if self.with_force:
            if hasattr(self, "ef_weight") and self.ef_weight is not None:
                ew, fw = self.ef_weight
            else:
                ew, fw = 0.8, 0.2
            ret = ew * result["e_mae"] + fw * result["f_mae"]
        else:
            ret = result["e_mae"]
        if torch.is_tensor(ret):
            ret = ret.detach().cpu().numpy()
        return ret


'''
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from qqtools.pipeline import qTaskBase

class TemplateTask(qTaskBase):

    def __init__(self, args):
        pass

    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        """
        Returns:
            out (dict): dict-like
        """
        pass

    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'metric_name': (metric_value, sample_count) }`
        """
        pass

    def batch_loss(self, out, batch_data, loss_fn) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'loss': (loss_value, sample_count) }`
        """
        pass

    def post_metrics_to_value(self, result) -> float:
        pass
'''
