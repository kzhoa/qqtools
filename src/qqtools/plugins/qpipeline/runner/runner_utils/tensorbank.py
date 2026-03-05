from typing import Dict, List, Any, Union
import warnings
import torch

from qqtools.torch import qdist


class TensorBank:
    """
    A utility class to collect tensors/scalers across batches and efficiently
    gather them at the end of an epoch, handling distributed collecting and preventing OOM.

    Keys supplied are mapped to Lists of values (scalers mapped to Tensors, Tensors mapped to CPU Tensors).
    """

    def __init__(self, logger=None):
        self.bank: Dict[str, List[torch.Tensor]] = {}
        self.logger = logger
        self._warned_high_dim = False

    def reset(self):
        """Clear all cached tensors."""
        self.bank.clear()

    def add(self, batch_dict: Dict[str, Any]):
        """
        Add batch metrics/tensors. Moves to CPU automatically to prevent
        GPU OOM during caching.

        Args:
            batch_dict: Dictionary mapping names to items (scalers, arrays, tensors).
        """
        if not batch_dict:
            return

        for k, v in batch_dict.items():
            if not isinstance(v, torch.Tensor):
                # Convert scalars/lists to tensor
                v = torch.as_tensor(v)

            # Warn if collecting huge tensors (e.g. image heatmaps) to avoid RAM OOM
            if not self._warned_high_dim and v.dim() > 2 and v.numel() > 10000:
                warning_msg = "[TensorBank] `batch_cache` should only be used for low-dimensional metrics (e.g., labels, logits). Do not cache high-dimensional feature maps as they will cause system RAM OOM!"
                if self.logger:
                    self.logger.warning(warning_msg)
                else:
                    warnings.warn(warning_msg)
                self._warned_high_dim = True

            # Detach and move to CPU immediately
            v_cpu = v.detach().cpu()

            # ensure batched (D=0 means scaler, wrap to 1D)
            if v_cpu.dim() == 0:
                v_cpu = v_cpu.unsqueeze(0)

            if k not in self.bank:
                self.bank[k] = []

            self.bank[k].append(v_cpu)

    def gather(self, ddp: bool = False, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Concatenate stored batches into unified tensors.
        If DDP is True, it will also gather and concatenate across all processes using NCCL.

        Args:
            ddp: Whether to apply distributed all-gather.
            device: Target device for DDP communication (NCCL requires GPU device).
                   If None, it tries to detect the current CUDA device.

        Returns:
            Dictionary mapped to concatenated full-epoch CPU Tensors.
        """
        result = {}
        if not self.bank:
            return result

        for k, tensor_list in self.bank.items():
            if not tensor_list:
                continue

            # 1. Cat on CPU first
            local_cat = torch.cat(tensor_list, dim=0)

            if ddp and qdist.is_dist_available_and_initialized():
                # 2. To avoid "all_gather is not implemented for CPU tensors with NCCL backend"
                # We offload it to GPU, gather, and bring it back to CPU
                gathered_cat = qdist.all_gather_tensor(local_cat, device)
                result[k] = gathered_cat.cpu()
            else:
                result[k] = local_cat

        # Clear the list immediately to free memory
        self.reset()

        return result
