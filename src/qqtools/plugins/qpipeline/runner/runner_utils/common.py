from typing import Any, Optional

import torch

import qqtools as qt

from .types import RunMode


def move_batch_to_device(batch_data, device: torch.device):
    """Move batch data to device, handling various data structures."""
    if batch_data is None:
        return batch_data

    if hasattr(batch_data, "to"):
        return batch_data.to(device)
    if isinstance(batch_data, dict):
        return qt.qDict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()})
    if isinstance(batch_data, (tuple, list)):
        return type(batch_data)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch_data)
    return batch_data


def _getattr_or_default(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute value and fallback when missing or None."""
    value = getattr(obj, key, None)
    if value is not None:
        return value
    return default() if callable(default) else default


def _is_periodic_trigger(
    run_mode: RunMode,
    interval: Optional[int],
    global_step: int,
    epoch: int,
    is_epoch_end: bool,
) -> bool:
    """Check if a periodic event should fire for current train state."""
    if interval is None or interval <= 0:
        return False

    if run_mode == RunMode.STEP:
        return global_step > 0 and global_step % interval == 0

    return is_epoch_end and (epoch + 1) % interval == 0
