from typing import Optional

import torch
import torch.nn as nn

from ...entry_utils.qema import qEMA
from ...qlogger import qLogger


def _should_enable_offload(device: torch.device, model: nn.Module, logger: Optional[qLogger]) -> bool:
    """Check if model offloading should be enabled based on GPU capacity."""
    if device.type != "cuda":
        return False

    from ...entry_utils.info import get_model_size_bytes

    model_size_bytes = get_model_size_bytes(model)
    try:
        gpu_capacity_bytes = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return False

    if model_size_bytes > (gpu_capacity_bytes * 0.5):
        if logger:
            logger.info(
                f"Model size ({model_size_bytes / 1024**2:.2f} MB) > 50% of GPU capacity. "
                "Enabling mutual offloading during evaluation."
            )
        return True
    return False


class _EMAEvaluationSession:
    """One-shot context session for a single evaluation run."""

    def __init__(
        self,
        eval_model: nn.Module,
        main_model: nn.Module,
        ema_model: Optional[qEMA],
        device: torch.device,
        use_ema: bool,
        use_offload: bool,
        logger: Optional[qLogger] = None,
    ):
        self.eval_model = eval_model
        self.main_model = main_model
        self.ema_model = ema_model
        self.device = device
        self.use_ema = use_ema
        self.use_offload = use_offload
        self.logger = logger

        self.ema_original_device = None
        self.offloaded = False

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.debug(msg)

    def __enter__(self) -> nn.Module:
        if not self.use_ema or self.ema_model is None:
            return self.eval_model

        if self.use_offload:
            self._log("Offloading main model to 'cpu' for EMA evaluation.")
            self.main_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.offloaded = True

        try:
            self.ema_original_device = next(self.ema_model.parameters()).device
            if self.ema_original_device != self.device:
                self._log(f"Moving EMA model from '{self.ema_original_device}' to '{self.device}' for evaluation.")
                self.ema_model.to(self.device)
        except (StopIteration, Exception):
            self.ema_original_device = None

        return self.ema_model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.use_ema and self.ema_model is not None:
            if self.ema_original_device is not None and self.ema_original_device != self.device:
                self._log(f"Moving EMA model back to '{self.ema_original_device}' after evaluation.")
                self.ema_model.to(self.ema_original_device)

            if self.offloaded:
                self._log(f"Restoring main model to '{self.device}' after evaluation.")
                self.main_model.to(self.device)


class EMAOffloadContext:
    """Factory that creates one-shot EMA/offload evaluation context sessions."""

    def __init__(
        self,
        main_model: nn.Module,
        ema_model: Optional[qEMA],
        device: torch.device,
        logger: Optional[qLogger] = None,
        allow_auto_offload: bool = True,
    ):
        self.main_model = main_model
        self.ema_model = ema_model
        self.device = device
        self.logger = logger

        self.should_allow_auto_offload = allow_auto_offload
        self._auto_offload_enabled = False
        if self.should_allow_auto_offload and self.ema_model is not None:
            self._auto_offload_enabled = _should_enable_offload(self.device, self.main_model, self.logger)

    def __call__(self, model: nn.Module, use_ema: bool) -> _EMAEvaluationSession:
        use_offload = self._auto_offload_enabled and (model is self.main_model)
        return _EMAEvaluationSession(
            eval_model=model,
            main_model=self.main_model,
            ema_model=self.ema_model,
            device=self.device,
            use_ema=use_ema,
            use_offload=use_offload,
            logger=self.logger,
        )
