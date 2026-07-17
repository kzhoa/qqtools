from typing import Optional

import torch
import torch.nn as nn

from ...entry_utils.qema import qEMA
from ...qlogger import qLogger


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


class _EMAEvaluationSession:
    """One-shot context session for a single evaluation run."""

    def __init__(
        self,
        eval_model: nn.Module,
        offload_target: nn.Module,
        ema_model: Optional[qEMA],
        device: torch.device,
        use_ema: bool,
        use_offload: bool,
        logger: Optional[qLogger] = None,
    ):
        self.eval_model = eval_model
        self.offload_target = offload_target
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
            self._log("Offloading model parameters to 'cpu' for EMA evaluation.")
            self.offload_target.cpu()
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
                self._log(f"Restoring model parameters to '{self.device}' after evaluation.")
                self.offload_target.to(self.device)


class EMAOffloadContext:
    """Factory that creates one-shot EMA/offload evaluation context sessions."""

    def __init__(
        self,
        main_model: nn.Module,
        ema_model: Optional[qEMA],
        device: torch.device,
        logger: Optional[qLogger] = None,
        auto_offload: bool = True,
    ):
        self.main_model = main_model
        self.ema_model = ema_model
        self.device = device
        self.logger = logger

        self._auto_offload_enabled = auto_offload and self.ema_model is not None

    def __call__(self, model: nn.Module, use_ema: bool) -> _EMAEvaluationSession:
        use_offload = self._auto_offload_enabled and (model is self.main_model)
        offload_target = _unwrap_model(self.main_model)
        return _EMAEvaluationSession(
            eval_model=model,
            offload_target=offload_target,
            ema_model=self.ema_model,
            device=self.device,
            use_ema=use_ema,
            use_offload=use_offload,
            logger=self.logger,
        )
