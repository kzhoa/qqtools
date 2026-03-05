from typing import Optional

import torch
import torch.nn as nn

from ...entry_utils.qema import qEMA
from ...qlogger import qLogger


class EMAOffloadContext:
    """Context manager that temporarily moves models for EMA evaluation."""

    def __init__(
        self,
        main_model: nn.Module,
        ema_model: Optional[qEMA],
        device: torch.device,
        use_ema: bool,
        use_offload: bool,
        logger: Optional[qLogger] = None,
    ):
        self.main_model = main_model
        self.ema_model = ema_model
        self.device = device
        self.use_ema = use_ema
        self.use_offload = use_offload
        self.logger = logger

        self.ema_original_device = None
        self.offloaded = False

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.debug(msg)

    def __enter__(self) -> nn.Module:
        if not self.use_ema or self.ema_model is None:
            return self.main_model

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
