import copy
import functools
import traceback
from typing import Optional, Union

import torch
from torch.optim.swa_utils import AveragedModel


def _avg_fn(avg_params, model_params, num_averaged, decay):
    """Efficient EMA update using linear interpolation."""
    return avg_params.lerp_(model_params.detach(), 1 - decay)


class qEMA(AveragedModel):
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.99,
        device: Optional[Union[str, torch.device]] = None,
        use_buffers: bool = False,
    ):
        super(AveragedModel, self).__init__()  # Call nn.Module.__init__

        # qq: Use a dictionary to store the reference model.
        # Assigning it directly as an attribute (e.g., self.model = model)
        # would cause nn.Module to automatically register it as a submodule,
        # resulting in duplicated parameters when calling .parameters().
        self._cache_dict = {"reference_model": model}

        # Determine and normalize device
        self.device = torch.device(device) if device else next(model.parameters()).device

        # Initialize core AveragedModel attributes manually to avoid redundant deepcopy
        self.module = self._create_safe_model_copy(model).to(self.device)
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=self.device))
        self.avg_fn = functools.partial(_avg_fn, decay=decay)
        self.multi_avg_fn = None
        self.use_buffers = use_buffers

        print(f"[qEMA] initialized with decay: {decay} on device: {self.device}")

    def _create_safe_model_copy(self, model):
        try:
            if hasattr(model, "get_init_args") and callable(getattr(model, "get_init_args")):
                init_args = model.get_init_args()
                model_class = type(model)
                safe_model = model_class(**init_args)
                safe_model.load_state_dict(model.state_dict())
                print(f"Created EMA model using get_init_args method for {type(model).__name__}")
                return safe_model
            else:
                return copy.deepcopy(model)
        except Exception as e:
            print(f"Failed to create model using get_init_args or deepcopy due to: {e}")
            traceback.print_exc()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.module(*args, **kwargs)

    def update(self):
        """Update EMA parameters using the reference model."""
        with torch.no_grad():
            self.update_parameters(self._cache_dict["reference_model"])
