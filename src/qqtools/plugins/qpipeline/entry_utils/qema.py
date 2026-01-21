import copy
import functools
import traceback

import torch
from torch.optim.swa_utils import AveragedModel


def _avg_fn(avg_params, model_params, num_averaged, decay):
    return decay * avg_params + (1 - decay) * model_params.data


class qEMA(AveragedModel):
    def __init__(
        self,
        model,
        decay=0.99,
        device=None,
    ):
        self._cache_dict = {"model": model}
        self.device = device

        # init
        avg_fn = functools.partial(_avg_fn, decay=decay)
        if device is None:
            device = next(model.parameters()).device

        multi_avg_fn = None
        use_buffers = False
        torch.nn.Module.__init__(self)

        assert avg_fn is None or multi_avg_fn is None, "Only one of avg_fn and multi_avg_fn should be provided"
        self.module = self._create_safe_model_copy(model)

        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

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
        with torch.no_grad():
            self.update_parameters(self._cache_dict["model"])
