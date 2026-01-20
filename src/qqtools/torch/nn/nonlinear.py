from typing import Optional

import torch
import torch.nn.functional as F


class ShiftedSoftplus(torch.nn.Module):
    """
    Shifted Softplus activation function.

    Computes `softplus(x) - log(2)`.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def get_nonlinear(name: str, args: Optional[dict] = None):
    if name == "silu":
        return torch.nn.SiLU()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "leakyrelu":
        return torch.nn.LeakyReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "softplus":
        return torch.nn.Softplus()
    elif name in ["ssp", "ShiftedSoftplus"]:
        return ShiftedSoftplus()
    else:
        raise ValueError(f"Non-linearity {name} not supported")
