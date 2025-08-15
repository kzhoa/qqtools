import torch


def get_nonlinear(name: str):
    if name == "silu":
        return torch.nn.SiLU()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "tanh":
        return torch.nn.Tanh()
    else:
        raise ValueError(f"Non-linearity {name} not supported")
