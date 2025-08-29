import numpy as np
import torch

__all__ = ["ensure_number"]


def str2number(inpt):
    if inpt is None:
        return None
    elif str.is_numeric(inpt):
        num = float(inpt)
        if num.is_integer():
            num = int(num)
        return num
    else:
        raise ValueError("inpt must be string of number")


def ensure_number(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.item()
    elif isinstance(x, str):
        return str2number(x)
    else:
        raise TypeError(f"type({x})")


def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, (int, float)):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise TypeError(f"type({x})")
