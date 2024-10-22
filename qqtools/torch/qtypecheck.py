import numpy as np
import torch

__all__ = ["ensureNumber"]


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


def ensureNumber(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.item()
    else:
        raise TypeError(f"type({x})")
