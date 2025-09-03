import numpy as np
import torch

__all__ = ["ensure_scala"]


def is_number(inpt) -> bool:
    if inpt is None or inpt == "":
        return False
    if isinstance(inpt, (float, int)):
        return True
    if isinstance(inpt, str):
        if inpt[0] == "-":
            inpt = inpt[1:]
        if "." in inpt:
            integ, _, frac = inpt.partition(".")
            return integ.isnumeric() and frac.isnumeric()
        else:
            return inpt.isnumeric()


def str2number(inpt):
    if inpt is None or inpt == "":
        return ValueError(f"input should not be None or empty")
    if not isinstance(inpt, str):
        raise TypeError(f"expect string input, got {type(inpt)}")
    if not is_number(inpt):
        raise ValueError(f"`{inpt}` is not a valid number")
    num = float(inpt)
    if num.is_integer():
        num = int(num)
    return num


def ensure_scala(x):
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
        return x.detach().cpu().numpy()
    elif isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise TypeError(f"type({x})")
