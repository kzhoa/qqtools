"""
Naming/behavior convention:
- `is_*()` functions: Return False on failure.
- `ensure_*()` functions: Raise an exception on failure.
"""

import math
from typing import List

import numpy as np
import torch

__all__ = [
    "is_number",
    "is_inf",
    "str2number",
    "ensure_scala",
    "ensure_numpy",
    "numpy_to_native",
]


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

    return False


def is_inf(x) -> bool:

    if isinstance(x, torch.Tensor):
        return torch.isinf(x).any().item()  # 返回 bool

    elif isinstance(x, (int, float)):
        return math.isinf(x)

    elif isinstance(x, (list, tuple)):
        return any(is_inf(item) for item in x)

    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()

    else:
        raise TypeError(f"Unsupported type: {type(x)}")


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


def numpy_to_native(obj):
    """
    Recursively convert numpy data types to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return complex(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.bytes_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_native(item) for item in obj)
    else:
        return obj
