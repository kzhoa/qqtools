import sys
from importlib import import_module

__all__ = ["LazyImport", "is_imported"]


def _try_import(pkg):
    try:
        _module = __import__(str(pkg))
        return _module
    except ImportError:
        return None


def is_imported(module_name: str):
    ans = module_name in sys.modules
    print(f"{module_name}{''if ans else ' not'} in sys.modules")
    return ans


class LazyImport:
    """LazyImport

    example:
        # import numpy as np
        np = LazyImport("numpy")
    """

    def __init__(self, module_name, fromlist=None):
        self.module_name = module_name
        self.fromlist = fromlist
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = import_module(self.module_name)
            # self.module = __import__(self.module_name)
        return getattr(self.module, name)


def common_imports():
    import os
    import pickle
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    g = globals()
    g["torch"] = torch
    g["np"] = np
    g["tqdm"] = tqdm
    g["pd"] = pd
    g["pickle"] = pickle
    g["Path"] = Path
    g["os"] = os
    g["sys"] = sys
    g["time"] = time
