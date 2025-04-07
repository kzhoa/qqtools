import numpy as np
import random
import torch

__all__ = ["freeze_rand"]


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
