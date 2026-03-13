from typing import List

import numpy as np


def qbalance_partion(sizes: np.ndarray, num_parts: int) -> List[List[int]]:
    sizes = np.asarray(sizes, dtype=np.float64)
    num_items = len(sizes)
    sort_idx = np.argsort(-sizes)
    assignment = np.empty(num_items, dtype=np.int32)
    assignment[sort_idx] = np.arange(num_items, dtype=np.int32) % num_parts
    return [np.where(assignment == i)[0].tolist() for i in range(num_parts)]
