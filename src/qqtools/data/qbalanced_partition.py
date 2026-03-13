from typing import List

import numpy as np


def qbalanced_partition(sizes: np.ndarray, num_parts: int) -> List[List[int]]:
    """
    Partition items into `num_parts` groups with a simple size-aware round-robin strategy.

    The input sizes are first converted to a `float64` NumPy array. Items are then
    sorted in descending order so larger elements are assigned first. After sorting,
    indices are distributed across partitions in round-robin order, which provides
    a lightweight and deterministic approximation to balanced partitioning.

    Parameters
    ----------
    sizes : np.ndarray
        One-dimensional array-like object containing the size or weight of each item.
    num_parts : int
        Number of partitions to create.

    Returns
    -------
    List[List[int]]
        A list of length `num_parts`. Each element contains the original item indices
        assigned to that partition.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    num_items = len(sizes)
    sort_idx = np.argsort(-sizes)
    assignment = np.empty(num_items, dtype=np.int32)
    assignment[sort_idx] = np.arange(num_items, dtype=np.int32) % num_parts
    return [np.where(assignment == i)[0].tolist() for i in range(num_parts)]
