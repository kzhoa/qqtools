from typing import Sequence, Tuple

import numpy as np

__all__ = [
    # --- naive implementation ---
    "random_split_train_valid",
    "random_split_train_valid_test",
    # --- advanced implementation ---
    "get_data_splits",
]


def random_split_train_valid(num_samples, ratio=0.8, seed=None) -> Tuple[Sequence, Sequence]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_samples)
    train_num = int(ratio * num_samples)
    train_idx = perm[:train_num]
    valid_idx = perm[train_num:]
    return train_idx, valid_idx


def random_split_train_valid_test(
    num_samples, ratio: Sequence = [0.8, 0.1, 0.1], seed=None
) -> Tuple[Sequence, Sequence]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_samples)
    train_num = int(ratio[0] * num_samples)
    val_num = int(ratio[1] * num_samples)
    train_idx = perm[:train_num]
    valid_idx = perm[train_num : train_num + val_num]
    test_idx = perm[train_num + val_num :]
    return train_idx, valid_idx, test_idx


def get_data_splits(
    total_num,
    sizes=None,
    ratios=None,
    seed=None,
):
    """Split data into train/validation/test sets based on either sizes or ratios.

    Args:
        total_num (_type_): _description_
        sizes (Sequence[int], optional): Absolute sizes for splits, e.g., [800, 100, 100].
            If sizes only has 2 elements, the return value of test indices will be None.
            If sizes[2] is -1, test size will be calculated as total_num - train - val.
            Mutually exclusive with ratios parameter.
        ratios (Sequence[float], optional): Relative ratios for splits, e.g., [0.8, 0.1, 0.1].
            If ratios only has 2 elements, the return value of test indices will be None.
            The sum of ratios should not exceed 1 (with small tolerance for floating point errors).
            Mutually exclusive with sizes parameter.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.

    Returns:
        tuple: Three elements containing:
            - train_idx (ndarray): Indices for training set
            - valid_idx (ndarray): Indices for validation set
            - test_idx (ndarray or None): Indices for test set (None if not specified)

    Examples:
        >>> # Size-based splitting
        >>> train, val, test = get_data_splits(1000, sizes=[800, 100, 100])
        >>> # Ratio-based splitting
        >>> train, val, test = get_data_splits(1000, ratios=[0.8, 0.1, 0.1])
        >>> # Without test set
        >>> train, val, _ = get_data_splits(1000, ratios=[0.9, 0.1])
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_num)

    if sizes is not None:
        # Handle size-based splitting
        assert len(sizes) in [2, 3], "sizes must have length 2 or 3"
        assert all(isinstance(s, int) for s in sizes), "sizes must be integers"
        assert sum(sizes) <= total_num, "Sum of sizes exceeds total number of samples"
        train_size = sizes[0]
        val_size = sizes[1]
        if len(sizes) > 2:
            if sizes[2] == -1:
                test_size = total_num - train_size - val_size
                assert test_size >= 0, "Calculated test size is negative"
            else:
                test_size = sizes[2]
        else:
            test_size = None
        # Verify total size
        total_split = train_size + val_size + (test_size if test_size is not None else 0)
        assert total_split <= total_num, f"Sum of sizes ({total_split}) exceeds total number of samples ({total_num})"
        # Split indices
        train_end = train_size
        valid_end = train_end + val_size
        train_idx = indices[:train_end]
        valid_idx = indices[train_end:valid_end]

        if test_size is not None:
            test_idx = indices[valid_end : valid_end + test_size]
        else:
            test_idx = None

    elif ratios is not None:
        assert len(ratios) in [2, 3], "ratio must have length 2 or 3"
        assert sum(ratios) <= 1 + 1e-6, "Sum of ratios exceeds 1"

        train_end = int(ratios[0] * total_num)
        valid_end = train_end + int(ratios[1] * total_num)
        train_idx = indices[:train_end]
        valid_idx = indices[train_end:valid_end]

        if len(ratios) > 2:
            test_end = valid_end + int(ratios[2] * total_num)
            test_idx = indices[valid_end:test_end]
        else:
            test_idx = None
    else:
        raise ValueError("Either sizes or ratio must be provided")
    return train_idx, valid_idx, test_idx
