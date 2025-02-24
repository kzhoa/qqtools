from typing import Sequence, Tuple

import numpy as np


def random_split_train_valid(num_samples, ratio=0.8) -> Tuple[Sequence, Sequence]:
    perm = np.random.permutation(num_samples)
    train_num = int(ratio * num_samples)
    train_idx = perm[:train_num]
    valid_idx = perm[train_num:]
    return train_idx, valid_idx


def random_split_train_valid_test(num_samples, ratio: Sequence = [0.8, 0.1, 0.1]) -> Tuple[Sequence, Sequence]:
    perm = np.random.permutation(num_samples)
    train_num = int(ratio[0] * num_samples)
    val_num = int(ratio[1] * num_samples)
    train_idx = perm[:train_num]
    valid_idx = perm[train_num : train_num + val_num]
    test_idx = perm[train_num + val_num :]
    return train_idx, valid_idx, test_idx
