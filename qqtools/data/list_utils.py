from typing import List

import numpy as np


def pad_sequences(sequences: List[List], max_len=None, dtype=float):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    n_sequences = len(sequences)
    result = np.zeros((n_sequences, max_len), dtype=dtype)
    for i, seq in enumerate(sequences):
        result[i, : len(seq)] = seq

    return result
