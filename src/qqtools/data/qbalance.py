from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

_BalanceStrategy = Callable[[np.ndarray, int], np.ndarray]


__all__ = [
    "assign_window_to_ranks",
    "compute_global_even_sort_order",
    "validate_balance_strategy",
]


def _normalize_sample_costs(sample_costs: np.ndarray | list[float]) -> np.ndarray:
    costs = np.asarray(sample_costs, dtype=np.float64)
    if costs.ndim != 1:
        raise ValueError(f"sample_costs must be 1D, got shape {costs.shape}")
    if not np.all(np.isfinite(costs)):
        raise ValueError("sample_costs must contain only finite values")
    if np.any(costs < 0):
        raise ValueError("sample_costs must be non-negative")
    return costs


def _validate_permutation(order: np.ndarray, total: int) -> np.ndarray:
    permutation = np.asarray(order, dtype=np.int64)
    if permutation.shape != (total,):
        raise ValueError(f"Unexpected permutation length: {permutation.shape}, expected {(total,)}")
    if total == 0:
        return permutation
    if len(np.unique(permutation)) != total:
        raise ValueError("Permutation contains duplicate indices.")
    if permutation.min(initial=0) != 0 or permutation.max(initial=-1) != total - 1:
        raise ValueError("Permutation contains missing indices.")
    return permutation


def _global_sort_v1(sample_costs: np.ndarray, seed: int) -> np.ndarray:
    total = int(len(sample_costs))
    if total == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.permutation(total).astype(np.int64)


def _global_sort_v2(sample_costs: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = int(len(sample_costs))
    if total == 0:
        return np.empty(0, dtype=np.int64)

    sorted_idx = np.argsort(sample_costs, kind="stable")
    num_bins = int(np.clip(np.sqrt(total), 64, 2048))
    bins = [chunk.tolist() for chunk in np.array_split(sorted_idx, num_bins)]

    for chunk in bins:
        if len(chunk) > 1:
            rng.shuffle(chunk)

    lows = bins[: num_bins // 2]
    highs = bins[num_bins // 2 :]
    order: list[int] = []
    low_ptr = 0
    high_ptr = 0
    should_take_low = True

    while len(order) < total:
        if should_take_low:
            source_bins = lows
            ptr = low_ptr
        else:
            source_bins = highs
            ptr = high_ptr

        attempts = 0
        while source_bins and attempts < len(source_bins):
            current = source_bins[ptr % len(source_bins)]
            if len(current) > 0:
                break
            ptr += 1
            attempts += 1
        else:
            should_take_low = not should_take_low
            continue

        order.append(current.pop(0))
        ptr += 1
        if should_take_low:
            low_ptr = ptr
        else:
            high_ptr = ptr
        should_take_low = not should_take_low

    return _validate_permutation(np.asarray(order, dtype=np.int64), total)


def _global_sort_v3(sample_costs: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = int(len(sample_costs))
    if total == 0:
        return np.empty(0, dtype=np.int64)

    global_mean = sample_costs.mean()
    sorted_idx = np.argsort(sample_costs, kind="stable")
    num_bins = int(np.clip(np.sqrt(total), 32, 4096))

    bins = np.array_split(sorted_idx, num_bins)
    for chunk in bins:
        if len(chunk) > 1:
            rng.shuffle(chunk)

    pointers = np.zeros(num_bins, dtype=np.int64)
    order = np.empty(total, dtype=np.int64)
    bin_means = np.asarray(
        [sample_costs[chunk].mean() if len(chunk) > 0 else 0.0 for chunk in bins],
        dtype=np.float64,
    )
    bin_priority = np.abs(bin_means - global_mean)
    if bin_priority.max() > 0:
        bin_priority /= bin_priority.sum()

    next_position = 0
    remaining = total
    window_size = max(8, num_bins // 4)

    while remaining > 0:
        for bin_idx in np.argsort(bin_priority):
            if pointers[bin_idx] >= len(bins[bin_idx]):
                continue
            order[next_position] = bins[bin_idx][pointers[bin_idx]]
            pointers[bin_idx] += 1
            next_position += 1
            remaining -= 1
            if remaining <= 0:
                break

        if next_position % window_size == 0:
            noise = rng.normal(0, 1e-3, size=num_bins)
            bin_priority = np.abs(bin_means - global_mean) + noise
            bin_priority = np.clip(bin_priority, 1e-6, None)
            bin_priority /= bin_priority.sum()

    return _validate_permutation(order.astype(np.int64), total)


_STRATEGIES: dict[str, _BalanceStrategy] = {
    "v1": _global_sort_v1,
    "v2": _global_sort_v2,
    "v3": _global_sort_v3,
}


def validate_balance_strategy(strategy: str) -> str:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy!r}. Expected one of {tuple(_STRATEGIES)}")
    return strategy


def compute_global_even_sort_order(
    sample_costs: np.ndarray | list[float],
    *,
    seed: int = 0,
    strategy: str = "v3",
) -> np.ndarray:
    costs = _normalize_sample_costs(sample_costs)
    validated_strategy = validate_balance_strategy(strategy)
    order = _STRATEGIES[validated_strategy](costs, int(seed))
    return _validate_permutation(order, int(costs.shape[0]))


def assign_window_to_ranks(
    window_indices: np.ndarray | list[int],
    sample_costs: np.ndarray | list[float],
    world_size: int,
    batch_size: int,
) -> list[list[int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    costs = _normalize_sample_costs(sample_costs)
    window = np.asarray(window_indices, dtype=np.int64)
    if window.ndim != 1:
        raise ValueError(f"window_indices must be 1D, got shape {window.shape}")
    if len(window) == 0:
        return [[] for _ in range(world_size)]
    if np.any(window < 0) or np.any(window >= costs.shape[0]):
        raise ValueError("window_indices contains out-of-range sample indices")

    sorted_positions = np.argsort(-costs[window], kind="stable")
    rank_to_items: list[list[int]] = [[] for _ in range(world_size)]
    rank_loads = np.zeros(world_size, dtype=np.float64)
    rank_counts = np.zeros(world_size, dtype=np.int64)

    def _locality_cost(rank_idx: int, sample_idx: int) -> float:
        if not rank_to_items[rank_idx]:
            return float("inf")
        return float(min(abs(sample_idx - existing_idx) for existing_idx in rank_to_items[rank_idx]))

    def _choose_rank(sample_idx: int, candidate_ranks: Iterable[int]) -> int:
        return min(
            candidate_ranks,
            key=lambda rank_idx: (
                float(rank_loads[rank_idx]),
                int(rank_counts[rank_idx]),
                _locality_cost(rank_idx, sample_idx),
                int(rank_idx),
            ),
        )

    for position in sorted_positions:
        sample_idx = int(window[position])
        chosen_rank = _choose_rank(sample_idx, range(world_size))
        if rank_counts[chosen_rank] >= batch_size:
            available_ranks = [rank_idx for rank_idx in range(world_size) if rank_counts[rank_idx] < batch_size]
            if not available_ranks:
                raise RuntimeError("No available rank left while assigning window")
            chosen_rank = _choose_rank(sample_idx, available_ranks)

        rank_to_items[chosen_rank].append(sample_idx)
        rank_loads[chosen_rank] += costs[sample_idx]
        rank_counts[chosen_rank] += 1

    return rank_to_items
