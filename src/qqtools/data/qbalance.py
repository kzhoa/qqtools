from __future__ import annotations

import heapq
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import fsum, nextafter
from typing import Callable, Iterable

import numpy as np

_BalanceStrategy = Callable[[np.ndarray, int], np.ndarray]


__all__ = [
    "assign_window_to_ranks",
    "compute_balanced_batch_indices",
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

    sorted_idx = np.argsort(sample_costs, kind="stable")
    num_bins = min(total, int(np.clip(np.sqrt(total), 32, 4096)))

    bins = np.array_split(sorted_idx, num_bins)
    for chunk in bins:
        if len(chunk) > 1:
            rng.shuffle(chunk)

    order = np.empty(total, dtype=np.int64)
    # Pair opposite cost quantiles instead of visiting high-cost bins consecutively.
    bucket_order = np.empty(num_bins, dtype=np.int64)
    low_count = (num_bins + 1) // 2
    bucket_order[::2] = np.arange(low_count)
    bucket_order[1::2] = np.arange(num_bins - 1, low_count - 1, -1)
    ordered_bins = [bins[index] for index in bucket_order]
    next_position = 0
    for position in range(len(bins[0])):
        for chunk in ordered_bins:
            if position < len(chunk):
                order[next_position] = chunk[position]
                next_position += 1

    return order


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


_LPT_STRATEGIES = ("lpt_fast", "lpt-medium", "lpt_best")


def _normalize_lpt_strategy(strategy: str) -> str:
    """Resolve the permanent short alias before planning or caching."""
    return "lpt-medium" if strategy == "lpt" else strategy


_LPT_REFINEMENT_PASSES = 4
_LPT_MAX_PAIRS_PER_PASS = 4096
_LPT_REPAIR_WINDOWS = 16
_LPT_SECONDARY_REPAIR_WINDOWS = 4
_LPT_REPAIR_CANDIDATES = 2
_LPT_REPAIR_SLOTS = 3


def _partition_layered_batches(
    costs: np.ndarray, order: np.ndarray, batch_size: int
) -> np.ndarray:
    """Give each batch one item per layer, largest item to the lightest batch."""
    batch_count = len(order) // batch_size
    layers = np.empty((batch_size, batch_count), dtype=np.int64)
    loads = np.zeros(batch_count, dtype=np.float64)
    with np.errstate(over="ignore"):
        for layer, items in enumerate(order.reshape(batch_size, batch_count)):
            targets = np.argsort(loads, kind="stable")
            layers[layer, targets] = items
            loads[targets] += costs[items]
    return layers.T


def _best_rank_batch_swaps(
    high_costs: np.ndarray, low_costs: np.ndarray, gaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prevalidated descending rows; at most 2*B-1 visits per pair, no B-by-B grid."""
    pair_count, batch_size = high_costs.shape
    high_pos = np.zeros(pair_count, dtype=np.int64)
    low_pos = np.zeros(pair_count, dtype=np.int64)
    best_high = np.full(pair_count, -1, dtype=np.int64)
    best_low = np.full(pair_count, -1, dtype=np.int64)
    best_gap = gaps.copy()
    active = np.flatnonzero(gaps > 0)
    for _ in range(2 * batch_size - 1):
        if not len(active):
            break
        delta = high_costs[active, high_pos[active]] - low_costs[active, low_pos[active]]
        # Clipping also keeps comparisons/subtractions finite for extreme costs.
        bounded_delta = np.clip(delta, 0.0, gaps[active])
        residual = np.abs((gaps[active] - bounded_delta) - bounded_delta)
        improved = active[residual < best_gap[active]]
        best_high[improved], best_low[improved] = high_pos[improved], low_pos[improved]
        best_gap[active] = np.minimum(best_gap[active], residual)
        # Descending high values decrease delta; descending low values increase it.
        should_advance_high = bounded_delta > gaps[active] - bounded_delta
        high_pos[active] += should_advance_high
        low_pos[active] += ~should_advance_high
        active = active[(high_pos[active] < batch_size) & (low_pos[active] < batch_size)
                        & (best_gap[active] > 0)]
    return best_high, best_low


def _swap_layered_rank_batches(
    costs: np.ndarray, batches: np.ndarray, world_size: int | None,
) -> np.ndarray:
    """Input rows come from layered LPT and are already descending by cost."""
    batch_count, batch_size = batches.shape
    if batch_count < 2 or batch_size == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if not np.all(np.isfinite(loads)) or np.all(loads == loads[0]):
        return batches
    order = np.argsort(loads, kind="stable")
    pair_count = batch_count // 2
    low, high = order[:pair_count], order[-pair_count:][::-1]
    high_slot, low_slot = _best_rank_batch_swaps(
        costs[batches[high]], costs[batches[low]], loads[high] - loads[low],
    )
    should_swap = high_slot >= 0
    if not np.any(should_swap):
        return batches
    high, low = high[should_swap], low[should_swap]
    high_slot, low_slot = high_slot[should_swap], low_slot[should_swap]
    trial = batches.copy()
    trial[high, high_slot], trial[low, low_slot] = (
        trial[low, low_slot], trial[high, high_slot],
    )
    trial_loads = _rank_batch_loads(costs, trial)
    before = _rank_batch_quality(loads, world_size)
    after = _rank_batch_quality(trial_loads, world_size)
    if after >= before:
        return batches
    if (world_size is not None and after[:2] == before[:2]
            and _has_worse_raw_step_sum(loads, trial_loads, world_size)):
        return batches
    return trial


def _partition_capacity_batches(
    costs: np.ndarray, order: np.ndarray, batch_size: int
) -> np.ndarray:
    """Assign descending items to the least-loaded batch with remaining capacity."""
    batch_count = len(order) // batch_size
    batches = np.empty((batch_count, batch_size), dtype=np.int64)
    available = [(0.0, 0, batch) for batch in range(batch_count)]
    for sample_idx in order:
        load, count, batch = available[0]
        batches[batch, count] = sample_idx
        next_load = load + float(costs[sample_idx])
        next_count = count + 1
        if next_count == batch_size:
            heapq.heappop(available)
        else:
            heapq.heapreplace(available, (next_load, next_count, batch))
    return batches


def _rank_batch_loads(costs: np.ndarray, batches: np.ndarray) -> np.ndarray:
    # An overflowing candidate must not prevent another tier candidate from winning.
    with np.errstate(over="ignore"):
        return costs[batches].sum(axis=1)


def _rank_batch_quality(
    loads: np.ndarray, world_size: int | None = None,
) -> tuple[float, float, float]:
    """Peak, P99, normalized squared loads; explicit world size scores step maxima."""
    if not np.all(np.isfinite(loads)):
        return (float("inf"),) * 3
    if not len(loads):
        return 0.0, 0.0, 0.0
    ordered = np.sort(loads)
    peak = float(ordered[-1])
    position = 0.99 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p99 = float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    if world_size is None:
        # Equal peak normalization keeps the third key finite and comparable.
        squared_sum = float(np.square(ordered / peak).sum()) if peak else 0.0
        return peak, p99, squared_sum
    # When the first two keys tie, dividing by their common peak preserves ordering
    # without overflowing on a large epoch whose individual batch costs are finite.
    step_max_sum = float((ordered[world_size - 1::world_size] / peak).sum()) if peak else 0.0
    return peak, p99, step_max_sum


def _swap_rank_batch_pair(
    costs: np.ndarray, batches: np.ndarray, loads: np.ndarray, first: int, second: int
) -> None:
    """Find a one-for-one exchange near half the load gap in O(B log B) time."""
    hi, lo = (first, second) if loads[first] >= loads[second] else (second, first)
    gap = float(loads[hi] - loads[lo])
    if gap <= 0:
        return
    low_positions = np.argsort(costs[batches[lo]], kind="stable")
    low_costs = costs[batches[lo, low_positions]].tolist()
    best_gap = gap
    best_swap = None
    for high_position, sample in enumerate(batches[hi]):
        weight = float(costs[sample])
        insertion = bisect_left(low_costs, weight - gap / 2)
        for candidate in (insertion - 1, insertion):
            if not 0 <= candidate < len(low_costs):
                continue
            delta = weight - low_costs[candidate]
            residual = abs((gap - delta) - delta) if 0 < delta < gap else gap
            if residual < best_gap:
                best_gap = residual
                best_swap = high_position, int(low_positions[candidate])
    if best_swap is not None:
        high_position, low_position = best_swap
        batches[hi, high_position], batches[lo, low_position] = (
            batches[lo, low_position], batches[hi, high_position]
        )


def _refine_rank_batches(
    costs: np.ndarray, batches: np.ndarray, world_size: int | None, seed: int,
    *, passes: int = _LPT_REFINEMENT_PASSES,
) -> np.ndarray:
    """Bounded paired swaps; retain the incumbent under the full tier objective."""
    trial = batches.copy()
    best = batches
    loads = _rank_batch_loads(costs, trial)
    best_quality = _rank_batch_quality(loads, world_size)
    rng = np.random.default_rng(seed)
    pair_count = min(len(trial) // 2, _LPT_MAX_PAIRS_PER_PASS)
    for pass_index in range(passes):
        if not pair_count or np.all(loads == loads[0]):
            break
        order = (np.argsort(loads, kind="stable") if pass_index % 2 == 0
                 else rng.permutation(len(trial)))
        for first, second in zip(order[:pair_count], order[-pair_count:][::-1]):
            _swap_rank_batch_pair(costs, trial, loads, int(first), int(second))
        loads = _rank_batch_loads(costs, trial)
        quality = _rank_batch_quality(loads, world_size)
        if quality < best_quality:
            best, best_quality = trial.copy(), quality
    return best


def _rank_batch_peak_lower_bound(
    costs: np.ndarray, order: np.ndarray, batch_size: int
) -> float:
    """Conservative float bound using the actual selected occurrences, sorted descending."""
    largest = float(costs[order[0]])
    batch_count = len(order) // batch_size
    try:
        mean = fsum(costs[order] / batch_count)
        # Allow for division/summation rounding and subnormal division errors.
        error = 2 * np.finfo(float).eps * mean + len(order) * nextafter(0.0, 1.0)
        mean = max(0.0, nextafter(mean - error, 0.0))
    except OverflowError:
        mean = 0.0
    try:
        companions = costs[order[-(batch_size - 1):]] if batch_size > 1 else ()
        paired = fsum((largest, *companions))
        paired = nextafter(nextafter(paired, 0.0), 0.0)
    except OverflowError:
        paired = 0.0
    return max(largest, mean, paired)


@lru_cache(maxsize=6)
def _small_batch_assignments(group_count: int, slots: int) -> np.ndarray:
    """All labelled assignments to two/three equal-capacity groups (at most 1680)."""
    items = tuple(range(group_count * slots))
    assignments = []
    for first in combinations(items, slots):
        rest = tuple(i for i in items if i not in first)
        if group_count == 2:
            assignments.append((first, rest))
            continue
        for second in combinations(rest, slots):
            third = tuple(i for i in rest if i not in second)
            assignments.append((first, second, third))
    result = np.asarray(assignments, dtype=np.int64)
    result.setflags(write=False)
    return result


def _rank_batch_repair_candidates(
    costs: np.ndarray, rows: np.ndarray, window: int
) -> list[np.ndarray]:
    """Reassign at most nine occurrences; shortlist distinct locally balanced load vectors."""
    group_count, batch_size = rows.shape
    slots = min(batch_size, _LPT_REPAIR_SLOTS)
    positions = np.argsort(costs[rows], axis=1, kind="stable")
    selected = (np.linspace(0, batch_size - 1, slots, dtype=int) + window) % batch_size
    positions = positions[:, selected]
    row_ids = np.arange(group_count)[:, None]
    pool = rows[row_ids, positions].ravel()
    is_fixed = np.ones(rows.shape, dtype=bool)
    is_fixed[row_ids, positions] = False
    with np.errstate(over="ignore"):
        residual = np.where(is_fixed, costs[rows], 0.0).sum(axis=1)
        assignments = _small_batch_assignments(group_count, slots)
        loads = costs[pool[assignments]].sum(axis=2) + residual
    ordered = np.sort(loads, axis=1)
    # Local scores only generate candidates; the complete epoch score decides acceptance.
    ranking = np.lexsort(tuple(ordered[:, j] for j in range(group_count)))
    ranking = ranking[np.all(np.isfinite(ordered[ranking]), axis=1)]
    if not len(ranking):
        return []
    ranked = ordered[ranking]
    is_distinct = np.r_[True, np.any(ranked[1:] != ranked[:-1], axis=1)]
    ranking = ranking[is_distinct][:_LPT_REPAIR_CANDIDATES]
    candidates = []
    for index in ranking:
        candidate = rows.copy()
        candidate[row_ids, positions] = pool[assignments[index]]
        candidates.append(candidate)
    return candidates


def _has_worse_raw_step_sum(loads: np.ndarray, trial: np.ndarray, world_size: int) -> bool:
    """Guard rounding disagreements between normalized and raw step-max sums."""
    with np.errstate(over="ignore"):
        before = np.sort(loads)[world_size - 1::world_size].sum()
        after = np.sort(trial)[world_size - 1::world_size].sum()
    return bool(after > before)


def _repair_rank_batches(
    costs: np.ndarray, batches: np.ndarray, world_size: int | None,
    seed: int, peak_lower_bound: float
) -> np.ndarray:
    """Bounded three-batch repair; retain the existing best under the full epoch objective."""
    if len(batches) < 2 or batches.shape[1] == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    best_quality = _rank_batch_quality(loads, world_size)
    best = batches.copy()
    rng = np.random.default_rng(seed)
    group_count = min(3, len(batches))
    small_groups = list(combinations(range(len(batches)), group_count)) if len(batches) <= 6 else []
    for window in range(_LPT_REPAIR_WINDOWS):
        is_peak_at_bound = best_quality[0] <= peak_lower_bound
        if is_peak_at_bound and window >= _LPT_SECONDARY_REPAIR_WINDOWS:
            break
        order = np.argsort(loads, kind="stable")
        if small_groups:
            groups = order[list(small_groups[window % len(small_groups)])]
        else:
            anchor = int(0.99 * (len(loads) - 1)) if is_peak_at_bound else len(loads) - 1
            anchor = max(0, anchor - window % min(4, anchor + 1))
            partners = rng.choice(len(loads) - 1, size=2, replace=False)
            partners += partners >= anchor
            groups = order[np.r_[anchor, partners]]
        rows = best[groups].copy()
        for candidate in _rank_batch_repair_candidates(costs, rows, window):
            candidate_loads = _rank_batch_loads(costs, candidate)
            if not np.all(np.isfinite(candidate_loads)):
                continue
            trial_loads = loads.copy()
            trial_loads[groups] = candidate_loads
            quality = _rank_batch_quality(trial_loads, world_size)
            if quality >= best_quality:
                continue
            if world_size is not None and quality[:2] == best_quality[:2] and _has_worse_raw_step_sum(
                loads, trial_loads, world_size
            ):
                continue
            best[groups] = candidate
            loads, best_quality = trial_loads, quality
    return best


@dataclass(frozen=True)
class _RankBatchPartition:
    """World-independent grouping: full rows plus fewer than B unassigned indices."""

    full_batches: np.ndarray
    remainder: np.ndarray


def _partition_selected_batches(
    costs: np.ndarray, selected: np.ndarray, batch_size: int, seed: int, strategy: str,
) -> np.ndarray:
    """Balance prevalidated occurrences divisible by B, without step-level objectives."""
    if not len(selected):
        return np.empty((0, batch_size), dtype=np.int64)
    order = selected[np.argsort(-costs[selected], kind="stable")]
    batches = _partition_layered_batches(costs, order, batch_size)
    batches = _swap_layered_rank_batches(costs, batches, None)
    loads = _rank_batch_loads(costs, batches)
    has_fixed_loads = (
        batch_size == 1 or len(batches) == 1
        or (np.all(np.isfinite(loads)) and np.all(loads == loads[0]))
    )
    if strategy != "lpt_fast" and not has_fixed_loads:
        candidate = _partition_capacity_batches(costs, order, batch_size)
        candidate_loads = _rank_batch_loads(costs, candidate)
        if _rank_batch_quality(candidate_loads) <= _rank_batch_quality(loads):
            batches, loads = candidate, candidate_loads
    if not np.all(np.isfinite(loads)):
        raise ValueError("Accumulated batch costs must remain finite")
    if strategy == "lpt_best" and not has_fixed_loads:
        batches = _refine_rank_batches(costs, batches, None, seed)
        lower_bound = float(costs[order[0]])
        if _rank_batch_loads(costs, batches).max() > lower_bound:
            lower_bound = _rank_batch_peak_lower_bound(costs, order, batch_size)
        batches = _repair_rank_batches(costs, batches, None, seed, lower_bound)
    return np.ascontiguousarray(batches)


def _validate_rank_batch_settings(batch_size: int, strategy: str) -> int:
    if strategy not in _LPT_STRATEGIES:
        raise ValueError(
            f"Unsupported rank-batch strategy {strategy!r}; expected {_LPT_STRATEGIES}"
        )
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError(f"batch_size must be an integer, got {batch_size!r}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return int(batch_size)


def _partition_rank_batches(
    sample_costs: np.ndarray | list[float],
    *,
    batch_size: int,
    seed: int = 0,
    strategy: str = "lpt-medium",
) -> _RankBatchPartition:
    """Partition every index exactly once, with no rank count, padding or dropping."""
    strategy = _normalize_lpt_strategy(strategy)
    batch_size = _validate_rank_batch_settings(batch_size, strategy)
    costs = _normalize_sample_costs(sample_costs)
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(costs))
    full_size = len(costs) // batch_size * batch_size
    batches = _partition_selected_batches(
        costs, selected[:full_size], batch_size, seed, strategy,
    )
    remainder = selected[full_size:].copy()
    batches.setflags(write=False)
    remainder.setflags(write=False)
    return _RankBatchPartition(batches, remainder)


def compute_balanced_batch_indices(
    sample_costs: np.ndarray | list[float],
    *,
    batch_size: int,
    seed: int = 0,
    strategy: str = "lpt",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed-size cost-balanced batches without rank assignment.

    Args:
        sample_costs: One-dimensional finite, nonnegative additive costs. Calculations
            use float64; balancing costs does not guarantee GPU memory safety.
        batch_size: Positive integer number of samples per complete batch.
        seed: Nonnegative random seed. Identical inputs reproduce the partition within
            an algorithm version, not necessarily across versions.
        strategy: "lpt_fast", "lpt-medium" (alias "lpt"), or "lpt_best".

    Returns:
        A plain tuple (batch_indices, remaining_indices) of read-only, contiguous int64
        arrays with shapes (N // batch_size, batch_size) and (N % batch_size,).
        Every input position occurs exactly once across the two arrays. Remainder
        positions are selected using seed before balancing and are not optimized as a
        batch. Empty input is supported. No padding, dropping, caching, epoch shuffle,
        or world-size-dependent refinement is performed. This is a heuristic, not an
        exact optimizer; reshuffling flattened indices loses the batch structure.

    Raises:
        TypeError: batch_size is not an integer, or an input has an invalid type.
        ValueError: Costs, batch_size, seed, or strategy are invalid, or accumulated
            batch costs overflow float64.
    """
    partition = _partition_rank_batches(
        sample_costs, batch_size=batch_size, seed=seed, strategy=strategy,
    )
    return partition.full_batches, partition.remainder


def _complete_rank_batch_tail(
    costs: np.ndarray, partition: _RankBatchPartition, world_size: int,
    seed: int, should_drop_last: bool, strategy: str,
) -> np.ndarray:
    """Repair at most 2R-1 base rows plus remainder; leave all other rows intact."""
    batches, remainder = partition.full_batches, partition.remainder
    batch_count, batch_size = batches.shape
    if not len(remainder) and batch_count % world_size == 0:
        return batches
    # Keep a whole number of steps outside the pool. One extra step supplies
    # companions for a heavy remainder instead of padding it in isolation.
    pool_count = min(batch_count, batch_count % world_size + world_size)
    rng = np.random.default_rng(seed)
    pool_ids = rng.choice(batch_count, size=pool_count, replace=False)
    is_kept = np.ones(batch_count, dtype=bool)
    is_kept[pool_ids] = False
    pool = np.concatenate((batches[pool_ids].ravel(), remainder))
    rng.shuffle(pool)
    global_size = batch_size * world_size
    target = ((len(pool) // global_size) if should_drop_last
              else ((len(pool) + global_size - 1) // global_size)) * global_size
    selected = pool[:target] if should_drop_last else np.resize(pool, target)
    repaired = _partition_selected_batches(costs, selected, batch_size, seed, strategy)
    return np.ascontiguousarray(np.concatenate((batches[is_kept], repaired)))


def _optimize_fast_step_batches(
    costs: np.ndarray, batches: np.ndarray, world_size: int,
) -> np.ndarray:
    """One sorted-row two-pointer pass on a derived copy; protect all three metrics."""
    if world_size == 1 or len(batches) < 2 or batches.shape[1] == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    positions = np.argsort(-costs[batches], axis=1, kind="stable")
    sorted_batches = np.take_along_axis(batches, positions, axis=1)
    candidate = _swap_layered_rank_batches(costs, sorted_batches, world_size)
    candidate_loads = _rank_batch_loads(costs, candidate)
    before = _rank_batch_quality(loads, world_size)
    after = _rank_batch_quality(candidate_loads, world_size)
    if (after < before and after[0] <= before[0] and after[1] <= before[1]
            and not _has_worse_raw_step_sum(loads, candidate_loads, world_size)):
        return candidate
    return batches


def _optimize_step_batches(
    costs: np.ndarray, batches: np.ndarray, world_size: int, seed: int,
    *, should_use_best: bool = True,
) -> np.ndarray:
    """Refine a derived plan without mutating base batches or worsening peak/P99/step sum."""
    if world_size == 1 or len(batches) < 2 or batches.shape[1] == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    best = batches
    quality = _rank_batch_quality(loads, world_size)
    for should_repair in ((False, True) if should_use_best else (False,)):
        if should_repair:
            candidate = _repair_rank_batches(costs, best, world_size, seed, 0.0)
        else:
            candidate = _refine_rank_batches(
                costs, best, world_size, seed,
                passes=_LPT_REFINEMENT_PASSES if should_use_best else 1,
            )
        candidate_loads = _rank_batch_loads(costs, candidate)
        candidate_quality = _rank_batch_quality(candidate_loads, world_size)
        if (candidate_quality < quality and candidate_quality[0] <= quality[0]
                and candidate_quality[1] <= quality[1]
                and not _has_worse_raw_step_sum(loads, candidate_loads, world_size)):
            best, loads, quality = candidate, candidate_loads, candidate_quality
    return best


def _plan_rank_batches(
    sample_costs: np.ndarray | list[float],
    *,
    batch_size: int,
    world_size: int = 1,
    seed: int = 0,
    should_drop_last: bool = False,
    should_shuffle: bool = True,
    strategy: str = "lpt-medium",
) -> np.ndarray:
    """Group independently of R, repair the small DDP tail, then compose steps.

    Base tiers minimize (peak, P99, squared loads), not step-level scores. Tail
    repair may change this ordering across tiers. Pad/drop counts match full global
    batches, but selected occurrences differ from the former global-prefix policy.
    All tiers refine derived copies for the actual rank count, protecting peak,
    P99 and raw step-max sum within each search. On divisible inputs higher tiers
    retain the lower derived candidate by lexicographic step quality; individual
    secondary metrics can trade off when a higher-priority metric improves.
    Adjacent loads then form steps.
    Optional traversal shuffle preserves the completed plan.
    No disk cache is read or written.
    """
    if isinstance(world_size, (bool, np.bool_)) or not isinstance(world_size, (int, np.integer)):
        raise TypeError(f"world_size must be an integer, got {world_size!r}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    world_size = int(world_size)
    strategy = _normalize_lpt_strategy(strategy)
    batch_size = _validate_rank_batch_settings(batch_size, strategy)
    costs = _normalize_sample_costs(sample_costs)
    rng = np.random.default_rng(seed)
    if should_drop_last and len(costs) < batch_size * world_size:
        return np.empty((0, world_size, batch_size), dtype=np.int64)
    partition = _partition_rank_batches(
        costs, batch_size=batch_size, seed=seed, strategy=strategy,
    )
    batches = _complete_rank_batch_tail(
        costs, partition, world_size, seed, should_drop_last, strategy,
    )
    if strategy == "lpt_fast":
        batches = _optimize_fast_step_batches(costs, batches, world_size)
    else:
        batches = _optimize_step_batches(
            costs, batches, world_size, seed, should_use_best=strategy == "lpt_best",
        )
    # With identical occurrences, retain the actual lower-tier derived candidate.
    # Tail pools differ by tier; do not compare plans that may drop different samples.
    if (strategy != "lpt_fast" and world_size > 1 and batch_size > 1
            and len(costs) % (batch_size * world_size) == 0 and len(batches) > 1
            and np.ptp(_rank_batch_loads(costs, batches)) > 0):
        middle = _plan_rank_batches(
            costs, batch_size=batch_size, world_size=world_size, seed=seed,
            should_drop_last=should_drop_last, should_shuffle=False,
            strategy="lpt-medium" if strategy == "lpt_best" else "lpt_fast",
        ).reshape(-1, batch_size)
        if _rank_batch_quality(_rank_batch_loads(costs, middle), world_size) < (
            _rank_batch_quality(_rank_batch_loads(costs, batches), world_size)
        ):
            batches = middle
    loads = _rank_batch_loads(costs, batches)

    step_batches = np.argsort(loads, kind="stable").reshape(-1, world_size)
    if should_shuffle:
        rng.shuffle(step_batches, axis=0)
        step_batches = rng.permuted(step_batches, axis=1)
        batches = rng.permuted(batches, axis=1)
    return np.ascontiguousarray(batches[step_batches])


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
