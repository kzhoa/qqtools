"""Experimental multi-start, wider-neighborhood refinement for exact comparison."""

from functools import lru_cache
from itertools import combinations
from numbers import Integral

import numpy as np

from qqtools.data.qbalance import (
    _partition_capacity_batches, _partition_layered_batches, _rank_batch_loads,
    _rank_batch_quality, _refine_rank_batches, _swap_rank_batch_pair,
)


@lru_cache(maxsize=8)
def _splits(batch_size):
    return np.array([(0, *rest) for rest in combinations(range(1, 2 * batch_size),
                                                       batch_size - 1)], dtype=np.int64)


def _repartition_pair(costs, trial, loads, first, second, world_size):
    batch_size = trial.shape[1]
    if batch_size <= 6:
        items = np.concatenate((trial[first], trial[second]))
        splits = _splits(batch_size)
        partial = costs[items[splits]].sum(1)
        total = loads[first] + loads[second]
        peaks = np.maximum(partial, total - partial)
        chosen = int(np.argmin(peaks))
        if len(splits) * len(loads) <= 65536:
            # Score all pair repartitions by the actual global objective, not variance.
            candidates = np.broadcast_to(loads, (len(splits), len(loads))).copy()
            candidates[:, first], candidates[:, second] = partial, total - partial
            ordered = np.sort(candidates, axis=1)
            position = 0.99 * (len(loads) - 1)
            lower = int(position)
            upper = min(lower + 1, len(loads) - 1)
            p99 = ordered[:, lower] + (ordered[:, upper] - ordered[:, lower]) * (position - lower)
            step_max = ordered[:, world_size - 1::world_size].sum(1)
            chosen = int(np.lexsort((step_max, p99, ordered[:, -1]))[0])
            score = _rank_batch_quality(candidates[chosen], world_size)
            if score >= _rank_batch_quality(loads, world_size):
                return
            mask = np.ones(2 * batch_size, dtype=bool)
            mask[splits[chosen]] = False
            trial[first], trial[second] = items[splits[chosen]], items[mask]
            loads[[first, second]] = costs[trial[[first, second]]].sum(1)
            return
        if peaks[chosen] < max(loads[first], loads[second]):
            mask = np.ones(2 * batch_size, dtype=bool)
            mask[splits[chosen]] = False
            trial[first], trial[second] = items[splits[chosen]], items[mask]
    else:
        _swap_rank_batch_pair(costs, trial, loads, first, second)
        loads[[first, second]] = costs[trial[[first, second]]].sum(1)
        hi, lo = (first, second) if loads[first] >= loads[second] else (second, first)
        gap = loads[hi] - loads[lo]
        positions = np.triu_indices(batch_size, k=1)
        high_sums = costs[trial[hi, positions[0]]] + costs[trial[hi, positions[1]]]
        low_sums = costs[trial[lo, positions[0]]] + costs[trial[lo, positions[1]]]
        order = np.argsort(low_sums, kind="stable")
        insertions = np.searchsorted(low_sums[order], high_sums - gap / 2)
        candidates = np.clip(np.stack((insertions - 1, insertions)), 0, len(order) - 1)
        deltas = high_sums - low_sums[order[candidates]]
        residual = np.where((deltas > 0) & (deltas < gap), np.abs(gap - 2 * deltas), gap)
        side, high = np.unravel_index(np.argmin(residual), residual.shape)
        if residual[side, high] < gap:
            low = order[candidates[side, high]]
            hp = [positions[0][high], positions[1][high]]
            lp = [positions[0][low], positions[1][low]]
            trial[hi, hp], trial[lo, lp] = trial[lo, lp].copy(), trial[hi, hp].copy()
    loads[[first, second]] = costs[trial[[first, second]]].sum(1)


def refine_wide(costs, batches, world_size, seed):
    trial = batches.copy()
    best = batches
    loads = _rank_batch_loads(costs, trial)
    best_quality = _rank_batch_quality(loads, world_size)
    rng = np.random.default_rng(seed)
    pair_count = min(len(trial) // 2, max(1, 65536 // (trial.shape[1] ** 2)))
    for pass_index in range(24):
        order = (np.argsort(loads, kind="stable") if pass_index % 2 == 0
                 else rng.permutation(len(trial)))
        for first, second in zip(order[:pair_count], order[-pair_count:][::-1]):
            _repartition_pair(costs, trial, loads, int(first), int(second), world_size)
        quality = _rank_batch_quality(loads, world_size)
        if quality < best_quality:
            best, best_quality = trial.copy(), quality
    return best


def prototype_best(costs, batch_size, world_size, seed=7):
    """Return experimental batches for divisible, bounded integer-cost inputs.

    Unlike the public sampler, this research prototype supports B <= 64, no tails,
    and integer costs whose epoch sum is below 2**53. It is not a replacement API.
    """
    for name, value in (("batch_size", batch_size), ("world_size", world_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or not len(costs) or len(costs) % (batch_size * world_size):
        raise ValueError("costs must be nonempty, 1D and divisible by batch_size * world_size")
    if batch_size > 64:
        raise ValueError("prototype supports batch_size <= 64")
    if (not np.all(np.isfinite(costs)) or np.any(costs < 0)
            or np.any(costs != np.floor(costs)) or sum(map(int, costs)) >= 2**53):
        raise ValueError("prototype requires nonnegative integer costs with sum < 2**53")
    order = np.random.default_rng(seed).permutation(len(costs))
    order = order[np.argsort(-costs[order], kind="stable")]
    starts = [_partition_layered_batches(costs, order, batch_size),
              _partition_capacity_batches(costs, order, batch_size)]
    candidates = []
    for start in starts:
        refined = _refine_rank_batches(costs, start, world_size, seed)
        candidates.extend((refined, refine_wide(costs, refined, world_size, seed)))
    best = min(candidates, key=lambda b: _rank_batch_quality(_rank_batch_loads(costs, b),
                                                           world_size))
    order = np.argsort(_rank_batch_loads(costs, best), kind="stable")
    return np.ascontiguousarray(best[order])
