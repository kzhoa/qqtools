"""One disjoint-pair swap pass with linear searches in descending batch costs."""

import numpy as np

from qqtools.data.qbalance import (
    _has_worse_raw_step_sum, _rank_batch_loads, _rank_batch_quality,
)
from layered_baseline import plan_layered_baseline as _plan_rank_batches


def _best_pair_swaps(
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


def _swap_once(costs: np.ndarray, batches: np.ndarray, world_size: int) -> np.ndarray:
    """Input rows come from layered LPT and are already descending by cost."""
    batch_count, batch_size = batches.shape
    if batch_count < 2 or batch_size == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    order = np.argsort(loads, kind="stable")
    pair_count = batch_count // 2
    low, high = order[:pair_count], order[-pair_count:][::-1]
    high_slot, low_slot = _best_pair_swaps(
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
    if after[:2] == before[:2] and _has_worse_raw_step_sum(loads, trial_loads, world_size):
        return batches
    return trial


def plan_fast_two_pointer(
    sample_costs: np.ndarray | list[float], *, batch_size: int, world_size: int = 1,
    seed: int = 0, should_drop_last: bool = False,
) -> np.ndarray:
    """Build a static fast plan and try the best gap-reducing exchange for each pair.

    Args:
        sample_costs: Nonnegative finite costs, validated by the production planner.
        batch_size: Fixed samples per rank-batch.
        world_size: Ranks per step.
        seed: Tail selection and equal-cost tie seed.
        should_drop_last: Drop instead of pad the incomplete global tail.

    Returns:
        Contiguous [step, rank, sample] indices, without traversal shuffling.

    Raises:
        ValueError: For invalid costs/dimensions or overflowing baseline loads.
        TypeError: For noninteger dimensions.
    """
    plan = _plan_rank_batches(
        sample_costs, batch_size=batch_size, world_size=world_size, seed=seed,
        should_drop_last=should_drop_last, should_shuffle=False, strategy="lpt_fast",
    )
    costs = np.asarray(sample_costs, dtype=np.float64)
    batches = plan.reshape(-1, batch_size)
    candidate = _swap_once(costs, batches, world_size)
    if candidate is batches:
        return plan
    order = np.argsort(_rank_batch_loads(costs, candidate), kind="stable")
    return np.ascontiguousarray(candidate[order].reshape(plan.shape))
