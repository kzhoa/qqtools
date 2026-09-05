"""One vectorized swap pass after layered LPT; experimental, not a public strategy."""

import numpy as np

from qqtools.data.qbalance import (
    _has_worse_raw_step_sum, _rank_batch_loads, _rank_batch_quality,
)
from layered_baseline import plan_layered_baseline as _plan_rank_batches


_SWAP_SLOTS = 4


def _swap_once(costs: np.ndarray, batches: np.ndarray, world_size: int) -> np.ndarray:
    """Pair opposite loads and search at most 16 exchanges per disjoint pair."""
    batch_count, batch_size = batches.shape
    if batch_count < 2 or batch_size == 1:
        return batches
    loads = _rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    order = np.argsort(loads, kind="stable")
    pair_count = batch_count // 2
    low, high = order[:pair_count], order[-pair_count:][::-1]
    positions = np.linspace(0, batch_size - 1, min(_SWAP_SLOTS, batch_size), dtype=int)
    high_costs = costs[batches[high[:, None], positions]]
    low_costs = costs[batches[low[:, None], positions]]
    gap = loads[high] - loads[low]
    delta = high_costs[:, :, None] - low_costs[:, None, :]
    delta = np.where((delta > 0) & (delta < gap[:, None, None]), delta, 0.0)
    residual = np.abs((gap[:, None, None] - delta) - delta).reshape(pair_count, -1)
    chosen = residual.argmin(axis=1)
    should_swap = residual[np.arange(pair_count), chosen] < gap
    if not np.any(should_swap):
        return batches
    high, low, chosen = high[should_swap], low[should_swap], chosen[should_swap]
    high_slot = positions[chosen // len(positions)]
    low_slot = positions[chosen % len(positions)]
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


def plan_fast_swap(
    sample_costs: np.ndarray | list[float], *, batch_size: int, world_size: int = 1,
    seed: int = 0, should_drop_last: bool = False,
) -> np.ndarray:
    """Build a static fast plan and try one vectorized swap pass.

    Args:
        sample_costs: Nonnegative finite sample costs, as in the production planner.
        batch_size: Samples per rank-batch.
        world_size: Ranks per step.
        seed: Tail selection and equal-cost tie seed.
        should_drop_last: Drop rather than pad the incomplete global tail.

    Returns:
        Contiguous [step, rank, sample] indices without traversal shuffling.

    Raises:
        ValueError: For invalid costs/dimensions or overflowing baseline loads.
        TypeError: For noninteger dimensions, as validated by the production planner.
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
