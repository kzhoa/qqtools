"""Frozen step-pass reference, compared against production without applying it twice."""

from unittest.mock import patch

import numpy as np

import qqtools.data.qbalance as balance


def refine_fast_steps(costs, batches, world_size):
    if world_size == 1 or len(batches) < 2 or batches.shape[1] == 1:
        return batches
    loads = balance._rank_batch_loads(costs, batches)
    if np.all(loads == loads[0]):
        return batches
    positions = np.argsort(-costs[batches], axis=1, kind="stable")
    sorted_batches = np.take_along_axis(batches, positions, axis=1)
    candidate = balance._swap_layered_rank_batches(costs, sorted_batches, world_size)
    candidate_loads = balance._rank_batch_loads(costs, candidate)
    before = balance._rank_batch_quality(loads, world_size)
    after = balance._rank_batch_quality(candidate_loads, world_size)
    if (after < before and after[0] <= before[0] and after[1] <= before[1]
            and not balance._has_worse_raw_step_sum(loads, candidate_loads, world_size)):
        return candidate
    return batches


def plan_fast(costs, batch_size, world_size, seed=7, should_optimize=True):
    replacement = refine_fast_steps if should_optimize else lambda costs, batches, ranks: batches
    with patch.object(balance, "_optimize_fast_step_batches", replacement):
        return balance._plan_rank_batches(
            costs, batch_size=batch_size, world_size=world_size,
            seed=seed, strategy="lpt_fast", should_shuffle=False,
        )
