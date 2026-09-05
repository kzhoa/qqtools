"""Opt-in fast-swap correctness checks; no timing assertions or production dependencies."""

import numpy as np
import pytest

from fast_swap import _swap_once, plan_fast_swap
from qqtools.data.qbalance import (
    _has_worse_raw_step_sum, _rank_batch_quality,
)
from layered_baseline import plan_layered_baseline as _plan_rank_batches


def _scalar_swap(costs, batches, world_size):
    if len(batches) < 2 or batches.shape[1] == 1:
        return batches
    loads = costs[batches].sum(1)
    order = np.argsort(loads, kind="stable")
    trial = batches.copy()
    positions = np.linspace(0, batches.shape[1] - 1, min(4, batches.shape[1]), dtype=int)
    for j in range(len(batches) // 2):
        low, high = order[j], order[-j - 1]
        gap = loads[high] - loads[low]
        best_gap, best_pair = gap, None
        for high_slot in positions:
            for low_slot in positions:
                delta = costs[batches[high, high_slot]] - costs[batches[low, low_slot]]
                residual = abs((gap - delta) - delta) if 0 < delta < gap else gap
                if residual < best_gap:
                    best_gap, best_pair = residual, (high_slot, low_slot)
        if best_pair is not None:
            high_slot, low_slot = best_pair
            trial[high, high_slot], trial[low, low_slot] = (
                trial[low, low_slot], trial[high, high_slot],
            )
    after_loads = costs[trial].sum(1)
    before = _rank_batch_quality(loads, world_size)
    after = _rank_batch_quality(after_loads, world_size)
    if after >= before or (after[:2] == before[:2]
                          and _has_worse_raw_step_sum(loads, after_loads, world_size)):
        return batches
    return trial


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 16, 64])
@pytest.mark.parametrize("batch_count", [1, 3, 8])
@pytest.mark.parametrize("seed", range(4))
def test_vector_pass_matches_scalar_search(batch_size, batch_count, seed):
    costs = np.random.default_rng(seed).uniform(0, 100, batch_size * batch_count)
    batches = np.arange(len(costs)).reshape(batch_count, batch_size)
    costs.setflags(write=False)
    batches.setflags(write=False)
    expected = _scalar_swap(costs, batches, 1)
    actual = _swap_once(costs, batches, 1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.arange(len(costs)))
    assert np.all((actual != batches).sum(1) <= 1)


@pytest.mark.parametrize("total", [0, 1, 12, 63, 128])
@pytest.mark.parametrize("should_drop", [False, True])
@pytest.mark.parametrize("world_size", [1, 4])
def test_plan_preserves_baseline_occurrences_quality_and_determinism(
    total, should_drop, world_size,
):
    costs = np.random.default_rng(42).lognormal(3, 1.5, total)
    costs.setflags(write=False)
    kwargs = dict(batch_size=4, world_size=world_size, seed=7, should_drop_last=should_drop)
    baseline = _plan_rank_batches(costs, strategy="lpt_fast", should_shuffle=False, **kwargs)
    actual = plan_fast_swap(costs, **kwargs)
    assert actual.shape == baseline.shape and actual.flags.c_contiguous
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.sort(baseline.ravel()))
    np.testing.assert_array_equal(actual, plan_fast_swap(costs, **kwargs))
    assert _rank_batch_quality(costs[actual].sum(2).ravel(), world_size) <= (
        _rank_batch_quality(costs[baseline].sum(2).ravel(), world_size)
    )


def test_cross_layer_example_improves_without_claiming_optimality():
    costs = np.array([100, 50, 40, 30, 20, 10, 9, 8, 7, 3, 2, 1])
    baseline = _plan_rank_batches(costs, batch_size=4, strategy="lpt_fast", should_shuffle=False)
    actual = plan_fast_swap(costs, batch_size=4)
    assert costs[baseline].sum(2).max() == 118
    assert costs[actual].sum(2).max() == 110


def test_equal_costs_are_unchanged():
    batches = np.arange(24).reshape(6, 4)
    assert _swap_once(np.ones(24), batches, 2) is batches


def test_large_finite_pair_gap_does_not_overflow_in_trial_arithmetic():
    costs = np.array([1e308, 1, 4e307, 1, 1, 1, 1, 1])
    batches = np.arange(8).reshape(2, 4)
    with np.errstate(over="raise", invalid="raise"):
        actual = _swap_once(costs, batches, 1)
    assert np.isfinite(costs[actual].sum(1)).all()
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.arange(8))


@pytest.mark.parametrize("costs,batch_size,world_size", [
    ([1, -1], 1, 1), ([np.nan], 1, 1), ([np.inf], 1, 1),
    ([1, 2], True, 1), ([1, 2], 1, 0), ([1e308, 1e308], 2, 1),
])
def test_invalid_inputs_use_production_validation(costs, batch_size, world_size):
    with pytest.raises((ValueError, TypeError)):
        plan_fast_swap(costs, batch_size=batch_size, world_size=world_size)
