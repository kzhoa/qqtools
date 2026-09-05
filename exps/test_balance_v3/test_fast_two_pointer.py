"""Opt-in correctness checks for the experimental linear pair search."""

import numpy as np
import pytest

from fast_two_pointer import _best_pair_swaps, _swap_once, plan_fast_two_pointer
from qqtools.data.qbalance import _rank_batch_quality
from layered_baseline import plan_layered_baseline as _plan_rank_batches


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 6, 8, 16, 64])
@pytest.mark.parametrize("distribution", ["integer", "float", "ties"])
@pytest.mark.parametrize("seed", range(5))
def test_two_pointer_matches_exhaustive_pair_gap(batch_size, distribution, seed):
    rng = np.random.default_rng(seed)
    values = (rng.uniform(0, 100, (14, batch_size)) if distribution == "float"
              else rng.integers(0, 5 if distribution == "ties" else 100, (14, batch_size)))
    values = np.sort(values.astype(float), axis=1)[:, ::-1]
    loads = values.sum(1)
    order = np.argsort(loads, kind="stable")
    high, low = values[order[7:][::-1]], values[order[:7]]
    gaps = loads[order[7:][::-1]] - loads[order[:7]]
    high.setflags(write=False)
    low.setflags(write=False)
    hpos, lpos = _best_pair_swaps(high, low, gaps)
    for pair, gap in enumerate(gaps):
        delta = high[pair, :, None] - low[pair, None, :]
        delta = np.clip(delta, 0, gap)
        expected = np.abs((gap - delta) - delta).min()
        if hpos[pair] < 0:
            assert expected == gap
        else:
            change = high[pair, hpos[pair]] - low[pair, lpos[pair]]
            assert 0 < change < gap
            assert abs((gap - change) - change) == expected


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 16, 64])
@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("total", [0, 1, 12, 127])
@pytest.mark.parametrize("should_drop", [False, True])
def test_plan_preserves_tail_cardinality_quality_and_determinism(
    batch_size, world_size, total, should_drop,
):
    costs = np.random.default_rng(42).lognormal(3, 1.5, total)
    costs.setflags(write=False)
    kwargs = dict(batch_size=batch_size, world_size=world_size, seed=7,
                  should_drop_last=should_drop)
    baseline = _plan_rank_batches(costs, strategy="lpt_fast", should_shuffle=False, **kwargs)
    assert np.all(np.diff(costs[baseline], axis=2) <= 0)
    actual = plan_fast_two_pointer(costs, **kwargs)
    assert actual.shape == baseline.shape and actual.flags.c_contiguous
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.sort(baseline.ravel()))
    np.testing.assert_array_equal(actual, plan_fast_two_pointer(costs, **kwargs))
    assert _rank_batch_quality(costs[actual].sum(2).ravel(), world_size) <= (
        _rank_batch_quality(costs[baseline].sum(2).ravel(), world_size)
    )


def test_finds_exchange_missed_by_four_positions():
    high = np.array([[36712, 113, 76, 55, 42, 32, 25, 20, 16, 12, 10, 7, 5, 4, 2, 0.]])
    low = np.array([[235, 175, 104, 71, 55, 39, 32, 25, 19, 15, 12, 9, 7, 5, 4, 1.]])
    hpos, lpos = _best_pair_swaps(high, low, high.sum(1) - low.sum(1))
    assert (hpos[0], lpos[0]) == (1, 15)
    assert high.sum() - high[0, hpos[0]] + low[0, lpos[0]] == 37019


def test_odd_middle_batch_is_not_touched():
    costs = np.array([100, 10, 7, 1, 50, 20, 9, 2, 40, 30, 8, 3.])
    batches = np.arange(12).reshape(3, 4)
    before = batches.copy()
    batches.setflags(write=False)
    actual = _swap_once(costs, batches, 1)
    order = np.argsort(costs[batches].sum(1), kind="stable")
    np.testing.assert_array_equal(actual[order[1]], before[order[1]])
    assert np.all((actual != before).sum(1) <= 1)
    np.testing.assert_array_equal(batches, before)


def test_extreme_and_subnormal_pair_arithmetic():
    high = np.array([[1e308, 4e307, 1, 0], [1e-323, 5e-324, 0, 0]])
    low = np.array([[1e307, 2, 1, 0], [5e-324, 0, 0, 0]])
    gap = high.sum(1) - low.sum(1)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        hpos, lpos = _best_pair_swaps(high, low, gap)
    for pair in range(2):
        delta = np.clip(high[pair, :, None] - low[pair, None, :], 0, gap[pair])
        expected = np.abs((gap[pair] - delta) - delta).min()
        chosen = high[pair, hpos[pair]] - low[pair, lpos[pair]] if hpos[pair] >= 0 else 0
        assert abs((gap[pair] - chosen) - chosen) == expected


@pytest.mark.parametrize("costs,batch_size,world_size", [
    ([1, -1], 1, 1), ([np.nan], 1, 1), ([np.inf], 1, 1),
    ([1, 2], True, 1), ([1, 2], 1, 0), ([1e308, 1e308], 2, 1),
])
def test_invalid_inputs_reuse_production_validation(costs, batch_size, world_size):
    with pytest.raises((ValueError, TypeError)):
        plan_fast_two_pointer(costs, batch_size=batch_size, world_size=world_size)
