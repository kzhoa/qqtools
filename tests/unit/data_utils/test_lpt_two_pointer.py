"""Correctness of the production linear pair search; no timing assertions."""

import numpy as np
import pytest

from qqtools.data.qbalance import (
    _best_rank_batch_swaps as _best_pair_swaps,
    _partition_layered_batches,
    _plan_rank_batches,
    _rank_batch_quality,
    _swap_layered_rank_batches,
)


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


def test_finds_exchange_missed_by_four_positions():
    high = np.array([[36712, 113, 76, 55, 42, 32, 25, 20, 16, 12, 10, 7, 5, 4, 2, 0.]])
    low = np.array([[235, 175, 104, 71, 55, 39, 32, 25, 19, 15, 12, 9, 7, 5, 4, 1.]])
    hpos, lpos = _best_pair_swaps(high, low, high.sum(1) - low.sum(1))
    assert (hpos[0], lpos[0]) == (1, 15)
    assert high.sum() - high[0, hpos[0]] + low[0, lpos[0]] == 37019


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_integrated_fast_preserves_occurrences_and_improves_layered(seed, batch_size):
    costs = np.random.default_rng(seed).lognormal(3, 1.5, batch_size * 12)
    selected = np.random.default_rng(seed).permutation(len(costs))
    order = selected[np.argsort(-costs[selected], kind="stable")]
    baseline = _partition_layered_batches(costs, order, batch_size)
    scores = []
    for strategy in ("lpt_fast", "lpt", "lpt_best"):
        plan = _plan_rank_batches(costs, batch_size=batch_size, world_size=4,
                                  seed=seed, strategy=strategy, should_shuffle=False)
        np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(len(costs)))
        scores.append(_rank_batch_quality(costs[plan].sum(2).ravel()))
    assert scores[2][:2] <= scores[1][:2] <= scores[0][:2]
    assert scores[0] <= _rank_batch_quality(costs[baseline].sum(1))


def test_fast_planner_actually_uses_swap():
    costs = np.array([100, 50, 40, 30, 20, 10, 9, 8, 7, 3, 2, 1])
    plan = _plan_rank_batches(costs, batch_size=4, strategy="lpt_fast", should_shuffle=False)
    assert costs[plan].sum(2).max() == 110  # Layered-only peak was 118.


def test_odd_middle_batch_is_not_touched():
    costs = np.array([100, 10, 7, 1, 50, 20, 9, 2, 40, 30, 8, 3.])
    batches = np.arange(12).reshape(3, 4)
    before = batches.copy()
    batches.setflags(write=False)
    actual = _swap_layered_rank_batches(costs, batches, 1)
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
