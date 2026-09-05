"""Production fast step pass preserves artifacts, occurrences and quality."""

import numpy as np
import pytest

import qqtools.data.qbalance as balance


@pytest.mark.parametrize("total", [0, 1, 48, 53, 256])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("seed", [7, 29])
def test_fast_step_production_contract(total, world_size, seed):
    costs = np.random.default_rng(seed).lognormal(3, 1.5, total)
    base = balance._partition_rank_batches(costs, batch_size=4, seed=seed, strategy="lpt_fast")
    batches = balance._complete_rank_batch_tail(costs, base, world_size, seed, False, "lpt_fast")
    before = batches.copy()
    batches.setflags(write=False)
    actual = balance._optimize_fast_step_batches(costs, batches, world_size)
    old, new = np.sort(costs[before].sum(1)), np.sort(costs[actual].sum(1))
    if len(old):
        assert new[-1] <= old[-1]
        assert np.percentile(new, 99) <= np.percentile(old, 99)
        assert new[world_size - 1::world_size].sum() <= old[world_size - 1::world_size].sum()
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.sort(before.ravel()))
    np.testing.assert_array_equal(batches, before)
    plan = balance._plan_rank_batches(
        costs, batch_size=4, world_size=world_size, seed=seed,
        strategy="lpt_fast", should_shuffle=False,
    )
    expected = actual[np.argsort(costs[actual].sum(1), kind="stable")]
    np.testing.assert_array_equal(plan.reshape(-1, 4), expected)


def test_fast_step_has_real_peak_improvement():
    costs = np.rint(np.random.default_rng(42).lognormal(3, 1.5, 16384))
    base = balance._partition_rank_batches(costs, batch_size=16, seed=7, strategy="lpt_fast")
    actual = balance._optimize_fast_step_batches(costs, base.full_batches, 4)
    assert costs[base.full_batches].sum(1).max() == 10471
    assert costs[actual].sum(1).max() == 10397
