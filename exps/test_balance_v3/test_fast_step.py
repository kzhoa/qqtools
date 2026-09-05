import numpy as np
import pytest

from fast_step import plan_fast, refine_fast_steps
from qqtools.data.qbalance import _plan_rank_batches


@pytest.mark.parametrize("total", [1, 48, 53, 256])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_fast_step_keeps_occurrences_and_all_metrics(total, world_size):
    costs = np.random.default_rng(7).integers(0, 100, total).astype(float)
    before = plan_fast(costs, 4, world_size, should_optimize=False)
    after = plan_fast(costs, 4, world_size)
    np.testing.assert_array_equal(np.sort(before.ravel()), np.sort(after.ravel()))
    old, new = costs[before].sum(2), costs[after].sum(2)
    assert new.max() <= old.max()
    assert np.percentile(new, 99) <= np.percentile(old, 99)
    assert new.max(1).sum() <= old.max(1).sum()
    np.testing.assert_array_equal(after, plan_fast(costs, 4, world_size))
    production = _plan_rank_batches(
        costs, batch_size=4, world_size=world_size, seed=7,
        strategy="lpt_fast", should_shuffle=False,
    )
    np.testing.assert_array_equal(after, production)
    batches = before.reshape(-1, 4).copy()
    batches.setflags(write=False)
    snapshot = batches.copy()
    refine_fast_steps(costs, batches, world_size)
    np.testing.assert_array_equal(batches, snapshot)
