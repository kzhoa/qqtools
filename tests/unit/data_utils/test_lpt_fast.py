"""Deterministic correctness checks for the production layered planner, not timing tests."""

import numpy as np
import pytest

from qqtools.data.qbalance import _partition_layered_batches, _plan_rank_batches


def _scalar_partition(costs, order, batch_size):
    batch_count = len(order) // batch_size
    batches = [[] for _ in range(batch_count)]
    loads = [0.0] * batch_count
    for layer in range(batch_size):
        targets = sorted(range(batch_count), key=loads.__getitem__)
        items = order[layer * batch_count:(layer + 1) * batch_count]
        for item, target in zip(items, targets):
            batches[target].append(item)
            loads[target] += float(costs[item])
    return np.asarray(batches, dtype=np.int64)


@pytest.mark.parametrize("seed", [0, 7, 29])
@pytest.mark.parametrize("batch_size", [1, 2, 16, 64])
@pytest.mark.parametrize("batch_count", [1, 8])
@pytest.mark.parametrize("distribution", ["zeros", "equal", "ties", "lognormal"])
def test_layered_partition_matches_scalar_reference(seed, batch_size, batch_count, distribution):
    rng = np.random.default_rng(seed)
    total = batch_size * batch_count
    if distribution == "zeros":
        costs = np.zeros(total)
    elif distribution == "equal":
        costs = np.ones(total)
    elif distribution == "ties":
        costs = rng.integers(0, 10, total).astype(np.float64)
    else:
        costs = rng.lognormal(3, 1.5, total)
    selected = rng.permutation(total)
    order = selected[np.argsort(-costs[selected], kind="stable")]
    costs.setflags(write=False)
    order.setflags(write=False)
    batches = _partition_layered_batches(costs, order, batch_size)
    np.testing.assert_array_equal(batches, _scalar_partition(costs, order, batch_size))
    np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(total))
    assert batches.shape == (batch_count, batch_size)
    assert batches.dtype == np.int64
    # Every layer contributes exactly one sample occurrence to each batch.
    for layer in range(batch_size):
        expected = order[layer * batch_count:(layer + 1) * batch_count]
        np.testing.assert_array_equal(np.sort(batches[:, layer]), np.sort(expected))


def test_fast_derived_plan_protects_metrics_for_actual_world_size():
    costs = np.random.default_rng(42).lognormal(size=256)
    kwargs = dict(batch_size=16, seed=7, strategy="lpt_fast", should_shuffle=False)
    single = _plan_rank_batches(costs, world_size=1, **kwargs)
    multi = _plan_rank_batches(costs, world_size=4, **kwargs)
    old = np.sort(costs[single].sum(2).ravel()).reshape(-1, 4)
    new = costs[multi].sum(2)
    assert new.max() <= old.max()
    assert np.percentile(new, 99) <= np.percentile(old, 99)
    assert new.max(1).sum() <= old.max(1).sum()
    np.testing.assert_array_equal(np.sort(multi.ravel()), np.arange(len(costs)))
    np.testing.assert_array_equal(multi, _plan_rank_batches(costs, world_size=4, **kwargs))
    assert multi.shape == (4, 4, 16)
    assert multi.flags.c_contiguous
    loads = costs[multi].sum(axis=2)
    np.testing.assert_allclose(loads, np.sort(loads.ravel()).reshape(-1, 4))


def test_fast_seed_changes_equal_cost_memberships():
    costs = np.ones(64)
    kwargs = dict(batch_size=4, strategy="lpt_fast", should_shuffle=False)
    first = _plan_rank_batches(costs, seed=7, **kwargs)
    second = _plan_rank_batches(costs, seed=29, **kwargs)
    assert {tuple(sorted(b)) for b in first[:, 0]} != {tuple(sorted(b)) for b in second[:, 0]}


def test_fast_pairs_opposite_extremes_for_two_sample_batches():
    costs = np.array([9, 9, 9, 3, 6, 6, 6, 12])
    plan = _plan_rank_batches(
        costs, batch_size=2, world_size=2, strategy="lpt_fast", should_shuffle=False
    )
    np.testing.assert_array_equal(costs[plan].sum(axis=2), np.full((2, 2), 15))
