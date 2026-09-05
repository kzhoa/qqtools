"""Opt-in checks for the experimental planner, outside the formal test suite.

Run: PYTHONPATH=src python -m pytest exps/test_balance_v3/test_layered_numpy.py -q
The production layered planner is covered independently in tests/unit/.
"""

import numpy as np
import pytest


from layered_numpy import _plan_layered_rank_batches as _plan


def _scalar_plan(costs, batch_size, world_size, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(costs)).tolist()
    order.sort(key=costs.__getitem__, reverse=True)
    batch_count = len(costs) // batch_size
    groups = [[] for _ in range(batch_count)]
    loads = [0.0] * batch_count
    targets = np.arange(batch_count)
    for layer in range(batch_size):
        rng.shuffle(targets)
        targets = np.asarray(sorted(targets, key=loads.__getitem__), dtype=np.int64)
        for item, target in zip(order[layer * batch_count:(layer + 1) * batch_count], targets):
            groups[target].append(item)
            loads[target] += float(costs[item])
    step_ids = sorted(range(batch_count), key=loads.__getitem__)
    return np.asarray(groups, dtype=np.int64)[step_ids].reshape(-1, world_size, batch_size)


@pytest.mark.parametrize("seed", [0, 7, 29])
@pytest.mark.parametrize("batch_size", [1, 2, 16, 64])
@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("distribution", ["zeros", "equal", "ties", "lognormal"])
def test_vectorization_matches_scalar_partition(seed, batch_size, world_size, distribution):
    rng = np.random.default_rng(42)
    total = batch_size * world_size * 5
    if distribution == "zeros":
        costs = np.zeros(total)
    elif distribution == "equal":
        costs = np.ones(total)
    elif distribution == "ties":
        costs = rng.integers(0, 10, total).astype(np.float64)
    else:
        costs = rng.lognormal(3, 1.5, total)
    costs.setflags(write=False)
    plan = _plan(costs, batch_size=batch_size, world_size=world_size, seed=seed)
    np.testing.assert_array_equal(plan, _scalar_plan(costs, batch_size, world_size, seed))
    np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(total))
    np.testing.assert_array_equal(
        plan, _plan(costs, batch_size=batch_size, world_size=world_size, seed=seed)
    )
    assert plan.shape == (5, world_size, batch_size)
    assert plan.dtype == np.int64 and plan.flags.c_contiguous
    loads = costs[plan].sum(axis=2)
    np.testing.assert_allclose(loads, np.sort(loads.ravel()).reshape(-1, world_size))


def test_world_size_only_changes_step_grouping():
    costs = np.random.default_rng(42).lognormal(size=256)
    single = _plan(costs, batch_size=16, seed=7)
    multi = _plan(costs, batch_size=16, world_size=4, seed=7)
    np.testing.assert_array_equal(single.reshape(-1, 16), multi.reshape(-1, 16))


def test_seed_changes_equal_cost_memberships():
    first = _plan(np.ones(64), batch_size=4, seed=7)
    second = _plan(np.ones(64), batch_size=4, seed=29)
    assert {tuple(sorted(b)) for b in first[:, 0]} != {tuple(sorted(b)) for b in second[:, 0]}


def test_empty_plan():
    assert _plan([], batch_size=16, world_size=4).shape == (0, 4, 16)


@pytest.mark.parametrize("costs", [[-1], [np.nan], [np.inf], [[1, 2]]])
def test_rejects_invalid_costs(costs):
    with pytest.raises(ValueError):
        _plan(costs, batch_size=1)


@pytest.mark.parametrize("name", ["batch_size", "world_size"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, 0, -1])
def test_rejects_invalid_dimensions(name, value):
    kwargs = {"batch_size": 1, "world_size": 1, name: value}
    with pytest.raises((TypeError, ValueError)):
        _plan([1], **kwargs)


def test_rejects_tail_and_accumulated_overflow():
    with pytest.raises(ValueError, match="divisible"):
        _plan([1, 2, 3], batch_size=2, world_size=2)
    with pytest.raises(ValueError, match="remain finite"):
        _plan([1e308, 1e308], batch_size=2)


def test_two_item_batches_pair_opposite_extremes():
    plan = _plan([9, 9, 9, 3, 6, 6, 6, 12], batch_size=2, world_size=2)
    np.testing.assert_array_equal(np.array([9, 9, 9, 3, 6, 6, 6, 12])[plan].sum(2), 15)
