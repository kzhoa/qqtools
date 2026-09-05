import numpy as np
import pytest

from qqtools.data.qbalance import _plan_rank_batches


def _batch_memberships(plan):
    return sorted(tuple(sorted(batch)) for batch in plan.reshape(-1, plan.shape[-1]))


@pytest.mark.parametrize("total", [0, 1, 3, 4, 5, 17, 64, 257])
@pytest.mark.parametrize("batch_size", [1, 2, 16])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("should_drop_last", [False, True])
def test_rank_plan_has_full_equal_batches_and_exact_tail_counts(
    total, batch_size, world_size, should_drop_last
):
    costs = np.random.default_rng(42).lognormal(size=total)
    costs.setflags(write=False)
    plan = _plan_rank_batches(
        costs,
        batch_size=batch_size,
        world_size=world_size,
        seed=7,
        should_drop_last=should_drop_last,
    )

    global_size = batch_size * world_size
    steps = total // global_size if should_drop_last else (total + global_size - 1) // global_size
    assert plan.shape == (steps, world_size, batch_size)
    assert plan.dtype == np.int64
    assert plan.flags.c_contiguous
    assert np.all((plan >= 0) & (plan < total))
    if should_drop_last:
        assert len(np.unique(plan)) == plan.size
    else:
        np.testing.assert_array_equal(np.unique(plan), np.arange(total))
    rank_sizes = [plan[:, rank, :].size for rank in range(world_size)]
    assert rank_sizes == [steps * batch_size] * world_size


def test_rank_plan_is_seed_reproducible_and_changes_traversal():
    costs = np.arange(128, dtype=np.float64)
    first = _plan_rank_batches(costs, batch_size=4, world_size=4, seed=7)
    second = _plan_rank_batches(costs, batch_size=4, world_size=4, seed=7)
    reseeded = _plan_rank_batches(costs, batch_size=4, world_size=4, seed=13)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, reseeded)
    # Unique costs leave the greedy partition fixed; shuffling is not uniform repartitioning.
    assert _batch_memberships(first) == _batch_memberships(reseeded)


@pytest.mark.parametrize("cost", [0.0, 1.0])
def test_rank_plan_handles_identical_costs(cost):
    costs = np.full(64, cost)
    plan = _plan_rank_batches(costs, batch_size=4, world_size=4)
    np.testing.assert_array_equal(costs[plan].sum(axis=2), np.full((4, 4), 4 * cost))
    np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(64))


def test_rank_plan_repartitions_when_batch_size_changes():
    costs = np.asarray([8, 7, 6, 5, 4, 3])
    pairs = _plan_rank_batches(costs, batch_size=2)
    triples = _plan_rank_batches(costs, batch_size=3)

    assert sorted(costs[pairs].sum(axis=2).ravel()) == [11, 11, 11]
    assert sorted(costs[triples].sum(axis=2).ravel()) == [16, 17]


def test_rank_plan_can_balance_across_previous_global_window_boundaries():
    costs = np.asarray([9, 6, 3, 2, 8, 7, 4, 1])
    plan = _plan_rank_batches(costs, batch_size=2, world_size=2)

    np.testing.assert_array_equal(costs[plan].sum(axis=2), np.full((2, 2), 10))


def test_rank_plan_isolates_outlier_with_small_samples():
    costs = np.ones(256)
    costs[0] = 1000
    plan = _plan_rank_batches(costs, batch_size=16, world_size=4)

    assert costs[plan].sum(axis=2).max() == 1015


def test_rank_plan_keeps_neighboring_batch_loads_in_same_step():
    costs = np.random.default_rng(42).lognormal(3, 1.5, 1024)
    plan = _plan_rank_batches(costs, batch_size=16, world_size=4, seed=7)
    loads = np.sort(costs[plan].sum(axis=2), axis=1)
    ordered_steps = loads[np.argsort(loads[:, 0], kind="stable")]

    np.testing.assert_allclose(ordered_steps, np.sort(loads.ravel()).reshape(-1, 4))


def test_middle_derived_plan_improves_metrics_for_actual_world_size():
    costs = np.random.default_rng(42).lognormal(size=256)
    single = _plan_rank_batches(costs, batch_size=16, world_size=1, seed=7)
    distributed = _plan_rank_batches(costs, batch_size=16, world_size=4, seed=7)

    old = np.sort(costs[single].sum(2).ravel()).reshape(-1, 4)
    new = costs[distributed].sum(2)
    assert new.max() <= old.max()
    assert np.percentile(new, 99) <= np.percentile(old, 99)
    assert new.max(1).sum() <= old.max(1).sum()
    np.testing.assert_array_equal(np.sort(distributed.ravel()), np.arange(len(costs)))


@pytest.mark.parametrize("costs", [[[1, 2]], [1, np.nan], [1, np.inf], [1, -1]])
def test_rank_plan_rejects_invalid_costs(costs):
    with pytest.raises(ValueError):
        _plan_rank_batches(costs, batch_size=2)


@pytest.mark.parametrize("name", ["batch_size", "world_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_rank_plan_rejects_nonpositive_dimensions_even_with_empty_costs(name, value):
    kwargs = {"batch_size": 2, "world_size": 1, name: value}
    with pytest.raises(ValueError, match=name):
        _plan_rank_batches([], **kwargs)


@pytest.mark.parametrize("name", ["batch_size", "world_size"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, "2"])
def test_rank_plan_rejects_noninteger_dimensions(name, value):
    kwargs = {"batch_size": 2, "world_size": 1, name: value}
    with pytest.raises(TypeError, match=name):
        _plan_rank_batches([1, 2], **kwargs)


def test_rank_plan_accepts_numpy_integer_dimensions():
    plan = _plan_rank_batches([1, 2], batch_size=np.int64(2), world_size=np.int64(1))
    assert plan.shape == (1, 1, 2)


def test_rank_plan_rejects_unrepresentable_batch_loads():
    with pytest.raises(ValueError, match="Accumulated batch costs"):
        _plan_rank_batches([np.finfo(np.float64).max] * 2, batch_size=2)
