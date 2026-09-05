import numpy as np
import pytest

import qqtools.torch.ddp.qbalancedsampler as qbs
from qqtools.torch.ddp import BalancedBatchSampler, BalancedDistributedSampler


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_lpt_builds_once_and_epoch_changes_only_step_and_rank_traversal(monkeypatch, strategy):
    build = qbs._plan_rank_batches
    calls = []

    def counted_build(*args, **kwargs):
        calls.append(kwargs)
        return build(*args, **kwargs)

    def unexpected_assignment(*args, **kwargs):
        raise AssertionError("LPT batches must not be repartitioned or globally reordered")

    monkeypatch.setattr(qbs, "_plan_rank_batches", counted_build)
    monkeypatch.setattr(qbs, "assign_window_to_ranks", unexpected_assignment)
    monkeypatch.setattr(qbs, "compute_global_even_sort_order", unexpected_assignment)
    costs = np.arange(128, dtype=float)
    samplers = [
        BalancedBatchSampler(
            costs, batch_size=4, world_size=4, rank=rank, strategy=strategy, seed=7
        )
        for rank in range(4)
    ]
    static = samplers[0]._plan_cache._lpt_plan.copy()
    before = [list(sampler) for sampler in samplers]
    expected_steps = sorted(tuple(sorted(tuple(batch) for batch in step)) for step in static)
    for epoch in (1, 2, 1):
        for sampler in samplers:
            sampler.set_epoch(epoch)
            assert sampler._plan_cache is sampler.sampler._plan_cache
            np.testing.assert_array_equal(sampler._plan_cache._lpt_plan, static)
        actual_steps = zip(*(list(sampler) for sampler in samplers))
        actual = sorted(tuple(sorted(tuple(batch) for batch in step)) for step in actual_steps)
        assert actual == expected_steps  # Also preserves sample order INSIDE each batch.
    assert [list(sampler) for sampler in samplers] != before
    assert len(calls) == 4
    assert all(call["should_shuffle"] is False for call in calls)
    expected_strategy = "lpt-medium" if strategy == "lpt" else strategy
    assert all(call["strategy"] == expected_strategy for call in calls)


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_non_shuffled_lpt_still_balances_and_matches_training_static_groups(strategy):
    costs = np.asarray([9.0, 6.0, 3.0, 2.0, 8.0, 7.0, 4.0, 1.0])
    plans = []
    for rank in range(2):
        sampler = BalancedBatchSampler(
            costs, batch_size=2, world_size=2, rank=rank, strategy=strategy, shuffle=False
        )
        training = BalancedBatchSampler(
            costs, batch_size=2, world_size=2, rank=rank, strategy=strategy, shuffle=True
        )
        before = list(sampler)
        sampler.set_epoch(123)
        assert list(sampler) == before
        assert sampler._plan_cache.epoch == 0
        np.testing.assert_array_equal(sampler._plan_cache._lpt_plan, training._plan_cache._lpt_plan)
        assert all(costs[batch].sum() == 10 for batch in sampler)
        plans.extend(before)
    assert sorted(index for batch in plans for index in batch) == list(range(len(costs)))


@pytest.mark.parametrize("total", [0, 1, 5, 8])
@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("should_shuffle", [True, False])
@pytest.mark.parametrize("should_drop", [True, False])
def test_lpt_tail_contract(total, world_size, should_shuffle, should_drop, strategy):
    kwargs = dict(
        sample_costs=np.arange(total), batch_size=2, world_size=world_size,
        shuffle=should_shuffle, drop_last=should_drop, strategy=strategy, seed=7,
    )
    global_size = 2 * world_size
    if total % global_size and not should_shuffle and not should_drop:
        with pytest.raises(ValueError, match="validation padding"):
            BalancedBatchSampler(rank=0, **kwargs)
        return
    samplers = [BalancedBatchSampler(rank=rank, **kwargs) for rank in range(world_size)]
    steps = total // global_size if should_drop else (total + global_size - 1) // global_size
    assert [len(sampler) for sampler in samplers] == [steps] * world_size
    target = steps * global_size
    actual = [i for sampler in samplers for batch in sampler for i in batch]
    assert len(actual) == target
    assert all(0 <= i < total for i in actual)
    if should_drop:
        assert len(set(actual)) == target
    else:
        assert sorted(set(actual)) == list(range(total))
    for sampler in samplers:
        sampler.set_epoch(2)
        assert all(len(batch) == 2 for batch in sampler)
    reseeded = [i for sampler in samplers for batch in sampler for i in batch]
    assert sorted(reseeded) == sorted(actual)


@pytest.mark.parametrize("should_shuffle", [True, False])
@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_lpt_rejects_sample_order_instead_of_ignoring_it(should_shuffle, strategy):
    with pytest.raises(ValueError, match="sample_order"):
        BalancedDistributedSampler(
            [1, 2], batch_size=1, rank=0, world_size=1, strategy=strategy,
            shuffle=should_shuffle, sample_order=[1, 0],
        )


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_lpt_snapshots_input_costs(strategy):
    costs = np.arange(64, dtype=float)
    sampler = BalancedBatchSampler(costs, batch_size=4, rank=0, world_size=1, strategy=strategy)
    costs[:] = np.nan
    sampler.set_epoch(2)
    assert np.all(np.isfinite(sampler._plan_cache.sample_costs))
    assert sorted(i for batch in sampler for i in batch) == list(range(64))


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_lpt_failed_epoch_update_does_not_commit_state(strategy):
    sampler = BalancedBatchSampler([1, 2], batch_size=1, rank=0, world_size=1, strategy=strategy)
    before = list(sampler)
    with pytest.raises(ValueError):
        sampler.set_epoch(-1)
    assert sampler._plan_cache.epoch == 0
    assert list(sampler) == before


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_lpt_is_not_a_dataset_order_strategy(strategy):
    with pytest.raises(ValueError, match="Unsupported strategy"):
        qbs.compute_global_even_sort_order([1, 2], strategy=strategy)


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
@pytest.mark.parametrize("name", ["batch_size", "rank", "world_size"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_lpt_rejects_noninteger_public_dimensions(strategy, name, value):
    kwargs = {"batch_size": 1, "rank": 0, "world_size": 1, name: value}
    with pytest.raises(TypeError, match=name):
        BalancedBatchSampler([1, 2], strategy=strategy, **kwargs)
