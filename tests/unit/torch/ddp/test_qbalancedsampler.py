import numpy as np
import pytest

import qqtools.torch.ddp.qbalancedsampler as qbs
from qqtools.torch.ddp import BalancedBatchSampler, BalancedDistributedSampler


def _mock_dist(
    monkeypatch,
    *,
    is_initialized: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    monkeypatch.setattr(qbs.dist, "is_available", lambda: True)
    monkeypatch.setattr(qbs.dist, "is_initialized", lambda: is_initialized)
    monkeypatch.setattr(qbs.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(qbs.dist, "get_world_size", lambda: world_size)


def _rank_plans(sample_costs, **kwargs):
    world_size = kwargs["world_size"]
    return [
        list(
            BalancedDistributedSampler(
                sample_costs,
                rank=rank,
                **kwargs,
            )
        )
        for rank in range(world_size)
    ]


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3", "lpt"])
def test_sampler_covers_each_sample_once_across_ranks_without_padding(
    monkeypatch,
    strategy,
):
    _mock_dist(monkeypatch)
    costs = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0]

    plans = _rank_plans(
        costs,
        batch_size=2,
        world_size=2,
        shuffle=True,
        seed=7,
        drop_last=False,
        strategy=strategy,
    )

    assert [len(plan) for plan in plans] == [4, 4]
    assert sorted(index for plan in plans for index in plan) == list(range(len(costs)))
    for batch_start in range(0, 4, 2):
        global_window = [
            index
            for plan in plans
            for index in plan[batch_start : batch_start + 2]
        ]
        assert len(global_window) == len(set(global_window)) == 4


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3", "lpt"])
def test_shuffle_epoch_is_deterministic_and_changes_plan(monkeypatch, strategy):
    _mock_dist(monkeypatch)
    costs = np.linspace(1.0, 256.0, num=256)
    kwargs = {
        "batch_size": 4,
        "rank": 0,
        "world_size": 2,
        "shuffle": True,
        "seed": 13,
        "strategy": strategy,
    }
    first = BalancedDistributedSampler(costs, **kwargs)
    second = BalancedDistributedSampler(costs, **kwargs)

    epoch0 = list(first)
    first.set_epoch(1)
    second.set_epoch(1)

    assert list(first) == list(second)
    assert list(first) != epoch0


def test_non_shuffled_sampler_ignores_epoch_and_respects_sample_order(monkeypatch):
    # QQTOOLS-COMPAT-0006: legacy sample_order support until v1.4.0.
    _mock_dist(monkeypatch)
    sampler = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        sample_order=[3, 2, 1, 0],
        strategy="v3",
    )

    before = list(sampler)
    sampler.set_epoch(99)

    assert list(sampler) == before
    assert set(before).issubset({0, 1, 2, 3})


def test_sampler_prefix_padding_equalizes_rank_lengths(monkeypatch):
    # QQTOOLS-COMPAT-0006: legacy validation padding until v1.4.0.
    _mock_dist(monkeypatch)
    costs = [5.0, 4.0, 3.0, 2.0, 1.0]

    plans = _rank_plans(
        costs,
        batch_size=2,
        world_size=2,
        shuffle=False,
        drop_last=False,
        strategy="v3",
    )

    assert [len(plan) for plan in plans] == [4, 4]
    assert set(range(len(costs))).issubset({index for plan in plans for index in plan})
    assert sum(len(plan) for plan in plans) == 8


def test_sampler_prefix_padding_handles_dataset_smaller_than_global_batch(monkeypatch):
    _mock_dist(monkeypatch)

    plans = _rank_plans(
        [4.0],
        batch_size=3,
        world_size=2,
        shuffle=True,
        seed=0,
        drop_last=False,
    )

    assert plans == [[0, 0, 0], [0, 0, 0]]


def test_sampler_drop_last_removes_incomplete_global_window(monkeypatch):
    # QQTOOLS-COMPAT-0006: legacy prefix truncation until v1.4.0.
    _mock_dist(monkeypatch)
    costs = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0]

    plans = _rank_plans(
        costs,
        batch_size=2,
        world_size=2,
        shuffle=False,
        drop_last=True,
        strategy="v3",
    )

    assert plans == [[0, 3], [1, 2]]
    assert [len(plan) for plan in plans] == [2, 2]


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3", "lpt"])
def test_batch_sampler_is_batched_view_of_distributed_sampler(monkeypatch, strategy):
    _mock_dist(monkeypatch)
    kwargs = {
        "sample_costs": [9.0, 1.0, 8.0, 2.0, 7.0],
        "batch_size": 2,
        "rank": 0,
        "world_size": 2,
        "shuffle": True,
        "seed": 3,
        "strategy": strategy,
        "drop_last": False,
    }
    distributed = BalancedDistributedSampler(**kwargs)
    batched = BalancedBatchSampler(**kwargs)

    batches = list(batched)

    assert [index for batch in batches for index in batch] == list(distributed)
    assert all(isinstance(batch, list) and len(batch) == 2 for batch in batches)
    assert len(batched) == len(batches)

    distributed.set_epoch(1)
    batched.set_epoch(1)
    assert [index for batch in batched for index in batch] == list(distributed)


def test_batch_sampler_drop_last_preserves_equal_full_batches_per_rank(monkeypatch):
    # QQTOOLS-COMPAT-0006: legacy prefix truncation until v1.4.0.
    _mock_dist(monkeypatch)
    costs = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0]
    samplers = [
        BalancedBatchSampler(
            costs,
            batch_size=2,
            rank=rank,
            world_size=2,
            shuffle=False,
            drop_last=True,
            strategy="v3",
        )
        for rank in range(2)
    ]

    batches_by_rank = [list(sampler) for sampler in samplers]

    assert batches_by_rank == [[[0, 3]], [[1, 2]]]
    assert [len(sampler) for sampler in samplers] == [1, 1]


def test_sampler_uses_initialized_distributed_rank_and_world_size(monkeypatch):
    _mock_dist(monkeypatch, is_initialized=True, rank=1, world_size=2)
    runtime_sampler = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0],
        batch_size=2,
        shuffle=False,
    )
    explicit_sampler = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0],
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=False,
    )

    assert list(runtime_sampler) == list(explicit_sampler)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_costs": [1.0, np.nan], "batch_size": 1},
        {"sample_costs": [1.0, 2.0], "batch_size": 0},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "world_size": 0},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "rank": 2, "world_size": 2},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "sample_order": [0, 0]},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "sample_order": [0]},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "sample_order": [0, 2]},
        {"sample_costs": [1.0, 2.0], "batch_size": 1, "strategy": "unknown"},
    ],
)
def test_sampler_rejects_invalid_configuration(monkeypatch, kwargs):
    _mock_dist(monkeypatch)

    with pytest.raises(ValueError):
        BalancedDistributedSampler(**kwargs)


def test_sampler_rejects_rank_override_that_conflicts_with_runtime(monkeypatch):
    _mock_dist(monkeypatch, is_initialized=True, rank=1, world_size=2)

    with pytest.raises(ValueError, match="does not match"):
        BalancedDistributedSampler(
            [1.0, 2.0],
            batch_size=1,
            rank=0,
            world_size=2,
        )
