import numpy as np
import pytest

import qqtools.torch.ddp.qbalancedsampler as qbs
from qqtools.torch.ddp import (
    BalancedBatchSampler,
    BalancedDistributedSampler,
    assign_chunk_lpt,
)


def _mock_dist_state(
    monkeypatch,
    *,
    is_available: bool,
    is_initialized: bool,
    rank: int = 0,
    world_size: int = 1,
):
    monkeypatch.setattr(qbs.dist, "is_available", lambda: is_available)
    monkeypatch.setattr(qbs.dist, "is_initialized", lambda: is_initialized)
    monkeypatch.setattr(qbs.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(qbs.dist, "get_world_size", lambda: world_size)


def test_assign_chunk_lpt_empty_chunk():
    assignment = assign_chunk_lpt([], [1.0, 2.0], world_size=2, batch_size=2)
    assert assignment == [[], []]


def test_assign_chunk_lpt_full_chunk_balances_normal_case():
    assignment = assign_chunk_lpt([0, 1, 2, 3], [9.0, 8.0, 7.0, 6.0], world_size=2, batch_size=2)
    assert assignment == [[0, 3], [1, 2]]


def test_assign_chunk_lpt_keeps_equal_cost_processing_stable():
    assignment = assign_chunk_lpt([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0], world_size=2, batch_size=2)
    assert assignment == [[0, 3], [1, 2]]


def test_assign_chunk_lpt_uses_locality_tie_break():
    sample_costs = np.zeros(11, dtype=np.float64)
    sample_costs[0] = 5.0
    sample_costs[10] = 5.0
    sample_costs[1] = 1.0

    assignment = assign_chunk_lpt([0, 10, 1], sample_costs, world_size=2, batch_size=2)

    assert assignment == [[0, 1], [10]]


def test_assign_chunk_lpt_falls_back_when_preferred_rank_is_full():
    sample_costs = np.zeros(12, dtype=np.float64)
    sample_costs[10] = 10.0
    sample_costs[0] = 2.0
    sample_costs[1] = 1.0
    sample_costs[11] = 1.0

    assignment = assign_chunk_lpt([0, 1, 10, 11], sample_costs, world_size=2, batch_size=2)

    assert assignment == [[10, 11], [0, 1]]


def test_assign_chunk_lpt_rejects_invalid_numeric_input():
    with pytest.raises(ValueError):
        assign_chunk_lpt([0, 1], [1.0, np.nan], world_size=2, batch_size=2)


def test_balanced_distributed_sampler_eager_validation_failures(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=True, is_initialized=True, rank=1, world_size=2)

    with pytest.raises(ValueError):
        BalancedDistributedSampler([1.0, np.nan], batch_size=2)

    with pytest.raises(ValueError):
        BalancedDistributedSampler([1.0, 2.0], batch_size=2, sample_order=[0, 0])

    with pytest.raises(ValueError):
        BalancedDistributedSampler([1.0, 2.0], batch_size=2, rank=0, world_size=2)


def test_balanced_distributed_sampler_shuffle_epoch_changes_and_preserves_equal_lengths(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)
    sample_costs = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0]

    rank0 = BalancedDistributedSampler(
        sample_costs,
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=True,
        seed=5,
        strategy="v1",
    )
    rank1 = BalancedDistributedSampler(
        sample_costs,
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=True,
        seed=5,
        strategy="v1",
    )

    epoch0_rank0 = list(rank0)
    epoch0_rank1 = list(rank1)

    rank0.set_epoch(1)
    rank1.set_epoch(1)

    epoch1_rank0 = list(rank0)
    epoch1_rank1 = list(rank1)

    assert len(epoch0_rank0) == len(epoch0_rank1)
    assert len(epoch1_rank0) == len(epoch1_rank1)
    assert epoch0_rank0 != epoch1_rank0
    assert epoch0_rank1 != epoch1_rank1


def test_balanced_distributed_sampler_prefix_padding_handles_tiny_dataset(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    rank0 = BalancedDistributedSampler(
        [4.0],
        batch_size=3,
        rank=0,
        world_size=2,
        shuffle=True,
        seed=0,
        drop_last=False,
    )
    rank1 = BalancedDistributedSampler(
        [4.0],
        batch_size=3,
        rank=1,
        world_size=2,
        shuffle=True,
        seed=0,
        drop_last=False,
    )

    assert list(rank0) == [0, 0, 0]
    assert list(rank1) == [0, 0, 0]
    assert len(rank0) == len(rank1) == 3


def test_balanced_distributed_sampler_without_shuffle_reuses_plan_and_ignores_epoch(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    sampler = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=False,
        sample_order=[4, 3, 2, 1, 0],
    )

    before = list(sampler)
    sampler.set_epoch(99)
    after = list(sampler)

    assert before == after


def test_balanced_distributed_sampler_drop_last_without_shuffle_drops_incomplete_global_chunk(
    monkeypatch,
):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    rank0 = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=True,
    )
    rank1 = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0],
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=False,
        drop_last=True,
    )

    assert list(rank0) == [0, 3]
    assert list(rank1) == [1, 2]
    assert len(rank0) == len(rank1) == 2


def test_balanced_distributed_sampler_len_matches_emitted_indices(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    rank0 = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=False,
    )
    rank1 = BalancedDistributedSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=False,
        drop_last=False,
    )

    assert list(rank0) == [0, 3, 0, 4]
    assert list(rank1) == [1, 2, 1, 2]
    assert len(rank0) == len(rank1) == 4


def test_balanced_batch_sampler_yields_list_batches(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    sampler = BalancedBatchSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=False,
    )

    batches = list(sampler)

    assert batches == [[0, 3], [0, 4]]
    assert all(isinstance(batch, list) for batch in batches)


def test_balanced_batch_sampler_is_view_over_distributed_plan(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)
    kwargs = dict(
        sample_costs=[9.0, 1.0, 8.0, 2.0, 7.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=True,
        seed=3,
        strategy="v1",
        drop_last=False,
    )

    distributed = BalancedDistributedSampler(**kwargs)
    batch_sampler = BalancedBatchSampler(**kwargs)

    flattened = [index for batch in batch_sampler for index in batch]

    assert flattened == list(distributed)


def test_balanced_batch_sampler_without_shuffle_drop_last_false_uses_prefix_padding(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    sampler = BalancedBatchSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=False,
    )

    assert list(sampler) == [[0, 3], [0, 4]]


def test_balanced_batch_sampler_drop_last_preserves_equal_batch_counts(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    rank0 = BalancedBatchSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=True,
    )
    rank1 = BalancedBatchSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0],
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=False,
        drop_last=True,
    )

    assert list(rank0) == [[0, 3]]
    assert list(rank1) == [[1, 2]]
    assert len(rank0) == len(rank1) == 1


def test_balanced_batch_sampler_len_reports_batch_count(monkeypatch):
    _mock_dist_state(monkeypatch, is_available=False, is_initialized=False)

    sampler = BalancedBatchSampler(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        drop_last=False,
    )

    assert len(sampler) == 2
