import numpy as np
import pytest
from torch.utils.data import DataLoader

from qqtools.torch.ddp import BalancedBatchSampler, BalancedDistributedSampler


@pytest.mark.integration
@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
@pytest.mark.parametrize("should_shuffle", [True, False])
@pytest.mark.parametrize("should_use_batch_sampler", [True, False])
def test_lpt_dataloaders_preserve_batches_and_epoch_coverage(
    should_shuffle, should_use_batch_sampler, strategy
):
    costs = np.random.default_rng(42).lognormal(size=48)
    dataset = list(range(len(costs)))
    sampler_class = BalancedBatchSampler if should_use_batch_sampler else BalancedDistributedSampler
    samplers = [
        sampler_class(
            costs, batch_size=3, rank=rank, world_size=4,
            strategy=strategy, shuffle=should_shuffle, seed=7,
        )
        for rank in range(4)
    ]
    if should_use_batch_sampler:
        loaders = [DataLoader(dataset, batch_sampler=sampler) for sampler in samplers]
    else:
        loaders = [DataLoader(dataset, sampler=sampler, batch_size=3) for sampler in samplers]
    first = None
    for epoch in (0, 1, 0):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        loaded = [[batch.tolist() for batch in loader] for loader in loaders]
        assert [len(batches) for batches in loaded] == [4] * 4
        assert sorted(i for batches in loaded for batch in batches for i in batch) == dataset
        for sampler, batches in zip(samplers, loaded):
            expected = list(sampler)
            if not should_use_batch_sampler:
                expected = [expected[start:start + 3] for start in range(0, len(expected), 3)]
            assert batches == expected
        if first is None:
            first = loaded
        elif not should_shuffle or epoch == 0:
            assert loaded == first
        else:
            assert loaded != first


@pytest.mark.integration
@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
@pytest.mark.parametrize("should_drop", [False, True])
@pytest.mark.parametrize("should_use_batch_sampler", [False, True])
def test_lpt_tail_repair_reaches_dataloaders_with_equal_steps(
    strategy, should_drop, should_use_batch_sampler,
):
    dataset = list(range(49))
    costs = np.random.default_rng(42).lognormal(size=len(dataset))
    sampler_class = BalancedBatchSampler if should_use_batch_sampler else BalancedDistributedSampler
    samplers = [sampler_class(
        costs, batch_size=3, world_size=4, rank=rank, strategy=strategy,
        shuffle=True, drop_last=should_drop, seed=7,
    ) for rank in range(4)]
    loaders = ([DataLoader(dataset, batch_sampler=sampler) for sampler in samplers]
               if should_use_batch_sampler else
               [DataLoader(dataset, batch_size=3, sampler=sampler) for sampler in samplers])
    first_counts = None
    for epoch in (0, 1, 2):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        loaded = [[batch.tolist() for batch in loader] for loader in loaders]
        assert [len(batches) for batches in loaded] == [4 if should_drop else 5] * 4
        assert all(len(batch) == 3 for batches in loaded for batch in batches)
        indices = [i for batches in loaded for batch in batches for i in batch]
        counts = np.bincount(indices, minlength=len(dataset))
        assert len(counts) == len(dataset)
        if should_drop:
            assert counts.max() == 1 and np.count_nonzero(counts == 0) == 1
        else:
            assert counts.min() >= 1 and (counts - 1).sum() == 11
        if first_counts is None:
            first_counts = counts
        else:
            np.testing.assert_array_equal(counts, first_counts)
