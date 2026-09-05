import warnings

import numpy as np
import pytest

from qqtools.data.qbalance import (
    _partition_rank_batches,
    _plan_rank_batches,
    compute_global_even_sort_order,
)
from qqtools.torch.ddp import BalancedBatchSampler, BalancedDistributedSampler


@pytest.mark.parametrize("sampler_type", [BalancedBatchSampler, BalancedDistributedSampler])
@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_legacy_sampler_warns_only_on_construction(sampler_type, strategy):
    # QQTOOLS-COMPAT-0006: delete legacy behavior fixtures with v1.4.0 removal.
    kwargs = {"strategy": strategy}
    with pytest.warns(FutureWarning, match="deprecated.*v1.4.0") as records:
        sampler = sampler_type(
            np.arange(32), batch_size=4, rank=0, world_size=1, **kwargs
        )
    assert len(records) == 1
    assert "lpt-medium" in str(records[0].message)
    assert sampler._plan_cache.strategy == strategy
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        first = list(sampler)
        sampler.set_epoch(1)
        assert len(list(sampler)) == len(first)
    assert not emitted


@pytest.mark.parametrize("sampler_type", [BalancedBatchSampler, BalancedDistributedSampler])
@pytest.mark.parametrize("shuffle", [False, True])
def test_medium_alias_normalizes_cache_and_matches_epochs(sampler_type, shuffle):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        samplers = [sampler_type(
            np.arange(48), batch_size=4, rank=1, world_size=2, shuffle=shuffle,
            seed=7, strategy=strategy,
        ) for strategy in ("lpt", "lpt-medium")]
        samplers.append(sampler_type(
            np.arange(48), batch_size=4, rank=1, world_size=2, shuffle=shuffle, seed=7,
        ))
        for epoch in (0, 1, 9):
            for sampler in samplers:
                sampler.set_epoch(epoch)
                assert sampler._plan_cache.strategy == "lpt-medium"
            assert list(samplers[0]) == list(samplers[1]) == list(samplers[2])
    assert not emitted


@pytest.mark.parametrize("size", [0, 48, 51])
def test_low_level_medium_alias_matches_base_and_derived_plans(size):
    costs = np.arange(size)
    base = [_partition_rank_batches(costs, batch_size=4, strategy=s)
            for s in ("lpt", "lpt-medium")]
    np.testing.assert_array_equal(base[0].full_batches, base[1].full_batches)
    np.testing.assert_array_equal(base[0].remainder, base[1].remainder)
    plans = [_plan_rank_batches(costs, batch_size=4, world_size=2, strategy=s)
             for s in ("lpt", "lpt-medium")]
    np.testing.assert_array_equal(*plans)


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt-medium", "lpt_best"])
def test_lpt_tiers_do_not_warn(strategy):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        sampler = BalancedBatchSampler(
            np.arange(32), batch_size=4, rank=0, world_size=2, strategy=strategy
        )
        assert len(list(sampler)) == 4
    assert not emitted


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_dataset_ordering_is_not_deprecated(strategy):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        order = compute_global_even_sort_order(np.arange(32), strategy=strategy)
    np.testing.assert_array_equal(np.sort(order), np.arange(32))
    assert not emitted
