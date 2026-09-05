import numpy as np
import pytest

import qqtools.torch.ddp.qbalancedsampler as qbs


@pytest.mark.parametrize("sampler_type", [qbs.BalancedBatchSampler, qbs.BalancedDistributedSampler])
@pytest.mark.parametrize("name", ["shuffle", "drop_last"])
@pytest.mark.parametrize("value", ["False", "True", 0, 1, None, [], np.array([True])])
def test_boolean_settings_reject_implicit_truthiness(sampler_type, name, value, monkeypatch):
    def unexpected_plan(*args, **kwargs):
        raise AssertionError("Invalid settings must fail before planning")

    monkeypatch.setattr(qbs, "_plan_rank_batches", unexpected_plan)
    with pytest.raises(TypeError, match=f"{name} must be a boolean"):
        sampler_type([1, 2, 3], batch_size=2, **{name: value})


@pytest.mark.parametrize("sampler_type", [qbs.BalancedBatchSampler, qbs.BalancedDistributedSampler])
@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.9, np.float64(1), "1", None])
def test_seed_rejects_noninteger_values(sampler_type, seed):
    with pytest.raises(TypeError, match="seed must be a non-negative integer"):
        sampler_type([1, 2], batch_size=1, seed=seed)


@pytest.mark.parametrize("sampler_type", [qbs.BalancedBatchSampler, qbs.BalancedDistributedSampler])
@pytest.mark.parametrize("seed", [-1, np.int64(-2)])
def test_seed_rejects_negative_values(sampler_type, seed):
    with pytest.raises(ValueError, match="seed must be non-negative"):
        sampler_type([1, 2], batch_size=1, seed=seed)


@pytest.mark.parametrize("sampler_type", [qbs.BalancedBatchSampler, qbs.BalancedDistributedSampler])
@pytest.mark.parametrize("should_shuffle", [False, True])
@pytest.mark.parametrize("should_drop", [False, True])
def test_numpy_scalar_settings_match_python_settings(sampler_type, should_shuffle, should_drop):
    numpy_sampler = sampler_type(
        [9, 1, 8, 2], batch_size=np.int64(2), rank=np.int64(0), world_size=np.int64(1),
        shuffle=np.bool_(should_shuffle), drop_last=np.bool_(should_drop), seed=np.uint64(7),
    )
    python_sampler = sampler_type(
        [9, 1, 8, 2], batch_size=2, rank=0, world_size=1,
        shuffle=should_shuffle, drop_last=should_drop, seed=7,
    )
    for epoch in (0, 1, 3):
        numpy_sampler.set_epoch(epoch)
        python_sampler.set_epoch(epoch)
        assert list(numpy_sampler) == list(python_sampler)
