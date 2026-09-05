"""Public NumPy batch partition contract, independent of distributed sampling."""

import numpy as np
import pytest

from qqtools.data import compute_balanced_batch_indices


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt-medium", "lpt_best"])
@pytest.mark.parametrize("total", [0, 1, 3, 4, 5, 32, 35])
def test_public_partition_covers_input_once(strategy, total):
    costs = np.random.default_rng(42).lognormal(size=total)
    original = costs.copy()
    result = compute_balanced_batch_indices(costs, batch_size=4, seed=7, strategy=strategy)
    assert type(result) is tuple
    batches, remainder = result
    assert batches.shape == (total // 4, 4)
    assert remainder.shape == (total % 4,)
    for array in result:
        assert array.dtype == np.int64
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    np.testing.assert_array_equal(np.sort(np.r_[batches.ravel(), remainder]), np.arange(total))
    np.testing.assert_array_equal(costs, original)
    repeated = compute_balanced_batch_indices(
        costs.tolist(), batch_size=4, seed=7, strategy=strategy,
    )
    for actual, expected in zip(result, repeated):
        np.testing.assert_array_equal(actual, expected)


def test_default_and_medium_alias_agree():
    costs = np.arange(35)
    expected = compute_balanced_batch_indices(costs, batch_size=4)
    for strategy in ("lpt", "lpt-medium"):
        actual = compute_balanced_batch_indices(costs, batch_size=4, strategy=strategy)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("strategy", ["lpt_fast", "lpt", "lpt_best"])
def test_opposite_weights_form_equal_batches(strategy):
    costs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    batches, remainder = compute_balanced_batch_indices(costs, batch_size=2, strategy=strategy)
    np.testing.assert_array_equal(costs[batches].sum(axis=1), np.full(4, 9))
    assert remainder.size == 0


@pytest.mark.parametrize("costs", [[-1], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_costs_rejected(costs):
    with pytest.raises(ValueError):
        compute_balanced_batch_indices(costs, batch_size=2)


@pytest.mark.parametrize("batch_size", [True, 1.5, "2"])
def test_noninteger_batch_size_rejected(batch_size):
    with pytest.raises(TypeError):
        compute_balanced_batch_indices([1, 2], batch_size=batch_size)


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"batch_size": -1},
    {"batch_size": 2, "seed": -1},
    {"batch_size": 2, "strategy": "v3"},
    {"batch_size": 2, "strategy": "unknown"},
])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_balanced_batch_indices([1, 2], **kwargs)
