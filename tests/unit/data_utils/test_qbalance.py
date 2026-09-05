import numpy as np
import pytest

from qqtools.data import (
    assign_window_to_ranks,
    compute_global_even_sort_order,
    validate_balance_strategy,
)


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_global_even_sort_returns_deterministic_full_permutation(strategy):
    costs = np.asarray([0.5, 3.0, 1.5, 4.5, 2.0, 6.0, 1.0, 5.5])

    first = compute_global_even_sort_order(costs, seed=7, strategy=strategy)
    second = compute_global_even_sort_order(costs, seed=7, strategy=strategy)

    assert first.dtype == np.int64
    assert sorted(first.tolist()) == list(range(costs.shape[0]))
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("v1", [0, 6, 7, 2, 4, 5, 1, 3]),
        ("v2", [0, 6, 2, 4, 1, 3, 7, 5]),
        ("v3", [0, 5, 6, 7, 2, 3, 4, 1]),
    ],
)
def test_global_even_sort_strategy_characterization(strategy, expected):
    costs = np.asarray([0.5, 3.0, 1.5, 4.5, 2.0, 6.0, 1.0, 5.5])

    order = compute_global_even_sort_order(costs, seed=7, strategy=strategy)

    assert order.tolist() == expected


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_global_even_sort_supports_empty_input(strategy):
    order = compute_global_even_sort_order([], strategy=strategy)

    assert order.dtype == np.int64
    assert order.shape == (0,)


@pytest.mark.parametrize("total", [1, 2, 31, 32, 33, 1025])
def test_v3_covers_small_and_unequal_bins(total):
    order = compute_global_even_sort_order(np.arange(total), seed=7, strategy="v3")
    assert np.array_equal(np.sort(order), np.arange(total))


def test_v3_pairs_opposite_cost_quantiles():
    costs = np.arange(1024, dtype=np.float64)
    order = compute_global_even_sort_order(costs, seed=7, strategy="v3")
    pairs = costs[order].reshape(-1, 2)
    assert np.all(pairs[:, 0] < 512)
    assert np.all(pairs[:, 1] >= 512)
    assert np.all(np.abs(pairs.sum(axis=1) - 1023) <= 31)


@pytest.mark.parametrize(
    "costs",
    [
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, np.inf]),
        np.asarray([1.0, -1.0]),
    ],
)
def test_global_even_sort_rejects_invalid_costs(costs):
    with pytest.raises(ValueError):
        compute_global_even_sort_order(costs)


def test_validate_balance_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        validate_balance_strategy("unknown")


def test_assign_window_to_ranks_balances_cost_and_preserves_capacity():
    assignment = assign_window_to_ranks(
        [0, 1, 2, 3],
        [9.0, 8.0, 7.0, 6.0],
        world_size=2,
        batch_size=2,
    )

    assert assignment == [[0, 3], [1, 2]]
    assert [sum([9.0, 8.0, 7.0, 6.0][idx] for idx in rank) for rank in assignment] == [15.0, 15.0]


def test_assign_window_to_ranks_uses_locality_for_equal_loads():
    costs = np.zeros(11, dtype=np.float64)
    costs[[0, 10]] = 5.0
    costs[1] = 1.0

    assignment = assign_window_to_ranks([0, 10, 1], costs, world_size=2, batch_size=2)

    assert assignment == [[0, 1], [10]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_indices": [0], "sample_costs": [1.0], "world_size": 0, "batch_size": 1},
        {"window_indices": [0], "sample_costs": [1.0], "world_size": 1, "batch_size": 0},
        {"window_indices": [[0]], "sample_costs": [1.0], "world_size": 1, "batch_size": 1},
        {"window_indices": [1], "sample_costs": [1.0], "world_size": 1, "batch_size": 1},
        {"window_indices": [0], "sample_costs": [np.nan], "world_size": 1, "batch_size": 1},
    ],
)
def test_assign_window_to_ranks_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        assign_window_to_ranks(**kwargs)
