import numpy as np
import pytest

from qqtools.data import compute_global_even_sort_order


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_compute_global_even_sort_order_empty_input(strategy):
    order = compute_global_even_sort_order([], strategy=strategy)
    assert order.dtype == np.int64
    assert order.shape == (0,)


@pytest.mark.parametrize(
    "sample_costs",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -1.0]),
    ],
)
def test_compute_global_even_sort_order_rejects_invalid_costs(sample_costs):
    with pytest.raises(ValueError):
        compute_global_even_sort_order(sample_costs)


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_compute_global_even_sort_order_returns_full_permutation(strategy):
    sample_costs = np.array([3.5, 1.0, 4.0, 2.0, 0.5, 7.0], dtype=np.float64)
    order = compute_global_even_sort_order(sample_costs, seed=11, strategy=strategy)

    assert order.dtype == np.int64
    assert sorted(order.tolist()) == list(range(sample_costs.shape[0]))


def test_compute_global_even_sort_order_is_deterministic_for_same_seed():
    sample_costs = np.array([0.5, 3.0, 1.5, 4.5, 2.0, 6.0, 1.0, 5.5], dtype=np.float64)

    first = compute_global_even_sort_order(sample_costs, seed=7, strategy="v3")
    second = compute_global_even_sort_order(sample_costs, seed=7, strategy="v3")

    assert np.array_equal(first, second)


def test_compute_global_even_sort_order_selects_distinct_strategies():
    sample_costs = np.array([0.5, 3.0, 1.5, 4.5, 2.0, 6.0, 1.0, 5.5], dtype=np.float64)

    v1 = compute_global_even_sort_order(sample_costs, seed=7, strategy="v1").tolist()
    v2 = compute_global_even_sort_order(sample_costs, seed=7, strategy="v2").tolist()
    v3 = compute_global_even_sort_order(sample_costs, seed=7, strategy="v3").tolist()

    assert len({tuple(v1), tuple(v2), tuple(v3)}) == 3


@pytest.mark.parametrize(
    ("sample_costs", "seed", "expected_orders"),
    [
        (
            np.array([0.5, 3.0, 1.5, 4.5, 2.0, 6.0, 1.0, 5.5], dtype=np.float64),
            7,
            {
                "v1": [0, 6, 7, 2, 4, 5, 1, 3],
                "v2": [0, 6, 2, 4, 1, 3, 7, 5],
                "v3": [1, 4, 3, 2, 6, 0, 7, 5],
            },
        ),
        (
            np.array([3.0, 3.0, 0.0, 7.0, 1.0, 2.0, 9.0, 4.0, 8.0], dtype=np.float64),
            0,
            {
                "v1": [4, 5, 2, 6, 3, 8, 7, 0, 1],
                "v2": [2, 4, 5, 0, 1, 7, 3, 8, 6],
                "v3": [7, 0, 1, 5, 3, 4, 8, 2, 6],
            },
        ),
    ],
)
def test_compute_global_even_sort_order_matches_characterization(
    sample_costs,
    seed,
    expected_orders,
):
    for strategy, expected in expected_orders.items():
        actual = compute_global_even_sort_order(
            sample_costs,
            seed=seed,
            strategy=strategy,
        ).tolist()
        assert actual == expected
