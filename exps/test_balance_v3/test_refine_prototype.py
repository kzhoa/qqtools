"""Opt-in checks; no dependency from the production unit suite on experiments."""

from itertools import combinations

import numpy as np
import pytest

from exact_rank_batches import integer_quality
from compare_exact import legacy_best
from inspect_neighborhood import inspect_trap
from refine_prototype import _repartition_pair, prototype_best
from qqtools.data.qbalance import _rank_batch_quality, _swap_rank_batch_pair


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 6, 7, 16, 64])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 7])
def test_prototype_preserves_inputs_cardinality_reproducibility_and_incumbent(
    batch_size, world_size, seed,
):
    costs = np.random.default_rng(42).integers(0, 100, batch_size * 8).astype(float)
    original = costs.copy()
    costs.setflags(write=False)
    batches = prototype_best(costs, batch_size, world_size, seed)
    assert batches.shape == (8, batch_size)
    np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(len(costs)))
    np.testing.assert_array_equal(batches, prototype_best(costs, batch_size, world_size, seed))
    np.testing.assert_array_equal(costs, original)
    baseline = legacy_best(costs, batch_size, world_size, seed)
    assert _rank_batch_quality(costs[batches].sum(1), world_size) <= (
        _rank_batch_quality(costs[baseline].sum(1), world_size)
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("seed", range(5))
def test_small_pair_repartition_minimizes_full_objective(batch_size, seed):
    costs = np.random.default_rng(seed).integers(0, 101, 4 * batch_size)
    batches = np.arange(len(costs)).reshape(4, batch_size)
    loads = costs[batches].sum(1).astype(float)
    items = tuple(range(2 * batch_size))
    expected = integer_quality(loads, 2)
    for chosen in combinations(items, batch_size):
        rest = tuple(i for i in items if i not in chosen)
        score = integer_quality([sum(costs[list(chosen)]), sum(costs[list(rest)]), *loads[2:]], 2)
        expected = min(expected, score)
    _repartition_pair(costs, batches, loads, 0, 1, 2)
    assert integer_quality(costs[batches].sum(1), 2) == expected
    np.testing.assert_array_equal(loads, costs[batches].sum(1))
    np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(len(costs)))


@pytest.mark.parametrize("batch_size", [7, 8, 16])
@pytest.mark.parametrize("seed", range(4))
def test_two_sample_exchange_matches_exhaustive_gap_minimization(batch_size, seed):
    costs = np.random.default_rng(seed).integers(0, 101, 2 * batch_size).astype(float)
    initial = np.arange(len(costs)).reshape(2, batch_size)
    batches = initial.copy()
    loads = costs[initial].sum(1)
    _swap_rank_batch_pair(costs, initial, loads, 0, 1)
    swapped_loads = costs[initial].sum(1)
    gap = swapped_loads[0] - swapped_loads[1]
    expected = abs(gap)
    for first in combinations(initial[0], 2):
        for second in combinations(initial[1], 2):
            delta = sum(costs[list(first)]) - sum(costs[list(second)])
            expected = min(expected, abs(gap - 2 * delta))
    _repartition_pair(costs, batches, loads, 0, 1, 1)
    assert abs(loads[0] - loads[1]) == expected
    np.testing.assert_array_equal(loads, costs[batches].sum(1))
    np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(len(costs)))


@pytest.mark.parametrize("costs,batch_size,world_size", [
    ([], 1, 1), ([1, 2, 3], 2, 1), ([1, -1], 1, 1), ([1.5, 2], 1, 1),
    ([np.nan, 1], 1, 1), ([np.inf, 1], 1, 1), ([2**54, 1], 1, 1),
    ([2**53 - 1, 1], 1, 1),
    ([1, 2], True, 1), ([1, 2], 1, 0), (np.ones(65), 65, 1), ([[1, 2]], 1, 1),
])
def test_prototype_rejects_unsupported_experimental_inputs(costs, batch_size, world_size):
    with pytest.raises(ValueError):
        prototype_best(costs, batch_size, world_size)


def test_remaining_trap_requires_more_than_two_batch_repartition():
    assert inspect_trap()[3][1][0] == 126
