"""Opt-in oracle checks: pytest exps/test_balance_v3/test_exact_rank_batches.py -q."""

from itertools import permutations

import pytest

from exact_rank_batches import integer_quality, solve_exact


@pytest.mark.parametrize("costs", [(0, 0, 0, 0, 0, 0), (9, 7, 6, 3, 2, 1)])
@pytest.mark.parametrize("batch_size,world_size", [(1, 2), (2, 3), (3, 2), (6, 1)])
def test_oracle_matches_independent_permutation_enumeration(costs, batch_size, world_size):
    expected = None
    for order in permutations(range(len(costs))):
        loads = [sum(costs[i] for i in order[j:j + batch_size])
                 for j in range(0, len(costs), batch_size)]
        # Enumerated batch order supplies step grouping; do not assume adjacent grouping.
        ordered = sorted(loads)
        lower, remainder = divmod(99 * (len(loads) - 1), 100)
        upper = min(lower + 1, len(loads) - 1)
        p99 = ordered[lower] * (100 - remainder) + ordered[upper] * remainder
        score = (max(loads), p99,
                 sum(max(loads[j:j + world_size]) for j in range(0, len(loads), world_size)))
        expected = score if expected is None else min(expected, score)
    result = solve_exact(costs, batch_size, world_size)
    assert result.quality == expected
    assert sorted(i for batch in result.batches for i in batch) == list(range(len(costs)))


def test_known_partition_and_count():
    result = solve_exact([9, 9, 9, 3, 12, 6, 6, 6], 2, 2)
    assert result.quality == (15, 1500, 30)
    assert result.partitions == 105


def test_integer_quality_does_not_round_large_integers():
    assert integer_quality([10**30, 10**30 + 1], 1) == (
        10**30 + 1, 100 * 10**30 + 99, 2 * 10**30 + 1,
    )


@pytest.mark.parametrize("costs,batch_size,world_size", [
    ([], 1, 1), ([1, 2, 3], 2, 1), ([1, -1], 1, 1), ([1.5, 2], 1, 1),
    ([True, 1], 1, 1), ([1, 2], True, 1), ([1, 2], 1, 0),
])
def test_invalid_oracle_inputs(costs, batch_size, world_size):
    with pytest.raises(ValueError):
        solve_exact(costs, batch_size, world_size)


def test_refuses_infeasible_exhaustion_instead_of_claiming_optimality():
    with pytest.raises(ValueError, match="exceed exhaustive budget"):
        solve_exact(list(range(16)), 4)
    with pytest.raises(ValueError, match="at most 24"):
        solve_exact(list(range(100)), 1)
