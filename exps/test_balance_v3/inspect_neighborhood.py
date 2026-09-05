"""Certify a remaining local-search trap, not the quality of arbitrary inputs.

PYTHONPATH=src python exps/test_balance_v3/inspect_neighborhood.py
Exhaust the two- and three-batch neighborhoods of a fixed observed incumbent.
"""

from itertools import combinations

from exact_rank_batches import integer_quality, solve_exact


COSTS = (56, 55, 34, 85, 41, 76, 21, 34, 1, 60, 2, 20)
# Observed prototype result on the held-out uniform/B3/seed109 case.
INCUMBENT = ((3, 2, 8), (5, 7, 11), (9, 0, 10), (4, 6, 1))


def _partitions(items, batch_size):
    if len(items) == batch_size:
        yield (items,)
        return
    for companions in combinations(items[1:], batch_size - 1):
        batch = (items[0], *companions)
        remaining = tuple(i for i in items if i not in batch)
        for rest in _partitions(remaining, batch_size):
            yield (batch, *rest)


def inspect_trap():
    loads = [sum(COSTS[i] for i in batch) for batch in INCUMBENT]
    quality = integer_quality(loads, 2)
    exact = solve_exact(COSTS, 3, 2)
    results = {}
    for width in (2, 3):
        best, count = quality, 0
        for groups in combinations(range(len(INCUMBENT)), width):
            items = tuple(i for j in groups for i in INCUMBENT[j])
            outside = [loads[j] for j in range(len(loads)) if j not in groups]
            for candidate in _partitions(items, 3):
                count += 1
                score = integer_quality(outside + [sum(COSTS[i] for i in g) for g in candidate], 2)
                best = min(best, score)
        results[width] = (count, best)
    assert quality == (130, 12970, 248)
    assert results[2] == (60, quality)
    assert results[3] == (1120, exact.quality)
    assert exact.quality == (126, 12591, 244)
    return results


if __name__ == "__main__":
    print("width -> (partitions examined, best (peak, P99 * 100, sum step maxima))")
    print(inspect_trap())
