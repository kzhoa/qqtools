"""Small-instance exhaustive oracle, independent of the production heuristics.

Enumerate unlabelled fixed-cardinality partitions exactly once by anchoring the
first remaining sample in the next batch. Integer arithmetic also represents
linear P99 exactly (scaled by 100). Similar-load step grouping minimizes the sum
of step maxima for fixed batch loads, so step permutations need not be enumerated.
This is an experiment, not a production solver; N > 24 or large searches are rejected.
"""

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class ExactPartition:
    batches: tuple[tuple[int, ...], ...]
    quality: tuple[int, int, int]  # peak, 100 * linear P99, sum of step maxima
    partitions: int


def integer_quality(loads: Sequence[int], world_size: int) -> tuple[int, int, int]:
    ordered = sorted(int(load) for load in loads)
    lower, remainder = divmod(99 * (len(ordered) - 1), 100)
    upper = min(lower + 1, len(ordered) - 1)
    p99_scaled = ordered[lower] * (100 - remainder) + ordered[upper] * remainder
    return ordered[-1], p99_scaled, sum(ordered[world_size - 1::world_size])


def solve_exact(
    costs: Sequence[int], batch_size: int, world_size: int = 1,
    *, max_partitions: int = 200_000,
) -> ExactPartition:
    """Exhaust all partitions or reject before starting; never report a partial optimum."""
    for name, value in (("batch_size", batch_size), ("world_size", world_size),
                        ("max_partitions", max_partitions)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not len(costs) or len(costs) % (batch_size * world_size):
        raise ValueError("costs must be nonempty and divisible by batch_size * world_size")
    if len(costs) > 24:
        raise ValueError("exhaustive oracle supports at most 24 samples")
    if any(isinstance(c, bool) or not isinstance(c, Integral) or c < 0 for c in costs):
        raise ValueError("oracle costs must be nonnegative integers")
    weights = tuple(map(int, costs))
    n = len(weights)
    batch_count = n // batch_size
    count = factorial(n) // (factorial(batch_size) ** batch_count * factorial(batch_count))
    if count > max_partitions:
        raise ValueError(f"{count} partitions exceed exhaustive budget {max_partitions}")
    best_quality = None
    best_batches = ()
    visited = 0

    def visit(remaining, batches, loads):
        nonlocal best_quality, best_batches, visited
        if len(remaining) == batch_size:
            visited += 1
            quality = integer_quality((*loads, sum(weights[i] for i in remaining)), world_size)
            if best_quality is None or quality < best_quality:
                best_quality = quality
                best_batches = (*batches, remaining)
            return
        first, rest = remaining[0], remaining[1:]
        for companions in combinations(rest, batch_size - 1):
            batch = (first, *companions)
            chosen = set(companions)
            visit(tuple(i for i in rest if i not in chosen), (*batches, batch),
                  (*loads, sum(weights[i] for i in batch)))

    visit(tuple(range(n)), (), ())
    assert visited == count and best_quality is not None
    return ExactPartition(best_batches, best_quality, visited)
