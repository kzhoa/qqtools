"""User-supplied reference algorithm; docstrings shortened, executable logic preserved.

Research fixture only, not a qqtools public API or a proposed replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from random import Random
from typing import Iterator, Sequence


def _integer(value: int, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _improve_by_swaps(
    groups: list[list[int]],
    loads: list[int],
    weights: Sequence[int],
    rng: Random,
    passes: int,
) -> None:
    """Preserve cardinality and never increase maximum load via paired exchanges."""
    m = len(groups)
    if m < 2:
        return
    for p in range(passes):
        order = list(range(m))
        rng.shuffle(order)
        if p % 2 == 0:
            order.sort(key=loads.__getitem__)
        for j in range(m // 2):
            a, b = order[j], order[-1 - j]
            hi, lo = (a, b) if loads[a] >= loads[b] else (b, a)
            gap = loads[hi] - loads[lo]
            if gap <= 1:
                continue
            best_gain = 0
            best: tuple[int, int, int] | None = None
            for u, x in enumerate(groups[hi]):
                for v, y in enumerate(groups[lo]):
                    delta = weights[x] - weights[y]
                    if 0 < delta < gap:
                        gain = 2 * delta * (gap - delta)
                        if gain > best_gain:
                            best_gain = gain
                            best = (u, v, delta)
            if best is not None:
                u, v, delta = best
                groups[hi][u], groups[lo][v] = groups[lo][v], groups[hi][u]
                loads[hi] -= delta
                loads[lo] += delta


def _partition_equal(
    weights: Sequence[int],
    n_groups: int,
    rng: Random,
    swap_passes: int,
) -> tuple[list[list[int]], list[int]]:
    """Layered inverse pairing followed by optional one-for-one exchanges."""
    n = len(weights)
    if n_groups <= 0 or n % n_groups != 0:
        raise ValueError("equal-sized partition requires exact divisibility")
    k = n // n_groups
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=weights.__getitem__, reverse=True)
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    loads = [0] * n_groups
    targets = list(range(n_groups))
    for layer in range(k):
        rng.shuffle(targets)
        targets.sort(key=loads.__getitem__)
        start = layer * n_groups
        for j, group in enumerate(targets):
            item = order[start + j]
            groups[group].append(item)
            loads[group] += weights[item]
    _improve_by_swaps(groups, loads, weights, rng, swap_passes)
    return groups, loads


@dataclass(frozen=True)
class BalancedSchedule:
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    rank_costs: tuple[tuple[int, ...], ...]
    batch_size: int
    n_ranks: int

    @property
    def n_iterations(self) -> int:
        return len(self.batches)

    @property
    def permutation(self) -> tuple[int, ...]:
        """Flatten in [iteration, rank, sample] order."""
        return tuple(i for step in self.batches for batch in step for i in batch)

    @property
    def interleaved_permutation(self) -> tuple[int, ...]:
        """Flatten in [iteration, sample, rank] order."""
        return tuple(
            step[r][j]
            for step in self.batches
            for j in range(self.batch_size)
            for r in range(self.n_ranks)
        )

    @property
    def iteration_costs(self) -> tuple[int, ...]:
        return tuple(sum(row) for row in self.rank_costs)

    @property
    def peak_cost(self) -> int:
        return max(max(row) for row in self.rank_costs)

    def for_rank(self, rank: int) -> RankBatchSampler:
        return RankBatchSampler(self, rank)


class RankBatchSampler:
    """Reusable iterable for DataLoader batch_sampler; rebuild for a new epoch."""

    def __init__(self, schedule: BalancedSchedule, rank: int) -> None:
        self.schedule = schedule
        self.rank = _integer(rank, "rank", minimum=0)
        if self.rank >= schedule.n_ranks:
            raise ValueError("rank must be less than n_ranks")

    def __len__(self) -> int:
        return self.schedule.n_iterations

    def __iter__(self) -> Iterator[list[int]]:
        for step in self.schedule.batches:
            yield list(step[self.rank])


def build_balanced_schedule(
    costs: Sequence[int],
    batch_size: int,
    n_ranks: int,
    *,
    seed: int = 0,
    epoch: int = 0,
    swap_passes: int = 0,
) -> BalancedSchedule:
    """Build an integer-cost, exactly divisible, fully shuffled two-level plan."""
    batch_size = _integer(batch_size, "batch_size", minimum=1)
    n_ranks = _integer(n_ranks, "n_ranks", minimum=1)
    seed = _integer(seed, "seed")
    epoch = _integer(epoch, "epoch", minimum=0)
    swap_passes = _integer(swap_passes, "swap_passes", minimum=0)
    c = tuple(_integer(v, f"costs[{i}]", minimum=0) for i, v in enumerate(costs))
    n = len(c)
    global_batch_size = batch_size * n_ranks
    if n == 0:
        raise ValueError("costs must not be empty")
    if n % global_batch_size != 0:
        raise ValueError(
            f"N={n} is not divisible by batch_size*n_ranks={global_batch_size}; "
            "choose an explicit tail/drop/pad policy before building the plan"
        )
    rng = Random(f"balanced-schedule:{seed}:{epoch}")
    n_iterations = n // global_batch_size
    n_local_batches = n_iterations * n_ranks
    local_batches, local_loads = _partition_equal(c, n_local_batches, rng, swap_passes)
    step_groups, _ = _partition_equal(local_loads, n_iterations, rng, swap_passes)
    rng.shuffle(step_groups)
    steps: list[tuple[tuple[int, ...], ...]] = []
    load_rows: list[tuple[int, ...]] = []
    for group in step_groups:
        rng.shuffle(group)
        row: list[tuple[int, ...]] = []
        for local_id in group:
            batch = local_batches[local_id]
            rng.shuffle(batch)
            row.append(tuple(batch))
        steps.append(tuple(row))
        load_rows.append(tuple(local_loads[local_id] for local_id in group))
    return BalancedSchedule(tuple(steps), tuple(load_rows), batch_size, n_ranks)


if __name__ == "__main__":
    demo_costs = [9, 9, 9, 3, 6, 6, 6, 12]
    plan = build_balanced_schedule(demo_costs, batch_size=2, n_ranks=2, seed=42)
    print("permutation:", list(plan.permutation))
    print("rank costs:", plan.rank_costs)
    print("iteration costs:", plan.iteration_costs)
    for t, step in enumerate(plan.batches):
        print(f"iteration {t}:", [[demo_costs[i] for i in batch] for batch in step])
