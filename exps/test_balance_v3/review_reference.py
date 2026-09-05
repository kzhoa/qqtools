"""Assert reference invariants and compare CPU planning on identical integer costs."""

import argparse
from itertools import product
from random import Random
from time import perf_counter

import numpy as np

from qqtools.data.qbalance import _plan_rank_batches
from reference_schedule import _improve_by_swaps, _partition_equal, build_balanced_schedule


def _check_plan(costs, plan):
    expected = list(range(len(costs)))
    assert sorted(plan.permutation) == expected
    assert len(plan.permutation) == len(costs)
    calculated = tuple(tuple(sum(costs[i] for i in batch) for batch in step)
                       for step in plan.batches)
    assert plan.rank_costs == calculated
    assert sum(plan.iteration_costs) == sum(costs)
    assert plan.peak_cost == max(max(row) for row in calculated)
    assert all(len(step) == plan.n_ranks for step in plan.batches)
    assert all(len(batch) == plan.batch_size for step in plan.batches for batch in step)
    for rank in range(plan.n_ranks):
        sampler = plan.for_rank(rank)
        assert len(sampler) == plan.n_iterations
        assert list(sampler) == list(sampler)
        flattened = tuple(i for batch in sampler for i in batch)
        assert flattened == plan.interleaved_permutation[rank::plan.n_ranks]


def _memberships(plan):
    return {tuple(sorted(batch)) for step in plan.batches for batch in step}


def _check_reference():
    cases = 0
    for seed, batch_size, ranks, steps in product(range(10), (1, 2, 4, 16), (1, 2, 4), (1, 3)):
        rng = Random(seed)
        costs = [rng.randrange(101) for _ in range(batch_size * ranks * steps)]
        before = build_balanced_schedule(costs, batch_size, ranks, seed=seed)
        after = build_balanced_schedule(costs, batch_size, ranks, seed=seed, swap_passes=2)
        _check_plan(costs, before)
        _check_plan(costs, after)
        assert after == build_balanced_schedule(costs, batch_size, ranks, seed=seed, swap_passes=2)
        assert after.peak_cost <= before.peak_cost
        cases += 2

    for costs in ([0] * 64, [1] * 64, [10**100] + [1] * 63):
        _check_plan(costs, build_balanced_schedule(costs, 16, 4, swap_passes=2))
        cases += 1
    for seed in range(100):
        rng = Random(seed)
        groups_count, group_size = rng.randrange(1, 17), rng.randrange(1, 17)
        costs = [rng.randrange(1000) for _ in range(groups_count * group_size)]
        groups, loads = _partition_equal(costs, groups_count, rng, 0)
        before_max, before_square_sum = max(loads), sum(load * load for load in loads)
        _improve_by_swaps(groups, loads, costs, rng, 3)
        assert max(loads) <= before_max
        assert sum(load * load for load in loads) <= before_square_sum
        assert loads == [sum(costs[i] for i in group) for group in groups]
        assert all(len(group) == group_size for group in groups)
        assert sorted(i for group in groups for i in group) == list(range(len(costs)))

    invalid = [
        ([], 1, 1, {}), ([1, 2, 3], 2, 2, {}), ([True], 1, 1, {}),
        ([1.0], 1, 1, {}), ([-1], 1, 1, {}), ([1], 0, 1, {}),
        ([1], 1, 0, {}), ([1], True, 1, {}), ([1], 1, 1, {"epoch": -1}),
        ([1], 1, 1, {"swap_passes": -1}),
    ]
    for costs, batch_size, ranks, kwargs in invalid:
        try:
            build_balanced_schedule(costs, batch_size, ranks, **kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Invalid configuration was accepted: {costs, kwargs}")
    sample_plan = build_balanced_schedule([1, 2], 1, 2)
    for rank in (-1, 2, True):
        try:
            sample_plan.for_rank(rank)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Invalid rank was accepted: {rank}")

    unique_costs = list(range(1, 257))
    first = build_balanced_schedule(unique_costs, 4, 4, seed=7, epoch=0)
    second = build_balanced_schedule(unique_costs, 4, 4, seed=7, epoch=1)
    overlap = len(_memberships(first) & _memberships(second))
    print(f"checks: {cases} plans, 100 swap invariants, 13 invalid configurations passed")
    print(f"unique-cost epoch membership overlap: {overlap}/64 batches", flush=True)


def _benchmark_case(total, distribution, batch_size, ranks):
    rng = np.random.default_rng(42)
    # Quantize ONCE; all methods receive the same Python integer list, below 2**53.
    values = (
        rng.uniform(1, 100, total)
        if distribution == "uniform" else rng.lognormal(3, 1.5, total)
    )
    costs = np.rint(values).astype(np.int64).tolist()
    cost_array = np.asarray(costs, dtype=np.float64)
    for method in ("lpt", "layered_0", "layered_2"):
        timings = []
        metrics = None
        for seed in (7, 13, 29):
            start = perf_counter()
            if method == "lpt":
                result = _plan_rank_batches(
                    costs, batch_size=batch_size, world_size=ranks, seed=seed
                )
            else:
                result = build_balanced_schedule(
                    costs, batch_size, ranks, seed=seed,
                    swap_passes=0 if method == "layered_0" else 2,
                )
            timings.append((perf_counter() - start) * 1000)
            batches = result if method == "lpt" else np.asarray(result.batches)
            assert batches.shape == (total // (batch_size * ranks), ranks, batch_size)
            assert np.array_equal(np.sort(batches.ravel()), np.arange(total))
            loads = cost_array[batches].sum(axis=2)
            if seed == 7:
                metrics = (
                    np.percentile(loads, 99), loads.max(), loads.sum(axis=1).max(),
                    np.ptp(loads, axis=1).mean(), loads.max(axis=1).sum(),
                )
            del result, batches
        print(
            f"{total},{distribution},{batch_size},{ranks},{method},{np.median(timings):.2f},"
            + ",".join(f"{metric:.2f}" for metric in metrics), flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", dest="should_measure_large", action="store_true")
    args = parser.parse_args()
    _check_reference()
    print("N,distribution,B,R,method,median_ms,p99_seed7,max_seed7,"
          "step_max_seed7,mean_span_seed7,sum_step_max_seed7", flush=True)
    for total, distribution, ranks in product((4096, 16384), ("uniform", "lognormal"), (1, 4)):
        _benchmark_case(total, distribution, 16, ranks)
    if args.should_measure_large:
        _benchmark_case(1_000_000, "lognormal", 16, 4)


if __name__ == "__main__":
    main()
