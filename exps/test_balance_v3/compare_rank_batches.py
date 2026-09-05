"""Compare complete CPU plans; no sampler API or persistent cache is changed."""

import argparse
from itertools import product
from time import perf_counter

import numpy as np

from qqtools.data.qbalance import (
    _plan_rank_batches,
    assign_window_to_ranks,
    compute_global_even_sort_order,
)


def _window_plan(costs, *, batch_size, world_size, seed, strategy):
    """Use the current window assignment, including its per-window validation cost."""
    global_size = batch_size * world_size
    if strategy == "global_lpt":
        windows = _plan_rank_batches(
            costs, batch_size=global_size, world_size=1, seed=seed
        ).reshape(-1, global_size)
    else:
        order = compute_global_even_sort_order(costs, strategy=strategy, seed=seed)
        target_size = (len(costs) + global_size - 1) // global_size * global_size
        windows = np.resize(order, target_size).reshape(-1, global_size)
    return np.asarray([
        assign_window_to_ranks(window, costs, world_size, batch_size) for window in windows
    ], dtype=np.int64)


def _adjacent_correlation(values):
    if values.size < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _compare(args):
    print(
        "distribution,N,B,R,seed,method,plan_ms,batch_p50,batch_p95,batch_p99,batch_max,"
        "step_p99,step_max,rank_span_mean,rank_span_max,rank0_adjacent_cost_corr",
        flush=True,
    )
    for total in args.sizes:
        for distribution in ("uniform", "lognormal"):
            rng = np.random.default_rng(42)
            costs = (
                rng.uniform(1, 100, total)
                if distribution == "uniform" else rng.lognormal(3, 1.5, total)
            )
            for batch_size, world_size, seed in product(
                args.batch_sizes, args.world_sizes, args.seeds
            ):
                _compare_case(costs, distribution, batch_size, world_size, seed)


def _compare_case(costs, distribution, batch_size, world_size, seed):
    for method in ("v1", "experimental_v3", "global_lpt_local_lpt", "direct_rank_lpt"):
        start = perf_counter()
        if method == "direct_rank_lpt":
            plan = _plan_rank_batches(
                costs, batch_size=batch_size, world_size=world_size, seed=seed
            )
        else:
            strategy = {
                "v1": "v1",
                "experimental_v3": "v3",
                "global_lpt_local_lpt": "global_lpt",
            }[method]
            plan = _window_plan(
                costs, batch_size=batch_size, world_size=world_size, seed=seed, strategy=strategy
            )
        elapsed_ms = (perf_counter() - start) * 1000
        expected_size = (len(costs) + batch_size * world_size - 1) // (batch_size * world_size)
        assert plan.shape == (expected_size, world_size, batch_size)
        assert plan.min() >= 0 and plan.max() < len(costs)
        assert np.array_equal(np.unique(plan), np.arange(len(costs)))
        if plan.size == len(costs):
            assert np.array_equal(np.sort(plan.ravel()), np.arange(len(costs)))
        loads = costs[plan].sum(axis=2)
        step_loads = loads.sum(axis=1)
        spans = np.ptp(loads, axis=1)
        correlation = _adjacent_correlation(costs[plan[:, 0, :]].ravel())
        print(
            f"{distribution},{len(costs)},{batch_size},{world_size},{seed},{method},"
            f"{elapsed_ms:.2f},{np.percentile(loads, 50):.2f},{np.percentile(loads, 95):.2f},"
            f"{np.percentile(loads, 99):.2f},{loads.max():.2f},"
            f"{np.percentile(step_loads, 99):.2f},{step_loads.max():.2f},"
            f"{spans.mean():.2f},{spans.max():.2f},{correlation:.4f}",
            flush=True,
        )


def _large_plan_timings(args):
    print("direct_plan_timing,N,B,R,median_ms,plan_bytes", flush=True)
    for total in (100_000, 1_000_000):
        costs = np.random.default_rng(42).lognormal(3, 1.5, total)
        timings = []
        for seed in args.seeds:
            start = perf_counter()
            plan = _plan_rank_batches(costs, batch_size=16, world_size=4, seed=seed)
            timings.append((perf_counter() - start) * 1000)
        print(f"direct_plan_timing,{total},16,4,{np.median(timings):.2f},{plan.nbytes}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--large", dest="should_measure_large", action="store_true")
    args = parser.parse_args()
    if min(args.sizes + args.batch_sizes + args.world_sizes) <= 0 or min(args.seeds) < 0:
        parser.error(
            "sizes, batch sizes and world sizes must be positive; seeds must be nonnegative"
        )
    _compare(args)
    if args.should_measure_large:
        _large_plan_timings(args)


if __name__ == "__main__":
    main()
