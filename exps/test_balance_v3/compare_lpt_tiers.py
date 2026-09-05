"""Benchmark public LPT tiers, asserting monotonic quality on identical inputs.

PYTHONPATH=src python exps/test_balance_v3/compare_lpt_tiers.py --large
Reports median planning time over seeds 7/13/29 and seed-7 quality metrics.
Timing excludes metric computation; all runs use static plans without traversal
shuffle. This measures CPU planning and additive costs, not GPU training speed.
"""

import argparse
from time import perf_counter

import numpy as np

from qqtools.data.qbalance import _LPT_STRATEGIES, _plan_rank_batches


def _benchmark_case(total, distribution, batch_size=16, world_size=4):
    rng = np.random.default_rng(42)
    if distribution == "uniform":
        costs = np.rint(rng.uniform(1, 100, total))
    elif distribution == "bimodal":
        costs = np.where(rng.random(total) < 0.1, 1000.0, 1.0)
    elif distribution == "outlier":
        costs = np.ones(total)
        costs[0] = 100_000
    else:
        costs = np.rint(rng.lognormal(3, 1.5, total))
    timings = {strategy: [] for strategy in _LPT_STRATEGIES}
    metrics = {}
    for repeat, seed in enumerate((7, 13, 29)):
        scores = {}
        strategies = _LPT_STRATEGIES[repeat:] + _LPT_STRATEGIES[:repeat]
        for strategy in strategies:
            start = perf_counter()
            plan = _plan_rank_batches(
                costs, batch_size=batch_size, world_size=world_size, seed=seed,
                strategy=strategy, should_shuffle=False,
            )
            timings[strategy].append((perf_counter() - start) * 1000)
            np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(total))
            loads = costs[plan].sum(2)
            scores[strategy] = (loads.max(), np.percentile(loads, 99), loads.max(1).sum())
            if seed == 7:
                metrics[strategy] = (*scores[strategy], np.ptp(loads, axis=1).mean())
        # Step sums remain diagnostics, not the grouping objective.
        assert scores["lpt_best"][:2] <= scores["lpt"][:2] <= scores["lpt_fast"][:2]
    for strategy in _LPT_STRATEGIES:
        row = (np.median(timings[strategy]), *metrics[strategy])
        print(f"{total},{distribution},{batch_size},{world_size},{strategy},"
              + ",".join(f"{value:.2f}" for value in row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", dest="should_measure_large")
    args = parser.parse_args()
    print("N,distribution,B,R,strategy,median_ms,peak_seed7,p99_seed7,"
          "sum_step_max_seed7,mean_span_seed7", flush=True)
    _benchmark_case(256, "uniform")
    for distribution in ("uniform", "lognormal", "bimodal", "outlier"):
        _benchmark_case(16384, distribution)
    for batch_size, world_size in ((1, 4), (4, 4), (64, 4), (16, 1)):
        _benchmark_case(16384, "lognormal", batch_size, world_size)
    if args.should_measure_large:
        _benchmark_case(1_000_000, "lognormal")


if __name__ == "__main__":
    main()
