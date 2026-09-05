"""Exact quality and end-to-end timing for a single vectorized fast swap pass.

PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --large
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --extended
Exact data reuse seeds 200..219 from the previous comparison (not a new holdout).
Timing uses seven paired repetitions, rotating order; it excludes result checks.
All timing runs include production lpt_best; output includes median/min/max latency.
"""

import argparse
from time import perf_counter

import numpy as np

from compare_exact import fixed_cases
from exact_rank_batches import integer_quality, solve_exact
from fast_swap import plan_fast_swap
from fast_two_pointer import plan_fast_two_pointer
from layered_baseline import plan_layered_baseline
from qqtools.data.qbalance import _plan_rank_batches, _rank_batch_quality


def _plan(costs, batch_size, world_size, seed, name):
    if name == "layered":
        return plan_layered_baseline(costs, batch_size=batch_size, world_size=world_size,
                                     seed=seed, should_shuffle=False)
    if name == "fast_swap":
        return plan_fast_swap(costs, batch_size=batch_size, world_size=world_size, seed=seed)
    if name == "two_pointer":
        return plan_fast_two_pointer(costs, batch_size=batch_size, world_size=world_size, seed=seed)
    return _plan_rank_batches(costs, batch_size=batch_size, world_size=world_size,
                              seed=seed, strategy=name, should_shuffle=False)


def extended_cases():
    for seed in range(200, 220):
        for total, batch_size in ((12, 6), (16, 8)):
            for distribution in ("uniform", "lognormal", "bimodal", "outlier", "ties"):
                rng = np.random.default_rng(seed)
                if distribution == "uniform":
                    costs = rng.integers(1, 101, total)
                elif distribution == "lognormal":
                    costs = np.rint(rng.lognormal(3, 1.5, total)).astype(np.int64)
                elif distribution == "bimodal":
                    costs = rng.integers(1, 10, total) + (rng.random(total) < 0.3) * 100
                elif distribution == "outlier":
                    costs = rng.integers(1, 10, total)
                    costs[0] = 1000
                else:
                    costs = rng.integers(0, 5, total)
                yield f"{distribution}/B{batch_size}/seed{seed}", costs, batch_size, 2


def exact_comparison(should_two_pointer=False, should_extended=False):
    names = ("layered", "fast_swap") + (("two_pointer",) if should_two_pointer else ())
    names += ("lpt_fast", "lpt", "lpt_best")
    results = {name: [] for name in names}
    cases = extended_cases() if should_extended else fixed_cases(should_fresh=True)
    for _, costs, batch_size, world_size in cases:
        optimum = solve_exact(costs, batch_size, world_size).quality
        scores = {}
        batch_scores = {}
        for name in names:
            plan = _plan(costs, batch_size, world_size, 7, name)
            np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(len(costs)))
            score = integer_quality(costs[plan].sum(2).ravel(), world_size)
            assert score >= optimum
            scores[name] = score
            batch_scores[name] = _rank_batch_quality(costs[plan].sum(2).ravel())
            gap = score[0] / optimum[0] - 1 if optimum[0] else 0.0
            results[name].append((score == optimum, score[0] == optimum[0], gap))
        assert scores["fast_swap"] <= scores["layered"]
        if should_two_pointer:
            assert scores["two_pointer"] <= scores["layered"]
        assert batch_scores["lpt_best"][:2] <= batch_scores["lpt"][:2] <= batch_scores["lpt_fast"][:2]
    print("strategy,cases,optimal_tuple,optimal_peak,mean_peak_gap_pct,max_peak_gap_pct")
    for name in names:
        rows = np.asarray(results[name])
        print(f"{name},{len(rows)},{int(rows[:, 0].sum())},{int(rows[:, 1].sum())},"
              f"{100 * rows[:, 2].mean():.4f},{100 * rows[:, 2].max():.4f}", flush=True)


def timing_comparison(total, distribution, batch_size=16, world_size=4, should_two_pointer=False):
    rng = np.random.default_rng(42)
    if distribution == "uniform":
        costs = np.rint(rng.uniform(1, 100, total))
    elif distribution == "outlier":
        costs = np.ones(total)
        costs[0] = 100_000
    else:
        costs = np.rint(rng.lognormal(3, 1.5, total))
    names = ("layered", "fast_swap") + (("two_pointer",) if should_two_pointer else ())
    names += ("lpt_fast", "lpt", "lpt_best")
    timings = {name: [] for name in names}
    metrics = {}
    for name in names:
        _plan(costs, batch_size, world_size, 7, name)  # Untimed warmup.
    for repeat, seed in enumerate((7, 13, 29, 7, 13, 29, 7)):
        scores = {}
        batch_scores = {}
        ordered_names = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in ordered_names:
            began = perf_counter()
            plan = _plan(costs, batch_size, world_size, seed, name)
            timings[name].append((perf_counter() - began) * 1000)
            np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(total))
            loads = costs[plan].sum(2)
            scores[name] = _rank_batch_quality(loads.ravel(), world_size)
            batch_scores[name] = _rank_batch_quality(loads.ravel())
            if repeat == 0:
                metrics[name] = (loads.max(), np.percentile(loads, 99), loads.max(1).sum())
        assert scores["fast_swap"] <= scores["layered"]
        if should_two_pointer:
            assert scores["two_pointer"] <= scores["layered"]
        assert batch_scores["lpt_best"][:2] <= batch_scores["lpt"][:2] <= batch_scores["lpt_fast"][:2]
    for name in names:
        peak, p99, step_sum = metrics[name]
        print(f"{total},{distribution},{batch_size},{world_size},{name},"
              f"{np.median(timings[name]):.3f},{min(timings[name]):.3f},"
              f"{max(timings[name]):.3f},{peak:.2f},{p99:.2f},{step_sum:.2f}", flush=True)
    for name in ("fast_swap", "two_pointer") if should_two_pointer else ("fast_swap",):
        delta_ms = np.median(timings[name]) - np.median(timings["layered"])
        print(f"{name},overhead_ms={delta_ms:.3f},"
              f"overhead_pct={100 * delta_ms / np.median(timings['layered']):.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="store_true", dest="should_exact")
    parser.add_argument("--large", action="store_true", dest="should_large")
    parser.add_argument("--two-pointer", action="store_true", dest="should_two_pointer")
    parser.add_argument("--extended", action="store_true", dest="should_extended")
    args = parser.parse_args()
    if args.should_exact or args.should_extended:
        exact_comparison(args.should_two_pointer, args.should_extended)
        return
    print("N,distribution,B,R,strategy,median_ms,min_ms,max_ms,"
          "peak_seed7,p99_seed7,sum_step_max_seed7")
    timing_comparison(12, "uniform", 3, 2, args.should_two_pointer)
    for distribution in ("uniform", "lognormal", "outlier"):
        timing_comparison(16384, distribution, should_two_pointer=args.should_two_pointer)
    for batch_size, world_size in ((1, 4), (4, 4), (64, 4), (16, 1)):
        timing_comparison(16384, "lognormal", batch_size, world_size, args.should_two_pointer)
    if args.should_large:
        for distribution in ("lognormal", "uniform", "outlier"):
            timing_comparison(1_000_000, distribution, should_two_pointer=args.should_two_pointer)


if __name__ == "__main__":
    main()
