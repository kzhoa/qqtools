"""Fast step-refinement quality/cost study. Run with --exact or --large."""

import argparse
from time import perf_counter

import numpy as np

from compare_exact import fixed_cases
from exact_rank_batches import integer_quality, solve_exact
from fast_step import plan_fast


def metrics(costs, plan):
    loads = costs[plan].sum(2)
    return np.array([loads.max(), np.percentile(loads, 99), loads.max(1).sum()])


def exact():
    hits = {False: [0, 0], True: [0, 0]}
    improved = 0
    for _, costs, batch_size, world_size in fixed_cases(should_fresh=True):
        optimum = solve_exact(costs, batch_size, world_size).quality
        scores = {}
        for enabled in (False, True):
            plan = plan_fast(costs, batch_size, world_size, should_optimize=enabled)
            np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(len(costs)))
            scores[enabled] = integer_quality(costs[plan].sum(2).ravel(), world_size)
            assert scores[enabled] >= optimum
            hits[enabled][0] += scores[enabled] == optimum
            hits[enabled][1] += scores[enabled][0] == optimum[0]
        assert all(a <= b for a, b in zip(scores[True], scores[False]))
        improved += scores[True] < scores[False]
    print(f"exact300 full/peak before={hits[False]} after={hits[True]} improved={improved}")


def timing(total, distribution, batch_size=16, world_size=4):
    rng = np.random.default_rng(42)
    if distribution == "uniform":
        costs = rng.integers(1, 101, total).astype(float)
    elif distribution == "bimodal":
        costs = np.where(rng.random(total) < .1, 1000., 1.)
    elif distribution == "outlier":
        costs = np.ones(total)
        costs[0] = 100000
    else:
        costs = np.rint(rng.lognormal(3, 1.5, total))
    times = {False: [], True: []}
    sample_metrics = {}
    improved = 0
    for enabled in (False, True):
        plan_fast(costs, batch_size, world_size, should_optimize=enabled)
    for repeat, seed in enumerate((7, 13, 29, 7, 13, 29, 7)):
        scores, plans = {}, {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            start = perf_counter()
            plan = plan_fast(costs, batch_size, world_size, seed, enabled)
            times[enabled].append((perf_counter() - start) * 1000)
            plans[enabled] = plan
            scores[enabled] = metrics(costs, plan)
        np.testing.assert_array_equal(np.sort(plans[False].ravel()), np.sort(plans[True].ravel()))
        assert np.all(scores[True] <= scores[False])
        improved += bool(np.any(scores[True] < scores[False]))
        if repeat == 0:
            sample_metrics = scores
    before, after = np.median(times[False]), np.median(times[True])
    values = [*sample_metrics[False], *sample_metrics[True]]
    print(f"{total},{distribution},{batch_size},{world_size},{before:.3f},{after:.3f},"
          f"{100 * (after / before - 1):.1f},{improved}/7," + ",".join(f"{v:.2f}" for v in values),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.exact:
        exact()
        return
    print("N,distribution,B,R,before_ms,after_ms,overhead_pct,improved_repeats,"
          "before_peak,before_p99,before_step,after_peak,after_p99,after_step")
    for distribution in ("uniform", "lognormal", "bimodal", "outlier"):
        timing(16384, distribution)
    timing(16385, "lognormal")
    timing(16384, "lognormal", 16, 1)
    timing(16384, "lognormal", 64, 4)
    timing(100096, "lognormal")
    if args.large:
        timing(1000000, "lognormal")


if __name__ == "__main__":
    main()
