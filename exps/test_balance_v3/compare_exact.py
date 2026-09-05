"""Compare fixed small instances with exhaustive optima; no timing assertions.

PYTHONPATH=src python exps/test_balance_v3/compare_exact.py
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --holdout

Seeds 0..9 and 100..119 are development/historical evaluation data.
Seeds 200..219 are fresh held-out evaluation data for bounded repair.
The legacy best implementation is frozen here to keep before/after reproducible.
"""

import argparse
from time import perf_counter

import numpy as np

from exact_rank_batches import integer_quality, solve_exact
from qqtools.data.qbalance import (
    _partition_capacity_batches, _partition_layered_batches, _plan_rank_batches,
    _rank_batch_loads, _rank_batch_quality, _swap_rank_batch_pair,
)


def legacy_best(costs, batch_size, world_size, seed=7):
    """Frozen four-pass refinement of the middle tier before the exact-oracle work."""
    # Preserve original batch IDs: final load sorting would change stable tie behavior.
    selected = np.random.default_rng(seed).permutation(len(costs))
    order = selected[np.argsort(-costs[selected], kind="stable")]
    best = _partition_layered_batches(costs, order, batch_size)
    loads = _rank_batch_loads(costs, best)
    if batch_size == 1 or len(best) == 1 or np.all(loads == loads[0]):
        return best
    capacity = _partition_capacity_batches(costs, order, batch_size)
    if _rank_batch_quality(_rank_batch_loads(costs, capacity), world_size) <= (
        _rank_batch_quality(loads, world_size)
    ):
        best = capacity
    trial = best.copy()
    loads = _rank_batch_loads(costs, trial)
    quality = _rank_batch_quality(loads, world_size)
    rng = np.random.default_rng(seed)
    pair_count = min(len(trial) // 2, 4096)
    for pass_index in range(4):
        if not pair_count or np.all(loads == loads[0]):
            break
        order = (np.argsort(loads, kind="stable") if pass_index % 2 == 0
                 else rng.permutation(len(trial)))
        for first, second in zip(order[:pair_count], order[-pair_count:][::-1]):
            _swap_rank_batch_pair(costs, trial, loads, int(first), int(second))
        loads = _rank_batch_loads(costs, trial)
        score = _rank_batch_quality(loads, world_size)
        if score < quality:
            best, quality = trial.copy(), score
    return best


def fixed_cases(should_holdout=False, should_fresh=False):
    seeds = range(200, 220) if should_fresh else (range(100, 120) if should_holdout else range(10))
    for seed in seeds:
        for batch_size, world_size in ((2, 2), (3, 2), (4, 3)):
            for distribution in ("uniform", "lognormal", "bimodal", "outlier", "ties"):
                rng = np.random.default_rng(seed)
                if distribution == "uniform":
                    costs = rng.integers(1, 101, 12)
                elif distribution == "lognormal":
                    costs = np.rint(rng.lognormal(3, 1.5, 12)).astype(np.int64)
                elif distribution == "bimodal":
                    costs = rng.integers(1, 10, 12) + (rng.random(12) < 0.3) * 100
                elif distribution == "outlier":
                    costs = rng.integers(1, 10, 12)
                    costs[0] = 1000
                else:
                    costs = rng.integers(0, 5, 12)
                yield f"{distribution}/B{batch_size}/seed{seed}", costs, batch_size, world_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true", dest="should_holdout")
    parser.add_argument("--fresh-holdout", action="store_true", dest="should_fresh")
    parser.add_argument("--prototype", action="store_true", dest="should_prototype")
    args = parser.parse_args()
    names = ("lpt_fast", "lpt", "legacy_best", "lpt_best")
    if args.should_prototype:
        names += ("prototype",)
    results = {name: [] for name in names}
    timings = {name: [] for name in names}
    worst_cases = {name: None for name in names}
    max_gaps = {name: -1.0 for name in names}
    prototype_gains = 0
    start = perf_counter()
    for label, costs, batch_size, world_size in fixed_cases(args.should_holdout, args.should_fresh):
        exact = solve_exact(costs, batch_size, world_size)
        scores = {}
        batch_scores = {}
        for name in names:
            began = perf_counter()
            if name == "prototype":
                from refine_prototype import prototype_best
                batches = prototype_best(costs.astype(float), batch_size, world_size)
            elif name == "legacy_best":
                batches = legacy_best(costs.astype(float), batch_size, world_size)
            else:
                batches = _plan_rank_batches(
                    costs, batch_size=batch_size, world_size=world_size, seed=7,
                    strategy=name, should_shuffle=False,
                ).reshape(-1, batch_size)
            timings[name].append(1000 * (perf_counter() - began))
            np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(len(costs)))
            quality = integer_quality(costs[batches].sum(1), world_size)
            assert quality >= exact.quality
            scores[name] = quality
            batch_scores[name] = _rank_batch_quality(costs[batches].sum(1))
            gap = (quality[0] / exact.quality[0] - 1) if exact.quality[0] else 0.0
            results[name].append((quality == exact.quality, quality[0] == exact.quality[0], gap))
            if gap > max_gaps[name]:
                max_gaps[name] = gap
                worst_cases[name] = (label, costs.tolist(), quality, exact.quality, exact.batches)
        # The oracle still reports the historical step objective for comparison;
        # base tiers retain world-independent quality; best's derived plan can
        # trade squared loads for step quality without worsening peak or P99.
        assert batch_scores["lpt_best"][:2] <= batch_scores["lpt"][:2] <= batch_scores["lpt_fast"][:2]
        if args.should_prototype:
            assert scores["prototype"] <= scores["legacy_best"]
            prototype_gains += scores["prototype"] < scores["lpt_best"]
        if scores["lpt_best"] > exact.quality and len(results["lpt_best"]) <= 30:
            print(f"miss {label}: costs={costs.tolist()} optimal={exact.quality} "
                  f"best={scores['lpt_best']}", flush=True)
    print("strategy,cases,optimal_tuple,optimal_peak,mean_peak_gap_pct,max_peak_gap_pct,median_ms")
    for name in names:
        rows = np.array(results[name])
        print(f"{name},{len(rows)},{int(rows[:, 0].sum())},{int(rows[:, 1].sum())},"
              f"{100 * rows[:, 2].mean():.4f},{100 * rows[:, 2].max():.4f},"
              f"{np.median(timings[name]):.3f}")
    if args.should_prototype:
        print(f"prototype_strict_improvements={prototype_gains}")
    for name in ("lpt_best", "prototype") if args.should_prototype else ("lpt_best",):
        print(f"worst {name}: {worst_cases[name]}")
    print(f"elapsed_s={perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
