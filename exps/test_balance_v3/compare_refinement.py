"""Scaling check for the experimental refinement (not an optimality benchmark).

PYTHONPATH=src python exps/test_balance_v3/compare_refinement.py --large
Median CPU planning time over seeds 7/13/29; additive-cost metrics use seed 7.
"""

import argparse
from time import perf_counter

import numpy as np

from refine_prototype import prototype_best
from compare_exact import legacy_best
from qqtools.data.qbalance import _plan_rank_batches, _rank_batch_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", dest="should_large")
    args = parser.parse_args()
    cases = [(16384, 4, 4), (16384, 16, 4), (16384, 64, 4), (16384, 16, 1)]
    if args.should_large:
        cases.append((1_000_000, 16, 4))
    print("N,B,R,strategy,median_ms,peak_seed7,peak_lower_bound,p99_seed7,sum_step_max_seed7",
          flush=True)
    for total, batch_size, world_size in cases:
        costs = np.rint(np.random.default_rng(42).lognormal(3, 1.5, total))
        ordered_costs = np.sort(costs)
        peak_lower_bound = max(costs.sum() / (total // batch_size),
                               ordered_costs[-1] + ordered_costs[:batch_size - 1].sum())
        timings = {name: [] for name in ("legacy_best", "lpt_best", "prototype")}
        metrics = {}
        for repeat, seed in enumerate((7, 13, 29)):
            scores = {}
            names = tuple(timings)
            names = names[repeat:] + names[:repeat]
            for name in names:
                began = perf_counter()
                if name == "prototype":
                    batches = prototype_best(costs, batch_size, world_size, seed)
                elif name == "legacy_best":
                    batches = legacy_best(costs, batch_size, world_size, seed)
                    order = np.argsort(costs[batches].sum(1), kind="stable")
                    batches = np.ascontiguousarray(batches[order])
                else:
                    batches = _plan_rank_batches(
                        costs, batch_size=batch_size, world_size=world_size,
                        strategy="lpt_best", seed=seed, should_shuffle=False,
                    ).reshape(-1, batch_size)
                timings[name].append(1000 * (perf_counter() - began))
                np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(total))
                loads = costs[batches].sum(1)
                scores[name] = _rank_batch_quality(loads, world_size)
                assert loads.max() >= peak_lower_bound
                if seed == 7:
                    metrics[name] = (loads.max(), peak_lower_bound, np.percentile(loads, 99),
                                     np.sort(loads)[world_size - 1::world_size].sum())
            assert scores["prototype"] <= scores["legacy_best"]
            assert scores["lpt_best"] <= scores["legacy_best"]
        for name in timings:
            print(f"{total},{batch_size},{world_size},{name},"
                  + ",".join(f"{v:.3f}" for v in (np.median(timings[name]), *metrics[name])),
                  flush=True)


if __name__ == "__main__":
    main()
