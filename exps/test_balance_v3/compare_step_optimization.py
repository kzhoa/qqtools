"""Compare full best planning with/without the derived step pass; CPU only."""

import argparse
from time import perf_counter
from unittest.mock import patch

import numpy as np

import qqtools.data.qbalance as balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("lpt", "lpt_best"), default="lpt_best")
    args = parser.parse_args()
    print("N,distribution,before_ms,after_ms,before_step_sum,after_step_sum")
    optimizer = balance._optimize_step_batches
    for distribution in ("uniform", "lognormal", "tail"):
        total = 16385 if distribution == "tail" else 16384
        rng = np.random.default_rng(42)
        costs = (rng.integers(1, 101, total).astype(float) if distribution == "uniform"
                 else np.rint(rng.lognormal(3, 1.5, total)))
        timings = {False: [], True: []}
        plans = {}
        for repeat in range(6):
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                replacement = optimizer if enabled else lambda costs, batches, *args, **kwargs: batches
                with patch.object(balance, "_optimize_step_batches", replacement):
                    start = perf_counter()
                    plan = balance._plan_rank_batches(
                        costs, batch_size=16, world_size=4, seed=7,
                        strategy=args.strategy, should_shuffle=False,
                    )
                    elapsed = (perf_counter() - start) * 1000
                if repeat:
                    timings[enabled].append(elapsed)
                plans[enabled] = plan
            np.testing.assert_array_equal(np.sort(plans[False].ravel()), np.sort(plans[True].ravel()))
            metrics = {}
            for enabled, plan in plans.items():
                loads = costs[plan].sum(2)
                metrics[enabled] = np.array([loads.max(), np.percentile(loads, 99), loads.max(1).sum()])
            assert np.all(metrics[True] <= metrics[False])
        print(f"{total},{distribution},{np.median(timings[False]):.3f},"
              f"{np.median(timings[True]):.3f},{metrics[False][2]:.0f},{metrics[True][2]:.0f}")


if __name__ == "__main__":
    main()
