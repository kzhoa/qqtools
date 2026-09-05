"""Compare static planners, separating vectorization from step grouping.

Run: PYTHONPATH=src python exps/test_balance_v3/compare_layered_numpy.py --large
No epoch shuffling, swaps, padding or disk caches are included. Scalar variants
use the supplied reference partitioner, but omit its schedule object/traversal
construction. All variants return int64 [step, rank, sample] arrays. Timings
include input conversion and output construction, exclude metrics/assertions.
Integer-valued synthetic costs and batch sums remain exactly representable in
float64, so the scalar and vectorized layered load multisets must agree even
though their random tie-breaking differs. Timings are medians of three runs;
quality metrics use seed 7, not an average across different plans.
"""

import argparse
from random import Random
from time import perf_counter

import numpy as np

from qqtools.data.qbalance import _plan_rank_batches
from layered_numpy import _plan_layered_rank_batches
from reference_schedule import _partition_equal


def _scalar_layered(costs, batch_size, world_size, seed, *, should_balance_steps):
    rng = Random(seed)
    batches, loads = _partition_equal(costs.tolist(), len(costs) // batch_size, rng, 0)
    if should_balance_steps:
        step_ids, _ = _partition_equal(loads, len(loads) // world_size, rng, 0)
    else:
        step_ids = np.argsort(loads, kind="stable").reshape(-1, world_size)
    return np.ascontiguousarray(np.asarray(batches, dtype=np.int64)[step_ids])


def _make_costs(total, distribution):
    rng = np.random.default_rng(42)
    if distribution == "uniform":
        values = rng.uniform(1, 100, total)
    elif distribution == "lognormal":
        values = rng.lognormal(3, 1.5, total)
    elif distribution == "bimodal":
        values = np.where(rng.random(total) < 0.1, 1000, 1)
    else:
        values = np.ones(total)
        values[0] = 100_000
    return np.rint(values).astype(np.int64)


def _benchmark_case(total, distribution, batch_size, world_size):
    costs = _make_costs(total, distribution)
    timings = {method: [] for method in METHODS}
    metrics = {}
    for repeat, seed in enumerate((7, 13, 29)):
        layered_loads = []
        # Rotate execution order to reduce systematic warm-up bias.
        for method in METHODS[repeat:] + METHODS[:repeat]:
            start = perf_counter()
            if method == "lpt":
                plan = _plan_rank_batches(
                    costs, batch_size=batch_size, world_size=world_size,
                    seed=seed, should_shuffle=False,
                )
            elif method == "layered_numpy_adjacent":
                plan = _plan_layered_rank_batches(
                    costs, batch_size=batch_size, world_size=world_size, seed=seed,
                )
            else:
                plan = _scalar_layered(
                    costs, batch_size, world_size, seed,
                    should_balance_steps=method == "layered_scalar_two_level",
                )
            timings[method].append((perf_counter() - start) * 1000)
            assert plan.shape == (total // (batch_size * world_size), world_size, batch_size)
            assert np.array_equal(np.sort(plan.ravel()), np.arange(total))
            loads = costs[plan].sum(axis=2)
            if method != "lpt":
                layered_loads.append(np.sort(loads.ravel()))
            if seed == 7:
                metrics[method] = (
                    np.percentile(loads, 99), loads.max(), np.ptp(loads, axis=1).mean(),
                    loads.sum(axis=1).max(), loads.max(axis=1).sum(),
                )
            del plan
        for loads in layered_loads[1:]:
            np.testing.assert_array_equal(loads, layered_loads[0])
    for method in METHODS:
        prefix = f"{total},{distribution},{batch_size},{world_size},{method}"
        row = (np.median(timings[method]), *metrics[method])
        print(prefix + "," + ",".join(f"{value:.2f}" for value in row), flush=True)


METHODS = (
    "lpt", "layered_scalar_two_level",
    "layered_scalar_adjacent", "layered_numpy_adjacent",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", dest="should_measure_large")
    args = parser.parse_args()
    print("N,distribution,B,R,method,median_ms,p99_seed7,peak_seed7,"
          "mean_span_seed7,step_peak_seed7,sum_step_max_seed7", flush=True)
    # Warm each implementation before collecting timings.
    _benchmark_case(256, "uniform", 16, 4)
    for distribution in ("uniform", "lognormal", "bimodal", "outlier"):
        _benchmark_case(16384, distribution, 16, 4)
    for batch_size, world_size in ((1, 4), (4, 4), (64, 4), (16, 1)):
        _benchmark_case(16384, "lognormal", batch_size, world_size)
    if args.should_measure_large:
        for total in (65536, 1_000_000):
            _benchmark_case(total, "lognormal", 16, 4)


if __name__ == "__main__":
    main()
