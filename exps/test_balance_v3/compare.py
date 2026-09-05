"""Compare the experimental V3 with the committed baseline; run from the repo root."""

import subprocess
from time import perf_counter

import numpy as np

from qqtools.data.qbalance import assign_window_to_ranks, compute_global_even_sort_order


baseline = {}
exec(subprocess.check_output(
    ["git", "show", "HEAD:src/qqtools/data/qbalance.py"], text=True
), baseline)
strategies = {
    "v1": lambda costs, seed: compute_global_even_sort_order(costs, seed=seed, strategy="v1"),
    "v2": lambda costs, seed: compute_global_even_sort_order(costs, seed=seed, strategy="v2"),
    "old_v3": lambda costs, seed: baseline["compute_global_even_sort_order"](
        costs, seed=seed, strategy="v3"
    ),
    "new_v3": lambda costs, seed: compute_global_even_sort_order(costs, seed=seed, strategy="v3"),
}

print("distribution,N,seed,world,strategy,order_ms,batch_p99,batch_max,span_mean")
for total in (4096, 16384):
    for distribution in ("uniform", "lognormal"):
        rng = np.random.default_rng(42)
        costs = (rng.uniform(1, 100, total) if distribution == "uniform"
                 else rng.lognormal(3, 1.5, total))
        for seed in (7, 13):
            for name, generate in strategies.items():
                start = perf_counter()
                order = generate(costs, seed)
                elapsed = (perf_counter() - start) * 1000
                for world_size in (1, 4):
                    loads = []
                    for start in range(0, total, 16 * world_size):
                        assignment = assign_window_to_ranks(
                            order[start:start + 16 * world_size], costs, world_size, 16
                        )
                        loads.append([costs[indices].sum() for indices in assignment])
                    loads = np.asarray(loads)
                    print(f"{distribution},{total},{seed},{world_size},{name},{elapsed:.2f},"
                          f"{np.percentile(loads, 99):.2f},{loads.max():.2f},"
                          f"{np.ptp(loads, axis=1).mean():.2f}", flush=True)

print("planning_only,N,strategy,median_ms")
for total in (100_000, 1_000_000):
    costs = np.random.default_rng(42).lognormal(3, 1.5, total)
    for name, generate in strategies.items():
        timings = []
        for seed in (7, 13, 29):
            start = perf_counter()
            generate(costs, seed)
            timings.append((perf_counter() - start) * 1000)
        print(f"planning_only,{total},{name},{np.median(timings):.2f}", flush=True)
