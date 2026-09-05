"""Historical layered-only benchmark, not a production compatibility path."""

import numpy as np

from qqtools.data.qbalance import (
    _normalize_sample_costs, _partition_layered_batches, _rank_batch_loads,
)


def plan_layered_baseline(
    sample_costs: np.ndarray | list[float],
    *,
    batch_size: int,
    world_size: int = 1,
    seed: int = 0,
    should_drop_last: bool = False,
    should_shuffle: bool = True,
    strategy: str = "lpt_fast",
) -> np.ndarray:
    """Frozen layered-only experimental baseline with production tail semantics."""
    if strategy != "lpt_fast":
        raise ValueError(
            f"Layered baseline only supports lpt_fast, got {strategy!r}"
        )
    costs = _normalize_sample_costs(sample_costs)
    for name, value in (("batch_size", batch_size), ("world_size", world_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    batch_size, world_size = int(batch_size), int(world_size)
    rng = np.random.default_rng(seed)
    total = len(costs)
    global_batch_size = batch_size * world_size
    if should_drop_last:
        target_size = total // global_batch_size * global_batch_size
    else:
        target_size = (total + global_batch_size - 1) // global_batch_size * global_batch_size
    if target_size == 0:
        return np.empty((0, world_size, batch_size), dtype=np.int64)

    selected = rng.permutation(total)
    if target_size < total:
        selected = selected[:target_size]
    elif target_size > total:
        selected = np.resize(selected, target_size)
    order = selected[np.argsort(-costs[selected], kind="stable")]

    batches = _partition_layered_batches(costs, order, batch_size)
    loads = _rank_batch_loads(costs, batches)
    if not np.all(np.isfinite(loads)):
        raise ValueError("Accumulated batch costs must remain finite")

    step_batches = np.argsort(loads, kind="stable").reshape(-1, world_size)
    if should_shuffle:
        rng.shuffle(step_batches, axis=0)
        step_batches = rng.permuted(step_batches, axis=1)
        batches = rng.permuted(batches, axis=1)
    return np.ascontiguousarray(batches[step_batches])
