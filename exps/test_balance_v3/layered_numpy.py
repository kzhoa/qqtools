"""Experimental vectorized layered partition; not a registered sampler strategy.

Uses qqtools' finite nonnegative float64 cost model, not arbitrary-precision
integer arithmetic. Requires exact divisibility: no implicit dropping/padding.
The static plan does not shuffle traversal or depend on an epoch. Seeded ties
affect memberships; consumers may shuffle steps/rank labels separately.
"""

import numpy as np

from qqtools.data.qbalance import _normalize_sample_costs


def _partition_layered(
    costs: np.ndarray, batch_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Partition validated, divisible costs with one item per batch per layer."""
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(costs))
    order = selected[np.argsort(-costs[selected], kind="stable")]
    batch_count = len(costs) // batch_size
    # Each layer writes into a contiguous backing row rather than a strided column.
    layers = np.empty((batch_size, batch_count), dtype=np.int64)
    loads = np.zeros(batch_count, dtype=np.float64)
    targets = np.arange(batch_count)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for layer, items in enumerate(order.reshape(batch_size, batch_count)):
                rng.shuffle(targets)
                targets = targets[np.argsort(loads[targets], kind="stable")]
                layers[layer, targets] = items
                loads[targets] += costs[items]
        except FloatingPointError as error:
            raise ValueError("Accumulated batch costs must remain finite") from error
    return layers.T, loads


def _plan_layered_rank_batches(
    sample_costs: np.ndarray | list[float],
    *,
    batch_size: int,
    world_size: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Return a static [step, rank, sample] plan using adjacent batch loads."""
    costs = _normalize_sample_costs(sample_costs)
    for name, value in (("batch_size", batch_size), ("world_size", world_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    batch_size, world_size = int(batch_size), int(world_size)
    if len(costs) % (batch_size * world_size):
        raise ValueError("sample count must be divisible by batch_size * world_size")
    batches, loads = _partition_layered(costs, batch_size, seed)
    step_ids = np.argsort(loads, kind="stable").reshape(-1, world_size)
    return np.ascontiguousarray(batches[step_ids])
