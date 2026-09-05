from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler

from qqtools.data.qbalance import (
    _LPT_STRATEGIES,
    _normalize_lpt_strategy,
    _plan_rank_batches,
    assign_window_to_ranks,
    compute_global_even_sort_order,
    validate_balance_strategy,
)


def _normalize_sample_costs(sample_costs: Sequence[float] | np.ndarray) -> np.ndarray:
    costs = np.asarray(sample_costs, dtype=np.float64)
    if costs.ndim != 1:
        raise ValueError(f"sample_costs must be 1D, got shape {costs.shape}")
    if not np.all(np.isfinite(costs)):
        raise ValueError("sample_costs must contain only finite values")
    if np.any(costs < 0):
        raise ValueError("sample_costs must be non-negative")
    return np.ascontiguousarray(costs)


def _validate_sample_order(
    sample_order: Sequence[int] | np.ndarray,
    total: int,
) -> np.ndarray:
    order = np.asarray(sample_order, dtype=np.int64)
    if order.ndim != 1:
        raise ValueError(f"sample_order must be 1D, got shape {order.shape}")
    if order.shape != (total,):
        raise ValueError(f"sample_order must have length {total}, got {order.shape[0]}")
    if total == 0:
        return np.ascontiguousarray(order)
    if len(np.unique(order)) != total:
        raise ValueError("sample_order must be a full permutation without duplicates")
    if order.min(initial=0) != 0 or order.max(initial=-1) != total - 1:
        raise ValueError("sample_order must be a full permutation covering [0, N)")
    return np.ascontiguousarray(order)


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _build_prefix_padding(order: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= order.shape[0]:
        return np.ascontiguousarray(order.astype(np.int64, copy=False))
    if order.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    repeat_count = int(np.ceil(target_size / order.shape[0]))
    tiled = np.tile(order, repeat_count)
    return np.ascontiguousarray(tiled[:target_size].astype(np.int64, copy=False))


def _resolve_rank_and_world_size(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    runtime_ready = _is_dist_ready()
    runtime_rank = dist.get_rank() if runtime_ready else 0
    runtime_world_size = dist.get_world_size() if runtime_ready else 1

    resolved_rank = runtime_rank if rank is None else int(rank)
    resolved_world_size = runtime_world_size if world_size is None else int(world_size)

    if runtime_ready:
        if resolved_rank != runtime_rank or resolved_world_size != runtime_world_size:
            raise ValueError(
                "Explicit rank/world_size does not match initialized torch.distributed runtime"
            )

    if resolved_world_size <= 0:
        raise ValueError(f"world_size must be positive, got {resolved_world_size}")
    if resolved_rank < 0 or resolved_rank >= resolved_world_size:
        raise ValueError(
            f"rank must be in [0, {resolved_world_size}), got {resolved_rank}"
        )
    return resolved_rank, resolved_world_size


@dataclass
class _BalancedPlanCache:
    sample_costs: np.ndarray
    batch_size: int
    rank: int
    world_size: int
    shuffle: bool
    seed: int
    drop_last: bool
    sample_order: np.ndarray | None
    strategy: str
    epoch: int = 0
    rank_local_plan: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    _lpt_plan: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.strategy in _LPT_STRATEGIES:
            if self.sample_order is not None:
                raise ValueError(f"strategy={self.strategy!r} cannot be combined with sample_order")
            remainder = len(self.sample_costs) % (self.batch_size * self.world_size)
            if remainder and not self.shuffle and not self.drop_last:
                raise ValueError(
                    "Non-shuffled LPT requires N divisible by batch_size * world_size "
                    "when drop_last=False; validation padding would duplicate samples. "
                    "Use drop_last=True only if discarding samples is intended."
                )
            self._lpt_plan = _plan_rank_batches(
                self.sample_costs,
                batch_size=self.batch_size,
                world_size=self.world_size,
                seed=self.seed,
                should_drop_last=self.drop_last,
                should_shuffle=False,
                strategy=self.strategy,
            )
            self._lpt_plan.setflags(write=False)
        self.rank_local_plan = self._build_rank_local_plan(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        if not self.shuffle:
            return
        epoch = int(epoch)
        if epoch == self.epoch:
            return
        self.rank_local_plan = self._build_rank_local_plan(epoch)
        self.epoch = epoch

    def _build_rank_local_plan(self, epoch: int) -> np.ndarray:
        if self._lpt_plan is not None:
            if not self.shuffle:
                return np.ascontiguousarray(self._lpt_plan[:, self.rank, :].reshape(-1))
            rng = np.random.default_rng(self.seed + epoch)
            steps = rng.permutation(len(self._lpt_plan))
            ranks = np.broadcast_to(np.arange(self.world_size), (len(steps), self.world_size))
            ranks = rng.permuted(ranks, axis=1)
            return np.ascontiguousarray(
                self._lpt_plan[steps, ranks[:, self.rank], :].reshape(-1)
            )
        total = int(self.sample_costs.shape[0])
        global_chunk_size = self.world_size * self.batch_size
        if self.shuffle:
            global_order = compute_global_even_sort_order(
                self.sample_costs,
                seed=self.seed + epoch,
                strategy=self.strategy,
            )
        elif self.sample_order is None:
            global_order = np.arange(total, dtype=np.int64)
        else:
            global_order = self.sample_order.copy()

        if global_chunk_size > 0:
            remainder = total % global_chunk_size
        else:
            remainder = 0

        if remainder and self.drop_last:
            global_order = global_order[: total - remainder]
        elif remainder:
            target_size = total + (global_chunk_size - remainder)
            global_order = _build_prefix_padding(global_order, target_size)

        if global_order.size == 0:
            return np.empty(0, dtype=np.int64)

        rank_chunks: list[np.ndarray] = []
        for start in range(0, int(global_order.shape[0]), global_chunk_size):
            window = global_order[start : start + global_chunk_size]
            assignment = assign_window_to_ranks(
                window_indices=window,
                sample_costs=self.sample_costs,
                world_size=self.world_size,
                batch_size=self.batch_size,
            )
            rank_chunk = np.asarray(assignment[self.rank], dtype=np.int64)
            if rank_chunk.size > 0:
                rank_chunks.append(rank_chunk)

        if not rank_chunks:
            return np.empty(0, dtype=np.int64)
        return np.ascontiguousarray(np.concatenate(rank_chunks).astype(np.int64, copy=False))


class BalancedDistributedSampler(Sampler[int]):
    """Balance fixed-size rank batches with order strategies or direct LPT planning.

    Prefer BalancedBatchSampler with DataLoader(batch_sampler=...) to preserve batch
    boundaries automatically. With this index sampler, the DataLoader batch_size must
    match the sampler's batch_size. Initialize the DDP process group before constructing
    the sampler for automatic rank detection; detection is not repeated later.

    Args:
        sample_costs: One-dimensional finite, nonnegative additive costs, in dataset
            index order. All ranks must supply identical costs and planning settings.
        batch_size: Positive integer sample count per rank, not the global batch size.
        rank: Rank index. None reads initialized torch.distributed, otherwise uses 0.
            An explicit value must agree with an initialized process group.
        world_size: Number of ranks. None reads initialized torch.distributed,
            otherwise uses 1. An explicit value must agree with the process group.
        shuffle: Boolean. For LPT, shuffle steps and rank assignments each epoch,
            not batch membership or within-batch order. False retains deterministic
            balanced order, not dataset order. Call set_epoch on every rank each epoch.
        seed: Nonnegative integer random seed, shared across ranks. Booleans are invalid.
        drop_last: Boolean. Drop the remainder of a global batch when True. With LPT,
            False pads training data but rejects non-divisible input if shuffle=False.
            Dropped or repeated sample occurrences stay fixed across epochs.
        sample_order: Legacy V-only permutation used when shuffle=False. LPT rejects it.
        strategy: Planning tier: lpt_fast, lpt-medium (alias/default lpt), or lpt_best.
            Legacy v1 through v3 are deprecated and scheduled for removal in v1.4.0.

    Raises:
        TypeError: Boolean or integer settings have unsupported types.
        ValueError: Costs, settings, rank configuration, or parameter combinations
            are invalid, or accumulated batch costs are not finite.

    ``lpt_fast``, ``lpt-medium`` (alias ``lpt``) and ``lpt_best`` trade planning work for
    lexicographic
    base-group quality (peak batch cost, P99, squared loads); higher tiers retain lower
    base candidates and may tie. DDP tail repair is separate and does not guarantee
    this tier ordering for final padded/dropped plans. Fast adds a derived two-pointer
    pass; middle adds one bounded pair
    pass on the derived plan; best adds multiple passes and three-batch repair.
    Each step search protects peak, P99 and step-max sum. On divisible inputs higher
    tiers retain the lower derived candidate under the lexicographic step score.
    Base groups remain unchanged. All three cache final batch membership in memory. ``shuffle=True``
    shuffles steps and rank labels using seed + epoch, not samples within batches.
    Non-shuffled LPT is deterministic balanced order, not original dataset order.
    LPT does not support ``sample_order``. Shuffled tails are selected/padded once
    from a small repair pool using the base seed, or discarded with ``drop_last=True``;
    at most 2*world_size-1 base batches are regrouped by tail repair (derived step
    optimization can touch other batches in the derived plan). Changing epochs does
    not change those occurrences. Non-shuffled LPT rejects non-divisible input unless
    dropping is explicitly requested, preventing silent validation duplicates.
    This strategy is sampler-only; dataset ordering and LMDB artifacts are unchanged.
    Legacy sampler strategies ``v1`` through ``v3`` are deprecated and will be removed
    in v1.4.0. The default is ``lpt``, normalized internally to ``lpt-medium``.
    """
    def __init__(
        self,
        sample_costs: Sequence[float] | np.ndarray,
        *,
        batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sample_order: Sequence[int] | np.ndarray | None = None,
        strategy: str = "lpt",
    ) -> None:
        for name, value in (("shuffle", shuffle), ("drop_last", drop_last)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean, got {value!r}")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        costs = _normalize_sample_costs(sample_costs)
        strategy = _normalize_lpt_strategy(strategy)
        validated_strategy = strategy if strategy in _LPT_STRATEGIES else validate_balance_strategy(
            strategy
        )
        if validated_strategy in _LPT_STRATEGIES:
            for name, value in (("batch_size", batch_size), ("rank", rank),
                                ("world_size", world_size)):
                if value is not None and (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                ):
                    raise TypeError(f"{name} must be an integer, got {value!r}")
            # Cached grouping must not depend on subsequent caller-side array mutations.
            costs = costs.copy()
            costs.setflags(write=False)
        validated_order = None
        if sample_order is not None:
            validated_order = _validate_sample_order(sample_order, int(costs.shape[0]))
        resolved_rank, resolved_world_size = _resolve_rank_and_world_size(rank, world_size)
        self._plan_cache = _BalancedPlanCache(
            sample_costs=costs,
            batch_size=int(batch_size),
            rank=resolved_rank,
            world_size=resolved_world_size,
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=bool(drop_last),
            sample_order=validated_order,
            strategy=validated_strategy,
        )
        # QQTOOLS-COMPAT-0006: remove legacy sampler strategies and warning in v1.4.0.
        if validated_strategy not in _LPT_STRATEGIES:
            warnings.warn(
                f"Sampler strategy={validated_strategy!r} is deprecated and will be removed "
                "in v1.4.0. Switch to 'lpt_fast', 'lpt-medium' (alias 'lpt'), or 'lpt_best'. "
                "LPT changes sample grouping/order, does not accept sample_order, and requires "
                "divisible input or drop_last=True when shuffle=False.",
                FutureWarning,
                stacklevel=2,
            )

    def __iter__(self):
        return iter(self._plan_cache.rank_local_plan.tolist())

    def __len__(self) -> int:
        return int(self._plan_cache.rank_local_plan.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self._plan_cache.set_epoch(epoch)


class BalancedBatchSampler(BatchSampler):
    """Balance rank-local batches for DataLoader(dataset, batch_sampler=sampler).

    Do not also supply batch_size, shuffle, sampler, or drop_last to DataLoader.
    Initialize the DDP process group first for automatic rank detection. All ranks
    must use identical costs and planning settings and call set_epoch each epoch.

    Args:
        sample_costs: Finite, nonnegative additive costs in dataset index order.
        batch_size: Positive integer number of samples per rank-local batch.
        rank: Rank index; None reads initialized DDP, otherwise uses 0.
        world_size: Number of ranks; None reads initialized DDP, otherwise uses 1.
            Explicit rank/world_size must match initialized DDP. Detection occurs once.
        shuffle: Boolean. LPT shuffles steps and rank assignments, not batch membership
            or within-batch order. False means deterministic balanced order.
        seed: Nonnegative integer shared across ranks; booleans are invalid.
        drop_last: Boolean. Drop an incomplete global batch when True. Otherwise LPT
            pads with shuffle=True and rejects non-divisible input with shuffle=False.
            Dropped/repeated occurrences remain fixed across epochs for LPT.
        sample_order: Legacy V-only permutation for shuffle=False; invalid with LPT.
        strategy: lpt_fast, lpt-medium (alias/default lpt), or lpt_best. Deprecated
            v1 through v3 remain available until v1.4.0.

    Raises:
        TypeError: Boolean or integer settings have unsupported types.
        ValueError: Costs, settings, or combinations are invalid. See
            BalancedDistributedSampler for the full planning contract.
    """
    def __init__(
        self,
        sample_costs: Sequence[float] | np.ndarray,
        *,
        batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sample_order: Sequence[int] | np.ndarray | None = None,
        strategy: str = "lpt",
    ) -> None:
        self.sampler = BalancedDistributedSampler(
            sample_costs,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            sample_order=sample_order,
            strategy=strategy,
        )
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self._plan_cache = self.sampler._plan_cache

    def __iter__(self):
        plan = self.sampler._plan_cache.rank_local_plan
        for start in range(0, int(plan.shape[0]), self.batch_size):
            batch = plan[start : start + self.batch_size]
            if batch.shape[0] < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()

    def __len__(self) -> int:
        plan_len = len(self.sampler)
        full_batches, remainder = divmod(plan_len, self.batch_size)
        if remainder and not self.drop_last:
            return full_batches + 1
        return full_batches

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
