from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler

from qqtools.data.qbalance import (
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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self.rank_local_plan = self._build_rank_local_plan(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        if not self.shuffle:
            return
        epoch = int(epoch)
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.rank_local_plan = self._build_rank_local_plan(epoch)

    def _build_rank_local_plan(self, epoch: int) -> np.ndarray:
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
        strategy: str = "v3",
    ) -> None:
        costs = _normalize_sample_costs(sample_costs)
        validated_strategy = validate_balance_strategy(strategy)
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

    def __iter__(self):
        return iter(self._plan_cache.rank_local_plan.tolist())

    def __len__(self) -> int:
        return int(self._plan_cache.rank_local_plan.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self._plan_cache.set_epoch(epoch)


class BalancedBatchSampler(BatchSampler):
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
        strategy: str = "v3",
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
        self._plan_cache = _BalancedPlanCache(
            sample_costs=self.sampler._plan_cache.sample_costs,
            batch_size=self.batch_size,
            rank=self.sampler._plan_cache.rank,
            world_size=self.sampler._plan_cache.world_size,
            shuffle=self.sampler._plan_cache.shuffle,
            seed=self.sampler._plan_cache.seed,
            drop_last=self.drop_last,
            sample_order=self.sampler._plan_cache.sample_order,
            strategy=self.sampler._plan_cache.strategy,
            epoch=self.sampler._plan_cache.epoch,
            rank_local_plan=self.sampler._plan_cache.rank_local_plan.copy(),
        )

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
        self._plan_cache.epoch = self.sampler._plan_cache.epoch
        self._plan_cache.rank_local_plan = self.sampler._plan_cache.rank_local_plan.copy()
