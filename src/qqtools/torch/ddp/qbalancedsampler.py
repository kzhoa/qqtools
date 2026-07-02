from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler

from qqtools.data.qbalance import compute_global_even_sort_order


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


def assign_chunk_lpt(
    chunk_indices: Sequence[int] | np.ndarray,
    sample_costs: Sequence[float] | np.ndarray,
    world_size: int,
    batch_size: int,
) -> list[list[int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    costs = _normalize_sample_costs(sample_costs)
    chunk = np.asarray(chunk_indices, dtype=np.int64)
    if chunk.ndim != 1:
        raise ValueError(f"chunk_indices must be 1D, got shape {chunk.shape}")
    if len(chunk) == 0:
        return [[] for _ in range(world_size)]
    if np.any(chunk < 0) or np.any(chunk >= costs.shape[0]):
        raise ValueError("chunk_indices contains out-of-range sample indices")

    sorted_positions = np.argsort(-costs[chunk], kind="stable")
    rank_to_items: list[list[int]] = [[] for _ in range(world_size)]
    rank_loads = np.zeros(world_size, dtype=np.float64)
    rank_counts = np.zeros(world_size, dtype=np.int64)

    def _locality_cost(rank_idx: int, sample_idx: int) -> float:
        if not rank_to_items[rank_idx]:
            return float("inf")
        return float(min(abs(sample_idx - existing_idx) for existing_idx in rank_to_items[rank_idx]))

    def _choose_rank(sample_idx: int, candidate_ranks: Iterable[int]) -> int:
        return min(
            candidate_ranks,
            key=lambda rank_idx: (
                float(rank_loads[rank_idx]),
                int(rank_counts[rank_idx]),
                _locality_cost(rank_idx, sample_idx),
                int(rank_idx),
            ),
        )

    for position in sorted_positions:
        sample_idx = int(chunk[position])
        chosen_rank = _choose_rank(sample_idx, range(world_size))
        if rank_counts[chosen_rank] >= batch_size:
            available_ranks = [
                rank_idx for rank_idx in range(world_size) if rank_counts[rank_idx] < batch_size
            ]
            if not available_ranks:
                raise RuntimeError("No available rank left while assigning chunk")
            chosen_rank = _choose_rank(sample_idx, available_ranks)

        rank_to_items[chosen_rank].append(sample_idx)
        rank_loads[chosen_rank] += costs[sample_idx]
        rank_counts[chosen_rank] += 1

    return rank_to_items


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
            chunk = global_order[start : start + global_chunk_size]
            assignment = assign_chunk_lpt(
                chunk_indices=chunk,
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
            strategy=strategy,
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
        costs = _normalize_sample_costs(sample_costs)
        validated_order = None
        if sample_order is not None:
            validated_order = _validate_sample_order(sample_order, int(costs.shape[0]))
        resolved_rank, resolved_world_size = _resolve_rank_and_world_size(rank, world_size)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self._plan_cache = _BalancedPlanCache(
            sample_costs=costs,
            batch_size=self.batch_size,
            rank=resolved_rank,
            world_size=resolved_world_size,
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=self.drop_last,
            sample_order=validated_order,
            strategy=strategy,
        )

    def __iter__(self):
        plan = self._plan_cache.rank_local_plan
        for start in range(0, int(plan.shape[0]), self.batch_size):
            batch = plan[start : start + self.batch_size]
            if batch.shape[0] < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()

    def __len__(self) -> int:
        plan_len = int(self._plan_cache.rank_local_plan.shape[0])
        full_batches, remainder = divmod(plan_len, self.batch_size)
        if remainder and not self.drop_last:
            return full_batches + 1
        return full_batches

    def set_epoch(self, epoch: int) -> None:
        self._plan_cache.set_epoch(epoch)
