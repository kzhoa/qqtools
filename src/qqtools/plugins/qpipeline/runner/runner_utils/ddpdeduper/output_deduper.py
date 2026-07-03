from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from torch.utils.data import DataLoader, IterableDataset, Sampler
from torch.utils.data.sampler import BatchSampler

import qqtools as qt

from .eval_contract import EvalBatch, EvalBatchControl, EvalDedupRuntime, _all_gather_object

__all__ = ["DDPOutputDeduper", "wrap_eval_dataloader_for_ddp_dedup"]


@dataclass(frozen=True)
class OccurrenceKey:
    slot: int
    logical_sample_id: Any


@dataclass(frozen=True)
class _SampleEnvelope:
    payload: Any
    occurrence: OccurrenceKey
    is_real: bool


class _WrappedDataset:
    def __init__(self, dataset: Any, slot_is_real: dict[int, bool]):
        self._dataset = dataset
        self._slot_is_real = slot_is_real

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, occurrence_key: OccurrenceKey) -> _SampleEnvelope:
        if not isinstance(occurrence_key, OccurrenceKey):
            raise TypeError(
                "DDPOutputDeduper wrapped datasets require OccurrenceKey indices. "
                f"Got {type(occurrence_key).__name__}."
            )
        payload = self._dataset[occurrence_key.logical_sample_id]
        return _SampleEnvelope(
            payload=payload,
            occurrence=occurrence_key,
            is_real=bool(self._slot_is_real[occurrence_key.slot]),
        )


class _OccurrenceBatchSampler(Sampler[list[OccurrenceKey]]):
    def __init__(self, occurrence_batches: Sequence[Sequence[OccurrenceKey]]):
        self._occurrence_batches = [list(batch) for batch in occurrence_batches]

    def __iter__(self) -> Iterator[list[OccurrenceKey]]:
        for batch in self._occurrence_batches:
            yield list(batch)

    def __len__(self) -> int:
        return len(self._occurrence_batches)


class _DedupCollate:
    def __init__(self, collate_fn, deduper: "DDPOutputDeduper"):
        self._collate_fn = collate_fn
        self._deduper = deduper

    def __call__(self, batch_list: Sequence[_SampleEnvelope]) -> EvalBatch:
        if not batch_list:
            raise ValueError("DDPOutputDeduper collate received an empty batch_list.")

        real_items = [item for item in batch_list if item.is_real]
        is_all_duplicate = len(real_items) == 0
        collate_source = [item.payload for item in batch_list] if is_all_duplicate else [item.payload for item in real_items]
        payload = self._collate_fn(collate_source)
        real_logical_ids = tuple(item.occurrence.logical_sample_id for item in real_items)
        self._deduper.record_real_logical_ids(real_logical_ids)
        return EvalBatch(
            payload=payload,
            control=EvalBatchControl(
                all_duplicate=is_all_duplicate,
                real_logical_sample_ids=real_logical_ids,
            ),
        )


class _DedupedDataLoader:
    def __init__(self, loader: DataLoader, deduper: "DDPOutputDeduper", runtime: EvalDedupRuntime):
        self._loader = loader
        self._deduper = deduper
        self.dedup_runtime = runtime

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        completed_naturally = False
        self._deduper.reset_iteration_state()
        try:
            for batch in self._loader:
                yield batch
            completed_naturally = True
        finally:
            if completed_naturally:
                self._deduper.assert_natural_completion()


class DDPOutputDeduper:
    """
    DDP eval/infer loader wrapper for sampler-padding dedup only.

    It preserves forward step counts while preventing duplicated logical samples introduced by
    DDP sampler padding from reaching metric/output collection.

    This is not a general-purpose deduper:
        - intended for duplicated samples injected only to equalize per-rank eval/infer steps
        - does not attempt to preserve intentional duplicate sampling semantics
        - should not be treated as a generic repeated-sample filter for arbitrary samplers
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        is_graph: bool = False,
        node_aligned_output_keys: Optional[Sequence[str]] = None,
    ):
        if isinstance(loader.dataset, IterableDataset):
            raise TypeError("DDPOutputDeduper does not support IterableDataset.")
        self.loader = loader
        self.is_graph = bool(is_graph)
        self.node_aligned_output_keys = tuple(node_aligned_output_keys or ())
        self.rank = qt.qdist.get_rank()
        self.world_size = qt.qdist.get_world_size()

        self._collected_real_logical_ids: list[Any] = []
        self._target_logical_sample_ids: set[Any] = set()

    def wrap(self):
        local_batches = self._build_local_batch_plan()
        occurrence_batches, slot_is_real, target_ids = self._build_occurrence_plan(local_batches)
        self._target_logical_sample_ids = target_ids

        wrapped_dataset = _WrappedDataset(self.loader.dataset, slot_is_real=slot_is_real)
        wrapped_collate = _DedupCollate(self.loader.collate_fn, deduper=self)
        wrapped_loader = DataLoader(
            dataset=wrapped_dataset,
            batch_sampler=_OccurrenceBatchSampler(occurrence_batches),
            collate_fn=wrapped_collate,
            num_workers=self.loader.num_workers,
            pin_memory=self.loader.pin_memory,
            timeout=self.loader.timeout,
            worker_init_fn=self.loader.worker_init_fn,
            multiprocessing_context=self.loader.multiprocessing_context,
            generator=self.loader.generator,
            prefetch_factor=self.loader.prefetch_factor,
            persistent_workers=self.loader.persistent_workers,
            pin_memory_device=self.loader.pin_memory_device,
            in_order=getattr(self.loader, "in_order", True),
        )

        runtime = EvalDedupRuntime(
            enabled=True,
            is_graph=self.is_graph,
            node_aligned_output_keys=self.node_aligned_output_keys,
            expected_real_sample_count=len(target_ids),
        )
        return _DedupedDataLoader(wrapped_loader, deduper=self, runtime=runtime)

    def reset_iteration_state(self) -> None:
        self._collected_real_logical_ids = []

    def record_real_logical_ids(self, logical_ids: Sequence[Any]) -> None:
        self._collected_real_logical_ids.extend(logical_ids)

    def assert_natural_completion(self) -> None:
        gathered_real_ids = _all_gather_object(list(self._collected_real_logical_ids))
        flattened = [logical_id for rank_ids in gathered_real_ids for logical_id in rank_ids]
        if len(flattened) != len(self._target_logical_sample_ids):
            raise AssertionError(
                "DDPOutputDeduper natural-completion assertion failed: "
                f"expected {len(self._target_logical_sample_ids)} real samples, got {len(flattened)}."
            )
        if set(flattened) != self._target_logical_sample_ids:
            raise AssertionError(
                "DDPOutputDeduper natural-completion assertion failed: "
                "collected real logical sample ids do not match the synchronized iteration plan."
            )

    def _build_local_batch_plan(self) -> list[list[Any]]:
        if self._can_use_sampler_path():
            local_indices = list(iter(self.loader.sampler))
            batch_size = self.loader.batch_size
            if batch_size is None:
                raise ValueError("Sampler path requires loader.batch_size to be set.")
            batches = [
                local_indices[start : start + batch_size]
                for start in range(0, len(local_indices), batch_size)
            ]
            if self.loader.drop_last and batches and len(batches[-1]) < batch_size:
                batches = batches[:-1]
            return batches

        batch_sampler = getattr(self.loader, "batch_sampler", None)
        if batch_sampler is None:
            raise ValueError("DDPOutputDeduper requires an observable sampler or batch_sampler.")
        return [list(batch) for batch in batch_sampler]

    def _can_use_sampler_path(self) -> bool:
        sampler = getattr(self.loader, "sampler", None)
        batch_sampler = getattr(self.loader, "batch_sampler", None)
        if sampler is None or batch_sampler is None:
            return False
        if self.loader.batch_size is None:
            return False
        return (
            isinstance(batch_sampler, BatchSampler)
            and getattr(batch_sampler, "sampler", None) is sampler
            and getattr(batch_sampler, "batch_size", None) == self.loader.batch_size
            and getattr(batch_sampler, "drop_last", None) == self.loader.drop_last
        )

    def _build_occurrence_plan(
        self,
        local_batches: Sequence[Sequence[Any]],
    ) -> tuple[list[list[OccurrenceKey]], dict[int, bool], set[Any]]:
        gathered_batches = _all_gather_object([list(batch) for batch in local_batches])
        max_steps = max((len(rank_batches) for rank_batches in gathered_batches), default=0)
        global_seen: set[Any] = set()
        local_real_slots: dict[int, bool] = {}
        local_occurrence_batches: list[list[OccurrenceKey]] = []
        local_slot = 0

        target_ids = {
            logical_id
            for rank_batches in gathered_batches
            for batch in rank_batches
            for logical_id in batch
        }

        for step_idx in range(max_steps):
            step_occurrence_batch: Optional[list[OccurrenceKey]] = None
            for rank_idx, rank_batches in enumerate(gathered_batches):
                if step_idx >= len(rank_batches):
                    continue
                batch = rank_batches[step_idx]
                for logical_id in batch:
                    is_real = logical_id not in global_seen
                    global_seen.add(logical_id)
                    if rank_idx != self.rank:
                        continue
                    occurrence_key = OccurrenceKey(slot=local_slot, logical_sample_id=logical_id)
                    if step_occurrence_batch is None:
                        step_occurrence_batch = []
                    step_occurrence_batch.append(occurrence_key)
                    local_real_slots[local_slot] = is_real
                    local_slot += 1
            if step_occurrence_batch is not None:
                local_occurrence_batches.append(step_occurrence_batch)

        return local_occurrence_batches, local_real_slots, target_ids


def wrap_eval_dataloader_for_ddp_dedup(
    loader: DataLoader,
    *,
    enabled: bool,
    is_graph: bool = False,
    node_aligned_output_keys: Optional[Sequence[str]] = None,
):
    if not enabled:
        return loader
    if getattr(loader, "dedup_runtime", None) is not None:
        return loader
    return DDPOutputDeduper(
        loader,
        is_graph=is_graph,
        node_aligned_output_keys=node_aligned_output_keys,
    ).wrap()
