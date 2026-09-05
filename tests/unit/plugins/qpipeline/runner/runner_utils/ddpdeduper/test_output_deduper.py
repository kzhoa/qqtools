from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from qqtools.torch.ddp import BalancedDistributedSampler
from qqtools.plugins.qpipeline.runner.runner_utils.avgbank import AvgBank
from qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract import EvalBatch, EvalDedupRuntime
from qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper import (
    DDPOutputDeduper,
    prepare_eval_loader_for_ddp,
)
from qqtools.plugins.qpipeline.runner.runner_utils.tensorbank import TensorBank


class DictDataset(Dataset):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {"id": torch.tensor([self.values[idx]]), "y": torch.tensor([self.values[idx]])}


class RawDataset(Dataset):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


def _collate_as_list(batch):
    return list(batch)


def _to_batches(indices, batch_size):
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def test_prepare_eval_loader_rejects_unequal_rank_batch_counts(monkeypatch):
    loader = DataLoader(DictDataset([0, 1, 2]), batch_size=2, shuffle=False, collate_fn=lambda batch: batch)
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        lambda value: [value, value - 1],
    )

    with pytest.raises(RuntimeError, match=r"stage=val.*counts=\{0: 2, 1: 1\}"):
        prepare_eval_loader_for_ddp(
            loader,
            enabled=False,
            stage="val",
            loader_name="val_id",
            distributed=True,
        )


def _mock_gather(monkeypatch, gathered_batches, gathered_real_ids):
    def _fake_gather(value):
        if value and isinstance(value[0], list):
            return gathered_batches
        return gathered_real_ids

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        _fake_gather,
    )


def test_deduper_sampler_path_marks_all_duplicate_tail_batch(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 2,
    )
    _mock_gather(monkeypatch, [[[0, 2]], [[1, 0]]], [[0, 2], [1]])

    loader = DataLoader(DictDataset([0, 1, 2]), batch_size=2, shuffle=False, collate_fn=lambda batch: batch)
    wrapped = DDPOutputDeduper(loader).wrap()

    batches = list(iter(wrapped))
    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch, EvalBatch)
    assert batch.control.all_duplicate is False
    assert [item["id"].item() for item in batch.payload] == [1]


def test_deduper_sampler_path_preserves_all_duplicate_step(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 2,
    )
    _mock_gather(monkeypatch, [[[0, 2], [4]], [[1, 3], [0]]], [[0, 2, 4], [1, 3]])

    loader = DataLoader(DictDataset([0, 1, 2, 3, 4]), batch_size=2, shuffle=False, collate_fn=lambda batch: batch)
    wrapped = DDPOutputDeduper(loader).wrap()

    batches = list(iter(wrapped))
    assert len(batches) == 2
    assert batches[0].control.all_duplicate is False
    assert [item["id"].item() for item in batches[0].payload] == [1, 3]
    assert batches[1].control.all_duplicate is True
    assert [item["id"].item() for item in batches[1].payload] == [0]


@dataclass
class ManualBatchSampler:
    batches: list[list[int]]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_deduper_batch_sampler_fallback(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 2,
    )
    _mock_gather(monkeypatch, [[[0, 2]], [[1, 0]]], [[0, 2], [1]])

    loader = DataLoader(
        RawDataset([10, 11, 12]),
        batch_sampler=ManualBatchSampler([[0, 2]]),
        collate_fn=lambda batch: list(batch),
    )
    wrapped = DDPOutputDeduper(loader).wrap()
    batches = list(iter(wrapped))

    assert batches[0].control.all_duplicate is False
    assert batches[0].payload == [10, 12]


def test_deduper_collects_worker_real_ids_in_main_process(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 2,
    )
    sample_costs = [5.0, 4.0, 3.0, 2.0, 1.0]
    local_indices = list(
        BalancedDistributedSampler(
            sample_costs,
            batch_size=2,
            rank=0,
            world_size=2,
            shuffle=False,
            strategy="v3",
        )
    )
    remote_indices = list(
        BalancedDistributedSampler(
            sample_costs,
            batch_size=2,
            rank=1,
            world_size=2,
            shuffle=False,
            strategy="v3",
        )
    )
    local_batches = _to_batches(local_indices, batch_size=2)
    remote_batches = _to_batches(remote_indices, batch_size=2)
    remote_real_ids = []
    seen_ids = set()
    for step_idx in range(max(len(local_batches), len(remote_batches))):
        for rank_idx, rank_batches in enumerate((local_batches, remote_batches)):
            if step_idx >= len(rank_batches):
                continue
            for logical_id in rank_batches[step_idx]:
                is_real = logical_id not in seen_ids
                seen_ids.add(logical_id)
                if rank_idx == 1 and is_real:
                    remote_real_ids.append(logical_id)

    collected_ids_by_main_process = []

    def _fake_gather(value):
        if value and isinstance(value[0], list):
            return [local_batches, remote_batches]
        collected_ids_by_main_process.append(list(value))
        return [list(value), remote_real_ids]

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        _fake_gather,
    )
    sampler = BalancedDistributedSampler(
        sample_costs,
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
        strategy="v3",
    )
    loader = DataLoader(
        RawDataset(range(len(sample_costs))),
        batch_size=2,
        sampler=sampler,
        collate_fn=_collate_as_list,
        num_workers=1,
    )

    batches = list(DDPOutputDeduper(loader).wrap())

    local_real_ids = []
    for batch in batches:
        local_real_ids.extend(batch.control.real_logical_sample_ids)
    assert collected_ids_by_main_process == [local_real_ids]
    assert set(local_real_ids).isdisjoint(remote_real_ids)
    assert set(local_real_ids).union(remote_real_ids) == set(range(len(sample_costs)))


def test_deduper_skips_completion_assertion_after_early_exit(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 1,
    )
    gather_calls = []

    def _fake_gather(value):
        gather_calls.append(value)
        return [value]

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        _fake_gather,
    )
    loader = DataLoader(RawDataset([0, 1, 2]), batch_size=1, collate_fn=_collate_as_list)
    wrapped = DDPOutputDeduper(loader).wrap()

    iterator = iter(wrapped)
    next(iterator)
    iterator.close()

    assert len(gather_calls) == 1


def test_deduper_repeated_iteration_keeps_audit_state_isolated(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 1,
    )
    completion_records = []

    def _fake_gather(value):
        if value and isinstance(value[0], list):
            return [value]
        completion_records.append(list(value))
        return [value]

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        _fake_gather,
    )
    loader = DataLoader(RawDataset([0, 1, 2]), batch_size=2, collate_fn=_collate_as_list)
    wrapped = DDPOutputDeduper(loader).wrap()

    list(wrapped)
    list(wrapped)

    assert completion_records == [[0, 1, 2], [0, 1, 2]]


def test_deduper_persistent_workers_keep_audit_state_isolated(monkeypatch):
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper.qt.qdist.get_world_size",
        lambda: 1,
    )
    completion_records = []

    def _fake_gather(value):
        if value and isinstance(value[0], list):
            return [value]
        completion_records.append(list(value))
        return [value]

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.output_deduper._all_gather_object",
        _fake_gather,
    )
    wrapped = DDPOutputDeduper(
        DataLoader(
            RawDataset([0, 1, 2]),
            batch_size=2,
            collate_fn=_collate_as_list,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    ).wrap()

    list(wrapped)
    list(wrapped)

    assert completion_records == [[0, 1, 2], [0, 1, 2]]


def test_deduper_rejects_iterable_dataset(monkeypatch):
    class IterableValues(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from range(3)

    loader = DataLoader(IterableValues(), batch_size=2)
    with pytest.raises(TypeError):
        DDPOutputDeduper(loader)


def test_runtime_gather_avg_bank_contributes_zero_for_missing_rank(monkeypatch):
    avg_bank = AvgBank()
    avg_bank.update_from_dict({"mse": (torch.tensor(2.0), 2)})
    runtime = EvalDedupRuntime(enabled=True)

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.is_dist_available_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract._all_gather_object",
        lambda value: [["mse"], []],
    )

    def _fake_all_reduce(tensor, op):
        del op
        tensor[0] = 2.0
        tensor[1] = 2.0

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.dist.all_reduce",
        _fake_all_reduce,
    )

    gathered = runtime.gather_avg_bank(avg_bank, distributed=True, device=torch.device("cpu"))
    assert gathered == {"mse": 1.0}


def test_runtime_gather_tensor_bank_uses_empty_tensor_for_missing_rank(monkeypatch):
    tensor_bank = TensorBank()
    tensor_bank.add({"pred": torch.tensor([[1.0], [2.0]])})
    runtime = EvalDedupRuntime(enabled=True)

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.is_dist_available_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract._all_gather_object",
        lambda value: [value, {}],
    )

    seen = {}

    def _fake_all_gather_tensor(tensor, device):
        del device
        seen["shape"] = tuple(tensor.shape)
        return tensor

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.all_gather_tensor",
        _fake_all_gather_tensor,
    )

    gathered = runtime.gather_tensor_bank(tensor_bank, distributed=True, device=torch.device("cpu"))
    assert seen["shape"] == (2, 1)
    assert tuple(gathered["pred"].shape) == (2, 1)
    assert tensor_bank.bank == {}


def test_runtime_gather_output_bank_returns_none_for_nonzero_rank(monkeypatch):
    output_bank = TensorBank()
    output_bank.add({"pred": torch.tensor([[1.0]])})
    runtime = EvalDedupRuntime(enabled=True)

    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.is_dist_available_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract._all_gather_object",
        lambda value: [value],
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.all_gather_tensor",
        lambda tensor, device: tensor,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qpipeline.runner.runner_utils.ddpdeduper.eval_contract.qt.qdist.get_rank",
        lambda: 1,
    )

    gathered = runtime.gather_output_bank(output_bank, distributed=True, device=torch.device("cpu"))
    assert gathered is None
