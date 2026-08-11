import multiprocessing
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

import qqtools.torch.qdataset as qdataset
from qqtools.torch.qdataset import qDictDataloader, qDictDataset


class DemoDataset(qDictDataset):
    pass


def _module_level_collate(batch_list):
    return batch_list


def test_qdictdataloader_default_collate_non_graph():
    dataset = DemoDataset(
        data_list=[
            {"x": torch.tensor([1.0, 2.0]), "label": 1},
            {"x": torch.tensor([3.0, 4.0]), "label": 0},
        ]
    )

    loader = qDictDataloader(dataset=dataset, batch_size=2, is_graph=False)
    assert loader.is_graph is False
    batch = next(iter(loader))

    assert batch["x"].shape == (2, 2)
    assert torch.equal(batch["label"], torch.tensor([1, 0]))


def test_qdictdataloader_custom_collate_takes_priority():
    dataset = DemoDataset(data_list=[{"x": 1}, {"x": 2}])

    def custom_collate(batch_list):
        return {"size": len(batch_list)}

    loader = qDictDataloader(dataset=dataset, batch_size=2, is_graph=True, collate_fn=custom_collate)
    assert loader.is_graph is True
    batch = next(iter(loader))
    assert batch == {"size": 2}


def test_qdictdataloader_graph_collate_uses_cached_key_types(monkeypatch):
    calls = {"determine": 0, "collate": 0}
    key_types = {"node": {"x"}, "edge": set(), "graph": set()}

    def fake_determine(batch_list):
        calls["determine"] += 1
        return key_types

    def fake_collate(batch_list, key_types=None):
        calls["collate"] += 1
        assert key_types is not None
        assert key_types == {"node": {"x"}, "edge": set(), "graph": set()}
        return {"batch_size": len(batch_list)}

    monkeypatch.setattr(qdataset, "determine_graph_key_types", fake_determine)
    monkeypatch.setattr(qdataset, "collate_graph_samples", fake_collate)

    samples = [
        {"x": torch.randn(2, 1), "num_nodes": 2},
        {"x": torch.randn(3, 1), "num_nodes": 3},
        {"x": torch.randn(1, 1), "num_nodes": 1},
    ]
    dataset = DemoDataset(data_list=samples)

    loader = qDictDataloader(dataset=dataset, batch_size=2, is_graph=True)
    batches = list(loader)

    assert len(batches) == 2
    assert calls["determine"] == 1
    assert calls["collate"] == 2


def test_qdictdataloader_graph_collator_is_pickle_safe():
    dataset = DemoDataset(data_list=[])
    loader = qDictDataloader(dataset=dataset, batch_size=1, is_graph=True)

    restored = pickle.loads(pickle.dumps(loader.collate_fn))

    assert isinstance(restored, qdataset._StatefulGraphCollator)


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_qdictdataloader_rejects_unpickleable_collate_for_pickle_based_worker(
    start_method,
):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")

    with pytest.raises(TypeError, match="module-level function"):
        qDictDataloader(
            dataset=DemoDataset(data_list=[]),
            batch_size=1,
            collate_fn=lambda batch: batch,
            num_workers=1,
            multiprocessing_context=start_method,
        )


def test_qdictdataloader_allows_lambda_without_pickle_based_worker():
    dataset = DemoDataset(data_list=[{"x": 1}])

    single_process_loader = qDictDataloader(
        dataset=dataset,
        batch_size=1,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )
    assert next(iter(single_process_loader)) == [{"x": 1}]

    if "fork" in multiprocessing.get_all_start_methods():
        fork_loader = qDictDataloader(
            dataset=dataset,
            batch_size=1,
            collate_fn=lambda batch: batch,
            num_workers=1,
            multiprocessing_context="fork",
        )
        assert fork_loader.collate_fn is not None


def test_qdictdataloader_accepts_pickleable_collate_for_spawn_worker():
    loader = qDictDataloader(
        dataset=DemoDataset(data_list=[]),
        batch_size=1,
        collate_fn=_module_level_collate,
        num_workers=1,
        multiprocessing_context="spawn",
    )

    assert loader.collate_fn is _module_level_collate


def test_qdictdataloader_supports_explicit_batch_sampler():
    dataset = DemoDataset(data_list=[{"x": 1}, {"x": 2}, {"x": 3}])
    batch_sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=2,
        drop_last=False,
    )

    loader = qDictDataloader(
        dataset=dataset,
        batch_size=None,
        batch_sampler=batch_sampler,
    )

    assert [batch["x"].tolist() for batch in loader] == [[1, 2], [3]]


@pytest.mark.parametrize(
    "loader_kwargs, message",
    [
        ({"batch_size": 2}, "batch_size"),
        ({"batch_size": None, "shuffle": True}, "shuffle"),
        ({"batch_size": None, "drop_last": True}, "drop_last"),
    ],
)
def test_qdictdataloader_rejects_batch_sampler_conflicts(loader_kwargs, message):
    dataset = DemoDataset(data_list=[{"x": 1}])
    batch_sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=1,
        drop_last=False,
    )

    with pytest.raises(ValueError, match=message):
        qDictDataloader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_qdictdataloader_graph_collate_with_pickle_based_worker(
    start_method,
    checkout_subprocess_env,
):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")

    probe_path = (
        Path(__file__).parents[3] / "fixtures" / "qdataset_graph_worker_probe.py"
    )
    worker_env = checkout_subprocess_env
    if os.name != "nt":
        worker_env.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "TMP": "/tmp"})

    subprocess.run(
        [sys.executable, str(probe_path), start_method],
        check=True,
        env=worker_env,
        timeout=60,
    )
