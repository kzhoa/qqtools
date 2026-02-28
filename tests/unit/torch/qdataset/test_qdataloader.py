import torch

import qqtools.torch.qdataset as qdataset
from qqtools.torch.qdataset import qDictDataloader, qDictDataset


class DemoDataset(qDictDataset):
    pass


def test_qdictdataloader_default_collate_non_graph():
    dataset = DemoDataset(
        data_list=[
            {"x": torch.tensor([1.0, 2.0]), "label": 1},
            {"x": torch.tensor([3.0, 4.0]), "label": 0},
        ]
    )

    loader = qDictDataloader(dataset=dataset, batch_size=2, is_graph=False)
    batch = next(iter(loader))

    assert batch["x"].shape == (2, 2)
    assert torch.equal(batch["label"], torch.tensor([1, 0]))


def test_qdictdataloader_custom_collate_takes_priority():
    dataset = DemoDataset(data_list=[{"x": 1}, {"x": 2}])

    def custom_collate(batch_list):
        return {"size": len(batch_list)}

    loader = qDictDataloader(dataset=dataset, batch_size=2, is_graph=True, collate_fn=custom_collate)
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
