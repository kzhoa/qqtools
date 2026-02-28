import numpy as np
import pytest
import torch

from qqtools.torch.qdataset import collate_dict_samples, qDictDataset


class DemoDataset(qDictDataset):
    pass


def test_collate_dict_samples_success_and_shape():
    batch = [
        {"x": torch.tensor([1.0, 2.0]), "label": 1},
        {"x": torch.tensor([3.0, 4.0]), "label": 0},
    ]

    merged = collate_dict_samples(batch)
    assert merged["x"].shape == (2, 2)
    assert torch.equal(merged["label"], torch.tensor([1, 0]))


def test_collate_dict_samples_inconsistent_keys_raises():
    batch = [{"x": torch.tensor([1])}, {"x": torch.tensor([2]), "y": torch.tensor([3])}]
    with pytest.raises(AssertionError):
        collate_dict_samples(batch)


def test_qdictdataset_index_select_and_getitem():
    dataset = DemoDataset(data_list=[{"v": 1}, {"v": 2}, {"v": 3}])

    subset = dataset.index_select([0, 2])
    assert len(subset) == 2
    assert subset[0] == {"v": 1}
    assert subset[1] == {"v": 3}

    bool_subset = dataset.index_select(torch.tensor([True, False, True]))
    assert len(bool_subset) == 2
    assert bool_subset[1] == {"v": 3}

    np_subset = dataset.index_select(np.array([1, 2], dtype=np.int64))
    assert len(np_subset) == 2
    assert np_subset[0] == {"v": 2}


def test_qdictdataset_get_norm_factor():
    dataset = DemoDataset(data_list=[{"target": torch.tensor(1.0)}, {"target": torch.tensor(3.0)}])
    mean, std = dataset.get_norm_factor("target")
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(torch.std(torch.tensor([1.0, 3.0])).item())
