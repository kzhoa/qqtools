import copy

import pytest
import torch

from qqtools.torch.qdataset import qData


def test_qdata_init_conflict_raises():
    with pytest.raises(ValueError):
        qData({"x": 1}, x=2)


def test_qdata_float_and_double_only_change_floating_tensors():
    data = qData({"x": torch.tensor([1.0], dtype=torch.float32), "y": torch.tensor([1], dtype=torch.int64)})

    data.double()
    assert data["x"].dtype == torch.float64
    assert data["y"].dtype == torch.int64

    data.float()
    assert data["x"].dtype == torch.float32
    assert data["y"].dtype == torch.int64


def test_qdata_copy_with_nested_dict_is_shallow():
    data = qData({"meta": {"lvl1": {"vals": [1, 2]}}, "x": torch.tensor([1.0])})

    copied = copy.copy(data)

    assert copied is not data
    assert copied["meta"] is data["meta"]
    assert copied["x"] is data["x"]

    copied["meta"]["lvl1"]["vals"].append(3)
    assert data["meta"]["lvl1"]["vals"] == [1, 2, 3]

    assert "_allow_notexist" not in copied.keys()
    assert "_default_function" not in copied.keys()
    assert copied.__dict__["_allow_notexist"] is False
    assert "_default_function" in copied.__dict__


def test_qdata_deepcopy_with_nested_dict_is_independent():
    data = qData({"meta": {"lvl1": {"vals": [1, 2]}}, "x": torch.tensor([1.0])})

    copied = copy.deepcopy(data)

    assert copied is not data
    assert copied["meta"] is not data["meta"]
    assert copied["meta"]["lvl1"] is not data["meta"]["lvl1"]
    assert copied["meta"]["lvl1"]["vals"] is not data["meta"]["lvl1"]["vals"]
    assert copied["x"] is not data["x"]
    assert torch.equal(copied["x"], data["x"])

    copied["meta"]["lvl1"]["vals"].append(9)
    assert data["meta"]["lvl1"]["vals"] == [1, 2]
    assert copied["meta"]["lvl1"]["vals"] == [1, 2, 9]

    assert "_allow_notexist" not in copied.keys()
    assert "_default_function" not in copied.keys()
    assert copied.__dict__["_allow_notexist"] is False
    assert "_default_function" in copied.__dict__
