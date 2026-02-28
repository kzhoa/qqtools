import pytest
import torch

from qqtools.torch.nn import functional as qfn


def test_l1_loss_ddp_mean_with_batch_size(monkeypatch):
    monkeypatch.setattr(qfn.qt.qdist, "all_reduce", lambda value, **kwargs: value)
    monkeypatch.setattr(qfn.qt.qdist, "get_world_size", lambda: 1)

    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    result = qfn.l1_loss_ddp(input_tensor, target_tensor, reduction="mean", batch_size=2)
    expected = torch.nn.functional.l1_loss(input_tensor, target_tensor) * 1 / 2
    assert torch.isclose(result, expected)


def test_l1_loss_ddp_sum_returns_raw_l1():
    input_tensor = torch.tensor([1.0, 3.0])
    target_tensor = torch.tensor([0.0, 1.0])

    result = qfn.l1_loss_ddp(input_tensor, target_tensor, reduction="sum")
    expected = torch.nn.functional.l1_loss(input_tensor, target_tensor)
    assert torch.isclose(result, expected)


def test_l1_loss_ddp_mean_asserts_for_high_dim(monkeypatch):
    monkeypatch.setattr(qfn.qt.qdist, "all_reduce", lambda value, **kwargs: value)
    monkeypatch.setattr(qfn.qt.qdist, "get_world_size", lambda: 1)

    input_tensor = torch.zeros(2, 2, 2)
    target_tensor = torch.zeros(2, 2, 2)

    with pytest.raises(AssertionError):
        qfn.l1_loss_ddp(input_tensor, target_tensor, reduction="mean", batch_size=None)
