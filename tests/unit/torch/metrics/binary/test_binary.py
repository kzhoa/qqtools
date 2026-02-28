import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from qqtools.torch.metrics.binary import binary_metrics, confusion_matrix, is_valid_shape


def test_is_valid_shape():
    assert is_valid_shape(np.array([0, 1, 1]))
    assert is_valid_shape(np.array([[0], [1], [1]]))
    assert not is_valid_shape(np.array([[0, 1], [1, 0]]))


def test_confusion_matrix_numpy_input():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    tp, fp, tn, fn = confusion_matrix(y_pred=y_pred, y_true=y_true)
    assert (tp, fp, tn, fn) == (3, 0, 2, 1)


def test_confusion_matrix_torch_input():
    y_true = torch.tensor([1, 0, 1, 1])
    y_pred = torch.tensor([1, 0, 0, 1])
    tp, fp, tn, fn = confusion_matrix(y_pred=y_pred, y_true=y_true)
    assert (tp, fp, tn, fn) == (2, 0, 1, 1)


def test_binary_metrics_basic_keys():
    preds = [0.9, 0.8, 0.2, 0.1]
    targets = [1, 1, 0, 0]
    result = binary_metrics(preds, targets)
    for key in ["auc", "aupr", "f1", "ppv", "recall", "fr100", "ttif20"]:
        assert key in result


def test_binary_metrics_constant_labels_return_negative_one_for_auc_family():
    preds = [0.9, 0.8, 0.2, 0.1]
    targets = [1, 1, 1, 1]
    result = binary_metrics(preds, targets, metrics=["auc", "aupr"])
    assert result["auc"] == -1.0
    assert result["aupr"] == -1.0


def test_binary_metrics_invalid_metric_raises():
    with pytest.raises(ValueError):
        binary_metrics([0.1, 0.9], [0, 1], metrics=["unknown_metric"])
