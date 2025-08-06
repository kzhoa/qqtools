import numpy as np
import torch


def is_valid_shape(arr):
    shape = arr.shape
    return len(shape) == 1 or (len(shape) >= 2 and shape[1] == 1)


def confusion_matrix(y_pred, y_true):
    """
    example
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    """

    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    assert is_valid_shape(y_pred)
    assert is_valid_shape(y_true)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, fp, tn, fn
