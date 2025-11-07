from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import qqtools as qt

from ...utils.check import check_values_allowed, is_alias_exists


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


def binary_metrics(
    preds: List[float],
    targets: List[int],
    threshold: float = 0.5,
    metrics: Optional[List[str]] = None,
    top_n: Optional[int] = None,
) -> qt.qDict:
    """
    Note:
    - Inputs will be automatically flattened if they have redundant single dimensions.
    - Returns -1.0 for metrics when labels are all 0 or all 1 (degenerate case).

    Args:
        preds (List[float]): A 1D vector of predicted probabilities, each value in the range [0, 1].
        targets (List[int]): A 1D vector of true labels, each value in the set {0, 1}.
        threshold (float) : Probability threshold for binary classification. Defaults to 0.5.
        metrics (List[str] | None) : List of metric names to compute. Supported metrics:
            - "auc", "auc-roc", "aucroc": Area under ROC curve
            - "aupr", "au-pr", "auc-pr", "aucpr": Area under Precision-Recall curve
            - "f1": F1 score
            - "ppv", "precision": Positive predictive value
            - "recall", "tpr": True positive rate (sensitivity)
            - "fr100": Fractional Rank of relative samples at top 100 preds
            - "ttif20": Time to identify first 20 true positives
            - If metrics is None, returns a comprehensive set of all above metrics.
        top_n (Optional[int]): If specified, evaluates metrics only on the top N predictions by probability.
               Useful for ranking metrics. If None, uses all predictions.

    Returns:
        dict: A dictionary containing the calculated metrics based on the provided input.
        Returns -1.0 for metrics that cannot be computed in degenerate cases (e.g., all negative labels).
    """

    def crop_topn(pds, gts, n):
        assert len(pds) == len(gts)
        if len(pds) < n:
            return pds, gts
        top_indices = np.argsort(pds)[-n:]
        pds = pds[top_indices]
        gts = gts[top_indices]
        return pds, gts

    WHOLE_METRICS = [
        "auc",  # AUC-ROC
        "aupr",  # AU-PR
        "f1",  # F1-score
        "ppv",  # PPV_n
        "recall",
        "fr100",
        "ttif20",
    ]

    if metrics is None:
        metrics = WHOLE_METRICS
    else:
        if isinstance(metrics, str) and "," in metrics:
            metrics = metrics.strip().split(",")  # try split
            metrics = [name.strip() for name in metrics]
        metrics = [name.lower() for name in metrics]
        check_values_allowed(metrics, WHOLE_METRICS)

    # basic check
    inpt_preds: np.ndarray = qt.ensure_numpy(preds).flatten()
    inpt_targets: np.ndarray = qt.ensure_numpy(targets).flatten()
    num_pos: int = np.sum(inpt_targets == 1, dtype=np.int64).item()
    num_neg: int = np.sum(inpt_targets == 0, dtype=np.int64).item()
    assert len(inpt_preds) == len(inpt_targets)
    assert num_pos + num_neg == len(inpt_targets), "labels must be 0 or 1"
    has_constant_values = (num_pos == 0) or (num_neg == 0)

    if top_n is not None and top_n < len(inpt_preds):
        cropped_preds, cropped_targets = crop_topn(inpt_preds, inpt_targets, top_n)
    else:
        cropped_preds, cropped_targets = inpt_preds, inpt_targets

    res = qt.qDict()
    if is_alias_exists(["auc", "auc-roc", "aucroc"], metrics):
        if has_constant_values:
            auc = -1.0  # return 1.0 instead of warning
        else:
            auc = roc_auc_score(cropped_targets, cropped_preds)
        res["auc"] = auc

    if is_alias_exists(["aupr", "au-pr", "auc-pr", "aucpr"], metrics):
        if has_constant_values:
            aupr = -1.0  # return 1.0 instead of warning
        else:
            aupr = average_precision_score(cropped_targets, cropped_preds)
        res["aupr"] = aupr

    # pred label - related
    pred_labels = cropped_preds > threshold
    if is_alias_exists(["ppv", "prec", "precision"], metrics):
        # return tp / (tp + fp)
        if num_pos == 0:
            ppv = -1.0
        else:
            ppv = precision_score(cropped_targets, pred_labels, zero_division=0.0)
        res["ppv"] = ppv

    if is_alias_exists(["recall", "tpr"], metrics):
        # return tp / (tp + fn)
        if num_pos == 0:
            recall = -1.0
        else:
            recall = recall_score(cropped_targets, pred_labels)
        res["recall"] = recall

    if is_alias_exists(["f1", "f1_score", "f1score"], metrics):
        if num_pos == 0:
            f1 = -1.0
        else:
            f1 = f1_score(cropped_targets, pred_labels)
        res["f1"] = f1

    # crop independently
    if "fr100" in metrics:
        # Fractional Rank
        if num_pos == 0:
            average_fr = -1.0
        else:
            top_limit = min(100, len(inpt_targets))
            cropped_preds, cropped_targets = crop_topn(inpt_preds, inpt_targets, top_limit)
            # from high to low
            sorted_indices = np.argsort(cropped_preds)[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, top_limit + 1)
            relevant_ranks = ranks[cropped_targets == 1]
            # qq: equal to `np.mean(relevant_ranks)/top_limit`
            fr_values = relevant_ranks / top_limit
            average_fr = np.mean(fr_values)
        res["fr100"] = average_fr

    if "ttif20" in metrics:
        # TODO: this implementation may not align with formal definition.
        top_limit = min(20, len(inpt_targets))
        if num_pos == 0:
            ttif20 = -1.0
        else:
            cropped_preds, cropped_targets = crop_topn(inpt_preds, inpt_targets, top_limit)
            cover_cnt = np.sum(cropped_targets == 1)
            ttif20 = cover_cnt / top_limit
        res["ttif20"] = ttif20

    return res
