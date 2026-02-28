import pytest

pytest.importorskip("sklearn")

from qqtools.torch import metrics
from qqtools.torch.metrics.binary import confusion_matrix


def test_metrics_init_exports_confusion_matrix():
    assert metrics.confusion_matrix is confusion_matrix
