from typing import Annotated, get_args, get_origin

import numpy as np
import numpy.typing as npt
import pytest

from qqtools.utils.qtyping import Float32Array


def test_float32_array_annotation_keeps_dtype_and_shape():
    annotation = Float32Array["batch", "feature"]

    assert get_origin(annotation) is Annotated
    assert get_args(annotation) == (npt.NDArray[np.float32], "batch, feature")


@pytest.mark.parametrize("shape", [1, ["batch", "feature"]])
def test_array_annotation_rejects_invalid_shape(shape):
    with pytest.raises(TypeError, match="shape must be tuple or str"):
        Float32Array[shape]
