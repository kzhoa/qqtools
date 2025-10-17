"""
qq:
`TypeVarTuple` supported by python>=3.11

Torch Usage:
    x_tensor: Float["b s d"]
    x_tensor: Long["b", "s", 256]

Numpy Usage:
    arr: FloatArray[3, 3]
    arr2: FloatArray["b", "s", "d"]
    arr3: FloatArray["b s d"]
"""

from typing import Annotated, Generic, TypeVar, TypeVarTuple, Unpack

import numpy as np
import numpy.typing as npt
from torch import Tensor

__all__ = ["Bool", "Float16", "Float32", "Float64", "Int32", "Int64"]


ShapeT = TypeVarTuple("ShapeT")

"""
torch
"""


class TensorType(Generic[Unpack[ShapeT]]):
    dtype: str

    @classmethod
    def __class_getitem__(cls, shapes):
        if isinstance(shapes, tuple):
            shape_str = ", ".join([str(t) for t in shapes])
        elif isinstance(shapes, str):
            shape_str = shapes
        else:
            raise TypeError("shape must be tuple or str")

        return Annotated[Tensor, cls.dtype, shape_str]


# fmt: off
class Float16(TensorType): dtype = "float16"
class Float32(TensorType): dtype = "float32"
class Float64(TensorType): dtype = "float64"
class Bool(TensorType): dtype = "bool"
class Int32(TensorType): dtype = "int32"
class Int64(TensorType): dtype = "int64"
Float = Float32
Long = Int64
# fmt: on

"""
numpy
"""


class Array(Generic[Unpack[ShapeT]]):
    def __repr__(self):
        args = self.__args__ if hasattr(self, "__args__") else ()
        return f"Shape[{', '.join(map(str, args))}]"


# fmt: off
class Float32Array(npt.NDArray[np.float32], Array): pass
class Float64Array(npt.NDArray[np.float64], Array): pass
class BoolArray(npt.NDArray[np.bool], Array): pass
class Int32Array(npt.NDArray[np.int32], Array): pass
class Int64Array(npt.NDArray[np.int64], Array): pass
# alias
FloatArray = Float32Array
LongArray = Int64Array
# fmt: on

if __name__ == "__main__":

    def process_image(a: Float32["b c h w"], b: Bool["b 1 h w"]) -> Float32["b c h w"]:
        return a * b
