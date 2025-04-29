from typing import Annotated, Generic, TypeVar

from torch import Tensor

__all__ = ["Bool", "Float16", "Float32", "Float64", "Int32", "Int64"]

DType = TypeVar("DType", bound=str)


class TensorType(Generic[DType]):
    dtype: str

    @classmethod
    def __class_getitem__(cls, shapes):
        if isinstance(shapes, tuple):
            shape_str = " ".join(shapes)
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

if __name__ == "__main__":

    def process_image(a: Float32["b c h w"], b: Bool["b 1 h w"]) -> Float32["b c h w"]:
        return a * b
