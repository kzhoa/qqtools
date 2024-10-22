"""
torch.jit friendly implementation of scatter()
can be a substantiate of `from torch_scatter import scatter`
from torch_geoemtric.utils
"""

from typing import Optional

import torch
from torch import Tensor


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)


@torch.jit.script
def scatter(
    ref: Tensor,
    index: Tensor,
    dim: int,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    if dim < 0:
        dim = torch.add(ref.dim(), dim)

    if dim < 0 or dim >= ref.dim():
        raise ValueError(f"dim out of range, got dim={dim}, but _ref.shape{ref.shape}")

    # handle _dim_size
    assert index.numel() > 0, "expect _index not empty"

    if dim_size is None:
        dim_size = torch.add(int(torch.max(index)), 1)

    # handle output _size
    _size = list(ref.shape)
    _size[dim] = dim_size

    # handle _index
    # torch.scatter_add_ requires that `_index.shape == _ref.shape`
    # broadcast
    if reduce == "sum" or reduce == "add":
        index = broadcast(index, ref, dim)
        out = ref.new_zeros(_size)
        out = out.scatter_add_(dim, index, ref)
        return out

    if reduce == "mean":
        count = ref.new_zeros(dim_size)
        count.scatter_add_(0, index, ref.new_ones(ref.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, ref, dim)
        out = ref.new_zeros(_size)
        out = out.scatter_add_(dim, index, ref)

        return out / broadcast(count, out, dim)

    raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")
