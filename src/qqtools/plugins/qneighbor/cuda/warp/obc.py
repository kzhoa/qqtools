"""Exact incoming neighbors in an open (nonperiodic) Cartesian graph."""

import math
import operator

import torch
from torch import Tensor


def _torch_edges(pos, batch, cutoff, k):
    # CUDA fallback for dtypes/caps outside the Warp kernel's supported range.
    sizes = torch.unique_consecutive(batch, return_counts=True)[1].tolist()
    edges = []
    start = 0
    for size in sizes:
        points = pos[start : start + size]
        if points.dtype in (torch.float16, torch.bfloat16):
            points = points.float()
        chunk = max(1, 262144 // size)
        for first in range(0, size, chunk):
            targets = points[first : first + chunk]
            squared = (targets[:, None] - points).square().sum(-1)
            squared[:, first : first + len(targets)].fill_diagonal_(torch.inf)
            indices = squared.argsort(dim=1, stable=True)[:, :k]
            row, slot = (squared.gather(1, indices) < cutoff**2).nonzero(as_tuple=True)
            edges.append(torch.stack((start + indices[row, slot], start + first + row)))
        start += size
    return torch.cat(edges, dim=1)


@torch.no_grad()
def open_radius_graph(
    positions: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    max_neighbors: int = 32,
) -> Tensor:
    """Find the nearest incoming neighbors of each node in an open system.

    Args:
        positions: Floating-point Cartesian coordinates of shape (N, 3).
        cutoff: Strict upper bound on neighbor distance.
        batch: Optional int64 graph IDs of shape (N,) on the positions device.
            By default all nodes belong to one graph. IDs may be interleaved.
        max_neighbors: Positive cap on incoming edges per target.

    Returns:
        Int64 edge index of shape (2, E) on the positions device. Each edge is
        (source, target). Self-edges are excluded. Edges are grouped by target
        and ordered by squared distance, then source ID.

    Raises:
        ImportError: If warp-lang is missing.
        ValueError: If positions, batch or search parameters are invalid.
    """
    if not positions.is_cuda:
        raise ValueError("The public neighbor API currently requires CUDA positions")
    k = operator.index(max_neighbors)
    if k < 1 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Expected positive cutoff and neighbor cap")
    if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_floating_point():
        raise ValueError("Expected floating-point Cartesian positions of shape (N, 3)")
    n = len(positions)
    if batch is None:
        batch = torch.zeros(n, device=positions.device, dtype=torch.long)
    if batch.shape != (n,) or batch.dtype != torch.long or batch.device != positions.device:
        raise ValueError("Expected one int64 batch ID per node on the positions device")
    if n and not bool(torch.isfinite(positions).all()):
        raise ValueError("Expected finite positions")
    if n < 2:
        return batch.new_empty((2, 0))
    k = min(k, n - 1)
    order = None
    if bool((batch[1:] < batch[:-1]).any()):
        order = batch.argsort(stable=True)
        positions, batch = positions[order], batch[order]

    if positions.dtype == torch.float32 and k < 256 and n * k <= 2**31 - 1:
        from ._obc_kernels import open_edges

        block_size = max(64, 1 << k.bit_length())
        edges = open_edges(positions, batch, cutoff, k, block_size)
    else:
        edges = _torch_edges(positions, batch, cutoff, k)
    if order is not None:
        edges = order[edges]
        edges = edges[:, edges[1].argsort(stable=True)]
    return edges
