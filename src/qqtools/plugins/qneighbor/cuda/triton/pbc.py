"""Public CUDA Triton periodic entry points."""

import math
import operator

import torch
from torch import Tensor

from ._periodic_bounded import build_pbc


def neighbor_list_pbc(
    positions: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    max_neighbors: int = 32,
    *,
    cell: Tensor | None = None,
    pbc: Tensor | bool = True,
) -> tuple[Tensor, Tensor]:
    """Build CUDA neighbors and integer periodic offsets.

    Args:
        positions: Cartesian CUDA coordinates (N, 3), float32 or float64.
        cutoff: Strict positive distance cutoff.
        batch: Optional nonnegative int64 graph IDs on positions.device;
            defaults to a single graph.
        max_neighbors: Positive incoming cap per target; defaults to 32.
        cell: Row-vector lattice (3, 3) or (B, 3, 3), required for periodic axes.
        pbc: True for full PBC (default), False for OBC, or axis flags (3,)/(B,3).

    Returns:
        Int64 (edge_index, cell_offsets), shapes (2, E) and (E, 3), on
        positions.device. Displacement is positions[target] - positions[source]
        + cell_offsets @ cell_for_target. OBC offsets are all zero.
    """
    if batch is None:
        batch = torch.zeros(len(positions), device=positions.device, dtype=torch.long)
    return build_neighbor_list_batch(positions, cell, pbc, batch, cutoff, max_neighbors)


def build_neighbor_list_batch(
    positions: Tensor,
    cell: Tensor | None,
    pbc: Tensor | bool,
    batch: Tensor,
    cutoff: float,
    max_neighbors: int = 32,
) -> tuple[Tensor, Tensor]:
    """Build incoming CUDA edges and integer offsets using bounded ghost queries.

    Args:
        positions: CUDA float32/float64 Cartesian coordinates (N, 3).
        cell: Row-vector cells (3, 3) or (B, 3, 3); None for all-open input.
        pbc: Boolean or axis flags (3,) / (B, 3). Single rows broadcast.
        batch: Nonnegative int64 graph IDs on the positions device.
        cutoff: Strict positive distance cutoff.
        max_neighbors: Positive incoming neighbor cap; defaults to 32.

    Returns:
        Int64 CUDA edge_index (2, E) and offsets (E, 3), with displacement
        positions[target] - positions[source] + offsets @ cell_for_target.
        Zero-offset self edges are excluded; periodic self images are retained.
        Equal distances are ordered by source and integer offset.

    Raises:
        ValueError: If geometry, metadata or search parameters are invalid,
            or periodic expansion exceeds supported indexing/tile limits.
    """
    if not math.isfinite(cutoff) or cutoff <= 0 or operator.index(max_neighbors) <= 0:
        raise ValueError("Expected positive cutoff and neighbor cap")
    if not isinstance(pbc, (bool, Tensor)):
        raise ValueError("Expected a Boolean or tensor PBC mask")
    if isinstance(pbc, Tensor) and (pbc.ndim not in (1, 2) or pbc.shape[-1] != 3 or pbc.numel() == 0):
        raise ValueError("Expected PBC metadata of shape (3,) or (B, 3)")
    if cell is not None and (cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3) or cell.numel() == 0):
        raise ValueError("Expected cell metadata of shape (3, 3) or (B, 3, 3)")
    if pbc is True and cell is None:
        raise ValueError("Periodic graphs require cell metadata")
    return build_pbc(positions, cell, pbc, batch, cutoff, max_neighbors)


def build_neighbor_list_batch_full_pbc(
    positions: Tensor,
    cell: Tensor,
    batch: Tensor,
    cutoff: float,
    max_neighbors: int = 32,
) -> tuple[Tensor, Tensor]:
    """Build CUDA neighbors assuming every axis is periodic."""
    if cell is None:
        raise ValueError("Periodic graphs require cell metadata")
    return build_neighbor_list_batch(positions, cell, True, batch, cutoff, max_neighbors)
