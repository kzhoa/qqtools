"""Public interface for periodic neighbor lists."""

import math
import operator

import torch
from torch import Tensor

from ._dependency import require_warp
from ._pbc_grid import grid_neighbors
from .obc import open_radius_graph


def neighbor_list_pbc(
    positions: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    max_neighbors: int = 32,
    *,
    cell: Tensor | None = None,
    pbc: Tensor | bool = True,
    bin_width: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Build CUDA neighbors and integer periodic offsets.

    Args:
        positions: Cartesian CUDA coordinates (N, 3), float32 or float64.
        cutoff: Strict positive distance cutoff.
        batch: Optional nonnegative int64 graph IDs on positions.device;
            defaults to a single graph. Assign consecutive IDs from 0 to B-1
            within each batch and use them to index per-graph cell and pbc rows.
            Nodes may be interleaved. GPU workspaces scale with max(batch) + 1,
            so gaps can increase memory use and runtime.
        max_neighbors: Positive incoming cap per target; defaults to 32.
        cell: Row-vector lattice (3, 3) or (B, 3, 3), required for periodic axes.
        pbc: True for full PBC (default), False for OBC, or axis flags (3,)/(B,3).
        bin_width: Optional positive grid-bin width for the Warp backend.

    Returns:
        Int64 (edge_index, cell_offsets), shapes (2, E) and (E, 3), on
        positions.device. Displacement is positions[target] - positions[source]
        + cell_offsets @ cell_for_target. OBC offsets are all zero.
    """
    if batch is None:
        batch = torch.zeros(len(positions), device=positions.device, dtype=torch.long)
    _validate_parameters(cutoff, max_neighbors, bin_width)
    _validate_inputs(positions, cell, batch)
    if pbc is True:
        if cell is None:
            raise ValueError("Periodic graphs require cell metadata")
        return grid_neighbors(positions, cell, None, batch, cutoff, max_neighbors, bin_width=bin_width)
    if pbc is False:
        pbc = torch.zeros(3, dtype=torch.bool)
    if not isinstance(pbc, Tensor):
        raise ValueError("Expected a Boolean or tensor PBC mask")
    if pbc.ndim not in (1, 2) or pbc.shape[-1] != 3 or pbc.numel() == 0:
        raise ValueError("Expected PBC metadata of shape (3,) or (B, 3)")
    if bin_width is None and not bool(pbc.any()):
        if pbc.ndim == 2 and len(pbc) > 1 and batch.numel() and int(batch.max()) >= len(pbc):
            raise ValueError("Missing PBC metadata for a batch graph")
        edges = open_radius_graph(positions, cutoff, batch, max_neighbors)
        return edges, edges.new_zeros((edges.shape[1], 3))
    return grid_neighbors(
        positions,
        cell,
        pbc,
        batch,
        cutoff,
        max_neighbors,
        bin_width=bin_width,
    )


def _validate_inputs(positions: Tensor, cell: Tensor | None, batch: Tensor) -> None:
    require_warp()
    if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_floating_point():
        raise ValueError("Expected floating-point Cartesian positions of shape (N, 3)")
    if batch.shape != (len(positions),) or batch.dtype != torch.long or batch.device != positions.device:
        raise ValueError("Expected one int64 batch ID per node on the positions device")
    if batch.numel() and bool((batch < 0).any()):
        raise ValueError("Expected nonnegative batch IDs")
    if cell is not None and (cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3) or cell.numel() == 0):
        raise ValueError("Expected cell metadata of shape (3, 3) or (B, 3, 3)")


def _validate_parameters(cutoff: float, cap: int, width: float | None) -> None:
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if operator.index(cap) <= 0:
        raise ValueError("max_neighbors must be positive")
    if width is not None and (not math.isfinite(width) or width <= 0):
        raise ValueError("bin_width must be finite and positive")
