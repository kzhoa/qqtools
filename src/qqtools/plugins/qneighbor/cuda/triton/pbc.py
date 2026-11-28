"""Public CUDA Triton periodic entry points."""

import math
import operator

import torch
from torch import Tensor

from ._periodic_bounded import search_bounded
from ._periodic_geometry import prepare_geometry
from ._periodic_whole import search_whole
from .obc import open_radius_graph


@torch.no_grad()
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
            defaults to a single graph. Assign consecutive IDs from 0 to B-1
            within each batch and use them to index per-graph cell and pbc rows.
            Nodes may be interleaved. GPU workspaces scale with max(batch) + 1,
            so gaps can increase memory use and runtime.
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
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Expected positive cutoff and neighbor cap")
    cap = operator.index(max_neighbors)
    if cap <= 0:
        raise ValueError("Expected positive cutoff and neighbor cap")
    if not isinstance(pbc, (bool, Tensor)):
        raise ValueError("Expected a Boolean or tensor PBC mask")
    if isinstance(pbc, Tensor) and (pbc.ndim not in (1, 2) or pbc.shape[-1] != 3 or pbc.numel() == 0):
        raise ValueError("Expected PBC metadata of shape (3,) or (B, 3)")
    if cell is not None and (cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3) or cell.numel() == 0):
        raise ValueError("Expected cell metadata of shape (3, 3) or (B, 3, 3)")
    if pbc is True and cell is None:
        raise ValueError("Periodic graphs require cell metadata")
    pos = positions
    if (
        not pos.is_cuda
        or pos.dtype not in (torch.float32, torch.float64)
        or pos.shape != (len(batch), 3)
        or batch.dtype != torch.long
        or batch.ndim != 1
        or batch.device != pos.device
    ):
        raise ValueError("Expected FP32 or FP64 CUDA positions and int64 graph IDs")
    if not len(pos):
        return batch.new_empty((2, 0)), batch.new_empty((0, 3))
    minimum, maximum = torch.aminmax(batch)
    first, last = torch.stack((minimum, maximum)).tolist()
    if first < 0:
        raise ValueError("Negative graph ID")
    graphs = last + 1
    mask = None
    if isinstance(pbc, bool):
        has_images = pbc
    else:
        mask = pbc.to(pos.device, torch.bool).reshape(-1, 3)
        mask = mask.expand(graphs, -1) if len(mask) == 1 else mask[:graphs]
        if len(mask) != graphs:
            raise ValueError("Missing PBC metadata")
        has_images = bool(mask.any())
    if not has_images:
        edges = open_radius_graph(pos, cutoff, batch, cap)
        return edges, edges.new_zeros((edges.shape[1], 3))
    if cell is None:
        raise ValueError("Expected periodic cells")
    cell = cell.to(pos.device, pos.dtype).reshape(-1, 3, 3)
    cell = cell.expand(graphs, -1, -1) if len(cell) == 1 else cell[:graphs]
    if len(cell) != graphs:
        raise ValueError("Missing cell metadata")
    if mask is not None:
        cell = torch.where(mask.any(1)[:, None, None], cell, torch.eye(3, device=pos.device, dtype=pos.dtype))
    pos, cell, batch = pos.contiguous(), cell.contiguous(), batch.contiguous()
    geometry = prepare_geometry(pos, cell, batch, cutoff)
    search = search_whole if cap > 256 else search_bounded
    return search(pos, cell, mask, batch, cutoff, cap, geometry)
