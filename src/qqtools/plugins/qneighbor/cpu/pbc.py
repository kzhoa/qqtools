"""CPU periodic neighbor search using explicit images and SciPy KDTree."""

import numpy as np
from torch import Tensor

from ._search import _assemble, _inputs


def neighbor_list_pbc(
    positions: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    max_neighbors: int = 32,
    *,
    cell: Tensor | None = None,
    pbc: Tensor | bool = True,
) -> tuple[Tensor, Tensor]:
    """Build incoming CPU edges and integer periodic offsets.

    Args:
        positions: Finite floating-point CPU coordinates (N, 3).
        cutoff: Strict positive distance cutoff; may exceed cell dimensions.
        batch: Optional nonnegative CPU int64 graph IDs; defaults to a single
            graph and may be interleaved.
        max_neighbors: Positive incoming cap per target; defaults to 32.
        cell: CPU row-vector lattice (3, 3) or (B, 3, 3). Periodic
            graphs require finite nonsingular cells, including partial PBC.
        pbc: True for full PBC (default), False for OBC, or CPU axis flags
            (3,) / (B, 3).

    Returns:
        Int64 CPU (edge_index, offsets), shapes (2, E) and (E, 3).
        Displacement is pos[target] - pos[source] + offsets @ cell.
        Results are sorted by target, float64 squared distance, source, offset.
        Only the zero-image self edge is excluded. All-false PBC needs no cell.
        Image storage grows with cutoff relative to cell dimensions. Dense
        neighborhoods use bounded nearest queries with complete tie repair;
        sparse neighborhoods use radius enumeration.
    """
    points, ids, cap = _inputs(positions, batch, cutoff, max_neighbors)
    if isinstance(pbc, bool):
        masks = np.full((1, 3), pbc, dtype=bool)
    elif (
        isinstance(pbc, Tensor)
        and pbc.device.type == "cpu"
        and pbc.ndim in (1, 2)
        and pbc.shape[-1] == 3
        and pbc.numel()
    ):
        masks = pbc.detach().bool().numpy().reshape(-1, 3)
    else:
        raise ValueError("Expected a Boolean or CPU PBC mask of shape (3,) or (B, 3)")
    if len(ids) and len(masks) != 1 and ids.max() >= len(masks):
        raise ValueError("Missing PBC metadata for a batch graph")
    cells = None
    if cell is not None:
        if cell.device.type != "cpu" or cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3) or not cell.numel():
            raise ValueError("Expected CPU cell metadata of shape (3, 3) or (B, 3, 3)")
        cells = cell.detach().double().numpy().reshape(-1, 3, 3)
        if len(ids) and len(cells) != 1 and ids.max() >= len(cells):
            raise ValueError("Missing cell metadata for a batch graph")
    graphs = np.unique(ids) if len(ids) else [0]
    for graph in graphs:
        if masks[0 if len(masks) == 1 else graph].any():
            if cells is None:
                raise ValueError("Periodic graphs require cell metadata")
            lattice = cells[0 if len(cells) == 1 else graph]
            if not np.isfinite(lattice).all() or np.linalg.matrix_rank(lattice) != 3:
                raise ValueError("Periodic graphs require finite nonsingular cells")
    return _assemble(points, ids, cutoff, cap, cells, masks)
