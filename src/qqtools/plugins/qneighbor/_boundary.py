"""Open and periodic boundary selection shared by the neighbor backends."""

from typing import Callable

import torch
from torch import Tensor


def make_neighbor_list(obc: Callable, periodic: Callable) -> Callable:
    def neighbor_list(
        positions: Tensor,
        cutoff: float,
        batch: Tensor | None = None,
        max_neighbors: int = 32,
        *,
        cell: Tensor | None = None,
        pbc: Tensor | bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Dispatch neighbor search from the boundary flags.

        When imported from qqtools.plugins.qneighbor, this function uses CUDA
        and does not automatically switch to CPU. For CPU inputs, explicitly use
        ``from qqtools.plugins.qneighbor.cpu import neighbor_list``.

        For known non-periodic systems, use ``neighbor_list_obc`` or its alias
        ``radius_graph`` to return only edges and avoid allocating zero offsets
        on the GPU. Tensor PBC flags require a reduction, potentially
        synchronizing CUDA; dedicated
        ``neighbor_list_obc/pbc`` entries bypass this boundary dispatch.

        Args:
            positions: Cartesian positions (N, 3) on the selected backend device.
            cutoff: Strict distance cutoff.
            batch: Optional int64 graph IDs; defaults to a single graph. For periodic
                CUDA searches, number the graphs in each batch from 0 to B-1, where B
                is the number of graphs. Nodes may be interleaved. Use these IDs to
                index per-graph cell and pbc rows. GPU workspaces scale with
                max(batch) + 1, so gaps can increase memory use and runtime.
            max_neighbors: Incoming nearest-neighbor cap.
            cell: Row-vector lattice required for periodic axes.
            pbc: False for OBC, True for full PBC, or axis flags (3,)/(B,3).

        Returns:
            Tuple of int64 (edge_index, cell_offsets), shapes (2, E) and (E, 3),
            on positions.device. OBC cell offsets are all zero.
        """
        if isinstance(pbc, bool):
            is_periodic = pbc
        elif isinstance(pbc, Tensor):
            if pbc.ndim not in (1, 2) or pbc.shape[-1] != 3 or pbc.numel() == 0:
                raise ValueError("Expected PBC metadata of shape (3,) or (B, 3)")
            is_periodic = bool(pbc.any())
        else:
            raise ValueError("Expected a Boolean or tensor PBC mask")
        if not is_periodic:
            if isinstance(pbc, Tensor) and pbc.ndim == 2 and len(pbc) > 1:
                if batch is not None and batch.numel() and int(batch.max()) >= len(pbc):
                    raise ValueError("Missing PBC metadata for a batch graph")
            edges = obc(positions, cutoff, batch, max_neighbors)
            return edges, edges.new_zeros((edges.shape[1], 3))
        return periodic(positions, cutoff, batch, max_neighbors, cell=cell, pbc=pbc)

    return neighbor_list
