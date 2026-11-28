"""CPU open-boundary neighbor search using SciPy."""

from torch import Tensor

from ._search import _assemble, _inputs


def neighbor_list_obc(positions: Tensor, cutoff: float, batch: Tensor | None = None, max_neighbors: int = 32) -> Tensor:
    """Build nearest incoming CPU edges, excluding self edges.

    Args:
        positions: Finite floating-point CPU coordinates (N, 3).
        cutoff: Strict positive distance cutoff.
        batch: Optional nonnegative int64 CPU graph IDs; may be interleaved.
        max_neighbors: Positive incoming neighbor cap per target.

    Returns:
        CPU int64 edge_index (2, E), ordered by target, distance, source.
        Search arithmetic uses float64. Distinct coincident atoms are neighbors.
    """
    points, ids, cap = _inputs(positions, batch, cutoff, max_neighbors)
    return _assemble(points, ids, cutoff, cap)[0]


open_radius_graph = neighbor_list_obc
radius_graph = neighbor_list_obc
