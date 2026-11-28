"""Explicit Warp CUDA neighbor functions; never select another backend."""

from qqtools.plugins.qneighbor._boundary import make_neighbor_list as _make_neighbor_list

from .obc import open_radius_graph as neighbor_list_obc
from .pbc import neighbor_list_pbc

__all__ = ["neighbor_list", "neighbor_list_obc", "neighbor_list_pbc", "radius_graph"]

radius_graph = neighbor_list_obc
neighbor_list = _make_neighbor_list(neighbor_list_obc, neighbor_list_pbc)
