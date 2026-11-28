"""CPU neighbor functions; SciPy is required, ASE and CUDA backends are not."""

from .._dispatch import _dispatcher
from .obc import neighbor_list_obc
from .pbc import neighbor_list_pbc

__all__ = ["neighbor_list", "neighbor_list_obc", "neighbor_list_pbc", "radius_graph"]

radius_graph = neighbor_list_obc
neighbor_list = _dispatcher(neighbor_list_obc, neighbor_list_pbc)
