"""CUDA neighbor search with one-time lazy Warp/Triton backend binding."""

__all__ = ["neighbor_list", "neighbor_list_obc", "neighbor_list_pbc", "radius_graph"]


def __getattr__(name: str):
    if name in __all__ or name in ("build_neighbor_list_batch", "build_neighbor_list_batch_full_pbc"):
        from . import cuda

        value = getattr(cuda, name)
    elif name in ("build_neighbor_list_batch_warp", "build_neighbor_list_batch_full_pbc_warp"):
        # Permanent explicit-Warp entry points, independent of the neutral backend.
        from .cuda.warp import pbc

        value = getattr(pbc, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
