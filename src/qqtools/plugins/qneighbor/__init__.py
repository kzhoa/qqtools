"""CUDA neighbor search with one-time lazy Warp/Triton backend binding."""

__all__ = ["neighbor_list", "neighbor_list_obc", "neighbor_list_pbc", "radius_graph"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import cuda

    value = getattr(cuda, name)
    globals()[name] = value
    return value
