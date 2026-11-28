"""Select and bind CUDA backends once; expose default and explicit functions.

Importing this module does not load Warp or Triton. Default functions prefer
Warp and fall back only when it is absent. Explicit aliases never select or
change the default backend. Resolved names bind directly to their functions.
"""

from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import Callable

from qqtools.qimport import LazyImport

from .._dispatch import _dispatcher

_DEFAULT_NAMES = ("neighbor_list", "neighbor_list_obc", "neighbor_list_pbc", "radius_graph")
_EXPLICIT = {
    "neighbor_list_obc_warp": ("warp", "neighbor_list_obc"),
    "neighbor_list_pbc_warp": ("warp", "neighbor_list_pbc"),
    "neighbor_list_obc_triton": ("triton", "neighbor_list_obc"),
    "neighbor_list_pbc_triton": ("triton", "neighbor_list_pbc"),
}
__all__ = [*_DEFAULT_NAMES, *_EXPLICIT]

neighbor_list: Callable
neighbor_list_obc: Callable
neighbor_list_pbc: Callable
radius_graph: Callable
build_neighbor_list_batch: Callable
build_neighbor_list_batch_full_pbc: Callable
_FUNCTIONS = ("radius_graph", "neighbor_list_pbc", "build_neighbor_list_batch", "build_neighbor_list_batch_full_pbc")
_ALIASES = {"neighbor_list_obc": "radius_graph"}
_LAZY_NAMES = (*_FUNCTIONS, *_ALIASES, "neighbor_list")
_lock = Lock()
_selected = None
_warp = LazyImport("warp")
_triton = LazyImport("triton")


@dataclass(frozen=True)
class _Backend:
    """Selected backend and directly callable CUDA search functions."""

    name: str
    radius_graph: Callable
    build_neighbor_list_batch: Callable
    build_neighbor_list_batch_full_pbc: Callable
    neighbor_list_pbc: Callable


def _load_backend(name: str) -> _Backend:
    if name == "warp":
        _warp.resolve()
    else:
        _triton.resolve()
    base = f"{__name__}.{name}"
    obc = import_module(f"{base}.obc")
    pbc = import_module(f"{base}.pbc")
    suffix = "_warp" if name == "warp" else ""
    return _Backend(
        name,
        obc.open_radius_graph,
        getattr(pbc, f"build_neighbor_list_batch{suffix}"),
        getattr(pbc, f"build_neighbor_list_batch_full_pbc{suffix}"),
        pbc.neighbor_list_pbc,
    )


def _is_missing(exc: ImportError, name: str) -> bool:
    cause = exc.__cause__
    return isinstance(cause, ModuleNotFoundError) and cause.name == name


def _ensure_backend() -> _Backend:
    """Select the default backend once and publish direct function references."""
    global _selected
    with _lock:
        if _selected is not None:
            return _selected
        try:
            selected = _load_backend("warp")
        except ImportError as exc:
            if not _is_missing(exc, "warp"):
                raise
            try:
                selected = _load_backend("triton")
            except ImportError as fallback_exc:
                if _is_missing(fallback_exc, "triton"):
                    raise ImportError(
                        "qneighbor needs warp-lang or triton; install qqtools[qneighbor] or qqtools[qneighbor-triton]"
                    ) from fallback_exc
                raise
        # Only publish after both OBC and PBC imports have succeeded.
        _selected = selected
        for name in _FUNCTIONS:
            globals()[name] = getattr(selected, name)
        for alias, target in _ALIASES.items():
            globals()[alias] = getattr(selected, target)
        globals()["neighbor_list"] = _dispatcher(selected.radius_graph, selected.neighbor_list_pbc)
        return selected


def __getattr__(name: str) -> Callable:
    if name in _LAZY_NAMES:
        _ensure_backend()
        return globals()[name]
    elif name in _EXPLICIT:
        backend, entry = _EXPLICIT[name]
        value = getattr(import_module(f"{__name__}.{backend}"), entry)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
