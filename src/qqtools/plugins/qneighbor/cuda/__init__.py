"""Default CUDA searches. Select a specific backend through cuda.warp or cuda.triton."""

from importlib import import_module
from threading import Lock
from types import ModuleType
from typing import Callable

from qqtools.plugins.qneighbor import __all__

_lock = Lock()
_selected: ModuleType | None = None


def _load_backend(name: str) -> ModuleType:
    # Explicit Warp imports defer loading warp-lang until search. For automatic
    # selection, resolve the dependency before binding its implementation.
    import_module(name)
    return import_module(f"{__name__}.{name}")


def _ensure_backend() -> ModuleType:
    """Bind the selected module's search functions together on first access."""
    global _selected
    with _lock:
        if _selected is not None:
            return _selected
        for name in ("warp", "triton"):
            try:
                backend = _load_backend(name)
            except ModuleNotFoundError as exc:
                if exc.name != name:
                    raise
                continue
            globals().update({export: getattr(backend, export) for export in __all__})
            _selected = backend
            return backend
        raise ImportError(
            "qneighbor needs warp-lang or triton; install qqtools[qneighbor] or qqtools[qneighbor-triton]"
        )


def __getattr__(name: str) -> Callable:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_ensure_backend(), name)
