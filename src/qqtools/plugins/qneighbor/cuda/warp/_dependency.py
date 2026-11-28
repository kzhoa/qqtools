"""Load the optional Warp dependency on first use."""

from types import ModuleType

from qqtools.qimport import LazyImport

_warp = LazyImport("warp")


def require_warp() -> ModuleType:
    try:
        return _warp.resolve()
    except ImportError as exc:
        cause = exc.__cause__
        if isinstance(cause, ModuleNotFoundError) and cause.name == "warp":
            raise ImportError(
                "qneighbor requires warp-lang. Install it with pip install 'qqtools[qneighbor]'."
            ) from cause
        raise
