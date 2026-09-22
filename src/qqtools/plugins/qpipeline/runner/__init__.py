"""Public qpipeline runner exports, loaded only when requested."""

from qqtools.qimport import LazyImport

_LAZY_EXPORTS = {
    "evaluate_runner": LazyImport(f"{__name__}.eval_runner", "evaluate_runner"),
    "infer_runner": LazyImport(f"{__name__}.eval_runner", "infer_runner"),
    "train_runner": LazyImport(f"{__name__}.runner", "train_runner"),
}


def __getattr__(name: str):
    """Load a public runner export on first access."""
    lazy_export = _LAZY_EXPORTS.get(name)
    if lazy_export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = lazy_export.resolve()
    globals()[name] = value
    return value


__all__ = ["train_runner", "evaluate_runner", "infer_runner"]
