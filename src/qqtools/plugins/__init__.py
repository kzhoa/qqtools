from __future__ import annotations

import importlib

__all__ = ["qexp", "qhyperconnect", "qpipeline"]


def __getattr__(name: str):
    if name == "qpipeline":
        return importlib.import_module(".qpipeline", __name__)
    if name == "qexp":
        return importlib.import_module(".qexp", __name__)
    if name == "qhyperconnect":
        module = importlib.import_module(".qhyperconnect", __name__)
        return getattr(module, "qhyperconnect")
    raise AttributeError(f"module {__name__} has no attribute {name}")
