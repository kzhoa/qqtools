"""Concurrent access to the shared CUDA and top-level search functions."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qqtools.plugins import qneighbor
from qqtools.plugins.qneighbor import cuda


@pytest.fixture
def exports(monkeypatch):
    namespaces = [(qneighbor, qneighbor.__all__), (cuda, cuda.__all__)]
    saved = [
        (module, names, {name: module.__dict__[name] for name in names if name in module.__dict__})
        for module, names in namespaces
    ]
    monkeypatch.setattr(cuda, "_selected", None)
    for module, names, _ in saved:
        for name in names:
            module.__dict__.pop(name, None)
    try:
        yield
    finally:
        for module, names, previous in saved:
            for name in names:
                module.__dict__.pop(name, None)
            module.__dict__.update(previous)


def test_cuda_and_top_level_share_one_concurrent_binding(exports, monkeypatch):
    obc, pbc = Mock(return_value="obc"), Mock(return_value="pbc")
    backend = SimpleNamespace(neighbor_list=Mock(), neighbor_list_obc=obc, radius_graph=obc, neighbor_list_pbc=pbc)
    loader = Mock(return_value=backend)
    monkeypatch.setattr(cuda, "_load_backend", loader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        functions = list(pool.map(lambda i: getattr(cuda if i % 2 else qneighbor, "neighbor_list"), range(32)))
    assert all(fn is backend.neighbor_list for fn in functions)
    loader.assert_called_once_with("warp")
    for name in qneighbor.__all__:
        assert getattr(cuda, name) is getattr(qneighbor, name)
    assert cuda.radius_graph is cuda.neighbor_list_obc is obc
    monkeypatch.setattr(cuda, "_ensure_backend", Mock(side_effect=AssertionError("Repeated selection")))
    monkeypatch.setattr(cuda, "import_module", Mock(side_effect=AssertionError("Repeated import")))
    assert cuda.neighbor_list_obc() == "obc"
    assert cuda.neighbor_list_pbc() == "pbc"
