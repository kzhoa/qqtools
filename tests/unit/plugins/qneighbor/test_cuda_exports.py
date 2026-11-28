"""Shared default binding and direct, isolated explicit CUDA exports."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qqtools.plugins import qneighbor
from qqtools.plugins.qneighbor import cuda


@pytest.fixture
def exports(monkeypatch):
    namespaces = [(qneighbor, qneighbor.__all__), (cuda, (*cuda.__all__, *cuda._FUNCTIONS))]
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
    loader = Mock(return_value=cuda._Backend("warp", obc, pbc, pbc, pbc))
    monkeypatch.setattr(cuda, "_load_backend", loader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        functions = list(pool.map(lambda i: getattr(cuda if i % 2 else qneighbor, "neighbor_list"), range(32)))
    assert all(fn is functions[0] for fn in functions)
    loader.assert_called_once_with("warp")
    for name in qneighbor.__all__:
        assert getattr(cuda, name) is getattr(qneighbor, name)
    assert cuda.radius_graph is cuda.neighbor_list_obc is obc
    monkeypatch.setattr(cuda, "_ensure_backend", Mock(side_effect=AssertionError("Repeated selection")))
    monkeypatch.setattr(cuda, "import_module", Mock(side_effect=AssertionError("Repeated import")))
    assert cuda.neighbor_list_obc() == "obc"
    assert cuda.neighbor_list_pbc() == "pbc"


@pytest.mark.parametrize("backend", ["warp", "triton"])
@pytest.mark.parametrize("mode", ["obc", "pbc"])
def test_explicit_cuda_exports_bind_without_default_selection(exports, monkeypatch, backend, mode):
    function = Mock(return_value=42)
    module = SimpleNamespace(**{f"neighbor_list_{mode}": function})
    loader = Mock(return_value=module)
    monkeypatch.setattr(cuda, "import_module", loader)
    monkeypatch.setattr(cuda, "_ensure_backend", Mock(side_effect=AssertionError("Default selection")))
    name = f"neighbor_list_{mode}_{backend}"
    assert getattr(cuda, name) is function
    assert cuda._selected is None
    loader.assert_called_once_with(f"qqtools.plugins.qneighbor.cuda.{backend}")
    monkeypatch.setattr(cuda, "import_module", Mock(side_effect=AssertionError("Repeated import")))
    assert getattr(cuda, name)() == 42


def test_explicit_import_failure_is_not_cached_or_redirected(exports, monkeypatch):
    function = Mock()
    error = ImportError("broken Triton")
    loader = Mock(side_effect=[error, SimpleNamespace(neighbor_list_pbc=function)])
    monkeypatch.setattr(cuda, "import_module", loader)
    with pytest.raises(ImportError, match="broken Triton"):
        cuda.neighbor_list_pbc_triton
    assert "neighbor_list_pbc_triton" not in cuda.__dict__
    assert cuda._selected is None
    assert cuda.neighbor_list_pbc_triton is function
    assert [call.args[0] for call in loader.call_args_list] == ["qqtools.plugins.qneighbor.cuda.triton"] * 2
