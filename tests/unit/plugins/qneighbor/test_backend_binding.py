"""One-time selection, precise fallback and direct-function binding."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from qqtools.plugins.qneighbor import cuda


@pytest.fixture
def binding(monkeypatch):
    previous = {name: cuda.__dict__.get(name) for name in cuda._LAZY_NAMES}
    monkeypatch.setattr(cuda, "_selected", None)
    for name in cuda._LAZY_NAMES:
        cuda.__dict__.pop(name, None)
    try:
        yield cuda
    finally:
        for name, value in previous.items():
            cuda.__dict__.pop(name, None)
            if value is not None:
                cuda.__dict__[name] = value


def backend(name):
    def graph(*args, **kwargs):
        return 42

    return cuda._Backend(name, graph, graph, graph, graph)


def missing(name):
    error = ImportError(f"Failed to import {name}")
    error.__cause__ = ModuleNotFoundError(name, name=name)
    return error


def test_first_access_binds_real_functions_without_dispatch(binding, monkeypatch):
    selected = backend("warp")
    loader = Mock(return_value=selected)
    monkeypatch.setattr(binding, "_load_backend", loader)
    fn = binding.radius_graph
    assert fn is selected.radius_graph
    assert binding._ensure_backend() is selected
    loader.assert_called_once_with("warp")
    monkeypatch.setattr(binding, "_ensure_backend", Mock(side_effect=AssertionError("No repeated init")))
    monkeypatch.setattr(binding, "_load_backend", Mock(side_effect=AssertionError("No repeated lookup")))
    for name in binding._FUNCTIONS:
        assert getattr(binding, name) is getattr(selected, name)
        assert getattr(binding, name)() == 42


def test_concurrent_init_selects_once(binding, monkeypatch):
    selected = backend("warp")
    loader = Mock(return_value=selected)
    monkeypatch.setattr(binding, "_load_backend", loader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(lambda _: binding._ensure_backend(), range(32)))
    assert all(item is selected for item in result)
    loader.assert_called_once_with("warp")


def test_absent_warp_falls_back_once(binding, monkeypatch):
    selected = backend("triton")
    loader = Mock(side_effect=[missing("warp"), selected])
    monkeypatch.setattr(binding, "_load_backend", loader)
    assert binding._ensure_backend() is selected
    assert binding._ensure_backend() is selected
    assert [call.args[0] for call in loader.call_args_list] == ["warp", "triton"]


@pytest.mark.parametrize(
    "error", [missing("warp_cuda_dependency"), ImportError("broken Warp"), RuntimeError("driver failure")]
)
def test_broken_warp_does_not_trigger_fallback(binding, monkeypatch, error):
    loader = Mock(side_effect=error)
    monkeypatch.setattr(binding, "_load_backend", loader)
    with pytest.raises(type(error)):
        binding._ensure_backend()
    loader.assert_called_once_with("warp")
    assert binding._selected is None
    assert all(name not in binding.__dict__ for name in binding._FUNCTIONS)


def test_both_missing_has_installation_guidance(binding, monkeypatch):
    monkeypatch.setattr(binding, "_load_backend", Mock(side_effect=[missing("warp"), missing("triton")]))
    with pytest.raises(ImportError, match="qneighbor-triton"):
        binding._ensure_backend()
    assert binding._selected is None


def test_named_routes_are_direct_aliases(binding, monkeypatch):
    selected = backend("warp")
    monkeypatch.setattr(binding, "_load_backend", Mock(return_value=selected))
    binding._ensure_backend()
    assert binding.neighbor_list_obc is binding.radius_graph is selected.radius_graph
    assert binding.neighbor_list_pbc is selected.neighbor_list_pbc


def test_short_name_dispatches_without_reselecting_backend(binding, monkeypatch):
    import torch

    edges = torch.tensor([[1, 0], [0, 1]])
    periodic_result = (edges, torch.ones((2, 3), dtype=torch.long))
    obc, pbc = Mock(return_value=edges), Mock(return_value=periodic_result)
    monkeypatch.setattr(binding, "_load_backend", Mock(return_value=cuda._Backend("warp", obc, pbc, pbc, pbc)))
    search = binding.neighbor_list
    monkeypatch.setattr(binding, "_ensure_backend", Mock(side_effect=AssertionError("Repeated init")))
    pos = torch.zeros((2, 3))
    for flags in (False, torch.tensor([False, False, False])):
        result, offsets = search(pos, 5.0, pbc=flags)
        assert result is edges
        torch.testing.assert_close(offsets, torch.zeros((2, 3), dtype=torch.long))
    pbc.assert_not_called()
    cell = torch.eye(3)
    for flags in (True, torch.tensor([True, False, False])):
        assert search(pos, 5.0, cell=cell, pbc=flags) is periodic_result
        assert pbc.call_args.args == (pos, 5.0, None, 32)
        assert pbc.call_args.kwargs["cell"] is cell
        assert pbc.call_args.kwargs["pbc"] is flags
    with pytest.raises(ValueError, match="PBC metadata"):
        search(pos, 5.0, pbc=torch.zeros(2))
