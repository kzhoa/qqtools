"""Common function contracts across the explicit CPU and CUDA namespaces."""

import importlib
import importlib.util
import inspect

import pytest
import torch

_BACKENDS = [
    pytest.param("cpu", "cpu", id="cpu"),
    *[
        pytest.param(
            f"cuda.{backend}",
            "cuda",
            id=backend,
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available() or importlib.util.find_spec(backend) is None,
                    reason=f"CUDA and {backend} required",
                ),
            ],
        )
        for backend in ("warp", "triton")
    ],
]


@pytest.mark.parametrize("namespace,device", _BACKENDS)
def test_default_periodic_cap_is_32(namespace, device):
    backend = importlib.import_module(f"qqtools.plugins.qneighbor.{namespace}")
    positions = torch.zeros((40, 3), device=device)
    cell = torch.eye(3, device=device) * 10
    batch = torch.zeros(40, device=device, dtype=torch.long)
    edges, offsets = backend.neighbor_list_pbc(positions, 1.0, batch, cell=cell, pbc=True)
    assert torch.equal(torch.bincount(edges[1], minlength=40), batch.new_full((40,), 32))
    assert not bool(offsets.any())
    assert not bool((edges[0] == edges[1]).any())
    explicit, explicit_offsets = backend.neighbor_list_pbc(positions, 1.0, batch, 32, cell=cell, pbc=True)
    torch.testing.assert_close(edges, explicit)
    torch.testing.assert_close(offsets, explicit_offsets)
    capped_edges, _ = backend.neighbor_list_pbc(positions, 1.0, batch, max_neighbors=1, cell=cell)
    assert torch.equal(torch.bincount(capped_edges[1], minlength=40), batch.new_ones(40))
    assert backend.radius_graph is backend.neighbor_list_obc


@pytest.mark.parametrize("namespace,device", _BACKENDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_singleton_open_graph_still_validates_coordinates(namespace, device, value):
    backend = importlib.import_module(f"qqtools.plugins.qneighbor.{namespace}")
    positions = torch.tensor([[value, 0, 0]], device=device)
    with pytest.raises(ValueError, match="finite positions"):
        backend.neighbor_list_obc(positions, 1.0)


@pytest.mark.parametrize("namespace,device", _BACKENDS)
@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("flags", [False, [False, False, False], [[False, False, False]], True, [True, False, False]])
def test_unified_return_structure_and_device(namespace, device, count, flags):
    backend = importlib.import_module(f"qqtools.plugins.qneighbor.{namespace}")
    # Exercise a non-current GPU when available, including empty-result allocation.
    if device == "cuda" and torch.cuda.device_count() > 1:
        device = f"cuda:{(torch.cuda.current_device() + 1) % torch.cuda.device_count()}"
    positions = torch.zeros((count, 3), device=device, dtype=torch.float64)
    cell = torch.eye(3, device=device, dtype=positions.dtype)
    pbc = flags if isinstance(flags, bool) else torch.tensor(flags, device=device)
    result = backend.neighbor_list(positions, 1.1, max_neighbors=32, cell=cell, pbc=pbc)
    assert isinstance(result, tuple) and len(result) == 2
    edges, offsets = result
    assert edges.shape[0] == 2
    assert offsets.shape == (edges.shape[1], 3)
    assert edges.dtype == offsets.dtype == torch.long
    assert edges.device == offsets.device == positions.device
    is_periodic = flags if isinstance(flags, bool) else bool(pbc.any())
    if is_periodic:
        expected = backend.neighbor_list_pbc(positions, 1.1, max_neighbors=32, cell=cell, pbc=pbc)
        torch.testing.assert_close(result, expected)
        if count:
            assert bool(offsets.any())
    else:
        torch.testing.assert_close(edges, backend.neighbor_list_obc(positions, 1.1, max_neighbors=32))
        assert not bool(offsets.any())


@pytest.mark.parametrize("namespace,device", _BACKENDS)
def test_shared_signature_and_single_structure_pbc_defaults(namespace, device):
    backend = importlib.import_module(f"qqtools.plugins.qneighbor.{namespace}")
    obc_parameters = list(inspect.signature(backend.neighbor_list_obc).parameters.values())
    assert [parameter.name for parameter in obc_parameters] == ["positions", "cutoff", "batch", "max_neighbors"]
    for search in (backend.neighbor_list, backend.neighbor_list_pbc):
        parameters = list(inspect.signature(search).parameters.values())
        assert [parameter.name for parameter in parameters[:4]] == [parameter.name for parameter in obc_parameters]
        assert parameters[2].default is None
        assert parameters[3].default == 32
        assert [parameter.name for parameter in parameters[4:6]] == ["cell", "pbc"]
        assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters[4:])

    positions = torch.zeros((1, 3), device=device)
    cell = torch.eye(3, device=device)
    edges, offsets = backend.neighbor_list_pbc(positions, 1.1, cell=cell)
    assert edges.shape == (2, 6)
    torch.testing.assert_close(offsets.abs().sum(1), torch.ones(6, device=device, dtype=torch.long))
    explicit_batch = torch.zeros(1, device=device, dtype=torch.long)
    expected = backend.neighbor_list_pbc(positions, 1.1, explicit_batch, 32, cell=cell, pbc=True)
    torch.testing.assert_close((edges, offsets), expected)
    assert backend.neighbor_list(positions, 1.1)[0].shape == (2, 0)
    with pytest.raises(ValueError, match="[Cc]ell"):
        backend.neighbor_list_pbc(positions, 1.1)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("warp") is None,
    reason="CUDA and Warp required",
)
def test_public_warp_pbc_forwards_grid_width():
    from qqtools.plugins.qneighbor.cuda.warp import neighbor_list_pbc

    positions = torch.zeros((1, 3), device="cuda")
    cell = torch.eye(3, device="cuda")
    expected = neighbor_list_pbc(positions, 1.1, cell=cell)
    torch.testing.assert_close(neighbor_list_pbc(positions, 1.1, cell=cell, bin_width=2.0), expected)
    with pytest.raises(ValueError, match="bin_width"):
        neighbor_list_pbc(positions, 1.1, cell=cell, bin_width=0)
