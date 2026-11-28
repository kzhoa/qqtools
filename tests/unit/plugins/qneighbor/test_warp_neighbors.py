import importlib.util
from itertools import product

import pytest
import torch

from qqtools.plugins.qneighbor.cuda.warp.obc import open_radius_graph
from qqtools.plugins.qneighbor.cuda.warp.pbc import build_neighbor_list_batch_warp

requires_warp = pytest.mark.skipif(importlib.util.find_spec("warp") is None, reason="qneighbor extra required")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _open_reference(positions, batch, cutoff, cap):
    squared = (positions[:, None] - positions[None, :]).square().sum(-1)
    squared.masked_fill_(batch[:, None] != batch, torch.inf)
    squared.fill_diagonal_(torch.inf)
    order = squared.argsort(dim=1, stable=True)[:, :cap]
    target, rank = (squared.gather(1, order) < cutoff**2).nonzero(as_tuple=True)
    return torch.stack((order[target, rank], target))


@requires_warp
@pytest.mark.parametrize(
    ("device", "dtype", "cap"),
    [
        ("cpu", torch.float64, 3),
        pytest.param("cuda", torch.float32, 3, marks=[requires_cuda, pytest.mark.gpu]),
        pytest.param("cuda", torch.float32, 64, marks=[requires_cuda, pytest.mark.gpu]),
        pytest.param("cuda", torch.float64, 128, marks=[requires_cuda, pytest.mark.gpu]),
    ],
)
def test_open_neighbors_match_pairwise_search(device, dtype, cap):
    # More candidates than threads for the CUDA cases, with ties and interleaved graphs.
    positions = torch.zeros((270, 3), dtype=dtype)
    positions[:, 0] = torch.arange(len(positions)) % 7 - 3
    batch = (torch.arange(len(positions)) % 2).mul(3)
    expected = _open_reference(positions, batch, 2.0, cap)
    from qqtools.plugins.qneighbor.cpu.obc import open_radius_graph as cpu_reference

    search = cpu_reference if device == "cpu" else open_radius_graph
    actual = search(positions.to(device), 2.0, batch.to(device), cap)
    torch.testing.assert_close(actual.cpu(), expected)


@requires_warp
@requires_cuda
@pytest.mark.gpu
def test_all_open_periodic_entry_point_returns_zero_offsets():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
    edges, offsets = build_neighbor_list_batch_warp(
        positions, None, torch.zeros(3, dtype=torch.bool), torch.zeros(2, dtype=torch.long, device="cuda"), 2.0, 1
    )
    torch.testing.assert_close(edges.cpu(), torch.tensor([[1, 0], [0, 1]]))
    torch.testing.assert_close(offsets.cpu(), torch.zeros((2, 3), dtype=torch.long))
    assert open_radius_graph(positions[:0], 2.0).shape == (2, 0)


def _periodic_reference(positions, cell, periodic, batch, cutoff, cap):
    expected = set()
    for target, graph in enumerate(batch.tolist()):
        images = tuple(product(*(range(-3, 4) if is_periodic else [0] for is_periodic in periodic[graph])))
        candidates = []
        for source in (batch == graph).nonzero(as_tuple=True)[0].tolist():
            for offset in images:
                if source == target and offset == (0, 0, 0):
                    continue
                delta = positions[target] - positions[source] + torch.tensor(offset, dtype=cell.dtype) @ cell[graph]
                squared = delta.square().sum().item()
                if squared < cutoff**2:
                    candidates.append((squared, source, offset))
        expected.update((source, target, *offset) for _, source, offset in sorted(candidates)[:cap])
    return expected


@requires_warp
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_neighbors_match_image_enumeration(dtype):
    cell = torch.tensor(
        [[[4, 1, 0], [0, 4, 0], [0, 0, 4]], [[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]] * 3],
        dtype=dtype,
    )
    periodic = torch.tensor([[True, True, True], [True, False, False], [False, False, False]])
    batch = torch.tensor([1, 0, 2, 0, 1, 2])
    positions = torch.tensor([[10, 0, 0], [0, 0, 0], [0, 0, 0], [3.5, 1, 0], [0, 1, 0], [0, 4, 0]], dtype=dtype)
    cutoff, cap = 4.25, 5
    expected = _periodic_reference(positions, cell, periodic, batch, cutoff, cap)
    edges, offsets = build_neighbor_list_batch_warp(
        positions.cuda(), cell.cuda(), periodic.cuda(), batch.cuda(), cutoff, cap
    )
    actual = list(map(tuple, torch.cat((edges.T, offsets), dim=1).cpu().tolist()))
    assert set(actual) == expected
    assert len(actual) == len(expected)


@requires_warp
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_pbc_entry_points_match_periodic_images(dtype):
    from qqtools.plugins.qneighbor import build_neighbor_list_batch_full_pbc_warp

    cell = torch.tensor([[4, 1, 0], [0, 4, 0], [0, 0, 4]], dtype=dtype)
    pos = torch.tensor([[10, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=dtype)
    batch = torch.tensor([1, 0, 1, 0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    expected = _periodic_reference(pos, cell.expand(2, -1, -1), mask, batch, 4.25, 7)
    pos, cell, batch = pos.cuda(), cell.cuda(), batch.cuda()
    direct = build_neighbor_list_batch_full_pbc_warp(pos, cell, batch, 4.25, 7)
    for pbc in (True, mask.cuda()):
        result = build_neighbor_list_batch_warp(pos, cell, pbc, batch, 4.25, 7)
        torch.testing.assert_close(result, direct)
    edges, shifts = direct
    keys = list(map(tuple, torch.cat((edges.T, shifts), dim=1).cpu().tolist()))
    assert len(keys) == len(expected)
    assert set(keys) == expected


@requires_warp
@requires_cuda
@pytest.mark.gpu
def test_periodic_grid_compacts_empty_rows_on_current_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        # Noncontiguous positions and graph IDs, one isolated target, custom open grid.
        positions = (
            torch.tensor([[0, 0, 0], [0.5, 0, 0], [3, 0, 0]], device="cuda", dtype=torch.float64)
            .T.contiguous()
            .T.requires_grad_()
        )
        batch = torch.zeros(6, device="cuda", dtype=torch.long)[::2]
        for cutoff, expected in [(0.5, [[], []]), (1.0, [[1, 0], [0, 1]])]:
            edges, shifts = build_neighbor_list_batch_warp(positions, None, False, batch, cutoff, 5, bin_width=0.25)
            torch.testing.assert_close(edges.cpu(), torch.tensor(expected, dtype=torch.long))
            torch.testing.assert_close(shifts.cpu(), torch.zeros((edges.shape[1], 3), dtype=torch.long))
        assert positions.grad is None


@requires_warp
@requires_cuda
@pytest.mark.gpu
def test_periodic_geometry_checks_positions_and_unoccupied_cells():
    # Graph 2 is the only occupied graph. All three periodic cells must still be valid.
    positions = torch.zeros((1, 3), device="cuda")
    batch = torch.tensor([2], device="cuda")
    cells = torch.eye(3, device="cuda").repeat(3, 1, 1)
    cells[1].zero_()
    with pytest.raises(ValueError, match="finite periodic cells"):
        build_neighbor_list_batch_warp(positions, cells, True, batch, 1.0, 3)
    with pytest.raises(ValueError, match="finite positions"):
        build_neighbor_list_batch_warp(positions.fill_(float("nan")), cells[0], True, batch, 1.0, 3)


@requires_warp
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", [1, 32, 128, 256])
def test_periodic_pruning_keeps_images_across_bin_faces_and_corners(dtype, cap):
    # This skew lattice places valid self-images on many shells, including ties.
    # The reciprocal-plane heights prove offsets outside [-3, 3] cannot qualify.
    positions = torch.tensor([[3.25, -1.0, 5.5]], dtype=dtype)
    cell = torch.tensor([[[2, 0.5, 0], [0, 2, 0], [0, 0, 2]]], dtype=dtype)
    periodic = torch.ones((1, 3), dtype=torch.bool)
    batch = torch.zeros(1, dtype=torch.long)
    cutoff = 6.25
    expected = _periodic_reference(positions, cell, periodic, batch, cutoff, cap)
    for width in (0.5, 2.0):
        edges, offsets = build_neighbor_list_batch_warp(
            positions.cuda(), cell.cuda(), True, batch.cuda(), cutoff, cap, bin_width=width
        )
        actual = list(map(tuple, torch.cat((edges.T, offsets), dim=1).cpu().tolist()))
        assert len(actual) == len(expected)
        assert set(actual) == expected


@pytest.mark.parametrize(
    "kwargs",
    [dict(cutoff=0), dict(cutoff=float("nan")), dict(max_neighbors=0), dict(bin_width=-1)],
)
def test_public_scalar_validation_even_for_empty_inputs(kwargs):
    args = dict(cutoff=1.0, max_neighbors=32)
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_neighbor_list_batch_warp(torch.empty(0, 3), torch.eye(3), True, torch.empty(0, dtype=torch.long), **args)


@pytest.mark.skipif(
    importlib.util.find_spec("warp") is None or not torch.cuda.is_available(), reason="Warp extra and CUDA required"
)
@pytest.mark.gpu
def test_public_shapes_and_metadata_rejected_before_kernel():
    pos = torch.zeros((2, 3), device="cuda")
    ids = torch.zeros(2, dtype=torch.long, device="cuda")
    cell = torch.eye(3, device="cuda") * 4
    for p, c, mask, b in (
        (pos, cell, True, ids - 1),
        (pos, cell, None, ids),
        (pos, cell[:2], True, ids),
        (pos, cell, torch.ones(2), ids),
        (pos, cell.repeat(2, 1, 1), True, ids + 2),
        (pos, cell, torch.ones(2, 3), ids + 2),
    ):
        with pytest.raises(ValueError):
            build_neighbor_list_batch_warp(p, c, mask, b, 1.0, 32)


@pytest.mark.skipif(
    importlib.util.find_spec("warp") is None or not torch.cuda.is_available(), reason="Warp extra and CUDA required"
)
@pytest.mark.gpu
@pytest.mark.parametrize("cap", [1, 24, 32, 64, 128, 129])
@pytest.mark.parametrize("mode", ["full", "mixed", "open"])
def test_periodic_current_stream_and_sparse_ids_match_images(cap, mode):
    with torch.cuda.stream(torch.cuda.Stream()):
        pos = (
            torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]], device="cuda")
            .T.contiguous()
            .T
        )
        ids = torch.tensor([2, 2, 0, 0, 2], device="cuda")
        cell = torch.tensor([[4.0, 1, 0], [0, 4, 0], [0, 0, 4]], device="cuda")
        mask = True if mode == "full" else torch.tensor([mode == "mixed", False, False], device="cuda")
        masks = torch.ones((3, 3), dtype=torch.bool) if mode == "full" else mask.cpu().expand(3, -1)
        expected = _periodic_reference(pos.cpu(), cell.cpu().expand(3, -1, -1), masks, ids.cpu(), 3.0, cap)
        edges, shifts = build_neighbor_list_batch_warp(pos, cell, mask, ids, 3.0, cap)
        actual = list(map(tuple, torch.cat((edges.T, shifts), dim=1).cpu().tolist()))
        assert len(actual) == len(expected)
        assert set(actual) == expected


@pytest.mark.skipif(
    importlib.util.find_spec("warp") is None or not torch.cuda.is_available(), reason="Warp extra and CUDA required"
)
@pytest.mark.gpu
def test_empty_public_graph_has_typed_empty_outputs():
    pos = torch.empty((0, 3), device="cuda")
    ids = torch.empty(0, device="cuda", dtype=torch.long)
    for mask, cell in ((True, torch.eye(3, device="cuda")), (False, None)):
        edges, shifts = build_neighbor_list_batch_warp(pos, cell, mask, ids, 1.0, 32)
        assert edges.shape == (2, 0) and shifts.shape == (0, 3)
        assert edges.dtype == shifts.dtype == torch.long
        assert edges.device == shifts.device == pos.device


@requires_warp
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("is_partial", [False, True])
def test_periodic_geometry_near_wrap_and_grid_boundaries(dtype, is_partial):
    """Cell-height rounding must not omit neighbors at wrap or bin boundaries."""
    one = torch.tensor(1.0, dtype=dtype)
    fractions = torch.zeros((4, 3), dtype=dtype)
    fractions[:, 0] = torch.stack((one * 0, torch.nextafter(one, one * 0), one, torch.nextafter(one, one * 2)))
    fractions[:, 1:] = 0.5
    batch = torch.zeros(4, dtype=torch.long)
    periodic = torch.tensor([[True, True, not is_partial]])
    for direction in (0.0, 8.0):
        side = torch.nextafter(torch.tensor(4.0, dtype=dtype), torch.tensor(direction, dtype=dtype))
        for skew in (0.0, 0.5):
            cell = torch.eye(3, dtype=dtype) * side
            cell[1, 0] = skew
            positions = fractions @ cell
            expected = _periodic_reference(positions, cell[None], periodic, batch, 5.37, 64)
            edges, offsets = build_neighbor_list_batch_warp(
                positions.cuda(), cell.cuda(), periodic.cuda(), batch.cuda(), 5.37, 64
            )
            actual = list(map(tuple, torch.cat((edges.T, offsets), dim=1).cpu().tolist()))
            assert len(actual) == len(expected)
            assert set(actual) == expected
