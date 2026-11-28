"""Triton CUDA contract against independent enumeration and OBC distances."""

import importlib.util
from itertools import product

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        importlib.util.find_spec("triton") is None or not torch.cuda.is_available(), reason="Triton and CUDA required"
    ),
]


def canonical(pair):
    return sorted(map(tuple, torch.cat((pair[0].T, pair[1]), dim=1).cpu().tolist()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", [1, 32, 128, 257])
@pytest.mark.parametrize("axes", [(True, True, True), (True, False, True)])
def test_periodic_matches_independent_images(dtype, cap, axes):
    from qqtools.plugins.qneighbor.cuda.triton import neighbor_list_pbc

    pos = torch.tensor([[8.0, 0, 0], [0.5, 0, 0], [0, 1.0, 0]], dtype=dtype)
    cell = torch.tensor([[4.0, 1, 0], [0, 4, 0], [0, 0, 4]], dtype=dtype)
    expected = []
    shifts = torch.tensor(list(product(*(range(-3, 4) if flag else [0] for flag in axes))), dtype=torch.long)
    for target in range(len(pos)):
        candidates = []
        for source in range(len(pos)):
            squared = (pos[target] - pos[source] + shifts.to(dtype) @ cell).square().sum(1)
            valid = squared < 5.0**2
            if source == target:
                valid &= shifts.ne(0).any(1)
            candidates.extend(
                (dist, source, tuple(shift)) for dist, shift in zip(squared[valid].tolist(), shifts[valid].tolist())
            )
        expected.extend((source, target, *shift) for _, source, shift in sorted(candidates)[:cap])
    with torch.cuda.stream(torch.cuda.Stream()):
        actual = neighbor_list_pbc(
            pos.cuda(),
            5.0,
            torch.zeros(len(pos), device="cuda", dtype=torch.long),
            cap,
            cell=cell.cuda(),
            pbc=torch.tensor(axes, device="cuda"),
        )
        assert canonical(actual) == sorted(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obc_ties_and_interleaved_batches(dtype):
    from qqtools.plugins.qneighbor.cuda.triton.obc import open_radius_graph

    pos = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], device="cuda", dtype=dtype)
    batch = torch.tensor([0, 2, 0, 0, 2, 2], device="cuda")
    squared = (pos[:, None] - pos).square().sum(-1)
    squared.masked_fill_(batch[:, None] != batch, torch.inf)
    squared.fill_diagonal_(torch.inf)
    for cap in (1, 32, 128):
        order = squared.argsort(dim=1, stable=True)[:, :cap]
        target, rank = (squared.gather(1, order) < 1.5**2).nonzero(as_tuple=True)
        expected = torch.stack((order[target, rank], target))
        torch.testing.assert_close(
            open_radius_graph(positions=pos, cutoff=1.5, batch=batch, max_neighbors=cap), expected
        )


def test_empty_and_invalid_geometry():
    from qqtools.plugins.qneighbor.cuda.triton import neighbor_list_pbc

    ids = torch.empty(0, dtype=torch.long, device="cuda")
    result = neighbor_list_pbc(
        torch.empty((0, 3), device="cuda"), 1.0, ids, 32, cell=torch.eye(3, device="cuda"), pbc=True
    )
    assert result[0].shape == (2, 0) and result[1].shape == (0, 3)
    with pytest.raises(ValueError, match="cell metadata"):
        neighbor_list_pbc(torch.empty((0, 3), device="cuda"), 1.0, ids, 32, cell=None, pbc=True)
    with pytest.raises(ValueError, match="cell metadata"):
        neighbor_list_pbc(
            torch.empty((0, 3), device="cuda"), 1.0, ids, 32, cell=torch.empty((2, 3), device="cuda"), pbc=True
        )
    pos = torch.zeros((1, 3), device="cuda")
    ids = torch.zeros(1, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match="geometry"):
        neighbor_list_pbc(pos, 1.0, ids, 32, cell=torch.zeros((3, 3), device="cuda"), pbc=True)
    with pytest.raises(ValueError, match="Negative"):
        neighbor_list_pbc(pos, 1.0, ids - 1, 32, cell=torch.eye(3, device="cuda"), pbc=True)
