"""CPU contracts checked against independent image enumeration."""

from itertools import product

import numpy as np
import pytest
import torch

from qqtools.plugins.qneighbor.cpu import neighbor_list, neighbor_list_obc, neighbor_list_pbc, radius_graph


def reference(pos, cell, flags, batch, cutoff, cap):
    edges, shifts = [], []
    for target in range(len(pos)):
        graph = int(batch[target])
        candidates = []
        for source in range(len(pos)):
            if batch[source] != graph:
                continue
            for shift in product(*(range(-5, 6) if flag else [0] for flag in flags[graph])):
                if source == target and shift == (0, 0, 0):
                    continue
                delta = pos[target] - pos[source] + np.asarray(shift) @ cell[graph]
                distance = delta @ delta
                if distance < cutoff**2:
                    candidates.append((distance, source, shift))
        for _, source, shift in sorted(candidates)[:cap]:
            edges.append((source, target))
            shifts.append(shift)
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T, torch.tensor(shifts, dtype=torch.long).reshape(-1, 3)


@pytest.mark.parametrize("cap", [1, 32, 128])
@pytest.mark.parametrize("cutoff", [1.0, 5.0, 12.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_enumeration(cap, cutoff, dtype):
    pos = torch.tensor([[8.0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 0], [0, 0, 0], [2, 0, 0]], dtype=dtype)
    cell = torch.tensor(
        [[[4.0, 1, 0], [0, 4, 0], [0, 0, 5]], [[4.0, 0, 0], [0, 4, 0], [0, 0, 4]], [[0.0, 0, 0]] * 3], dtype=dtype
    )
    flags = torch.tensor([[True, True, True], [True, False, False], [False, False, False]])
    batch = torch.tensor([0, 1, 2, 0, 1, 2])
    expected = reference(pos.double().numpy(), cell.double().numpy(), flags.numpy(), batch.numpy(), cutoff, cap)
    actual = neighbor_list_pbc(pos, cutoff, batch, cap, cell=cell, pbc=flags)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cap", [1, 4, 128])
def test_open_and_unified_agree(cap):
    pos = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 0, 0], [2, 0, 0]])
    batch = torch.tensor([3, 3, 0, 3, 3, 0])
    expected = reference(pos.numpy(), np.zeros((4, 3, 3)), np.zeros((4, 3), bool), batch.numpy(), 2.0, cap)[0]
    assert radius_graph is neighbor_list_obc
    torch.testing.assert_close(neighbor_list_obc(pos, 2.0, batch, cap), expected)
    edge, shifts = neighbor_list(pos, 2.0, batch, cap)
    torch.testing.assert_close(edge, expected)
    torch.testing.assert_close(shifts, torch.zeros((expected.shape[1], 3), dtype=torch.long))
    edge, shifts = neighbor_list_pbc(pos, 2.0, batch, cap, cell=None, pbc=False)
    torch.testing.assert_close(edge, expected)
    assert not shifts.any()


def test_empty_singleton_and_periodic_self_images():
    for pos in (torch.empty((0, 3)), torch.zeros((1, 3))):
        assert neighbor_list_obc(pos, 1).shape == (2, 0)
    edge, shift = neighbor_list_pbc(torch.zeros((1, 3)), 1.1, None, 32, cell=torch.eye(3), pbc=True)
    assert edge.shape == (2, 6)
    assert torch.equal(shift.abs().sum(1), torch.ones(6, dtype=torch.long))
    assert neighbor_list_pbc(torch.empty((0, 3)), 1, None, 32, cell=torch.eye(3), pbc=True)[0].shape == (2, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cutoff=0),
        dict(cutoff=float("nan")),
        dict(max_neighbors=0),
        dict(batch=torch.tensor([-1])),
        dict(batch=torch.tensor([0.0])),
        dict(positions=torch.tensor([[float("nan"), 0, 0]])),
    ],
)
def test_invalid_open(kwargs):
    arguments = dict(positions=torch.zeros((1, 3)), cutoff=1.0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        neighbor_list_obc(**arguments)


@pytest.mark.parametrize(
    "cell,mask,batch",
    [
        (None, True, None),
        (torch.zeros(3, 3), True, None),
        (torch.eye(3), torch.ones(2), None),
        (torch.eye(3), torch.ones(2, 3), torch.tensor([2])),
        (torch.eye(3).repeat(2, 1, 1), True, torch.tensor([2])),
    ],
)
def test_invalid_periodic(cell, mask, batch):
    with pytest.raises(ValueError):
        neighbor_list_pbc(torch.zeros((1, 3)), 1.0, batch, 32, cell=cell, pbc=mask)


def test_skewed_cell_winding_and_strided_autograd_input():
    generator = torch.Generator().manual_seed(734)
    cell = torch.tensor([[3.0, 0, 0], [2.5, 2.0, 0], [-1.0, 1.0, 3.0]], dtype=torch.float64)
    fractional = torch.rand((9, 3), generator=generator, dtype=torch.float64)
    pos = fractional @ cell
    storage = torch.zeros((9, 6), dtype=torch.float64)
    storage[:, ::2] = pos
    strided = storage[:, ::2].requires_grad_()
    flags = torch.tensor([True, True, False])
    batch = torch.zeros(9, dtype=torch.long)
    expected = reference(pos.numpy(), cell[None].numpy(), flags[None].numpy(), batch.numpy(), 3.5, 32)
    actual = neighbor_list_pbc(strided, 3.5, batch, 32, cell=cell, pbc=flags)
    torch.testing.assert_close(actual, expected)
    winding = torch.randint(-100, 100, (9, 3), generator=generator)
    winding[:, 2] = 0
    edge, offset = neighbor_list_pbc(pos + winding.double() @ cell, 3.5, batch, 32, cell=cell, pbc=flags)
    torch.testing.assert_close(edge, actual[0])
    torch.testing.assert_close(offset, actual[1] + winding[edge[0]] - winding[edge[1]])


def test_strict_cutoff_float64():
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    assert neighbor_list_obc(pos, 1.0).shape == (2, 0)
    assert neighbor_list_obc(pos, np.nextafter(1.0, 2.0)).shape == (2, 2)
    edge, _ = neighbor_list_pbc(pos, 1.0, None, 32, cell=torch.eye(3) * 10, pbc=True)
    assert edge.shape == (2, 0)


@pytest.mark.parametrize("size", [1, 16, 63, 64, 65, 128])
@pytest.mark.parametrize("cap", [1, 32, 128])
def test_mixed_sizes_preserve_nearest_order_and_batch_isolation(size, cap):
    rng = np.random.default_rng(734)
    # Integer coordinates produce exact ties and duplicate positions; sparse,
    # interleaved IDs exercise source-order tie breaking in both query paths.
    pos = torch.from_numpy(rng.integers(-3, 4, (size + 7, 3)).astype(np.float64))
    batch = torch.tensor([9] * size + [2] * 7)
    permutation = torch.from_numpy(rng.permutation(len(pos)))
    pos, batch = pos[permutation], batch[permutation]
    expected = reference(pos.numpy(), np.zeros((10, 3, 3)), np.zeros((10, 3), bool), batch.numpy(), 4.0, cap)[0]
    actual = neighbor_list_obc(pos, 4.0, batch, cap)
    torch.testing.assert_close(actual, expected)
    periodic, offsets = neighbor_list_pbc(pos, 4.0, batch, cap, cell=None, pbc=False)
    torch.testing.assert_close(periodic, actual)
    assert not offsets.any()


@pytest.mark.parametrize("cap,count", [(1, 20), (32, 100), (128, 300)])
def test_periodic_nearest_query_keeps_all_boundary_ties(cap, count):
    # More coincident sources than the initial nearest query can hold. The
    # selected source IDs must still be deterministic, excluding only self.
    pos = torch.zeros((count, 3), dtype=torch.float64)
    edge, offsets = neighbor_list_pbc(pos, 1.0, None, cap, cell=torch.eye(3) * 10, pbc=True)
    expected = [
        (source, target) for target in range(count) for source in [i for i in range(count) if i != target][:cap]
    ]
    torch.testing.assert_close(edge, torch.tensor(expected).T)
    assert not offsets.any()


@pytest.mark.parametrize("cap", [1, 6, 32])
def test_periodic_nearest_query_orders_periodic_image_ties(cap):
    pos = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)
    flags = torch.ones(3, dtype=torch.bool)
    expected = reference(pos.numpy(), cell[None].numpy(), flags[None].numpy(), np.zeros(1, dtype=int), 3.1, cap)
    torch.testing.assert_close(neighbor_list_pbc(pos, 3.1, None, cap, cell=cell, pbc=flags), expected)


def test_periodic_chunked_images_match_independent_enumeration():
    # Small lattice forces more than one image-generation block; includes
    # nonperiodic axes and nonzero winding in the original coordinates.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.diag(torch.tensor([0.25, 3.0, 4.0], dtype=torch.float64))
    edge, shift = neighbor_list_pbc(pos, 2100.0, None, 8, cell=cell, pbc=torch.tensor([True, False, False]))
    # Enumerate in wrapped coordinates to include original winding correctly.
    wrapped = torch.zeros((2, 3), dtype=torch.float64)
    expected_edge, expected_shift = reference(
        wrapped.numpy(), cell[None].numpy(), np.array([[True, False, False]]), np.zeros(2, dtype=int), 1.0, 8
    )
    winding = torch.tensor([[0, 0, 0], [5, 0, 0]])
    torch.testing.assert_close(edge, expected_edge)
    torch.testing.assert_close(shift, expected_shift + winding[edge[0]] - winding[edge[1]])
