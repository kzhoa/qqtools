import torch

from qqtools.torch.qmgraph import qtriplets


def build_manual_case(keep_periodic_self_loop: bool = False):
    edge_index = torch.tensor(
        [
            [0, 2, 1, 0, 1, 2, 3],
            [1, 1, 2, 2, 0, 0, 3],
        ],
        dtype=torch.long,
    )
    cell_offsets = torch.zeros((edge_index.size(1), 3), dtype=torch.long)
    if keep_periodic_self_loop:
        cell_offsets[0, 0] = 1
        cell_offsets[4, 0] = -1
    return edge_index, cell_offsets, 4


def canonicalize_result(result):
    col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji = result
    num_edges = row.numel()
    combined_id = idx_ji.long() * num_edges + idx_kj.long()
    sort_idx = torch.argsort(combined_id)
    return (
        col,
        row,
        idx_i[sort_idx],
        idx_j[sort_idx],
        idx_k[sort_idx],
        idx_kj[sort_idx],
        idx_ji[sort_idx],
    )


def test_qtriplets_returns_expected_triplets_for_manual_graph():
    edge_index, cell_offsets, num_nodes = build_manual_case(keep_periodic_self_loop=False)

    col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji = canonicalize_result(
        qtriplets(edge_index, cell_offsets, num_nodes)
    )

    assert torch.equal(col, torch.tensor([1, 1, 2, 2, 0, 0, 3]))
    assert torch.equal(row, torch.tensor([0, 2, 1, 0, 1, 2, 3]))
    assert torch.equal(idx_i, torch.tensor([1, 1, 2, 2, 0, 0]))
    assert torch.equal(idx_j, torch.tensor([0, 2, 1, 0, 1, 2]))
    assert torch.equal(idx_k, torch.tensor([1, 1, 2, 2, 0, 0]))
    assert torch.equal(idx_kj, torch.tensor([4, 4, 1, 1, 3, 3]))
    assert torch.equal(idx_ji, torch.tensor([0, 5, 2, 5, 0, 5]))


def test_qtriplets_removes_zero_offset_return_triplets():
    edge_index, cell_offsets, num_nodes = build_manual_case(keep_periodic_self_loop=False)

    _, _, idx_i, _, idx_k, _, _ = qtriplets(edge_index, cell_offsets, num_nodes)

    assert not torch.any(idx_i == idx_k)


def test_qtriplets_keeps_periodic_return_triplets_with_nonzero_offset():
    edge_index, cell_offsets, num_nodes = build_manual_case(keep_periodic_self_loop=True)

    _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = canonicalize_result(
        qtriplets(edge_index, cell_offsets, num_nodes)
    )

    assert torch.any(idx_i == idx_k)
    periodic_mask = idx_i == idx_k
    assert torch.equal(idx_i[periodic_mask], torch.tensor([0]))
    assert torch.equal(idx_j[periodic_mask], torch.tensor([1]))
    assert torch.equal(idx_k[periodic_mask], torch.tensor([0]))
    assert torch.equal(idx_kj[periodic_mask], torch.tensor([4]))
    assert torch.equal(idx_ji[periodic_mask], torch.tensor([0]))


def test_qtriplets_returns_empty_triplets_when_graph_has_no_chain():
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=torch.long,
    )
    cell_offsets = torch.zeros((2, 3), dtype=torch.long)

    _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = qtriplets(edge_index, cell_offsets, num_nodes=3)

    assert idx_i.numel() == 0
    assert idx_j.numel() == 0
    assert idx_k.numel() == 0
    assert idx_kj.numel() == 0
    assert idx_ji.numel() == 0
