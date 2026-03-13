from typing import Tuple

import torch


def qtriplets(
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build graph triplets with native PyTorch tensor operations.

    The input edge list follows the `j -> i` convention used by many message-passing
    pipelines. For each edge `j -> i`, this function enumerates all incoming edges
    `k -> j` and produces candidate triplets `k -> j -> i`.

    Triplets that collapse into `d -> b -> d` are removed when the combined periodic
    offset is zero. If the summed offset is non-zero, the triplet is kept because it
    represents a valid periodic-image interaction.

    Parameters
    ----------
    edge_index : torch.Tensor
        Integer tensor with shape `(2, num_edges)`. The first row stores source node
        indices `j`, and the second row stores target node indices `i`.
    cell_offsets : torch.Tensor
        Integer tensor with shape `(num_edges, 3)` describing the periodic cell offset
        attached to each directed edge.
    num_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    tuple[torch.Tensor, ...]
        A 7-tuple `(col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji)` where:

        - `col`: target node indices `i` for the original edges.
        - `row`: source node indices `j` for the original edges.
        - `idx_i`: triplet target node indices `i`.
        - `idx_j`: triplet center node indices `j`.
        - `idx_k`: triplet source node indices `k`.
        - `idx_kj`: edge indices for `k -> j`.
        - `idx_ji`: edge indices for `j -> i`.
    """
    row, col = edge_index  # j -> i
    num_edges = row.size(0)

    deg_in = torch.bincount(col, minlength=num_nodes)
    num_triplets = deg_in[row]

    idx_ji = torch.arange(num_edges, device=row.device).repeat_interleave(num_triplets)
    idx_j = row[idx_ji]
    idx_i = col[idx_ji]

    _, sort_idx_kj = torch.sort(col)

    ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=row.device)
    torch.cumsum(deg_in, dim=0, out=ptr[1:])

    starts = ptr[row].repeat_interleave(num_triplets)

    count_ptr = torch.zeros(num_edges + 1, dtype=torch.long, device=row.device)
    torch.cumsum(num_triplets, dim=0, out=count_ptr[1:])
    group_starts = count_ptr[:-1].repeat_interleave(num_triplets)
    local_idx = torch.arange(count_ptr[-1], device=row.device) - group_starts

    idx_kj = sort_idx_kj[starts + local_idx]
    idx_k = row[idx_kj]

    cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
    mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    idx_k = idx_k[mask]
    idx_kj = idx_kj[mask]
    idx_ji = idx_ji[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
