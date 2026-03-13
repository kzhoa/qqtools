"""
tested by qq
date: 2026-03-13

results:
+--------+---------------+-------+--------+-------------+-----------+---------+
| Device | Case          | Nodes | AvgDeg | Sparse (ms) | Pure (ms) | Speedup |
+--------+---------------+-------+--------+-------------+-----------+---------+
| CUDA   | small_sparse  |  2000 |      8 |     36.7875 |   32.2134 |   1.14x |
| CUDA   | small_dense   |  2000 |     24 |     37.1302 |   32.3654 |   1.15x |
| CUDA   | medium_sparse |  5000 |     12 |     37.0028 |   32.2280 |   1.15x |
| CUDA   | medium_dense  |  5000 |     32 |     39.0903 |   34.6479 |   1.13x |
| CUDA   | large_sparse  | 10000 |     20 |     38.6570 |   34.1343 |   1.13x |
| CUDA   | large_dense   | 10000 |     40 |     47.9191 |   46.3588 |   1.03x |
+--------+---------------+-------+--------+-------------+-----------+---------+
"""

import torch

try:
    from torch_sparse import SparseTensor
except ImportError as exc:
    SparseTensor = None
    TORCH_SPARSE_IMPORT_ERROR = exc
else:
    TORCH_SPARSE_IMPORT_ERROR = None

from qqtools.qtimer import Timer

BENCHMARK_CONFIGS = [
    ("small_sparse", 2_000, 8),
    ("small_dense", 2_000, 24),
    ("medium_sparse", 5_000, 12),
    ("medium_dense", 5_000, 32),
    ("large_sparse", 10_000, 20),
    ("large_dense", 10_000, 40),
]


def ensure_runtime_prerequisites():
    if TORCH_SPARSE_IMPORT_ERROR is not None or SparseTensor is None:
        raise RuntimeError(
            "torch_sparse is required for qtriplets comparison; SparseTensor is unavailable, "
            "so the sparse vs pure benchmark cannot run."
        ) from TORCH_SPARSE_IMPORT_ERROR

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA support is required for qtriplets benchmark, but torch.cuda.is_available() is False.")


def triplets_sparse(edge_index, cell_offsets, num_nodes):
    row, col = edge_index  # j -> i
    value = torch.arange(row.size(0), device=row.device)

    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices for triplets k -> j -> i.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Edge indices for k -> j and j -> i.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()

    # Remove d -> b -> d self-loop triplets unless periodic offsets differ.
    cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
    mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def triplets_pure(edge_index, cell_offsets, num_nodes):
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

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def generate_mock_data(num_nodes=5000, avg_degree=30, device="cpu", seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    num_edges = num_nodes * avg_degree
    row = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    col = torch.randint(0, num_nodes, (num_edges,), generator=generator)

    mask = row != col
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0).to(device)

    num_edges = edge_index.size(1)
    cell_offsets = torch.randint(-1, 2, (num_edges, 3), generator=generator).to(device)

    return edge_index, cell_offsets, num_nodes


def canonicalize_result(result, num_edges):
    col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji = result
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


def assert_triplets_match(edge_index, cell_offsets, num_nodes, case_name):
    sparse_result = canonicalize_result(
        triplets_sparse(edge_index, cell_offsets, num_nodes),
        edge_index.size(1),
    )
    pure_result = canonicalize_result(
        triplets_pure(edge_index, cell_offsets, num_nodes),
        edge_index.size(1),
    )

    tensor_names = ["col", "row", "idx_i", "idx_j", "idx_k", "idx_kj", "idx_ji"]
    for name, sparse_tensor, pure_tensor in zip(tensor_names, sparse_result, pure_result):
        assert torch.equal(sparse_tensor, pure_tensor), f"{case_name}: mismatch found in {name}"


def build_manual_case(device, keep_periodic_self_loop):
    edge_index = torch.tensor(
        [
            [0, 2, 1, 0, 1, 2, 3],
            [1, 1, 2, 2, 0, 0, 3],
        ],
        dtype=torch.long,
        device=device,
    )
    cell_offsets = torch.zeros((edge_index.size(1), 3), dtype=torch.long, device=device)
    if keep_periodic_self_loop:
        cell_offsets[0, 0] = 1
        cell_offsets[4, 0] = -1
    return edge_index, cell_offsets, 4


def run_correctness_suite(device):
    print(f"\n[{device.upper()}] Running correctness checks...")

    deterministic_cases = [
        ("manual_self_loop_filtered", *build_manual_case(device, keep_periodic_self_loop=False)),
        ("manual_self_loop_preserved", *build_manual_case(device, keep_periodic_self_loop=True)),
    ]

    for case_name, edge_index, cell_offsets, num_nodes in deterministic_cases:
        assert_triplets_match(edge_index, cell_offsets, num_nodes, case_name)

    random_cases = [
        ("random_small_sparse", 32, 4, 7),
        ("random_medium_sparse", 128, 8, 11),
        ("random_dense", 96, 16, 19),
    ]
    for case_name, num_nodes, avg_degree, seed in random_cases:
        edge_index, cell_offsets, graph_num_nodes = generate_mock_data(
            num_nodes=num_nodes,
            avg_degree=avg_degree,
            device=device,
            seed=seed,
        )
        assert_triplets_match(edge_index, cell_offsets, graph_num_nodes, case_name)

    print(f"[{device.upper()}] All correctness checks passed.")


def time_func(func, edge_index, cell_offsets, num_nodes, device, num_warmup, num_iters):
    for _ in range(num_warmup):
        func(edge_index, cell_offsets, num_nodes)

    use_cuda_timer = device == "cuda"
    with Timer(cuda=use_cuda_timer, verbose=False) as timer:
        for _ in range(num_iters):
            func(edge_index, cell_offsets, num_nodes)

    return timer.duration * 1000.0 / num_iters


def benchmark(device, num_nodes=10000, avg_degree=40, seed=101, num_warmup=10, num_iters=100):
    edge_index, cell_offsets, num_nodes = generate_mock_data(
        num_nodes=num_nodes,
        avg_degree=avg_degree,
        device=device,
        seed=seed,
    )

    time_sparse = time_func(
        triplets_sparse,
        edge_index,
        cell_offsets,
        num_nodes,
        device,
        num_warmup,
        num_iters,
    )
    time_pure = time_func(
        triplets_pure,
        edge_index,
        cell_offsets,
        num_nodes,
        device,
        num_warmup,
        num_iters,
    )

    return time_sparse, time_pure


def run_benchmark_suite(device, num_warmup=5, num_iters=10):
    results = []

    for index, (case_name, num_nodes, avg_degree) in enumerate(BENCHMARK_CONFIGS):
        print(
            f"[{device.upper()}] Benchmarking {case_name} "
            f"(num_nodes={num_nodes}, avg_degree={avg_degree}, iterations={num_iters})..."
        )
        time_sparse, time_pure = benchmark(
            device=device,
            num_nodes=num_nodes,
            avg_degree=avg_degree,
            seed=101 + index,
            num_warmup=num_warmup,
            num_iters=num_iters,
        )
        results.append(
            {
                "device": device.upper(),
                "case_name": case_name,
                "num_nodes": num_nodes,
                "avg_degree": avg_degree,
                "time_sparse": time_sparse,
                "time_pure": time_pure,
                "speedup": time_sparse / time_pure,
            }
        )

    return results


def print_results_table(results):
    print("\n" + "=" * 94)
    print(
        f"{'Device':<8} | {'Case':<14} | {'Nodes':>8} | {'AvgDeg':>6} | "
        f"{'Sparse (ms)':>11} | {'Pure (ms)':>10} | {'Speedup':>8}"
    )
    print("-" * 94)

    for result in results:
        color = "\033[92m" if result["speedup"] >= 1 else "\033[91m"
        reset = "\033[0m"
        speedup_str = f"{result['speedup']:.2f}x"
        print(
            f"{result['device']:<8} | {result['case_name']:<14} | {result['num_nodes']:>8} | "
            f"{result['avg_degree']:>6} | {result['time_sparse']:>11.4f} | "
            f"{result['time_pure']:>10.4f} | {color}{speedup_str:>8}{reset}"
        )

    print("=" * 94)


if __name__ == "__main__":
    ensure_runtime_prerequisites()
    devices = ["cuda"]

    results = []

    for device in devices:
        run_correctness_suite(device)
        results.extend(run_benchmark_suite(device, num_warmup=5, num_iters=10))

    print_results_table(results)
