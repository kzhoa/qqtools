import random
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

import qqtools as qt


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)


def scatter_mean0(
    ref: Tensor,
    index: Tensor,
    dim: int,
    dim_size: Optional[int] = None,
):
    """count before sum, then divide with broadcast"""
    if dim < 0:
        dim = torch.add(ref.dim(), dim)

    if dim < 0 or dim >= ref.dim():
        raise ValueError(f"dim out of range, got dim={dim}, but _ref.shape{ref.shape}")

    # handle _dim_size
    assert index.numel() > 0, "expect _index not empty"

    if dim_size is None:
        dim_size = torch.add(torch.max(index).to(torch.int64), 1)

    # handle output _size
    _size = list(ref.shape)
    _size[dim] = dim_size

    count = ref.new_zeros(dim_size)
    count.scatter_add_(0, index, ref.new_ones(ref.size(dim)))
    count = count.clamp(min=1)

    index = broadcast(index, ref, dim)
    out = ref.new_zeros(_size)
    out = out.scatter_add_(dim, index, ref)

    return out / broadcast(count, out, dim)


def scatter_mean1(
    ref: Tensor,
    index: Tensor,
    dim: int,
    dim_size: Optional[int] = None,
):
    """sum, then fullsize count, then divide"""
    if dim < 0:
        dim = torch.add(ref.dim(), dim)

    if dim < 0 or dim >= ref.dim():
        raise ValueError(f"dim out of range, got dim={dim}, but _ref.shape{ref.shape}")

    # handle _dim_size
    assert index.numel() > 0, "expect _index not empty"

    if dim_size is None:
        dim_size = torch.add(torch.max(index).to(torch.int64), 1)

    # handle output _size
    _size = list(ref.shape)
    _size[dim] = dim_size

    index = broadcast(index, ref, dim)
    out = ref.new_zeros(_size)
    out = out.scatter_add_(dim, index, ref)

    ones = ref.new_ones(ref.shape)
    count = ref.new_zeros(_size)
    count.scatter_add_(dim, index, ones)
    count.clamp_(min=1)

    return out / count


def scatter_mean2(
    ref: Tensor,
    index: Tensor,
    dim: int,
    dim_size: Optional[int] = None,
):
    """PyTorch native"""
    if dim < 0:
        dim = ref.dim() + dim

    if dim < 0 or dim >= ref.dim():
        raise ValueError(f"dim out of range, got dim={dim}, but ref.shape {ref.shape}")

    assert index.numel() > 0, "expect index not empty"

    if dim_size is None:
        dim_size = torch.add(torch.max(index).to(torch.int64), 1)

    _size = list(ref.shape)
    _size[dim] = dim_size

    index = broadcast(index, ref, dim)
    out = ref.new_zeros(_size)

    return out.scatter_reduce_(dim, index, ref, reduce="mean", include_self=False)


# ----- mock data -----


def generate_scatter_data(num_nodes: int, feature_dim: int, target_groups: int, dim_for_scatter: int = 0):

    ref_tensor = torch.randn(num_nodes, feature_dim)

    if dim_for_scatter == 0:
        # randomly assign each node to a target group
        index_values = [random.randint(0, target_groups - 1) for _ in range(num_nodes)]
        index_tensor = torch.tensor(index_values, dtype=torch.long)
        actual_dim_size = target_groups
    else:
        raise NotImplementedError("Currently only supports dim_for_scatter=0")

    return ref_tensor, index_tensor, actual_dim_size


def benchmark_scatter(num_nodes=1_0000, feature_dim=256, target_groups=32, n_iters=100):

    ref, index, dim_size = generate_scatter_data(
        num_nodes,
        feature_dim=feature_dim,
        target_groups=target_groups,
        dim_for_scatter=0,
    )
    print("Verifying correctness...")
    try:
        res0 = scatter_mean0(ref, index, dim=0, dim_size=dim_size)
        res1 = scatter_mean1(ref, index, dim=0, dim_size=dim_size)
        res2 = scatter_mean2(ref, index, dim=0, dim_size=dim_size)

        is_correct0 = torch.allclose(res0, res1, atol=1e-5)
        is_correct1 = torch.allclose(res1, res2, atol=1e-5)

        assert is_correct0, "Results mismatch: scatter_mean0 vs scatter_mean1"
        assert is_correct1, "Results mismatch: scatter_mean1 vs scatter_mean2"
        print("Correctness check passed.")
    except Exception as e:
        pytest.fail(f"Verification step failed: {e}")

    cpu_times: Dict[str, float] = {}
    gpu_times: Dict[str, float] = {}

    # CPU Benchmarking
    print("Benchmarking CPU...")
    for name, func in [
        ("scatter_mean0", scatter_mean0),
        ("scatter_mean1", scatter_mean1),
        ("scatter_mean2", scatter_mean2),
    ]:
        for _ in range(5):  # warm-up runs
            func(ref, index, dim=0, dim_size=dim_size)

        with qt.Timer(cuda=False, verbose=False) as t:
            for _ in range(n_iters):
                func(ref, index, dim=0, dim_size=dim_size)
        cpu_times[name] = t.duration * 1000  # ms

    # GPU Benchmarking
    if torch.cuda.is_available():
        print("Benchmarking GPU...")
        ref_gpu = ref.to("cuda")
        index_gpu = index.to("cuda")

        for name, func in [
            ("scatter_mean0", scatter_mean0),
            ("scatter_mean1", scatter_mean1),
            ("scatter_mean2", scatter_mean2),
        ]:
            for _ in range(5):  # warm-up runs
                func(ref, index, dim=0, dim_size=dim_size)

            torch.cuda.synchronize()
            with qt.Timer(cuda=True, verbose=False) as t:
                for _ in range(n_iters):
                    func(ref_gpu, index_gpu, dim=0, dim_size=dim_size)
            gpu_times[name] = t.duration  # Timer returns ms for CUDA
    else:
        print("CUDA not available, skipping GPU benchmarks.")
        gpu_times = {name: float("inf") for name in ["scatter_mean0", "scatter_mean1", "scatter_mean2"]}

    return cpu_times, gpu_times


# --- Pytest Configurations ---

benchmark_configs: List[Tuple[str, int, int, int]] = [
    ("Tiny0", 500, 128, 16),
    ("Tiny1", 500, 256, 16),
    ("Tiny2", 500, 256, 32),
    ("Tiny3", 1_000, 128, 16),
    ("Tiny4", 1_000, 256, 32),
    ("Smaller", 5_000, 256, 32),
    ("Small", 1_0000, 256, 32),
    ("Medium", 5_0000, 512, 64),
    ("Large", 10_0000, 1024, 128),
    ("Larger", 20_0000, 512, 128),
]


def generate_params(configs: List[Tuple[str, int, int, int]]):
    params = []
    for desc, num_nodes, feature_dim, target_groups in configs:
        test_id = f"{desc}_N{num_nodes}_F{feature_dim}_G{target_groups}"
        params.append((test_id, num_nodes, feature_dim, target_groups))
    return params


@pytest.mark.parametrize("test_id, num_nodes, feature_dim, target_groups", generate_params(benchmark_configs))
def test_benchmark_scatter(test_id, num_nodes, feature_dim, target_groups, n_iters=100):
    print(f"\n--- Running benchmark for: {test_id} ---")
    print(f"Config: N={num_nodes}, F={feature_dim}, G={target_groups}")

    cpu_times, gpu_times = benchmark_scatter(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        target_groups=target_groups,
        n_iters=n_iters,
    )

    print(f"[{test_id}] CPU times (ms): {cpu_times}")
    print(f"[{test_id}] GPU times (ms): {gpu_times}")


# --- non pytest function ---
def run_all_benchmarks(configs: List[Tuple[str, int, int, int]], n_iters: int = 100):
    """
    A non-pytest function that directly runs all defined benchmark configurations and prints results.
    """
    all_results = {}
    params = generate_params(configs)

    print(f"Starting benchmarks with n_iters={n_iters}...")  # Replaced logger.info

    for test_id, num_nodes, feature_dim, target_groups in params:
        print(f"\n--- Running benchmark for: {test_id} ---")  # Replaced logger.info
        print(f"Config: N={num_nodes}, F={feature_dim}, G={target_groups}")  # Replaced logger.info

        cpu_times, gpu_times = benchmark_scatter(
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            target_groups=target_groups,
            n_iters=n_iters,
        )
        all_results[test_id] = {"cpu": cpu_times, "gpu": gpu_times}

    print("\n--- Benchmark Summary ---")  # Replaced logger.info
    for test_id, results in all_results.items():
        print(f"\n{test_id}:")
        print(f"  CPU: {results['cpu']}")
        print(f"  GPU: {results['gpu']}")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks(benchmark_configs, n_iters=100)
