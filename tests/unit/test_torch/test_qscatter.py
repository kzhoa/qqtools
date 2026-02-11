import time
from typing import Optional

import torch
from torch import Tensor


def broadcast(src: Tensor, other: Tensor, dim: int):
    if src.dim() == 1:
        for _ in range(other.dim() - 1):
            src = src.unsqueeze(-1)
    return src.expand_as(other)


def scatter_mean1(ref: Tensor, index: Tensor, dim: int, dim_size: Optional[int] = None):
    if dim < 0:
        dim = ref.dim() + dim
    if dim_size is None:
        dim_size = int(torch.max(index)) + 1

    _size = list(ref.shape)
    _size[dim] = dim_size

    idx_broadcasted = broadcast(index, ref, dim)

    out = ref.new_zeros(_size)
    out.scatter_add_(dim, idx_broadcasted, ref)

    ones = torch.ones_like(ref)
    count = ref.new_zeros(_size)
    count.scatter_add_(dim, idx_broadcasted, ones)
    count.clamp_(min=1)

    return out / count


def scatter_mean2(ref: Tensor, index: Tensor, dim: int, dim_size: Optional[int] = None):
    """PyTorch native"""
    if dim < 0:
        dim = ref.dim() + dim
    if dim_size is None:
        dim_size = int(torch.max(index)) + 1

    _size = list(ref.shape)
    _size[dim] = dim_size

    idx_broadcasted = broadcast(index, ref, dim)

    out = ref.new_zeros(_size)
    return out.scatter_reduce_(dim, idx_broadcasted, ref, reduce="mean", include_self=False)


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模拟 GNN 场景:
    # ref: [1,000,000 个边/节点, 128 维特征]
    # index: 对应 50,000 个目标槽位
    num_elements = 1_000_000
    num_features = 128
    num_clusters = 50_000

    ref = torch.randn(num_elements, num_features, device=device)
    index = torch.randint(0, num_clusters, (num_elements,), device=device)
    dim = 0

    # 1. 结果校验
    res1 = scatter_mean1(ref, index, dim, num_clusters)
    res2 = scatter_mean2(ref, index, dim, num_clusters)
    # 注意：浮点数计算顺序不同可能有极小误差，使用 allclose
    is_correct = torch.allclose(res1, res2, atol=1e-6)
    print(f"结果校验是否一致: {is_correct}")

    # 2. 性能测试函数
    def benchmark(fn, name, iterations=100):
        # 预热
        for _ in range(10):
            fn(ref, index, dim, num_clusters)

        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            for _ in range(iterations):
                fn(ref, index, dim, num_clusters)

            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / iterations
        else:
            start = time.perf_counter()
            for _ in range(iterations):
                fn(ref, index, dim, num_clusters)
            end = time.perf_counter()
            return (end - start) * 1000 / iterations

    t1 = benchmark(scatter_mean1, "scatter_mean1 (Manual)")
    t2 = benchmark(scatter_mean2, "scatter_mean2 (Native)")

    print(f"Method 1 (Manual) Average Time: {t1:.4f} ms")
    print(f"Method 2 (Native) Average Time: {t2:.4f} ms")
    print(f"Speedup: {t1/t2:.2f}x")


if __name__ == "__main__":
    run_benchmark()
