"""Exact CUDA top-k radius graph for open boundaries."""

import math
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _select(d2, source, valid, K: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    if d2.dtype == tl.float32:
        key = (d2.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | source.to(tl.uint64)
        candidates = tl.where(valid, ~key, 0)
        if K == 1:
            winners = tl.max(candidates, 0) + tl.full((1,), 0, tl.uint64)
        else:
            winners = tl.topk(candidates, K)
        return tl.where(winners != 0, (~winners).to(tl.uint32).to(tl.int32), -1)
    else:
        # Explicit lexicographic compare avoids quantizing FP64 distances.
        dist = tl.where(valid, d2, float("inf"))
        ids = tl.where(valid, source, -1).to(tl.int32)
        lane = tl.arange(0, B)
        for level in tl.static_range(1, LOG + 1):
            for step in tl.static_range(level):
                stride = 1 << (level - 1 - step)
                partner = lane ^ stride
                other = tl.gather(dist, partner, 0)
                idx = tl.gather(ids, partner, 0)
                take_min = ((lane & (1 << level)) == 0) == ((lane & stride) == 0)
                less = (other < dist) | ((other == dist) & (idx < ids))
                swap = take_min == less
                dist = tl.where(swap, other, dist)
                ids = tl.where(swap, idx, ids)
        return tl.gather(ids, tl.arange(0, K), 0)


@triton.jit
def _open(pos, order, starts, ends, r2, out, K: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    t = tl.program_id(0)
    offset = tl.load(starts + t) + tl.arange(0, B)
    valid = offset < tl.load(ends + t)
    src = tl.load(order + offset, valid, 0)
    dx = tl.load(pos + src * 3) - tl.load(pos + t * 3)
    dy = tl.load(pos + src * 3 + 1) - tl.load(pos + t * 3 + 1)
    dz = tl.load(pos + src * 3 + 2) - tl.load(pos + t * 3 + 2)
    d2 = tl.fma(dz, dz, tl.fma(dx, dx, dy * dy))
    selected = _select(d2, src, valid & (src != t) & (d2 < tl.load(r2)), K, B, LOG)
    tl.store(out + t * K + tl.arange(0, K), selected)


@torch.no_grad()
def open_radius_graph(
    positions: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None = None,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Return incoming CUDA FP32/FP64 edges sorted by distance and source ID."""
    pos = positions
    k = operator.index(max_neighbors)
    if not pos.is_cuda or pos.ndim != 2 or pos.shape[1] != 3 or pos.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected FP32 or FP64 CUDA positions")
    if batch is None:
        batch = torch.zeros(len(pos), device=pos.device, dtype=torch.long)
    if batch.shape != (len(pos),) or batch.dtype != torch.long or batch.device != pos.device:
        raise ValueError("Invalid graph IDs")
    if not math.isfinite(cutoff) or cutoff <= 0 or k < 1:
        raise ValueError("Invalid cutoff or neighbor cap")
    if len(pos) and not bool(pos.isfinite().all()):
        raise ValueError("Expected finite positions")
    if len(pos) < 2:
        return batch.new_empty((2, 0))
    groups, order = batch.sort(stable=True)
    starts = torch.searchsorted(groups, batch.contiguous())
    ends = torch.searchsorted(groups, batch.contiguous(), right=True)
    maximum = int((ends - starts).max())
    cap = min(k, maximum)
    K = triton.next_power_of_2(cap)
    B = triton.next_power_of_2(maximum)
    if B > 65536:
        raise ValueError("Graph exceeds Triton tile limit")
    selected = torch.empty((len(pos), K), device=pos.device, dtype=torch.int32)
    with torch.cuda.device(pos.device):
        _open[(len(pos),)](
            pos.contiguous(), order, starts, ends, pos.new_tensor(cutoff**2), selected, K, B, B.bit_length() - 1
        )
    selected = selected[:, :cap]
    row, slot = (selected >= 0).nonzero(as_tuple=True)
    return torch.stack((selected[row, slot].long(), row))
