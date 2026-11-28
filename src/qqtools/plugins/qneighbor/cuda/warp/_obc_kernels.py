"""Exact tiled Cartesian neighbor search for batch-sorted FP32 CUDA positions."""

import math

import torch
import warp as wp


@wp.func_native("return __float_as_uint(distance2);")
def _float_bits(distance2: float) -> wp.uint32: ...


@wp.func
def _distance_key(pos: wp.array(dtype=wp.vec3), dst: int, src: int, end: int, cutoff2: float):
    if src < end and src != dst:
        delta = pos[src] - pos[dst]
        squared = wp.dot(delta, delta)
        if squared < cutoff2:
            # Nonnegative float bits have the same ordering as their numeric values.
            return (wp.uint64(_float_bits(squared)) << wp.uint64(32)) | wp.uint64(src)
    return wp.uint64(0xFFFFFFFFFFFFFFFF)


@wp.kernel(enable_backward=False)
def _select_neighbors(
    pos: wp.array(dtype=wp.vec3),
    batch: wp.array(dtype=wp.int64),
    cutoff2: float,
    k: int,
    slots: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
):
    dst, lane = wp.tid()
    group = batch[dst]
    start = wp.lower_bound(batch, group)
    end = batch.shape[0]
    if group != batch[end - 1]:
        end = wp.lower_bound(batch, group + wp.int64(1))
    src = start + lane
    keys = wp.tile(_distance_key(pos, dst, src, end, cutoff2))
    values = wp.tile(src)
    wp.tile_sort(keys, values)
    best_key = wp.untile(keys)
    best_src = wp.untile(values)

    # Retain the previous k winners and merge the next block of candidates.
    next_src = start + wp.block_dim()
    while next_src < end:
        if lane >= k:
            best_src = next_src + lane - k
            best_key = _distance_key(pos, dst, best_src, end, cutoff2)
        keys_next = wp.tile(best_key)
        values_next = wp.tile(best_src)
        wp.tile_sort(keys_next, values_next)
        best_key = wp.untile(keys_next)
        best_src = wp.untile(values_next)
        next_src += wp.block_dim() - k
    valid = lane < k and best_key != wp.uint64(0xFFFFFFFFFFFFFFFF)
    count = wp.tile_sum(wp.tile(wp.int32(valid)))
    if lane == 0:
        counts[dst] = count[0]
    if lane < k:
        slots[dst, lane] = wp.where(valid, best_src, -1)


@wp.kernel(enable_backward=False)
def _compact(slots: wp.array2d(dtype=wp.int32), ends: wp.array(dtype=wp.int32), edges: wp.array2d(dtype=wp.int64)):
    dst, rank = wp.tid()
    src = slots[dst, rank]
    if src >= 0:
        start = int(0)
        if dst > 0:
            start = ends[dst - 1]
        edges[0, start + rank] = wp.int64(src)
        edges[1, start + rank] = wp.int64(dst)


@torch.no_grad()
def open_edges(positions, batch, cutoff, k, block_size=64):
    """Build COO edges for finite positions grouped by sorted batch IDs.

    Nodes at exactly the cutoff are excluded. Equal squared distances use source ID order.
    """
    if not positions.is_cuda or positions.dtype != torch.float32 or positions.shape != (len(batch), 3):
        raise ValueError("Expected one FP32 CUDA position of shape (3,) per node")
    if batch.dtype != torch.long or batch.ndim != 1 or batch.device != positions.device:
        raise ValueError("Expected int64 batch IDs on the same GPU as positions")
    if block_size not in (32, 64, 128, 256) or not 0 < k < block_size:
        raise ValueError("Expected 0 < k < block_size with 32, 64, 128 or 256 threads")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Expected a positive finite cutoff")
    n = len(positions)
    if n == 0:
        return batch.new_empty((2, 0))
    if n * k > torch.iinfo(torch.int32).max:
        raise ValueError("Edge capacity exceeds the int32 compact index range")
    wp.init()
    slots = torch.empty((n, k), dtype=torch.int32, device=positions.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
    counts = torch.empty(n, dtype=torch.int32, device=positions.device)
    slot_array = wp.from_torch(slots, return_ctype=True)
    wp.launch(
        _select_neighbors,
        dim=(n, block_size),
        block_dim=block_size,
        inputs=[
            wp.from_torch(positions.contiguous(), dtype=wp.vec3, return_ctype=True),
            wp.from_torch(batch.contiguous(), return_ctype=True),
            cutoff**2,
            k,
            slot_array,
            wp.from_torch(counts, return_ctype=True),
        ],
        stream=stream,
    )
    ends = torch.cumsum(counts, dim=0, dtype=torch.int32)
    total = int(ends[-1])
    edges = batch.new_empty((2, total))
    if total:
        wp.launch(
            _compact,
            dim=(n, k),
            inputs=[
                slot_array,
                wp.from_torch(ends, return_ctype=True),
                wp.from_torch(edges, return_ctype=True),
            ],
            stream=stream,
        )
    return edges
