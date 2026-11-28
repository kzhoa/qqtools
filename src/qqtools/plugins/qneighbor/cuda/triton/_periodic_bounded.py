"""Periodic radius graph with automatic whole, streaming or spatial queries."""

import math
import operator

import torch
import triton
import triton.language as tl

from ._periodic_whole import _images
from ._periodic_whole import build_pbc as _whole_neighbors
from .obc import open_radius_graph


@triton.jit
def _visit(
    pos,
    target,
    owners,
    trans,
    shifts,
    order,
    start,
    end,
    r2,
    best,
    best_id,
    threshold,
    K: tl.constexpr,
    B: tl.constexpr,
    SORTED: tl.constexpr,
):
    lane = tl.arange(0, B)
    tx = tl.load(pos + target * 3)
    ty = tl.load(pos + target * 3 + 1)
    tz = tl.load(pos + target * 3 + 2)
    for cursor in range(start, end, B - K):
        j = cursor + lane - K
        active = (lane >= K) & (j < end)
        if SORTED:
            ghost = tl.load(order + j, active, 0)
        else:
            ghost = tl.where(active, j, 0)
        src = tl.load(owners + ghost, active, 0)
        dx = tx - tl.load(pos + src * 3) + tl.load(trans + 3 * ghost, active, 0.0)
        dy = ty - tl.load(pos + src * 3 + 1) + tl.load(trans + 3 * ghost + 1, active, 0.0)
        dz = tz - tl.load(pos + src * 3 + 2) + tl.load(trans + 3 * ghost + 2, active, 0.0)
        identity = src == target
        for a in tl.static_range(3):
            identity &= tl.load(shifts + 3 * ghost + a, active, 0) == 0
        d = tl.fma(dz, dz, tl.fma(dx, dx, dy * dy))
        valid = active & ~identity & (d < r2)
        if pos.dtype.element_ty == tl.float32:
            score = ~((d.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | ghost.to(tl.uint64))
            score = tl.where(valid, score, 0)
            if tl.max(score, 0) > threshold:
                best = tl.sort(tl.where(lane < K, best, score), descending=True)
                threshold = tl.max(tl.where(lane == K - 1, best, 0), 0)
        else:
            score = tl.where(valid, ~d.to(tl.uint64, bitcast=True), 0)
            highest = tl.max(score, 0)
            if (highest != 0) & (highest >= threshold):
                best = tl.where(lane < K, best, score)
                best_id = tl.where(lane < K, best_id, tl.where(valid, ghost, -1))
                for level in tl.static_range(1, tl.constexpr(B.bit_length())):
                    for step in tl.static_range(level):
                        partner = lane ^ (1 << (level - 1 - step))
                        other = tl.gather(best, partner, 0)
                        ids = tl.gather(best_id, partner, 0)
                        high = ((lane & (1 << level)) == 0) == ((lane & (1 << (level - 1 - step))) == 0)
                        better = (other > best) | ((other == best) & (ids < best_id))
                        worse = (other < best) | ((other == best) & (ids > best_id))
                        swap = tl.where(high, better, worse)
                        best = tl.where(swap, other, best)
                        best_id = tl.where(swap, ids, best_id)
                threshold = tl.max(tl.where(lane == K - 1, best, 0), 0)
    return best, best_id, threshold


@triton.jit
def _query(
    pos,
    batch,
    ends,
    owners,
    trans,
    shifts,
    order,
    cell_ends,
    base,
    shape,
    target_bins,
    radius,
    selected,
    degrees,
    cap,
    K: tl.constexpr,
    B: tl.constexpr,
    SORTED: tl.constexpr,
):
    target = tl.program_id(0)
    g = tl.load(batch + target)
    best = tl.full((B,), 0, tl.uint64)
    best_id = tl.full((B,), -1, tl.int32)
    threshold = tl.full((), 0, tl.uint64)
    r2 = tl.load(radius)
    if SORTED:
        sx = tl.load(shape + g * 3)
        sy = tl.load(shape + g * 3 + 1)
        sz = tl.load(shape + g * 3 + 2)
        x = tl.load(target_bins + 3 * target)
        y = tl.load(target_bins + 3 * target + 1)
        z = tl.load(target_bins + 3 * target + 2)
        off = tl.load(base + g)
        for xx in range(tl.maximum(0, x - 1), tl.minimum(sx, x + 2)):
            for yy in range(tl.maximum(0, y - 1), tl.minimum(sy, y + 2)):
                col = off + (xx * sy + yy) * sz
                begin = col + tl.maximum(0, z - 1)
                last = col + tl.minimum(sz - 1, z + 1)
                first = tl.load(cell_ends + begin - 1, begin > 0, 0)
                stop = tl.load(cell_ends + last)
                best, best_id, threshold = _visit(
                    pos, target, owners, trans, shifts, order, first, stop, r2, best, best_id, threshold, K, B, SORTED
                )
    else:
        first = tl.load(ends + g - 1, g > 0, 0).to(tl.int32)
        stop = tl.load(ends + g).to(tl.int32)
        best, best_id, threshold = _visit(
            pos, target, owners, trans, shifts, order, first, stop, r2, best, best_id, threshold, K, B, SORTED
        )
    if pos.dtype.element_ty == tl.float32:
        winners = tl.gather(best, tl.arange(0, K), 0)
        chosen = tl.where(winners != 0, (~winners).to(tl.uint32).to(tl.int32), -1)
    else:
        chosen = tl.gather(best_id, tl.arange(0, K), 0)
    tl.store(selected + target * K + tl.arange(0, K), chosen)
    tl.store(degrees + target, tl.sum(((tl.arange(0, K) < cap) & (chosen >= 0)).to(tl.int32), 0))


@triton.jit
def _pack(selected, owners, shifts, ends, edge, offset, nedges, cap, K: tl.constexpr):
    target = tl.program_id(0)
    lane = tl.arange(0, K)
    idx = tl.load(selected + target * K + lane)
    valid = (lane < cap) & (idx >= 0)
    start = tl.load(ends + target - 1, target > 0, 0)
    j = start + lane
    src = tl.load(owners + idx, valid, 0)
    tl.store(edge + j, src, valid)
    tl.store(edge + nedges + j, target, valid)
    for a in tl.static_range(3):
        v = tl.load(shifts + 3 * idx + a, valid, 0)
        tl.store(offset + 3 * j + a, v, valid)


@triton.jit
def _bins(
    pos,
    owners,
    trans,
    batch,
    origin,
    dims,
    base,
    width,
    keys,
    counts,
    target_bins,
    n,
    B: tl.constexpr,
    GHOST: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < n
    if GHOST:
        src = tl.load(owners + i, valid, 0)
    else:
        src = i
    g = tl.load(batch + src, valid, 0)
    key = tl.load(base + g)
    rel = tl.full((B,), 0, tl.int32)
    for a in tl.static_range(3):
        p = tl.load(pos + src * 3 + a, valid, 0.0).to(tl.float64)
        if GHOST:
            p -= tl.load(trans + i * 3 + a, valid, 0.0).to(tl.float64)
        dim = tl.load(dims + g * 3 + a)
        origin_a = tl.load(origin + g * 3 + a)
        b = tl.minimum(tl.maximum(tl.floor((p - origin_a) / tl.load(width + g)), 0.0), (dim - 1).to(tl.float64)).to(
            tl.int32
        )
        rel = rel * dim + b
        if not GHOST:
            tl.store(target_bins + i * 3 + a, b, valid)
    if GHOST:
        tl.store(keys + i, key + rel, valid)
        tl.atomic_add(counts + key + rel, 1, valid, sem="relaxed")


@triton.jit
def _halo_bounds(
    frac, graph, order, lo, hi, span, axes, low, shape, sizes, counts, n, MASKED: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < n
    src = tl.load(order + i, valid, 0)
    g = tl.load(graph + i, valid, 0)
    size = tl.full((B,), 1, tl.int64)
    for a in tl.static_range(3):
        f = tl.load(frac + 3 * src + a)
        pad = tl.load(span + 3 * g + a)
        first = tl.ceil(f - (tl.load(hi + 3 * g + a) + pad))
        last = tl.floor(f - (tl.load(lo + 3 * g + a) - pad))
        if MASKED:
            active = tl.load(axes + 3 * g + a)
            first = tl.where(active, first, 0.0)
            last = tl.where(active, last, 0.0)
        width = last - first + 1
        ok = (first >= -2147483648.0) & (last <= 2147483647.0) & (width >= 1.0) & (width <= 65536.0)
        # Flag unsupported geometry in the existing atom_max host check.
        dim = tl.where(ok, width, 65537.0).to(tl.int32)
        size = tl.minimum(65537, size * dim.to(tl.int64))
        tl.store(low + 3 * i + a, tl.where(ok, first, 0.0).to(tl.int32), valid)
        tl.store(shape + 3 * i + a, dim, valid)
    tl.store(sizes + i, size, valid)
    tl.atomic_add(counts + g, size, valid, sem="relaxed")


@torch.no_grad()
def build_pbc(pos, cell, pbc, batch, cutoff, cap):
    cap = operator.index(cap)
    if cap > 256:
        return _whole_neighbors(pos, cell, pbc, batch, cutoff, cap)
    if (
        not pos.is_cuda
        or pos.dtype not in (torch.float32, torch.float64)
        or pos.shape != (len(batch), 3)
        or batch.dtype != torch.long
        or batch.ndim != 1
        or batch.device != pos.device
    ):
        raise ValueError("Expected FP32 or FP64 CUDA positions and int64 graph IDs")
    if cap < 1 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Invalid cutoff or cap")
    if not len(pos):
        return batch.new_empty((2, 0)), batch.new_empty((0, 3))
    minimum, maximum = torch.aminmax(batch)
    first, last = torch.stack((minimum, maximum)).tolist()
    if first < 0:
        raise ValueError("Negative graph ID")
    graphs = last + 1
    mask = None
    if isinstance(pbc, bool):
        has_images = pbc
    else:
        if pbc.ndim not in (1, 2) or pbc.shape[-1] != 3:
            raise ValueError("Invalid PBC mask")
        mask = pbc.to(pos.device, torch.bool).reshape(-1, 3)
        mask = mask.expand(graphs, -1) if len(mask) == 1 else mask[:graphs]
        if len(mask) != graphs:
            raise ValueError("Missing PBC metadata")
        has_images = bool(mask.any())
    if not has_images:
        edges = open_radius_graph(pos, cutoff, batch, cap)
        return edges, edges.new_zeros((edges.shape[1], 3))
    if cell is None or cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3):
        raise ValueError("Expected periodic cells")
    cell = cell.to(pos.device, pos.dtype).reshape(-1, 3, 3)
    cell = cell.expand(graphs, -1, -1) if len(cell) == 1 else cell[:graphs]
    if len(cell) != graphs:
        raise ValueError("Missing cell metadata")
    if mask is not None:
        cell = torch.where(mask.any(1)[:, None, None], cell, torch.eye(3, device=pos.device, dtype=pos.dtype))
    pos, cell, batch = pos.contiguous(), cell.contiguous(), batch.contiguous()
    inv, info = torch.linalg.inv_ex(cell.double())
    frac = (pos.double()[:, None] @ inv[batch]).squeeze(1)
    norms = inv.norm(dim=1)
    if not bool((info == 0).all() & torch.isfinite(frac).all() & torch.isfinite(norms).all()):
        raise ValueError("Invalid periodic geometry")
    lo = frac.new_full((len(cell), 3), torch.inf)
    hi = -lo
    ids = batch[:, None].expand_as(frac)
    lo.scatter_reduce_(0, ids, frac, reduce="amin")
    hi.scatter_reduce_(0, ids, frac, reduce="amax")
    span = (cutoff + 1e-4) * norms
    graphs, order = batch.sort(stable=True)
    low = torch.empty((len(pos), 3), device=pos.device, dtype=torch.int32)
    shape = torch.empty_like(low)
    sizes = torch.empty_like(batch)
    per_graph = batch.new_zeros(len(cell))
    with torch.cuda.device(pos.device):
        _halo_bounds[(triton.cdiv(len(pos), 128),)](
            frac,
            graphs,
            order,
            lo,
            hi,
            span,
            mask.contiguous() if mask is not None else low,
            low,
            shape,
            sizes,
            per_graph,
            len(pos),
            mask is not None,
            128,
            enable_fp_fusion=False,
        )
    starts = sizes.cumsum(0) - sizes
    ends = per_graph.cumsum(0)
    total, atom_max, graph_max = torch.stack((ends[-1], sizes.max(), per_graph.max())).tolist()
    K = triton.next_power_of_2(cap)
    if total >= 2**31 // 3 or atom_max > 65536 or len(pos) * K >= 2**31:
        raise ValueError("Expansion exceeds the supported index or image tile budget")
    owners = low.new_empty(total)
    shifts = low.new_empty(total, 3)
    trans = pos.new_empty(total, 3)
    selected = low.new_empty(len(pos), K)
    with torch.cuda.device(pos.device):
        _images[(len(pos),)](
            cell, batch, order, starts, low, shape, owners, shifts, trans, triton.next_power_of_2(atom_max)
        )
        has_spatial_bins = graph_max > 8192
        if graph_max <= 1024:
            from ._periodic_whole import _nearest

            B = triton.next_power_of_2(max(graph_max, K))
            _nearest[(len(pos),)](
                pos, batch, ends, owners, shifts, trans, pos.new_tensor(cutoff**2), selected, K, B, B.bit_length() - 1
            )
        else:
            perm = cell_ends = base = dims = target_bins = owners
            if has_spatial_bins:
                origins = pos.new_full((len(cell), 3), torch.inf)
                hi = -origins
                idx = batch[:, None].expand_as(pos)
                origins.scatter_reduce_(0, idx, pos, reduce="amin")
                hi.scatter_reduce_(0, idx, pos, reduce="amax")
                origins = torch.where(origins.isfinite(), origins, 0).double()
                hi = torch.where(hi.isfinite(), hi, 0).double()
                # For a true edge, rounding in (target-source)+translation
                # can move an axis by O(eps * |target-source|). Use one padded
                # cutoff-width bin and compute actual bin coordinates in FP64.
                extent = (hi - origins).clamp_min(0)
                width = cutoff + (extent.amax(1) + cutoff) * 1e-5 + 1e-12
                # A larger width on elongated/sparse graphs bounds metadata.
                axis_cap = (
                    (batch.new_zeros(len(cell)).scatter_add_(0, batch, torch.ones_like(batch)) * 8 + 64)
                    .double()
                    .pow(1 / 3)
                    .ceil()
                )
                width = torch.maximum(width, extent.amax(1) / axis_cap)
                dims = (extent / width[:, None]).floor().int() + 1
                volumes = dims.prod(1)
                base = (volumes.cumsum(0) - volumes).int()
                count = int(volumes.sum())
                if count >= 2**31:
                    raise ValueError("Too many spatial bins")
                target_bins = low.new_empty(len(pos), 3)
                keys = low.new_empty(total)
                counts = low.new_zeros(count)
                for ghost, n in ((False, len(pos)), (True, total)):
                    _bins[(triton.cdiv(n, 128),)](
                        pos, owners, trans, batch, origins, dims, base, width, keys, counts, target_bins, n, 128, ghost
                    )
                perm = keys.argsort().int()
                cell_ends = counts.cumsum(0, dtype=torch.int32)
            B = max(256, 2 * K)
            degrees = low.new_empty(len(pos))
            _query[(len(pos),)](
                pos,
                batch,
                ends,
                owners,
                trans,
                shifts,
                perm,
                cell_ends,
                base,
                dims,
                target_bins,
                pos.new_tensor(cutoff**2),
                selected,
                degrees,
                cap,
                K,
                B,
                has_spatial_bins,
            )
            edge_ends = degrees.cumsum(0)
            edge_count = int(edge_ends[-1])
            edge = batch.new_empty((2, edge_count))
            offsets = batch.new_empty((edge_count, 3))
            if edge_count:
                _pack[(len(pos),)](selected, owners, shifts, edge_ends, edge, offsets, edge_count, cap, K)
            return edge, offsets
    selected = selected[:, :cap]
    target, slot = (selected >= 0).nonzero(as_tuple=True)
    chosen = selected[target, slot].long()
    return torch.stack((owners[chosen].long(), target)), shifts[chosen].long()
