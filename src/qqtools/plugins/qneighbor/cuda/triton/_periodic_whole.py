"""Whole-graph Triton periodic query and fallback for large neighbor caps."""

import torch
import triton
import triton.language as tl

from ._periodic_geometry import PeriodicGeometry
from .obc import _select


@triton.jit
def _images(cell, batch, order, starts, low, shape, owners, shifts, translation, B: tl.constexpr):
    atom = tl.program_id(0)
    src = tl.load(order + atom)
    graph = tl.load(batch + src)
    lane = tl.arange(0, B)
    ny = tl.load(shape + 3 * atom + 1)
    nz = tl.load(shape + 3 * atom + 2)
    active = lane < tl.load(shape + 3 * atom) * ny * nz
    sx = tl.load(low + 3 * atom) + lane // (ny * nz)
    sy = tl.load(low + 3 * atom + 1) + lane // nz % ny
    sz = tl.load(low + 3 * atom + 2) + lane % nz
    ghost = tl.load(starts + atom) + lane
    tl.store(owners + ghost, src, active)
    tl.store(shifts + ghost * 3, sx, active)
    tl.store(shifts + ghost * 3 + 1, sy, active)
    tl.store(shifts + ghost * 3 + 2, sz, active)
    for axis in tl.static_range(3):
        base = cell + graph * 9 + axis
        t = tl.fma(
            sz.to(cell.dtype.element_ty),
            tl.load(base + 6),
            tl.fma(sy.to(cell.dtype.element_ty), tl.load(base + 3), sx.to(cell.dtype.element_ty) * tl.load(base)),
        )
        tl.store(translation + ghost * 3 + axis, t, active)


@triton.jit
def _nearest(
    pos,
    batch,
    ends,
    owners,
    shifts,
    translation,
    radius_sq,
    output,
    K: tl.constexpr,
    B: tl.constexpr,
    LOG: tl.constexpr,
):
    target = tl.program_id(0)
    graph = tl.load(batch + target)
    start = tl.load(ends + graph - 1, graph > 0, 0)
    stop = tl.load(ends + graph)
    ghost = start + tl.arange(0, B)
    mask = ghost < stop
    src = tl.load(owners + ghost, mask, 0)
    identity = src == target
    dx = tl.load(pos + target * 3) - tl.load(pos + src * 3) + tl.load(translation + ghost * 3, mask, 0)
    dy = tl.load(pos + target * 3 + 1) - tl.load(pos + src * 3 + 1) + tl.load(translation + ghost * 3 + 1, mask, 0)
    dz = tl.load(pos + target * 3 + 2) - tl.load(pos + src * 3 + 2) + tl.load(translation + ghost * 3 + 2, mask, 0)
    for axis in tl.static_range(3):
        identity = identity & (tl.load(shifts + ghost * 3 + axis, mask, 0) == 0)
    d2 = tl.fma(dz, dz, tl.fma(dx, dx, dy * dy))
    # The low word orders equally distant ghosts by source and integer offset.
    selected = _select(d2, ghost, mask & ~identity & (d2 < tl.load(radius_sq)), K, B, LOG)
    tl.store(output + target * K + tl.arange(0, K), selected)


def search_whole(pos, cell, mask, batch, cutoff, cap, geometry: PeriodicGeometry):
    graphs, order = geometry.sorted_graphs, geometry.order
    f = geometry.fractional[order]
    low = (f - (geometry.upper + geometry.span)[graphs]).ceil().int()
    shape = (f - (geometry.lower - geometry.span)[graphs]).floor().int() - low + 1
    if mask is not None:
        active = mask[graphs]
        low = torch.where(active, low, 0)
        shape = torch.where(active, shape, 1)
    sizes = shape.prod(dim=1)
    starts = sizes.cumsum(0) - sizes
    per_graph = batch.new_zeros(len(cell)).scatter_add_(0, graphs, sizes)
    ends = per_graph.cumsum(0)
    total, atom_max, graph_max = torch.stack((ends[-1], sizes.max(), per_graph.max())).tolist()
    K = triton.next_power_of_2(cap)
    B = triton.next_power_of_2(max(graph_max, K))
    if total >= 2**31 // 3 or B > 65536:
        raise ValueError("Expanded batch exceeds Triton indexing/tile limit")
    owners = low.new_empty(total)
    shifts = low.new_empty(total, 3)
    translation = pos.new_empty(total, 3)
    selected = low.new_empty(len(pos), K)
    with torch.cuda.device(pos.device):
        _images[(len(pos),)](
            cell, batch, order, starts, low, shape, owners, shifts, translation, triton.next_power_of_2(atom_max)
        )
        _nearest[(len(pos),)](
            pos,
            batch,
            ends,
            owners,
            shifts,
            translation,
            pos.new_tensor(cutoff * cutoff),
            selected,
            K,
            B,
            B.bit_length() - 1,
        )
    selected = selected[:, :cap]
    target, slot = (selected >= 0).nonzero(as_tuple=True)
    chosen = selected[target, slot].long()
    return torch.stack((owners[chosen].long(), target)), shifts[chosen].long()
