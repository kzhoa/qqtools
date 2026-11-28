"""Exact radius-limited top-k PBC neighbors using a Warp branch-and-bound query.

No radius graph is materialized. A fractional-coordinate cell grid indexes the
real atoms once; periodic images are visited implicitly. Each query maintains a
max heap of k candidates and prunes cells using conservative distance bounds.
"""

import math
import operator
from functools import lru_cache

import torch
from torch import Tensor

from ._dependency import require_warp


@lru_cache(maxsize=16)
def _query_kernel(dtype, periodic_images, full_pbc=False):
    wp = require_warp()

    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)
    mat = wp.types.matrix(shape=(3, 3), dtype=scalar)

    @wp.func
    def worse(d1: scalar, a1: int, s1: wp.vec3i, d2: scalar, a2: int, s2: wp.vec3i):
        # Deterministic lexicographic ordering for exact distance ties.
        return d1 > d2 or (
            d1 == d2
            and (
                a1 > a2
                or (
                    a1 == a2
                    and (s1[0] > s2[0] or (s1[0] == s2[0] and (s1[1] > s2[1] or (s1[1] == s2[1] and s1[2] > s2[2]))))
                )
            )
        )

    @wp.kernel(enable_backward=False)
    def query(
        pos: wp.array(dtype=vec),
        frac: wp.array(dtype=vec),
        wraps: wp.array(dtype=wp.vec3i),
        batch: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=mat),
        periodic: wp.array(dtype=wp.vec3i),
        origin: wp.array(dtype=vec),
        width: wp.array(dtype=vec),
        heights: wp.array(dtype=vec),
        dims: wp.array(dtype=wp.vec3i),
        grid_offset: wp.array(dtype=wp.int32),
        atom_bin: wp.array(dtype=wp.vec3i),
        starts: wp.array(dtype=wp.int32),
        ends: wp.array(dtype=wp.int32),
        ordered_atoms: wp.array(dtype=wp.int32),
        cutoff: scalar,
        k: int,
        indices: wp.array2d(dtype=wp.int32),
        shifts: wp.array2d(dtype=wp.vec3i),
        distances: wp.array2d(dtype=scalar),
        counts: wp.array(dtype=wp.int32),
    ):
        target = wp.tid()
        graph = batch[target]
        f = frac[target]
        c = mat()
        if wp.static(periodic_images):
            c = cell[graph]
        shape = dims[graph]
        center = atom_bin[target]
        step = width[graph]
        org = origin[graph]
        height = heights[graph]
        pbc = wp.vec3i(1) if wp.static(full_pbc) else periodic[graph]
        count = int(0)
        ring = int(0)
        limit2 = cutoff * cutoff
        # Relax lower bounds to avoid pruning on fractional-coordinate roundoff.
        geometry_scale = cutoff
        if wp.static(periodic_images):
            for a in range(3):
                for b in range(3):
                    geometry_scale = wp.max(geometry_scale, wp.abs(c[a, b]))
        else:
            for axis in range(3):
                geometry_scale = wp.max(geometry_scale, wp.abs(f[axis]) + wp.abs(org[axis]))
        margin = scalar(0.00001) * geometry_scale
        # Cartesian AABB enclosing a grid bin, including skew lattice vectors.
        # Its distance bound complements the reciprocal-plane bound below.
        half_extent = step * scalar(0.5)
        if wp.static(periodic_images):
            half_extent = vec()
            for axis in range(3):
                for component in range(3):
                    half_extent[component] += wp.abs(step[axis] * c[axis, component]) * scalar(0.5)
        while True:
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    for dz in range(-ring, ring + 1):
                        if wp.max(wp.abs(dx), wp.max(wp.abs(dy), wp.abs(dz))) != ring:
                            continue
                        ijk = center + wp.vec3i(dx, dy, dz)
                        valid = bool(True)
                        image = wp.vec3i(0)
                        wrapped = ijk
                        bound = scalar(0)
                        for axis in range(3):
                            if pbc[axis] == 0 and (ijk[axis] < 0 or ijk[axis] >= shape[axis]):
                                valid = False
                            if wp.static(periodic_images):
                                image[axis] = int(wp.floor(scalar(ijk[axis]) / scalar(shape[axis])))
                                wrapped[axis] = ijk[axis] - image[axis] * shape[axis]
                            lo = org[axis] + scalar(ijk[axis]) * step[axis]
                            hi = lo + step[axis]
                            gap = wp.max(scalar(0), wp.max(lo - f[axis], f[axis] - hi))
                            bound = wp.max(bound, gap * height[axis])
                        if not valid:
                            continue
                        relaxed_bound = wp.max(scalar(0), bound - margin)
                        if relaxed_bound * relaxed_bound > limit2:
                            continue
                        bin_center = org + wp.cw_mul(
                            vec(scalar(ijk[0]), scalar(ijk[1]), scalar(ijk[2])) + vec(scalar(0.5)), step
                        )
                        box_delta = bin_center - f
                        if wp.static(periodic_images):
                            box_delta = box_delta * c
                        box_gap = vec()
                        for axis in range(3):
                            box_gap[axis] = wp.max(scalar(0), wp.abs(box_delta[axis]) - half_extent[axis] - margin)
                        # Strict rejection preserves equal-distance tie candidates.
                        if wp.dot(box_gap, box_gap) > limit2:
                            continue
                        key = grid_offset[graph] + (wrapped[0] * shape[1] + wrapped[1]) * shape[2] + wrapped[2]
                        for cursor in range(starts[key], ends[key]):
                            source = ordered_atoms[cursor]
                            if source == target and image == wp.vec3i(0):
                                continue
                            # Return offsets for edge source -> target, matching
                            # pos[target] - pos[source] + offset @ cell.
                            shift = wp.vec3i(0)
                            delta = pos[target] - pos[source]
                            if wp.static(periodic_images):
                                shift = wraps[source] - wraps[target] - image
                                shift_cart = vec(scalar(shift[0]), scalar(shift[1]), scalar(shift[2])) * c
                                delta += shift_cart
                            d2 = wp.dot(delta, delta)
                            if d2 >= cutoff * cutoff:
                                continue
                            if count < k:
                                at = count
                                count += 1
                                while at > 0:
                                    parent = (at - 1) // 2
                                    if not worse(
                                        d2,
                                        source,
                                        shift,
                                        distances[target, parent],
                                        indices[target, parent],
                                        shifts[target, parent],
                                    ):
                                        break
                                    distances[target, at] = distances[target, parent]
                                    indices[target, at] = indices[target, parent]
                                    shifts[target, at] = shifts[target, parent]
                                    at = parent
                                distances[target, at] = d2
                                indices[target, at] = source
                                shifts[target, at] = shift
                            elif worse(distances[target, 0], indices[target, 0], shifts[target, 0], d2, source, shift):
                                at = int(0)
                                while 2 * at + 1 < k:
                                    child = 2 * at + 1
                                    right = child + 1
                                    if right < k and worse(
                                        distances[target, right],
                                        indices[target, right],
                                        shifts[target, right],
                                        distances[target, child],
                                        indices[target, child],
                                        shifts[target, child],
                                    ):
                                        child = right
                                    if not worse(
                                        distances[target, child],
                                        indices[target, child],
                                        shifts[target, child],
                                        d2,
                                        source,
                                        shift,
                                    ):
                                        break
                                    distances[target, at] = distances[target, child]
                                    indices[target, at] = indices[target, child]
                                    shifts[target, at] = shifts[target, child]
                                    at = child
                                distances[target, at] = d2
                                indices[target, at] = source
                                shifts[target, at] = shift
                            if count == k:
                                limit2 = distances[target, 0]
            # Everything outside the visited bin box crosses at least one of
            # its six planes. Fractional reciprocal-plane distances are valid
            # lower bounds even for a triclinic cell (no minimum-image shortcut).
            remaining = scalar(1.0e30)
            for axis in range(3):
                lower_bin = center[axis] - ring
                upper_bin = center[axis] + ring + 1
                if pbc[axis] != 0 or lower_bin > 0:
                    gap = (f[axis] - (org[axis] + scalar(lower_bin) * step[axis])) * height[axis]
                    remaining = wp.min(remaining, gap)
                if pbc[axis] != 0 or upper_bin < shape[axis]:
                    gap = (org[axis] + scalar(upper_bin) * step[axis] - f[axis]) * height[axis]
                    remaining = wp.min(remaining, gap)
            relaxed_remaining = wp.max(scalar(0), remaining - margin)
            if relaxed_remaining * relaxed_remaining > limit2:
                break
            ring += 1
        counts[target] = count

    return query


@torch.no_grad()
def grid_neighbors(
    positions: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
    batch: Tensor,
    cutoff: float,
    max_neighbors: int,
    *,
    bin_width: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Return incoming top-k edges inside cutoff; ties use (source, offset).

    All-false PBC uses Cartesian bins and accepts cell=None.
    Geometry is detached, as for existing graph builders. The model computes
    differentiable edge vectors from the returned indices and integer shifts.
    """
    wp = require_warp()

    if positions.device.type != "cuda" or positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("Warp top-k requires CUDA float32 or float64 positions")
    k = operator.index(max_neighbors)
    if k <= 0 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Warp top-k requires a positive neighbor cap and cutoff")
    n = positions.shape[0]
    if n == 0:
        result = (
            torch.empty(2, 0, dtype=torch.long, device=positions.device),
            torch.empty(0, 3, dtype=torch.long, device=positions.device),
        )
        return result
    # Import decorated kernels only after the optional dependency is available.
    from ._grid_io import assign_kernel, pack

    # CUDA from_torch accesses Warp's device table directly. Do not rely on
    # another package/import having initialized it first (fresh worker startup).
    # wp.init() is idempotent; subsequent graph builds reuse the runtime.
    wp.init()
    device, dtype = positions.device, positions.dtype
    batch = batch.to(device=device, dtype=torch.long)
    last_graph = batch.max()
    if pbc is None:
        # The caller declares every axis periodic; no device mask to inspect.
        graphs = int(last_graph) + 1
        periodic_images, open_axes = True, False
        periodic = None
    else:
        periodic = pbc.to(device=device, dtype=torch.bool).reshape(-1, 3)
        # Read the batch size and boundary modes in one device synchronization.
        active = (torch.arange(len(periodic), device=device) <= last_graph)[:, None]
        last, periodic_images, open_axes = torch.stack(
            (last_graph, (periodic & active).any(), (~periodic & active).any())
        ).tolist()
        graphs = last + 1
        periodic_images = bool(periodic_images)
        periodic = periodic.expand(graphs, -1) if periodic.shape[0] == 1 else periodic[:graphs]
        if periodic.shape[0] != graphs:
            raise ValueError("Missing PBC metadata for a batch graph")
    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))

    def descriptor(tensor, warp_dtype=None):
        return wp.from_torch(tensor, dtype=warp_dtype, requires_grad=False, return_ctype=True)

    mat = wp.types.matrix(shape=(3, 3), dtype=scalar)
    positions = positions.contiguous()
    batch32 = batch.int()
    mask = periodic.int() if open_axes else None
    occupied = batch.new_zeros(graphs)
    wraps = None
    if periodic_images:
        if cell is None:
            raise ValueError("Periodic graphs require cell metadata")
        cell = cell.reshape(-1, 3, 3)
        cell = cell.expand(graphs, -1, -1) if cell.shape[0] == 1 else cell[:graphs]
        if cell.shape[0] != graphs:
            raise ValueError("Missing cell metadata for a batch graph")
        cell = cell.to(device=device, dtype=dtype).contiguous()
        if open_axes:
            # A nonperiodic member of a mixed batch may have a zero placeholder cell.
            cell = torch.where(periodic.any(dim=1)[:, None, None], cell, torch.eye(3, device=device, dtype=dtype))
        inv, info = torch.linalg.inv_ex(cell)
        height = torch.empty((graphs, 3), device=device, dtype=dtype)
        from ._grid_geometry import geometry_kernel

        frac = torch.empty_like(positions)
        wraps = torch.empty((n, 3), device=device, dtype=torch.int32)
        invalid = torch.zeros(1, device=device, dtype=torch.int32)
        inverse_transpose = inv.transpose(-1, -2).contiguous()
        wp.launch(
            geometry_kernel(dtype),
            dim=max(n, graphs),
            inputs=[
                descriptor(positions, vec),
                descriptor(batch32),
                descriptor(inverse_transpose, mat),
                descriptor(height, vec),
                descriptor(info),
                descriptor(mask, wp.vec3i) if open_axes else None,
                int(open_axes),
                descriptor(frac, vec),
                descriptor(wraps, wp.vec3i),
                descriptor(occupied),
                descriptor(invalid),
            ],
            stream=stream,
        )
        if bool(invalid[0]):
            raise ValueError("Warp top-k requires finite positions and nonsingular finite periodic cells")
    else:
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("Warp top-k requires finite positions")
        cell = None
        frac = positions  # Axis-aligned Cartesian bins have unit plane distances.
        height = positions.new_ones((graphs, 3))
        occupied.scatter_add_(0, batch, torch.ones(1, dtype=torch.long, device=device).expand_as(batch))
    if open_axes:
        expanded_batch = batch[:, None].expand(-1, 3)
        lo = torch.full((graphs, 3), float("inf"), device=device, dtype=dtype)
        hi = -lo
        lo.scatter_reduce_(0, expanded_batch, frac, reduce="amin", include_self=True)
        hi.scatter_reduce_(0, expanded_batch, frac, reduce="amax", include_self=True)
        lo = torch.where(occupied[:, None] > 0, lo, 0)
        hi = torch.where(occupied[:, None] > 0, hi, 0)
        origin = lo - 1e-5
        extent = hi - lo + 2e-5
        if periodic_images:
            origin = torch.where(periodic, 0, origin)
            extent = torch.where(periodic, 1, extent)
    else:
        # Wrapped fractional coordinates already occupy the unit cell.
        origin = torch.zeros_like(height)
        extent = torch.ones_like(height)
    bin_width = 2.0 if bin_width is None else float(bin_width)
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive")
    dims = (extent * height / bin_width).floor().clamp_min(1).long()
    # Bound dense grid metadata on elongated or mostly empty cells.
    axis_cap = (occupied * 8 + 64).double().pow(1 / 3).ceil().long()
    dims = torch.minimum(dims, axis_cap[:, None])
    width = extent / dims

    grid_sizes = dims.prod(dim=1)
    offsets = grid_sizes.cumsum(0) - grid_sizes
    total_cells = int(grid_sizes.sum())
    if total_cells > torch.iinfo(torch.int32).max:
        raise ValueError("Periodic grid exceeds the supported int32 bin index range")
    dims32, offsets32 = dims.int(), offsets.int()
    frac, origin, width = frac.contiguous(), origin.contiguous(), width.contiguous()
    atom_bin = torch.empty((n, 3), device=device, dtype=torch.int32)
    keys = torch.empty(n, device=device, dtype=torch.int32)
    bin_counts = torch.zeros(total_cells, device=device, dtype=torch.int32)
    wp.launch(
        assign_kernel(dtype),
        dim=n,
        inputs=[
            descriptor(frac, vec),
            descriptor(batch32),
            descriptor(origin, vec),
            descriptor(width, vec),
            descriptor(dims32, wp.vec3i),
            descriptor(offsets32),
            descriptor(atom_bin, wp.vec3i),
            descriptor(keys),
            descriptor(bin_counts),
        ],
        stream=stream,
    )
    order = torch.argsort(keys, stable=True).int()
    ends = bin_counts.cumsum(0, dtype=torch.int32)
    starts = ends - bin_counts
    indices = torch.empty((n, k), device=device, dtype=torch.int32)
    shifts = torch.empty((n, k, 3), device=device, dtype=torch.int32)
    distances = torch.empty((n, k), device=device, dtype=dtype)
    counts = torch.empty(n, device=device, dtype=torch.int32)
    arrays = [
        (positions, vec),
        (frac, vec),
        (wraps, wp.vec3i),
        (batch32, wp.int32),
        (cell, mat),
        (mask, wp.vec3i),
        (origin, vec),
        (width, vec),
        (height.contiguous(), vec),
        (dims32, wp.vec3i),
        (offsets32, wp.int32),
        (atom_bin, wp.vec3i),
        (starts, wp.int32),
        (ends, wp.int32),
        (order, wp.int32),
    ]
    wp.launch(
        _query_kernel(dtype, periodic_images, not open_axes),
        dim=n,
        inputs=[descriptor(t, w) if t is not None else None for t, w in arrays] + [cutoff, k],
        outputs=[
            descriptor(indices),
            descriptor(shifts, wp.vec3i),
            descriptor(distances),
            descriptor(counts),
        ],
        stream=stream,
        block_dim=64,
    )
    edge_ends = counts.cumsum(0)
    total = int(edge_ends[-1])
    # Warp array descriptors use signed 32-bit byte strides. For very
    # large COO outputs the row stride of the int64 edge index exceeds that.
    if total > torch.iinfo(torch.int32).max // edge_ends.element_size():
        row, slot = torch.nonzero(torch.arange(k, device=device)[None] < counts[:, None], as_tuple=True)
        return torch.stack((indices[row, slot].long(), row)), shifts[row, slot].long()
    edge_index = torch.empty((2, total), device=device, dtype=torch.long)
    cell_offsets = torch.empty((total, 3), device=device, dtype=torch.long)
    if total:
        wp.launch(
            pack,
            dim=(n, k),
            inputs=[
                descriptor(edge_ends),
                descriptor(counts),
                descriptor(indices),
                descriptor(shifts, wp.vec3i),
                descriptor(edge_index),
                descriptor(cell_offsets, wp.vec3l),
            ],
            stream=stream,
        )
    return edge_index, cell_offsets
