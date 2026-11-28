"""CUDA kernels for assigning atoms to grid bins and compacting incoming edges."""

from functools import lru_cache

import torch
import warp as wp


@lru_cache(maxsize=2)
def assign_kernel(dtype):
    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)

    @wp.kernel(enable_backward=False)
    def assign(
        frac: wp.array(dtype=vec),
        batch: wp.array(dtype=int),
        origin: wp.array(dtype=vec),
        widths: wp.array(dtype=vec),
        shape: wp.array(dtype=wp.vec3i),
        offsets: wp.array(dtype=int),
        atom_bin: wp.array(dtype=wp.vec3i),
        keys: wp.array(dtype=int),
        bin_counts: wp.array(dtype=int),
    ):
        t = wp.tid()
        g = batch[t]
        dims = shape[g]
        coord = frac[t] - origin[g]
        bins = wp.vec3i()
        for axis in range(3):
            bins[axis] = wp.clamp(int(wp.floor(coord[axis] / widths[g][axis])), 0, dims[axis] - 1)
        atom_bin[t] = bins
        key = offsets[g] + (bins[0] * dims[1] + bins[1]) * dims[2] + bins[2]
        keys[t] = key
        wp.atomic_add(bin_counts, key, 1)

    return assign


@wp.kernel(enable_backward=False)
def pack(
    ends: wp.array(dtype=wp.int64),
    counts: wp.array(dtype=int),
    indices: wp.array2d(dtype=int),
    shifts: wp.array2d(dtype=wp.vec3i),
    edges: wp.array2d(dtype=wp.int64),
    out_shifts: wp.array(dtype=wp.vec3l),
):
    target, slot = wp.tid()
    if slot < counts[target]:
        dst = ends[target] - wp.int64(counts[target]) + wp.int64(slot)
        edges[0, dst] = wp.int64(indices[target, slot])
        edges[1, dst] = wp.int64(target)
        s = shifts[target, slot]
        out_shifts[dst] = wp.vec3l(wp.int64(s[0]), wp.int64(s[1]), wp.int64(s[2]))
