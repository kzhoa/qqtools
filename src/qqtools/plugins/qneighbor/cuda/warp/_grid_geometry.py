"""Fused cell heights, coordinate conversion and validation for periodic grids."""

from functools import lru_cache

import torch
import warp as wp

# Match the separately rounded products of the PyTorch fractional-coordinate sum.
wp.set_module_options({"fuse_fp": False})


@lru_cache(maxsize=2)
def geometry_kernel(dtype):
    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)
    mat = wp.types.matrix(shape=(3, 3), dtype=scalar)

    @wp.kernel(enable_backward=False)
    def prepare(
        positions: wp.array(dtype=vec),
        batch: wp.array(dtype=int),
        inverse_transpose: wp.array(dtype=mat),
        heights: wp.array(dtype=vec),
        info: wp.array(dtype=int),
        periodic: wp.array(dtype=wp.vec3i),
        mixed: int,
        frac: wp.array(dtype=vec),
        wraps: wp.array(dtype=wp.vec3i),
        occupied: wp.array(dtype=wp.int64),
        invalid: wp.array(dtype=int),
    ):
        i = wp.tid()
        if i < info.shape[0]:
            inverse = inverse_transpose[i]
            h = vec()
            for axis in range(3):
                a = inverse[axis, 0]
                b = inverse[axis, 1]
                c = inverse[axis, 2]
                h[axis] = scalar(1) / wp.sqrt(a * a + b * b + c * c)
            heights[i] = h
            if info[i] != 0 or not wp.isfinite(h) or wp.min(h) <= scalar(0):
                wp.atomic_max(invalid, 0, 1)
        if i < positions.shape[0]:
            graph = batch[i]
            p = positions[i]
            inv = inverse_transpose[graph]
            axes = wp.vec3i(1)
            if mixed != 0:
                axes = periodic[graph]
            f = vec()
            w = wp.vec3i()
            for axis in range(3):
                x = p[0] * inv[axis, 0] + p[1] * inv[axis, 1] + p[2] * inv[axis, 2]
                if not wp.isfinite(x):
                    wp.atomic_max(invalid, 0, 1)
                elif axes[axis] != 0:
                    w[axis] = int(wp.floor(x))
                f[axis] = x - scalar(w[axis])
            frac[i] = f
            wraps[i] = w
            wp.atomic_add(occupied, graph, wp.int64(1))

    return prepare
