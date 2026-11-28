"""CPU dense/spatial queries and deterministic incoming-edge selection."""

import math
import operator
from itertools import islice, product

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from torch import Tensor

# Keep dense distance storage at 4096 float64 entries per graph.
_DENSE_ATOM_LIMIT = 64
_IMAGE_BLOCK_POINT_LIMIT = 32768
_DENSITY_SAMPLE_TARGETS = 16
_NEAREST_CANDIDATE_FACTOR = 2
_DISTANCE_TIE_TOLERANCE = 32 * np.finfo(float).eps


def _inputs(positions: Tensor, batch: Tensor | None, cutoff: float, cap: int) -> tuple[np.ndarray, np.ndarray, int]:
    cap = operator.index(cap)
    if cap < 1 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Expected positive finite cutoff and neighbor cap")
    if positions.device.type != "cpu" or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected CPU positions of shape (N, 3)")
    if not positions.is_floating_point():
        raise ValueError("Expected floating-point positions")
    points = positions.detach().double().numpy()
    if not np.isfinite(points).all():
        raise ValueError("Expected finite positions")
    if batch is None:
        batch_ids = np.zeros(len(points), dtype=np.int64)
    else:
        if batch.device.type != "cpu" or batch.shape != (len(points),) or batch.dtype != torch.long:
            raise ValueError("Expected one CPU int64 batch ID per node")
        batch_ids = batch.numpy()
        if np.any(batch_ids < 0):
            raise ValueError("Expected nonnegative batch IDs")
    return points, batch_ids, cap


def _cap_incoming(target, order, cap):
    if not len(order):
        return order
    sorted_targets = target[order]
    ranks = np.arange(len(order))
    starts = np.maximum.accumulate(np.where(np.r_[True, sorted_targets[1:] != sorted_targets[:-1]], ranks, 0))
    return order[ranks - starts < cap]


def _select(source, target, shifts, squared, cutoff, cap):
    valid = (squared < cutoff * cutoff) & ((source != target) | np.any(shifts != 0, axis=1))
    source, target, shifts, squared = source[valid], target[valid], shifts[valid], squared[valid]
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], source, squared, target))
    keep = _cap_incoming(target, order, cap)
    return np.stack((source[keep], target[keep])), shifts[keep]


def _open(points, cutoff, cap):
    if len(points) <= _DENSE_ATOM_LIMIT:
        squared = cdist(points, points, metric="sqeuclidean")
        squared[squared >= cutoff * cutoff] = np.inf
        np.fill_diagonal(squared, np.inf)
        # Original source order plus stable sorting resolves equal distances.
        nearest = np.argsort(squared, axis=1, kind="stable")[:, : min(cap, len(points))]
        target, rank = np.nonzero(np.take_along_axis(squared, nearest, axis=1) < cutoff * cutoff)
        return np.stack((nearest[target, rank], target))
    # Each unordered pair is discovered once; both incoming directions share
    # its distance. Unlike periodic search this needs only one tree.
    pairs = cKDTree(points).query_pairs(cutoff, output_type="ndarray")
    target = np.concatenate((pairs[:, 0], pairs[:, 1]))
    source = np.concatenate((pairs[:, 1], pairs[:, 0]))
    delta = points[target] - points[source]
    squared = np.einsum("ij,ij->i", delta, delta)
    valid = squared < cutoff * cutoff
    source, target, squared = source[valid], target[valid], squared[valid]
    order = np.lexsort((source, squared, target))
    keep = _cap_incoming(target, order, cap)
    return np.stack((source[keep], target[keep]))


def _periodic_pairs(
    points: np.ndarray, ghost_positions: np.ndarray, cutoff: float, cap: int
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(ghost_positions)
    # Sampling chooses a cost strategy only. Both query paths are exact;
    # sampling every target would duplicate much of a sparse query's cost.
    sample_targets = np.linspace(0, len(points) - 1, min(_DENSITY_SAMPLE_TARGETS, len(points)), dtype=np.int64)
    candidate_counts = tree.query_ball_point(points[sample_targets], cutoff, return_length=True)
    if candidate_counts.mean() <= _NEAREST_CANDIDATE_FACTOR * cap + 1:
        pairs = cKDTree(points).sparse_distance_matrix(tree, cutoff, output_type="ndarray")
        return pairs["i"], pairs["j"]
    return _nearest_pairs(tree, points, cutoff, cap)


def _nearest_pairs(tree: cKDTree, points: np.ndarray, cutoff: float, cap: int) -> tuple[np.ndarray, np.ndarray]:
    # Allow for one excluded zero-image self and one extra distance boundary.
    query_size = min(cap + 2, tree.n)
    distances, nearest_indices = tree.query(points, k=query_size, distance_upper_bound=np.nextafter(cutoff, np.inf))
    distances = distances.reshape(len(points), query_size)
    nearest_indices = nearest_indices.reshape(len(points), query_size)
    target, slot = np.nonzero(nearest_indices < tree.n)
    ghost_indices = nearest_indices[target, slot]
    if query_size == tree.n:
        return target, ghost_indices

    boundary_distance, previous_distance = distances[:, -1], distances[:, -2]
    is_finite = np.isfinite(boundary_distance)
    is_tied = np.zeros(len(points), dtype=bool)
    finite_boundary = boundary_distance[is_finite]
    roundoff_margin = _DISTANCE_TIE_TOLERANCE * np.maximum(1, finite_boundary)
    is_tied[is_finite] = finite_boundary <= previous_distance[is_finite] + roundoff_margin
    tied_targets = np.flatnonzero(is_tied)
    if len(tied_targets):
        # KDTree does not guarantee source/offset order within distance ties.
        # Include the whole boundary shell, then let _select order it exactly.
        radii = np.nextafter(
            boundary_distance[tied_targets] + _DISTANCE_TIE_TOLERANCE * np.maximum(1, boundary_distance[tied_targets]),
            np.inf,
        )
        neighbors = tree.query_ball_point(points[tied_targets], radii)
        lengths = np.fromiter(map(len, neighbors), dtype=np.int64)
        keep = ~is_tied[target]
        target = np.concatenate((target[keep], np.repeat(tied_targets, lengths)))
        ghost_indices = np.concatenate((ghost_indices[keep], np.concatenate(neighbors).astype(np.int64)))
    return target, ghost_indices


def _periodic(points, cell, flags, cutoff, cap):
    if not np.any(flags):
        edge = _open(points, cutoff, cap)
        return edge, np.zeros((edge.shape[1], 3), dtype=np.int64)
    inverse = np.linalg.inv(cell)
    winding_float = np.where(flags, np.floor(points @ inverse), 0)
    if np.any(np.abs(winding_float) >= 2**62):
        raise ValueError("Periodic winding exceeds supported int64 range")
    winding = winding_float.astype(np.int64)
    wrapped = points - winding @ cell
    # Reciprocal face distances bound all images, including skewed cells and
    # multiple images per atom. Nonperiodic axes are never replicated.
    extent_float = np.where(flags, np.ceil(cutoff * np.linalg.norm(inverse, axis=0)), 0)
    if not np.isfinite(extent_float).all() or np.any(extent_float >= 2**62):
        raise ValueError("Periodic image range exceeds supported int64 range")
    extent = extent_float.astype(np.int64)
    low, high = wrapped.min(axis=0) - cutoff, wrapped.max(axis=0) + cutoff
    image_iter = product(*(range(-int(n), int(n) + 1) if flag else (0,) for n, flag in zip(extent, flags)))
    ghost_parts, source_parts, shift_parts = [], [], []
    # Bound expanded-coordinate temporaries; do not materialize the entire
    # image lattice before pruning, especially for strongly skewed cells.
    image_block_size = max(1, _IMAGE_BLOCK_POINT_LIMIT // len(points))
    while image_block := list(islice(image_iter, image_block_size)):
        image_offsets = np.asarray(image_block, dtype=np.int64)
        # Preserve the original per-image matrix multiplication's rounding.
        translations = np.asarray([image @ cell for image in image_offsets])
        expanded = wrapped[None, :, :] - translations[:, None, :]
        keep = np.all((expanded >= low) & (expanded <= high), axis=2)
        image_id, source_id = np.nonzero(keep)
        if len(source_id):
            ghost_parts.append(expanded[keep])
            source_parts.append(source_id)
            shift_parts.append(image_offsets[image_id])
    ghost_positions = np.concatenate(ghost_parts)
    sources = np.concatenate(source_parts)
    images = np.concatenate(shift_parts)
    target, ghost_indices = _periodic_pairs(wrapped, ghost_positions, cutoff, cap)
    source = sources[ghost_indices]
    shifts = images[ghost_indices] + winding[source] - winding[target]
    delta = wrapped[target] - ghost_positions[ghost_indices]
    return _select(source, target, shifts, np.einsum("ij,ij->i", delta, delta), cutoff, cap)


def _graph_neighbors(points, graph, cutoff, cap, cells, masks):
    if masks is None:
        return _open(points, cutoff, cap), None
    flags = masks[0 if len(masks) == 1 else graph]
    cell = cells[0 if len(cells) == 1 else graph] if cells is not None else None
    return _periodic(points, cell, flags, cutoff, cap)


def _assemble(points, batch_ids, cutoff, cap, cells=None, masks=None) -> tuple[Tensor, Tensor | None]:
    if not len(batch_ids):
        edge = torch.empty((2, 0), dtype=torch.long)
        return edge, edge.new_empty((0, 3)) if masks is not None else None

    if np.all(batch_ids == batch_ids[0]):
        edge, shift = _graph_neighbors(points, batch_ids[0], cutoff, cap, cells, masks)
        return torch.from_numpy(edge), torch.from_numpy(shift) if shift is not None else None

    # Stable grouping preserves original source order for equal-distance neighbors.
    indices = np.argsort(batch_ids, kind="stable")
    sorted_ids = batch_ids[indices]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    edges, shifts = [], []
    for index in np.split(indices, boundaries):
        edge, shift = _graph_neighbors(points[index], batch_ids[index[0]], cutoff, cap, cells, masks)
        edges.append(index[edge])
        if shift is not None:
            shifts.append(shift)
    edge = np.concatenate(edges, axis=1)
    order = np.argsort(edge[1], kind="stable")
    sorted_edges = torch.from_numpy(edge[:, order].copy())
    sorted_shifts = torch.from_numpy(np.concatenate(shifts)[order]) if shifts else None
    return sorted_edges, sorted_shifts
