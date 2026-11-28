"""Fractional coordinates and per-graph image bounds for Triton PBC queries."""

from typing import NamedTuple

import torch
from torch import Tensor


class PeriodicGeometry(NamedTuple):
    fractional: Tensor
    lower: Tensor
    upper: Tensor
    span: Tensor
    sorted_graphs: Tensor
    order: Tensor


def prepare_geometry(pos: Tensor, cell: Tensor, batch: Tensor, cutoff: float) -> PeriodicGeometry:
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
    return PeriodicGeometry(frac, lo, hi, span, graphs, order)
