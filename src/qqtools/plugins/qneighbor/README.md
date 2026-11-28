# Choosing a qneighbor import

**The default import is CUDA-only.** Use
`from qqtools.plugins.qneighbor import neighbor_list` for CUDA tensors.
CPU tensors require an explicit CPU import:
`from qqtools.plugins.qneighbor.cpu import neighbor_list`.

Read `from qqtools.plugins.qneighbor.cuda.warp import neighbor_list_pbc` in two
parts. The path `cuda.warp` selects the device and backend. The function name
`neighbor_list_pbc` selects the periodic neighbor search. Switching to Triton
changes only the import path.

## Choose the function by boundary condition

| Function | When to use it | Returns |
| --- | --- | --- |
| `neighbor_list` | General entry point. Pass `pbc` to select the boundary condition at call time. The default is open. | `(edge_index, cell_offsets)`, with zero offsets for open boundaries |
| `neighbor_list_obc` | Open boundary conditions (OBC). Search the given positions directly and skip allocating offsets. | `edge_index` |
| `neighbor_list_pbc` | Periodic boundary conditions (PBC). Supply a lattice `cell` to include periodic images. All axes are periodic by default. | `(edge_index, cell_offsets)` |

For example, these two atoms are 0.9 units apart in open space and 0.1 units
apart across the boundary of a unit cell:

```python
import torch
from qqtools.plugins.qneighbor.cpu import neighbor_list_obc, neighbor_list_pbc

positions = torch.tensor([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]])
edges = neighbor_list_obc(positions, cutoff=0.2)  # No neighbors.
edges, shifts = neighbor_list_pbc(positions, cutoff=0.2, cell=torch.eye(3))
# Each atom has the other as a neighbor across the periodic x boundary.
```

## Choose the import path by device and backend

Each path below exports all three functions. The examples use the general entry
point so you can see exactly which part of the import changes.

| Device and backend | Import |
| --- | --- |
| CUDA, default selection | `from qqtools.plugins.qneighbor import neighbor_list` |
| CUDA, explicit Warp | `from qqtools.plugins.qneighbor.cuda.warp import neighbor_list` |
| CUDA, explicit Triton | `from qqtools.plugins.qneighbor.cuda.triton import neighbor_list` |
| CPU, SciPy | `from qqtools.plugins.qneighbor.cpu import neighbor_list` |

The top-level path and `qqtools.plugins.qneighbor.cuda` select Warp on first
function access and fall back to Triton if Warp is absent.
