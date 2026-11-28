# qneighbor CUDA implementation and validation

## Public contract

The CUDA functions are `neighbor_list`, `neighbor_list_obc`,
`neighbor_list_pbc`, and `radius_graph` (an exact alias of `neighbor_list_obc`).
The shortest function dispatches on PBC and always returns `(edge_index, cell_offsets)`
on the positions device, including zero offsets for OBC. For known OBC, use
`neighbor_list_obc` or its alias `radius_graph` to avoid allocating zero offsets on the GPU. Dedicated functions
avoid the outer boundary dispatch. All three entries share
`(positions, cutoff, batch=None, max_neighbors=32)`; `cell` and `pbc` are
keyword-only. Omitting `batch` selects one structure. The unified entry defaults
to `pbc=False`, and the explicit PBC entry defaults to `pbc=True`.
The same four names in `qqtools.plugins.qneighbor.cuda` share the top-level
function objects and one-time backend selection. The CUDA namespace also exposes
`neighbor_list_obc_warp`, `neighbor_list_pbc_warp`, `neighbor_list_obc_triton`, and
`neighbor_list_pbc_triton`. These are lazily bound direct aliases of the existing
backend functions, with no call wrapper, fallback, or default-backend selection.
A failed resolution is not cached; its error propagates to the caller.
The top-level namespace remains CUDA-only. CPU functions are available separately
from `qqtools.plugins.qneighbor.cpu`; see [CPU validation](qneighbor-cpu-validation.md).

The neutral CUDA interface selects Warp on first function access and uses Triton
only when Warp itself is absent. It binds actual functions, without a public
initialization step or repeated backend selection. An installed but broken Warp
raises its original error. Explicit functions in `cuda.warp` and `cuda.triton`
never switch backends. Importing qqtools or the qneighbor module does not import
Warp. The qneighbor extra installs `warp-lang` without a version constraint,
independently of the full extra. Backend selection does not gate on a Warp
version number. Use a wheel compatible with the host driver.

Warp PBC uses the grid/heap implementation. The Warp `neighbor_list_pbc` also
accepts the backend-specific keyword `bin_width`. The lower-level
`build_neighbor_list_batch_warp` and `build_neighbor_list_batch_full_pbc_warp`
entry points accept an optional `bin_width`, defaulting to a target width of 2 Å.
There is no public algorithm selector. The pre-release auto selector and scan
implementation were withdrawn after performance validation; they are not part
of the installed package. OBC and Triton implementations are unchanged.

All public builders default to K32. They select the nearest incoming neighbors
strictly within cutoff, excluding only zero-offset self edges. Periodic outputs
are int64 edges `(2, E)` and cell offsets `(E, 3)`; the displacement convention is
`positions[target] - positions[source] + offset @ cell_for_target`.
Warp uses source index then lexicographic offset to resolve exact distance ties.
Edge column order is not part of the shared API contract.

Warp periodic positions must be CUDA float32 or float64. Batch IDs must be
nonnegative int64 on the positions device. Cells use row vectors, with shape
`(3, 3)` or `(B, 3, 3)`; a single row broadcasts. PBC accepts a Python bool or
an axis mask of shape `(3,)` or `(B, 3)`. All-open input accepts `cell=None`.
Periodic cells must be finite and nonsingular, including unused periodic cells.
Search records no autograd; models compute differentiable edge vectors from the
returned structure. Execution follows the current Torch CUDA stream. Compact
output allocation requires synchronization, so CUDA graph capture is unsupported.

CPU imports do not require Warp, Triton or ASE. Triton PBC uses its own automatic
whole/stream/spatial query selection; the removal of Warp auto does not change
that internal implementation. Ghost expansion consumes memory and has explicit
indexing/image-tile limits. Triton OBC supports at most 65,536 atoms per graph.
No backend promises universal performance superiority.

## Production layout

Top-level `__init__.py` lazily re-exports the default CUDA functions.
`cuda/__init__.py` owns the sole backend selection state and explicit aliases.
`_dispatch.py` contains shared OBC/PBC dispatch for CPU and CUDA. CPU imports do
not load the CUDA namespace. The former `api.py` was removed; users import
functions from `qneighbor`, `qneighbor.cuda`, or the explicit backend namespaces.

Warp exposes `obc.py` and `pbc.py`. Its private modules are `_dependency.py`,
`_obc_kernels.py`, `_pbc_grid.py`, `_grid_geometry.py`, and `_grid_io.py`.
The backend and boundary condition are explicit in the directory/module names.
Public long-form Warp aliases and the explicit full-PBC builder remain available.

The experimental auto/scan modules, their helpers, and a frozen grid baseline
were moved to ignored `exps/qneighbor/warp_auto_archive/`. That independent local
package includes the old interface and self-contained experiment tests. It is
not imported by production code or required by repository tests.

## Why auto was withdrawn

The original policy used batch-average grid density, total atom count, cutoff,
and neighbor cap. It did not meet the <=10% worst-case loss target against the
faster validated grid/scan route. At 102,278 atoms, cutoff 6 Å, K128, a repeated
public-call measurement gave about 3.46 ms for auto versus 2.44 ms for grid.
Scan itself remains useful experimentally: at cutoff 12 Å, K128, the same batch
measured about 6.86 ms for scan versus 13.78 ms for grid.

Changing scan to 32 threads for K<=64 improved some cases. A final matrix of
348 configurations with 11 randomized paired rounds found a maximum improvement
of 21.6% over the previous auto and maximum regression of 4.7%, with nearly
unchanged median latency. However, updated auto still lost up to 67.6% against
the fastest validated route. Changing density thresholds or using largest graph
size introduced other routing failures, including nearly doubling latency for a
60,000-atom slab at cutoff 7 Å, K112. Those routing changes were rejected.

These are historical H800/Warp 1.19 CUDA 12 measurements, including calibration
and regression cases; they are not an independent universal performance claim.
The simpler public grid implementation is retained until an experimental
selector has evidence for a clearly defined performance envelope.

## Test-data boundary

Repository tests construct inputs in code and use independent pairwise/image
references or controlled dependency mocks. They do not read the ignored OC22
and QO2Mol 10,000-structure files. Private fixture comparisons and performance
sweeps are local experiments, not pytest or CI coverage, and a fresh checkout
needs neither file. CUDA tests require their documented device and dependencies.

Tests for removed auto routing, scan buffers and fused auto metadata moved with
the experimental code. Production tests retain scalar/shape validation, lazy
imports, empty outputs, custom streams, sparse graph IDs, mixed PBC, float32/64,
custom widths, exact nearest-K behavior and periodic-image pruning. Auto/grid
parameter duplications were removed; surviving cases test the production API
against independent expected results rather than comparing a function to itself.

Focused validation commands, using an environment with the relevant extras:

```bash
./scripts/dev test tests/unit/plugins/qneighbor/test_backend_binding.py -q
PYTHONPATH=src python -m pytest tests/unit/plugins/qneighbor tests/integration/plugins/test_qneighbor_imports.py -q
ruff check src tests scripts
ruff format --check src tests scripts
```

Local CUDA validation additionally selects the isolated Warp 1.19 dependency and
GPU 0. The standard binding checks use the managed development environment.

Withdrawal validation on 2026-11-28: the production suite passed **148 tests**
(24.01 s), with no skips. The standard developer command additionally selected
the backend-binding and empty-input scalar-validation nodes: **14 passed**.
The separate ignored archive suite passed **84 tests** (19.15 s), confirming
that the experimental implementation remains runnable after removal. These
archive tests are not part of production CI and overlap some production cases.
The previous 188-to-148 production count change reflects removal of retired
algorithm cases/duplicates and transfer of generic boundary coverage into
`test_warp_neighbors.py`; it is not an exemption for surviving behavior.
Full-repository Ruff checks, formatting, and `git diff --check` passed.

API self-review: calls are simpler because there is only one Warp PBC search
implementation; existing full/partial/open and batch input cases remain covered;
CUDA, dependency, cell, output and synchronization constraints remain explicit.
No temporary compatibility layer or additional public initialization is added.

## Bounded geometry-preparation experiment

A short follow-up retained one small optimization: compute reciprocal-plane
heights in the existing geometry kernel, rather than launching separate Torch
norm and reciprocal operations. The inverse-transpose layout and `fuse_fp=False`
remain unchanged. Coordinate conversion, search, routing and public signatures
are unchanged. Height calculation still includes unused cells so invalid metadata
is rejected consistently. Enabling FMA or changing the inverse layout did not
show a stable benefit and was not retained; the original transpose is normally
already a contiguous view of the column-major inverse.

The final paired measurement used 36 configurations and 31 randomized rounds,
including B1/8/128, a 102,278-atom batch, cutoff 1/6/12, K32/128, FP32/FP64 and
partial PBC. Timings include complete synchronized public calls after warmup.
New small-structure fixture rows were used for the final check; the large-cell
regression remained the same. Updated/baseline median latency ratios had median
0.9830, p95 0.9975 and maximum 0.9991. The largest observed reduction was 5.4%.
This is a small measured gain in the bounded matrix, not a universal speed claim.
All canonical edges/offsets matched the frozen baseline.

The production suite passed **152 tests** (22.95 s), including four new
self-contained FP32/FP64 full/partial-PBC cases that exercise values immediately
above/below wrap and grid-size boundaries against independent image enumeration.
The standard backend-binding lane passed 10 tests; full-repository Ruff and diff
checks passed. Local experiments are ignored under `exps/qneighbor/geometry-quick/`.
Exploration stopped after this limited comparison.

## CUDA namespace exports

The CUDA namespace adds the four shared default names and four explicit
backend-suffixed aliases described above. Regression coverage checks import with
both optional backends blocked, concurrent access through both default
namespaces, direct alias identity, absence of repeated resolution, explicit
backend isolation, and retry after an import error. Real CUDA contract tests
also verify the suffixed aliases are the existing Warp/Triton function objects.

Validation on 2026-11-28: the managed export/binding selection passed **17 tests**;
the complete qneighbor suite with CUDA and Warp 1.19 passed **159 tests**
(24.74 s). Full-repository Ruff and diff checks passed. No kernels or dependency
requirements changed, so performance benchmarks were not repeated.

## API module consolidation

The former `api.py` responsibilities now live in `cuda/__init__.py` (backend
selection and binding) and `_dispatch.py` (shared boundary dispatch). Top-level
exports resolve directly from CUDA. Existing top-level, CPU and explicit CUDA
function imports retain their signatures and identities; the removed pre-release
`qneighbor.api` import path has no compatibility shim.

The CPU subprocess regression additionally confirms that importing and calling
CPU builders does not load `qqtools.plugins.qneighbor.cuda`. Concurrent default
binding, precise missing-dependency fallback, explicit aliases and repeated-call
behavior remain covered after retargeting tests to the CUDA state owner.
The managed binding/export/CPU selection passed **78 tests**; the complete CUDA
qneighbor suite passed **159 tests** (26.27 s). Full-repository Ruff and diff
checks passed. No search kernels changed.


## Test-lane placement

Fresh-process dependency isolation and backend binding live in
`tests/integration/plugins/test_qneighbor_imports.py`. Unit tests retain
in-process numerical, validation, mocked selection and alias checks. The five
process-test functions (six parametrized cases) were moved without removing
assertions. Their CUDA cases retain the GPU marker and optional dependency
requirements; CPU-only preflight runs the four non-GPU import cases. The full
local qneighbor selection includes both Unit and Integration paths above.
