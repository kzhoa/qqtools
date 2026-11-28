# qneighbor CPU search

## Test-data boundary

The repository Unit suite under `tests/unit/plugins/qneighbor/` and fresh-process
Integration checks in `tests/integration/plugins/test_qneighbor_imports.py` construct
their inputs in code and uses independent geometric references or mocked backend
bindings. It does not load OC22/QO2Mol files or import scripts from `exps/`.
The normal test/CI entry points do not run the local real-data experiments.

`exps/qneighbor/data/oc22-10k.pt` and `qo2mol-10k.pt` are untracked local
benchmark inputs. All real-data comparison counts and performance tables below
are historical local experiment evidence, **not repository pytest cases or CI
requirements**. The data and associated scripts remain outside the repository test
suite; a fresh checkout needs neither file to run the regression tests.
CUDA tests still require their documented hardware and optional dependencies.

A one-time audit-hook check denied Python file reads of both datasets and the
local data directory in the pytest process and its Python subprocesses. All
172 regression tests passed (22.33 s), with no tests removed or weakened.
The audit hook was temporary; it is not a new repository test requirement.

## Contract and implementation

The explicit CPU namespace exports `neighbor_list`, `neighbor_list_obc`,
`neighbor_list_pbc`, and `radius_graph` (the exact OBC alias). No initialization
step is needed. CPU imports and searches require neither ASE nor either CUDA
backend. SciPy is already a core qqtools dependency. The unified `neighbor_list`
always returns `(edge_index, cell_offsets)`, with zero offsets for OBC.
`neighbor_list_obc` returns only the edge index. All three entries share
`(positions, cutoff, batch=None, max_neighbors=32)`; `cell` and `pbc` are
keyword-only. Omitting `batch` selects one structure. The unified entry defaults
to `pbc=False`, and the explicit PBC entry defaults to `pbc=True`.

```python
from qqtools.plugins.qneighbor.cpu import neighbor_list_obc, neighbor_list_pbc

edges = neighbor_list_obc(positions, cutoff=5.0, batch=batch, max_neighbors=32)
edges, offsets = neighbor_list_pbc(positions, 6.0, batch, 32, cell=cell, pbc=pbc)
```

Inputs and results are CPU Torch tensors. Search uses float64 geometry, a strict
cutoff, nonnegative int64 graph IDs (including interleaved and sparse IDs), and
nearest incoming K neighbors. Edges are `(source, target)`; the displacement is
`positions[target] - positions[source] + offsets @ cell`. Equal distances are
ordered by source, then lexicographic integer offset. Distinct coincident atoms
and periodic self images are retained; only zero-image self edges are removed.
The graph itself is nondifferentiable; callers can compute differentiable edge
vectors from its indices and offsets.

OBC uses direct `scipy.spatial.distance.cdist` squared distances and stable
row-wise nearest selection for graphs with at most 64 atoms. Larger graphs use
`cKDTree.query_pairs`: discover an unordered pair once and emit both directions. PBC wraps only periodic fractional axes, enumerates lattice images
using reciprocal face-distance bounds, prunes against an expanded Cartesian
bounding box, and queries the wrapped atoms against an image KDTree. Dense neighborhoods
use nearest-K queries with complete boundary-tie repair; sparse neighborhoods
use radius enumeration. Returned
shifts restore the original unwrapped coordinates. Partial PBC, skewed cells,
and cutoffs larger than the cell are supported. Periodic cells must be finite
and nonsingular, including partial PBC; all-false masks need no cell.

Validation, batch assembly and deterministic nearest-neighbor selection are
shared. The all-open path of the PBC entry delegates to the OBC query. Separate
query paths avoid imposing image construction on molecules. The CUDA default
selection and direct CUDA imports retain their existing behavior.

The radius-query paths materialize candidates before selecting K. Their memory
therefore scales with all in-cutoff neighbors, not just the returned list.
Dense periodic queries normally retain only K plus two candidates per target,
but a large distance-tie shell can still require a full radius query. Periodic
image storage is unchanged by this candidate reduction.
The small OBC dense path bounds its distance matrix to 4096 float64 entries. Small/skewed
cells and large cutoffs can substantially increase image storage. These results
do not establish universal performance superiority or a bound for pathological
geometry.

## Baseline and method

DeepChem's `src/utils/qqtools/qchem/pbc/pbc_neighbor.py` calls ASE's
`primitive_neighbor_list`. In ASE 3.26 this function uses spatial bins, not a
KDTree. The benchmark invokes this same ASE primitive, converts shifts to the
qneighbor convention and applies the same strict nearest-K contract. ASE does
not natively apply that cap. Timings include input conversion, tree/bin
construction, enumeration, selection and Torch result construction; no graph or
tree caching is used. Input fixture loading is excluded.

Environment: Intel Xeon Platinum 8480+, SciPy 1.16.1, ASE 3.26.0, Torch
2.8.0+cu128 / Python 3.12. CPU calls use one thread; OpenBLAS and OMP threads
are set to one, with no hardware-specific code. CPU affinity is recorded in raw
results; the process is not pinned to one core.

The performance matrix uses 128 randomly selected structures per dataset
(seed 734), drawn from the existing 10,000-structure QO2Mol and OC22 fixtures.
Batch sizes are 1/8/32/128, cutoffs 1/5/6/12 Å, and K=32/128. Each configuration
uses up to four batches, a warmup and three alternating repetitions. Reported
values are medians over batches and repetitions. Small-batch configurations use
fewer than 128 structures; this is a sampled performance matrix, not a timing
of all 10,000 structures.

The dedicated OBC experiment compares the shared two-tree radius query with
one-tree unordered pairs on all 128 molecules. Across eight cutoff/K settings,
pair queries reduced query-plus-selection time by 15–38%. This supports sharing
infrastructure while keeping different OBC/PBC query paths, rather than forcing
both through periodic-image machinery. Complete-call measurements also compare
the dedicated OBC entry with the all-false PBC entry.

Reproducible experiment scripts and raw data are in ignored
`exps/qneighbor/cpu/` and `exps/qneighbor/results/cpu-*`. The public implementation
and regression tests do not depend on those files or on ASE.

## Results and acceptance decision

Both CPU implementations are published through `qqtools.plugins.qneighbor.cpu`.
They meet the ASE speed criterion on the measured fixtures. This decision does
not introduce automatic CPU/CUDA routing at the package root.

The initial KDTree-only OBC / image-tree PBC version was measured as follows.
The full ASE-bin matrix covered 64 configurations. Final OBC remeasurement
covered all 32 OBC configurations after replacing the initial two-tree query;
it was 3.50–17.36 times faster than ASE bins. The 32 PBC configurations were
3.78–13.94 times faster than ASE bins; six representative PBC configurations
were remeasured after the shared selection cleanup. The all-false PBC entry
was a median 4.1% slower than the direct OBC entry across the final OBC matrix;
both use the same OBC query after dispatch.

A second, stronger baseline uses ASE's **`PrimitiveNeighborList` class**, which
really does use `cKDTree` (unlike the similarly named bin-based function).
It constructs a fresh list with `skin=0`, `self_interaction=False` and
`bothways=False` for each graph, then expands unordered pairs in NumPy. This
avoids ASE's expensive Python `bothways` loop and gives the baseline the same
vectorized reversal advantage as qneighbor. Its conversion and nearest-K
selection are timed. Graph/tree reuse is still disabled.

The KDTree comparison covers batch 1/128, cutoff 1/5/6/12 and K=32/128.
Representative complete-call medians below use **batch 128, K=32** and compare
against this faster ASE KDTree baseline, not just ASE bins:

| Mode | Cutoff (Å) | qneighbor (ms) | ASE KDTree (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| OBC | 5 | 22.51 | 95.94 | 4.26× |
| PBC | 5 | 241.42 | 551.54 | 2.28× |
| PBC | 6 | 332.56 | 705.22 | 2.12× |
| PBC | 12 | 1427.93 | 2222.74 | 1.56× |

The OBC batch contains 3136 atoms and the PBC batch 10266 atoms. These are
local shared-host measurements, not hardware-independent performance bounds.
An independent one-core fixture verifier overlapped later timing runs.

The full real-data check compared both caps against independently selected ASE
edges and offsets on **all 10,000 QO2Mol structures (cutoff 5)** and **all
10,000 OC22 structures (cutoff 6)**: **40,000 comparisons passed** in 466.47 s.
The full 10k check does not cover every cutoff in the performance matrix; those
additional cutoffs are checked on the sampled performance inputs and in unit
geometry cases.

Fixture SHA-256 values:

- QO2Mol: `34a231acc4b74614499fb992631fdfb8354641d9b053c42604843ee465f17d10`
- OC22: `e2a67ccd18c3777352bc387c08f54aa29e5f1c43f9e81ba2fb6d80fffddd2089`

Raw outputs: `cpu-benchmark.json` (initial matrix), `cpu-final-obc.json`,
`cpu-final-pbc.json`, `cpu-obc-routes.json`, `cpu-ase-tree.json`,
`cpu-ase-tree-k128.json`, and `cpu-full-fixtures.json` under the ignored results
folder. `cpu/initial_search.py` preserves the initial query implementation.

## Regression checks and review

The CPU regression suite uses independent periodic-image enumeration, without
ASE. It covers FP32/FP64 inputs, K=1/32/128, cutoffs 1/5/12, skewed and partial
PBC, multiple images, winding restoration, interleaved/sparse batches,
coincident atoms, deterministic ties, strict cutoff boundaries, empty inputs,
strided tensors requiring gradients, invalid metadata and subprocess imports
with ASE/Warp/Triton blocked.

Regression commands from a checkout with the appropriate CPU/CUDA dependencies installed:

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/qneighbor/test_cpu_neighbors.py -q
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  python -m pytest tests/unit/plugins/qneighbor -q
ruff check src tests scripts
ruff format --check src tests scripts
git diff --check
```

The full qneighbor run passed 133 tests after the implementation change. Two
additional CPU geometry/boundary tests were then added; the final focused CPU
suite passed all 36 tests. The moved Torch fallback was exercised on CUDA FP64
in the full run. Repository Ruff and diff-whitespace checks passed. This is
feature validation, not an integration/release preflight or installed-wheel gate.

Self-review: the four-function import surface needs no setup step; explicit
CPU imports do not probe CUDA dependencies. Real-world input coverage includes
mixed periodic axes and unwrapped coordinates. Hidden constraints (CPU-only
inputs, float64 arithmetic, nonsingular periodic cells, and candidate/image
memory growth) are documented and validated at the boundary where applicable.
The old Torch helper was still needed for Warp's unsupported dtype/cap path;
it now lives with that CUDA caller instead of the CPU implementation.

## Small-graph direct-distance optimization

The current OBC implementation selects direct squared-distance calculation for
individual graphs with **at most 64 atoms**, retaining the established KDTree
query above that limit. `scipy.spatial.distance.cdist` performs distance work in
compiled code; NumPy masks the diagonal and out-of-cutoff entries, then stably
sorts each target's row. Stable source order preserves deterministic ties.
The 64-atom bound limits each distance matrix to 32 KiB; it is a portable
algorithm/memory threshold, not a CPU-model dispatch rule. There is no JIT,
new dependency, graph cache, thread-pool change, or public configuration argument.

Screening compared padded NumPy batches, padded Torch CPU batches, SciPy
`cdist` batches and per-graph direct queries. Torch was slower. Padded batches
can reduce some large-batch overhead, but hurt single graphs and sparse larger
graphs; the small per-graph implementation was selected for a simpler bounded
path with consistently positive measured gains. Large dense-matrix candidates
were rejected as defaults because short-cutoff sparse cases regressed sharply.

The final complete-call matrix has 216 configurations: 104 real-data batch
configurations and 112 synthetic size/batch configurations. Real batches use
1/8/32/128 graphs, up to four batches per size, from the same seeded 128-molecule
sample. Synthetic graphs have 16/48/64/65/128/512/2048 atoms, batch 1/8, and
uniform density 0.05 atoms/Å³. Both sets cover cutoff 1/5/6/12 and K=32/128.
Five alternating repetitions follow an exact-output comparison. Both methods
use one CPU thread and the same host-resident inputs; timing includes validation,
per-graph selection, assembly and Torch outputs. The baseline is the frozen
previous public KDTree implementation, **not ASE**.

Representative final medians, K=32:

| Batch | Cutoff (Å) | Previous KDTree (ms) | Current OBC (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.206 | 0.109 | 1.89× |
| 8 | 5 | 1.579 | 0.635 | 2.49× |
| 32 | 5 | 5.759 | 2.006 | 2.87× |
| 128 | 1 | 6.205 | 4.301 | 1.44× |
| 128 | 5 | 21.973 | 7.858 | 2.80× |
| 128 | 6 | 29.730 | 10.535 | 2.82× |
| 128 | 12 | 31.593 | 11.653 | 2.71× |

All 10,000 QO2Mol fixtures were compared at batch 1 and 128, all four cutoffs
and both caps: **80,632 API comparisons passed**, with exact edge order matching
the frozen implementation (49.12 s). The current CPU unit suite passes **54
tests**, including mixed-size batches around 63/64/65 atoms, duplicate positions,
exact distance ties and the all-false PBC route. The earlier full module/CUDA
run above belongs to the preceding implementation; CUDA code was not changed
by this optimization. Repository Ruff checks and diff whitespace checks passed.

Review: public imports and signatures are unchanged; each input graph selects
its own route, so mixed sizes need no caller conventions. Dense allocation is
bounded, and the existing KDTree path remains available for large graphs.
No new lifecycle, dependency or CPU-thread ownership contract is introduced.

Artifacts under ignored `exps/qneighbor/`: `cpu/pre_dense_search.py` freezes the
baseline; `cpu/dense_validate.py` and `cpu/dense_final_bench.py` reproduce full
verification and timings. Results are `cpu-dense-full-fixtures.json`,
`cpu-dense-final.json`, and `cpu-dense-tests.log`. Candidate screening files are
explicitly marked partial; they are not the final acceptance matrix.

Large-graph timings retain the same KDTree algorithm. Three initial noisy
65-atom cases were repeated 31 times; the largest 2048-atom/batch-8/K128/cutoff12
case was repeated nine times after the initial run showed a 14.7% difference.
The largest case remeasured at 1649.19 ms (previous) versus 1637.55 ms (current);
the worst ratio among the four rechecked cases was 1.024. These rechecks support
retaining the unchanged large-graph route and distinguish timing noise from an
algorithm change. Raw records: `cpu-dense-recheck.json` and
`cpu-dense-recheck-large.json`.

## Periodic nearest-query optimization

The current periodic implementation adds two changes without changing any
public signature or requiring setup:

1. Generate/prune periodic images in array blocks, targeting at most 32,768
   expanded coordinates per block (or one image when the graph itself is larger).
   Consume lattice indices lazily rather than allocating the whole image lattice.
   Keep each image's original vector/matrix multiplication to preserve rounding.
2. Estimate candidate density on at most 16 evenly spaced targets. If the mean
   in-cutoff candidate count exceeds `2*K+1`, request K+2 nearest candidates per
   target; otherwise retain radius enumeration. The sample chooses a cost
   strategy only: both paths compute exact neighbors for every target.

One extra slot allows removal of the zero-image self edge, and another detects
an incomplete distance boundary. Since KDTree tie order is unspecified, rows
with a tied/nearly tied last boundary are repaired by a radius query that
includes the entire shell. The existing strict squared-distance filter and
source/offset tie ordering then select K. Periodic self images remain valid.
This is not a minimum-image algorithm: multiple images of one atom remain
supported even when cutoff exceeds the lattice dimensions.

Screening found that simply replacing radius enumeration with nearest queries
regressed short cutoffs and large K. Query-ball lists were also slower, and a
full neighbor-count pass added too much work on large sparse graphs. The sampled
selector avoids that full pass. On the 128-structure profile at cutoff12/K32,
the previous path spent approximately 1125 of 1429 ms in candidate preparation
and final selection (79%); reducing candidates addresses that measured cost.

### Complete-call performance

The final comparison uses the frozen previous public CPU PBC entry, not ASE.
It includes validation, image construction, tree construction, query, exact
selection, batch assembly and Torch output allocation. Both methods run with
one CPU thread, no graph/tree cache, five alternating repetitions, and an
untimed exact-output comparison first. Timing runs use CPU affinity 223 on the
same Xeon 8480+ host; the full-fixture verifier can run concurrently elsewhere
on the shared host. These measurements do not promise identical ratios on
other machines or pathological inputs.

There are **192 configurations**: 104 real-data cases (batch 1/8/32/128,
up to four batches each), 64 uniform synthetic cases (16/64/256/1024 atoms,
batch 1/4), and 24 cluster/slab/skew stress cases. All cover cutoff1/5/6/12
and K32/128. The real-data sample is the same seeded 128 OC22 structures used
in earlier CPU experiments. Every timed case matched previous edges and shifts
exactly, including order.

Batch 128 complete-call medians (10266 atoms):

| Cutoff (Å) | K | Previous (ms) | Current (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 77.55 | 57.47 | 1.35× |
| 5 | 32 | 229.29 | 214.45 | 1.07× |
| 6 | 32 | 326.15 | 292.00 | 1.12× |
| 12 | 32 | 1410.71 | 339.53 | 4.15× |
| 1 | 128 | 75.14 | 56.85 | 1.32× |
| 5 | 128 | 240.38 | 232.37 | 1.03× |
| 6 | 128 | 348.44 | 316.77 | 1.10× |
| 12 | 128 | 1653.55 | 1149.15 | 1.44× |

For cutoff12/K32, batch 1/8/32 medians improve from 10.62/89.92/354.71 ms to
2.82/21.28/84.31 ms. Across the whole matrix, the median current/previous ratio
is 0.832, p95 is 0.995, and the worst initial ratio is 1.032. Repeating the
worst three cases 31 times gives a worst ratio of 1.0002. The sampling heuristic
can still choose a less efficient route on other geometries; it does not change
correctness and is not a universal speed guarantee.

### Correctness, memory and review

The CPU unit suite passes **61 tests**, adding large coincident-source tie
shells at K1/32/128, periodic-image ties, and an image range that crosses the
chunk boundary. Existing strict-cutoff, skew/partial-PBC, winding, mixed-batch,
invalid-input and missing-optional-backend tests still pass. Repository Ruff
checks and whitespace checks pass. CUDA implementation code is unchanged;
this is feature validation, not a release preflight.

Periodic image storage remains proportional to the required images. Nearest
queries normally reduce candidate arrays to roughly N*(K+2); radius enumeration
and large tie-repair shells can still materialize all in-cutoff neighbors.
There is no claim of peak memory bounded only by K.

Self-review: the direct CPU API needs no new user choices; cost sampling depends
on the current geometry rather than persistent cached state. All original
periodic-image and stable-order semantics are retained. Input requirements and
memory limits remain documented. SciPy is already a required dependency; no
extra backend, compiler, thread-pool setting or machine-specific tuning was added.

Frozen sources and experiment scripts live under ignored `exps/qneighbor/cpu/`:
`pre_pbc_nearest.py`, `pre_pbc_entry.py`, `pbc_nearest_verify.py`,
`pbc_nearest_bench.py`, and `pbc_nearest_recheck.py`. Final result records are
`cpu-pbc-nearest-final.json`, `cpu-pbc-nearest-recheck.json` and
`cpu-pbc-nearest-tests.log` in the results folder. Screening/profiling records
are retained separately under `cpu-pbc-*-screen.json` and the size sweeps.

Full OC22 verification completed on all 10,000 stored structures at cutoff
1/5/6/12 and K32/128, batched in groups of 128: **632 public-call comparisons,
covering 80,000 graph results**, matched the frozen previous implementation
exactly for edge order and integer offsets. Runtime was 514.21 s. The record is
`cpu-pbc-nearest-full-fixtures.json`; no ASE dependency is needed for this
comparison because the baseline had already been validated against ASE.
