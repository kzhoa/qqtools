---
doc_type: spec
status: active
updated_at: 2026-09-05
archived_at:
---

# Balanced Sampler

This specification covers `BalancedDistributedSampler` and `BalancedBatchSampler`, with
the LPT planning tiers documented below as sampler strategies rather than separate samplers.

## Sampler initialization

For normal use, only `sample_costs` and the per-rank `batch_size` are required.
Prefer `BalancedBatchSampler` with `DataLoader(dataset, batch_sampler=sampler)`;
do not also pass batch size, shuffle, sampler, or drop policy to the DataLoader.
With `BalancedDistributedSampler`, the DataLoader batch size must match the sampler's.

`shuffle` and `drop_last` accept Python or NumPy boolean scalars only, not strings
or integers. `seed` accepts a nonnegative Python or NumPy integer, excluding booleans;
floats are rejected rather than truncated. Validation occurs before plan generation.

Initialize the distributed process group before constructing the sampler for automatic
`rank`/`world_size` detection. Without initialized DDP these default to 0/1; subsequent
DDP initialization does not update an existing sampler. Explicit values must agree
with an initialized process group. All ranks must use the same planning settings.

LPT `shuffle` changes step traversal and rank assignments, not batch membership;
call `set_epoch` on every rank. `shuffle=False` means deterministic balanced order,
not original dataset order. The tail policy below applies to a global batch, and
dropped/repeated occurrences remain fixed across epochs. `sample_order` is a legacy
V-only option, not an LPT tuning parameter.

## Public batch grouping API

```python
from qqtools.data import compute_balanced_batch_indices

batch_indices, remaining_indices = compute_balanced_batch_indices(
    sample_costs, batch_size=16, strategy="lpt", seed=777,
)
```

The result is a plain two-element tuple, not a public result class. Both arrays are
read-only, contiguous int64 indices: `batch_indices` has shape `(N // B, B)` and
`remaining_indices` has shape `(N % B,)`. Every input position appears exactly once
across them. Empty input and `N < B` are supported. Remaining indices are selected
using the seed before balancing; they are not optimized as a partial batch. Nothing
is silently padded or dropped. Call `.copy()` if mutable output is needed.

Costs must be one-dimensional, finite and nonnegative; calculations use float64 and
reject non-finite accumulated batch loads. `batch_size` is a positive integer.
The strategies are `lpt_fast`, `lpt-medium` (permanent alias/default `lpt`), and
`lpt_best`. Identical input, strategy and seed reproduce the partition within an
algorithm version; exact grouping is not guaranteed across versions.

This NumPy grouping API has no PyTorch/DDP dependency, world-size parameter, disk cache,
epoch shuffle, or step-level refinement. Its quality is not the complete sampler's
post-refinement quality. It is heuristic cost balancing, not a mathematical optimum or
GPU memory guarantee. Preserve batch boundaries when consuming its output; arbitrary
reshuffling or changing the batch size invalidates the planned grouping.

`compute_global_even_sort_order` continues to return a global one-dimensional sample
permutation without a batch-size parameter. `assign_window_to_ranks` continues to
assign an existing window to ranks. Their names and behavior remain unchanged; the
sampler's V-strategy deprecation does not retire these independent APIs.

## LPT Strategies

`BalancedDistributedSampler` and `BalancedBatchSampler` accept `strategy="lpt_fast"`,
`"lpt-medium"`, and `"lpt_best"`. `"lpt"` is a permanent input alias for
`"lpt-medium"`; initialization normalizes it before planning and in-memory caching.
Historical benchmark tables use `lpt` for this same medium tier.
The default is `"lpt"` (normalized to `"lpt-medium"`). LPT is not supported by
`compute_global_even_sort_order()`
or the dataset-level `balance_strategy` setting used for LMDB artifacts.

`qLmdbDataset.to_dataloader()` delegates strategy selection to the sampler default;
it passes neither `strategy` nor the dataset's cached `sample_order`. Dataset-level
`balance_strategy` controls only dataset artifacts. The loader inherits the sampler's
tail rules: with the current LPT default, `shuffle=False` and `drop_last=False`
require `N` divisible by `batch_size * world_size`, rather than padding validation data.

### Legacy sampler strategy deprecation

Starting in v1.3.15, sampler settings `v1`, `v2`, and `v3` emit a visible
`FutureWarning` once per successful sampler construction (not per epoch). They remain
functional when explicitly selected until removal in v1.4.0. Default construction uses
`lpt` and does not warn. This transition does not deprecate dataset
ordering strategies or `compute_global_even_sort_order`, which cannot consume LPT plans.

Choose `lpt_fast`, `lpt-medium` (or `lpt`), or `lpt_best` explicitly to migrate.
Migration changes sample grouping and traversal; LPT rejects `sample_order`, preserves
batch membership across epochs, and rejects non-divisible validation input unless
`drop_last=True`. It is the recommended fixed-size cost-balancing family, not a guarantee
of better runtime or peak cost for every input, nor of equivalent training randomness.
GPU memory safety and convergence remain unverified. The sampler default changes from
`v3` to `lpt` in this release, so callers omitting `strategy` also adopt the grouping,
shuffle and validation contracts above. Historical statements below about unchanged defaults
describe earlier algorithm work and are superseded by this default-selection decision.

### v1.4.0 maintainer removal checklist

This checklist retires `QQTOOLS-COMPAT-0006`; follow the shared
[compatibility governance](compatibility-governance.md) and
[publish pipeline](../../.github/publish.md). Removing the registry entry alone is not
an implementation cleanup.

1. Inspect the release obligations and locate temporary implementation/test markers:

   ```bash
   python scripts/checks/check_compatibility_registry.py plan --release-version 1.4.0
   rg -n 'QQTOOLS-COMPAT-0006' src tests scripts
   ```

2. Remove `v1`, `v2`, and `v3` acceptance and legacy planning branches from both
   `BalancedDistributedSampler` and `BalancedBatchSampler`, together with their
   deprecation warning. Inspect `src/qqtools/torch/ddp/qbalancedsampler.py` and callers
   of any helper before deleting it: a marker is a navigation aid, not an exhaustive
   dependency inventory. Preserve dataset-level `balance_strategy`,
   `compute_global_even_sort_order()`, their V implementations and shared helpers still
   used by them. Preserve all LPT tiers, the default `lpt`, and its permanent alias
   relationship to `lpt-medium`.
3. Remove tests that exclusively assert legacy sampler support or its warning, including
   marked fixtures in `tests/unit/torch/ddp/test_qbalancedsampler.py`. Update
   `tests/unit/torch/ddp/test_balance_strategy_migration.py` to assert rejection of each
   retired strategy by both sampler classes. Retain default/alias equivalence, LPT
   behavior, and dataset/global-ordering V coverage; do not delete the migration test
   file wholesale.
4. Remove the temporary ID markers from source, tests and scripts, and delete only the
   `QQTOOLS-COMPAT-0006` item from `compatibility-registry.toml`. Do not decrement
   `next_id`, reuse the ID, or introduce a completed registry state. Both removal
   deadlines are v1.4.0, so no separate `legacy_removed` transition release is needed.
   Update this specification to describe the completed removal and record the breaking
   change in `CHANGELOG.md`; Git history and release notes retain the historical record.
5. Verify behavior and registry consistency:

   ```bash
   PYTHONPATH=src python -m pytest tests/unit/torch/ddp tests/integration/torch/test_lpt_dataloader.py -q
   python scripts/checks/check_compatibility_registry.py validate
   python scripts/checks/check_compatibility_registry.py check --release-version 1.4.0
   ```

   Also run the existing dataset/global-ordering tests if their shared code changes.
   Resolve every other due registry item reported by the release check independently;
   clearing this sampler item alone does not guarantee a passing release gate.
6. Commit the cleanup candidate, then run
   `python scripts/release_preflight.py --target-version 1.4.0` from a clean worktree
   **before** bumping the source version to 1.4.0. Only after it passes, finalize the
   version/changelog release commit and tag as described in the publish pipeline.

The gate verifies lifecycle state and marker cleanup, not the semantic removal of every
legacy path. Code review and rejection tests remain required. The tag publish workflow
does not repeat this local compatibility gate.

```python
from torch.utils.data import DataLoader
from qqtools.torch.ddp import BalancedBatchSampler

sampler = BalancedBatchSampler(
    sample_costs,
    batch_size=16,
    strategy="lpt",
    shuffle=True,
    seed=777,
)
loader = DataLoader(dataset, batch_sampler=sampler)
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

Rank and world size are inferred from initialized torch.distributed or supplied explicitly.
All ranks must use identical strategy, costs, base seed, batch size, world size and tail settings, and the
same epoch. Without initialized distributed state, defaults are rank 0 and world size 1.
When using `BalancedDistributedSampler` instead, the DataLoader batch size must match the sampler's.

## Planning and Shuffle

The fast step experiment is now integrated into production. Every tier has a world-size-aware
derived stage: fast uses one sorted-row two-pointer pass, middle one bounded pair pass, best
multi-pass swaps and three-batch repair. On divisible inputs, middle retains the actual fast
derived candidate and best retains middle, using lexicographic (peak, P99, step sum) quality.
This can add lower-tier planning work to higher tiers. Tail-dependent occurrence sets are
not compared. Base arrays remain unchanged and world-independent; all final derived plans
may depend on world size. Each search independently protects peak, P99 and raw step sum;
lower-tier candidate selection is lexicographic, not componentwise dominance.

Fast re-sorts rows before searching because previous exchanges can disrupt cost order.
It preserves the original plan on non-improvement and skips single-rank, B1, fewer-than-two
batch and equal-load cases. The search runs at initialization only. Defaults, public
arguments, tail occurrence counts and epoch behavior are unchanged; no disk cache is added.
Historical sections describing fast as experimental are superseded by this paragraph.

As of 2026-09-05, planning is split into world-independent grouping, bounded DDP tail
repair, optional best-tier step refinement, and step composition. Historical comparison sections
below predate these changes;
their quality and timing numbers are historical, not measurements of the current planner.

Costs are finite, nonnegative values; floating-point estimates are supported. Each rank-batch
contains exactly `batch_size` samples. Completed batches of similar cost are grouped into steps.
Best can exchange samples across batches in a derived, world-size-specific plan before
step composition. This never modifies the world-independent base partition.

| Strategy | Planning work |
|---|---|
| `lpt_fast` | Layered pairing and base exchange, then one derived step-aware two-pointer pass |
| `lpt` | Compare fast with capacity LPT, then run one bounded step-aware pair pass |
| `lpt_best` | Base paired swaps and three-batch repair, then bounded step-aware refinement of the derived plan |

Base candidates use the same seeded sample occurrences, excluding fewer than B remainder
indices. Candidate quality is lexicographic: minimize maximum rank-batch cost first,
linear-interpolated P99 second, and the sum of squared batch loads third. Loads are divided
by the common peak before squaring to avoid overflow. Higher tiers cannot worsen this tuple
for the base partition; they may tie. Individual secondary metrics can worsen when a
higher-priority metric improves. Tail repair uses tier-specific pools, so the final padded
or dropped plan does not carry that cross-tier guarantee. Step-max sums are not used to
select base partitions and need not improve across tiers. Computation uses float64,
not arbitrary-precision integer arithmetic; comparisons follow the computed floating-point scores.

The fast pass pairs batches from opposite ends of the load ordering. It uses the layered
rows' descending costs to find the best single gap-reducing exchange per pair in at most
2B-1 pointer visits, without a B-by-B candidate grid. Each batch participates at most once;
an odd middle batch is untouched. Actual loads and full-epoch quality are recomputed, retaining
the layered candidate on non-improvement or nonfinite trial loads. Overflowing layered loads
skip the pass so a finite capacity
candidate can still rescue a higher tier. The pass adds O(N) search and O(N log N) scoring,
with O(N) memory overall. It is not an optimum over arbitrary pairings or multiple swaps.

This integration changes LPT sample membership/order for the same input and seed. All higher
tiers start with the improved fast candidate. No new argument, default change or disk cache
is introduced. Exchange search runs only during static planning, never at each epoch traversal.

The best tier first uses four passes, alternating extreme-load and seeded random pairings,
with at most 4096 disjoint pairs per pass. Each pair searches for a swap near half its load gap
using sorted costs and binary search. The incumbent is retained after every pass according to
the full objective, not merely a local squared-load improvement. This previous four-pass result
remains the incumbent for the additional repair stage:

- At most 16 windows, each involving two or three batches and at most three sample occurrences
  per batch. All unselected positions are frozen; selected per-batch cardinalities are preserved.
- Enumerate at most 1,680 labelled assignments to the selected slots. The tiny assignment-index
  tables are cached read-only; they contain no dataset-specific data.
- Rank distinct local load vectors lexicographically by descending loads and submit at most
  two candidates per window to the full epoch score: at most 32 candidate-score evaluations.
  Local rankings only generate a shortlist, not a claim of global optimality.
- Compute a conservative float64 lower bound from the selected occurrences:
  `max(largest_cost, mean_batch_cost, largest_cost + sum(smallest B-1 costs))`. Mean and paired
  sums are rounded down with error allowance; an overflowing component is discarded, not used
  as an optimality certificate. If the largest sample already matches the peak, skip the more
  expensive full bound calculation entirely.
- When the computed peak reaches the bound, use at most four windows / eight candidate scores
  for secondary metrics. Larger plans target P99-near batches in this mode, otherwise the
  heaviest few batches; two seeded partners complete each window. Plans with at most six batches
  cycle through small batch combinations instead. The total window budget never resets.
- Recompute actual affected batch loads before acceptance, rejecting nonfinite candidates.
  Retain improvements only under the world-independent batch quality tuple. Step-level
  scores are available for diagnostics but do not influence this stage.

These are internal budgets, not new user parameters. With N denoting post-tail occurrences and
M=N/B batches, the fixed repair budgets preserve O(N log N) total planning time and O(N) memory.
This bounds scaling, not absolute latency or the approximation ratio. `best` means best among
the retained candidates, not a global optimum or a strict improvement on every input.
The lower bound controls internal search effort; the sampler exposes no public 1%-optimality
certificate. Floating-point batch scores remain subject to the numerical semantics above.
Planning-time order is a tendency, not a latency guarantee.
Further planning is skipped when batch size is one, there is only one batch, or the fast
candidate already has identical finite loads on every batch.

Bounded repair can change `lpt_best` batch membership and sample order for the same input/seed
compared with the earlier four-pass implementation. It still builds once at sampler initialization;
epoch traversal never reruns repair. The stage split also changes tail occurrences and may
change memberships for divisible inputs because the third quality key changed.

The fast tier actively balances costs; it does not promise to beat random sampling on all inputs.
The middle tier now evaluates the fast candidate too, so its sample order can differ from the
initial capacity-only `lpt` implementation. V1-V3 and the default selection are unaffected.

The sampler snapshots costs and caches the complete static plan once per instance in memory.
Base seed controls tail selection and equal-cost tie-breaking. Epoch is not part of static
grouping. `BalancedBatchSampler` shares its underlying sampler's cache instead of rebuilding it.
Changing the caller's cost array does not update a live sampler; construct a new sampler when
costs, batch size, world size or the dataset/index mapping change.

| Setting | Behavior |
|---|---|
| `shuffle=True` | Shuffle step order and rank assignments with `seed + epoch` |
| `shuffle=False` | Fixed balanced order; `set_epoch()` has no effect |
| Within-batch order | Fixed across epochs in either mode |
| `sample_order` | Rejected with LPT in either shuffle mode |

Non-shuffled order is not original dataset order. Preserve sample indices when prediction outputs
need to be restored to their original ordering. Other strategies retain their `sample_order`
behavior; LPT rejects rather than silently disregarding it.

## Tail Contract

Let B be `batch_size`, R be `world_size`, and G=B*R. The base partition contains
`full_batches[M, B]` and `remainder[N % B]`, together covering every original index exactly
once. Remainder selection precedes cost sorting using the base seed. These arrays are
read-only and independent of R, drop policy, shuffle and epoch. Strategy, costs, B and seed
still affect grouping; a future persistent cache must account for algorithm version too.

When needed, the DDP stage takes `min(M, M % R + R)` seeded random full batches plus
the remainder as a repair pool. The untouched batch count is divisible by R. Pool indices
are shuffled, truncated or repeated to a multiple of G, then balanced again. This changes
at most 2R-1 existing full batches; a small dataset may fit entirely in the pool. Taking
one extra step provides companions for a large remainder, but is not a hard peak guarantee.
Tail repair changes no full batches if the base already divides into complete steps.
The selected occurrences remain fixed across epochs.

| Input/settings | Result |
|---|---|
| Empty dataset | Empty plan |
| N divisible by G | Each sample exactly once; equal complete batches on all ranks |
| Non-divisible, `drop_last=True` | Drop exactly N % G samples from the repair pool, with no duplicates |
| Non-divisible, `shuffle=True, drop_last=False` | Preserve every sample and add exactly G - N % G occurrences from the pool |
| Non-divisible, `shuffle=False, drop_last=False` | Raise `ValueError` before iteration |

Dropping omits the same samples each epoch; padding repeats the same selected occurrences and may
place duplicates within a batch. Neither should be interpreted as exact-once evaluation.

After tail repair and optional best-tier refinement, batches sorted by load are grouped adjacently into steps. This step
composition is reused even with epoch shuffle, which only changes traversal and rank labels.
The stage split prepares a cache boundary but adds no disk persistence, rank-zero writer,
broadcast, or cross-process shared plan. Every sampler instance still computes its own plan.

### Middle and Best Step Refinement (2026-09-05)

Middle now runs one extreme-load pair pass (at most 4096 disjoint pairs) after tail
completion, with independent peak/P99/raw-step guards. It does not run three-batch repair.
On divisible inputs, best also evaluates the actual middle derived plan and retains the
better lexicographic (peak, P99, step sum) candidate. This adds middle planning work to best
when needed; a lower peak can trade off secondary metrics during this candidate selection.
Tail-dependent occurrence sets are not compared. Base arrays remain world-independent;
middle and best derived memberships may now depend on world size even without a tail.

`lpt_best` now passes the completed batches through `_optimize_step_batches` before sorting
them into steps. It reuses bounded four-pass pair swaps and up to 16 three-batch repair
windows with the actual world size and the step-aware objective. This search runs on derived
copies; base arrays, remainder, occurrence counts and batch cardinality are preserved.
Unlike tail repair, it can touch batches outside the small tail pool.

After each search, acceptance requires a strict lexicographic improvement plus independent
non-increase checks for peak, P99 and the raw sum of adjacent-step maximum loads. Normalized
step scores provide finite comparisons; the raw-sum guard rejects rounding disagreements.
Squared loads may increase because they are the base-stage objective, not the derived-stage
objective. This is a bounded heuristic, not a guarantee of reproducing all historical choices.
One rank, fewer than two batches, B=1 and identical loads skip the extra search. Fast now
runs the separately described two-pointer step pass. No extra public arguments, disk writes or per-epoch search
are introduced; the sampler retains the final derived plan in memory.

Before middle integration, the same 300 small cases gave best 282 full historical-objective hits, up from 262
immediately after the stage split, and still 299 optimal peaks. This restores the previous
hit count, not necessarily the identical set of winning cases. Fast/middle remain 139/188
full hits and 196/260 peak hits. Actual training effects remain unverified.

Before middle integration, CPU-only full-initialization comparisons (five rotating measured repetitions after warmup,
seed 7, B16/R4; `compare_step_optimization.py`) measured:

| Input | Without step pass | With step pass | Step-max sum before / after |
|---|---:|---:|---:|
| N16384 integer uniform | 28.322 ms | 41.930 ms | 205786 / 205786 |
| N16384 rounded lognormal | 29.800 ms | 49.393 ms | 259082 / 259081 |
| N16385 rounded lognormal, padded tail | 33.103 ms | 51.652 ms | 260008 / 260005 |

These are historical local CPU measurements, not GPU throughput results. The extra work
can produce no improvement on a particular input.

### Middle Integration and Experimental Fast Step Pass (2026-09-05)

This section records the pre-integration experiment. Production middle included one step
pass; fast was unchanged at measurement time. The experimental
`exps/test_balance_v3/fast_step.py` sorts each completed batch by descending cost, tries one
vectorized disjoint-pair two-pointer exchange pass, and retains the original unless peak,
P99 and step sum do not worsen and step-aware quality strictly improves. Base arrays and
tail occurrences are preserved. This is measured before final step sorting, not by repeating
the entire planner. Actual DDP training and convergence are unverified.

Fast's 300 small-case full/peak optimum hits change from 139/196 to 156/226; 39 cases
strictly improve the historical tuple. Peak hit rate rises from 65.3% to 75.3%. In the
following CPU measurements, B16/R4 unless noted, times are full planning medians over seven
rotating baseline/prototype repetitions with seeds 7/13/29 repeated, after warmup. They are
not seven independent quality seeds. Metrics shown are seed 7.

| Fast input | Before / after ms | Overhead | Peak before / after | Step sum before / after |
|---|---:|---:|---:|---:|
| 16384 uniform | 2.203 / 2.898 | 31.6% | 805 / 805 | 205786 / 205786 |
| 16384 lognormal | 2.302 / 3.908 | 69.8% | 10471 / 10397 | 259209 / 259166 |
| 16384 bimodal | 1.168 / 1.921 | 64.5% | 2014 / 2014 | 401698 / 401698 |
| 16384 single outlier | 0.906 / 1.307 | 44.2% | 100015 / 100015 | 104095 / 104095 |
| 16385 lognormal, padded | 3.408 / 4.267 | 25.2% | 10471 / 10397 | 260106 / 260058 |
| 16384 lognormal, B64 | 3.081 / 5.150 | 67.2% | 12550 / 12297 | 258858 / 258725 |
| 100096 lognormal | 13.777 / 19.218 | 39.5% | 37018 / 36943 | 1591678 / 1591621 |
| 1000000 lognormal | 216.206 / 313.947 | 45.2% | 37019 / 36944 | 15529736 / 15529693 |

R1 skips this search (2.388/2.319 ms measured, unchanged quality; the apparent speedup is
timing noise). All measured multi-rank lognormal seeds improve; uniform, bimodal and single
outlier inputs do not. Fast integration was subsequently approved and completed.

Middle's one-pass initialization comparison, five measured repetitions after warmup at
seed 7, gives 9.424/11.950 ms for uniform, 9.575/14.594 ms for lognormal and
10.059/14.515 ms for padded lognormal (N16384/16384/16385, B16/R4). Step sums tie on
these three cases. A fixed 48-sample regression instead improves middle peak from 191 to
187, with best still 185. Public arguments and epoch-time work are unchanged.
The refreshed 300-case production comparison gives middle 194 full/265 peak hits (previously
188/260); best remains 282/299. Middle's mean peak gap falls from 0.1843% to 0.1427%.

### Stage-Split Verification (2026-09-05)

After integrating fast step refinement, the directly affected verification set passed 1219
checks, including read-only base protection, componentwise metric guards, an actual step
improvement regression and DataLoader tail coverage:

```bash
PYTHONPATH=src python -m pytest \
  tests/unit/data_utils/test_lpt_fast_steps.py \
  tests/unit/data_utils/test_lpt_step_optimization.py \
  tests/unit/data_utils/test_lpt_stages.py \
  tests/unit/data_utils/test_lpt_two_pointer.py \
  tests/unit/data_utils/test_lpt_fast.py \
  tests/unit/data_utils/test_lpt_tiers.py \
  tests/unit/data_utils/test_lpt_repair.py \
  tests/unit/data_utils/test_rank_batch_plan.py \
  tests/unit/torch/ddp/test_lpt_sampler.py \
  tests/unit/torch/ddp/test_qbalancedsampler.py \
  tests/integration/torch/test_lpt_dataloader.py \
  exps/test_balance_v3/test_compare_fast_swap.py \
  exps/test_balance_v3/test_fast_step.py -q --tb=short
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --fresh-holdout
PYTHONPATH=src python exps/test_balance_v3/compare_step_optimization.py --strategy lpt
PYTHONPATH=src python exps/test_balance_v3/compare_fast_step.py --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_step.py --large
```

The integration reran the pytest command and `compare_exact.py --fresh-holdout`; timing
commands above document the earlier experiment, not a new timing run. Production full/peak
hits on 300 cases are fast 156/226, middle 198/267 and best 282/299. Mean peak gaps are
0.3282%, 0.1360% and 0.0015%. The experimental reference and production fast match in
regression tests. Review confirms unchanged public call sites, input/tail protection and
explicit disclosure of world-dependent derived plans. GPU training remains unverified.

Immediately after the stage split (before derived refinement), on the same 300 small
divisible cases, optimal-peak hits remained 196/260/299 for
fast/middle/best, with mean peak gaps 0.4790%/0.1843%/0.0015%. Under the historical
objective including step maxima, best's full-tuple hits fall from 282 to 262. This is
a measured trade-off from removing step scores, not evidence of non-regression on all
metrics. The oracle still optimizes that historical tuple, not the new squared-load key.
These cases do not measure tail-repair quality or training convergence.
At that point the nine-case CPU tier smoke benchmark also passed coverage and peak/P99 ordering checks;
step-max sums remain reported diagnostics, not monotonic tier assertions.

Review: public call sites remain unchanged; empty/tiny datasets, exact tail counts,
read-only base arrays, world-size independence, and epoch coverage have regression checks;
changed tail identities, the base-only tier guarantee, and the derived best plan's world-size
dependence are documented explicitly. Initialization-only search preserves simple call sites
and avoids repeating optimization each epoch; no high-confidence review issue remains open.
Integration uses actual DataLoaders with ranks simulated in one process, not real multi-GPU
training. Persistent-cache integrity and cross-process cache coordination are out of scope
because no persistent cache has been introduced.
For validation with `shuffle=False, drop_last=False`, non-divisible inputs are rejected to avoid
silently biased metrics. Do not enable shuffling or dropping merely to bypass this check for
validation. General exact-once DDP validation tails require a separate execution/masking design.

## Scope and Limitations

- This implementation adds no disk cache or changes to `balance_meta.npz`, `balance_order.npy`
  or materialized LMDB data. Independently constructed samplers do not share a persistent plan.
- Batch membership is fixed. Traversal shuffle is not uniform random repartitioning.
- Similar-load step grouping can increase variation in total load between steps.
- LPT is a heuristic on additive cost estimates, not an optimality or GPU-memory guarantee.
- Real GPU/DDP throughput, memory and training convergence are **假设/未验证**. Integration tests
  cover real CPU DataLoaders with explicit ranks, not a multi-process distributed training run.

## Test Placement

- `tests/unit/data_utils/test_lpt_fast.py` checks the production layered partition against
  an independent scalar reference and protects its grouping and seed behavior. It imports
  production code only, with no dependency on experimental files.
- `tests/unit/data_utils/test_lpt_tiers.py` protects the tier quality contract. Batch-cost P99
  is an output-quality assertion, not an execution-time measurement.
- `tests/unit/data_utils/test_lpt_repair.py` protects the bounded repair: search/score budgets,
  the certified three-batch counterexample, exact local assignment enumeration, preservation
  of unselected slots and occurrences, conservative bounds, overflow, determinism and laziness.
  It imports production code only and contains no timing assertions.
- `exps/test_balance_v3/test_layered_numpy.py` preserves the experimental implementation's
  scalar-equivalence and boundary checks. Run it explicitly; it is outside the default
  `testpaths = tests` collection. The former unit-test path is no longer used.
- Planning-time measurements and large-scale comparisons remain in the `compare_*.py`
  experimental scripts, not the unit suite.

The placement-only change collected 214 checks and passed both sets independently (99 production
checks and 115 experimental checks). No production algorithm or sampler API changed. Commands:

```bash
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_fast.py exps/test_balance_v3/test_layered_numpy.py --collect-only -q
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_fast.py -q
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_layered_numpy.py -q
```

## Initial Tier Verification (2026-09-05, Before Bounded Repair)

The minimum affected unit and CPU DataLoader integration set passed (459 tests), along with
two representative existing dataset-loader tests. Commands:

```bash
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_tiers.py tests/unit/data_utils/test_rank_batch_plan.py tests/unit/torch/ddp/test_lpt_sampler.py tests/unit/torch/ddp/test_qbalancedsampler.py tests/integration/torch/test_lpt_dataloader.py -q
PYTHONPATH=src python -m pytest tests/unit/torch/qdataset/test_qlmdbdataset.py::test_balance_assets_are_deterministic_and_used_by_loader tests/unit/torch/qdataset/test_qlmdbdataset.py::test_balanced_subset_uses_local_cost_and_order_coordinates -q
PYTHONPATH=src python exps/test_balance_v3/compare_lpt_tiers.py --large
PYTHONPATH=src python exps/test_balance_v3/compare_layered_numpy.py
PYTHONPATH=src python exps/test_balance_v3/review_reference.py
```

The tier benchmark asserts quality monotonicity for each seed and covers uniform, lognormal,
bimodal and outlier costs, batch sizes 1/4/16/64, single-rank execution, and one million samples.
Reference comparison scripts now label the middle tier `lpt`, not capacity-only LPT.

Representative one-million-sample rounded lognormal results, B=16 and R=4 (CPU planning time
is a three-run median; quality uses seed 7):

| Tier | Planning ms | Maximum rank-batch cost | P99 | Sum of step maxima |
|---|---:|---:|---:|---:|
| `lpt_fast` | 115.92 | 37131 | 2928.01 | 15529796 |
| `lpt` | 1091.65 | 36712 | 2524.01 | 15529590 |
| `lpt_best` | 1292.43 | 36712 | 2524.01 | 15529586 |

These are synthetic CPU results, not universal dominance over other samplers or GPU throughput
measurements. A separate 48-sample regression fixture improves the peak from 192 to 186 with
the best tier; ties in the million-sample peak/P99 are expected, not a failure of the contract.

Post-change review checked simplicity (one strategy argument, no search-budget knobs), input
coverage (tails, empty inputs, finite costs, integer dimensions, overflow), and hidden contracts
(lexicographic quality, changed middle-tier ordering, static caches, and validation restrictions).

## Exact-Oracle Refinement Experiment (2026-09-05)

This historical section describes the earlier wide-search prototype, **not the later bounded
repair now used by production `lpt_best`**. Its "current" baseline and measurements refer to the
original four-pass implementation. Experimental assets live under `exps/test_balance_v3/`;
no production unit test imports them. See the bounded-repair verification below for current results.

### Exact baseline

`exact_rank_batches.py` enumerates every unlabelled, fixed-cardinality sample partition once.
It returns a feasible witness, the optimal full quality tuple and the number of partitions
examined. Costs use Python integers; linear P99 is represented exactly as `100 * P99`.
Grouping sorted adjacent batch loads minimizes the sum of step maxima for fixed loads, so
the oracle does not enumerate equivalent batch/rank permutations. Independent permutation
enumeration on six-sample cases checks this reduction, including multiple-step grouping.

The oracle requires nonnegative integer costs, exact divisibility, N <= 24, and at most 200,000
partitions by default. It rejects oversized searches before enumeration; it never labels a
budget-limited result as optimal. These are experiment-only restrictions, not sampler changes.
Large fixed datasets are not automatically feasible to exhaust: N=16, B=4 already has
2,627,625 unlabelled partitions.

`compare_exact.py` fixes N=12, (B,R)=(2,2)/(3,2)/(4,3), and five cost distributions: uniform,
rounded lognormal, bimodal, one outlier and repeated small costs. Dataset seeds 0..9 form 150
development configurations; seeds 100..119 form 300 held-out configurations. Planning seed
is always 7. Each dataset cost vector is reused across the three shapes; these are not 450
independent data draws. The holdout was evaluated after choosing the prototype's search policy.
Future tuning on these observed cases must use a fresh holdout for independent evaluation.

The comparator reports exact full-tuple hits, exact peak hits, average and worst relative peak
gap, and worst-case inputs with optimal witnesses. `legacy_best` snapshots the original
four-pass policy while reusing existing low-level production primitives. It matches the
current production baseline in this experiment. No oracle result is fed into the prototype.

### Experimental refinement and results

`refine_prototype.py` evaluates both layered and capacity-LPT starting points, each with the
existing refinement and an additional 24-pass bounded search. For B <= 6 it enumerates complete
two-batch repartitions; when candidate-count times batch-count is at most 65,536, it ranks all
of them by the full objective instead of pair variance. Otherwise it chooses the minimum pair
peak. For larger B it adds vectorized two-for-two exchanges after one-for-one swaps. Each pass
retains the global incumbent; its disjoint-pair budget is min(M//2, max(1, 65536//B**2)).

This prototype only supports divisible, nonempty inputs, B <= 64, and nonnegative integer costs
with total below 2**53. It is not ready to replace the public float-cost/tail-aware implementation.
Its sorted, contiguous result groups adjacent batch loads just like the production planner.

| Dataset split | Algorithm | Full-tuple optimum | Peak optimum | Mean peak gap | Worst peak gap |
|---|---|---:|---:|---:|---:|
| Development: 150 | Current `lpt_best` | 100 | 139 | 0.1005% | 3.5000% |
| Development: 150 | Prototype | 132 | 144 | 0.0297% | 1.1450% |
| Held-out: 300 | Current `lpt_best` | 214 | 273 | 0.1405% | 3.1746% |
| Held-out: 300 | Prototype | 275 | 286 | 0.0644% | 3.1746% |

The prototype strictly improves the full tuple in 37 development and 70 held-out cases, with
no regressions on these inputs. Its worst held-out peak gap remains unchanged; improvements
in average quality are not a worst-case approximation guarantee.

### Certified two-batch search trap

`inspect_neighborhood.py` preserves the remaining uniform/B3/seed109 counterexample:

```text
costs = [56, 55, 34, 85, 41, 76, 21, 34, 1, 60, 2, 20]
B = 3; R = 2
observed prototype loads = [120, 130, 118, 117]
observed quality = (130, 129.70, 248)
global optimum = (126, 125.91, 244)
```

Exhausting all 60 two-batch repartitions cannot improve the incumbent's full tuple. Exhausting
all 1,120 three-batch repartitions reaches the global optimum. Thus more strictly improving
two-batch rounds alone cannot escape this incumbent. Neutral/worse intermediate moves or a
larger neighborhood could; the next supported research direction is bounded joint replanning
of selected batches when pair search stalls. This is a concrete counterexample, not proof that
three-batch search suffices for arbitrary inputs.

### Scaling and interpretation

`compare_refinement.py --large` checks N=16,384 at B=4/16/64 and R=1/4, plus N=1,000,000 at
B=16/R=4. All use rounded lognormal costs from dataset seed 42. It asserts coverage and
non-regression for planning seeds 7/13/29; timings are three-run medians and metrics use seed 7.
Timing includes prototype validation and output sorting, but excludes result assertions.

The script also reports the peak lower bound
`max(sum(costs)/M, largest_cost + sum(smallest B-1 costs))`, for M complete rank-batches.
In all these scaling cases, current `lpt_best` already reaches that lower bound: 10,167 for
N=16,384 and 36,712 for N=1,000,000. Their peak is therefore globally optimal, even though
these large datasets were not exhausted. This certificate does not establish optimal P99 or
step-max sums. An unchanged peak on these datasets is not evidence of a weak search algorithm.

The prototype substantially increases planning cost while obtaining only small secondary-metric
improvements on this workload. Final local measurements:

| N / B / R | Algorithm | Median planning ms | Peak | P99 | Sum of step maxima |
|---|---|---:|---:|---:|---:|
| 16,384 / 16 / 4 | Current `lpt_best` | 28.467 | 10167 | 2436.13 | 259081 |
| 16,384 / 16 / 4 | Prototype | 897.888 | 10167 | 2436.13 | 259079 |
| 1,000,000 / 16 / 4 | Current `lpt_best` | 1470.557 | 36712 | 2524.01 | 15529586 |
| 1,000,000 / 16 / 4 | Prototype | 2829.916 | 36712 | 2524.01 | 15529583 |

It remains experimental. Broader real-data quality, memory,
GPU/DDP throughput and convergence effects are **假设/未验证**; synthetic exact results do not
establish a universal approximation ratio or practical training benefit.

### Verification and reproduction

116 opt-in checks passed: oracle validation against independent permutation enumeration,
partition counts, integer precision, invalid/budgeted inputs, prototype cardinality/coverage,
read-only inputs, determinism, incumbent preservation, exhaustive pair/exchange comparisons,
and the certified search trap. Both exact-comparison splits and the scaling benchmark passed
their behavioral assertions. No production code changed, so no broader product suite was run.

```bash
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_exact_rank_batches.py exps/test_balance_v3/test_refine_prototype.py -q
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --prototype
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --prototype --holdout
PYTHONPATH=src python exps/test_balance_v3/compare_refinement.py --large
PYTHONPATH=src python exps/test_balance_v3/inspect_neighborhood.py
```

Self-review: simplicity is preserved because no sampler options were added; supported oracle
and prototype input restrictions are checked before searching; hidden contracts distinguish
exact optimality, heuristic non-regression, float scoring, held-out evaluation, and CPU timings.

## Bounded Repair Verification (2026-09-05)

Production `lpt_best` now uses the bounded repair described above; the much heavier 24-pass,
multi-start prototype remains experimental. Only the internal LPT planner changes. Existing
sampler API/defaults, finite float-cost support, tail handling and static-cache contracts remain.

### Exact results

The earlier 150 development and 300 historical evaluation configurations are retained as
regression data. A fresh set of 300 configurations uses dataset seeds 200..219, with the same
N=12, three B/R shapes and five distributions, and fixed planning seed 7. Its first evaluation
followed selection of the repair policy; later work only removed redundant calculations and
refactored validation without retuning the search. Configurations reuse cost vectors across shapes.

| Split | Algorithm | Full-tuple optimum | Peak optimum | Mean peak gap | Worst peak gap |
|---|---|---:|---:|---:|---:|
| Development: 150 | Previous four-pass best | 100 | 139 | 0.1005% | 3.5000% |
| Development: 150 | Bounded repair | 135 | 150 | 0.0000% | 0.0000% |
| Historical: 300 | Previous four-pass best | 214 | 273 | 0.1405% | 3.1746% |
| Historical: 300 | Bounded repair | 276 | 297 | 0.0106% | 1.4815% |
| Fresh: 300 | Previous four-pass best | 206 | 273 | 0.1136% | 4.0000% |
| Fresh: 300 | Bounded repair | 282 | 299 | 0.0015% | 0.4566% |

The comparator asserts non-regression against the previous best, not against the independent
wide-search prototype. Neither new algorithm universally dominates the other. The frozen
two-batch trap is repaired from peak 130 to the exact optimum 126, including the full optimal
tuple (126, 125.91, 244). This assertion is also a production unit regression.

The remaining fresh-set peak miss is uniform/B4/seed219: bounded repair gives 220 versus the
exact optimum 219. These are measured small-instance outcomes, not a general 0.46% guarantee.
The experiment's exhaustive oracle remains separate from production and never guides its search.

### Planning cost

Final `compare_refinement.py --large` measurements use identical rounded lognormal costs,
three-run medians over planning seeds 7/13/29, and seed-7 quality. The previous four-pass
baseline includes final contiguous load-sorted output, as does production. Measurements exclude
the oracle and result assertions, and remain hardware/load-dependent CPU timings.

| N / B / R | Algorithm | Median ms | Peak | P99 | Sum of step maxima |
|---|---|---:|---:|---:|---:|
| 16,384 / 16 / 4 | Previous four-pass best | 25.231 | 10167 | 2436.13 | 259081 |
| 16,384 / 16 / 4 | Bounded repair | 28.258 | 10167 | 2436.13 | 259080 |
| 16,384 / 16 / 4 | Wide-search prototype | 877.874 | 10167 | 2436.13 | 259079 |
| 1,000,000 / 16 / 4 | Previous four-pass best | 1429.893 | 36712 | 2524.01 | 15529586 |
| 1,000,000 / 16 / 4 | Bounded repair | 1511.029 | 36712 | 2524.01 | 15529585 |
| 1,000,000 / 16 / 4 | Wide-search prototype | 2882.952 | 36712 | 2524.01 | 15529583 |

The million-sample increase is about 5.7%, not a general overhead guarantee. Peak already
matches its lower bound on this workload. Bounded repair does not match every secondary gain
of the much slower prototype: at N=16,384/B=64, P99 remains 3936 versus the prototype's 3935.
On fresh N=12 configurations, median planning time is approximately 6.46 ms versus 0.19 ms
for the four-pass baseline; fixed search overhead is proportionally larger on tiny datasets.

### Scope of verification

The minimum affected helper, sampler and real CPU DataLoader tests passed (513 checks),
including 54 focused bounded-repair checks. Budgets are checked by invocation counts, not
elapsed-time assertions. The 116 experimental oracle/prototype checks passed and remain opt-in.
No GPU memory, multi-process DDP throughput or training convergence was measured; those remain
**假设/未验证**. No whole-repository or installed-artifact suite was needed for this internal change.

```bash
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_repair.py tests/unit/data_utils/test_lpt_tiers.py tests/unit/data_utils/test_rank_batch_plan.py tests/unit/torch/ddp/test_lpt_sampler.py tests/unit/torch/ddp/test_qbalancedsampler.py tests/integration/torch/test_lpt_dataloader.py -q
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_exact_rank_batches.py exps/test_balance_v3/test_refine_prototype.py -q
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --prototype --holdout
PYTHONPATH=src python exps/test_balance_v3/compare_exact.py --fresh-holdout
PYTHONPATH=src python exps/test_balance_v3/compare_refinement.py --large
PYTHONPATH=src python exps/test_balance_v3/compare_lpt_tiers.py --large
```

Post-change review: simplicity (same strategy argument, no new search knobs), input coverage
(tails, repeated occurrences, arbitrary B, floats, subnormals and overflow), and hidden contracts
(fixed work budgets, shortlist limitations, changed ordering, float scores and no global-optimum
promise) are documented and checked. Initialization still performs all planning; epoch traversal
does not introduce new search branches or cache invalidation.

## Single-Pass Fast Swap Experiment (2026-09-05)

`exps/test_balance_v3/fast_swap.py` experiments with one vectorized exchange pass after the
current `lpt_fast` plan. **This is not integrated into production or registered as a strategy.**
The public three tiers, defaults and sampler contracts above remain unchanged.

### Algorithm and quality

Sort completed batch loads, pair the lightest with the heaviest, and leave an odd middle batch
untouched. Each batch participates in at most one exchange. Select up to four evenly spaced
sample positions per batch (the layered baseline keeps sample costs descending within each
batch). Evaluate at most 4x4 exchanges per pair in vectorized NumPy operations. Select the
exchange minimizing the pair load gap, apply all disjoint exchanges once, then recompute actual
loads and retain the new plan only if the full epoch objective improves. Use the same raw-sum
rounding guard as bounded repair when peak and P99 tie. Re-sort accepted batch loads into steps.
This preserves cardinality and selected occurrences, with O(N log N) time and O(N) memory.
Candidate positions are deliberately bounded; this is not exhaustive one-for-one local search.

`compare_fast_swap.py --exact` reuses the same 300 N=12 configurations (dataset seeds 200..219,
five distributions, three B/R shapes, planning seed 7) as the latest tier comparison. These are
known comparison data, not a new independent holdout. No exact result feeds the algorithm.

| Algorithm | Full-tuple optimum | Peak optimum | Mean peak gap | Worst peak gap |
|---|---:|---:|---:|---:|
| Current `lpt_fast` | 114/300 | 156/300 | 1.3143% | 14.8515% |
| One-pass experiment | 139/300 | 196/300 | 0.4790% | 9.8361% |
| Current `lpt` | 175/300 | 245/300 | 0.4484% | 11.1111% |
| Current `lpt_best` | 282/300 | 299/300 | 0.0015% | 0.4566% |

Peak-optimum hit rate rises from 52% to 65.33%; mean peak gap falls by about 63.6%. The
cross-layer example from the discussion improves from 118 to 110, not the exact optimum 106.
The experiment can beat the current middle tier on individual inputs; any eventual integration
must propagate the new fast candidate into higher-tier comparisons to preserve tier semantics.

### End-to-end timing

Final measurements ran separately from pytest, with one warmup per algorithm and seven timed
repetitions using planning seeds 7/13/29/7/13/29/7. Algorithm order rotates. Costs use dataset
seed 42; quality below uses planning seed 7. The table reports medians, not timing guarantees.
The timings include baseline planning, candidate generation, full-score acceptance and final
sorting; assertions and input generation are excluded. No GPU training is timed.

| N / distribution / B / R | Fast ms | One pass ms | Change | Peak before -> after | P99 before -> after |
|---|---:|---:|---:|---|---|
| 12 / uniform / 3 / 2 | 0.057 | 0.109 | +90.19% | 195 -> 188 | 194.79 -> 187.94 |
| 16,384 / uniform / 16 / 4 | 1.231 | 1.525 | +23.95% | 807 -> 807 | 807 -> 807 |
| 16,384 / lognormal / 16 / 4 | 1.699 | 1.649 | -2.98% | 10583 -> 10553 | 2838.13 -> 2808.13 |
| 16,384 / lognormal / 4 / 4 | 1.491 | 2.866 | +92.17% | 10194 -> 10180 | 1414.25 -> 1400.25 |
| 16,384 / lognormal / 64 / 4 | 1.463 | 1.669 | +14.10% | 12886 -> 12851 | 6242.85 -> 6207.85 |
| 1,000,000 / lognormal / 16 / 4 | 130.149 | 165.408 | +27.09% | 37131 -> 37100 | 2928.01 -> 2897.01 |
| 1,000,000 / uniform / 16 / 4 | 114.950 | 148.656 | +29.32% | 809 -> 809 | 809 -> 809 |
| 1,000,000 / outlier / 16 / 4 | 48.325 | 61.484 | +27.23% | 100015 -> 100015 | 16 -> 16 |

The small lognormal apparent speedup is measurement variability, not an algorithmic speedup:
the preliminary run measured 1.532 -> 1.880 ms on that same case. Millisecond-scale differences
should not be overinterpreted. The million-sample lognormal overhead was +33.97% in the
preliminary run and +27.09% in the final run; both show appreciable added work for limited
large-input peak improvement (31, about 0.0835%). The single-outlier baseline peak is already
optimal, so no peak improvement is possible on that distribution.

The experiment remains much faster than the middle tier on the large lognormal case
(165.408 ms versus 1101.739 ms), but its peak and P99 remain higher. Conclusion: the one-pass
idea improves small-instance hit rates, yet this bounded-position implementation does not
currently justify replacing the production fast tier on the measured large-data workload.
It remains an opt-in experiment rather than an automatic library change.

### Verification and limits

101 opt-in checks passed: vectorized versus scalar candidate search, one exchange per batch,
coverage/cardinality, odd and empty plans, seeded tails including padding/drop, read-only inputs,
determinism, global non-regression, invalid inputs and large finite arithmetic. The exact and
timing benchmarks assert coverage and non-regression on every compared input/seed. Production
code was not edited for this experiment, so no product-wide suite was run.

```bash
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_fast_swap.py -q
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --large
```

Self-review: the experimental interface reuses production validation and makes static traversal
explicit; finite costs, tails and fixed cardinality remain supported. It does not expose a new
public strategy, assume uniform permutations, or claim global optimality. The implementation
wraps the existing planner, so timing includes its initial final-sort plus any subsequent
candidate scoring/re-sorting. Integrating and reusing intermediate arrays might reduce overhead,
but that benefit is **假设/未验证**, as are real GPU/DDP throughput and convergence effects.

## Two-Pointer Fast Swap Experiment (2026-09-05)

`exps/test_balance_v3/fast_two_pointer.py` replaces the experimental fixed-four-position search
with a linear two-pointer search. It still performs only one disjoint-pair exchange pass and
is **not registered or integrated into production**. The original fast and both experimental
alternatives are compared in the same run; no sampler API or default changes.

### Search contract

Layered LPT produces descending sample costs within each batch, so no extra per-batch sort is
needed. For each heavy/light batch pair, move two pointers to find the exchange minimizing the
absolute remaining load difference. Moving the heavy pointer decreases the cost difference;
moving the light pointer increases it. Each pair visits at most 2B-1 positions. All active pairs
are evaluated in NumPy vectors, with completed pairs removed from further iterations.

The additional search is O(N), not O(NB), and uses O(N) memory including gathered batch costs.
Overall planning remains O(N log N). This does not imply the same constant cost as four-position
search: there can be up to 2B-1 vectorized iterations, and each examines an active subset.

The local optimum covers one-for-one exchanges for the chosen pairing, not arbitrary pairings,
multiple exchanges or the full epoch objective. Ties may choose different samples from the
four-position implementation. Actual batch costs and the full score are recomputed after the
pass, with the original fast plan retained on non-improvement and the existing raw-sum rounding
guard applied. The experiment does not claim full-objective dominance over four-position search
on arbitrary inputs. It preserves tail occurrences, fixed cardinality and static traversal.

### Exact comparison

The existing 300 cases have B=2/3/4: four selected positions already cover all samples, so both
exchange methods obtain identical quality scores on that set. To exercise the larger search
space, add 200 configurations: (N,B,R)=(12,6,2) and (16,8,2), dataset seeds 200..219, and the same
five distributions, with planning seed 7. Each has two rank-batches and is small enough for the
existing exhaustive oracle. The seeds are reused; this is an extension, not a new independent
holdout. Its results should not be generalized to arbitrary batch counts or real datasets.

| Algorithm | Peak optimum, existing 300 | Peak optimum, extended 200 | Extended mean peak gap | Extended worst peak gap |
|---|---:|---:|---:|---:|
| Original `lpt_fast` | 156 | 64 | 2.1771% | 25.5294% |
| Four-position pass | 196 | 83 | 0.9532% | 18.5882% |
| Two-pointer pass | 196 | 100 | 0.6628% | 9.5385% |
| Current `lpt` | 245 | 134 | 0.2563% | 3.0405% |
| Current `lpt_best` | 299 | 194 | 0.0097% | 0.4478% |

Full-tuple optimum counts on the existing set are 114/139/139/175/282 in table order; on the
extension they equal the peak-optimum counts. The latter is specific to this two-batch data,
not a general equivalence between the objectives. Extended hit rate rises from 41.5% for the
four-position pass to 50% for the two-pointer pass.

### Same-run timing and quality

Benchmarks ran separately from tests, with one untimed warmup per algorithm and seven timed
repetitions (seeds 7/13/29/7/13/29/7), rotating algorithm order. Times are end-to-end medians,
including baseline planning, pair search, quality checks and output re-sorting. Dataset seed is
42; reported quality uses planning seed 7. Timings vary with machine load and should be compared
within this table, not against earlier isolated runs.

For N=1,000,000, B=16, R=4, rounded lognormal costs:

| Algorithm | Median ms | Peak | P99 | Sum of step maxima |
|---|---:|---:|---:|---:|
| Original `lpt_fast` | 141.064 | 37131 | 2928.01 | 15529796 |
| Four-position pass | 200.837 | 37100 | 2897.01 | 15529779 |
| Two-pointer pass | 244.358 | 37019 | 2816.01 | 15529736 |
| Current `lpt` | 1199.705 | 36712 | 2524.01 | 15529590 |

Two-pointer planning adds approximately 43.5 ms (+21.7%) over the four-position pass, and
103.3 ms (+73.2%) over the original fast tier, while remaining about 4.9 times faster than the
middle tier. Its peak improvement from the original is 112 rather than 31. Peak is still above
the lower bound 36712: best single-pair exchanges do not eliminate the whole residual overload.

Other same-run timings (milliseconds):

| N / distribution / B / R | Original fast | Four-position | Two-pointer | Middle `lpt` |
|---|---:|---:|---:|---:|
| 12 / uniform / 3 / 2 | 0.130 | 0.382 | 0.467 | 0.265 |
| 16,384 / lognormal / 16 / 4 | 1.613 | 2.200 | 2.396 | 9.583 |
| 16,384 / lognormal / 4 / 4 | 1.522 | 3.622 | 3.153 | 12.425 |
| 16,384 / lognormal / 64 / 4 | 1.653 | 2.204 | 3.955 | 9.244 |
| 1,000,000 / uniform / 16 / 4 | 126.948 | 160.103 | 170.074 | 1148.400 |
| 1,000,000 / outlier / 16 / 4 | 54.683 | 72.215 | 79.906 | 1049.443 |

Uniform and single-outlier large cases show no quality improvement. At N=16,384/B=64,
two-pointer peak/P99 improve to 12550/5906.30 versus 12851/6207.85 with four positions, but
time is about 79% higher than four-position search. At B=4 the two exchange variants have the
same quality; different vectorization constants and timing noise affect which is faster.
Tiny-input overhead is proportionally large, so the experiment is not faster than `lpt` on
every shape. No universal latency ratio or approximation guarantee is established.

Given the accepted preference for a stronger fast tier while retaining a substantial speed gap
to the middle tier, two-pointer search is a promising candidate. Integration still requires an
explicit follow-up and propagation of the improved fast candidate into higher-tier comparisons.
Neither experimental variant changes production behavior in this evaluation.

### Verification

225 new opt-in checks passed, including linear search versus exhaustive B-by-B pair search on
integers/floats/ties, B=1..64, descending-order preconditions generated by the planner, empty/odd
plans, padding/drop, read-only inputs, deterministic output, occurrence preservation, invalid
inputs, and extreme/subnormal arithmetic. The missed-113-versus-1 exchange is a fixed regression.
Together with the original four-position tests, 326 checks passed. Both exact comparison sets
and the timing matrix passed coverage and non-regression assertions against original fast.
Production code was not modified, so no product-wide tests were run.

```bash
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_fast_swap.py exps/test_balance_v3/test_fast_two_pointer.py -q
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --extended
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --large
```

Post-change review confirms the same experimental interface and inherited validation, explicit
internal sorted-row requirements, fixed exchange count, and full-score fallback. Searches run
only during static planning, never during traversal. Real GPU memory, DDP throughput, convergence
and savings from integrating/reusing planner intermediates remain **假设/未验证**.

## Complete Five-Way Comparison (2026-09-05)

This follow-up adds current production `lpt_best` to every timing case, reports minimum and
maximum alongside median latency, and asserts production tier ordering as well as experimental
non-regression. It does not integrate either experimental pass. Earlier timing tables above
are historical runs; use this section for the complete same-run comparison.

Environment: AMD Ryzen 7 7840HS, WSL2 Linux 6.6.87.2, Python 3.13.12, NumPy 2.4.3.
Each algorithm receives one warmup and seven timed calls, with rotating execution order and
planning seeds 7/13/29/7/13/29/7. Input seed is 42. Input generation and output assertions are
excluded; complete static planning is included. Tests/exhaustive searches were run after timing.
The host is not isolated: observed min/max are variability indicators, not confidence intervals.

### Planning time

All values are medians in milliseconds. B is per-rank batch size; R is rank count.

| N / distribution / B / R | Original fast | Four-position | Two-pointer | `lpt` | `lpt_best` |
|---|---:|---:|---:|---:|---:|
| 12 / uniform / 3 / 2 | 0.119 | 0.234 | 0.207 | 0.087 | 7.362 |
| 16,384 / uniform / 16 / 4 | 1.196 | 1.558 | 2.145 | 7.957 | 32.539 |
| 16,384 / lognormal / 16 / 4 | 1.399 | 1.673 | 2.396 | 8.159 | 29.079 |
| 16,384 / outlier / 16 / 4 | 0.469 | 0.582 | 0.823 | 7.384 | 17.447 |
| 16,384 / lognormal / 1 / 4 | 1.232 | 1.241 | 1.163 | 1.148 | 1.136 |
| 16,384 / lognormal / 4 / 4 | 1.701 | 2.638 | 2.401 | 9.797 | 56.273 |
| 16,384 / lognormal / 64 / 4 | 1.547 | 1.660 | 3.474 | 8.022 | 19.827 |
| 16,384 / lognormal / 16 / 1 | 1.347 | 1.793 | 2.562 | 8.338 | 27.505 |
| 1,000,000 / lognormal / 16 / 4 | 136.592 | 192.154 | 220.530 | 1178.085 | 1469.637 |
| 1,000,000 / uniform / 16 / 4 | 127.823 | 153.047 | 170.713 | 1132.462 | 1455.115 |
| 1,000,000 / outlier / 16 / 4 | 59.748 | 75.282 | 80.064 | 1013.057 | 1155.891 |

Million-sample lognormal latency ranges in the same column order:
134.551–154.736 / 174.541–216.046 / 206.204–236.367 /
1075.601–1209.264 / 1428.707–1506.538 ms.

Two pointers cost 28.376 ms (+14.8%) over four positions and 83.938 ms (+61.5%) over original
fast on this long-tailed input, while remaining 5.34x faster than `lpt` and 6.66x faster than
`lpt_best`. On large uniform/outlier inputs the overhead over original fast is about 34%.
These are one-time planning costs, not per-step training slowdowns. Tiny N and B=1 do not
establish a speed ladder: fixed overhead and noise dominate, and B=1 cannot change batch loads.

### Quality on large data

For N=1,000,000, lognormal, B=16/R=4, planning seed 7:

| Algorithm | Peak | P99 | Sum of step maxima |
|---|---:|---:|---:|
| Original fast | 37131 | 2928.01 | 15529796 |
| Four-position | 37100 | 2897.01 | 15529779 |
| Two-pointer | 37019 | 2816.01 | 15529736 |
| `lpt` | 36712 | 2524.01 | 15529590 |
| `lpt_best` | 36712 | 2524.01 | 15529585 |

The largest sample costs 36712, a peak lower bound. Both higher tiers attain that bound on
this input; two-pointer peak remains 0.836% above it. Relative to original fast, two pointers
reduce peak by 0.302% and P99 by 3.825%. The peak improvement is modest because one outlier
dominates. No general approximation guarantee follows from this case.

On the large uniform case all five methods tie at peak/P99 809 and step-max sum 12625538;
on the large outlier case all tie at peak 100015, P99 16 and step-max sum 349999. Extra search
does not always buy quality. At N=16,384/B=64/R=4, two pointers improve peak/P99 to
12550/5906.30 from four-position 12851/6207.85, but median time rises from 1.660 to 3.474 ms.
Here `lpt_best` has lower P99 than `lpt` (3936 vs 3960.35), but higher step-max sum
(258025 vs 258017): the quality contract is lexicographic, not independent metric dominance.

### Exact-optimum reference and interpretation

The unchanged exhaustive datasets are rerun with explicit production-tier ordering assertions.
The 300-case set uses B=2/3/4; the 200-case extension uses B=6/8 and only two local batches.
They reuse seeds 200..219 and are not independent holdouts. Peak hit rates, respectively:

| Algorithm | Existing 300 | Extended 200 | Extended mean / worst peak gap |
|---|---:|---:|---:|
| Original fast | 52.0% | 32.0% | 2.1771% / 25.5294% |
| Four-position | 65.3% | 41.5% | 0.9532% / 18.5882% |
| Two-pointer | 65.3% | 50.0% | 0.6628% / 9.5385% |
| `lpt` | 81.7% | 67.0% | 0.2563% / 3.0405% |
| `lpt_best` | 99.7% | 97.0% | 0.0097% / 0.4478% |

Recommendation: the two-pointer pass is a useful fast-tier candidate under the accepted
planning-overhead tradeoff. It is neither globally optimal nor always faster than `lpt` on
tiny inputs. Higher tiers must inherit the improved fast candidate when integration happens;
the current middle tier can lose to experimental fast on individual cases. Production remains
unchanged. GPU memory, actual DDP throughput and convergence are **假设/未验证**.

### Reproduction and review

```bash
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --large
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --extended
PYTHONPATH=src python -m pytest exps/test_balance_v3/test_compare_fast_swap.py exps/test_balance_v3/test_fast_swap.py exps/test_balance_v3/test_fast_two_pointer.py -q
```

Review: the CLI stays simple (same flags); inputs cover multiple scales, distributions, B and R;
timing scope, seed reuse, static planning and non-optimality are explicit. The benchmark checks
coverage and full-score ordering outside timed regions. New opt-in checks validate all-tier
reporting and min/median/max field semantics without asserting unstable speed thresholds.
All 328 opt-in tests passed; both exact datasets and all 11 timing configurations passed their
coverage and ordering assertions. No production code changed in this follow-up.

## Two-Pointer Production Integration (2026-09-05)

The fast pass now executes directly between layered partitioning and higher-tier comparison,
before final step sorting. Experimental wrappers use `layered_baseline.py`, a frozen research
baseline, rather than inadvertently exchanging twice on the new production output. Comparison
output labels the baseline `layered` and reports production `lpt_fast` separately. The baseline
is not a public strategy or a production compatibility path.

Rerunning the same exhaustive sets gives:

| Production strategy | Peak hits / 300 | Peak hits / 200 |
|---|---:|---:|
| `lpt_fast` | 196 | 100 |
| `lpt` | 260 | 162 |
| `lpt_best` | 299 | 192 |

The earlier respective counts were 156/245/299 and 64/134/194. The changed starting point
improves middle-tier results but changes the best-tier local search trajectory: extended-set
best hits fall from 194 to 192, with mean peak gap rising from 0.0097% to 0.0115%; worst gap
remains 0.4478%. The contract orders current tiers, not results across algorithm revisions.
No additional old-start search was added. Historical timing/quality figures are not silently
relabelled as current results.

An eight-shape same-run timing smoke matrix passed coverage and tier ordering. For N=16,384,
lognormal B=16/R=4, medians are 2.305/8.882/28.145 ms for production fast/middle/best. Production
fast matches the experimental two-pointer peak/P99 (10471/2726.13). This is not a new
million-sample benchmark or a GPU throughput measurement.

Verification commands for the integrated planner and real DataLoader consumption:

```bash
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_two_pointer.py tests/unit/data_utils/test_lpt_fast.py tests/unit/data_utils/test_lpt_tiers.py tests/unit/data_utils/test_lpt_repair.py tests/unit/data_utils/test_rank_batch_plan.py tests/unit/torch/ddp/test_lpt_sampler.py tests/unit/torch/ddp/test_qbalancedsampler.py tests/integration/torch/test_lpt_dataloader.py -q
PYTHONPATH=src python -m pytest tests/unit/data_utils/test_lpt_tiers.py exps/test_balance_v3/test_fast_swap.py exps/test_balance_v3/test_fast_two_pointer.py exps/test_balance_v3/test_compare_fast_swap.py -q
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --exact
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer --extended
PYTHONPATH=src python exps/test_balance_v3/compare_fast_swap.py --two-pointer
```

Self-review: usage remains unchanged; correctness covers tails, floating-point extremes,
odd batch counts, fixed cardinality, tier ordering, determinism and cached DataLoader traversal.
The descending-row prerequisite is internal and satisfied immediately after layered planning;
there is no new caller requirement. Pair search matches exhaustive B-by-B search in unit tests.
The changed membership/order and cross-version best-tier limitation are explicit. Performance
experiments remain under `exps/`, not unit tests. GPU memory/convergence remain **假设/未验证**.
The integrated unit/DataLoader suite passed 776 tests; the separate experimental/tier run
passed 443 tests (including 328 experimental checks). Both exact runs and the timing smoke
matrix passed. No high-confidence review findings remain; retaining an additional old-start
best search is a separate quality/latency tradeoff, not part of this integration.
