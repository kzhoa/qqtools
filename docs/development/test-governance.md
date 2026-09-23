# Test governance

This guide defines how contributors and coding agents place, select, execute,
and report validation. The root [AGENTS.md](../../AGENTS.md) owns mandatory
integration and release gates. [Development workflow](development-workflow.md)
describes promotion; [developer tooling](developer-tooling.md) describes setup.

## Executable sources of truth

- [scripts/dev](../../scripts/dev) is the standard local entry point.
- [pytest.ini](../../pytest.ini) defines source collection and markers;
  [tox.ini](../../tox.ini) defines test environments.
- [run_preflight.py](../../scripts/ci/run_preflight.py) is the shared preflight
  command manifest. Local and CI setup must run the same gate.
- [CONTRACT_MATRIX.md](../../tests/CONTRACT_MATRIX.md) maps public behavior to
  regression evidence. Coverage alone does not prove a public contract.
- The stable workflows and exact-SHA release profile define installed-artifact
  and versioned-release validation.

Historical main-first ADRs are background, not an alternative to the current
feature/dev/main contract. Investigate and repair drift between current policy
and executable configuration rather than bypassing either.

## Layers and asset placement

Placement follows the behavior boundary, not the source directory or desired
execution frequency. Only Unit, Integration, and E2E are formal test layers.

| Location | Evidence or asset |
| --- | --- |
| `tests/unit/` | Deterministic local logic, models, decisions, and state transitions |
| `tests/integration/` | Real component collaboration, including filesystem, locks, processes, and lightweight training |
| `tests/e2e/` | Installed public CLI, YAML, or Python workflows |
| `tests/helpers/` | Shared fixture construction, mock data, and assertion helpers; no test cases |
| `tests/fixtures/` | Static or executable fixtures consumed by tests |
| `tests/demo/` | Manual demonstrations, not regression protection |
| `scripts/checks/` | Stable repository, configuration, and test-governance checks |

Unit tests should be fast and repeatable, without real network access, long
training, or complete multi-component workflows. Integration tests must assert
effective state, output, or side effects, not merely absence of exceptions.
E2E covers public delivery boundaries without duplicating internal combinations
already proven by Unit or Integration.

Do not add regression tests solely to preserve the cleanup history of a completed
refactor, such as asserting that old module files, imports, names, or source text
remain absent. Verify these cleanup tasks with a one-time search or review; do
not turn them into recurring repository scans. Retire existing checks when their
migration purpose ends instead of moving them behind `slow` or `stress` markers.
Keep tests that protect current behavior, supported compatibility contracts, or
explicit architectural boundaries; those tests must identify the continuing
contract they enforce rather than merely forbid a historical implementation.

Historical `tests/integration/functional/` remains Integration; do not create a
fourth layer. Do not add test cases directly under `tests/` or into helpers.
Exploratory scripts are not sufficient long-term regression protection; move
durable behavior checks into the appropriate formal layer.

Use `test_*.py` filenames and group tests by behavior. Register markers in the
applicable pytest configuration. Markers describe ownership, capabilities, or
runtime constraints; they must not conceal flaky tests or misclassify a layer.

## Development validation

For each executable-behavior increment, run the smallest test set that can
disprove the change. Relevant tests directly exercise the changed behavior,
assert its state/output/side effects, or protect a contract it implements.
Directory proximity alone is not evidence that a whole suite is affected.

Use the standard entry point with explicit paths or node IDs for narrow checks:

```bash
./scripts/dev test tests/unit/test_dev_entry.py -q
./scripts/dev test tests/integration/qexp/test_resource_isolation.py -q
```

Without explicit paths, `test` selects Unit tests. Maintainers may invoke tox or
pytest directly with the correct interpreter and dependencies; source checks
must import the current checkout. Do not widen a check solely to use a named
tox environment. Installed-artifact checks must instead reject checkout `src/`
imports, as described below.

Run added or changed tests; writing a test or collecting it is not execution
evidence. For moves, renames, or marker changes, collect the affected paths and
then run them. For shared fixtures, hooks, and helpers, identify actual consumers
and run representative affected tests; expand when consumers cannot be reliably
enumerated. Add or update effective regression coverage for executable-behavior
defects fixed in the task. Record unrelated defects without automatically
expanding scope; documentation defects use applicable static checks.

During iteration, pure documentation, formatting, and non-behavioral metadata
changes need applicable static/path/consistency checks, not product suites.
This does not waive mandatory integration or release gates.

## Expand verification when impact requires it

Expand beyond the directly changed tests when public API/CLI behavior, schema,
persisted state, process protocols, or shared infrastructure changes; when a
refactor's impact cannot be bounded; or when narrow checks reveal adjacent
regressions. Run the smallest broader set that addresses that impact. Cross-module
protocol changes need Integration or equivalent protocol evidence.

User requests and mandatory integration, compatibility, CI, or release policies
also require the corresponding broader checks. Review or commit preparation
alone does not justify unrelated suites. Development stopping rules must never
be used to omit a configured candidate gate.

## Integration, CI, and release gates

`./scripts/dev preflight` runs the complete shared source gate and accepts no
extra arguments. It requires Python 3.13, Linux, and system tmux. Missing
prerequisites must fail explicitly; do not skip real lifecycle coverage.

Before requesting feature promotion, run repository governance,
`ruff check src tests scripts`, `ruff format --check src tests scripts`, and all
relevant focused tests locally. A local `./scripts/dev preflight` run is
recommended for broad or high-risk changes and may be required by an explicit
task or policy, but it is not a universal local prerequisite for promotion.

A pre-existing baseline failure may be documented and investigated while work
continues, but it does not waive a mandatory integration or release gate.

The configured feature-promotion workflow must run complete preflight against
the exact candidate with `.dev/**` removed and must not change `dev` unless it
succeeds. It attests the resulting squash commit through GitHub OIDC/Sigstore;
the `dev` push verifies that provenance instead of repeating the identical source
gate. Promotion from `dev` to `main` validates repository preflight and
installed-artifact E2E against the exact `dev` commit before the fast-forward. A
provenance-only Dev Preflight push run is not reusable preflight evidence, so the
release runs the complete source gate. Main push artifact CI may reuse the
release artifact jobs under the
[evidence rules](development-workflow.md#reusing-gate-evidence).
Post-push checks are not substitutes for these pre-promotion gates. The sole
exception is the standard administrator release's owner-only commit limited to
`src/qqtools/version.py` and `CHANGELOG.md`: compatibility planning runs before
the version bump, then the direct `dev` push must obtain complete release-profile
Dev Preflight evidence for its exact SHA before `dev`-to-`main` promotion can
begin.

Ordinary main/PR artifact CI validates installed wheels using the matrix in its
stable workflow. Canonical Python runs `artifact-e2e`; the other configured
versions run `artifact-smoke`. Source Unit/Integration coverage belongs to the
shared preflight, including the dev and promotion workflows.

`artifact-e2e` builds a non-editable wheel from the current checkout.
`release-e2e --installpkg <wheel>` validates the exact selected release artifact.
Installed tests clear `PYTHONPATH` and reject checkout `src/` imports. Use these
lanes for affected public delivery boundaries and whenever policy requires them.

For a versioned release, Dev Preflight recognizes the owner-only metadata commit
and runs the release source profile against its exact SHA. That profile checks
version and changelog consistency, compatibility and export contracts, static and
governance rules, Unit, general Integration, the marked slow qpipeline/qexp
real-training and LMDB process cases, and qexp Integration.
Tagged publishing separately runs `release-e2e` against the selected wheel before
publication. These distinct gates cannot substitute for one another.

The qpipeline/qexp real-training case and LMDB child-process/GC cases are marked
`slow`. Feature preflight excludes them from the general Integration collection;
run their nodes explicitly when changing those workflows. Release preflight runs
them once. Source and installed-artifact pytest configurations print the 20 slowest
setup/call/teardown durations in every phase. Override with `--durations=100`
when investigating a larger set.

The complete local responsibility storage crash/replay, power-cut, persisted-image,
initialization, cleanup, and stage-build matrices are marked `slow`. Feature
preflight retains representative real-process crashes immediately before and at
the durable-intent barrier for publication, handoff, and cross-page retirement.
Run the complete storage file when changing storage/recovery behavior; release
preflight explicitly selects every slow matrix once. The randomized removal and
page-reuse test keeps a 65-record, one-reuse-round functional case; its original
140-record, four-round qualification is opt-in `stress`.

## Avoid repeated test startup work

Feature preflight collects qexp resource isolation, store crash boundaries, and
machine-lab checks in one pytest process. Torch multiprocessing tests and the
lifecycle gate retain separate processes for their isolation and gate semantics.
Release runs the full qexp gate instead of repeating the feature subset.

The local responsibility storage crash matrix preloads pytest and storage module
definitions into the forkserver before its first child. Each crash boundary still
uses a fresh child, real filesystem operations, and the original recovery
assertions. If a forkserver is already running, the preload request does not
change it; this optimization is not required for correctness.

The qdataset process checks share one parent probe per start method, while graph
collation and file-lock serialization still use separate fresh workers. Both
`spawn` and `forkserver` remain covered; the forkserver preloads module definitions.
These two cases are marked `slow` and selected explicitly by release preflight.
Feature preflight keeps the lightweight `fork` multi-worker read check. When
changing Dataset, DataLoader, file-lock behavior, or their dependencies, run
`./scripts/dev test tests/integration/torch/test_qdataset_process_boundaries.py`.
This selection is manual; feature preflight does not detect affected files.
Progress retry tests use short per-reporter retry delays while preserving retry,
backoff, cap, reset, and bounded shutdown assertions. Production timing defaults
are unchanged.

The JSON scanner keeps a three-block fragmentation case in normal validation;
the two 8 MiB memory qualifications are `stress`. Authority overload coverage
keeps a small saturated-cache/backlog scenario with a reduced test work budget;
the original 256-active-attempt scenario is `stress`. Both retain mixed lease
modes, renewal deadlines, deferred admission, continued discovery, and stale-clock
isolation assertions. These scale cases remain available with `--run-stress`.

## Optional stress qualification

When adding or maintaining tests, contributors must classify expensive cases:

- Apply `@pytest.mark.stress` to scale, sustained-load, throughput, and large
  working-set qualifications whose purpose is measuring capacity or performance.
- Apply `@pytest.mark.slow` to functional correctness tests whose execution cost
  makes them unsuitable for the fast development loop. Use pytest duration
  reports to inform this classification; do not classify by filename alone.
- For mixed parameter matrices, mark only the expensive parameters with
  `pytest.param(..., marks=pytest.mark.stress)` or `pytest.mark.slow`; retain
  ordinary functional and boundary parameters in normal validation.

Choose markers by test purpose, not merely elapsed time. A slow crash-recovery
correctness test requires `slow`, not `stress`. Neither marker permits weakening
assertions or removing required lifecycle coverage.

`stress` marks scale, sustained-load, throughput, and large working-set
qualification. Source pytest runs deselect these cases by default, including
feature and release preflight. Opt in explicitly:

```bash
./scripts/dev test tests/integration/qexp --run-stress -m stress -q --durations=20
```

Select a file or node to bound an experiment; `--run-stress` permits stress cases,
while `-m stress` selects only those cases. A path or `-m stress` alone does not
opt in. Deselected cases are not passing evidence. These experiments are manual,
not an additional release gate. Keep each selected case within ten minutes.

`slow` is separate: it describes functional tests unsuitable for a fast loop and
does not globally exclude them. Unit currently includes slow functional tests;
general Integration and the release profile use their documented selections.
Keep small functional and boundary regressions in normal validation when moving
large parameters to `stress`. Crash-recovery, durability, and lifecycle tests
must not be reclassified solely because they start subprocesses or run slowly.

## qexp resource and lifecycle protection

Before implementing or changing qexp Integration tests, apply the isolation and
cleanup requirements in this section. Lightweight checks on restricted platforms
must not be reported as real Linux lifecycle evidence.

Use `./scripts/dev test tests/unit/qexp -q` for qexp Unit tests and select
Integration files by changed collaboration, persistence, CLI, or scheduling
behavior. Use complete `qexp-integration` when impact requires it or the user
requests it; [tests/readme.md](../../tests/readme.md) lists entry points. The
release profile also excludes opt-in `stress` cases. Machine lab tests
remain part of qexp Integration and can be selected by path.

Every qexp Integration test must own isolated temporary roots, HOME/XDG,
runtime roots, tmux resources, and ledger-based cleanup checks. Never use the
production default authority, default tmux socket, or resources that can collide
with a developer's real qexp agents. Leaked participants, process groups, cleanup
diagnostics, or test-owned tmux sockets must fail the owning test.

`machine_lab` belongs only to Integration; `host_exclusive` belongs only to
installed-wheel E2E. `qexp_fast_io` changes fixture I/O behavior, not assertions
or test classification. Production default host authority is verified only
through installed-wheel public entry points.

Read the [product compatibility policy](../spec/qexp_product_spec.md#compatibility-policy)
and runtime invariants before changing protected behavior. Do not rewrite a
protected test to accommodate a regression. Refactoring test infrastructure
must preserve the protected behavior.

Non-Linux or restricted environments provide only the evidence they can
actually execute. Distinguish static/lightweight checks from real Linux process
and lifecycle coverage; never report a platform limitation as a passing test.

## Failures, reruns, and reporting

Classify failures as product regressions, test defects, or environment problems,
then diagnose the smallest failing case. Record reproductions for flaky tests;
a passing retry is not proof of stability. Never weaken assertions or hide real
collaboration boundaries to make a lane pass.

After a fix, rerun failed tests and checks whose evidence the fix invalidated.
Expand when it touches shared behavior or reveals adjacent risk. Avoid repeating
unaffected checks without a reason. During development, stop once relevant
behavior and applicable escalation conditions have sufficient evidence and no
later edit has invalidated it. Record blocked checks and residual risk; a blocked
mandatory gate remains a blocker to integration or release.

Report the behavior changed, the reason for the validation scope, actual commands
and results, and required checks that could not run with their limitations.
Collection, a new test file, or a manual observation alone cannot replace
behavioral assertions. Unrelated suites need not be listed as missing evidence.
