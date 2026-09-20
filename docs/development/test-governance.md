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
- The stable workflows and [release_preflight.py](../../scripts/release_preflight.py)
  define installed-artifact and versioned-release validation.

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
succeeds. A subsequent Dev Preflight run verifies the push. Promotion from `dev`
to `main` validates repository preflight and installed-artifact E2E against the
exact `dev` commit before the fast-forward. A release may reuse recent successful
Dev Preflight evidence for that same SHA, and main push artifact CI may reuse the
release artifact jobs, under the [evidence rules](development-workflow.md#reusing-gate-evidence).
Post-push checks are not substitutes for these pre-promotion gates.

Ordinary main/PR artifact CI validates installed wheels using the matrix in its
stable workflow. Canonical Python runs `artifact-e2e`; the other configured
versions run `artifact-smoke`. Source Unit/Integration coverage belongs to the
shared preflight, including the dev and promotion workflows.

`artifact-e2e` builds a non-editable wheel from the current checkout.
`release-e2e --installpkg <wheel>` validates the exact selected release artifact.
Installed tests clear `PYTHONPATH` and reject checkout `src/` imports. Use these
lanes for affected public delivery boundaries and whenever policy requires them.

For a versioned release, `scripts/release_preflight.py --target-version X.Y.Z`
requires a clean committed candidate, checks compatibility and export contracts,
and runs Unit, general Integration, and complete qexp Integration. Tagged
publishing separately runs `release-e2e` against the selected wheel before
publication. These distinct gates cannot substitute for one another.

## qexp resource and lifecycle protection

Before implementing or changing qexp Integration tests, apply the isolation and
cleanup requirements in this section. Lightweight checks on restricted platforms
must not be reported as real Linux lifecycle evidence.

Use the stable `qexp-unit`, `qexp-integration`, and `qexp-machine-lab` tox lanes
for their corresponding scopes; [tests/readme.md](../../tests/readme.md) lists
entry points. `qexp-unit` is the default module check. Select Integration files
by changed collaboration, persistence, CLI, or scheduling behavior. Use complete
`qexp-integration` when impact requires it, the user requests it, or release
policy mandates it. `qexp-machine-lab` is a convenience lane within Integration.

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
