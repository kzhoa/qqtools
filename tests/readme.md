# Test suite navigation

See [test governance](../docs/development/test-governance.md) for test placement,
verification selection, isolation, mandatory gates, and reporting requirements.
See [developer tooling](../docs/development/developer-tooling.md) for setup.
Run the following commands from the repository root.

## Directories

| Directory | Purpose |
| --- | --- |
| `unit/` | Local deterministic behavior |
| `integration/` | Real component collaboration; historical `functional/` remains here |
| `e2e/` | Installed public workflows, including qexp in `e2e/qexp/` |
| `helpers/` and `fixtures/` | Shared test support and consumed assets |
| `demo/` | Manual demonstrations |

[CONTRACT_MATRIX.md](CONTRACT_MATRIX.md) maps public contracts to test evidence.

## Common local commands

```bash
./scripts/dev test
./scripts/dev test tests/unit/test_dev_entry.py -q
./scripts/dev test tests/integration/qexp/test_resource_isolation.py -q
./scripts/dev preflight
```

`test` defaults to Unit tests; explicit paths/node IDs select other source tests.
`preflight` is the complete shared gate and accepts no extra arguments.
The qpipeline/qexp real-training case and LMDB process/GC cases carry the
`slow` marker: feature preflight excludes them, release preflight runs them once,
and maintainers run their node IDs explicitly when changing those workflows.
All source and installed-artifact pytest phases report their 20 longest durations.
Use `--durations=100` for a longer report. Complete storage crash matrices are
`slow`: run the storage file when changing recovery behavior; release runs all
matrices, while feature preflight retains representative crash coverage.

## qexp startup profiling

The isolated real-runner workload can retain startup phase and write measurements:

```bash
./scripts/dev test tests/integration/qexp/test_authority_workload.py -q \
  -p tests.helpers.qexp.authority_measurement --authority-profile-startup \
  --authority-workload-profile='{"bindings":4,"hold_seconds":5}' \
  --authority-workload-output=/tmp/qexp-startup.json
python -m tests.helpers.qexp.authority_reports /tmp/qexp-startup.json
```

Profiling lives in `tests/helpers/qexp/`; production entrypoints have no profiling
switches. It observes agent imports, tmux window creation and command sending,
shell/Python entry, runner imports, authority-lock acquisition, and intent publication.
The startup wrapper invokes the real runner with unchanged arguments and deadlines.
Agent file-write observations include nested fsync counts and elapsed time. They do
not include runner/guardian I/O, and nested or concurrent durations must not be added
as disjoint wall time. The observer adds overhead, so compare identical harnesses in
alternating runs and confirm behavior with profiling disabled.
The measurement agent uses a 0.1-second loop. Since heartbeat timestamps have
whole-second precision, this workload has more identical snapshot writes than the
normal five-second loop; do not extrapolate its write-reduction percentage to that
cadence.

Shell phase timestamps require Bash's `EPOCHREALTIME` and a stable host clock;
unavailable shell timestamps and missing runner reports remain explicit. Reports
retain invalid/incomplete profile status separately from the workload outcome, so
a broken profile cannot replace the original workload failure. The option is
explicitly opt-in; inherited profiling environment variables are cleared when it
is disabled. Reports
include isolated filesystem paths and should be summarized before publication.
Keep the existing 15-second convergence assertion; a failed profile is failure
evidence, not permission to extend its budget.

## Maintainer lanes

With Python 3.13 and the tooling dependencies prepared, maintainers may invoke
repository-defined tox environments directly:

```bash
./scripts/dev test tests/unit/qexp -q
tox run -e qexp-integration
./scripts/dev test tests/integration/qexp/test_machine_lab.py -q
tox run -e artifact-e2e
tox run -e artifact-smoke
tox run -e release-e2e --installpkg /path/to/selected.whl
```

Bare `tox` runs only the complete source preflight. Run `artifact-smoke` under
each supported Python version for installed-package imports and CLI startup;
missing interpreters fail.

The complete qexp Integration gate runs ordinary and lifecycle collections as
two four-worker phases. It lets both phases finish and rejects a successful run
that exceeds the shared 600-second soft budget, preserving their complete timing
reports. A shared 1200-second hard timeout still terminates a stuck process tree.
Reports go to `qexp-gate-reports/`; `QEXP_GATE_REPORT_DIR` overrides that location.

Default source pytest collection excludes E2E. `artifact-e2e` builds a wheel
from the checkout; `release-e2e` validates the selected exact wheel. See
[test governance](../docs/development/test-governance.md#integration-ci-and-release-gates)
for when each gate is required.

### Optional stress experiments

Large scale and memory qualifications marked `stress` are excluded by default,
including release. Run a selected experiment explicitly, for example:

```bash
./scripts/dev test tests/integration/qexp/test_observation_scale.py --run-stress -m stress -q --durations=20
```

Small boundary cases remain in regular validation. See
[test governance](../docs/development/test-governance.md#optional-stress-qualification).
