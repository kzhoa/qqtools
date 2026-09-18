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

## Maintainer lanes

With Python 3.13 and the tooling dependencies prepared, maintainers may invoke
repository-defined tox environments directly:

```bash
tox run -e qexp-unit
tox run -e qexp-integration
tox run -e qexp-machine-lab
tox run -e artifact-e2e
tox run-parallel -e 'py{311,312,313,314}-artifact-smoke'
tox run -e release-e2e --installpkg /path/to/selected.whl
```

Bare `tox` runs only the complete source preflight. The explicit Python matrix
checks installed-package imports and CLI startup; missing interpreters fail.

The complete qexp Integration gate runs ordinary and lifecycle collections as
two four-worker phases within a shared 600-second budget. Reports go to
`qexp-gate-reports/`; `QEXP_GATE_REPORT_DIR` overrides that location.

Default source pytest collection excludes E2E. `artifact-e2e` builds a wheel
from the checkout; `release-e2e` validates the selected exact wheel. See
[test governance](../docs/development/test-governance.md#integration-ci-and-release-gates)
for when each gate is required.
