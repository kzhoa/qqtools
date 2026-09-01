# Test Suite Convention

Test placement is determined by the behavior boundary under test, never by the
location of production code.

## Test Layers

- `tests/unit`: one function, class, or local module. Avoid complete training,
  network access, and persistent side effects.
- `tests/integration`: two or more real modules collaborating. Tests must assert
  effective state or output, not only that no exception was raised.
- `tests/e2e`: a public YAML, CLI, or Python entry point completing a user-visible
  workflow with a tiny local model/dataset.
- `tests/e2e/qexp`: installed-wheel public qexp workflows.
- `tests/demo`: manually runnable demonstrations. Demos do not provide regression
  protection.

Historical functional coverage lives in `tests/integration/functional` until it can be split more
precisely. Do not add a fourth test layer.

## Markers and Commands

Use module-level markers for suite ownership and add `slow`, `gpu`, or `ddp` only
when the runtime requirement is real. Markers must not hide a flaky test.

```bash
PYTHONPATH=src python -m pytest tests/unit -q
PYTHONPATH=src python -m pytest tests/integration -q
tox run -e preflight
tox run -e unit
tox run -e integration
tox run -e artifact-e2e
tox run -e release-e2e
```

Test placement and execution frequency are separate decisions. An integration test remains an
integration test even when it is intentionally excluded from the fast pull-request loop.

For qexp, use these stable module entry points:

```bash
tox run -e qexp-unit
tox run -e qexp-integration
tox run -e qexp-full
tox run -e qexp-machine-lab
```

During development, run `qexp-unit` and the specific integration files that protect the changed
behavior. `qexp-integration` is the complete integration layer, while `qexp-full` combines all
qexp unit and integration tests for release validation. `preflight` is the local push gate.

Main CI runs only installed-artifact evidence. The canonical Python runs `artifact-e2e`; declared
non-canonical Python versions run `artifact-smoke`. Unit and Integration evidence belongs to the
local preflight, not normal CI.

Default pytest collection excludes `tests/e2e`. `tox run -e artifact-e2e` builds and validates a
wheel from the current checkout; use
`tox run -e release-e2e --installpkg <wheel>` when validating a selected,
non-editable release artifact rather than a checkout or arbitrary installed package. The publish
workflow is the release gate that runs this installed-wheel E2E against the exact tagged artifact.

`CONTRACT_MATRIX.md` tracks public behavior and links it to protecting tests.
Coverage reports only show executed code and do not prove public behavior is protected.
