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
- `tests/e2e/qexp`: release E2E tests that validate a built, installed wheel.
- `tests/demo`: manually runnable demonstrations. Demos do not provide regression
  protection.

`tests/functional` is a migration-only directory. Do not add new tests there;
classify new coverage as unit, integration, or e2e instead.

## Markers and Commands

Use module-level markers for suite ownership and add `slow`, `gpu`, or `ddp` only
when the runtime requirement is real. Markers must not hide a flaky test.

```bash
PYTHONPATH=src python -m pytest tests/unit -q
PYTHONPATH=src python -m pytest tests/integration -q
PYTHONPATH=src python -m pytest tests/e2e -q
tox run -e unit
tox run -e integration
tox run -e e2e
tox run -e release-e2e
```

Test placement and execution frequency are separate decisions. An integration test remains an
integration test even when it is intentionally excluded from the fast pull-request loop.

For qexp, use these stable module entry points:

```bash
tox run -e qexp-unit
tox run -e qexp-integration
tox run -e qexp-full
```

During development, run `qexp-unit` and the specific integration files that protect the changed
behavior. `qexp-integration` is the complete integration layer, while `qexp-full` combines all
qexp unit and integration tests for release validation. Pull-request CI does not run the complete
qexp integration layer; the publish gate does.

Default pytest collection excludes `tests/e2e/qexp`. `tox run -e release-e2e`
builds and validates a wheel from the current checkout; use
`tox run -e release-e2e --installpkg <wheel>` when validating a selected,
non-editable release artifact rather than a checkout or arbitrary installed package. The publish
workflow is the release gate that runs this installed-wheel E2E against the exact tagged artifact.

`CONTRACT_MATRIX.md` tracks public behavior and links it to protecting tests.
`TODOLIST.md` tracks unfinished test work. Coverage reports only show executed
code and do not prove public behavior is protected.
