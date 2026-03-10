# Tests Directory Convention

This document defines how test files should be organized under `tests/` and what
source scope each test suite is responsible for.

## Directory Responsibilities

- `tests/unit`
  - Fast, isolated tests for single modules/functions/classes.
  - Prefer no filesystem/network side effects.
  - Default home for core modules under `src/qqtools/**` unless explicitly
    classified as `full` (functional/integration scope).

- `tests/functional`
  - Integration and user-side behavior tests.
  - Can cover interactions across multiple modules and realistic workflows.
  - Includes plugin behavior, plugin-composed flows, and CLI behavior that
    depends on plugin execution paths.

- `tests/demo`
  - Runnable examples for logging/format/output demonstration.
  - Not a substitute for assertions in `tests/unit` or `tests/functional`.

## Coverage Classification Rule

- `base`
  - Modules that are not `full` and do not rely on optional std deps listed in
    `pyproject.toml` (`lmdb`, `tqdm`, `requests`).
  - Expected primary coverage: `tests/unit`.

- `standard`
  - Modules depending on optional std deps (`lmdb`, `tqdm`, `requests`).
  - Expected primary coverage: `tests/unit` (with dependency-aware test design).

- `full`
  - Modules requiring functional/integration tests in `tests/functional`.
  - Includes:
    - plugin module tests (`src/qqtools/plugins/**`)
    - plugin-based multi-module tests or user-side simulation tests
    - CLI command tests that involve plugin flows (`src/qqtools/cli/**`)

## Module-to-Suite Mapping

- `src/qqtools/plugins/**` -> `tests/functional`
- `src/qqtools/cli/**` -> `tests/functional` (when command path depends on plugins)
- `src/qqtools/qm/**` -> `tests/unit`
- other `src/qqtools/**` modules -> `tests/unit` by default, unless promoted to
  `full` for integration reasons

## Tracking and Source of Truth

- Detailed uncovered-module tracking lives in:
  - `tests/TODOLIST.md`
- When rules in this file and TODO entries conflict, this file is the convention
  source; update TODO entries accordingly.

## Historical Notes

- Older rough planning content was replaced by this convention-focused document.
- Keep future updates here concise and policy-oriented; keep progress snapshots in
  `tests/TODOLIST.md`.
