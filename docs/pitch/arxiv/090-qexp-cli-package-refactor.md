---
doc_type: pitch
status: completed
updated_at: 2026-09-22
archived_at: 2026-09-22
implementation_authorized: true
---

# qexp CLI package refactor

## Decision and scope

Replace the oversized `src/qqtools/plugins/qexp/cli.py` module with a cohesive
`src/qqtools/plugins/qexp/cli/` package. Split parser construction, entry-point
orchestration, submission adaptation, and command-family dispatch along existing
responsibility boundaries while preserving all documented CLI behavior.

This is an internal structural refactor. It does not authorize changes to command
names, option placement, Project selection, machine authority, activation behavior,
output schemas, human presentation, exit codes, persisted data, or runtime
protocols. The active
[product](../../spec/qexp_product_spec.md) and
[runtime](../../spec/qexp_runtime_spec.md) specifications remain authoritative for
those contracts. The completed
[CLI consolidation pitch](089-qexp-cli-consolidation-and-daily-operations.md)
records historical rationale and delivery context; it is not a normative contract.
The current implementation and regression tests provide the characterization
baseline for behavior not otherwise specified, without overriding the public
specifications.

Implementation is authorized only for the internal refactor and behavior-preserving
test changes defined here. Do not modify workflow infrastructure or publish changes
under this authorization.

## Evidence and motivation

The inspected working tree on 2026-09-22 contains a 2,417-line `cli.py` with two
dominant functions:

- `build_parser()` spans 441 lines and owns the complete argparse tree, common
  option placement, parser bindings, help text integration, and parser-specific
  value conversion.
- `main()` spans 934 lines and combines raw-argv classification, parse-error
  translation, context resolution, machine-local commands, Project commands,
  submission orchestration, agent activation, rendering, and exit-code policy.

The module also contains submission-only normalization and result/error payload
construction, continuous observation validation, output-kind enforcement, and
small command-family adapters. Existing domain operations already live in
`commands/`, `agent/`, `runtime/`, and other qexp modules, so the problem is not
missing service extraction. The remaining issue is that the CLI adapter layer is
concentrated in one file.

The installed entry point is currently:

```toml
qexp = "qqtools.plugins.qexp.cli:main"
```

Tests import `main` and `build_parser` from that path. Several tests also patch
private implementation symbols through `qqtools.plugins.qexp.cli`; those patch
paths are internal test coupling rather than user-facing Python API contracts.

## Goals

- Make each CLI module readable without loading the complete command tree and
  every command implementation into one working context.
- Keep one visible CLI package rather than adding multiple `cli_*.py` files to
  the qexp package root.
- Make the installed entry point and repository imports name the modules that
  actually own `main` and `build_parser`.
- Keep context resolution, output validation, exception translation, and exit
  policy centralized rather than duplicating them across handlers.
- Make dependency direction explicit and prevent handler modules from importing
  the package initializer or entry point.

## Non-goals

- No CLI redesign, command rename, option change, or output change.
- No new command framework, plugin system, event bus, abstract base class, or
  declarative parser DSL.
- No movement of domain behavior from existing `commands/`, `agent/`, or
  `runtime/` modules into the CLI package.
- No one-file-per-command decomposition.
- No forwarding module, compatibility facade, or re-export shim for the former
  `cli.py` import surface, including `main`, `build_parser`, underscored helpers,
  or monkeypatch targets.
- No public specification, compatibility-registry, or persisted-format change
  unless implementation discovers an unavoidable behavioral change and obtains
  a separate decision.

## Target package

```text
src/qqtools/plugins/qexp/
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── entrypoint.py
│   ├── parser.py
│   ├── submission.py
│   ├── local_handlers.py
│   └── project_handlers.py
├── commands/
├── agent/
├── runtime/
└── ...
```

`cli.py` and `cli/` must not coexist. The implementation must replace the file
with the package in one coherent change so Python never has two candidates for
the same import path.

### `cli/__init__.py`

Remain empty apart from an optional package docstring. It must not import or
re-export `main`, `build_parser`, handlers, service aliases, or private helpers.
Callers use the owning modules directly:

```python
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.cli.parser import build_parser
```

The former `qqtools.plugins.qexp.cli:main` and
`from qqtools.plugins.qexp.cli import main, build_parser` paths intentionally
stop working. qexp's internal Python module layout is not a supported public API,
and this refactor must update repository callers instead of preserving obsolete
ownership through a shell.

### `cli/__main__.py`

Retain direct module execution without duplicating startup behavior:

```python
from .entrypoint import main

raise SystemExit(main())
```

This is the standard Python package execution adapter, not a compatibility
facade: it exists only for `python -m qqtools.plugins.qexp.cli` and exposes no
imported API from `cli.__init__`.

### `cli/parser.py`

Own the complete argparse boundary:

- `_QexpArgumentParser` and CLI-specific parse exceptions;
- raw-argv classification needed to select structured parse diagnostics;
- duplicate common-option detection and normalization;
- parser-only value converters;
- common-option registration;
- `build_parser()` and all `CommandSpec` bindings.

It may import fixed option vocabularies needed to construct choices, but it must
not import handler modules or execute domain operations. Parser construction
continues to produce the canonical `CommandSpec.handler`, `context`, and `output`
metadata used by dispatch and output validation.

### `cli/entrypoint.py`

Own cross-command orchestration only:

- acquire raw argv and invoke the parser;
- translate parser failures into the established human or JSON diagnostics;
- install and reset the active `CommandSpec` output contract;
- perform shared pre-dispatch validation;
- resolve `RootConfig` and `ExecutionContext` exactly once when required;
- select the local, submission, or Project handler;
- retain the single outer exception-to-output and exit-code boundary.

The entry point must not contain the Task, Group, agent, upgrade, migration, or
submission implementation branches after their handlers are extracted. Handler
exceptions continue to flow to this module unless a command owns a documented
special outcome such as submission interruption or Task wait JSON failure.

### `cli/submission.py`

Own only the CLI adaptation around the existing submission service:

- raw submission option and payload-boundary inspection;
- command/file mode validation;
- `SubmissionRequest` construction;
- preview, committed-result, and failure payload construction;
- submission-specific output and exit mapping;
- post-commit local activation without relabeling a verified commit.

Durable preparation, idempotency, staging, commit, recovery, and Task/Group
truth remain in their current domain modules. This split must preserve the
protected submission workflows and schema-version-1 JSON result contract.

### `cli/local_handlers.py`

Own handlers that do not use ordinary selected-Project resolution:

- machine initialization;
- Project initialization, registration, inventory, enablement, disablement, and
  removal;
- saved CLI context operations;
- machine-global agent lifecycle and resource configuration;
- machine-global upgrade coordination;
- explicit migration routes and bounded retired-command diagnostics.

These handlers may resolve an explicit Project where the command contract
requires it, but they must not fall through to ordinary cwd, environment, or
saved-context Project selection.

### `cli/project_handlers.py`

Own handlers that run after ordinary or explicitly required Project context has
been resolved:

- typed Project configuration;
- Task operations and observation;
- Group and Worker Set operations;
- bounded status and machine observation;
- doctor, cleanup, and operation inspection.

The module receives resolved context and explicitly supplied collaborators for
cross-cutting actions such as emission and local-agent activation. It must not
re-resolve Project identity or reconstruct machine authority from argparse
values.

## Dependency direction

```text
cli.__main__ ───────────────→ cli.entrypoint
cli.entrypoint ─────────────→ cli.parser
             ├──────────────→ cli.local_handlers
             ├──────────────→ cli.project_handlers
             └──────────────→ cli.submission

cli.parser ─────────────────→ commands.registry and fixed option vocabularies
cli.*_handlers ─────────────→ existing commands / agent / runtime services
```

`cli.__init__` has no outgoing imports. No CLI submodule imports it explicitly.
`parser.py` does not import handlers. Handler modules do not import
`entrypoint.py`. Cross-cutting callables that would otherwise create a cycle are
passed explicitly or moved to the narrowest neutral owner only when they are
genuinely shared.

## Supported execution and internal import paths

The user-facing execution surfaces remain stable:

- the `qexp` console script;
- direct module execution through `python -m qqtools.plugins.qexp.cli`;
- every documented command, option, help contract, output, and exit code.

The console entry in `pyproject.toml` changes to the true owner:

```toml
qexp = "qqtools.plugins.qexp.cli.entrypoint:main"
```

Repository imports change to `cli.entrypoint`, `cli.parser`, or the other owning
submodule. Underscored functions, imported service aliases, module globals, and
test monkeypatch paths are internal. Tests patch the module that consumes a
dependency after the split. Repository code outside tests must also move to the
owning module; it must not cause a forwarding facade to be retained.

## Implementation sequence

1. Add or strengthen characterization tests for parser leaves, common-option
   placement, JSON parse failures, help disclosure, installed entry-point import,
   and direct module execution.
2. In one atomic tree transition, replace `cli.py` with the `cli/` package, move
   the complete parser boundary into `parser.py`, and move the remaining
   orchestration and not-yet-extracted command branches into `entrypoint.py`.
   Add `__main__.py`, update `pyproject.toml` to `cli.entrypoint:main`, and update
   repository imports of `main` and `build_parser` to those existing owning
   modules. Keep `cli/__init__.py` empty and do not introduce placeholder
   re-exports or forwarding modules.
3. Extract `local_handlers.py` and `project_handlers.py` one command family at a
   time. Keep resolution and the outer error boundary in `entrypoint.py`; after
   each move, update the affected callers and monkeypatch targets to the module
   that now consumes the dependency.
4. Extract `submission.py` last because its JSON outcomes, exit codes,
   interruption behavior, idempotency diagnostics, and post-commit activation
   form the most sensitive CLI contract.
5. Verify all repository imports and patch targets name their final owning
   modules, then verify the dependency graph has no CLI-package cycles or
   compatibility forwarding layers.

Each step must leave a runnable CLI and independently reviewable tests. Do not
combine package conversion with behavioral cleanup or command redesign.

## Acceptance criteria

- [ ] `src/qqtools/plugins/qexp/cli.py` is replaced by the target `cli/` package;
      no same-name file/package pair exists.
- [ ] The installed `qexp` entry point resolves through
      `qqtools.plugins.qexp.cli.entrypoint:main` and invokes the same `main(argv)`
      behavior.
- [ ] `cli/__init__.py` imports and exports no implementation symbols.
- [ ] Repository callers import `main`, `build_parser`, helpers, and patch targets
      from their true owning modules; the former `cli.py` import surface is not
      forwarded.
- [ ] Direct module execution remains available through `cli/__main__.py`.
- [ ] The parser exposes the same command leaves, `CommandSpec` values, option
      placement, help text, defaults, and conflict behavior.
- [ ] Human output, JSON payloads, stderr diagnostics, and exit codes remain
      behaviorally identical for all finite commands.
- [ ] Continuous Task watch/log behavior, terminal restoration, broken-pipe
      handling, and interruption exits remain unchanged.
- [ ] Submission command/file validation, dry-run, quiet mode, schema-version-1
      JSON, idempotency results, and post-commit activation behavior remain
      unchanged.
- [ ] Read-only Project commands remain usable without a local binding, while
      mutating commands continue to require verified authority.
- [ ] Machine initialization, enrollment, agent lifecycle, migrations, and
      upgrade commands preserve their distinct resolution and activation rules.
- [ ] No CLI submodule imports the package initializer, parser-to-handler dependency,
      or handler-to-entrypoint dependency.
- [ ] Existing domain services do not acquire CLI rendering or argparse
      responsibilities.

## Validation plan

### Characterization and package-boundary coverage

Before moving implementation, strengthen
`tests/unit/qexp/test_cli_consolidated_tree.py` with a checked-in baseline of
every command path and its `CommandSpec.handler`, `context`, and `output`, plus
parser defaults, common-option placement, duplicate-option behavior, normalized
help text, and the `submit --` payload boundary. Add a raw-argv matrix covering
human and JSON parse failures so exception routing can be compared without
depending on implementation symbols.

As part of the atomic module-to-package transition, add focused package-structure
coverage that runs in clean Python processes and verifies:

- `cli.__init__` does not expose `main`, `build_parser`, handlers, or service
  aliases;
- `cli.parser`, `cli.entrypoint`, each handler module, and `cli.submission` import
  independently without a cycle;
- `python -m qqtools.plugins.qexp.cli --help` succeeds through `cli.__main__`;
- the installed distribution declares
  `qexp = qqtools.plugins.qexp.cli.entrypoint:main` and both that console script
  and direct module execution load code from the installed wheel rather than the
  source checkout; and
- a static dependency check rejects parser-to-handler, handler-to-entrypoint, and
  CLI-submodule-to-package-initializer imports.

These tests protect observable behavior and declared dependency boundaries. They
must not retain old import paths or patch aliases solely to make the refactor pass.

### Focused iteration

After the atomic package transition and after each handler-family extraction, run
the directly affected characterization and CLI contract tests:

```bash
./scripts/dev test \
  tests/unit/qexp/test_cli_consolidated_tree.py \
  tests/unit/qexp/test_cli_compatibility.py \
  tests/unit/qexp/test_tmux_cli.py \
  tests/integration/qexp/test_cli_contract.py \
  tests/integration/qexp/test_output_format.py \
  tests/integration/qexp/test_cli_project_selection.py
```

Run the owning contract files when their handler family moves. At minimum,
machine/enrollment and Project control extraction must include:

```bash
./scripts/dev test \
  tests/integration/qexp/test_machine_enrollment.py \
  tests/integration/qexp/test_dynamic_gpu_policy.py \
  tests/integration/qexp/test_availability_transitions.py \
  tests/integration/qexp/test_indexed_group_cancel.py \
  tests/integration/qexp/test_indexed_worker_removal.py
```

After submission extraction, run its transactional and Project-resolution
contracts:

```bash
./scripts/dev test \
  tests/integration/qexp/test_submission_contracts_and_transactions.py \
  tests/integration/qexp/test_explicit_home_submission.py
```

Run continuous observation, including the real subprocess path that invokes the
package with `python -m`:

```bash
./scripts/dev test \
  tests/integration/qexp/test_continuous_observation_cli.py \
  tests/integration/qexp/test_task_observation_cli.py \
  tests/integration/qexp/test_continuous_observation_processes.py
```

### Final candidate gates

Because every CLI dispatch path and the installed entry point move, the final
impact cannot be bounded to the focused files. Before requesting integration,
run the repository checks, the complete qexp Unit and Integration lanes, and the
installed-wheel E2E lane:

```bash
python scripts/checks/check_repository_governance.py
ruff check src tests scripts
ruff format --check src tests scripts
tox run -e qexp-unit
tox run -e qexp-integration
tox run -e artifact-e2e
```

`artifact-e2e` is the required local evidence for package discovery and the
console entry point; a source-tree E2E invocation does not replace it. The
configured promotion workflow must still run complete preflight against the
exact candidate before `dev` advances. Local `./scripts/dev preflight` remains
recommended but is not an additional prerequisite unless required by the active
workflow or a later risk decision.

Any observed CLI behavior change must stop the refactor. Update the authoritative
public specification and obtain the required compatibility decision before
continuing; do not rewrite characterization or protected tests to accept a
regression.

## Risks and controls

### Same-name module/package transition

Python must not encounter both `cli.py` and `cli/`. Perform the replacement as
one coherent tree change and test both console-script loading and ordinary
imports immediately.

### Circular imports through the package initializer

A non-empty `cli/__init__.py` could recreate the monolith as an import hub. Keep
it free of implementation imports, enforce the dependency direction above, and
test each submodule import in a clean Python process.

### Argparse default and option-placement drift

Common options currently appear at multiple parser levels so they may occur
before or after command paths. Mechanical movement can change inherited
defaults, duplicate handling, payload separation, or help output. Freeze these
behaviors before moving parser construction.

### Exception and output drift

Moving handlers can accidentally catch an exception at a different layer or
bypass active `CommandSpec` output validation. Preserve one outer exception
boundary and pass the canonical emitter into handlers rather than adding local
generic error wrappers.

### Test seams mistaken for compatibility promises

Existing tests patch service names imported into `cli.py`. After extraction,
patch the consuming handler module or pass a dependency explicitly. Avoid
private re-export shims; repository callers must migrate to the owning module.

### Submission regression

Submission combines raw argv, Project discovery, normalization, durable commit,
presentation, and activation failure handling. Extract it last, preserve the
existing domain boundary, and require decoded JSON and exit-code regression
tests for every outcome.

## Documentation impact and lifecycle

No public documentation update is expected for a behavior-preserving
implementation. The package structure is internal, and public operation must
remain understandable without this ignored pitch. If implementation changes a
documented CLI or compatibility contract, update the authoritative public spec
before integration rather than treating this pitch as the contract.

When implementation and required validation are complete, update this pitch's
status and archive it under `docs/pitch/arxiv/` using the next sequential archive
number. Do not archive it merely because the package directory has been created;
all acceptance criteria and regression coverage must be complete.
