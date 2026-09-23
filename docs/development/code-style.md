# Code style

This guide describes how to write and review maintainable qqtools code. The root
[AGENTS.md](../../AGENTS.md) owns execution rules and required gates;
[pyproject.toml](../../pyproject.toml) owns formatter and lint configuration.
Apply the guidance to new and changed code without unrelated style rewrites.
Existing implementation patterns are context, not automatically requirements.

## Automated formatting and lint

Let Ruff handle layout and import ordering. The current configuration uses a
120-character formatting target, LF line endings, and the Python 3.11 target.
Use the formatter's output rather than manually enforcing competing wrapping,
quote, or blank-line conventions. The line-length target is not a guarantee
that every formatted line fits within 120 characters.

The lint gate selects a limited set of checks for syntax, name resolution,
unused local assignments, selected expressions and naming, and import ordering.
It does not enforce every recommendation in this guide. Assigned lambdas are
allowed; do not introduce new lint restrictions merely to enforce a preference.

From an environment prepared using [developer tooling](developer-tooling.md),
the repository checks are:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
```

During editing, format the files you changed with `ruff format <paths>` and
review any automatic fixes. Full integration gates remain defined by the root
contract and [development workflow](development-workflow.md).

## Names and public interfaces

Use `snake_case` for functions, methods, variables, and modules; use `PascalCase`
for new classes and `UPPER_SNAKE_CASE` for constants. Existing public names such
as `qDict` and `qPipeline` are established API spellings. Preserve them, and use
the `q` prefix only when extending an existing naming family where it helps
users recognize the interface.

Choose names that explain the value or operation at the call site. Short names
such as `k` and `v` are reasonable in small, obvious dictionary operations;
state, ownership, units, and lifecycle concepts deserve descriptive names.
A leading underscore marks an internal implementation detail, not an access
control mechanism. Predicate names should describe the question they answer;
validation functions should make their failure behavior clear.

Keep exports deliberate. When changing a public API, review package re-exports,
`__all__` where present, type stubs, tests, and documentation together. Do not
rename an established interface solely to satisfy this guide; assess its
compatibility impact through [compatibility governance](compatibility-governance.md).

## Types and data contracts

Annotate new or substantially changed public interfaces and nontrivial internal
boundaries. Include meaningful return types and represent optional values
explicitly. Prefer Python 3.11-compatible forms such as `list[str]`,
`dict[str, int]`, and `str | None` in new code; do not bulk-convert untouched
annotations for consistency alone.

Package metadata allows installation on Python 3.11 and later, but CI tests only
Python 3.12 and later; Python 3.11 compatibility is not verified. Standard
development tooling uses Python 3.13. Do not introduce newer syntax or standard-library APIs
into library code without accounting for the package's declared minimum version.

Use `Any` only where a boundary is intentionally dynamic. Prefer a concrete
type, a protocol, or a small structured record when consumers rely on known
fields or operations. Avoid redundant local annotations when inference is
clear. Type hints do not replace runtime validation of external data.

## Imports and dependency boundaries

Use explicit imports and let Ruff group standard-library, third-party, and
first-party imports. Keep optional dependencies behind the feature boundary
that needs them; unrelated imports should not eagerly require an optional extra.
Check dependency declarations and relevant import tests when changing that
boundary.

Reuse an existing lazy-import facility when its deferred behavior is needed.
Dynamic global injection and lazy proxies are not the default for new modules:
ordinary imports are easier to inspect when startup cost and optional dependency
contracts do not require deferral. Catch import failures narrowly enough that a
broken dependency is not silently reported as an absent optional package.

## Control flow, validation, and errors

Keep state transitions and side effects visible. Use comprehensions for simple
transformations and loops for branching or multiple effects. Extract helpers
around meaningful operations, not arbitrary line counts. Prefer standard Python
and existing project utilities before adding custom machinery.

Use dynamic class creation, monkey-patching, or custom performance paths only
when a concrete requirement justifies them. Explain the constraint and provide
appropriate behavioral tests; performance claims need measurements. An existing
specialized implementation is not a reason to repeat the technique elsewhere.

Validate external inputs with explicit exceptions that identify the invalid
value or violated constraint. Reserve `assert` for internal invariants; checks
needed for correct behavior must survive Python's optimized execution mode.
Catch specific exceptions where recovery or useful translation is possible,
preserve the original cause when translating, and do not hide unexpected failures
behind a success-shaped default.

Use context managers or `try`/`finally` for resources that require cleanup.
Keep ownership and cleanup responsibility clear, especially for processes,
temporary files, and runtime state. Test guidance belongs in
[test governance](test-governance.md).

## Comments, documentation, and output

Write new comments and docstrings in English. Explain intent, constraints,
non-obvious behavior, and public contracts rather than narrating each statement.
Document important inputs, return values, side effects, and errors when they
cannot be understood from the signature. Follow the surrounding docstring
convention; no repository-wide Google or NumPy format is required.

Preserve existing license notices. Author-specific comment prefixes are not a
project convention to propagate. Temporary compatibility behavior must follow
the compatibility registry policy rather than relying on a TODO alone.

Use the owning subsystem's output and logging interfaces. Keep CLI presentation
at the CLI boundary and avoid incidental `print` calls in reusable library code.
Respect structured-output and log contracts; readable status text is not a
substitute for a documented machine-readable event. Update affected user guides
and specs according to the [documentation guide](documentation-guide.md).
