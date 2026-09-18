# Documentation guide

Public documentation must let users and contributors use, operate, verify, and
maintain qqtools without access to ignored planning documents. The root
[AGENTS.md](../../AGENTS.md) defines the execution contract and privacy boundaries.

## Document locations

| Location | Responsibility |
| --- | --- |
| Root `AGENTS.md` | Durable execution constraints, mandatory gates, and scenario navigation |
| `docs/development/` | Engineering workflows, code style, tooling, documentation, testing, compatibility, and repository governance |
| `docs/spec/` | Product behavior, runtime contracts, persisted formats, and the compatibility lifecycle registry |
| `docs/` user guides and root `README.md` | Installation, usage, and user-facing explanations |
| `tests/readme.md` | Test directory navigation and common execution entry points |
| `.dev/work-item.md` on feature branches | Disposable, non-sensitive implementation state, stripped during promotion |
| Ignored private planning and agent configuration | Personal rationale, exploration, templates, and collaboration preferences |

The compatibility policy lives in
[compatibility-governance.md](compatibility-governance.md); its machine-readable
ledger remains [compatibility-registry.toml](../spec/compatibility-registry.toml).
Product compatibility promises belong in the relevant product/runtime specs.
Existing ADRs provide historical rationale; current execution follows the root
contract and current specifications, not superseded workflow descriptions.

## When documentation changes are required

Before implementing a public contract change, establish its scope, invariants,
and acceptance criteria in reviewable, non-sensitive context. Update the public
specification and operating instructions when users or contributors need them
to understand changed CLI/API behavior, configuration, persisted formats,
migration, recovery, or development procedures. Obtain any approval required by
the relevant protected-workflow or governance policy before implementation.

An internal refactor that preserves documented behavior does not automatically
need a new permanent design document. Ordinary fixes may be sufficiently
described by the change and its regression tests. Feature execution state may
live in `.dev/`; private exploration may remain private. Neither may be the only
source of information required for long-term correctness.

When changing packaging, entry points, optional dependencies, examples, or
configuration, inspect the implementation and public instructions together.
Check actual repository paths and behavior before documenting them. Do not copy
directory layouts or experiment templates from unrelated projects.

## Write verifiable contracts

Write documentation in English, including headings, tables, templates, and
example prose. Translate existing non-English content when maintaining the
related documents; preserve technical identifiers and literal product output.

Scale the document to the change. Cover the goal and affected scope, relevant
inputs and outputs, state transitions, failure/boundary behavior, compatibility
impact, and observable acceptance criteria. Clearly distinguish assumptions or
unverified behavior from established facts.

Requirements should say what the system must do and how that can be verified.
Avoid vague promises, speculative scope expansion, and mandatory boilerplate
that does not help assess the actual change. Use examples, tables, or diagrams
when they make a contract easier to inspect; no fixed chapter count or personal
planning template is required.

## Naming, links, and maintenance

Use descriptive `kebab-case` names for new development guides. Follow existing
conventions when maintaining established specs; this is not a requirement to
rename all existing documents. Avoid temporary suffixes such as `final` or `new`.

Use relative Markdown links to public repository files. Keep commands rooted at
the repository root unless explicitly stated otherwise. Link to the authoritative
policy rather than copying its detailed rules into multiple documents.

When moving or renaming a document, update incoming links, outgoing relative
links, and execution entry points that reference it. Check both tracked files
and new public files. Private references should point to the public policy;
public validation must not depend on those private references.

Before delivery, check the changed documentation for path and contract
consistency. Report relevant validation and material gaps. Documentation-only
changes do not need product tests during iteration, but mandatory integration
and release gates still apply.
