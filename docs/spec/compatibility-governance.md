---
doc_type: spec
status: active
updated_at: 2026-08-31
archived_at:
---

# Compatibility Governance

## Authority and scope

This policy covers temporary compatibility behavior across qqtools. The tracked
`compatibility-registry.toml` is the machine-readable lifecycle authority. Product specifications
remain authoritative for behavior and protected workflows; pitches explain complex delivery work
but do not replace the registry.

Every temporary reader, writer, CLI alias, migration path, protocol adapter, or behavior fixture
must have one registry item. Permanent multi-format support is a product contract and must not be
misrepresented as temporary compatibility.

## Lifecycle

Each item moves through these states:

```text
planned -> compatibility_active -> legacy_removed -> removed from registry
```

- `planned`: the future contract is registered but no compatibility implementation or marker may
  exist in source, tests, or scripts.
- `compatibility_active`: the new contract is the write/default path and the temporary legacy
  boundary remains available.
- `legacy_removed`: the legacy public entry or writer has been removed; only an explicitly planned
  reader, migrator, or transition check may remain.
- Registry removal: temporary runtime support, warnings, markers, and old-behavior fixtures are
  absent, and the target release has reached `transition_purged_in`.

Registry items use exact `X.Y.Z` versions. `introduced_in` must precede `legacy_removed_in`, and
`transition_purged_in` must not precede `legacy_removed_in`. Equal removal and purge versions are
allowed when no separate migration release is needed.

## Registry contract

The top-level `next_id` is the next numeric compatibility ID to allocate. It only increases, even
after older items leave the registry. This prevents ID reuse without retaining completed records.
The registry may contain no `[[items]]` entries when no compatibility work is unfinished.

Every unfinished `[[items]]` entry contains:

- a unique `QQTOOLS-COMPAT-NNNN` ID and identical `marker`;
- `component`, `kind`, and module-level `owner`;
- lifecycle `status` and the three release versions;
- tracked or non-ignored `decision_refs` and `verification` files; the clean committed preflight
  guarantees they are tracked before release;
- optional append-only `extensions` for approved deadline changes.

An extension names `legacy_removed_in` or `transition_purged_in`, chains from the previous effective
value to a later version, records the approval version and reason, and points to a tracked decision.
Original deadlines are not overwritten.

Temporary implementation and behavior fixtures carry their compatibility ID. The checker scans
tracked and non-ignored files under `src/`, `tests/`, and `scripts/`. Planned items must have no
marker; active and legacy-removed items must retain at least one marker.

Completed items are not a registry state and are not retained indefinitely. At release check time,
the checker reads the registry from the exact `v<current source version>` tag. An item may disappear
only when the target reaches its effective `transition_purged_in` and its markers are absent. The
same baseline prevents `next_id` rollback and reuse of retired IDs. If that prior tag predates the
registry, the comparison bootstraps without historical items. Git history, release tags, and the
CHANGELOG are the historical record.

## Commands

```bash
python scripts/checks/check_compatibility_registry.py validate
python scripts/checks/check_compatibility_registry.py plan --release-version X.Y.Z
python scripts/checks/check_compatibility_registry.py check --release-version X.Y.Z
```

`validate` checks registry structure and repository evidence. `plan` reports the state or registry
removal required at a target release. `check` fails when the recorded state does not match that
target, an item was removed early, a retired marker remains, or the ID watermark regresses.

The release operator must inspect `plan`, resolve every due action, commit the candidate, and run:

```bash
python scripts/release_preflight.py --target-version X.Y.Z
```

The local preflight is the only hard compatibility gate. The tag publish workflow trusts it and
does not repeat compatibility validation.

## When a dedicated decision document is required

A normal CLI alias or bounded local reader needs only a registry entry and ordinary feature
documentation. A dedicated pitch or ADR is required when compatibility changes shared persisted
truth, permits different versions to run concurrently, performs an irreversible rewrite, changes
ownership, or requires locks, markers, recovery, or rollback protocols.
