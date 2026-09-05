---
doc_type: spec
status: drafting
updated_at: 2026-09-05
archived_at:
---

# qexp Task Dependencies Compatibility Transition

This is the planned transition contract for Task dependencies in 1.3.15. Global schema
`version` and `minimum_reader_version` remain 6. `QQTOOLS-COMPAT-0007` independently tracks
temporary compatibility; the [registry](compatibility-registry.toml) owns exact deadlines.
The [dependency pitch](../pitch/qexp-task-dependencies.md) defines the proposed feature semantics.

## Release contract

| Release | Legacy boundary | Required lifecycle state |
| --- | --- | --- |
| 1.3.15 | Restricted read-only legacy inspection and explicit, resumable upgrader; canonical writes after activation | compatibility_active after implementation |
| 1.3.16 | No ordinary legacy parsing; only the restricted upgrader reads legacy records | legacy_removed |
| 1.3.17 | No legacy decoder or upgrader; reject unconverted roots and direct operators to 1.3.16 | Remove temporary implementation, markers, fixtures and registry item |

Until implementation the item stays `planned`, with no compatibility marker in source, tests or
scripts. Extending a deadline requires an append-only approved registry extension. Permanent
canonical validation and capability gates survive retirement of temporary compatibility.

## Canonical protocol and activation

Canonical Task truth explicitly stores `depends_on_task_ids: string[]`; empty means no dependencies.
Missing fields mean empty dependencies only inside the temporary legacy boundary. Missing fields
on a canonical root are integrity errors. Normalize all durable copies consumed by submission
replay, retry and recovery as well as queued and terminal Tasks; preserve identities, history,
Attempt numbering and fencing. Inventory embedded copies before implementation.

Require the permanent root capability `task-dependencies-v1`, unioned with existing capabilities
such as `cpu-lane-v1`. Do not change schema numbers or rely on `writer_capabilities` alone. Every
CLI, API, agent and background entry obtains a validated root context; unknown required
capabilities reject access on the process's first open of each root. Include rejection of binaries
that understand CPU lanes but not dependencies.

Reuse the [CPU lane activation constraints](qexp_cpu_lane_compatibility.md): explicit target root,
freeze admission, drain execution and unfinished operations, stop all normal root clients and
suppress restart, collect activation/runtime-bound participant attestations, and recheck under
the schema lock. Missing or offline evidence blocks activation. Expired heartbeats do not prove
drain. First-open cache validity requires no normal root context to survive a protocol change;
do not add periodic gate reads to claim or heartbeat paths.

Track dependency activation durably as `legacy -> preparing -> canonical`. Persist a resumable
journal and install the capability gate before canonical writes. Only the upgrader and restricted
diagnostics may access partial conversion. Normalize in bounded recoverable batches, rebuild
affected derived state, audit, and atomically commit canonical before restarting normal clients.
Interruption retains the gate and progress; renew attestations and resume forward. No automatic
gate removal or rollback to old writers is permitted.

Both CPU lanes and dependencies ship in 1.3.15. Fresh roots initialize all applicable canonical
protocols and required capabilities before admission. Existing roots serialize feature upgrades
through the shared coordination boundary; sessions cannot interleave conversions. CPU canonical
state does not imply dependency canonical state. Opening dependency dispatch requires completion
of all applicable activations. Operator entry points must expose feature/session identity and
partial completion clearly while reusing existing transition primitives.

## Verification obligations

The registry names existing schema and machine-runtime suites as implementation locations, not
evidence that dependency transition tests already exist. Cover legacy no-dependency records,
canonical writes, terminal history, embedded submission replay, old CLI/API/agent rejection,
capability union, CPU-only-capable binary rejection, and coordinated activation of both features.
Inject crashes at journal/gate installation, conversion and canonical commit; verify forward
recovery, offline/live-client refusal and no dispatch during preparation. Verify first-open
context isolation across processes/roots and no periodic gate reads after validation. Check
the release-specific legacy removal and final purge without removing permanent gates.

**假设/未验证**：依赖协议激活、嵌套副本清单、双协议协调与旧二进制拒绝尚未实现或验证。

## References

- [Compatibility governance](compatibility-governance.md)
- [CPU lane rollout decision](../adr/qexp/0009-staged-cpu-lane-rollout.md)
