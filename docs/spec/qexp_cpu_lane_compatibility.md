---
doc_type: spec
status: drafting
updated_at: 2026-09-05
archived_at:
---

# qexp CPU Lane Compatibility Transition

This is the CPU transition contract. Global qexp schema
remains 6. QQTOOLS-COMPAT-0005 tracks temporary compatibility; the
[registry](compatibility-registry.toml) is authoritative for exact release deadlines.

## Release contract

1.3.15 includes the explicit operator upgrader, so existing projects can activate CPU-only Tasks
in the feature release after a drained, attested transition.

| Release | Legacy read boundary | Writes | Lifecycle obligation |
| --- | --- | --- | --- |
| 1.3.15 | Explicit upgrader reads drained legacy roots; new roots are canonical | Canonical writes after activation | Mark `compatibility_active`; no automatic conversion |
| 1.3.16 | No legacy decoder or upgrader | Canonical records only | Remove registry item, temporary code, markers, warnings and legacy fixtures |

1.3.16 is the planned cleanup release. Delay requires an append-only registry extension backed by
an approved decision, not an overwritten deadline. Until implementation, the item stays `planned`
and its marker must not appear in source, tests or scripts.

Compatibility permits new code to read or upgrade old GPU data within this window. It does not
authorize pre-1.3.15 binaries to use activated roots. Canonical protocol and gate remain permanent
after 1.3.16. Indefinite reading of unconverted legacy roots would be permanent multi-format
support, not this temporary transition.

## Canonical data and upgrade scope

Legacy Task records lack lane and contain positive `requested_gpus`; the temporary reader treats
them as GPU lane without adding a CPU request. Canonical Tasks explicitly store lane and only its
applicable request. Canonical Attempt and claim records identify lane and reservation acquisition
where required. CPU Tasks never use legacy GPU encoding. GPU reservation format stays unchanged.

Conversion covers queued Tasks, terminal Task/Attempt history, archived claims and other durable
embedded copies consumed by ordinary readers, inspect, retry and recovery. Preserve identities,
attempt numbering, fencing, retry/cancellation policy and terminal results. Submission replay,
retry and clone must not restore legacy embedded records. Rebuild derived ready markers, catalogs,
lane-specific primary-demand evidence and cursors before activation.

Inventory all affected readers, writers and embedded copies before implementation. Final audit
must prove every ordinary-reader input canonical; converting only ready Tasks would strand history
when the temporary reader is removed.

## Activation and permanent gate

Use a durable CPU protocol marker with states `legacy`, `preparing`, `canonical`. Missing marker
means legacy during 1.3.15. Fresh roots initialize canonical before accepting work. 1.3.16
rejects existing legacy or preparing roots and directs operators to complete upgrade using 1.3.15;
it must not treat an old root as empty.

Add `required_capabilities: [cpu-lane-v1]` to the schema envelope; keep `version` and
`minimum_reader_version` at 6 and preserve existing `writer_capabilities`. New binaries reject
unknown required capabilities on the process's first open of each root, before normal access. Old binaries reject
the additional envelope field. Appending only to the existing writer-capability list is
insufficient. CLI, agent, Python API and background execution must obtain a validated root context.

### Check frequency and process lifetime

Each process validates each root once on first open and caches the successful result in its root
context. Bind the cache to process identity and resolved root/project identity; do not share success
across roots or inherit it as valid after fork. Concurrent first opens share one initialization;
failed checks never create a usable context. A new agent project binding validates its root before
use. A new CLI process or restarted agent checks again.

Once initialized, scheduler slices, claim and lease heartbeats add no protocol-gate file reads,
filesystem polling or periodic revalidation. Existing authorization, fencing, schema locks and
capacity checks remain in force; this is not permission to remove them or a zero-total-I/O promise.
No configurable check interval, separate governance registry or general-purpose cache is introduced.

This cache requires the root protocol to remain unchanged for the lifetime of every open context.
Before activation, protocol changes or root replacement, stop all processes with that root open,
including readers, Python clients and agents, and block their restart. Only the designated
transition process and transition diagnostics may operate until commit. Restart normal processes
after completion so they validate afresh. Unsupported hot root replacement must not reuse a cached
context. Transition verification and journal reads are outside normal scheduling and are not
limited to a single check. If process quiescence cannot be established, refuse the transition;
elapsed time or a stale heartbeat is not evidence that cached contexts are gone.

Activation follows this sequence:

1. A first normal root open checks protocol state without starting an upgrade. In 1.3.15,
   explicitly read-only inspection may use the temporary legacy reader; normal mutation and
   dispatch require canonical state. In 1.3.16 normal legacy access is rejected. Both releases
   report the explicit upgrade command below. Never silently terminate tasks; existing supervision
   continues until the operator drains the target project.
2. Freeze new target-project admission, submissions and registration changes. Confirm no active
   claim, nonterminal Attempt, unfinished submission/control operation, project reservation or
   live process/launch/termination evidence on all participating machines. Stop all normal
   processes with the root open, including readers and new-version clients, and prevent restart
   until activation completes. Missing/offline evidence blocks
   activation; heartbeat expiry is not proof of drain. Do not remove other projects' reservations;
   account for supervision impact when stopping a shared machine-global agent.
3. Under the schema lock, recheck shared state and the participant set, durably create a transition
   journal, then install the gate before any canonical write. Crashes between gate and marker
   writes resume from the journal. Lock exclusion cannot replace stopping old processes with
   cached authorization or open handles.
4. Set `preparing`; normalize records in bounded resumable batches, preserve recoverable originals
   and journal progress, atomically replace changed records, and rebuild ready state. Block normal
   writers and dispatch throughout preparation; only the upgrader and transition diagnostics may
   access partial data. Initialize missing machine CPU policy to 0, preserving explicit capacity.
5. Audit canonical records and derived state. Require every registered participant to acknowledge
   capability through transition-only registration, bound to this activation and runtime identity;
   commit `canonical` atomically before restarting normal processes and reopening dispatch.
   Resume forward after interruption and retain the gate. Obsolete offline
   registrations require formal retirement after execution is accounted for, not automatic skipping.

This normalizes the affected protocol without cloning/replacing the whole root or invoking a
schema-7 migration. 1.3.16 does not repeat activation or reset capacity on canonical roots.
Rollback of an activated root to pre-1.3.15 binaries is unsupported; old unactivated GPU roots
remain usable by their old binaries.

## Operator entry points (1.3.15)

Provide a feature-specific `qexp upgrade cpu-lane` command group. It is independent of the
existing `qexp migrate --to-schema` interface and the machine CPU-capacity configuration commands.
All upgrade commands require explicit global `--shared-root`; saved CLI context must not silently
choose a mutation target. They support the existing `--format human|json` convention. These are
planned interfaces, not commands currently available in the installed CLI.

```bash
# Read-only inventory and preflight; no marker, lock-file or policy creation.
qexp --shared-root /mnt/share/myproject/.qexp upgrade cpu-lane check

# After draining and stopping normal clients, create the activation session.
qexp --shared-root /mnt/share/myproject/.qexp upgrade cpu-lane start

# On EACH registered machine, use the activation ID returned by start.
qexp --shared-root /mnt/share/myproject/.qexp --machine gpu-a \
  upgrade cpu-lane attest --activation-id <id> --confirm-clients-stopped

# On the coordinator, verify all attestations and perform or resume conversion.
qexp --shared-root /mnt/share/myproject/.qexp upgrade cpu-lane resume --activation-id <id>

# Read-only progress and recovery instructions, including during interruption.
qexp --shared-root /mnt/share/myproject/.qexp upgrade cpu-lane status --format json
```

| Action | Contract |
| --- | --- |
| `check` | Read-only checks of schema, protocol, shared blockers, participant set and locally available evidence. Remote evidence not verified is explicitly unknown; a passing inventory is not authorization to convert. |
| `start` | After shared preflight, create one journal/session under the upgrade coordinator lock with a frozen participant set and activation ID. Session phase is `awaiting_attestations`; do not normalize records yet. It does not remotely stop clients or claim that a legacy binary obeys the session marker. |
| `attest` | Runs locally, resolves this machine's registered runtime, checks reservation/process evidence and scheduler authority exclusion, and publishes a capability/drain acknowledgment bound to activation, project and runtime identity. The explicit confirmation covers operator-controlled external readers/clients and restart suppression that qexp cannot discover reliably. Failed checks publish no positive acknowledgment; the flag never bypasses local blockers. |
| `resume` | Requires the matching session ID, all participant acknowledgments and a fresh shared/local coordinator preflight. Install the permanent gate before canonical writes; advance bounded journaled batches through conversion, ready rebuild, audit and canonical commit. Repeating resumes the same session and never allocates a second migration. |
| `status` | Read-only journal/protocol inspection through the restricted transition reader; works for legacy, preparing, canonical and damaged/incomplete sessions. Report inconsistencies as blockers, not success. |

`start` on an unfinished session returns its ID and the resume/attestation instructions without
overwriting it. `start` on a verified canonical root is an idempotent no-op. Concurrent mutation
commands serialize through one coordinator lock; contention reports `upgrade_busy`. Attestation
writes use the same session coordination boundary and cannot change the frozen participant set.
`resume` after commit is an idempotent success for that session. A missing/mismatched session ID,
changed participant/runtime identity or reopened normal client invalidates the relevant evidence
and blocks continuation; collect fresh attestations before retrying. Operators must repeat
attestation after an interruption before resuming; acknowledgments are not timeless drain proof.

All output identifies project/root, protocol state, activation ID, phase, completed record/batch
counts, total count if known, participant acknowledgment state, blockers and a concrete next
command. Do not invent a percentage before inventory is complete. Stable phases are
`awaiting_attestations`, `normalizing`, `rebuilding_ready`, `auditing`, `completed`; errors retain
the last recoverable phase and record the failure. Read-only `status` exits 0 when the report is
readable even if blocked; `check` and mutations exit nonzero when their prerequisites fail.
Malformed arguments use the existing CLI usage-error convention.

SIGINT or process death preserves journal/progress and keeps dispatch closed. Query `status`,
correct the reported blocker, renew attestations and rerun `resume` with the same ID. No `--force`
or automatic rollback bypasses drain, removes the permanent gate, clears reservations or kills
tasks. The mutating CLI and any Python entry use the same protocol primitives; ordinary first-open
checks never invoke them implicitly.

Operator sequence: stop new submissions and disable target-project admission using existing
project controls, let supervision drain tasks (or explicitly cancel them), stop normal clients
and suppress automatic restart, then run `check`, `start`, local `attest` on every participant,
and `resume`. `status` with phase `completed` is the signal to restart agents/clients and restore
any project binding that the operator disabled. Upgrade does not re-enable bindings, start agents
or set positive CPU capacity; allocate CPU slots separately with `agent cpu-lane set`.

1.3.16 removes this temporary command group with the upgrader. Permanent first-open diagnostics
on legacy/preparing roots direct operators to use 1.3.15 and the same commands to finish conversion;
canonical roots need no upgrade command. Release cleanup includes parser entries and command
fixtures as well as the underlying decoder and journal recovery implementation.

**假设/未验证**：旧版本及已运行进程的门禁覆盖、完整嵌套记录清单、升级入口覆盖和跨机器
排空确认尚未实现或验证；必须通过实现阶段的故障注入验收后才能宣称安全过渡。

## Release verification

The existing schema compatibility and machine-runtime suites named in the registry are planned
verification locations. Extend them during implementation; their existence does not imply current
CPU transition coverage.

- 1.3.15: legacy GPU reads; canonical GPU/CPU writes; missing/nonzero CPU policy; submission replay;
  old CLI/API/agent rejection before writes; offline participant refusal.
- 1.3.16: canonical roots remain unchanged; legacy/preparing roots reject without mutation and
  direct operators to 1.3.15.
- Both transition releases: crashes after journal/gate installation, within batches and before
  canonical commit; forward recovery; no premature dispatch or old writer resurrection.
- Release preflight enforces the exact lifecycle. Retirement obligations for earlier compatibility
  items are independent of this new item's introduction.
- Count gate-specific file reads: concurrent initial opens perform one validation per process/root;
  repeated slices, claim and lease ticks do not increase the count after initialization. Distinguish
  gate reads from existing correctness checks, and verify new bindings, other roots and restarted
  or forked processes validate independently. Incompatible roots yield no usable context.
- Refuse activation while any normal root context remains live; after quiescence and commit,
  restarted processes validate the new protocol. Do not use cache expiry or periodic polling as a
  substitute for the stop constraint.
- Verify `check`/`status` cause no filesystem writes, including on damaged roots; first normal
  opens never start conversion. Verify explicit target selection, start idempotency, competing
  coordinators, wrong activation IDs, missing/changed attestations, interruption/resume and final
  restart instructions. A confirmation flag cannot bypass an observed running process/reservation.

## References

- [ADR-QEXP-0008](../adr/qexp/0008-cpu-lane-schema6-compatibility.md)
- [Compatibility governance](compatibility-governance.md)
- [CPU-only Task Lane pitch](../pitch/arxiv/053-qexp-cpu-only-task-scheduling.md)
