---
doc_type: pitch
status: drafting
updated_at: 2026-09-09
archived_at:
---

# qexp Agent Lifecycle Independence and Restart Reconciliation

## Background and goal

An operator must be able to stop, restart, or lose the machine agent without terminating
already launched training or losing its eventual correct qexp outcome. This property is called
agent lifecycle independence: execution continuity plus restart reconciliation from durable evidence.
It does not mean the agent is stateless or that scheduling and status remain live while it is absent.

The core workflow is: launch training, stop the agent, let training finish while the agent is
absent, start the agent, and observe correct Task/Attempt terminal metadata without manual repair
or another training launch. An agent outage longer than a claim lease must also be covered.

The [rolling upgrade coordinator](qexp-machine-rolling-upgrade-coordinator.md) depends on this
property and extends it across supported package/protocol versions. This pitch owns the permanent
same-version runtime contract and its regression gate, independently of whether any migration exists.

This is delivery 1 of the three-pitch design. Delivery 2,
[Machine write eligibility and logical-name re-registration](qexp-machine-write-eligibility-and-reregistration.md),
owns clean/stale machine entry and reuse of names such as g3. Delivery 3 is the rolling coordinator.
Project discovery and batch add-project scripts remain user-owned. These later deliveries reuse
this baseline and must not reinterpret agent restart as a new machine registration.

## Existing evidence and verification gap

- `src/qqtools/plugins/qexp/runner.py` launches a private-session guardian, publishes local launch
  and process identities, waits for training, and writes an exit observation independently of agent
  terminal publication. The guardian is tied to runner survival, not a promise to survive runner loss.
- `src/qqtools/plugins/qexp/machine_agent.py` requests graceful stop by signalling the agent PID.
- `docs/spec/qexp_runtime_spec.md` sections 15.4 and 15.6 describe recovery of the same live
  Attempt and recovery-finalize for a finished process. Recovery may issue a new fencing token.
- `tests/integration/qexp/test_runner_pipeline.py` exercises evidence publication and supervisor
  reconciliation, including simulated child processes. These cases alone do not prove the complete
  real-agent stop/offline-completion/start workflow.
- `tests/CONTRACT_MATRIX.md` maps existing authority and crash contracts, but does not explicitly
  name this lifecycle contract. This is a coverage inventory, not a claim that all related tests
  are absent or that the current implementation fails.

## Scope and operating assumptions

In scope: graceful stop/start, restart, agent-only SIGKILL, repeated interruptions, and recovery
across launch/publication boundaries on supported Linux with real runner, guardian, and tmux execution.
The primary guarantee applies to an already launched training process; interruptions during launch
must preserve at-most-one launch and must not invent successful execution.

The host, runner/guardian, training process, runtime evidence, and shared project data remain intact.
The same machine runtime and registered project identity are used on restart. Storage must be
available for durable exit evidence and eventually for reconciliation. There is no required agent
return before lease expiry and no cleanup deadline that discards evidence merely due to agent absence.

Out of scope: host reboot, runner death, deleted runtime/evidence, arbitrary data corruption,
cross-host execution transfer, checkpoint recovery, and arbitrary version upgrades. Explicit task
cancellation, manual retry/supersession, retirement, and durable termination decisions retain their
existing authority semantics and are not undone by restarting an agent.

Agent downtime and protocol-eligibility expiry do not release project-local name ownership.
Automatic restart recovery assumes that registration has not been explicitly adopted elsewhere.
After explicit adoption, the returning environment reports the lost project identity and preserves
evidence under delivery 2's diagnostics contract; unaffected project bindings continue normally.
Replacing that stale binding must not detach old execution evidence or local capacity accounting.
Delivery 2 owns the old-execution completion path after re-registration; a new name alone does not
grant authority over an old Attempt. This boundary does not weaken intact restart recovery here.

For host loss with missing evidence, retain the existing orphan/blocked workflow and explicit
manual retry. No new destruction-confirmation or failure-finalization interface is introduced.
A replacement may reuse the logical machine name through the eligibility pitch without resolving
historical orphans first. Evidence-based terminal reconciliation is not inherently restricted to
the original physical machine; recovering a live process still requires verified process ownership.

## Required behavior

### Execution and authority

- Stopping or crashing only the agent must not signal or tear down training, its runner, guardian,
  process group, or tmux execution container. Already launched work continues while the agent is absent.
- Queue dispatch, renewal, and shared status may pause. A peer may project an expired claim as
  orphaned/blocked, but expiry alone must not authorize duplicate execution or free occupied capacity.
- Agent stop must not itself commit a training termination decision. Existing committed termination
  or supersession prevents unconditional recovery; return must honor those decisions.
- On return, reconcile surviving execution and reservations before making affected capacity
  available for new dispatch. Live-process recovery must not restart the training command.
- Preserve stable project, Task, Attempt, and launch identity. Fencing tokens, lease timestamps,
  agent PID, and recovery timestamps may legitimately change through the existing authority protocol.
  Lifecycle independence requires semantic correctness, not byte-identical metadata.

### Offline completion and convergence

- The runner records success or failure without depending on an agent IPC acknowledgement or a
  live agent. Evidence needed for reconciliation survives until the corresponding durable outcome
  and required accounting effects are safely committed.
- On agent start, automatically reconcile matching exit evidence into the original Attempt and Task,
  including after peer-observed lease expiry. Publish the actual exit code and applicable result,
  retain log references and identity, archive the claim as required, and reconcile reservations.
  Do not require a manual doctor, retry, re-registration, or migration command in the intact case.
- Repeated restart or interruption during terminal publication must converge idempotently: no
  successor launch, contradictory terminal outcome, lost exit evidence, or premature/double release
  of capacity. Existing lifecycle delivery semantics apply; this does not promise exactly-once
  delivery of external notifications or application side effects.
- If authority was explicitly superseded, identity cannot be verified, or required evidence is
  missing, preserve available evidence and report the precise blocker. Do not guess success or
  overwrite a newer Attempt. These are diagnosed boundary failures, not successful automatic recovery.
- Define a numerical reconciliation latency budget for a specified healthy-storage workload before
  acceptance. Status while offline may be stale; after return it must converge within that budget.

## Verification and permanent gate

Protect this as a permanent product/runtime invariant, not a temporary compatibility registry entry.
During implementation, add the workflow to the product Protected workflows and runtime invariants,
and map concrete tests in `tests/CONTRACT_MATRIX.md`. Preserve existing ownership, cancellation,
and orphan-recovery rules; any necessary change to them requires an explicit compatibility decision.

Use a focused Integration file proposed as
`tests/integration/qexp/test_agent_lifecycle_independence.py` for real isolated agents, filesystem
evidence, processes, and peer interleavings. Reuse existing qexp fixtures and machine-lab facilities.
Do not substitute test-written exit observations for the runner in the core offline-completion cases.
Use a short CPU command with a launch counter, progress/exit handshake, and known result; GPU capacity
accounting can use existing isolated fixtures without requiring a long GPU training job.

Add the bounded representative lifecycle cases to the existing `tox run -e preflight` source gate;
keep the complete crash/race matrix in `qexp-integration`, which release preflight runs. Extend the
existing installed-wheel protected workflow E2E with one public CLI stop/offline-completion/start
case on the real Linux/tmux path. This checks packaging and actual lifecycle commands without
duplicating the entire Integration matrix. Do not introduce a new permanent monitoring service.

Required gate environments must fail if the contract cases are absent, unexpectedly skipped, or
their real-process prerequisites are missing. Unsupported local environments may report unverified
results, but cannot count them as passing release evidence. Fault injection must use deterministic
barriers and bounded timeouts, retain failure artifacts, and clean only test-owned resources.

| ID | Scenario | Operation | Required result |
| --- | --- | --- | --- |
| LI-01 | Training remains live | Stop the real agent after a verified launch; observe progress; start it again | Same process identity and Attempt; launch count one; no premature capacity release |
| LI-02 | Offline completion | Parameterize successful and nonzero exit while agent is stopped, then start | Original Task/Attempt receive correct outcome, exit code, logs, claim and reservation accounting automatically |
| LI-03 | Outage beyond lease | Keep a peer active until it observes lease expiry; return the owner with either live training or completed evidence | No peer replacement launch; same Attempt recovers or finalizes with valid fencing |
| LI-04 | Agent-only crash | SIGKILL the agent while training runs; let training exit; restart | Runner/guardian survive agent death and publish evidence; outcome converges without relaunch |
| LI-05 | Launch boundaries | Interrupt before launch, after authorization, and between process creation and registration | No duplicate command; verified launches recover; ambiguous evidence is isolated without fabricated completion |
| LI-06 | Terminal crash boundaries | Interrupt between exit evidence, Attempt/Task publication, and reservation cleanup; restart repeatedly | Durable outcome and accounting converge; evidence is not deleted too early |
| LI-07 | Multiple projects | Stop one global agent with multiple active project bindings and mixed live/finished work | Each identity and reservation is reconciled in its own project; no cross-project release |
| LI-08 | Superseded or incomplete evidence | Return after explicit supersession/termination, or with missing/mismatched evidence | Stale authority cannot overwrite newer truth; precise blocker and safe evidence retention |
| LI-09 | Installed public workflow | Run submit, agent stop, offline completion, agent start through installed CLI | Real Linux/tmux execution survives; public status converges and no manual recovery is needed |

## Cost and non-functional constraints

The intended primary cost is one-time regression infrastructure and ongoing test execution. Add no
background daemon, polling loop, history scan, or per-training heartbeat merely to prove this property.
Reuse runner evidence and ordinary startup/supervision reconciliation. Bound recovery work fairly
across projects and measure restart-to-convergence under a declared count of active/unreconciled
Attempts, rather than requiring a full historical scan.

Retaining unresolved evidence and reservations during an outage consumes disk and can reduce
available scheduling capacity; this is necessary to avoid loss and duplicate execution. Declare
evidence size and cleanup conditions. Do not reclaim based only on elapsed agent downtime.
Distinguish retained evidence from retained GPU occupancy: once local process absence and cleanup
are verified, local capacity may release idempotently even if shared terminal publication is blocked.
Keep the evidence/accounting recovery records needed for later convergence; do not hold otherwise
free capacity merely because shared finalization is pending. Registry/cursor state can be rebuilt,
but whole-runtime loss is a failure with possible evidence loss, not a lossless cache reset. Clarify
the runtime specification's disposable-state terminology during implementation without adding a
backup service or changing the existing orphan/no-automatic-retry outcome.

**Assumption / unverified:** The existing runtime may need correctness fixes to satisfy the full
contract, particularly return after lease expiry and interruption around terminal publication.
Development effort, test duration, recovery latency, and any additional durable writes are not yet
measured. This pitch does not authorize redesigning execution authority merely to simplify a test.
Fix numerical acceptance limits after baseline measurement but before evaluating the changed runtime;
record baseline and candidate separately. Missing limits or an exceeded limit is not passing evidence,
and limits must not be raised after a failure merely to accept the observed result.

## Implementation plan

### Global lifecycle conformance follow-through

Static inspection of `run_machine_agent_loop` on 2026-09-09 shows a main loop continuing until stop
without a true-idle exit branch, while product section 15.3 and runtime section 17 specify on-demand
idle exit. This is a concrete source/spec conformance gap; **Assumption / unverified:** complete
real-process behavior and the intended global multi-binding mode policy still require verification.
Do not treat the current loop as implicit approval to replace on-demand semantics with daemon mode.

Delivery 1 owns resolving this gap before claiming global lifecycle conformance: trace the public
entry paths, reproduce idle behavior with real agents, and record one approved contract, fixing
implementation or explicitly approving a protected-specification change as appropriate. Define
global behavior across multiple bindings rather than assigning permanent control to the first one.
If tracked as separate conformance work, record its concrete owner/dependency and retain this gate;
moving the task does not satisfy it. This follow-through is not authorization to implement a broad
lifecycle redesign during pitch editing.

Delivery 2 owns only the process-local first-registration wait for explicit empty startup. Failed
registration keeps waiting; the first successfully published binding must be validated and consumed
before idle exit becomes possible. After consumption, later registry emptiness does not re-enter
the wait, and the approved global lifecycle contract applies. Its ME-26 does not prove ordinary
on-demand behavior. Derived live status must not create a persisted wait-state protocol or mistake
later empty registry for first registration. Runtime blockers and running execution remain protected.

### LI-03 recovery boundaries

Recovery of an existing Attempt is separate from admission of new work. An expired lease never
revives its old fencing token; recovery commits a higher fencing token by CAS while preserving the
Attempt ID, launch count, and reservation identity. An ineligible or superseded registration cannot
claim new work or renew ordinary authority, but may perform limited recovery only when local process
evidence, Attempt identity, Group state, protocol barriers, and reservation ownership validate.

Draining workers may finish or recover existing Attempts but cannot admit new Tasks. Removing or
disabled workers cannot resume execution; they may publish verified terminal results. Missing or
mismatched evidence remains orphaned/blocked. Released reservations are not automatically revived;
they require explicit repair. Pending recovery or repair prevents true-idle reporting. Rejected
recovery records a durable reason code and never bypasses upgrade writer exclusion or protocol floors.

### Phase 1: establish the contract and baseline

- [ ] Reproduce the global true-idle conformance gap and record the approved lifecycle decision,
      including multi-binding and post-first-registration behavior, before claiming conformance.
- [ ] Inventory existing lifecycle/recovery tests against LI-01 through LI-09 and reuse valid coverage.
- [x] Add tracked product/runtime contract text and contract-matrix mapping with honest pending status.
- [ ] Reproduce LI-01 through LI-04 with real processes and record baseline, state transitions,
      evidence ownership, numerical convergence budget, and gate runtime budget.

### Phase 2: close verified gaps

- [x] Add deterministic interruption coverage and multi-project/boundary cases in the focused
      Integration suite; fix only demonstrated lifecycle/evidence/recovery defects.
- [ ] Verify runner survival separately from runner-death containment; retain existing guardian safety.
- [ ] Record any required persisted-protocol or authority change before implementing it, applying
      existing compatibility governance and the upgrade pitch where relevant.

### Phase 3: enforce and document

- [x] Wire representative tests into existing source preflight and full cases into release verification.
- [x] Add the installed CLI lifecycle case and verify required gate collection/skip handling.
- [x] Update the operator explanation of offline status, restart recovery, and evidence-loss boundaries.

## Acceptance checklist

- [ ] Global true-idle behavior has a single approved product/runtime contract and real-process
      evidence; first-registration tests are not substituted for this separate conformance gate.
- [ ] LI-01 through LI-09 have passing real-boundary evidence on supported Linux.
      Current rerun still has an LI-03 convergence failure; retain this gate until the
      live and completed lease-expiry cases pass within the 15-second budget.
- [x] Stop/offline completion/start needs no manual repair in the intact, unsuperseded case.
- [x] Same-version lifecycle correctness is gated independently of migration implementation.
- [ ] Test runtime, evidence retention, and convergence budgets are recorded and satisfied.
- [ ] Tracked specifications and test mappings describe verified behavior; no pitch-only gate claim.
