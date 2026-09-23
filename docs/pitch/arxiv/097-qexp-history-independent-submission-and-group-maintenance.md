---
doc_type: pitch
status: archived
updated_at: 2026-09-23
archived_at: 2026-09-23
---

# Long-lived machine agents with history-independent working sets

## Archived outcome

Phase A was implemented and promoted to `dev` as
`38de2bbb9ca3b4feaac1cd318ee93fbda9dcdd01`. Dependency validation now reads only
the candidate-reachable dependency closure and has regression coverage for
unrelated malformed history, deep graphs, cycles, and record identity.

Phase B1 produced the reviewed
[Group service quiescence protocol](../../spec/qexp_group_service_quiescence.md).
Its protocol and compatibility decision were approved on 2026-09-23. Phase B2
now routes active discovery through three generation-fenced locator lanes owned by
the machine upgrade coordinator, with bounded residency, per-Project fairness,
durable maintenance checkpoints and explicit degraded repair. Phase C adds the
0/1,000/10,000/100,000 Group fixtures, a 100,000-lifecycle stress case and the
retained 24-hour soak command. The accelerated scale and lifecycle cases passed;
the full 24-hour wall-clock soak remains separate retained release evidence.
The proposal below is retained as historical planning context; current public
specifications describe shipped behavior.

## Decision

A machine agent should remain healthy through months of submission, execution, completion, and
retry. Its routine working set must follow unfinished obligations, not every Group or Task ever
observed. Users commonly leave completed Groups open; performance must not depend on remembering
to seal, pause, delete history, or periodically restart the agent.

Automatically enroll new obligations, service them with bounded resources, and retire their runtime
state after durable completion or handoff. Historical truth remains available on demand. Group
quiescence describes the absence of routine service obligations, not a new user-visible admission
state, automatic sealing, or deletion. Prefer completing existing debt/responsibility mechanisms
over introducing a separate persistent Group `inactive` flag or a new general-purpose queue system.

Deliver targeted submission validation first, obligation-driven Group service second, and long-run
resource/retention qualification third. All three are required for the overall objective; background
retirement is not an optional optimization dependent on a user first proving a slowdown. Measurement
determines budgets and implementation choices. This proposal does not authorize implementation or
claim an already measured production latency regression.

## Relationship to prior pitches

| Document | Existing scope | Relationship to this proposal |
| --- | --- | --- |
| [Active/history separation](070-qexp-active-history-separation.md) | Completed startup, dispatch, discovery, control-operation and indexed-history work | Preserve its delivered contracts; this proposal addresses a remaining submission scan and recurring idle-Group cost |
| [Group ready membership](055-qexp-group-ready-membership-projection.md) | Replaced historical Task scans for primary/borrow ready-member updates | Same scaling principle, different lookup; ready-only membership cannot answer dependencies on completed Tasks |
| [Task dependencies](058-qexp-task-dependencies.md) | Dependency semantics and validation | Preserve identity, same-Group, commitment, cleanup and cycle safety |
| [Frozen success completion actions](../frozen/qexp-success-completion-actions.md) | Proposed commands after successful Task or sealed-Group completion | Not an idle-Group retirement mechanism; do not use its seal requirement as a performance prerequisite |

No matching automatic Group quiescence proposal was found in the current `docs/pitch/frozen/`
inventory. Do not reopen or rewrite completed pitches or thaw completion actions as part of this
work. Maintained public specs, not these private references, remain authoritative.

## Source evidence and impact

The inspected implementation has two distinct cost sources:

1. `runtime/dependencies.py::_group_tasks` enumerates the Project's Task directory, reads and parses
   each Task, then filters by Group. `validate_group_dependencies` invokes it for grouped candidates
   even when they declare no dependencies. The normal successful submission path calls validation
   before staging and again inside `stage_and_commit` in `runtime/submission.py`.
2. `runtime/group_discovery/service.py::_next_group` sweeps the Group directory and admits published
   Groups without filtering by admission state. Retained entries own discovery and maintenance
   sessions. Per-step budgets bound individual visits, not total cost across all historical Groups.

For the first path, two validations can read approximately twice the retained Project Task count,
in addition to work for the actual submission. The growth is with retained Tasks, not simply Group
count. Sealing old Groups does not eliminate this scan. Even unrelated malformed historical Task
records can currently affect this lookup because parsing precedes Group filtering.

For the second path, recurring enumeration and per-Group service state are potential I/O, memory,
and progress-latency costs. Background service is separate from scheduling and renewal, but shares
machine and filesystem resources. Its contribution to submit latency has not been measured.
Do not claim global lock serialization or a specific slowdown without profiling.

## Required behavior

- Keep `open/sealed` admission and `active/paused` dispatch semantics unchanged. An open Group with
  terminal Tasks may receive new work without an explicit reopen or resume operation.
- For a fixed submission and relevant dependency graph, unrelated history must not increase
  dependency-validation directory enumeration or authoritative record reads.
- Never replace a full scan with an unchecked cache, incomplete membership view, or silent
  assumption that missing dependency truth is success.
- Idle maintenance must not require user action; retained Group names and historical truth remain
  available. Failed or cancelled terminal Tasks may be quiescent, but unresolved safety or control
  obligations must not disappear.
- Preserve transaction visibility, revalidation, lock ordering, retry/idempotency behavior, and
  crash recovery. Do not reduce durability to obtain a performance improvement.
- At fixed enabled Project bindings, active load, relevant dependency graph, and maintenance backlog,
  increasing unrelated history must not cause unbounded routine service work, live objects, threads,
  descriptors, or queues. Retained disk history is a separate capacity concern.

## Runtime ownership model

Reuse ready indexes, local Attempt responsibilities, active operations, and discovery/cleanup debt
as the entry points for unfinished work. These indexes locate work; authoritative truth and existing
fences still authorize mutation and execution. This is a common lifecycle contract for existing
services, not a requirement to consolidate them into one storage format.

| Obligation | Routine service ownership ends when |
| --- | --- |
| Ready or delayed Task | Claimed, cancelled, or durably handed to its appropriate waiting/due-work mechanism |
| Local Attempt | Terminal/accounting and required recovery-evidence handoff complete |
| Submission or control operation | Completion is durable and active responsibility is retired |
| Membership projection or metadata maintenance | Certified progress is current and remaining debt is complete or durably handed off |

A Group is the scope of these obligations, not an unconditional permanent service allocation.
Retire each service independently: a running Attempt need not keep an otherwise idle membership
discovery session open. Conversely, all Tasks being terminal does not erase pending cancellation,
publication, recovery, or cleanup work. Uncertain ownership retains a discoverable obligation, not
an indefinitely pinned heavyweight session. Eviction of a cache entry is safe only when unfinished
progress remains durably recoverable.

## Phase A: targeted dependency validation

Use the frozen candidate set plus exact Task-ID reads for required existing dependency truth.
Overlay candidates onto existing truth when evaluating the proposed graph. Traverse the reachable
dependency closure needed to establish cycle safety, including paths back into candidates; do not
assume one-hop reads prove acyclicity. Retain same-Group, committed-Submission, cleanup, missing
record, and self-dependency checks at their existing transaction boundaries.

The target work is proportional to candidate count plus the vertices and edges of the relevant
dependency closure, not all Project history. A genuinely large relevant graph may still be costly;
do not promise constant-time dependency validation. Prefer exact reads before introducing another
persisted membership index. Existing live-ready membership excludes terminal Tasks and is not a
complete dependency oracle.

Before implementing a dependency-free fast path, establish which existing invariant the complete
Group cycle check owns, including existing candidate IDs, retries, dependency editing, and corrupt
reachable state. Document the intended boundary: ordinary submission validates affected truth;
unrelated historical corruption belongs to explicit audit/repair. If this narrows protected
behavior, obtain the compatibility decision before implementing it.

Retain both preflight validation and locked publication-time revalidation unless a separate proof
shows a check is redundant. Audit concurrent dependency edits and cleanup under the actual lock
protocol; no new time-of-check/time-of-use gap is acceptable.

## Phase B: obligation-driven Group service and reliable wake-up

### B1: protocol design and review gate

Phase B is not implementation-ready. Deliver a reviewable protocol design before authorizing B2;
the following decisions must not be delegated to incidental coding choices:

- Exact persisted records, identity/revision fields, authority versus advisory ownership, and
  per-service retirement predicates; identify reused mechanisms and justify each new record.
- Complete producer/consumer inventory, including retries, due work, control changes, recovery,
  repair, and supported released writers; distinguish shared acknowledgements from local ones.
- Lock acquisition order and publication/acknowledgement state machines, with durable ordering,
  concurrent-generation cases, crash windows, replay, and discoverability after agent restart.
- Mixed-version qualification and activation/fencing conditions, exceptional bootstrap/rebuild,
  degraded-index behavior, and any compatibility decisions requiring explicit approval.
- Numerical session/I/O/queue budgets and fairness rules, including due-work progress, bounded
  retries, and how idle proofs survive cache eviction without historical Group scans.

Review must establish no lost wake-up, no cross-machine acknowledgement of local responsibility,
and no routine historical-scan dependency. Keep the design available in non-private delivery context
and incorporate accepted durable contracts into public specs. A checklist of desired properties
alone does not pass this gate. Phase A may proceed independently under its own implementation gates.

### B2: implementation after protocol approval

First measure cost as completed Group count grows, separately from Task count. Inventory the
existing discovery debt, change, coverage, and maintenance mechanisms before adding state.

For each service, retirement requires certified progress at its owning revision and no remaining
obligation for that service, or a durable handoff to another owner. An elapsed idle timeout, an empty
ready index, or `sealed` alone is insufficient proof. Maintenance debt has an independently scheduled
lane; it must not keep every discovery session resident. When no service needs a Group, release its
cached objects, parsers, buffers, and file handles, and remove it from routine round-robin visits.

New submissions, retries, dependency/control changes, and newly due obligations must make work
discoverable without waiting for a complete historical Group sweep. Reuse existing durable debt
publication where possible. Define a revision-fenced handoff so publication racing with quiescence
cannot lose a wake-up, and make process restart reconstruct unfinished obligations.

The protocol review must establish these properties at every producing transition:

- No effective work may lack a durable discovery owner. Publish intent before the relevant commit
  or prove that an existing discoverable operation owns the commit-to-notification crash window.
  A memory callback or filesystem notification alone is insufficient.
- A consumer acknowledges only the identity/generation/revision it actually processed. Completion
  racing with a new publication cannot delete the newer obligation. Coalescing notifications is
  allowed only when the pending revision and all required effects remain recoverable.
- Duplicate delivery is safe through authoritative revalidation and idempotent progress. Restart
  resumes pending obligations without replaying all completed Group payloads. Failed publication
  follows the owning transaction's recovery rules; it is not silently treated as successful wake-up.
- Distinguish shared projection completion from machine-local execution/recovery responsibilities.
  One machine completing shared work cannot acknowledge another machine's local obligations.
  Define consumer eligibility and acknowledgement ownership before choosing queue layout.
- Delayed work has a durable due-time discovery path. Blocked work retains reason, retry policy,
  and an operator-visible locator without busy polling or silently becoming completed.

Do not merely stop visiting known idle entries while continuing to reopen all Group records every
refresh. The design must address both recurring enumeration and retained per-Group memory. Any
residual historical audit must have its own budget and must not be the correctness mechanism for
ordinary wake-up. Recovery may use resumable bounded bootstrap, never a hidden submit-time scan.

Implement only the reviewed B1 protocol. If supported writers cannot publish wake-up debt, do not
enable quiescence silently; satisfy the approved activation/fencing decision first. Protocol changes
discovered during implementation must return to review rather than silently widening assumptions.

### Agent residency is separate from service retirement

Releasing a Group service object does not retire its durable obligations and does not authorize
agent process exit. Preserve the existing machine-global `daemon` and `on_demand` policies and
true-idleness proof in the [runtime lifecycle contract](../../spec/qexp_runtime_spec.md#171-on-demand-mode).
Here, "wake-up" means re-enrolling service work in a running agent, or discovering it on a normal
agent restart; it does not introduce an external process launcher.

| Work classification | Service/cache behavior | Agent exit consequence |
| --- | --- | --- |
| Retained Group history with no service obligation | Retire routine sessions and release caches | History alone does not prevent on-demand exit; daemon remains running |
| Eligible queued work or capacity-deferred work | Keep durable demand; load bounded batches | Pending demand cannot become idle merely because no claim was made |
| Delayed or dependency-waiting work | Retain due-time/change discovery; evict only reconstructible caches | Preserve existing demand classification; unresolved work cannot be reclassified as idle by this optimization |
| Blocked execution safety, recovery, or failed maintenance affecting idle proof | Retain discoverable responsibility and bounded retry/diagnostics | Backoff or session eviction does not establish true idleness |
| Running processes, provisional reservations, termination or reconciliation | Continue the owning service or a durable safe handoff | Preserve existing idle blockers |
| All existing idle predicates proven | No remaining mandatory work | On-demand may exit only after the existing full-loop idle interval; daemon does not auto-exit |

Do not make all optional background work a new process-liveness blocker. Existing idle-neutral
Group discovery remains idle-neutral unless its work owns a responsibility required by the current
idle contract. B1 must map concrete waiting/blocked categories to those existing predicates; it must
not use a blanket "not runnable now" rule. Unresolved demand can legitimately retain an on-demand
process while using bounded memory and low-rate checks. Changing that behavior is a separate
compatibility decision, not a prerequisite for history-independent service.

There is no new "exit now, automatically start at the deadline" guarantee. Without an independently
qualified external launcher, a durable timestamp cannot restart a stopped process. Preserve existing
local-submit activation, `--no-activate`, and explicit start behavior; do not assume retry or a remote
write starts an absent local agent. Due work must progress while the agent is resident and be
rediscovered after an explicit restart. Automatic post-exit timer activation is out of scope.

## Phase C: bounded resources, reclamation, and long-run health

### Service budgets and failure isolation

Define explicit numerical limits before implementation for resident Group sessions, parser buffers,
open descriptors, storage workers, prefetched work, and queued requests. Overflow stays durably
pending and is admitted fairly; never drop obligations or allocate one thread per Group. Fixed
Project and worker overhead must be accounted for separately from per-obligation cost.

Reserve service capacity for local supervision, renewal, and terminal/resource reconciliation.
New execution admission and background discovery cannot consume those reservations. Maintenance also receives
a nonzero guaranteed share so temporary evidence cannot accumulate indefinitely during busy periods.
Rotate Projects and obligations fairly; repeated failures use capped backoff and bounded diagnostic
output, without starving healthy work or allocating unbounded retry entries in memory.

Track pending count, oldest pending age, retry state, and completion rate using bounded maintained
metadata rather than scanning history to produce metrics. Persistent backlog growth or disk pressure
must become visible. Backpressure in this proposal applies to new execution claims and background
work loading, not acceptance of `qexp submit`. Define execution/resource thresholds and recovery
behavior explicitly; defer new claims before acquiring reservations or authorizing launch, rather
than claiming work and parking it indefinitely. Existing authorized work keeps its lifecycle rules.

Submissions that can be durably persisted retain the existing Operation transitions and CLI result;
committed Tasks remain in the existing queue while execution capacity is unavailable. Introduce no
capacity-specific Submission state, new rejection result, or CLI wait for execution capacity.
Same-key replay and interrupted-submission recovery retain their existing semantics. Submission
storage failure still follows the existing failure/uncertain-commit protocol; it is never reported
as successfully queued merely because an in-memory request exists.

Fair execution must resume when capacity is available. Any incompatible eligibility or lifecycle
change still requires approval. Submission quotas or storage-capacity rejection policies would
require a separate public contract and compatibility decision and are not part of this feature.

Blocking storage calls are not cancelled merely because a soft deadline expires. Inventory shared
filesystem calls on critical threads, bound outstanding I/O, and define which supervision can
continue during a stalled Project. Do not abandon ownership, release resources, or launch replacement
work to hide storage failure. State the supported filesystem and outage envelope; arbitrary storage
hangs cannot carry an unconditional wall-clock guarantee.

### Runtime garbage versus user history

Automatically reclaim only implementation-owned scratch, obsolete derived generations, completed
temporary receipts, and cache state after their existing proof and writer-quiescence requirements
are satisfied. Durable cleanup debt survives interruption until deletion and directory durability
are confirmed. Do not introduce broad recursive cleanup of Project or runtime roots.

Task/Attempt history, user logs, and experiment artifacts remain subject to explicit retention
policy, not automatic Group retirement. This feature does not authorize their deletion. Unlimited
history cannot fit finite disk forever; document capacity requirements, configurable operator-owned
retention where supported, and early disk-pressure diagnostics. Runtime-owned rotating diagnostics
and temporary storage need explicit byte/file limits. Storage exhaustion must not discard the last
execution or recovery evidence.

### Long-run acceptance contract

At matched active load after warm-up, live object/descriptor/queue counts must remain within their
declared caps across repeated create/run/finish cycles. RSS need not return exactly to its initial
value, but must remain within an agreed envelope rather than trend with cumulative Group count.
With no new debt, healthy storage, eligible Project bindings, and a continuously running agent
providing the guaranteed maintenance share, reclaimable runtime debris and maintenance backlog
must converge to the documented steady baseline. Protected user history is measured separately.
Input stopping and storage recovering alone are insufficient if no agent remains to service debt.

On-demand exit preserves unfinished optional maintenance debt and durable progress without making
that debt a new idle blocker. While the agent is stopped, no cleanup progress or wall-clock drain
deadline is promised. A later start resumes the debt; complete drain additionally requires enough
eligible service time. Repeated short runs must not lose or repeatedly reset acknowledged progress,
but a restart alone is not a guarantee that all debt clears before the next exit.

B1 must classify mandatory idle-blocking responsibilities separately from optional maintenance,
using the existing residency contract. Mandatory responsibilities continue to prevent premature
exit. Do not introduce a timer launcher or expand idle blockers merely to satisfy a cleanup target.

Specify the workload rate and supported capacity envelope. No design can bound durable backlog if
new obligations arrive permanently faster than they can be serviced. Qualification must include
overload detection/backpressure and successful drain, not just low-rate idle operation.

## Acceptance evidence

| Area | Required evidence |
| --- | --- |
| Unrelated history | Fixed new/existing-Group submission with 0, 1k, 10k, and 100k unrelated terminal Tasks; dependency-validation reads do not grow with history and no Task-directory enumeration occurs |
| Group count | Vary completed Group count independently of Task count; include open, sealed, and paused Groups; record background reads, resident entries/memory, and wake-up progress |
| Dependency safety | Dependency-free candidates, same-batch edges, terminal prerequisites, cross-Group/missing/uncommitted/cleanup references, self-cycle, multi-hop cycle, and a large relevant closure |
| Concurrency | Submit versus dependency edits, cleanup, retries, and same-key replay; rejection before publication and existing atomic visibility remain intact |
| Service retirement | All-terminal open Groups leave routine service; an idle discovery session can retire while an Attempt remains correctly supervised; unfinished debt survives session eviction |
| Wake-up | Append and retry after quiescence, race at every handoff, crash/restart, corrupt advisory state, and supported mixed writers; no lost or duplicate authorized execution |
| Multi-machine ownership | Shared-work acknowledgement cannot retire another machine's local recovery obligation; offline consumer and concurrent producer cases retain required discovery |
| Submission under execution pressure | Durable submission result, Operation state, same-key replay, activation, and crash recovery match existing behavior; no new claim is parked waiting for service capacity and queued work resumes fairly |
| On-demand residency | Pure history does not retain the agent; waiting/delayed demand, unresolved safety, and failed idle-proof maintenance retain existing semantics even after cache eviction; true idle lasts a full loop before exit |
| Due work and process restart | Large waiting/blocked sets use capped memory and fair bounded service; due work progresses in a resident agent; crash/restart restores pending work; idle exit followed by ordinary local-submit activation retains existing behavior without a new timer launcher |
| Long-run churn | Repeatedly create and finish Groups with fixed live load while retaining history; session, buffer, descriptor, thread and queue counts stay capped, RSS stays within the declared envelope |
| Maintenance and overload | Continuous submissions do not starve cleanup; excess input produces visible backlog/backpressure; with no new debt, healthy storage, eligible bindings, and continuous agent service, eligible debris and debt drain without deleting user history |
| Optional maintenance across exit | On-demand may exit with optional debt; debt and acknowledged progress survive exit/crash and resume on restart without becoming new idle blockers; sufficient eligible service time drains the backlog, while short runs do not promise full drain |
| Failure isolation | Storage stalls, disk pressure, malformed records, repeated failed cleanup, and agent restart preserve evidence and allow unaffected work to progress within the declared isolation envelope |
| Measurement | Count reads, bytes, enumeration, lock hold/wait, and background work separately; latency is supporting evidence with filesystem and cache conditions recorded |

Use isolated temporary Projects, not customer data or real training. Establish the current baseline
before implementation. Add deterministic operation-count regressions to the normal focused suite;
run larger scale qualification separately through standard tooling. No fixed millisecond claim is
made until the environment and budget are defined.

Before qualification, freeze numerical resource caps, latency/oldest-work-age budgets, overload
thresholds, churn count, soak duration, and permitted RSS drift. Use a proposed baseline of at least
100,000 accelerated obligation lifecycles plus a 24-hour real-process soak with periodic new Groups;
these complement, not replace, each other. Record cache/filesystem conditions and include restart
and injected-failure phases. Such evidence tests convergence and does not certify months of uptime
or physical power-loss safety. Keep small structural regressions in ordinary gates and retain larger
qualification commands/results as reproducible release evidence under test governance.

## Delivery checklist and boundaries

- [x] Confirm Phase A call graph and relevant dependency mutation/locking invariants; capture baseline.
- [x] Implement targeted validation with focused safety and history-independence regression tests.
- [x] Deliver and review B1 records, producer/consumer inventory, locks, state machines, crash matrix,
  residency classification, budgets, and mixed-version activation; obtain required compatibility approvals.
- [x] Only after B1 approval, implement B2 service retirement/wake-up and validate every handoff.
- [ ] Establish Phase C resource budgets, fair service, failure isolation, safe reclamation, and
  pressure diagnostics; run accelerated churn and real-time soak qualification.
- [x] Align public runtime/product specs, contract matrix, and active/history acceptance evidence with
  delivered scope. Register temporary compatibility behavior only if introduced.
- [x] Run appropriate focused checks and required promotion gates; report all three phases
  independently. Phase A may ship first, but the long-lived-agent objective is not complete until
  obligation-driven service and long-run resource convergence are qualified.

Automatic seal/pause, automatic post-exit timer activation, new Submission capacity quotas/states,
archival or deletion of authoritative history, completion actions, general logging, and the separate
submission-failure diagnostic feature are out of scope. User-history
cleanup remains an explicit retention operation, not a prerequisite for acceptable daily submission
performance; safe runtime-owned garbage reclamation is part of Phase C.
