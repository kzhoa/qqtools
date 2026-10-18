---
doc_type: pitch
status: archived
updated_at: 2026-10-18
archived_at: 2026-10-18
---

# qexp Project-Wide Task Dependencies

## Decision

Extend Task dependencies from a same-Group graph to a same-Project graph. A
Task may name direct prerequisite Tasks from another Group or with no Group.
The downstream Task may likewise be grouped or ungrouped. Cross-Project
dependencies remain unsupported.

Keep the existing explicit `all-of` dependency list. Users must name every
independent direct prerequisite. They need not repeat transitive ancestors,
but may name an ancestor when it is intentionally also a direct prerequisite.
Do not add implicit dependency inference or a live "depend on this Group"
relationship in this delivery.

For example:

```text
A ----\
       > C    C directly depends on both independent Tasks: [A, B]
B ----/

A -> B -> C   C directly depends only on B; A is already a transitive ancestor
```

This pitch proposes a public behavior change and implementation boundaries. It
does not itself change the CLI or runtime.

## Why change the Group restriction

The current implementation stores Project-wide stable Task IDs but accepts a
dependency edge only when both Tasks belong to the same Group. An ungrouped
Task cannot declare any dependency. This makes Group serve two unrelated
purposes: an admission and management boundary, and a dependency namespace.

That coupling is costly for ordinary experiment structure. A data preparation
or baseline Task may be shared by several experiment Groups; evaluation may be
managed separately from training; and a later control Group may consume a
result from an earlier sealed Group. Today these workflows must either combine
unrelated work into one Group or move dependency correctness outside qexp.

The Product Spec already defines Group as the primary observation and
management boundary and Task identity as stable. Dependency edges should
follow Task identity. Group admission, dispatch, Worker Set, cancellation, and
placement remain scoped to the Task's own Group; a dependency does not merge or
inherit those policies.

## Current behavior and affected contracts

Canonical Task truth stores sorted, unique `depends_on_task_ids`. Submission,
dependency editing, cycle validation, claim locking, the dependency gate, and
cleanup reference protection all assume same-Group edges. Ungrouped Task
records currently reject nonempty dependency lists. The CLI already accepts
repeated `--depends-on TASK_ID`, and manifests already accept a list. No new
dependency identifier or command is needed.

The archived [Task dependency pitch](058-qexp-task-dependencies.md)
records the original same-Group decision. Current user-visible behavior belongs
in the [Product Spec](../../spec/qexp_product_spec.md), while locking, persisted
projections, recovery, and cleanup protocols belong in the
[Runtime Spec](../../spec/qexp_runtime_spec.md). Implementation must update both
specifications rather than treating this pitch as normative runtime truth.

## Proposed product contract

### Project-wide direct edges

`depends_on_task_ids` continues to contain exact Task IDs. Each referenced Task
must be in the same Project and either belong to a committed Submission or be a
candidate in the same atomic Submission. A Task may reference a Task in any
Group or an ungrouped Task. Project identity remains an external selection
boundary, so an ID from another Project is missing rather than remotely
resolved.

Task IDs remain the only persisted dependency identity. Group names are loaded
from current Task truth for locking and diagnostics; they are not copied into
the edge, and Group display names or Task names never participate in identity.

Dependency semantics remain:

- every direct prerequisite must be `succeeded` before the downstream Task can
  be claimed;
- a nonterminal or recoverably blocked prerequisite produces `waiting`;
- a failed or cancelled prerequisite produces `blocked`;
- missing, malformed, cleaning, or otherwise inconsistent prerequisite truth
  produces `invalid`;
- only `ready` permits claim, reservation, or Attempt creation.

The graph is transitively enforced. If B could succeed only after A succeeded,
C depending on B is sufficient to order C after both B and A. Listing A on C
is valid when A is intentionally also a direct prerequisite, but is not
required merely to preserve execution order.

### Group lifecycle isolation

A dependency edge grants no control over the prerequisite's Group. It does not
add workers, change placement, reopen or resume a Group, retry a failed Task, or
propagate cancellation. Specifically:

- sealing a Group prevents new members but does not invalidate its existing
  Tasks as prerequisites;
- pausing an upstream Group may leave downstream Tasks waiting indefinitely;
- cancelling or failing an upstream Task blocks downstream Tasks but does not
  automatically cancel them;
- Group cancellation targets only the Group's own membership snapshot;
- Task cleanup is rejected while any Task anywhere in the Project directly
  references the target.

Observation must show the prerequisite Task ID and reason regardless of Group.
Human detail may also show its Group for orientation, but persisted identity and
JSON correctness must not depend on a mutable display label or Task name.

### CLI and manifest behavior

Keep the existing interface:

```bash
qexp submit --group eval --depends-on TRAIN_TASK_ID -- python evaluate.py

qexp task dependencies add EVAL_TASK_ID --depends-on PREP_TASK_ID
```

Repeated `--depends-on` values and manifest `depends_on_task_ids` retain
explicit `all-of` semantics. Submission preview, idempotency digests, persisted
resolved context, dependency editing, list/show/wait output, and replay must all
preserve the exact normalized direct-edge set.

Errors should identify the downstream Task, prerequisite Task, and their Groups
when available. The old "not in Group" rejection is removed; cross-Project or
missing IDs, cycles, cleanup conflicts, and edit-ineligible Tasks remain errors.

This expansion does not change dependency-edit eligibility: only a committed,
nonterminal Task that has never created an Attempt, has no active claim, and is
not entering cleanup may have its dependency set edited. Claim, retry, cancel,
and terminal-transition semantics otherwise remain unchanged.

## Why not infer or abbreviate multiple dependencies now

Independent prerequisites cannot be inferred safely from submission order,
Task names, Group membership, or timestamps. Such inference would make adding a
Task or reopening a Group silently change another Task's execution condition.
The explicit list is auditable, idempotent, and already supports normal fan-in.

A live Group dependency is especially ambiguous: it is unclear whether it
means current members, members added before sealing, or all future members; and
whether failed, cancelled, retried, or cleaned members remain in the condition.
It also turns Group reopening into a retroactive dependency mutation.

If real manifests show that large independent fan-in is a recurring usability
problem, consider a later convenience selector with this narrower contract:

- accept only a sealed source Group;
- snapshot its committed Task IDs during submission preparation;
- expand and persist those IDs in `depends_on_task_ids` before the Submission
  Operation commits;
- freeze the expansion in preview, idempotency, and replay;
- create no durable Group-to-Task dependency type.

That selector is deferred. It does not reduce the runtime edge set and should
be added only with evidence that explicit independent fan-in is materially
burdensome. Documentation should immediately clarify that transitive ancestors
do not need to be repeated.

## Runtime and consistency requirements

Cross-Group support must not be implemented by deleting the current Group
equality checks alone. The implementation needs one Project-scoped dependency
mutation fence. In the existing global order it follows the schema writer fence
and, for submission, the idempotency-key lock; it precedes every affected Group
and Task lock. Dependency submission and editing, dependency-projection
reconciliation, and cleanup-start validation use that fence so concurrent edge
creation, cycle creation, and cleanup cannot commit contradictory results.

An operation that takes the dependency fence then acquires all affected
non-null Group locks in stable Group-name order, followed by all affected Task
locks in stable Task-ID order. An ungrouped Task contributes no Group lock.
Single-Group operations must remain compatible with this order, and no path may
acquire the dependency fence or a Group lock while holding a Task lock. Claim
need not serialize unrelated graph mutations globally: it acquires the affected
Group and Task locks in the same order, then re-reads the downstream Task and
every direct prerequisite and verifies the dependency identity before creating
a reservation or Attempt. It must not hold those shared locks across process
launch or network I/O.

Cycle validation follows the complete Project-local graph reachable from the
candidate edges by exact Task ID. It must include paths that cross Groups and
paths through ungrouped Tasks, without scanning unrelated Task history. Two
concurrent edits that would form a cycle must serialize so at most one commits.

Downstream Task truth remains the sole authority for an edge. A rebuildable
reverse dependency projection is required for bounded Project-wide cleanup
discovery, but it must not independently create or delete edges. Projection
updates and their recovery state are serialized by the dependency fence.
Cleanup validates every projected dependent against live downstream Task truth
and must fail closed before deleting the target when projection coverage is
stale, incomplete, unreadable, or under repair. Repair rebuilds the projection
from canonical Task truth; the Runtime Spec must define its bounded traversal,
crash recovery, coverage proof, and degraded-state behavior before
implementation.

Scheduling liveness continues to use the existing persistent ready marker and
bounded cyclic candidate traversal. A dependency-blocked queued Task retains
its marker; cursor progress skips it without deleting it, and later traversal
re-evaluates the dependency gate even when no downstream Task revision changed.
Cross-Group support must preserve the existing dependency-only primary-demand
recheck so waiting dependencies neither create false primary demand nor starve
other candidates. A notification may reduce latency, but correctness and
restart liveness must not depend on delivery of a best-effort notification or
on the reverse cleanup projection. Final claim locking remains authoritative.

## Compatibility and activation

This is a backward-compatible CLI expansion but a persisted semantic change.
An older `task-dependencies-v1` agent interprets a valid cross-Group edge as
invalid and cannot provide the new liveness or cleanup guarantees. Therefore a
root must not admit mixed v1/v2 dependency writers or schedulers.

Introduce the permanent required capability `task-dependencies-v2` through the
existing qualified upgrade and activation workflow. It is additive: v2-capable
software continues to support `task-dependencies-v1`, and canonical v2 roots
list both capabilities. Existing roots continue with v1 semantics until an
explicit activation has drained incompatible processes, added the v2 gate,
built and audited the reverse projection, and opened v2 scheduling. Fresh
roots install both dependency capabilities before admission. Existing
same-Group Task records require no semantic rewrite; their ID lists are valid
v2 edges.

Do not add a reader shim that silently treats cross-Group edges as same-Group
or empty. Activation is forward-only after the v2 capability is installed;
rollback requires restoring a pre-activation root snapshot and compatible
process set, not removing the gate in place. If temporary upgrade code is
required, register its lifecycle before integration. Product and Runtime Spec
changes must record the activation prerequisites, operator-visible failure
modes, and recovery procedure before implementation begins.

## Effort and sequencing assessment

The cross-Group change has clear product value, but it is a medium-to-large
runtime change rather than a parser relaxation. The highest-risk work is the
concurrency boundary around cycle creation and cleanup, followed by liveness
when a prerequisite changes after the scheduler cursor has passed a downstream
marker. Persisted Task shape and normal CLI syntax can remain stable.

Deliver it in two independently reviewable steps:

1. Clarify direct versus transitive prerequisites in user documentation and
   examples. This resolves avoidable duplicate declarations without changing
   behavior.
2. Implement and activate the Project-wide dependency protocol with the full
   locking, recovery, cleanup, and mixed-version gates in this pitch.

Do not bundle a Group snapshot selector into step 2. It adds a second product
decision but does not simplify the Project-wide runtime protocol.

## Acceptance criteria

| ID | Scenario | Required result |
| --- | --- | --- |
| PD-01 | Group `eval` Task C depends on Group `train` Task A | C waits, then becomes claimable only after A succeeds |
| PD-02 | In separate acyclic cases, a grouped Task depends on an ungrouped Task and an ungrouped Task depends on a grouped Task | Both same-Project combinations follow the same dependency gate |
| PD-03 | C depends on independent A and B | C remains unclaimable until both succeed |
| PD-04 | A -> B -> C and C lists only B | C cannot run before B succeeds; C need not repeat A |
| PD-05 | Edges across two or more Groups form a cycle | Submission or edit fails atomically without changing Task truth |
| PD-06 | Concurrent opposite cross-Group edits would form a cycle | At most one edit commits; operations do not deadlock |
| PD-07 | An upstream Task succeeds while no agent is currently inspecting the downstream marker | Later bounded cyclic traversal makes the downstream Task schedulable without resubmission or notification delivery |
| PD-08 | Cleanup starts for a Task referenced from another Group or by an ungrouped Task | Cleanup is rejected; diagnostics identify confirmed direct dependents without trusting the projection as edge authority |
| PD-09 | A new dependency races with cleanup start | Exactly one operation commits; no dangling reference or deleted prerequisite is possible |
| PD-10 | An upstream Group is sealed, paused, reopened, or cancelled | Only its documented lifecycle behavior occurs; no control action propagates across the edge |
| PD-11 | Submission replay or dependency editing contains cross-Group edges | Preview, digest, replay, and committed direct-edge sets remain identical |
| PD-12 | A v1 process opens a v2 root | Required-capability validation rejects it before mutation or scheduling |
| PD-13 | Reverse dependency projection coverage is missing, stale, damaged, or under repair | Claim and scheduling remain safe, cleanup retains Task truth, and bounded repair can rebuild the projection from Tasks |
| PD-14 | A prerequisite ID belongs to another Project or is missing | The operation fails without attempting cross-Project discovery |
| PD-15 | A Task references another candidate in the same atomic Submission | Validation overlays the complete candidate set; commit publishes either the whole graph or none of it |
| PD-16 | A v2 edge/projection update crashes between durable writes | Recovery under the dependency fence converges Task truth and the projection before cleanup can trust coverage |

## Delivery outline

- [ ] Update Product and Runtime Specs with Project-wide dependency semantics,
      Group isolation, lock order, cleanup, traversal, and activation contracts.
- [ ] Define the v2 capability and any temporary compatibility lifecycle before
      implementing persisted or mixed-version behavior.
- [ ] Implement Project-scoped dependency mutation fencing, ordered multi-Group
      and Task locking, reachable-closure cycle validation, and final claim gate.
- [ ] Implement rebuildable reverse cleanup discovery with explicit coverage and
      degraded-state evidence; preserve ready-marker traversal and dependency
      rechecks; update cleanup, observation, doctor, and recovery paths.
- [ ] Preserve the existing CLI and manifest fields, improve diagnostics, and
      document direct versus transitive prerequisites.
- [ ] Add focused normal, failure, concurrency, crash-recovery, traversal,
      cleanup, idempotency, and mixed-version rejection tests for PD-01 through
      PD-16.

The aggregate sealed-Group snapshot selector is deliberately outside this
delivery. Revisit it only after the Project-wide explicit-edge change has usage
evidence and a separate acceptance contract.
