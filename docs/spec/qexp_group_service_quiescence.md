# qexp Group Service Quiescence Protocol

Status: proposed; implementation requires protocol and compatibility approval.

## Purpose and boundary

This protocol makes routine Group discovery and metadata maintenance proportional to unfinished
obligations. Retained Group and Task history remains authoritative and queryable, but history alone
does not allocate a resident service session or require a recurring Group-directory sweep.

The protocol preserves the existing `open|sealed` admission states, `active|paused` dispatch states,
Submission Operation commit point, Group and Task lock order, ready and due-work indexes, local
Attempt responsibility ledger, Group control operations, and machine-global `daemon|on_demand`
residency policy. It introduces no user-visible inactive state, automatic seal or pause, Task-history
retention policy, external timer launcher, or execution-capacity rejection state.

The new records locate service work. They never authorize Task mutation, execution, cancellation,
resource release, cleanup, or Group membership. Consumers must re-read the existing authoritative
records and retain their current fences before producing an effect.

## Existing mechanisms and ownership

The implementation must reuse these owners rather than copy their truth into the locator protocol:

| Obligation | Existing durable owner | Service use |
| --- | --- | --- |
| Newly committed Group membership | Submission Operation, Group `pending_submission_commit`, membership discovery debt, Group coverage | Membership lane locates projection work; coverage remains authoritative for certified membership |
| Task transition affecting active Group control | Revisioned Task/Attempt truth and the Group recheck journal | Control lane locates unsettled journal and active-operation work |
| Group cancel or Worker removal | Active Group control operation, receipts, coverage and recheck cursors | Control lane keeps required coverage and settlement discoverable |
| Source, receipt, journal and generation cleanup | Existing per-Group maintenance debt and retention records | Maintenance lane locates cleanup work; debt records remain deletion authority |
| Ready or capacity-deferred Task | Ready indexes and authoritative Task truth | Outside this protocol; scheduler admission and demand classification are unchanged |
| Delayed availability | Durable offer-deadline index and Task truth | Outside this protocol; the resident agent continues current due-time service |
| Submission recovery | Submission-control active responsibility and Submission Operation | Recovery publishes or refreshes the membership locator before retiring its existing responsibility |
| Local Attempt launch, supervision or recovery | Machine-local responsibility ledger, process evidence and reservations | Never acknowledged through shared Group service state |

The current `operations/group-discovery/active/` records remain immutable membership-source debt.
The current coverage `rechecks`, source-cleanup debt, receipts and diagnostics retain their formats.
The proposed locator is needed because those records live below per-Group directories and cannot be
found after cache eviction without reopening every retained Group.

## Persisted locator protocol

### Shared layout

An activated Project has three shared lanes:

```text
operations/group-service-v1/
  membership/<00..ff>/<group-digest>.json
  control/<00..ff>/<group-digest>.json
  maintenance/<00..ff>/<group-digest>.json
```

`group-digest` is lowercase SHA-256 of the UTF-8 Group name. The first digest byte selects one of
256 fixed shards and the remaining digest names the record. Hashing bounds directory fan-out but
does not establish identity: every read verifies the Group name and canonical Group-authority
identity stored in the record. A path collision or mismatched identity makes that lane unavailable;
it never replaces the existing record.

Each path contains one coalesced wake-up locator:

```json
{
  "version": 1,
  "identity": {
    "project_id": "project-id",
    "group_directory_identity": {},
    "group": "experiment"
  },
  "service": "membership",
  "generation": 7,
  "published_at": "2026-09-23T00:00:00+00:00",
  "reason": "submission_commit"
}
```

The exact top-level fields are `version`, `identity`, `service`, `generation`, `published_at`, and
`reason`. `version` is `1`; `service` matches its lane; `generation` is a positive integer below
`2^63`; `reason` is one of the producer reasons defined below and is diagnostic, not authority.
The identity binds the current Project, canonical Group directory generation, and Group name.
Records are limited to 16 KiB.

A layout coordinator creates all three lane directories and all 768 shard directories before any
writer may treat locator publication as the owner of a transition. It synchronizes every shard,
each lane, `group-service-v1`, `operations`, and the shared root, then publishes a 16 KiB
`group-service-v1/layout.json` record bound to Project and canonical Group-directory identity.
Writers never create a lane or shard. A missing directory, symlink, wrong identity, or absent layout
record makes publication unavailable. Before activation, the existing historical service path still
owns discovery when dual-publication is unavailable; after the writer fence is installed, failure to
publish prevents the covered effect.

`layout.json` has exactly `version`, `identity`, `revision`, `lanes`, `shard_count`, `path_scheme`,
`manifest_digest`, and `created_at`. Version and revision are 1. Identity contains exactly
`project_id` and `group_directory_identity`. Lanes is the sorted list `control`, `maintenance`,
`membership`; shard count is 256; path scheme is `sha256-first-byte-v1`. `manifest_digest` is SHA-256
of canonical JSON containing every expected `lane/00..ff` relative path in lexical order.

Before publishing the marker, the coordinator opens every expected path with directory, close-on-
exec and no-follow flags, rejects any unexpected type, synchronizes bottom-up, and repeats the exact
path checks. The activation record stores layout revision and digest. A writer validates the exact
marker, its current Group-authority identity, and its selected lane/shard path without following a
symlink. A marker change, target-path failure, or mismatch with activation evidence rejects
publication. The fixed manifest does not treat client-local inode or device values as shared
identity.

A producer holding the Group writer fence reads the existing matching record, increments its
generation, and atomically replaces it before committing the transition that creates or changes the
obligation. Absence starts generation 1. Parent creation and replacement are synchronized through
the precreated layout and existing durable atomic-write primitive. The Group fence serializes
generation changes and acknowledgement; no separate locator lock is introduced.

A consumer may unlink a locator only while holding the same schema and Group writer fences, after
it has:

1. re-read the exact locator and remembered its generation;
2. proved the lane-specific retirement predicate from authoritative truth;
3. re-read the locator and verified that identity, service and generation are unchanged;
4. unlinked the record and synchronized its shard directory.

A later producer cannot be deleted by an older acknowledgement because it must acquire the same
Group fence and publish a higher generation. A crash before unlink causes duplicate delivery. A
crash after unlink but before directory synchronization may make the same locator reappear. Both
cases are safe because consumers revalidate authoritative truth and all existing effects are
idempotent. There is no cross-machine acknowledgement record.

The locator is authoritative only for ordinary discovery of its lane. Loss or external deletion is
storage corruption and requires explicit audit/rebuild. Its payload is never authority for the
underlying work.

### Machine-local traversal state

Each machine runtime may retain an advisory cursor per Project, binding generation and lane:
`group-service-cursors/<project-id>/<lane>.json`. It records shard and last digest only. Cursor loss
or corruption restarts that active-locator traversal and cannot acknowledge shared work. The worker
persists at most once per 64 locator visits or on a completed shard, so cursor writes are bounded.

Traversal uses one lazy `scandir` owner per selected lane and shard. It does not materialize a
directory listing. Restart may repeat records. Within a continuously healthy run, each lane advances
by digest and wraps through all 256 shards, so a repeatedly failing early record cannot starve later
records.

## Producer inventory and publication order

Every supported writer below must publish before its owning effect. Publication failure prevents
that effect unless the row identifies an already durable owner for the gap.

| Producer transition | Lane and reason | Required order and recovery owner |
| --- | --- | --- |
| New grouped Submission begins its commit interval | `membership/submission_commit` | Locator precedes the first durable Group pending-commit or staged membership effect. Submission recovery retains the locator through abort or finalization. |
| Committed Submission finalization or replay | `membership/submission_finalize` | Existing membership-source debt remains durable before Group pending state clears. Recovery refreshes the locator before retiring submission-control responsibility. |
| Retry, claim, launch, terminal, cancellation, availability, cleanup or recovery transition that publishes a Group recheck | `control/task_change` | Locator precedes the recheck ticket; ticket precedes the covered effect under the current Group/Task fences. Failed ticket publication retains current invalidation behavior. |
| Group cancellation or Worker removal creation/retry | `control/group_operation` | Locator precedes the active operation or its new generation. The active operation remains the idle blocker and effect owner. |
| Source completion creates source-cleanup debt | `maintenance/source_cleanup` | Locator precedes cleanup debt; cleanup debt precedes retirement of the active source locator. |
| Receipt, recheck-prefix or obsolete-generation cleanup becomes eligible | `maintenance/metadata_cleanup` | The transition that first makes cleanup safe publishes or refreshes the locator before dropping its prior recovery owner. |
| Explicit audit, repair or activation bootstrap finds unfinished work | Matching lane with `repair` or `bootstrap` | The resumable coordinator publishes locators before advancing its certified scan cursor. |

Task submissions without a Group publish no Group locator. Ready index publication, offer deadlines,
lease renewal, process supervision and local recovery do not publish a Group locator unless they
also execute a transition already listed above. Lease-only updates remain outside the recheck
journal.

The supported writer inventory is: submit command and Python submission API, Submission recovery
and doctor, task retry/cancel/dependency/availability commands, scheduler claim/launch/claim-loss
paths, terminal transition and recovery paths, Group cancellation and Worker changes, cleanup, and
the discovery/maintenance worker itself. B2 may not activate until tests demonstrate that every
listed writer either publishes the required locator or is fenced from the activated Project.

## Consumer state machines and retirement

### Membership lane

An admitted membership locator creates or reuses one bounded `GroupDiscoveryService` in activated
locator mode. That constructor requires matching active `schema/group-service.json` evidence and
never creates a `SubmissionSourceSweep`. It resumes an already durable active source and processes
only membership-source debt. Missing or mismatched activation evidence fails closed. The exceptional
bootstrap/rebuild coordinator is the only component allowed to enumerate retained Submission
sources after this mode exists.

The activation bootstrap certifies coverage for every Group present before the writer fence, while
target writers publish locators for every later change. The global active record is therefore the
bootstrap proof; a per-Group missing `background.bootstrap_complete` marker cannot select a source
sweep after activation. B2 must make the mode explicit in the service interface and must not infer
it merely from a caller or an absent background file.

The service may acknowledge the observed generation only when all of these are true under the Group
writer fence:

- no `active-source.json` owner remains;
- the Group membership-debt bucket is empty and its absence is synchronized;
- certified consecutive coverage reaches the current committed Group membership tail;
- no `pending_submission_commit` or unresolved Group creation operation can extend that tail;
- no gap, ambiguity or unavailable coverage diagnostic remains.

An aborted producer is acknowledged only after its Submission recovery owner is terminal and no
membership effect remains. A large source, replaced source, corrupt receipt or missing Task locator
keeps the generation pending with bounded retry and diagnostics.

### Control lane

The control service advances existing cancellation/removal consumers and typed recheck settlement.
It may acknowledge the observed generation only when:

- every active Group control operation for the Group is terminal or has durably handed remaining
  execution/recovery work to its existing owner;
- the recheck journal has no in-flight ticket at or below the observed tail;
- every active consumer cursor has reached the certified tail or has a durable terminal receipt;
- current membership coverage required by an active operation is complete;
- no invalidation, unexplained missing event, or failed settlement remains.

An acknowledgement is shared because all predicates and effects in this lane are shared Project
truth. It cannot acknowledge machine-local launch, process, termination or recovery responsibility;
those remain in the eligible machine's local ledger even after the shared operation advances.

### Maintenance lane

`GroupMaintenance` processes its four existing lanes. It may acknowledge a locator only after one
complete certified pass, started after the observed generation, proves:

- no reclaimable resolved recheck prefix remains;
- no eligible completed-operation receipt remains;
- no source-cleanup debt or cleanup-owned scratch remains;
- no obsolete generation eligible for removal remains;
- no lane owns an open iterator, deletion job, retryable error or unpersisted diagnostic.

New cleanup eligibility during the pass publishes a higher generation and prevents unlink of the
older observation. Optional cleanup may remain pending when an on-demand agent exits; its locator
and completed pass cursor survive and resume on the next explicit agent start.

The pass cursor is a new rebuildable checkpoint at the identity-bound coverage path
`maintenance/quiescence-v1.json`. Its exact fields are `version`, `identity`,
`locator_generation`, `pass_id`, `next_lane`, and `lanes`. `version` is 1; `pass_id` is a positive
UUID integer; `next_lane` is one of `journal`, `receipts`, `sources`, or `generations`. Each lane
stores `state` (`pending|scanning|complete`), the source directory identity and revision, a
nonnegative Linux directory cookie, the exact current job identity or null, and the lane-specific
certification accumulator. The entire record is limited to 64 KiB.

B2 reuses `runtime/directory_capture.py::read_directory_entry` so each step opens one directory,
seeks to the durable cookie, reads at most one entry, closes the descriptor, and checkpoints only
after the inspected effect or no-effect classification is durable. Nonmutating census lanes retain
their cursor only while the directory identity/revision remains unchanged; a change resets that
lane and its accumulator to offset zero. A mutating lane persists the selected job before its first
deletion. After the job finishes, its authoritative debt deletion is the durable progress proof;
the lane restarts at offset zero against the new revision rather than trusting a cookie across
directory mutation. Thus crash and eviction may repeat inspection, but never repeat an authorized
effect or forget a completed deletion.

Stable EOF plus an unchanged final directory revision marks one lane complete. All four lanes must
be complete for the observed locator generation before acknowledgement. A higher locator generation
resets the pass. Corrupt checkpoint state is discarded and rebuilt from authoritative debt while
the locator remains; it never authorizes deletion. This checkpoint, the existing debt records, and
the one-entry cursor allow safe session eviction and preserve progress across repeated short runs
without retaining an open iterator or rescanning Group history.

### Session eviction

Lane acknowledgement and object eviction are separate. A worker may close a service object after a
bounded quantum even when its locator remains, provided all unfinished progress is durable in the
existing service checkpoint/debt, the maintenance pass checkpoint when applicable, and the locator
still exists. Closing releases parsers, iterators, buffers and descriptors but does not mark the
lane complete. A Group with no admitted locator owns no routine discovery or maintenance entry.

A running Attempt does not retain a Group discovery session. Its supervision remains machine-local.
Conversely, all current Tasks being terminal does not permit locator acknowledgement while control,
recovery, publication or cleanup predicates remain unfinished.

## Worker scheduling, caps and fairness

The first B2 implementation uses these machine-wide hard caps:

| Resource | Cap |
| --- | ---: |
| Resident Group entries across all Projects | 64 |
| Resident source parser/recovery owners within those entries | 16 |
| Concurrent cooperative close owners | 16 |
| Open descriptors attributable to Group service | 256 |
| In-memory locator candidates | 64 |
| Pending member candidates per source | 64 |
| Group service threads | 1 |
| Source slice | 256 KiB, 32 I/O operations, 64 KiB parsed bytes, 20 ms soft deadline |
| One locator or ordinary maintenance record | 16 KiB |
| Maintenance quiescence checkpoint | 64 KiB |

The worker must refuse a new admission before exceeding a cap. The locator remains durable and is
revisited after another entry evicts. It may not allocate one thread, unbounded retry object, queue,
or diagnostic per Group.

Four turns form the minimum fair cycle: two control turns, one membership turn and one maintenance
turn. Projects rotate within each lane; shards and digests rotate within each Project. When a lane
has no work its turn may be borrowed, but maintenance receives at least one turn per four while it
has eligible work. Local supervision, renewal, terminal reconciliation and reservation release run
outside this worker and retain their current capacity reservation.

One failing locator receives exponential retry delays of 1, 2, 4, 8, 16, 32 and at most 60 seconds.
Retry state exists only for resident entries; eviction discards that cache, while durable diagnostic
and locator state remain. A reloaded failure starts at one second and cannot prevent traversal of
later locators. Diagnostic text is capped by the existing 2 KiB reason bound and replaces the prior
reason rather than appending unbounded history.

The implementation must expose per lane: pending locators encountered, resident entries, oldest
observed pending age, completions, retries, evictions, bytes, reads, writes, descriptor high-water,
and cap refusals. Metrics use maintained counters and sampled active-lane traversal; they do not
scan Group or Task history.

## Locks, concurrency and acknowledgement

The global order remains schema writer fence, Group writer lock, then sorted Task locks. Locator
publication and acknowledgement occur inside the Group portion of that order. Directory traversal
holds no schema, Group or Task lock. A consumer closes its iterator before acquiring the Group lock
when the filesystem implementation cannot guarantee an independent descriptor.

Two machines may deliver the same shared locator. Both revalidate authoritative truth; only the
machine holding the Group writer lock may settle a step or acknowledge. The later machine observes
the removed record or repeats an idempotent check. No machine identity is stored in shared
acknowledgement because completion is shared. Machine-local responsibility always uses the existing
ledger and eligible machine identity, and no shared consumer may retire it.

Delayed or blocked work keeps its current owner and retry policy. A locator timestamp is never a
launch deadline, scheduler authorization, or promise to start an absent agent.

## Crash and race matrix

| Boundary | Required recovery result |
| --- | --- |
| Crash before locator publication | Owning effect has not committed; retry may publish generation 1 |
| Locator durable, effect absent or aborted | Consumer revalidates, waits for the producer's terminal recovery state, then safely acknowledges no-effect work |
| Locator and recheck/debt durable, effect absent | Existing transition owner settles or aborts its exact ticket/debt; duplicate delivery is safe |
| Effect durable, producer crashes before later publication/finalization | The pre-effect locator and existing operation/ticket own recovery; no historical Group sweep is needed |
| Consumer completes work, crashes before acknowledgement | Locator redelivers; authoritative predicates make the repeated work idempotent |
| Consumer unlinks, crash before directory synchronization | Locator may reappear and redeliver; it cannot authorize a duplicate effect |
| Producer races with acknowledgement | Group lock serializes them; either acknowledgement removes the old generation before a higher one is published, or generation mismatch preserves the new work |
| Agent exits or crashes with open sessions | Cooperative close releases in-memory owners when possible; locators, checkpoints, debts and local responsibility ledgers reconstruct unfinished work after restart |
| One machine completes shared work while another is offline | Shared locator may be acknowledged; the offline machine's local ledger remains and is processed only by an eligible owner after return |
| Locator record is malformed, collides or has the wrong authority identity | Lane becomes degraded for that Project; no acknowledgement or mutation is authorized; explicit audit/repair rebuilds from truth |

## Residency classification

The Group service worker remains idle-neutral. Pure history and optional maintenance locators do not
prevent an `on_demand` agent from exiting. The existing full-loop true-idleness interval remains
required.

| Work | Existing residency owner |
| --- | --- |
| Ready, capacity-deferred or dependency-waiting demand | Scheduler demand classification and ready/due indexes |
| Active Group cancel/removal or cleanup control | Existing active operation namespaces checked by true-idle logic |
| Local running, starting, termination or recovery responsibility | Local ledger, process evidence and reservations |
| Failed mandatory maintenance that prevents a current idle proof | Existing maintenance/control diagnostic owner |
| Optional metadata cleanup only | Idle-neutral locator; progress resumes while an agent is resident |
| Retained Group/Task history only | No locator and no residency effect |

This protocol does not add an external timer. Due work progresses while the agent is resident and is
rediscovered from its existing durable index after an explicit restart or normal local-submit
activation. `daemon` remains resident; `on_demand` may exit only through its existing global proof.

## Activation, mixed versions and repair

### Compatibility decision proposed for approval

The protocol requires a new permanent `group-service-v1` writer capability and a temporary
`QQTOOLS-COMPAT-0017` rolling-activation item. The supported source is 1.3.21 and the target is
1.3.22. Normal and recovery coordination are L1: install the target package and restart the global
agent on each machine. Running training processes continue; scheduling may pause on only the
restarted machine within the existing one-second interruption budget.

The target release initially dual-publishes locators while retaining the historical Group sweep.
`schema/group-service.json` progresses through `preparing`, `fenced`, `building`, `active`, or
`degraded`. It is limited to 16 KiB and has exactly `version`, `revision`, `protocol`, `identity`,
`state`, `activation_epoch`, `writer_floor`, `layout`, `bootstrap`, `diagnostic`, and `updated_at`.
Version is 1, revision is a positive monotonically increasing integer, protocol is
`group-service-v1`, and activation epoch is a positive UUID integer. Identity contains exactly
`project_id` and `group_directory_identity`. Writer floor is `1.3.22`. Layout contains exactly the
accepted `revision` and `manifest_digest` from `layout.json`.

Bootstrap contains exactly `generation`, `phase`, `cursors`, `stable_pass`, `post_fence`, and
`completed_at`. Generation is a positive UUID integer. Phase is one of `groups`, `submissions`,
`active_namespaces`, `stable_pass`, or `complete`. Cursors contains exactly `groups`, `submissions`,
`submission_control`, `group_control`, `cleanup`, and `discovery_debt`. Each cursor contains exactly
`directory_revision`, `cookie`, and `complete`; directory revision is null before opening or the
exact `device`, `inode`, `size`, `mtime_ns`, and `ctime_ns` witness, cookie is a nonnegative Linux
directory cookie, and complete is boolean. A revision mismatch resets only that namespace to cookie
zero; a machine whose shared-filesystem device view differs may repeat work but cannot skip it.

The `groups` cursor certifies canonical Group records, `submissions` certifies retained Submission
sources, and the four active cursors cover Submission control, Group control, cleanup, and existing
membership-discovery debt. Phase advances only when every cursor owned by the current phase is
complete. Stable-pass certification reopens every namespace and compares its final revision with
the cursor witness; any change resets the affected cursor and phase. `stable_pass` is a nonnegative
integer, `post_fence` is boolean, and `completed_at` is null until a fully post-fence stable pass.
All cursors use the bounded durable directory-capture primitive and checkpoint after at most one
entry, so restart never has to infer which directory a numeric cookie belongs to.

Diagnostic is null or contains exactly bounded `code` and `detail` strings. Every transition holds
the exclusive schema fence, verifies current Project and canonical Group authority identity, and
increments revision. The allowed transitions are
`preparing -> fenced -> building -> active`, `active -> degraded`, and
`degraded -> building -> active`; other regressions or an activation-epoch change fail closed.

An activated service requires state `active`, matching protocol/identity/layout, completed
post-fence bootstrap evidence with at least one stable pass, the required `group-service-v1`
capability in current schema truth, and an installed writer meeting the floor. Fixed metadata may be
cached only with the schema record revision and is revalidated at the existing writer/open fences.
The global machine upgrade coordinator advances every registered Project; no per-Project command is
part of the normal path.

Activation requires all of the following:

1. every registered participant and supported writer reports the target locator capability at its
   current registration generation;
2. the coordinator durably precreates and verifies the complete locator layout;
3. under the existing all-participant and exclusive schema fences, the coordinator selects the new
   required writer capability and records `fenced`; a returning 1.3.21 process can no longer mutate
   the Project, while target writers continue dual-publication and the old service path remains
   active;
4. only after `fenced`, a resumable bounded bootstrap scans every canonical Group, retained
   Submission source, and active control/recovery namespace, publishes locators for all unfinished
   obligations, certifies membership coverage, and completes a stable final pass;
5. the final pass runs entirely after old-writer exclusion; target writers dual-publish throughout,
   so a change racing behind the cursor already has a locator;
6. the schema, writer floor, layout and canonical Group authority identities still match the
   bootstrap evidence when the activation commit selects `active`.

Only the activation commit selects locator-only admission and activated debt-only service
construction. Before it, the old worker continues its Group sweep, including while the Project is
`fenced` or `building`. After it, no routine fallback scans Group or Submission history. A
participant that cannot be fenced keeps the Project in `preparing`; compatible service continues
through the old path. B2 must not begin bootstrap before the fence or silently activate on partial
writer evidence.

If the fixed metadata, active lane roots or authority identity become unavailable, the Project
enters `degraded`. New capable writers continue dual-publication. Local Attempt supervision,
renewal, terminal evidence collection and unaffected Projects continue. Affected Group control
completion waits rather than inferring absent work. `qexp admin check` reports the failed lane and
known discovery boundary.

`qexp admin repair` requests a resumable machine-coordinated rebuild. Rebuild scans canonical Groups
and active operation/debt namespaces under bounded cursors, recreates locators before advancing each
cursor, then repeats the activation stability checks. It never runs on submit, deletes user history,
or treats a periodic audit as the ordinary wake-up mechanism. Corruption outside all available
discovery sources remains an explicit integrity error.

Temporary dual-read/build logic may be removed only through the compatibility registry lifecycle.
The permanent locator reader, writer capability fence, active-state check and explicit rebuild
remain supported contracts.

## Qualification and approval gate

B2 implementation is authorized only after review accepts these decisions:

1. three coalesced, generation-fenced locator lanes and their authority boundary;
2. shared acknowledgement with strict exclusion of machine-local responsibility;
3. the producer inventory, pre-effect publication order and lane retirement predicates;
4. the numerical caps, 2:1:1 fairness cycle and bounded retry policy;
5. the 1.3.21 to 1.3.22 L1 dual-publication, bootstrap, activation and degraded-rebuild contract.

Implementation must add deterministic tests for every crash-matrix row and operation counts with
0, 1,000, 10,000 and 100,000 unrelated completed Groups. Qualification varies Group count
independently of Task count and covers open, sealed and paused Groups; append/retry after quiescence;
offline consumers; concurrent publishers; session eviction while an Attempt remains supervised;
on-demand exit; storage failure; and active-locator overload.

Before activation, freeze the qualification envelope: at least 100,000 accelerated obligation
lifecycles, a 24-hour real-process soak, maximum 64 resident entries, maximum 256 Group-service file
descriptors, no extra service threads, and no growth in resident entries, queues or descriptors with
cumulative completed Group count. RSS may vary within a measured allocator envelope but must not
trend with retained Group history. The run records filesystem/cache conditions, reads, bytes, lock
wait/hold, oldest-work age, completion rate, restart phases and injected slow-I/O failures.
