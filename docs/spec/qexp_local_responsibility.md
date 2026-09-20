# Local recovery membership and qualified discovery

The local membership store provides direct discovery of unfinished Attempts.
Machine-owned bounded supervision uses it as the primary discovery source only
after exact-generation retained capture qualification. Before qualification,
existing evidence lanes and startup completion rules remain enabled. Membership
never authorizes a claim, process creation, renewal, terminal transition or
resource release. Missing or corrupt membership cannot imply that an Attempt is
absent, and cannot justify launching a replacement process.

## Integration and compatibility

The executor requires durable publication after launch authorization and before
creating the runner or tmux window, outside shared authority locks. Existing
reservations cover the earlier claim/authorization crash window. The per-project
directory is `runtime_root/recovery-responsibilities`; each entry stores Attempt
identity, Task ID and Attempt number, without copying changing lease or fencing
state. Exact retries write nothing. Publication failure increments
`responsibility.publication_unavailable` in dispatch diagnostics and fails the
launch before runner/window creation. Existing launch-failure reconciliation owns
the authorized Attempt and reservation; index failure does not grant replay rights.
Compensation may mark an unstarted Attempt failed only after its locked check finds
no local launch intent, registration, process manifest or exit observation. If a
handoff observation fails after the runner has published intent, authority and
capacity remain with that Attempt for recovery; the pending wrapper cannot run
after a compensation-induced resource release.
The runner also requires the same membership before taking shared authority locks,
then revalidates its claim and Attempt before publishing the launch intent and
creating a workload. This covers a runner invoked by an older Executor. The second
publication is idempotent and adds no writes. Already running wrappers continue to
publish their final observations without requiring an available membership store.
Required publication by current code does not prove complete old-writer coverage.

The supervisor prefetches up to 64 candidates using bounded background storage
work. At most four storage threads may run per machine process, with no executor
job backlog, and at most one outstanding read and 64 buffered candidates per
project. Before qualification, membership discovery shares the non-active half of the semantic-step
budget with the six evidence/maintenance lanes. Each fourteen-step round reserves
seven active service turns and one turn for each other lane; membership readiness
or failure cannot reduce the active share. The advisory membership lane consumes
at most one candidate every four ticks, deferring until its next allocated turn.
Existing evidence paths provide all process and
ownership checks. Index errors appear as `responsibility_discovery_failures`;
they never turn incomplete startup into completed startup.

After initial active acknowledgement, each read turn reserves up to 48 candidates for active work and 16 for maintenance,
using independent reverse cursors and round-robin bucket turns. The active budget
is split across four buckets with up to 12 candidates each, so a new launch just
behind the cursor does not wait sixteen machine cycles. Maintenance retains its
separate 16-candidate turn. A failed maintenance
page does not suppress the active page. The active traversal's initial sweep does
not wait for the maintenance sweep; neither sweep proves complete writer capture.
Base-only buckets receive one restartable build slice of at most eight entries per
worker turn, before stage reads. Build failures preserve existing evidence discovery.

Index I/O and redo do not run on the supervisor thread. Closing a supervisor does
not wait for a blocked storage operation; storage workers are daemon threads and
cannot invoke supervisor callbacks or mutate execution authority. Publication
still runs on the dispatch thread and may wait for local storage. This change does
not promise bounded admission latency or a wall-clock bound for arbitrary local
filesystem operations. Existing evidence reads and shared writes retain their
existing latency limitations.

The bounded supervisor queues cleanup only after terminal/accounting checks and
known writer quiescence. The storage worker durably hands membership to maintenance
with an immutable `qexp-local-cleanup-v1` receipt **before deleting evidence**.
Each slice removes at most eight termination decisions and the six fixed evidence
files. All affected directory namespaces are synced before membership retirement.
An incomplete or failed slice retains the receipt. Restart can finish cleanup from
that receipt even after every original evidence file and shared Task has gone.
The 64-item request queue is only an optimization: before durable handoff the
original evidence remains; afterward the stored receipt owns replay. A cleanup
error does not suppress unrelated membership reads.

A receipt contains its format, Task ID, Attempt ID and proof basis; the entry
retains the exact Attempt number when established by Attempt truth or canonical
identity. Explicit Task cleanup can retain an unknown number as null under the
`task_cleanup` basis, which additionally records the durable cleanup operation ID.
Receipt encoding is limited to 4 KiB within the
existing 64 KiB entry bound. Existing maintenance entries without a receipt grant
no deletion permission. A recorded receipt cannot be replaced by a different proof;
replay checks membership generation before deletion and uses CAS retirement.

Wrapper PID/start identity and process-group absence are checked before handoff.
Unreadable process state, a live writer, or an unclassified legacy wrapper record
retains responsibility. Existing protocol-1 recovered manifests that lack both
wrapper fields may instead use an absent recorded process group plus a matching,
valid final exit observation: protocol 1 performs no further inbox writes after
that observation. A partial wrapper identity, missing exit, unknown protocol or
live group does not satisfy this alternative. These checks do not kill processes.
For deleted Tasks, a
matching completed cleanup tombstone, required machine acknowledgements, absence
of matching local reservations, and a valid final exit observation provide the
alternative proof. Old acknowledgements alone do not prove writer quiescence:
without that final write or an already stored receipt, cleanup stays unresolved.
An opaque Attempt identity additionally requires an explicit matching Task ID in
the final observation. Its operation-bound `task_cleanup` receipt may retain an
unknown Attempt number; the supervisor does not invent one from the current Task
counter or treat advisory membership as ownership proof.

For a surviving Task, an active membership's captured Attempt number allows direct
validation of a historical noncanonical identity after a retry. The loaded Attempt
must match the requested identity, number, Task, machine and terminal state; the
Task sequencing, claim and accounting checks still apply. A locator for a successor
or an out-of-range number does not grant cleanup permission. This path does not
enumerate the Task's Attempt directory. Bounded supervision uses the prefetched
locator, including when no local evidence file remains, and never performs index
lookup or redo on its own thread. Maintenance without a receipt is not promoted
back into an active candidate by this path.
When a prefetched locator retains only the Attempt number, late local evidence
can supply the Task ID without losing that number. Direct truth validation still
precedes cleanup, and filling the persisted locator occurs in the storage worker.
This also applies after the corresponding backfill lane has already completed.

The cleanup command also uses this handoff for matching local Attempt truth,
including noncanonical identities recovered from the Task's Attempt records. It
holds the existing Task cleanup fence and validates the terminal phase, exact
membership locator and writer quiescence before deleting evidence. A live or
unresolved writer retains its evidence and delays the machine acknowledgement;
an unfinished eight-decision slice reports `local_cleanup_pending`. Failed storage
does not acknowledge cleanup or delete Task truth. An existing receipt supports
retry after partial evidence deletion. Clean also releases matching active and
provisional CPU reservations, alongside GPU reservations, before acknowledgement.
Stable operation aliases and retained import copies do not grant an active cleanup
operation a second work slice in the same legacy-inclusive discovery pass.

Explicit Task cleanup inventories all seven evidence lanes and unresolved local
memberships to capture records with no matching shared Attempt. It does not delete
those records through a direct-unlink fallback. The caller holds the existing Task
cleanup fence; the durable operation must name this Task and machine, record a
terminal Task state, and match surviving Task control with no active claim. Initial
handoff additionally requires Task ownership from local evidence or canonical
Attempt identity, a valid final protocol-1 observation, and writer quiescence.
Advisory membership alone cannot establish Task ownership. Missing final evidence
or a live/unclassified writer retains files and prevents the machine acknowledgement.
Corrupt shared Attempt records fail instead of being reclassified as absent truth.

The operation-bound receipt is durable before evidence deletion. It supports retry
after both the local records and shared Task/Attempt truth have disappeared. A
different cleanup operation or conflicting local identity cannot reuse the receipt.
This inventory is part of explicit cleanup maintenance and is not a replacement for
normal active discovery. Known Attempt truth can fill partial local-backfill locators
as well as imported locators; neither may overwrite an already known field.

Unbounded supervisor APIs use the same proof and deletion protocol synchronously.
They service one membership candidate per startup/tick for receipt-only replay;
no surviving evidence filename or Task record is required. Candidates owned by
proven cleanup are not rematerialized or replayed as live termination work.
Storage errors preserve evidence/receipts and produce diagnostics. An unavailable
optional membership does not prevent existing live-evidence supervision. These
legacy APIs retain their full evidence sweeps and synchronous storage latency;
only the bounded machine-agent path isolates membership work in background threads.

Complete old-writer capture and historical identities with no known direct locator
require the qualification below before primary discovery activation. Bare membership
or an advisory backfill EOF does not establish complete local obligation coverage.

Legacy evidence import captures an immutable absolute `legacy_source` in the
Attempt's membership before copying or unlinking a source record. Capture shares
the bounded redo transaction with publication and the evidence-write guard with
cleanup. A canonical Attempt ID supplies its exact Task/number locator; an
unclassified identity retains unknown fields as null. Later evidence or matching
authoritative Attempt truth may fill unknown fields while the membership is
active, but cannot change known identity or replace the source. Exact retries
write nothing. Cleanup handoff freezes both locator and source.

The importer syncs destination content, its directory and its ancestor namespace
before durably unlinking the source. Interrupted directory barriers preserve the
source, and retries repeat those barriers even when the target already exists.
Evidence targets, event targets and reservation release receipts must be regular
files reached through real directories within their owned runtime root. Symlinked
leaves or ancestors cannot prove a retained copy and prevent source retirement.
Late inbox conflicts preserve both copies and responsibility. Once a cleanup
receipt owns the Attempt, import leaves source evidence for that owner and does
not recreate target evidence. Existing initial-migration destination precedence
applies only to matching identities: an existing target's Attempt, Task and
explicit Attempt number must agree with the source's known fields and canonical
path identity. Both files must contain the lane's expected object envelope;
missing, wrong-lane or malformed envelopes are rejected. Conflicts retain both
records without publishing a new cleanup owner for the conflicting path.
Matching target evidence may fill an unknown
locator field. Late-record conflict behavior remains in force.

While writer capture retains a legacy inbox, each background membership read
refreshes the three direct launch-intent, registration and exit-observation paths
for that identity. It copies missing records after confirming durable source
membership; it never enumerates or deletes the source. Exact copies produce no
writes. Conflicting records remain unresolved, and a cleanup receipt prohibits
refresh from recreating evidence. The short active observation lock excludes
refresh; a durable pending capture alone permits it. Read failures retain the
membership and do not suppress other candidates.

Machine dispatch defers legacy inbox moves and destructive cleanup on capture
lock contention, recording `legacy_inbox.capture_deferred` or
`cleanup.capture_deferred`, while continuing compatible scheduling. Other import
or maintenance errors retain their existing failure behavior. The background
refresh lets a captured old runner finish through the original Attempt while
source retention still prevents destructive import. It does not prove that every
old writer was captured or permit primary discovery activation.

Cleanup checks known source writers before handoff and durably cleans the source
before the destination. One slice processes at most one populated root and eight
termination decisions; a failed barrier retains the receipt and target evidence.
Replay retains the immutable source after source files disappear. Binding removal
checks both CPU and GPU reservations and refuses to discard remaining membership
or an unavailable membership store; membership presence uses only the fixed
sixteen bucket headers. This does not certify an empty source before initial
capture or cover every delayed writer. Local inventory backfill is described below;
its completion is not a writer-completeness certificate.
Before qualification, legacy import retains its existing directory scans. After
qualification, routine dispatch leaves Attempt inboxes to direct membership refresh
and transports only the pending event outbox. Ancestor barriers add import-only
write cost beyond the membership transaction counts below.

Binding removal and on-demand idle exit share local recovery discovery. Incomplete
or unreadable legacy migration records, uncaptured source evidence/reservations,
target evidence, and remaining or unavailable membership retain responsibility.
Pending local diagnostic events retain their outbox responsibility until shared
publication. Late legacy events join the imported inbox and preserve their source
until the destination path and its ancestors are durable.
Idle checks stop at the first blocker; diagnostic binding removal enumerates the
blockers. Existing reservation snapshots already include CPU and GPU occupancy.
Before qualification these checks scan evidence directories. Qualified on-demand
idle uses the exact binding/source capture proof and sixteen Ledger headers for
Attempt obligations, including both active and maintenance members. Invalid
capture, owner/generation/source drift or a lost shared capability blocks idle.
These read-only checks perform no completion-directory barrier; they neither
grant execution authority nor delete evidence. Binding-removal diagnostics retain
the explicit evidence inventory.

Capture does not own event outboxes or legacy CPU/GPU reservations. Qualified idle
continues to check those pending directories, as well as machine reservations and
existing pending operation/demand gates. An unreleased legacy source hold also
blocks exit. An absent source hold proves release only with the matching
digest-bound release receipt; missing or damaged receipt metadata blocks idle.
Routine qualified inbox service probes the pending event outbox before
entering destructive import, so an empty outbox causes no capture synchronization.
Late events retain the existing destination/ancestor durability and source deletion
protocol. Late Attempt records remain in their retained inbox and are discovered
through each captured identity; indexed source cleanup owns their final deletion.

Explicit legacy-project migration transfers active and provisional reservations
in both capacity domains. It acquires source then target locks within one domain
at a time; GPU and CPU locks are never nested. It rejects conflicting source
ownership and target identities before modifying records in that domain. CPU
import preserves the target machine's configured capacity; imported existing
occupancy may exceed that capacity and prevents additional admission until enough
slots become available. It never increases capacity or terminates imported work.

Destination copies and their ancestor namespaces become durable before source
unlink. Source directory barriers run even when a retry finds no remaining files.
A matching destination release record owns a partially imported reservation;
retry validates its unchanged identity/content and syncs it instead of recreating
active occupancy. An identical leftover target occupancy file from interrupted
release is deleted and its directory synced before the source copy is removed;
failed retirement barriers therefore retain the source for retry. Conflicts retain
the source and keep migration disabled. If CPU
import fails after GPU transfer, transferred GPU records remain held at the
destination and retry continues with the retained CPU source. This is resumable
explicit reservation transport, not complete historical membership backfill.

## Restartable local inventory capture

### Pre-observation retention checkpoint

`WriterCaptureCheckpoint.observe()` durably creates
`responsibility-writer-capture.json` before admitting any caller observation.
The checkpoint binds a capture ID to the canonical runtime root and exact Ledger
instance. It remains pending when the scope exits, including when no writer was
found. Returning from an observation is not completion of the inventory.

Each `record()` batch contains at most 64 exact identity/locator/writer records
and is written to the checkpoint before membership publication. Reopen replays
the entire retained batch idempotently, including writers whose processes have
since exited. The batch is cleared only after every membership transaction has
completed. A failed or uncertain write invalidates the active observation scope;
the caller must reopen and replay before submitting another batch. Reopen retries
the checkpoint directory barrier before admitting new observations. A missing or
replaced Ledger never initializes a replacement through this API.

The checkpoint guard uses a read-only descriptor for each runtime root's parent
directory and nonblocking `flock`. The parent inode survives partition removal;
acquisition neither creates lock files nor recreates a removed runtime root.
Observation/batch processing and binding removal acquire it exclusively, while
evidence cleanup and import acquire it shared. Sibling roots therefore share this
brief exclusion boundary. Legacy operations acquire distinct parents in canonical
path order. Its durable pending state prohibits current evidence
cleanup, previously handed-off cleanup replay, legacy source import/deletion,
Task cleanup acknowledgement even without visible evidence, shared finalization
of an already acknowledged cleanup, and binding removal.
Both source and destination roots are guarded for legacy moves; direct source
cleanup also consults the source checkpoint. Invalid or dangling checkpoint files
retain the same prohibition. Read-only discovery and runner final writes continue.
Task cleanup holds the shared guard through acknowledgement publication and
shared finalization; a new observation cannot begin between those operations.
For a managed binding this includes the legacy root named by its durable migration
record, even after every imported membership has retired or the operation was
already acknowledged. It also includes the binding's authoritative target when a
caller supplies an old runtime configuration and an explicit machine reservation
backend. That target checkpoint retains cleanup during interrupted source-hold
establishment, before the source marker exists. Source provenance is recovered
from that retained record, not by scanning historical Attempts or extant memberships. An explicit managed
partition without its binding, or a present migration without a valid absolute source,
cannot proceed. An ordinary binding created without migration has no legacy root.
The parent guard also covers active-operation enumeration and its local cursor
writes. Binding validation precedes that enumeration, so a stale managed caller
cannot recreate a removed runtime partition. A busy root-removal or observation
scope rejects this metadata work; a durable pending checkpoint still permits
non-destructive progress reporting. This guard lasts for one bounded operation
selection/reconciliation call and adds no lock-file or snapshot writes.
Binding removal holds its exclusive guard through the blocker check and deletion,
excluding concurrent cleanup or import even after the runtime root is unlinked.

The caller establishes retention under runtime lifecycle ownership. Machine
enrollment holds the current registry and binding fences, which exclude binding
removal/replacement, and accepts only an immutable completed migration. Subsequent
bounded slices rely on the durable hold and nonblocking parent locks; they do not
hold the machine-wide migration or registry lock across process/history reads.
Checkpoint code may then take evidence, Attempt-control and Ledger bucket locks;
it must not acquire Task/authority or registry locks from inside the observation
scope. Existing Task cleanup may enter the shared checkpoint guard from its outer
Task fence. All checkpoint acquisition by cleanup is nonblocking.

This primitive establishes retention and replays observed batches. Production
enrollment now schedules it after the shared admission fence and commits the
separate completion certificate described below. The released-source qualification
tool also establishes this checkpoint before its process census. The bounded process sweep below captures supported
runner commands; a fresh final source/evidence sweep, verified old-writer admission
fence and coordinated activation remain required. The optional post-process
backfill below supplies that new evidence traversal, without certifying writer
admission or activating primary discovery. No observation EOF or empty
pending batch permits removing the checkpoint or dropping the existing discovery paths.

### Machine registration preparation

`MachineRuntime.prepare_recovery_registration` is an enrollment primitive for
the process holding machine scheduler authority. It updates the current owned,
eligible registration to version/protocol version 2 with
`recovery_protocol = qexp-local-responsibility-v1`, preserving its generation,
runtime identity and binding enablement. Scheduler authority cannot be released
while a preparation publication is in progress. A forked child cannot reuse its
parent's process-local authority proof.

Ordinary preparation and dispatch take the shared side of the machine migration
fence; actual migration and removal retain its exclusive side. Scheduler authority
still serializes dispatch, and preparation holds the registry lock and existing
registration/machine fences. Ordinary preparation therefore need not wait for an
entire multi-project dispatch cycle. An interrupted registration transaction uses
the exclusive migration fence before rollback; the shared fence is released first,
never upgraded in place. The primitive defaults to nonblocking local locks; a
background caller can wait for migration or registration to finish. Stale ownership, legacy machine metadata and incomplete legacy migration
defer preparation. Any interrupted registration transaction is replayed first; its
retirement directory is synced before publishing version 2 so a reboot cannot
restore an old version-1 rollback snapshot behind the new admission fence.
Publication uses the normal atomic file and directory durability barriers.
An already visible version-2 record is directory-synced before preparation
returns success, including retry after an uncertain publication barrier.

The target permanently accepts versions 1 and 2; version 2 additionally requires
the exact recovery protocol. Ordinary new registrations still start at version 1.
Renewal, reactivation, repeated registration and explicit replacement of a
version-2 logical name preserve the protocol floor. Matching prepared state is
not republished, apart from an independently due eligibility renewal.

Released versions 1.3.17 and 1.3.18 reject version 2 under their registration
write fence, including a dispatcher that cached eligibility before preparation.
A version-1 peer on the same project remains eligible before shared capability
activation. Registration alone does not fence a passive old runner: qualification
confirms it can still reach the process factory. This primitive does not install
a shared capability, certify captured writers, release source retention, or
enable sole-index discovery. The remaining shared admission/capture protocol is
still required before cutover.
After preparation, downgrading the machine agent to either released source
version cannot restore its write eligibility. Keep a version-2-capable agent;
lowering the persisted version or deleting registration state is not a supported
rollback. Existing old runner evidence production is still retained.

The global machine agent automatically prepares all locally registered bindings
while holding scheduler authority. A demand-started background worker visits at
most four projects per pass and owns its scheduling independently of foreground
dispatch. Each capture step returns `advanced`, `waiting`, or `complete`.
Productive passes yield for 50 milliseconds before continuing; an external wait
backs off only that binding for one second. Waiting bindings cannot delay useful
work on other bindings. New bindings
receive one registration preparation attempt before capture work can delay other
unvisited bindings. An inaccessible binding does not prevent prepared projects
from later advancing capture. The worker performs shared-root I/O separately from foreground polling and the
heartbeat/authority control loops. Foreground polling reads local registry state
and an immutable completion snapshot; only the worker owns capture handles,
round-robin position, and retry deadlines. Failures remain pending and retry fairly; disabled bindings can
prepare without becoming enabled. Superseded local generations stop retrying
without advertising recovery readiness. Exact successful bindings are cached for
the agent lifetime; changed bindings are revalidated, and ordinary restart
rechecks durable registration and capture state. When capture and source retention
are settled, no worker or timer remains. Pending enrollment prevents on-demand
idle exit between batches.
Stopping the agent prevents the worker from starting another project and retains
scheduler ownership until the entire service thread has stopped, including
intermediate capture I/O outside the short publication fences. A join timeout may
not release authority to a successor while the old service still mutates capture.

The per-project eligibility object in agent status includes `registration_version`
and `recovery_protocol`. These describe admission preparation, not process census
completion or discovery activation. No per-project operator activation step is
introduced. After registration preparation, the worker attempts the shared
admission fence below, then advances retained process/evidence capture and
source-retention release without a per-project operator command.

### Shared admission fence

`local-recovery-v1` is a required root capability. The target understands it;
released 1.3.17/1.3.18 readers reject it. The machine-owned enrollment worker adds
it only after every machine directory contains matching machine metadata and a
valid registration with either the prepared version-2 protocol or explicit
supersession. An expired version-1 registration still blocks activation. Neither
missing heartbeat nor advertised heartbeat capabilities prove preparation. A
missing, malformed or legacy participant retains the waiting state. Waiting does
not disable compatible scheduling, and another registered project can advance.

Activation holds current scheduler ownership, the local registry lock and the
existing shared registration/machine fence. It then tries the exclusive schema
lock nonblockingly. This respects the registered claim path's outer registration
fence; schema contention releases the fences and defers the attempt rather than
waiting with an inverted lock order. Binding identity and local write eligibility
are revalidated. The existing upgrade metadata must be ready, with no pending or
blocked coordinator operation. The transition reads machine metadata, never
Task/Attempt or local evidence history; its one-time work scales with the number
of project machine participants.

Every accepted participant registration directory is synced before the capability
publication. This completes a peer's potentially uncertain registration or
supersession rename barrier. The schema update uses atomic file replacement and
file/directory sync. A retry that sees the capability completes its schema
directory barrier before reporting success, without enumerating participants.
The target's upgrade-journal metadata remains valid: its terminal manifest binds
the base schema, excluding only this independently fenced capability. Audit and
repair-plan evidence still bind the complete schema record, and unrelated schema
changes remain drift.

Agent status includes a read-only `recovery_enrollment` snapshot with `waiting`,
`admission_fenced`, `superseded` or `unavailable` state and specific blockers. It is
diagnostic, not a capture or activation certificate. The background worker caches
successful admission fencing for the exact local binding and otherwise retries
in its existing bounded project batches.

This capability fences old claim and runner admission, including an old runner
that begins after the boundary. It does not certify an inventory of pre-boundary
processes, fence every cached old Task mutation API, or authorize removal of
history discovery. Existing old runners may still publish final evidence. Their
capture, fresh retained evidence sweep, local activation certificate and retained
source cleanup remain separate requirements. Removing this capability or
downgrading participants to readers that reject it is not a supported rollback.

### Automatic retained capture completion

After shared admission is fenced, each enrollment pass visits at most 64 process
entries and spends at most 64 evidence work units per selected project. Empty
lanes consume one unit; visited entries and processed records each stay within
that budget. Evidence lanes may share one retention/lock scope within a slice,
avoiding repeated contention between empty lanes. The advisory membership reader
stops its ordinary backfill once retained capture owns the work. Existing authority
evidence lanes remain enabled until the qualification described below succeeds.

Initial retention and completion publication require current scheduler PID,
registry membership, eligible prepared registration and the shared capability.
The registry excludes binding removal; completed legacy migration metadata supplies
the exact source and must match project, machine and shared-root identity. Missing
migration means no legacy source; present malformed metadata is unavailable, never
silently interpreted as absence. Intermediate discovery does not hold machine-wide
migration/registry locks. Changes of binding or source prevent final publication.

First admission atomically stamps the writer checkpoint with the exact owning
binding, including registration generation, and resets process-sweep progress at
a strictly newer revision. Previously pending writers replay before this reset;
the revision also forces a fresh evidence sweep after replaying any prior pending
evidence batch. Completed prototype checkpoints without this stamp cannot satisfy
post-fence capture. An uncertain admission publication resumes its stamped progress
after the directory barrier rather than starting another sweep.

Completion is persisted as `responsibility-capture-complete.json`, format
`qexp-local-capture-complete-v1`, capped at 16 KiB when read. It binds the local
runtime, Ledger instance, project/machine/shared root, owning MachineRuntime
identity, registration generation at publication, immutable legacy source, and
SHA-256 digests of the completed process and fresh evidence checkpoints. Publication
holds their retention/checkpoint locks and requires no pending batches, the final
evidence lane and the exact completed process revision. These digests detect
checkpoint drift; they are not signatures against a malicious filesystem owner.

Reading completion verifies fixed metadata and finishes a possibly uncertain
parent-directory barrier. Restart does not enumerate processes, history or members.
A same-host reboot preserves completed capture; another host or a changed PID
namespace within the same boot is unavailable. A malformed, missing or replaced
checkpoint/Ledger cannot release cleanup. Completed process capture cannot reopen
or mutate the checkpoints bound by its certificate. Read-only status omits the
barrier and remains advisory.
Reusing completion requires the exact registration generation. Returning to the
same runtime identity after another generation owned the logical name does not
reuse the earlier coverage. The current scheduler validates its new registration
and shared admission fence, then starts a fresh bounded capture automatically.
An unfinished capture first replays pending observations and atomically resets its
admission stamp and process revision. A scanner retaining an earlier admission
stamp cannot continue its old iterator or certify a sweep in the new generation.

For completed capture, `responsibility-capture-generation.json` records a bounded
replay intent containing the new owner stamp, old completion/checkpoint, and any
valid old source-release receipt, capped at 1,081,344 bytes on both write and read.
Its presence makes completion unavailable and
retains cleanup. Under both parent guards, recovery retains or reacquires and syncs
the exact source hold before publishing this target-side intent. This source hold
uses phase `generation_pending` and the new admission stamp, so the old completion
cannot release it. A crash before intent publication leaves conservative retention,
preventing standalone cleanup from removing the source needed for retry. Recovery
deletes old completion and release metadata, atomically resets the
checkpoint at the next process revision, restores the normal pending source hold,
then removes the intent. Every transition
finishes its directory barrier. Missing unreleased source retention cannot be
recreated; foreign source ownership, replaced Ledger/checkpoints, and different
persistent binding ownership fail closed. Existing memberships remain intact.
An already released source may be retained again only from its validated release
receipt. A crash resumes the exact recorded revision. If registration advances
again while recovery is pending, the current owner finishes that local reset before
resetting the now-unfinished capture for its latest generation. Neither intermediate
state grants discovery coverage or authority. Source release revalidates completion
after acquiring its parent guards, excluding a race with generation replacement.

The target certificate permits normal proof-based per-Attempt cleanup. A legacy
source hold remains pending: direct standalone cleanup and whole-root deletion
cannot bypass it. The exact owning target may clean captured source evidence under
both parent guards, after the existing receipt and writer-quiescence checks. A live
captured writer still prevents handoff even if it has not written any file. After
all Ledger memberships retire, enrollment durably publishes a local
`responsibility-source-release.json` receipt binding the completion digest and
source, then removes and directory-syncs the source hold. Later target-only launches
do not reopen source ownership. A missing hold before that proof cannot settle
unresolved memberships. An uncertain unlink is retried by syncing the source directory. Target
binding removal continues to require its ordinary reservation/evidence/Ledger
blockers to be clear.
If only capture is pending and a scheduler is still running, removal releases its
lifecycle locks and retries for at most 30 seconds so normal background completion
can finish. Active evidence fails removal immediately. A stopped scheduler or a
capture that does not finish within that bound leaves the binding and its evidence
intact.

Agent status exposes `recovery_capture` as `not_started`, `capturing_processes`,
`capturing_evidence`, `captured_source_retained`, `captured` or `unavailable`, with
`diagnostic_only=true`. Neither this status nor the capture certificate means the
supervisor's initial active traversal or reservation reconciliation has completed.
Primary discovery and old-lane removal therefore require their own guarded
activation; full-machine scale evidence is recorded separately below.

### Bounded runner process capture

`RunnerProcessCapture.take(limit)` visits at most 1–64 raw `/proc` directory
entries per slice, including non-PID entries. It considers the current user's
published `python -m qqtools.plugins.qexp.runner` command shape, matches project,
machine and current/declared legacy runtime, and skips the guardian command, which
does not publish runner evidence. It does not claim to discover arbitrary embedded
Python callers. PID/start ticks are read before and after the command; reused,
departed and zombie processes are not captured as the observed live writer.
Access errors, malformed matching commands or unreadable identity prevent progress
without removing retention. Command reads are capped at 64 KiB plus the overflow
probe and stat reads at 8 KiB plus the probe. Large unrelated commands are ignored;
a truncated runner command cannot be accepted. No shared Task/Attempt history,
process launch or process signal is used by this API.

The pending checkpoint optionally declares one immutable canonical `legacy_source`.
Before any process observation, capture locks distinct target/source parents in
canonical order and persists a source hold at the same checkpoint filename. Its
format is `qexp-pending-writer-source-v1`; it binds source root, target root, exact
target Ledger instance and capture ID. It does not create a second Ledger in the
source runtime. An existing hold owned by another capture is never overwritten.
Partial hold establishment admits no observation and is retried before the next
slice. Once progress or a pending observation batch exists, a missing source hold
is an unavailable capture; it cannot be silently recreated while reusing the old
sweep or replaying its batch. Both target and source remain protected between
slices and after EOF.

Observed legacy writers carry the declared `legacy_source` in their pending batch.
One Ledger transaction publishes both the process identity and source locator;
the source mapping cannot be lost between two membership commits. Exact retries
do not rewrite the member. Optional producer `progress` is journaled together with
the observations and becomes visible to the next observer only after pending
memberships replay. The process producer uses `qexp-runner-process-sweep-v1`, exact
project/machine/host/boot/PID-namespace context, monotonic revision, and
`is_sweep_complete`. Changed context or malformed progress cannot reuse a completed
sweep. The lifecycle owner may call `restart_after_reboot()` to replay the durable
pending writer batch and begin another process sweep on a new boot of the same
host. It preserves the enrollment ID, membership and source hold, advances the
process revision and clears sweep completion in one checkpoint publication.
An unchanged scope needs no checkpoint write. A foreign host, malformed identity
or changed PID namespace within the same boot remains unavailable. This boundary
neither activates discovery nor releases retained evidence.

The live `/proc` iterator is process-local. After interruption an unfinished sweep
replays its exact retained batch and starts a fresh iterator; it does not claim a
durable directory seek or avoid re-reading the interrupted prefix. Journal changes
from another scanner do not reset a surviving iterator, so alternating scanners
can each finish a finite pass. Reopen of a completed sweep performs no new process
enumeration or journal write, while retrying the retention directory barriers.
Each slice may replay at most 64 prior observations before visiting at most `limit`
new entries. The returned page counts describe new visits/publications, excluding
that replay. These are work bounds, not a wall-clock guarantee for filesystem I/O.

An empty process page commits one progress update; a page with observations commits
pending intent and its clearance around membership transactions. Source retention
adds one initial file publication and a directory barrier on reopen, not one source
Ledger or a file per PID. Measurement belongs to `scripts/qualification/`, outside
production. The enrollment worker schedules this producer after admission
fencing and preserves compatible service while capture holds its per-slice parent locks.

### Captured process identities

An active membership may retain `captured_writers`, a bounded list of observed
Linux process identities. Each contains the existing machine-id based `host_id`,
a canonical `boot_id`, positive `pid_namespace` inode and `pid`, and nonnegative
`start_time_ticks`. The namespace
is the PID view in which the PID was observed. At most 64 writers fit one entry;
an overflow or encoding error fails capture without discarding prior writers.
This limit does not authorize treating an overflowing inventory as complete.
A deterministic refusal first persists a sticky `writer_capture_incomplete`
marker that forbids cleanup handoff. If there is no existing entry or it cannot
fit the marker, the fixed bucket header retains the marker and conservatively
blocks handoff for that bucket. Repeated refusals do not rewrite an existing
marker, and a later successful individual capture cannot clear it. Clearing
incomplete capture requires a separately qualified rebuild/capture protocol;
no automatic clearance or primary activation is implemented here.

This refusal marker is not a replacement for the migration's durable pending
checkpoint established before enumerating processes. Storage failure or a crash
before any capture intent is durable cannot certify the newly observed writer;
the pending migration must retain discovery and replay responsibility.

`capture_local_writer` serializes publication with the evidence-cleanup guard.
The existing bucket redo transaction atomically creates or extends membership
and its known Task/Attempt locator. Exact retries perform no writes, conflicting
locators are rejected, and a new writer cannot be added after maintenance handoff.
Multiple observed writers remain recorded independently. Attaching one writer to
an existing member costs three file and three directory fsync calls; it is a
capture transition, not a lease-renewal write.

Before publishing cleanup proof, the worker checks every captured writer even
if the original intent, registration and process files have disappeared. A
different valid boot ID on the same verified host, or an exited/reused PID in the
same boot and namespace, proves that writer cannot publish again. A namespace
mismatch on the same boot, different/unverifiable host, malformed identity or
inaccessible process/boot metadata retains active responsibility.
Existing terminal/accounting proof is still required; process
exit alone never authorizes Task finalization or resource release.

This primitive records known writers and is exercised by the released-runner
qualification tool. It does not run a process census in normal supervision,
certify a complete writer inventory, or activate sole-index discovery. Fenced,
restartable process enumeration and all-participant rolling activation remain
required before removing existing discovery paths.

### Evidence directory capture

The existing background responsibility reader performs one capture slice before
its membership read until the local sweep completes. Each slice visits at most
64 directory entries (including unrelated files and nested directories) and
processes at most 64 selected evidence records. It uses the same four-worker
bound as discovery. A failed capture is reported without suppressing already
indexed responsibilities; existing evidence discovery remains enabled.

`runtime_root/responsibility-backfill.json` records format
`qexp-local-responsibility-backfill-v1`, the exact ledger instance, capture ID,
checkpoint revision, current lane, a pending batch of relative paths and whether
that lane reached EOF. The fixed lane order is process manifests, registrations,
exit observations, launch intents, termination decisions, wrappers and authority
diagnostics. The checkpoint is capped at 1 MiB and 64 pending paths. Unrecognized
formats, invalid paths and replaced/missing ledgers fail closed rather than
reinitializing a store behind an existing checkpoint.

Capture persists its pending batch before updating membership. It reads and
validates each record under the per-Attempt evidence guard, merges only unknown
locator fields, and durably publishes membership before clearing the batch or
advancing to the next lane. Existing maintenance receipts and known identity
fields cannot be overwritten. Source evidence and Task/Attempt truth are never
deleted or rewritten by capture, and it never launches work.

Restart first replays the pending batch, then rescans the unfinished lane because
filesystem directory cursors are not durable. Completed lanes remain checkpointed;
exact membership retries perform no writes. This is resumable capture, not a
persistent directory seek cursor: repeated interruptions can repeat enumeration
of the unfinished lane. Another scanner's checkpoint revision does not reset a
surviving iterator in the same capture/lane; alternating instances can finish
their finite passes without starving each other. The only complete-sweep result means that this capture
visited its lanes; it is never sufficient to disable old discovery. Later writes
behind the sweep require publication coverage or a new fenced capture before
activation. Writer fencing, arbitrary-corruption rebuild and physical separation
of active versus maintenance traversal remain separate requirements.

An explicitly supplied `process_capture` selects a fresh, retained evidence sweep.
Each slice enters `RunnerProcessCapture.completed_sweep()` to replay pending
writers, validate the original host/boot/namespace and completed process progress,
and hold target/source retention before evidence reads. An incomplete process
sweep cannot create the evidence checkpoint. The new checkpoint is
`responsibility-capture-backfill.json`; it uses the same bounded batch protocol,
binds the exact writer capture ID and ordered roots, and cannot reuse an earlier
ordinary backfill's completion. Its lanes cover all seven evidence kinds in the
target, then all seven in the declared legacy source, when present. A missing
record from a retained pending batch is an error and retains that batch; it is
not treated as a successful cleanup.

The evidence checkpoint also records `writer_sweep_revision`, binding its scan to
the completed process sweep. After a reboot restarts that sweep, evidence capture
waits for its new completion. Any old pending evidence batch is replayed first,
within the ordinary page limit; only then may a new evidence capture ID and lane
zero replace the old progress. A crash during either replay or replacement keeps
all responsibility discoverable. Existing v1 evidence checkpoints without this
field receive the same conservative fresh sweep after pending replay. Invalid or
future revisions are unavailable rather than reset. Once the matching evidence
sweep completes, reopening again performs no enumeration or checkpoint write.

All records publish into the target Ledger. A source record atomically retains
its legacy root with the recovery locator; source files are not moved or deleted
and no source Ledger is created. This includes terminal evidence written after
the process sweep, when an ordinary backfill may already have completed. The
returned progress includes the total number of lanes. Completed reopen performs
no evidence traversal or checkpoint write, but still validates enrollment and
retention. Capture ID/source changes cannot adopt the old checkpoint.
The enrollment worker now schedules this API after the shared admission fence.
The sweep alone does not prove admission fencing or authorize sole-index discovery;
the coordinated completion protocol binds its ordering and exact checkpoints.

The backfill format is additive and local. The automatic enrollment protocol
uses the separately qualified shared admission fence above; it does not change
machine ownership or require a new operator command. Already-running old runners
continue writing the same evidence paths; they need not publish entries.
No temporary reader adapter or compatibility lifecycle is added. Existing
discovery remains an independently correct path rather than interpreting old
writers as migrated. Initial publication uses a private fixed sibling directory,
`.recovery-responsibilities.building`, under the existing initialization lock.
Interrupted creation resumes only that bounded empty layout. The fully initialized
directory is renamed into place; an `initializing` record remains until the parent
directory barrier succeeds. Retry finishes that barrier before acknowledging new
publication. An existing published store without a valid marker remains unavailable;
normal launch and recovery still use the original protocol. Initialization never
erases occupied buckets or reinterprets an unknown published directory as empty.

Terminal cleanup resolves a canonical historical Attempt directly from its encoded
number, including after retry clears the current Attempt reference or a successor
has claimed the Task. It validates the exact Task/Attempt/number, terminal phase,
claim archive and reservation disposition. A successor's reservation is not the
old Attempt's accounting obligation. Noncanonical historical identities without a
captured locator remain unresolved. Missing Task truth requires the separate
cleanup proof above; mere absence is not a cleanup authorization.

The bounded authority cutover below removes normal flat evidence scans. Qualified
outage observation uses direct responsibility membership. Group cancellation and
Worker removal use the certified Group membership/change protocol documented in
the runtime specification. Diagnostic binding removal retains its explicit
inventory; routine qualified idle and legacy Attempt inbox service use the
coverage described above. Synchronous unbounded supervisor APIs retain their full
scans. Whole-machine history qualification and released-writer lifecycle evidence
are recorded in the delivery acceptance report rather than inferred from this
storage protocol alone.

## Qualified bounded authority discovery

The machine control plane supplies the current binding's persistent owner and
registration generation. Before indexed authority service starts, a background
worker validates the fixed-size completion, process/evidence checkpoints, Ledger
instance and shared `local-recovery-v1` capability. A first successful qualification
finishes the completion directory barrier; unchanged qualification performs no
write. No history enumeration is needed to validate this metadata.

Without a completed capture, ordinary evidence discovery remains active. An
invalid proof, replaced Ledger, pending generation transition or lost capability
reports unavailable and prevents startup admission. If a previously qualified
proof disappears, that reader stays unavailable until revalidated; it cannot
reinterpret missing proof as a successful empty legacy startup. Existing fences
still authorize each mutation, independently of discovery qualification.

A qualified supervisor serves the active membership traversal and direct evidence
paths instead of the five flat authority lanes and six cleanup-directory scans.
Half its semantic turns remain reserved for cached active Attempts; the other
half alternates membership and termination service. Empty active turns can service
membership. The existing 256-entry active cache and 64-candidate prefetch bounds
remain. Index and qualification I/O never block the supervision thread or close.

Initial readiness requires all sixteen active buckets to finish, every prefetched
initial candidate to receive a successful foreground acknowledgement, and pending
control work to finish. Background completion flags cannot acknowledge unpublished
or unconsumed candidates. Newly materialized manifests are supervised before their
candidate is acknowledged. Active discovery or semantic failures reset the initial
coverage epoch and require a fresh traversal. While a control is pending, primary
membership service leaves prefetched candidates buffered until a slot is free;
it never repeatedly consumes and restarts them behind the same long control.
Cached short supervision can still finish without retaining a second control.
An occupied slot cannot acknowledge another unfinished long control. Unsuccessful cancellation or
a later generator failure also resets coverage and clears that Attempt's service
throttle. Only successful control completion preserves acknowledgement.
Maintenance failures do not reset successful active coverage. Initial traversal
visits up to four active buckets with at most 16 members each per background read.
This keeps the 64-member budget while avoiding sixteen foreground cycle waits for
an empty project's initial coverage. Afterward independent active/maintenance
cursors receive 48/16 candidates; the active budget spans four buckets per read.

Termination-only members directly enqueue their own decision directory; primary
service never enumerates the outer termination root. One current directory and
at most 256 queued directories are retained. The initial barrier also waits for
successful EOF of each initial directory; signalling/read failures restart that
cursor and retain the barrier. Queue overflow is a retryable failed candidate,
not a silently discarded directory. Later arrivals do not indefinitely extend an
already completed initial barrier.

The control plane revokes its ready-generation marker whenever startup readiness
is false or authority service fails; an unavailable registry revokes all ready
generations while preserving supervision state. Certificate presence is only a cheap hint
for entering supervision for a disabled/ineligible binding; it does not authorize
admission or skip actual qualification. Reservations and shared authority remain
protected by their existing reconciliation and write guards.

## Storage and finite service traversal

Sixteen hash buckets contain dense pages of 64 identity hashes and direct entry
locators. Each bucket retains all-member `pN` pages and separate active `aN` and
maintenance `mN` pages. Locators carry both the all-member slot and their stage
slot. Publication appends to the active index; handoff removes that active slot and
appends to maintenance; retirement removes both all-member and maintenance slots.
Each index independently moves its tail into the vacated slot and removes empty
tail pages. Both compactions and locator updates share one redo transaction;
if the same locator moves in both indexes, both slot changes survive. Physical
names are reused; normal traversal never enumerates the data
directory. Byte limits are 8 KiB/header, 64 KiB/page or entry, 60 KiB/initial entry,
and 512 KiB/transaction. These constrain the optional index, not qexp's accepted
identifier contract; an unrepresentable entry uses the existing discovery path.

The service cursor scans from the initial tail toward slot zero. Its upper bound
only decreases; later appends never extend that sweep. Compaction moves a retained
member only toward a smaller slot. Thus an unvisited retained member stays ahead
of the reverse cursor until selected. Every continuously retained member present
at sweep start is visited before that finite sweep ends, even if every slice is
followed by mutations. Previously visited members can move ahead of the cursor
and be selected again; consumers must reconcile idempotently. Members added after
a sweep starts are guaranteed service by a later sweep if they remain present.
Buckets take independent round-robin turns, including after a bucket fails.

Stage cursors are bound to their stage as well as ledger instance and bucket.
Active reads access only active pages and their locators, never maintenance or
all-member pages. The finite-sweep guarantee applies to membership continuously
retained in that stage; handoff durably transfers ownership to the other stage.

The v1 base layout remains supported without stage projections. Optional header
`stages` metadata records projection version 1, both counts and a decreasing
`before` build boundary. A bounded reverse base-page sweep constructs missing
stage slots without changing membership generations or cleanup receipts. Each
entry and cursor advance commit together; a crash may replay but cannot skip the
captured member. Concurrent publication/handoff creates stage slots, while
compaction only moves old members toward the remaining build interval. Stage reads
fail unavailable until both stage counts sum to the base count. Complete builds
perform no further writes. This is permanent support for a base-only v1 store,
not a license to discard corrupt projections or a mixed-writer activation fence.

The strict revision-checked `page` API remains available for static inspection;
it is not used for service scheduling. Cross-bucket snapshot semantics are not
provided. The progress proof concerns selected records under successful storage
operations, not an I/O latency guarantee or recovery from arbitrary corruption.

## Crash protocol and locks

A bucket `flock` is a leaf lock. Store methods never acquire Task, Group,
reservation, Attempt-control or machine authority locks while holding it.
Readers finish pending redo under the same bucket fence before returning pages.

Background cleanup first tries the per-Attempt evidence-write guard, then the
existing Attempt-control lock, without waiting on either. Manifest and intent
materialization use the same evidence guard before reading their source record,
so concurrent deletion cannot resurrect a manifest from a cached registration.
The lock order is evidence guard -> Attempt control -> membership bucket. No
shared authority locks are acquired by cleanup workers. Normal authority work
defers candidates with an outstanding cleanup request while continuing other lanes.

1. Write and file-sync a checksum-protected pending transaction, rename it, and
   directory-sync it before changing any live record.
2. File-sync and rename all non-deleted after-images; perform requested unlinks;
   replace the header last. One directory fsync then commits the whole namespace.
3. Unlink the pending transaction and directory-sync the removal.

The directory barrier does not imply atomic multi-file rename: before it, any
subset of after-images can survive, including the new header alone. The already
durable pending transaction repairs every such combination. Recovery accepts
only the transaction's base or resulting header revision and rejects applying an
old transaction over a newer generation. An unsynced pending unlink can reappear
after a crash and replay the same committed transaction safely.

Publish, including initial source capture, costs eight fsync calls. Attaching a
source to existing membership or filling its unknown locator costs six; exact
capture/resolution retries write nothing. Active-to-maintenance handoff costs seven
to ten, and retirement five to eleven, depending on page/entry changes. A transaction
contains at most eight after-images within the unchanged 512 KiB limit. Restartable production initialization is separate:
it creates all fixed bucket metadata with 55 fsync calls, including the durable
rename completion record. The strict create-only experiment API retains its
52-call initialization. Quiet reads and exact publication retries perform zero writes.
No per-renewal or per-runner-event membership write is required.
Evidence deletion adds its own directory barriers; the membership-only figures
above are not the cost of a complete terminal cleanup.

## Verification

[Capture integration tests](../../tests/integration/qexp/test_recovery_capture.py)
assert that qualified idle and routine inbox service enumerate no Attempt evidence
and perform no unchanged completion sync. They retain active/maintenance members,
legacy CPU/GPU reservations and source/target events as blockers; reject damaged
proof, Ledger, capability, generation or migration source; preserve late direct
runner observations; and retry interrupted event destination barriers without loss.

[Primary discovery tests](../../tests/integration/qexp/test_primary_responsibility.py)
cover no flat authority scans on qualified startup/steady service, semantic
acknowledgement, uncollected worker completion, failed candidates, damaged or
missing qualification, maintenance failure isolation, direct termination and its
initial retry barrier, queue fairness, and nonblocking qualification/close.

[Storage tests](../../tests/integration/test_local_responsibility_storage.py)
exercise the same implementation through instrumentation retained outside `src/`:
finite traversal under appends and compaction, round-robin selection, real process
crash boundaries, repeated replay interruption, every mixture of persisted
after-images, a directory-durability loss model, concurrent publication,
generation checks and compact cleanup. Initialization tests interrupt a real
process at every instrumented persistence boundary and retry a failed root rename
barrier; unknown and occupied storage must remain unchanged.
Cleanup tests interrupt real processes at every instrumented storage boundary,
retry after the final deletion barrier fails, reject stale generation proofs,
and distinguish bare maintenance from a durable cleanup receipt.
[Integration tests](../../tests/integration/qexp/test_responsibility_discovery.py)
cover publication before runner creation, missing/corrupt-index fallback,
asynchronous retirement, delayed index I/O, unchanged startup completeness and
retention during partial cleanup, and historical terminal cleanup after retry with
or without a successor. They also cover live/unknown writers, partial cleanup
across reader restart, deleted Task proof, blocked cleanup without manifest
resurrection, and fair discovery despite repeated cleanup failure.
Unbounded-supervisor cases retain live-writer evidence, replay a standalone receipt
after final-barrier interruption and deletion of shared truth, reject bare
maintenance as proof, and complete terminal publication despite a corrupt index.
[Cleanup command tests](../../tests/integration/qexp/test_cleanup_responsibility.py)
cover delayed runner writes, retained proof and absent acknowledgements after
storage failures, bounded decision cleanup, noncanonical Attempt records,
mismatched membership rejection, and CPU reservation cleanup with Task and project
isolation. Active-operation discovery tests reject duplicate service through
stable aliases or interrupted-import copies.
[Legacy import tests](../../tests/integration/qexp/test_legacy_responsibility_import.py)
cover real process interruption during capture/copy/unlink, failed ancestor and
source-deletion barriers, late source writes, live source writers, conflicting
identities, immutable source/receipt handoff, and reservation/membership-only
binding removal blockers. Storage fault tests also exercise source attachment
and partial-locator resolution at every instrumented persistence boundary.
Actual hardware power loss and released old
writer migration are not established by these tests.

[Recovery retention tests](../../tests/integration/qexp/test_recovery_retention.py)
cover membership-only idle/disable behavior, unreadable stores, incomplete
migration before capture, uncaptured sources, CPU-only occupancy, and stopping idle
discovery at the first blocker.
Evidence import and retention checks distinguish absent lanes from inaccessible
or malformed lanes, including nested termination evidence. Enumeration and
metadata errors block migration and binding retirement; they do not certify an
empty responsibility set. These checks stream entries and close their directory
handles when idle detection stops at its first blocker. Symlinked evidence is
unavailable rather than silently omitted or followed outside the local lane.
[Reservation import tests](../../tests/integration/qexp/test_legacy_reservation_import.py)
cover both domains and phases, source locking, unchanged CPU policy, failed
directory barriers, real process crashes during copy/unlink, release-aware retry,
cross-phase identity conflicts, source ownership and partial cross-domain transfer.
Released-source retry also rejects a conflicting retained target before deletion.
Unknown reservation ownership blocks binding removal. Nested reservation files,
duplicate source IDs across phases and malformed target CPU occupancy block
migration before source deletion in the affected domain.

[Backfill tests](../../tests/integration/qexp/test_responsibility_backfill.py)
cover all seven lanes without shared Task truth, partial locator merging, immutable
maintenance ownership, pending-batch replay, completed-lane resume, bounded raw
enumeration, invalid records and replaced stores, worker discovery through capture
failure, and real process interruption after pending publication, membership
publication and lane checkpointing. Storage fault tests also exercise local
capture and locator completion at every transaction persistence boundary.

[Captured writer tests](../../tests/integration/qexp/test_captured_recovery_writers.py)
retain a real live process after all original evidence has disappeared, reject
ambiguous host/boot/namespace identity, and exercise PID reuse, overflow and
encoding refusal. Storage fault tests cover both member and bucket incomplete
markers at every transaction and replay boundary. Once the refusal intent is
durable, interrupted replay still prevents cleanup handoff; a crash before that
intent remains the responsibility of the pending migration protocol.

[Checkpoint tests](../../tests/integration/qexp/test_writer_capture_checkpoint.py)
interrupt real processes after initial retention, observation, batch journaling,
membership publication and batch clearance. Reopen retains cleanup exclusion and
replays the exact departed process identity. They also cover uncertain writes,
failed directory barriers, cleanup/capture lock exclusion, malformed checkpoints,
and source deletion through another runtime. Cleanup, retention and legacy-import
tests cover empty local acknowledgement, binding removal/idle retention and both
record and event moves during pending capture.

[Process capture tests](../../tests/integration/qexp/test_process_capture.py) cover
raw page limits, target/source mapping without shared truth, process identity
faults, alternating scanners, completed-sweep reopen and real process interruption
during target/source hold publication, observed-batch commit, member publication
and batch clearance. Reopen captures a journaled writer even after its `/proc`
entry disappears. Storage crash/replay matrices cover atomic writer/source
publication. The released-source probe runs actual v1.3.17/v1.3.18 wrappers whose
Task/Attempt/evidence were erased, independently enumerates them using this
producer into a different target runtime, and verifies continued single execution
and retained legacy inbox ownership. It does not certify all-writer activation.

[Profiling instructions](../../scripts/qualification/README.md) retain the measurement
and fault-injection tools outside production code. No new dependency is required.

[Active/history delivery acceptance](qexp_active_history_acceptance.md) records the
matched history matrix, active-load evidence and validated filesystem envelope.
