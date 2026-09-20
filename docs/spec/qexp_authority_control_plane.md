---
doc_type: spec
status: active
updated_at: 2026-09-18
archived_at:
---

# Machine authority control-plane scheduling

The machine agent uses resumable authority evidence lanes. Attempt identity,
registration-generation fencing, lease expiry, process identity, and durable
terminal truth retain the contracts in [qexp_runtime_spec.md](qexp_runtime_spec.md).
Validation methods, measured costs, and operating limits are recorded below.

## Ownership and admission

One authority thread owns a supervisor for each serviced binding generation.
Binding removal or generation replacement closes its advisory discovery cursors.
The existing shared-write locks and identity checks remain mandatory; cached
supervisors, active locators, and cursor positions never authorize writes.

Startup constructs the same lanes used during normal operation. Dispatch waits
for the first successful complete registration, process, observation, intent,
and termination sweeps of the current generation. Maintenance remains available
while startup recovery progresses. A malformed or unavailable record prevents
that lane from completing its initial successful sweep; other lanes and projects
continue. Local evidence and failed maintenance still prevent false idle exit.

Nested process, registration, launch-intent, termination, and capacity records
must be JSON objects before their fields are read. Wrong container types enter
the existing diagnostic/error paths, retain evidence and capacity, and do not
terminate service for healthy records in other lanes. Malformed machine capacity
snapshots produce a heartbeat `capacity_unavailable` observation; invalid fields
detected during publication produce `publication_unavailable`. Both retain their
files and retry next cycle rather than advertising empty capacity or
terminating the heartbeat thread. A malformed termination
record rejects the recovery check rather than proving absence of commitment.

The standalone supervisor's default complete-scan API remains available. The
machine control plane uses a 256-step startup slice, then `work_limit=64` after
the initial sweeps succeed; it does not synchronously drain
startup inventory before allowing other projects to run.

## Project service

Registry discovery reads machine-local registry data, without first reading
registration state from every shared project. Per-project shared status,
configuration, eligibility renewal, and supervision run inside that project's
service opportunity. Each authority cycle services at most 16 registered projects.
For larger stable registries, the next cycle starts after the selected group;
for smaller registries, the leading project rotates. Empty registries reset the
cursor; removal of the next project resumes at the current first project.

An eligible enabled binding needs no preliminary process inventory probe.
Ineligible bindings use resumable one-entry process probes before recovery or
reactivation. Disabled bindings use bounded local evidence probes so retained
execution can still drain. Reservation-only draining belongs to machine maintenance:
its disabled/draining selection and reservation reconciliation run before authority
readiness gates admission. Expired provisional and identity-verified orphaned
active reservations therefore drain even without local execution evidence or an
authority supervisor. Superseded registrations do not run a supervisor.
Write eligibility is revalidated independently of cached supervisor existence.
When no process manifest is found or reactivation is unavailable, a non-superseded
binding still reconciles up to eight local exit observations. This may release
identity-verified finished capacity from registration-only evidence; it retains
that evidence and does not grant shared write eligibility or startup readiness.

The registry's existing JSON record is parsed as a whole. Registry parsing,
local active-reservation snapshots, and dependent authority transactions are not
constant-size operations. This candidate does not introduce a registry format
change or claim that its entire I/O cost is independent of active project count.

## Evidence work and fairness

Each project slice executes at most 256 semantic steps during initial recovery,
then 64 during ordinary service. The larger startup slice amortizes per-visit
configuration/eligibility costs without removing the finite work cap. Generation
replacement restores the startup budget; each project sample reports `step_limit`.
Selection rotates across:

1. cached active supervision;
2. registration and process discovery;
3. exit observations and launch intents;
4. termination replay;
5. terminal-evidence cleanup.

The implementation uses eight lanes. Each fourteen-step round alternates seven active
turns with one turn each for registration, process, observation, intent,
termination, cleanup, and local responsibility discovery. Thus a 64-step slice provides 32 active service
opportunities and at least four opportunities for every other lane. A pending
locked inventory consumes one step first, leaving at least 31 active opportunities. An error consumes
its turn and advances discovery. If the active cache is empty, its turn is lent
to an unfinished startup lane in rotating order, or to the following
discovery/cleanup lane once startup completes. The total step limit is unchanged.
A startup lane can receive up to 37 turns in a 64-step slice (147 in a 256-step
slice); after startup, loans permit up to ten turns per ordinary slice. Advisory
membership processing is capped at one candidate every four ticks and may replay
one additional evidence record, without borrowing an active turn.
Cached Attempts always retain their active share. `idle_steps_reassigned` counts these loans. Repeated failing
records cannot prevent other
lanes from receiving service. Registration replay must succeed before that
Attempt's ordinary supervision can finalize shared truth. Failure still permits
identity-verified local-capacity reconciliation while preserving exit evidence.
When a selected launch intent has a matching registration but no manifest, bounded
service directly materializes that registration through the existing fenced path.
It need not wait for an unrelated retained-registration sweep to reach the new file.

The active locator cache holds at most 256 Attempt IDs per project. Once full,
new discoveries do not evict members awaiting their next service. Overflow
Attempts receive direct service through discovery; a missing cached process
manifest frees a slot for a later discovery. Pacing timestamps are retained only
for cached Attempts, keeping that auxiliary state bounded too.
Repeated service of a known process is paced by
`min(1 second, renewal interval)`; a newly observed exit bypasses that advisory
pacing. Neither authority mode is excluded: `holder_bound` Attempts still receive
supervision and terminal processing despite having no lease-renewal deadline.

For a stable set of at most 256 cached Attempts, active rotation revisits each
within nine project slices even when registration arrivals exceed their lane's
throughput and cleanup fails continuously. Valid arrivals cannot displace that
cached rotation. This is a selection bound, subject to
the service pacing above, not a renewal or wall-clock guarantee. Overflow Attempts
depend on directory discovery and do not have the nine-slice bound. A project's
elapsed service gap also includes other projects' work and returning storage I/O.
Load acceptance must therefore account for measured cycle duration and the
unchanged lease policy, not infer lease safety from the cache size.

`EvidenceScan` visits at most N directory entries per call, including unrelated
files, directories, and symlinks. It never builds a sorted inventory. Lanes use
N=1 and retain one iterator per directory. Directory device/inode replacement
invalidates that iterator. Machine supervision checks directory identity once per
bounded project slice, using an ephemeral slice token; standalone scans check on
every call. EOF closes the iterator, and the next slice starts another sweep.
An exhausted lane records only one completed sweep per slice, so repeated idle
turns cannot erase a failed sweep and grant startup readiness. Directory replacement
or creation during a slice is discovered by its next slice.
Additions do not restart an in-progress sweep, so they cannot continuously push
its tail back to the beginning. Entries missed during concurrent mutation are
found on a later sweep. Restart discards every cursor and repeats discovery.
No notification channel is required for correctness.

Termination-tree discovery consumes directory and child-record turns separately.
Terminal decision cleanup removes at most eight regular JSON files per attempt
cleanup call and never recursively traverses the tree. Identifying flat evidence
remains until the decision directory is gone. Unsupported nested entries are
retained rather than recursively deleted.

The portable discovery target is at most N entry visits and N selected paths
per `EvidenceScan.take(N)`, demonstrated for N=1/4/64. A semantic authority step
can still perform dependent reads, locks, and writes; the discovery limit is not
a claim that total slice I/O is bounded by N filesystem calls.

Recovery and ordinary supervision inspect termination decisions in pages of at
most eight directory entries, counting unsupported entries too. They retain the
Attempt lock continuously across pages and the final Recovery CAS or renewal.
No shared lock is held across a yield. Lock acquisition is nonblocking. A yielded
page is never a negative proof: only EOF permits recovery or renewal. A positive
irreversible commitment rejects recovery and instead advances ordinary termination,
including a crash after commitment but before `signal_committed` was recorded.

Each project retains one unfinished locked inventory, shared between recovery
and ordinary supervision. Its next page consumes the first step of the next
slice. An ordinary inventory that completes within its first page proceeds in
the same turn, so a long inventory does not prevent unrelated short inventories
from renewing. If the pending slot is occupied, another long inventory closes its
first-page probe and retries through its regular service opportunities; no queue
or additional persistent cursor is allocated. This bounds memory, not completion
latency under arbitrary numbers of long inventories. Successful completion resets
that cached Attempt's pacing time so it cannot immediately reclaim the slot.

Before delayed supervision acts, it reloads the manifest and current shared claim
identity. Recovery additionally revalidates the live process and clock evidence;
ordinary renewal retains the existing fresh clock checks. Other lanes defer the
same Attempt by its record identity while a proof is pending, avoiding lock
reentry. Exit evidence cancels recovery, but ordinary supervision completes its
termination check before classifying that exit. Terminal publication and hooks
run after releasing the local lock and retain their own shared-state fences.
If the manifest is missing, malformed, unreadable, or has a different Task,
Attempt, or fencing token after terminal CAS, the committed Task remains terminal,
lifecycle hooks still run, and `terminal_manifest_unreadable` records the local
publication problem. A failed local publication also preserves hooks and evidence.
Later active and cleanup turns revalidate the manifest identity before deleting
evidence. Registration-only completion can still clean up without a manifest.
Malformed evidence is retained for repair; its shape cannot terminate the
authority loop or authorize local cleanup.

Stop, generation replacement, unserviceable binding/registry state, and scan errors
discard the proof and close the iterator and lock. Restart scans again from
durable decisions. Startup is not complete while a locked inventory is pending.
A rejected CAS keeps the Task blocked and its local evidence/reservation intact;
completion of initial evidence sweeps is not a claim that every orphan recovered.
Readiness alone cannot authorize its next claim or prove idleness. Diagnostics
expose `pending_control_attempt`, `pending_control_kind`, and
`control_pending_age_seconds`. Scanning H decisions still takes O(H) total work
and may delay another tool requiring that Attempt lock until completion.
Synchronous doctor and standalone reconciliation retain the complete-scan API.

A new mutation marker cannot safely invalidate these scans unless every decision
writer honors it. Older recovery tools do not observe such a marker. Introducing
one as proof of an unchanged scan therefore requires an enforceable compatibility
boundary for those writers; package installation or agent readiness alone does
not supply that boundary. This candidate has not introduced that protocol.

Terminal accounting and cleanup check the Attempt's named reservation with at
most four JSON reads across the active/provisional GPU and CPU paths. Each lane
uses its existing capacity lock; locks are not nested. Expired provisional records
do not count as usage, active records do, and malformed matching records fail the
check rather than implying free capacity. This lookup neither enumerates nor
expires unrelated reservations.

Pending claim archives use one project-wide outer Task-directory cursor and one
selected Task's record cursor. A call visits one outer entry or at most eight
child entries, including unsupported entries. Each complete child sweep advances
the outer cursor even if malformed records remain; an unreadable child directory
also relinquishes its turn. Finite histories and repeated service therefore reach
later Tasks without per-Task cursor eviction. A large child history still requires
multiple calls before the next Task is selected; this is not a constant per-Task
latency guarantee.

Atomic removal of the caller's empty pending directory verifies
completion; timestamps and cursor exhaustion alone cannot authorize evidence
cleanup. Failed or malformed records stay pending while other records in the
page can progress. Unexpected non-record entries also retain the directory and
prevent cleanup; they are not recursively deleted. Archive creation still uses
the existing immutable create/fsync protocol. At most two cursors are retained
per project. Replay errors produce local diagnostics while evidence remains.
Restart loses only discovery progress, never pending archive evidence.

Bounded local exit reconciliation searches at most four active GPU and four
active CPU reservation entries per call triggered by verified finished evidence.
One resumable cursor per capacity lane belongs to the supervisor, rather than to
each Attempt. All eligible calls advance that pair; arrivals cannot evict its
progress, and a failed capacity directory cannot exclude the other lane.
Each selected reservation loads its own current local process/registration and
exit evidence. This can release another finished Attempt in the same project,
but still requires its verified process absence and matching Task, Attempt and
fencing token; mutation rechecks the full reservation identity under its capacity lock.
Missing evidence, foreign-project records, and token mismatches retain capacity.
Malformed entries are diagnosed and retained. Provisional reservations are not
searched because this release operation only mutates active reservations.

## Termination and deadlines

Bounded supervisors advance committed signal decisions without sleeping while
holding the Attempt control lock. Every step reloads the durable decision and
rechecks process-group identity. SIGTERM is durably recorded before the grace
period is tracked with a process-local monotonic deadline. SIGKILL is not sent
before that deadline. Loss of the advisory deadline on restart grants a fresh
grace period. The agent never infers process absence from elapsed time.

The complete-scan API retains its blocking signal helper. Both paths preserve
signal commitment, identity checks, and durable termination states; no persisted
termination fields or old-reader requirements change.

`advance_deadline(scheduled_at, interval, finished_at)` drops already-due slots
and returns a deadline strictly after completion. It does not replay missed
cycles. Periods must be finite and positive; timestamps must be finite. For a
cycle scheduled at 0, interval 0.25, and completion at 0.60, the next deadline
is 0.75 and two slots are skipped. Heartbeat uses the same arithmetic.

These monotonic values never replace wall-clock/clock-evidence lease checks.
Machine-registration eligibility remains independently renewed on project
service and heartbeat; it is not inferred from an Attempt's authority mode.
Every guarded operation still locks and reloads the registration, validating
expiry, generation, runtime instance, and runtime root through the authoritative
commit. Renewal persistence is due at the durable expiry minus TTL plus the
renewal interval capped at half the TTL. This retains a renewal opportunity
for ordinary polling when a valid Attempt renewal interval is near the TTL.
The authority service period is also capped at one quarter of the TTL, even
when the configured loop and Attempt renewal intervals are longer. Each project
visit derives this cadence from current shared policy, including idle policy
changes, rather than the supervisor's cached policy. Failure to read this advisory
cadence input leaves the existing eligibility and cached local-recovery paths in
place. This does
not bound delays from blocking I/O or overloaded project rotation. A reduced TTL or backward clock movement also shortens an
expiry beyond the current policy horizon. Unchanged expiry values do not need
another publication. Explicit registration and reactivation retain their existing
semantics; an ordinary guard cannot revive an expired or superseded binding.

The following targets were registered before adding this renewal scheduling,
after the initial instrumented implementation: with a stable default policy and
generation, fresh guards perform zero registration publications, the first due
guard performs one, and subsequent guards before the next due time perform zero.
A 25-second steady window permits at most four registration publications per
binding, allowing whole-second expiry quantization. Policy changes, explicit
registration, and reactivation are excluded from that stationary-load target.
Every guard must still detect a replaced generation or expired eligibility.
The existing 15-second lifecycle convergence gate remains unchanged. Wider-load
measurements below are separate from that protected four-Attempt contract.

Machine heartbeat snapshots (`agent.json`, `gpu.json`, and `summary.json`) and
machine-process `status.json` suppress replacement when persisted content is
identical. Heartbeat timestamps retain their existing precision and cadence;
state or identity changes within one timestamp still publish immediately. Missing
or malformed snapshots are rebuilt. Comparison uses the persisted document under
the existing writer serialization, without a process-local content cache or a
background write queue. Changed documents retain atomic replacement and fsync.
Authority commits, reservations, launch intents, and process registrations do not
use snapshot write suppression.


## Storage failures and operating limits

This implementation uses synchronous cooperative service, not worker-based
storage isolation. It promises bounded evidence selection when operations
return. An indefinitely blocked filesystem call can still delay other projects
on that thread. It makes no hard real-time or worker-pool saturation guarantee.
Heartbeat remains a separate thread, with its own synchronous I/O limitations.

Configuration/shared-store failures use a retained local observation cursor to
reconcile at most eight exit observations per fallback turn. Local resource
release still requires matching durable identity and verified process absence.
Released capacity never substitutes for durable Task/Attempt terminal truth.
Errors and missed deadlines remain explicit diagnostics.

Finite returning I/O, finite directory populations, and supported arrival rates
are assumptions for eventual sweep completion. Unlimited arrivals or more due
work than a slice can service can increase lateness. The mixed-authority overload
model uses 256 cached Attempts, half in each authority mode, one 64-step slice per
modeled second, eight valid protocol-v1 registrations from newly claimed and authorized Tasks per
slice, and failing maintenance.
Under these returning-I/O assumptions, the resident bounded-lease transactions advance
expiry at least every 18 modeled seconds with the default ten-second renewal
interval and retain more than 100 seconds of lease headroom. Holder-bound Attempts
retain no timed expiry. A 130-second service interruption with stale clock evidence
prevents further Attempt publication, marks bounded leases isolated, and preserves
holder-bound local-safe state and every fencing token. This is a scheduling
and transaction model, not a wall-time guarantee for 256 real processes.

## Work avoidance and diagnostics

Equivalent Attempt writes were already suppressed under the existing phase and
identity checks. The candidate additionally avoids equivalent local policy-cache
and process-authority-state writes. Comparison uses the persisted JSON value;
missing or malformed local records are rewritten. Token changes and incomplete
records are never hidden by an attempt-ID-only cache.

The machine-local `authority_control_plane.json` sidecar remains diagnostic-only.
The authority thread publishes a complete in-memory snapshot; heartbeat attempts
local persistence after its shared heartbeat work, at most once per
`max(1 second, loop_interval)`. Failed writes are throttled too. Existing atomic
persistence and fsync behavior remain intact.

The envelope includes producer instance, sequence, sample time, registry coverage,
project service order, phase durations, and requested/observed waits, cycle time,
start lateness, and skipped intervals. Project observations additionally include:

Each project sample identifies its registration generation and reports the
start-to-start `service_gap_seconds` and cumulative `maximum_service_gap_seconds`
for that generation. First service has an unknown (`null`) gap. Failed and
ineligible service attempts count as visits; the metric does not imply successful
renewal. Generation replacement resets the observations, and registry removal
discards them. These gaps include intervening project work and cycle waits.

| Field | Interpretation |
| --- | --- |
| `startup_complete` | Successful initial sweeps for this supervisor generation |
| Per-lane entries, processed work, failures, sweeps, duration | Cumulative observations for this supervisor; not total filesystem-call counts |
| `sweep_age_seconds` | Age of the current directory sweep; not an invented age for undiscovered records |
| `oldest_failed_work_age_seconds` | Time since a failure without a subsequent successful whole sweep |
| Active-cache size/limit and oldest active due age | Advisory known-work pressure; not complete inventory coverage |
| `active_admission_deferred` | Discovery encounters that could not join the full cache; counts encounters, not distinct Attempts |
| Maximum service gap and renewal lateness | Observed scheduling delay; never authority evidence |
| Registration-to-manifest/running, exit-to-terminal, terminal-to-accounting | Count, latest, and maximum observed stage delays |
| Equivalent local writes avoided and renewal outcomes | Work avoided and actual outcome counts |
| `operations.counters` and `operations.timings` | Scoped JSON-store calls, fsync calls, legacy inventory entries, and lock-acquisition attempts; cumulative count, total duration, and maximum duration |
| `eligibility_operations` | Per-visit registration renewal/reactivation publications, renewal lateness observations, and scoped storage/lock operations |

Successful registration publication observes renewal lateness against the previous
expiry minus the current policy's TTL plus its renewal interval. The observation
is taken after publication returns, so it includes publication delay. Reactivation
and ordinary renewal have separate counters; failed publications contribute no
success or lateness observation. An unreadable previous timestamp produces an
unavailable-timing counter without changing renewal behavior. Lateness uses UTC
and is diagnostic only, including when policy or wall-clock values change.
`registration.renewal_lateness` uses observation counts and nanosecond total/max
fields, not measured operation-call durations. The per-project envelope covers
authority-thread eligibility and reactivation. The sidecar's separate `heartbeat`
sample aggregates heartbeat-thread operations, including both preliminary
eligibility refresh and renewal inside the guarded heartbeat publication. These
are distinct registration writes and each contributes an observation. Other write
guards outside these two control-plane scopes are not included.

Heartbeat samples have their own sequence, UTC sample time, monotonic duration,
and status (`returned`, `registry_unavailable`, `capacity_unavailable`, `publication_unavailable`, or
`raised`). Individual skipped projects are counted as ineligible, unavailable,
or rejected by the final write guard;
`returned` does not assert successful publication for every binding. Each sample
replaces its predecessor, including on failure, and does not mutate the authority
thread's snapshot. Sidecar publication still occurs after heartbeat work and uses
the existing throttle; its authority and heartbeat samples can have different ages.

Operation collection covers the cooperative supervisor tick and its synchronous
callees. Failed calls are counted and timed; lock-acquisition time includes waiting
inside `flock`. JSON-store writes include their nested fsync durations, so timings
must not be added together as disjoint costs. These counters do not cover every
filesystem operation or work in other threads, and are never authority evidence.

Stage delays using durable UTC timestamps are diagnostic and can be affected by
wall-clock changes. Sweep/work durations use monotonic time. `tick_returned` only
means the method returned; it does not certify authority health. Snapshots may
remain after shutdown or be republished unchanged. Consumers must inspect
producer identity, sequence, and age; file mtime is not a liveness proof.

## Verification

Evidence currently lives in the following tests:

- `tests/integration/qexp/test_authority_scan.py`: actual iterator-call limits,
  directory replacement, restart, error recovery, symlinks, and lost-wakeup discovery.
- `tests/integration/qexp/test_authority_work.py`: bounded startup inventory,
  terminal progress despite large registration history, malformed-record fairness,
  both authority modes, restart, publication failure, and incremental cleanup.
- `tests/unit/test_qexp_authority_control_plane.py`: deadline arithmetic, rotation,
  registry changes, the 16-project cap, generation replacement, and diagnostics.
- `tests/integration/qexp/test_machine_runtime.py`: admission after current-generation
  recovery, independent maintenance/control progress, and resource behavior.
- `tests/integration/qexp/test_lease_hardening.py`: durable termination and nonblocking
  signal grace, including conservative restart of a lost deadline.
- `tests/integration/qexp/test_authority_overload.py`: real mixed-mode lease
  transactions under saturated discovery and failed maintenance, followed by
  stale-clock isolation without token changes or unauthorized lease publication.
- `tests/integration/qexp/test_authority_workload.py`: real tmux/Python runners,
  actual fsync, isolated roots and resource cleanup, four-Attempt default, and
  independently configurable binding/Attempt/local-history/shared-history counts,
  plus an optional 0–30-second live hold that checks running state and unchanged
  fencing tokens before requesting completion.

The workload test's optional measurement plugin wraps the same production entry
point and store, scandir, metadata, fsync, and flock boundaries in both versions.
It retains raw event timestamps, per-thread operation counts/durations, source SHA
and file fingerprints, and instrumentation fingerprints. Attempt lease and
machine-registration expiry changes are retained as distinct publication events;
`hold_seconds` keeps runners alive to observe these renewals. `all_running_seconds`
measures startup separately from the later completion request. Parent observations
during the hold check each Attempt once per second and are retained in the report. Durations overlap for
nested operations. Registration and exit source timestamps precede their runner
publication calls; resulting stage differences include that publication cost and
are not exact durable-publication-to-publication measurements. Parent polling
uses a 10-ms interval. GPU visibility is simulated; the training command is a real
Python process, not a GPU performance workload. Flock counts include acquisition
and release calls; their duration is not a direct measurement of contention.

The observer also captures immutable cumulative counter snapshots every 250 ms,
with a bounded 4,096-snapshot buffer and an explicit dropped-sample count. The
parent records the steady phase's monotonic start/end. The report summarizer uses
the first and last snapshots inside that phase, subtracts counters per thread,
and divides by their actual elapsed time. It reports window coverage and leaves
missing or too-short windows unavailable. These steady rates separate startup
cost from ongoing service; whole-run totals alone are not comparable rates.
A separate bounded buffer retains the latest 16,384 write observations, with
path, thread, start/end, write duration, nested fsync count/duration, and failures.
It reports dropped observations explicitly. Successful immutable creation has a
`None` return value and is counted; an explicit failed-create result is not a
publication. Agent instrumentation does not include runner-process I/O.
Token and registration-generation changes remain distinct events even when an
expiry timestamp is unchanged. Source UTC timestamps can have one-second
quantization, so subsecond stage comparisons require that uncertainty to be
considered separately from monotonic end-to-end measurements.

For example, collect a retained-history sample with:

```bash
./scripts/dev test -p tests.helpers.qexp.authority_measurement \
  tests/integration/qexp/test_authority_workload.py \
  --authority-workload-profile='{"bindings":4,"local_history":1024,"hold_seconds":25}' \
  --authority-workload-output=/tmp/authority-workload.json -q
.tox/unit/bin/python -m tests.helpers.qexp.authority_reports \
  /tmp/authority-workload.json > /tmp/authority-summary.json
```

The summarizer retains the complete last control-plane snapshot, including
separate authority/heartbeat eligibility scopes, scheduler observations, per-lane
ages, and cumulative service gaps and admission deferrals. It reports missing
snapshots and checks project/generation coverage against observed registrations.
Whole-run per-thread operation counters retain failed-call counts and timings.
A latest sample is not a historical series: sweep, failed-work, and oldest-due
ages and per-visit eligibility counters must not be presented as run maxima or
totals. Validate sample time/sequence against the final observer capture when
using these diagnostics as acceptance evidence. Expiry-event lateness assumes
the workload's fixed default policy (TTL 120 seconds, renewal interval ten seconds).
The workload has separate 15-second startup/terminal/accounting waits; the
protected lifecycle tests establish the four-Attempt convergence gate.

The optional `--authority-profile-startup` observer adds runner import, tmux,
shell/Python entry, authority-lock, and intent-publication phases; usage and
limitations are in [the test guide](../../tests/readme.md#qexp-startup-profiling).
The test environment clears inherited `TMUX`/`TMUX_PANE` before creating its own
server. Startup profiles verify both the runner's source checkout and isolated
HOME. Earlier measurements taken through an inherited server are invalid for
baseline/candidate comparison: its existing environment can import a different
checkout and run unrelated login initialization.

On 2026-09-18, an isolated P4/A1 workload with a 0.1-second agent loop, no retained
history, and a five-second steady phase ran three alternating pairs against pre-optimization `20caa93`.
All six passed the unchanged 15-second startup wait with valid runner origin.
Median per-run measurements were:

| Measurement | Before | Snapshot suppression and lazy download export |
| --- | --- | --- |
| Steady atomic replacements/second | 42.84 | 17.43 |
| Steady fsync calls/second | 85.69 | 34.86 |
| Runner import | 51.15 ms | 42.21 ms |
| All four Attempts running | 3.630 s | 4.071 s |

These samples demonstrate reduced write traffic, not an end-to-end startup
speedup. The high-frequency workload often repeats the same whole-second heartbeat
timestamp. At the normal five-second cadence, timestamps usually change on every
heartbeat, so snapshot suppression saves substantially fewer writes; unchanged
process status can still be suppressed. Runner authority-lock acquisition and intent publication each took about
one millisecond; command-to-shell medians ranged from 482 to 758 ms, including
command transmission. A separate isolated-shell check measured login startup at
657–1,030 ms versus 4–5 ms without startup files; this host's existing clock-wait
hook alone took 356–489 ms. These host-specific costs are not repository timing
guarantees. These are historical pre-repair shell measurements. Final comparisons below use
the same repaired host login hook for both variants. Launch timeouts and runtime
clock qualification remain unchanged.

A shared synchronous snapshot comparison captures the measured write reduction
without introducing queue ownership, flushing, or cross-lock ordering. Batched
directory fsync could theoretically reduce three changed snapshots from six
fsync calls to four, but it would not eliminate file fsync or shell initialization;
these measurements do not justify a general asynchronous writer. The independent
scale/history comparisons below and the mixed-mode transaction model provide
separate evidence for scheduling and steady-state cost.

## Matched runtime and cost evidence (2026-09-18)

Baseline `d6e0e6fdb362977f3d9e46e0e3737cd50b6a2364` and candidate production
source `47f717e9dc054c53f88aeaf6cd4e739ca38f0aac` ran three alternating pairs
per profile (36 runs; all passed). Later delivery edits affect tests and this
report, not the measured production code. Each profile used Python 3.13, Linux
5.15, real tmux/Python children, fsync-enabled local storage, a 0.1-second agent
loop and a 25-second live hold. P is bindings; A is Attempts per binding; H is
retained records per binding. Dimensions varied independently; this is not a
claim about every cross-product of these loads. Task setup is outside startup
measurement. Training completion, unchanged fencing tokens, one execution per
Task, terminal publication, accounting and test-owned resource cleanup were checked.

Both variants used identical observation and isolation helpers. Every runner
profile verified the expected source and HOME. The instrumentation SHA-256 is
`674e1008044db0cdd053c21463efd685d820647323135282072a9562cd46d546`;
the workload test SHA-256 is
`8393ee84efbbc5d19452c853ba1dbea23c55e47c1afd30764a419fb7035e1f54`.
Raw reports retain source-file fingerprints, publication events, operation
snapshots, startup phases and diagnostics; the retained test commands above
reproduce the collection and summarization. Initial clock qualification was
26.3 ms against the unchanged 100 ms limit. A host repair changed the root login
hook from waiting for fresh NTP queries to ensuring the measurement-only chronyd
process exists; actual authority clock qualification remains fresh. Both sides
used that repaired hook. Older shell-handoff failures and invalid inherited-tmux
samples are retained separately and excluded from these matched aggregates.

Values below are baseline / candidate medians of the three per-run measurements.
Steady rates use interior counter differences divided by actual sampled duration.

| Profile | All running (s) | JSON reads/s | Atomic replacements/s | fsync/s | stat + lstat/s |
| --- | --- | --- | --- | --- | --- |
| P1/A1 | 1.56 / 1.02 | 920.43 / 744.94 | 81.77 / 13.83 | 163.53 / 27.65 | 8894.13 / 7253.35 |
| P4/A1 | 3.19 / 3.57 | 1143.17 / 1135.53 | 82.17 / 17.97 | 164.39 / 35.94 | 10402.41 / 10428.81 |
| P16/A1 | 12.00 / 13.06 | 1033.56 / 1144.79 | 73.64 / 37.63 | 147.28 / 75.27 | 9734.42 / 9596.24 |
| P4/A4 | 7.22 / 8.14 | 1468.30 / 1586.66 | 83.93 / 19.05 | 167.85 / 38.10 | 7642.28 / 7795.22 |
| P4/A1/local H1024 | 4.63 / 6.84 | 3118.14 / 1156.56 | 83.58 / 18.71 | 167.12 / 37.41 | 13791.03 / 11382.37 |
| P4/A1/shared H1024 | 2.95 / 3.36 | 1140.13 / 1185.87 | 83.26 / 18.33 | 166.52 / 36.67 | 10473.44 / 10953.41 |

Write avoidance reduces steady I/O, but does not establish an overall startup
speedup. Startup recovery and additional identity checks can cost more than the
baseline. At the normal five-second loop, timestamp changes reduce the benefit
of suppressing identical heartbeat snapshots. No asynchronous writer was added.

| Profile | Registration→manifest (s) | Registration→running (s) | Exit→terminal (s) | Terminal→accounting (s) |
| --- | --- | --- | --- | --- |
| P1/A1 | 0.605 / 0.320 | 0.610 / 0.339 | 0.677 / 0.611 | 0.007 / 0.006 |
| P4/A1 | 0.851 / 0.810 | 0.864 / 0.824 | 0.635 / 0.907 | 0.019 / 0.012 |
| P16/A1 | 0.997 / 0.982 | 1.009 / 0.991 | 1.783 / 2.295 | 0.021 / 0.025 |
| P4/A4 | 0.813 / 0.871 | 0.828 / 0.876 | 1.412 / 1.213 | 0.017 / 0.018 |
| P4/A1/local H1024 | 2.172 / 0.692 | 2.187 / 0.697 | 3.497 / 1.091 | 0.005 / 0.020 |
| P4/A1/shared H1024 | 0.695 / 0.903 | 0.702 / 0.908 | 0.937 / 0.729 | 0.021 / 0.011 |

Stage values are medians of per-run medians, not pooled percentiles. Source
UTC timestamps precede publication and have whole-second quantization. Each
variant has 135 measured Attempts; raw summaries retain the coverage count of
each stage. These stages do not replace monotonic lifecycle-gate timings.

| Candidate profile | Maximum project service gap (s) | Maximum Attempt renewal lateness (s) | Maximum eligibility renewal lateness (s) |
| --- | --- | --- | --- |
| P1/A1 | 0.215 | 1.332 | 0.028 |
| P4/A1 | 1.160 | 2.537 | 0.170 |
| P16/A1 | 5.524 | 4.293 | 0.813 |
| P4/A4 | 2.714 | 3.044 | 0.199 |
| P4/A1/local H1024 | 1.250 | 2.108 | 0.175 |
| P4/A1/shared H1024 | 1.138 | 1.830 | 0.164 |

All candidate bindings published registration two or three times during each
25-second hold, satisfying the previously registered maximum of four. No write
or counter snapshots were dropped. Interior steady windows covered at least
98.2% of the requested phase. Every final candidate diagnostic included all
expected projects and current registration generations. Authority/heartbeat
samples were 0.144–2.246 seconds old at the final monotonic observer capture
(UTC mapped through the last event; one-second timestamp quantization applies).
Positive sequence numbers and cumulative service-gap counters show progressed
service. Final oldest-active-due ages were zero, with no pending control proof
or deferred cache admissions in these profiles. Final lane ages, failures,
eligibility operations and heartbeat operations remain in the raw snapshots;
no historical peak queue-age claim follows from those final samples.

An additional P16/A1/local-H1024 interaction stress test failed the unchanged
15-second diagnostic startup wait on both versions. The candidate had no early
claims while initial recovery was incomplete; disabling operation instrumentation
also failed to complete startup in that interval. This combination is outside
the protected at-most-four-Attempt convergence promise and is not advertised as
a 15-second supported load. Finite returning I/O still governs eventual progress;
synchronous storage can delay every project. The mixed-authority overload model
above separately establishes selection/lease behavior under valid arrivals and
failed maintenance, without claiming wall-clock process capacity.

Integration requires full qexp Integration (including LI-03 and protected
lifecycle/crash cases), installed-wheel public-entrypoint coverage, and complete
preflight with `.dev/**` removed. Known baseline failures never waive these gates.

## Compatibility and rollback

No authority record, registration, or termination format changes. Cursors and
signal deadlines are advisory process-local state. The new local sidecar can be
ignored by an older package. Existing single-owner restrictions remain in force;
this change does not authorize simultaneous old/new machine-agent owners.

Reverting the candidate restores the complete-scan control path. Disabling
observability alone does not undo scheduling. Neither path cancels already
running training on agent restart; durable registration and exit evidence remain
available to the returning authority.
