# Active work and indexed history delivery acceptance

Qualification snapshot: 2026-09-20.

The delivered runtime uses qualified active responsibility discovery, certified
Group membership/change indexes, bounded Submission committed-state proofs, and
opt-in Task-ID keyset pagination. Authoritative records and existing ownership
fences still authorize every mutation. See the [runtime specification](qexp_runtime_spec.md),
[local responsibility protocol](qexp_local_responsibility.md) and
[reproducible qualification commands](../../scripts/qualification/README.md).

## Deployment and measurement envelope

Qualification ran on Linux with Python 3.13.15, real isolated processes and tmux,
real file/directory fsync, and both the source checkout and isolated runtime data
on `/tmp` on the host's overlay filesystem. Dependencies use the prepared project
environment. Caches were
warm/uncontrolled. This establishes same-host behavior; it does not certify
cross-host hcufs ordering, physical power loss, or cold-cache tail latency.
The runtime source is `3e2ae36`; subsequent delivery changes relocate test tools,
strengthen qualification evidence and update documentation without changing runtime.

The real machine matrix uses four registered projects and one live Attempt per
project. History counts are per project, so the largest case retains 400000 historical Tasks plus four new live Tasks.
All cases require primary discovery, one launch per live Task, successful original Attempt
results, released reservations, and zero unrelated retained Task opens across all
agent threads. Runner I/O is not instrumented. Fixture construction and automatic
capture qualification occur before the timed healthy-startup workload; upgrade
convergence has separate lifecycle and released-writer coverage.

## History and admission scale

| History per project | Submission shape | All running (s) | Accounting complete (s) | Agent retained-history opens |
| ---: | --- | ---: | ---: | ---: |
| 0 | bulk | 8.14 | 9.84 | 0 |
| 1000 | bulk | 9.98 | 11.57 | 0 |
| 1000 | single | 10.15 | 12.21 | 0 |
| 10000 | bulk | 9.18 | 11.49 | 0 |
| 10000 | single | 10.06 | 11.51 | 0 |
| 100000 | bulk | 9.82 | 11.69 | 0 |
| 100000 | single | 10.60 | 12.13 | 0 |

Times start at the agent launch request; accounting time includes execution.
Fixture construction is excluded. The existing 15-second per-stage assertions
remain unchanged. All seven cases use identical runtime and fixture fingerprints.

The earlier mixed-filesystem run (source on hcufs, runtime data on `/tmp`) passed
six cases but failed the 100k single-Submission startup deadline: one Attempt was
running and three were starting at 15 seconds. Agent import alone took 5.91 seconds;
no unrelated retained-history files were opened. This failed run is retained as
diagnostic evidence, not a passing result. The final matched matrix uses the local
source checkout throughout, without changing code, deadlines, assertions or fsync.
No claim is made that the original deployment's startup-latency gate passed.

The matrix collectively covers one large bulk Submission and many single-Task
Submissions; each case selects one shape. Its live workload consists of newly submitted Tasks.
The separate `test_submission_control_scale.py` matrix exercises the surviving
active Task inside an enlarged retained bulk Submission: classification,
eligibility and actual claim may read a source only below 64 KiB; larger cases
must read zero source payload bytes. This is an independent source-size dimension,
not evidence that the machine fixture's new Tasks belong to the retained batch.

The four pagination cases passed. Nonempty histories exercise ten query shapes,
including every populated phase/Group intersection and an absent Group; the empty
history case has seven applicable shapes. Full quiescent traversal equals truth.
Each page is bounded to 50 results, at most 1089 counted index pages and less than
13 MiB of counted index reads, with directory enumeration forbidden. Unfiltered
first-page index reads at 0/1k/10k/100k were 563/7957/8231/8494 bytes respectively.

The surviving-Task bulk admission matrix also passed. Source sizes were
1692/482695/4811696/48101697 bytes; the three admission consumers read 5076 bytes
in total for the small source and zero source payload bytes in every larger case.
Classification plus actual claim took 0.019–0.033 seconds in these samples.

## Active load, crash recovery and maintenance

The active-set envelope uses complementary evidence: empty/one-member storage and
primary startup regressions, four real simultaneous Attempts across four projects,
16-member fair primary service, and production-ledger traversal at 256 members.
The 256-member case is structural service evidence, not 256 real training processes
meeting the four-Attempt 15-second latency budget.

A 10000-cycle publish/handoff/retire profile with four retained active members
finished with 39 files (33 fixed files, four locators and two dense pages).
The unchanged production ledger hash is
`eafb250b94de55fce985247626450b9ead4fc92f73832fdc207099a3761148b1`.
Median/p95 publish was 5.23/8.63 ms and retirement 4.17/6.95 ms in that run.
These small, warm-cache samples are cost observations, not latency guarantees.
Normal idempotent publication and reads write nothing. The maintained profiler
reports file/directory fsync, bytes and lock hold/wait cost per transition.

Formal regressions cover partial durable publication, interrupted replay and
capture, late writer evidence, stale generations, cleanup failure, Group fairness,
recheck handoff, change-index reclamation, malformed/missing indexes, and bounded
rebuild. Local evidence and shared history have separate discovery paths; direct
reads of historical dependencies are legitimate authoritative work and remain
separately budgeted. Query-index failure cannot authorize work or stop safe local
supervision and terminal collection.

Released v1.3.17/v1.3.18 writer and running-workload probes previously passed with
automatic Group cutover, original process/Attempt identity, one launch, offline
exit and terminal reconciliation. The final coordination change also has targeted
registration, capture, lifecycle and shutdown-owner regression evidence. Migration
and admission authority are preserved independently of background progress state.

## Compatibility and user-visible limits

Group authority automatically moves to the canonical `groups-v2` namespace under
the recorded upgrade protocol; old cached writers cannot mutate canonical truth.
Unfinished old worker-removal operations without worker incarnation evidence are
blocked. Tasks, processes and resources remain intact; operators may reissue
`qexp group worker remove <group> <machine>` when removal is still needed.
This compatibility change was explicitly approved on 2026-09-19 and is documented
in the public product/runtime upgrade contracts.

Task pagination is opt-in live Task-ID keyset traversal, not a snapshot. It supports
unfiltered, phase, Group and combined partitions; short/empty continuing pages and
explicit invalid/degraded cursors preserve bounded work. Existing unlimited lists,
full Attempt detail and `top` aggregates retain their documented behavior and are
outside the bounded-history promise. Standalone `run_dispatch_cycle` with legacy
starting recovery enabled retains its explicit Task scan; normal machine admission
disables that branch and uses reservation recovery. There is no new exact-count or
search API.

## Delivery checks and retained tools

Qualification helpers live in `tests/helpers/qexp`; real integration tests retain
their crash and process coverage. Maintained cost and released-writer tools live in
`scripts/qualification`. Production imports none of those helpers. The obsolete
Group census draft and superseded private design/execution pitches are removed
only after delivery acceptance; the feature has one main archived pitch.

The complete qexp Integration invocation produced 2,136 passes and two failures
in 722.21 seconds. Both failures concerned test measurement boundaries: a
process-global memory peak included allocations outside the audit, and an idle
exit timer started at capture completion before Group activation and source
release had settled. Repairs preserve the 100,000-record input, 2 MiB audit bound,
and original enrollment and exit deadlines; production code is unchanged.
The isolated audit module passed all 30 tests. Enrollment/capture/coordination
checks passed 81 tests, followed by all nine enrollment tests after strengthening
the exact capture-identity/source-release receipt check. Independent review found
no remaining actionable findings. These targeted results complement the 2,136
unaffected passes; the original invocation is not reported as an all-green run.

Promotion additionally requires governance, Ruff and complete candidate preflight.
Its actual result and promoted revision belong to delivery evidence; this report
does not predeclare that gate or a release to main.
