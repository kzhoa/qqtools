---
doc_type: adr
status: active
updated_at: 2026-08-28
archived_at:
---

# Ready Index and Portable Work Budget

## Context

The scheduler currently discovers work by scanning Task truth. Completed Task history therefore
adds directory enumeration, JSON reads, and parsing to every maintenance and dispatch cycle. No
single benchmark machine or filesystem can supply a latency threshold that is valid for all users
of an open-source library.

## Decision

Scheduling will use a durable, rebuildable `ready` liveness projection. Task, Attempt, Submission,
Group, claim, and reservation records remain authoritative. A missing or stale ready marker may
delay scheduling and is therefore a liveness defect, but it cannot authorize execution.

Each Task owns a monotonically increasing `ready_generation`. A producer writes the new marker
before committing Task or Submission truth, then removes the old marker after commit. Marker paths
are:

```text
indexes/ready/home/<machine>/<partition>/<task-id>.<generation>.json
indexes/ready/shared/<partition>/<task-id>.<generation>.json
```

Writers allocate each marker into a monotonically numbered partition record with at most 64 durable
slots. Active partition IDs are themselves held in linked catalog pages of at most 64 IDs. Scheduler
consumers follow catalog and partition records and open exact marker paths; they do not glob an
unbounded directory. Slot allocation is durable and occurs under a short ready allocator lock, which
is released before any Task or Submission lock/I/O. A partition is sealed when its slots are allocated
and removed only after its markers and reservations drain. Consumers compare marker identity and
generation with authoritative truth and classify each record as claimable, temporarily unavailable,
permanently stale, or corrupt. Only permanently stale markers may be removed, after rechecking that a
concurrent writer has not replaced the generation.

The additive layout persists an exact per-generation slot reservation. Terminal transitions can
therefore remove a marker and release its partition slot without scanning catalogs; sealed empty
partitions are removed from their catalog page. Existing schema-6 Task records without the additive
field read as generation zero and remain on legacy discovery until the builder activates the index.

Project and per-project candidate cursors are durable advisory progress. Every inspected marker
advances the candidate cursor, including errors and claim races. A crash may repeat inspection;
claim fencing prevents duplicate execution.

The machine scheduler runs deterministic fair rounds. Each enabled project may win at most one
claim in a round; remaining GPU capacity and work budget permit an immediate next round. Candidate
inspection shares one slice budget across projects, while each project retains disposable adaptive
batch state. Full capacity bypasses ready candidate reads entirely.

Discovery is state-gated: `absent` and `building` use compatibility Task discovery, `active` uses
ready-only discovery, and `degraded` rejects new claims for that project while control and repair
continue. Corrupt active catalog, partition, or marker data never activates a full-Task
fallback. Permanently stale deletion rechecks authoritative generation before removing the exact
marker reservation.

Index cutover is `absent -> building -> active | degraded`. Building records a Task watermark,
backfills through it, catches up writers, performs a final consistency check, and enables ready-only
scheduling only after the writer capability gate is active. An incompatible writer fails before
mutation. Repair may rebuild projections from truth, but ordinary active scheduling never scans all
Tasks to compensate for projection loss.

Work is constrained by environment-independent record, Task-read, Attempt-read, reservation-read,
operation, and memory hard limits. A 50 ms monotonic soft deadline controls when another independent
record may start; it cannot interrupt an already-blocking synchronous filesystem operation. An
in-process adaptive batch begins conservatively, shrinks immediately on slowdown, grows one record
only after repeated faster observations, and never exceeds hard limits. Adaptive state is disposable
and never enters shared truth. Yielded due work is rescheduled immediately after control-plane threads
receive an execution opportunity.

Starting Attempt recovery is driven by machine-wide active reservations plus exact launch/process
evidence. Timed offers and maintenance consume bounded active sets. No `pending-launch` or generic
`maintenance` marker is introduced.

Startup runs scheduling immediately. Capacity released within a cycle and ready markers produced by
local maintenance feed the next immediate fair round. Registry state is reloaded on every scheduling
cycle, and the normal loop interval remains the portable idle probe for remote writes and later
binding changes; correctness does not depend on filesystem notifications.

Availability, Group control, and cleanup store their complete unfinished truth in type-specific
`active/` directories and publish terminal history at stable paths. Timed offers are partitioned by
home machine and UTC-hour bucket. Compatibility symlinks preserve existing exact-path diagnostics
without duplicating authority. Regular maintenance consumes no more than 64 active records and does
not scan terminal operation or Task history.

The machine-wide capacity gate snapshots reservation identity while holding only the local
reservation lock, performs shared Task and Attempt verification after releasing that lock, and uses
a full-identity compare-before-mutate operation for retag or release. A second locked snapshot counts
active and unexpired provisional GPU occupancy. Unknown, malformed, or unverifiable ownership stays
reserved. When no visible qexp GPU remains free, the agent skips ordinary scheduler work but keeps
control, authority, recovery, and maintenance paths active. Starting recovery opens only the Task and
Attempt named by an active reservation; machine-agent recovery never discovers `starting` work by
scanning the Task directory.

Cross-host use additionally requires the deployment-specific, two-host qualification contract in
[0001-shared-filesystem-coordination.md](0001-shared-filesystem-coordination.md). Probe evidence cannot
tune partitions or work budgets.

## Consequences

- Steady-state scheduling cost is bounded by active work and protocol limits, not terminal history.
- Marker writes add projection I/O and cross-record crash windows; generation ordering and repair
  make those windows explicit.
- Fixed-size partition/catalog pages and conservative defaults may require more immediate slices for a large backlog,
  but do not weaken fairness or resource bounds.
- Local benchmarks detect operation-count regressions and provide diagnostics; they do not certify
  universal latency.
- Ready-index activation is a schema capability transition and must preserve protected `init`,
  `submit`, and migration workflows without a new mandatory user step.

## Rejected Alternatives

- Scanning Task truth with a larger interval: reduces frequency but retains history-dependent cost.
- Persisting machine-specific throughput tuning: leaks one deployment's filesystem behavior into
  shared policy and makes recovery non-portable.
- One unpartitioned marker directory: lacks a useful enumeration and memory bound for large queues.
- Treating markers as authority: duplicates Task truth and creates unsafe split-brain decisions.
- Adding generic pending-launch or maintenance markers: duplicates recovery evidence already owned
  by reservations and durable operation records.
