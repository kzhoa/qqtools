---
doc_type: pitch
status: archived
updated_at: 2026-10-04
archived_at: 2026-10-04
---

# qexp Group-Service Large Submission Activation

## Goal and scope

Allow the fenced `group-service-v1` activation bootstrap to inspect every valid
Submission Operation with bounded, restartable streaming, including records larger
than the 64 KiB direct-read boundary. Preserve 64 KiB as a maximum individual read
chunk rather than treating it as the maximum legal Submission size.

The correction covers both the historical `operations/submissions` sweep and the
authoritative Submission referenced by the submission-control pending namespace. It
also lets a patched agent automatically retry the exact 1.3.24 failure where a valid
source above 64 KiB was recorded as `repair_required`.

This change does not enlarge locator, activation, Group-control, cleanup, request, or
result record limits; mutate or delete Submission truth; replay Tasks; relax writer
fences; or require a manual Project repair.

## Observed failure

The 1.3.24 activation reader correctly separated the 16 KiB metadata limit from the
64 KiB direct Submission-source boundary. A production Project nevertheless contains
six valid committed bulk Submission Operations larger than 64 KiB. Their sizes range
from 65,846 to 213,184 bytes and they are referenced by 330 Tasks. Every source is
valid JSON and has a revision-bound submission-control receipt.

The activation bootstrap advances one historical Submission at a time. When it
reached the first 115,354-byte source, it failed with:

```text
JSON record exceeds its 65536-byte limit: operations/submissions/96522abab62647c1.json.
```

The coordinator then durably recorded `phase=activation`,
`state=repair_required`, and `admission_blocked=false`. Deleting the source would
leave Task and idempotency references dangling and would only expose the next large
record.

## Root cause

A Submission Operation embeds the resolved bulk plan: Task IDs, complete Task specs,
worker and Group decisions, observation metadata, and live-progress selection. Its
encoded size therefore grows with the number and shape of Tasks in one bulk submit.
The canonical writer deliberately supports sources above 64 KiB by publishing a
pending intent, the authoritative source, and a small revision-bound receipt.

The activation bootstrap still calls a whole-record JSON reader with the 64 KiB
direct-source limit to obtain only `submission.operation_id`, `submission.state`, and
`submission.target_group`. This confuses a bounded direct-read policy with a source
validity rule. Raising the constant to the 256 KiB migration-slice budget would repair
the observed Project but fail again for a larger valid bulk Submission.

## Contract

### Source validity and resource bounds

- A Submission is not invalid merely because its encoded source exceeds 64 KiB.
- Sources at or below 64 KiB may retain the current whole-record direct path.
- Larger sources are lexed and projected in chunks no larger than 64 KiB. No slice
  retains the full source, Task-spec array, or unrelated scalar values.
- One coordinator phase call keeps the existing limits of one source work item,
  128 metadata operations, 256 KiB total I/O, and one-second interruption budget.
- Locator and ordinary Group-service metadata retain their 16 KiB record limit.
- Malformed JSON, a non-regular source, a symlink, a mismatched operation identity,
  an invalid state, or an invalid Group name fails closed with source-specific
  diagnostics.

### Projection and progress

The large-source projector validates one JSON root and extracts only:

- `submission.operation_id`, which must match the filename;
- `submission.state`; and
- `submission.target_group`, which may be null only where the existing activation
  behavior permits it.

Projection progress is bound to the source's regular-file revision and to the
bootstrap namespace and operation ID. A compact coordinator-owned checkpoint stores
the lexical offset and structural projection state outside authoritative Submission
truth. A canonical checksum rejects altered or corrupted checkpoint fields before
any offset or projected value is trusted. Checkpoint and source opens are no-follow,
nonblocking reads whose descriptor type is verified as a regular file. The existing
activation directory cursor does not advance until projection finishes and any
required membership locator is durable.

If the source revision changes, stale projected state is discarded and parsing
restarts from byte zero. A process crash may repeat a chunk or locator publication;
both paths are idempotent. The checkpoint is replaced for the next source and removed
when activation no longer needs it. No open file descriptor or parser object is
required across slices or agent restarts.

For sources within 64 KiB, output and cursor behavior remain byte-for-byte compatible
with the existing direct path. For larger sources, states that do not require a
membership locator still receive complete structural and identity validation before
the cursor advances.

### Released-failure recovery

Writable discovery may convert a migration from `repair_required` back to `runnable`
only when all existing recovery predicates hold and the stored error exactly reports
the 1.3.24 65,536-byte Submission-source limit for a canonical regular file larger
than that boundary. The activation record must remain in its fenced `building` state,
with no pause, repair plan, or in-flight slice.

Recovery preserves the journal, audit evidence, activation generation, cursor, work
counters, and fence history. It clears only the stale failure and retry timing. The
next ordinary slice must stream and validate the source; reconciliation itself does
not claim progress or completion.

This is a permanent correction for a released reader defect. It does not retain an
alternate writer or temporary data format and therefore does not add a compatibility
registry item.

## Implementation boundaries

- Add a Group-service activation source projector beside the existing discovery
  streaming components. It owns bounded lexical/structural extraction and a compact,
  versioned checkpoint; it must not own migration scheduling or locator publication.
- Extend migration storage with only the regular-file, no-follow, budget-accounted
  byte-read and checkpoint-cleanup operations required by that projector.
- Integrate the projector into the `submissions` and `submission_control` activation
  namespaces. Keep the directory cookie fixed while a source is incomplete.
- Extend the exact Group-service historical retry recognizer from the released
  16 KiB false positive to the released 64 KiB false positive. Do not recognize
  unrelated errors or generic oversized metadata.
- Update the runtime specification and upgrade guide to distinguish direct-read,
  streaming, per-slice, and source-validity boundaries.

No dependency, schema version, CLI command, protected workflow, or public branch
policy changes are required.

## Verification

Regression coverage must establish:

- direct sources at the 64 KiB boundary retain their current behavior;
- valid grouped sources at 64 KiB + 1, above one parsing chunk, and above one
  migration slice converge without whole-record reads;
- a fresh coordinator resumes a partial projection from its durable checkpoint;
- source replacement invalidates the checkpoint and restarts projection safely;
- both the historical Submission namespace and submission-control pending namespace
  use the streaming path;
- required locator publication is durable before the directory cursor advances;
- terminal/nonmatching sources advance without publishing a locator;
- malformed, identity-mismatched, symlink, and structurally invalid sources fail
  closed without advancing the cursor;
- 16 KiB limits remain enforced for locator and ordinary control records;
- an exact 1.3.24 64 KiB `repair_required` journal becomes runnable and completes,
  while unrelated errors and noncanonical paths remain blocked; and
- repeated interruption and retry neither duplicate authority nor mutate Submission,
  Task, Attempt, idempotency, or receipt truth.

Run focused Group-service activation and upgrade-coordinator integration suites,
broader qexp integration, repository governance, full Ruff checks, and the complete
promotion preflight.

## Acceptance criteria

- The six observed 65,846–213,184-byte committed Submission Operations need no
  deletion, rewriting, or manual repair.
- Package upgrade plus agent restart automatically retries the recorded 1.3.24
  failure and resumes bounded activation.
- A valid source larger than any individual read chunk progresses across durable
  slices and agent restarts.
- Resource bounds remain explicit, enforced, and independent: 64 KiB maximum read
  chunk, 256 KiB maximum migration-slice I/O, and no 64 KiB Submission validity cap.
- Existing running workloads, admission, scheduling, Group authority, and release
  compatibility behavior remain unchanged.
