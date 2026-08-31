---
doc_type: adr
adr_id: ADR-QEXP-0004
status: accepted
updated_at: 2026-08-31
archived_at:
supersedes: []
superseded_by:
---

# ADR-QEXP-0004: Separate Machine Identity from Task Placement

## Context

qexp currently accepts a global `--machine` option and resolves it before `submit`. The value is
used as the CLI's current machine identity and, because a Task defaults its `home_machine` to the
submitting machine, also influences Task placement.

This overload creates two problems:

- `qexp --machine g4 submit ...` naturally reads as “submit to g4”, although the implementation
  interprets it as “execute this command as g4”.
- normal `submit` activation verifies that the selected identity has a matching local project
  binding, but `submit --no-activate` skips that activation path and can persist a Task whose
  submitting identity and home machine were supplied by the caller rather than derived from the
  local binding.

The product also has a legitimate placement requirement that the overloaded option cannot express
safely: a user operating on g3 may need to create a Task whose only eligible execution machine is
g4. In that scenario, g3 remains the submitting machine and g4 is the Task home machine.

Machine identity, Task placement policy, and the machine that eventually wins an execution claim
are separate facts and require separate contracts.

## Decision

### Machine identity is derived local authority

For operational commands, including `submit`, qexp derives the current machine identity from the
unique local `MachineRuntime` project binding selected by the canonical shared root and stable
project ID. A caller-provided machine name is not an authority source.

Operational context separates project location from identity. `--shared-root`, `QEXP_SHARED_ROOT`,
and the saved context shared root may locate the Project. `--machine` and `QEXP_MACHINE` are
compatibility assertions only: each must equal the binding-derived machine or the command fails
before mutation. The saved context machine/runtime root, `--runtime-root`, and `QEXP_RUNTIME_ROOT`
are non-authoritative legacy inputs; they do not select identity, activation target, or reservation
ownership. `--machine-runtime-root` locates the local registry but does not provide a machine
identity.

`--no-activate` suppresses the request to start the local agent only. It does not bypass project
binding resolution or machine-identity validation.

`--machine` remains the identity-establishment term for commands such as `init` and `migrate`.
`qexp use` selects only a local default shared root; during 1.3.13 it accepts machine/runtime
inputs only as warned compatibility no-ops (`QQTOOLS-COMPAT-0002`). During compatibility, an
explicitly supplied global `--machine` on an operational command is an identity assertion:

- if it equals the machine name derived from the local binding, the command may continue;
- if it differs, the command fails before mutation and directs the user to `--home-machine` when
  Task placement was intended.

### Task home is explicit placement policy

`qexp submit` gains `--home-machine <name>`. Omission, or the explicit value `current`, resolves to
the verified submitting machine. A different value selects the Task's home machine without changing
the submitting identity and without impersonating or remotely activating that machine.

For a private Task, the home machine is the complete execution-eligibility set. Therefore an
ungrouped Task may use a non-current home machine:

```bash
# Executed on g3; only g4 may execute the Task.
qexp submit --home-machine g4 -- python train.py
```

This records `original_submitting_machine = g3`, `home_machine = g4`, and
`sharing_mode = private`. The target must have valid current-generation machine metadata in the
Project, but its agent need not be online when the Task is committed. qexp does not remotely start
the target agent or transfer project files. Shared target metadata is not proof that the target has
an enabled local binding or an available agent; until the target registers locally and its agent is
available, the Task remains queued at home.

Spillover placement continues to require a Group. The explicit home and every fallback candidate
must be authorized by the Group Worker Set. A machine does not become a Group worker merely because
the submission command originated there.

### Group resource pools remain explicitly dynamic

The Group Worker Set remains the dynamically changeable resource pool. `qexp group machines add`,
`drain`, and `remove` retain their existing safety semantics, including protection against stranding
queued work; active membership still controls subsequent claim eligibility.

Submission is not a Worker Set transition. A submit, regardless of local or remote home, never
adds its origin machine to a Group. Batch manifest `group.workers` remains an explicit, audited
Worker Set addition path. A single-task caller must use the existing Group creation or member
management commands before relying on a new worker. This removes implicit execution authorization
without reducing the resource pool's ability to expand, drain, or contract.

### Persisted facts remain distinct

Submission and execution records preserve these separate values:

- `original_submitting_machine`: verified local identity that created the Submission Operation;
- `placement_policy.home_machine`: user-selected first and, for private placement, exclusive
  execution machine;
- Attempt execution machine: machine that wins a valid claim.

Idempotent retries reuse the initially resolved submission context and do not reinterpret
`current` or a caller's later local identity.

## Compatibility Decision

The protected `qexp init -> qexp submit -- <command>` workflow remains unchanged: omission of both
identity and placement options resolves the local binding and keeps the local machine as Task home.
Existing scripts that explicitly pass the correct local `--machine` remain valid during the
compatibility period because the option is treated as an assertion.

The documented behavior that a submission from the current machine may atomically add that machine
to an open Group is intentionally retired. Group growth remains supported, but must be explicit via
Group member operations or a manifest's explicit worker additions. This compatibility change is
necessary because implicit admission turns task creation into unrequested execution authorization.

A previously successful mismatch, including the `--no-activate` path that could persist work while
claiming another machine's identity, becomes an error. This is an approved corrective compatibility
change: the old behavior violated project-binding ownership and produced false submission audit
facts. It is not retained as a remote-submission interface.

No persisted schema field is repurposed. The existing distinct submitting-machine and home-machine
fields retain their meanings.

## Consequences

- CLI identity cannot be changed into Task placement by changing command context.
- `--no-activate` no longer provides an identity-validation bypass.
- A user can create a private g4-only Task while operating on g3 without YAML or a Group.
- Offline target machines can accumulate private home work; execution waits until the target's
  locally registered agent becomes available.
- Group resource pools remain dynamically expandable, drainable, and removable through explicit
  Worker Set operations; submission itself is no longer a resource-pool mutation.
- Cross-machine submission does not imply remote shell, remote wake-up, file transfer, or source
  snapshot behavior.
- CLI help, README examples, product specification, runtime invariants, submission validation, and
  compatibility tests must be updated together.
- Reverse binding lookup becomes part of operational command context resolution and must fail
  closed on missing or ambiguous local bindings.

## Rejected Alternatives

- **Interpret global `--machine` as the Task destination.** This changes an identity option into
  placement, loses the real submitting machine, and breaks agent and local-runtime ownership.
- **Permit caller-selected identity only with `--no-activate`.** Activation policy is unrelated to
  identity authority and cannot safely serve as its validation boundary.
- **Retain origin auto-admission for local-home submissions only.** The authorization effect is
  still implicit and makes the meaning of a submit depend on placement details rather than an
  explicit Group membership action.
- **Require a Group for every non-current private home.** A private Task already has the bounded
  eligibility set `{home_machine}`; requiring a Group adds ceremony without adding authorization.
- **Model g4-only execution as g3 home with g4 fallback.** The home machine remains eligible by
  definition, so this does not meet the requirement that only g4 may execute the Task.
- **Use a one-Task batch manifest as the ordinary interface.** The runtime may share the normalized
  TaskSpec pipeline, but single-Task submission must remain YAML-free.

## Related Requirement

- [qexp-explicit-home-machine-submission.md](../../pitch/qexp-explicit-home-machine-submission.md)
