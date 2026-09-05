---
doc_type: adr
status: accepted
updated_at: 2026-09-05
archived_at:
adr_id: ADR-QEXP-0007
supersedes: []
superseded_by:
---

# ADR-QEXP-0007: Isolate CPU-only Tasks in an Opt-in CPU Lane

## Context

qexp is a GPU-first experiment queue. Its existing scheduling, reservation, recovery, and
observability contracts use GPU capacity as the admission resource. Users also need to queue
CPU-only work, such as preprocessing and evaluation, including on GPU hosts with deliberately
reserved spare CPU capacity.

Treating CPU as a second required resource of every Task would change GPU scheduling semantics:
GPU Task CPU consumption is not currently declared, cannot be inferred safely from process
telemetry, and would make existing GPU admission depend on a new and potentially inaccurate CPU
model. Conversely, running unbounded CPU-only work on a GPU host can compete with training
dataloaders, communication, and other host work.

The design must add controlled CPU-only execution without changing the resource contract of
existing GPU Tasks or turning qexp into a general heterogeneous-resource scheduler.

## Decision

qexp has two mutually exclusive scheduling lanes:

1. The GPU lane remains the default and existing behavior. GPU Tasks request GPUs only; GPU
   scheduling, reservation, and admission do not read or reserve CPU capacity.
2. The CPU-only lane accepts only Tasks explicitly submitted with `--gpus 0 --cpus N`, where
   `N` is a positive declared logical CPU slot budget. A Task cannot request both GPU and CPU
   resources in this design.

The CPU-only lane is machine-local and disabled by default on every Machine. A machine agent has
a persistent machine-global `cpu_only_capacity`, initially `0`; a Machine does not automatically
derive or enable this capacity from visible GPUs, logical CPU count, affinity, or utilization.
Only a machine operator may opt in by setting a positive capacity, initially through
`qexp init --cpu-lane-capacity N` or later through the explicit local agent CPU-lane configuration
interface.

CPU-only admission is limited to the configured CPU-only capacity: active CPU-only Task requests
must not exceed it. This capacity is a static budget reserved by the operator, not a measurement
of currently idle CPUs and not a CPU reservation for GPU Tasks. GPU Tasks may run while CPU-only
Tasks run on the same Machine, but neither lane's admission consumes or gates the other.

## Consequences

- Existing GPU-only commands, reservations, scheduling throughput, and compatibility behavior do
  not gain a CPU dependency.
- CPU-only work can run concurrently, including on GPU hosts, but only after an operator has
  explicitly allocated a safe CPU-only budget for that host.
- qexp guarantees that its CPU-only Tasks do not exceed `cpu_only_capacity`; it cannot guarantee
  that an incorrectly sized budget will not contend with CPU work performed by GPU Tasks or other
  processes.
- Reducing `cpu_only_capacity` below active CPU-only reservations, including setting it to zero,
  must be rejected until the CPU-only lane is drained. Configuration changes must never silently
  cancel Tasks, remove reservations, or terminate processes.
- Mixed GPU-plus-CPU Task requests, CPU affinity, cgroup enforcement, dynamic CPU utilization
  inference, Group CPU quotas, and general multi-resource scheduling remain out of scope.

## References

- [CPU-only Task Lane pitch](../../pitch/arxiv/053-qexp-cpu-only-task-scheduling.md)
- [ADR-QEXP-0002: Ready Index and Portable Work Budget](0002-ready-index-and-portable-work-budget.md)
- [qexp Product Spec](../../spec/qexp_product_spec.md)
- [qexp Runtime Spec](../../spec/qexp_runtime_spec.md)
