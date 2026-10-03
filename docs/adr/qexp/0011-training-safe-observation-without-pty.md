---
doc_type: adr
adr_id: ADR-QEXP-0011
status: accepted
updated_at: 2026-09-29
archived_at:
supersedes: []
superseded_by:
---

# ADR-QEXP-0011: Training-Safe Observation Without PTY Capture

## Context

The original proposal sought to reproduce a training application's native Rich
terminal display through a PTY broker and a read-only tmux relay. A bounded
node-local spool would retain terminal bytes while shared-log delivery lagged.

On 2026-09-23, the controlling requirement was clarified: observation is
auxiliary. Training performance and continuity take precedence over display
availability and observation-data retention. The PTY/spool proposal could not
meet that requirement:

- A finite, lossless spool eventually fills when its sink stalls. Stopping PTY
  draining then backpressures the application. More buffering only delays this
  boundary.
- Dropping captured bytes avoids spool backpressure but does not remove the
  broker's ownership of the application's output path. Broker suspension or
  failure can still affect training output.
- Replaying a bounded tail of ANSI bytes cannot guarantee reconstruction of an
  arbitrary terminal screen after the viewer is destroyed.

The historical fault-injection experiments demonstrated both the benefit of
buffering short sink delays and blocking once the finite spool filled. They did
not establish production lifecycle safety or GPU-training throughput guarantees.
No production PTY path was shipped by that proposal.

## Decision

1. Reject PTY capture and durable terminal-byte spooling as the implementation
   of auxiliary qexp live observation. Do not introduce the proposed output-mode
   selector, PTY broker, delivery obligations, or log-delivery cleanup gates.
2. Preserve the detached runner and guardian, existing cancellation authority,
   and combined stdout/stderr file output. A tmux window remains an observer;
   enabling observation does not change the application's TTY environment.
3. Use framework-neutral, bounded, disposable progress snapshots and an
   independent viewer. qexp owns presentation through Rich or text; the viewer
   does not reproduce the application's original terminal byte stream.
4. Observation delivery and rendering cannot authorize execution, determine
   Task success, or send process-control signals to training. A stalled writer,
   missing viewer, or unavailable snapshot must degrade observation without
   creating a training wait or termination dependency.
5. Keep observation work bounded and optional, preserve progress-v1
   compatibility, and validate training-side overhead and failure isolation
   against the current public contracts. This decision does not claim zero
   overhead or waive measured acceptance requirements.

## Consequences

Training does not gain a new output broker or durable observation-delivery
dependency. Multiple clients can observe the same training run, and a viewer
can be recreated from available structured state without replaying terminal
history. Applications without structured reporting retain ordinary log viewing.

The tradeoff is explicit: qexp displays reported progress and metrics, not every
field or visual detail of the application's native interface. Snapshots may be
stale, incomplete, or unavailable; they are not a lossless historical metric
store. Existing application-log I/O remains part of the execution baseline;
this ADR does not claim to eliminate that baseline's storage dependencies.

The historical PTY experiments remain diagnostic evidence, not an active
implementation plan. A future native-terminal feature would require a separate
decision explaining how its output ownership and failure behavior satisfy the
training-continuity requirement.

## References

- [Live application progress contract](../../spec/qexp_live_progress.md)
- [Product specification](../../spec/qexp_product_spec.md)
- [Runtime specification](../../spec/qexp_runtime_spec.md)
- [Training-safe live observation validation](../../development/qexp-live-progress-validation.md)

These public documents own behavior and validation evidence. This decision does
not depend on private pitch or experiment files.
