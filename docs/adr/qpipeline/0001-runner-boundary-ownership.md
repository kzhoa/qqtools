# ADR-QPIPELINE-0001: Runner Boundary Ownership

- Status: Accepted
- Date: 2026-04-21
- Owners: qpipeline maintainers

## Context

The runner supports two execution modes (`epoch` and `step`) and two stopping boundaries (`max_epochs` and `max_steps`).
Historically, logs stated that one boundary was ignored in a given mode, while actual stopping behavior could still be constrained by both boundaries in practice.

This created a semantics mismatch:
- user expectation from logs/config comments
- actual stop condition in the execution loop

Later, the orchestration layer enforced full mutual exclusion to remove that mismatch.
That policy made `run_mode='step'` ignore `max_epochs` entirely, but it also removed a previously useful bounded-step workflow:
- step-oriented progress/evaluation cadence driven by optimizer updates
- an optional epoch cap to stop overly long runs on very large or irregular datasets

## Decision

Boundary resolution remains a business policy owned by the orchestration layer (`train_runner`), not by the execution engine (`RunningAgent`).

Policy:
- In `run_mode='epoch'`: keep `max_epochs`, ignore `max_steps`.
- In `run_mode='step'`: require `max_steps` as the primary boundary, and optionally keep `max_epochs` as a secondary stopping boundary.

Implementation rule:
- `train_runner` resolves effective boundaries before creating `RunConfig`.
- `RunningAgent` remains policy-agnostic and only enforces the boundaries it receives.

## Consequences

Positive:
- Log semantics and actual stop behavior are aligned.
- Clear separation of concerns: policy in orchestrator, mechanics in agent.
- Future policy changes can be made in one place without modifying loop internals.
- Step-mode callers can combine optimizer-step cadence with an explicit epoch cap.

Trade-offs:
- Step-mode callers passing both boundaries must understand that stopping is now controlled by whichever boundary arrives first.
- Policy must be documented clearly in PRD and release notes.

## Non-Goals

- This ADR does not redefine evaluation/checkpoint trigger semantics.
- This ADR does not change early-stopping logic.

## Follow-ups

- Keep PRD concise and reference ADRs for rationale/history.
- Add/maintain tests that verify effective boundary behavior in both modes.
