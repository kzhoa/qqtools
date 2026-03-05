# ADR-QPIPELINE-0001: Runner Boundary Ownership

- Status: Accepted
- Date: 2026-03-05
- Owners: qpipeline maintainers

## Context

The runner supports two execution modes (`epoch` and `step`) and two stopping boundaries (`max_epochs` and `max_steps`).
Historically, logs stated that one boundary was ignored in a given mode, while actual stopping behavior could still be constrained by both boundaries in practice.

This created a semantics mismatch:
- user expectation from logs/config comments
- actual stop condition in the execution loop

## Decision

Boundary mutual exclusion is a business policy and is owned by the orchestration layer (`train_runner`), not by the execution engine (`RunningAgent`).

Policy:
- In `run_mode='epoch'`: keep `max_epochs`, ignore `max_steps`.
- In `run_mode='step'`: keep `max_steps`, ignore `max_epochs`.

Implementation rule:
- `train_runner` resolves effective boundaries before creating `RunConfig`.
- `RunningAgent` remains policy-agnostic and only enforces the boundaries it receives.

## Consequences

Positive:
- Log semantics and actual stop behavior are aligned.
- Clear separation of concerns: policy in orchestrator, mechanics in agent.
- Future policy changes can be made in one place without modifying loop internals.

Trade-offs:
- Callers passing both boundaries must understand that one is intentionally discarded based on mode.
- Policy must be documented clearly in PRD and release notes.

## Non-Goals

- This ADR does not redefine evaluation/checkpoint trigger semantics.
- This ADR does not change early-stopping logic.

## Follow-ups

- Keep PRD concise and reference ADRs for rationale/history.
- Add/maintain tests that verify effective boundary behavior in both modes.
