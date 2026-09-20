"""Finite semantic model for the reviewed Group-removal G0 contract.

This is a one-Task, one-remove-operation model with a fixed two-generation
horizon.  It models lock-boundary interleavings only: crash/restart does not
change durable state.  It does not prove fsync behavior, producer fairness,
storage behavior, or rollout behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Phase = Literal["terminal", "queued", "deleted"]
Obligation = Literal["none", "in_flight", "effected", "resolved", "aborted"]
MAX_GENERATIONS = 2


@dataclass(frozen=True, slots=True)
class GroupRemovalState:
    """Immutable state for one Group removal and its publication receipt."""

    phase: Phase = "terminal"
    generation: int = 0
    obligation: Obligation = "none"
    consumed_generation: int = 0
    receipt_phase: Phase = "terminal"
    receipt_generation: int = 0
    outcome_in_task: frozenset[int] = frozenset()
    outcome_in_cleanup: frozenset[int] = frozenset()
    counted: frozenset[int] = frozenset()
    completed: bool = False


INITIAL_STATE = GroupRemovalState()
Successor = tuple[str, GroupRemovalState]


def successors(
    state: GroupRemovalState,
    *,
    allow_inflight_ack: bool = False,
    allow_unwitnessed_delete: bool = False,
    reuse_aborted_ticket: bool = False,
) -> tuple[Successor, ...]:
    """Return deterministic, state-changing actions from ``state``.

    The three keyword switches deliberately add only the mutation-test defect
    described by their names.  Correct completed states have no successors;
    the reuse defect alone admits its specified stale post-completion action.
    """

    transitions: list[Successor] = []

    def add(action: str, candidate: GroupRemovalState) -> None:
        if candidate != state:
            transitions.append((action, candidate))

    can_reuse = (
        reuse_aborted_ticket
        and state.phase == "terminal"
        and state.obligation == "aborted"
        and state.consumed_generation == state.generation
    )
    if state.completed:
        if can_reuse:
            add("reuse_aborted_ticket", replace(state, phase="queued", obligation="effected"))
        return tuple(transitions)

    if (
        state.phase == "terminal"
        and state.obligation in {"none", "resolved", "aborted"}
        and state.consumed_generation == state.generation
        and state.generation < MAX_GENERATIONS
    ):
        add("publish", replace(state, generation=state.generation + 1, obligation="in_flight"))

    if state.phase == "terminal" and state.obligation == "in_flight":
        add("mutation_effect", replace(state, phase="queued", obligation="effected"))

    if state.obligation == "effected":
        add("resolve_effect", replace(state, obligation="resolved"))

    if state.obligation == "in_flight":
        add("abort", replace(state, obligation="aborted"))

    recheck_obligations = {"none", "resolved", "aborted"}
    if allow_inflight_ack:
        recheck_obligations.add("in_flight")
    if state.obligation in recheck_obligations:
        add(
            "recheck",
            replace(
                state,
                consumed_generation=state.generation,
                receipt_phase=state.phase,
                receipt_generation=state.generation,
                counted=state.counted | state.outcome_in_task | state.outcome_in_cleanup,
            ),
        )

    if state.phase == "queued" and state.obligation == "resolved":
        add(
            "converge",
            replace(state, phase="terminal", outcome_in_task=state.outcome_in_task | {state.generation}),
        )

    if state.phase == "terminal" and state.outcome_in_task:
        add(
            "cleanup_handoff",
            replace(state, outcome_in_cleanup=state.outcome_in_cleanup | state.outcome_in_task),
        )

    if state.phase == "terminal" and (state.outcome_in_task <= state.outcome_in_cleanup or allow_unwitnessed_delete):
        add("cleanup_delete", replace(state, phase="deleted", outcome_in_task=frozenset()))

    complete_obligations = {"none", "resolved", "aborted"}
    if allow_inflight_ack:
        complete_obligations.update({"in_flight", "effected"})
    if (
        state.obligation in complete_obligations
        and state.consumed_generation == state.generation
        and state.receipt_generation == state.generation
        and state.receipt_phase == state.phase
        and state.phase != "queued"
        and (state.outcome_in_task | state.outcome_in_cleanup) <= state.counted
    ):
        add("complete", replace(state, completed=True))

    if can_reuse:
        add("reuse_aborted_ticket", replace(state, phase="queued", obligation="effected"))

    return tuple(transitions)
