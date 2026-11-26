"""Closed Project I/O parameters and shared steps for an accepted takeover."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..runtime.records import validate_group_name
from ..runtime.takeover_registration import (
    TakeoverRegistration,
    TakeoverRegistrationConflict,
    abort_takeover_registration,
    commit_takeover_registration,
    observe_takeover_registration,
    release_takeover_registration,
    reserve_takeover_registration,
)
from .takeover_drain import observe_takeover_drain
from .takeover_groups import apply_takeover_group
from .takeover_intent import TakeoverIntent, TakeoverIntentConflict, read_takeover_intent

_PLAN_ID = re.compile(r"[0-9a-f]{64}")
_STEP_OUTCOMES = {
    "reserve": {"reserved", "stale", "intent_changed"},
    "observe": {"successor_observed", "unchanged", "changed", "intent_changed"},
    "drain": {"drained", "drain_pending", "drain_unavailable", "intent_changed"},
    "group": {"applied", "stale", "intent_changed"},
    "commit": {"committed", "shared_commit_pending", "stale", "intent_changed"},
    "release": {"released", "intent_changed"},
    "abort": {"successor_observed", "departing_current", "departing_changed", "intent_changed"},
}


class _TakeoverIntentChanged(RuntimeError):
    """The durable intent changed after the worker entered a shared step."""


def _parse_parameters(value: object) -> tuple[str, str, TakeoverRegistration, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError("takeover_registration parameters must be an object.")
    fields = {"plan_id", "step", "transition"}
    if value.get("step") == "group":
        fields.add("group_name")
    if value.keys() != fields:
        raise ValueError("takeover_registration parameters have missing or unknown fields.")
    plan_id = value["plan_id"]
    if not isinstance(plan_id, str) or _PLAN_ID.fullmatch(plan_id) is None:
        raise ValueError("takeover_registration plan_id must be 64 lowercase hexadecimal characters.")
    step = value["step"]
    if not isinstance(step, str) or step not in _STEP_OUTCOMES:
        raise ValueError("takeover_registration step is invalid.")
    raw_transition = value["transition"]
    if not isinstance(raw_transition, Mapping):
        raise ValueError("takeover_registration transition must be an object.")
    transition = dict(raw_transition)
    for field in ("target", "departing"):
        witness = transition.get(field)
        if isinstance(witness, Mapping):
            transition[field] = dict(witness)
    group_name = value.get("group_name")
    if step == "group":
        if group_name is None:
            raise ValueError("takeover group step requires a Group name.")
        validate_group_name(group_name)
    return plan_id, step, TakeoverRegistration.from_dict(transition), group_name


def takeover_registration_parameters(value: object) -> dict[str, Any]:
    """Validate and detach the exact transition for one request."""
    plan_id, step, transition, group_name = _parse_parameters(value)
    parameters = {"plan_id": plan_id, "step": step, "transition": transition.to_dict()}
    if group_name is not None:
        parameters["group_name"] = group_name
    return parameters


def takeover_registration_evidence(step: str, value: Mapping[str, Any]) -> dict[str, str]:
    """Accept only the bounded observations available to the requested step."""
    if value.keys() != {"outcome"}:
        raise ValueError("takeover_registration evidence has missing or unknown fields.")
    outcome = value["outcome"]
    if not isinstance(outcome, str) or outcome not in _STEP_OUTCOMES[step]:
        raise ValueError("takeover_registration evidence outcome does not match the step.")
    return {"outcome": outcome}


def _allows_step(intent: TakeoverIntent, step: str) -> bool:
    phase = intent.phase
    if step == "reserve":
        return phase == "accepted" and not intent.is_cancel_requested
    if step == "drain":
        return phase == "verifying_drain" and not intent.is_cancel_requested
    if step == "group":
        return phase == "applying_groups"
    if step == "commit":
        return phase == "shared_committed" or (phase == "committing" and not intent.is_cancel_requested)
    if step == "release":
        return phase == "releasing"
    if step == "abort":
        return phase in {"cancelling", "aborting_stale"}
    return phase not in {"completed", "cancelled"}


def run_takeover_step(
    runtime_root: Path,
    parameters: Mapping[str, Any],
    *,
    before_mutation: Callable[[], None],
) -> dict[str, str]:
    """Run one shared takeover step while the exact local intent authorizes it."""
    plan_id, step, transition, group_name = _parse_parameters(parameters)
    if transition.successor_runtime_root != runtime_root:
        return {"outcome": "intent_changed"}

    def active_intent() -> TakeoverIntent | None:
        try:
            return read_takeover_intent(runtime_root)
        except TakeoverIntentConflict:
            return None

    intent = active_intent()
    if intent is None or intent.plan_id != plan_id or intent.transition != transition or not _allows_step(intent, step):
        return {"outcome": "intent_changed"}

    if transition.shared_root.resolve() != transition.shared_root:
        raise ValueError("takeover_registration shared_root no longer resolves to the canonical Project root.")

    def require_current() -> None:
        if active_intent() != intent:
            raise _TakeoverIntentChanged
        before_mutation()

    if step == "drain":
        outcome = observe_takeover_drain(transition)
        return {"outcome": outcome if active_intent() == intent else "intent_changed"}
    if step == "observe":
        observed = observe_takeover_registration(transition)
        return {"outcome": "successor_observed" if observed == "committed" else observed}
    if step == "group":
        index = len(intent.group_outcomes)
        if index >= len(intent.group_approvals) or intent.group_approvals[index].group_name != group_name:
            return {"outcome": "intent_changed"}
        try:
            outcome = apply_takeover_group(
                transition, intent.group_approvals[index], plan_id, before_mutation=require_current
            )
            return {"outcome": outcome}
        except _TakeoverIntentChanged:
            return {"outcome": "intent_changed"}
    if step == "release":
        release_takeover_registration(transition, before_mutation=require_current)
        return {"outcome": "released"}
    if step == "abort":
        try:
            aborted = abort_takeover_registration(transition, before_mutation=require_current)
        except _TakeoverIntentChanged:
            return {"outcome": "intent_changed"}
        return {"outcome": "successor_observed" if aborted == "committed" else aborted}
    try:
        if step == "reserve":
            reserve_takeover_registration(transition, before_mutation=require_current)
            return {"outcome": "reserved"}
        commit_takeover_registration(
            transition, before_mutation=require_current, is_replay_only=intent.phase == "shared_committed"
        )
        return {"outcome": "committed"}
    except TakeoverRegistrationConflict:
        # Re-probing also verifies that Project identity and capability still
        # allow a definitive decision about the registration conflict.
        observed = observe_takeover_registration(transition)
        if intent.phase == "shared_committed" and observed != "committed":
            raise
        outcome = "shared_commit_pending" if step == "commit" and observed == "committed" else "stale"
        return {"outcome": outcome}
