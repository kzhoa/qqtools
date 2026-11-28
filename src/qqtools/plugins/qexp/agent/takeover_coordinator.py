"""Advance an accepted Project takeover within ordinary machine agent turns."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from typing import TYPE_CHECKING

from ..runtime.authority_scan import is_path_present, iter_evidence_files
from ..runtime.store import read_json
from .takeover_binding import disable_departing_binding, publish_takeover_binding, restore_departing_binding
from .takeover_intent import (
    _CANCELLABLE_PHASES,
    _TERMINAL_PHASES,
    TakeoverIntent,
    TakeoverIntentConflict,
    finish_takeover_intent,
    read_takeover_intent,
    save_takeover_intent,
)

if TYPE_CHECKING:
    from .context import MachineRuntime
    from .project_io.controller import ProjectIOController

_FROZEN_PHASES = {"verifying_drain", "committing", "shared_committed", "publishing_binding"}
_DRAIN_RETRY_SECONDS = 1.0
_REGISTRATION_STEPS = {
    "accepted": ("reserve", {"reserved": "reserved", "stale": "aborting_stale"}),
    "shared_committed": ("commit", {"committed": "publishing_binding"}),
}
_COMMIT_TRANSITIONS = {
    ("observe", "changed"): "aborting_stale",
    ("observe", "successor_observed"): "shared_committed",
    ("commit", "stale"): "aborting_stale",
    ("commit", "committed"): "shared_committed",
    ("commit", "shared_commit_pending"): "shared_committed",
}


def takeover_frozen_owner(intent: TakeoverIntent) -> tuple[str, str, str] | None:
    """Return the departing request owner when new work must stay fenced."""
    departing = intent.transition.departing
    is_settling_grant = intent.phase in {"cancelling", "aborting_stale"} and intent.drain_observation == "drained"
    if departing is None or (intent.phase not in _FROZEN_PHASES and not is_settling_grant):
        return None
    return departing.runtime_id, intent.transition.project_id, departing.generation


class TakeoverCoordinator:
    """Reconcile the durable takeover using local effects and isolated shared steps."""

    def __init__(self, runtime: MachineRuntime) -> None:
        self.runtime = runtime
        self._commit_plan_id: str | None = None
        self._drain_retry: tuple[str, float] | None = None

    def advance(self, controller: ProjectIOController, registry_revision: int) -> bool:
        """Advance one phase and report whether an active intent retains demand."""
        intent = read_takeover_intent(self.runtime.root)
        if intent is None or intent.phase != "committing" or intent.plan_id != self._commit_plan_id:
            self._commit_plan_id = None
        if intent is None or (self._drain_retry is not None and intent.plan_id != self._drain_retry[0]):
            self._drain_retry = None
        if intent is None:
            return False
        if intent.transition.successor_runtime_id != self.runtime.observe_instance_id():
            self._commit_plan_id = None
            raise TakeoverIntentConflict("Project takeover belongs to a different MachineRuntime identity.")

        phase = intent.phase
        if phase in _TERMINAL_PHASES:
            self._finish(intent)
        elif intent.is_cancel_requested and phase in _CANCELLABLE_PHASES:
            self._commit_plan_id = None
            self._set_phase(intent, "cancelling")
        elif phase == "reserved":
            self._set_phase(intent, "draining", apply=disable_departing_binding)
        elif phase == "draining":
            if self._drain_retry_is_due():
                self._advance_drain(intent)
        elif phase == "verifying_drain":
            self._verify_drain(controller, intent, registry_revision)
        elif phase == "committing":
            self._advance_commit(controller, intent, registry_revision)
        elif phase == "publishing_binding":
            self._set_phase(intent, "releasing", apply=publish_takeover_binding)
        elif phase == "releasing":
            if controller.advance_takeover_registration(intent, registry_revision, "release") == (
                "release",
                "released",
            ):
                self._set_phase(intent, "applying_groups" if intent.group_approvals else "completed")
        elif phase == "applying_groups":
            self._advance_groups(controller, intent, registry_revision)
        elif phase in {"aborting_stale", "cancelling"}:
            self._advance_abort(controller, intent, registry_revision)
        else:
            step, transitions = _REGISTRATION_STEPS[phase]
            result = controller.advance_takeover_registration(intent, registry_revision, step)
            if result is not None and result[0] == step and (next_phase := transitions.get(result[1])) is not None:
                self._set_phase(intent, next_phase)
        return True

    def _set_phase(
        self,
        intent: TakeoverIntent,
        phase: str,
        *,
        apply: Callable[[MachineRuntime, TakeoverIntent], bool] | None = None,
        drain_observation: str | None = None,
    ) -> None:
        updated = replace(intent, phase=phase, drain_observation=drain_observation or intent.drain_observation)
        if updated == intent:
            return
        with self.runtime.agent_lifecycle_guard(blocking=False) as acquired:
            if acquired and read_takeover_intent(self.runtime.root) == intent:
                if apply is None or apply(self.runtime, intent):
                    save_takeover_intent(self.runtime.root, updated, expected=intent)

    def _finish(self, intent: TakeoverIntent) -> None:
        with self.runtime.agent_lifecycle_guard(blocking=False) as acquired:
            if acquired and read_takeover_intent(self.runtime.root) == intent:
                finish_takeover_intent(self.runtime.root, intent)

    def _drain_retry_is_due(self) -> bool:
        return self._drain_retry is None or time.monotonic() >= self._drain_retry[1]

    def _local_drain_is_clear(self, intent: TakeoverIntent) -> bool:
        project_id = intent.transition.project_id
        if intent.transition.departing is None and is_path_present(self.runtime.project_paths(project_id)["root"]):
            raise TakeoverIntentConflict("The unenrolled Project has a retained local partition.")
        if any(pending_project == project_id for pending_project, _ in self.runtime.pending_launch_handoffs):
            return False
        # Attachment publishes active evidence before removing the provisional file.
        for lane in ("provisional", "active", "cpu_provisional", "cpu_active"):
            directory = self.runtime.paths[lane]
            with closing(iter_evidence_files(directory, recursive=True)) as paths:
                for path in paths:
                    reservation = read_json(path).get("reservation")
                    if (
                        path.parent != directory
                        or path.stem in {".", ".."}
                        or not isinstance(reservation, dict)
                        or reservation.get("reservation_id") != path.stem
                        or not isinstance(reservation.get("project_id"), str)
                        or not reservation["project_id"]
                    ):
                        raise ValueError(f"Project takeover cannot verify reservation evidence at {path}.")
                    if reservation["project_id"] == project_id:
                        return False
        return True

    def _advance_drain(self, intent: TakeoverIntent) -> None:
        try:
            is_clear = self._local_drain_is_clear(intent)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._set_phase(intent, "draining", drain_observation="drain_unavailable")
        else:
            if is_clear:
                self._set_phase(intent, "verifying_drain")
            else:
                self._set_phase(intent, "draining", drain_observation="waiting_for_work")

    def _verify_drain(self, controller: ProjectIOController, intent: TakeoverIntent, registry_revision: int) -> None:
        if not self._drain_retry_is_due():
            return
        owner = takeover_frozen_owner(intent)
        if owner is not None and any(
            (request.runtime_id, request.project_id, request.registration_generation) == owner
            for request in controller.executor.unresolved_requests()
        ):
            return
        try:
            is_clear = self._local_drain_is_clear(intent)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._set_phase(intent, "draining", drain_observation="drain_unavailable")
            return
        if not is_clear:
            self._set_phase(intent, "draining", drain_observation="waiting_for_work")
            return
        result = controller.advance_takeover_registration(intent, registry_revision, "drain")
        if result == ("drain", "drained"):
            self._set_phase(intent, "committing", drain_observation="drained")
        elif result in {("drain", "drain_pending"), ("drain", "drain_unavailable")}:
            self._drain_retry = (intent.plan_id, time.monotonic() + _DRAIN_RETRY_SECONDS)
            self._set_phase(intent, "draining", drain_observation=result[1])

    def _advance_commit(self, controller: ProjectIOController, intent: TakeoverIntent, registry_revision: int) -> None:
        # After a lost result or restart, shared observation decides whether a
        # new commit is safe or whether only replay of the successor is needed.
        permitted_plan = self._commit_plan_id
        step = "commit" if permitted_plan == intent.plan_id else "observe"
        self._commit_plan_id = None
        result = controller.advance_takeover_registration(intent, registry_revision, step)
        if result is None:
            self._commit_plan_id = permitted_plan
            return
        if result[0] != step:
            return
        if result == ("observe", "unchanged"):
            self._commit_plan_id = intent.plan_id
        elif (next_phase := _COMMIT_TRANSITIONS.get(result)) is not None:
            self._set_phase(intent, next_phase)

    def _advance_abort(self, controller: ProjectIOController, intent: TakeoverIntent, registry_revision: int) -> None:
        result = controller.advance_takeover_registration(intent, registry_revision, "abort")
        terminal = "cancelled" if intent.phase == "cancelling" else "stale"
        if result == ("abort", "successor_observed"):
            self._set_phase(intent, "shared_committed")
        elif result == ("abort", "departing_current"):
            self._set_phase(intent, terminal, apply=restore_departing_binding)
        elif result == ("abort", "departing_changed"):
            self._set_phase(intent, terminal)

    def _advance_groups(self, controller: ProjectIOController, intent: TakeoverIntent, registry_revision: int) -> None:
        index = len(intent.group_outcomes)
        if index == len(intent.group_approvals):
            self._set_phase(intent, "completed")
            return
        approval = intent.group_approvals[index]
        result = controller.advance_takeover_registration(
            intent, registry_revision, "group", group_name=approval.group_name
        )
        if result is None or result[0] != "group" or result[1] not in {"applied", "stale"}:
            return
        updated = replace(intent, group_outcomes=(*intent.group_outcomes, result[1]))
        with self.runtime.agent_lifecycle_guard(blocking=False) as acquired:
            if acquired and read_takeover_intent(self.runtime.root) == intent:
                save_takeover_intent(self.runtime.root, updated, expected=intent)
