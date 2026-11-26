"""Durable machine-local coordination records for Project takeover."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime.records import validate_identifier
from ..runtime.registration_protocol import MAX_REGISTRATION_RECORD_BYTES, _read_protocol_record
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, check_mutation_fence, require_json_size
from ..runtime.takeover_registration import TakeoverRegistration
from ..takeover_contract import TAKEOVER_CANCELLABLE_PHASES as _CANCELLABLE_PHASES
from ..takeover_contract import TAKEOVER_NEXT_PHASES as _NEXT_PHASES
from ..takeover_contract import TAKEOVER_PHASES as _PHASES
from ..takeover_contract import TAKEOVER_TERMINAL_PHASES as _TERMINAL_PHASES
from .takeover_plan import TakeoverGroupApproval, takeover_plan_nonce

_INTENT_NAME = "project-registration-replacement.json"
_INTENT_KEY = "project_takeover"
_LEGACY_KEY = "project_registration_replacement"
_VERSION = 1
_FIELDS = {
    "version",
    "plan_id",
    "transition",
    "original_enabled",
    "phase",
    "is_cancel_requested",
    "group_approvals",
    "group_outcomes",
    "drain_observation",
}
_LONGEST_PHASE = max(_PHASES, key=len)
_DRAIN_OBSERVATIONS = frozenset({"waiting_for_work", "drain_pending", "drain_unavailable", "drained"})


class TakeoverIntentConflict(RuntimeError):
    """The takeover record, runtime ownership, or phase differs from the expected state."""


@dataclass(frozen=True, slots=True)
class TakeoverIntent:
    """The exact inputs and durable progress of one accepted Project takeover."""

    plan_id: str
    transition: TakeoverRegistration
    original_enabled: bool
    phase: str = "accepted"
    is_cancel_requested: bool = False
    group_approvals: tuple[TakeoverGroupApproval, ...] = ()
    group_outcomes: tuple[str, ...] = ()
    drain_observation: str | None = None

    def __post_init__(self) -> None:
        takeover_plan_nonce(self.plan_id)
        if not isinstance(self.transition, TakeoverRegistration):
            raise ValueError("Project takeover transition must be a registration transaction.")
        shared_root = str(self.transition.shared_root)
        if (
            len(self.transition.project_id) > 128
            or len(shared_root) > 4096
            or len(self.transition.successor_generation) > 128
            or re.fullmatch(r"[0-9a-f]{64}", self.transition.successor_runtime_id) is None
        ):
            raise ValueError("Project takeover identity cannot be carried by Project I/O.")
        validate_identifier(self.transition.successor_generation, "successor_generation")
        # The worker also repeats the Project scope outside the transition and
        # adds its process identity or result. Leave room for those fields.
        require_json_size(
            {"shared_root": shared_root, "transition": self.transition.to_dict()},
            max_bytes=MAX_REGISTRATION_RECORD_BYTES - 8192,
            record_type="takeover_transition",
        )
        if type(self.group_approvals) is not tuple or any(
            not isinstance(approval, TakeoverGroupApproval)
            or approval.machine_name != self.transition.target.machine_name
            for approval in self.group_approvals
        ):
            raise ValueError("Project takeover intent has invalid Group approvals.")
        names = [approval.group_name for approval in self.group_approvals]
        if names != sorted(set(names)):
            raise ValueError("Project takeover intent requires unique Group approvals in name order.")
        if type(self.original_enabled) is not bool or type(self.is_cancel_requested) is not bool:
            raise ValueError("Project takeover enablement and cancellation flags must be booleans.")
        if not isinstance(self.phase, str) or self.phase not in _PHASES:
            raise ValueError("Project takeover intent has an invalid phase.")
        if self.phase in ("cancelling", "cancelled") and not self.is_cancel_requested:
            raise ValueError("Project takeover cancellation phases require a cancellation request.")
        if (
            type(self.group_outcomes) is not tuple
            or len(self.group_outcomes) > len(self.group_approvals)
            or any(outcome not in ("applied", "stale") for outcome in self.group_outcomes)
            or (self.group_outcomes and self.phase not in {"applying_groups", "completed"})
            or (self.phase == "completed" and len(self.group_outcomes) != len(self.group_approvals))
        ):
            raise ValueError("Project takeover intent has invalid Group outcomes.")
        if self.drain_observation is not None and self.drain_observation not in _DRAIN_OBSERVATIONS:
            raise ValueError("Project takeover intent has an invalid drain observation.")
        # Any accepted record must still fit after switching to a longer phase.
        document = self.to_dict()
        document[_INTENT_KEY]["phase"] = _LONGEST_PHASE
        document[_INTENT_KEY]["is_cancel_requested"] = False
        document[_INTENT_KEY]["group_outcomes"] = ["applied"] * len(self.group_approvals)
        document[_INTENT_KEY]["drain_observation"] = max(_DRAIN_OBSERVATIONS, key=len)
        require_json_size(document, max_bytes=MAX_REGISTRATION_RECORD_BYTES, record_type=_INTENT_KEY)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the versioned Project takeover envelope."""
        return {
            _INTENT_KEY: {
                "version": _VERSION,
                "plan_id": self.plan_id,
                "transition": self.transition.to_dict(),
                "original_enabled": self.original_enabled,
                "phase": self.phase,
                "is_cancel_requested": self.is_cancel_requested,
                "group_approvals": [approval.to_dict() for approval in self.group_approvals],
                "group_outcomes": list(self.group_outcomes),
                "drain_observation": self.drain_observation,
            }
        }

    @classmethod
    def from_dict(cls, value: object) -> TakeoverIntent:
        """Validate and reconstruct a versioned Project takeover envelope."""
        if not isinstance(value, dict) or value.keys() != {_INTENT_KEY}:
            raise ValueError("Project takeover record has an invalid envelope.")
        payload = value[_INTENT_KEY]
        if not isinstance(payload, dict) or payload.keys() != _FIELDS:
            raise ValueError("Project takeover record has an invalid field set.")
        if type(payload["version"]) is not int or payload["version"] != _VERSION:
            raise ValueError("Project takeover record has an unsupported version.")
        if not isinstance(payload["group_approvals"], list) or not isinstance(payload["group_outcomes"], list):
            raise ValueError("Project takeover Group approvals and outcomes must be lists.")
        return cls(
            plan_id=payload["plan_id"],
            transition=TakeoverRegistration.from_dict(payload["transition"]),
            original_enabled=payload["original_enabled"],
            phase=payload["phase"],
            is_cancel_requested=payload["is_cancel_requested"],
            group_approvals=tuple(TakeoverGroupApproval.from_dict(value) for value in payload["group_approvals"]),
            group_outcomes=tuple(payload["group_outcomes"]),
            drain_observation=payload["drain_observation"],
        )


def _require_owner(machine_root: Path, intent: TakeoverIntent) -> None:
    if machine_root != intent.transition.successor_runtime_root:
        raise TakeoverIntentConflict("Project takeover belongs to a different MachineRuntime root.")


def _read_active(machine_root: Path, *, allow_legacy: bool) -> TakeoverIntent | None:
    document = _read_protocol_record(machine_root / _INTENT_NAME)
    if document is None:
        return None
    if document.keys() == {_LEGACY_KEY} and isinstance(document[_LEGACY_KEY], dict):
        if allow_legacy:
            return None
        raise TakeoverIntentConflict("A Project registration replacement owns the active intent record.")
    intent = TakeoverIntent.from_dict(document)
    _require_owner(machine_root, intent)
    return intent


def read_takeover_intent(machine_root: Path) -> TakeoverIntent | None:
    """Read this runtime's active takeover, leaving legacy replacement records to their owner."""
    return _read_active(machine_root, allow_legacy=True)


def _validate_update(expected: TakeoverIntent, intent: TakeoverIntent) -> None:
    if (expected.plan_id, expected.transition, expected.original_enabled, expected.group_approvals) != (
        intent.plan_id,
        intent.transition,
        intent.original_enabled,
        intent.group_approvals,
    ):
        raise TakeoverIntentConflict(
            "Project takeover plan, registration transition, original enablement, or Group approval changed."
        )
    if expected.group_outcomes != intent.group_outcomes and (
        expected.phase != "applying_groups"
        or intent.phase != "applying_groups"
        or expected.group_outcomes != intent.group_outcomes[:-1]
    ):
        raise TakeoverIntentConflict("Project takeover Group outcomes must advance one approved change at a time.")
    if expected.phase != intent.phase and intent.phase not in _NEXT_PHASES.get(expected.phase, ()):
        raise TakeoverIntentConflict(f"Project takeover cannot advance from {expected.phase} to {intent.phase}.")
    if expected.is_cancel_requested and not intent.is_cancel_requested:
        raise TakeoverIntentConflict("Project takeover cannot clear its cancellation request.")
    if (
        intent.is_cancel_requested
        and not expected.is_cancel_requested
        and (expected.phase not in _CANCELLABLE_PHASES or intent.phase == "shared_committed")
    ):
        raise TakeoverIntentConflict("Project takeover cannot request cancellation after shared commit.")


def _write_intent(path: Path, intent: TakeoverIntent) -> None:
    document = intent.to_dict()
    require_json_size(document, max_bytes=MAX_REGISTRATION_RECORD_BYTES, record_type=_INTENT_KEY)
    atomic_replace(path, document)


def save_takeover_intent(machine_root: Path, intent: TakeoverIntent, *, expected: TakeoverIntent | None) -> None:
    """Compare and save the active takeover while the caller holds the lifecycle lock."""
    _require_owner(machine_root, intent)
    current = _read_active(machine_root, allow_legacy=False)
    if current != expected:
        raise TakeoverIntentConflict("Project takeover active intent changed.")
    if current is None:
        if read_takeover_result(machine_root, intent.plan_id) is not None:
            raise TakeoverIntentConflict("This Project takeover plan already has a terminal result.")
        if intent.phase != "accepted" or intent.is_cancel_requested:
            raise TakeoverIntentConflict(
                "A new Project takeover must start at accepted without a cancellation request."
            )
    else:
        _validate_update(current, intent)
        if current == intent:
            DurableIO().sync_directory(machine_root, "takeover_intent_replay")
            return
    _write_intent(machine_root / _INTENT_NAME, intent)


def _result_path(machine_root: Path, plan_id: str) -> Path:
    takeover_plan_nonce(plan_id)
    directory = machine_root / "takeover-results"
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Project takeover results path is not a directory: {directory}")
    return directory / f"{plan_id}.json"


def _require_terminal(intent: TakeoverIntent) -> None:
    if intent.phase not in _TERMINAL_PHASES:
        raise TakeoverIntentConflict("Project takeover result requires a terminal phase.")


def _read_result(machine_root: Path, path: Path) -> TakeoverIntent | None:
    document = _read_protocol_record(path)
    if document is None:
        return None
    result = TakeoverIntent.from_dict(document)
    _require_owner(machine_root, result)
    _require_terminal(result)
    if result.plan_id != path.stem:
        raise TakeoverIntentConflict("Project takeover result belongs to a different plan ID.")
    return result


def read_takeover_result(machine_root: Path, plan_id: str) -> TakeoverIntent | None:
    """Read the durable terminal receipt for one exact takeover plan."""
    return _read_result(machine_root, _result_path(machine_root, plan_id))


def finish_takeover_intent(machine_root: Path, intent: TakeoverIntent) -> None:
    """Persist the terminal receipt and clear the active intent under the caller's lifecycle lock."""
    _require_owner(machine_root, intent)
    _require_terminal(intent)
    current = _read_active(machine_root, allow_legacy=False)
    if current is not None and current != intent:
        raise TakeoverIntentConflict("Project takeover active intent changed before finishing.")

    path = _result_path(machine_root, intent.plan_id)
    result = _read_result(machine_root, path)
    if result is None and current is not None:
        _write_intent(path, intent)
    elif result != intent:
        raise TakeoverIntentConflict("Project takeover has a missing or conflicting terminal receipt.")

    durable_io = DurableIO()
    durable_io.sync_directory(path.parent, "takeover_result")
    durable_io.sync_directory(machine_root, "takeover_result_parent")
    if current is not None:
        active_path = machine_root / _INTENT_NAME
        check_mutation_fence(active_path)
        active_path.unlink()
        durable_io.sync_directory(machine_root, "takeover_intent_cleared")


def takeover_issue_path(machine_root: Path) -> Path:
    """Locate the bounded local diagnostic for the current operation."""
    return machine_root / "agent" / "takeover-issue.json"


def decode_takeover_issue(document: object, intent: TakeoverIntent) -> str | None:
    """Use only a bounded issue scoped to this exact plan and phase."""
    if not isinstance(document, dict) or intent.phase in _TERMINAL_PHASES:
        return None
    if (
        document.get("plan_id") != intent.plan_id
        or document.get("phase") != intent.phase
        or not isinstance(document.get("reason"), str)
        or not 0 < len(document["reason"]) <= 512
    ):
        return None
    return document["reason"]


def read_takeover_issue(machine_root: Path, intent: TakeoverIntent) -> str | None:
    """Read best-effort local diagnostic evidence without affecting operation status."""
    try:
        return decode_takeover_issue(_read_protocol_record(takeover_issue_path(machine_root)), intent)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
