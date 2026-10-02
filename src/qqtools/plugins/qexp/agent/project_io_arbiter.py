"""Fairly select transient Project I/O requests for known binding owners."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from .project_io_protocol import PROJECT_IO_CAPACITY, PROJECT_IO_OPERATIONS

BindingOwner = tuple[str, str, str]
ServiceClass = Literal["authority", "primary", "background"]
WorkFamily = Literal["default", "gpu_primary", "cpu_primary", "gpu_borrow", "cpu_borrow"]
WORK_FAMILIES = frozenset({"default", "gpu_primary", "cpu_primary", "gpu_borrow", "cpu_borrow"})

_SERVICE_CLASSES: tuple[ServiceClass, ...] = ("authority", "primary", "background")
_SERVICE_CLASS_CYCLE: tuple[ServiceClass, ...] = (
    "authority",
    "primary",
    "authority",
    "background",
)
_MAX_RAW_CANDIDATES = 64


def _validate_owner(owner: object) -> BindingOwner:
    if type(owner) is not tuple or len(owner) != 3 or any(type(part) is not str or not part for part in owner):
        raise ValueError("owner must be a tuple of three non-empty strings.")
    return owner


def _validate_work_key(work_key: object) -> tuple[str, ...]:
    if type(work_key) is not tuple or any(type(part) is not str or not part for part in work_key):
        raise ValueError("work_key must be a tuple of non-empty strings.")
    return work_key


def _validate_intent(intent: object) -> ServiceIntent:
    if type(intent) is not ServiceIntent:
        raise ValueError("intents must contain ServiceIntent values.")
    _validate_owner(intent.owner)
    if type(intent.service_class) is not str or intent.service_class not in _SERVICE_CLASSES:
        raise ValueError("service_class must be 'authority', 'primary', or 'background'.")
    if type(intent.operation_kind) is not str or intent.operation_kind not in PROJECT_IO_OPERATIONS:
        raise ValueError("operation_kind must name a supported Project I/O operation.")
    _validate_work_key(intent.work_key)
    if type(intent.work_family) is not str or intent.work_family not in WORK_FAMILIES:
        raise ValueError("work_family must name a supported scheduling lane or default.")
    return intent


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence.")
    return value


@dataclass(frozen=True, slots=True)
class ServiceIntent:
    """Describe one typed request with its caller-assigned service class.

    Operation kind does not determine service class; dependent requests inherit
    the requesting work's class.
    """

    owner: BindingOwner
    service_class: ServiceClass
    operation_kind: str
    work_key: tuple[str, ...]
    work_family: WorkFamily = "default"
    deadline: float | None = None

    def __post_init__(self) -> None:
        _validate_owner(self.owner)
        if type(self.service_class) is not str or self.service_class not in _SERVICE_CLASSES:
            raise ValueError("service_class must be 'authority', 'primary', or 'background'.")
        if type(self.operation_kind) is not str or self.operation_kind not in PROJECT_IO_OPERATIONS:
            raise ValueError("operation_kind must name a supported Project I/O operation.")
        _validate_work_key(self.work_key)
        if type(self.work_family) is not str or self.work_family not in WORK_FAMILIES:
            raise ValueError("work_family must name a supported scheduling lane or default.")
        if self.deadline is not None and (
            self.service_class != "authority"
            or type(self.deadline) not in {float, int}
            or not math.isfinite(self.deadline)
            or self.deadline < 0
        ):
            raise ValueError("deadline must be a finite nonnegative authority-service timestamp.")


class ProjectIOArbiter:
    """Keep owner fairness and choose transient request grants.

    Callers reconcile current owners before selecting. Selection never enrolls
    unknown owners. Callers discover due work, submit bounded intents, and
    fairly rediscover omitted work with their existing cursors. A grant permits
    an attempt to prepare one typed request; the executor remains the definitive
    capacity and fence. Each binding owner is selected at most once per call.
    No request payload or work queue is retained here.
    """

    __slots__ = ("_rosters", "_class_cursor")

    def __init__(self) -> None:
        self._rosters: tuple[OrderedDict[BindingOwner, None], ...] = tuple(OrderedDict() for _ in _SERVICE_CLASSES)
        self._class_cursor = 0

    def reconcile(self, owners: Sequence[BindingOwner]) -> None:
        """Reconcile registered owners while preserving incumbent roster order."""
        owner_sequence = _require_sequence(owners, "owners")
        reconciled_owners: list[BindingOwner] = []
        seen_owners: set[BindingOwner] = set()
        for value in owner_sequence:
            owner = _validate_owner(value)
            if owner not in seen_owners:
                seen_owners.add(owner)
                reconciled_owners.append(owner)

        new_rosters: list[OrderedDict[BindingOwner, None]] = []
        for roster in self._rosters:
            retained = [owner for owner in roster if owner in seen_owners]
            retained_owners = set(retained)
            retained.extend(owner for owner in reconciled_owners if owner not in retained_owners)
            new_rosters.append(OrderedDict.fromkeys(retained))

        self._rosters = tuple(new_rosters)

    def candidate_priorities(self) -> dict[tuple[BindingOwner, ServiceClass], int]:
        """Rank discovered interests without a second, independently rotating window.

        Interleave per-class owner order so a bounded collector retains the
        oldest eligible owners of every class. Class service weights still
        belong to select(), not to this discovery priority.
        """
        return {
            (owner, service_class): rank * len(_SERVICE_CLASSES) + class_index
            for class_index, (service_class, roster) in enumerate(zip(_SERVICE_CLASSES, self._rosters, strict=True))
            for rank, owner in enumerate(roster)
        }

    def select(
        self,
        intents: Sequence[ServiceIntent],
        *,
        blocked: Collection[BindingOwner] = (),
        limit: int = PROJECT_IO_CAPACITY,
    ) -> tuple[ServiceIntent, ...]:
        """Return fair, per-owner transient grants from currently discovered work.

        The caller must continue discovering omitted work on later passes. A
        selected owner is blocked across every service class for this call.
        """
        if type(limit) is not int or not 0 <= limit <= PROJECT_IO_CAPACITY:
            raise ValueError(f"limit must be an integer from 0 to {PROJECT_IO_CAPACITY}.")

        intent_sequence = _require_sequence(intents, "intents")
        if len(intent_sequence) > _MAX_RAW_CANDIDATES:
            raise ValueError(f"intents may contain at most {_MAX_RAW_CANDIDATES} candidates.")
        raw_intents = tuple(intent_sequence)
        if len(raw_intents) > _MAX_RAW_CANDIDATES:
            raise ValueError(f"intents may contain at most {_MAX_RAW_CANDIDATES} candidates.")

        if not isinstance(blocked, Collection) or isinstance(blocked, str | bytes | bytearray):
            raise ValueError("blocked must be a collection of binding owners.")
        blocked_values = tuple(blocked)
        blocked_owners = {_validate_owner(owner) for owner in blocked_values}

        candidates_by_class: dict[ServiceClass, dict[BindingOwner, ServiceIntent]] = {
            service_class: {} for service_class in _SERVICE_CLASSES
        }
        for raw_intent in raw_intents:
            intent = _validate_intent(raw_intent)
            class_candidates = candidates_by_class[intent.service_class]
            class_candidates.setdefault(intent.owner, intent)

        if limit == 0:
            return ()

        selected: list[ServiceIntent] = []
        class_cursor = self._class_cursor
        while len(selected) < limit:
            grant: tuple[int, ServiceClass, BindingOwner, ServiceIntent] | None = None
            for offset in range(len(_SERVICE_CLASS_CYCLE)):
                cycle_slot = (class_cursor + offset) % len(_SERVICE_CLASS_CYCLE)
                service_class = _SERVICE_CLASS_CYCLE[cycle_slot]
                class_index = _SERVICE_CLASSES.index(service_class)
                class_candidates = candidates_by_class[service_class]
                roster = self._rosters[class_index]
                for owner in roster:
                    if owner in blocked_owners:
                        continue
                    intent = class_candidates.get(owner)
                    if intent is not None:
                        grant = (cycle_slot, service_class, owner, intent)
                        break
                if grant is not None:
                    break

            if grant is None:
                break

            cycle_slot, service_class, owner, intent = grant
            class_index = _SERVICE_CLASSES.index(service_class)
            self._rosters[class_index].move_to_end(owner)
            blocked_owners.add(owner)
            selected.append(intent)
            class_cursor = (cycle_slot + 1) % len(_SERVICE_CLASS_CYCLE)
            self._class_cursor = class_cursor

        return tuple(selected)
