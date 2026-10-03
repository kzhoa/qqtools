"""Coordinate bounded, transient admission of Project I/O service intents."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from .project_io_arbiter import WORK_FAMILIES, BindingOwner, ProjectIOArbiter, ServiceClass, ServiceIntent
from .project_io_protocol import PROJECT_IO_CAPACITY, PROJECT_IO_OPERATIONS

_SERVICE_CLASSES: tuple[ServiceClass, ...] = ("authority", "primary", "background")
_MAX_PENDING_INTENTS = 64
_MAX_CRITICAL_CHAIN_GRANTS = 3
_MAX_DUE_AUTHORITY_GRANTS = 3
_FAMILY_TIER_CONTINUATION = -1
_FAMILY_TIER_DUE = 0
_FAMILY_TIER_ORDINARY = 1
_CRITICAL_CHAIN_OPERATIONS = frozenset(
    {"scheduler_claim", "scheduler_launch_authorize", "scheduler_reservation_reconcile"}
)
_BACKGROUND_FAMILY_ORDER = {
    "recovery_admission": 0,
    "upgrade_service": 1,
    "progress_projection": 2,
    "activation_consumer_register": 3,
    "activation_observe": 4,
    "group_service_probe": 5,
    "group_service_advance": 6,
    "recovery_group_authority": 7,
    "submission_control_service": 8,
    "observation_service": 9,
    "maintenance_descriptor_advance": 10,
    "activation_consumer_ack": 11,
}
_PRIMARY_FAMILY_ORDER = {
    # Finish an already-produced empty scan before starting another proof.
    # Until this compare-and-commit runs, discovery is pinned to the same
    # durable cursor and no later scan can make useful progress.
    ("scheduler_cursor_commit", "cpu_primary"): 0,
    ("scheduler_cursor_commit", "gpu_primary"): 0,
    ("scheduler_cursor_commit", "cpu_borrow"): 0,
    ("scheduler_cursor_commit", "gpu_borrow"): 0,
    # The closed proof is itself a complete non-authorizing demand census. Run
    # it before position-based discovery so an empty Project does not pay for
    # four role/lane scans and cursor commits merely to prove the same absence.
    # An active result backs off for one second, allowing ordinary discovery to
    # authorize work on the following pass.
    ("scheduler_quiescence_probe", "default"): 1,
    ("scheduler_observe", "cpu_primary"): 2,
    ("scheduler_observe", "gpu_primary"): 2,
    ("scheduler_observe", "cpu_borrow"): 3,
    ("scheduler_observe", "gpu_borrow"): 3,
}


@dataclass(slots=True)
class _CriticalChainCredit:
    epoch: tuple[str, str] | None
    grants: int = 0


@dataclass(slots=True)
class _ProgressChainCredit:
    grants: int = 0


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence.")
    return value


def _require_collection(value: object, label: str) -> Collection[object]:
    if not isinstance(value, Collection) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be a collection.")
    return value


def _validate_owner(owner: object) -> BindingOwner:
    if type(owner) is not tuple or len(owner) != 3 or any(type(part) is not str or not part for part in owner):
        raise ValueError("owner must be a tuple of three non-empty strings.")
    return owner


def _validate_intent(intent: object) -> ServiceIntent:
    if type(intent) is not ServiceIntent:
        raise ValueError("intent must be a ServiceIntent value.")
    _validate_owner(intent.owner)
    if type(intent.service_class) is not str or intent.service_class not in _SERVICE_CLASSES:
        raise ValueError("service_class must be 'authority', 'primary', or 'background'.")
    if type(intent.operation_kind) is not str or intent.operation_kind not in PROJECT_IO_OPERATIONS:
        raise ValueError("operation_kind must name a supported Project I/O operation.")
    if type(intent.work_key) is not tuple or any(type(part) is not str or not part for part in intent.work_key):
        raise ValueError("work_key must be a tuple of non-empty strings.")
    if type(intent.work_family) is not str or intent.work_family not in WORK_FAMILIES:
        raise ValueError("work_family must name a supported scheduling lane or default.")
    return intent


class ProjectIOAdmission:
    """Fairly select up to four ephemeral local preparation actions per pass.

    Call ``begin`` once before a complete machine service pass, ``offer`` at
    each new preparation gate, and ``finish`` after discovery. The bounded
    collector retains at most 64 intent/action pairs, sharing the candidate
    budget between nonempty classes and preferring each class's oldest owners.
    A finite operation/lane cursor rotates work within each owner/class. The
    first offered item within one operation/lane wins; its existing producer
    cursor owns Attempt or record fairness. Registry metadata
    itself is not subject to the bound. Candidate budget balance is separate
    from the arbiter's weighted execution opportunities.

    Actions live only through the current pass and are never persisted or sent
    over the worker protocol. Executor capacity and per-binding limits remain
    authoritative. Wrapping the whole machine pass keeps fixed service-call
    order from controlling owner or class fairness. Occupancy is checked when
    granting: a result consumed later in the pass can unblock earlier interests.
    """

    __slots__ = (
        "_arbiter",
        "_active",
        "_draining",
        "_collected",
        "_seen",
        "_seen_families",
        "_work_cursors",
        "_urgent_families",
        "_monotonic",
        "_pass_time",
        "_priorities",
        "_blocked",
        "_free_slots",
        "_omitted_candidates",
        "_has_deferred_work",
        "_critical_chain_grants",
        "_critical_burst_grants",
        "_critical_yielded_classes",
        "_due_authority_burst_grants",
        "_due_authority_yielded_classes",
        "_progress_chain_grants",
    )

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._arbiter = ProjectIOArbiter()
        self._active = False
        self._draining = False
        self._collected: OrderedDict[tuple[BindingOwner, ServiceClass], tuple[ServiceIntent, Callable[[], None]]] = (
            OrderedDict()
        )
        self._priorities: dict[tuple[BindingOwner, ServiceClass], int] = {}
        self._seen: set[tuple[BindingOwner, ServiceClass]] = set()
        self._seen_families: set[tuple[BindingOwner, ServiceClass, tuple[str, str]]] = set()
        self._work_cursors: dict[tuple[BindingOwner, ServiceClass], tuple[str, str]] = {}
        self._urgent_families: dict[tuple[BindingOwner, ServiceClass], tuple[str, str]] = {}
        self._monotonic = monotonic
        self._pass_time = 0.0
        self._blocked: set[BindingOwner] = set()
        self._free_slots = 0
        self._omitted_candidates = False
        self._has_deferred_work = False
        self._critical_chain_grants: dict[BindingOwner, _CriticalChainCredit] = {}
        self._critical_burst_grants = 0
        self._critical_yielded_classes: set[ServiceClass] = set()
        self._due_authority_burst_grants = 0
        self._due_authority_yielded_classes: set[ServiceClass] = set()
        self._progress_chain_grants: dict[BindingOwner, _ProgressChainCredit] = {}

    @property
    def active(self) -> bool:
        """Whether a machine service pass is currently open."""
        return self._active

    @property
    def pending_count(self) -> int:
        """Return the number of candidates retained during the active pass."""
        return len(self._collected) if self._active else 0

    @property
    def has_deferred_work(self) -> bool:
        """Whether the last completed pass left eligible work to rediscover."""
        return self._has_deferred_work

    def candidate_priority(self, owner: BindingOwner, service_class: ServiceClass) -> int | None:
        """Return this turn's roster rank for one known owner and class.

        Producers with their own bounded discovery windows use this rank so
        they retain the same owners the arbiter can actually grant next.  The
        value is transient and is valid only for the active admission turn.
        """
        if not self._active or self._draining:
            raise RuntimeError("candidate priority is available only during collection.")
        checked_owner = _validate_owner(owner)
        if type(service_class) is not str or service_class not in _SERVICE_CLASSES:
            raise ValueError("service_class must be 'authority', 'primary', or 'background'.")
        return self._priorities.get((checked_owner, service_class))

    def begin(
        self,
        owners: Sequence[BindingOwner],
        *,
        blocked: Collection[BindingOwner],
        free_slots: int,
    ) -> None:
        """Open a machine pass after validating owners, blocked bindings, and capacity.

        Args:
            owners: Current registered binding owners.
            blocked: Owners whose unresolved work already occupies a slot.
            free_slots: Available executor capacity, from zero through four.

        Raises:
            RuntimeError: If another pass is active.
            ValueError: If an owner or capacity value is invalid.
        """
        if self._active:
            raise RuntimeError("a Project I/O admission pass is already active.")

        owner_values = _require_sequence(owners, "owners")
        checked_owners: list[BindingOwner] = []
        seen_owners: set[BindingOwner] = set()
        for raw_owner in owner_values:
            owner = _validate_owner(raw_owner)
            if owner not in seen_owners:
                seen_owners.add(owner)
                checked_owners.append(owner)
        owner_tuple = tuple(checked_owners)

        blocked_values = _require_collection(blocked, "blocked")
        checked_blocked = tuple(_validate_owner(owner) for owner in blocked_values)
        if type(free_slots) is not int or not 0 <= free_slots <= PROJECT_IO_CAPACITY:
            raise ValueError(f"free_slots must be an integer from 0 to {PROJECT_IO_CAPACITY}.")

        self._arbiter.reconcile(owner_tuple)
        self._work_cursors = {key: cursor for key, cursor in self._work_cursors.items() if key[0] in seen_owners}
        self._urgent_families = {key: family for key, family in self._urgent_families.items() if key[0] in seen_owners}
        self._critical_chain_grants = {
            owner: state for owner, state in self._critical_chain_grants.items() if owner in seen_owners
        }
        if not self._critical_chain_grants:
            self._critical_burst_grants = 0
            self._critical_yielded_classes.clear()
        self._progress_chain_grants = {
            owner: state for owner, state in self._progress_chain_grants.items() if owner in seen_owners
        }
        self._pass_time = self._monotonic()
        priorities = self._arbiter.candidate_priorities()

        self._draining = False
        self._collected = OrderedDict()
        self._seen = set()
        self._seen_families = set()
        self._priorities = priorities
        self._blocked = set(checked_blocked)
        self._free_slots = free_slots
        self._omitted_candidates = False
        self._has_deferred_work = False
        self._active = True

    def offer(self, intent: ServiceIntent, action: Callable[[], None]) -> None:
        """Collect one known local preparation interest for this pass.

        Args:
            intent: Typed owner, service class, operation, and work identity.
            action: Local controller action to call if the arbiter selects it.

        Raises:
            RuntimeError: If no pass is active or an action is being drained.
            ValueError: If the intent or action is invalid.
        """
        if not self._active:
            raise RuntimeError("begin must be called before offering Project I/O work.")
        if self._draining:
            raise RuntimeError("Project I/O actions cannot offer work while finishing a pass.")

        checked_intent = _validate_intent(intent)
        if not callable(action):
            raise ValueError("action must be callable.")

        critical_credit = self._critical_chain_grants.get(checked_intent.owner)
        critical_epoch = self._critical_chain_epoch(checked_intent)
        if (
            checked_intent.service_class == "primary"
            and checked_intent.operation_kind in _CRITICAL_CHAIN_OPERATIONS
            and critical_credit is not None
            and critical_credit.epoch != critical_epoch
            and critical_credit.epoch is not None
            and critical_epoch is not None
            and critical_credit.epoch[0] != critical_epoch[0]
        ):
            self._critical_burst_grants = 0
            self._critical_yielded_classes.clear()

        key = (checked_intent.owner, checked_intent.service_class)
        priority = self._priorities.get(key)
        if priority is None:
            return
        family = (checked_intent.operation_kind, checked_intent.work_family)
        family_key = (*key, family)
        if family_key in self._seen_families:
            return
        self._seen_families.add(family_key)
        if key in self._seen:
            self._omitted_candidates = True
            retained = self._collected.get(key)
            if retained is not None:
                candidate_rank = self._family_rank(key, checked_intent)
                previous_rank = self._family_rank(key, retained[0])
                if candidate_rank < previous_rank:
                    self._collected[key] = (checked_intent, action)
            elif self._can_prioritize_critical_chain(checked_intent):
                self._retain_critical_continuation(key, checked_intent, action)
            return
        self._seen.add(key)

        if len(self._collected) < _MAX_PENDING_INTENTS:
            self._collected[key] = (checked_intent, action)
            return

        if self._can_prioritize_critical_chain(checked_intent) and self._retain_critical_continuation(
            key,
            checked_intent,
            action,
        ):
            self._omitted_candidates = True
            return

        counts = {service_class: 0 for service_class in _SERVICE_CLASSES}
        for _owner, service_class in self._collected:
            counts[service_class] += 1
        largest_class = max(_SERVICE_CLASSES, key=counts.__getitem__)
        incoming_class = checked_intent.service_class
        # Idle owners ahead in another class's roster must not inflate this
        # class's admission cost. Compare owner ranks only within one class.
        evict_class = largest_class if counts[incoming_class] < counts[largest_class] else incoming_class
        worst_key = max(
            (collected_key for collected_key in self._collected if collected_key[1] == evict_class),
            key=self._priorities.__getitem__,
        )
        if evict_class != incoming_class or priority < self._priorities[worst_key]:
            del self._collected[worst_key]
            self._collected[key] = (checked_intent, action)
        self._omitted_candidates = True

    def _retain_critical_continuation(
        self,
        key: tuple[BindingOwner, ServiceClass],
        intent: ServiceIntent,
        action: Callable[[], None],
    ) -> bool:
        """Keep a proven finite continuation inside a full candidate window."""
        victims = [
            retained_key
            for retained_key, (retained, _action) in self._collected.items()
            if retained.service_class == intent.service_class and not self._can_prioritize_critical_chain(retained)
        ]
        if not victims:
            return False
        victim = max(victims, key=self._priorities.__getitem__)
        del self._collected[victim]
        self._collected[key] = (intent, action)
        return True

    def _family_rank(self, key: tuple[BindingOwner, ServiceClass], intent: ServiceIntent) -> tuple:
        """Rank continuations, one urgent deadline, then cursor-rotated ordinary work."""
        family = (intent.operation_kind, intent.work_family)
        cursor = self._work_cursors.get(key)
        previous_urgent = self._urgent_families.get(key)
        is_due = intent.deadline is not None and intent.deadline <= self._pass_time
        if previous_urgent is None and is_due:
            return (_FAMILY_TIER_DUE, intent.deadline, family)
        if self._can_prioritize_critical_chain(intent):
            return (_FAMILY_TIER_CONTINUATION, cursor is not None and family <= cursor, family)
        if self._can_prefer_progress_chain(intent):
            return (_FAMILY_TIER_CONTINUATION, cursor is not None and family <= cursor, family)
        # Finish finite activation and service-lane closure before periodic
        # metadata. The cursor still visits every waiting family before a
        # continuously eligible prerequisite or closure family can repeat.
        family_order = self._ordinary_family_order(intent.service_class, family)
        cursor_order = None if cursor is None else self._ordinary_family_order(intent.service_class, cursor)
        # An urgent override cannot reset the ordinary family cursor or take
        # consecutive opportunities away from an already waiting finite set.
        is_previous_urgent = family == previous_urgent
        is_at_or_before_cursor = cursor_order is not None and family_order <= cursor_order
        return (_FAMILY_TIER_ORDINARY, is_previous_urgent, is_at_or_before_cursor, family_order)

    @staticmethod
    def _is_progress_chain(intent: ServiceIntent) -> bool:
        return (
            intent.service_class == "background"
            and intent.operation_kind == "progress_projection"
            and len(intent.work_key) >= 3
            and intent.work_key[-3] == "progress"
            and intent.work_key[-1] in {"observe", "publish"}
        )

    def _can_prefer_progress_chain(self, intent: ServiceIntent) -> bool:
        if not self._is_progress_chain(intent):
            return False
        credit = self._progress_chain_grants.get(intent.owner)
        return credit is None or credit.grants < 2

    @staticmethod
    def _ordinary_family_order(service_class: ServiceClass, family: tuple[str, str]) -> tuple[int, tuple[str, str]]:
        if service_class == "background":
            return (_BACKGROUND_FAMILY_ORDER.get(family[0], len(_BACKGROUND_FAMILY_ORDER)), family)
        if service_class == "primary":
            return (_PRIMARY_FAMILY_ORDER.get(family, len(_PRIMARY_FAMILY_ORDER)), family)
        return (0, family)

    def _can_prefer_critical_chain(self, intent: ServiceIntent) -> bool:
        if intent.service_class != "primary" or intent.operation_kind not in _CRITICAL_CHAIN_OPERATIONS:
            return False
        epoch = self._critical_chain_epoch(intent)
        credit = self._critical_chain_grants.get(intent.owner)
        return credit is None or credit.epoch != epoch or credit.grants < _MAX_CRITICAL_CHAIN_GRANTS

    def _can_prioritize_critical_chain(self, intent: ServiceIntent) -> bool:
        """Return whether a chain has both Attempt credit and global burst room."""
        return self._critical_burst_grants < _MAX_CRITICAL_CHAIN_GRANTS and self._can_prefer_critical_chain(intent)

    @staticmethod
    def _critical_chain_epoch(intent: ServiceIntent) -> tuple[str, str] | None:
        if not intent.work_key:
            return None
        # The executor epoch fences the machine incarnation; the final work
        # key component identifies the exact Attempt chain.  Credits therefore
        # carry claim -> launch -> reconciliation without leaking exhaustion to
        # a later Task owned by the same binding.
        return (intent.work_key[0], intent.work_key[-1])

    def _record_critical_chain_grant(self, intent: ServiceIntent, *, used_preference: bool) -> None:
        epoch = self._critical_chain_epoch(intent)
        credit = self._critical_chain_grants.get(intent.owner)
        is_critical = intent.service_class == "primary" and intent.operation_kind in _CRITICAL_CHAIN_OPERATIONS
        if is_critical and credit is not None and credit.epoch != epoch:
            self._critical_chain_grants.pop(intent.owner)
            credit = None
        if is_critical:
            if used_preference:
                self._critical_burst_grants = min(
                    _MAX_CRITICAL_CHAIN_GRANTS,
                    self._critical_burst_grants + 1,
                )
            if credit is None:
                credit = _CriticalChainCredit(epoch)
                self._critical_chain_grants[intent.owner] = credit
            credit.grants = min(_MAX_CRITICAL_CHAIN_GRANTS, credit.grants + 1)
        elif self._critical_burst_grants >= _MAX_CRITICAL_CHAIN_GRANTS and intent.service_class in {
            "primary",
            "background",
        }:
            self._critical_yielded_classes.add(intent.service_class)
            if self._critical_yielded_classes == {"primary", "background"}:
                self._critical_burst_grants = 0
                self._critical_yielded_classes.clear()
                self._critical_chain_grants.clear()

        progress_credit = self._progress_chain_grants.get(intent.owner)
        if self._is_progress_chain(intent):
            if progress_credit is None:
                progress_credit = _ProgressChainCredit()
                self._progress_chain_grants[intent.owner] = progress_credit
            progress_credit.grants = min(2, progress_credit.grants + 1)
        elif intent.service_class == "background" and progress_credit is not None:
            self._progress_chain_grants.pop(intent.owner)

    def refresh_capacity(self, *, blocked: Collection[BindingOwner], free_slots: int) -> None:
        """Refresh occupancy after result consumption without restarting fairness."""
        if not self._active or self._draining:
            raise RuntimeError("capacity can only be refreshed during collection.")
        checked_blocked = {_validate_owner(owner) for owner in _require_collection(blocked, "blocked")}
        if type(free_slots) is not int or not 0 <= free_slots <= PROJECT_IO_CAPACITY:
            raise ValueError("free_slots must be within executor capacity.")
        self._blocked = checked_blocked
        self._free_slots = free_slots

    def finish(self, *, failed: bool = False) -> None:
        """Select and run current-pass actions, then release all transient state.

        Args:
            failed: Drop all offers without selecting actions when true.

        Raises:
            RuntimeError: If no pass is active or this pass is already draining.
            ValueError: If ``failed`` is not a bool.

        Exceptions raised by selected actions propagate after transient state is cleared.
        """
        if not self._active:
            raise RuntimeError("begin must be called before finishing Project I/O admission.")
        if self._draining:
            raise RuntimeError("Project I/O admission is already finishing this pass.")
        if type(failed) is not bool:
            raise ValueError("failed must be a bool.")

        self._draining = True
        try:
            if failed:
                self._has_deferred_work = False
                return

            self._collected = OrderedDict(
                (key, item) for key, item in self._collected.items() if key[0] not in self._blocked
            )
            self._omitted_candidates = self._omitted_candidates and any(
                owner not in self._blocked for owner, _service in self._seen
            )

            self._has_deferred_work = self._omitted_candidates or bool(self._collected)
            for _ in range(self._free_slots):
                eligible = tuple(
                    intent for intent, _action in self._collected.values() if intent.owner not in self._blocked
                )
                due_authority = tuple(
                    intent
                    for intent in eligible
                    if intent.service_class == "authority"
                    and intent.deadline is not None
                    and intent.deadline <= self._pass_time
                )
                prefer_critical = any(self._can_prioritize_critical_chain(intent) for intent in eligible)
                prefer_due_authority = (
                    len(due_authority) > 1 and self._due_authority_burst_grants < _MAX_DUE_AUTHORITY_GRANTS
                )
                candidates = (
                    due_authority
                    # A finite deadline burst reduces expiry risk, then the
                    # ordinary class cycle must regain an opportunity even if
                    # multiple owners remain continuously due.
                    if prefer_due_authority
                    else tuple(
                        intent
                        for intent in eligible
                        if not (
                            prefer_critical
                            and intent.service_class == "primary"
                            and not self._can_prefer_critical_chain(intent)
                        )
                        and not (prefer_critical and intent.service_class == "background")
                    )
                )
                selected = self._arbiter.select(
                    candidates,
                    blocked=self._blocked,
                    limit=1,
                )
                if not selected:
                    break
                intent = selected[0]
                key = (intent.owner, intent.service_class)
                action = self._collected.pop(key)[1]
                self._blocked.add(intent.owner)
                if prefer_due_authority:
                    self._due_authority_burst_grants += 1
                    if self._due_authority_burst_grants == _MAX_DUE_AUTHORITY_GRANTS:
                        self._due_authority_yielded_classes.clear()
                elif (
                    self._due_authority_burst_grants >= _MAX_DUE_AUTHORITY_GRANTS
                    and intent.service_class != "authority"
                ):
                    self._due_authority_yielded_classes.add(intent.service_class)
                    waiting_classes = {
                        candidate.service_class
                        for candidate in eligible
                        if candidate.service_class in {"primary", "background"}
                    }
                    if waiting_classes <= self._due_authority_yielded_classes:
                        self._due_authority_burst_grants = 0
                        self._due_authority_yielded_classes.clear()
                family = (intent.operation_kind, intent.work_family)
                is_urgent = (
                    key not in self._urgent_families
                    and intent.deadline is not None
                    and intent.deadline <= self._pass_time
                )
                if is_urgent or self._can_prefer_progress_chain(intent):
                    # Continuations may finish a report, but cannot rewind the
                    # ordinary cursor past pending upgrade/recovery families.
                    self._urgent_families[key] = family
                else:
                    self._urgent_families.pop(key, None)
                    self._work_cursors[key] = family
                self._record_critical_chain_grant(
                    intent,
                    used_preference=prefer_critical and self._can_prefer_critical_chain(intent),
                )
                # Advance only an attempted grant. An exception must not spend
                # every unattempted peer's turn and recreate the same ordering.
                action()
            self._has_deferred_work = self._omitted_candidates or bool(self._collected)
        finally:
            self._collected = OrderedDict()
            self._seen = set()
            self._seen_families = set()
            self._priorities = {}
            self._blocked = set()
            self._free_slots = 0
            self._omitted_candidates = False
            self._draining = False
            self._active = False
