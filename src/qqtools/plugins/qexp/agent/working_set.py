"""Process-local binding residency coordinated by Project activation checkpoints."""

from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from ..runtime.records import utc_now
from ..runtime.store import atomic_replace
from .bindings import ProjectBinding

if TYPE_CHECKING:
    from .context import MachineRuntime

SERVICE_LANES = ("scheduler", "authority", "group", "observation", "submission", "maintenance")
_RECORD_VERSION = 1
_MAX_REASON_BYTES = 128
_COLD_RECONCILE_SECONDS = 60.0
_OBSERVATION_INTERVAL_SECONDS = 5.0
_monotonic = time.monotonic
_CheckpointSignature = tuple[str, int] | None
_Identity = tuple[str, str, str | None]


@dataclass(frozen=True, slots=True)
class BindingTurn:
    """One service lane's observation of a binding activation checkpoint."""

    runtime_id: str
    project_id: str
    registration_generation: str | None
    shared_root: Path
    lane: str
    checkpoint: _CheckpointSignature
    unknown: bool = False
    wake_generation: int = 0

    @property
    def identity(self) -> _Identity:
        return self.runtime_id, self.project_id, self.registration_generation


@dataclass(frozen=True, slots=True)
class ActivationObservationIntent:
    """Exact durable cursor handed to one asynchronous activation observation."""

    binding: ProjectBinding
    replay_epoch: str | None
    replay_sequence: int
    expected_checkpoint: _CheckpointSignature


@dataclass(frozen=True, slots=True)
class ActivationAcknowledgementIntent:
    """Exact replay acknowledgement and local quiescence snapshot for one binding."""

    binding: ProjectBinding
    epoch: str
    sequence: int
    reconstructed_floor: int | None
    require_current: bool
    checkpoint: _CheckpointSignature
    wake_generation: int
    lane_revision: int


@dataclass(slots=True)
class _BindingState:
    binding: ProjectBinding
    state: str = "resident"
    checkpoint: _CheckpointSignature = None
    acknowledgements: dict[str, _CheckpointSignature] = field(default_factory=lambda: dict.fromkeys(SERVICE_LANES))
    acknowledged_lanes: set[str] = field(default_factory=set)
    reason: str = "startup"
    updated_at: str = field(default_factory=utc_now)
    startup_pending: bool = True
    unknown: bool = False
    next_cold_reconcile_at: float = 0.0
    consumer_registered: bool = False
    replay_epoch: str | None = None
    replay_sequence: int = 0
    reconstructed_floor: int | None = None
    activation_observed: bool = False
    replay_proposal: tuple[str, int, int | None, bool] | None = None
    acknowledgement_intent: ActivationAcknowledgementIntent | None = None
    lane_revision: int = 0
    next_observation_at: float = 0.0
    observation_due: bool = True


class BindingWorkingSet:
    """Track resident bindings until every service lane proves quiescence."""

    def __init__(self, runtime: MachineRuntime, *, process_fence: str | None = None) -> None:
        self._runtime = runtime
        self._runtime_id: str | None = None
        if process_fence is not None and (not isinstance(process_fence, str) or not process_fence):
            raise ValueError("process_fence must be a nonempty string when supplied.")
        self._process_fence = process_fence or uuid.uuid4().hex
        self._lock = RLock()
        self._states: dict[_Identity, _BindingState] = {}
        self._resident_members: dict[_Identity, None] = {}
        self._dormant_roster: OrderedDict[_Identity, None] = OrderedDict()
        self._dormant_renewal_roster: OrderedDict[_Identity, None] = OrderedDict()
        self._dormant_members: set[_Identity] = set()
        self._dormant_enabled: set[_Identity] = set()
        self._registry_revision: int | None = None
        self._reconcile_target_revision: int | None = None
        self._wake_generations: dict[_Identity, int] = {}
        self._reconcile_generation = 0
        self._activation_offsets: dict[str, int] = {}
        self._cold_probes = 0
        self._activations = 0

    def reconcile(self, bindings: Sequence[ProjectBinding], *, revision: int | None = None) -> None:
        """Replace stale generations and make newly observed bindings resident."""
        with self._lock:
            if revision is not None and self._registry_revision is not None and revision <= self._registry_revision:
                return
        current: dict[_Identity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._identity(binding)
            previous = current.get(identity)
            if previous is not None and previous.shared_root != binding.shared_root:
                raise ValueError("the working set cannot reconcile one binding identity to multiple roots.")
            current[identity] = binding

        with self._lock:
            if revision is not None and (
                (self._registry_revision is not None and revision <= self._registry_revision)
                or (self._reconcile_target_revision is not None and revision <= self._reconcile_target_revision)
            ):
                return
            if revision is not None:
                self._reconcile_target_revision = revision
            self._reconcile_generation += 1
            reconcile_generation = self._reconcile_generation
            additions = [
                (identity, binding)
                for identity, binding in current.items()
                if (state := self._states.get(identity)) is None or state.binding.shared_root != binding.shared_root
            ]

        with self._lock:
            if reconcile_generation != self._reconcile_generation:
                return
            if revision is not None and self._registry_revision is not None and revision <= self._registry_revision:
                return

            retained = {
                identity: state
                for identity, state in self._states.items()
                if identity in current and state.binding.shared_root == current[identity].shared_root
            }
            self._states = retained
            self._resident_members.clear()
            self._dormant_roster.clear()
            self._dormant_renewal_roster.clear()
            self._dormant_members.clear()
            self._dormant_enabled.clear()
            for identity in current:
                state = retained.get(identity)
                if state is None:
                    continue
                previous_enabled = state.binding.enabled
                state.binding = current[identity]
                if state.state == "dormant":
                    self._add_dormant_locked(identity)
                    if not previous_enabled and state.binding.enabled:
                        self._wake_locked(identity, state, reason="binding_enabled", unknown=False)
                else:
                    self._add_resident_locked(identity)
            self._wake_generations = {
                identity: generation for identity, generation in self._wake_generations.items() if identity in retained
            }
            for identity, binding in additions:
                if identity in retained:
                    continue
                state = _BindingState(
                    binding=binding,
                    unknown=True,
                    consumer_registered=False,
                )
                state.reason = "startup_activation_unknown"
                self._states[identity] = state
                self._add_resident_locked(identity)
                self._persist_locked(identity, state)
            self._registry_revision = revision
            if revision is not None and self._reconcile_target_revision == revision:
                self._reconcile_target_revision = None

    def resident_bindings(self, bindings: Sequence[ProjectBinding] | None = None) -> list[ProjectBinding]:
        """Return current resident bindings in the caller's registry order."""
        with self._lock:
            if bindings is None:
                return [
                    state.binding
                    for identity in self._resident_members
                    if (state := self._states.get(identity)) is not None and state.state == "resident"
                ]
            return [
                state.binding
                for binding in bindings
                if (state := self._states.get(self._identity(binding))) is not None
                and state.binding == binding
                and state.state == "resident"
            ]

    @property
    def is_reconciled(self) -> bool:
        """Return whether this process has consumed a durable registry revision."""
        with self._lock:
            return self._registry_revision is not None

    def has_dormant_enabled_binding(self) -> bool:
        """Return whether borrow proof omits any enabled dormant binding."""
        with self._lock:
            return bool(self._dormant_enabled)

    @property
    def process_fence(self) -> str:
        """Return this process's stable activation-consumer fence."""
        return self._process_fence

    def select_dormant_registration_renewals(
        self,
        *,
        limit: int = 64,
        heartbeat_interval_seconds: float = 5.0,
    ) -> tuple[tuple[ProjectBinding, ...], float]:
        """Select a bounded fair dormant slice without touching Project storage."""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if (
            not isinstance(heartbeat_interval_seconds, (int, float))
            or isinstance(heartbeat_interval_seconds, bool)
            or not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat_interval_seconds must be a finite positive number.")
        selected: list[ProjectBinding] = []
        with self._lock:
            roster_size = len(self._dormant_renewal_roster)
            rounds_to_revisit = max(1, math.ceil(roster_size / limit))
            renewal_horizon = (rounds_to_revisit + 1) * heartbeat_interval_seconds
            for _ in range(min(limit, roster_size)):
                identity, _marker = self._dormant_renewal_roster.popitem(last=False)
                self._dormant_renewal_roster[identity] = None
                state = self._states.get(identity)
                if state is not None and state.state == "dormant" and state.binding.enabled:
                    selected.append(state.binding)
        return tuple(selected), renewal_horizon

    def select_registration_renewals(
        self,
        *,
        limit: int = 64,
        heartbeat_interval_seconds: float = 5.0,
    ) -> tuple[tuple[ProjectBinding, ...], float]:
        """Select enabled registrations fairly, independent of Project work state."""
        if (
            type(heartbeat_interval_seconds) not in {int, float}
            or not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat_interval_seconds must be a finite positive number.")
        with self._lock:
            selected = self._fair_states_locked("registration_renewal", lambda state: state.binding.enabled, limit)
            count = sum(state.binding.enabled for state in self._states.values())
            rounds_to_revisit = max(1, math.ceil(count / limit))
            horizon = (rounds_to_revisit + 1) * heartbeat_interval_seconds
            return tuple(state.binding for _identity, state in selected), horizon

    def select_activation_registrations(self, *, limit: int = 64) -> tuple[ProjectBinding, ...]:
        """Select enabled unregistered bindings fairly without Project I/O."""
        with self._lock:
            selected = self._fair_states_locked(
                "registration",
                lambda state: state.binding.enabled and not state.consumer_registered,
                limit,
            )
            return tuple(state.binding for _identity, state in selected)

    def apply_activation_registrations(
        self,
        bindings: Sequence[ProjectBinding],
        results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Apply exact isolated consumer-registration completions locally."""
        by_project = {binding.project_id: binding for binding in bindings}
        with self._lock:
            for project_id, evidence in results.items():
                binding = by_project.get(project_id)
                if binding is None:
                    continue
                identity = self._identity(binding)
                state = self._states.get(identity)
                if state is None or state.binding != binding or evidence.get("outcome") != "registered":
                    continue
                acknowledgement = evidence.get("acknowledgement")
                state.consumer_registered = True
                if acknowledgement is None:
                    state.replay_epoch = None
                    state.replay_sequence = 0
                elif self._is_checkpoint_record(acknowledgement):
                    state.replay_epoch = acknowledgement["epoch"]
                    state.replay_sequence = acknowledgement["sequence"]
                else:
                    continue
                state.activation_observed = False
                state.replay_proposal = None
                state.observation_due = True
                state.reason = "consumer_registered"
                state.updated_at = utc_now()
                self._persist_locked(identity, state)

    def select_activation_observations(self, *, limit: int = 64) -> tuple[ActivationObservationIntent, ...]:
        """Select due observation cursors fairly without touching Project storage."""
        now = _monotonic()
        with self._lock:
            selected = self._fair_states_locked(
                "observation",
                lambda state: (
                    state.binding.enabled
                    and state.consumer_registered
                    and (state.observation_due or (state.state == "resident" and now >= state.next_observation_at))
                ),
                limit,
            )
            return tuple(
                ActivationObservationIntent(
                    binding=state.binding,
                    replay_epoch=state.replay_epoch,
                    replay_sequence=state.replay_sequence,
                    expected_checkpoint=state.checkpoint,
                )
                for _identity, state in selected
            )

    def retain_activation_observation_intent(
        self,
        binding: ProjectBinding,
        *,
        replay_epoch: str | None,
        replay_sequence: int,
    ) -> ActivationObservationIntent | None:
        """Rebuild an exact still-current intent for unresolved worker ownership."""
        identity = self._identity(binding)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.binding != binding
                or state.replay_epoch != replay_epoch
                or state.replay_sequence != replay_sequence
            ):
                return None
            return ActivationObservationIntent(binding, replay_epoch, replay_sequence, state.checkpoint)

    def is_activation_observation_due(
        self,
        binding: ProjectBinding,
        *,
        replay_epoch: str | None,
        replay_sequence: int,
    ) -> bool:
        """Revalidate a collected observation intent immediately before grant."""
        now = _monotonic()
        with self._lock:
            state = self._states.get(self._identity(binding))
            return bool(
                state is not None
                and state.binding == binding
                and state.binding.enabled
                and state.consumer_registered
                and state.replay_epoch == replay_epoch
                and state.replay_sequence == replay_sequence
                and (state.observation_due or (state.state == "resident" and now >= state.next_observation_at))
            )

    def apply_activation_observations(
        self,
        intents: Sequence[ActivationObservationIntent],
        results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Apply bounded replay observations and wake changed dormant bindings."""
        by_project = {intent.binding.project_id: intent for intent in intents}
        now = _monotonic()
        with self._lock:
            for project_id, evidence in results.items():
                intent = by_project.get(project_id)
                if intent is None:
                    continue
                binding = intent.binding
                identity = self._identity(binding)
                state = self._states.get(identity)
                if (
                    state is None
                    or state.binding != binding
                    or state.replay_epoch != intent.replay_epoch
                    or state.replay_sequence != intent.replay_sequence
                ):
                    continue
                if evidence.get("outcome") != "observed":
                    self._wake_locked(identity, state, reason="checkpoint_unreadable", unknown=True)
                    continue
                checkpoint_record = evidence.get("checkpoint")
                if checkpoint_record is None:
                    checkpoint = None
                    proposal = None
                else:
                    replay = evidence.get("replay")
                    if not self._is_checkpoint_record(checkpoint_record) or not isinstance(replay, Mapping):
                        continue
                    if (
                        replay.get("epoch") != checkpoint_record["epoch"]
                        or type(replay.get("sequence")) is not int
                        or replay["sequence"] < 1
                        or type(replay.get("complete")) is not bool
                    ):
                        continue
                    floor = replay.get("reconstructed_floor")
                    if floor is not None and (type(floor) is not int or floor < 1 or floor > replay["sequence"]):
                        continue
                    checkpoint = (checkpoint_record["epoch"], checkpoint_record["sequence"])
                    proposal = (replay["epoch"], replay["sequence"], floor, replay["complete"])
                previous_checkpoint = state.checkpoint
                was_dormant = state.state == "dormant"
                if checkpoint != previous_checkpoint:
                    self._wake_locked(
                        identity,
                        state,
                        reason="checkpoint_changed",
                        unknown=False,
                        checkpoint=checkpoint,
                    )
                elif was_dormant and now >= state.next_cold_reconcile_at:
                    self._wake_locked(identity, state, reason="cold_reconciliation", unknown=False)
                if checkpoint is not None and state.replay_epoch is not None and state.replay_epoch != checkpoint[0]:
                    # register_consumer is the mutation that deliberately clears
                    # a durable acknowledgement from an obsolete epoch.  Do not
                    # attempt to acknowledge the replacement epoch until that
                    # fenced mutation has completed in an isolated worker.
                    state.consumer_registered = False
                    state.activation_observed = False
                    state.replay_proposal = None
                    state.observation_due = True
                    state.unknown = True
                    state.reason = "activation_epoch_registration_pending"
                    state.updated_at = utc_now()
                    self._persist_locked(identity, state)
                    continue
                state.activation_observed = True
                state.replay_proposal = proposal
                state.observation_due = False
                state.next_observation_at = now + _OBSERVATION_INTERVAL_SECONDS
                state.unknown = not state.consumer_registered
                state.updated_at = utc_now()
                all_acknowledged = self._all_lanes_acknowledged(state)
                if all_acknowledged and checkpoint is None and state.consumer_registered:
                    self._enter_dormant_locked(identity, state)
                else:
                    self._persist_locked(identity, state)

    def select_activation_acknowledgements(
        self,
        *,
        limit: int = 64,
    ) -> tuple[ActivationAcknowledgementIntent, ...]:
        """Select exact replay acknowledgements after every lane is quiescent."""
        with self._lock:
            selected = self._fair_states_locked(
                "acknowledgement",
                lambda state: (
                    state.binding.enabled
                    and state.consumer_registered
                    and state.checkpoint is not None
                    and self._all_lanes_acknowledged(state)
                    and (state.acknowledgement_intent is not None or state.replay_proposal is not None)
                ),
                limit,
            )
            intents: list[ActivationAcknowledgementIntent] = []
            for identity, state in selected:
                intent = state.acknowledgement_intent
                if intent is None:
                    proposal = state.replay_proposal
                    if proposal is None or state.checkpoint is None:
                        continue
                    epoch, sequence, floor, complete = proposal
                    intent = ActivationAcknowledgementIntent(
                        binding=state.binding,
                        epoch=epoch,
                        sequence=sequence,
                        reconstructed_floor=floor,
                        require_current=bool(complete and state.checkpoint == (epoch, sequence)),
                        checkpoint=state.checkpoint,
                        wake_generation=self._wake_generations.get(identity, 0),
                        lane_revision=state.lane_revision,
                    )
                    state.acknowledgement_intent = intent
                intents.append(intent)
            return tuple(intents)

    def retain_activation_acknowledgement_intent(
        self,
        binding: ProjectBinding,
        *,
        epoch: str,
        sequence: int,
        reconstructed_floor: int | None,
        require_current: bool,
    ) -> ActivationAcknowledgementIntent | None:
        """Return the local intent still owning one unresolved shared ack."""
        identity = self._identity(binding)
        with self._lock:
            state = self._states.get(identity)
            intent = None if state is None or state.binding != binding else state.acknowledgement_intent
            if (
                intent is None
                or intent.epoch != epoch
                or intent.sequence != sequence
                or intent.reconstructed_floor != reconstructed_floor
                or intent.require_current != require_current
            ):
                return None
            return intent

    def apply_activation_acknowledgements(
        self,
        intents: Sequence[ActivationAcknowledgementIntent],
        results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Absorb monotonic shared progress; only exact current completions retire."""
        by_project = {intent.binding.project_id: intent for intent in intents}
        with self._lock:
            for project_id, evidence in results.items():
                intent = by_project.get(project_id)
                acknowledgement = evidence.get("acknowledgement")
                if intent is None:
                    continue
                identity = self._identity(intent.binding)
                state = self._states.get(identity)
                if state is None or state.binding != intent.binding:
                    continue
                if (
                    evidence.get("outcome") != "acknowledged"
                    or not self._is_checkpoint_record(acknowledgement)
                    or acknowledgement != {"epoch": intent.epoch, "sequence": intent.sequence}
                ):
                    if state.acknowledgement_intent == intent:
                        state.acknowledgement_intent = None
                        state.replay_proposal = None
                        state.observation_due = True
                        state.unknown = True
                        state.reason = "activation_ack_unavailable"
                        state.updated_at = utc_now()
                        self._persist_locked(identity, state)
                    continue
                # The shared cursor is monotonic durable truth even when local
                # service turns changed while the worker was running.
                current_epoch = state.checkpoint[0] if state.checkpoint is not None else None
                if state.replay_epoch == intent.epoch and state.replay_sequence <= intent.sequence:
                    state.replay_sequence = intent.sequence
                elif state.replay_epoch is None or current_epoch == intent.epoch:
                    state.replay_epoch = intent.epoch
                    state.replay_sequence = intent.sequence
                is_exact = state.acknowledgement_intent == intent
                if is_exact:
                    state.acknowledgement_intent = None
                    state.replay_proposal = None
                if (
                    is_exact
                    and intent.require_current
                    and state.checkpoint == intent.checkpoint == (intent.epoch, intent.sequence)
                    and self._wake_generations.get(identity, 0) == intent.wake_generation
                    and state.lane_revision == intent.lane_revision
                    and self._all_lanes_acknowledged(state)
                ):
                    self._enter_dormant_locked(identity, state)
                else:
                    state.observation_due = True
                    state.reason = "activation_replay_pending"
                    state.updated_at = utc_now()
                    self._persist_locked(identity, state)

    def begin_turn(self, binding: ProjectBinding, lane: str) -> BindingTurn:
        """Capture one lane turn using only the latest isolated observation."""
        if lane not in SERVICE_LANES:
            raise ValueError(f"unknown working-set service lane: {lane!r}")
        identity = self._identity(binding)
        runtime_id = identity[0]
        with self._lock:
            state = self._states.get(identity)
            if state is None or state.binding.shared_root != binding.shared_root:
                return BindingTurn(
                    runtime_id,
                    binding.project_id,
                    binding.registration_generation,
                    binding.shared_root,
                    lane,
                    None,
                    unknown=True,
                )
            if state.state == "dormant":
                self._wake_locked(identity, state, reason="service_turn", unknown=state.unknown)
            state.state = "resident"
            state.acknowledgements[lane] = None
            state.acknowledged_lanes.discard(lane)
            state.lane_revision += 1
            state.startup_pending = True
            state.reason = "service_turn"
            state.updated_at = utc_now()
            self._add_resident_locked(identity)
            self._persist_locked(identity, state)
            return BindingTurn(
                runtime_id,
                binding.project_id,
                binding.registration_generation,
                binding.shared_root,
                lane,
                state.checkpoint,
                state.unknown or not state.activation_observed or not state.consumer_registered,
                self._wake_generations.get(identity, 0),
            )

    def is_current_turn(self, turn: BindingTurn) -> bool:
        """Check local proof identity without waking a binding or advancing a lane."""
        with self._lock:
            state = self._states.get(turn.identity)
            return bool(
                state is not None
                and state.binding.shared_root == turn.shared_root
                and turn.lane in SERVICE_LANES
                and turn.wake_generation == self._wake_generations.get(turn.identity, 0)
                and turn.checkpoint == state.checkpoint
                and not turn.unknown
                and not state.unknown
                and state.consumer_registered
            )

    def has_current_activation(self, binding: ProjectBinding) -> bool:
        """Return whether service turns can bind to a fully observed activation."""
        with self._lock:
            state = self._states.get(self._identity(binding))
            return bool(
                state is not None
                and state.binding == binding
                and not state.unknown
                and state.consumer_registered
                and state.activation_observed
            )

    def is_lane_quiescent(self, binding: ProjectBinding, lane: str) -> bool:
        """Recognize a settled lane without certifying final activation replay."""
        if lane not in SERVICE_LANES:
            raise ValueError("unknown binding service lane.")
        with self._lock:
            state = self._states.get(self._identity(binding))
            return bool(
                state is not None
                and state.binding == binding
                and not state.unknown
                and state.consumer_registered
                and lane in state.acknowledged_lanes
                and state.acknowledgements[lane] == state.checkpoint
            )

    def is_turn_observation_pending(self, turn: BindingTurn) -> bool:
        """Recognize an unchanged unknown turn without certifying quiescence."""
        with self._lock:
            state = self._states.get(turn.identity)
            return bool(
                state is not None
                and state.binding.shared_root == turn.shared_root
                and turn.lane in SERVICE_LANES
                and turn.wake_generation == self._wake_generations.get(turn.identity, 0)
                and turn.checkpoint == state.checkpoint
                and turn.unknown
                and (state.unknown or not state.activation_observed or not state.consumer_registered)
            )

    def acknowledge(self, turn: BindingTurn, *, quiescent: bool) -> bool:
        """Record local quiescence; shared acknowledgement is asynchronous."""
        identity = turn.identity
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.binding.shared_root != turn.shared_root
                or turn.lane not in SERVICE_LANES
                or turn.runtime_id != identity[0]
                or turn.wake_generation != self._wake_generations.get(identity, 0)
                or turn.checkpoint != state.checkpoint
            ):
                return False
            if turn.unknown or state.unknown or not state.activation_observed or not state.consumer_registered:
                state.observation_due = True
                state.reason = "activation_observation_pending"
                state.updated_at = utc_now()
                self._persist_locked(identity, state)
                return False
            if not quiescent:
                state.acknowledgements[turn.lane] = None
                state.acknowledged_lanes.discard(turn.lane)
                state.lane_revision += 1
                state.state = "resident"
                state.startup_pending = True
                state.reason = "lane_active"
                state.updated_at = utc_now()
                self._add_resident_locked(identity)
                self._persist_locked(identity, state)
                return False

            state.acknowledgements[turn.lane] = state.checkpoint
            state.acknowledged_lanes.add(turn.lane)
            state.lane_revision += 1
            state.reason = "quiescent"
            state.updated_at = utc_now()
            all_acknowledged = state.acknowledged_lanes == set(SERVICE_LANES) and all(
                state.acknowledgements[lane] == state.checkpoint for lane in SERVICE_LANES
            )
            state.state = "resident"
            state.startup_pending = True
            self._add_resident_locked(identity)
            if all_acknowledged:
                state.reason = "activation_ack_pending"
                if state.checkpoint is None:
                    # A final isolated observation must confirm that no
                    # checkpoint appeared after the service turns.
                    state.activation_observed = False
                    state.replay_proposal = None
                    state.observation_due = True
            self._persist_locked(identity, state)
            return False

    def poll_dormant(
        self,
        bindings: Sequence[ProjectBinding] | None = None,
        *,
        limit: int = 4,
    ) -> list[ProjectBinding]:
        """Queue a bounded fair dormant slice for isolated observation."""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if bindings is not None and not bindings:
            return []
        now = _monotonic()

        eligibility: dict[_Identity, dict[Path, ProjectBinding]] | None = None
        if bindings is not None:
            eligibility = {}
            for binding in bindings:
                identity = self._identity(binding)
                eligibility.setdefault(identity, {})[binding.shared_root] = binding

        with self._lock:
            for _ in range(min(limit, len(self._dormant_roster))):
                identity, _marker = self._dormant_roster.popitem(last=False)
                self._dormant_roster[identity] = None
                if identity not in self._dormant_members:
                    continue
                state = self._states.get(identity)
                if state is None or state.state != "dormant":
                    continue
                if now < state.next_cold_reconcile_at:
                    continue
                roots = eligibility.get(identity) if eligibility is not None else None
                binding = (
                    state.binding if eligibility is None else roots.get(state.binding.shared_root) if roots else None
                )
                if eligibility is None and binding is not None and not binding.enabled:
                    continue
                if binding is not None:
                    state.observation_due = True
                    self._cold_probes += 1
        # Observation completion, not selection, decides whether to wake.
        return []

    def activate(self, binding: ProjectBinding, reason: str) -> None:
        """Restore exact current binding residency after local work appears."""
        self._validate_reason(reason)
        identity = self._identity(binding)
        with self._lock:
            state = self._states.get(identity)
            if state is None or state.binding.shared_root != binding.shared_root:
                raise ValueError("cannot activate a stale or mismatched project binding.")
            self._advance_wake_generation_locked(identity)
            is_discovery_failure = reason.startswith("discovery_")
            was_dormant = state.state == "dormant"
            state.state = "resident"
            state.acknowledgements = dict.fromkeys(SERVICE_LANES)
            state.acknowledged_lanes.clear()
            state.lane_revision += 1
            state.replay_proposal = None
            state.observation_due = True
            state.startup_pending = True
            state.reason = reason
            state.updated_at = utc_now()
            if is_discovery_failure:
                state.unknown = True
            elif not state.unknown:
                state.unknown = False
            if was_dormant:
                self._activations += 1
            self._add_resident_locked(identity)
            self._persist_locked(identity, state)

    def snapshot(self) -> dict[str, int]:
        """Return bounded coordinator counters for diagnostics and tests."""
        with self._lock:
            return {
                "dormant_bindings": sum(state.state == "dormant" for state in self._states.values()),
                "startup_pending_bindings": sum(state.startup_pending for state in self._states.values()),
                "unknown_bindings": sum(state.unknown for state in self._states.values()),
                "cold_probes": self._cold_probes,
                "activations": self._activations,
            }

    def _fair_states_locked(
        self,
        service: str,
        predicate: Any,
        limit: int,
    ) -> list[tuple[_Identity, _BindingState]]:
        if type(limit) is not int or limit < 1 or limit > 64:
            raise ValueError("activation selection limit must be an integer from 1 to 64.")
        candidates = [(identity, state) for identity, state in self._states.items() if predicate(state)]
        if not candidates:
            self._activation_offsets.pop(service, None)
            return []
        offset = self._activation_offsets.get(service, 0) % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        selected = ordered[:limit]
        self._activation_offsets[service] = (offset + len(selected)) % len(candidates)
        return selected

    @staticmethod
    def _is_checkpoint_record(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {"epoch", "sequence"}:
            return False
        epoch = value.get("epoch")
        try:
            parsed = uuid.UUID(hex=epoch) if isinstance(epoch, str) else None
        except ValueError:
            return False
        return bool(
            parsed is not None
            and parsed.int != 0
            and parsed.hex == epoch
            and type(value.get("sequence")) is int
            and value["sequence"] >= 1
        )

    @staticmethod
    def _all_lanes_acknowledged(state: _BindingState) -> bool:
        return state.acknowledged_lanes == set(SERVICE_LANES) and all(
            state.acknowledgements[lane] == state.checkpoint for lane in SERVICE_LANES
        )

    def _enter_dormant_locked(self, identity: _Identity, state: _BindingState) -> None:
        state.state = "dormant"
        state.startup_pending = False
        state.unknown = False
        state.reason = "quiescent"
        state.updated_at = utc_now()
        state.next_cold_reconcile_at = _monotonic() + _COLD_RECONCILE_SECONDS
        if self._persist_locked(identity, state):
            self._add_dormant_locked(identity)

    def _identity(self, binding: ProjectBinding) -> _Identity:
        with self._lock:
            if self._runtime_id is None:
                self._runtime_id = self._runtime.instance_id
            return self._runtime_id, binding.project_id, binding.registration_generation

    @staticmethod
    def _validate_reason(reason: str) -> None:
        if type(reason) is not str or not reason.strip():
            raise ValueError("working-set reason must be a nonempty string.")
        try:
            encoded = reason.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("working-set reason must be valid UTF-8.") from exc
        if len(encoded) > _MAX_REASON_BYTES or any(ord(char) < 0x20 or ord(char) == 0x7F for char in reason):
            raise ValueError("working-set reason must be at most 128 bytes without control characters.")

    def _progress_path(self, binding: ProjectBinding) -> Path:
        return self._runtime.project_paths(binding.project_id)["root"] / "working-set-v1.json"

    def _wake_locked(
        self,
        identity: _Identity,
        state: _BindingState,
        *,
        reason: str,
        unknown: bool,
        checkpoint: _CheckpointSignature | object = ...,
    ) -> None:
        self._advance_wake_generation_locked(identity)
        was_dormant = state.state == "dormant"
        state.state = "resident"
        state.acknowledgements = dict.fromkeys(SERVICE_LANES)
        state.acknowledged_lanes.clear()
        state.lane_revision += 1
        state.startup_pending = True
        state.reason = reason
        state.unknown = unknown
        state.updated_at = utc_now()
        if checkpoint is not ...:
            state.checkpoint = checkpoint  # type: ignore[assignment]
        state.replay_proposal = None
        state.observation_due = True
        if was_dormant:
            self._activations += 1
        self._add_resident_locked(identity)
        self._persist_locked(identity, state)

    def _advance_wake_generation_locked(self, identity: _Identity) -> None:
        self._wake_generations[identity] = self._wake_generations.get(identity, 0) + 1

    def _add_dormant_locked(self, identity: _Identity) -> None:
        self._resident_members.pop(identity, None)
        if identity not in self._dormant_members:
            self._dormant_members.add(identity)
        state = self._states.get(identity)
        if state is not None and state.binding.enabled:
            self._dormant_enabled.add(identity)
            self._dormant_roster.setdefault(identity, None)
            self._dormant_renewal_roster.setdefault(identity, None)
        else:
            self._dormant_enabled.discard(identity)
            self._dormant_roster.pop(identity, None)
            self._dormant_renewal_roster.pop(identity, None)

    def _add_resident_locked(self, identity: _Identity) -> None:
        self._remove_dormant_locked(identity)
        self._resident_members[identity] = None

    def _remove_dormant_locked(self, identity: _Identity) -> None:
        self._dormant_enabled.discard(identity)
        self._dormant_roster.pop(identity, None)
        self._dormant_renewal_roster.pop(identity, None)
        if identity in self._dormant_members:
            self._dormant_members.remove(identity)

    def _persist_locked(self, identity: _Identity, state: _BindingState) -> bool:
        path = self._progress_path(state.binding)
        checkpoint = self._signature_record(state.checkpoint)
        acknowledgements = {lane: self._signature_record(state.acknowledgements[lane]) for lane in SERVICE_LANES}
        try:
            atomic_replace(
                path,
                {
                    "working_set": {
                        "version": _RECORD_VERSION,
                        "identity": {
                            "runtime_id": identity[0],
                            "project_id": identity[1],
                            "registration_generation": identity[2],
                        },
                        "process_fence": self._process_fence,
                        "state": state.state,
                        "checkpoint": checkpoint,
                        "acknowledgements": acknowledgements,
                        "acknowledged_lanes": sorted(state.acknowledged_lanes),
                        "reason": state.reason,
                        "updated_at": state.updated_at,
                    }
                },
            )
        except (OSError, RuntimeError, ValueError, TypeError):
            state.state = "resident"
            state.acknowledgements = dict.fromkeys(SERVICE_LANES)
            state.acknowledged_lanes.clear()
            state.startup_pending = True
            state.reason = "local_progress_unavailable"
            state.unknown = True
            state.updated_at = utc_now()
            self._add_resident_locked(identity)
            return False
        return True

    @staticmethod
    def _signature_record(signature: _CheckpointSignature) -> dict[str, Any] | None:
        if signature is None:
            return None
        return {"epoch": signature[0], "sequence": signature[1]}
