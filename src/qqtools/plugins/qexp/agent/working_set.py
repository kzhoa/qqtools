"""Process-local binding residency coordinated by Project activation checkpoints."""

from __future__ import annotations

import math
import os
import stat
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from ..runtime.project_activation import (
    read_project_activation,
    read_project_activation_events,
    read_project_activation_snapshot,
)
from ..runtime.project_activation_consumers import ack_consumer, register_consumer
from ..runtime.records import utc_now
from ..runtime.store import atomic_replace, read_json_limited
from .bindings import ProjectBinding

if TYPE_CHECKING:
    from .context import MachineRuntime

SERVICE_LANES = ("scheduler", "authority", "group", "observation", "submission", "maintenance")
_LEGACY_SERVICE_LANES = frozenset({"scheduler", "authority", "group", "observation", "submission"})
_RECORD_VERSION = 1
_MAX_REASON_BYTES = 128
_MAX_RECORD_BYTES = 16 * 1024
_COLD_RECONCILE_SECONDS = 60.0
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

    @property
    def identity(self) -> _Identity:
        return self.runtime_id, self.project_id, self.registration_generation


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
            retries = [
                (identity, binding)
                for identity, binding in current.items()
                if (state := self._states.get(identity)) is not None
                and state.binding.shared_root == binding.shared_root
                and not state.consumer_registered
            ]

        # Read old local progress without holding the coordinator lock. Its state is
        # never trusted as readiness; only its integrity affects the unknown marker.
        progress_validity = {identity: self._valid_saved_progress(binding) for identity, binding in additions}
        registrations = {identity: self._register_consumer(binding) for identity, binding in additions + retries}

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
            for identity, _binding in retries:
                state = retained.get(identity)
                if state is not None and registrations[identity] is not None:
                    state.consumer_registered = True
                    state.replay_epoch, state.replay_sequence = registrations[identity]
                    state.unknown = False
                    state.reason = "consumer_registration_recovered"
                    state.updated_at = utc_now()
                    self._persist_locked(identity, state)
            self._wake_generations = {
                identity: generation for identity, generation in self._wake_generations.items() if identity in retained
            }
            for identity, binding in additions:
                if identity in retained:
                    continue
                state = _BindingState(
                    binding=binding,
                    unknown=not progress_validity[identity] or registrations[identity] is None,
                    consumer_registered=registrations[identity] is not None,
                )
                if registrations[identity] is not None:
                    state.replay_epoch, state.replay_sequence = registrations[identity]
                if state.unknown:
                    state.reason = "startup_progress_unknown"
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

    def renew_dormant_registrations(
        self,
        *,
        limit: int = 64,
        heartbeat_interval_seconds: float = 5.0,
    ) -> tuple[int, int]:
        """Renew a bounded fair slice without restoring hot Project context."""
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
        renewed = 0
        for binding in selected:
            try:
                renewed += bool(
                    self._runtime.binding_write_eligible(
                        binding,
                        renew=True,
                        renewal_horizon_seconds=renewal_horizon,
                    )
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
        return len(selected), renewed

    def begin_turn(self, binding: ProjectBinding, lane: str) -> BindingTurn:
        """Observe activation before one lane performs service work."""
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

        signature, unknown = self._read_signature(binding)

        with self._lock:
            state = self._states.get(identity)
            if state is None or state.binding.shared_root != binding.shared_root:
                return BindingTurn(
                    runtime_id,
                    binding.project_id,
                    binding.registration_generation,
                    binding.shared_root,
                    lane,
                    signature,
                    unknown=True,
                )
            if unknown:
                self._wake_locked(identity, state, reason="checkpoint_unreadable", unknown=True)
            elif state.state == "dormant":
                reason = "checkpoint_changed" if signature != state.checkpoint else "service_turn"
                self._wake_locked(identity, state, reason=reason, unknown=False, checkpoint=signature)
            else:
                if signature != state.checkpoint:
                    state.acknowledgements = dict.fromkeys(SERVICE_LANES)
                    state.acknowledged_lanes.clear()
                    state.startup_pending = True
                state.checkpoint = signature
                state.state = "resident"
                state.acknowledgements[lane] = None
                state.acknowledged_lanes.discard(lane)
                state.startup_pending = True
                state.reason = "service_turn"
                state.unknown = False
                state.updated_at = utc_now()
                self._persist_locked(identity, state)
        return BindingTurn(
            runtime_id,
            binding.project_id,
            binding.registration_generation,
            binding.shared_root,
            lane,
            signature,
            unknown,
        )

    def acknowledge(self, turn: BindingTurn, *, quiescent: bool) -> bool:
        """Commit a lane acknowledgement only if its checkpoint is still current."""
        identity = turn.identity
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.binding.shared_root != turn.shared_root
                or turn.lane not in SERVICE_LANES
                or turn.runtime_id != identity[0]
            ):
                return False
            wake_generation = self._wake_generations.get(identity, 0)

        signature, unknown = self._read_signature(state.binding)

        with self._lock:
            current_state = self._states.get(identity)
            if (
                current_state is not state
                or current_state.binding.shared_root != turn.shared_root
                or turn.runtime_id != identity[0]
                or self._wake_generations.get(identity, 0) != wake_generation
            ):
                return False
            state = current_state
            if unknown or turn.unknown:
                self._wake_locked(identity, state, reason="checkpoint_unreadable", unknown=True)
                return False
            if signature != turn.checkpoint:
                self._wake_locked(
                    identity,
                    state,
                    reason="checkpoint_changed",
                    unknown=False,
                    checkpoint=signature,
                )
                return False
            if not quiescent:
                state.acknowledgements[turn.lane] = None
                state.acknowledged_lanes.discard(turn.lane)
                state.state = "resident"
                state.startup_pending = True
                state.reason = "lane_active"
                state.updated_at = utc_now()
                self._add_resident_locked(identity)
                self._persist_locked(identity, state)
                return False

            state.checkpoint = signature
            state.acknowledgements[turn.lane] = signature
            state.acknowledged_lanes.add(turn.lane)
            state.reason = "quiescent"
            state.updated_at = utc_now()
            all_acknowledged = state.acknowledged_lanes == set(SERVICE_LANES) and all(
                state.acknowledgements[lane] == signature for lane in SERVICE_LANES
            )
            if all_acknowledged and not self._persist_locked(identity, state):
                self._wake_locked(identity, state, reason="local_progress_unavailable", unknown=True)
                return False
            replay_before = (state.replay_epoch, state.replay_sequence, state.reconstructed_floor)
            replay_complete = True
            if all_acknowledged and signature is not None:
                try:
                    if state.replay_epoch != signature[0]:
                        registration = self._register_consumer(state.binding)
                        if registration is None:
                            raise RuntimeError("Project activation consumer re-registration failed")
                        state.consumer_registered = True
                        state.replay_epoch, state.replay_sequence = registration
                    replay_complete = self._replay_events(state, signature)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    self._wake_locked(identity, state, reason="activation_replay_unavailable", unknown=True)
                    return False
            if all_acknowledged and replay_complete:
                was_dormant = state.state == "dormant"
                state.state = "dormant"
                state.startup_pending = False
                state.unknown = False
                if not was_dormant:
                    state.next_cold_reconcile_at = _monotonic() + _COLD_RECONCILE_SECONDS
            else:
                state.state = "resident"
                state.startup_pending = True
                if all_acknowledged:
                    state.reason = "activation_replay_pending"
                self._add_resident_locked(identity)
            persisted = self._persist_locked(identity, state)
            if persisted and all_acknowledged:
                if not state.consumer_registered or turn.registration_generation is None:
                    persisted = False
                elif signature is not None:
                    try:
                        ack_consumer(
                            turn.shared_root,
                            runtime_id=turn.runtime_id,
                            project_id=turn.project_id,
                            registration_generation=turn.registration_generation,
                            process_fence=self._process_fence,
                            epoch=signature[0],
                            sequence=state.replay_sequence,
                            reconstructed_floor=state.reconstructed_floor,
                            require_current=state.state == "dormant",
                        )
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        persisted = False
                if not persisted:
                    state.replay_epoch, state.replay_sequence, state.reconstructed_floor = replay_before
                    self._wake_locked(identity, state, reason="consumer_ack_unavailable", unknown=True)
                    return False
            if not replay_complete:
                return False
            if persisted and state.state == "dormant":
                self._add_dormant_locked(identity)
            else:
                self._remove_dormant_locked(identity)
            return persisted

    def poll_dormant(
        self,
        bindings: Sequence[ProjectBinding] | None = None,
        *,
        limit: int = 4,
    ) -> list[ProjectBinding]:
        """Cold-check a bounded fair slice of dormant bindings."""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if bindings is not None and not bindings:
            return []

        eligibility: dict[_Identity, dict[Path, ProjectBinding]] | None = None
        registry_identities: list[_Identity] = []
        if bindings is not None:
            eligibility = {}
            for binding in bindings:
                identity = self._identity(binding)
                registry_identities.append(identity)
                eligibility.setdefault(identity, {})[binding.shared_root] = binding

        selected: list[tuple[_Identity, _BindingState, ProjectBinding, _CheckpointSignature]] = []
        with self._lock:
            for _ in range(min(limit, len(self._dormant_roster))):
                if len(selected) >= limit:
                    break
                identity, _marker = self._dormant_roster.popitem(last=False)
                self._dormant_roster[identity] = None
                if identity not in self._dormant_members:
                    continue
                state = self._states.get(identity)
                if state is None or state.state != "dormant":
                    continue
                roots = eligibility.get(identity) if eligibility is not None else None
                binding = (
                    state.binding if eligibility is None else roots.get(state.binding.shared_root) if roots else None
                )
                if eligibility is None and binding is not None and not binding.enabled:
                    continue
                if binding is not None:
                    selected.append((identity, state, binding, state.checkpoint))

        observations: list[
            tuple[_Identity, _BindingState, ProjectBinding, _CheckpointSignature, _CheckpointSignature, bool]
        ] = []
        for identity, observed_state, binding, checkpoint in selected:
            try:
                signature, unknown = self._read_signature(binding)
            finally:
                with self._lock:
                    self._cold_probes += 1
            observations.append((identity, observed_state, binding, checkpoint, signature, unknown))

        with self._lock:
            awakened: set[tuple[_Identity, Path]] = set()
            for identity, observed_state, binding, checkpoint, signature, unknown in observations:
                state = self._states.get(identity)
                if (
                    state is not observed_state
                    or state.binding.shared_root != binding.shared_root
                    or state.state != "dormant"
                    or state.checkpoint != checkpoint
                ):
                    continue
                if unknown:
                    self._wake_locked(identity, state, reason="checkpoint_unreadable", unknown=True)
                    awakened.add((identity, binding.shared_root))
                elif signature != state.checkpoint:
                    self._wake_locked(
                        identity,
                        state,
                        reason="checkpoint_changed",
                        unknown=False,
                        checkpoint=signature,
                    )
                    awakened.add((identity, binding.shared_root))
                elif _monotonic() >= state.next_cold_reconcile_at:
                    self._wake_locked(identity, state, reason="cold_reconciliation", unknown=False)
                    awakened.add((identity, binding.shared_root))
            if bindings is None:
                return [
                    binding
                    for identity, _state, binding, _checkpoint, _signature, _unknown in observations
                    if (identity, binding.shared_root) in awakened
                ]
            return [
                binding
                for binding, identity in zip(bindings, registry_identities, strict=True)
                if (identity, binding.shared_root) in awakened
            ]

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

    def _identity(self, binding: ProjectBinding) -> _Identity:
        with self._lock:
            if self._runtime_id is None:
                self._runtime_id = self._runtime.instance_id
            return self._runtime_id, binding.project_id, binding.registration_generation

    def _register_consumer(self, binding: ProjectBinding) -> tuple[str | None, int] | None:
        if binding.registration_generation is None:
            return None
        try:
            record = register_consumer(
                binding.shared_root,
                runtime_id=self._identity(binding)[0],
                project_id=binding.project_id,
                registration_generation=binding.registration_generation,
                process_fence=self._process_fence,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return None
        acknowledgement = record["project_activation_consumer"]["ack"]
        if acknowledgement is None:
            return None, 0
        return acknowledgement["epoch"], acknowledgement["sequence"]

    @staticmethod
    def _replay_events(state: _BindingState, signature: tuple[str, int]) -> bool:
        epoch, sequence = signature
        if state.replay_epoch != epoch:
            state.replay_epoch = epoch
            state.replay_sequence = 0
        state.reconstructed_floor = None
        if state.replay_sequence > sequence:
            raise ValueError("activation replay cursor exceeds the current checkpoint")
        snapshot = read_project_activation_snapshot(state.binding.shared_root)
        if snapshot is not None:
            snapshot_record = snapshot["project_activation_snapshot"]
            if snapshot_record["epoch"] == epoch and state.replay_sequence < snapshot_record["floor_sequence"]:
                # This path is reached only after every service lane has durably
                # reconciled the binding's authoritative active/waiting indexes.
                state.replay_sequence = snapshot_record["floor_sequence"]
                state.reconstructed_floor = snapshot_record["floor_sequence"]
        if state.replay_sequence == sequence:
            return True
        batch = read_project_activation_events(
            state.binding.shared_root,
            epoch=epoch,
            after_sequence=state.replay_sequence,
            limit=min(256, sequence - state.replay_sequence),
        )
        if not batch:
            raise ValueError("activation replay made no progress")
        state.replay_sequence += len(batch)
        return state.replay_sequence == sequence

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

    @staticmethod
    def _read_signature(binding: ProjectBinding) -> tuple[_CheckpointSignature, bool]:
        try:
            record = read_project_activation(binding.shared_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return None, True
        if record is None:
            return None, False
        value = record["project_activation"]
        return (value["epoch"], value["sequence"]), False

    def _valid_saved_progress(self, binding: ProjectBinding) -> bool:
        path = self._progress_path(binding)
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECORD_BYTES:
                return False
            value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES, record_type="working_set")
            record = value.get("working_set")
            if type(record) is not dict or set(record) != {
                "version",
                "identity",
                "process_fence",
                "state",
                "checkpoint",
                "acknowledgements",
                "acknowledged_lanes",
                "reason",
                "updated_at",
            }:
                return False
            identity = record["identity"]
            if type(identity) is not dict or set(identity) != {"runtime_id", "project_id", "registration_generation"}:
                return False
            if identity != {
                "runtime_id": self._runtime_id,
                "project_id": binding.project_id,
                "registration_generation": binding.registration_generation,
            }:
                return False
            if type(record["version"]) is not int or record["version"] != _RECORD_VERSION:
                return False
            if type(record["process_fence"]) is not str or not record["process_fence"]:
                return False
            if record["state"] not in {"resident", "dormant"}:
                return False
            if not self._valid_signature(record["checkpoint"]):
                return False
            acks = record["acknowledgements"]
            if type(acks) is not dict or (set(acks) != set(SERVICE_LANES) and set(acks) != _LEGACY_SERVICE_LANES):
                return False
            if not all(self._valid_signature(signature) for signature in acks.values()):
                return False
            acknowledged_lanes = record["acknowledged_lanes"]
            if (
                type(acknowledged_lanes) is not list
                or len(acknowledged_lanes) != len(set(acknowledged_lanes))
                or any(lane not in SERVICE_LANES for lane in acknowledged_lanes)
            ):
                return False
            if record["state"] == "dormant" and (
                set(acknowledged_lanes) != set(SERVICE_LANES)
                or any(acks[lane] != record["checkpoint"] for lane in SERVICE_LANES)
            ):
                return False
            self._validate_reason(record["reason"])
            if type(record["updated_at"]) is not str or not record["updated_at"]:
                return False
            timestamp = datetime.fromisoformat(record["updated_at"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                return False
            return True
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            return False

    @staticmethod
    def _valid_signature(value: object) -> bool:
        if value is None:
            return True
        epoch = value.get("epoch") if type(value) is dict else None
        if not isinstance(epoch, str):
            return False
        try:
            parsed_epoch = uuid.UUID(hex=epoch)
        except (ValueError, AttributeError):
            return False
        return (
            type(value) is dict
            and set(value) == {"epoch", "sequence"}
            and epoch == parsed_epoch.hex
            and parsed_epoch.int != 0
            and type(value["sequence"]) is int
            and value["sequence"] >= 1
        )

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
        state.startup_pending = True
        state.reason = reason
        state.unknown = unknown
        state.updated_at = utc_now()
        if checkpoint is not ...:
            state.checkpoint = checkpoint  # type: ignore[assignment]
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
