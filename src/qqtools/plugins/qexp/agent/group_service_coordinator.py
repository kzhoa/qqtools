"""Group-service round state and closed transition decisions."""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..runtime.group_discovery.probe import initial_group_service_probe_state
from .bindings import ProjectBinding
from .group_service_transport import (
    GroupServiceRequestSpec,
    group_service_advance_request,
    group_service_probe_request,
    initial_group_service_continuation,
)
from .project_io_protocol import ProjectIORequest
from .working_set import BindingTurn


def _json_copy(value: Any) -> Any:
    """Return a detached mutable copy of protocol-frozen JSON data."""
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_copy(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class GroupServiceTransition:
    """Intent produced by one Group-service state transition."""

    acknowledge_quiescent: bool = False
    restart_turn: bool = False


class _GroupWorkingSet(Protocol):
    def begin_turn(self, binding: ProjectBinding, lane: str) -> BindingTurn: ...

    def is_current_turn(self, turn: BindingTurn) -> bool: ...

    def is_lane_quiescent(self, binding: ProjectBinding, lane: str) -> bool: ...

    def is_turn_observation_pending(self, turn: BindingTurn) -> bool: ...

    def acknowledge(self, turn: BindingTurn, *, quiescent: bool) -> bool: ...


class _GroupServiceRequestPreparer(Protocol):
    def prepare_closed_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        request_spec: GroupServiceRequestSpec,
    ) -> ProjectIORequest: ...


@dataclass(slots=True)
class GroupServiceRound:
    """State retained while advancing one captured Group-service turn."""

    turn: BindingTurn
    probe_state: Mapping[str, Any]
    is_quiescent: bool = False
    candidate: Mapping[str, Any] | None = None
    continuation: Mapping[str, Any] | None = None
    settled_legacy: set[str] = field(default_factory=set)
    retry_at: float = 0.0

    def apply_probe(self, evidence: Mapping[str, Any]) -> GroupServiceTransition:
        """Apply validated probe evidence and return its acknowledgement intent."""
        self.probe_state = _json_copy(evidence["probe_state"])
        self.is_quiescent = evidence["state"] == "quiescent"
        candidate = _json_copy(evidence["candidate"])
        if candidate is not None and candidate["lane"] == "legacy" and candidate["group"] in self.settled_legacy:
            candidate = None
        self.candidate = candidate
        self.continuation = None if candidate is None else initial_group_service_continuation(candidate)
        self.retry_at = 0.0
        return GroupServiceTransition(acknowledge_quiescent=self.is_quiescent)

    def apply_advance(
        self,
        evidence: Mapping[str, Any],
        *,
        monotonic_time: float,
    ) -> GroupServiceTransition:
        """Apply validated advance evidence and return its transition intent."""
        candidate = self.candidate
        state = evidence["state"]
        follow_up = evidence["probe"]
        if state == "quiescent" and candidate is not None and candidate["lane"] == "legacy":
            self.settled_legacy.add(candidate["group"])
        if state in {"quiescent", "stale"}:
            self.candidate = None
            self.continuation = None
        else:
            self.continuation = _json_copy(evidence["continuation"])

        if follow_up is not None:
            self.probe_state = _json_copy(follow_up["probe_state"])
            self.is_quiescent = follow_up["state"] == "quiescent"
            next_candidate = _json_copy(follow_up["candidate"])
            self.candidate = next_candidate
            self.continuation = None if next_candidate is None else initial_group_service_continuation(next_candidate)
            if self.is_quiescent:
                acknowledge_quiescent = True
                restart_turn = False
            else:
                acknowledge_quiescent = False
                restart_turn = False
        else:
            acknowledge_quiescent = False
            restart_turn = candidate is None or candidate["lane"] != "legacy"

        self.retry_at = monotonic_time + 1.0 if state == "blocked" else 0.0
        return GroupServiceTransition(
            acknowledge_quiescent=acknowledge_quiescent,
            restart_turn=restart_turn,
        )

    def restart(self, turn: BindingTurn, probe_state: Mapping[str, Any]) -> None:
        """Replace the captured turn and probe state while retaining round progress."""
        self.turn = turn
        self.probe_state = _json_copy(probe_state)
        self.is_quiescent = False


@dataclass(frozen=True, slots=True)
class GroupServiceOffer:
    """Deferred Group-service offer carrying its exact expected round."""

    identity: Hashable
    operation_kind: str
    _expected_round: GroupServiceRound | None = field(default=None, repr=False, compare=False)


class GroupServiceCoordinator:
    """Own Group-service rounds, request associations, and local turn effects."""

    def __init__(self, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        self._rounds: dict[Hashable, GroupServiceRound] = {}
        self._requests: dict[str, GroupServiceRound] = {}
        self._current_identities: set[Hashable] | None = None

    def is_quiescent(self, identity: Hashable, working_set: _GroupWorkingSet) -> bool:
        """Return whether an owned round proves its current captured turn quiescent."""
        round_state = self._rounds.get(identity)
        return bool(
            round_state is not None and round_state.is_quiescent and working_set.is_current_turn(round_state.turn)
        )

    def reconcile(
        self,
        current: Mapping[Hashable, ProjectBinding],
        unresolved_request_ids: Collection[str],
        working_set: _GroupWorkingSet,
    ) -> None:
        """Prune Group rounds and request associations against current execution state."""
        self._current_identities = set(current)
        self._rounds = {
            identity: round_state
            for identity, round_state in self._rounds.items()
            if identity in current
            and (
                working_set.is_current_turn(round_state.turn)
                or working_set.is_turn_observation_pending(round_state.turn)
            )
        }
        unresolved = set(unresolved_request_ids)
        self._requests = {
            request_id: round_state for request_id, round_state in self._requests.items() if request_id in unresolved
        }

    def discard_request(self, request_id: str) -> None:
        """Forget one successfully resolved stale request association."""
        self._requests.pop(request_id, None)

    def request_round(self, request_id: str) -> GroupServiceRound | None:
        """Return the round associated with one unresolved request, if retained."""
        return self._requests.get(request_id)

    def apply_failure(
        self,
        identity: Hashable,
        request_id: str,
        binding: ProjectBinding,
        operation_kind: str,
        working_set: _GroupWorkingSet,
    ) -> bool:
        """Apply one accepted failure and restart only an exact advance association."""
        self._validate_operation_kind(operation_kind)
        round_state = self._requests.pop(request_id, None)
        if round_state is None or self._rounds.get(identity) is not round_state:
            return False
        if operation_kind == "group_service_advance":
            self._restart(round_state, binding, working_set)
        return True

    def apply_result(
        self,
        identity: Hashable,
        request_id: str,
        binding: ProjectBinding,
        operation_kind: str,
        evidence: Mapping[str, Any],
        working_set: _GroupWorkingSet,
    ) -> bool:
        """Apply one accepted completed result and its captured-turn effects."""
        self._validate_operation_kind(operation_kind)
        round_state = self._requests.pop(request_id, None)
        if round_state is None or self._rounds.get(identity) is not round_state:
            return False
        transition = (
            round_state.apply_probe(evidence)
            if operation_kind == "group_service_probe"
            else round_state.apply_advance(evidence, monotonic_time=self._monotonic())
        )
        if transition.acknowledge_quiescent and working_set.is_current_turn(round_state.turn):
            working_set.acknowledge(round_state.turn, quiescent=True)
        if transition.restart_turn:
            self._restart(round_state, binding, working_set)
        return True

    def next_offer(
        self,
        identity: Hashable,
        binding: ProjectBinding,
        working_set: _GroupWorkingSet,
    ) -> GroupServiceOffer | None:
        """Return the next Group offer after applying local turn state effects."""
        if self._current_identities is not None and identity not in self._current_identities:
            return None
        if working_set.is_lane_quiescent(binding, "group"):
            return None
        round_state = self._rounds.get(identity)
        if round_state is not None and round_state.is_quiescent:
            if working_set.is_current_turn(round_state.turn):
                working_set.acknowledge(round_state.turn, quiescent=True)
                return None
            if working_set.is_turn_observation_pending(round_state.turn):
                return None
            self._restart(round_state, binding, working_set)
        if round_state is not None and round_state.retry_at > self._monotonic():
            return None
        operation_kind = (
            "group_service_advance"
            if round_state is not None and round_state.candidate is not None
            else "group_service_probe"
        )
        return GroupServiceOffer(identity, operation_kind, round_state)

    def prepare_request(
        self,
        offer: GroupServiceOffer,
        binding: ProjectBinding,
        registry_revision: int,
        preparer: _GroupServiceRequestPreparer,
        working_set: _GroupWorkingSet,
    ) -> ProjectIORequest | None:
        """Prepare and associate one deferred offer after its exact-state recheck."""
        self._validate_operation_kind(offer.operation_kind)
        if self._current_identities is not None and offer.identity not in self._current_identities:
            return None
        round_state = self._rounds.get(offer.identity)
        if round_state is not offer._expected_round:
            return None
        if round_state is not None and round_state.candidate is not None:
            request_spec = group_service_advance_request(
                binding.machine_name,
                round_state.candidate,
                round_state.continuation,
                round_state.probe_state,
            )
        else:
            state = initial_group_service_probe_state() if round_state is None else round_state.probe_state
            request_spec = group_service_probe_request(binding.machine_name, state)
        request = preparer.prepare_closed_request(
            binding,
            registry_revision,
            request_spec,
        )
        if round_state is None:
            state = initial_group_service_probe_state()
            round_state = GroupServiceRound(working_set.begin_turn(binding, "group"), state)
            self._rounds[offer.identity] = round_state
        self._requests[request.request_id] = round_state
        return request

    @staticmethod
    def _validate_operation_kind(operation_kind: str) -> None:
        if operation_kind not in {"group_service_probe", "group_service_advance"}:
            raise ValueError(f"unsupported Group-service operation kind: {operation_kind!r}")

    @staticmethod
    def _restart(round_state: GroupServiceRound, binding: ProjectBinding, working_set: _GroupWorkingSet) -> None:
        if working_set.is_current_turn(round_state.turn):
            working_set.acknowledge(round_state.turn, quiescent=False)
        turn = working_set.begin_turn(binding, "group")
        round_state.restart(turn, initial_group_service_probe_state())


__all__ = [
    "GroupServiceCoordinator",
    "GroupServiceOffer",
    "GroupServiceRound",
    "GroupServiceTransition",
]
