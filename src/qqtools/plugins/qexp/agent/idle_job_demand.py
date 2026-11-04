"""Asynchronous, capacity-aware demand proofs for Machine Idle Job."""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from .bindings import ProjectBinding
from .dispatch_probe import PrimaryProbeSession
from .primary_probe_transport import encode_probe_session
from .project_io_protocol import PROJECT_IO_CAPACITY, ProjectIORequest
from .project_io_transport_support import ProjectIOProtocolError
from .working_set import BindingTurn

_MAX_CAPACITY = 4096
_MAX_CAPACITY_KEY_LENGTH = 4096
_MAX_SELECTION_WINDOW = 64
_ACTIVE_RETRY_SECONDS = 1.0


def _json_copy(value: Any) -> Any:
    """Copy protocol-shaped state before retaining it across a controller turn."""
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_copy(item) for item in value]
    return value


@dataclass(slots=True)
class _IdleJobRound:
    """One binding's independent capacity-scoped demand census."""

    turn: BindingTurn
    probe_state: Mapping[str, Any]
    state: str = "pending"
    retry_at: float = 0.0


@dataclass(frozen=True, slots=True)
class _RequestAssociation:
    """The exact round to which one unresolved probe belongs."""

    identity: Hashable
    round: _IdleJobRound


class _DemandController(Protocol):
    """Controller surface used by the independent demand coordinator."""

    runtime: Any
    executor: Any

    def _poll_admission_executor(self, *, force_poll: bool = False) -> Mapping[str, Any]: ...

    def _binding_identity(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        executor_epoch: str | None,
    ) -> Hashable | None: ...

    def _request_identity(self, request: ProjectIORequest) -> Hashable: ...

    def validated_config(self, binding: ProjectBinding, registry_revision: int) -> Any | None: ...

    def _offer_new_request(
        self,
        identity: Hashable,
        kind: str,
        action: Callable[[], None],
        *parts: str,
    ) -> None: ...

    def _service_retry_is_due(self, key: tuple[Hashable, str, str, str]) -> bool: ...

    def _record_service_failure(self, key: tuple[Hashable, str, str, str]) -> None: ...

    def _start_observation_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[Hashable, str, str, str],
    ) -> None: ...

    def scheduler_is_quiescent(self, binding: ProjectBinding, registry_revision: int) -> bool: ...


class IdleJobDemandCoordinator:
    """Coordinate bounded, non-mutating Idle Job demand probes.

    This coordinator deliberately has no activation acknowledgement side
    effects.  A demand result is useful only while the local scheduler turn
    captured for that result remains current.
    """

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._rounds: dict[tuple[Hashable, str], _IdleJobRound] = {}
        self._requests: dict[str, _RequestAssociation] = {}
        self._capacity_key: str | None = None
        self._capacity_values: tuple[int, int] | None = None
        self._selection_cursor: tuple[str, str, str] | None = None

    def advance(
        self,
        controller: _DemandController,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        gpu_capacity: int,
        cpu_capacity: int,
        capacity_key: str,
    ) -> str:
        """Advance demand proofs and return ``idle``, ``busy``, or ``unknown``."""
        self._validate_inputs(
            bindings,
            registry_revision,
            gpu_capacity=gpu_capacity,
            cpu_capacity=cpu_capacity,
            capacity_key=capacity_key,
        )

        status, unresolved = self._poll_executor(controller)
        if status is None:
            return "unknown"
        executor_epoch = status["executor_epoch"]
        try:
            controller._observed_executor_epoch = executor_epoch
        except (AttributeError, TypeError):
            pass

        capacity_changed = self._capacity_key != capacity_key or self._capacity_values != (
            gpu_capacity,
            cpu_capacity,
        )
        if capacity_changed:
            # Existing workers are still fenced through their exact requests;
            # their old associations remain long enough to stale-resolve them.
            self._rounds.clear()
            self._selection_cursor = None
        self._capacity_key = capacity_key
        self._capacity_values = (gpu_capacity, cpu_capacity)

        current, validation_blocked = self._current_bindings(
            controller,
            bindings,
            registry_revision,
            executor_epoch,
            capacity_key,
        )
        current_keys = set(current)
        self._prune_rounds(controller, current_keys)
        unresolved_ids = {request.request_id for request in unresolved}
        self._requests = {
            request_id: association
            for request_id, association in self._requests.items()
            if request_id in unresolved_ids
        }

        remaining_unresolved = self._advance_requests(
            controller,
            unresolved,
            current,
            registry_revision,
            capacity_key,
        )

        # A no-project daemon can be idle only once the executor itself is
        # known active and every retained request has been reconciled.
        if not current:
            if validation_blocked or remaining_unresolved:
                return "unknown"
            return "idle"

        self._offer_probes(
            controller,
            current,
            registry_revision,
            gpu_capacity,
            cpu_capacity,
            capacity_key,
            status,
        )

        has_active = False
        complete = True
        for cache_identity, (base_identity, binding) in current.items():
            round_state = self._rounds.get(cache_identity)
            if self._scheduler_proof_is_current(controller, binding, registry_revision):
                continue
            if round_state is None or not self._turn_is_current(controller, round_state.turn):
                complete = False
                continue
            if round_state.state == "active":
                has_active = True
            elif round_state.state != "quiescent":
                complete = False

        if has_active:
            return "busy"
        if complete and not validation_blocked:
            return "idle"
        return "unknown"

    @staticmethod
    def _validate_inputs(
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        gpu_capacity: int,
        cpu_capacity: int,
        capacity_key: str,
    ) -> None:
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes | bytearray):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        for value, label in ((gpu_capacity, "gpu_capacity"), (cpu_capacity, "cpu_capacity")):
            if type(value) is not int or not 0 <= value <= _MAX_CAPACITY:
                raise ValueError(f"{label} must be a bounded nonnegative integer.")
        if (
            not isinstance(capacity_key, str)
            or not capacity_key
            or len(capacity_key) > _MAX_CAPACITY_KEY_LENGTH
            or "\x00" in capacity_key
        ):
            raise ValueError("capacity_key must be a nonempty bounded string.")

    @staticmethod
    def _poll_executor(
        controller: _DemandController,
    ) -> tuple[Mapping[str, Any] | None, tuple[ProjectIORequest, ...]]:
        try:
            status = controller._poll_admission_executor()
            if not isinstance(status, Mapping):
                return None, ()
            epoch = status.get("executor_epoch")
            envelope = status.get("envelope")
            if not isinstance(epoch, str) or not epoch or not isinstance(envelope, str) or envelope == "unknown":
                return None, ()
            unresolved = tuple(controller.executor.unresolved_requests())
            if any(not isinstance(request, ProjectIORequest) for request in unresolved):
                return None, ()
            # The executor exposes this bound as part of its status.  Reject a
            # malformed value rather than accidentally creating an unbounded
            # preparation window.
            capacity = status.get("capacity", PROJECT_IO_CAPACITY)
            if type(capacity) is not int or not 0 <= capacity <= PROJECT_IO_CAPACITY:
                return None, ()
            return status, unresolved
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            return None, ()

    def _current_bindings(
        self,
        controller: _DemandController,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        executor_epoch: str,
        capacity_key: str,
    ) -> tuple[dict[tuple[Hashable, str], tuple[Hashable, ProjectBinding]], bool]:
        current: dict[tuple[Hashable, str], tuple[Hashable, ProjectBinding]] = {}
        validation_blocked = False
        for binding in bindings:
            if binding.enabled is not True:
                continue
            try:
                config = controller.validated_config(binding, registry_revision)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                config = None
            if config is None:
                validation_blocked = True
                continue
            try:
                base_identity = controller._binding_identity(binding, registry_revision, executor_epoch)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                base_identity = None
            if base_identity is None:
                validation_blocked = True
                continue
            cache_identity = (base_identity, capacity_key)
            if cache_identity in current:
                continue
            current[cache_identity] = (base_identity, binding)
        return current, validation_blocked

    def _prune_rounds(
        self,
        controller: _DemandController,
        current: set[tuple[Hashable, str]],
    ) -> None:
        for identity, round_state in tuple(self._rounds.items()):
            if identity not in current:
                self._rounds.pop(identity, None)
                continue
            if self._turn_is_current(controller, round_state.turn):
                continue
            if self._turn_observation_pending(controller, round_state.turn):
                continue
            self._rounds.pop(identity, None)

    def _advance_requests(
        self,
        controller: _DemandController,
        unresolved: Sequence[ProjectIORequest],
        current: Mapping[tuple[Hashable, str], tuple[Hashable, ProjectBinding]],
        registry_revision: int,
        capacity_key: str,
    ) -> tuple[ProjectIORequest, ...]:
        current_by_base = {
            base_identity: (cache_identity, binding) for cache_identity, (base_identity, binding) in current.items()
        }
        remaining: list[ProjectIORequest] = []
        for request in unresolved:
            if request.operation_kind != "idle_job_probe":
                remaining.append(request)
                continue
            association = self._requests.get(request.request_id)
            try:
                request_identity = controller._request_identity(request)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                request_identity = None
            expected = None if request_identity is None else current_by_base.get(request_identity)
            is_current = (
                association is not None
                and expected is not None
                and association.identity == expected[0]
                and self._rounds.get(association.identity) is association.round
            )
            if not is_current:
                if self._resolve_stale(controller, request):
                    self._drop_request(controller, request.request_id)
                else:
                    remaining.append(request)
                continue

            service_key = self._service_key(request_identity, capacity_key)
            try:
                result = controller.executor.consume(request.request_id, request)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
                remaining.append(request)
                continue
            if result is None:
                if request.request_id not in self._started_requests(controller) and self._retry_due(
                    controller, service_key
                ):
                    self._start_existing(controller, request, service_key)
                remaining.append(request)
                continue
            self._started_requests(controller).discard(request.request_id)
            self._requests.pop(request.request_id, None)
            if result.request != request:
                raise ProjectIOProtocolError("Idle Job demand result differs from its exact request.")
            round_state = association.round
            if result.status != "completed":
                self._record_failure(controller, service_key)
                if self._rounds.get(association.identity) is round_state:
                    round_state.state = "pending"
                continue
            evidence = result.evidence
            if not isinstance(evidence, Mapping) or evidence.get("state") not in {"quiescent", "active", "pending"}:
                raise ProjectIOProtocolError("Idle Job demand evidence has an invalid state.")
            if self._rounds.get(association.identity) is not round_state:
                continue
            round_state.probe_state = _json_copy(evidence["probe_state"])
            if evidence["state"] == "quiescent":
                round_state.state = "quiescent"
                round_state.retry_at = 0.0
            elif evidence["state"] == "active":
                round_state.state = "active"
                round_state.retry_at = self._monotonic() + _ACTIVE_RETRY_SECONDS
            else:
                round_state.state = "pending"
                round_state.retry_at = 0.0
            self._clear_failure(controller, service_key)
        return tuple(remaining)

    def _offer_probes(
        self,
        controller: _DemandController,
        current: Mapping[tuple[Hashable, str], tuple[Hashable, ProjectBinding]],
        registry_revision: int,
        gpu_capacity: int,
        cpu_capacity: int,
        capacity_key: str,
        status: Mapping[str, Any],
    ) -> None:
        ordered = sorted(current.items(), key=lambda item: self._binding_sort_key(item[1][1]))
        if not ordered:
            return
        tokens = [self._binding_sort_key(binding) for _identity, (_base, binding) in ordered]
        start = 0
        if self._selection_cursor in tokens:
            start = (tokens.index(self._selection_cursor) + 1) % len(tokens)
        window_size = min(len(ordered), _MAX_SELECTION_WINDOW)
        window = [ordered[(start + offset) % len(ordered)] for offset in range(window_size)]
        self._selection_cursor = self._binding_sort_key(window[-1][1][1])

        offer_limit = min(int(status.get("capacity", PROJECT_IO_CAPACITY)), PROJECT_IO_CAPACITY)
        if status.get("overdue_worker_count", 0) > status.get("supported_hang_limit", 0):
            offer_limit = 0
        offered = 0
        for cache_identity, (base_identity, binding) in window:
            if offered >= offer_limit:
                break
            if self._scheduler_proof_is_current(controller, binding, registry_revision):
                continue
            round_state = self._rounds.get(cache_identity)
            if round_state is not None:
                if round_state.state == "quiescent" and self._turn_is_current(controller, round_state.turn):
                    continue
                if round_state.state == "active" and round_state.retry_at > self._monotonic():
                    continue
            if any(association.identity == cache_identity for association in self._requests.values()):
                continue
            service_key = self._service_key(base_identity, capacity_key)
            if not self._retry_due(controller, service_key):
                continue
            controller._offer_new_request(
                base_identity,
                "idle_job_probe",
                partial(
                    self._prepare_probe,
                    controller,
                    cache_identity,
                    round_state,
                    binding,
                    registry_revision,
                    gpu_capacity,
                    cpu_capacity,
                    capacity_key,
                    service_key,
                ),
                capacity_key,
                "demand",
            )
            offered += 1

    def _prepare_probe(
        self,
        controller: _DemandController,
        cache_identity: tuple[Hashable, str],
        expected_round: _IdleJobRound | None,
        binding: ProjectBinding,
        registry_revision: int,
        gpu_capacity: int,
        cpu_capacity: int,
        capacity_key: str,
        service_key: tuple[Hashable, str, str, str],
    ) -> None:
        if self._capacity_key != capacity_key or self._rounds.get(cache_identity) is not expected_round:
            return
        state = (
            encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
            if expected_round is None
            else expected_round.probe_state
        )
        try:
            request = controller.executor.prepare_idle_job_probe(
                binding,
                registry_revision,
                probe_state=state,
                gpu_capacity=gpu_capacity,
                cpu_capacity=cpu_capacity,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._record_failure(controller, service_key)
            return
        round_state = expected_round
        if round_state is None:
            round_state = _IdleJobRound(
                controller.runtime.working_set.begin_turn(binding, "scheduler"), _json_copy(state)
            )
            self._rounds[cache_identity] = round_state
        request_id = request.request_id
        self._requests[request_id] = _RequestAssociation(cache_identity, round_state)
        try:
            controller._start_observation_request(request, service_key)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._record_failure(controller, service_key)

    @staticmethod
    def _binding_sort_key(binding: ProjectBinding) -> tuple[str, str, str]:
        return (
            binding.project_id,
            str(binding.shared_root),
            binding.registration_generation or "",
        )

    @staticmethod
    def _service_key(identity: Hashable | None, capacity_key: str) -> tuple[Hashable, str, str, str]:
        if identity is None:
            identity = "invalid"
        return identity, "idle_job_probe", capacity_key, "demand"

    @staticmethod
    def _turn_is_current(controller: _DemandController, turn: BindingTurn) -> bool:
        try:
            return bool(controller.runtime.working_set.is_current_turn(turn))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return False

    @staticmethod
    def _turn_observation_pending(controller: _DemandController, turn: BindingTurn) -> bool:
        try:
            return bool(controller.runtime.working_set.is_turn_observation_pending(turn))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return False

    @staticmethod
    def _scheduler_proof_is_current(
        controller: _DemandController,
        binding: ProjectBinding,
        registry_revision: int,
    ) -> bool:
        checker = getattr(controller, "scheduler_is_quiescent", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(binding, registry_revision))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return False

    @staticmethod
    def _started_requests(controller: _DemandController) -> set[str]:
        started = getattr(controller, "_started_requests", None)
        return started if isinstance(started, set) else set()

    @staticmethod
    def _retry_due(controller: _DemandController, key: tuple[Hashable, str, str, str]) -> bool:
        checker = getattr(controller, "_service_retry_is_due", None)
        if not callable(checker):
            return True
        try:
            return bool(checker(key))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False

    @staticmethod
    def _record_failure(controller: _DemandController, key: tuple[Hashable, str, str, str]) -> None:
        recorder = getattr(controller, "_record_service_failure", None)
        if callable(recorder):
            recorder(key)

    @staticmethod
    def _clear_failure(controller: _DemandController, key: tuple[Hashable, str, str, str]) -> None:
        backoff = getattr(controller, "_service_backoff", None)
        if isinstance(backoff, dict):
            backoff.pop(key, None)

    def _start_existing(
        self,
        controller: _DemandController,
        request: ProjectIORequest,
        service_key: tuple[Hashable, str, str, str],
    ) -> None:
        try:
            controller._start_observation_request(request, service_key)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._record_failure(controller, service_key)

    def _resolve_stale(self, controller: _DemandController, request: ProjectIORequest) -> bool:
        resolver = getattr(controller.executor, "resolve_stale_idle_job_probe", None)
        if not callable(resolver):
            resolver = getattr(controller.executor, "resolve_stale_request", None)
        if not callable(resolver):
            return False
        try:
            return bool(resolver(request.request_id, request))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            return False

    @staticmethod
    def _drop_request(controller: _DemandController, request_id: str) -> None:
        started = getattr(controller, "_started_requests", None)
        if isinstance(started, set):
            started.discard(request_id)


__all__ = ["IdleJobDemandCoordinator"]
