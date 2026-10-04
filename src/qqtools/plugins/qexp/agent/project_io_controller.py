"""Controller-side state for asynchronous Project binding validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, Iterator

from ..config_types import RootConfig
from ..runtime.authority_scan import validate_evidence_path
from ..runtime.paths import local_paths, machine_project_paths
from ..runtime.records import validate_identifier
from ..runtime.resources.cpu_lane import cpu_reservation_snapshot, reserve_cpu, retag_cpu_if_matches
from ..runtime.resources.reservations import (
    ReservationIdentity,
    attach_executor_offer,
    classify_exact_reservation,
    classify_executor_offer,
    release_executor_offer,
    release_if_matches,
    reservation_snapshot,
    reserve,
    reserve_admitted,
    retag_if_matches,
)
from ..runtime.store import read_json_limited
from ..runtime.submission_control_continuation import submission_control_continuation
from .bindings import ProjectBinding
from .context import MachineRuntime
from .dispatch_probe import PrimaryProbeSession
from .group_service_coordinator import GroupServiceCoordinator, GroupServiceOffer
from .primary_probe_transport import encode_probe_session
from .process_reservation_recovery import registration_reservation
from .progress_coordinator import ProgressObservationCoordinator
from .project_io_admission import ProjectIOAdmission, ServiceIntent
from .project_io_executor import ProjectIOExecutor, ProjectIOProtocolError
from .project_io_protocol import ProjectIORequest
from .ready_cursor_validation import validate_ready_cursor_transition
from .working_set import BindingTurn

_INITIAL_BACKOFF_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 300.0
_RECOVERY_WAIT_RETRY_SECONDS = 1.0
_MAX_RETAINED_SCHEDULER_INTENTS = 64
_MAX_RETAINED_RESERVATION_RECONCILIATIONS = 256
_EVENT_FILENAME = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{32})\.json\Z")
_AUTHORITY_OPERATION_KINDS = frozenset(
    {
        "authority_service",
        "authority_renewal",
        "authority_orphan_recovery",
        "authority_terminal_observe",
        "authority_termination_commit",
        "authority_terminal_publish",
        "authority_running_publish",
    }
)
_AUTHORITY_MUTATION_OPERATION_KINDS = frozenset(
    {
        "authority_renewal",
        "authority_orphan_recovery",
        "authority_termination_commit",
        "authority_terminal_publish",
        "authority_running_publish",
    }
)
_SERVICE_CLASS_BY_OPERATION_KIND = {
    "upgrade_service": "background",
    "submission_control_service": "background",
    "observation_service": "background",
    "notification_service": "background",
    "group_service_probe": "background",
    "group_service_advance": "background",
    "progress_projection": "background",
    "legacy_capture_read": "background",
    "legacy_capture_scan": "background",
    "recovery_source_hold": "background",
    "recovery_admission": "background",
    "recovery_source_release": "background",
    "recovery_group_authority": "background",
    "recovery_capture_transition": "background",
    "validate_binding": "authority",
    "registration_renew": "authority",
    "authority_service": "authority",
    "authority_renewal": "authority",
    "authority_orphan_recovery": "authority",
    "authority_terminal_observe": "authority",
    "authority_termination_commit": "authority",
    "authority_terminal_publish": "authority",
    "authority_running_publish": "authority",
    "scheduler_observe": "primary",
    "scheduler_primary_probe": "primary",
    "scheduler_cursor_commit": "primary",
    "scheduler_claim": "primary",
    "scheduler_launch_authorize": "primary",
    "scheduler_reservation_reconcile": "primary",
    "scheduler_due_offer": "background",
    "scheduler_quiescence_probe": "primary",
    "scheduler_ready_index_build": "background",
    "maintenance_descriptor_advance": "background",
    "maintenance_flush_event": "background",
    "machine_snapshot_publish": "background",
    "activation_observe": "background",
    "activation_consumer_register": "background",
    "activation_consumer_ack": "background",
    "activation_consumer_retire": "background",
}


def _admission_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run one controller operation inside the shared admission turn."""

    @wraps(method)
    def wrapped(self: ProjectIOController, *args: Any, **kwargs: Any) -> Any:
        with self.admission_turn():
            return method(self, *args, **kwargs)

    return wrapped


def _json_copy(value: Any) -> Any:
    """Return a detached mutable copy of protocol-frozen JSON data."""
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_copy(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _ValidationIdentity:
    runtime_id: str
    executor_epoch: str
    project_id: str
    shared_root: str
    machine_name: str
    registration_generation: str
    runtime_root: str
    registry_revision: int

    @property
    def request_owner(self) -> tuple[str, str, str]:
        return (self.runtime_id, self.project_id, self.registration_generation)


@dataclass(frozen=True, slots=True)
class _FailureBackoff:
    consecutive_failures: int
    retry_at: float


@dataclass(slots=True)
class _ClaimBatchBudget:
    remaining_cpu_slots: int


@dataclass(frozen=True, slots=True)
class BorrowAdmissionGrant:
    """One-turn capability proving complete primary-demand absence."""

    runtime_id: str
    executor_epoch: str
    registry_revision: int
    lane: str
    identities: frozenset[_ValidationIdentity]
    capacity_digest: str


@dataclass(slots=True)
class _PrimaryProbeRound:
    round_id: str
    identities: frozenset[_ValidationIdentity]
    capacity_digest: str
    states: dict[_ValidationIdentity, Mapping[str, Any]]
    scanned: set[_ValidationIdentity]
    verified: set[_ValidationIdentity]
    phase: str = "scan"


@dataclass(slots=True)
class _SchedulerQuiescenceRound:
    turn: BindingTurn
    probe_state: Mapping[str, Any]
    is_quiescent: bool = False
    retry_at: float = 0.0


@dataclass(frozen=True, slots=True)
class _PendingClaimOffer:
    identity: _ValidationIdentity
    binding: ProjectBinding
    candidate: Mapping[str, Any]
    cursor: Mapping[str, Any]
    evidence: Mapping[str, Any]
    offer: Mapping[str, Any]
    service_key: tuple[_ValidationIdentity, str, str, str]


class ProjectIOController:
    """Advance binding validation using only MachineRuntime and executor state."""

    def __init__(
        self,
        runtime: MachineRuntime,
        executor: ProjectIOExecutor,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
        self._admission = ProjectIOAdmission(monotonic=monotonic)
        self._admission_scope_active = False
        self._initial_request_ids: set[str] = set()
        self._needs_successor_turn = False
        self._monotonic = monotonic
        self._observed_executor_epoch: str | None = None
        self._validated: set[_ValidationIdentity] = set()
        self._backoff: dict[_ValidationIdentity, _FailureBackoff] = {}
        self._service_backoff: dict[tuple[_ValidationIdentity, str, str, str], _FailureBackoff] = {}
        self._reservation_reconcile_backoff: dict[tuple[_ValidationIdentity, str, str, str], _FailureBackoff] = {}
        self._observations: dict[tuple[_ValidationIdentity, str, str, str], Mapping[str, Any]] = {}
        self._empty_observations: dict[tuple[_ValidationIdentity, str, str, str], Mapping[str, Any]] = {}
        self._observation_offsets: dict[tuple[str, str], int] = {}
        self._started_requests: set[str] = set()
        self._authorized_reservations: set[ReservationIdentity] = set()
        self._trusted_reservations: dict[ReservationIdentity, None] = {}
        self._reservation_reconcile_binding_cursor: tuple[str, str] | None = None
        self._reservation_reconcile_cursors: dict[tuple[str, str], str] = {}
        self._due_offer_binding_cursor: tuple[str, str] | None = None
        self._ready_index_binding_cursor: tuple[str, str] | None = None
        self._maintenance_descriptor_binding_cursor: tuple[str, str] | None = None
        self._maintenance_descriptor_settled: dict[_ValidationIdentity, BindingTurn] = {}
        self._maintenance_descriptor_dirty: set[_ValidationIdentity] = set()
        self._maintenance_descriptor_turns: dict[str, BindingTurn] = {}
        self._ready_index_settled: dict[_ValidationIdentity, None] = {}
        self._ready_index_needed: dict[_ValidationIdentity, None] = {}
        self._snapshot_last_published: dict[_ValidationIdentity, tuple[str, float]] = {}
        self._activation_offsets: dict[str, int] = {}
        self._primary_rounds: dict[str, _PrimaryProbeRound] = {}
        self._scheduler_quiescence_rounds: dict[_ValidationIdentity, _SchedulerQuiescenceRound] = {}
        self._scheduler_quiescence_requests: dict[str, _SchedulerQuiescenceRound] = {}
        self._group_service = GroupServiceCoordinator(monotonic=monotonic)
        self._borrow_grants: dict[str, BorrowAdmissionGrant] = {}
        self._pending_claim_offers: dict[str, _PendingClaimOffer] = {}
        self._offer_recovery_cursor: str | None = None
        self._upgrade_status: dict[_ValidationIdentity, Mapping[str, Any]] = {}
        self._upgrade_due: dict[_ValidationIdentity, float] = {}
        self._submission_cursors: dict[_ValidationIdentity, Mapping[str, Any]] = {}
        self._submission_due: dict[_ValidationIdentity, float] = {}
        self._observation_due: dict[_ValidationIdentity, float] = {}
        self._notification_due: dict[_ValidationIdentity, float] = {}
        self._observation_turns: dict[_ValidationIdentity, BindingTurn] = {}
        self._registration_due: dict[_ValidationIdentity, float] = {}
        self._registration_deadlines: dict[_ValidationIdentity, float] = {}
        self._submission_turns: dict[_ValidationIdentity, BindingTurn] = {}
        self.progress = ProgressObservationCoordinator(runtime, clock=monotonic)

    @_admission_operation
    def advance_upgrade_work(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> None:
        """Apply exact journal summaries locally and offer one fair upgrade slice."""
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if not isinstance(epoch, str) or status.get("envelope") == "unknown":
            self.runtime.upgrade_discovery_unknown = True
            return
        self._observed_executor_epoch = epoch
        current = {
            identity: binding
            for binding in bindings
            if (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
        }
        self._upgrade_status = {key: value for key, value in self._upgrade_status.items() if key in current}
        self._upgrade_due = {key: value for key, value in self._upgrade_due.items() if key in current}
        unresolved = self.executor.unresolved_requests()
        owners = {self._request_identity(request).request_owner for request in unresolved}
        for request in unresolved:
            if request.operation_kind != "upgrade_service":
                continue
            identity = self._request_identity(request)
            if identity not in current:
                if self.executor.resolve_stale_upgrade_service(request.request_id, request):
                    owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            key = (identity, "upgrade_service", "upgrade", "advance")
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                pass
            else:
                continue
            result = self.executor.load_result(request.request_id)
            if result is not None and result.request != request:
                raise ProjectIOProtocolError("upgrade service result changed request identity.")
            if result is not None and result.status == "outcome_unknown":
                if self.executor.reset_ambiguous_upgrade_service_for_retry(request.request_id, request):
                    self._record_service_failure(key)
                result = None
            if result is None:
                if self._service_retry_is_due(key):
                    self._start_upgrade_request(request, key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                self._upgrade_status[identity] = consumed.evidence
                self._service_backoff.pop(key, None)
                delay = 1.0
                next_probe = consumed.evidence["next_probe_at"]
                if next_probe is not None:
                    due_at = datetime.fromisoformat(next_probe.replace("Z", "+00:00"))
                    delay = max(delay, (due_at - datetime.now(timezone.utc)).total_seconds())
                self._upgrade_due[identity] = self._monotonic() + delay
            else:
                self._record_service_failure(key)
                self._upgrade_status.pop(identity, None)

        # Missing evidence blocks only its own Project, never healthy peers.
        unknown = {identity.project_id for identity in current if identity not in self._upgrade_status}
        self.runtime.upgrade_discovery_unknown = False
        self.runtime.upgrade_pending_projects = unknown | {
            identity.project_id for identity, evidence in self._upgrade_status.items() if evidence["pending"]
        }
        self.runtime.upgrade_idle_blocked_projects = unknown | {
            identity.project_id for identity, evidence in self._upgrade_status.items() if evidence["idle_blocking"]
        }
        self.runtime.upgrade_admission_blocked_projects = unknown | {
            identity.project_id for identity, evidence in self._upgrade_status.items() if evidence["admission_blocked"]
        }
        self.runtime.upgrade_runnable_projects = {
            identity.project_id for identity, evidence in self._upgrade_status.items() if evidence["can_run"]
        }
        self.runtime.upgrade_probe_deadlines = {
            identity.project_id: due
            for identity, due in self._upgrade_due.items()
            if self._upgrade_status.get(identity, {}).get("pending")
        }
        self.runtime.upgrade_discovery_complete = not unknown
        self.runtime.upgrade_registry_revision = registry_revision
        for identity, binding in current.items():
            evidence = self._upgrade_status.get(identity)
            if (evidence is not None and not evidence["pending"]) or self._upgrade_due.get(
                identity, 0.0
            ) > self._monotonic():
                continue
            key = (identity, "upgrade_service", "upgrade", "advance")
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                "upgrade_service",
                partial(self._prepare_upgrade_request, binding, registry_revision, key),
                "upgrade",
            )

    def _prepare_upgrade_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_upgrade_service(binding, registry_revision)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(key)
            return
        self._start_upgrade_request(request, key)

    def _start_upgrade_request(
        self,
        request: ProjectIORequest,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            process = None
        if process is None:
            self._record_service_failure(key)
        else:
            self._started_requests.add(request.request_id)
            self._service_backoff.pop(key, None)

    @property
    def has_pending_admission(self) -> bool:
        """Return whether discovery or result consumption needs another service turn."""
        return (
            self._admission.pending_count > 0
            or self._admission.has_deferred_work
            or self._needs_successor_turn
            or bool(self._pending_claim_offers)
        )

    def _mark_ready_index_needed(self, identity: _ValidationIdentity) -> None:
        """Retain one bounded, insertion-ordered ready-index repair intent."""
        if identity in self._ready_index_needed:
            return
        if len(self._ready_index_needed) >= _MAX_RETAINED_SCHEDULER_INTENTS:
            victim = next(
                (
                    retained
                    for retained in self._ready_index_needed
                    if (retained, "scheduler_ready_index_build", "ready_index", "advance") in self._service_backoff
                ),
                next(iter(self._ready_index_needed)),
            )
            del self._ready_index_needed[victim]
        self._ready_index_needed[identity] = None

    def pending_wait_seconds(self, default: float) -> float:
        """Bound idle waits while admitted work or responsive workers need polling."""
        if self._admission.pending_count or self._pending_claim_offers:
            return min(default, 0.1)
        if self._admission.has_deferred_work or self._needs_successor_turn:
            return min(default, 0.1)
        if self.executor.has_ready_result():
            return min(default, 0.1)
        status = self.executor.status_view()
        if status["active_worker_count"] > status["overdue_worker_count"] + status["exit_unverified_worker_count"] or (
            status["free_slot_count"] < status["capacity"] and status["overdue_worker_count"] == 0
        ):
            return min(default, 0.1)
        return default

    @contextmanager
    def admission_turn(self) -> Iterator[None]:
        """Collect one pass, initializing its executor snapshot after input validation."""
        if self._admission_scope_active:
            yield
            return

        self._admission_scope_active = True
        self._needs_successor_turn = False
        try:
            try:
                yield
                if self._admission.active:
                    # The last producer may have consumed a result after its
                    # snapshot. Grant against final occupancy, not call order.
                    self._poll_admission_executor()
                    current_request_ids = {request.request_id for request in self.executor.unresolved_requests()}
                    self._needs_successor_turn = bool(self._initial_request_ids - current_request_ids)
            except BaseException:
                if self._admission.active:
                    self._admission.finish(failed=True)
                raise
            else:
                if self._admission.active:
                    self._admission.finish()
        finally:
            # A capability not selected by this turn's arbiter must expire.
            # Keep discovery progress, but require a fresh verification pass
            # before another capability can be issued.
            for lane in self._borrow_grants:
                round_state = self._primary_rounds.get(lane)
                if round_state is not None:
                    round_state.verified.clear()
            self._borrow_grants.clear()
            self._admission_scope_active = False
            self._initial_request_ids.clear()

    def _poll_admission_executor(self) -> dict[str, Any]:
        """Open the lazy admission snapshot at the existing validated I/O boundary."""
        status = self.executor.poll()
        unresolved = self.executor.unresolved_requests()
        blocked = {(request.runtime_id, request.project_id, request.registration_generation) for request in unresolved}
        free_slots = max(0, status["capacity"] - len(unresolved))
        if status["envelope"] == "unknown" or status["overdue_worker_count"] > status["supported_hang_limit"]:
            free_slots = 0
        if self._admission.active:
            self._admission.refresh_capacity(blocked=blocked, free_slots=free_slots)
            return status
        _registry_revision, bindings = self.runtime.load_registry_snapshot()
        retained_request_identities = {self._request_identity(request) for request in unresolved}
        # Only observed registry/epoch changes invalidate retry state. A busy
        # worker lock yields an unknown epoch, not evidence of a new epoch.
        # A producer's resident/admission subset is not the registry and must
        # never erase another lane's still-current backoff.
        self._service_backoff = {
            key: state
            for key, state in self._service_backoff.items()
            if (
                key[0].registry_revision == _registry_revision
                and (status.get("executor_epoch") is None or key[0].executor_epoch == status["executor_epoch"])
            )
            or key[0] in retained_request_identities
        }
        self._initial_request_ids = {request.request_id for request in unresolved}
        owners = [
            (self.executor.runtime_id, binding.project_id, binding.registration_generation)
            for binding in bindings
            if isinstance(binding.registration_generation, str) and binding.registration_generation
        ]
        owners.extend(
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        )
        owners.extend(
            (self.executor.runtime_id, binding.project_id, binding.registration_generation)
            for binding in self.runtime.activation_consumer_retirements.select_pending(bindings, limit=64)
            if isinstance(binding.registration_generation, str) and binding.registration_generation
        )
        self._admission.begin(owners, blocked=blocked, free_slots=free_slots)
        return status

    def _offer_new_request(
        self,
        identity: _ValidationIdentity,
        kind: str,
        action: Callable[[], None],
        *parts: str,
        deadline: float | None = None,
    ) -> None:
        service_class = _SERVICE_CLASS_BY_OPERATION_KIND.get(kind)
        if service_class is None:
            raise ValueError(f"operation kind {kind!r} has no Project I/O admission class.")
        self._admission.offer(
            ServiceIntent(
                owner=identity.request_owner,
                service_class=service_class,
                operation_kind=kind,
                deadline=deadline,
                work_family=(
                    f"{parts[0]}_{parts[1]}"
                    if kind
                    in {"scheduler_observe", "scheduler_primary_probe", "scheduler_cursor_commit", "scheduler_claim"}
                    else "default"
                ),
                work_key=(
                    identity.executor_epoch,
                    identity.shared_root,
                    identity.machine_name,
                    identity.runtime_root,
                    str(identity.registry_revision),
                    *parts,
                ),
            ),
            action,
        )

    @_admission_operation
    def advance_binding_validation(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, RootConfig]:
        """Consume current validation results, then start eligible validation requests."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            self._observed_executor_epoch = None
            return {}
        self._observed_executor_epoch = executor_epoch
        unresolved = self.executor.unresolved_requests()
        self._advance_prevalidation_authority_replays(unresolved, executor_epoch)
        retry_start_permitted = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        ordered_bindings: list[tuple[_ValidationIdentity, ProjectBinding]] = []
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or identity in current_bindings:
                continue
            current_bindings[identity] = binding
            ordered_bindings.append((identity, binding))

        unresolved_request_ids = {request.request_id for request in unresolved}
        self._started_requests.intersection_update(unresolved_request_ids)
        for request in unresolved:
            identity = self._request_identity(request)
            if request.operation_kind == "scheduler_launch_authorize" and (
                identity not in current_bindings or identity not in self._validated
            ):
                # A finished launch request occupies this owner's admission
                # slot, but consuming its authority requires validation. Retire
                # definitive evidence without launching; fresh validation and
                # exact authorization replay break that circular dependency,
                # including requests issued before a registry revision change.
                if self.executor.resolve_stale_scheduler_launch_authorize(request.request_id, request):
                    unresolved_request_ids.discard(request.request_id)
                    self._started_requests.discard(request.request_id)
            if request.operation_kind != "validate_binding":
                continue
            result = self.executor.consume(request.request_id, request)
            if result is None:
                identity = self._request_identity(request)
                if (
                    retry_start_permitted
                    and identity in current_bindings
                    and self._start_retry_is_due(identity, request.request_id)
                ):
                    self._start_existing(request, identity)
                continue
            unresolved_request_ids.discard(request.request_id)
            self._started_requests.discard(request.request_id)
            if result.request != request:
                raise ProjectIOProtocolError("consumed result does not repeat its exact request identity.")

            identity = self._request_identity(request)
            if identity not in current_bindings:
                continue
            if result.status == "completed":
                self._validated.add(identity)
                self._backoff.pop(identity, None)
            else:
                self._record_failure(identity)

        current_identities = set(current_bindings)
        self._validated.intersection_update(current_identities)
        self._backoff = {identity: state for identity, state in self._backoff.items() if identity in current_identities}

        overdue_count = executor_status["overdue_worker_count"]
        supported_hang_limit = executor_status["supported_hang_limit"]
        can_start = executor_status["envelope"] != "unknown" and overdue_count <= supported_hang_limit
        now = self._monotonic()

        if can_start:
            for identity, binding in ordered_bindings:
                if identity in self._validated:
                    continue
                failure = self._backoff.get(identity)
                if failure is not None and failure.retry_at > now:
                    continue
                self._offer_new_request(
                    identity,
                    "validate_binding",
                    partial(self._prepare_binding_validation_request, identity, binding, registry_revision),
                )

        return {
            binding.project_id: self._root_config(binding)
            for identity, binding in ordered_bindings
            if identity in self._validated
        }

    def _prepare_binding_validation_request(
        self,
        identity: _ValidationIdentity,
        binding: ProjectBinding,
        registry_revision: int,
    ) -> None:
        try:
            request = self.executor.prepare_validate_binding(binding, registry_revision)
        except (OSError, RuntimeError, ValueError):
            self._record_failure(identity)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_failure(identity)
            return
        if process is None:
            self._record_failure(identity)
        else:
            self._started_requests.add(request.request_id)
            self._backoff.pop(identity, None)

    def _advance_prevalidation_authority_replays(
        self,
        unresolved: Sequence[ProjectIORequest],
        executor_epoch: str,
    ) -> None:
        """Drain retained cross-epoch mutations before their owner can validate.

        Admission correctly prevents two requests for one Project owner. A
        retained mutation from the prior executor epoch must therefore replay
        before validation, rather than waiting for authority service that can
        run only after validation.
        """
        resolvers = {
            "authority_renewal": self.executor.resolve_stale_authority_renewal,
            "authority_orphan_recovery": self.executor.resolve_stale_authority_orphan_recovery,
            "authority_termination_commit": self.executor.resolve_stale_authority_termination_commit,
            "authority_terminal_publish": self.executor.resolve_stale_authority_terminal_publish,
        }
        for request in unresolved:
            resolver = resolvers.get(request.operation_kind)
            if resolver is None or request.executor_epoch == executor_epoch:
                continue
            try:
                resolver(request.request_id, request)
            except (OSError, RuntimeError, ValueError):
                continue

    @_admission_operation
    def advance_scheduler_observations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        lane: str,
        admission_role: str,
    ) -> dict[str, Mapping[str, Any]]:
        """Consume matching observations once and start bounded current requests."""
        if lane not in {"gpu", "cpu"}:
            raise ValueError("lane must be 'gpu' or 'cpu'.")
        if admission_role not in {"primary", "borrow"}:
            raise ValueError("admission_role must be 'primary' or 'borrow'.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            self._observed_executor_epoch = None
            return {}
        self._observed_executor_epoch = executor_epoch

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            if binding.enabled is not True or not self.runtime.working_set.has_current_activation(binding):
                continue
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or identity not in self._validated or identity in current_bindings:
                continue
            current_bindings[identity] = binding

        current_identities = set(current_bindings)
        selected_service = ("scheduler_observe", lane, admission_role)
        self._observations = {
            key: evidence
            for key, evidence in self._observations.items()
            if key[1:] != selected_service or key[0] in current_identities
        }
        valid_scheduler_services = {
            ("scheduler_observe", service_lane, service_role)
            for service_lane in ("gpu", "cpu")
            for service_role in ("primary", "borrow")
        }
        self._empty_observations = {
            key: evidence
            for key, evidence in self._empty_observations.items()
            if key[1:] in valid_scheduler_services and (key[1:] != selected_service or key[0] in current_identities)
        }

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._started_requests.intersection_update(unresolved_by_id)
        observations: dict[str, Mapping[str, Any]] = {
            binding.project_id: _json_copy(self._observations[service_key])
            for identity, binding in current_bindings.items()
            if (service_key := self._observation_service_key(identity, lane, admission_role)) in self._observations
        }

        def retained_candidate_count() -> int:
            return sum(key[1:] == ("scheduler_observe", lane, admission_role) for key in self._observations)

        def primary_priority(identity: _ValidationIdentity) -> int:
            priority = self._admission.candidate_priority(identity.request_owner, "primary")
            return priority if priority is not None else len(current_bindings) * 3 + 1

        def worst_retained_candidate() -> tuple[_ValidationIdentity, str, str, str]:
            return max(
                (
                    retained_key
                    for retained_key in self._observations
                    if retained_key[1:] == ("scheduler_observe", lane, admission_role)
                ),
                key=lambda retained_key: primary_priority(retained_key[0]),
            )

        for request in unresolved:
            if request.operation_kind != "scheduler_observe":
                continue
            if request.parameters["lane"] != lane or request.parameters["admission_role"] != admission_role:
                continue

            identity = self._request_identity(request)
            binding = current_bindings.get(identity)
            expected_service_namespace = f"scheduler-{request.project_id}-{admission_role}-{lane}"
            if request.parameters["cursor_namespace"] != expected_service_namespace:
                continue

            result = self.executor.consume(request.request_id, request)
            if result is None:
                service_key = self._observation_service_key(identity, lane, admission_role)
                if (
                    identity in current_bindings
                    and request.request_id not in self._started_requests
                    and self._service_retry_is_due(service_key)
                ):
                    self._start_observation_request(request, service_key)
                continue
            unresolved_by_id.pop(request.request_id, None)
            self._started_requests.discard(request.request_id)
            if result.request != request:
                raise ProjectIOProtocolError("consumed result does not repeat its exact request identity.")
            if binding is None:
                # Finished reads must not hold the slot needed for fresh
                # activation or validation. Discarding their transient evidence
                # grants no authority and leaves the durable cursor unchanged.
                continue

            service_key = self._observation_service_key(identity, lane, admission_role)
            if result.status != "completed":
                self._observations.pop(service_key, None)
                self._record_service_failure(service_key)
                continue
            self._service_backoff.pop(service_key, None)
            evidence = dict(result.evidence)
            ready_index_inactive = evidence.get("reason") == "ready_index_inactive"
            if ready_index_inactive:
                self._ready_index_settled.pop(identity, None)
                self._mark_ready_index_needed(identity)
                self._observations = {
                    retained_key: retained
                    for retained_key, retained in self._observations.items()
                    if retained_key[0] != identity
                }
                self._empty_observations = {
                    retained_key: retained
                    for retained_key, retained in self._empty_observations.items()
                    if retained_key[0] != identity
                }
            else:
                self._ready_index_needed.pop(identity, None)
            if evidence["outcome"] == "candidate":
                quiescence = self._scheduler_quiescence_rounds.pop(identity, None)
                if quiescence is not None and (
                    self.runtime.working_set.is_current_turn(quiescence.turn)
                    or self.runtime.working_set.is_turn_observation_pending(quiescence.turn)
                ):
                    self.runtime.working_set.activate(binding, "scheduler_candidate")
                self._empty_observations.pop(service_key, None)
                if service_key not in self._observations and retained_candidate_count() >= (
                    _MAX_RETAINED_SCHEDULER_INTENTS
                ):
                    # A completed observation is a finite continuation and
                    # receives one same-turn claim opportunity.  Its owner was
                    # moved to the back when observation started, so rejecting
                    # it by the *current* roster rank would discard every result
                    # from the 65th owner.  Evict old rediscoverable evidence;
                    # per-Attempt critical credit lets this fresh result reach
                    # claim before another discovery rotation can displace it.
                    evicted_key = worst_retained_candidate()
                    self._observations.pop(evicted_key)
                    evicted_binding = current_bindings.get(evicted_key[0])
                    if evicted_binding is not None:
                        observations.pop(evicted_binding.project_id, None)
                self._observations[service_key] = _json_copy(evidence)
                observations[binding.project_id] = _json_copy(evidence)
            else:
                # Empty scans are transient observations, not durable intents.
                # Keeping one would permanently hide work submitted afterwards.
                self._observations.pop(service_key, None)
                # An inactive index has no cursor progress to commit. Retaining
                # this result would create useless commit work while the exact
                # repair it requested is waiting for the same binding slot.
                if (
                    not ready_index_inactive
                    and service_key not in self._empty_observations
                    and len(self._empty_observations) >= _MAX_RETAINED_SCHEDULER_INTENTS
                ):
                    # Empty evidence is likewise transient. Its cursor has not
                    # committed yet, so eviction loses no authoritative progress
                    # and lets an omitted owner enter the bounded window.
                    evicted_key = next(iter(self._empty_observations))
                    self._empty_observations.pop(evicted_key)
                    evicted_binding = current_bindings.get(evicted_key[0])
                    if evicted_binding is not None:
                        observations.pop(evicted_binding.project_id, None)
                if not ready_index_inactive:
                    self._empty_observations[service_key] = _json_copy(evidence)
                observations[binding.project_id] = _json_copy(evidence)

        can_start = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )
        ordered_bindings = list(current_bindings.items())
        offset_key = (lane, admission_role)
        offset = self._observation_offsets.get(offset_key, 0)
        if ordered_bindings:
            offset %= len(ordered_bindings)
            ordered_bindings = ordered_bindings[offset:] + ordered_bindings[:offset]
            self._observation_offsets[offset_key] = (offset + 1) % len(ordered_bindings)
        retained_candidates = retained_candidate_count()
        rotated_full_window = False
        for identity, binding in ordered_bindings:
            if not can_start:
                break
            if identity in self._ready_index_needed:
                continue
            if binding.project_id in observations:
                continue
            service_key = self._observation_service_key(identity, lane, admission_role)
            if service_key in self._observations or service_key in self._empty_observations:
                # An empty observation owns its cursor until the typed cursor
                # commit consumes it. Starting another observation here keeps
                # the owner perpetually occupied and can starve that commit,
                # so the scan would repeat from the same durable position
                # forever while ready work before the cursor remains hidden.
                continue
            if not self._service_retry_is_due(service_key):
                continue
            if retained_candidates >= _MAX_RETAINED_SCHEDULER_INTENTS:
                # Discovery has its own bounded round-robin offset because an
                # owner cannot gain arbiter priority for a claim it has never
                # been allowed to observe.  Rotate one omitted owner into the
                # window unconditionally; the resulting finite continuation
                # receives per-Attempt claim credit in this same turn.
                evicted_key = worst_retained_candidate()
                self._observations.pop(evicted_key)
                evicted_binding = current_bindings.get(evicted_key[0])
                if evicted_binding is not None:
                    observations.pop(evicted_binding.project_id, None)
                retained_candidates -= 1
                rotated_full_window = True
            cursor_namespace = f"scheduler-{binding.project_id}-{admission_role}-{lane}"
            self._offer_new_request(
                identity,
                "scheduler_observe",
                partial(
                    self._prepare_scheduler_observation_request,
                    identity,
                    binding,
                    registry_revision,
                    lane,
                    admission_role,
                    cursor_namespace,
                    service_key,
                ),
                lane,
                admission_role,
            )
            if rotated_full_window:
                break

        return observations

    def _prepare_scheduler_observation_request(
        self,
        identity: _ValidationIdentity,
        binding: ProjectBinding,
        registry_revision: int,
        lane: str,
        admission_role: str,
        cursor_namespace: str,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        # Admission offers are collected before completed quiescence probes are
        # consumed later in the same machine pass.  Recheck the closed proof at
        # grant time so a stale observation offer cannot reopen a settled lane.
        if self.scheduler_is_quiescent(binding, registry_revision):
            return
        try:
            request = self.executor.prepare_scheduler_observe(
                binding,
                registry_revision,
                lane=lane,
                admission_role=admission_role,
                cursor_namespace=cursor_namespace,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
        else:
            self._started_requests.add(request.request_id)
            self._service_backoff.pop(service_key, None)

    @_admission_operation
    def advance_scheduler_cursor_commits(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        lane: str,
        admission_role: str,
        observations: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Commit controller-produced empty-scan cursors through the typed executor."""
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if not isinstance(lane, str) or lane not in {"gpu", "cpu"}:
            raise ValueError("lane must be 'gpu' or 'cpu'.")
        if not isinstance(admission_role, str) or admission_role not in {"primary", "borrow"}:
            raise ValueError("admission_role must be 'primary' or 'borrow'.")
        if not isinstance(observations, Mapping):
            raise ValueError("observations must be a mapping.")
        if len(observations) > _MAX_RETAINED_SCHEDULER_INTENTS:
            raise ValueError("observations exceed the per-pass scheduler intent limit.")
        for project_id, evidence in observations.items():
            self._validate_empty_observation(project_id, evidence, lane, admission_role)

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            self._observed_executor_epoch = None
            return {}
        self._observed_executor_epoch = executor_epoch

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        bindings_by_project: dict[str, tuple[_ValidationIdentity, ProjectBinding]] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if (
                identity is None
                or binding.enabled is not True
                or identity not in self._validated
                or identity in current_bindings
            ):
                continue
            current_bindings[identity] = binding
            bindings_by_project[binding.project_id] = (identity, binding)
        current_identities = set(current_bindings)
        valid_scheduler_services = {
            ("scheduler_observe", service_lane, service_role)
            for service_lane in ("gpu", "cpu")
            for service_role in ("primary", "borrow")
        }
        self._empty_observations = {
            key: evidence
            for key, evidence in self._empty_observations.items()
            if key[0] in current_identities and key[1:] in valid_scheduler_services
        }

        for project_id, evidence in observations.items():
            current = bindings_by_project.get(project_id)
            if current is None:
                raise ValueError("cursor observation does not belong to a current validated binding.")
            identity, _binding = current
            observation_key = self._observation_service_key(identity, lane, admission_role)
            provenance = self._empty_observations.get(observation_key)
            if provenance is None or dict(provenance) != dict(evidence):
                raise ValueError("cursor observation was not produced for this binding and scheduler service.")

        # Empty observations are durable controller intents until their exact
        # cursor commit is prepared. Admission offers are intentionally
        # transient, so a commit that lost this pass's four-slot arbitration
        # must be rediscovered from retained provenance on the next pass.
        pending_observations = {
            binding.project_id: evidence
            for identity, binding in current_bindings.items()
            if (evidence := self._empty_observations.get(self._observation_service_key(identity, lane, admission_role)))
            is not None
        }

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._started_requests.intersection_update(unresolved_by_id)
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        completions: dict[str, Mapping[str, Any]] = {}

        for request in unresolved:
            if request.operation_kind != "scheduler_cursor_commit":
                continue
            identity = self._request_identity(request)
            binding = current_bindings.get(identity)
            service = self._cursor_namespace_service(
                request.project_id,
                request.parameters["cursor"]["namespace"],
                lane,
                admission_role,
            )
            if service is None:
                continue
            service_key = self._cursor_commit_service_key(identity, lane, admission_role)
            if binding is None:
                try:
                    resolved = self.executor.resolve_stale_cursor_commit(request.request_id, request)
                except (OSError, RuntimeError, ValueError):
                    resolved = False
                if resolved:
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            retry_due = self._service_retry_is_due(service_key)
            try:
                reset_for_retry = self.executor.reset_ambiguous_cursor_commit_for_retry(
                    request.request_id,
                    request,
                )
            except (OSError, RuntimeError, ValueError):
                reset_for_retry = False
            if reset_for_retry:
                if service_key not in self._service_backoff:
                    self._record_service_failure(service_key)
                elif retry_due:
                    self._start_cursor_commit_request(request, service_key)
                continue
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                self._started_requests.discard(request.request_id)
                result = self.executor.load_result(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue

            if result is None or result.status == "outcome_unknown":
                continue
            if result.request != request:
                raise ProjectIOProtocolError("cursor commit result does not repeat its exact request identity.")
            if result.status == "completed":
                self._validate_cursor_commit_result(result.evidence)
                consumed = self.executor.consume(request.request_id, request)
                if consumed is None:
                    continue
                if consumed.request != request or consumed.status != "completed":
                    raise ProjectIOProtocolError("consumed cursor result differs from its reconciled result.")
                completions[request.project_id] = dict(consumed.evidence)
                unresolved_by_id.pop(request.request_id, None)
                unresolved_owners.discard(identity.request_owner)
                self._started_requests.discard(request.request_id)
                self._service_backoff.pop(service_key, None)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is not None:
                unresolved_by_id.pop(request.request_id, None)
                unresolved_owners.discard(identity.request_owner)
                self._started_requests.discard(request.request_id)
            self._record_service_failure(service_key)

        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation)
            for request in unresolved_by_id.values()
        }
        can_start = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )
        for project_id, evidence in pending_observations.items():
            if not can_start:
                break
            identity, binding = bindings_by_project[project_id]
            service_key = self._cursor_commit_service_key(identity, lane, admission_role)
            if not self._service_retry_is_due(service_key):
                continue
            observation_key = self._observation_service_key(identity, lane, admission_role)
            self._offer_new_request(
                identity,
                "scheduler_cursor_commit",
                partial(
                    self._prepare_scheduler_cursor_commit_request,
                    binding,
                    registry_revision,
                    evidence["cursor"],
                    evidence["source_revisions"],
                    observation_key,
                    service_key,
                ),
                lane,
                admission_role,
            )

        return completions

    def _prepare_scheduler_cursor_commit_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        cursor: Mapping[str, Any],
        source_revisions: Mapping[str, object],
        observation_key: tuple[_ValidationIdentity, str, str, str],
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_scheduler_cursor_commit(
                binding,
                registry_revision,
                cursor=cursor,
                source_revisions=source_revisions,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._empty_observations.pop(observation_key, None)
        self._start_cursor_commit_request(request, service_key)

    def scheduler_is_quiescent(self, binding: ProjectBinding, registry_revision: int) -> bool:
        """Check a cached exact activation-bound scan without shared reads."""
        if not isinstance(binding, ProjectBinding):
            raise ValueError("binding must be a ProjectBinding value.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        epoch = self._observed_executor_epoch
        identity = None if epoch is None else self._binding_identity(binding, registry_revision, epoch)
        round_state = self._scheduler_quiescence_rounds.get(identity)
        return bool(
            round_state is not None
            and round_state.is_quiescent
            and self.runtime.working_set.is_current_turn(round_state.turn)
        )

    def group_service_is_quiescent(self, binding: ProjectBinding, registry_revision: int) -> bool:
        """Check one cached activation-bound Group-service census locally."""
        if not isinstance(binding, ProjectBinding):
            raise ValueError("binding must be a ProjectBinding value.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        epoch = self._observed_executor_epoch
        identity = None if epoch is None else self._binding_identity(binding, registry_revision, epoch)
        return self._group_service.is_quiescent(identity, self.runtime.working_set)

    @_admission_operation
    def advance_group_service_probes(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Advance closed censuses and exact bounded Group transactions."""
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if not isinstance(epoch, str) or status.get("envelope") == "unknown":
            return
        self._observed_executor_epoch = epoch
        current = {
            identity: binding
            for binding in bindings
            if binding.enabled
            and (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
            and identity in self._validated
        }
        unresolved = self.executor.unresolved_requests()
        unresolved_ids = {request.request_id for request in unresolved}
        self._group_service.reconcile(current, unresolved_ids, self.runtime.working_set)
        for request in unresolved:
            if request.operation_kind not in {"group_service_probe", "group_service_advance"}:
                continue
            identity = self._request_identity(request)
            key = (
                identity,
                request.operation_kind,
                "group",
                "census" if request.operation_kind.endswith("probe") else "advance",
            )
            if identity not in current:
                if self.executor.resolve_stale_request(request.request_id, request):
                    self._started_requests.discard(request.request_id)
                    self._group_service.discard_request(request.request_id)
                continue
            pending_result = self.executor.load_result(request.request_id)
            if (
                request.operation_kind == "group_service_advance"
                and pending_result is not None
                and pending_result.status == "outcome_unknown"
            ):
                try:
                    reset = self.executor.reset_ambiguous_replayable_request_for_retry(
                        request.request_id,
                        request,
                    )
                except (OSError, RuntimeError, ValueError):
                    reset = False
                if reset:
                    self._started_requests.discard(request.request_id)
                    self._record_service_failure(key)
                    if self._service_retry_is_due(key):
                        self._start_observation_request(request, key)
                continue
            result = self.executor.consume(request.request_id, request)
            if result is None:
                if request.request_id not in self._started_requests and self._service_retry_is_due(key):
                    self._start_observation_request(request, key)
                continue
            self._started_requests.discard(request.request_id)
            if result.request != request:
                raise ProjectIOProtocolError("Group-service probe result differs from its exact request.")
            if result.status != "completed":
                if self._group_service.apply_failure(
                    identity,
                    request.request_id,
                    current[identity],
                    request.operation_kind,
                    self.runtime.working_set,
                ):
                    self._record_service_failure(key)
                continue
            if self._group_service.apply_result(
                identity,
                request.request_id,
                current[identity],
                request.operation_kind,
                result.evidence,
                self.runtime.working_set,
            ):
                self._service_backoff.pop(key, None)
        for identity, binding in current.items():
            offer = self._group_service.next_offer(identity, binding, self.runtime.working_set)
            if offer is None:
                continue
            operation_kind = offer.operation_kind
            key = (identity, operation_kind, "group", "advance" if operation_kind.endswith("advance") else "census")
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                operation_kind,
                partial(self._prepare_group_service_request, offer, binding, registry_revision, key),
            )

    def _prepare_group_service_request(
        self,
        offer: GroupServiceOffer,
        binding: ProjectBinding,
        registry_revision: int,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self._group_service.prepare_request(
                offer,
                binding,
                registry_revision,
                self.executor,
                self.runtime.working_set,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(key)
            return
        if request is None:
            return
        self._start_observation_request(request, key)

    @_admission_operation
    def advance_scheduler_quiescence(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Advance independent complete scans and acknowledge their captured turns."""
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if not isinstance(epoch, str) or status.get("envelope") == "unknown":
            return
        self._observed_executor_epoch = epoch
        census_dependencies = set(self._ready_index_needed) | {
            key[0] for key, evidence in self._observations.items() if evidence.get("outcome") == "candidate"
        }
        current = {
            identity: binding
            for binding in bindings
            if binding.enabled
            and self.runtime.working_set.has_current_activation(binding)
            and (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
            and identity in self._validated
            and identity not in census_dependencies
        }
        self._scheduler_quiescence_rounds = {
            identity: round_state
            for identity, round_state in self._scheduler_quiescence_rounds.items()
            if identity in current
            and (
                self.runtime.working_set.is_current_turn(round_state.turn)
                or self.runtime.working_set.is_turn_observation_pending(round_state.turn)
            )
        }
        unresolved = self.executor.unresolved_requests()
        unresolved_ids = {request.request_id for request in unresolved}
        self._scheduler_quiescence_requests = {
            request_id: round_state
            for request_id, round_state in self._scheduler_quiescence_requests.items()
            if request_id in unresolved_ids
        }
        for request in unresolved:
            if request.operation_kind != "scheduler_quiescence_probe":
                continue
            identity = self._request_identity(request)
            key = (identity, "scheduler_quiescence_probe", "scheduler", "quiescence")
            if identity not in current:
                if self.executor.resolve_stale_scheduler_quiescence_probe(request.request_id, request):
                    self._started_requests.discard(request.request_id)
                    self._scheduler_quiescence_requests.pop(request.request_id, None)
                continue
            result = self.executor.consume(request.request_id, request)
            if result is None:
                if request.request_id not in self._started_requests and self._service_retry_is_due(key):
                    self._start_observation_request(request, key)
                continue
            self._started_requests.discard(request.request_id)
            round_state = self._scheduler_quiescence_requests.pop(request.request_id, None)
            if result.request != request:
                raise ProjectIOProtocolError("scheduler quiescence result differs from its exact request.")
            if round_state is None or self._scheduler_quiescence_rounds.get(identity) is not round_state:
                continue
            if result.status != "completed":
                self._record_service_failure(key)
                continue
            self._service_backoff.pop(key, None)
            evidence = result.evidence
            round_state.probe_state = _json_copy(evidence["probe_state"])
            round_state.is_quiescent = evidence["state"] == "quiescent"
            if not round_state.is_quiescent:
                self.runtime.working_set.acknowledge(round_state.turn, quiescent=False)
                if evidence["state"] == "active" or evidence["probe_state"] == request.parameters["probe_state"]:
                    round_state.retry_at = self._monotonic() + 1.0
            elif self.runtime.working_set.is_current_turn(round_state.turn):
                self.runtime.working_set.acknowledge(round_state.turn, quiescent=True)
        for identity, binding in current.items():
            round_state = self._scheduler_quiescence_rounds.get(identity)
            key = (identity, "scheduler_quiescence_probe", "scheduler", "quiescence")
            if round_state is not None and (round_state.is_quiescent or round_state.retry_at > self._monotonic()):
                continue
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                "scheduler_quiescence_probe",
                partial(
                    self._prepare_scheduler_quiescence_probe, identity, binding, registry_revision, round_state, key
                ),
            )

    def _prepare_scheduler_quiescence_probe(
        self,
        identity: _ValidationIdentity,
        binding: ProjectBinding,
        registry_revision: int,
        round_state: _SchedulerQuiescenceRound | None,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if self._scheduler_quiescence_rounds.get(identity) is not round_state:
            return
        state = (
            encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
            if round_state is None
            else round_state.probe_state
        )
        try:
            request = self.executor.prepare_scheduler_quiescence_probe(binding, registry_revision, probe_state=state)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(key)
            return
        if round_state is None:
            round_state = _SchedulerQuiescenceRound(self.runtime.working_set.begin_turn(binding, "scheduler"), state)
            self._scheduler_quiescence_rounds[identity] = round_state
        self._scheduler_quiescence_requests[request.request_id] = round_state
        self._start_observation_request(request, key)

    @_admission_operation
    def issue_borrow_admission(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        lane: str,
        visible_capacity: int,
        free_capacity: int,
    ) -> BorrowAdmissionGrant | None:
        """Advance independent scan/verification rounds before issuing a capability."""
        if lane not in {"gpu", "cpu"}:
            raise ValueError("lane must be 'gpu' or 'cpu'.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if (
            type(visible_capacity) is not int
            or type(free_capacity) is not int
            or not 0 <= free_capacity <= visible_capacity <= 4096
        ):
            raise ValueError("primary probe capacities are invalid.")
        status = self._poll_admission_executor()
        executor_epoch = status.get("executor_epoch")
        if status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            self.invalidate_borrow_admission(lane)
            return None
        current_revision, registry_bindings = self.runtime.load_registry_snapshot()
        if current_revision != registry_revision:
            self.invalidate_borrow_admission(lane)
            return None
        enabled = [binding for binding in registry_bindings if binding.enabled]
        supplied = list(bindings)
        if len(supplied) != len(enabled) or set(supplied) != set(enabled):
            self.invalidate_borrow_admission(lane)
            return None
        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in supplied:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or identity not in self._validated:
                self.invalidate_borrow_admission(lane)
                return None
            current[identity] = binding
        identities = frozenset(current)
        if not identities:
            self.invalidate_borrow_admission(lane)
            return None
        local_digest, usage = self._primary_capacity_snapshot()
        # Include policy capacity and currently usable headroom in the round key.
        digest = hashlib.sha256(f"{local_digest}:{visible_capacity}:{free_capacity}".encode()).hexdigest()
        round_state = self._primary_rounds.get(lane)
        if round_state is None or round_state.identities != identities or round_state.capacity_digest != digest:
            round_state = _PrimaryProbeRound(uuid.uuid4().hex, identities, digest, {}, set(), set())
            self._primary_rounds[lane] = round_state
        unresolved = self.executor.unresolved_requests()
        for request in unresolved:
            if request.operation_kind != "scheduler_primary_probe" or request.parameters["lane"] != lane:
                continue
            result = self.executor.consume(request.request_id, request)
            identity = self._request_identity(request)
            key = (identity, "scheduler_primary_probe", lane, "primary")
            if result is None:
                if request.request_id not in self._started_requests and self._service_retry_is_due(key):
                    self._start_observation_request(request, key)
                continue
            self._started_requests.discard(request.request_id)
            if result.request != request:
                raise ProjectIOProtocolError("primary probe result differs from its exact request.")
            if (
                identity not in current
                or request.parameters["round_id"] != round_state.round_id
                or request.parameters["capacity_digest"] != digest
                or request.parameters["phase"] != round_state.phase
            ):
                continue
            if result.status != "completed":
                self._record_service_failure(key)
                continue
            self._service_backoff.pop(key, None)
            round_state.states[identity] = _json_copy(result.evidence["probe_state"])
            if result.evidence["demand"] == "no_primary_demand":
                target = round_state.scanned if round_state.phase == "scan" else round_state.verified
                target.add(identity)
            else:
                round_state.scanned.discard(identity)
                round_state.verified.discard(identity)
                if round_state.phase == "verify":
                    round_state.phase = "scan"
                    round_state.verified.clear()
        if round_state.scanned == identities and round_state.phase == "scan":
            round_state.phase = "verify"
        if round_state.verified != identities:
            completed = round_state.scanned if round_state.phase == "scan" else round_state.verified
            for identity, binding in current.items():
                key = (identity, "scheduler_primary_probe", lane, "primary")
                if identity in completed or not self._service_retry_is_due(key):
                    continue
                state = round_state.states.get(
                    identity, encode_probe_session(PrimaryProbeSession(), binding.project_id, lane)
                )
                self._offer_new_request(
                    identity,
                    "scheduler_primary_probe",
                    partial(
                        self._prepare_primary_probe,
                        binding,
                        registry_revision,
                        lane,
                        round_state,
                        visible_capacity,
                        free_capacity,
                        usage.get(binding.project_id, {}),
                        state,
                        key,
                    ),
                    lane,
                    "primary",
                )
            return None
        if self._primary_capacity_snapshot()[0] != local_digest:
            self._primary_rounds.pop(lane, None)
            return None
        grant = BorrowAdmissionGrant(
            runtime_id=self.executor.runtime_id,
            executor_epoch=executor_epoch,
            registry_revision=registry_revision,
            lane=lane,
            identities=identities,
            capacity_digest=local_digest,
        )
        self._borrow_grants[lane] = grant
        return grant

    def _primary_capacity_snapshot(self) -> tuple[str, dict[str, dict[str, int]]]:
        snapshot = reservation_snapshot(self.runtime.root)
        policy, _cpu_records = cpu_reservation_snapshot(self.runtime.root)
        records = sorted(snapshot.reservations, key=lambda record: record["reservation_id"])
        encoded = json.dumps(
            {"reservations": records, "cpu_capacity": policy.capacity}, sort_keys=True, separators=(",", ":")
        )
        usage: dict[str, dict[str, int]] = {}
        for record in records:
            project, group = record.get("project_id"), record.get("group_name")
            if isinstance(project, str) and isinstance(group, str):
                groups = usage.setdefault(project, {})
                groups[group] = groups.get(group, 0) + len(record.get("gpu_ids") or [])
        return hashlib.sha256(encoded.encode()).hexdigest(), usage

    @_admission_operation
    def invalidate_borrow_admission(self, lane: str) -> None:
        """Discard proof progress when an admission prerequisite is unavailable."""
        if lane not in {"gpu", "cpu"}:
            raise ValueError("lane must be 'gpu' or 'cpu'.")
        self._primary_rounds.pop(lane, None)
        self._borrow_grants.pop(lane, None)
        self._poll_admission_executor()
        for request in self.executor.unresolved_requests():
            if request.operation_kind == "scheduler_primary_probe" and request.parameters["lane"] == lane:
                if self.executor.resolve_stale_primary_probe(request.request_id, request):
                    self._started_requests.discard(request.request_id)

    def _prepare_primary_probe(
        self,
        binding: ProjectBinding,
        revision: int,
        lane: str,
        round_state: _PrimaryProbeRound,
        visible: int,
        free: int,
        usage: Mapping[str, int],
        state: Mapping[str, Any],
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if self._primary_rounds.get(lane) is not round_state:
            return
        try:
            request = self.executor.prepare_scheduler_primary_probe(
                binding,
                revision,
                lane=lane,
                round_id=round_state.round_id,
                phase=round_state.phase,
                capacity_digest=round_state.capacity_digest,
                visible_capacity=visible,
                free_capacity=free,
                group_gpu_usage=usage,
                probe_state=state,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_observation_request(request, service_key)

    @_admission_operation
    def advance_scheduler_claims(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        lane: str,
        admission_role: str,
        observations: Mapping[str, Mapping[str, Any]],
        available_gpu_ids: Sequence[int],
        available_cpu_slots: int,
        borrow_admission: BorrowAdmissionGrant | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        """Reconcile durable claims before reserving offers for current observations."""
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if not isinstance(lane, str) or lane not in {"gpu", "cpu"}:
            raise ValueError("lane must be 'gpu' or 'cpu'.")
        if not isinstance(admission_role, str) or admission_role not in {"primary", "borrow"}:
            raise ValueError("admission_role must be 'primary' or 'borrow'.")
        if not isinstance(observations, Mapping):
            raise ValueError("observations must be a mapping.")
        if not isinstance(available_gpu_ids, Sequence) or isinstance(available_gpu_ids, str | bytes):
            raise ValueError("available_gpu_ids must be a sequence of unique nonnegative integers.")
        gpu_ids = list(available_gpu_ids)
        if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("available_gpu_ids must be a sequence of unique nonnegative integers.")
        if type(available_cpu_slots) is not int or available_cpu_slots < 0:
            raise ValueError("available_cpu_slots must be a nonnegative integer.")

        candidate_intent_count = 0
        for project_id, evidence in observations.items():
            if not isinstance(project_id, str) or not project_id or not isinstance(evidence, Mapping):
                raise ValueError("observations must map project IDs to evidence objects.")
            if set(evidence) != {"outcome", "reason", "source_revisions", "candidate", "cursor"}:
                raise ValueError("scheduler observation evidence has missing or unknown fields.")
            if not isinstance(evidence["reason"], str) or evidence["reason"] not in {
                "candidate_ready",
                "no_candidate",
                "ready_index_inactive",
                "candidate_unresolved",
                "dependency_not_ready",
                "placement_rejected",
                "admission_role_mismatch",
                "borrow_admission_unavailable",
                "slice_exhausted",
            }:
                raise ValueError("scheduler observation reason is invalid.")
            if not isinstance(evidence["source_revisions"], Mapping):
                raise ValueError("scheduler observation source_revisions must be a mapping.")
            cursor = evidence["cursor"]
            if not isinstance(cursor, Mapping) or cursor.get("namespace") != (
                f"scheduler-{project_id}-{admission_role}-{lane}"
            ):
                raise ValueError("scheduler observation cursor does not match its project, role, and lane.")
            if set(cursor) != {"namespace", "routes"} or not isinstance(cursor["routes"], Mapping):
                raise ValueError("scheduler observation cursor is malformed.")
            if set(cursor["routes"]) != {"home", "shared"}:
                raise ValueError("scheduler observation cursor routes are malformed.")
            for scope, route in cursor["routes"].items():
                if not isinstance(route, Mapping) or set(route) != {"observed", "next"}:
                    raise ValueError("scheduler observation cursor route is malformed.")
                validate_ready_cursor_transition(route["observed"], route["next"], f"cursor.routes.{scope}")
            if not isinstance(evidence["outcome"], str) or evidence["outcome"] not in {"candidate", "none"}:
                raise ValueError("scheduler observation outcome is invalid.")
            candidate = evidence["candidate"]
            if evidence["outcome"] == "candidate":
                candidate_intent_count += 1
                if candidate_intent_count > _MAX_RETAINED_SCHEDULER_INTENTS:
                    raise ValueError("observations exceed the per-pass candidate limit.")
                if not isinstance(candidate, Mapping):
                    raise ValueError("candidate observation must contain a candidate object.")
                candidate_fields = {
                    "task_id",
                    "task_revision",
                    "ready_identity",
                    "ready_generation",
                    "ready_scope",
                    "ready_revision",
                    "catalog_revision",
                    "catalog_page",
                    "partition",
                    "marker_name",
                    "home_machine",
                    "attempt_number",
                    "attempt_id",
                    "fencing_token",
                    "lane",
                    "requested_gpus",
                    "requested_cpus",
                    "admission_role",
                    "group_name",
                    "group_revision",
                    "group_dispatch_epoch",
                    "group_worker_set_epoch",
                    "worker_state_epoch",
                    "worker_scheduling_role",
                    "gpu_limit_gpus",
                }
                if set(candidate) != candidate_fields:
                    raise ValueError("scheduler candidate has missing or unknown fields.")
                if candidate.get("lane") != lane or candidate.get("admission_role") != admission_role:
                    raise ValueError("candidate lane and admission role must match the claim request.")
                if type(candidate.get("requested_gpus")) is not int or type(candidate.get("requested_cpus")) is not int:
                    raise ValueError("candidate resource counts must be integers.")
                if candidate["requested_gpus"] < 0 or candidate["requested_cpus"] < 0:
                    raise ValueError("candidate resource counts must be nonnegative.")
                if (lane == "gpu" and candidate["requested_cpus"] != 0) or (
                    lane == "cpu" and candidate["requested_gpus"] != 0
                ):
                    raise ValueError("candidate resource counts do not match its lane.")
                if not isinstance(candidate.get("task_id"), str) or not isinstance(candidate.get("attempt_id"), str):
                    raise ValueError("candidate task identity is invalid.")
                if type(candidate.get("attempt_number")) is not int or candidate["attempt_number"] < 1:
                    raise ValueError("candidate attempt_number must be positive.")
                if type(candidate.get("fencing_token")) is not int or candidate["fencing_token"] < 1:
                    raise ValueError("candidate fencing_token must be positive.")
                ready_scope = candidate.get("ready_scope")
                if not isinstance(ready_scope, str) or ready_scope not in {"home", "shared"}:
                    raise ValueError("candidate ready_scope is invalid.")
                route = cursor["routes"].get(ready_scope)
                if not isinstance(route, Mapping):
                    raise ValueError("candidate ready_scope is invalid.")
                next_position = route["next"]
                if (
                    next_position["catalog_page"] != candidate.get("catalog_page")
                    or next_position["partition"] != candidate.get("partition")
                    or next_position["after_name"] != candidate.get("marker_name")
                    or next_position["revision"] <= route["observed"]["revision"]
                ):
                    raise ValueError("candidate does not match the observed cursor advance.")
                if evidence["reason"] != "candidate_ready":
                    raise ValueError("candidate observation must use candidate_ready reason.")
            elif candidate is not None:
                raise ValueError("empty scheduler observation cannot contain a candidate.")
            elif evidence["reason"] == "candidate_ready":
                raise ValueError("empty scheduler observation cannot use candidate_ready reason.")

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            self._observed_executor_epoch = None
            return {}
        self._observed_executor_epoch = executor_epoch

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or identity in current_bindings:
                continue
            if binding.enabled is not True or identity not in self._validated:
                continue
            current_bindings[identity] = binding
        current_identities = set(current_bindings)
        self._recover_unprepared_claim_offers()
        borrow_admission_valid = admission_role != "borrow"
        if admission_role == "borrow" and isinstance(borrow_admission, BorrowAdmissionGrant):
            current_revision, registry_bindings = self.runtime.load_registry_snapshot()
            enabled = {binding for binding in registry_bindings if binding.enabled}
            borrow_admission_valid = (
                self._borrow_grants.get(lane) is borrow_admission
                and borrow_admission.runtime_id == self.executor.runtime_id
                and borrow_admission.executor_epoch == executor_epoch
                and borrow_admission.registry_revision == registry_revision == current_revision
                and borrow_admission.lane == lane
                and borrow_admission.identities == frozenset(current_identities)
                and set(bindings) == enabled
            )

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._started_requests.intersection_update(unresolved_by_id)
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        processed_projects: set[str] = set()
        completions: dict[str, Mapping[str, Any]] = {}

        for request in unresolved:
            if request.operation_kind != "scheduler_claim":
                continue
            parameters = request.parameters
            if parameters["lane"] != lane or parameters["admission_role"] != admission_role:
                continue
            identity = self._request_identity(request)
            binding = current_bindings.get(identity)
            expected_namespace = f"scheduler-{request.project_id}-{admission_role}-{lane}"
            candidate = parameters["candidate"]
            if (
                parameters["cursor"]["namespace"] != expected_namespace
                or candidate["lane"] != lane
                or candidate["admission_role"] != admission_role
            ):
                continue
            processed_projects.add(request.project_id)
            service_key = self._claim_service_key(identity, lane, admission_role)
            retry_due = self._service_retry_is_due(service_key)
            try:
                reset_for_retry = self.executor.reset_ambiguous_claim_for_retry(request.request_id, request)
            except (OSError, RuntimeError, ValueError):
                reset_for_retry = False
            if reset_for_retry:
                if service_key not in self._service_backoff:
                    self._record_service_failure(service_key)
                elif retry_due:
                    self._start_claim_request(request, service_key, reconcile_only=binding is None)
                continue
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
                result = self.executor.load_result(request.request_id)
                if result is None:
                    if was_started or service_key not in self._service_backoff:
                        self._record_service_failure(service_key)
                    if executor_status["overdue_worker_count"] <= executor_status[
                        "supported_hang_limit"
                    ] and self._service_retry_is_due(service_key):
                        self._start_claim_request(request, service_key, reconcile_only=binding is None)
                    continue
            else:
                self._started_requests.add(request.request_id)
                continue

            if result.request != request:
                raise ProjectIOProtocolError("consumed result does not repeat its exact request identity.")
            if result.status == "fenced":
                try:
                    offer_identity = self._offer_identity(request)
                    release_executor_offer(self.runtime.root, offer_identity, "claim_fenced_before_mutation")
                    if classify_executor_offer(self.runtime.root, offer_identity) != "matching_released":
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
                consumed = self.executor.consume(request.request_id, request)
                if consumed is None:
                    continue
                unresolved_by_id.pop(request.request_id, None)
                unresolved_owners.discard(identity.request_owner)
                self._started_requests.discard(request.request_id)
                self._record_service_failure(service_key)
                continue
            if result.status != "completed":
                # An unknown claim result can follow a shared-side mutation.
                # Keep the request and offer until stronger evidence is available.
                continue

            try:
                offer_identity = self._offer_identity(request)
                outcome = result.evidence["outcome"]
                if outcome == "claimed":
                    attempt_id = result.evidence["attempt_id"]
                    fencing_token = result.evidence["fencing_token"]
                    attach_executor_offer(self.runtime.root, offer_identity, attempt_id, fencing_token)
                    if classify_executor_offer(self.runtime.root, offer_identity) != "matching_active":
                        continue
                    self._trust_reservation(offer_identity)
                elif outcome == "no_claim":
                    release_executor_offer(self.runtime.root, offer_identity)
                    if classify_executor_offer(self.runtime.root, offer_identity) != "matching_released":
                        continue
                else:
                    continue
            except (OSError, RuntimeError, ValueError):
                continue

            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            if consumed.request != request or consumed.status != "completed":
                raise ProjectIOProtocolError("consumed result differs from the reconciled claim result.")
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                self._service_backoff.pop(service_key, None)
            else:
                self._record_service_failure(service_key)
            if binding is not None:
                completions[binding.project_id] = dict(consumed.evidence)

        retry_start_permitted = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )
        if not retry_start_permitted:
            return completions

        pending_projects = self._advance_pending_claim_offers(current_bindings, lane, admission_role)
        batch_budget = _ClaimBatchBudget(available_cpu_slots)
        for identity, binding in current_bindings.items():
            if (
                binding.project_id in processed_projects
                or binding.project_id in completions
                or binding.project_id in pending_projects
            ):
                continue
            evidence = observations.get(binding.project_id)
            if not isinstance(evidence, Mapping) or evidence["outcome"] != "candidate":
                continue
            observation_key = self._observation_service_key(identity, lane, admission_role)
            if self._observations.get(observation_key) != evidence:
                continue
            if admission_role == "borrow" and not borrow_admission_valid:
                continue
            candidate = evidence["candidate"]
            cursor = evidence["cursor"]
            if candidate["lane"] != lane or candidate["admission_role"] != admission_role:
                continue
            requested_gpus = candidate["requested_gpus"]
            requested_cpus = candidate["requested_cpus"]
            if lane == "gpu":
                if requested_gpus < 1 or len(gpu_ids) < requested_gpus:
                    # This evidence is transient; the shared ready marker is
                    # authoritative.  Releasing an unclaimable cached candidate
                    # lets the bounded owner window inspect a feasible peer.
                    self._observations.pop(observation_key, None)
                    continue
            else:
                if requested_cpus < 1 or available_cpu_slots < requested_cpus:
                    # This evidence is transient; the shared ready marker is
                    # authoritative.  Releasing an unclaimable cached candidate
                    # lets the bounded owner window inspect a feasible peer.
                    self._observations.pop(observation_key, None)
                    continue

            service_key = self._claim_service_key(identity, lane, admission_role)
            if not self._service_retry_is_due(service_key):
                continue
            if admission_role == "borrow":
                # Resource allocation is a bounded local result effect, not a
                # new Project request. Spend this turn's proof now, then let
                # the common arbiter select the shared claim independently.
                self._prepare_scheduler_claim_request(
                    identity,
                    binding,
                    candidate,
                    cursor,
                    evidence,
                    lane,
                    admission_role,
                    list(gpu_ids),
                    batch_budget,
                    registry_revision,
                    observation_key,
                    service_key,
                    borrow_admission,
                    defer_request=True,
                )
                continue
            self._offer_new_request(
                identity,
                "scheduler_claim",
                partial(
                    self._prepare_scheduler_claim_request,
                    identity,
                    binding,
                    candidate,
                    cursor,
                    evidence,
                    lane,
                    admission_role,
                    list(gpu_ids),
                    batch_budget,
                    registry_revision,
                    observation_key,
                    service_key,
                    borrow_admission,
                ),
                lane,
                admission_role,
                candidate["attempt_id"],
            )

        return completions

    def _prepare_scheduler_claim_request(
        self,
        identity: _ValidationIdentity,
        binding: ProjectBinding,
        candidate: Mapping[str, Any],
        cursor: Mapping[str, Any],
        evidence: Mapping[str, Any],
        lane: str,
        admission_role: str,
        allowed_gpu_ids: list[int],
        batch_budget: _ClaimBatchBudget,
        registry_revision: int,
        observation_key: tuple[_ValidationIdentity, str, str, str],
        service_key: tuple[_ValidationIdentity, str, str, str],
        borrow_admission: BorrowAdmissionGrant | None,
        *,
        defer_request: bool = False,
    ) -> None:
        if defer_request and (
            len(self._pending_claim_offers) >= 4
            or any(
                pending.identity.request_owner == identity.request_owner
                for pending in self._pending_claim_offers.values()
            )
        ):
            return
        if admission_role == "borrow":
            current_revision, current_bindings = self.runtime.load_registry_snapshot()
            if (
                borrow_admission is None
                or self._borrow_grants.get(lane) is not borrow_admission
                or self._primary_capacity_snapshot()[0] != borrow_admission.capacity_digest
                or current_revision != borrow_admission.registry_revision
                or frozenset(
                    self._binding_identity(item, current_revision, borrow_admission.executor_epoch)
                    for item in current_bindings
                    if item.enabled
                )
                != borrow_admission.identities
            ):
                return
            # A negative proof authorizes one new reservation, not later turns or a batch.
            self._borrow_grants.pop(lane, None)
            self._primary_rounds.pop(lane, None)
        requested_gpus = candidate["requested_gpus"]
        requested_cpus = candidate["requested_cpus"]
        if lane == "gpu":
            try:
                reserved_gpu_ids = reservation_snapshot(self.runtime.root).reserved_gpu_ids
            except (OSError, RuntimeError, ValueError):
                self._record_service_failure(service_key)
                return
            selected_gpu_ids = [gpu_id for gpu_id in allowed_gpu_ids if gpu_id not in reserved_gpu_ids][:requested_gpus]
            if requested_gpus < 1 or len(selected_gpu_ids) < requested_gpus:
                return
            cpu_slots = 0
        else:
            if requested_cpus < 1 or batch_budget.remaining_cpu_slots < requested_cpus:
                return
            selected_gpu_ids = []
            cpu_slots = requested_cpus

        request_id = uuid.uuid4().hex
        reservation: dict[str, Any] | None = None
        try:
            reservation = self._reserve_claim_offer(
                binding,
                identity,
                candidate,
                selected_gpu_ids,
                cpu_slots,
                request_id,
            )
            batch_budget.remaining_cpu_slots -= cpu_slots
            reservation_record = reservation["reservation"]
            offer = {
                "offer_id": reservation_record["reservation_id"],
                "acquisition_id": reservation_record["acquisition_id"],
                "reservation_id": reservation_record["reservation_id"],
                "executor_epoch": identity.executor_epoch,
                "request_id": request_id,
                "project_id": binding.project_id,
                "shared_root": identity.shared_root,
                "registration_generation": binding.registration_generation,
                "task_id": candidate["task_id"],
                "attempt_id": candidate["attempt_id"],
                "attempt_number": candidate["attempt_number"],
                "fencing_token": candidate["fencing_token"],
                "lane": lane,
                "gpu_ids": selected_gpu_ids,
                "cpu_slots": cpu_slots,
                "group_name": candidate["group_name"],
                "group_dispatch_epoch": candidate["group_dispatch_epoch"],
                "group_worker_set_epoch": candidate["group_worker_set_epoch"],
                "worker_state_epoch": candidate["worker_state_epoch"],
                "worker_scheduling_role": candidate["worker_scheduling_role"],
                "gpu_limit_gpus": candidate["gpu_limit_gpus"],
                "admitted_as_borrow": reservation_record.get("admission", {}).get("admitted_as_borrow", False),
            }
            if defer_request:
                pending = _PendingClaimOffer(
                    identity,
                    binding,
                    _json_copy(candidate),
                    _json_copy(cursor),
                    _json_copy(evidence),
                    offer,
                    service_key,
                )
                self._pending_claim_offers[request_id] = pending
                self._observations.pop(observation_key, None)
                self._offer_pending_claim(pending)
                return
            request = self.executor.prepare_scheduler_claim(
                binding,
                registry_revision,
                lane=lane,
                admission_role=admission_role,
                candidate=candidate,
                offer=offer,
                cursor=cursor,
                request_id=request_id,
                source_revisions=evidence["source_revisions"],
            )
        except (OSError, RuntimeError, ValueError):
            if reservation is not None:
                try:
                    unresolved_ids = {item.request_id for item in self.executor.unresolved_requests()}
                    if request_id not in unresolved_ids:
                        offer_identity = ReservationIdentity.from_record(reservation["reservation"])
                        release_executor_offer(self.runtime.root, offer_identity)
                        if classify_executor_offer(self.runtime.root, offer_identity) == "matching_released":
                            batch_budget.remaining_cpu_slots += cpu_slots
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
                    pass
            self._record_service_failure(service_key)
            return

        self._observations.pop(observation_key, None)
        self._start_claim_request(request, service_key)

    def _recover_unprepared_claim_offers(self) -> None:
        """Recover the local allocation/publication gap without touching a Project."""
        identities = sorted(
            (
                identity
                for record in reservation_snapshot(self.runtime.root).provisional
                if (identity := ReservationIdentity.from_record(record)).executor_request_id is not None
                and identity.executor_request_id not in self._pending_claim_offers
            ),
            key=lambda identity: identity.reservation_id,
        )
        start = (
            bisect_right([identity.reservation_id for identity in identities], self._offer_recovery_cursor)
            if self._offer_recovery_cursor is not None
            else 0
        )
        for identity in (identities[start:] + identities[:start])[:4]:
            self._offer_recovery_cursor = identity.reservation_id
            self.executor.release_unprepared_offer(identity)

    def _advance_pending_claim_offers(
        self,
        bindings: Mapping[_ValidationIdentity, ProjectBinding],
        lane: str,
        role: str,
    ) -> set[str]:
        current_revision, current_bindings = self.runtime.load_registry_snapshot()
        registered = {binding.project_id: binding for binding in current_bindings if binding.enabled}
        pending_projects: set[str] = set()
        for request_id, pending in tuple(self._pending_claim_offers.items()):
            if pending.candidate["lane"] != lane or pending.candidate["admission_role"] != role:
                continue
            identity = pending.identity
            binding = registered.get(identity.project_id)
            if (
                binding is None
                or self._binding_identity(binding, current_revision, self._observed_executor_epoch) != identity
            ):
                if self.executor.release_unprepared_offer(self._pending_offer_identity(pending)):
                    self._pending_claim_offers.pop(request_id, None)
                continue
            pending_projects.add(identity.project_id)
            if identity in bindings and self._service_retry_is_due(pending.service_key):
                self._offer_pending_claim(pending)
        return pending_projects

    def _offer_pending_claim(self, pending: _PendingClaimOffer) -> None:
        self._offer_new_request(
            pending.identity,
            "scheduler_claim",
            partial(self._prepare_pending_claim, pending),
            pending.candidate["lane"],
            pending.candidate["admission_role"],
            pending.candidate["attempt_id"],
        )

    @staticmethod
    def _pending_offer_identity(pending: _PendingClaimOffer) -> ReservationIdentity:
        offer = pending.offer
        return ReservationIdentity.from_record(
            {
                "reservation_id": offer["reservation_id"],
                "acquisition_id": offer["acquisition_id"],
                "project_id": offer["project_id"],
                "shared_root": offer["shared_root"],
                "task_id": offer["task_id"],
                "attempt_id": offer["attempt_id"],
                "fencing_token": offer["fencing_token"],
                "gpu_ids": list(offer["gpu_ids"]) if offer["lane"] == "gpu" else None,
                "cpu_slots": offer["cpu_slots"] if offer["lane"] == "cpu" else None,
                "executor_owner": {
                    "executor_epoch": offer["executor_epoch"],
                    "request_id": offer["request_id"],
                    "registration_generation": offer["registration_generation"],
                },
            }
        )

    def _prepare_pending_claim(self, pending: _PendingClaimOffer) -> None:
        request_id = pending.offer["request_id"]
        if self._pending_claim_offers.get(request_id) is not pending:
            return
        try:
            current_revision, bindings = self.runtime.load_registry_snapshot()
            if (
                current_revision != pending.identity.registry_revision
                or pending.binding not in bindings
                or pending.identity.executor_epoch != self._observed_executor_epoch
            ):
                if self.executor.release_unprepared_offer(self._pending_offer_identity(pending)):
                    self._pending_claim_offers.pop(request_id, None)
                return
            request = self.executor.prepare_scheduler_claim(
                pending.binding,
                current_revision,
                lane=pending.candidate["lane"],
                admission_role=pending.candidate["admission_role"],
                candidate=pending.candidate,
                offer=pending.offer,
                cursor=pending.cursor,
                request_id=request_id,
                source_revisions=pending.evidence["source_revisions"],
            )
        except (OSError, RuntimeError, ValueError):
            # Publication can fail after the exact request is durable. In that
            # case leave its offer to ordinary typed claim reconciliation.
            if any(request.request_id == request_id for request in self.executor.unresolved_requests()):
                self._pending_claim_offers.pop(request_id, None)
            self._record_service_failure(pending.service_key)
            return
        self._pending_claim_offers.pop(request_id, None)
        self._start_claim_request(request, pending.service_key)

    @staticmethod
    def _claim_service_key(
        identity: _ValidationIdentity,
        lane: str,
        admission_role: str,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "scheduler_claim", lane, admission_role)

    def _start_claim_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
        *,
        reconcile_only: bool = False,
    ) -> None:
        try:
            process = self.executor.start(
                request.request_id,
                reconcile_fenced_claim=(reconcile_only or request.executor_epoch != self._observed_executor_epoch),
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    @staticmethod
    def _offer_identity(request: ProjectIORequest) -> ReservationIdentity:
        offer = request.parameters["offer"]
        reservation = {
            "reservation_id": offer["reservation_id"],
            "acquisition_id": offer["acquisition_id"],
            "project_id": offer["project_id"],
            "shared_root": offer["shared_root"],
            "task_id": offer["task_id"],
            "attempt_id": offer["attempt_id"],
            "fencing_token": offer["fencing_token"],
            "gpu_ids": list(offer["gpu_ids"]) if offer["lane"] == "gpu" else None,
            "cpu_slots": offer["cpu_slots"] if offer["lane"] == "cpu" else None,
            "executor_owner": {
                "executor_epoch": request.executor_epoch,
                "request_id": request.request_id,
                "registration_generation": request.registration_generation,
            },
        }
        return ReservationIdentity.from_record(reservation)

    def _reserve_claim_offer(
        self,
        binding: ProjectBinding,
        identity: _ValidationIdentity,
        candidate: Mapping[str, Any],
        gpu_ids: list[int],
        cpu_slots: int,
        request_id: str,
    ) -> dict[str, Any]:
        owner = {
            "executor_epoch": identity.executor_epoch,
            "executor_request_id": request_id,
            "registration_generation": binding.registration_generation,
        }
        if candidate["lane"] == "cpu":
            return reserve_cpu(
                self.runtime.root,
                candidate["task_id"],
                cpu_slots,
                attempt_id=candidate["attempt_id"],
                fencing_token=candidate["fencing_token"],
                project_id=binding.project_id,
                shared_root=identity.shared_root,
                machine_name=binding.machine_name,
                group_name=candidate["group_name"],
                admitted_as_borrow=candidate["admission_role"] == "borrow",
                worker_scheduling_role=candidate["worker_scheduling_role"],
                group_worker_set_epoch=candidate["group_worker_set_epoch"],
                worker_state_epoch=candidate["worker_state_epoch"],
                **owner,
            )
        if candidate["group_name"] is not None:
            return reserve_admitted(
                self.runtime.root,
                candidate["task_id"],
                gpu_ids,
                project_id=binding.project_id,
                group_name=candidate["group_name"],
                machine_name=binding.machine_name,
                gpu_limit_gpus=candidate["gpu_limit_gpus"],
                worker_scheduling_role=candidate["worker_scheduling_role"],
                group_worker_set_epoch=candidate["group_worker_set_epoch"],
                worker_state_epoch=candidate["worker_state_epoch"],
                attempt_id=candidate["attempt_id"],
                fencing_token=candidate["fencing_token"],
                shared_root=identity.shared_root,
                admitted_as_borrow=candidate["admission_role"] == "borrow",
                **owner,
            )
        return reserve(
            self.runtime.root,
            candidate["task_id"],
            gpu_ids,
            attempt_id=candidate["attempt_id"],
            fencing_token=candidate["fencing_token"],
            project_id=binding.project_id,
            shared_root=identity.shared_root,
            machine_name=binding.machine_name,
            **owner,
        )

    @_admission_operation
    def advance_scheduler_reservation_reconciliations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        reservations: Sequence[ReservationIdentity],
    ) -> tuple[ReservationIdentity, ...]:
        """Classify shared ownership, then apply only exact local reservation effects."""
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if not isinstance(reservations, Sequence) or isinstance(reservations, str | bytes):
            raise ValueError("reservations must be a sequence.")
        if any(not isinstance(item, ReservationIdentity) for item in reservations):
            raise ValueError("reservations must contain ReservationIdentity values.")
        reservation_by_id = {item.reservation_id: item for item in reservations}
        if len(reservation_by_id) != len(reservations):
            raise ValueError("reservations must have unique reservation IDs.")
        current_reservations = set(reservations)
        self._trusted_reservations = {item: None for item in self._trusted_reservations if item in current_reservations}

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            return tuple(sorted(self._trusted_reservations, key=lambda item: item.reservation_id))
        self._observed_executor_epoch = executor_epoch

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        bindings_by_project: dict[str, tuple[_ValidationIdentity, ProjectBinding]] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or identity not in self._validated:
                continue
            current_bindings[identity] = binding
            bindings_by_project[binding.project_id] = (identity, binding)

        unresolved = self.executor.unresolved_requests()
        retained_reconcile_identities = set(current_bindings) | {
            self._request_identity(request)
            for request in unresolved
            if request.operation_kind == "scheduler_reservation_reconcile"
        }
        self._reservation_reconcile_backoff = {
            key: state
            for key, state in self._reservation_reconcile_backoff.items()
            if key[0] in retained_reconcile_identities
        }
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._started_requests.intersection_update(unresolved_by_id)
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        processed_reservations: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "scheduler_reservation_reconcile":
                continue
            requested = self._reservation_identity_from_request(request)
            processed_reservations.add(requested.reservation_id)
            identity = self._request_identity(request)
            binding = current_bindings.get(identity)
            current = reservation_by_id.get(requested.reservation_id)
            service_key = self._reservation_reconcile_service_key(identity, requested.reservation_id)
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if binding is not None and current == requested:
                    if was_started or service_key not in self._reservation_reconcile_backoff:
                        self._record_reservation_reconcile_failure(service_key)
                    if self._reservation_reconcile_retry_is_due(service_key):
                        self._start_reservation_reconcile_request(request, service_key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("reservation reconciliation result changed request identity.")
            applied = True
            trusted: ReservationIdentity | None = None
            if result.status == "completed" and binding is not None:
                outcome = result.evidence.get("outcome")
                try:
                    local_state = classify_exact_reservation(self.runtime.root, requested)
                    if outcome == "retained":
                        if local_state == "matching_active" and current == requested:
                            trusted = requested
                    elif outcome == "retag":
                        token = result.evidence.get("target_fencing_token")
                        attempt_id = result.evidence.get("target_attempt_id")
                        if type(token) is not int or not isinstance(attempt_id, str):
                            applied = False
                        else:
                            target = replace(requested, attempt_id=attempt_id, fencing_token=token)
                            target_state = classify_exact_reservation(self.runtime.root, target)
                            if target_state != "matching_active" and local_state == "matching_active":
                                retagged = (
                                    retag_cpu_if_matches(self.runtime.root, requested, attempt_id, token)
                                    if requested.cpu_slots is not None
                                    else retag_if_matches(self.runtime.root, requested, attempt_id, token)
                                )
                                if retagged:
                                    target_state = classify_exact_reservation(self.runtime.root, target)
                            if target_state == "matching_active":
                                trusted = target
                    elif outcome == "release":
                        reason = result.evidence.get("reason")
                        if not isinstance(reason, str):
                            applied = False
                        elif local_state in {"matching_active", "matching_release_pair"}:
                            released = release_if_matches(self.runtime.root, requested, reason)
                            if not released and classify_exact_reservation(self.runtime.root, requested) not in {
                                "matching_released",
                                "absent",
                                "conflict",
                            }:
                                applied = False
                    elif outcome != "isolated":
                        applied = False
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    applied = False
            if not applied:
                self._record_reservation_reconcile_failure(service_key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed" and consumed.evidence.get("outcome") != "isolated":
                self._reservation_reconcile_backoff.pop(service_key, None)
            else:
                self._record_reservation_reconcile_failure(service_key)
            self._trusted_reservations.pop(requested, None)
            self._reservation_reconcile_cursors[(request.project_id, request.registration_generation)] = (
                requested.reservation_id
            )
            if trusted is not None:
                self._trust_reservation(trusted)

        can_start = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )
        current_cursor_owners = {
            (identity.project_id, identity.registration_generation) for identity in current_bindings
        }
        self._reservation_reconcile_cursors = {
            owner: cursor
            for owner, cursor in self._reservation_reconcile_cursors.items()
            if owner in current_cursor_owners
        }
        reservations_by_project: dict[str, list[ReservationIdentity]] = {}
        for reservation in reservations:
            if reservation.project_id is not None:
                reservations_by_project.setdefault(reservation.project_id, []).append(reservation)
        for owned in reservations_by_project.values():
            owned.sort(key=lambda item: item.reservation_id)

        ordered_owners = sorted(
            (identity.project_id, identity.registration_generation) for identity in current_bindings
        )
        if ordered_owners:
            start = 0
            if self._reservation_reconcile_binding_cursor is not None:
                start = bisect_right(ordered_owners, self._reservation_reconcile_binding_cursor)
                if start == len(ordered_owners):
                    start = 0
            selected_owners = (ordered_owners[start:] + ordered_owners[:start])[:_MAX_RETAINED_SCHEDULER_INTENTS]
            self._reservation_reconcile_binding_cursor = selected_owners[-1]
        else:
            selected_owners = []
            self._reservation_reconcile_binding_cursor = None

        if can_start:
            for owner in selected_owners:
                current = bindings_by_project.get(owner[0])
                if current is None or current[0].registration_generation != owner[1]:
                    continue
                identity, binding = current
                owned = reservations_by_project.get(identity.project_id, [])
                if not owned:
                    continue
                cursor = self._reservation_reconcile_cursors.get(owner)
                reservation_ids = [item.reservation_id for item in owned]
                candidate_index = bisect_right(reservation_ids, cursor) if cursor is not None else 0
                if candidate_index == len(owned):
                    candidate_index = 0
                reservation = owned[candidate_index]
                self._reservation_reconcile_cursors[owner] = reservation.reservation_id
                if reservation in self._trusted_reservations or reservation.reservation_id in processed_reservations:
                    continue
                service_key = self._reservation_reconcile_service_key(identity, reservation.reservation_id)
                if not self._reservation_reconcile_retry_is_due(service_key):
                    continue
                self._offer_new_request(
                    identity,
                    "scheduler_reservation_reconcile",
                    partial(
                        self._prepare_scheduler_reservation_reconcile_request,
                        binding,
                        registry_revision,
                        reservation,
                        service_key,
                    ),
                    reservation.attempt_id or reservation.reservation_id,
                )
        self._trim_trusted_reservations()
        return tuple(sorted(self._trusted_reservations, key=lambda item: item.reservation_id))

    @_admission_operation
    def advance_scheduler_launch_authorizations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        reservations: Sequence[ReservationIdentity],
    ) -> dict[str, Mapping[str, Any]]:
        """Advance exact active-reservation launch gates without Project I/O."""
        if not isinstance(bindings, Sequence) or isinstance(bindings, str | bytes):
            raise ValueError("bindings must be a sequence.")
        if any(not isinstance(binding, ProjectBinding) for binding in bindings):
            raise ValueError("bindings must contain ProjectBinding values.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if not isinstance(reservations, Sequence) or isinstance(reservations, str | bytes):
            raise ValueError("reservations must be a sequence.")
        if any(not isinstance(item, ReservationIdentity) for item in reservations):
            raise ValueError("reservations must contain ReservationIdentity values.")
        reservation_by_id = {item.reservation_id: item for item in reservations}
        if len(reservation_by_id) != len(reservations):
            raise ValueError("reservations must have unique reservation IDs.")
        self._authorized_reservations.intersection_update(reservations)

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status["executor_epoch"]
        if executor_status["envelope"] == "unknown" or not isinstance(executor_epoch, str):
            self._observed_executor_epoch = None
            return {}
        self._observed_executor_epoch = executor_epoch

        current_bindings: dict[_ValidationIdentity, ProjectBinding] = {}
        bindings_by_project: dict[str, tuple[_ValidationIdentity, ProjectBinding]] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if (
                identity is None
                or binding.enabled is not True
                or identity not in self._validated
                or identity in current_bindings
            ):
                continue
            current_bindings[identity] = binding
            bindings_by_project[binding.project_id] = (identity, binding)

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._started_requests.intersection_update(unresolved_by_id)
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        completions: dict[str, Mapping[str, Any]] = {}
        processed_reservations: set[str] = set()

        for request in unresolved:
            if request.operation_kind != "scheduler_launch_authorize":
                continue
            reservation_id = request.provisional_offer_id
            if not isinstance(reservation_id, str):
                continue
            processed_reservations.add(reservation_id)
            identity = self._request_identity(request)
            binding = current_bindings.get(identity)
            reservation = reservation_by_id.get(reservation_id)
            if (
                binding is None
                or reservation is None
                or not self._launch_request_matches_reservation(request, reservation)
            ):
                continue
            service_key = self._launch_service_key(identity)
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if was_started or service_key not in self._service_backoff:
                    self._record_service_failure(service_key)
                if self._service_retry_is_due(service_key):
                    self._start_launch_request(request, service_key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("launch result does not repeat its exact request identity.")
            if result.status == "outcome_unknown":
                try:
                    reset = self.executor.reset_ambiguous_scheduler_launch_authorize_for_retry(
                        request.request_id,
                        request,
                    )
                except (OSError, RuntimeError, ValueError):
                    reset = False
                if reset:
                    self._record_service_failure(service_key)
                    if self._service_retry_is_due(service_key):
                        self._start_launch_request(request, service_key)
                continue
            if result.status == "completed" and result.evidence.get("outcome") == "authorized":
                try:
                    if classify_exact_reservation(self.runtime.root, reservation) != "matching_active":
                        continue
                except (OSError, RuntimeError, ValueError):
                    continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                completions[reservation_id] = _json_copy(consumed.evidence)
                if consumed.evidence.get("outcome") == "authorized":
                    self._authorized_reservations.add(reservation)
                self._service_backoff.pop(service_key, None)
            else:
                self._record_service_failure(service_key)

        can_start = (
            executor_status["envelope"] != "unknown"
            and executor_status["overdue_worker_count"] <= executor_status["supported_hang_limit"]
        )
        if not can_start:
            return completions
        for reservation in reservations:
            if reservation.reservation_id in processed_reservations:
                continue
            if reservation in self._authorized_reservations:
                continue
            current = bindings_by_project.get(reservation.project_id or "")
            if current is None:
                continue
            identity, binding = current
            claim_identity = self._claim_identity_for_reservation(reservation, identity)
            if claim_identity is None:
                continue
            service_key = self._launch_service_key(identity)
            if self._reservation_has_registered_process(binding, reservation):
                continue
            if not self._service_retry_is_due(service_key):
                continue
            try:
                if classify_exact_reservation(self.runtime.root, reservation) != "matching_active":
                    continue
            except (OSError, RuntimeError, ValueError):
                self._record_service_failure(service_key)
                continue
            self._offer_new_request(
                identity,
                "scheduler_launch_authorize",
                partial(
                    self._prepare_scheduler_launch_authorization_request,
                    binding,
                    registry_revision,
                    claim_identity,
                    reservation,
                    service_key,
                ),
                reservation.attempt_id or reservation.reservation_id,
            )
        return completions

    def _reservation_has_registered_process(self, binding: ProjectBinding, reservation: ReservationIdentity) -> bool:
        """Do not reoffer launch work for an immutable matching registration.

        This suppresses duplicate launch requests only; it grants no authority and
        leaves renewal, completion, and reservation reconciliation to their lanes.
        """
        if reservation.attempt_id is None:
            return False
        paths = machine_project_paths(self.runtime.root, binding.project_id)
        path = paths["registrations"] / f"{reservation.attempt_id}.json"
        try:
            if not validate_evidence_path(path, paths["root"]):
                return False
            record = read_json_limited(path, max_bytes=65_536, record_type="registration").get("process_registration")
            return (
                isinstance(record, dict)
                and type(record.get("protocol_version")) is int
                and record["protocol_version"] == 1
                and record.get("task_id") == reservation.task_id
                and record.get("attempt_id") == reservation.attempt_id
                and type(record.get("fencing_token")) is int
                and record["fencing_token"] == reservation.fencing_token
                and record.get("machine_name") == binding.machine_name
                and registration_reservation(record, paths["root"]) == reservation.reservation_id
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return False

    def _prepare_scheduler_launch_authorization_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        claim_identity: ReservationIdentity,
        reservation: ReservationIdentity,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            if classify_exact_reservation(self.runtime.root, reservation) != "matching_active":
                return
            request = self.executor.prepare_scheduler_launch_authorize(
                binding,
                registry_revision,
                claim_identity=claim_identity,
                reservation_identity=reservation,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_launch_request(request, service_key)

    def retry_scheduler_launch_authorization(self, reservation: ReservationIdentity) -> None:
        """Allow exact authorization replay after a proven-local launch failure."""
        if not isinstance(reservation, ReservationIdentity):
            raise ValueError("launch authorization retry requires a ReservationIdentity.")
        self._authorized_reservations.discard(reservation)

    @_admission_operation
    def advance_scheduler_due_offers(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one fairly selected due-offer transaction."""
        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is not None and binding.enabled and identity in self._validated:
                current[identity] = binding
        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "scheduler_due_offer":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                if self.executor.resolve_stale_scheduler_due_offer(request.request_id, request):
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            processed.add(binding.project_id)
            key = (identity, "scheduler_due_offer", "deadline", "advance")
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if was_started or key not in self._service_backoff:
                    self._record_service_failure(key)
                if self._service_retry_is_due(key):
                    self._start_scheduler_due_offer_request(request, key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("due-offer result changed request identity.")
            if result.status == "outcome_unknown":
                try:
                    reset = self.executor.reset_ambiguous_scheduler_due_offer_for_retry(
                        request.request_id,
                        request,
                    )
                except (OSError, RuntimeError, ValueError):
                    reset = False
                if reset:
                    self._record_service_failure(key)
                    if self._service_retry_is_due(key):
                        self._start_scheduler_due_offer_request(request, key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                completions[binding.project_id] = dict(consumed.evidence)
                self._service_backoff.pop(key, None)
            else:
                self._record_service_failure(key)

        can_start = executor_status.get("overdue_worker_count", 0) <= executor_status.get("supported_hang_limit", 0)
        owners = sorted((identity.project_id, identity.registration_generation) for identity in current)
        if not can_start or not owners:
            if not owners:
                self._due_offer_binding_cursor = None
            return completions
        start = 0
        if self._due_offer_binding_cursor is not None:
            start = bisect_right(owners, self._due_offer_binding_cursor)
            if start == len(owners):
                start = 0
        selected = (owners[start:] + owners[:start])[:_MAX_RETAINED_SCHEDULER_INTENTS]
        by_owner = {
            (identity.project_id, identity.registration_generation): (identity, binding)
            for identity, binding in current.items()
        }
        for owner in selected:
            self._due_offer_binding_cursor = owner
            identity, binding = by_owner[owner]
            if binding.project_id in processed:
                continue
            key = (identity, "scheduler_due_offer", "deadline", "advance")
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                "scheduler_due_offer",
                partial(
                    self._prepare_scheduler_due_offer_request,
                    binding,
                    registry_revision,
                    key,
                ),
                "deadline",
            )
            break
        return completions

    def _prepare_scheduler_due_offer_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if self.scheduler_is_quiescent(binding, registry_revision):
            return
        try:
            request = self.executor.prepare_scheduler_due_offer(binding, registry_revision)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_scheduler_due_offer_request(request, service_key)

    def _start_scheduler_due_offer_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    @_admission_operation
    def advance_scheduler_ready_index_builds(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        observed_only: bool = False,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one fairly selected ready-index build slice."""
        if type(observed_only) is not bool:
            raise ValueError("observed_only must be a boolean.")
        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        eligible: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is not None and binding.enabled and identity in self._validated:
                eligible[identity] = binding
        self._ready_index_settled = {identity: None for identity in self._ready_index_settled if identity in eligible}
        self._ready_index_needed = {identity: None for identity in self._ready_index_needed if identity in eligible}
        current = {
            identity: binding
            for identity, binding in eligible.items()
            if not observed_only or identity in self._ready_index_needed
        }

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "scheduler_ready_index_build":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                if self.executor.resolve_stale_scheduler_ready_index_build(request.request_id, request):
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            processed.add(binding.project_id)
            key = (identity, "scheduler_ready_index_build", "ready_index", "advance")
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if was_started or key not in self._service_backoff:
                    self._record_service_failure(key)
                if self._service_retry_is_due(key):
                    self._start_scheduler_ready_index_request(request, key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("ready-index result changed request identity.")
            if result.status == "outcome_unknown":
                try:
                    reset = self.executor.reset_ambiguous_scheduler_ready_index_build_for_retry(
                        request.request_id,
                        request,
                    )
                except (OSError, RuntimeError, ValueError):
                    reset = False
                if reset:
                    self._record_service_failure(key)
                    if self._service_retry_is_due(key):
                        self._start_scheduler_ready_index_request(request, key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                completions[binding.project_id] = dict(consumed.evidence)
                if consumed.evidence.get("state") in {"active", "degraded"}:
                    self._ready_index_needed.pop(identity, None)
                    self._ready_index_settled.pop(identity, None)
                    self._ready_index_settled[identity] = None
                    while len(self._ready_index_settled) > _MAX_RETAINED_SCHEDULER_INTENTS:
                        del self._ready_index_settled[next(iter(self._ready_index_settled))]
                self._service_backoff.pop(key, None)
            else:
                self._record_service_failure(key)

        can_start = executor_status.get("overdue_worker_count", 0) <= executor_status.get("supported_hang_limit", 0)
        owners = sorted((identity.project_id, identity.registration_generation) for identity in current)
        if not can_start or not owners:
            if not owners:
                self._ready_index_binding_cursor = None
            return completions
        start = 0
        if self._ready_index_binding_cursor is not None:
            start = bisect_right(owners, self._ready_index_binding_cursor)
            if start == len(owners):
                start = 0
        selected = (owners[start:] + owners[:start])[:_MAX_RETAINED_SCHEDULER_INTENTS]
        by_owner = {
            (identity.project_id, identity.registration_generation): (identity, binding)
            for identity, binding in current.items()
        }
        for owner in selected:
            self._ready_index_binding_cursor = owner
            identity, binding = by_owner[owner]
            if identity in self._ready_index_settled or binding.project_id in processed:
                continue
            key = (identity, "scheduler_ready_index_build", "ready_index", "advance")
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                "scheduler_ready_index_build",
                partial(
                    self._prepare_scheduler_ready_index_request,
                    binding,
                    registry_revision,
                    key,
                ),
                "ready_index",
            )
            break
        return completions

    def _prepare_scheduler_ready_index_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_scheduler_ready_index_build(binding, registry_revision)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_scheduler_ready_index_request(request, service_key)

    def _start_scheduler_ready_index_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    @_admission_operation
    def advance_maintenance_descriptor_work(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one descriptor-maintenance slice for a fairly selected binding."""
        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is not None and binding.enabled and identity in self._validated:
                current[identity] = binding
        self._maintenance_descriptor_settled = {
            identity: turn
            for identity, turn in self._maintenance_descriptor_settled.items()
            if identity in current
            and (
                self.runtime.working_set.is_current_turn(turn)
                or self.runtime.working_set.is_turn_observation_pending(turn)
            )
        }
        self._maintenance_descriptor_dirty.intersection_update(current)

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        self._maintenance_descriptor_turns = {
            request_id: turn
            for request_id, turn in self._maintenance_descriptor_turns.items()
            if request_id in unresolved_by_id
        }
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "maintenance_descriptor_advance":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                if self.executor.resolve_stale_maintenance_descriptor(request.request_id, request):
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                    self._maintenance_descriptor_dirty.discard(identity)
                    self._maintenance_descriptor_turns.pop(request.request_id, None)
                continue
            processed.add(binding.project_id)
            key = self._maintenance_descriptor_service_key(identity)
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                was_started = request.request_id in self._started_requests
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue

            result = self.executor.load_result(request.request_id)
            if result is None:
                if was_started or key not in self._service_backoff:
                    self._record_service_failure(key)
                if self._service_retry_is_due(key):
                    self._start_maintenance_descriptor_request(request, key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("descriptor maintenance result changed request identity.")
            if result.status == "outcome_unknown":
                if self.executor.reset_ambiguous_maintenance_descriptor_for_retry(request.request_id, request):
                    self._record_service_failure(key)
                    if self._service_retry_is_due(key):
                        self._start_maintenance_descriptor_request(request, key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            turn = self._maintenance_descriptor_turns.pop(request.request_id, None)
            if consumed.status == "completed":
                evidence = dict(consumed.evidence)
                completions[binding.project_id] = evidence
                state = evidence.get("maintenance_state")
                observed_newer_activation = identity in self._maintenance_descriptor_dirty
                self._maintenance_descriptor_dirty.discard(identity)
                if turn is not None:
                    self.runtime.working_set.acknowledge(
                        turn,
                        quiescent=(
                            not observed_newer_activation
                            and (
                                state in {"idle", "completed", "intervention"}
                                and not evidence.get("more")
                                or state == "waiting"
                                and not evidence.get("idle_blocking")
                            )
                        ),
                    )
                if (
                    not observed_newer_activation
                    and state in {"idle", "completed", "intervention"}
                    and not evidence.get("more")
                    and turn is not None
                    and (
                        self.runtime.working_set.is_current_turn(turn)
                        or self.runtime.working_set.is_turn_observation_pending(turn)
                    )
                ):
                    self._maintenance_descriptor_settled.pop(identity, None)
                    self._maintenance_descriptor_settled[identity] = turn
                    while len(self._maintenance_descriptor_settled) > _MAX_RETAINED_SCHEDULER_INTENTS:
                        del self._maintenance_descriptor_settled[next(iter(self._maintenance_descriptor_settled))]
                    self._service_backoff.pop(key, None)
                    self.runtime.maintenance_retry_deadlines.pop(binding.project_id, None)
                    self.runtime.maintenance_retry_idle_blocked_projects.discard(binding.project_id)
                elif state == "waiting" and isinstance(evidence.get("next_due_at"), str):
                    due_at = datetime.fromisoformat(evidence["next_due_at"].replace("Z", "+00:00"))
                    delay = max(0.0, (due_at - datetime.now(timezone.utc)).total_seconds())
                    retry_at = self._monotonic() + delay
                    self._service_backoff[key] = _FailureBackoff(0, retry_at)
                    self.runtime.maintenance_retry_deadlines[binding.project_id] = (binding, retry_at)
                    if evidence.get("idle_blocking"):
                        self.runtime.maintenance_retry_idle_blocked_projects.add(binding.project_id)
                    else:
                        self.runtime.maintenance_retry_idle_blocked_projects.discard(binding.project_id)
                else:
                    self._service_backoff.pop(key, None)
                    self.runtime.maintenance_retry_deadlines.pop(binding.project_id, None)
                    self.runtime.maintenance_retry_idle_blocked_projects.discard(binding.project_id)
            else:
                self._record_service_failure(key)

        can_start = executor_status.get("envelope") != "unknown" and executor_status.get(
            "overdue_worker_count", 0
        ) <= executor_status.get("supported_hang_limit", 0)
        owners = sorted((identity.project_id, identity.registration_generation) for identity in current)
        if not can_start or not owners:
            if not owners:
                self._maintenance_descriptor_binding_cursor = None
            return completions
        start = 0
        if self._maintenance_descriptor_binding_cursor is not None:
            start = bisect_right(owners, self._maintenance_descriptor_binding_cursor)
            if start == len(owners):
                start = 0
        selected = (owners[start:] + owners[:start])[:_MAX_RETAINED_SCHEDULER_INTENTS]
        by_owner = {
            (identity.project_id, identity.registration_generation): (identity, binding)
            for identity, binding in current.items()
        }
        for owner in selected:
            self._maintenance_descriptor_binding_cursor = owner
            identity, binding = by_owner[owner]
            if identity in self._maintenance_descriptor_settled or binding.project_id in processed:
                continue
            key = self._maintenance_descriptor_service_key(identity)
            if not self._service_retry_is_due(key):
                continue
            self._offer_new_request(
                identity,
                "maintenance_descriptor_advance",
                partial(
                    self._prepare_maintenance_descriptor_request,
                    binding,
                    registry_revision,
                    key,
                ),
                "descriptor",
            )
        return completions

    def _prepare_maintenance_descriptor_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_maintenance_descriptor_advance(binding, registry_revision)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._maintenance_descriptor_turns[request.request_id] = self.runtime.working_set.begin_turn(
            binding, "maintenance"
        )
        self._start_maintenance_descriptor_request(request, service_key)

    def _start_maintenance_descriptor_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    @_admission_operation
    def advance_maintenance_event_flushes(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one digest-bound local diagnostic flush per current binding."""
        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is not None and binding.enabled and identity in self._validated:
                current[identity] = binding
        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "maintenance_flush_event":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                if self.executor.resolve_stale_maintenance_event(request.request_id, request):
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            processed.add(binding.project_id)
            key = (identity, "maintenance_flush_event", "events", "flush")
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if self._service_retry_is_due(key):
                    self._start_maintenance_event_request(request, key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("maintenance event result changed request identity.")
            if result.status == "outcome_unknown":
                if self.executor.reset_ambiguous_maintenance_event_for_retry(request.request_id, request):
                    self._record_service_failure(key)
                    if self._service_retry_is_due(key):
                        self._start_maintenance_event_request(request, key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                evidence = dict(consumed.evidence)
                completions[binding.project_id] = evidence
                if evidence.get("outcome") == "flushed":
                    if self.executor.retire_maintenance_event_source(request):
                        self._service_backoff.pop(key, None)
                    else:
                        self._record_service_failure(key)
                elif evidence.get("reason") in {"invalid_event", "identity_mismatch"}:
                    if self.executor.retire_maintenance_event_source(request):
                        self._service_backoff.pop(key, None)
                    else:
                        self._record_service_failure(key)
                else:
                    self._record_service_failure(key)
            else:
                self._record_service_failure(key)

        can_start = executor_status.get("envelope") != "unknown" and executor_status.get(
            "overdue_worker_count", 0
        ) <= executor_status.get("supported_hang_limit", 0)
        if not can_start:
            return completions
        for identity, binding in current.items():
            if binding.project_id in processed:
                continue
            key = (identity, "maintenance_flush_event", "events", "flush")
            if not self._service_retry_is_due(key):
                continue
            candidate = self._next_local_event(binding.project_id)
            if candidate is None:
                continue
            bucket, filename, event_id, digest = candidate
            self._offer_new_request(
                identity,
                "maintenance_flush_event",
                partial(
                    self._prepare_maintenance_event_flush_request,
                    binding,
                    registry_revision,
                    bucket,
                    filename,
                    event_id,
                    digest,
                    key,
                ),
                bucket,
                filename,
                event_id,
                digest,
            )
        return completions

    def _prepare_maintenance_event_flush_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        bucket: str,
        filename: str,
        event_id: str,
        digest: str,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_maintenance_flush_event(
                binding,
                registry_revision,
                bucket=bucket,
                filename=filename,
                event_id=event_id,
                sha256=digest,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_maintenance_event_request(request, service_key)

    @_admission_operation
    def advance_machine_snapshot_publications(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        instance_id: str,
        pid: int | None,
        visible_gpu_ids: Sequence[int],
        reservations: Sequence[Mapping[str, Any]],
        heartbeat_interval_seconds: int | float,
        started_at: str,
        gpu_policy: object,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one isolated advisory snapshot publication per validated binding."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        visible = self._snapshot_gpu_ids(visible_gpu_ids, "visible_gpu_ids")
        if pid is not None and (type(pid) is not int or pid <= 0):
            raise ValueError("pid must be a positive integer or null.")
        if (
            type(heartbeat_interval_seconds) not in {int, float}
            or (type(heartbeat_interval_seconds) is float and not math.isfinite(heartbeat_interval_seconds))
            or heartbeat_interval_seconds <= 0
            or heartbeat_interval_seconds > 86_400
        ):
            raise ValueError("heartbeat_interval_seconds is invalid.")
        if not isinstance(started_at, str):
            raise ValueError("started_at must be an ISO-8601 timestamp.")
        reservation_values = self._snapshot_reservations(reservations)
        reserved = sorted({gpu_id for item in reservation_values for gpu_id in item["gpu_ids"]})
        summaries_by_project: dict[str, list[dict[str, Any]]] = {}
        for item in reservation_values:
            project_id = item["project_id"]
            if project_id is None:
                continue
            summaries_by_project.setdefault(project_id, []).append(
                {
                    key: item[key]
                    for key in (
                        "reservation_id",
                        "project_id",
                        "group_name",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "gpu_ids",
                        "state",
                        "admission",
                    )
                }
            )
        policy_view = getattr(gpu_policy, "with_reservations", None)
        if callable(policy_view):
            gpu_policy = policy_view(reserved)
        policy_to_dict = getattr(gpu_policy, "to_dict", None)
        if callable(policy_to_dict):
            gpu_policy = policy_to_dict()
        if not isinstance(gpu_policy, Mapping):
            raise ValueError("gpu_policy must be a mapping or policy view.")
        policy = _json_copy(gpu_policy)
        if not isinstance(policy, dict):
            raise ValueError("gpu_policy must be an object.")

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if (
                identity is not None
                and binding.enabled is True
                and identity in self._validated
                and identity not in current
            ):
                current[identity] = binding
        current_identities = set(current)
        self._snapshot_last_published = {
            identity: state
            for identity, state in self._snapshot_last_published.items()
            if identity in current_identities
        }

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed_projects: set[str] = set()
        for request in unresolved:
            if request.operation_kind != "machine_snapshot_publish":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                if self.executor.resolve_stale_machine_snapshot(request.request_id, request):
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue
            processed_projects.add(binding.project_id)
            service_key = self._snapshot_service_key(identity)
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue
            result = self.executor.load_result(request.request_id)
            if result is None:
                if self._service_retry_is_due(service_key):
                    self._start_machine_snapshot_request(request, service_key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("machine snapshot result changed request identity.")
            if result.status == "outcome_unknown":
                if self.executor.reset_ambiguous_machine_snapshot_for_retry(request.request_id, request):
                    self._record_service_failure(service_key)
                    if self._service_retry_is_due(service_key):
                        self._start_machine_snapshot_request(request, service_key)
                continue
            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                evidence = dict(consumed.evidence)
                completions[binding.project_id] = evidence
                if evidence.get("outcome") == "published":
                    self._service_backoff.pop(service_key, None)
                    self._snapshot_last_published[identity] = (
                        self._machine_snapshot_signature(request.parameters),
                        self._monotonic(),
                    )
                else:
                    self._validated.discard(identity)
                    self._snapshot_last_published.pop(identity, None)
                    self._record_service_failure(service_key)
            else:
                self._record_service_failure(service_key)

        can_start = executor_status.get("envelope") != "unknown" and executor_status.get(
            "overdue_worker_count", 0
        ) <= executor_status.get("supported_hang_limit", 0)
        if not can_start:
            return completions
        now = self._monotonic()
        for identity, binding in current.items():
            if binding.project_id in processed_projects:
                continue
            service_key = self._snapshot_service_key(identity)
            if not self._service_retry_is_due(service_key):
                continue
            project_summaries = summaries_by_project.get(binding.project_id, [])
            signature = self._machine_snapshot_signature(
                {
                    "instance_id": instance_id,
                    "pid": pid,
                    "visible_gpu_ids": visible,
                    "reserved_gpu_ids": reserved,
                    "reservation_summaries": project_summaries,
                    "heartbeat_interval_seconds": heartbeat_interval_seconds,
                    "started_at": started_at,
                    "gpu_policy": policy,
                }
            )
            last_published = self._snapshot_last_published.get(identity)
            if (
                last_published is not None
                and last_published[0] == signature
                and now - last_published[1] < heartbeat_interval_seconds
            ):
                continue
            self._offer_new_request(
                identity,
                "machine_snapshot_publish",
                partial(
                    self._prepare_machine_snapshot_publish_request,
                    binding,
                    registry_revision,
                    instance_id,
                    pid,
                    visible,
                    reserved,
                    project_summaries,
                    heartbeat_interval_seconds,
                    started_at,
                    policy,
                    service_key,
                ),
            )
        return completions

    def _prepare_machine_snapshot_publish_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        instance_id: str,
        pid: int | None,
        visible_gpu_ids: list[int],
        reserved_gpu_ids: list[int],
        reservation_summaries: list[dict[str, Any]],
        heartbeat_interval_seconds: int | float,
        started_at: str,
        gpu_policy: dict[str, Any],
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        # A recurring heartbeat must not keep an on-demand agent alive after
        # this binding has completed every service lane. Shutdown publishes
        # the authoritative stop snapshot after the executor is fenced.
        from .config import load_agent_config

        try:
            is_on_demand = load_agent_config(self.runtime).exit_when_idle
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            is_on_demand = False
        if is_on_demand and binding not in self.runtime.working_set.resident_bindings():
            return
        try:
            request = self.executor.prepare_machine_snapshot_publish(
                binding,
                registry_revision,
                instance_id=instance_id,
                pid=pid,
                visible_gpu_ids=visible_gpu_ids,
                reserved_gpu_ids=reserved_gpu_ids,
                reservation_summaries=reservation_summaries,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                started_at=started_at,
                gpu_policy=gpu_policy,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_machine_snapshot_request(request, service_key)

    @_admission_operation
    def advance_registration_renewals(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        renewal_horizon_seconds: int | float,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance registration renewal requests for exact validated bindings."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if (
            type(renewal_horizon_seconds) not in {int, float}
            or (type(renewal_horizon_seconds) is float and not math.isfinite(renewal_horizon_seconds))
            or renewal_horizon_seconds < 0
            or renewal_horizon_seconds > 86_400
        ):
            raise ValueError("renewal_horizon_seconds is invalid.")

        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch
        for retained in (self._registration_due, self._registration_deadlines):
            for identity in tuple(retained):
                if identity.executor_epoch != executor_epoch or identity.registry_revision != registry_revision:
                    retained.pop(identity, None)

        current: dict[_ValidationIdentity, ProjectBinding] = {}
        for binding in bindings:
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is not None and binding.enabled is True and identity not in current:
                current[identity] = binding

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed_projects: set[str] = set()

        for request in unresolved:
            if request.operation_kind != "registration_renew":
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            if binding is None:
                try:
                    resolved = self.executor.resolve_stale_registration_renew(request.request_id, request)
                except (OSError, RuntimeError, ValueError):
                    resolved = False
                if resolved:
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                continue

            processed_projects.add(binding.project_id)
            service_key = self._registration_renew_service_key(identity)
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue

            result = self.executor.load_result(request.request_id)
            if result is None:
                if self._service_retry_is_due(service_key) and executor_status.get(
                    "overdue_worker_count", 0
                ) < executor_status.get("supported_hang_limit", 0):
                    self._start_registration_renew_request(request, service_key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("registration renewal result changed request identity.")
            if result.status == "outcome_unknown":
                try:
                    reset_for_retry = self.executor.reset_ambiguous_registration_renew_for_retry(
                        request.request_id,
                        request,
                    )
                except (OSError, RuntimeError, ValueError):
                    reset_for_retry = False
                if reset_for_retry:
                    self._record_service_failure(service_key)
                continue

            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                evidence = dict(consumed.evidence)
                completions[binding.project_id] = evidence
                if evidence.get("outcome") == "eligible":
                    expires_at = datetime.fromisoformat(evidence["eligibility_expires_at"].replace("Z", "+00:00"))
                    remaining = max(0.0, (expires_at - datetime.now(timezone.utc)).total_seconds())
                    self._service_backoff.pop(service_key, None)
                    # A successor still waits for a worker and the other
                    # authority families. Do not spend half a short lease in
                    # local cooldown before entering that queue.
                    self._registration_due[identity] = self._monotonic() + evidence["renew_after_seconds"]
                    self._registration_deadlines[identity] = self._monotonic() + max(
                        0.0, remaining - min(2.0, remaining * 3 / 4)
                    )
                else:
                    self._validated.discard(identity)
                    self._record_service_failure(service_key)
            else:
                if consumed.status == "fenced":
                    self._validated.discard(identity)
                self._record_service_failure(service_key)

        can_start = executor_status.get("envelope") != "unknown" and executor_status.get(
            "overdue_worker_count", 0
        ) <= executor_status.get("supported_hang_limit", 0)
        if not can_start:
            return completions
        for identity, binding in current.items():
            if binding.project_id in processed_projects:
                continue
            service_key = self._registration_renew_service_key(identity)
            if self._registration_due.get(identity, 0.0) > self._monotonic():
                # Local scheduling acknowledgement only, not fresh shared
                # eligibility evidence or a reusable authority grant.
                completions[binding.project_id] = {"outcome": "not_due"}
                continue
            if not self._service_retry_is_due(service_key):
                continue
            self._offer_new_request(
                identity,
                "registration_renew",
                partial(
                    self._prepare_registration_renew_request,
                    binding,
                    registry_revision,
                    renewal_horizon_seconds,
                    service_key,
                ),
                deadline=self._registration_deadlines.get(identity, self._monotonic()),
            )
        return completions

    def _prepare_registration_renew_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        renewal_horizon_seconds: int | float,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            request = self.executor.prepare_registration_renew(
                binding,
                registry_revision,
                renewal_horizon_seconds=renewal_horizon_seconds,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_registration_renew_request(request, service_key)

    def advance_activation_observations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        replay_cursors: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        """Observe activation checkpoints only for enabled, locally validated bindings."""
        if replay_cursors is not None and not isinstance(replay_cursors, Mapping):
            raise ValueError("replay_cursors must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        for binding in bindings:
            cursor = None if replay_cursors is None else replay_cursors.get(binding.project_id)
            if cursor is None:
                parameters[binding.project_id] = {"replay_epoch": None, "replay_sequence": 0}
                continue
            if not isinstance(cursor, Mapping) or set(cursor) != {"epoch", "sequence"}:
                raise ValueError("activation replay cursor fields are invalid.")
            parameters[binding.project_id] = {
                "replay_epoch": cursor["epoch"],
                "replay_sequence": cursor["sequence"],
            }
        return self._advance_activation_io(
            "activation_observe",
            bindings,
            registry_revision,
            parameters,
            require_validated=False,
        )

    def advance_activation_consumer_registrations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        process_fence: str,
    ) -> dict[str, Mapping[str, Any]]:
        """Register activation consumers using isolated, idempotent Project I/O."""
        try:
            validate_identifier(process_fence, "process_fence")
        except ValueError as exc:
            raise ValueError("process_fence is invalid.") from exc
        parameters = {binding.project_id: {"process_fence": process_fence} for binding in bindings}
        return self._advance_activation_io(
            "activation_consumer_register",
            bindings,
            registry_revision,
            parameters,
            require_validated=False,
        )

    def advance_activation_consumer_acks(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        acknowledgements: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Persist supplied per-Project activation acknowledgements asynchronously.

        Each mapping value contains process_fence, epoch, sequence,
        reconstructed_floor, and require_current. Only bindings with a supplied
        acknowledgement are candidates in this pass.
        """
        if not isinstance(acknowledgements, Mapping):
            raise ValueError("acknowledgements must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        for project_id, value in acknowledgements.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != {
                "process_fence",
                "epoch",
                "sequence",
                "reconstructed_floor",
                "require_current",
            }:
                raise ValueError("activation acknowledgement fields are invalid.")
            parameters[project_id] = dict(value)
        return self._advance_activation_io(
            "activation_consumer_ack",
            bindings,
            registry_revision,
            parameters,
            require_validated=False,
        )

    def advance_activation_consumer_retirements(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Retire exact removed activation-consumer generations asynchronously."""
        project_ids = [binding.project_id for binding in bindings]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("activation consumer retirement accepts at most one generation per Project.")
        return self._advance_activation_io(
            "activation_consumer_retire",
            bindings,
            registry_revision,
            {binding.project_id: {} for binding in bindings},
            require_validated=False,
        )

    def advance_authority_services(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        attempts: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Observe exact Attempts point-in-time without granting reusable authority."""
        if not isinstance(attempts, Mapping):
            raise ValueError("attempts must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
        }
        for project_id, value in attempts.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority service Attempt fields are invalid.")
            parameters[project_id] = {"service_action": "observe_current_attempt", **dict(value)}
        return self._advance_activation_io(
            "authority_service",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_renewals(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        observations: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Renew only exact current point-in-time authority observations."""
        if not isinstance(observations, Mapping):
            raise ValueError("observations must be a mapping by project_id.")
        bindings_by_project = {binding.project_id: binding for binding in bindings}
        parameters: dict[str, dict[str, Any]] = {}
        for project_id, evidence in observations.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(evidence, Mapping) or evidence.get("outcome") != "observed_current":
                raise ValueError("authority renewal requires observed_current evidence.")
            binding = bindings_by_project.get(project_id)
            if binding is None:
                raise ValueError("authority renewal observation has no current binding.")
            expected_provenance = {
                "runtime_id": self.executor.runtime_id,
                "executor_epoch": self._observed_executor_epoch,
                "project_id": project_id,
                "canonical_shared_root": str(binding.shared_root),
                "registration_generation": binding.registration_generation,
                "registry_revision": registry_revision,
                "machine_name": binding.machine_name,
                "service_action": "observe_current_attempt",
                "authority_granted": False,
                "attempt_phase": "running",
                "reason": None,
            }
            for field, expected in expected_provenance.items():
                if evidence.get(field) != expected:
                    raise ValueError(f"authority renewal observation provenance {field} is inconsistent.")
            local_effects = evidence.get("local_effects")
            if not isinstance(local_effects, (list, tuple)) or local_effects:
                raise ValueError("authority renewal observation provenance local_effects is inconsistent.")
            revisions = evidence.get("source_revisions")
            if (
                not isinstance(revisions, Mapping)
                or set(revisions) != {"task", "attempt_digest"}
                or type(revisions.get("task")) is not int
                or not isinstance(revisions.get("attempt_digest"), str)
            ):
                raise ValueError("authority renewal requires exact source revisions.")
            parameters[project_id] = {
                "task_id": evidence.get("task_id"),
                "attempt_id": evidence.get("attempt_id"),
                "attempt_number": evidence.get("attempt_number"),
                "fencing_token": evidence.get("fencing_token"),
                "reservation_id": evidence.get("reservation_id"),
                "process_identity": _json_copy(evidence.get("process_identity")),
                "source_revisions": dict(revisions),
            }
        return self._advance_activation_io(
            "authority_renewal",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_orphan_recoveries(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        attempts: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Recover exact live orphaned Attempts through shared Project I/O."""
        if not isinstance(attempts, Mapping):
            raise ValueError("attempts must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "binding_signature",
            "source_revisions",
            "replay_only",
        }
        for project_id, value in attempts.items():
            validate_identifier(project_id, "project_id")
            if isinstance(value, Mapping):
                value = {"replay_only": False, **value}
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority orphan recovery Attempt fields are invalid.")
            parameters[project_id] = dict(value)
        return self._advance_activation_io(
            "authority_orphan_recovery",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_terminal_observations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        attempts: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Observe exact terminal authority state without granting authority."""
        if not isinstance(attempts, Mapping):
            raise ValueError("attempts must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "mode",
        }
        for project_id, value in attempts.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority terminal observation Attempt fields are invalid.")
            parameters[project_id] = dict(value)
        return self._advance_activation_io(
            "authority_terminal_observe",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_running_publications(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        registrations: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Publish exact local process registrations through bounded Project I/O."""
        if not isinstance(registrations, Mapping):
            raise ValueError("registrations must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "process_created_at",
        }
        for project_id, value in registrations.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority running publication registration fields are invalid.")
            parameters[project_id] = {**dict(value), "source_revisions": {}}
        return self._advance_activation_io(
            "authority_running_publish",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_termination_commits(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        decisions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Commit only the supplied exact shared termination decisions."""
        if not isinstance(decisions, Mapping):
            raise ValueError("decisions must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "decision_id",
            "decision_token",
            "authority_outcome",
            "reason",
            "process_identity",
            "source_revisions",
        }
        for project_id, value in decisions.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority termination decision fields are invalid.")
            parameters[project_id] = dict(value)
        return self._advance_activation_io(
            "authority_termination_commit",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    def advance_authority_terminal_publications(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        transitions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Publish only the supplied exact terminal transitions."""
        if not isinstance(transitions, Mapping):
            raise ValueError("transitions must be a mapping by project_id.")
        parameters: dict[str, dict[str, Any]] = {}
        expected = {
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "mode",
            "phase",
            "reason",
            "exit_code",
            "termination_result",
            "source_revisions",
            "transition_digest",
        }
        for project_id, value in transitions.items():
            validate_identifier(project_id, "project_id")
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("authority terminal transition fields are invalid.")
            parameters[project_id] = dict(value)
        return self._advance_activation_io(
            "authority_terminal_publish",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )

    @_admission_operation
    def advance_submission_control(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Apply shared proof progress and fenced local service acknowledgements."""
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if status.get("envelope") == "unknown" or not isinstance(epoch, str):
            return
        current = {
            identity: binding
            for binding in bindings
            if (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
        }
        for retained in (self._submission_cursors, self._submission_due, self._submission_turns):
            for identity in tuple(retained):
                if identity not in current:
                    retained.pop(identity, None)
        parameters = {}
        for identity, binding in current.items():
            if self.runtime.working_set.is_lane_quiescent(binding, "submission"):
                continue
            if not binding.enabled:
                turn = self.runtime.working_set.begin_turn(binding, "submission")
                self.runtime.working_set.acknowledge(turn, quiescent=True)
            elif (
                identity in self._validated
                and binding.project_id not in self.runtime.upgrade_admission_blocked_projects
                and self._submission_due.get(identity, 0.0) <= self._monotonic()
            ):
                parameters[binding.project_id] = {
                    "continuation": self._submission_cursors.get(identity, submission_control_continuation()),
                }
        results = self._advance_activation_io(
            "submission_control_service",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )
        by_project = {identity.project_id: identity for identity in current}
        for project_id, evidence in results.items():
            identity = by_project[project_id]
            turn = self._submission_turns.pop(identity, None)
            if "continuation" not in evidence:
                if turn is not None:
                    self.runtime.working_set.acknowledge(turn, quiescent=False)
                continue
            self._submission_cursors[identity] = dict(evidence["continuation"])
            quiescent = evidence["quiescent"]
            if turn is not None:
                self.runtime.working_set.acknowledge(turn, quiescent=quiescent)
            if quiescent or evidence["reason_code"] == "blocked":
                self._submission_due[identity] = self._monotonic() + 1.0
            else:
                self._submission_due.pop(identity, None)

    @_admission_operation
    def advance_observation_maintenance(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Apply projection maintenance completion to a captured local observation turn."""
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if status.get("envelope") == "unknown" or not isinstance(epoch, str):
            return
        current = {
            identity: binding
            for binding in bindings
            if (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
        }
        for retained in (self._observation_due, self._observation_turns):
            for identity in tuple(retained):
                if identity not in current:
                    retained.pop(identity, None)
        parameters = {}
        for identity, binding in current.items():
            if self.runtime.working_set.is_lane_quiescent(binding, "observation"):
                continue
            if not binding.enabled:
                turn = self.runtime.working_set.begin_turn(binding, "observation")
                self.runtime.working_set.acknowledge(turn, quiescent=True)
            elif (
                identity in self._validated
                and binding.project_id not in self.runtime.upgrade_admission_blocked_projects
                and self._observation_due.get(identity, 0.0) <= self._monotonic()
            ):
                parameters[binding.project_id] = {}
        results = self._advance_activation_io(
            "observation_service",
            bindings,
            registry_revision,
            parameters,
            require_validated=True,
        )
        by_project = {identity.project_id: identity for identity in current}
        for project_id, evidence in results.items():
            identity = by_project[project_id]
            turn = self._observation_turns.pop(identity, None)
            quiescent = evidence.get("quiescent") is True
            if turn is not None:
                self.runtime.working_set.acknowledge(turn, quiescent=quiescent)
            if quiescent or evidence.get("reason_code") == "blocked":
                self._observation_due[identity] = self._monotonic() + 1.0
            else:
                self._observation_due.pop(identity, None)

    @_admission_operation
    def advance_notification_maintenance(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Reconcile registered legacy sources through the common background arbiter."""
        status = self._poll_admission_executor()
        epoch = status.get("executor_epoch")
        if status.get("envelope") == "unknown" or not isinstance(epoch, str):
            return
        current = {
            identity: binding
            for binding in bindings
            if (identity := self._binding_identity(binding, registry_revision, epoch)) is not None
        }
        for identity in tuple(self._notification_due):
            if identity not in current:
                self._notification_due.pop(identity, None)
        parameters = {
            binding.project_id: {}
            for identity, binding in current.items()
            if binding.enabled
            and identity in self._validated
            and binding.project_id not in self.runtime.upgrade_admission_blocked_projects
            and self._notification_due.get(identity, 0.0) <= self._monotonic()
        }
        results = self._advance_activation_io(
            "notification_service", bindings, registry_revision, parameters, require_validated=True
        )
        by_project = {identity.project_id: identity for identity in current}
        for project_id, evidence in results.items():
            if "state" in evidence:
                self._notification_due[by_project[project_id]] = self._monotonic() + 5.0

    @_admission_operation
    def advance_progress_projection(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Serve local-captured progress through the common background arbiter."""
        return self._advance_activation_io(
            "progress_projection", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_recovery_admission(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Serve registration/root recovery preparation through common admission."""
        return self._advance_activation_io(
            "recovery_admission", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_recovery_capture_transitions(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        return self._advance_activation_io(
            "recovery_capture_transition", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_recovery_group_authority(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        return self._advance_activation_io(
            "recovery_group_authority", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_recovery_source_releases(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        return self._advance_activation_io(
            "recovery_source_release", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_recovery_source_holds(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Serve exact source retention through the common background arbiter."""
        return self._advance_activation_io(
            "recovery_source_hold", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_legacy_capture_scans(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        return self._advance_activation_io(
            "legacy_capture_scan", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    @_admission_operation
    def advance_legacy_capture_reads(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Serve retained source reads through the common background arbiter."""
        return self._advance_activation_io(
            "legacy_capture_read", bindings, registry_revision, parameters_by_project, require_validated=False
        )

    def advance_progress_work(self, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        self.progress.advance(self, bindings, registry_revision)

    @_admission_operation
    def _advance_activation_io(
        self,
        operation_kind: str,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        parameters_by_project: Mapping[str, Mapping[str, Any]],
        *,
        require_validated: bool,
    ) -> dict[str, Mapping[str, Any]]:
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        executor_status = self._poll_admission_executor()
        executor_epoch = executor_status.get("executor_epoch")
        if executor_status.get("envelope") == "unknown" or not isinstance(executor_epoch, str):
            return {}
        self._observed_executor_epoch = executor_epoch

        current: dict[_ValidationIdentity, ProjectBinding] = {}
        parameters_for_identity: dict[_ValidationIdentity, dict[str, Any]] = {}
        allows_disabled_binding = operation_kind in _AUTHORITY_OPERATION_KINDS or operation_kind in {
            "activation_consumer_retire",
            "progress_projection",
            "legacy_capture_read",
            "legacy_capture_scan",
            "recovery_source_hold",
            "recovery_admission",
            "recovery_source_release",
            "recovery_group_authority",
            "recovery_capture_transition",
        }
        for binding in bindings:
            if binding.enabled is not True and not allows_disabled_binding:
                continue
            raw_parameters = parameters_by_project.get(binding.project_id)
            if raw_parameters is None:
                continue
            identity = self._binding_identity(binding, registry_revision, executor_epoch)
            if identity is None or (require_validated and identity not in self._validated) or identity in current:
                continue
            current[identity] = binding
            parameters_for_identity[identity] = {
                "machine_name": binding.machine_name,
                **(dict(raw_parameters) if raw_parameters is not None else {}),
            }

        unresolved = self.executor.unresolved_requests()
        unresolved_by_id = {request.request_id: request for request in unresolved}
        maintenance_descriptor_unresolved = {
            self._request_identity(request)
            for request in unresolved
            if request.operation_kind == "maintenance_descriptor_advance"
        }
        unresolved_owners = {
            (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
        }
        self._started_requests.intersection_update(unresolved_by_id)
        completions: dict[str, Mapping[str, Any]] = {}
        processed_projects: set[str] = set()

        for request in unresolved:
            if request.operation_kind != operation_kind:
                continue
            identity = self._request_identity(request)
            binding = current.get(identity)
            expected_parameters = parameters_for_identity.get(identity)
            request_matches = dict(request.parameters) == expected_parameters
            if operation_kind in {"progress_projection", "legacy_capture_scan"}:
                request_matches = _json_copy(request.parameters) == expected_parameters
            if operation_kind in _AUTHORITY_MUTATION_OPERATION_KINDS and expected_parameters is not None:
                request_matches = dict(request.parameters) == {
                    key: value for key, value in expected_parameters.items() if key != "source_revisions"
                } and dict(request.source_revisions) == expected_parameters.get("source_revisions")
            if binding is None or not request_matches:
                stale_service_key = (
                    self._authority_service_key(identity, operation_kind)
                    if operation_kind in _AUTHORITY_OPERATION_KINDS
                    else self._activation_service_key(identity, operation_kind)
                )
                stale_result = self.executor.load_result(request.request_id)
                if (
                    operation_kind in _AUTHORITY_MUTATION_OPERATION_KINDS
                    and stale_result is not None
                    and stale_result.status == "retryable_error"
                ):
                    if stale_service_key not in self._service_backoff:
                        self._record_service_failure(stale_service_key)
                        continue
                    if not self._service_retry_is_due(stale_service_key):
                        continue
                try:
                    if operation_kind == "activation_observe":
                        resolved = self.executor.resolve_stale_activation_observe(request.request_id, request)
                    elif operation_kind == "submission_control_service":
                        resolved = self.executor.resolve_stale_submission_control_service(request.request_id, request)
                    elif operation_kind == "observation_service":
                        resolved = self.executor.resolve_stale_observation_service(request.request_id, request)
                    elif operation_kind == "notification_service":
                        resolved = self.executor.resolve_stale_notification_service(request.request_id, request)
                    elif operation_kind == "progress_projection":
                        resolved = self.executor.resolve_stale_progress_projection(request.request_id, request)
                    elif operation_kind == "legacy_capture_read":
                        resolved = self.executor.resolve_stale_legacy_capture_read(request.request_id, request)
                    elif operation_kind == "legacy_capture_scan":
                        resolved = self.executor.resolve_stale_legacy_capture_scan(request.request_id, request)
                    elif operation_kind == "recovery_source_hold":
                        resolved = self.executor.resolve_stale_recovery_source_hold(request.request_id, request)
                    elif operation_kind == "recovery_admission":
                        resolved = self.executor.resolve_stale_recovery_admission(request.request_id, request)
                    elif operation_kind == "recovery_source_release":
                        resolved = self.executor.resolve_stale_recovery_source_release(request.request_id, request)
                    elif operation_kind == "recovery_group_authority":
                        resolved = self.executor.resolve_stale_recovery_group_authority(request.request_id, request)
                    elif operation_kind == "recovery_capture_transition":
                        resolved = self.executor.resolve_stale_recovery_capture_transition(request.request_id, request)
                    elif operation_kind == "activation_consumer_register":
                        resolved = self.executor.resolve_stale_activation_consumer_register(request.request_id, request)
                    elif operation_kind == "activation_consumer_retire":
                        resolved = self.executor.resolve_stale_activation_consumer_retire(request.request_id, request)
                    elif operation_kind == "authority_service":
                        resolved = self.executor.resolve_stale_authority_service(request.request_id, request)
                    elif operation_kind == "authority_renewal":
                        resolved = self.executor.resolve_stale_authority_renewal(request.request_id, request)
                    elif operation_kind == "authority_orphan_recovery":
                        resolved = self.executor.resolve_stale_authority_orphan_recovery(request.request_id, request)
                    elif operation_kind == "authority_terminal_observe":
                        resolved = self.executor.resolve_stale_authority_terminal_observe(request.request_id, request)
                    elif operation_kind == "authority_termination_commit":
                        resolved = self.executor.resolve_stale_authority_termination_commit(request.request_id, request)
                    elif operation_kind == "authority_terminal_publish":
                        resolved = self.executor.resolve_stale_authority_terminal_publish(request.request_id, request)
                    elif operation_kind == "authority_running_publish":
                        resolved = self.executor.resolve_stale_authority_running_publish(request.request_id, request)
                    else:
                        resolved = self.executor.resolve_stale_activation_consumer_ack(request.request_id, request)
                except (OSError, RuntimeError, ValueError):
                    resolved = False
                if resolved:
                    unresolved_by_id.pop(request.request_id, None)
                    unresolved_owners.discard(identity.request_owner)
                    self._started_requests.discard(request.request_id)
                    self._service_backoff.pop(stale_service_key, None)
                elif (
                    operation_kind in _AUTHORITY_MUTATION_OPERATION_KINDS
                    and stale_result is not None
                    and stale_result.status == "retryable_error"
                ):
                    self._record_service_failure(stale_service_key)
                continue

            processed_projects.add(binding.project_id)
            service_key = (
                self._authority_service_key(identity, operation_kind)
                if operation_kind in _AUTHORITY_OPERATION_KINDS
                else self._activation_service_key(identity, operation_kind)
            )
            try:
                self.executor._load_process(request.request_id)
            except FileNotFoundError:
                self._started_requests.discard(request.request_id)
            else:
                self._started_requests.add(request.request_id)
                continue

            result = self.executor.load_result(request.request_id)
            if result is None:
                if self._service_retry_is_due(service_key):
                    self._start_activation_request(request, service_key)
                continue
            if result.request != request:
                raise ProjectIOProtocolError("activation I/O result changed request identity.")
            if result.status == "outcome_unknown" and operation_kind in {
                "submission_control_service",
                "observation_service",
                "notification_service",
                "progress_projection",
                "legacy_capture_read",
                "legacy_capture_scan",
                "recovery_source_hold",
                "recovery_admission",
                "recovery_source_release",
                "recovery_group_authority",
                "recovery_capture_transition",
                "activation_consumer_register",
                "activation_consumer_ack",
                "activation_consumer_retire",
                *_AUTHORITY_MUTATION_OPERATION_KINDS,
            }:
                reset = False
                try:
                    if operation_kind == "activation_consumer_register":
                        reset = self.executor.reset_ambiguous_activation_consumer_register_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "submission_control_service":
                        reset = self.executor.reset_ambiguous_submission_control_service_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "observation_service":
                        reset = self.executor.reset_ambiguous_observation_service_for_retry(request.request_id, request)
                    elif operation_kind == "notification_service":
                        reset = self.executor.reset_ambiguous_notification_service_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "progress_projection":
                        reset = self.executor.reset_ambiguous_progress_projection_for_retry(request.request_id, request)
                    elif operation_kind == "legacy_capture_read":
                        reset = self.executor.reset_ambiguous_legacy_capture_read_for_retry(request.request_id, request)
                    elif operation_kind == "legacy_capture_scan":
                        reset = self.executor.reset_ambiguous_legacy_capture_scan_for_retry(request.request_id, request)
                    elif operation_kind == "recovery_source_hold":
                        reset = self.executor.reset_ambiguous_recovery_source_hold_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "recovery_admission":
                        reset = self.executor.reset_ambiguous_recovery_admission_for_retry(request.request_id, request)
                    elif operation_kind == "recovery_source_release":
                        reset = self.executor.reset_ambiguous_recovery_source_release_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "recovery_group_authority":
                        reset = self.executor.reset_ambiguous_recovery_group_authority_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "recovery_capture_transition":
                        reset = self.executor.reset_ambiguous_recovery_capture_transition_for_retry(
                            request.request_id, request
                        )
                    elif operation_kind == "activation_consumer_ack":
                        reset = self.executor.reset_ambiguous_activation_consumer_ack_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "activation_consumer_retire":
                        reset = self.executor.reset_ambiguous_activation_consumer_retire_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "authority_renewal":
                        reset = self.executor.reset_ambiguous_authority_renewal_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "authority_orphan_recovery":
                        reset = self.executor.reset_ambiguous_authority_orphan_recovery_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "authority_termination_commit":
                        reset = self.executor.reset_ambiguous_authority_termination_commit_for_retry(
                            request.request_id,
                            request,
                        )
                    elif operation_kind == "authority_running_publish":
                        reset = self.executor.reset_ambiguous_authority_running_publish_for_retry(
                            request.request_id,
                            request,
                        )
                    else:
                        reset = self.executor.reset_ambiguous_authority_terminal_publish_for_retry(
                            request.request_id,
                            request,
                        )
                except (OSError, RuntimeError, ValueError):
                    reset = False
                if reset:
                    self._record_service_failure(service_key)
                continue

            consumed = self.executor.consume(request.request_id, request)
            if consumed is None:
                continue
            unresolved_by_id.pop(request.request_id, None)
            unresolved_owners.discard(identity.request_owner)
            self._started_requests.discard(request.request_id)
            if consumed.status == "completed":
                evidence = dict(consumed.evidence)
                if operation_kind in {
                    "progress_projection",
                    "legacy_capture_read",
                    "legacy_capture_scan",
                    "recovery_source_hold",
                }:
                    evidence = _json_copy(consumed.evidence)
                completions[binding.project_id] = evidence
                if operation_kind == "activation_observe":
                    replay = evidence.get("replay")
                    if (
                        isinstance(replay, Mapping)
                        and type(replay.get("sequence")) is int
                        and replay["sequence"] > request.parameters["replay_sequence"]
                    ):
                        self._maintenance_descriptor_settled.pop(identity, None)
                        self._service_backoff.pop(self._maintenance_descriptor_service_key(identity), None)
                        self.runtime.maintenance_retry_deadlines.pop(binding.project_id, None)
                        self.runtime.maintenance_retry_idle_blocked_projects.discard(binding.project_id)
                        if identity in maintenance_descriptor_unresolved:
                            self._maintenance_descriptor_dirty.add(identity)
                if evidence.get("outcome") in {"stale", "observed_stale"}:
                    if operation_kind in _AUTHORITY_OPERATION_KINDS:
                        self._service_backoff.pop(service_key, None)
                    else:
                        self._validated.discard(identity)
                        self._record_service_failure(service_key)
                elif (
                    operation_kind in {"recovery_admission", "recovery_group_authority"}
                    and evidence.get("state") == "waiting"
                ):
                    self._service_backoff[service_key] = _FailureBackoff(
                        0, self._monotonic() + _RECOVERY_WAIT_RETRY_SECONDS
                    )
                else:
                    self._service_backoff.pop(service_key, None)
            else:
                if consumed.status == "fenced":
                    self._validated.discard(identity)
                self._record_service_failure(service_key)
                completions[binding.project_id] = {
                    "outcome": "unavailable",
                    "reason": consumed.reason_code,
                }

        can_start = executor_status.get("envelope") != "unknown" and executor_status.get(
            "overdue_worker_count", 0
        ) <= executor_status.get("supported_hang_limit", 0)
        if not can_start:
            return completions

        ordered = list(current.items())
        if ordered:
            offset = self._activation_offsets.get(operation_kind, 0) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
            self._activation_offsets[operation_kind] = (offset + min(64, len(ordered))) % len(ordered)
        else:
            self._activation_offsets.pop(operation_kind, None)
        for identity, binding in ordered[:64]:
            if binding.project_id in processed_projects:
                continue
            service_key = (
                self._authority_service_key(identity, operation_kind)
                if operation_kind in _AUTHORITY_OPERATION_KINDS
                else self._activation_service_key(identity, operation_kind)
            )
            if not self._service_retry_is_due(service_key):
                continue
            self._offer_new_request(
                identity,
                operation_kind,
                partial(
                    self._prepare_activation_io_request,
                    operation_kind,
                    identity,
                    binding,
                    registry_revision,
                    parameters_for_identity[identity],
                    service_key,
                ),
                *(
                    (
                        "progress",
                        parameters_for_identity[identity]["context"]["attempt_id"],
                        ("publish" if parameters_for_identity[identity].get("projection") is not None else "observe"),
                    )
                    if operation_kind == "progress_projection"
                    else ()
                ),
            )
        return completions

    def _prepare_activation_io_request(
        self,
        operation_kind: str,
        identity: _ValidationIdentity,
        binding: ProjectBinding,
        registry_revision: int,
        parameters: Mapping[str, Any],
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if operation_kind == "activation_observe" and not self.runtime.working_set.is_activation_observation_due(
            binding,
            replay_epoch=parameters["replay_epoch"],
            replay_sequence=parameters["replay_sequence"],
        ):
            return
        try:
            if operation_kind == "activation_observe":
                request = self.executor.prepare_activation_observe(
                    binding,
                    registry_revision,
                    replay_epoch=parameters["replay_epoch"],
                    replay_sequence=parameters["replay_sequence"],
                )
            elif operation_kind == "submission_control_service":
                request = self.executor.prepare_submission_control_service(
                    binding,
                    registry_revision,
                    continuation=parameters["continuation"],
                )
                self._submission_turns[identity] = self.runtime.working_set.begin_turn(binding, "submission")
            elif operation_kind == "observation_service":
                request = self.executor.prepare_observation_service(binding, registry_revision)
                self._observation_turns[identity] = self.runtime.working_set.begin_turn(binding, "observation")
            elif operation_kind == "notification_service":
                request = self.executor.prepare_notification_service(binding, registry_revision)
            elif operation_kind == "progress_projection":
                request = self.executor.prepare_progress_projection(
                    binding, registry_revision, context=parameters["context"], projection=parameters["projection"]
                )
            elif operation_kind == "recovery_admission":
                request = self.executor.prepare_recovery_admission(binding, registry_revision)
            elif operation_kind == "recovery_source_release":
                request = self.executor.prepare_recovery_source_release(
                    binding, registry_revision, completion_digest=parameters["completion_digest"]
                )
            elif operation_kind == "recovery_group_authority":
                request = self.executor.prepare_recovery_group_authority(
                    binding, registry_revision, completion_digest=parameters["completion_digest"]
                )
            elif operation_kind == "recovery_capture_transition":
                request = self.executor.prepare_recovery_capture_transition(
                    binding,
                    registry_revision,
                    completion_digest=parameters["completion_digest"],
                    phase=parameters["phase"],
                )
            elif operation_kind == "recovery_source_hold":
                request = self.executor.prepare_recovery_source_hold(
                    binding, registry_revision, capture_id=parameters["capture_id"]
                )
            elif operation_kind in {"legacy_capture_read", "legacy_capture_scan"}:
                prepare = (
                    self.executor.prepare_legacy_capture_read
                    if operation_kind == "legacy_capture_read"
                    else self.executor.prepare_legacy_capture_scan
                )
                request = prepare(
                    binding,
                    registry_revision,
                    **{key: value for key, value in parameters.items() if key != "machine_name"},
                )
            elif operation_kind == "authority_service":
                request = self.executor.prepare_authority_service(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                )
            elif operation_kind == "authority_renewal":
                request = self.executor.prepare_authority_renewal(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                    source_revisions=parameters["source_revisions"],
                )
            elif operation_kind == "authority_orphan_recovery":
                request = self.executor.prepare_authority_orphan_recovery(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                    binding_signature=parameters["binding_signature"],
                    source_revisions=parameters["source_revisions"],
                    replay_only=parameters.get("replay_only", False),
                )
            elif operation_kind == "authority_terminal_observe":
                request = self.executor.prepare_authority_terminal_observe(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                    mode=parameters["mode"],
                )
            elif operation_kind == "authority_termination_commit":
                request = self.executor.prepare_authority_termination_commit(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    decision_id=parameters["decision_id"],
                    decision_token=parameters["decision_token"],
                    authority_outcome=parameters["authority_outcome"],
                    reason=parameters["reason"],
                    process_identity=parameters["process_identity"],
                    source_revisions=parameters["source_revisions"],
                )
            elif operation_kind == "authority_terminal_publish":
                request = self.executor.prepare_authority_terminal_publish(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                    mode=parameters["mode"],
                    phase=parameters["phase"],
                    reason=parameters["reason"],
                    exit_code=parameters["exit_code"],
                    termination_result=parameters["termination_result"],
                    source_revisions=parameters["source_revisions"],
                    transition_digest=parameters["transition_digest"],
                )
            elif operation_kind == "authority_running_publish":
                request = self.executor.prepare_authority_running_publish(
                    binding,
                    registry_revision,
                    task_id=parameters["task_id"],
                    attempt_id=parameters["attempt_id"],
                    attempt_number=parameters["attempt_number"],
                    fencing_token=parameters["fencing_token"],
                    reservation_id=parameters["reservation_id"],
                    process_identity=parameters["process_identity"],
                    process_created_at=parameters["process_created_at"],
                )
            elif operation_kind == "activation_consumer_register":
                request = self.executor.prepare_activation_consumer_register(
                    binding,
                    registry_revision,
                    process_fence=parameters["process_fence"],
                )
            elif operation_kind == "activation_consumer_retire":
                request = self.executor.prepare_activation_consumer_retire(binding, registry_revision)
            else:
                request = self.executor.prepare_activation_consumer_ack(
                    binding,
                    registry_revision,
                    process_fence=parameters["process_fence"],
                    epoch=parameters["epoch"],
                    sequence=parameters["sequence"],
                    reconstructed_floor=parameters["reconstructed_floor"],
                    require_current=parameters["require_current"],
                )
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        self._start_activation_request(request, service_key)

    @staticmethod
    def _snapshot_gpu_ids(value: Sequence[int], label: str) -> list[int]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes)
            or len(value) > 4096
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(f"{label} must be a bounded sequence of unique nonnegative integers.")
        return sorted(value)

    @staticmethod
    def _snapshot_reservations(reservations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(reservations, Sequence) or isinstance(reservations, str | bytes) or len(reservations) > 4096:
            raise ValueError("reservations must be a bounded sequence.")
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in reservations:
            if not isinstance(item, Mapping):
                raise ValueError("reservations must contain reservation objects.")
            reservation_id = item.get("reservation_id")
            try:
                validate_identifier(reservation_id, "reservation_id")
            except ValueError as exc:
                raise ValueError("reservation identity is invalid.") from exc
            if reservation_id in seen_ids:
                raise ValueError("reservations must have unique reservation IDs.")
            seen_ids.add(reservation_id)
            project_id = item.get("project_id")
            if project_id is not None:
                try:
                    validate_identifier(project_id, "project_id")
                except ValueError as exc:
                    raise ValueError("reservation project identity is invalid.") from exc
            state = item.get("state")
            if state not in ("active", "provisional"):
                raise ValueError("reservation state is invalid.")
            gpu_ids_value = item.get("gpu_ids", [])
            if (
                not isinstance(gpu_ids_value, Sequence)
                or isinstance(gpu_ids_value, str | bytes)
                or len(gpu_ids_value) > 4096
                or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids_value)
                or len(set(gpu_ids_value)) != len(gpu_ids_value)
            ):
                raise ValueError("reservation GPU identity is invalid.")
            summary: dict[str, Any] = {
                "reservation_id": reservation_id,
                "project_id": project_id,
                "gpu_ids": list(gpu_ids_value),
                "state": state,
            }
            for field in ("group_name", "machine_name", "task_id", "attempt_id"):
                value = item.get(field)
                if value is not None:
                    try:
                        validate_identifier(value, field)
                    except ValueError as exc:
                        raise ValueError(f"reservation {field} is invalid.") from exc
                summary[field] = value
            admission = item.get("admission")
            if admission is not None and not isinstance(admission, Mapping):
                raise ValueError("reservation admission must be an object or null.")
            summary["admission"] = _json_copy(admission) if admission is not None else None
            normalized.append(summary)
        return normalized

    @staticmethod
    def _snapshot_service_key(identity: _ValidationIdentity) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "machine_snapshot_publish", "snapshot", "machine")

    @staticmethod
    def _registration_renew_service_key(identity: _ValidationIdentity) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "registration_renew", "renewal", "registration")

    @staticmethod
    def _maintenance_descriptor_service_key(
        identity: _ValidationIdentity,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "maintenance_descriptor_advance", "descriptor", "advance")

    @staticmethod
    def _activation_service_key(
        identity: _ValidationIdentity,
        operation_kind: str,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, operation_kind, "activation", "consumer")

    @staticmethod
    def _authority_service_key(
        identity: _ValidationIdentity,
        operation_kind: str = "authority_service",
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, operation_kind, "authority", "attempt")

    def _start_activation_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    @staticmethod
    def _machine_snapshot_signature(parameters: Mapping[str, Any]) -> str:
        names = (
            "instance_id",
            "pid",
            "visible_gpu_ids",
            "reserved_gpu_ids",
            "reservation_summaries",
            "heartbeat_interval_seconds",
            "started_at",
            "gpu_policy",
        )
        try:
            encoded = json.dumps(
                {name: _json_copy(parameters[name]) for name in names},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("machine snapshot input is not bounded JSON data.") from exc
        return hashlib.sha256(encoded).hexdigest()

    def _start_machine_snapshot_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    def _start_registration_renew_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    def _next_local_event(self, project_id: str) -> tuple[str, str, str, str] | None:
        root = local_paths(self.runtime.project_paths(project_id)["root"])["events"]
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            root_fd = os.open(root, directory_flags)
        except (FileNotFoundError, OSError):
            return None
        try:
            root_metadata = os.fstat(root_fd)
            if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != os.geteuid():
                return None
            with os.scandir(root_fd) as entries:
                buckets = sorted(
                    (entry.name, entry.inode())
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False) and entry.name not in {".", ".."}
                )
            for bucket_name, bucket_inode in buckets:
                try:
                    bucket_fd = os.open(bucket_name, directory_flags, dir_fd=root_fd)
                except (FileNotFoundError, OSError):
                    continue
                try:
                    bucket_metadata = os.fstat(bucket_fd)
                    if (
                        not stat.S_ISDIR(bucket_metadata.st_mode)
                        or bucket_metadata.st_uid != os.geteuid()
                        or bucket_metadata.st_ino != bucket_inode
                    ):
                        continue
                    with os.scandir(bucket_fd) as entries:
                        candidates = sorted(
                            (entry.name, entry.inode())
                            for entry in entries
                            if _EVENT_FILENAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False)
                        )
                    for filename, file_inode in candidates:
                        try:
                            metadata = os.stat(filename, dir_fd=bucket_fd, follow_symlinks=False)
                            if (
                                not stat.S_ISREG(metadata.st_mode)
                                or metadata.st_uid != os.geteuid()
                                or metadata.st_nlink != 1
                                or metadata.st_size > 65_536
                                or metadata.st_ino != file_inode
                            ):
                                continue
                            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                            descriptor = os.open(filename, flags, dir_fd=bucket_fd)
                            with os.fdopen(descriptor, "rb") as handle:
                                opened = os.fstat(handle.fileno())
                                encoded = handle.read(65_537)
                        except (FileNotFoundError, OSError):
                            continue
                        if len(encoded) > 65_536 or (opened.st_dev, opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            continue
                        return (
                            bucket_name,
                            filename,
                            filename.removesuffix(".json"),
                            hashlib.sha256(encoded).hexdigest(),
                        )
                finally:
                    os.close(bucket_fd)
            return None
        finally:
            os.close(root_fd)

    def _start_maintenance_event_request(
        self,
        request: ProjectIORequest,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(key)
            return
        if process is None:
            self._record_service_failure(key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(key, None)

    @staticmethod
    def _reservation_identity_from_request(request: ProjectIORequest) -> ReservationIdentity:
        value = request.parameters["reservation_identity"]
        return ReservationIdentity(
            reservation_id=value["reservation_id"],
            acquisition_id=value["acquisition_id"],
            project_id=value["project_id"],
            task_id=value["task_id"],
            attempt_id=value["attempt_id"],
            fencing_token=value["fencing_token"],
            gpu_ids=tuple(value["gpu_ids"] or ()),
            cpu_slots=value["cpu_slots"],
            shared_root=value["shared_root"],
            registration_generation=value["registration_generation"],
            executor_epoch=value["executor_epoch"],
            executor_request_id=value["executor_request_id"],
        )

    @staticmethod
    def _reservation_reconcile_service_key(
        identity: _ValidationIdentity,
        reservation_id: str,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "scheduler_reservation_reconcile", "reservation", reservation_id)

    def _prepare_scheduler_reservation_reconcile_request(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        reservation: ReservationIdentity,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        self._reservation_reconcile_cursors[(binding.project_id, binding.registration_generation)] = (
            reservation.reservation_id
        )
        try:
            if classify_exact_reservation(self.runtime.root, reservation) != "matching_active":
                return
            request = self.executor.prepare_scheduler_reservation_reconcile(
                binding,
                registry_revision,
                reservation_identity=reservation,
            )
        except (OSError, RuntimeError, ValueError):
            self._record_reservation_reconcile_failure(service_key)
            return
        self._start_reservation_reconcile_request(request, service_key)

    def _start_reservation_reconcile_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_reservation_reconcile_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_reservation_reconcile_failure(service_key)
            return
        if process is None:
            self._record_reservation_reconcile_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._reservation_reconcile_backoff.pop(service_key, None)

    @staticmethod
    def _claim_identity_for_reservation(
        reservation: ReservationIdentity,
        identity: _ValidationIdentity,
    ) -> dict[str, Any] | None:
        prefix = f"{reservation.task_id}-attempt-"
        attempt_id = reservation.attempt_id
        if (
            reservation.project_id != identity.project_id
            or reservation.shared_root != identity.shared_root
            or reservation.registration_generation not in {None, identity.registration_generation}
            or not isinstance(attempt_id, str)
            or not attempt_id.startswith(prefix)
            or type(reservation.fencing_token) is not int
            or reservation.fencing_token < 1
        ):
            return None
        suffix = attempt_id[len(prefix) :]
        if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0"):
            return None
        attempt_number = int(suffix)
        if attempt_number < 1:
            return None
        return {
            "task_id": reservation.task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": reservation.fencing_token,
            "reservation_id": reservation.reservation_id,
        }

    @staticmethod
    def _launch_request_matches_reservation(
        request: ProjectIORequest,
        reservation: ReservationIdentity,
    ) -> bool:
        value = request.parameters.get("claim_identity")
        attempt_id = reservation.attempt_id
        prefix = f"{reservation.task_id}-attempt-"
        if not isinstance(value, Mapping) or not isinstance(attempt_id, str) or not attempt_id.startswith(prefix):
            return False
        suffix = attempt_id[len(prefix) :]
        if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0"):
            return False
        return (
            request.project_id == reservation.project_id
            and request.canonical_shared_root == reservation.shared_root
            and reservation.registration_generation in {None, request.registration_generation}
            and dict(value)
            == {
                "task_id": reservation.task_id,
                "attempt_id": attempt_id,
                "attempt_number": int(suffix),
                "fencing_token": reservation.fencing_token,
                "reservation_id": reservation.reservation_id,
            }
        )

    @staticmethod
    def _launch_service_key(identity: _ValidationIdentity) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "scheduler_launch_authorize", "launch", "active")

    def _start_launch_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    def validated_config(self, binding: ProjectBinding, registry_revision: int) -> RootConfig | None:
        """Return a locally built config only for an exact cached validation identity."""
        if type(registry_revision) is not int or registry_revision < 0:
            return None
        identity = self._binding_identity(binding, registry_revision, self._observed_executor_epoch)
        if identity is None or identity not in self._validated:
            return None
        return self._root_config(binding)

    def _binding_identity(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        executor_epoch: str | None,
    ) -> _ValidationIdentity | None:
        if not binding.registration_generation or not executor_epoch:
            return None
        if binding.runtime_instance_id != self.executor.runtime_id or binding.runtime_root != str(self.runtime.root):
            return None
        return _ValidationIdentity(
            self.executor.runtime_id,
            executor_epoch,
            binding.project_id,
            str(binding.shared_root),
            binding.machine_name,
            binding.registration_generation,
            str(self.runtime.root),
            registry_revision,
        )

    def _request_identity(self, request: ProjectIORequest) -> _ValidationIdentity:
        return _ValidationIdentity(
            request.runtime_id,
            request.executor_epoch,
            request.project_id,
            request.canonical_shared_root,
            request.parameters["machine_name"],
            request.registration_generation,
            str(self.runtime.root),
            request.registry_revision,
        )

    def _start_retry_is_due(self, identity: _ValidationIdentity, request_id: str) -> bool:
        if request_id in self._started_requests:
            return False
        failure = self._backoff.get(identity)
        return failure is None or failure.retry_at <= self._monotonic()

    def _start_existing(self, request: ProjectIORequest, identity: _ValidationIdentity) -> None:
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_failure(identity)
            return
        if process is None:
            self._record_failure(identity)
            return
        self._started_requests.add(request.request_id)
        self._backoff.pop(identity, None)

    def _record_failure(self, identity: _ValidationIdentity) -> None:
        previous = self._backoff.get(identity)
        failures = previous.consecutive_failures + 1 if previous is not None else 1
        exponent = min(failures - 1, 6)
        delay = min(_INITIAL_BACKOFF_SECONDS * (2**exponent), _MAX_BACKOFF_SECONDS)
        self._backoff[identity] = _FailureBackoff(failures, self._monotonic() + delay)

    @staticmethod
    def _observation_service_key(
        identity: _ValidationIdentity,
        lane: str,
        admission_role: str,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "scheduler_observe", lane, admission_role)

    def _service_retry_is_due(self, key: tuple[_ValidationIdentity, str, str, str]) -> bool:
        failure = self._service_backoff.get(key)
        return failure is None or failure.retry_at <= self._monotonic()

    def _record_service_failure(self, key: tuple[_ValidationIdentity, str, str, str]) -> None:
        previous = self._service_backoff.get(key)
        failures = previous.consecutive_failures + 1 if previous is not None else 1
        exponent = min(failures - 1, 6)
        delay = min(_INITIAL_BACKOFF_SECONDS * (2**exponent), _MAX_BACKOFF_SECONDS)
        self._service_backoff[key] = _FailureBackoff(failures, self._monotonic() + delay)

    def _reservation_reconcile_retry_is_due(
        self,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> bool:
        failure = self._reservation_reconcile_backoff.get(key)
        return failure is None or failure.retry_at <= self._monotonic()

    def _record_reservation_reconcile_failure(
        self,
        key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        previous = self._reservation_reconcile_backoff.get(key)
        failures = previous.consecutive_failures + 1 if previous is not None else 1
        exponent = min(failures - 1, 6)
        delay = min(_INITIAL_BACKOFF_SECONDS * (2**exponent), _MAX_BACKOFF_SECONDS)
        self._reservation_reconcile_backoff[key] = _FailureBackoff(
            failures,
            self._monotonic() + delay,
        )

    def _trust_reservation(self, reservation: ReservationIdentity) -> None:
        self._trusted_reservations.pop(reservation, None)
        self._trusted_reservations[reservation] = None
        self._trim_trusted_reservations()

    def _trim_trusted_reservations(self) -> None:
        while len(self._trusted_reservations) > _MAX_RETAINED_RESERVATION_RECONCILIATIONS:
            self._trusted_reservations.pop(next(iter(self._trusted_reservations)))

    @staticmethod
    def _cursor_commit_service_key(
        identity: _ValidationIdentity,
        lane: str,
        admission_role: str,
    ) -> tuple[_ValidationIdentity, str, str, str]:
        return (identity, "scheduler_cursor_commit", lane, admission_role)

    @staticmethod
    def _cursor_namespace_service(
        project_id: str,
        namespace: object,
        lane: str,
        admission_role: str,
    ) -> tuple[str, str] | None:
        if namespace != f"scheduler-{project_id}-{admission_role}-{lane}":
            return None
        return (lane, admission_role)

    @staticmethod
    def _validate_empty_observation(
        project_id: object,
        evidence: object,
        lane: str,
        admission_role: str,
    ) -> None:
        if not isinstance(project_id, str) or not project_id or not isinstance(evidence, Mapping):
            raise ValueError("empty scheduler observations must map nonempty project IDs to evidence objects.")
        if set(evidence) != {"outcome", "reason", "source_revisions", "candidate", "cursor"}:
            raise ValueError("empty scheduler observation has missing or unknown fields.")
        if evidence["outcome"] != "none" or evidence["candidate"] is not None:
            raise ValueError("cursor commits require an empty scheduler observation.")
        reasons = {
            "no_candidate",
            "ready_index_inactive",
            "candidate_unresolved",
            "dependency_not_ready",
            "placement_rejected",
            "admission_role_mismatch",
            "borrow_admission_unavailable",
            "slice_exhausted",
        }
        if evidence["reason"] not in reasons:
            raise ValueError("empty scheduler observation reason is invalid.")

        source_revisions = evidence["source_revisions"]
        if (
            not isinstance(source_revisions, Mapping)
            or set(source_revisions) != {"ready_index"}
            or type(source_revisions["ready_index"]) is not int
            or source_revisions["ready_index"] < 0
        ):
            raise ValueError("empty scheduler observation ready-index revision is invalid.")

        cursor = evidence["cursor"]
        expected_namespace = f"scheduler-{project_id}-{admission_role}-{lane}"
        if (
            not isinstance(cursor, Mapping)
            or set(cursor) != {"namespace", "routes"}
            or cursor["namespace"] != expected_namespace
            or not isinstance(cursor["routes"], Mapping)
            or set(cursor["routes"]) != {"home", "shared"}
        ):
            raise ValueError("empty scheduler observation cursor namespace or routes are invalid.")

        for scope in ("home", "shared"):
            route = cursor["routes"][scope]
            if not isinstance(route, Mapping) or set(route) != {"observed", "next"}:
                raise ValueError("empty scheduler observation cursor route is malformed.")
            validate_ready_cursor_transition(route["observed"], route["next"], f"cursor.routes.{scope}")

    @staticmethod
    def _validate_cursor_commit_result(evidence: object) -> None:
        if not isinstance(evidence, Mapping) or set(evidence) != {"routes"}:
            raise ProjectIOProtocolError("cursor commit result evidence is malformed.")
        routes = evidence["routes"]
        if not isinstance(routes, Mapping) or set(routes) != {"home", "shared"}:
            raise ProjectIOProtocolError("cursor commit result routes are malformed.")
        if any(routes[scope] not in {"committed", "already_applied", "stale"} for scope in ("home", "shared")):
            raise ProjectIOProtocolError("cursor commit result route outcome is invalid.")

    def _start_cursor_commit_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        if request.executor_epoch != self._observed_executor_epoch:
            self._record_service_failure(service_key)
            return
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    def _start_observation_request(
        self,
        request: ProjectIORequest,
        service_key: tuple[_ValidationIdentity, str, str, str],
    ) -> None:
        try:
            process = self.executor.start(request.request_id)
        except (OSError, RuntimeError, ValueError):
            self._record_service_failure(service_key)
            return
        if process is None:
            self._record_service_failure(service_key)
            return
        self._started_requests.add(request.request_id)
        self._service_backoff.pop(service_key, None)

    def _root_config(self, binding: ProjectBinding) -> RootConfig:
        return RootConfig.from_canonical_paths(
            binding.shared_root,
            binding.shared_root.parent,
            binding.machine_name,
            self.runtime.project_paths(binding.project_id)["root"],
        )
