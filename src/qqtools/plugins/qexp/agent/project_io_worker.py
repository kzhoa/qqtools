"""Fresh-interpreter worker for the closed Project I/O operation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config_types import RootConfig
from ..domain.policies import task_machine_matches
from ..events import flush_local_event
from ..layout import (
    load_machine_record,
    load_machine_registration,
    load_root_config,
    machine_state_path,
    validate_root_contract,
)
from ..lease import load_lease_policy
from ..machine_config import load_machine_policy
from ..runtime.authority_lock import authority_locks
from ..runtime.paths import attempt_path, local_paths, machine_project_paths, machine_runtime_paths, shared_paths
from ..runtime.ready import advance_ready_index_build
from ..runtime.ready import routes as ready_routes
from ..runtime.ready import state as ready_state
from ..runtime.ready.group_members import is_group_ready_member_projection_usable
from ..runtime.ready.index import classify_ready_marker
from ..runtime.ready.records import ReadyMarkerRef
from ..runtime.ready.traversal import ReadyCursor, compare_and_commit_ready_cursor, load_ready_cursor, peek_ready_marker
from ..runtime.records import AttemptRecord, utc_now
from ..runtime.registration_authority import RegistrationIdentity, registration_state, registration_write_guard
from ..runtime.resources.reservations import ReservationIdentity
from ..runtime.store import atomic_replace, fenced_mutations, read_json, read_json_limited, require_json_size
from ..runtime.tasks import load_task
from ..runtime.work_budget import SliceBudget, WorkBudgetPolicy
from .bindings import ProjectBinding, decode_registry
from .primary_probe_transport import decode_probe_session, encode_probe_session
from .project_io_protocol import (
    PROJECT_IO_MAX_RECORD_BYTES,
    PROJECT_IO_PROTOCOL_VERSION,
    ProjectIOEpoch,
    ProjectIOProcess,
    ProjectIORequest,
    ProjectIOResult,
    authority_terminal_transition_digest,
)
from .project_io_transport_support import ProjectIOProtocolError, _process_start_time_ticks, _resolve_runtime_id

_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_HANDSHAKE_SECONDS = 2.0
_HANDSHAKE_POLL_SECONDS = 0.025
_MAX_SHARED_RECORD_BYTES = 8 * 1024 * 1024


class _ExecutorEpochFenced(RuntimeError):
    """Raised when a worker loses its local executor mutation fence."""


class _BindingAuthorityChanged(RuntimeError):
    """Raised when binding authority changes across a claim mutation window."""


class _RecoveryPreparationDeferred(RuntimeError):
    """Local rollback or migration has not yet established recovery eligibility."""


class _AuthorityReplayBlocked(RuntimeError):
    """Raised when a stale mutation has no exact persisted target to finish."""


def _registration_is_eligible(registration: dict[str, Any]) -> bool:
    try:
        expires_at = datetime.fromisoformat(registration["eligibility_expires_at"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return registration.get("state") == "eligible" and expires_at > datetime.now(timezone.utc)


@contextmanager
def _bounded_project_reads(shared_root: Path) -> Iterator[None]:
    """Reject oversized Project records before any legacy JSON reader parses them."""
    original_open = Path.open

    def bounded_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any):
        handle = original_open(path, mode, *args, **kwargs)
        if "r" not in mode or any(flag in mode for flag in ("w", "a", "+")):
            return handle
        try:
            path.relative_to(shared_root)
        except ValueError:
            return handle
        try:
            size = os.fstat(handle.fileno()).st_size
        except Exception:
            handle.close()
            raise
        if size > _MAX_SHARED_RECORD_BYTES:
            handle.close()
            raise ValueError("Project JSON record exceeds the isolated worker read limit.")
        return handle

    Path.open = bounded_open
    try:
        yield
    finally:
        Path.open = original_open


def _read_request(paths: dict[str, Path], request_id: str) -> ProjectIORequest:
    path = paths["project_io_requests"] / f"{request_id}.json"
    return ProjectIORequest.from_dict(
        read_json_limited(path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type="project_io_request")
    )


def _read_process(paths: dict[str, Path], request_id: str) -> ProjectIOProcess:
    path = paths["project_io_processes"] / f"{request_id}.json"
    return ProjectIOProcess.from_dict(
        read_json_limited(path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type="project_io_process")
    )


def _read_epoch(paths: dict[str, Path]) -> ProjectIOEpoch:
    return ProjectIOEpoch.from_dict(
        read_json_limited(
            paths["project_io_epoch"], max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type="project_io_epoch"
        )
    )


def _validate_worker_record_paths(paths: dict[str, Path], request_id: str) -> None:
    root = paths["project_io_root"]
    for directory in (
        root,
        paths["project_io_requests"],
        paths["project_io_processes"],
    ):
        metadata = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError("executor handshake directory is not locally owned.")
    for lane in ("project_io_requests", "project_io_processes"):
        path = paths[lane] / f"{request_id}.json"
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise ProjectIOProtocolError("executor handshake record is not locally owned.")


def _matching_handshake(
    paths: dict[str, Path], request_id: str, runtime_id: str, pid: int, ticks: int
) -> tuple[ProjectIORequest, ProjectIOProcess, ProjectIOEpoch] | None:
    request = _read_request(paths, request_id)
    process = _read_process(paths, request_id)
    epoch = _read_epoch(paths)
    if (
        request.request_id != request_id
        or request.runtime_id != runtime_id
        or request.protocol_version != PROJECT_IO_PROTOCOL_VERSION
        or process.request != request
        or process.pid != pid
        or process.start_time_ticks != ticks
        or process.state != "running"
        or epoch.runtime_id != runtime_id
    ):
        return None
    return request, process, epoch


def _wait_for_handshake(
    paths: dict[str, Path], request_id: str, runtime_id: str, pid: int, ticks: int
) -> tuple[ProjectIORequest, ProjectIOProcess, ProjectIOEpoch] | None:
    deadline = time.monotonic() + _HANDSHAKE_SECONDS
    while time.monotonic() < deadline:
        try:
            _validate_worker_record_paths(paths, request_id)
            result = _matching_handshake(paths, request_id, runtime_id, pid, ticks)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            result = None
        if result is not None:
            return result
        time.sleep(min(_HANDSHAKE_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    return None


def _validate_binding(
    request: ProjectIORequest,
    runtime_root: Path,
    *,
    validate_registration: bool = True,
) -> dict[str, Any]:
    if request.operation_kind not in {
        "validate_binding",
        "upgrade_service",
        "submission_control_service",
        "observation_service",
        "notification_service",
        "group_service_probe",
        "group_service_advance",
        "progress_projection",
        "legacy_capture_read",
        "legacy_capture_scan",
        "recovery_source_hold",
        "recovery_admission",
        "recovery_source_release",
        "recovery_group_authority",
        "recovery_capture_transition",
        "scheduler_observe",
        "scheduler_primary_probe",
        "scheduler_quiescence_probe",
        "scheduler_claim",
        "scheduler_cursor_commit",
        "scheduler_launch_authorize",
        "scheduler_reservation_reconcile",
        "scheduler_due_offer",
        "scheduler_ready_index_build",
        "maintenance_descriptor_advance",
        "maintenance_flush_event",
        "machine_snapshot_publish",
        "registration_renew",
        "activation_observe",
        "activation_consumer_register",
        "activation_consumer_ack",
        "activation_consumer_retire",
        "authority_service",
        "authority_renewal",
        "authority_orphan_recovery",
        "authority_termination_commit",
        "authority_terminal_observe",
        "authority_terminal_publish",
        "authority_running_publish",
    }:
        raise ProjectIOProtocolError("worker operation is not allowlisted.")
    shared_root = Path(request.canonical_shared_root)
    cfg = load_root_config(
        shared_root,
        request.parameters["machine_name"],
    )
    validate_root_contract(cfg)
    project_identity = read_json_limited(
        shared_paths(shared_root)["project"] / "identity.json",
        max_bytes=PROJECT_IO_MAX_RECORD_BYTES,
        record_type="project_identity",
    )
    if set(project_identity) != {"project"} or not isinstance(project_identity["project"], dict):
        raise ValueError("Project identity record is malformed.")
    identity = project_identity["project"]
    if set(identity) != {"project_id", "shared_root"}:
        raise ValueError("Project identity fields are malformed.")
    if identity["project_id"] != request.project_id or identity["shared_root"] != request.canonical_shared_root:
        raise ValueError("Project identity does not match the request binding.")
    schema_record = read_json_limited(
        shared_paths(shared_root)["schema"] / "version.json",
        max_bytes=PROJECT_IO_MAX_RECORD_BYTES,
        record_type="project_schema",
    )
    schema = schema_record.get("schema")
    if not isinstance(schema, dict) or type(schema.get("version")) is not int or schema["version"] <= 0:
        raise ValueError("Project schema identity is malformed.")
    required_capabilities = schema.get("required_capabilities", [])
    if (
        not isinstance(required_capabilities, list)
        or len(required_capabilities) > 32
        or any(not isinstance(item, str) or not item or len(item) > 128 for item in required_capabilities)
        or len(set(required_capabilities)) != len(required_capabilities)
    ):
        raise ValueError("Project schema capabilities are malformed.")
    if validate_registration:
        registration_envelope = load_machine_registration(cfg)
        registration = registration_envelope.get("registration") if isinstance(registration_envelope, dict) else None
        if (
            not isinstance(registration, dict)
            or registration.get("project_id") != request.project_id
            or registration.get("shared_root") != request.canonical_shared_root
            or registration.get("machine_name") != request.parameters["machine_name"]
            or registration.get("generation") != request.registration_generation
            or registration.get("runtime_instance_id") != request.runtime_id
            or registration.get("runtime_root") != str(runtime_root)
            or not _registration_is_eligible(registration)
        ):
            raise ValueError("machine registration does not match the current request binding.")
    return {
        "project_id": identity["project_id"],
        "shared_root": identity["shared_root"],
        "schema_version": schema["version"],
        "required_capabilities": list(required_capabilities),
    }


def _authority_renewal_task_commit_matches_request(request: ProjectIORequest, cfg: object) -> bool:
    """Recognize only the exact Task-first commit that an old renewal may finish."""
    if request.operation_kind != "authority_renewal":
        return False
    parameters = request.parameters
    try:
        task = load_task(cfg, parameters["task_id"])
    except FileNotFoundError:
        return False
    claim = task.claim_control.get("active_claim") or {}
    task_matches = bool(
        task.attempt_control.get("current_attempt_id") == parameters["attempt_id"]
        and task.attempt_control.get("current_attempt_number") == parameters["attempt_number"]
        and claim.get("attempt_id") == parameters["attempt_id"]
        and claim.get("attempt_number") == parameters["attempt_number"]
        and claim.get("fencing_token") == parameters["fencing_token"]
        and claim.get("machine_name") == parameters["machine_name"]
        and claim.get("reservation_id") == parameters["reservation_id"]
        and claim.get("clock_observation_id") == request.request_id
    )
    if not task_matches:
        return False
    # Once Task carries this immutable marker, an unreadable companion Attempt
    # is not proof that the partial commit disappeared. Surface that failure so
    # stale reconciliation preserves the exact request and retries it.
    attempt = AttemptRecord.from_dict(
        read_json(attempt_path(cfg.shared_root, parameters["task_id"], parameters["attempt_number"]))
    )
    return bool(
        attempt.reservation_id == parameters["reservation_id"]
        and attempt.task_id == parameters["task_id"]
        and attempt.attempt_id == parameters["attempt_id"]
        and attempt.attempt_number == parameters["attempt_number"]
        and attempt.current_fencing_token == parameters["fencing_token"]
        and attempt.machine_name == parameters["machine_name"]
        and attempt.phase == "running"
        and all(
            attempt.process.get(field) == parameters["process_identity"].get(field)
            for field in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        )
    )


def _read_worker_registry(runtime_root: Path) -> tuple[int, tuple[ProjectBinding, ...]]:
    path = machine_runtime_paths(runtime_root)["registry"]
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError("machine registry must be a regular non-symlink file.")
    return decode_registry(read_json_limited(path, max_bytes=_MAX_SHARED_RECORD_BYTES, record_type="machine_registry"))


def _registry_binding_is_current(request: ProjectIORequest, runtime_root: Path) -> bool:
    """Recheck captured local ownership without shared I/O or a local lock."""
    try:
        revision, bindings = _read_worker_registry(runtime_root)
        if revision != request.registry_revision:
            return False
        matching = [
            item
            for item in bindings
            if item.project_id == request.project_id
            and str(item.shared_root) == request.canonical_shared_root
            and item.machine_name == request.parameters["machine_name"]
        ]
        if len(matching) != 1:
            return False
        binding = matching[0]
        allows_disabled_binding = request.operation_kind in {
            "upgrade_service",
            "progress_projection",
            "legacy_capture_read",
            "legacy_capture_scan",
            "recovery_source_hold",
            "recovery_admission",
            "recovery_source_release",
            "recovery_group_authority",
            "recovery_capture_transition",
            "authority_service",
            "authority_renewal",
            "authority_orphan_recovery",
            "authority_termination_commit",
            "authority_terminal_observe",
            "authority_terminal_publish",
            "authority_running_publish",
        }
        return bool(
            (binding.enabled or allows_disabled_binding)
            and binding.registration_generation == request.registration_generation
            and binding.runtime_instance_id == request.runtime_id
            and binding.runtime_root == str(runtime_root)
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _claim_binding_is_current(
    request: ProjectIORequest, runtime_root: Path, cfg: object, *, require_eligible: bool = True
) -> bool:
    """Revalidate both local binding generation and its Project registration."""
    if not _registry_binding_is_current(request, runtime_root):
        return False
    try:
        envelope = load_machine_registration(cfg) or {}
        registration = envelope.get("registration")
        return bool(
            isinstance(registration, dict)
            and registration.get("project_id") == request.project_id
            and registration.get("shared_root") == request.canonical_shared_root
            and registration.get("machine_name") == request.parameters["machine_name"]
            and registration.get("generation") == request.registration_generation
            and registration.get("runtime_instance_id") == request.runtime_id
            and registration.get("runtime_root") == str(runtime_root)
            and (
                _registration_is_eligible(registration)
                or (
                    not require_eligible
                    and registration.get("state") == "eligible"
                    and registration_state(registration) == "expired"
                )
            )
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return False


def _authority_service_observe(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Observe one Task/Attempt pair without granting reusable authority.

    The evidence is point-in-time only. Any follow-on mutation or local process
    action must revalidate the returned revisions and every authority fence in
    its own typed operation immediately before applying effects.
    """
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def evidence(
        outcome: str,
        reason: str | None,
        *,
        task_revision: int | None = None,
        attempt_digest: str | None = None,
        attempt_phase: str | None = None,
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reason": reason,
            "runtime_id": request.runtime_id,
            "executor_epoch": request.executor_epoch,
            "project_id": request.project_id,
            "canonical_shared_root": request.canonical_shared_root,
            "registration_generation": request.registration_generation,
            "registry_revision": request.registry_revision,
            "machine_name": parameters["machine_name"],
            "service_action": parameters["service_action"],
            "task_id": parameters["task_id"],
            "attempt_id": parameters["attempt_id"],
            "attempt_number": parameters["attempt_number"],
            "fencing_token": parameters["fencing_token"],
            "reservation_id": parameters["reservation_id"],
            "process_identity": dict(parameters["process_identity"]),
            "source_revisions": {"task": task_revision, "attempt_digest": attempt_digest},
            "attempt_phase": attempt_phase,
            "authority_granted": False,
            "local_effects": [],
        }

    def require_current_epoch() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced

    def load_records() -> tuple[Any | None, AttemptRecord | None, str | None]:
        try:
            task = load_task(cfg, parameters["task_id"])
        except FileNotFoundError:
            return None, None, None
        try:
            raw_attempt = attempt_path(
                cfg.shared_root,
                parameters["task_id"],
                parameters["attempt_number"],
            ).read_bytes()
            attempt_value = json.loads(raw_attempt.decode("utf-8"))
            if not isinstance(attempt_value, dict):
                raise ValueError("Attempt record must contain a JSON object.")
            attempt = AttemptRecord.from_dict(attempt_value)
            attempt_digest = hashlib.sha256(raw_attempt).hexdigest()
        except FileNotFoundError:
            attempt = None
            attempt_digest = None
        return task, attempt, attempt_digest

    def records_payload(records: tuple[Any | None, AttemptRecord | None, str | None]) -> tuple[Any, Any, str | None]:
        task, attempt, attempt_digest = records
        return (
            None if task is None else task.to_dict(),
            None if attempt is None else attempt.to_dict(),
            attempt_digest,
        )

    require_current_epoch()
    if not _claim_binding_is_current(request, runtime_root, cfg):
        return evidence("observed_stale", "binding_fence")

    try:
        initial_task = load_task(cfg, parameters["task_id"])
    except FileNotFoundError:
        first_missing = load_records()
        second_missing = load_records()
        if records_payload(first_missing) != records_payload(second_missing):
            raise RuntimeError("Task changed during authority observation.")
        initial_task = second_missing[0]
        if initial_task is None:
            require_current_epoch()
            if not _claim_binding_is_current(request, runtime_root, cfg):
                return evidence("observed_stale", "binding_fence")
            return evidence("observed_stale", "task_changed")
    with authority_locks(cfg, initial_task):
        first = load_records()
        if first[0] is not None and first[0].depends_on_task_ids != initial_task.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while authority locks were acquired.")
        second = load_records()
        if records_payload(first) != records_payload(second):
            raise RuntimeError("Task or Attempt changed during authority observation.")
        task, attempt, attempt_digest = second
        task_revision = None if task is None else task.meta.get("revision")
        if task_revision is not None and type(task_revision) is not int:
            raise ValueError("Task revision is malformed.")

        stale_reason: str | None = None
        if task is None:
            stale_reason = "task_changed"
        else:
            claim = task.claim_control.get("active_claim") or {}
            if (
                task.attempt_control.get("current_attempt_id") != parameters["attempt_id"]
                or task.attempt_control.get("current_attempt_number") != parameters["attempt_number"]
                or claim.get("attempt_id") != parameters["attempt_id"]
                or claim.get("attempt_number") != parameters["attempt_number"]
                or claim.get("fencing_token") != parameters["fencing_token"]
                or claim.get("machine_name") != parameters["machine_name"]
                or (attempt is not None and attempt.machine_name != parameters["machine_name"])
            ):
                stale_reason = "task_changed"
            elif (
                attempt is None
                or attempt.task_id != parameters["task_id"]
                or attempt.attempt_id != parameters["attempt_id"]
                or attempt.attempt_number != parameters["attempt_number"]
                or attempt.current_fencing_token != parameters["fencing_token"]
            ):
                stale_reason = "attempt_changed"
            elif (
                parameters["reservation_id"] is not None and attempt.reservation_id != parameters["reservation_id"]
            ) or claim.get("reservation_id") != attempt.reservation_id:
                stale_reason = "reservation_changed"
            elif any(
                attempt.process.get(field) != parameters["process_identity"][field]
                for field in (
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                )
            ):
                stale_reason = "process_changed"
            elif attempt.phase != "running":
                stale_reason = "attempt_not_running"
            elif (
                task.control.get("terminate_running") is True
                or task.control.get("cancellation_requested_at") is not None
                or attempt.termination.get("requested_at") is not None
                or attempt.termination.get("requested_by_operation_id") is not None
            ):
                stale_reason = "termination_pending"

        require_current_epoch()
        if not _claim_binding_is_current(request, runtime_root, cfg):
            return evidence(
                "observed_stale",
                "binding_fence",
                task_revision=task_revision,
                attempt_digest=attempt_digest,
            )
        if stale_reason is not None:
            return evidence(
                "observed_stale",
                stale_reason,
                task_revision=task_revision,
                attempt_digest=attempt_digest,
                attempt_phase=None if attempt is None else attempt.phase,
            )
        return evidence(
            "observed_current",
            None,
            task_revision=task_revision,
            attempt_digest=attempt_digest,
            attempt_phase=attempt.phase if attempt is not None else None,
        )


def _authority_renewal(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
    *,
    replay_only: bool = False,
    active_executor_epoch: str | None = None,
) -> dict[str, Any]:
    """Commit one replayable renewal without performing machine-local effects."""
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def mutation_fence() -> None:
        epoch = _read_epoch(paths)
        expected_epoch = active_executor_epoch if replay_only else request.executor_epoch
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != expected_epoch:
            raise _ExecutorEpochFenced
        if replay_only:
            if not _authority_renewal_task_commit_matches_request(request, cfg):
                raise _BindingAuthorityChanged
        elif not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    from ..scheduler import renew_project_io_attempt_lease

    result = renew_project_io_attempt_lease(
        cfg,
        request_id=request.request_id,
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        expected_task_revision=request.source_revisions["task"],
        expected_attempt_digest=request.source_revisions["attempt_digest"],
        mutation_fence=mutation_fence,
        replay_only=replay_only,
    )
    return {
        **result,
        "machine_name": parameters["machine_name"],
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "attempt_number": parameters["attempt_number"],
        "fencing_token": parameters["fencing_token"],
        "reservation_id": parameters["reservation_id"],
        "process_identity": dict(parameters["process_identity"]),
        "source_revisions": dict(request.source_revisions),
        "authority_granted": False,
        "local_effects": [],
    }


def _authority_terminal_publish_partial_target_matches(
    request: ProjectIORequest,
    cfg: object,
) -> bool:
    """Recognize the exact Attempt-first or fully committed terminal target."""
    parameters = request.parameters
    try:
        task = load_task(cfg, parameters["task_id"])
    except FileNotFoundError:
        return False
    try:
        attempt = AttemptRecord.from_dict(
            read_json(attempt_path(cfg.shared_root, parameters["task_id"], parameters["attempt_number"]))
        )
    except FileNotFoundError:
        return False
    if (
        attempt.task_id != parameters["task_id"]
        or attempt.attempt_id != parameters["attempt_id"]
        or attempt.attempt_number != parameters["attempt_number"]
        or attempt.current_fencing_token != parameters["fencing_token"]
        or attempt.machine_name != parameters["machine_name"]
        or attempt.reservation_id != parameters["reservation_id"]
        or any(
            attempt.process.get(field) != parameters["process_identity"].get(field)
            for field in parameters["process_identity"]
        )
        or attempt.phase != parameters["phase"]
        or attempt.result.get("reason") != parameters["reason"]
        or attempt.result.get("exit_code") != parameters["exit_code"]
        or attempt.termination.get("result") != parameters["termination_result"]
        or attempt.termination.get("project_io_publication_id") != request.request_id
    ):
        return False
    task_phase = task.state.get("projection")
    active_claim = task.claim_control.get("active_claim") or {}
    if (
        task_phase == parameters["phase"]
        and task.state.get("reason") == parameters["reason"]
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == parameters["attempt_number"]
        and not active_claim
        and task.control.get("termination_result") == parameters["termination_result"]
        and task.control.get("project_io_publication_id") == request.request_id
    ):
        return True
    if parameters["mode"] == "active":
        return bool(
            task_phase == "running"
            and task.attempt_control.get("current_attempt_id") == parameters["attempt_id"]
            and task.attempt_control.get("current_attempt_number") == parameters["attempt_number"]
            and active_claim.get("attempt_id") == parameters["attempt_id"]
            and active_claim.get("attempt_number") == parameters["attempt_number"]
            and active_claim.get("fencing_token") == parameters["fencing_token"]
            and active_claim.get("machine_name") == parameters["machine_name"]
            and active_claim.get("reservation_id") == parameters["reservation_id"]
        )
    return bool(
        parameters["mode"] == "detached_orphan"
        and task_phase == "blocked"
        and not active_claim
        and task.attempt_control.get("current_attempt_id") is None
        and task.attempt_control.get("current_attempt_number") == parameters["attempt_number"]
        and task.attempt_control.get("next_attempt_number") == parameters["attempt_number"] + 1
    )


def _authority_orphan_recovery(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
    *,
    replay_only: bool = False,
    active_executor_epoch: str | None = None,
) -> dict[str, Any]:
    """Commit shared orphan recovery; MachineRuntime effects stay in controller."""
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def mutation_fence() -> None:
        epoch = _read_epoch(paths)
        expected_epoch = active_executor_epoch if replay_only else request.executor_epoch
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != expected_epoch:
            raise _ExecutorEpochFenced
        if not replay_only and not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    from ..scheduler import recover_project_io_orphaned_attempt

    source = request.source_revisions
    result = recover_project_io_orphaned_attempt(
        cfg,
        machine_name=parameters["machine_name"],
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        binding_signature=parameters["binding_signature"],
        expected_task_revision=source.get("task"),
        expected_attempt_digest=source.get("attempt_digest"),
        mutation_fence=mutation_fence,
        replay_only=replay_only or parameters["replay_only"],
    )
    if not replay_only:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
    return result


def _authority_termination_commit(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
    *,
    replay_only: bool = False,
    active_executor_epoch: str | None = None,
) -> dict[str, Any]:
    """Commit the exact Task termination marker without signaling processes."""
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def mutation_fence() -> None:
        epoch = _read_epoch(paths)
        expected_epoch = active_executor_epoch if replay_only else request.executor_epoch
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != expected_epoch:
            raise _ExecutorEpochFenced
        if replay_only:
            # An exact Task marker returns before the first write boundary.
            raise _AuthorityReplayBlocked
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    from ..scheduler import commit_project_io_shared_termination

    result = commit_project_io_shared_termination(
        cfg,
        machine_name=parameters["machine_name"],
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
        expected_task_revision=request.source_revisions["task"],
        expected_attempt_digest=request.source_revisions["attempt_digest"],
        mutation_fence=mutation_fence,
    )
    if not replay_only:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
    return result


def _authority_terminal_observe(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Observe one active or orphaned Attempt without granting reusable authority."""
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def require_current_epoch() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced

    require_current_epoch()
    from ..scheduler import observe_project_io_terminal_state

    evidence = observe_project_io_terminal_state(
        cfg,
        machine_name=parameters["machine_name"],
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        mode=parameters["mode"],
    )
    if not _claim_binding_is_current(request, runtime_root, cfg):
        evidence.update({"outcome": "stale", "reason": "binding_fence"})
    require_current_epoch()
    return evidence


def _authority_terminal_publish(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
    *,
    replay_only: bool = False,
    active_executor_epoch: str | None = None,
) -> dict[str, Any]:
    """Publish a canonical terminal transition through shared domain helpers."""
    parameters = request.parameters
    computed_digest = authority_terminal_transition_digest(
        mode=parameters["mode"],
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=parameters["machine_name"],
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=parameters["phase"],
        reason=parameters["reason"],
        exit_code=parameters["exit_code"],
        termination_result=parameters["termination_result"],
    )
    if parameters["transition_digest"] != computed_digest:
        raise ProjectIOProtocolError("terminal transition digest differs from canonical semantic fields.")
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def mutation_fence() -> None:
        epoch = _read_epoch(paths)
        expected_epoch = active_executor_epoch if replay_only else request.executor_epoch
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != expected_epoch:
            raise _ExecutorEpochFenced
        if replay_only:
            if not _authority_terminal_publish_partial_target_matches(request, cfg):
                raise _AuthorityReplayBlocked
        elif not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    from ..scheduler import publish_project_io_terminal_transition

    result = publish_project_io_terminal_transition(
        cfg,
        request_id=request.request_id,
        machine_name=parameters["machine_name"],
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
        expected_task_revision=request.source_revisions["task"],
        expected_attempt_digest=request.source_revisions["attempt_digest"],
        transition_digest=parameters["transition_digest"],
        mutation_fence=mutation_fence,
    )
    if result.get("outcome") in {"committed", "already_committed"} and result.get("lifecycle_event") is not None:
        from ..lifecycle import TaskLifecycleEvent, dispatch_task_lifecycle_hooks_noexcept
        from ..notifications import notification_delivery_fence, notification_runtime
        from ..runtime.store import fenced_mutations

        def notification_fence() -> None:
            mutation_fence()
            # Exact old terminal recovery does not authorize a new external
            # side effect after its registration or registry revision changed.
            if not _claim_binding_is_current(request, runtime_root, cfg):
                raise _BindingAuthorityChanged

        def notification_local_fence() -> None:
            epoch = _read_epoch(paths)
            expected_epoch = active_executor_epoch if replay_only else request.executor_epoch
            if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != expected_epoch:
                raise _ExecutorEpochFenced
            if not _registry_binding_is_current(request, runtime_root):
                raise _BindingAuthorityChanged
            write_possible[0] = True

        try:
            with (
                notification_runtime(
                    runtime_root,
                    project_id=request.project_id,
                    before_shared_capture=notification_fence,
                    before_local_write=notification_local_fence,
                ),
                notification_delivery_fence(notification_fence),
                fenced_mutations(cfg.shared_root, notification_fence),
            ):
                notification_fence()
                dispatch_task_lifecycle_hooks_noexcept(cfg, TaskLifecycleEvent(**result["lifecycle_event"]))
        except _BindingAuthorityChanged:
            if not replay_only:
                raise
            # Recovery has proven the old shared transaction complete. Skipping
            # an obsolete notification must not make that proof ambiguous again.
    if not replay_only:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
    return result


def _authority_running_publish(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Publish the exact registered process as running through shared domain logic."""
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])
    manifest = local_paths(cfg.runtime_root)["processes"] / f"{parameters['attempt_id']}.json"
    registration = {
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "fencing_token": parameters["fencing_token"],
        "reservation_id": parameters["reservation_id"],
        **dict(parameters["process_identity"]),
        "process_created_at": parameters["process_created_at"],
    }

    def require_current() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    def mutation_fence() -> None:
        require_current()
        write_possible[0] = True

    require_current()
    from ..runtime.running_publication import publish_running_registration

    transitioned_to_running = publish_running_registration(
        cfg,
        registration,
        manifest,
        mutation_fence=mutation_fence,
    )
    require_current()
    return {
        "machine_name": parameters["machine_name"],
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "attempt_number": parameters["attempt_number"],
        "fencing_token": parameters["fencing_token"],
        "reservation_id": parameters["reservation_id"],
        "process_identity": dict(parameters["process_identity"]),
        "process_created_at": parameters["process_created_at"],
        "outcome": "processed",
        "reason": None,
        "transitioned_to_running": transitioned_to_running,
        "authority_granted": False,
        "local_effects": [],
    }


def _authority_replay_blocked_evidence(request: ProjectIORequest) -> dict[str, Any]:
    """Return bounded stale evidence when an old epoch has no target to finish."""
    parameters = request.parameters
    revisions = dict(request.source_revisions)
    if request.operation_kind == "authority_termination_commit":
        return {
            "outcome": "stale",
            "reason": "executor_epoch_fence",
            "machine_name": parameters["machine_name"],
            "task_id": parameters["task_id"],
            "attempt_id": parameters["attempt_id"],
            "attempt_number": parameters["attempt_number"],
            "fencing_token": parameters["fencing_token"],
            "reservation_id": parameters["reservation_id"],
            "decision_id": parameters["decision_id"],
            "decision_token": parameters["decision_token"],
            "authority_outcome": parameters["authority_outcome"],
            "decision_reason": parameters["reason"],
            "process_identity": dict(parameters["process_identity"]),
            "source_revisions": revisions,
            "committed_revisions": revisions,
            "shared_commitment": None,
            "authority_granted": False,
            "local_effects": [],
        }
    return {
        "outcome": "stale",
        "reason": "executor_epoch_fence",
        "machine_name": parameters["machine_name"],
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "attempt_number": parameters["attempt_number"],
        "fencing_token": parameters["fencing_token"],
        "reservation_id": parameters["reservation_id"],
        "reservation_machine_name": parameters["machine_name"],
        "process_identity": dict(parameters["process_identity"]),
        "mode": parameters["mode"],
        "phase": parameters["phase"],
        "transition_reason": parameters["reason"],
        "exit_code": parameters["exit_code"],
        "termination_result": parameters["termination_result"],
        "source_revisions": revisions,
        "committed_revisions": revisions,
        "lifecycle_event": None,
        "transition_digest": parameters["transition_digest"],
        "authority_granted": False,
        "local_effects": [],
    }


def _scheduler_claim(request: ProjectIORequest, runtime_root: Path) -> dict[str, Any]:
    parameters = request.parameters
    candidate = parameters["candidate"]
    offer = parameters["offer"]
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])
    committed = _committed_claim_evidence(cfg, candidate, offer)
    if committed is not None:
        return committed
    if not _claim_binding_is_current(request, runtime_root, cfg):
        return {"outcome": "no_claim", "reason": "candidate_stale"}
    from ..scheduler import claim_project_io_candidate

    def require_current_authority() -> None:
        epoch = _read_epoch(machine_runtime_paths(runtime_root))
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    return claim_project_io_candidate(
        cfg,
        candidate,
        offer,
        project_id=request.project_id,
        registration_generation=request.registration_generation,
        admission_role=parameters["admission_role"],
        source_revisions=dict(request.source_revisions),
        mutation_fence=require_current_authority,
    )


def _scheduler_launch_authorize(
    request: ProjectIORequest,
    runtime_root: Path,
    mutation_fence: Any,
) -> dict[str, Any]:
    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])
    from ..launch_policy import resolve_launch_handoff_policy, validate_launch_handoff_timeout_seconds

    policy = resolve_launch_handoff_policy(cfg.shared_root)
    timeout_seconds = validate_launch_handoff_timeout_seconds(policy["timeout_seconds"])
    from ..scheduler import authorize_project_io_launch

    return authorize_project_io_launch(
        cfg,
        parameters["claim_identity"],
        launch_handoff_timeout_seconds=timeout_seconds,
        mutation_fence=mutation_fence,
    )


def _scheduler_reservation_reconcile(request: ProjectIORequest) -> dict[str, Any]:
    parameters = request.parameters
    value = parameters["reservation_identity"]
    identity = ReservationIdentity(
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
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    from ..project_maintenance import classify_reservation

    result = classify_reservation(cfg, identity)
    return {
        "outcome": result.outcome,
        "reservation_identity": dict(value),
        "reason": result.reason,
        "target_attempt_id": result.attempt_id,
        "target_fencing_token": result.fencing_token,
    }


def _maintenance_flush_event(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    parameters = request.parameters
    machine_name = parameters["machine_name"]
    cfg = load_root_config(Path(request.canonical_shared_root), machine_name)
    project_runtime_root = machine_project_paths(runtime_root, request.project_id)["root"]
    cfg = replace(cfg, runtime_root=project_runtime_root)

    def before_shared_write() -> bool:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            return False
        write_possible[0] = True
        return True

    return flush_local_event(
        cfg,
        bucket=parameters["bucket"],
        filename=parameters["filename"],
        event_id=parameters["event_id"],
        sha256=parameters["sha256"],
        machine_name=machine_name,
        before_shared_write=before_shared_write,
    )


def _machine_snapshot_publish(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    from ..machine_state import publish_shared_machine_snapshots, publish_shared_machine_stop_snapshot

    # Protocol records freeze nested objects. Snapshot writers require ordinary
    # JSON containers, including reservation admission and policy warnings.
    parameters = request.to_dict()["project_io_request"]["parameters"]
    machine_name = parameters["machine_name"]
    cfg = load_root_config(Path(request.canonical_shared_root), machine_name)
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    @contextmanager
    def shared_write_guard() -> Iterator[bool]:
        with _request_registration_guard(
            request,
            runtime_root,
            paths,
            cfg,
            write_possible,
            renewal_horizon_seconds=float(parameters["heartbeat_interval_seconds"]) * 2,
        ) as registration:
            yield registration is not None

    summaries = [dict(item) for item in parameters["reservation_summaries"]]
    reserved_gpu_ids = list(parameters["reserved_gpu_ids"])
    active_attempt_ids = sorted({item["attempt_id"] for item in summaries if isinstance(item.get("attempt_id"), str)})
    idle_since_at = None
    if not reserved_gpu_ids:
        idle_since_at = parameters["snapshot_at"]
        try:
            previous = read_json(machine_state_path(cfg, "agent.json")).get("agent", {})
        except (FileNotFoundError, OSError, ValueError, TypeError):
            previous = {}
        if previous.get("instance_id") == parameters["instance_id"] and previous.get("observed_state") == "idle":
            previous_idle_since = previous.get("idle_since_at")
            if isinstance(previous_idle_since, str):
                idle_since_at = previous_idle_since

    gpu_policy = dict(parameters["gpu_policy"])
    if "visible_status" in gpu_policy:
        if gpu_policy.get("visible_status") == "pending":
            gpu_policy["reserved_gpu_ids"] = reserved_gpu_ids
            gpu_policy["unreserved_gpu_ids"] = None
            gpu_policy["draining_gpu_ids"] = []
        else:
            policy_visible = gpu_policy.get("visible_gpu_ids")
            if policy_visible is None:
                visible = set()
            elif (
                isinstance(policy_visible, Sequence)
                and not isinstance(policy_visible, str | bytes)
                and all(type(gpu_id) is int and gpu_id >= 0 for gpu_id in policy_visible)
                and len(set(policy_visible)) == len(policy_visible)
            ):
                visible = set(policy_visible)
            else:
                raise ValueError("gpu_policy visible_gpu_ids is invalid.")
            reserved = set(reserved_gpu_ids)
            gpu_policy["reserved_gpu_ids"] = reserved_gpu_ids
            gpu_policy["unreserved_gpu_ids"] = sorted(visible - reserved)
            gpu_policy["draining_gpu_ids"] = sorted(reserved - visible)

    stop_reason = parameters["stop_reason"]
    if stop_reason is None:
        published = publish_shared_machine_snapshots(
            cfg,
            instance_id=parameters["instance_id"],
            pid=parameters["pid"],
            agent_mode=load_machine_policy(cfg).agent_mode,
            observed_state="active" if reserved_gpu_ids else "idle",
            active_attempt_ids=active_attempt_ids,
            visible_gpu_ids=parameters["visible_gpu_ids"],
            reserved_gpu_ids=reserved_gpu_ids,
            heartbeat_interval_seconds=parameters["heartbeat_interval_seconds"],
            started_at=parameters["started_at"],
            idle_since_at=idle_since_at,
            reservation_summaries=summaries,
            gpu_policy=gpu_policy,
            observed_at=parameters["snapshot_at"],
            shared_write_guard=shared_write_guard,
        )
    else:
        published = publish_shared_machine_stop_snapshot(
            cfg,
            instance_id=parameters["instance_id"],
            pid=parameters["pid"],
            agent_mode=load_machine_policy(cfg).agent_mode,
            visible_gpu_ids=parameters["visible_gpu_ids"],
            reserved_gpu_ids=reserved_gpu_ids,
            heartbeat_interval_seconds=parameters["heartbeat_interval_seconds"],
            started_at=parameters["started_at"],
            idle_since_at=None if reserved_gpu_ids else parameters["snapshot_at"],
            stop_reason=stop_reason,
            gpu_policy=gpu_policy,
            shared_write_guard=shared_write_guard,
        )
    if not published:
        return {"outcome": "stale", "snapshot_at": parameters["snapshot_at"], "reason": "binding_fence"}
    return {"outcome": "published", "snapshot_at": parameters["snapshot_at"], "reason": None}


@contextmanager
def _request_registration_guard(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    cfg: Any,
    write_possible: list[bool],
    *,
    renewal_horizon_seconds: float = 0.0,
    allow_reactivation: bool = False,
) -> Iterator[dict | None]:
    """Fence shared registration work without holding any MachineRuntime lock."""
    identity = RegistrationIdentity(
        request.project_id, request.registration_generation, request.runtime_id, str(runtime_root)
    )

    def is_current() -> bool:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        return _claim_binding_is_current(request, runtime_root, cfg, require_eligible=not allow_reactivation)

    def before_shared_write() -> None:
        if not is_current():
            raise _BindingAuthorityChanged
        write_possible[0] = True

    with fenced_mutations(cfg.shared_root, before_shared_write):
        with registration_write_guard(
            cfg,
            identity,
            is_current=is_current,
            renewal_horizon_seconds=renewal_horizon_seconds,
            before_shared_write=before_shared_write,
            allow_reactivation=allow_reactivation,
        ) as registration:
            yield registration


def _registration_renew(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Renew write eligibility for one exact current registration binding."""
    parameters = request.parameters
    machine_name = parameters["machine_name"]
    cfg = load_root_config(Path(request.canonical_shared_root), machine_name)

    def stale_evidence() -> dict[str, Any]:
        return {
            "outcome": "stale",
            "renewed": False,
            "eligibility_expires_at": None,
            "renew_after_seconds": None,
            "reason": "binding_fence",
        }

    with _request_registration_guard(
        request,
        runtime_root,
        paths,
        cfg,
        write_possible,
        renewal_horizon_seconds=parameters["renewal_horizon_seconds"],
        allow_reactivation=True,
    ) as registration:
        if registration is None:
            if write_possible[0]:
                raise RuntimeError("registration renewal lost eligibility after its shared write began.")
            return stale_evidence()
        policy = load_lease_policy(cfg)
        remaining = max(
            0.0,
            (
                datetime.fromisoformat(registration["eligibility_expires_at"].replace("Z", "+00:00"))
                - datetime.now(timezone.utc)
            ).total_seconds(),
        )
        return {
            "outcome": "eligible",
            "renewed": write_possible[0],
            "eligibility_expires_at": registration["eligibility_expires_at"],
            "renew_after_seconds": min(policy.renew_interval_seconds, policy.ttl_seconds / 2, remaining / 4),
            "reason": None,
        }


def _activation_checkpoint_observe(request: ProjectIORequest) -> dict[str, Any]:
    from ..runtime.project_activation import observe_project_activation

    return observe_project_activation(
        Path(request.canonical_shared_root),
        replay_epoch=request.parameters["replay_epoch"],
        replay_sequence=request.parameters["replay_sequence"],
    )


def _activation_consumer_mutation(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Commit one fenced consumer mutation for the exact current machine binding."""
    from ..runtime.project_activation_consumers import ack_consumer, register_consumer

    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])

    def stale_evidence() -> dict[str, Any]:
        return {"outcome": "stale", "acknowledgement": None, "reason": "binding_fence"}

    with _request_registration_guard(request, runtime_root, paths, cfg, write_possible) as registration:
        if registration is None:
            if write_possible[0]:
                raise _BindingAuthorityChanged
            return stale_evidence()

        def before_shared_write() -> None:
            epoch = _read_epoch(paths)
            if (
                not epoch.active
                or epoch.runtime_id != request.runtime_id
                or epoch.executor_epoch != request.executor_epoch
            ):
                raise _ExecutorEpochFenced
            if not _claim_binding_is_current(request, runtime_root, cfg):
                raise _BindingAuthorityChanged
            write_possible[0] = True

        if request.operation_kind == "activation_consumer_register":
            record = register_consumer(
                Path(request.canonical_shared_root),
                runtime_id=request.runtime_id,
                project_id=request.project_id,
                registration_generation=request.registration_generation,
                process_fence=parameters["process_fence"],
                before_shared_write=before_shared_write,
            )
            acknowledgement = record["project_activation_consumer"]["ack"]
            return {"outcome": "registered", "acknowledgement": acknowledgement}
        record = ack_consumer(
            Path(request.canonical_shared_root),
            runtime_id=request.runtime_id,
            project_id=request.project_id,
            registration_generation=request.registration_generation,
            process_fence=parameters["process_fence"],
            epoch=parameters["epoch"],
            sequence=parameters["sequence"],
            reconstructed_floor=parameters["reconstructed_floor"],
            require_current=parameters["require_current"],
            before_shared_write=before_shared_write,
        )
        acknowledgement = record["project_activation_consumer"]["ack"]
        return {"outcome": "acknowledged", "acknowledgement": acknowledgement}


def _activation_consumer_retirement(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Retire one exact generation after proving its durable local removal."""
    from ..runtime.project_activation_consumers import retire_consumer_after_registration_removal

    def require_current_epoch() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced

    def binding_is_exact(binding: ProjectBinding) -> bool:
        return (
            binding.runtime_instance_id == request.runtime_id
            and binding.project_id == request.project_id
            and binding.registration_generation == request.registration_generation
            and str(binding.shared_root) == request.canonical_shared_root
            and binding.machine_name == request.parameters["machine_name"]
            and binding.runtime_root == str(runtime_root)
        )

    def require_absent() -> bool:
        require_current_epoch()
        revision, bindings = _read_worker_registry(runtime_root)
        if revision < request.registry_revision:
            raise RuntimeError("machine registry revision moved backwards during consumer retirement")
        return not any(binding_is_exact(binding) for binding in bindings)

    if not require_absent():
        return {"outcome": "stale", "consumer_existed": False}

    def before_shared_write() -> None:
        if not require_absent():
            raise _BindingAuthorityChanged
        write_possible[0] = True

    record = retire_consumer_after_registration_removal(
        Path(request.canonical_shared_root),
        runtime_id=request.runtime_id,
        project_id=request.project_id,
        registration_generation=request.registration_generation,
        before_shared_write=before_shared_write,
    )
    require_current_epoch()
    return {"outcome": "retired", "consumer_existed": record is not None}


def _committed_claim_evidence(cfg: object, candidate: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any] | None:
    """Recover only an exact claim that still owns current execution authority."""
    initial = load_task(cfg, candidate["task_id"])
    with authority_locks(cfg, initial):
        task = load_task(cfg, candidate["task_id"])
        claim = task.claim_control.get("active_claim") or {}
        owns_current_authority = (
            claim.get("attempt_id") == candidate["attempt_id"]
            and claim.get("attempt_number") == candidate["attempt_number"]
            and claim.get("reservation_id") == offer["reservation_id"]
            and claim.get("fencing_token") == candidate["fencing_token"]
            and task.attempt_control.get("current_attempt_id") == candidate["attempt_id"]
            and task.attempt_control.get("current_attempt_number") == candidate["attempt_number"]
        )
        if not owns_current_authority:
            return None

        path = attempt_path(cfg.shared_root, candidate["task_id"], candidate["attempt_number"])
        try:
            attempt = read_json(path)["attempt"]
        except FileNotFoundError as exc:
            raise RuntimeError("matching shared claim is not yet durably materialized") from exc
        if not isinstance(attempt, dict):
            raise RuntimeError("matching shared claim has invalid attempt evidence")
        exact_attempt = (
            attempt.get("attempt_id") == candidate["attempt_id"]
            and attempt.get("task_id") == candidate["task_id"]
            and attempt.get("attempt_number") == candidate["attempt_number"]
            and attempt.get("reservation_id") == offer["reservation_id"]
            and attempt.get("current_fencing_token") == candidate["fencing_token"]
            and attempt.get("assigned_gpus") == offer["gpu_ids"]
        )
        if not exact_attempt:
            raise RuntimeError("matching shared claim has inconsistent attempt evidence")
        if attempt.get("phase") not in {"claimed", "starting", "running"}:
            return None
        return {
            "outcome": "claimed",
            "attempt_id": candidate["attempt_id"],
            "attempt_number": candidate["attempt_number"],
            "fencing_token": candidate["fencing_token"],
            "reservation_id": offer["reservation_id"],
            "gpu_ids": offer["gpu_ids"],
            "cpu_slots": offer["cpu_slots"],
        }


def _cursor_from_position(
    namespace: str,
    machine_name: str,
    scope: str,
    position: dict[str, Any],
) -> ReadyCursor:
    return ReadyCursor(
        namespace,
        machine_name,
        scope,
        position["catalog_page"],
        position["partition"],
        position["after_name"],
        position["revision"],
    )


def _scheduler_cursor_commit(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
    *,
    binding_fence_is_error: bool = False,
    should_retain_observed: bool = False,
) -> dict[str, Any]:
    """Compare-and-commit both observed route cursors in deterministic order."""
    parameters = request.parameters
    machine_name = parameters["machine_name"]
    cfg = load_root_config(Path(request.canonical_shared_root), machine_name)
    if not _claim_binding_is_current(request, runtime_root, cfg):
        if binding_fence_is_error:
            raise _BindingAuthorityChanged
        return {"routes": {"home": "stale", "shared": "stale"}}

    cursor = parameters["cursor"]
    namespace = cursor["namespace"]
    outcomes: dict[str, str] = {}

    def require_current_epoch() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    for scope in ("home", "shared"):
        positions = cursor["routes"][scope]
        observed = _cursor_from_position(namespace, machine_name, scope, positions["observed"])
        next_cursor = (
            observed
            if should_retain_observed
            else _cursor_from_position(namespace, machine_name, scope, positions["next"])
        )
        outcomes[scope] = compare_and_commit_ready_cursor(
            cfg,
            namespace,
            scope,
            observed,
            next_cursor,
            mutation_guard=require_current_epoch,
        )
    return {"routes": outcomes}


def _cursor_position(cursor: ReadyCursor) -> dict[str, Any]:
    return {
        "catalog_page": cursor.catalog_page,
        "partition": cursor.partition,
        "after_name": cursor.after_name,
        "revision": cursor.revision,
    }


def _ready_reference_revisions(cfg: object, reference: ReadyMarkerRef) -> tuple[int, int]:
    """Read the exact catalog and partition revisions named by one reference."""
    route_key = ready_routes.route_key(reference.queue_scope, cfg.machine_name)
    catalog = read_json(ready_routes.catalog_path(cfg.shared_root, route_key, reference.catalog_page))["ready_catalog"]
    if (
        catalog.get("route") != route_key
        or catalog.get("page") != reference.catalog_page
        or type(catalog.get("revision")) is not int
        or reference.partition not in catalog.get("partitions", [])
    ):
        raise ValueError("ready catalog changed while observing a candidate.")
    partition = read_json(
        ready_routes.partition_record_path(
            cfg.shared_root,
            reference.queue_scope,
            cfg.machine_name,
            reference.partition,
        )
    )["ready_partition"]
    if (
        partition.get("route") != route_key
        or partition.get("partition") != reference.partition
        or partition.get("catalog_page") != reference.catalog_page
        or type(partition.get("revision")) is not int
        or reference.marker_name not in partition.get("slots", [])
    ):
        raise ValueError("ready partition changed while observing a candidate.")
    return catalog["revision"], partition["revision"]


def _scheduler_candidate(
    request: ProjectIORequest,
    cfg: object,
    reference: ReadyMarkerRef,
    cursor: ReadyCursor,
    *,
    is_quiescence_probe: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, int], str | None]:
    """Project exact eligible Task and Group truth into a bounded candidate record."""
    classification = classify_ready_marker(cfg, reference, read_only=True)
    if classification.classification == "permanently_stale":
        return None, {}, None
    if classification.classification == "temporarily_unavailable":
        if classification.reason.startswith("dependency_"):
            return None, {}, "dependency_not_ready"
        return None, {}, "candidate_unresolved"
    if classification.classification != "claimable" or classification.task is None:
        return None, {}, "candidate_unresolved"

    task = classification.task
    lane = ("cpu" if task.spec.is_cpu_only else "gpu") if is_quiescence_probe else request.parameters["lane"]
    admission_role = None if is_quiescence_probe else request.parameters["admission_role"]
    if (lane == "cpu") != task.spec.is_cpu_only:
        return None, {}, None
    if not task_machine_matches(task, cfg.machine_name):
        return None, {}, "placement_rejected"

    group_record: dict[str, Any] | None = None
    worker: dict[str, Any] | None = None
    group_revision: int | None = None
    group_dispatch_epoch: int | None = None
    group_worker_set_epoch: int | None = None
    worker_state_epoch: int | None = None
    worker_scheduling_role: str | None = None
    gpu_limit_gpus: int | None = None
    if task.group_name is not None:
        from ..runtime.group_namespace import read_group
        from ..runtime.records import normalize_group_record

        group_value = normalize_group_record(read_group(cfg.shared_root, task.group_name))
        group_record = group_value["group"]
        group_meta = group_value.get("meta")
        if not isinstance(group_meta, dict) or type(group_meta.get("revision")) is not int:
            return None, {}, "candidate_unresolved"
        group_revision = group_meta["revision"]
        group_dispatch_epoch = group_record.get("dispatch_epoch")
        group_worker_set_epoch = group_record.get("worker_set_epoch")
        if (
            group_record.get("dispatch_state") != "active"
            or type(group_dispatch_epoch) is not int
            or type(group_worker_set_epoch) is not int
        ):
            return None, {}, "candidate_unresolved"
        worker = group_record["worker_set"].get(cfg.machine_name)
        if worker is None or worker.get("state") != "active":
            return None, {}, "placement_rejected"
        worker_scheduling_role = worker.get("scheduling_role")
        if admission_role is None:
            admission_role = worker_scheduling_role
        if worker_scheduling_role != admission_role:
            return None, {}, "admission_role_mismatch"
        worker_state_epoch = worker.get("state_epoch", 0)
        gpu_limit_gpus = worker.get("gpu_limit_gpus")
        if type(worker_state_epoch) is not int or worker_state_epoch < 0:
            return None, {}, "candidate_unresolved"
        if gpu_limit_gpus is not None and type(gpu_limit_gpus) is not int:
            return None, {}, "candidate_unresolved"
        if lane == "gpu" and gpu_limit_gpus is not None and task.spec.requested_gpus > gpu_limit_gpus:
            return None, {}, "placement_rejected"
    elif admission_role is None:
        admission_role = "primary"
    elif admission_role != "primary":
        return None, {}, "admission_role_mismatch"

    task_revision = task.meta.get("revision")
    attempt_number = task.attempt_control.get("next_attempt_number")
    fencing_epoch = task.claim_control.get("fencing_epoch")
    if (
        type(task_revision) is not int
        or task_revision < 0
        or type(attempt_number) is not int
        or attempt_number <= 0
        or type(fencing_epoch) is not int
        or fencing_epoch < 0
    ):
        return None, {}, "candidate_unresolved"
    catalog_revision, partition_revision = _ready_reference_revisions(cfg, reference)
    requested_gpus = task.spec.requested_gpus if lane == "gpu" else 0
    requested_cpus = (task.spec.requested_cpus or 0) if lane == "cpu" else 0
    candidate = {
        "task_id": task.task_id,
        "task_revision": task_revision,
        "ready_identity": reference.identity,
        "ready_generation": reference.generation,
        "ready_scope": reference.queue_scope,
        "ready_revision": partition_revision,
        "catalog_revision": catalog_revision,
        "catalog_page": reference.catalog_page,
        "partition": reference.partition,
        "marker_name": reference.marker_name,
        "home_machine": reference.home_machine,
        "attempt_number": attempt_number,
        "attempt_id": f"{task.task_id}-attempt-{attempt_number}",
        "fencing_token": fencing_epoch + 1,
        "lane": lane,
        "requested_gpus": requested_gpus,
        "requested_cpus": requested_cpus,
        "admission_role": admission_role,
        "group_name": task.group_name,
        "group_revision": group_revision,
        "group_dispatch_epoch": group_dispatch_epoch,
        "group_worker_set_epoch": group_worker_set_epoch,
        "worker_state_epoch": worker_state_epoch,
        "worker_scheduling_role": worker_scheduling_role,
        "gpu_limit_gpus": gpu_limit_gpus,
    }
    revisions = {
        "task": task_revision,
        "ready_catalog": catalog_revision,
        "ready_partition": partition_revision,
    }
    if group_revision is not None:
        revisions["group"] = group_revision
    return candidate, revisions, None


def _scheduler_quiescence_probe(request: ProjectIORequest) -> dict[str, Any]:
    """Prove complete ready-route absence independently of dispatch positions."""
    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    # The codec's lane label only namespaces local continuation keys here. Both
    # resource lanes and roles are classified from exact Task/Group truth.
    session = decode_probe_session(request.parameters["probe_state"], request.project_id, cfg.machine_name, "gpu")
    keys = [(request.project_id, scope, "gpu") for scope in ("home", "shared")]
    session.begin_round("gpu", keys)
    budget = SliceBudget(
        WorkBudgetPolicy(
            record_hard_limit=64,
            operation_hard_limit=256,
            soft_deadline_ms=50,
            minimum_batch_size=1,
            initial_batch_size=4,
            growth_observations=3,
        )
    )

    def result(state: str) -> dict[str, Any]:
        return {"state": state, "probe_state": encode_probe_session(session, request.project_id, "gpu")}

    if ready_state.read_ready_index_status(cfg).get("state") != "active" or not is_group_ready_member_projection_usable(
        cfg
    ):
        return result("pending")
    try:
        for key in keys:
            scope = key[1]
            revision = ready_routes.ready_index_route_revision(cfg, scope, budget)
            previous = session.route(key)
            if previous.revision is not None and previous.revision != revision:
                # Primary probes recheck completed baselines. Retirement must
                # also restart an unfinished census when its prefix changes.
                session.invalidate_routes([key])
            decision = session.begin_route(key, revision)
            if not decision.should_scan:
                continue
            cursor = session.route(key).cursor
            while True:
                before = cursor
                peek = peek_ready_marker(cfg, request.project_id, scope, cursor, budget, read_only=True)
                if peek.exhausted or peek.unresolved:
                    session.record_progress(key, before)
                    return result("pending")
                cursor = peek.cursor
                if peek.reference is None:
                    session.record_progress(key, cursor)
                    session.finish_route(key, ready_routes.ready_index_route_revision(cfg, scope, budget))
                    break
                if not budget.can_start_record():
                    session.record_progress(key, before)
                    return result("pending")
                budget.consume_record()
                candidate, _revisions, reason = _scheduler_candidate(
                    request, cfg, peek.reference, cursor, is_quiescence_probe=True
                )
                if candidate is not None:
                    session.record_progress(key, before)
                    return result("active")
                if reason in {"candidate_unresolved", "dependency_not_ready"}:
                    session.record_progress(key, before)
                    return result("pending")
                session.record_progress(key, cursor)
        completed = session.completed_revisions("gpu", keys)
        if completed is None:
            return result("pending")
        for key, revision in completed.items():
            current = ready_routes.ready_index_route_revision(cfg, key[1], budget)
            if current != revision:
                session.invalidate_routes([key])
                return result("pending")
    except ready_routes.ReadyProbeBudgetExhausted:
        return result("pending")
    return result("quiescent")


def _group_service_mutation_fence(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> Callable[[RootConfig], None]:
    """Build the existing executor and binding fence for Group mutations."""

    def before_shared_mutation(cfg: RootConfig) -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    return before_shared_mutation


def _scheduler_primary_probe(request: ProjectIORequest) -> dict[str, Any]:
    """Scan primary truth without dispatch cursors or MachineRuntime effects."""
    from .primary_demand import probe_primary_demand

    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    project_id, lane = request.project_id, parameters["lane"]
    session = decode_probe_session(parameters["probe_state"], project_id, parameters["machine_name"], lane)
    keys = [(project_id, scope, lane) for scope in ("shared", "home")]
    budget = SliceBudget(
        WorkBudgetPolicy(
            record_hard_limit=64,
            operation_hard_limit=256,
            soft_deadline_ms=50,
            minimum_batch_size=1,
            initial_batch_size=4,
            growth_observations=3,
        )
    )
    demand = "unresolved"
    revisions = None
    if parameters["phase"] != "verify" or session.completed_revisions(lane, keys) is not None:
        demand = probe_primary_demand(
            session,
            {project_id},
            {project_id: cfg},
            {project_id: cfg},
            list(range(parameters["visible_capacity"])),
            list(range(parameters["free_capacity"])),
            budget,
            lane=lane,
            group_gpu_usage=parameters["group_gpu_usage"],
        ).state
    if demand == "no_primary_demand":
        completed = session.completed_revisions(lane, keys)
        if completed is None:
            demand = "unresolved"
        else:
            revisions = {key[1]: revision for key, revision in completed.items()}
            try:
                for key, revision in completed.items():
                    observed = ready_routes.ready_index_route_revision(
                        cfg, key[1], budget, primary_only=True, lane=lane
                    )
                    if observed != revision:
                        session.invalidate_routes([key])
                        demand = "unresolved"
            except ready_routes.ReadyProbeBudgetExhausted:
                demand = "unresolved"
            if demand != "no_primary_demand":
                revisions = None
    return {
        "demand": demand,
        "probe_state": encode_probe_session(session, project_id, lane),
        "route_revisions": revisions,
    }


def _scheduler_observe(request: ProjectIORequest) -> dict[str, Any]:
    """Observe one bounded candidate slice without changing Project data."""
    machine_name = request.parameters["machine_name"]
    cfg = load_root_config(Path(request.canonical_shared_root), machine_name)
    status = ready_state.read_ready_index_status(cfg)
    state_revision = status.get("revision")
    if type(state_revision) is not int or state_revision < 0:
        raise ValueError("ready index revision is invalid.")
    budget = SliceBudget(
        WorkBudgetPolicy(
            record_hard_limit=64,
            operation_hard_limit=256,
            soft_deadline_ms=50,
            minimum_batch_size=1,
            initial_batch_size=4,
            growth_observations=3,
        )
    )
    observed: dict[str, ReadyCursor] = {}
    next_positions: dict[str, ReadyCursor] = {}
    cursor_namespace = request.parameters["cursor_namespace"]
    for scope in ("home", "shared"):
        cursor = load_ready_cursor(cfg, cursor_namespace, scope)
        observed[scope] = cursor
        next_positions[scope] = cursor

    def evidence(
        outcome: str, reason: str, candidate: dict[str, Any] | None, revisions: dict[str, int]
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reason": reason,
            "source_revisions": {"ready_index": state_revision, **revisions},
            "candidate": candidate,
            "cursor": {
                "namespace": cursor_namespace,
                "routes": {
                    scope: {
                        "observed": _cursor_position(observed[scope]),
                        "next": _cursor_position(next_positions[scope]),
                    }
                    for scope in ("home", "shared")
                },
            },
        }

    if status.get("state") != "active":
        return evidence("none", "ready_index_inactive", None, {})

    encountered: set[str] = set()
    for scope in ("home", "shared"):
        cursor = next_positions[scope]
        while True:
            if not budget.can_start_operation():
                return evidence("none", "slice_exhausted", None, {})
            peek = peek_ready_marker(cfg, cursor_namespace, scope, cursor, budget, read_only=True)
            cursor = peek.cursor
            next_positions[scope] = cursor
            if peek.exhausted:
                return evidence("none", "slice_exhausted", None, {})
            if peek.unresolved:
                return evidence("none", "candidate_unresolved", None, {})
            if peek.reference is None:
                break
            if not budget.can_start_record():
                return evidence("none", "slice_exhausted", None, {})
            budget.consume_record()
            candidate, revisions, skip_reason = _scheduler_candidate(request, cfg, peek.reference, cursor)
            if skip_reason == "candidate_unresolved":
                return evidence("none", "candidate_unresolved", None, {})
            if skip_reason is not None:
                encountered.add(skip_reason)
                continue
            if candidate is None:
                continue
            return evidence("candidate", "candidate_ready", candidate, revisions)

    for reason in ("dependency_not_ready", "placement_rejected", "admission_role_mismatch"):
        if reason in encountered:
            return evidence("none", reason, None, {})
    return evidence("none", "no_candidate", None, {})


def _scheduler_due_offer(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Advance one due offer under repeated executor and binding fences."""
    from ..project_maintenance import advance_due_offer

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    progress = advance_due_offer(cfg, before_shared_mutation=before_shared_mutation)
    return {"outcome": progress.outcome, "reason": progress.reason, "task_id": progress.task_id}


def _scheduler_ready_index_build(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Advance one ready-index slice under repeated executor and binding fences."""
    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        status = advance_ready_index_build(cfg, max_tasks=64, bounded_initialization=True)
    state = status.get("state")
    revision = status.get("revision")
    build = status.get("build") if state == "building" else None
    build_id = build.get("build_id") if isinstance(build, dict) else None
    phase = build.get("phase") if isinstance(build, dict) else None
    return {"state": state, "revision": revision, "build_id": build_id, "phase": phase}


def _submission_control_service(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Advance one journal-backed proof slice, with no machine-local effects."""
    from ..runtime import submission_control
    from ..runtime.submission_control_maintenance import SubmissionControlMaintenance

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    owner = SubmissionControlMaintenance(cfg, continuation=request.parameters["continuation"])
    try:
        with fenced_mutations(cfg.shared_root, before_shared_mutation):
            result = owner.advance()
            state = submission_control.read_control_state(cfg)
        quiescent = result.get("reason") == "pending_idle" and state is not None and state["state"] == "active"
        progress = result["state"] in {"active", "building"} or result.get("reason") in {
            "pending_scan_complete",
            "pending_entry_skipped",
            "pending_directory_changed",
            "source_checkpointed",
        }
        progress = progress or dict(request.parameters["continuation"]) != owner.continuation
        return {
            "state": result["state"],
            "quiescent": quiescent,
            "reason_code": "idle" if quiescent else "progress" if progress else "blocked",
            "continuation": owner.continuation,
        }
    finally:
        owner.close()


def _observation_service(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Advance one shared observation slice with all handles closed before return."""
    from ..runtime.observation.maintenance import ObservationMaintenance

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    owner = ObservationMaintenance(cfg)
    try:
        with fenced_mutations(cfg.shared_root, before_shared_mutation):
            result = owner.advance()
        quiescent = result["state"] == "active" and result.get("reason") == "idle"
        has_progress = result["state"] == "building" or result.get("reason") in {
            "garbage_collecting",
            "reclaiming_observation_generation",
        }
        return {
            "state": result["state"],
            "quiescent": quiescent,
            "reason_code": "idle" if quiescent else "progress" if has_progress else "blocked",
        }
    finally:
        owner.close()


def _notification_service(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Capture shared legacy truth before the narrow private bookkeeping transaction."""
    from ..notification_policy import NotificationPolicyBusyError
    from ..notification_reconciliation import (
        LegacyConflictError,
        LegacySourceBusyError,
        LegacySourceInvalidError,
        reconcile_captured_legacy,
    )

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])

    def before_local_write() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _registry_binding_is_current(request, runtime_root):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    def before_shared_capture() -> None:
        # Recheck registration under its shared machine fence, never the policy lock.
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    try:
        reconcile_captured_legacy(
            runtime_root,
            cfg,
            request.project_id,
            before_shared_capture=before_shared_capture,
            before_local_write=before_local_write,
        )
    except (_ExecutorEpochFenced, _BindingAuthorityChanged):
        raise
    except LegacyConflictError:
        return {"state": "conflict"}
    except LegacySourceInvalidError:
        return {"state": "source_invalid"}
    except (LegacySourceBusyError, NotificationPolicyBusyError):
        return {"state": "blocked"}
    return {"state": "ready"}


def _recovery_admission(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Prepare shared recovery without MachineRuntime locks or record writes."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.recovery_admission import prepare_shared_recovery_admission
    from ..runtime.responsibility_store import DurableIO, Unavailable

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    journal_path = machine_runtime_paths(runtime_root)["registration_transaction"]
    migration_path = project_root / "migration.json"
    identity = RegistrationIdentity(
        request.project_id, request.registration_generation, request.runtime_id, str(runtime_root)
    )

    def migration_context() -> dict | None:
        if not is_path_present(migration_path):
            return None
        if not stat.S_ISREG(migration_path.lstat().st_mode):
            raise Unavailable("recovery preparation requires regular migration metadata")
        migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
        if not isinstance(migration, dict):
            raise Unavailable("recovery preparation migration metadata is invalid")
        if migration.get("state") != "active" or any(
            migration.get(key) != value
            for key, value in {
                "project_id": request.project_id,
                "shared_root": request.canonical_shared_root,
                "machine_name": request.parameters["machine_name"],
            }.items()
        ):
            raise _RecoveryPreparationDeferred
        return migration

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    waiting = {"state": "waiting", "registration_prepared": False, "admission_fenced": False}

    def rollback_pending_registration() -> bool:
        """Replay one durable CLI snapshot under machine-local exclusion."""
        from ..runtime.locks import exclusive
        from .registration import MachineRegistration

        registration = MachineRegistration(
            runtime_root,
            ensure_layout=lambda: None,
            current_instance_id=lambda: request.runtime_id,
        )
        with exclusive(paths["locks"] / "migration.lock", blocking=False) as migration_acquired:
            if not migration_acquired:
                return False
            with registration.registry_guard(blocking=False) as registry_acquired:
                if not registry_acquired or not is_path_present(journal_path):
                    return False
                current_epoch = _read_epoch(paths)
                if (
                    not current_epoch.active
                    or current_epoch.runtime_id != request.runtime_id
                    or current_epoch.executor_epoch != request.executor_epoch
                ):
                    raise _ExecutorEpochFenced
                if not _registry_binding_is_current(request, runtime_root):
                    raise _BindingAuthorityChanged

                def before_effect() -> None:
                    effect_epoch = _read_epoch(paths)
                    if (
                        not effect_epoch.active
                        or effect_epoch.runtime_id != request.runtime_id
                        or effect_epoch.executor_epoch != request.executor_epoch
                    ):
                        raise _ExecutorEpochFenced
                    write_possible[0] = True

                registration.rollback_pending_locked(before_effect=before_effect)
                return True

    epoch = _read_epoch(paths)
    if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
        raise _ExecutorEpochFenced
    if not _registry_binding_is_current(request, runtime_root):
        raise _BindingAuthorityChanged
    try:
        if is_path_present(journal_path):
            if rollback_pending_registration():
                return waiting
            if is_path_present(journal_path):
                return waiting
        migration = migration_context()
        from ..runtime.registration_authority import observe_registration_owner

        def observation_is_current() -> bool:
            current_epoch = _read_epoch(paths)
            return (
                current_epoch.active
                and current_epoch.runtime_id == request.runtime_id
                and current_epoch.executor_epoch == request.executor_epoch
                and _registry_binding_is_current(request, runtime_root)
                and not is_path_present(journal_path)
                and migration_context() == migration
            )

        if observe_registration_owner(cfg, identity, is_current=observation_is_current) == "superseded":
            return {"state": "superseded", "registration_prepared": False, "admission_fenced": False}
        fence()
        machine_record = load_machine_record(cfg)
        machine = machine_record.get("machine") if isinstance(machine_record, dict) else None
        if not isinstance(machine, dict) or machine.get("agent_runtime") != "machine":
            return waiting

        def local_current() -> bool:
            return (
                _registry_binding_is_current(request, runtime_root)
                and not is_path_present(journal_path)
                and migration_context() == migration
            )

        def before_shared_write() -> None:
            fence()
            if not local_current() or load_machine_record(cfg) != machine_record:
                raise _RecoveryPreparationDeferred
            write_possible[0] = True

        def before_registration_conversion() -> None:
            fence()
            if not local_current():
                raise _RecoveryPreparationDeferred
            # Shared registration exclusion prevents a current CLI from taking
            # an old snapshot in the gap between this absence barrier and v2.
            # This sync changes no MachineRuntime file and needs no local guard.
            DurableIO().sync_directory(journal_path.parent, "recovery_registration")
            fence()
            if not local_current():
                raise _RecoveryPreparationDeferred

        prepared = prepare_shared_recovery_admission(
            cfg,
            identity,
            is_current=local_current,
            before_registration_conversion=before_registration_conversion,
            before_shared_write=before_shared_write,
        )
        fence()
        if not local_current():
            raise _RecoveryPreparationDeferred
    except _RecoveryPreparationDeferred:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return waiting
    return {
        "state": "ready" if prepared.admission.is_fenced else "waiting",
        "registration_prepared": prepared.is_registration_prepared,
        "admission_fenced": prepared.admission.is_fenced,
    }


def _recovery_capture_transition(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Fence generation handoff source effects; local reset belongs to the coordinator."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.responsibility_backfill import _canonical_capture_root
    from ..runtime.responsibility_capture import CAPTURE_BYTES, CAPTURE_FILE, GENERATION_FILE
    from ..runtime.responsibility_completion import COMPLETION_FILE, SOURCE_RELEASE_FILE, _digest
    from ..runtime.responsibility_generation import (
        load_capture_generation_transition,
        normalize_generation_source,
        retain_generation_source,
    )
    from ..runtime.responsibility_source_read import SourceCaptureStale
    from ..runtime.responsibility_store import Unavailable

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    target = machine_project_paths(runtime_root, request.project_id)["root"]
    migration_path = target / "migration.json"
    owner = {
        "project_id": request.project_id,
        "shared_root": request.canonical_shared_root,
        "machine_name": request.parameters["machine_name"],
        "owner_root": str(runtime_root),
        "owner_instance": request.runtime_id,
        "registration_generation": request.registration_generation,
    }

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    def transition_context():
        migration = None
        source = None
        if is_path_present(migration_path):
            if not stat.S_ISREG(migration_path.lstat().st_mode):
                raise Unavailable("capture transition requires regular migration metadata")
            migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
            if not isinstance(migration, dict) or any(
                migration.get(key) != value
                for key, value in {
                    "project_id": request.project_id,
                    "shared_root": request.canonical_shared_root,
                    "machine_name": request.parameters["machine_name"],
                    "state": "active",
                }.items()
            ):
                raise SourceCaptureStale("capture transition migration is not current")
            raw_source = migration.get("legacy_runtime_root")
            if not isinstance(raw_source, str) or str(Path(raw_source)) != raw_source:
                raise Unavailable("capture transition source is not canonical")
            source = _canonical_capture_root(Path(raw_source))
        transition = load_capture_generation_transition(target, legacy_source=source, owner=owner)
        if _digest(transition.state["previous_completion"]) != request.parameters["completion_digest"]:
            raise SourceCaptureStale("capture transition completion identity changed")
        current = read_json_limited(target / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
        presence = tuple(
            is_path_present(target / name) for name in (GENERATION_FILE, COMPLETION_FILE, SOURCE_RELEASE_FILE)
        )
        if request.parameters["phase"] == "normalize":
            if not presence[0] or presence[1] or presence[2] or current != transition.after:
                raise SourceCaptureStale("source normalization requires the completed local reset")
        elif current != transition.before:
            raise SourceCaptureStale("source retention phase was already locally reset")
        return transition, migration, presence

    fence()
    try:
        context = transition_context()

        def before_source_write() -> None:
            fence()
            if transition_context() != context:
                raise SourceCaptureStale("capture transition local context changed before source effect")
            write_possible[0] = True

        transaction = (
            retain_generation_source if request.parameters["phase"] == "retain" else normalize_generation_source
        )
        transaction(context[0], before_source_write=before_source_write)
        fence()
        if transition_context() != context:
            raise SourceCaptureStale("capture transition local context changed before result publication")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "retained" if request.parameters["phase"] == "retain" else "normalized"}


def _recovery_group_authority(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Activate shared Group truth only from this binding's exact local completion."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.group_namespace import activate_group_authority_locked
    from ..runtime.locks import schema_lock
    from ..runtime.recovery_admission import _upgrade_blocker
    from ..runtime.registration_authority import RECOVERY_REGISTRATION_VERSION, registration_write_guard
    from ..runtime.responsibility_capture import GENERATION_FILE, capture_admission
    from ..runtime.responsibility_completion import _digest, read_capture_completion
    from ..runtime.responsibility_source_read import SourceCaptureStale

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    target = machine_project_paths(runtime_root, request.project_id)["root"]
    identity = RegistrationIdentity(
        request.project_id, request.registration_generation, request.runtime_id, str(runtime_root)
    )
    owner = capture_admission(
        {
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
            "owner_root": str(runtime_root),
            "owner_instance": request.runtime_id,
            "registration_generation": request.registration_generation,
        }
    )

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    def completion_context() -> dict:
        if is_path_present(target / GENERATION_FILE):
            raise SourceCaptureStale("Group activation capture is transitioning")
        proof = read_capture_completion(target, should_sync=False)
        if (
            proof is None
            or capture_admission(proof) != owner
            or _digest(proof) != request.parameters["completion_digest"]
        ):
            raise SourceCaptureStale("Group activation requires this exact binding's completion")
        return proof

    fence()
    try:
        proof = completion_context()

        def is_current() -> bool:
            fence()
            return completion_context() == proof

        def before_shared_write() -> None:
            if not is_current():
                raise SourceCaptureStale("Group activation completion changed before mutation")
            write_possible[0] = True

        with registration_write_guard(
            cfg, identity, is_current=is_current, before_shared_write=before_shared_write
        ) as registration:
            if registration is None or registration["version"] != RECOVERY_REGISTRATION_VERSION:
                return {"state": "waiting"}
            with schema_lock(cfg.shared_root, blocking=False) as acquired:
                if not acquired or _upgrade_blocker(cfg) is not None:
                    return {"state": "waiting"}
                with fenced_mutations(cfg.shared_root, before_shared_write):
                    is_active = activate_group_authority_locked(cfg)
                if not is_current():
                    raise SourceCaptureStale("Group activation completion changed before result")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "active" if is_active else "waiting"}


def _recovery_source_release(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Consume a local durable retirement intent and release only source retention."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.responsibility_capture import GENERATION_FILE, capture_admission
    from ..runtime.responsibility_completion import (
        SOURCE_RELEASE_FILE,
        _digest,
        load_source_release_context,
        read_capture_completion,
        release_source_retention,
    )
    from ..runtime.responsibility_source_read import SourceCaptureStale

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    target = machine_project_paths(runtime_root, request.project_id)["root"]
    owner = capture_admission(
        {
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
            "owner_root": str(runtime_root),
            "owner_instance": request.runtime_id,
            "registration_generation": request.registration_generation,
        }
    )

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    def capture_context():
        if is_path_present(target / GENERATION_FILE) or not is_path_present(target / SOURCE_RELEASE_FILE):
            raise SourceCaptureStale("source release has no current local retirement intent")
        proof = read_capture_completion(target, should_sync=False)
        if (
            proof is None
            or capture_admission(proof) != owner
            or _digest(proof) != request.parameters["completion_digest"]
        ):
            raise SourceCaptureStale("source release completion belongs to another owner or revision")
        return load_source_release_context(target, completion_digest=request.parameters["completion_digest"])

    fence()
    try:
        context = capture_context()

        def before_source_write() -> None:
            fence()
            if capture_context() != context:
                raise SourceCaptureStale("source release context changed before source effect")
            write_possible[0] = True

        is_released = release_source_retention(context, before_source_write=before_source_write)
        fence()
        if capture_context() != context:
            raise SourceCaptureStale("source release context changed before result publication")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "released" if is_released else "waiting"}


def _recovery_source_hold(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Own source retention only; exact target metadata is read-only."""
    from ..runtime.responsibility_capture import retain_capture_source
    from ..runtime.responsibility_source_read import SourceCaptureStale, load_source_hold_context
    from ..runtime.responsibility_store import Unavailable

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    owner = {
        "project_id": request.project_id,
        "shared_root": request.canonical_shared_root,
        "machine_name": request.parameters["machine_name"],
        "owner_root": str(runtime_root),
        "owner_instance": request.runtime_id,
        "registration_generation": request.registration_generation,
    }

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    def capture_context() -> tuple[dict, dict]:
        capture = load_source_hold_context(project_root, owner, capture_id=request.parameters["capture_id"])
        migration_path = project_root / "migration.json"
        if not stat.S_ISREG(migration_path.lstat().st_mode):
            raise Unavailable("source retention requires regular completed migration metadata")
        migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
        expected = {
            "state": "active",
            "legacy_runtime_root": capture["legacy_source"],
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
        }
        if not isinstance(migration, dict) or any(migration.get(key) != value for key, value in expected.items()):
            raise SourceCaptureStale("source retention migration identity changed")
        return capture, migration

    fence()
    try:
        capture, migration = capture_context()

        def before_source_write() -> None:
            fence()
            if capture_context() != (capture, migration):
                raise _BindingAuthorityChanged
            write_possible[0] = True

        hold = retain_capture_source(capture, before_source_write=before_source_write)
        fence()
        if capture_context() != (capture, migration):
            raise _BindingAuthorityChanged
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale", "hold": None}
    return {"state": "retained", "hold": hold.to_dict()}


def _legacy_capture_read(request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """Read/discover retained source evidence without any target mutation."""
    from ..runtime.responsibility_source_read import (
        SourceCaptureStale,
        load_source_read_context,
        read_retained_source_locator,
    )
    from ..runtime.responsibility_source_scan import (
        initial_source_scan_cursor,
        scan_retained_source,
        source_scan_cursor,
    )
    from .recovery_transport import encode_capture_locator

    parameters = dict(request.parameters)
    cfg = load_root_config(Path(request.canonical_shared_root), parameters.pop("machine_name"))
    is_scan = request.operation_kind == "legacy_capture_scan"
    cursor = source_scan_cursor(parameters.pop("cursor")) if is_scan else None
    if is_scan:
        parameters["relative"] = None
    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    owner = {
        "project_id": request.project_id,
        "shared_root": request.canonical_shared_root,
        "machine_name": request.parameters["machine_name"],
        "owner_root": str(runtime_root),
        "owner_instance": request.runtime_id,
        "registration_generation": request.registration_generation,
    }

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    fence()
    try:
        context = load_source_read_context(project_root, owner, **parameters)
        if is_scan:
            if context.backfill.get("source_cursor", initial_source_scan_cursor()) != cursor:
                raise SourceCaptureStale("source discovery cursor differs from its local checkpoint")
            scan = scan_retained_source(context, parameters["lane"], cursor)
        else:
            captured = read_retained_source_locator(context, parameters["lane"], parameters["relative"])
        fence()
        if load_source_read_context(project_root, owner, **parameters) != context:
            raise SourceCaptureStale("retained source context changed during the read")
    except SourceCaptureStale:
        return {"state": "stale", "scan" if is_scan else "locator": None}
    if is_scan:
        return {"state": "observed", "scan": scan}
    return {"state": "observed", "locator": encode_capture_locator(captured)}


def _progress_projection(
    request: ProjectIORequest, runtime_root: Path, paths: dict[str, Path], write_possible: list[bool]
) -> dict[str, Any]:
    """Resolve shared identity or publish one exact projection; local state is read-only."""
    from qqtools.qexp._progress_protocol import read_advisory_snapshot

    from ..runtime.progress import resolve_progress_binding
    from ..runtime.progress_projection import observe_progress_snapshot, publish_progress_snapshot

    parameters = request.to_dict()["project_io_request"]["parameters"]
    context = dict(parameters["context"])
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    directory = (
        "progress-contexts" if context["protocol_version"] == 1 else f"progress-v{context['protocol_version']}-contexts"
    )
    context_path = project_root / directory / f"{context['attempt_id']}.json"

    def context_alive() -> bool:
        try:
            return read_advisory_snapshot(context_path) == context
        except (OSError, ValueError, TypeError, RecursionError):
            return False

    def fence() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged

    fence()
    if not context_alive():
        return {"state": "retired", "binding": None, "snapshot": None}
    projection = parameters["projection"]
    if projection is None:
        result = observe_progress_snapshot(cfg, context, request.registration_generation)
        fence()
        if not context_alive():
            return {"state": "retired", "binding": None, "snapshot": None}
        return result
    captured_binding = {
        **context,
        "registration_generation": request.registration_generation,
        "fencing_token": projection["fencing_token"],
    }

    def before_replace() -> None:
        fence()
        if not context_alive():
            raise _BindingAuthorityChanged
        current = resolve_progress_binding(cfg, context)
        if current is None or current["fencing_token"] != projection["fencing_token"]:
            raise _BindingAuthorityChanged
        write_possible[0] = True

    with fenced_mutations(cfg.shared_root, before_replace):
        result = publish_progress_snapshot(
            cfg, context, captured_binding, dict(projection), context_alive=context_alive, before_replace=before_replace
        )
    fence()
    return result


def _upgrade_service(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Use the shared upgrade journal as the sole recovery authority."""
    from ..runtime.upgrade.framework import UpgradeCoordinator, pending_upgrade_requires_completion

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        coordinator = UpgradeCoordinator(cfg)
        status = coordinator.discover()
        if status.get("pending") and status.get("can_run"):
            status = coordinator.advance()
    return {
        "state": status["state"],
        "pending": bool(status.get("pending")),
        "can_run": bool(status.get("can_run")),
        "admission_blocked": bool(status.get("admission_blocked")) or status["state"] == "inaccessible",
        "idle_blocking": pending_upgrade_requires_completion(status),
        "next_probe_at": status.get("next_probe_at"),
    }


def _maintenance_descriptor_advance(
    request: ProjectIORequest,
    runtime_root: Path,
    paths: dict[str, Path],
    write_possible: list[bool],
) -> dict[str, Any]:
    """Advance one Project maintenance descriptor through the isolated worker."""
    from ..runtime.maintenance import advance_maintenance_work

    parameters = request.parameters
    cfg = load_root_config(Path(request.canonical_shared_root), parameters["machine_name"])
    cfg = replace(cfg, runtime_root=machine_project_paths(runtime_root, request.project_id)["root"])

    def before_shared_mutation() -> None:
        epoch = _read_epoch(paths)
        if not epoch.active or epoch.runtime_id != request.runtime_id or epoch.executor_epoch != request.executor_epoch:
            raise _ExecutorEpochFenced
        if not _claim_binding_is_current(request, runtime_root, cfg):
            raise _BindingAuthorityChanged
        write_possible[0] = True

    with fenced_mutations(cfg.shared_root, before_shared_mutation):
        progress = advance_maintenance_work(
            cfg,
            reservation_runtime_root=runtime_root,
            max_scan=1,
        )
    maintenance_state = progress.get("maintenance_state")
    if maintenance_state not in {"idle", "waiting", "pending", "running", "completed", "intervention"}:
        raise ValueError("maintenance descriptor returned an invalid state.")
    next_due_at = progress.get("next_due_at") if maintenance_state == "waiting" else None
    more = progress.get("more")
    if more is None:
        # A terminal descriptor may have a due successor in the same Project
        # outbox. Require one bounded confirmation slice before settling the
        # Project instead of treating a missing hint as proof of global idle.
        more = maintenance_state in {"pending", "running", "completed", "intervention"}
    if type(more) is not bool:
        raise ValueError("maintenance descriptor returned an invalid more hint.")
    idle_blocking = progress.get("idle_blocking", maintenance_state == "waiting")
    if type(idle_blocking) is not bool or (idle_blocking and maintenance_state != "waiting"):
        raise ValueError("maintenance descriptor returned an invalid idle blocker.")
    return {
        "maintenance_state": maintenance_state,
        "next_due_at": next_due_at,
        "more": more,
        "idle_blocking": idle_blocking,
    }


def _safe_exception_type(exc: Exception) -> str:
    value = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,95}", value):
        return "Exception"
    return value


def _publish_result(
    root: Path,
    paths: dict[str, Path],
    request: ProjectIORequest,
    process: ProjectIOProcess,
    status: str,
    reason_code: str | None,
    evidence: dict[str, Any],
    *,
    reconcile_fenced_claim: bool = False,
    reconcile_authority_renewal: bool = False,
    reconcile_authority_orphan_recovery: bool = False,
    reconcile_authority_termination_commit: bool = False,
    reconcile_authority_terminal_publish: bool = False,
    reconciliation_epoch: str | None = None,
    shared_write_possible: bool = False,
) -> bool:
    from ..runtime.locks import exclusive

    result_path = paths["project_io_results"] / f"{request.request_id}.json"
    with exclusive(paths["project_io_lock"]):
        current_request = _read_request(paths, request.request_id)
        current_process = _read_process(paths, request.request_id)
        epoch = _read_epoch(paths)
        if (
            current_request != request
            or current_process.request != request
            or current_process.pid != process.pid
            or current_process.start_time_ticks != process.start_time_ticks
            or current_process.state not in {"running", "stop_targeted", "exit_unverified"}
            or epoch.runtime_id != request.runtime_id
        ):
            return False
        if (
            (not epoch.active or epoch.executor_epoch != request.executor_epoch)
            and status != "outcome_unknown"
            and not (
                reconcile_fenced_claim
                and epoch.active
                and request.operation_kind == "scheduler_claim"
                and epoch.runtime_id == request.runtime_id
            )
            and not (
                reconcile_authority_renewal
                and epoch.active
                and request.operation_kind == "authority_renewal"
                and epoch.runtime_id == request.runtime_id
                and epoch.executor_epoch == reconciliation_epoch
            )
            and not (
                reconcile_authority_orphan_recovery
                and epoch.active
                and request.operation_kind == "authority_orphan_recovery"
                and epoch.runtime_id == request.runtime_id
                and epoch.executor_epoch == reconciliation_epoch
            )
            and not (
                reconcile_authority_termination_commit
                and epoch.active
                and request.operation_kind == "authority_termination_commit"
                and epoch.runtime_id == request.runtime_id
                and epoch.executor_epoch == reconciliation_epoch
            )
            and not (
                reconcile_authority_terminal_publish
                and epoch.active
                and request.operation_kind == "authority_terminal_publish"
                and epoch.runtime_id == request.runtime_id
                and epoch.executor_epoch == reconciliation_epoch
            )
        ):
            if shared_write_possible and request.operation_kind in {
                "upgrade_service",
                "submission_control_service",
                "observation_service",
                "notification_service",
                "progress_projection",
                "recovery_source_hold",
                "recovery_admission",
                "recovery_source_release",
                "recovery_group_authority",
                "recovery_capture_transition",
                "maintenance_descriptor_advance",
                "maintenance_flush_event",
                "machine_snapshot_publish",
                "registration_renew",
                "activation_consumer_register",
                "activation_consumer_ack",
                "activation_consumer_retire",
                "authority_renewal",
                "authority_orphan_recovery",
                "authority_termination_commit",
                "authority_terminal_publish",
                "authority_running_publish",
            }:
                status = "outcome_unknown"
                reason_code = "project_io_outcome_unknown"
            else:
                status = "fenced"
                reason_code = "project_io_executor_epoch_fenced"
            evidence = {}
        elif status == "fenced":
            reason_code = "project_io_executor_epoch_fenced"
            evidence = {}
        try:
            existing = read_json_limited(
                result_path,
                max_bytes=PROJECT_IO_MAX_RECORD_BYTES,
                record_type="project_io_result",
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            parsed = ProjectIOResult.from_dict(existing)
            if parsed.request != request:
                return False
            if not (
                (
                    reconcile_authority_renewal
                    or reconcile_authority_orphan_recovery
                    or reconcile_authority_termination_commit
                    or reconcile_authority_terminal_publish
                )
                and parsed.status in {"outcome_unknown", "retryable_error"}
            ):
                return True
        result = ProjectIOResult(
            request=request,
            status=status,
            reason_code=reason_code,
            completed_at=utc_now(),
            evidence=evidence,
        )
        record = result.to_dict()
        require_json_size(record, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type="project_io_result")
        return atomic_replace(result_path, record) is not None


def _run(
    root: Path,
    request_id: str,
    *,
    reconcile_fenced_claim: bool = False,
    reconcile_authority_renewal: bool = False,
    reconcile_authority_orphan_recovery: bool = False,
    reconcile_authority_termination_commit: bool = False,
    reconcile_authority_terminal_publish: bool = False,
) -> int:
    paths = machine_runtime_paths(root)
    runtime_id = _resolve_runtime_id(root)
    pid = os.getpid()
    deadline = time.monotonic() + _HANDSHAKE_SECONDS
    ticks = _process_start_time_ticks(pid)
    while ticks is None and time.monotonic() < deadline:
        time.sleep(_HANDSHAKE_POLL_SECONDS)
        ticks = _process_start_time_ticks(pid)
    if ticks is None:
        return 1
    handshake = _wait_for_handshake(paths, request_id, runtime_id, pid, ticks)
    if handshake is None:
        return 1
    request, process, epoch = handshake
    if reconcile_authority_renewal and (
        request.operation_kind != "authority_renewal" or not epoch.active or epoch.runtime_id != request.runtime_id
    ):
        return 1
    if reconcile_authority_orphan_recovery and (
        request.operation_kind != "authority_orphan_recovery"
        or not epoch.active
        or epoch.runtime_id != request.runtime_id
    ):
        return 1
    if reconcile_authority_termination_commit and (
        request.operation_kind != "authority_termination_commit"
        or not epoch.active
        or epoch.runtime_id != request.runtime_id
    ):
        return 1
    if reconcile_authority_terminal_publish and (
        request.operation_kind != "authority_terminal_publish"
        or not epoch.active
        or epoch.runtime_id != request.runtime_id
    ):
        return 1
    if reconcile_fenced_claim:
        if not epoch.active or epoch.runtime_id != request.runtime_id or request.operation_kind != "scheduler_claim":
            return 1
        try:
            with _bounded_project_reads(Path(request.canonical_shared_root)):
                cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
                if epoch.executor_epoch == request.executor_epoch and _claim_binding_is_current(request, root, cfg):
                    return 1
                committed = _committed_claim_evidence(
                    cfg,
                    request.parameters["candidate"],
                    request.parameters["offer"],
                )
        except Exception:
            return (
                0
                if _publish_result(
                    root,
                    paths,
                    request,
                    process,
                    "outcome_unknown",
                    "project_io_outcome_unknown",
                    {},
                    reconcile_fenced_claim=True,
                )
                else 1
            )
        if committed is None:
            return (
                0
                if _publish_result(
                    root,
                    paths,
                    request,
                    process,
                    "fenced",
                    "project_io_executor_epoch_fenced",
                    {},
                    reconcile_fenced_claim=True,
                )
                else 1
            )
        return (
            0
            if _publish_result(
                root,
                paths,
                request,
                process,
                "completed",
                None,
                {**committed, "cursor_routes": None},
                reconcile_fenced_claim=True,
            )
            else 1
        )
    replay_authority_mutation = (
        reconcile_authority_renewal
        or reconcile_authority_orphan_recovery
        or reconcile_authority_termination_commit
        or reconcile_authority_terminal_publish
    )
    if not epoch.active or (epoch.executor_epoch != request.executor_epoch and not replay_authority_mutation):
        return (
            0
            if _publish_result(
                root,
                paths,
                request,
                process,
                "fenced",
                "project_io_executor_epoch_fenced",
                {},
            )
            else 1
        )
    fenced = False
    cursor_write_possible = [False]
    claim_cursor_commit_started = [False]
    launch_authorization_write_possible = [False]
    due_offer_write_possible = [False]
    ready_index_write_possible = [False]
    descriptor_write_possible = [False]
    upgrade_write_possible = [False]
    event_write_possible = [False]
    snapshot_write_possible = [False]
    registration_write_possible = [False]
    activation_consumer_write_possible = [False]
    authority_renewal_write_possible = [False]
    authority_orphan_recovery_write_possible = [False]
    authority_termination_write_possible = [False]
    authority_terminal_write_possible = [False]
    authority_running_write_possible = [False]
    try:
        with _bounded_project_reads(Path(request.canonical_shared_root)):
            activation_mutation = request.operation_kind in {
                "activation_consumer_register",
                "activation_consumer_ack",
                "activation_consumer_retire",
            }
            binding_evidence = _validate_binding(
                request,
                root,
                validate_registration=not (
                    activation_mutation
                    or replay_authority_mutation
                    or request.operation_kind in {"registration_renew", "recovery_admission"}
                ),
            )
            if request.operation_kind == "validate_binding":
                evidence = binding_evidence
            elif request.operation_kind == "upgrade_service":
                evidence = _upgrade_service(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "submission_control_service":
                evidence = _submission_control_service(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "observation_service":
                evidence = _observation_service(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "notification_service":
                evidence = _notification_service(request, root, paths, upgrade_write_possible)
            elif request.operation_kind in {"group_service_probe", "group_service_advance"}:
                before_shared_mutation = (
                    _group_service_mutation_fence(request, root, paths, upgrade_write_possible)
                    if request.operation_kind == "group_service_advance"
                    else None
                )
                from .group_service_worker import handle_group_service_request

                evidence = handle_group_service_request(request, root, before_shared_mutation)
            elif request.operation_kind == "progress_projection":
                evidence = _progress_projection(request, root, paths, upgrade_write_possible)
            elif request.operation_kind in {"legacy_capture_read", "legacy_capture_scan"}:
                evidence = _legacy_capture_read(request, root, paths)
            elif request.operation_kind == "recovery_source_hold":
                evidence = _recovery_source_hold(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "recovery_admission":
                evidence = _recovery_admission(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "recovery_source_release":
                evidence = _recovery_source_release(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "recovery_group_authority":
                evidence = _recovery_group_authority(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "recovery_capture_transition":
                evidence = _recovery_capture_transition(request, root, paths, upgrade_write_possible)
            elif request.operation_kind == "activation_observe":
                evidence = _activation_checkpoint_observe(request)
            elif request.operation_kind in {"activation_consumer_register", "activation_consumer_ack"}:
                evidence = _activation_consumer_mutation(
                    request,
                    root,
                    paths,
                    activation_consumer_write_possible,
                )
            elif request.operation_kind == "activation_consumer_retire":
                evidence = _activation_consumer_retirement(
                    request,
                    root,
                    paths,
                    activation_consumer_write_possible,
                )
            elif request.operation_kind == "scheduler_observe":
                evidence = _scheduler_observe(request)
            elif request.operation_kind == "scheduler_primary_probe":
                evidence = _scheduler_primary_probe(request)
            elif request.operation_kind == "scheduler_quiescence_probe":
                evidence = _scheduler_quiescence_probe(request)
            elif request.operation_kind == "scheduler_claim":
                # This last check deliberately takes no MachineRuntime lock:
                # fencing may race after it, in which case result publication
                # reports fenced and shared recovery evidence owns ambiguity.
                current_epoch = _read_epoch(paths)
                if (
                    not current_epoch.active
                    or current_epoch.runtime_id != request.runtime_id
                    or current_epoch.executor_epoch != request.executor_epoch
                ):
                    fenced = True
                else:
                    claim_evidence = _scheduler_claim(request, root)
                    if claim_evidence.get("outcome") == "claimed":
                        evidence = {**claim_evidence, "cursor_routes": None}
                    elif claim_evidence.get("outcome") == "no_claim":
                        claim_cursor_commit_started[0] = True
                        cursor_evidence = _scheduler_cursor_commit(
                            request,
                            root,
                            paths,
                            cursor_write_possible,
                            binding_fence_is_error=True,
                            # Rejecting an obsolete observation does not prove
                            # the current candidate can be skipped. Re-observe
                            # it before advancing past this position.
                            should_retain_observed=claim_evidence.get("reason")
                            in {
                                "ready_changed",
                                "task_changed",
                                "group_changed",
                                "admission_role_changed",
                            },
                        )
                        evidence = {**claim_evidence, "cursor_routes": cursor_evidence["routes"]}
                    else:
                        raise ProjectIOProtocolError("scheduler_claim returned an unsupported outcome.")
            elif request.operation_kind == "scheduler_launch_authorize":

                def require_current_launch_authority() -> None:
                    current_epoch = _read_epoch(paths)
                    if (
                        not current_epoch.active
                        or current_epoch.runtime_id != request.runtime_id
                        or current_epoch.executor_epoch != request.executor_epoch
                    ):
                        raise _ExecutorEpochFenced
                    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
                    if not _claim_binding_is_current(request, root, cfg):
                        raise _BindingAuthorityChanged

                def before_launch_shared_write() -> None:
                    require_current_launch_authority()
                    launch_authorization_write_possible[0] = True

                require_current_launch_authority()
                evidence = _scheduler_launch_authorize(request, root, before_launch_shared_write)
                require_current_launch_authority()
            elif request.operation_kind == "scheduler_reservation_reconcile":
                evidence = _scheduler_reservation_reconcile(request)
            elif request.operation_kind == "scheduler_due_offer":
                evidence = _scheduler_due_offer(request, root, paths, due_offer_write_possible)
            elif request.operation_kind == "scheduler_ready_index_build":
                evidence = _scheduler_ready_index_build(request, root, paths, ready_index_write_possible)
            elif request.operation_kind == "maintenance_descriptor_advance":
                evidence = _maintenance_descriptor_advance(request, root, paths, descriptor_write_possible)
            elif request.operation_kind == "scheduler_cursor_commit":
                evidence = _scheduler_cursor_commit(request, root, paths, cursor_write_possible)
            elif request.operation_kind == "maintenance_flush_event":
                evidence = _maintenance_flush_event(request, root, paths, event_write_possible)
            elif request.operation_kind == "machine_snapshot_publish":
                evidence = _machine_snapshot_publish(request, root, paths, snapshot_write_possible)
            elif request.operation_kind == "registration_renew":
                evidence = _registration_renew(request, root, paths, registration_write_possible)
            elif request.operation_kind == "authority_service":
                evidence = _authority_service_observe(request, root, paths)
            elif request.operation_kind == "authority_renewal":
                evidence = _authority_renewal(
                    request,
                    root,
                    paths,
                    authority_renewal_write_possible,
                    replay_only=reconcile_authority_renewal,
                    active_executor_epoch=epoch.executor_epoch,
                )
            elif request.operation_kind == "authority_orphan_recovery":
                evidence = _authority_orphan_recovery(
                    request,
                    root,
                    paths,
                    authority_orphan_recovery_write_possible,
                    replay_only=reconcile_authority_orphan_recovery,
                    active_executor_epoch=epoch.executor_epoch,
                )
            elif request.operation_kind == "authority_termination_commit":
                evidence = _authority_termination_commit(
                    request,
                    root,
                    paths,
                    authority_termination_write_possible,
                    replay_only=reconcile_authority_termination_commit,
                    active_executor_epoch=epoch.executor_epoch,
                )
            elif request.operation_kind == "authority_terminal_observe":
                evidence = _authority_terminal_observe(request, root, paths)
            elif request.operation_kind == "authority_terminal_publish":
                evidence = _authority_terminal_publish(
                    request,
                    root,
                    paths,
                    authority_terminal_write_possible,
                    replay_only=reconcile_authority_terminal_publish,
                    active_executor_epoch=epoch.executor_epoch,
                )
            elif request.operation_kind == "authority_running_publish":
                evidence = _authority_running_publish(
                    request,
                    root,
                    paths,
                    authority_running_write_possible,
                )
            else:
                raise ProjectIOProtocolError("worker operation is not implemented by this version.")
    except _ExecutorEpochFenced:
        if (
            request.operation_kind
            in {
                "upgrade_service",
                "submission_control_service",
                "observation_service",
                "notification_service",
                "progress_projection",
                "recovery_source_hold",
                "recovery_admission",
                "recovery_source_release",
                "recovery_group_authority",
                "recovery_capture_transition",
            }
            and upgrade_write_possible[0]
        ):
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_renewal" and authority_renewal_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_orphan_recovery" and authority_orphan_recovery_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                reconcile_authority_orphan_recovery=reconcile_authority_orphan_recovery,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind in {
            "authority_termination_commit",
            "authority_terminal_publish",
        } and (authority_termination_write_possible[0] or authority_terminal_write_possible[0]):
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                reconcile_authority_termination_commit=reconcile_authority_termination_commit,
                reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_running_publish" and authority_running_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if (
            request.operation_kind
            in {"activation_consumer_register", "activation_consumer_ack", "activation_consumer_retire"}
            and activation_consumer_write_possible[0]
        ):
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "registration_renew" and registration_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "machine_snapshot_publish" and snapshot_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "maintenance_flush_event" and event_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_due_offer" and due_offer_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_ready_index_build" and ready_index_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "maintenance_descriptor_advance" and descriptor_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                shared_write_possible=True,
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_launch_authorize" and launch_authorization_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_claim" and claim_cursor_commit_started[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_cursor_commit" and cursor_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_scheduler_cursor_commit_failed",
                {},
            )
            return 0 if published else 1
        published = _publish_result(
            root,
            paths,
            request,
            process,
            "fenced",
            "project_io_executor_epoch_fenced",
            {},
            reconcile_authority_termination_commit=reconcile_authority_termination_commit,
            reconcile_authority_orphan_recovery=reconcile_authority_orphan_recovery,
            reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
            reconciliation_epoch=epoch.executor_epoch,
        )
        return 0 if published else 1
    except _AuthorityReplayBlocked:
        if authority_termination_write_possible[0] or authority_terminal_write_possible[0]:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
                reconcile_authority_termination_commit=reconcile_authority_termination_commit,
                reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=True,
            )
        else:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "completed",
                None,
                _authority_replay_blocked_evidence(request),
                reconcile_authority_termination_commit=reconcile_authority_termination_commit,
                reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
                reconciliation_epoch=epoch.executor_epoch,
            )
        return 0 if published else 1
    except Exception as exc:
        if request.operation_kind in {
            "upgrade_service",
            "submission_control_service",
            "observation_service",
            "notification_service",
            "progress_projection",
            "recovery_source_hold",
            "recovery_admission",
            "recovery_source_release",
            "recovery_group_authority",
            "recovery_capture_transition",
        }:
            write_possible = upgrade_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else f"project_io_{request.operation_kind}_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_renewal":
            status = "outcome_unknown" if authority_renewal_write_possible[0] else "retryable_error"
            reason_code = (
                "project_io_outcome_unknown"
                if authority_renewal_write_possible[0]
                else "project_io_authority_renewal_failed"
            )
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                {} if authority_renewal_write_possible[0] else {"exception_type": _safe_exception_type(exc)},
                reconcile_authority_renewal=reconcile_authority_renewal,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=authority_renewal_write_possible[0],
            )
            return 0 if published else 1
        if request.operation_kind == "authority_orphan_recovery":
            write_possible = authority_orphan_recovery_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else "project_io_authority_orphan_recovery_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                reconcile_authority_orphan_recovery=reconcile_authority_orphan_recovery,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind in {
            "authority_termination_commit",
            "authority_terminal_publish",
        }:
            write_possible = authority_termination_write_possible[0] or authority_terminal_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown"
                if write_possible
                else (
                    "project_io_authority_termination_commit_failed"
                    if request.operation_kind == "authority_termination_commit"
                    else "project_io_authority_terminal_publish_failed"
                ),
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                reconcile_authority_termination_commit=reconcile_authority_termination_commit,
                reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
                reconciliation_epoch=epoch.executor_epoch,
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_running_publish":
            write_possible = authority_running_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else "project_io_authority_running_publish_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "authority_terminal_observe":
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "retryable_error",
                "project_io_authority_terminal_observe_failed",
                {"exception_type": _safe_exception_type(exc)},
            )
            return 0 if published else 1
        if request.operation_kind == "maintenance_flush_event":
            if event_write_possible[0]:
                status = "outcome_unknown"
                reason_code = "project_io_outcome_unknown"
                evidence_payload: dict[str, Any] = {}
            else:
                status = "retryable_error"
                reason_code = "project_io_maintenance_flush_event_failed"
                evidence_payload = {"exception_type": _safe_exception_type(exc)}
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                evidence_payload,
                shared_write_possible=event_write_possible[0],
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_due_offer":
            write_possible = due_offer_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else "project_io_scheduler_due_offer_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_ready_index_build":
            write_possible = ready_index_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else "project_io_scheduler_ready_index_build_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "maintenance_descriptor_advance":
            write_possible = descriptor_write_possible[0]
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown" if write_possible else "retryable_error",
                "project_io_outcome_unknown" if write_possible else "project_io_maintenance_descriptor_advance_failed",
                {} if write_possible else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=write_possible,
            )
            return 0 if published else 1
        if request.operation_kind == "machine_snapshot_publish":
            status = "outcome_unknown" if snapshot_write_possible[0] else "retryable_error"
            reason_code = (
                "project_io_outcome_unknown"
                if snapshot_write_possible[0]
                else "project_io_machine_snapshot_publish_failed"
            )
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                {} if snapshot_write_possible[0] else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=snapshot_write_possible[0],
            )
            return 0 if published else 1
        if request.operation_kind == "registration_renew":
            status = "outcome_unknown" if registration_write_possible[0] else "retryable_error"
            reason_code = (
                "project_io_outcome_unknown"
                if registration_write_possible[0]
                else "project_io_registration_renew_failed"
            )
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                {} if registration_write_possible[0] else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=registration_write_possible[0],
            )
            return 0 if published else 1
        if request.operation_kind in {
            "activation_consumer_register",
            "activation_consumer_ack",
            "activation_consumer_retire",
        }:
            status = "outcome_unknown" if activation_consumer_write_possible[0] else "retryable_error"
            reason_code = (
                "project_io_outcome_unknown"
                if activation_consumer_write_possible[0]
                else (
                    "project_io_activation_consumer_register_failed"
                    if request.operation_kind == "activation_consumer_register"
                    else "project_io_activation_consumer_ack_failed"
                    if request.operation_kind == "activation_consumer_ack"
                    else "project_io_activation_consumer_retire_failed"
                )
            )
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                {} if activation_consumer_write_possible[0] else {"exception_type": _safe_exception_type(exc)},
                shared_write_possible=activation_consumer_write_possible[0],
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_launch_authorize":
            if launch_authorization_write_possible[0]:
                status = "outcome_unknown"
                reason_code = "project_io_outcome_unknown"
                evidence_payload: dict[str, Any] = {}
            else:
                status = "retryable_error"
                reason_code = "project_io_scheduler_launch_authorize_failed"
                evidence_payload = {"exception_type": _safe_exception_type(exc)}
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                reason_code,
                evidence_payload,
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_claim":
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "outcome_unknown",
                "project_io_outcome_unknown",
                {},
            )
            return 0 if published else 1
        if request.operation_kind == "scheduler_cursor_commit":
            status = "outcome_unknown" if cursor_write_possible[0] else "retryable_error"
            published = _publish_result(
                root,
                paths,
                request,
                process,
                status,
                "project_io_scheduler_cursor_commit_failed",
                {} if cursor_write_possible[0] else {"exception_type": _safe_exception_type(exc)},
            )
            return 0 if published else 1
        reason_code = {
            "scheduler_observe": "project_io_scheduler_observation_failed",
            "scheduler_primary_probe": "project_io_scheduler_observation_failed",
            "scheduler_quiescence_probe": "project_io_scheduler_observation_failed",
            "scheduler_claim": "project_io_scheduler_claim_failed_before_commit",
            "scheduler_reservation_reconcile": "project_io_scheduler_reservation_reconcile_failed",
            "scheduler_due_offer": "project_io_scheduler_due_offer_failed",
            "scheduler_ready_index_build": "project_io_scheduler_ready_index_build_failed",
            "machine_snapshot_publish": "project_io_machine_snapshot_publish_failed",
            "registration_renew": "project_io_registration_renew_failed",
            "activation_observe": "project_io_activation_observe_failed",
            "activation_consumer_retire": "project_io_activation_consumer_retire_failed",
            "authority_service": "project_io_authority_service_failed",
            "authority_renewal": "project_io_authority_renewal_failed",
            "authority_orphan_recovery": "project_io_authority_orphan_recovery_failed",
            "authority_running_publish": "project_io_authority_running_publish_failed",
            "legacy_capture_read": "project_io_legacy_capture_read_failed",
            "legacy_capture_scan": "project_io_legacy_capture_scan_failed",
        }.get(request.operation_kind, "project_io_binding_validation_failed")
        published = _publish_result(
            root,
            paths,
            request,
            process,
            "retryable_error",
            reason_code,
            {"exception_type": _safe_exception_type(exc)},
        )
    else:
        if fenced:
            published = _publish_result(
                root,
                paths,
                request,
                process,
                "fenced",
                "project_io_executor_epoch_fenced",
                {},
            )
            return 0 if published else 1
        published = _publish_result(
            root,
            paths,
            request,
            process,
            "completed",
            None,
            evidence,
            reconcile_authority_renewal=reconcile_authority_renewal,
            reconcile_authority_orphan_recovery=reconcile_authority_orphan_recovery,
            reconcile_authority_termination_commit=reconcile_authority_termination_commit,
            reconcile_authority_terminal_publish=reconcile_authority_terminal_publish,
            reconciliation_epoch=epoch.executor_epoch,
            shared_write_possible=(
                upgrade_write_possible[0]
                or event_write_possible[0]
                or descriptor_write_possible[0]
                or snapshot_write_possible[0]
                or registration_write_possible[0]
                or activation_consumer_write_possible[0]
                or authority_renewal_write_possible[0]
                or authority_orphan_recovery_write_possible[0]
                or authority_termination_write_possible[0]
                or authority_terminal_write_possible[0]
                or authority_running_write_possible[0]
            ),
        )
    return 0 if published else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--reconcile-fenced-claim", action="store_true")
    parser.add_argument("--reconcile-authority-renewal", action="store_true")
    parser.add_argument("--reconcile-authority-orphan-recovery", action="store_true")
    parser.add_argument("--reconcile-authority-termination-commit", action="store_true")
    parser.add_argument("--reconcile-authority-terminal-publish", action="store_true")
    args = parser.parse_args(argv)
    if not _REQUEST_ID.fullmatch(args.request_id):
        return 2
    try:
        root = Path(args.runtime_root).expanduser().resolve()
        if not root.is_absolute() or str(root) != args.runtime_root:
            return 2
        return _run(
            root,
            args.request_id,
            reconcile_fenced_claim=args.reconcile_fenced_claim,
            reconcile_authority_renewal=args.reconcile_authority_renewal,
            reconcile_authority_orphan_recovery=args.reconcile_authority_orphan_recovery,
            reconcile_authority_termination_commit=args.reconcile_authority_termination_commit,
            reconcile_authority_terminal_publish=args.reconcile_authority_terminal_publish,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
