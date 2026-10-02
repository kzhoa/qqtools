"""Bounded version-1 records for machine-local Project I/O isolation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..runtime.records import validate_identifier
from ..runtime.store import require_json_size
from ..runtime.submission_control_continuation import submission_control_continuation
from .group_service_transport import (
    group_service_advance_evidence,
    group_service_advance_parameters,
    group_service_probe_evidence,
    group_service_probe_parameters,
)
from .primary_probe_transport import validate_probe_state
from .progress_transport import progress_context, progress_evidence, progress_projection
from .recovery_transport import (
    legacy_capture_evidence,
    legacy_capture_parameters,
    legacy_scan_evidence,
    legacy_scan_parameters,
    recovery_admission_evidence,
    recovery_completion_evidence,
    recovery_completion_parameters,
    recovery_generation_parameters,
    source_hold_evidence,
    source_hold_parameters,
)

PROJECT_IO_PROTOCOL_VERSION = 1
PROJECT_IO_CAPACITY = 4
PROJECT_IO_SUPPORTED_HANG_LIMIT = 2
PROJECT_IO_MAX_RECORD_BYTES = 65_536
PROJECT_IO_MAX_RESOLVED_RECORDS = 256
PROJECT_IO_MAX_RESOLVED_BYTES = 8 * 1024 * 1024
PROJECT_IO_OVERDUE_SECONDS = 10.0
PROJECT_IO_STOP_GRACE_SECONDS = 2.0

PROJECT_IO_OPERATIONS = frozenset(
    {
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
    }
)
PROJECT_IO_RESULT_STATUSES = frozenset({"completed", "retryable_error", "outcome_unknown", "fenced"})
PROJECT_IO_PROCESS_STATES = frozenset({"running", "stop_targeted", "exit_unverified"})
PROJECT_IO_ENVELOPES = frozenset({"healthy", "degraded", "exceeded", "unknown"})

_HEX_ID = re.compile(r"^[0-9a-f]{32}$")
_EVENT_ID = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_ID = re.compile(r"^[0-9a-f]{64}$")
_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,95}$")
_REQUEST_FIELDS = frozenset(
    {
        "protocol_version",
        "runtime_id",
        "executor_epoch",
        "request_id",
        "operation_kind",
        "project_id",
        "canonical_shared_root",
        "registration_generation",
        "registry_revision",
        "source_revisions",
        "provisional_offer_id",
        "prepared_at",
        "parameters",
    }
)

JSONScalar = str | int | float | bool | None

_CANDIDATE_FIELDS = frozenset(
    {
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
)
_OFFER_FIELDS = frozenset(
    {
        "offer_id",
        "acquisition_id",
        "reservation_id",
        "executor_epoch",
        "request_id",
        "project_id",
        "shared_root",
        "registration_generation",
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "lane",
        "gpu_ids",
        "cpu_slots",
        "group_name",
        "group_dispatch_epoch",
        "group_worker_set_epoch",
        "worker_state_epoch",
        "worker_scheduling_role",
        "gpu_limit_gpus",
        "admitted_as_borrow",
    }
)
_OBSERVATION_REASONS = frozenset(
    {
        "candidate_ready",
        "no_candidate",
        "ready_index_inactive",
        "candidate_unresolved",
        "dependency_not_ready",
        "placement_rejected",
        "admission_role_mismatch",
        "borrow_admission_unavailable",
        "slice_exhausted",
    }
)
_CLAIM_REASONS = frozenset(
    {
        "claimed",
        "candidate_stale",
        "task_changed",
        "ready_changed",
        "dependency_not_ready",
        "group_changed",
        "admission_role_changed",
        "borrow_admission_unavailable",
        "offer_mismatch",
    }
)
_LAUNCH_AUTHORIZATION_DENIAL_REASONS = frozenset(
    {
        "claim_not_current",
        "attempt_not_current",
        "cleanup_pending",
        "cancellation_pending",
        "group_gate_closed",
        "authority_unavailable",
        "clock_unhealthy",
        "reservation_unavailable",
    }
)
_AUTHORITY_SERVICE_OUTCOMES = frozenset({"observed_current", "observed_stale"})
_AUTHORITY_SERVICE_REASONS = frozenset(
    {
        "task_changed",
        "attempt_changed",
        "reservation_changed",
        "process_changed",
        "termination_pending",
        "attempt_not_running",
        "binding_fence",
        "renewal_plan_expired",
    }
)
_AUTHORITY_RENEWAL_OUTCOMES = frozenset({"renewed", "not_required", "observed_stale", "termination_requested"})
_AUTHORITY_ORPHAN_RECOVERY_OUTCOMES = frozenset({"recovered", "already_recovered", "stale"})
_PROJECT_IO_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})
_PROJECT_IO_TERMINAL_MODES = frozenset({"active", "detached_orphan"})
_PROJECT_IO_STALE_REASONS = frozenset(
    {
        "task_missing",
        "attempt_missing",
        "task_identity_mismatch",
        "attempt_identity_mismatch",
        "claim_not_current",
        "reservation_mismatch",
        "process_identity_mismatch",
        "task_phase_mismatch",
        "attempt_phase_mismatch",
        "decision_conflict",
        "source_revision_mismatch",
        "terminal_conflict",
        "binding_fence",
        "executor_epoch_fence",
        "recovery_clock_unhealthy",
        "recovery_process_not_alive",
        "recovery_worker_state_invalid",
        "recovery_termination_blocked",
        "recovery_reservation_mismatch",
        "recovery_identity_mismatch",
        "recovery_target_mismatch",
    }
)


def authority_terminal_transition_digest(
    *,
    mode: str,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    machine_name: str,
    reservation_id: str | None,
    process_identity: Mapping[str, Any],
    phase: str,
    reason: str,
    exit_code: int | None,
    termination_result: str | None,
) -> str:
    """Hash the exact semantic terminal transition in canonical JSON form."""
    semantic_fields = {
        "mode": mode,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "machine_name": machine_name,
        "reservation_id": reservation_id,
        "process_identity": dict(process_identity),
        "phase": phase,
        "reason": reason,
        "exit_code": exit_code,
        "termination_result": termination_result,
    }
    encoded = json.dumps(
        semantic_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_copy(item) for item in value]
    return value


def _require_record_size(
    value: dict[str, Any],
    record_type: str,
    *,
    max_bytes: int = PROJECT_IO_MAX_RECORD_BYTES,
) -> None:
    try:
        require_json_size(value, max_bytes=max_bytes, record_type=record_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_type} exceeds its encoded JSON limit or is not JSON data.") from exc


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has missing or unknown fields.")


def _require_envelope(value: object, envelope: str, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    if set(record) != {envelope}:
        raise ValueError(f"{label} must contain only the {envelope!r} envelope.")
    return _require_mapping(record[envelope], f"{label}.{envelope}")


def _require_text(value: object, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is invalid.")
    return value


def _require_identifier(value: object, label: str, *, maximum: int = 128) -> str:
    text = _require_text(value, label, maximum=maximum)
    try:
        validate_identifier(text, label)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid.") from exc
    return text


def _require_timestamp(value: object, label: str) -> str:
    text = _require_text(value, label, maximum=40)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC.")
    return text


def _require_nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is invalid.")
    return value


def _validate_supervised_process_identity(value: object, label: str) -> dict[str, int | None]:
    process_identity = dict(_require_mapping(value, label))
    _require_exact_keys(
        process_identity,
        frozenset(
            {
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            }
        ),
        label,
    )
    for field, item in process_identity.items():
        if item is not None:
            process_identity[field] = _require_nonnegative_int(
                item, f"{label}.{field}", positive=field in {"wrapper_pid", "process_group_id"}
            )
    return process_identity


def _bounded_json_value(value: object, label: str, *, depth: int = 0) -> Any:
    """Copy a small JSON value while rejecting objects with unbounded shape."""
    if depth > 8:
        raise ValueError(f"{label} is nested too deeply.")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > 9_223_372_036_854_775_807:
            raise ValueError(f"{label} integer is out of range.")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} number must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > 4096 or "\x00" in value:
            raise ValueError(f"{label} text is invalid.")
        return value
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise ValueError(f"{label} object has too many fields.")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128 or "\x00" in key:
                raise ValueError(f"{label} object key is invalid.")
            normalized[key] = _bounded_json_value(item, label, depth=depth + 1)
        return normalized
    if isinstance(value, list | tuple):
        if len(value) > 4096:
            raise ValueError(f"{label} array has too many items.")
        return [_bounded_json_value(item, label, depth=depth + 1) for item in value]
    raise ValueError(f"{label} must contain only bounded JSON data.")


def _validate_machine_reservation(value: object) -> dict[str, Any]:
    """Project only bounded advisory identity fields from one local reservation."""
    reservation = _require_mapping(value, "machine snapshot reservation")
    reservation_id = _require_identifier(reservation.get("reservation_id"), "reservation.reservation_id")
    project_id_value = reservation.get("project_id")
    project_id = (
        _require_identifier(project_id_value, "reservation.project_id") if project_id_value is not None else None
    )
    gpu_ids_value = reservation.get("gpu_ids", [])
    if (
        not isinstance(gpu_ids_value, list | tuple)
        or len(gpu_ids_value) > 4096
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids_value)
        or len(set(gpu_ids_value)) != len(gpu_ids_value)
    ):
        raise ValueError("reservation.gpu_ids is invalid.")
    state = reservation.get("state")
    if state not in ("active", "provisional"):
        raise ValueError("reservation.state is invalid.")
    nullable_identifiers: dict[str, str | None] = {}
    for field in ("group_name", "machine_name", "task_id", "attempt_id"):
        item = reservation.get(field)
        nullable_identifiers[field] = _require_identifier(item, f"reservation.{field}") if item is not None else None
    admission_value = reservation.get("admission")
    admission = None if admission_value is None else _bounded_json_value(admission_value, "reservation.admission")
    if admission is not None and not isinstance(admission, dict):
        raise ValueError("reservation.admission must be an object or null.")
    return {
        "reservation_id": reservation_id,
        "project_id": project_id,
        **nullable_identifiers,
        "gpu_ids": list(gpu_ids_value),
        "state": state,
        "admission": admission,
    }


def _validate_claim_identity(value: object) -> dict[str, Any]:
    identity = dict(_require_mapping(value, "claim_identity"))
    _require_exact_keys(
        identity,
        frozenset({"task_id", "attempt_id", "attempt_number", "fencing_token", "reservation_id"}),
        "claim_identity",
    )
    for field in ("task_id", "attempt_id", "reservation_id"):
        identity[field] = _require_identifier(identity[field], f"claim_identity.{field}")
    identity["attempt_number"] = _require_nonnegative_int(
        identity["attempt_number"], "claim_identity.attempt_number", positive=True
    )
    identity["fencing_token"] = _require_nonnegative_int(
        identity["fencing_token"], "claim_identity.fencing_token", positive=True
    )
    if identity["attempt_id"] != f"{identity['task_id']}-attempt-{identity['attempt_number']}":
        raise ValueError("claim_identity attempt identity is inconsistent.")
    return identity


def _validate_reservation_identity(value: object) -> dict[str, Any]:
    identity = dict(_require_mapping(value, "reservation_identity"))
    fields = frozenset(
        {
            "reservation_id",
            "acquisition_id",
            "project_id",
            "task_id",
            "attempt_id",
            "fencing_token",
            "gpu_ids",
            "cpu_slots",
            "shared_root",
            "registration_generation",
            "executor_epoch",
            "executor_request_id",
        }
    )
    _require_exact_keys(identity, fields, "reservation_identity")
    for field in ("reservation_id", "acquisition_id", "project_id", "task_id", "attempt_id"):
        identity[field] = _require_identifier(identity[field], f"reservation_identity.{field}")
    identity["fencing_token"] = _require_nonnegative_int(
        identity["fencing_token"], "reservation_identity.fencing_token", positive=True
    )
    gpu_ids = identity["gpu_ids"]
    cpu_slots = identity["cpu_slots"]
    if gpu_ids is not None:
        if (
            not isinstance(gpu_ids, list | tuple)
            or not gpu_ids
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)
            or cpu_slots is not None
        ):
            raise ValueError("reservation_identity GPU resources are invalid.")
        identity["gpu_ids"] = list(gpu_ids)
    else:
        identity["cpu_slots"] = _require_nonnegative_int(cpu_slots, "reservation_identity.cpu_slots", positive=True)
    shared_root = identity["shared_root"]
    if not isinstance(shared_root, str) or not shared_root or len(shared_root) > 4096:
        raise ValueError("reservation_identity.shared_root is invalid.")
    owner_fields = ("registration_generation", "executor_epoch", "executor_request_id")
    owner_values = tuple(identity[field] for field in owner_fields)
    if any(value is None for value in owner_values):
        if not all(value is None for value in owner_values):
            raise ValueError("reservation_identity executor owner is incomplete.")
    else:
        for field in owner_fields:
            identity[field] = _require_identifier(identity[field], f"reservation_identity.{field}")
    return identity


def _validate_candidate(value: object) -> dict[str, Any]:
    candidate = dict(_require_mapping(value, "scheduler candidate"))
    _require_exact_keys(candidate, _CANDIDATE_FIELDS, "scheduler candidate")
    candidate["task_id"] = _require_identifier(candidate["task_id"], "candidate.task_id")
    candidate["attempt_id"] = _require_identifier(candidate["attempt_id"], "candidate.attempt_id")
    candidate["ready_identity"] = _require_text(candidate["ready_identity"], "candidate.ready_identity", maximum=256)
    candidate["ready_scope"] = _require_text(candidate["ready_scope"], "candidate.ready_scope", maximum=16)
    if candidate["ready_scope"] not in {"home", "shared"}:
        raise ValueError("candidate.ready_scope is invalid.")
    if candidate["lane"] not in {"gpu", "cpu"}:
        raise ValueError("candidate.lane is invalid.")
    if candidate["admission_role"] not in {"primary", "borrow"}:
        raise ValueError("candidate.admission_role is invalid.")
    for field in (
        "task_revision",
        "ready_generation",
        "ready_revision",
        "catalog_revision",
        "attempt_number",
        "fencing_token",
    ):
        candidate[field] = _require_nonnegative_int(
            candidate[field],
            f"candidate.{field}",
            positive=field in {"ready_generation", "attempt_number", "fencing_token"},
        )
    candidate["catalog_page"] = _require_nonnegative_int(candidate["catalog_page"], "candidate.catalog_page")
    candidate["partition"] = _require_identifier(candidate["partition"], "candidate.partition")
    candidate["marker_name"] = _require_text(candidate["marker_name"], "candidate.marker_name", maximum=256)
    candidate["home_machine"] = _require_identifier(candidate["home_machine"], "candidate.home_machine")
    for field in ("requested_gpus", "requested_cpus"):
        candidate[field] = _require_nonnegative_int(candidate[field], f"candidate.{field}")
        if candidate[field] > 4096:
            raise ValueError(f"candidate.{field} is too large.")
    if (candidate["lane"] == "gpu") != (candidate["requested_cpus"] == 0):
        raise ValueError("candidate resource counts do not match its lane.")
    if (candidate["lane"] == "cpu") != (candidate["requested_gpus"] == 0):
        raise ValueError("candidate resource counts do not match its lane.")
    if candidate["group_name"] is not None:
        candidate["group_name"] = _require_identifier(candidate["group_name"], "candidate.group_name")
    for field in ("group_revision", "group_dispatch_epoch", "group_worker_set_epoch", "worker_state_epoch"):
        item = candidate[field]
        if item is not None:
            candidate[field] = _require_nonnegative_int(item, f"candidate.{field}")
    role = candidate["worker_scheduling_role"]
    if role is not None and role not in {"primary", "borrow"}:
        raise ValueError("candidate.worker_scheduling_role is invalid.")
    limit = candidate["gpu_limit_gpus"]
    if limit is not None:
        candidate["gpu_limit_gpus"] = _require_nonnegative_int(limit, "candidate.gpu_limit_gpus", positive=True)
    if candidate["attempt_id"] != f"{candidate['task_id']}-attempt-{candidate['attempt_number']}":
        raise ValueError("candidate attempt identity is inconsistent.")
    if candidate["ready_identity"] != f"{candidate['task_id']}.{candidate['ready_generation']}":
        raise ValueError("candidate ready identity is inconsistent.")
    if candidate["marker_name"] != f"{candidate['ready_identity']}.json":
        raise ValueError("candidate ready marker name is inconsistent.")
    if candidate["group_name"] is None and any(
        candidate[field] is not None
        for field in (
            "group_revision",
            "group_dispatch_epoch",
            "group_worker_set_epoch",
            "worker_state_epoch",
            "worker_scheduling_role",
            "gpu_limit_gpus",
        )
    ):
        raise ValueError("ungrouped candidate contains Group admission evidence.")
    if candidate["group_name"] is not None and any(
        candidate[field] is None
        for field in (
            "group_revision",
            "group_dispatch_epoch",
            "group_worker_set_epoch",
            "worker_state_epoch",
            "worker_scheduling_role",
        )
    ):
        raise ValueError("grouped candidate lacks Group admission evidence.")
    if candidate["group_name"] is None and candidate["admission_role"] != "primary":
        raise ValueError("ungrouped candidate must use primary admission.")
    if candidate["group_name"] is not None and candidate["worker_scheduling_role"] != candidate["admission_role"]:
        raise ValueError("candidate admission role differs from its Worker role.")
    return candidate


def _validate_cursor_position(value: object, label: str) -> dict[str, Any]:
    position = dict(_require_mapping(value, label))
    _require_exact_keys(position, frozenset({"catalog_page", "partition", "after_name", "revision"}), label)
    page = position["catalog_page"]
    if page is not None:
        page = _require_nonnegative_int(page, f"{label}.catalog_page")
        if page > 1_000_000_000_000_000:
            raise ValueError(f"{label}.catalog_page is too large.")
    partition = position["partition"]
    if partition is not None:
        partition = _require_identifier(partition, f"{label}.partition", maximum=256)
    after_name = position["after_name"]
    if after_name is not None:
        after_name = _require_text(after_name, f"{label}.after_name", maximum=256)
    revision = _require_nonnegative_int(position["revision"], f"{label}.revision")
    return {"catalog_page": page, "partition": partition, "after_name": after_name, "revision": revision}


def _validate_observation_cursor(value: object, request: ProjectIORequest | None = None) -> dict[str, Any]:
    cursor = dict(_require_mapping(value, "scheduler_observe cursor"))
    _require_exact_keys(cursor, frozenset({"namespace", "routes"}), "scheduler_observe cursor")
    namespace = _require_identifier(cursor["namespace"], "cursor.namespace")
    if request is not None:
        expected_namespace = (
            request.parameters["cursor_namespace"]
            if request.operation_kind == "scheduler_observe"
            else request.parameters["cursor"]["namespace"]
        )
        if namespace != expected_namespace:
            raise ValueError("scheduler cursor namespace differs from its request.")
    routes_value = _require_mapping(cursor["routes"], "scheduler_observe cursor routes")
    _require_exact_keys(routes_value, frozenset({"home", "shared"}), "scheduler_observe cursor routes")
    routes: dict[str, Any] = {}
    for scope in ("home", "shared"):
        positions = dict(_require_mapping(routes_value[scope], f"cursor.routes.{scope}"))
        _require_exact_keys(positions, frozenset({"observed", "next"}), f"cursor.routes.{scope}")
        observed = _validate_cursor_position(positions["observed"], f"cursor.routes.{scope}.observed")
        next_position = _validate_cursor_position(positions["next"], f"cursor.routes.{scope}.next")
        if next_position["revision"] < observed["revision"]:
            raise ValueError(f"cursor.routes.{scope}.next revision precedes observed revision.")
        position_fields = ("catalog_page", "partition", "after_name")
        if (
            any(next_position[field] != observed[field] for field in position_fields)
            and next_position["revision"] <= observed["revision"]
        ):
            raise ValueError(f"cursor.routes.{scope}.next position changed without advancing its revision.")
        routes[scope] = {"observed": observed, "next": next_position}
    return {"namespace": namespace, "routes": routes}


def _validate_offer(value: object) -> dict[str, Any]:
    offer = dict(_require_mapping(value, "executor-owned offer"))
    _require_exact_keys(offer, _OFFER_FIELDS, "executor-owned offer")
    for field in ("offer_id", "acquisition_id", "reservation_id"):
        identifier = _require_text(offer[field], f"offer.{field}", maximum=16)
        if not re.fullmatch(r"[0-9a-f]{16}", identifier):
            raise ValueError(f"offer.{field} must be a lowercase 16-character hexadecimal ID.")
    for field in ("executor_epoch", "request_id"):
        identifier = _require_text(offer[field], f"offer.{field}", maximum=32)
        if not _HEX_ID.fullmatch(identifier):
            raise ValueError(f"offer.{field} must be a lowercase 32-character hexadecimal ID.")
    for field in ("project_id", "registration_generation", "task_id", "attempt_id"):
        offer[field] = _require_identifier(offer[field], f"offer.{field}")
    offer["shared_root"] = _require_text(offer["shared_root"], "offer.shared_root", maximum=4096)
    root = Path(offer["shared_root"])
    if not root.is_absolute() or os.path.normpath(offer["shared_root"]) != offer["shared_root"]:
        raise ValueError("offer.shared_root must already be absolute and normalized.")
    if offer["lane"] not in {"gpu", "cpu"}:
        raise ValueError("offer.lane is invalid.")
    offer["attempt_number"] = _require_nonnegative_int(offer["attempt_number"], "offer.attempt_number", positive=True)
    offer["fencing_token"] = _require_nonnegative_int(offer["fencing_token"], "offer.fencing_token", positive=True)
    gpu_ids = offer["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or len(gpu_ids) > 4096
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("offer.gpu_ids is invalid.")
    offer["cpu_slots"] = _require_nonnegative_int(offer["cpu_slots"], "offer.cpu_slots")
    if offer["offer_id"] != offer["reservation_id"]:
        raise ValueError("offer_id must equal reservation_id in protocol version 1.")
    if (offer["lane"] == "gpu") != (offer["cpu_slots"] == 0) or (offer["lane"] == "cpu") != (not gpu_ids):
        raise ValueError("offer resource counts do not match its lane.")
    if offer["attempt_id"] != f"{offer['task_id']}-attempt-{offer['attempt_number']}":
        raise ValueError("offer attempt identity is inconsistent.")
    if offer["group_name"] is not None:
        offer["group_name"] = _require_identifier(offer["group_name"], "offer.group_name")
    for field in ("group_dispatch_epoch", "group_worker_set_epoch", "worker_state_epoch"):
        item = offer[field]
        if item is not None:
            offer[field] = _require_nonnegative_int(item, f"offer.{field}")
    role = offer["worker_scheduling_role"]
    if role is not None and role not in {"primary", "borrow"}:
        raise ValueError("offer.worker_scheduling_role is invalid.")
    limit = offer["gpu_limit_gpus"]
    if limit is not None:
        offer["gpu_limit_gpus"] = _require_nonnegative_int(limit, "offer.gpu_limit_gpus", positive=True)
    if type(offer["admitted_as_borrow"]) is not bool:
        raise ValueError("offer.admitted_as_borrow must be a boolean.")
    if offer["admitted_as_borrow"] != (role == "borrow"):
        raise ValueError("offer borrow admission differs from its Worker role.")
    return offer


def _request_parameters(operation_kind: str, value: object) -> dict[str, Any]:
    parameters = dict(_require_mapping(value, f"{operation_kind} parameters"))
    if operation_kind == "validate_binding":
        _require_exact_keys(parameters, frozenset({"machine_name"}), "validate_binding parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "registration_renew":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "renewal_horizon_seconds"}),
            "registration_renew parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        horizon = parameters["renewal_horizon_seconds"]
        if (
            type(horizon) not in {int, float}
            or (type(horizon) is float and not math.isfinite(horizon))
            or horizon < 0
            or horizon > 86_400
        ):
            raise ValueError("renewal_horizon_seconds is invalid.")
    elif operation_kind == "upgrade_service":
        _require_exact_keys(parameters, frozenset({"machine_name"}), "upgrade_service parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind in {"observation_service", "notification_service", "recovery_admission"}:
        _require_exact_keys(parameters, frozenset({"machine_name"}), f"{operation_kind} parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "group_service_probe":
        parameters = group_service_probe_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "group_service_advance":
        parameters = group_service_advance_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "recovery_source_hold":
        parameters = source_hold_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind in {"recovery_source_release", "recovery_group_authority"}:
        parameters = recovery_completion_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "recovery_capture_transition":
        parameters = recovery_generation_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "legacy_capture_read":
        parameters = legacy_capture_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "legacy_capture_scan":
        parameters = legacy_scan_parameters(parameters)
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "progress_projection":
        _require_exact_keys(parameters, frozenset({"machine_name", "context", "projection"}), "progress parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["context"] = progress_context(parameters["context"])
        if parameters["context"]["machine_name"] != parameters["machine_name"]:
            raise ValueError("progress producer belongs to another machine")
    elif operation_kind == "submission_control_service":
        _require_exact_keys(
            parameters, frozenset({"machine_name", "continuation"}), "submission_control_service parameters"
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        if not isinstance(parameters["continuation"], Mapping):
            raise ValueError("Submission-control continuation must be an object.")
        parameters["continuation"] = submission_control_continuation(parameters["continuation"])
    elif operation_kind == "activation_observe":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "replay_epoch", "replay_sequence"}),
            "activation_observe parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        replay_epoch = parameters["replay_epoch"]
        if replay_epoch is not None:
            replay_epoch = _validate_activation_epoch(replay_epoch, "activation_observe.replay_epoch")
        replay_sequence = _require_nonnegative_int(
            parameters["replay_sequence"],
            "activation_observe.replay_sequence",
        )
        if replay_epoch is None and replay_sequence != 0:
            raise ValueError("activation_observe replay_sequence requires replay_epoch.")
        parameters["replay_epoch"] = replay_epoch
        parameters["replay_sequence"] = replay_sequence
    elif operation_kind == "activation_consumer_register":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "process_fence"}),
            "activation_consumer_register parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["process_fence"] = _require_identifier(parameters["process_fence"], "process_fence")
    elif operation_kind == "activation_consumer_ack":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "process_fence", "epoch", "sequence", "reconstructed_floor", "require_current"}),
            "activation_consumer_ack parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["process_fence"] = _require_identifier(parameters["process_fence"], "process_fence")
        parameters["epoch"] = _validate_activation_epoch(parameters["epoch"], "activation_consumer_ack.epoch")
        sequence = _require_nonnegative_int(
            parameters["sequence"],
            "activation_consumer_ack.sequence",
            positive=True,
        )
        floor = parameters["reconstructed_floor"]
        if floor is not None:
            floor = _require_nonnegative_int(
                floor,
                "activation_consumer_ack.reconstructed_floor",
                positive=True,
            )
            if floor > sequence:
                raise ValueError("activation_consumer_ack.reconstructed_floor exceeds sequence.")
        parameters["reconstructed_floor"] = floor
        if type(parameters["require_current"]) is not bool:
            raise ValueError("activation_consumer_ack.require_current must be a boolean.")
    elif operation_kind == "activation_consumer_retire":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name"}),
            "activation_consumer_retire parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "authority_service":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "service_action",
                    "task_id",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "process_identity",
                }
            ),
            "authority_service parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        if parameters["service_action"] != "observe_current_attempt":
            raise ValueError("authority_service service_action is unsupported.")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_service Attempt identity is inconsistent.")
        process_identity = dict(_require_mapping(parameters["process_identity"], "process_identity"))
        _require_exact_keys(
            process_identity,
            frozenset(
                {
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                }
            ),
            "authority_service process_identity",
        )
        for field in ("wrapper_pid", "process_group_id"):
            value = process_identity[field]
            if value is not None:
                process_identity[field] = _require_nonnegative_int(value, f"process_identity.{field}", positive=True)
        for field in ("wrapper_start_time_ticks", "process_group_start_time_ticks"):
            value = process_identity[field]
            if value is not None:
                process_identity[field] = _require_nonnegative_int(value, f"process_identity.{field}")
        parameters["process_identity"] = process_identity
    elif operation_kind == "authority_renewal":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "task_id",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "process_identity",
                }
            ),
            "authority_renewal parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_renewal Attempt identity is inconsistent.")
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        process_identity = dict(_require_mapping(parameters["process_identity"], "process_identity"))
        _require_exact_keys(
            process_identity,
            frozenset(
                {
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                }
            ),
            "authority_renewal process_identity",
        )
        for field in ("wrapper_pid", "process_group_id"):
            value = process_identity[field]
            if value is not None:
                process_identity[field] = _require_nonnegative_int(value, f"process_identity.{field}", positive=True)
        for field in ("wrapper_start_time_ticks", "process_group_start_time_ticks"):
            value = process_identity[field]
            if value is not None:
                process_identity[field] = _require_nonnegative_int(value, f"process_identity.{field}")
        parameters["process_identity"] = process_identity
    elif operation_kind == "authority_orphan_recovery":
        parameters.setdefault("replay_only", False)
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "task_id",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "process_identity",
                    "binding_signature",
                    "replay_only",
                }
            ),
            "authority_orphan_recovery parameters",
        )
        if type(parameters["replay_only"]) is not bool:
            raise ValueError("authority_orphan_recovery replay_only must be a boolean.")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_orphan_recovery Attempt identity is inconsistent.")
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        parameters["process_identity"] = _validate_supervised_process_identity(
            parameters["process_identity"], "authority_orphan_recovery process_identity"
        )
        signature = parameters["binding_signature"]
        if (
            not isinstance(signature, list)
            or not 1 <= len(signature) <= 16
            or any(not isinstance(item, str) or not item or len(item) > 4096 or "\x00" in item for item in signature)
        ):
            raise ValueError("authority_orphan_recovery binding_signature is invalid.")
        parameters["binding_signature"] = list(signature)
    elif operation_kind == "authority_termination_commit":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
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
                }
            ),
            "authority_termination_commit parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_termination_commit Attempt identity is inconsistent.")
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        parameters["decision_id"] = _require_identifier(parameters["decision_id"], "decision_id")
        parameters["decision_token"] = _require_nonnegative_int(
            parameters["decision_token"], "decision_token", positive=True
        )
        if parameters["decision_token"] != parameters["fencing_token"]:
            raise ValueError("authority_termination_commit decision_token differs from fencing_token.")
        parameters["authority_outcome"] = _require_text(
            parameters["authority_outcome"], "authority_outcome", maximum=128
        )
        parameters["reason"] = _require_text(parameters["reason"], "reason", maximum=256)
        parameters["process_identity"] = _validate_supervised_process_identity(
            parameters["process_identity"], "authority_termination_commit process_identity"
        )
    elif operation_kind == "authority_terminal_observe":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "task_id",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "process_identity",
                    "mode",
                }
            ),
            "authority_terminal_observe parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_terminal_observe Attempt identity is inconsistent.")
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        parameters["process_identity"] = _validate_supervised_process_identity(
            parameters["process_identity"], "authority_terminal_observe process_identity"
        )
        if parameters["mode"] not in _PROJECT_IO_TERMINAL_MODES:
            raise ValueError("authority_terminal_observe mode is invalid.")
    elif operation_kind == "authority_terminal_publish":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
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
                    "transition_digest",
                }
            ),
            "authority_terminal_publish parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_terminal_publish Attempt identity is inconsistent.")
        reservation_id = parameters["reservation_id"]
        if reservation_id is not None:
            parameters["reservation_id"] = _require_identifier(reservation_id, "reservation_id")
        parameters["process_identity"] = _validate_supervised_process_identity(
            parameters["process_identity"], "authority_terminal_publish process_identity"
        )
        if parameters["mode"] not in _PROJECT_IO_TERMINAL_MODES:
            raise ValueError("authority_terminal_publish mode is invalid.")
        if parameters["phase"] not in _PROJECT_IO_TERMINAL_PHASES:
            raise ValueError("authority_terminal_publish phase is invalid.")
        parameters["reason"] = _require_text(parameters["reason"], "reason", maximum=256)
        exit_code = parameters["exit_code"]
        if exit_code is not None and type(exit_code) is not int:
            raise ValueError("authority_terminal_publish exit_code must be an integer or null.")
        termination_result = parameters["termination_result"]
        if termination_result is not None:
            parameters["termination_result"] = _require_text(termination_result, "termination_result", maximum=128)
        digest = _require_text(parameters["transition_digest"], "transition_digest", maximum=64)
        if not _SHA256_DIGEST.fullmatch(digest):
            raise ValueError("transition_digest must be a lowercase 64-character hexadecimal digest.")
        parameters["transition_digest"] = digest
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
        if digest != computed_digest:
            raise ValueError("transition_digest differs from the canonical terminal transition.")
    elif operation_kind == "authority_running_publish":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "task_id",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "process_identity",
                    "process_created_at",
                }
            ),
            "authority_running_publish parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["task_id"] = _require_identifier(parameters["task_id"], "task_id")
        parameters["attempt_id"] = _require_identifier(parameters["attempt_id"], "attempt_id")
        parameters["attempt_number"] = _require_nonnegative_int(
            parameters["attempt_number"], "attempt_number", positive=True
        )
        parameters["fencing_token"] = _require_nonnegative_int(
            parameters["fencing_token"], "fencing_token", positive=True
        )
        parameters["reservation_id"] = _require_identifier(parameters["reservation_id"], "reservation_id")
        if parameters["attempt_id"] != f"{parameters['task_id']}-attempt-{parameters['attempt_number']}":
            raise ValueError("authority_running_publish Attempt identity is inconsistent.")
        parameters["process_identity"] = _validate_supervised_process_identity(
            parameters["process_identity"], "authority_running_publish process_identity"
        )
        parameters["process_created_at"] = _require_timestamp(parameters["process_created_at"], "process_created_at")
    elif operation_kind == "scheduler_due_offer":
        _require_exact_keys(parameters, frozenset({"machine_name"}), "scheduler_due_offer parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "scheduler_ready_index_build":
        _require_exact_keys(parameters, frozenset({"machine_name"}), "scheduler_ready_index_build parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "maintenance_descriptor_advance":
        _require_exact_keys(parameters, frozenset({"machine_name"}), "maintenance_descriptor_advance parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
    elif operation_kind == "scheduler_quiescence_probe":
        _require_exact_keys(
            parameters, frozenset({"machine_name", "probe_state"}), "scheduler_quiescence_probe parameters"
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["probe_state"] = validate_probe_state(parameters["probe_state"])
        if any(route["recheck"] is not None for route in parameters["probe_state"]["routes"].values()):
            raise ValueError("scheduler quiescence probe cannot carry primary dependency rechecks.")
    elif operation_kind == "scheduler_primary_probe":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "lane",
                    "round_id",
                    "phase",
                    "capacity_digest",
                    "visible_capacity",
                    "free_capacity",
                    "group_gpu_usage",
                    "probe_state",
                }
            ),
            "scheduler_primary_probe parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        if parameters["lane"] not in {"gpu", "cpu"} or parameters["phase"] not in {"scan", "verify"}:
            raise ValueError("primary probe lane or phase is invalid.")
        if not isinstance(parameters["round_id"], str) or re.fullmatch(r"[0-9a-f]{32}", parameters["round_id"]) is None:
            raise ValueError("primary probe round_id is invalid.")
        if (
            not isinstance(parameters["capacity_digest"], str)
            or re.fullmatch(r"[0-9a-f]{64}", parameters["capacity_digest"]) is None
        ):
            raise ValueError("primary probe capacity_digest is invalid.")
        for field in ("visible_capacity", "free_capacity"):
            parameters[field] = _require_nonnegative_int(parameters[field], field)
            if parameters[field] > 4096:
                raise ValueError("primary probe capacity exceeds its bound.")
        if parameters["free_capacity"] > parameters["visible_capacity"]:
            raise ValueError("primary probe free capacity exceeds visible capacity.")
        usage = dict(_require_mapping(parameters["group_gpu_usage"], "group_gpu_usage"))
        if len(usage) > 256:
            raise ValueError("primary probe group usage exceeds its bound.")
        for key, count in usage.items():
            _require_identifier(key, "group_gpu_usage group")
            if _require_nonnegative_int(count, "group_gpu_usage count") > 4096:
                raise ValueError("primary probe group usage count exceeds its bound.")
        parameters["group_gpu_usage"] = usage
        parameters["probe_state"] = validate_probe_state(parameters["probe_state"])
    elif operation_kind == "scheduler_observe":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "lane", "admission_role", "cursor_namespace"}),
            "scheduler_observe parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        if parameters["lane"] not in {"gpu", "cpu"}:
            raise ValueError("scheduler_observe lane is invalid.")
        if parameters["admission_role"] not in {"primary", "borrow"}:
            raise ValueError("scheduler_observe admission_role is invalid.")
        parameters["cursor_namespace"] = _require_identifier(parameters["cursor_namespace"], "cursor_namespace")
    elif operation_kind == "scheduler_cursor_commit":
        _require_exact_keys(parameters, frozenset({"machine_name", "cursor"}), "scheduler_cursor_commit parameters")
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["cursor"] = _validate_observation_cursor(parameters["cursor"])
    elif operation_kind == "scheduler_claim":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "lane", "admission_role", "candidate", "offer", "cursor"}),
            "scheduler_claim parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        if parameters["lane"] not in {"gpu", "cpu"}:
            raise ValueError("scheduler_claim lane is invalid.")
        if parameters["admission_role"] not in {"primary", "borrow"}:
            raise ValueError("scheduler_claim admission_role is invalid.")
        parameters["candidate"] = _validate_candidate(parameters["candidate"])
        parameters["offer"] = _validate_offer(parameters["offer"])
        parameters["cursor"] = _validate_observation_cursor(parameters["cursor"])
        candidate = parameters["candidate"]
        offer = parameters["offer"]
        if candidate["lane"] != parameters["lane"] or candidate["admission_role"] != parameters["admission_role"]:
            raise ValueError("scheduler_claim role or lane differs from its observation.")
        if offer["lane"] != candidate["lane"] or offer["task_id"] != candidate["task_id"]:
            raise ValueError("scheduler_claim offer differs from its observation.")
        if (
            offer["attempt_id"] != candidate["attempt_id"]
            or offer["attempt_number"] != candidate["attempt_number"]
            or offer["fencing_token"] != candidate["fencing_token"]
            or len(offer["gpu_ids"]) != candidate["requested_gpus"]
            or offer["cpu_slots"] != candidate["requested_cpus"]
            or offer["group_name"] != candidate["group_name"]
            or offer["group_dispatch_epoch"] != candidate["group_dispatch_epoch"]
            or offer["group_worker_set_epoch"] != candidate["group_worker_set_epoch"]
            or offer["worker_state_epoch"] != candidate["worker_state_epoch"]
            or offer["worker_scheduling_role"] != candidate["worker_scheduling_role"]
            or offer["gpu_limit_gpus"] != candidate["gpu_limit_gpus"]
        ):
            raise ValueError("scheduler_claim offer identity differs from its observation.")
        if candidate["ready_scope"] == "home" and candidate["home_machine"] != parameters["machine_name"]:
            raise ValueError("home-scope candidate does not match the claim machine.")
        next_position = parameters["cursor"]["routes"][candidate["ready_scope"]]["next"]
        observed_position = parameters["cursor"]["routes"][candidate["ready_scope"]]["observed"]
        if next_position["revision"] <= observed_position["revision"]:
            raise ValueError("scheduler_claim candidate cursor does not advance.")
        if (
            next_position["catalog_page"] != candidate["catalog_page"]
            or next_position["partition"] != candidate["partition"]
            or next_position["after_name"] != candidate["marker_name"]
        ):
            raise ValueError("scheduler_claim cursor does not name the observed candidate.")
    elif operation_kind == "scheduler_launch_authorize":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "claim_identity"}),
            "scheduler_launch_authorize parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["claim_identity"] = _validate_claim_identity(parameters["claim_identity"])
    elif operation_kind == "scheduler_reservation_reconcile":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "reservation_identity"}),
            "scheduler_reservation_reconcile parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["reservation_identity"] = _validate_reservation_identity(parameters["reservation_identity"])
    elif operation_kind == "maintenance_flush_event":
        _require_exact_keys(
            parameters,
            frozenset({"machine_name", "bucket", "filename", "event_id", "sha256"}),
            "maintenance_flush_event parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["bucket"] = _require_identifier(parameters["bucket"], "bucket")
        if parameters["bucket"] in {".", ".."}:
            raise ValueError("bucket must be one local event directory name.")
        event_id = _require_text(parameters["event_id"], "event_id", maximum=32)
        if not _EVENT_ID.fullmatch(event_id):
            raise ValueError("event_id must be a lowercase 16- or 32-character hexadecimal ID.")
        parameters["event_id"] = event_id
        filename = _require_text(parameters["filename"], "filename", maximum=37)
        if filename != f"{event_id}.json":
            raise ValueError("filename must be the exact JSON filename for event_id.")
        parameters["filename"] = filename
        digest = _require_text(parameters["sha256"], "sha256", maximum=64)
        if not _SHA256_DIGEST.fullmatch(digest):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest.")
        parameters["sha256"] = digest
    elif operation_kind == "machine_snapshot_publish":
        _require_exact_keys(
            parameters,
            frozenset(
                {
                    "machine_name",
                    "instance_id",
                    "pid",
                    "visible_gpu_ids",
                    "reserved_gpu_ids",
                    "reservation_summaries",
                    "heartbeat_interval_seconds",
                    "started_at",
                    "snapshot_at",
                    "gpu_policy",
                    "stop_reason",
                }
            ),
            "machine_snapshot_publish parameters",
        )
        parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")
        parameters["instance_id"] = _require_identifier(parameters["instance_id"], "instance_id")
        pid = parameters["pid"]
        if pid is not None:
            parameters["pid"] = _require_nonnegative_int(pid, "pid", positive=True)
        for field in ("visible_gpu_ids", "reserved_gpu_ids"):
            gpu_ids = parameters[field]
            if (
                not isinstance(gpu_ids, list | tuple)
                or len(gpu_ids) > 4096
                or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
                or len(set(gpu_ids)) != len(gpu_ids)
            ):
                raise ValueError(f"{field} is invalid.")
            parameters[field] = sorted(gpu_ids)
        summaries = parameters["reservation_summaries"]
        if not isinstance(summaries, list | tuple) or len(summaries) > 4096:
            raise ValueError("reservation_summaries is invalid.")
        parameters["reservation_summaries"] = [_validate_machine_reservation(item) for item in summaries]
        interval = parameters["heartbeat_interval_seconds"]
        if (
            type(interval) not in {int, float}
            or (type(interval) is float and not math.isfinite(interval))
            or interval <= 0
            or interval > 86_400
        ):
            raise ValueError("heartbeat_interval_seconds is invalid.")
        parameters["started_at"] = _require_timestamp(parameters["started_at"], "started_at")
        parameters["snapshot_at"] = _require_timestamp(parameters["snapshot_at"], "snapshot_at")
        policy = _bounded_json_value(parameters["gpu_policy"], "gpu_policy")
        if not isinstance(policy, dict):
            raise ValueError("gpu_policy must be an object.")
        parameters["gpu_policy"] = policy
        stop_reason = parameters["stop_reason"]
        if stop_reason is not None:
            stop_reason = _require_text(stop_reason, "stop_reason", maximum=64)
            if stop_reason not in {"idle", "stopped", "stopped_by_signal", "unhandled_exception"}:
                raise ValueError("stop_reason is invalid.")
        parameters["stop_reason"] = stop_reason
    else:
        raise ValueError("operation_kind is not supported by this protocol version.")
    return parameters


def _request_fields(value: object) -> dict[str, Any]:
    payload = _require_mapping(value, "Project I/O request identity")
    _require_exact_keys(payload, _REQUEST_FIELDS, "Project I/O request identity")
    if type(payload["protocol_version"]) is not int or payload["protocol_version"] != PROJECT_IO_PROTOCOL_VERSION:
        raise ValueError("Project I/O protocol version is unsupported.")
    runtime_id = _require_text(payload["runtime_id"], "runtime_id", maximum=64)
    if not _RUNTIME_ID.fullmatch(runtime_id):
        raise ValueError("runtime_id must be a lowercase 64-character hexadecimal ID.")
    for name in ("executor_epoch", "request_id"):
        identifier = _require_text(payload[name], name, maximum=32)
        if not _HEX_ID.fullmatch(identifier):
            raise ValueError(f"{name} must be a lowercase 32-character hexadecimal ID.")
    operation_kind = payload["operation_kind"]
    if operation_kind not in PROJECT_IO_OPERATIONS:
        raise ValueError("operation_kind is not supported by this protocol version.")
    project_id = _require_identifier(payload["project_id"], "project_id")
    canonical_root = _require_text(payload["canonical_shared_root"], "canonical_shared_root", maximum=4096)
    root_path = Path(canonical_root)
    # This parser runs in the controller for poll/status/replay. Canonicality
    # must therefore be checked lexically: Path.resolve() can enter a stalled
    # shared mount and defeat the isolation boundary this protocol establishes.
    if not root_path.is_absolute() or os.path.normpath(canonical_root) != canonical_root:
        raise ValueError("canonical_shared_root must already be absolute and normalized.")
    registration_generation = _require_identifier(payload["registration_generation"], "registration_generation")
    registry_revision = payload["registry_revision"]
    if type(registry_revision) is not int or registry_revision < 0:
        raise ValueError("registry_revision must be a nonnegative integer.")
    source_revisions = _require_mapping(payload["source_revisions"], "source_revisions")
    if len(source_revisions) > 16:
        raise ValueError("source_revisions contains too many entries.")
    revisions: dict[str, JSONScalar] = {}
    for key, revision in source_revisions.items():
        name = _require_identifier(key, "source_revisions key", maximum=64)
        if revision is None or type(revision) in {int, bool}:
            revisions[name] = revision
        elif type(revision) is float and math.isfinite(revision):
            revisions[name] = revision
        elif isinstance(revision, str) and len(revision) <= 128 and "\x00" not in revision:
            revisions[name] = revision
        else:
            raise ValueError("source_revisions values must be bounded JSON scalars.")
    offer_id = payload["provisional_offer_id"]
    if offer_id is not None:
        offer_id = _require_identifier(offer_id, "provisional_offer_id")
    prepared_at = _require_timestamp(payload["prepared_at"], "prepared_at")
    parameters = _request_parameters(operation_kind, payload["parameters"])
    if operation_kind == "authority_renewal":
        _require_exact_keys(
            revisions,
            frozenset({"task", "attempt_digest"}),
            "authority_renewal source_revisions",
        )
        revisions["task"] = _require_nonnegative_int(revisions["task"], "source_revisions.task")
        attempt_digest = revisions["attempt_digest"]
        if not isinstance(attempt_digest, str) or not _SHA256_DIGEST.fullmatch(attempt_digest):
            raise ValueError("source_revisions.attempt_digest is invalid.")
    elif operation_kind == "authority_orphan_recovery":
        _require_exact_keys(
            revisions,
            frozenset({"task", "attempt_digest"}),
            "authority_orphan_recovery source_revisions",
        )
        task_revision = revisions["task"]
        attempt_digest = revisions["attempt_digest"]
        # The first request is built from machine-local evidence and may bind
        # the shared source revision only after the worker acquires authority
        # locks. A populated pair remains a strict CAS for replay/tests.
        if task_revision is None and attempt_digest is None:
            pass
        else:
            revisions["task"] = _require_nonnegative_int(task_revision, "source_revisions.task")
            if not isinstance(attempt_digest, str) or not _SHA256_DIGEST.fullmatch(attempt_digest):
                raise ValueError("source_revisions.attempt_digest is invalid.")
    elif operation_kind == "authority_termination_commit":
        _validate_authority_source_revision_pair(
            revisions,
            "authority_termination_commit source_revisions",
        )
    elif operation_kind == "authority_terminal_observe":
        _require_exact_keys(revisions, frozenset(), "authority_terminal_observe source_revisions")
    elif operation_kind == "authority_terminal_publish":
        _validate_authority_source_revision_pair(
            revisions,
            "authority_terminal_publish source_revisions",
        )
    elif operation_kind == "authority_running_publish":
        _require_exact_keys(revisions, frozenset(), "authority_running_publish source_revisions")
    elif operation_kind == "activation_consumer_retire":
        _require_exact_keys(revisions, frozenset(), "activation_consumer_retire source_revisions")
        if offer_id is not None:
            raise ValueError("activation_consumer_retire cannot name a provisional offer.")
    if operation_kind in {
        "upgrade_service",
        "submission_control_service",
        "observation_service",
        "notification_service",
        "group_service_probe",
        "group_service_advance",
        "legacy_capture_read",
        "legacy_capture_scan",
        "recovery_source_hold",
        "recovery_admission",
        "recovery_source_release",
        "recovery_group_authority",
        "recovery_capture_transition",
    }:
        _require_exact_keys(revisions, frozenset(), f"{operation_kind} source_revisions")
        if offer_id is not None:
            raise ValueError(f"{operation_kind} cannot name a provisional offer.")
    if operation_kind == "progress_projection":
        _require_exact_keys(revisions, frozenset(), "progress source_revisions")
        if offer_id is not None:
            raise ValueError("progress cannot name a provisional offer")
        parameters["projection"] = progress_projection(
            _json_copy(parameters["projection"]), parameters["context"], registration_generation
        )
    if operation_kind == "scheduler_primary_probe":
        _require_exact_keys(revisions, frozenset(), "scheduler_primary_probe source_revisions")
    if operation_kind == "scheduler_quiescence_probe":
        _require_exact_keys(revisions, frozenset(), "scheduler_quiescence_probe source_revisions")
        if offer_id is not None:
            raise ValueError("scheduler quiescence probe cannot name a provisional offer.")
    if operation_kind in {"group_service_probe", "group_service_advance"}:
        _require_exact_keys(revisions, frozenset(), f"{operation_kind} source_revisions")
        if offer_id is not None:
            raise ValueError("Group-service operation cannot name a provisional offer.")
    if operation_kind == "scheduler_claim":
        if offer_id != parameters["offer"]["offer_id"]:
            raise ValueError("provisional_offer_id differs from the claim offer.")
    elif operation_kind == "scheduler_launch_authorize":
        if offer_id != parameters["claim_identity"]["reservation_id"]:
            raise ValueError("provisional_offer_id differs from the launch reservation.")
    elif operation_kind == "scheduler_reservation_reconcile":
        _require_exact_keys(
            revisions,
            frozenset(),
            "scheduler_reservation_reconcile source_revisions",
        )
        identity = parameters["reservation_identity"]
        if identity["project_id"] != project_id or identity["shared_root"] != canonical_root:
            raise ValueError("reservation_identity differs from its request binding.")
        if offer_id is not None:
            raise ValueError("scheduler_reservation_reconcile cannot name a provisional offer.")
    elif operation_kind == "scheduler_due_offer":
        _require_exact_keys(revisions, frozenset(), "scheduler_due_offer source_revisions")
        if offer_id is not None:
            raise ValueError("scheduler_due_offer cannot name a provisional offer.")
    elif operation_kind == "scheduler_ready_index_build":
        _require_exact_keys(revisions, frozenset(), "scheduler_ready_index_build source_revisions")
        if offer_id is not None:
            raise ValueError("scheduler_ready_index_build cannot name a provisional offer.")
    elif operation_kind == "maintenance_descriptor_advance":
        _require_exact_keys(revisions, frozenset(), "maintenance_descriptor_advance source_revisions")
        if offer_id is not None:
            raise ValueError("maintenance_descriptor_advance cannot name a provisional offer.")
    elif operation_kind == "machine_snapshot_publish":
        if offer_id is not None:
            raise ValueError("machine_snapshot_publish cannot name a provisional offer.")
    elif offer_id is not None:
        raise ValueError("this operation cannot name a provisional offer.")
    return {
        "protocol_version": PROJECT_IO_PROTOCOL_VERSION,
        "runtime_id": runtime_id,
        "executor_epoch": payload["executor_epoch"],
        "request_id": payload["request_id"],
        "operation_kind": operation_kind,
        "project_id": project_id,
        "canonical_shared_root": canonical_root,
        "registration_generation": registration_generation,
        "registry_revision": registry_revision,
        "source_revisions": MappingProxyType(revisions),
        "provisional_offer_id": offer_id,
        "prepared_at": prepared_at,
        "parameters": _freeze_json(parameters),
    }


@dataclass(frozen=True, slots=True)
class ProjectIORequest:
    protocol_version: int
    runtime_id: str
    executor_epoch: str
    request_id: str
    operation_kind: str
    project_id: str
    canonical_shared_root: str
    registration_generation: str
    registry_revision: int
    source_revisions: Mapping[str, JSONScalar]
    provisional_offer_id: str | None
    prepared_at: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = _request_fields(
            {
                "protocol_version": self.protocol_version,
                "runtime_id": self.runtime_id,
                "executor_epoch": self.executor_epoch,
                "request_id": self.request_id,
                "operation_kind": self.operation_kind,
                "project_id": self.project_id,
                "canonical_shared_root": self.canonical_shared_root,
                "registration_generation": self.registration_generation,
                "registry_revision": self.registry_revision,
                "source_revisions": self.source_revisions,
                "provisional_offer_id": self.provisional_offer_id,
                "prepared_at": self.prepared_at,
                "parameters": self.parameters,
            }
        )
        for key, value in normalized.items():
            object.__setattr__(self, key, value)
        if self.operation_kind in {"scheduler_claim", "scheduler_cursor_commit"}:
            _validate_observation_cursor(self.parameters["cursor"], self)
        if self.operation_kind == "machine_snapshot_publish":
            parameters = self.parameters
            if any(
                item["project_id"] != self.project_id or item["machine_name"] not in {None, parameters["machine_name"]}
                for item in parameters["reservation_summaries"]
            ):
                raise ValueError("machine snapshot reservation summaries differ from their project binding.")
        _require_record_size({"project_io_request": self._payload()}, "project_io_request")

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "runtime_id": self.runtime_id,
            "executor_epoch": self.executor_epoch,
            "request_id": self.request_id,
            "operation_kind": self.operation_kind,
            "project_id": self.project_id,
            "canonical_shared_root": self.canonical_shared_root,
            "registration_generation": self.registration_generation,
            "registry_revision": self.registry_revision,
            "source_revisions": dict(self.source_revisions),
            "provisional_offer_id": self.provisional_offer_id,
            "prepared_at": self.prepared_at,
            "parameters": _json_copy(self.parameters),
        }

    def to_dict(self) -> dict[str, Any]:
        value = {"project_io_request": self._payload()}
        _require_record_size(value, "project_io_request")
        return value

    @classmethod
    def from_dict(cls, value: object) -> ProjectIORequest:
        envelope = _require_envelope(value, "project_io_request", "project_io_request")
        _require_exact_keys(envelope, _REQUEST_FIELDS, "project_io_request")
        _require_record_size({"project_io_request": dict(envelope)}, "project_io_request")
        return cls(**dict(envelope))


@dataclass(frozen=True, slots=True)
class ProjectIOProcess:
    request: ProjectIORequest
    pid: int
    start_time_ticks: int
    started_at: str
    state: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProjectIORequest):
            raise ValueError("process request identity is invalid.")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("process pid must be a positive integer.")
        if type(self.start_time_ticks) is not int or self.start_time_ticks <= 0:
            raise ValueError("process start_time_ticks must be a positive integer.")
        _require_timestamp(self.started_at, "started_at")
        if self.state not in PROJECT_IO_PROCESS_STATES:
            raise ValueError("process state is invalid.")
        _require_record_size({"project_io_process": self._payload()}, "project_io_process")

    def _payload(self) -> dict[str, Any]:
        return {
            **self.request._payload(),
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "started_at": self.started_at,
            "state": self.state,
        }

    def to_dict(self) -> dict[str, Any]:
        value = {"project_io_process": self._payload()}
        _require_record_size(value, "project_io_process")
        return value

    @classmethod
    def from_dict(cls, value: object) -> ProjectIOProcess:
        envelope = _require_envelope(value, "project_io_process", "project_io_process")
        _require_exact_keys(
            envelope,
            frozenset({*_REQUEST_FIELDS, "pid", "start_time_ticks", "started_at", "state"}),
            "project_io_process",
        )
        _require_record_size({"project_io_process": dict(envelope)}, "project_io_process")
        request = ProjectIORequest(**{key: envelope[key] for key in _REQUEST_FIELDS})
        return cls(
            request=request,
            pid=envelope["pid"],
            start_time_ticks=envelope["start_time_ticks"],
            started_at=envelope["started_at"],
            state=envelope["state"],
        )


@dataclass(frozen=True, slots=True)
class ProjectIOResult:
    request: ProjectIORequest
    status: str
    reason_code: str | None
    completed_at: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProjectIORequest):
            raise ValueError("result request identity is invalid.")
        if self.status not in PROJECT_IO_RESULT_STATUSES:
            raise ValueError("result status is invalid.")
        if self.reason_code is not None and self.reason_code not in {
            "project_io_binding_validation_failed",
            "project_io_upgrade_service_failed",
            "project_io_submission_control_service_failed",
            "project_io_observation_service_failed",
            "project_io_notification_service_failed",
            "project_io_progress_projection_failed",
            "project_io_legacy_capture_read_failed",
            "project_io_legacy_capture_scan_failed",
            "project_io_recovery_source_hold_failed",
            "project_io_recovery_admission_failed",
            "project_io_recovery_source_release_failed",
            "project_io_recovery_group_authority_failed",
            "project_io_recovery_capture_transition_failed",
            "project_io_scheduler_observation_failed",
            "project_io_scheduler_claim_failed_before_commit",
            "project_io_scheduler_cursor_commit_failed",
            "project_io_scheduler_launch_authorize_failed",
            "project_io_scheduler_reservation_reconcile_failed",
            "project_io_scheduler_due_offer_failed",
            "project_io_scheduler_ready_index_build_failed",
            "project_io_maintenance_descriptor_advance_failed",
            "project_io_maintenance_flush_event_failed",
            "project_io_machine_snapshot_publish_failed",
            "project_io_registration_renew_failed",
            "project_io_activation_observe_failed",
            "project_io_activation_consumer_register_failed",
            "project_io_activation_consumer_ack_failed",
            "project_io_activation_consumer_retire_failed",
            "project_io_authority_service_failed",
            "project_io_authority_renewal_failed",
            "project_io_authority_orphan_recovery_failed",
            "project_io_authority_termination_commit_failed",
            "project_io_authority_terminal_observe_failed",
            "project_io_authority_terminal_publish_failed",
            "project_io_authority_running_publish_failed",
            "project_io_worker_exited_without_result",
            "project_io_executor_epoch_fenced",
            "project_io_outcome_unknown",
            "project_io_protocol_invalid",
        }:
            raise ValueError("result reason_code is invalid.")
        if (self.status == "completed" and self.reason_code is not None) or (
            self.status == "fenced" and self.reason_code != "project_io_executor_epoch_fenced"
        ):
            raise ValueError("result status and reason_code do not agree.")
        if self.status in {"retryable_error", "outcome_unknown"} and self.reason_code is None:
            raise ValueError("retryable or unknown results require a stable reason_code.")
        _require_timestamp(self.completed_at, "completed_at")
        evidence = _require_mapping(self.evidence, "result evidence")
        normalized_evidence = _validate_evidence(self.request, self.status, evidence)
        object.__setattr__(self, "evidence", _freeze_json(normalized_evidence))
        _require_record_size({"project_io_result": self._payload()}, "project_io_result")

    def _payload(self) -> dict[str, Any]:
        return {
            **self.request._payload(),
            "status": self.status,
            "reason_code": self.reason_code,
            "completed_at": self.completed_at,
            "evidence": _json_copy(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        value = {"project_io_result": self._payload()}
        _require_record_size(value, "project_io_result")
        return value

    @classmethod
    def from_dict(cls, value: object) -> ProjectIOResult:
        envelope = _require_envelope(value, "project_io_result", "project_io_result")
        result_keys = frozenset({*_REQUEST_FIELDS, "status", "reason_code", "completed_at", "evidence"})
        _require_exact_keys(envelope, result_keys, "project_io_result")
        _require_record_size({"project_io_result": dict(envelope)}, "project_io_result")
        request = ProjectIORequest(**{key: envelope[key] for key in _REQUEST_FIELDS})
        return cls(
            request=request,
            status=envelope["status"],
            reason_code=envelope["reason_code"],
            completed_at=envelope["completed_at"],
            evidence=envelope["evidence"],
        )


def _validate_authority_source_revision_pair(value: object, label: str) -> dict[str, JSONScalar]:
    revisions = _require_mapping(value, label)
    _require_exact_keys(revisions, frozenset({"task", "attempt_digest"}), label)
    task_revision = _require_nonnegative_int(revisions["task"], f"{label}.task")
    attempt_digest = revisions["attempt_digest"]
    if not isinstance(attempt_digest, str) or not _SHA256_DIGEST.fullmatch(attempt_digest):
        raise ValueError(f"{label}.attempt_digest is invalid.")
    return {"task": task_revision, "attempt_digest": attempt_digest}


def _validate_observed_authority_revisions(value: object, label: str) -> dict[str, JSONScalar]:
    revisions = _require_mapping(value, label)
    _require_exact_keys(revisions, frozenset({"task", "attempt_digest"}), label)
    task_revision = revisions["task"]
    if task_revision is not None:
        task_revision = _require_nonnegative_int(task_revision, f"{label}.task")
    attempt_digest = revisions["attempt_digest"]
    if attempt_digest is not None and (
        not isinstance(attempt_digest, str) or not _SHA256_DIGEST.fullmatch(attempt_digest)
    ):
        raise ValueError(f"{label}.attempt_digest is invalid.")
    return {"task": task_revision, "attempt_digest": attempt_digest}


def _validate_terminal_lifecycle_event(value: object, label: str) -> dict[str, Any]:
    event = dict(_require_mapping(value, label))
    expected = frozenset(
        {
            "event_type",
            "task_id",
            "attempt_id",
            "attempt_number",
            "previous_task_phase",
            "phase",
            "reason",
            "exit_code",
            "execution_machine_name",
            "dispatching_machine_name",
            "finished_at",
            "task_revision",
            "execution_started_at",
            "duration_ms",
            "project_id",
            "project",
            "task_name",
        }
    )
    _require_exact_keys(event, expected, label)
    if event["event_type"] != "task_terminal":
        raise ValueError(f"{label}.event_type is invalid.")
    event["task_id"] = _require_identifier(event["task_id"], f"{label}.task_id")
    event["attempt_id"] = _require_identifier(event["attempt_id"], f"{label}.attempt_id")
    event["attempt_number"] = _require_nonnegative_int(
        event["attempt_number"], f"{label}.attempt_number", positive=True
    )
    if event["previous_task_phase"] not in {"running", "blocked"}:
        raise ValueError(f"{label}.previous_task_phase is invalid.")
    if event["phase"] not in _PROJECT_IO_TERMINAL_PHASES:
        raise ValueError(f"{label}.phase is invalid.")
    event["reason"] = _require_text(event["reason"], f"{label}.reason", maximum=256)
    if event["exit_code"] is not None and type(event["exit_code"]) is not int:
        raise ValueError(f"{label}.exit_code is invalid.")
    for field in ("execution_machine_name", "dispatching_machine_name"):
        event[field] = _require_identifier(event[field], f"{label}.{field}")
    event["finished_at"] = _require_timestamp(event["finished_at"], f"{label}.finished_at")
    event["task_revision"] = _require_nonnegative_int(event["task_revision"], f"{label}.task_revision", positive=True)
    started_at = event["execution_started_at"]
    if started_at is not None:
        event["execution_started_at"] = _require_timestamp(started_at, f"{label}.execution_started_at")
    duration_ms = event["duration_ms"]
    if duration_ms is not None:
        event["duration_ms"] = _require_nonnegative_int(duration_ms, f"{label}.duration_ms")
    project_id = event["project_id"]
    if project_id is not None:
        event["project_id"] = _require_identifier(project_id, f"{label}.project_id")
    event["project"] = _require_text(event["project"], f"{label}.project", maximum=4096)
    task_name = event["task_name"]
    if task_name is not None:
        if not isinstance(task_name, str) or len(json.dumps(task_name, ensure_ascii=False).encode("utf-8")) > 16_384:
            raise ValueError(f"{label}.task_name is invalid.")
    return event


def _validate_evidence(request: ProjectIORequest, status: str, value: Mapping[str, Any]) -> dict[str, Any]:
    evidence = dict(value)
    if status == "completed":
        if request.operation_kind == "validate_binding":
            expected = {"project_id", "shared_root", "schema_version", "required_capabilities"}
            _require_exact_keys(evidence, frozenset(expected), "validate_binding evidence")
            if evidence["project_id"] != request.project_id or evidence["shared_root"] != request.canonical_shared_root:
                raise ValueError("completed binding evidence does not match the request identity.")
            if type(evidence["schema_version"]) is not int or evidence["schema_version"] <= 0:
                raise ValueError("schema_version evidence must be a positive integer.")
            capabilities = evidence["required_capabilities"]
            if (
                not isinstance(capabilities, list)
                or len(capabilities) > 32
                or any(not isinstance(item, str) or not item or len(item) > 128 for item in capabilities)
                or len(set(capabilities)) != len(capabilities)
            ):
                raise ValueError("required_capabilities evidence is invalid.")
            evidence["required_capabilities"] = list(capabilities)
        elif request.operation_kind == "scheduler_due_offer":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "reason", "task_id"}),
                "scheduler_due_offer evidence",
            )
            reasons = {
                "offered",
                "no_due_deadline",
                "task_missing",
                "task_not_queued",
                "claim_active",
                "not_home_machine",
                "deadline_not_due",
                "elapsed_offer_unproven",
                "offer_rejected",
            }
            if evidence["outcome"] not in {"offered", "noop"} or evidence["reason"] not in reasons:
                raise ValueError("scheduler_due_offer outcome or reason is invalid.")
            if (evidence["outcome"] == "offered") != (evidence["reason"] == "offered"):
                raise ValueError("scheduler_due_offer outcome and reason do not agree.")
            task_id = evidence["task_id"]
            if evidence["reason"] == "no_due_deadline":
                if task_id is not None:
                    raise ValueError("scheduler_due_offer empty result cannot name a Task.")
            else:
                evidence["task_id"] = _require_identifier(task_id, "scheduler_due_offer task_id")
        elif request.operation_kind == "scheduler_ready_index_build":
            _require_exact_keys(
                evidence,
                frozenset({"state", "revision", "build_id", "phase"}),
                "scheduler_ready_index_build evidence",
            )
            if evidence["state"] not in {"absent", "building", "active", "degraded"}:
                raise ValueError("scheduler_ready_index_build state is invalid.")
            revision = evidence["revision"]
            if type(revision) is not int or revision < 0:
                raise ValueError("scheduler_ready_index_build revision is invalid.")
            build_id = evidence["build_id"]
            phase = evidence["phase"]
            if build_id is not None:
                build_id = _require_identifier(build_id, "scheduler_ready_index_build build_id")
            if phase is not None and phase not in {
                "reset-projection",
                "inventory",
                "backfill",
                "audit",
                "primary-rebuild",
            }:
                raise ValueError("scheduler_ready_index_build phase is invalid.")
            if evidence["state"] != "building" and (build_id is not None or phase is not None):
                raise ValueError("scheduler_ready_index_build non-building evidence cannot name a build.")
            if evidence["state"] == "building" and (build_id is None or phase is None):
                raise ValueError("scheduler_ready_index_build building evidence requires build identity.")
            evidence["build_id"] = build_id
            evidence["phase"] = phase
        elif request.operation_kind == "maintenance_descriptor_advance":
            _require_exact_keys(
                evidence,
                frozenset({"maintenance_state", "next_due_at", "more", "idle_blocking"}),
                "maintenance_descriptor_advance evidence",
            )
            if evidence["maintenance_state"] not in {
                "idle",
                "waiting",
                "pending",
                "running",
                "completed",
                "intervention",
            }:
                raise ValueError("maintenance_descriptor_advance maintenance_state is invalid.")
            if type(evidence["more"]) is not bool:
                raise ValueError("maintenance_descriptor_advance more must be a boolean.")
            if type(evidence["idle_blocking"]) is not bool:
                raise ValueError("maintenance_descriptor_advance idle_blocking must be a boolean.")
            if evidence["idle_blocking"] and evidence["maintenance_state"] != "waiting":
                raise ValueError("maintenance_descriptor_advance idle_blocking requires waiting state.")
            next_due_at = evidence["next_due_at"]
            if evidence["maintenance_state"] == "waiting":
                if next_due_at is not None:
                    evidence["next_due_at"] = _require_timestamp(
                        next_due_at,
                        "maintenance_descriptor_advance next_due_at",
                    )
            elif next_due_at is not None:
                raise ValueError("maintenance_descriptor_advance next_due_at must be null outside waiting state.")
        elif request.operation_kind == "scheduler_quiescence_probe":
            _require_exact_keys(evidence, frozenset({"state", "probe_state"}), "scheduler quiescence evidence")
            if evidence["state"] not in {"quiescent", "active", "pending"}:
                raise ValueError("scheduler quiescence state is invalid.")
            state = validate_probe_state(evidence["probe_state"])
            if any(route["recheck"] is not None for route in state["routes"].values()):
                raise ValueError("scheduler quiescence evidence cannot carry dependency rechecks.")
            if evidence["state"] == "quiescent" and (
                state["pending_scopes"] or any(not route["is_complete"] for route in state["routes"].values())
            ):
                raise ValueError("scheduler quiescence requires both complete routes.")
            evidence["probe_state"] = state
        elif request.operation_kind == "scheduler_primary_probe":
            _require_exact_keys(
                evidence, frozenset({"demand", "probe_state", "route_revisions"}), "scheduler_primary_probe evidence"
            )
            if evidence["demand"] not in {"runnable_now", "waiting_for_aggregation", "no_primary_demand", "unresolved"}:
                raise ValueError("primary probe demand is invalid.")
            state = validate_probe_state(evidence["probe_state"])
            revisions = evidence["route_revisions"]
            if evidence["demand"] == "no_primary_demand":
                revisions = dict(_require_mapping(revisions, "primary probe route_revisions"))
                _require_exact_keys(revisions, frozenset({"home", "shared"}), "primary probe route_revisions")
                for scope, revision in revisions.items():
                    _require_nonnegative_int(revision, "primary probe route revision")
                    if not state["routes"][scope]["is_complete"] or state["routes"][scope]["revision"] != revision:
                        raise ValueError("primary absence proof differs from its completed baseline.")
            elif revisions is not None:
                raise ValueError("non-absence primary probe result cannot supply an absence proof.")
            evidence["probe_state"] = state
            evidence["route_revisions"] = revisions
        elif request.operation_kind == "scheduler_observe":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "reason", "source_revisions", "candidate", "cursor"}),
                "scheduler_observe evidence",
            )
            if evidence["outcome"] not in {"candidate", "none"}:
                raise ValueError("scheduler_observe outcome is invalid.")
            if evidence["reason"] not in _OBSERVATION_REASONS:
                raise ValueError("scheduler_observe reason is invalid.")
            revisions = _validate_source_revisions(evidence["source_revisions"])
            cursor = _validate_observation_cursor(evidence["cursor"], request)
            candidate = evidence["candidate"]
            if evidence["outcome"] == "candidate":
                candidate = _validate_candidate(candidate)
                if candidate["lane"] != request.parameters["lane"]:
                    raise ValueError("observed candidate lane differs from the request.")
                if candidate["admission_role"] != request.parameters["admission_role"]:
                    raise ValueError("observed candidate role differs from the request.")
                if candidate["task_id"] == "":
                    raise ValueError("observed candidate task identity is invalid.")
                if (
                    candidate["ready_scope"] == "home"
                    and candidate["home_machine"] != request.parameters["machine_name"]
                ):
                    raise ValueError("home-scope candidate does not match the request machine.")
                next_position = cursor["routes"][candidate["ready_scope"]]["next"]
                if (
                    next_position["catalog_page"] != candidate["catalog_page"]
                    or next_position["partition"] != candidate["partition"]
                    or next_position["after_name"] != candidate["marker_name"]
                ):
                    raise ValueError("observed candidate reference differs from its next cursor position.")
                if evidence["reason"] != "candidate_ready":
                    raise ValueError("candidate observation must use candidate_ready reason.")
            elif candidate is not None:
                raise ValueError("no-candidate observation must not contain a candidate.")
            elif evidence["reason"] == "candidate_ready":
                raise ValueError("empty observation cannot use candidate_ready reason.")
            evidence["source_revisions"] = revisions
            evidence["candidate"] = candidate
            evidence["cursor"] = cursor
        elif request.operation_kind == "scheduler_claim":
            outcome = evidence.get("outcome")
            if outcome == "claimed":
                fields = {
                    "outcome",
                    "attempt_id",
                    "attempt_number",
                    "fencing_token",
                    "reservation_id",
                    "gpu_ids",
                    "cpu_slots",
                    "cursor_routes",
                }
                _require_exact_keys(evidence, frozenset(fields), "scheduler_claim evidence")
                if evidence["cursor_routes"] is not None:
                    raise ValueError("claimed scheduler_claim evidence cannot advance cursors.")
                for field in ("attempt_id", "reservation_id"):
                    evidence[field] = _require_identifier(evidence[field], f"scheduler_claim.{field}")
                candidate = request.parameters["candidate"]
                offer = request.parameters["offer"]
                if (
                    evidence["attempt_id"] != candidate["attempt_id"]
                    or evidence["attempt_number"] != candidate["attempt_number"]
                    or evidence["fencing_token"] != candidate["fencing_token"]
                    or evidence["reservation_id"] != offer["reservation_id"]
                ):
                    raise ValueError("scheduler_claim evidence does not match its request.")
                gpu_ids = evidence["gpu_ids"]
                if (
                    not isinstance(gpu_ids, list)
                    or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
                    or len(set(gpu_ids)) != len(gpu_ids)
                    or tuple(gpu_ids) != tuple(offer["gpu_ids"])
                    or evidence["cpu_slots"] != offer["cpu_slots"]
                ):
                    raise ValueError("scheduler_claim resources differ from its offer.")
                evidence["gpu_ids"] = list(gpu_ids)
            elif outcome == "no_claim":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "reason", "cursor_routes"}),
                    "scheduler_claim evidence",
                )
                if evidence["reason"] not in _CLAIM_REASONS - {"claimed"}:
                    raise ValueError("scheduler_claim reason is invalid.")
                cursor_routes = _require_mapping(evidence["cursor_routes"], "scheduler_claim cursor_routes")
                _require_exact_keys(
                    cursor_routes,
                    frozenset({"home", "shared"}),
                    "scheduler_claim cursor_routes",
                )
                outcomes = {"committed", "already_applied", "stale"}
                if any(cursor_routes[scope] not in outcomes for scope in ("home", "shared")):
                    raise ValueError("scheduler_claim cursor route outcome is invalid.")
                evidence["cursor_routes"] = {scope: cursor_routes[scope] for scope in ("home", "shared")}
            else:
                raise ValueError("scheduler_claim outcome is invalid.")
        elif request.operation_kind == "scheduler_cursor_commit":
            _require_exact_keys(evidence, frozenset({"routes"}), "scheduler_cursor_commit evidence")
            routes = _require_mapping(evidence["routes"], "scheduler_cursor_commit routes")
            _require_exact_keys(routes, frozenset({"home", "shared"}), "scheduler_cursor_commit routes")
            outcomes = {"committed", "already_applied", "stale"}
            if any(routes[scope] not in outcomes for scope in ("home", "shared")):
                raise ValueError("scheduler_cursor_commit route outcome is invalid.")
            evidence["routes"] = {scope: routes[scope] for scope in ("home", "shared")}
        elif request.operation_kind == "scheduler_launch_authorize":
            outcome = evidence.get("outcome")
            expected_identity = dict(request.parameters["claim_identity"])
            if outcome == "authorized":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "claim_identity", "launch_id", "launch_handoff_timeout_seconds"}),
                    "scheduler_launch_authorize evidence",
                )
                identity = _validate_claim_identity(evidence["claim_identity"])
                if identity != expected_identity:
                    raise ValueError("launch authorization identity differs from its request.")
                launch_id = _require_text(evidence["launch_id"], "launch_id", maximum=32)
                if not _HEX_ID.fullmatch(launch_id):
                    raise ValueError("launch_id must be a lowercase 32-character hexadecimal ID.")
                timeout = evidence["launch_handoff_timeout_seconds"]
                if (
                    type(timeout) not in {int, float}
                    or (type(timeout) is float and not math.isfinite(timeout))
                    or timeout < 1
                    or timeout > 300
                ):
                    raise ValueError("launch_handoff_timeout_seconds is invalid.")
                evidence["claim_identity"] = identity
            elif outcome == "denied":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "claim_identity", "reason"}),
                    "scheduler_launch_authorize evidence",
                )
                identity = _validate_claim_identity(evidence["claim_identity"])
                if identity != expected_identity:
                    raise ValueError("launch authorization identity differs from its request.")
                if evidence["reason"] not in _LAUNCH_AUTHORIZATION_DENIAL_REASONS:
                    raise ValueError("scheduler_launch_authorize reason is invalid.")
                evidence["claim_identity"] = identity
            else:
                raise ValueError("scheduler_launch_authorize outcome is invalid.")
        elif request.operation_kind == "scheduler_reservation_reconcile":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reservation_identity",
                        "reason",
                        "target_attempt_id",
                        "target_fencing_token",
                    }
                ),
                "scheduler_reservation_reconcile evidence",
            )
            identity = _validate_reservation_identity(evidence["reservation_identity"])
            expected_identity = _validate_reservation_identity(request.parameters["reservation_identity"])
            if identity != expected_identity:
                raise ValueError("reservation reconciliation identity differs from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            target_attempt_id = evidence["target_attempt_id"]
            target_fencing_token = evidence["target_fencing_token"]
            if outcome in {"retained", "isolated"}:
                if reason is not None or target_attempt_id is not None or target_fencing_token is not None:
                    raise ValueError("reservation reconciliation outcome contains invalid authority.")
            elif outcome == "release":
                if (
                    reason not in {"task_missing", "claim_missing"}
                    or target_attempt_id is not None
                    or target_fencing_token is not None
                ):
                    raise ValueError("reservation release evidence is invalid.")
            elif outcome == "retag":
                target_attempt_id = _require_identifier(
                    target_attempt_id, "scheduler_reservation_reconcile target_attempt_id"
                )
                target_fencing_token = _require_nonnegative_int(
                    target_fencing_token,
                    "scheduler_reservation_reconcile target_fencing_token",
                    positive=True,
                )
                if (
                    reason is not None
                    or target_attempt_id != identity["attempt_id"]
                    or target_fencing_token <= identity["fencing_token"]
                ):
                    raise ValueError("reservation retag evidence is invalid.")
            else:
                raise ValueError("scheduler_reservation_reconcile outcome is invalid.")
            evidence["reservation_identity"] = identity
        elif request.operation_kind == "maintenance_flush_event":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "event_id", "sha256", "reason"}),
                "maintenance_flush_event evidence",
            )
            if evidence["outcome"] not in {"flushed", "stale", "missing"}:
                raise ValueError("maintenance_flush_event outcome is invalid.")
            if evidence["event_id"] != request.parameters["event_id"]:
                raise ValueError("maintenance_flush_event event_id differs from its request.")
            if evidence["sha256"] != request.parameters["sha256"]:
                raise ValueError("maintenance_flush_event digest differs from its request.")
            reason = evidence["reason"]
            valid_reasons = {
                "source_missing",
                "source_untrusted",
                "digest_mismatch",
                "invalid_event",
                "identity_mismatch",
                "binding_fence",
            }
            if (
                (evidence["outcome"] == "flushed" and reason is not None)
                or (evidence["outcome"] == "missing" and reason != "source_missing")
                or (evidence["outcome"] == "stale" and reason not in valid_reasons - {"source_missing"})
            ):
                raise ValueError("maintenance_flush_event reason does not match its outcome.")
        elif request.operation_kind == "machine_snapshot_publish":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "snapshot_at", "reason"}),
                "machine_snapshot_publish evidence",
            )
            if evidence["outcome"] not in {"published", "stale"}:
                raise ValueError("machine_snapshot_publish outcome is invalid.")
            if evidence["snapshot_at"] != request.parameters["snapshot_at"]:
                raise ValueError("machine_snapshot_publish timestamp differs from its request.")
            reason = evidence["reason"]
            if (evidence["outcome"] == "published" and reason is not None) or (
                evidence["outcome"] == "stale" and reason != "binding_fence"
            ):
                raise ValueError("machine_snapshot_publish reason does not match its outcome.")
        elif request.operation_kind == "registration_renew":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "renewed", "eligibility_expires_at", "renew_after_seconds", "reason"}),
                "registration_renew evidence",
            )
            if evidence["outcome"] not in {"eligible", "stale"} or type(evidence["renewed"]) is not bool:
                raise ValueError("registration_renew outcome or renewed flag is invalid.")
            if evidence["outcome"] == "eligible":
                if evidence["reason"] is not None:
                    raise ValueError("eligible registration_renew evidence cannot have a reason.")
                evidence["eligibility_expires_at"] = _require_timestamp(
                    evidence["eligibility_expires_at"], "eligibility_expires_at"
                )
                renew_after = evidence["renew_after_seconds"]
                if (
                    type(renew_after) not in {int, float}
                    or not math.isfinite(renew_after)
                    or renew_after < 0
                    or renew_after > 86_400
                ):
                    raise ValueError("registration_renew renew_after_seconds is invalid.")
            else:
                if evidence["renewed"] is not False:
                    raise ValueError("stale registration_renew evidence cannot be renewed.")
                if (
                    evidence["eligibility_expires_at"] is not None
                    or evidence["renew_after_seconds"] is not None
                    or evidence["reason"] != "binding_fence"
                ):
                    raise ValueError("stale registration_renew evidence is invalid.")
        elif request.operation_kind == "recovery_admission":
            evidence = recovery_admission_evidence(evidence)
        elif request.operation_kind == "recovery_source_release":
            evidence = recovery_completion_evidence(evidence, completed_state="released")
        elif request.operation_kind == "recovery_group_authority":
            evidence = recovery_completion_evidence(evidence, completed_state="active")
        elif request.operation_kind == "recovery_capture_transition":
            evidence = recovery_completion_evidence(
                evidence, completed_state="retained" if request.parameters["phase"] == "retain" else "normalized"
            )
        elif request.operation_kind == "recovery_source_hold":
            evidence = source_hold_evidence(evidence, request.parameters["capture_id"])
        elif request.operation_kind == "legacy_capture_read":
            evidence = legacy_capture_evidence(evidence, request.parameters)
        elif request.operation_kind == "legacy_capture_scan":
            evidence = legacy_scan_evidence(evidence, request.parameters)
        elif request.operation_kind == "progress_projection":
            evidence = progress_evidence(
                _json_copy(evidence), _json_copy(request.parameters), request.registration_generation
            )
        elif request.operation_kind == "notification_service":
            _require_exact_keys(evidence, frozenset({"state"}), "notification_service evidence")
            if evidence["state"] not in {"ready", "conflict", "source_invalid", "blocked"}:
                raise ValueError("Notification service state is invalid.")
        elif request.operation_kind == "group_service_probe":
            evidence = group_service_probe_evidence(evidence)
        elif request.operation_kind == "group_service_advance":
            evidence = group_service_advance_evidence(evidence, request.parameters["candidate"])
        elif request.operation_kind == "observation_service":
            _require_exact_keys(
                evidence, frozenset({"state", "quiescent", "reason_code"}), "observation_service evidence"
            )
            if evidence["state"] not in {"active", "building", "waiting", "degraded", "closed"}:
                raise ValueError("Observation service state is invalid.")
            if type(evidence["quiescent"]) is not bool:
                raise ValueError("Observation service quiescence must be boolean.")
            if evidence["reason_code"] not in {"progress", "idle", "blocked", "closed"}:
                raise ValueError("Observation service reason is invalid.")
            if evidence["quiescent"] != (evidence["reason_code"] == "idle"):
                raise ValueError("Observation service quiescence lacks idle evidence.")
            if evidence["quiescent"] and evidence["state"] != "active":
                raise ValueError("Observation idle evidence must be active.")
            if (evidence["state"] == "closed") != (evidence["reason_code"] == "closed"):
                raise ValueError("Observation service closed state is inconsistent.")
        elif request.operation_kind == "submission_control_service":
            _require_exact_keys(
                evidence,
                frozenset({"state", "quiescent", "reason_code", "continuation"}),
                "submission_control_service evidence",
            )
            if evidence["state"] not in {"active", "building", "waiting", "closed"}:
                raise ValueError("Submission-control service state is invalid.")
            if type(evidence["quiescent"]) is not bool:
                raise ValueError("Submission-control service quiescence must be boolean.")
            if evidence["reason_code"] not in {"progress", "idle", "blocked", "closed"}:
                raise ValueError("Submission-control service reason is invalid.")
            if evidence["quiescent"] != (evidence["reason_code"] == "idle"):
                raise ValueError("Submission-control service quiescence lacks idle evidence.")
            if not isinstance(evidence["continuation"], Mapping):
                raise ValueError("Submission-control continuation must be an object.")
            cursor = submission_control_continuation(evidence["continuation"])
            if evidence["quiescent"] and (cursor["pending_offset"] or cursor["pending_had_work"]):
                raise ValueError("Submission-control idle evidence has an incomplete sweep.")
        elif request.operation_kind == "upgrade_service":
            _require_exact_keys(
                evidence,
                frozenset({"state", "pending", "can_run", "admission_blocked", "idle_blocking", "next_probe_at"}),
                "upgrade_service evidence",
            )
            if evidence["state"] not in {
                "idle",
                "runnable",
                "waiting",
                "paused",
                "pause_pending",
                "repair_required",
                "ready_to_resume",
                "completed",
                "inaccessible",
            }:
                raise ValueError("upgrade_service state is invalid.")
            for name in ("pending", "can_run", "admission_blocked", "idle_blocking"):
                if type(evidence[name]) is not bool:
                    raise ValueError(f"upgrade_service {name} must be a boolean.")
            if evidence["next_probe_at"] is not None:
                _require_timestamp(evidence["next_probe_at"], "upgrade_service next_probe_at")
        elif request.operation_kind == "activation_observe":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "checkpoint", "replay"}),
                "activation_observe evidence",
            )
            if evidence["outcome"] != "observed":
                raise ValueError("activation_observe outcome is invalid.")
            checkpoint = evidence["checkpoint"]
            if checkpoint is not None:
                checkpoint = _validate_activation_checkpoint(checkpoint, "activation_observe.checkpoint")
            replay = evidence["replay"]
            if checkpoint is None:
                if replay is not None:
                    raise ValueError("activation_observe replay requires a checkpoint.")
            else:
                replay = dict(_require_mapping(replay, "activation_observe.replay"))
                _require_exact_keys(
                    replay,
                    frozenset({"epoch", "sequence", "reconstructed_floor", "complete"}),
                    "activation_observe.replay",
                )
                replay["epoch"] = _validate_activation_epoch(replay["epoch"], "activation_observe.replay.epoch")
                replay["sequence"] = _require_nonnegative_int(
                    replay["sequence"],
                    "activation_observe.replay.sequence",
                    positive=True,
                )
                floor = replay["reconstructed_floor"]
                if floor is not None:
                    floor = _require_nonnegative_int(
                        floor,
                        "activation_observe.replay.reconstructed_floor",
                        positive=True,
                    )
                if (
                    replay["epoch"] != checkpoint["epoch"]
                    or replay["sequence"] > checkpoint["sequence"]
                    or (floor is not None and floor > replay["sequence"])
                    or type(replay["complete"]) is not bool
                    or replay["complete"] != (replay["sequence"] == checkpoint["sequence"])
                ):
                    raise ValueError("activation_observe replay evidence is inconsistent.")
                if (
                    request.parameters["replay_epoch"] == checkpoint["epoch"]
                    and replay["sequence"] < request.parameters["replay_sequence"]
                ):
                    raise ValueError("activation_observe replay moved backwards.")
                replay["reconstructed_floor"] = floor
            evidence["checkpoint"] = checkpoint
            evidence["replay"] = replay
        elif request.operation_kind == "activation_consumer_register":
            outcome = evidence.get("outcome")
            if outcome == "registered":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "acknowledgement"}),
                    "activation_consumer_register evidence",
                )
                if evidence["acknowledgement"] is not None:
                    evidence["acknowledgement"] = _validate_activation_checkpoint(
                        evidence["acknowledgement"], "activation_consumer_register.acknowledgement"
                    )
            elif outcome == "stale":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "acknowledgement", "reason"}),
                    "activation_consumer_register stale evidence",
                )
                if evidence["acknowledgement"] is not None or evidence["reason"] != "binding_fence":
                    raise ValueError("activation_consumer_register stale evidence is invalid.")
            else:
                raise ValueError("activation_consumer_register outcome is invalid.")
        elif request.operation_kind == "activation_consumer_ack":
            outcome = evidence.get("outcome")
            if outcome == "acknowledged":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "acknowledgement"}),
                    "activation_consumer_ack evidence",
                )
                acknowledgement = _validate_activation_checkpoint(
                    evidence["acknowledgement"], "activation_consumer_ack.acknowledgement"
                )
                if acknowledgement != {
                    "epoch": request.parameters["epoch"],
                    "sequence": request.parameters["sequence"],
                }:
                    raise ValueError("activation_consumer_ack evidence differs from its request.")
                evidence["acknowledgement"] = acknowledgement
            elif outcome == "stale":
                _require_exact_keys(
                    evidence,
                    frozenset({"outcome", "acknowledgement", "reason"}),
                    "activation_consumer_ack stale evidence",
                )
                if evidence["acknowledgement"] is not None or evidence["reason"] != "binding_fence":
                    raise ValueError("activation_consumer_ack stale evidence is invalid.")
            else:
                raise ValueError("activation_consumer_ack outcome is invalid.")
        elif request.operation_kind == "activation_consumer_retire":
            _require_exact_keys(
                evidence,
                frozenset({"outcome", "consumer_existed"}),
                "activation_consumer_retire evidence",
            )
            if evidence["outcome"] not in {"retired", "stale"}:
                raise ValueError("activation_consumer_retire outcome is invalid.")
            if type(evidence["consumer_existed"]) is not bool:
                raise ValueError("activation_consumer_retire consumer_existed must be a boolean.")
            if evidence["outcome"] == "stale" and evidence["consumer_existed"]:
                raise ValueError("stale activation_consumer_retire cannot report an existing consumer.")
        elif request.operation_kind == "authority_service":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "runtime_id",
                        "executor_epoch",
                        "project_id",
                        "canonical_shared_root",
                        "registration_generation",
                        "registry_revision",
                        "machine_name",
                        "service_action",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                        "source_revisions",
                        "attempt_phase",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_service evidence",
            )
            parameters = request.parameters
            provenance = {
                "runtime_id": request.runtime_id,
                "executor_epoch": request.executor_epoch,
                "project_id": request.project_id,
                "canonical_shared_root": request.canonical_shared_root,
                "registration_generation": request.registration_generation,
                "registry_revision": request.registry_revision,
            }
            for field, expected in provenance.items():
                if evidence[field] != expected:
                    raise ValueError(f"authority_service evidence {field} differs from its request.")
            for field in (
                "machine_name",
                "service_action",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_identity",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_service evidence {field} differs from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in _AUTHORITY_SERVICE_OUTCOMES or (
                (outcome == "observed_current" and reason is not None)
                or (outcome == "observed_stale" and reason not in _AUTHORITY_SERVICE_REASONS)
            ):
                raise ValueError("authority_service outcome and reason are invalid.")
            revisions = dict(_require_mapping(evidence["source_revisions"], "authority_service.source_revisions"))
            _require_exact_keys(
                revisions,
                frozenset({"task", "attempt_digest"}),
                "authority_service.source_revisions",
            )
            task_revision = revisions["task"]
            if task_revision is not None:
                revisions["task"] = _require_nonnegative_int(task_revision, "source_revisions.task")
            attempt_digest = revisions["attempt_digest"]
            if attempt_digest is not None and (
                not isinstance(attempt_digest, str) or not _SHA256_DIGEST.fullmatch(attempt_digest)
            ):
                raise ValueError("source_revisions.attempt_digest is invalid.")
            phase = evidence["attempt_phase"]
            if phase is not None and phase not in {
                "claimed",
                "starting",
                "running",
                "succeeded",
                "failed",
                "cancelled",
                "orphaned",
            }:
                raise ValueError("authority_service attempt_phase is invalid.")
            if evidence["authority_granted"] is not False:
                raise ValueError("read-only authority_service cannot grant authority.")
            if outcome == "observed_current" and (
                phase != "running" or revisions["task"] is None or revisions["attempt_digest"] is None
            ):
                raise ValueError("current authority_service evidence requires running versioned truth.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("read-only authority_service cannot return local effects.")
            evidence["source_revisions"] = revisions
        elif request.operation_kind == "authority_renewal":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                        "source_revisions",
                        "committed_revisions",
                        "lease_expires_at",
                        "renew_after_seconds",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_renewal evidence",
            )
            parameters = request.parameters
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_identity",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_renewal evidence {field} differs from its request.")
            if evidence["source_revisions"] != request.source_revisions:
                raise ValueError("authority_renewal source revisions differ from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in _AUTHORITY_RENEWAL_OUTCOMES:
                raise ValueError("authority_renewal outcome is invalid.")
            if outcome in {"renewed", "not_required"} and reason is not None:
                raise ValueError("authority_renewal successful outcome cannot have a reason.")
            if outcome == "observed_stale" and reason not in _AUTHORITY_SERVICE_REASONS:
                raise ValueError("authority_renewal stale reason is invalid.")
            if outcome == "termination_requested" and reason != "termination_pending":
                raise ValueError("authority_renewal termination reason is invalid.")
            committed = dict(_require_mapping(evidence["committed_revisions"], "authority_renewal.committed_revisions"))
            _require_exact_keys(
                committed,
                frozenset({"task", "attempt_digest"}),
                "authority_renewal.committed_revisions",
            )
            if committed["task"] is not None:
                committed["task"] = _require_nonnegative_int(committed["task"], "committed_revisions.task")
            digest = committed["attempt_digest"]
            if digest is not None and (not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest)):
                raise ValueError("committed_revisions.attempt_digest is invalid.")
            if outcome != "observed_stale" and (committed["task"] is None or digest is None):
                raise ValueError("authority_renewal current outcome requires exact committed revisions.")
            expiry = evidence["lease_expires_at"]
            renew_after = evidence["renew_after_seconds"]
            if outcome == "renewed":
                _require_timestamp(expiry, "authority_renewal.lease_expires_at")
                if (
                    type(renew_after) not in {int, float}
                    or not math.isfinite(renew_after)
                    or not 0 < renew_after <= 86_400
                ):
                    raise ValueError("authority_renewal renew_after_seconds is invalid.")
            elif expiry is not None:
                raise ValueError("authority_renewal non-renewed outcome cannot have an expiry.")
            elif renew_after is not None:
                raise ValueError("authority_renewal non-renewed outcome cannot have a cadence.")
            if evidence["authority_granted"] is not False:
                raise ValueError("authority_renewal evidence cannot grant reusable authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("authority_renewal worker cannot return local effects.")
            evidence["committed_revisions"] = committed
        elif request.operation_kind == "authority_orphan_recovery":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                        "binding_signature",
                        "source_revisions",
                        "committed_revisions",
                        "recovered_fencing_token",
                        "lease_expires_at",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_orphan_recovery evidence",
            )
            parameters = request.parameters
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_identity",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_orphan_recovery evidence {field} differs from its request.")
            if list(evidence["binding_signature"]) != list(parameters["binding_signature"]):
                raise ValueError("authority_orphan_recovery evidence binding_signature differs from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in _AUTHORITY_ORPHAN_RECOVERY_OUTCOMES:
                raise ValueError("authority_orphan_recovery outcome is invalid.")
            if (outcome == "stale" and reason not in _PROJECT_IO_STALE_REASONS) or (
                outcome != "stale" and reason is not None
            ):
                raise ValueError("authority_orphan_recovery reason is invalid.")
            source = _validate_observed_authority_revisions(
                evidence["source_revisions"], "authority_orphan_recovery.source_revisions"
            )
            committed = _validate_observed_authority_revisions(
                evidence["committed_revisions"], "authority_orphan_recovery.committed_revisions"
            )
            if outcome != "stale" and (
                source["task"] is None
                or source["attempt_digest"] is None
                or committed["task"] is None
                or committed["attempt_digest"] is None
            ):
                raise ValueError("recovered orphan evidence requires exact revisions.")
            token = evidence["recovered_fencing_token"]
            if outcome == "stale":
                if token is not None or evidence["lease_expires_at"] is not None:
                    raise ValueError("stale orphan recovery evidence cannot contain a target lease.")
            else:
                token = _require_nonnegative_int(token, "recovered_fencing_token", positive=True)
                if token <= parameters["fencing_token"]:
                    raise ValueError("recovered_fencing_token must advance the expired token.")
                _require_timestamp(evidence["lease_expires_at"], "authority_orphan_recovery.lease_expires_at")
            signature = evidence["binding_signature"]
            if (
                not isinstance(signature, list)
                or not 1 <= len(signature) <= 16
                or any(not isinstance(item, str) or not item or len(item) > 4096 for item in signature)
            ):
                raise ValueError("authority_orphan_recovery evidence binding_signature is invalid.")
            if evidence["authority_granted"] is not False:
                raise ValueError("orphan recovery cannot grant reusable authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("orphan recovery cannot return local effects.")
            evidence["source_revisions"] = source
            evidence["committed_revisions"] = committed
        elif request.operation_kind == "authority_termination_commit":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "decision_id",
                        "decision_token",
                        "authority_outcome",
                        "decision_reason",
                        "process_identity",
                        "source_revisions",
                        "committed_revisions",
                        "shared_commitment",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_termination_commit evidence",
            )
            parameters = request.parameters
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "decision_id",
                "decision_token",
                "authority_outcome",
                "process_identity",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_termination_commit evidence {field} differs from its request.")
            if evidence["decision_reason"] != parameters["reason"]:
                raise ValueError("authority_termination_commit decision_reason differs from its request.")
            if evidence["source_revisions"] != request.source_revisions:
                raise ValueError("authority_termination_commit source revisions differ from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in {"committed", "already_committed", "stale"}:
                raise ValueError("authority_termination_commit outcome is invalid.")
            if (outcome == "stale" and reason not in _PROJECT_IO_STALE_REASONS) or (
                outcome != "stale" and reason is not None
            ):
                raise ValueError("authority_termination_commit reason is invalid.")
            committed = _validate_observed_authority_revisions(
                evidence["committed_revisions"], "authority_termination_commit.committed_revisions"
            )
            if outcome != "stale" and (committed["task"] is None or committed["attempt_digest"] is None):
                raise ValueError("committed termination evidence requires exact revisions.")
            expected_commitment = "committed" if outcome in {"committed", "already_committed"} else None
            if evidence["shared_commitment"] != expected_commitment:
                raise ValueError("authority_termination_commit shared_commitment is invalid.")
            if evidence["authority_granted"] is not False:
                raise ValueError("termination commit cannot grant reusable authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("termination commit cannot return local effects.")
            evidence["committed_revisions"] = committed
        elif request.operation_kind == "authority_terminal_observe":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                        "mode",
                        "task_phase",
                        "attempt_phase",
                        "attempt_result_reason",
                        "attempt_exit_code",
                        "execution_machine_name",
                        "reservation_machine_name",
                        "termination_result",
                        "cancel_requested",
                        "source_revisions",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_terminal_observe evidence",
            )
            parameters = request.parameters
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "process_identity",
                "mode",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_terminal_observe evidence {field} differs from its request.")
            outcome = evidence["outcome"]
            if outcome != "stale" and evidence["reservation_id"] != parameters["reservation_id"]:
                raise ValueError("authority_terminal_observe reservation_id differs from its request.")
            if evidence["reservation_id"] is not None:
                evidence["reservation_id"] = _require_identifier(
                    evidence["reservation_id"], "authority_terminal_observe.reservation_id"
                )
            if type(evidence["cancel_requested"]) is not bool:
                raise ValueError("authority_terminal_observe cancel_requested must be a boolean.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in {"current", "already_terminal", "settled_terminal", "stale"}:
                raise ValueError("authority_terminal_observe outcome is invalid.")
            if (outcome == "stale" and reason not in _PROJECT_IO_STALE_REASONS) or (
                outcome != "stale" and reason is not None
            ):
                raise ValueError("authority_terminal_observe reason is invalid.")
            for field, choices in (
                ("task_phase", {"queued", "running", "succeeded", "failed", "cancelled", "blocked"}),
                ("attempt_phase", {"claimed", "starting", "running", "succeeded", "failed", "cancelled", "orphaned"}),
            ):
                phase = evidence[field]
                if phase is not None and phase not in choices:
                    raise ValueError(f"authority_terminal_observe {field} is invalid.")
            attempt_result_reason = evidence["attempt_result_reason"]
            if attempt_result_reason is not None:
                evidence["attempt_result_reason"] = _require_text(
                    attempt_result_reason,
                    "authority_terminal_observe.attempt_result_reason",
                    maximum=128,
                )
            if evidence["attempt_exit_code"] is not None and type(evidence["attempt_exit_code"]) is not int:
                raise ValueError("authority_terminal_observe.attempt_exit_code must be an integer or null.")
            for field in ("execution_machine_name", "reservation_machine_name"):
                machine = evidence[field]
                if machine is not None:
                    evidence[field] = _require_identifier(machine, f"authority_terminal_observe.{field}")
            terminal_result = evidence["termination_result"]
            if terminal_result is not None:
                evidence["termination_result"] = _require_text(
                    terminal_result, "authority_terminal_observe.termination_result", maximum=128
                )
            revisions = _validate_observed_authority_revisions(
                evidence["source_revisions"], "authority_terminal_observe.source_revisions"
            )
            if outcome in {"current", "settled_terminal"} and (
                revisions["task"] is None or revisions["attempt_digest"] is None
            ):
                raise ValueError(f"{outcome} terminal observation requires exact source revisions.")
            if outcome == "already_terminal" and (
                evidence["task_phase"] not in _PROJECT_IO_TERMINAL_PHASES
                or evidence["attempt_phase"] != evidence["task_phase"]
            ):
                raise ValueError("already_terminal evidence requires matching terminal Task and Attempt phases.")
            if outcome == "settled_terminal" and (
                evidence["task_phase"] is None
                or evidence["attempt_phase"] not in _PROJECT_IO_TERMINAL_PHASES
                or evidence["execution_machine_name"] != parameters["machine_name"]
                or evidence["reservation_machine_name"] != parameters["machine_name"]
            ):
                raise ValueError("settled_terminal evidence requires settled Task and Attempt machine identity.")
            if evidence["authority_granted"] is not False:
                raise ValueError("read-only terminal observation cannot grant authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("read-only terminal observation cannot return local effects.")
            evidence["source_revisions"] = revisions
        elif request.operation_kind == "authority_terminal_publish":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "outcome",
                        "reason",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "reservation_machine_name",
                        "process_identity",
                        "mode",
                        "phase",
                        "transition_reason",
                        "exit_code",
                        "termination_result",
                        "source_revisions",
                        "committed_revisions",
                        "lifecycle_event",
                        "transition_digest",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_terminal_publish evidence",
            )
            parameters = request.parameters
            outcome = evidence["outcome"]
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "process_identity",
                "mode",
                "phase",
                "exit_code",
                "termination_result",
                "transition_digest",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_terminal_publish evidence {field} differs from its request.")
            if evidence["transition_reason"] != parameters["reason"]:
                raise ValueError("authority_terminal_publish transition_reason differs from its request.")
            if outcome != "stale" and evidence["reservation_id"] != parameters["reservation_id"]:
                raise ValueError("authority_terminal_publish reservation_id differs from its request.")
            if evidence["reservation_id"] is not None:
                evidence["reservation_id"] = _require_identifier(
                    evidence["reservation_id"], "authority_terminal_publish.reservation_id"
                )
            if evidence["reservation_machine_name"] is not None:
                evidence["reservation_machine_name"] = _require_identifier(
                    evidence["reservation_machine_name"], "authority_terminal_publish.reservation_machine_name"
                )
            if evidence["source_revisions"] != request.source_revisions:
                raise ValueError("authority_terminal_publish source revisions differ from its request.")
            outcome = evidence["outcome"]
            reason = evidence["reason"]
            if outcome not in {"committed", "already_committed", "stale"}:
                raise ValueError("authority_terminal_publish outcome is invalid.")
            if (outcome == "stale" and reason not in _PROJECT_IO_STALE_REASONS) or (
                outcome != "stale" and reason is not None
            ):
                raise ValueError("authority_terminal_publish reason is invalid.")
            committed = _validate_observed_authority_revisions(
                evidence["committed_revisions"], "authority_terminal_publish.committed_revisions"
            )
            if outcome != "stale" and (committed["task"] is None or committed["attempt_digest"] is None):
                raise ValueError("committed terminal publication requires exact revisions.")
            event = evidence["lifecycle_event"]
            if outcome != "stale" and event is None:
                raise ValueError("committed terminal publication requires a lifecycle event.")
            if event is not None:
                event = _validate_terminal_lifecycle_event(event, "authority_terminal_publish.lifecycle_event")
                if (
                    event["task_id"] != parameters["task_id"]
                    or event["attempt_id"] != parameters["attempt_id"]
                    or event["attempt_number"] != parameters["attempt_number"]
                    or event["phase"] != parameters["phase"]
                    or event["reason"] != parameters["reason"]
                    or event["exit_code"] != parameters["exit_code"]
                    or event["task_revision"] != committed["task"]
                ):
                    raise ValueError("lifecycle event differs from the committed terminal transition.")
            if evidence["authority_granted"] is not False:
                raise ValueError("terminal publication cannot grant reusable authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("terminal publication cannot return local effects.")
            evidence["committed_revisions"] = committed
            evidence["lifecycle_event"] = event
        elif request.operation_kind == "authority_running_publish":
            _require_exact_keys(
                evidence,
                frozenset(
                    {
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                        "process_created_at",
                        "outcome",
                        "reason",
                        "transitioned_to_running",
                        "authority_granted",
                        "local_effects",
                    }
                ),
                "authority_running_publish evidence",
            )
            parameters = request.parameters
            evidence["machine_name"] = _require_identifier(
                evidence["machine_name"], "authority_running_publish evidence.machine_name"
            )
            evidence["task_id"] = _require_identifier(evidence["task_id"], "authority_running_publish evidence.task_id")
            evidence["attempt_id"] = _require_identifier(
                evidence["attempt_id"], "authority_running_publish evidence.attempt_id"
            )
            evidence["attempt_number"] = _require_nonnegative_int(
                evidence["attempt_number"], "authority_running_publish evidence.attempt_number", positive=True
            )
            evidence["fencing_token"] = _require_nonnegative_int(
                evidence["fencing_token"], "authority_running_publish evidence.fencing_token", positive=True
            )
            evidence["reservation_id"] = _require_identifier(
                evidence["reservation_id"], "authority_running_publish evidence.reservation_id"
            )
            evidence["process_created_at"] = _require_timestamp(
                evidence["process_created_at"], "authority_running_publish evidence.process_created_at"
            )
            for field in (
                "machine_name",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_created_at",
            ):
                if evidence[field] != parameters[field]:
                    raise ValueError(f"authority_running_publish evidence {field} differs from its request.")
            evidence["process_identity"] = _validate_supervised_process_identity(
                evidence["process_identity"], "authority_running_publish evidence process_identity"
            )
            if evidence["process_identity"] != parameters["process_identity"]:
                raise ValueError("authority_running_publish evidence process_identity differs from its request.")
            if evidence["outcome"] != "processed" or evidence["reason"] is not None:
                raise ValueError("authority_running_publish outcome or reason is invalid.")
            if type(evidence["transitioned_to_running"]) is not bool:
                raise ValueError("authority_running_publish transitioned_to_running must be a boolean.")
            if evidence["authority_granted"] is not False:
                raise ValueError("running publication cannot grant reusable authority.")
            if type(evidence["local_effects"]) is not list or evidence["local_effects"]:
                raise ValueError("running publication cannot return local effects.")
        else:
            raise ValueError("completed result operation is unsupported.")
    elif status == "retryable_error":
        if set(evidence) not in (set(), {"exception_type"}):
            raise ValueError("retryable_error evidence has unknown fields.")
        if "exception_type" in evidence and (
            not isinstance(evidence["exception_type"], str) or not _EXCEPTION_TYPE.fullmatch(evidence["exception_type"])
        ):
            raise ValueError("exception_type evidence is invalid.")
    elif evidence:
        raise ValueError(f"{status} results must not carry evidence.")
    return evidence


def _validate_source_revisions(value: object) -> dict[str, JSONScalar]:
    revisions = _require_mapping(value, "operation source_revisions")
    if len(revisions) > 16:
        raise ValueError("operation source_revisions contains too many entries.")
    normalized: dict[str, JSONScalar] = {}
    for key, revision in revisions.items():
        name = _require_identifier(key, "operation source_revisions key", maximum=64)
        if revision is None or type(revision) in {int, bool}:
            normalized[name] = revision
        elif type(revision) is float and math.isfinite(revision):
            normalized[name] = revision
        elif isinstance(revision, str) and len(revision) <= 128 and "\x00" not in revision:
            normalized[name] = revision
        else:
            raise ValueError("operation source_revisions values must be bounded JSON scalars.")
    return normalized


def _validate_activation_epoch(value: object, label: str) -> str:
    text = _require_text(value, label, maximum=32)
    try:
        parsed = uuid.UUID(hex=text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical nonzero UUID hex value.") from exc
    if parsed.int == 0 or text != parsed.hex:
        raise ValueError(f"{label} must be a canonical nonzero UUID hex value.")
    return text


def _validate_activation_checkpoint(value: object, label: str) -> dict[str, Any]:
    checkpoint = dict(_require_mapping(value, label))
    _require_exact_keys(checkpoint, frozenset({"epoch", "sequence"}), label)
    checkpoint["epoch"] = _validate_activation_epoch(checkpoint["epoch"], f"{label}.epoch")
    checkpoint["sequence"] = _require_nonnegative_int(checkpoint["sequence"], f"{label}.sequence", positive=True)
    return checkpoint


@dataclass(frozen=True, slots=True)
class ProjectIOEpoch:
    protocol_version: int
    runtime_id: str
    executor_epoch: str
    active: bool
    created_at: str
    completion_sequence: int

    def __post_init__(self) -> None:
        payload = self._payload()
        _require_exact_keys(
            payload,
            frozenset(
                {"protocol_version", "runtime_id", "executor_epoch", "active", "created_at", "completion_sequence"}
            ),
            "project_io_epoch",
        )
        if type(self.protocol_version) is not int or self.protocol_version != PROJECT_IO_PROTOCOL_VERSION:
            raise ValueError("Project I/O epoch protocol version is unsupported.")
        runtime_id = _require_text(self.runtime_id, "runtime_id", maximum=64)
        if not _RUNTIME_ID.fullmatch(runtime_id):
            raise ValueError("runtime_id must be a lowercase 64-character hexadecimal ID.")
        epoch = _require_text(self.executor_epoch, "executor_epoch", maximum=32)
        if not _HEX_ID.fullmatch(epoch):
            raise ValueError("executor_epoch must be a lowercase 32-character hexadecimal ID.")
        if type(self.active) is not bool:
            raise ValueError("epoch active must be a boolean.")
        _require_timestamp(self.created_at, "created_at")
        if type(self.completion_sequence) is not int or self.completion_sequence < 0:
            raise ValueError("completion_sequence must be a nonnegative integer.")
        _require_record_size({"project_io_epoch": payload}, "project_io_epoch")

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "runtime_id": self.runtime_id,
            "executor_epoch": self.executor_epoch,
            "active": self.active,
            "created_at": self.created_at,
            "completion_sequence": self.completion_sequence,
        }

    def to_dict(self) -> dict[str, Any]:
        value = {"project_io_epoch": self._payload()}
        _require_record_size(value, "project_io_epoch")
        return value

    @classmethod
    def from_dict(cls, value: object) -> ProjectIOEpoch:
        envelope = _require_envelope(value, "project_io_epoch", "project_io_epoch")
        keys = frozenset(
            {"protocol_version", "runtime_id", "executor_epoch", "active", "created_at", "completion_sequence"}
        )
        _require_exact_keys(envelope, keys, "project_io_epoch")
        _require_record_size({"project_io_epoch": dict(envelope)}, "project_io_epoch")
        return cls(**dict(envelope))
