"""Machine-local bounded executor for typed Project I/O requests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..runtime.locks import exclusive
from ..runtime.paths import machine_project_paths, machine_runtime_paths
from ..runtime.records import utc_now, validate_identifier
from ..runtime.resources.reservations import (
    ReservationIdentity,
    classify_exact_reservation,
    classify_executor_offer,
    release_executor_offer,
)
from ..runtime.store import atomic_replace, read_json_limited, require_json_size
from .group_service_transport import GroupServiceRequestSpec
from .project_io_protocol import (
    PROJECT_IO_CAPACITY,
    PROJECT_IO_MAX_RECORD_BYTES,
    PROJECT_IO_MAX_RESOLVED_BYTES,
    PROJECT_IO_MAX_RESOLVED_RECORDS,
    PROJECT_IO_OPERATIONS,
    PROJECT_IO_OVERDUE_SECONDS,
    PROJECT_IO_PROTOCOL_VERSION,
    PROJECT_IO_STOP_GRACE_SECONDS,
    PROJECT_IO_SUPPORTED_HANG_LIMIT,
    ProjectIOEpoch,
    ProjectIOProcess,
    ProjectIORequest,
    ProjectIOResult,
    authority_terminal_transition_digest,
)
from .project_io_transport_support import (
    _PROCESS_ABSENT,
    _PROCESS_LIVE,
    _PROCESS_UNVERIFIED,
    ProjectIOProtocolError,
    _process_start_time_ticks,
    _resolve_runtime_id,
    inspect_process_identity,
)

_RECORD_ID = re.compile(r"^[0-9a-f]{32}$")
_RESOLVED_NAME = re.compile(r"^(\d{20})-([0-9a-f]{32})\.json$")
_WORKER_HANDSHAKE_SECONDS = 2.0
_START_IDENTITY_SECONDS = 1.0
_EVENT_ID = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# Every typed operation has one explicit recovery policy when a positively
# absent worker published no result. Keep this partition exhaustive so adding
# a protocol operation cannot silently retain a dead process record forever.
_OUTCOME_UNKNOWN_ON_WORKER_EXIT = frozenset(
    {
        "upgrade_service",
        "submission_control_service",
        "observation_service",
        "notification_service",
        "group_service_advance",
        "progress_projection",
        "recovery_source_hold",
        "recovery_admission",
        "recovery_source_release",
        "recovery_group_authority",
        "recovery_capture_transition",
        "scheduler_claim",
        "scheduler_cursor_commit",
        "scheduler_launch_authorize",
        "scheduler_due_offer",
        "scheduler_ready_index_build",
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
    }
)
_RETRYABLE_ON_WORKER_EXIT = frozenset(
    {
        "validate_binding",
        "group_service_probe",
        "legacy_capture_read",
        "legacy_capture_scan",
        "scheduler_observe",
        "scheduler_primary_probe",
        "scheduler_quiescence_probe",
        "scheduler_reservation_reconcile",
        "activation_observe",
        "authority_service",
        "authority_terminal_observe",
    }
)
_REPLAYABLE_REQUEST_OPERATION_KINDS = frozenset({"group_service_advance"})
if (
    _OUTCOME_UNKNOWN_ON_WORKER_EXIT & _RETRYABLE_ON_WORKER_EXIT
    or _OUTCOME_UNKNOWN_ON_WORKER_EXIT | _RETRYABLE_ON_WORKER_EXIT != PROJECT_IO_OPERATIONS
):
    raise RuntimeError("Project I/O worker-exit recovery policy must partition every operation.")


def _request_digest(request: ProjectIORequest) -> str:
    encoded = json.dumps(
        request.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


class ProjectIOExecutor:
    """Controller and bounded durable store for one MachineRuntime."""

    def __init__(self, runtime: object, *, runtime_id: str | None = None) -> None:
        self.runtime = runtime
        root_value = getattr(runtime, "root", runtime)
        self.root = Path(root_value).expanduser().resolve()
        self.paths = machine_runtime_paths(self.root)
        self.runtime_id = runtime_id or _resolve_runtime_id(self.root)
        self._children: dict[str, subprocess.Popen[bytes]] = {}
        self._reconciliation_unknown = False

    def _ensure_layout(self) -> None:
        ensure_layout = getattr(self.runtime, "ensure_layout", None)
        if callable(ensure_layout):
            ensure_layout(create_identity=False)
        else:
            for name in (
                "project_io_root",
                "project_io_requests",
                "project_io_results",
                "project_io_processes",
                "project_io_resolved",
            ):
                self.paths[name].mkdir(parents=True, exist_ok=True)
            self.paths["project_io_lock"].parent.mkdir(parents=True, exist_ok=True)

    def _record_path(self, lane: str, request_id: str) -> Path:
        if not _RECORD_ID.fullmatch(request_id):
            raise ValueError("request_id must be a lowercase 32-character hexadecimal ID.")
        return self.paths[f"project_io_{lane}"] / f"{request_id}.json"

    def _read_record(self, path: Path, record_type: str) -> dict[str, Any]:
        self._validate_local_path(path, must_exist=True)
        return read_json_limited(path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type=record_type)

    def _validate_local_path(self, path: Path, *, must_exist: bool) -> bool:
        try:
            relative = path.relative_to(self.paths["project_io_root"])
        except ValueError as exc:
            raise ProjectIOProtocolError("executor evidence path escaped its fixed root.") from exc
        if not relative.parts or ".." in relative.parts:
            raise ProjectIOProtocolError("executor evidence path escaped its fixed root.")
        directory = self.paths["project_io_root"]
        for part in relative.parts[:-1]:
            try:
                metadata = directory.stat(follow_symlinks=False)
            except FileNotFoundError:
                if must_exist:
                    raise
                return False
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ProjectIOProtocolError("executor evidence directory is not a locally owned real directory.")
            directory = directory / part
        try:
            parent_metadata = path.parent.stat(follow_symlinks=False)
        except FileNotFoundError:
            if must_exist:
                raise
            return False
        if not stat.S_ISDIR(parent_metadata.st_mode) or parent_metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError("executor evidence parent is not a locally owned real directory.")
        try:
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            if must_exist:
                raise
            return False
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise ProjectIOProtocolError("executor evidence is not a locally owned regular file.")
        return True

    def _write_record(self, path: Path, value: dict[str, Any], record_type: str) -> None:
        require_json_size(value, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type=record_type)
        result = atomic_replace(path, value)
        if result is None:
            raise OSError(f"durability could not be established for {record_type}.")

    def _load_epoch(self, *, missing_ok: bool = False) -> ProjectIOEpoch | None:
        path = self.paths["project_io_epoch"]
        try:
            value = self._read_record(path, "project_io_epoch")
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ProjectIOProtocolError("executor epoch record is missing.") from None
        epoch = ProjectIOEpoch.from_dict(value)
        if epoch.runtime_id != self.runtime_id:
            raise ProjectIOProtocolError("executor epoch belongs to a foreign runtime identity.")
        return epoch

    def _load_request(self, request_id: str) -> ProjectIORequest:
        request = ProjectIORequest.from_dict(
            self._read_record(self._record_path("requests", request_id), "project_io_request")
        )
        if request.request_id != request_id or request.runtime_id != self.runtime_id:
            raise ProjectIOProtocolError("request record identity does not match its local runtime path.")
        return request

    def _load_process(self, request_id: str) -> ProjectIOProcess:
        process = ProjectIOProcess.from_dict(
            self._read_record(self._record_path("processes", request_id), "project_io_process")
        )
        if process.request.request_id != request_id or process.request.runtime_id != self.runtime_id:
            raise ProjectIOProtocolError("process record identity does not match its local runtime path.")
        return process

    def _load_result(self, request_id: str) -> ProjectIOResult:
        result = ProjectIOResult.from_dict(
            self._read_record(self._record_path("results", request_id), "project_io_result")
        )
        if result.request.request_id != request_id or result.request.runtime_id != self.runtime_id:
            raise ProjectIOProtocolError("result record identity does not match its local runtime path.")
        return result

    def _directory_ids(self, lane: str, *, limit: int = PROJECT_IO_CAPACITY) -> tuple[list[str], bool]:
        directory = self.paths[f"project_io_{lane}"]
        try:
            metadata = directory.stat(follow_symlinks=False)
        except FileNotFoundError:
            return [], False
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError(f"executor {lane} path is not a locally owned real directory.")
        names: list[str] = []
        too_many = False
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(names) >= limit:
                    too_many = True
                    break
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ProjectIOProtocolError(f"executor {lane} contains a non-regular record.")
                match = re.fullmatch(r"([0-9a-f]{32})\.json", entry.name)
                if match is None:
                    raise ProjectIOProtocolError(f"executor {lane} contains an invalid record filename.")
                names.append(match.group(1))
        return names, too_many

    def _scan_records(
        self,
    ) -> tuple[dict[str, ProjectIORequest], dict[str, ProjectIOProcess], dict[str, ProjectIOResult]]:
        request_ids, requests_overflow = self._directory_ids("requests", limit=PROJECT_IO_CAPACITY + 1)
        process_ids, processes_overflow = self._directory_ids("processes", limit=PROJECT_IO_CAPACITY + 1)
        result_ids, results_overflow = self._directory_ids("results", limit=PROJECT_IO_CAPACITY + 1)
        all_ids = set(request_ids) | set(process_ids) | set(result_ids)
        if requests_overflow or processes_overflow or results_overflow or len(all_ids) > PROJECT_IO_CAPACITY:
            raise ProjectIOProtocolError("executor has more than four unresolved request identities.")
        requests = {request_id: self._load_request(request_id) for request_id in request_ids}
        processes = {request_id: self._load_process(request_id) for request_id in process_ids}
        results = {request_id: self._load_result(request_id) for request_id in result_ids}
        if set(processes) - set(requests) or set(results) - set(requests):
            raise ProjectIOProtocolError("executor evidence has a missing linked request.")
        for request_id, process in processes.items():
            if process.request != requests[request_id]:
                raise ProjectIOProtocolError("process record does not repeat its exact request identity.")
        for request_id, result in results.items():
            if result.request != requests[request_id]:
                raise ProjectIOProtocolError("result record does not repeat its exact request identity.")
        return requests, processes, results

    def _sequence_from_history(self) -> int:
        directory = self.paths["project_io_resolved"]
        try:
            metadata = directory.stat(follow_symlinks=False)
        except FileNotFoundError:
            return 0
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError("executor resolved path is not a locally owned real directory.")
        records: list[tuple[int, Path, int]] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(records) >= PROJECT_IO_MAX_RESOLVED_RECORDS + 1:
                    raise ProjectIOProtocolError("resolved history exceeds its bounded record limit.")
                match = _RESOLVED_NAME.fullmatch(entry.name)
                if match is None or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ProjectIOProtocolError("resolved history contains invalid local evidence.")
                info = entry.stat(follow_symlinks=False)
                if info.st_uid != os.geteuid() or info.st_nlink != 1:
                    raise ProjectIOProtocolError("resolved history contains foreign file ownership.")
                records.append((int(match.group(1)), directory / entry.name, info.st_size))
        return max((sequence for sequence, _path, _size in records), default=0)

    def _next_epoch(
        self,
        *,
        executor_epoch: str | None = None,
        active: bool,
        sequence: int | None = None,
    ) -> ProjectIOEpoch:
        if sequence is None:
            try:
                sequence = self._load_epoch().completion_sequence
            except (FileNotFoundError, ProjectIOProtocolError, OSError, ValueError, TypeError):
                sequence = self._sequence_from_history()
            sequence = max(sequence, self._sequence_from_history())
        return ProjectIOEpoch(
            protocol_version=PROJECT_IO_PROTOCOL_VERSION,
            runtime_id=self.runtime_id,
            executor_epoch=executor_epoch or uuid.uuid4().hex,
            active=active,
            created_at=utc_now(),
            completion_sequence=sequence,
        )

    def begin_epoch(self) -> str:
        """Durably fence prior work, publish a fresh epoch, then reconcile evidence."""
        self._ensure_layout()
        with exclusive(self.paths["project_io_lock"]):
            try:
                sequence = self._load_epoch().completion_sequence
            except (FileNotFoundError, ProjectIOProtocolError, OSError, ValueError, TypeError):
                sequence = self._sequence_from_history()
            epoch = self._next_epoch(active=True, sequence=sequence)
            self._write_record(self.paths["project_io_epoch"], epoch.to_dict(), "project_io_epoch")
            try:
                self._reconcile_locked()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
                self._reconciliation_unknown = True
            return epoch.executor_epoch

    def prepare_validate_binding(
        self,
        binding: object,
        registry_revision: int,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one read-only binding validation request within fixed limits."""
        machine_name = getattr(binding, "machine_name", None)
        registration_generation = getattr(binding, "registration_generation", None)
        if registration_generation is None:
            raise ValueError("validate_binding requires a registration_generation.")
        return self._prepare_request(
            binding,
            registry_revision,
            "validate_binding",
            {"machine_name": machine_name},
            source_revisions=source_revisions,
        )

    def prepare_scheduler_observe(
        self,
        binding: object,
        registry_revision: int,
        *,
        lane: str,
        admission_role: str,
        cursor_namespace: str,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one bounded read-only scheduling observation request."""
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_observe",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "lane": lane,
                "admission_role": admission_role,
                "cursor_namespace": cursor_namespace,
            },
            source_revisions=source_revisions,
        )

    def prepare_scheduler_quiescence_probe(
        self,
        binding: object,
        registry_revision: int,
        *,
        probe_state: Mapping[str, Any],
    ) -> ProjectIORequest:
        """Persist a read-only full-route scheduler retirement continuation."""
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_quiescence_probe",
            {"machine_name": getattr(binding, "machine_name", None), "probe_state": probe_state},
            source_revisions=None,
        )

    def prepare_scheduler_primary_probe(
        self,
        binding: object,
        registry_revision: int,
        *,
        lane: str,
        round_id: str,
        phase: str,
        capacity_digest: str,
        visible_capacity: int,
        free_capacity: int,
        group_gpu_usage: Mapping[str, int],
        probe_state: Mapping[str, Any],
    ) -> ProjectIORequest:
        """Persist an independent read-only primary-demand continuation."""
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_primary_probe",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "lane": lane,
                "round_id": round_id,
                "phase": phase,
                "capacity_digest": capacity_digest,
                "visible_capacity": visible_capacity,
                "free_capacity": free_capacity,
                "group_gpu_usage": group_gpu_usage,
                "probe_state": probe_state,
            },
            source_revisions=None,
        )

    def prepare_scheduler_cursor_commit(
        self,
        binding: object,
        registry_revision: int,
        *,
        cursor: Mapping[str, Any],
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one compare-and-commit request for advisory ready cursors."""
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_cursor_commit",
            {"machine_name": getattr(binding, "machine_name", None), "cursor": cursor},
            source_revisions=source_revisions,
        )

    def prepare_scheduler_claim(
        self,
        binding: object,
        registry_revision: int,
        *,
        lane: str,
        admission_role: str,
        candidate: Mapping[str, Any],
        offer: Mapping[str, Any],
        cursor: Mapping[str, Any],
        request_id: str,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one exact shared claim request for a preallocated local offer."""
        reservation_record = {
            "reservation_id": offer.get("reservation_id"),
            "acquisition_id": offer.get("acquisition_id"),
            "project_id": offer.get("project_id"),
            "shared_root": offer.get("shared_root"),
            "task_id": offer.get("task_id"),
            "attempt_id": offer.get("attempt_id"),
            "fencing_token": offer.get("fencing_token"),
            "gpu_ids": offer.get("gpu_ids") if offer.get("lane") == "gpu" else None,
            "cpu_slots": offer.get("cpu_slots") if offer.get("lane") == "cpu" else None,
            "executor_owner": {
                "executor_epoch": offer.get("executor_epoch"),
                "request_id": offer.get("request_id"),
                "registration_generation": offer.get("registration_generation"),
            },
        }
        identity = ReservationIdentity.from_record(reservation_record)
        if offer.get("offer_id") != identity.reservation_id:
            raise ProjectIOProtocolError("claim offer ID differs from its reservation identity.")
        if classify_executor_offer(self.runtime.root, identity) != "matching_provisional":
            raise ProjectIOProtocolError("claim offer is not the exact live provisional reservation.")
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_claim",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "lane": lane,
                "admission_role": admission_role,
                "candidate": candidate,
                "offer": offer,
                "cursor": cursor,
            },
            source_revisions=source_revisions,
            request_id=request_id,
            provisional_offer_id=offer.get("offer_id"),
        )

    def prepare_scheduler_launch_authorize(
        self,
        binding: object,
        registry_revision: int,
        *,
        claim_identity: Mapping[str, Any],
        reservation_identity: ReservationIdentity,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one exact launch-authorization request for an active local reservation."""
        if not isinstance(reservation_identity, ReservationIdentity):
            raise ValueError("launch authorization requires an exact ReservationIdentity.")
        claim_fields = {"task_id", "attempt_id", "attempt_number", "fencing_token", "reservation_id"}
        if not isinstance(claim_identity, Mapping) or set(claim_identity) != claim_fields:
            raise ValueError("launch authorization claim identity has missing or unknown fields.")
        claim = dict(claim_identity)
        shared_root = getattr(binding, "shared_root", None)
        canonical_shared_root = str(shared_root) if shared_root is not None else None
        if (
            reservation_identity.reservation_id != claim["reservation_id"]
            or reservation_identity.task_id != claim["task_id"]
            or reservation_identity.attempt_id != claim["attempt_id"]
            or reservation_identity.fencing_token != claim["fencing_token"]
            or reservation_identity.project_id != getattr(binding, "project_id", None)
            or reservation_identity.shared_root != canonical_shared_root
            or reservation_identity.registration_generation
            not in {None, getattr(binding, "registration_generation", None)}
        ):
            raise ProjectIOProtocolError(
                "launch authorization reservation does not match its claim and current binding identity."
            )
        try:
            offer_state = classify_exact_reservation(self.runtime.root, reservation_identity)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProjectIOProtocolError("launch authorization reservation could not be validated locally.") from exc
        if offer_state != "matching_active":
            raise ProjectIOProtocolError("launch authorization requires the exact active local reservation.")
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_launch_authorize",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "claim_identity": claim,
            },
            source_revisions=source_revisions,
            provisional_offer_id=reservation_identity.reservation_id,
        )

    def prepare_scheduler_reservation_reconcile(
        self,
        binding: object,
        registry_revision: int,
        *,
        reservation_identity: ReservationIdentity,
    ) -> ProjectIORequest:
        """Persist one shared-only classification request for an active reservation."""
        if not isinstance(reservation_identity, ReservationIdentity):
            raise ValueError("reservation reconciliation requires an exact ReservationIdentity.")
        shared_root = str(getattr(binding, "shared_root", None))
        if (
            reservation_identity.project_id != getattr(binding, "project_id", None)
            or reservation_identity.shared_root != shared_root
        ):
            raise ProjectIOProtocolError("reservation reconciliation identity differs from its binding.")
        try:
            state = classify_exact_reservation(self.runtime.root, reservation_identity)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProjectIOProtocolError("reservation reconciliation could not validate local identity.") from exc
        if state != "matching_active":
            raise ProjectIOProtocolError("reservation reconciliation requires the exact active local reservation.")
        identity = {
            "reservation_id": reservation_identity.reservation_id,
            "acquisition_id": reservation_identity.acquisition_id,
            "project_id": reservation_identity.project_id,
            "task_id": reservation_identity.task_id,
            "attempt_id": reservation_identity.attempt_id,
            "fencing_token": reservation_identity.fencing_token,
            "gpu_ids": list(reservation_identity.gpu_ids) if reservation_identity.cpu_slots is None else None,
            "cpu_slots": reservation_identity.cpu_slots,
            "shared_root": reservation_identity.shared_root,
            "registration_generation": reservation_identity.registration_generation,
            "executor_epoch": reservation_identity.executor_epoch,
            "executor_request_id": reservation_identity.executor_request_id,
        }
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_reservation_reconcile",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "reservation_identity": identity,
            },
            source_revisions={},
        )

    def prepare_scheduler_due_offer(
        self,
        binding: object,
        registry_revision: int,
    ) -> ProjectIORequest:
        """Persist one bounded due-offer maintenance request."""
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("due-offer maintenance requires an enabled current binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("due-offer maintenance binding belongs to a different runtime.")
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_due_offer",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions={},
        )

    def prepare_scheduler_ready_index_build(
        self,
        binding: object,
        registry_revision: int,
    ) -> ProjectIORequest:
        """Persist one bounded ready-index build request."""
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("ready-index build requires an enabled current binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("ready-index build binding belongs to a different runtime.")
        binding_runtime_root = getattr(binding, "runtime_root", None)
        if not isinstance(binding_runtime_root, str) or Path(binding_runtime_root).expanduser().resolve() != self.root:
            raise ProjectIOProtocolError("ready-index build binding has a different runtime root.")
        return self._prepare_request(
            binding,
            registry_revision,
            "scheduler_ready_index_build",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions={},
        )

    def prepare_maintenance_descriptor_advance(
        self,
        binding: object,
        registry_revision: int,
    ) -> ProjectIORequest:
        """Persist one bounded Project maintenance-descriptor advancement request."""
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("descriptor maintenance requires an enabled current binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("descriptor maintenance binding belongs to a different runtime.")
        binding_runtime_root = getattr(binding, "runtime_root", None)
        if not isinstance(binding_runtime_root, str) or Path(binding_runtime_root).expanduser().resolve() != self.root:
            raise ProjectIOProtocolError("descriptor maintenance binding has a different runtime root.")
        return self._prepare_request(
            binding,
            registry_revision,
            "maintenance_descriptor_advance",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions={},
        )

    def prepare_maintenance_flush_event(
        self,
        binding: object,
        registry_revision: int,
        *,
        bucket: str,
        filename: str,
        event_id: str,
        sha256: str,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist a digest-bound request for one current binding-local event."""
        project_id = getattr(binding, "project_id", None)
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("maintenance event flush requires an enabled current binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("maintenance event flush binding belongs to a different runtime.")
        binding_runtime_root = getattr(binding, "runtime_root", None)
        if not isinstance(binding_runtime_root, str) or Path(binding_runtime_root).expanduser().resolve() != self.root:
            raise ProjectIOProtocolError("maintenance event flush binding has a different runtime root.")
        registration_generation = getattr(binding, "registration_generation", None)
        if registration_generation is None:
            raise ValueError("maintenance event flush requires a registration_generation.")
        try:
            validate_identifier(bucket, "bucket")
            validate_identifier(project_id, "project_id")
        except ValueError as exc:
            raise ValueError("maintenance event flush identity is invalid.") from exc
        if bucket in {".", ".."} or not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
            raise ValueError("maintenance event flush event identity is invalid.")
        if filename != f"{event_id}.json":
            raise ValueError("maintenance event flush filename must match event_id.")
        if not isinstance(sha256, str) or not _SHA256_DIGEST.fullmatch(sha256):
            raise ValueError("maintenance event flush sha256 must be a lowercase 64-character digest.")

        self._validate_project_event_source(project_id, bucket, filename, sha256)
        return self._prepare_request(
            binding,
            registry_revision,
            "maintenance_flush_event",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "bucket": bucket,
                "filename": filename,
                "event_id": event_id,
                "sha256": sha256,
            },
            source_revisions=source_revisions,
        )

    def prepare_machine_snapshot_publish(
        self,
        binding: object,
        registry_revision: int,
        *,
        instance_id: str,
        pid: int | None,
        visible_gpu_ids: list[int],
        reserved_gpu_ids: list[int],
        reservation_summaries: list[dict[str, Any]],
        heartbeat_interval_seconds: int | float,
        started_at: str,
        gpu_policy: Mapping[str, Any],
        stop_reason: str | None = None,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one bounded advisory snapshot for an exact enabled binding."""
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("machine snapshot publication requires an enabled binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("machine snapshot binding belongs to a different runtime.")
        binding_runtime_root = getattr(binding, "runtime_root", None)
        if not isinstance(binding_runtime_root, str) or Path(binding_runtime_root).expanduser().resolve() != self.root:
            raise ProjectIOProtocolError("machine snapshot binding has a different runtime root.")
        if getattr(binding, "registration_generation", None) is None:
            raise ValueError("machine snapshot publication requires a registration_generation.")
        snapshot_at = utc_now()
        return self._prepare_request(
            binding,
            registry_revision,
            "machine_snapshot_publish",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "instance_id": instance_id,
                "pid": pid,
                "visible_gpu_ids": visible_gpu_ids,
                "reserved_gpu_ids": reserved_gpu_ids,
                "reservation_summaries": reservation_summaries,
                "heartbeat_interval_seconds": heartbeat_interval_seconds,
                "started_at": started_at,
                "snapshot_at": snapshot_at,
                "gpu_policy": gpu_policy,
                "stop_reason": stop_reason,
            },
            source_revisions=source_revisions,
        )

    def prepare_registration_renew(
        self,
        binding: object,
        registry_revision: int,
        *,
        renewal_horizon_seconds: int | float,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one exact current-binding registration renewal request."""
        if getattr(binding, "enabled", None) is not True:
            raise ProjectIOProtocolError("registration renewal requires an enabled binding.")
        if getattr(binding, "runtime_instance_id", None) != self.runtime_id:
            raise ProjectIOProtocolError("registration renewal binding belongs to a different runtime.")
        binding_runtime_root = getattr(binding, "runtime_root", None)
        if not isinstance(binding_runtime_root, str) or Path(binding_runtime_root).expanduser().resolve() != self.root:
            raise ProjectIOProtocolError("registration renewal binding has a different runtime root.")
        if getattr(binding, "registration_generation", None) is None:
            raise ValueError("registration renewal requires a registration_generation.")
        return self._prepare_request(
            binding,
            registry_revision,
            "registration_renew",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "renewal_horizon_seconds": renewal_horizon_seconds,
            },
            source_revisions=source_revisions,
        )

    def prepare_upgrade_service(self, binding: object, registry_revision: int) -> ProjectIORequest:
        """Persist one journal-backed discovery and bounded upgrade slice."""
        return self._prepare_request(
            binding,
            registry_revision,
            "upgrade_service",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions=None,
        )

    def prepare_submission_control_service(
        self,
        binding: object,
        registry_revision: int,
        *,
        continuation: Mapping[str, Any],
    ) -> ProjectIORequest:
        """Persist one resumable, shared-only Submission proof repair slice."""
        return self._prepare_request(
            binding,
            registry_revision,
            "submission_control_service",
            {"machine_name": getattr(binding, "machine_name", None), "continuation": continuation},
            source_revisions=None,
        )

    def prepare_observation_service(self, binding: object, registry_revision: int) -> ProjectIORequest:
        """Persist one shared-only observation projection maintenance slice."""
        return self._prepare_request(
            binding,
            registry_revision,
            "observation_service",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions=None,
        )

    def prepare_notification_service(self, binding: object, registry_revision: int) -> ProjectIORequest:
        """Capture shared legacy configuration without transporting credentials."""
        return self._prepare_request(
            binding,
            registry_revision,
            "notification_service",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions=None,
        )

    def reset_ambiguous_notification_service_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="notification_service")

    def prepare_progress_projection(
        self,
        binding: object,
        registry_revision: int,
        *,
        context: Mapping[str, Any],
        projection: Mapping[str, Any] | None,
    ) -> ProjectIORequest:
        """Observe or publish one frozen producer context without a local worker write."""
        return self._prepare_request(
            binding,
            registry_revision,
            "progress_projection",
            {"machine_name": getattr(binding, "machine_name", None), "context": context, "projection": projection},
            source_revisions={},
        )

    def prepare_recovery_admission(self, binding: object, registry_revision: int) -> ProjectIORequest:
        """Prepare shared recovery ownership and root admission, not local capture."""
        return self._prepare_request(
            binding,
            registry_revision,
            "recovery_admission",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions={},
        )

    def resolve_stale_recovery_admission(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "recovery_admission":
            raise ValueError("stale recovery admission resolution requires recovery_admission")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_recovery_admission_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="recovery_admission")

    def prepare_recovery_capture_transition(
        self, binding: object, registry_revision: int, *, completion_digest: str, phase: str
    ) -> ProjectIORequest:
        return self._prepare_request(
            binding,
            registry_revision,
            "recovery_capture_transition",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "completion_digest": completion_digest,
                "phase": phase,
            },
            source_revisions={},
        )

    def resolve_stale_recovery_capture_transition(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "recovery_capture_transition":
            raise ValueError("stale capture transition resolution requires recovery_capture_transition")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_recovery_capture_transition_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(
            request_id, request, operation_kind="recovery_capture_transition"
        )

    def prepare_recovery_group_authority(
        self, binding: object, registry_revision: int, *, completion_digest: str
    ) -> ProjectIORequest:
        return self._prepare_request(
            binding,
            registry_revision,
            "recovery_group_authority",
            {"machine_name": getattr(binding, "machine_name", None), "completion_digest": completion_digest},
            source_revisions={},
        )

    def resolve_stale_recovery_group_authority(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "recovery_group_authority":
            raise ValueError("stale Group authority resolution requires recovery_group_authority")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_recovery_group_authority_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="recovery_group_authority")

    def prepare_recovery_source_release(
        self, binding: object, registry_revision: int, *, completion_digest: str
    ) -> ProjectIORequest:
        return self._prepare_request(
            binding,
            registry_revision,
            "recovery_source_release",
            {"machine_name": getattr(binding, "machine_name", None), "completion_digest": completion_digest},
            source_revisions={},
        )

    def resolve_stale_recovery_source_release(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "recovery_source_release":
            raise ValueError("stale source release resolution requires recovery_source_release")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_recovery_source_release_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="recovery_source_release")

    def prepare_recovery_source_hold(
        self, binding: object, registry_revision: int, *, capture_id: str
    ) -> ProjectIORequest:
        """Establish/replay one retained source hold without target worker writes."""
        return self._prepare_request(
            binding,
            registry_revision,
            "recovery_source_hold",
            {"machine_name": getattr(binding, "machine_name", None), "capture_id": capture_id},
            source_revisions={},
        )

    def resolve_stale_recovery_source_hold(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "recovery_source_hold":
            raise ValueError("stale source retention resolution requires recovery_source_hold")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_recovery_source_hold_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="recovery_source_hold")

    def prepare_legacy_capture_scan(
        self,
        binding: object,
        registry_revision: int,
        *,
        capture_id: str,
        backfill_id: str,
        backfill_revision: int,
        lane: str,
        cursor: Mapping[str, Any],
    ) -> ProjectIORequest:
        return self._prepare_request(
            binding,
            registry_revision,
            "legacy_capture_scan",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "capture_id": capture_id,
                "backfill_id": backfill_id,
                "backfill_revision": backfill_revision,
                "lane": lane,
                "cursor": dict(cursor),
            },
            source_revisions={},
        )

    def resolve_stale_legacy_capture_scan(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "legacy_capture_scan":
            raise ValueError("stale source discovery resolution requires legacy_capture_scan")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_legacy_capture_scan_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="legacy_capture_scan")

    def prepare_legacy_capture_read(
        self,
        binding: object,
        registry_revision: int,
        *,
        capture_id: str,
        backfill_id: str,
        backfill_revision: int,
        lane: str,
        relative: str,
    ) -> ProjectIORequest:
        """Read one exact retained source record without local worker mutations."""
        return self._prepare_request(
            binding,
            registry_revision,
            "legacy_capture_read",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "capture_id": capture_id,
                "backfill_id": backfill_id,
                "backfill_revision": backfill_revision,
                "lane": lane,
                "relative": relative,
            },
            source_revisions={},
        )

    def resolve_stale_legacy_capture_read(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "legacy_capture_read":
            raise ValueError("stale source read resolution requires legacy_capture_read")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_legacy_capture_read_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="legacy_capture_read")

    def reset_ambiguous_progress_projection_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="progress_projection")

    def resolve_stale_progress_projection(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "progress_projection":
            raise ValueError("stale progress resolution requires progress_projection")
        return self._resolve_stale_request(request_id, request)

    def resolve_stale_notification_service(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "notification_service":
            raise ValueError("stale notification resolution requires notification_service.")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_observation_service_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="observation_service")

    def resolve_stale_observation_service(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "observation_service":
            raise ValueError("stale observation repair resolution requires observation_service.")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_submission_control_service_for_retry(self, request_id: str, request: ProjectIORequest) -> bool:
        return self._reset_ambiguous_request_for_retry(request_id, request, operation_kind="submission_control_service")

    def resolve_stale_submission_control_service(self, request_id: str, request: ProjectIORequest) -> bool:
        if request.operation_kind != "submission_control_service":
            raise ValueError("stale Submission repair resolution requires submission_control_service.")
        return self._resolve_stale_request(request_id, request)

    def reset_ambiguous_upgrade_service_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Resume the durable upgrade journal only after exact worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="upgrade_service",
        )

    def resolve_stale_upgrade_service(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Discard stale scheduling evidence; the shared journal owns migration recovery."""
        if current_request.operation_kind != "upgrade_service":
            raise ValueError("stale resolution requires upgrade_service.")
        return self._resolve_stale_request(request_id, current_request)

    def prepare_activation_observe(
        self,
        binding: object,
        registry_revision: int,
        *,
        replay_epoch: str | None = None,
        replay_sequence: int = 0,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one bounded read-only activation checkpoint observation."""
        return self._prepare_request(
            binding,
            registry_revision,
            "activation_observe",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "replay_epoch": replay_epoch,
                "replay_sequence": replay_sequence,
            },
            source_revisions=source_revisions,
        )

    def prepare_activation_consumer_register(
        self,
        binding: object,
        registry_revision: int,
        *,
        process_fence: str,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one fenced activation-consumer registration request."""
        return self._prepare_request(
            binding,
            registry_revision,
            "activation_consumer_register",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "process_fence": process_fence,
            },
            source_revisions=source_revisions,
        )

    def prepare_activation_consumer_ack(
        self,
        binding: object,
        registry_revision: int,
        *,
        process_fence: str,
        epoch: str,
        sequence: int,
        reconstructed_floor: int | None = None,
        require_current: bool = False,
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one bounded activation acknowledgement request."""
        return self._prepare_request(
            binding,
            registry_revision,
            "activation_consumer_ack",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "process_fence": process_fence,
                "epoch": epoch,
                "sequence": sequence,
                "reconstructed_floor": reconstructed_floor,
                "require_current": require_current,
            },
            source_revisions=source_revisions,
        )

    def prepare_activation_consumer_retire(
        self,
        binding: object,
        registry_revision: int,
    ) -> ProjectIORequest:
        """Persist one exact removed-generation consumer retirement request."""
        return self._prepare_request(
            binding,
            registry_revision,
            "activation_consumer_retire",
            {"machine_name": getattr(binding, "machine_name", None)},
            source_revisions={},
        )

    def prepare_authority_service(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        process_identity: Mapping[str, object],
        source_revisions: Mapping[str, object] | None = None,
    ) -> ProjectIORequest:
        """Persist one bounded point-in-time observation that grants no authority."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_service",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "service_action": "observe_current_attempt",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
            },
            source_revisions=source_revisions,
        )

    def prepare_authority_renewal(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        process_identity: Mapping[str, object],
        source_revisions: Mapping[str, object],
    ) -> ProjectIORequest:
        """Persist one exact replayable bounded-lease renewal."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_renewal",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
            },
            source_revisions=source_revisions,
        )

    def prepare_authority_orphan_recovery(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        process_identity: Mapping[str, object],
        binding_signature: Sequence[str],
        source_revisions: Mapping[str, object] | None = None,
        replay_only: bool = False,
    ) -> ProjectIORequest:
        """Prepare one shared-only live orphan recovery CAS."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_orphan_recovery",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
                "binding_signature": list(binding_signature),
                "replay_only": replay_only,
            },
            source_revisions=source_revisions or {"task": None, "attempt_digest": None},
        )

    def prepare_authority_termination_commit(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        decision_id: str,
        decision_token: int,
        authority_outcome: str,
        reason: str,
        process_identity: Mapping[str, object],
        source_revisions: Mapping[str, object],
    ) -> ProjectIORequest:
        """Persist one exact replayable shared termination commitment."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_termination_commit",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "decision_id": decision_id,
                "decision_token": decision_token,
                "authority_outcome": authority_outcome,
                "reason": reason,
                "process_identity": dict(process_identity),
            },
            source_revisions=source_revisions,
        )

    def prepare_authority_terminal_observe(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        process_identity: Mapping[str, object],
        mode: str,
    ) -> ProjectIORequest:
        """Persist one read-only observation of exact terminal authority state."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_terminal_observe",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
                "mode": mode,
            },
            source_revisions={},
        )

    def prepare_authority_terminal_publish(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str | None,
        process_identity: Mapping[str, object],
        mode: str,
        phase: str,
        reason: str,
        exit_code: int | None,
        termination_result: str | None,
        source_revisions: Mapping[str, object],
        transition_digest: str,
    ) -> ProjectIORequest:
        """Persist one exact replayable terminal publication request."""
        digest = authority_terminal_transition_digest(
            mode=mode,
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            fencing_token=fencing_token,
            machine_name=getattr(binding, "machine_name", None),
            reservation_id=reservation_id,
            process_identity=process_identity,
            phase=phase,
            reason=reason,
            exit_code=exit_code,
            termination_result=termination_result,
        )
        if transition_digest != digest:
            raise ValueError("transition_digest differs from the canonical terminal transition.")
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_terminal_publish",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
                "mode": mode,
                "phase": phase,
                "reason": reason,
                "exit_code": exit_code,
                "termination_result": termination_result,
                "transition_digest": transition_digest,
            },
            source_revisions=source_revisions,
        )

    def prepare_authority_running_publish(
        self,
        binding: object,
        registry_revision: int,
        *,
        task_id: str,
        attempt_id: str,
        attempt_number: int,
        fencing_token: int,
        reservation_id: str,
        process_identity: Mapping[str, object],
        process_created_at: str,
    ) -> ProjectIORequest:
        """Persist one bounded request to publish the registered process as running."""
        return self._prepare_request(
            binding,
            registry_revision,
            "authority_running_publish",
            {
                "machine_name": getattr(binding, "machine_name", None),
                "task_id": task_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": reservation_id,
                "process_identity": dict(process_identity),
                "process_created_at": process_created_at,
            },
            source_revisions={},
        )

    def reset_ambiguous_authority_running_publish_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact idempotent running publication after worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="authority_running_publish",
        )

    def resolve_stale_authority_running_publish(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Archive an obsolete running publication after proving worker absence."""
        if current_request.operation_kind != "authority_running_publish" or current_request.request_id != request_id:
            raise ValueError("stale resolution requires authority_running_publish.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_authority_termination_commit_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact termination marker after proving worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="authority_termination_commit",
        )

    def resolve_stale_authority_termination_commit(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "authority_termination_commit" or current_request.request_id != request_id:
            raise ValueError("stale resolution requires authority_termination_commit.")
        result = self.load_result(request_id)
        if result is None or result.status in {"outcome_unknown", "retryable_error"}:
            self.start(request_id, reconcile_authority_termination_commit=True)
            return False
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_authority_terminal_observe(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "authority_terminal_observe":
            raise ValueError("stale resolution requires authority_terminal_observe.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_authority_terminal_publish_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact terminal transition after proving worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="authority_terminal_publish",
        )

    def resolve_stale_authority_terminal_publish(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "authority_terminal_publish" or current_request.request_id != request_id:
            raise ValueError("stale resolution requires authority_terminal_publish.")
        result = self.load_result(request_id)
        if result is None or result.status in {"outcome_unknown", "retryable_error"}:
            self.start(request_id, reconcile_authority_terminal_publish=True)
            return False
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_activation_observe(self, request_id: str, current_request: ProjectIORequest) -> bool:
        if current_request.operation_kind != "activation_observe":
            raise ValueError("stale resolution requires activation_observe.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_primary_probe(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Discard non-authorizing probe progress only after exact worker absence."""
        if current_request.operation_kind != "scheduler_primary_probe":
            raise ValueError("stale resolution requires scheduler_primary_probe.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_scheduler_quiescence_probe(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Discard non-authorizing scan progress only after exact worker absence."""
        if current_request.operation_kind != "scheduler_quiescence_probe":
            raise ValueError("stale resolution requires scheduler_quiescence_probe.")
        return self._resolve_stale_request(request_id, current_request)

    def _validate_project_event_source(self, project_id: str, bucket: str, filename: str, sha256: str) -> None:
        """Validate the exact source path and digest without following local links."""
        path = machine_project_paths(self.root, project_id)["events"] / bucket / filename
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts[:-1]:
            try:
                metadata = current.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise ProjectIOProtocolError("maintenance event source directory is missing.") from exc
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ProjectIOProtocolError("maintenance event source directory is not locally owned.")
            current = current / part
        try:
            parent_metadata = current.stat(follow_symlinks=False)
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ProjectIOProtocolError("maintenance event source is missing.") from exc
        if not stat.S_ISDIR(parent_metadata.st_mode) or parent_metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError("maintenance event source parent is not locally owned.")
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise ProjectIOProtocolError("maintenance event source is not a locally owned regular file.")

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened_metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or opened_metadata.st_uid != os.geteuid()
                or opened_metadata.st_nlink != 1
                or (opened_metadata.st_dev, opened_metadata.st_ino) != (metadata.st_dev, metadata.st_ino)
            ):
                raise ProjectIOProtocolError("maintenance event source changed during validation.")
            encoded = handle.read(PROJECT_IO_MAX_RECORD_BYTES + 1)
        if len(encoded) > PROJECT_IO_MAX_RECORD_BYTES:
            raise ProjectIOProtocolError("maintenance event source exceeds the local event size limit.")
        if hashlib.sha256(encoded).hexdigest() != sha256:
            raise ProjectIOProtocolError("maintenance event source digest changed before request preparation.")

    def retire_maintenance_event_source(self, request: ProjectIORequest) -> bool:
        """Delete only the exact digest-bound source of a consumed flush result."""
        if request.operation_kind != "maintenance_flush_event" or request.runtime_id != self.runtime_id:
            raise ValueError("maintenance event retirement requires a current flush request.")
        parameters = request.parameters
        events_root = machine_project_paths(self.root, request.project_id)["events"]
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            events_fd = os.open(events_root, directory_flags)
        except (FileNotFoundError, OSError):
            return False
        try:
            root_metadata = os.fstat(events_fd)
            if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != os.geteuid():
                return False
            try:
                bucket_fd = os.open(parameters["bucket"], directory_flags, dir_fd=events_fd)
            except (FileNotFoundError, OSError):
                return False
            try:
                bucket_metadata = os.fstat(bucket_fd)
                if not stat.S_ISDIR(bucket_metadata.st_mode) or bucket_metadata.st_uid != os.geteuid():
                    return False
                filename = parameters["filename"]
                quarantine = f".{filename}.{uuid.uuid4().hex}.retire"
                try:
                    os.replace(filename, quarantine, src_dir_fd=bucket_fd, dst_dir_fd=bucket_fd)
                except (FileNotFoundError, OSError):
                    return False
                is_exact = False
                try:
                    metadata = os.stat(quarantine, dir_fd=bucket_fd, follow_symlinks=False)
                    if stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid() and metadata.st_nlink == 1:
                        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                        descriptor = os.open(quarantine, flags, dir_fd=bucket_fd)
                        with os.fdopen(descriptor, "rb") as handle:
                            opened = os.fstat(handle.fileno())
                            encoded = handle.read(PROJECT_IO_MAX_RECORD_BYTES + 1)
                        is_exact = (
                            (opened.st_dev, opened.st_ino) == (metadata.st_dev, metadata.st_ino)
                            and len(encoded) <= PROJECT_IO_MAX_RECORD_BYTES
                            and hashlib.sha256(encoded).hexdigest() == parameters["sha256"]
                        )
                except (FileNotFoundError, OSError):
                    is_exact = False
                if is_exact:
                    os.unlink(quarantine, dir_fd=bucket_fd)
                    os.fsync(bucket_fd)
                    return True
                try:
                    os.link(
                        quarantine,
                        filename,
                        src_dir_fd=bucket_fd,
                        dst_dir_fd=bucket_fd,
                        follow_symlinks=False,
                    )
                except (FileExistsError, FileNotFoundError, OSError):
                    # A concurrent replacement remains canonical. Preserve the
                    # displaced source under its recovery name.
                    return False
                os.unlink(quarantine, dir_fd=bucket_fd)
                os.fsync(bucket_fd)
                return False
            finally:
                os.close(bucket_fd)
        finally:
            os.close(events_fd)

    def reset_ambiguous_maintenance_event_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make an exact idempotent event publication replayable after exit."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="maintenance_flush_event",
        )

    def reset_ambiguous_maintenance_descriptor_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact descriptor-advance request replayable after worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="maintenance_descriptor_advance",
        )

    def resolve_stale_maintenance_event(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive an obsolete event request only after proving worker absence."""
        if current_request.operation_kind != "maintenance_flush_event":
            raise ValueError("stale event resolution requires maintenance_flush_event.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_maintenance_descriptor(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive obsolete descriptor maintenance only after proving worker absence."""
        if current_request.operation_kind != "maintenance_descriptor_advance":
            raise ValueError("stale descriptor resolution requires maintenance_descriptor_advance.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_scheduler_due_offer(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive obsolete due-offer work only after proving worker absence."""
        if current_request.operation_kind != "scheduler_due_offer":
            raise ValueError("stale due-offer resolution requires scheduler_due_offer.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_scheduler_ready_index_build(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Archive obsolete ready-index work only after proving worker absence."""
        if current_request.operation_kind != "scheduler_ready_index_build":
            raise ValueError("stale ready-index resolution requires scheduler_ready_index_build.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_machine_snapshot_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the same fixed-time advisory replace after proving worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="machine_snapshot_publish",
        )

    def resolve_stale_machine_snapshot(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive an obsolete snapshot request only after proving worker absence."""
        if current_request.operation_kind != "machine_snapshot_publish":
            raise ValueError("stale snapshot resolution requires machine_snapshot_publish.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_registration_renew_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the same exact registration renewal after proving worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="registration_renew",
        )

    def resolve_stale_registration_renew(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive an obsolete registration renewal only after proving worker absence."""
        if current_request.operation_kind != "registration_renew":
            raise ValueError("stale resolution requires registration_renew.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_activation_consumer_register_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact idempotent consumer registration after worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="activation_consumer_register",
        )

    def resolve_stale_activation_consumer_register(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "activation_consumer_register":
            raise ValueError("stale resolution requires activation_consumer_register.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_activation_consumer_ack_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact idempotent acknowledgement after worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="activation_consumer_ack",
        )

    def resolve_stale_activation_consumer_ack(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "activation_consumer_ack":
            raise ValueError("stale resolution requires activation_consumer_ack.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_activation_consumer_retire_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="activation_consumer_retire",
        )

    def resolve_stale_activation_consumer_retire(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        if current_request.operation_kind != "activation_consumer_retire":
            raise ValueError("stale resolution requires activation_consumer_retire.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_authority_service(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive an obsolete read-only authority observation after worker absence."""
        if current_request.operation_kind != "authority_service":
            raise ValueError("stale resolution requires authority_service.")
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_authority_renewal_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay the exact idempotent renewal after proving worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="authority_renewal",
        )

    def resolve_stale_authority_renewal(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Replay ambiguous exact Task-first commits, then archive them as stale."""
        if current_request.operation_kind != "authority_renewal" or current_request.request_id != request_id:
            raise ValueError("stale resolution requires authority_renewal.")
        result = self.load_result(request_id)
        if result is None or result.status in {"outcome_unknown", "retryable_error"}:
            # The replay-only worker may complete only an already committed Task
            # marker. It cannot begin a lease renewal from a stale local manifest.
            # start() reconciles the exact process record under the executor lock
            # and will not create a concurrent worker while the old one may live.
            self.start(request_id, reconcile_authority_renewal=True)
            return False
        return self._resolve_stale_request(request_id, current_request)

    def reset_ambiguous_authority_orphan_recovery_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay an exact orphan recovery after worker absence."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="authority_orphan_recovery",
        )

    def resolve_stale_authority_orphan_recovery(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay an ambiguous orphan CAS before archiving its result."""
        if current_request.operation_kind != "authority_orphan_recovery" or current_request.request_id != request_id:
            raise ValueError("stale resolution requires authority_orphan_recovery.")
        result = self.load_result(request_id)
        if result is None or result.status in {"outcome_unknown", "retryable_error"}:
            self.start(request_id, reconcile_authority_orphan_recovery=True)
            return False
        return self._resolve_stale_request(request_id, current_request)

    def prepare_closed_request(
        self,
        binding: object,
        registry_revision: int,
        request_spec: GroupServiceRequestSpec,
    ) -> ProjectIORequest:
        """Persist one validated request specification from a closed service boundary."""
        if not isinstance(request_spec, GroupServiceRequestSpec):
            raise ValueError("closed request preparation requires a GroupServiceRequestSpec.")
        return self._prepare_request(
            binding,
            registry_revision,
            request_spec.operation_kind,
            request_spec.parameters,
            source_revisions=None,
        )

    def _prepare_request(
        self,
        binding: object,
        registry_revision: int,
        operation_kind: str,
        parameters: Mapping[str, Any],
        *,
        source_revisions: Mapping[str, object] | None,
        request_id: str | None = None,
        provisional_offer_id: str | None = None,
    ) -> ProjectIORequest:
        project_id = getattr(binding, "project_id", None)
        shared_root = getattr(binding, "shared_root", None)
        registration_generation = getattr(binding, "registration_generation", None)
        if registration_generation is None:
            raise ValueError(f"{operation_kind} requires a registration_generation.")
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        canonical_root = str(shared_root)
        request_id = request_id or uuid.uuid4().hex
        with exclusive(self.paths["project_io_lock"]):
            epoch = self._load_epoch()
            if not epoch.active:
                raise ProjectIOProtocolError("executor epoch is fenced.")
            self._reconcile_locked()
            requests, _processes, _results = self._scan_records()
            if len(requests) >= PROJECT_IO_CAPACITY:
                raise ProjectIOProtocolError("executor capacity is saturated.")
            identity = (self.runtime_id, project_id, registration_generation)
            if any(
                (request.runtime_id, request.project_id, request.registration_generation) == identity
                for request in requests.values()
            ):
                raise ProjectIOProtocolError("an unresolved request already exists for this exact binding.")
            request = ProjectIORequest(
                protocol_version=PROJECT_IO_PROTOCOL_VERSION,
                runtime_id=self.runtime_id,
                executor_epoch=epoch.executor_epoch,
                request_id=request_id,
                operation_kind=operation_kind,
                project_id=project_id,
                canonical_shared_root=canonical_root,
                registration_generation=registration_generation,
                registry_revision=registry_revision,
                source_revisions=source_revisions if source_revisions is not None else {},
                provisional_offer_id=provisional_offer_id,
                prepared_at=utc_now(),
                parameters=parameters,
            )
            if operation_kind == "scheduler_claim":
                offer = request.parameters["offer"]
                if (
                    offer["executor_epoch"] != epoch.executor_epoch
                    or offer["request_id"] != request.request_id
                    or offer["project_id"] != request.project_id
                    or offer["shared_root"] != request.canonical_shared_root
                    or offer["registration_generation"] != request.registration_generation
                    or offer["offer_id"] != request.provisional_offer_id
                ):
                    raise ProjectIOProtocolError("claim offer does not match its executor request identity.")
            path = self._record_path("requests", request_id)
            if self._validate_local_path(path, must_exist=False):
                raise ProjectIOProtocolError("random request ID collided with durable evidence.")
            self._write_record(path, request.to_dict(), "project_io_request")
            return request

    def start(
        self,
        request_id: str,
        *,
        reconcile_fenced_claim: bool = False,
        reconcile_authority_renewal: bool = False,
        reconcile_authority_orphan_recovery: bool = False,
        reconcile_authority_termination_commit: bool = False,
        reconcile_authority_terminal_publish: bool = False,
    ) -> ProjectIOProcess | None:
        """Start one fresh interpreter for a prepared typed request without waiting."""
        child: subprocess.Popen[bytes] | None = None
        with exclusive(self.paths["project_io_lock"]):
            epoch = self._load_epoch()
            if not epoch.active:
                raise ProjectIOProtocolError("executor epoch is fenced.")
            self._reconcile_locked()
            requests, processes, results = self._scan_records()
            request = requests.get(request_id)
            if request is None:
                raise ProjectIOProtocolError("prepared request does not exist.")
            is_fenced_claim_reconciliation = reconcile_fenced_claim and request.operation_kind == "scheduler_claim"
            is_authority_renewal_reconciliation = (
                reconcile_authority_renewal and request.operation_kind == "authority_renewal"
            )
            is_authority_orphan_reconciliation = (
                reconcile_authority_orphan_recovery and request.operation_kind == "authority_orphan_recovery"
            )
            is_authority_termination_reconciliation = (
                reconcile_authority_termination_commit and request.operation_kind == "authority_termination_commit"
            )
            is_authority_terminal_reconciliation = (
                reconcile_authority_terminal_publish and request.operation_kind == "authority_terminal_publish"
            )
            if reconcile_authority_renewal and not is_authority_renewal_reconciliation:
                raise ProjectIOProtocolError("authority renewal reconciliation requires authority_renewal.")
            if reconcile_authority_orphan_recovery and not is_authority_orphan_reconciliation:
                raise ProjectIOProtocolError(
                    "authority orphan recovery reconciliation requires authority_orphan_recovery."
                )
            if reconcile_authority_termination_commit and not is_authority_termination_reconciliation:
                raise ProjectIOProtocolError(
                    "authority termination reconciliation requires authority_termination_commit."
                )
            if reconcile_authority_terminal_publish and not is_authority_terminal_reconciliation:
                raise ProjectIOProtocolError("authority terminal reconciliation requires authority_terminal_publish.")
            if (
                request.executor_epoch != epoch.executor_epoch
                and not is_fenced_claim_reconciliation
                and not is_authority_renewal_reconciliation
                and not is_authority_orphan_reconciliation
                and not is_authority_termination_reconciliation
                and not is_authority_terminal_reconciliation
            ):
                raise ProjectIOProtocolError("prepared request belongs to a fenced executor epoch.")
            if request_id in processes:
                return processes.get(request_id)
            if request_id in results and not (
                (
                    is_authority_renewal_reconciliation
                    or is_authority_orphan_reconciliation
                    or is_authority_termination_reconciliation
                    or is_authority_terminal_reconciliation
                )
                and results[request_id].status in {"outcome_unknown", "retryable_error"}
            ):
                return None
            previous_child = self._children.get(request_id)
            if previous_child is not None:
                if previous_child.poll() is None:
                    return None
                self._children.pop(request_id, None)
            command = [
                sys.executable,
                "-m",
                "qqtools.plugins.qexp.agent.project_io_worker",
                "--runtime-root",
                str(self.root),
                "--request-id",
                request_id,
            ]
            if is_fenced_claim_reconciliation:
                command.append("--reconcile-fenced-claim")
            if is_authority_renewal_reconciliation:
                command.append("--reconcile-authority-renewal")
            if is_authority_orphan_reconciliation:
                command.append("--reconcile-authority-orphan-recovery")
            if is_authority_termination_reconciliation:
                command.append("--reconcile-authority-termination-commit")
            if is_authority_terminal_reconciliation:
                command.append("--reconcile-authority-terminal-publish")
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Retain the child before any later identity or record operation can
            # fail. A retry may then observe this exact process instead of
            # spawning an unbounded replacement with no ownership evidence.
            self._children[request_id] = child
            deadline = time.monotonic() + _START_IDENTITY_SECONDS
            ticks = _process_start_time_ticks(child.pid)
            while ticks is None and time.monotonic() < deadline:
                if child.poll() is not None:
                    self._children.pop(request_id, None)
                    return None
                time.sleep(0.01)
                ticks = _process_start_time_ticks(child.pid)
            if ticks is None:
                return None
            process = ProjectIOProcess(
                request=request,
                pid=child.pid,
                start_time_ticks=ticks,
                started_at=utc_now(),
                state="running",
            )
            path = self._record_path("processes", request_id)
            if self._validate_local_path(path, must_exist=False):
                self._children[request_id] = child
                raise ProjectIOProtocolError("process evidence already exists for this request.")
            self._write_record(path, process.to_dict(), "project_io_process")
            return process

    def poll(self) -> dict[str, Any]:
        """Reconcile current process identities without waiting for any worker."""
        try:
            with exclusive(self.paths["project_io_lock"], blocking=False) as acquired:
                if not acquired:
                    return self._unknown_status()
                self._reconcile_locked()
                self._reconciliation_unknown = False
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            self._reconciliation_unknown = True
            return self._unknown_status()
        return self.status_view()

    def unresolved_requests(self) -> tuple[ProjectIORequest, ...]:
        """Return exact unresolved requests without resolving or consuming them."""
        with exclusive(self.paths["project_io_lock"]):
            epoch = self._load_epoch()
            if not epoch.active:
                raise ProjectIOProtocolError("executor epoch is fenced.")
            self._reconcile_locked(replay_archives=False)
            current_epoch = self._load_epoch()
            if not current_epoch.active or current_epoch.executor_epoch != epoch.executor_epoch:
                raise ProjectIOProtocolError("executor epoch changed while reading unresolved requests.")
            requests, _processes, _results = self._scan_records()
            return tuple(sorted(requests.values(), key=lambda request: request.request_id))

    def _reconcile_locked(self, *, replay_archives: bool = True) -> None:
        # Reap children owned by this controller before consulting /proc. An
        # exited but unreaped child still has the same PID/start identity and
        # would otherwise be misclassified as a live worker indefinitely.
        for request_id, child in tuple(self._children.items()):
            if child.poll() is not None:
                self._children.pop(request_id, None)
        requests, processes, results = self._scan_records()
        remaining_process_ids = set(processes)
        for request_id, process in processes.items():
            state = inspect_process_identity(process.pid, process.start_time_ticks)
            result = results.get(request_id)
            if state == _PROCESS_UNVERIFIED:
                if process.state not in {"exit_unverified", "stop_targeted"}:
                    self._write_record(
                        self._record_path("processes", request_id),
                        replace(process, state="exit_unverified").to_dict(),
                        "project_io_process",
                    )
                continue
            if state == _PROCESS_LIVE:
                if process.state == "exit_unverified":
                    self._write_record(
                        self._record_path("processes", request_id),
                        replace(process, state="running").to_dict(),
                        "project_io_process",
                    )
                continue
            if result is None:
                request = requests[request_id]
                if request.operation_kind in _OUTCOME_UNKNOWN_ON_WORKER_EXIT:
                    result = ProjectIOResult(
                        request=request,
                        status="outcome_unknown",
                        reason_code="project_io_outcome_unknown",
                        completed_at=utc_now(),
                        evidence={},
                    )
                    self._write_record(self._record_path("results", request_id), result.to_dict(), "project_io_result")
                    # Keep the in-memory snapshot aligned with the just-written
                    # outcome. The archive pass below must see this synthesized
                    # ambiguous result (not None) or it could discard a
                    # Task-first authority-renewal commit during epoch restart.
                    results[request_id] = result
                elif request.operation_kind in _RETRYABLE_ON_WORKER_EXIT:
                    result = ProjectIOResult(
                        request=request,
                        status="retryable_error",
                        reason_code="project_io_worker_exited_without_result",
                        completed_at=utc_now(),
                        evidence={},
                    )
                    self._write_record(self._record_path("results", request_id), result.to_dict(), "project_io_result")
                else:  # pragma: no cover - guarded by the exhaustive partition above.
                    raise ProjectIOProtocolError("worker-exit recovery policy is missing.")
            _durable_unlink(self._record_path("processes", request_id))
            remaining_process_ids.discard(request_id)
            self._children.pop(request_id, None)
        if replay_archives:
            epoch = self._load_epoch()
            for request_id, request in requests.items():
                if request_id in remaining_process_ids:
                    continue
                archived = self._find_resolution(request_id, request)
                if archived is None:
                    if request.executor_epoch == epoch.executor_epoch:
                        continue
                    result = results.get(request_id)
                    if (
                        request.operation_kind == "authority_renewal"
                        and result is not None
                        and result.status in {"outcome_unknown", "retryable_error"}
                    ):
                        # Preserve the exact replay identity across executor restart.
                        # The replay-only worker will verify the Task marker and either
                        # finish that commit or return a stale result without writing.
                        continue
                    if (
                        request.operation_kind == "authority_orphan_recovery"
                        and result is not None
                        and result.status in {"outcome_unknown", "retryable_error"}
                    ):
                        continue
                    if (
                        request.operation_kind
                        in {
                            "authority_termination_commit",
                            "authority_orphan_recovery",
                            "authority_terminal_publish",
                        }
                        and result is not None
                        and result.status in {"outcome_unknown", "retryable_error"}
                    ):
                        # Preserve the exact intent for replay-only reconciliation.
                        continue
                    if request.operation_kind == "scheduler_claim" and classify_executor_offer(
                        self.runtime.root, _claim_offer_identity(request)
                    ) not in {"matching_active", "matching_released"}:
                        # A new epoch does not erase ambiguous capacity. Keep
                        # the old exact request until claim reconciliation has
                        # converged its executor-owned offer.
                        continue
                    self._commit_resolution(request_id, "stale", request, results.get(request_id))
                else:
                    resolution, _result = archived
                    if resolution == "consumed" and request.executor_epoch == epoch.executor_epoch and epoch.active:
                        # The exact consumer may not have observed the return
                        # from the archive commit yet. Preserve its replayable
                        # result for this epoch; a later epoch may reclaim it.
                        continue
                for lane in ("processes", "results", "requests"):
                    _durable_unlink(self._record_path(lane, request_id))
                self._children.pop(request_id, None)
        self._reconciliation_unknown = False

    def load_result(self, request_id: str) -> ProjectIOResult | None:
        try:
            result = self._load_result(request_id)
        except FileNotFoundError:
            return None
        request = self._load_request(request_id)
        if result.request != request:
            raise ProjectIOProtocolError("result record does not repeat its exact request identity.")
        return result

    def release_unprepared_offer(self, identity: ReservationIdentity) -> bool:
        """Release an exact local offer only when no transport was ever prepared.

        Request publication and process start both use this lock. Missing all
        three live records proves that this offer cannot have a worker; existing
        or malformed evidence is never treated as absence.
        """
        if not isinstance(identity, ReservationIdentity) or identity.executor_request_id is None:
            raise ValueError("unprepared offer release requires an executor-owned identity.")
        request_id = identity.executor_request_id
        with exclusive(self.paths["project_io_lock"]):
            self._load_epoch()
            if request_id in self._children:
                return False
            if any(
                self._validate_local_path(self._record_path(lane, request_id), must_exist=False)
                for lane in ("requests", "processes", "results")
            ):
                return False
            # The resource API validates the complete identity and also repairs
            # its exact released+provisional crash image. Do not pre-classify:
            # that recoverable pair is deliberately reported as a conflict to
            # read-only callers until the transition is replayed.
            release_executor_offer(self.runtime.root, identity, "claim_offer_not_prepared")
            return classify_executor_offer(self.runtime.root, identity) == "matching_released"

    def reset_ambiguous_claim_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact current-epoch claim replayable after process absence."""
        if (
            not isinstance(current_request, ProjectIORequest)
            or current_request.request_id != request_id
            or current_request.operation_kind != "scheduler_claim"
        ):
            raise ValueError("claim retry requires the exact scheduler_claim request.")
        with exclusive(self.paths["project_io_lock"]):
            epoch = self._load_epoch()
            if not epoch.active or self._load_request(request_id) != current_request:
                return False
            try:
                process = self._load_process(request_id)
            except FileNotFoundError:
                process = None
            if process is None:
                child = self._children.get(request_id)
                if child is not None:
                    if child.poll() is None:
                        return False
                    self._children.pop(request_id, None)
            if process is not None:
                state = inspect_process_identity(process.pid, process.start_time_ticks)
                if state != _PROCESS_ABSENT:
                    return False
                _durable_unlink(self._record_path("processes", request_id))
                self._children.pop(request_id, None)
            try:
                result = self._load_result(request_id)
            except FileNotFoundError:
                result = None
            if result is not None:
                if result.request != current_request or result.status != "outcome_unknown":
                    return False
                _durable_unlink(self._record_path("results", request_id))
            return True

    def reset_ambiguous_cursor_commit_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact current-epoch idempotent cursor CAS replayable."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="scheduler_cursor_commit",
        )

    def reset_ambiguous_replayable_request_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Replay one explicitly authorized request after exact worker absence."""
        if not isinstance(current_request, ProjectIORequest) or current_request.request_id != request_id:
            raise ValueError("retry requires the exact ProjectIORequest identity.")
        operation_kind = current_request.operation_kind
        if operation_kind not in _REPLAYABLE_REQUEST_OPERATION_KINDS:
            raise ValueError(f"operation {operation_kind!r} is not replayable through this entry.")
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind=operation_kind,
        )

    def reset_ambiguous_scheduler_due_offer_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact fenced due-offer advance replayable."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="scheduler_due_offer",
        )

    def reset_ambiguous_scheduler_ready_index_build_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact fenced ready-index slice replayable."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="scheduler_ready_index_build",
        )

    def reset_ambiguous_scheduler_launch_authorize_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Make one exact fenced launch authorization replayable."""
        return self._reset_ambiguous_request_for_retry(
            request_id,
            current_request,
            operation_kind="scheduler_launch_authorize",
        )

    def _reset_ambiguous_request_for_retry(
        self,
        request_id: str,
        current_request: ProjectIORequest,
        *,
        operation_kind: str,
    ) -> bool:
        if (
            not isinstance(current_request, ProjectIORequest)
            or current_request.request_id != request_id
            or current_request.operation_kind != operation_kind
        ):
            raise ValueError(f"retry requires the exact {operation_kind} request.")
        with exclusive(self.paths["project_io_lock"]):
            epoch = self._load_epoch()
            if (
                not epoch.active
                or current_request.executor_epoch != epoch.executor_epoch
                or self._load_request(request_id) != current_request
            ):
                return False
            try:
                process = self._load_process(request_id)
            except FileNotFoundError:
                process = None
            if process is None:
                child = self._children.get(request_id)
                if child is not None:
                    if child.poll() is None:
                        return False
                    self._children.pop(request_id, None)
            if process is not None:
                state = inspect_process_identity(process.pid, process.start_time_ticks)
                if state != _PROCESS_ABSENT:
                    return False
                _durable_unlink(self._record_path("processes", request_id))
                self._children.pop(request_id, None)
            try:
                result = self._load_result(request_id)
            except FileNotFoundError:
                result = None
            if result is not None:
                if result.request != current_request or result.status != "outcome_unknown":
                    return False
                _durable_unlink(self._record_path("results", request_id))
            return True

    def resolve_stale_cursor_commit(
        self,
        request_id: str,
        current_request: ProjectIORequest,
    ) -> bool:
        """Archive an obsolete cursor intent only after proving process absence."""
        if current_request.operation_kind != "scheduler_cursor_commit":
            raise ValueError("stale cursor resolution requires scheduler_cursor_commit.")
        return self._resolve_stale_request(request_id, current_request)

    def resolve_stale_request(self, request_id: str, current_request: ProjectIORequest) -> bool:
        """Archive an exact stale request after the private process-absence checks."""
        return self._resolve_stale_request(request_id, current_request)

    def _resolve_stale_request(self, request_id: str, current_request: ProjectIORequest) -> bool:
        if not isinstance(current_request, ProjectIORequest) or current_request.request_id != request_id:
            raise ValueError("stale resolution requires the exact request identity.")
        with exclusive(self.paths["project_io_lock"]):
            try:
                stored = self._load_request(request_id)
            except FileNotFoundError:
                return True
            if stored != current_request:
                return False
            try:
                process = self._load_process(request_id)
            except FileNotFoundError:
                process = None
            if process is None:
                child = self._children.get(request_id)
                if child is not None:
                    if child.poll() is None:
                        return False
                    self._children.pop(request_id, None)
            if process is not None:
                if inspect_process_identity(process.pid, process.start_time_ticks) != _PROCESS_ABSENT:
                    return False
                _durable_unlink(self._record_path("processes", request_id))
            try:
                result = self._load_result(request_id)
            except FileNotFoundError:
                result = None
            if result is not None and result.request != stored:
                return False
            self._commit_resolution(request_id, "stale", stored, result)
            for lane in ("processes", "results", "requests"):
                _durable_unlink(self._record_path(lane, request_id))
            self._children.pop(request_id, None)
            return True

    def consume(self, request_id: str, current_request: ProjectIORequest) -> ProjectIOResult | None:
        """Archive only an exact current result, then durably reclaim transient records."""
        if not isinstance(current_request, ProjectIORequest) or current_request.request_id != request_id:
            raise ValueError("consume requires the exact current typed request identity.")
        with exclusive(self.paths["project_io_lock"]):
            self._reconcile_locked(replay_archives=False)
            try:
                stored = self._load_request(request_id)
            except FileNotFoundError:
                # A restart/new epoch may already have archived and reclaimed
                # this stale identity. Absence cannot grant authority.
                return None
            result: ProjectIOResult | None
            try:
                result = self._load_result(request_id)
            except FileNotFoundError:
                result = None
            try:
                process = self._load_process(request_id)
            except FileNotFoundError:
                process = None
            if process is not None:
                # Publication precedes worker exit. Retain every linked record until
                # reconciliation has positively observed PID/start-time absence.
                return None

            archived = self._find_resolution(request_id, stored)
            if archived is not None:
                for lane in ("processes", "results", "requests"):
                    _durable_unlink(self._record_path(lane, request_id))
                self._children.pop(request_id, None)
                resolution, archived_result = archived
                epoch = self._load_epoch()
                return (
                    archived_result
                    if resolution == "consumed"
                    and stored == current_request
                    and stored.executor_epoch == epoch.executor_epoch
                    and epoch.active
                    else None
                )

            epoch = self._load_epoch()
            authority_current = (
                stored == current_request
                and stored.executor_epoch == epoch.executor_epoch
                and epoch.active
                and (result is None or result.request == stored)
            )
            if not authority_current:
                self._commit_resolution(request_id, "stale", stored, result)
                for lane in ("processes", "results", "requests"):
                    _durable_unlink(self._record_path(lane, request_id))
                self._children.pop(request_id, None)
                return None
            if result is None or result.status == "outcome_unknown":
                return None
            self._commit_resolution(request_id, "consumed", stored, result)
            for lane in ("processes", "results", "requests"):
                _durable_unlink(self._record_path(lane, request_id))
            self._children.pop(request_id, None)
            return result

    def _find_resolution(
        self,
        request_id: str,
        expected_request: ProjectIORequest,
    ) -> tuple[str, ProjectIOResult | None] | None:
        directory = self.paths["project_io_resolved"]
        try:
            metadata = directory.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ProjectIOProtocolError("resolved history path is not a locally owned real directory.")
        matches: list[Path] = []
        record_count = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                record_count += 1
                if record_count > PROJECT_IO_MAX_RESOLVED_RECORDS:
                    raise ProjectIOProtocolError("resolved history exceeds its bounded record limit.")
                match = _RESOLVED_NAME.fullmatch(entry.name)
                if match is None or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ProjectIOProtocolError("resolved history contains invalid local evidence.")
                if match.group(2) == request_id:
                    matches.append(directory / entry.name)
        found: tuple[str, ProjectIOResult | None] | None = None
        for path in matches:
            value = self._read_record(path, "project_io_resolution")
            if set(value) != {"project_io_resolution"} or not isinstance(value["project_io_resolution"], dict):
                raise ProjectIOProtocolError("resolved history record is malformed.")
            resolution = value["project_io_resolution"]
            expected_keys = {
                "sequence",
                "request_id",
                "resolution",
                "recorded_at",
                "request_sha256",
                "result",
            }
            if set(resolution) != expected_keys or resolution["request_id"] != request_id:
                raise ProjectIOProtocolError("resolved history identity is malformed.")
            resolution_kind = resolution["resolution"]
            if resolution_kind not in {"consumed", "stale"}:
                raise ProjectIOProtocolError("resolved history has an invalid resolution kind.")
            if resolution["request_sha256"] != _request_digest(expected_request):
                raise ProjectIOProtocolError("resolved history repeats a conflicting request identity.")
            request = expected_request
            result_value = resolution["result"]
            if result_value is None:
                result = None
            else:
                if not isinstance(result_value, dict) or set(result_value) != {
                    "status",
                    "reason_code",
                    "completed_at",
                    "evidence",
                }:
                    raise ProjectIOProtocolError("resolved history result is malformed.")
                result = ProjectIOResult(
                    request=request,
                    status=result_value["status"],
                    reason_code=result_value["reason_code"],
                    completed_at=result_value["completed_at"],
                    evidence=result_value["evidence"],
                )
            if result is not None and result.request != request:
                raise ProjectIOProtocolError("resolved history result does not repeat its exact request.")
            if resolution_kind == "consumed" and (result is None or result.status == "outcome_unknown"):
                raise ProjectIOProtocolError("consumed resolution does not contain its exact result.")
            candidate = (resolution_kind, result)
            if found is not None and found != candidate:
                raise ProjectIOProtocolError("resolved history contains conflicting resolutions.")
            found = candidate
        return found

    def _commit_resolution(
        self,
        request_id: str,
        resolution: str,
        request: ProjectIORequest,
        result: ProjectIOResult | None,
    ) -> None:
        epoch = self._load_epoch()
        sequence = epoch.completion_sequence + 1
        next_epoch = replace(epoch, completion_sequence=sequence)
        self._write_record(self.paths["project_io_epoch"], next_epoch.to_dict(), "project_io_epoch")
        result_payload = result.to_dict()["project_io_result"] if result is not None else None
        record = {
            "project_io_resolution": {
                "sequence": sequence,
                "request_id": request_id,
                "resolution": resolution,
                "recorded_at": utc_now(),
                "request_sha256": _request_digest(request),
                # The request is already stored above. Repeating it inside the
                # result would make a pair of individually valid 64 KiB records
                # impossible to archive within the same bounded record limit.
                "result": (
                    {
                        "status": result.status,
                        "reason_code": result.reason_code,
                        "completed_at": result.completed_at,
                        "evidence": result_payload["evidence"],
                    }
                    if result is not None
                    else None
                ),
            }
        }
        require_json_size(record, max_bytes=PROJECT_IO_MAX_RECORD_BYTES, record_type="project_io_resolution")
        path = self.paths["project_io_resolved"] / f"{sequence:020d}-{request_id}.json"
        atomic_result = atomic_replace(path, record)
        if atomic_result is None:
            raise OSError("durability could not be established for project_io_resolution.")
        self._evict_resolved_history()

    def _evict_resolved_history(self) -> None:
        directory = self.paths["project_io_resolved"]
        entries: list[tuple[int, Path, int]] = []
        total_bytes = 0
        with os.scandir(directory) as scanned:
            for entry in scanned:
                if len(entries) >= PROJECT_IO_MAX_RESOLVED_RECORDS + 1:
                    raise ProjectIOProtocolError("resolved history exceeds its bounded record limit.")
                match = _RESOLVED_NAME.fullmatch(entry.name)
                if match is None or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ProjectIOProtocolError("resolved history contains invalid local evidence.")
                info = entry.stat(follow_symlinks=False)
                if info.st_uid != os.geteuid() or info.st_nlink != 1:
                    raise ProjectIOProtocolError("resolved history contains foreign file ownership.")
                size = info.st_size
                if size <= 0 or size > PROJECT_IO_MAX_RECORD_BYTES:
                    raise ProjectIOProtocolError("resolved history record has an invalid encoded size.")
                entries.append((int(match.group(1)), directory / entry.name, size))
                total_bytes += size
        entries.sort(key=lambda item: item[0])
        while entries and (
            len(entries) > PROJECT_IO_MAX_RESOLVED_RECORDS or total_bytes > PROJECT_IO_MAX_RESOLVED_BYTES
        ):
            _sequence, path, size = entries.pop(0)
            _durable_unlink(path)
            total_bytes -= size

    def status_view(self) -> dict[str, Any]:
        """Return a read-only, bounded status projection without touching Project roots."""
        if self._reconciliation_unknown:
            return self._unknown_status()
        try:
            epoch = self._load_epoch()
            requests, processes, results = self._scan_records()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            return self._unknown_status()
        if epoch is None:
            return self._unknown_status()
        now = time.time()
        active_count = 0
        overdue_requests: list[ProjectIORequest] = []
        exit_unverified_count = 0
        unreaped_count = 0
        blocking_project_ids: list[str] = []
        unknown = False
        for request_id, request in requests.items():
            if request.project_id not in blocking_project_ids and len(blocking_project_ids) < PROJECT_IO_CAPACITY:
                blocking_project_ids.append(request.project_id)
            if request_id not in results and now - _timestamp_seconds(request.prepared_at) > PROJECT_IO_OVERDUE_SECONDS:
                overdue_requests.append(request)
            process = processes.get(request_id)
            if process is None:
                continue
            state = inspect_process_identity(process.pid, process.start_time_ticks)
            if state == _PROCESS_LIVE:
                active_count += 1
                if process.state == "stop_targeted":
                    unreaped_count += 1
            elif state == _PROCESS_UNVERIFIED:
                exit_unverified_count += 1
                unknown = True
                if process.state == "stop_targeted":
                    unreaped_count += 1
        overdue_count = len(overdue_requests)
        if epoch is None:
            envelope = "unknown"
            executor_epoch = None
        elif unknown:
            envelope = "unknown"
            executor_epoch = epoch.executor_epoch
        elif not epoch.active or requests:
            envelope = "degraded"
            executor_epoch = epoch.executor_epoch
        elif overdue_count > PROJECT_IO_SUPPORTED_HANG_LIMIT:
            envelope = "exceeded"
            executor_epoch = epoch.executor_epoch
        else:
            envelope = "healthy"
            executor_epoch = epoch.executor_epoch
        # A third overdue request exceeds the supported envelope even though it
        # remains bounded by the four fixed slots.
        if overdue_count > PROJECT_IO_SUPPORTED_HANG_LIMIT and envelope != "unknown":
            envelope = "exceeded"
        return {
            "protocol_version": PROJECT_IO_PROTOCOL_VERSION,
            "executor_epoch": executor_epoch,
            "capacity": PROJECT_IO_CAPACITY,
            "active_worker_count": active_count,
            "overdue_worker_count": overdue_count,
            "exit_unverified_worker_count": exit_unverified_count,
            "unreaped_worker_count": unreaped_count,
            "free_slot_count": max(0, PROJECT_IO_CAPACITY - len(requests)),
            "supported_hang_limit": PROJECT_IO_SUPPORTED_HANG_LIMIT,
            "envelope": envelope,
            "oldest_overdue_at": min((request.prepared_at for request in overdue_requests), default=None),
            "blocking_project_ids": blocking_project_ids[:PROJECT_IO_CAPACITY],
        }

    @staticmethod
    def _unknown_status() -> dict[str, Any]:
        return {
            "protocol_version": PROJECT_IO_PROTOCOL_VERSION,
            "executor_epoch": None,
            "capacity": PROJECT_IO_CAPACITY,
            "active_worker_count": 0,
            "overdue_worker_count": 0,
            "exit_unverified_worker_count": 0,
            "unreaped_worker_count": 0,
            "free_slot_count": 0,
            "supported_hang_limit": PROJECT_IO_SUPPORTED_HANG_LIMIT,
            "envelope": "unknown",
            "oldest_overdue_at": None,
            "blocking_project_ids": [],
        }

    def has_unfinished_work(self) -> bool:
        """Keep idle exit blocked unless the executor has a healthy empty epoch."""
        return self.status_view()["envelope"] != "healthy"

    def has_ready_result(self) -> bool:
        """Return whether a published result can be consumed without waiting on its worker."""
        try:
            requests, processes, results = self._scan_records()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            return False
        for request_id in requests.keys() & results.keys():
            child = self._children.get(request_id)
            if child is not None and child.poll() is not None:
                return True
            process = processes.get(request_id)
            if process is None or inspect_process_identity(process.pid, process.start_time_ticks) == _PROCESS_ABSENT:
                return True
        return False

    def fence_epoch(self) -> str:
        """Durably replace the active epoch before controller or worker stop."""
        self._ensure_layout()
        with exclusive(self.paths["project_io_lock"]):
            try:
                current = self._load_epoch()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
                current = None
            if current is not None:
                fenced = replace(current, active=False)
            else:
                fenced = self._next_epoch(active=False)
            self._write_record(self.paths["project_io_epoch"], fenced.to_dict(), "project_io_epoch")
            return fenced.executor_epoch

    def shutdown(self) -> dict[str, Any]:
        """Fence workers, allow bounded graceful exit, signal exact identities, and re-probe."""
        self.fence_epoch()
        status = self.poll()
        if (
            status["envelope"] != "unknown"
            and status["active_worker_count"] == 0
            and status["exit_unverified_worker_count"] == 0
            and status["unreaped_worker_count"] == 0
        ):
            return status
        self._grace_period(PROJECT_IO_STOP_GRACE_SECONDS)
        targets = self._mark_live_workers_stop_targeted()
        for process in targets:
            self._signal_exact_process(process, signal.SIGTERM)
        if targets:
            self._grace_period(PROJECT_IO_STOP_GRACE_SECONDS)
        targets = self._mark_live_workers_stop_targeted()
        for process in targets:
            self._signal_exact_process(process, signal.SIGKILL)
        self.poll()
        return self.status_view()

    def _grace_period(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _mark_live_workers_stop_targeted(self) -> list[ProjectIOProcess]:
        targets: list[ProjectIOProcess] = []
        try:
            with exclusive(self.paths["project_io_lock"]):
                _requests, processes, _results = self._scan_records()
                for request_id, process in processes.items():
                    state = inspect_process_identity(process.pid, process.start_time_ticks)
                    if state == _PROCESS_LIVE:
                        targeted = replace(process, state="stop_targeted")
                        self._write_record(
                            self._record_path("processes", request_id),
                            targeted.to_dict(),
                            "project_io_process",
                        )
                        targets.append(targeted)
                    elif state == _PROCESS_UNVERIFIED and process.state not in {
                        "exit_unverified",
                        "stop_targeted",
                    }:
                        self._write_record(
                            self._record_path("processes", request_id),
                            replace(process, state="exit_unverified").to_dict(),
                            "project_io_process",
                        )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ProjectIOProtocolError):
            return []
        return targets

    @staticmethod
    def _signal_exact_process(process: ProjectIOProcess, signum: int) -> bool:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            return False
        descriptor: int | None = None
        try:
            descriptor = os.pidfd_open(process.pid, 0)
            if inspect_process_identity(process.pid, process.start_time_ticks) != _PROCESS_LIVE:
                return False
            signal.pidfd_send_signal(descriptor, signum)
            return True
        except (OSError, ProcessLookupError, PermissionError, ValueError):
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _timestamp_seconds(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value).timestamp()


def _claim_offer_identity(request: ProjectIORequest) -> ReservationIdentity:
    offer = request.parameters["offer"]
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
                "executor_epoch": request.executor_epoch,
                "request_id": request.request_id,
                "registration_generation": request.registration_generation,
            },
        }
    )
