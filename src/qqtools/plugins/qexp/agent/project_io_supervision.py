"""Local Attempt authority coordination over isolated shared operations."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import stat
import struct
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..runtime.authority_scan import EvidenceScan, is_path_present, validate_evidence_path
from ..runtime.local_exit_reconciliation import LocalExitReconciler
from ..runtime.paths import local_paths, machine_project_paths
from ..runtime.process_evidence import inspect_local_group_identity, inspect_wrapper_identity
from ..runtime.records import utc_now, validate_identifier
from ..runtime.resources.cpu_lane import retag_cpu_if_matches
from ..runtime.resources.reservations import ReservationIdentity, retag_if_matches
from ..runtime.responsibility_cleanup import evidence_write_guard
from ..runtime.store import atomic_replace, read_json_limited
from ..runtime.termination import (
    RootConfig,
    advance_signals,
    attempt_control_lock,
    commit_signal,
    create_decision,
    decision_path,
    update_decision,
)
from ..runtime.work_budget import diagnostic_increment
from .context import MachineRuntime, ProjectBinding
from .process_reservation_recovery import ProcessReservationRecovery, registration_reservation
from .project_io_controller import ProjectIOController
from .project_io_process import PROCESS_IDENTITY_FIELDS
from .project_io_process import binding_prefix as _authority_due_binding_prefix
from .project_io_process import binding_signature as _authority_binding_signature
from .project_io_process import read_process_manifest as _read_process_manifest
from .project_io_protocol import PROJECT_IO_CAPACITY, ProjectIORequest, authority_terminal_transition_digest
from .project_io_terminal_proof import (
    TerminalObservationProof,
    TerminalPublicationProof,
    build_terminal_observation_proof,
    build_terminal_publication_proof,
    matches_terminal_lifecycle_event,
    matches_terminal_observation_identity,
    terminal_proof_matches_candidate,
)
from .project_io_terminal_proof import build_terminal_transition as _construct_terminal_transition
from .project_io_terminal_proof import has_exact_terminal_revisions as _has_exact_terminal_revisions
from .project_io_terminal_proof import terminal_transition_target as _select_terminal_transition_target
from .project_io_termination_convergence import (
    TerminationConvergence,
    TerminationConvergenceEvidence,
    classify_termination_convergence,
)
from .working_set import BindingTurn

_MAX_AUTHORITY_DUE_KEYS = 256
_MAX_AUTHORITY_SCAN_STATES = 256
_AUTHORITY_DUE_INFLIGHT_RESERVE = PROJECT_IO_CAPACITY

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_ONLYDIR = 0x01000000
_INOTIFY_CENSUS_MASK = (
    _IN_CLOSE_WRITE | _IN_MOVED_FROM | _IN_MOVED_TO | _IN_CREATE | _IN_DELETE | _IN_DELETE_SELF | _IN_MOVE_SELF
)
_INOTIFY_EVENT = struct.Struct("iIII")


class _DirectoryMutationWatch:
    """Fail-closed Linux mutation witness for one bounded local census."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._failed = False
        self._dirty = False
        self._filters: dict[int, set[str | None]] = {}
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            init = libc.inotify_init1
            init.argtypes = [ctypes.c_int]
            init.restype = ctypes.c_int
            add = libc.inotify_add_watch
            add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            add.restype = ctypes.c_int
            fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
            if fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        except (AttributeError, OSError):
            self._failed = True
            return
        self._fd = fd
        self._add_watch = add

    def watch(self, path: Path) -> None:
        if self._failed or self._fd is None:
            return
        target = path
        selected_name: str | None = None
        try:
            metadata = target.stat(follow_symlinks=False)
        except FileNotFoundError:
            selected_name = target.name
            target = target.parent
        else:
            if not stat.S_ISDIR(metadata.st_mode):
                self._failed = True
                return
        watch = self._add_watch(
            self._fd,
            os.fsencode(target),
            _INOTIFY_CENSUS_MASK | _IN_ONLYDIR,
        )
        if watch < 0:
            self._failed = True
            return
        self._filters.setdefault(watch, set()).add(selected_name)

    def unchanged(self) -> bool:
        if self._failed or self._fd is None or self._dirty:
            return False
        while True:
            try:
                event = os.read(self._fd, 4096)
            except BlockingIOError:
                break
            except OSError:
                self._failed = True
                return False
            if not event:
                self._failed = True
                return False
            offset = 0
            while offset < len(event):
                if len(event) - offset < _INOTIFY_EVENT.size:
                    self._failed = True
                    return False
                watch, _mask, _cookie, name_length = _INOTIFY_EVENT.unpack_from(event, offset)
                offset += _INOTIFY_EVENT.size
                end = offset + name_length
                if end > len(event):
                    self._failed = True
                    return False
                name = os.fsdecode(event[offset:end].split(b"\0", 1)[0])
                offset = end
                filters = self._filters.get(watch)
                if watch == -1 or filters is None or None in filters or name in filters:
                    self._dirty = True
        return not self._dirty

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _discard_authority_due_for_manifest(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    path: Path,
    *,
    keep_key: tuple[str, ...] | None = None,
) -> None:
    """Forget cadence for one local attempt after it disappears or changes identity."""
    prefix = _authority_due_binding_prefix(binding)
    for key in tuple(runtime.authority_process_next_due):
        if key[:6] == prefix and len(key) > 7 and key[7] == path.stem and key != keep_key:
            runtime.authority_process_next_due.pop(key, None)


def _read_isolated_authority_manifest(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    path: Path,
    binding_signature: tuple[str, ...],
) -> dict[str, Any] | None:
    """Read one exact local process manifest; never follow it into Project truth."""
    entry = _read_process_manifest(runtime, binding, path, binding_signature)
    if entry is None:
        return None
    paths = machine_project_paths(runtime.root, binding.project_id)
    try:
        observation_path = paths["observations"] / f"{entry['parameters']['attempt_id']}.json"
        if is_path_present(observation_path):
            if not validate_evidence_path(observation_path, paths["root"]):
                return None
            return None
        wrapper = inspect_wrapper_identity(entry["process"])
        group = inspect_local_group_identity(entry["process"])
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
    if wrapper.state != "alive" and group.state != "alive":
        return None
    return entry


def _authority_evidence_matches_intent(evidence: Mapping[str, Any], intent: Mapping[str, Any]) -> bool:
    parameters = intent["parameters"]
    return all(
        evidence.get(field) == parameters[field]
        for field in (
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
        )
    )


def _is_current_authority_observation(evidence: Mapping[str, Any], intent: Mapping[str, Any]) -> bool:
    revisions = evidence.get("source_revisions")
    if (
        evidence.get("outcome") != "observed_current"
        or evidence.get("attempt_phase") != "running"
        or evidence.get("authority_granted") is not False
        or evidence.get("local_effects") not in ([], ())
        or not _authority_evidence_matches_intent(evidence, intent)
        or not isinstance(revisions, Mapping)
        or set(revisions) != {"task", "attempt_digest"}
        or type(revisions.get("task")) is not int
    ):
        return False
    digest = revisions.get("attempt_digest")
    return isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _publish_local_authority_state(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    intent: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Update only a still-matching local manifest after an authority result."""
    outcome = evidence.get("outcome")
    if outcome == "renewed":
        authority_state = "healthy"
        expires_at = evidence.get("lease_expires_at")
        if not isinstance(expires_at, str) or not expires_at or len(expires_at) > 40:
            authority_state = "suspect"
    elif outcome == "not_required":
        authority_state = "local_safe"
        expires_at = None
    elif outcome in {"termination_requested", "observed_stale"}:
        authority_state = "isolated"
        expires_at = None
    elif outcome == "unavailable":
        authority_state = "suspect"
        expires_at = None
    else:
        return
    path = intent["path"]
    try:
        current = _read_isolated_authority_manifest(runtime, binding, path, intent["binding_signature"])
        if current is None or current["intent_signature"] != intent["intent_signature"]:
            return
        process = dict(current["process"])
        process["authority_state"] = authority_state
        if outcome == "renewed" and authority_state == "healthy":
            process["lease_expires_at"] = expires_at
        atomic_replace(path, {"process": process})
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        diagnostic_increment("scheduler.isolated.authority_manifest_publish_failed")


def advance_authority_renewals(
    runtime: MachineRuntime,
    controller: ProjectIOController,
    bindings: Sequence[ProjectBinding],
    registry_revision: int,
) -> None:
    """Observe and renew live local manifests before this cycle's scheduler admission."""
    if type(registry_revision) is not int or registry_revision < 0:
        return
    eligible = list(bindings)
    current_signatures = {
        binding.project_id: _authority_binding_signature(binding, registry_revision) for binding in eligible
    }
    binding_by_project = {binding.project_id: binding for binding in eligible}
    intents: dict[str, dict[str, Any]] = runtime.authority_process_intents
    observations: dict[str, dict[str, Any]] = runtime.authority_process_observations
    for project_id in tuple(intents):
        if current_signatures.get(project_id) != intents[project_id]["binding_signature"]:
            intents.pop(project_id, None)
            observations.pop(project_id, None)

    scan_keys = set(current_signatures.values())
    for key in tuple(runtime.authority_process_scans):
        if key not in scan_keys:
            runtime.authority_process_scans.pop(key).close()
    live_due_prefixes = {_authority_due_binding_prefix(binding) for binding in eligible}
    now = time.monotonic()
    for key, due_at in tuple(runtime.authority_process_next_due.items()):
        # Cadence entries expire when due; this bounds stale-attempt retention by
        # the maximum renewal horizon even if a manifest is deleted without a scan.
        if key[:6] not in live_due_prefixes or due_at <= now:
            runtime.authority_process_next_due.pop(key, None)

    try:
        unresolved = controller.executor.unresolved_requests()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        diagnostic_increment("scheduler.isolated.authority_unresolved_read_failed")
        return
    pinned: list[ProjectBinding] = []
    pinned_ids: set[str] = set()
    for request in unresolved:
        if request.operation_kind not in {"authority_service", "authority_renewal"}:
            continue
        binding = binding_by_project.get(request.project_id)
        signature = current_signatures.get(request.project_id)
        if (
            binding is None
            or signature is None
            or request.registry_revision != registry_revision
            or request.canonical_shared_root != str(binding.shared_root)
            or request.registration_generation != binding.registration_generation
        ):
            continue
        path = machine_project_paths(runtime.root, binding.project_id)["processes"] / (
            f"{request.parameters['attempt_id']}.json"
        )
        entry = _read_isolated_authority_manifest(runtime, binding, path, signature)
        if entry is None or any(
            entry["parameters"].get(field) != request.parameters.get(field)
            for field in (
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_identity",
            )
        ):
            continue
        intents[binding.project_id] = entry
        if request.operation_kind == "authority_renewal":
            observations[binding.project_id] = {
                "binding_signature": signature,
                "intent_signature": entry["intent_signature"],
                "executor_epoch": request.executor_epoch,
                "registry_revision": registry_revision,
                "evidence": {
                    "outcome": "observed_current",
                    "reason": None,
                    "runtime_id": request.runtime_id,
                    "executor_epoch": request.executor_epoch,
                    "project_id": request.project_id,
                    "canonical_shared_root": request.canonical_shared_root,
                    "registration_generation": request.registration_generation,
                    "registry_revision": request.registry_revision,
                    "machine_name": request.parameters["machine_name"],
                    "service_action": "observe_current_attempt",
                    **entry["parameters"],
                    "source_revisions": dict(request.source_revisions),
                    "attempt_phase": "running",
                    "authority_granted": False,
                    "local_effects": [],
                },
            }
        if binding.project_id not in pinned_ids:
            pinned.append(binding)
            pinned_ids.add(binding.project_id)

    if eligible:
        start = runtime.authority_process_offset % len(eligible)
        rotated = eligible[start:] + eligible[:start]
        selected = pinned + [binding for binding in rotated if binding.project_id not in pinned_ids][: 64 - len(pinned)]
        runtime.authority_process_offset = (start + max(1, min(64, len(eligible)))) % len(eligible)
    else:
        selected = []
        runtime.authority_process_offset = 0

    for binding in selected:
        project_id = binding.project_id
        signature = current_signatures[project_id]
        process_paths = machine_project_paths(runtime.root, project_id)
        selected_entry: dict[str, Any] | None = None
        cached = intents.get(project_id)
        if cached is not None:
            try:
                selected_entry = _read_isolated_authority_manifest(
                    runtime,
                    binding,
                    cached["path"],
                    signature,
                )
            except (OSError, RuntimeError, ValueError, TypeError, KeyError):
                diagnostic_increment("scheduler.isolated.authority_manifest_read_failed")
            if selected_entry is None or selected_entry["intent_signature"] != cached["intent_signature"]:
                _discard_authority_due_for_manifest(runtime, binding, cached["path"])
                intents.pop(project_id, None)
                observations.pop(project_id, None)
                selected_entry = None
        if selected_entry is None:
            scan = runtime.authority_process_scans.get(signature)
            if scan is None:
                if len(runtime.authority_process_scans) >= _MAX_AUTHORITY_SCAN_STATES:
                    diagnostic_increment("scheduler.isolated.authority_scan_capacity_exhausted")
                    continue
                scan = EvidenceScan(process_paths["processes"])
                runtime.authority_process_scans[signature] = scan
            try:
                page = scan.take(1)
                if page.paths:
                    selected_entry = _read_isolated_authority_manifest(
                        runtime,
                        binding,
                        page.paths[0],
                        signature,
                    )
                    if selected_entry is None:
                        _discard_authority_due_for_manifest(runtime, binding, page.paths[0])
            except (OSError, RuntimeError, ValueError, TypeError, KeyError):
                runtime.authority_process_scans.pop(signature, scan).close()
                diagnostic_increment("scheduler.isolated.authority_manifest_scan_failed")
            else:
                if page.is_complete:
                    runtime.authority_process_scans.pop(signature, scan).close()
        if selected_entry is not None:
            if runtime.authority_process_next_due.get(selected_entry["due_key"], 0.0) > time.monotonic():
                continue
            if (
                selected_entry["due_key"] not in runtime.authority_process_next_due
                and len(runtime.authority_process_next_due) >= _MAX_AUTHORITY_DUE_KEYS - _AUTHORITY_DUE_INFLIGHT_RESERVE
                and project_id not in pinned_ids
            ):
                # Preserve space for every already-submitted executor request's
                # completion cadence; never evict a live identity and renew it early.
                intents.pop(project_id, None)
                observations.pop(project_id, None)
                continue
            _discard_authority_due_for_manifest(
                runtime,
                binding,
                selected_entry["path"],
                keep_key=selected_entry["due_key"],
            )
            intents[project_id] = selected_entry

    selected_intents = {
        binding.project_id: intents[binding.project_id] for binding in selected if binding.project_id in intents
    }
    service_bindings = [
        selected_intents[binding.project_id]["binding"]
        for binding in selected
        if binding.project_id in selected_intents
    ]
    attempts = {project_id: intent["parameters"] for project_id, intent in selected_intents.items()}
    try:
        observed = controller.advance_authority_services(service_bindings, registry_revision, attempts)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        observed = {}
        diagnostic_increment("scheduler.isolated.authority_observation_failed")
    executor_epoch = getattr(controller, "_observed_executor_epoch", None)
    runtime_id = getattr(controller.executor, "runtime_id", None)
    for project_id, evidence in observed.items():
        intent = selected_intents.get(project_id)
        if intent is None or not isinstance(evidence, Mapping):
            continue
        if _is_current_authority_observation(evidence, intent):
            if (
                evidence.get("runtime_id") == runtime_id
                and evidence.get("executor_epoch") == executor_epoch
                and evidence.get("project_id") == project_id
                and evidence.get("canonical_shared_root") == str(intent["binding"].shared_root)
                and evidence.get("registration_generation") == intent["binding"].registration_generation
                and evidence.get("registry_revision") == registry_revision
                and executor_epoch is not None
                and runtime_id is not None
            ):
                observations[project_id] = {
                    "binding_signature": intent["binding_signature"],
                    "intent_signature": intent["intent_signature"],
                    "executor_epoch": executor_epoch,
                    "registry_revision": registry_revision,
                    "evidence": dict(evidence),
                }
        elif evidence.get("outcome") in {"observed_stale", "unavailable"}:
            _publish_local_authority_state(runtime, intent["binding"], intent, evidence)
            runtime.authority_process_next_due[intent["due_key"]] = time.monotonic() + 1.0
            observations.pop(project_id, None)
            intents.pop(project_id, None)

    current_observations: dict[str, Mapping[str, Any]] = {}
    for project_id, intent in selected_intents.items():
        cached = observations.get(project_id)
        if (
            cached is None
            or cached.get("binding_signature") != intent["binding_signature"]
            or cached.get("intent_signature") != intent["intent_signature"]
            or cached.get("executor_epoch") != executor_epoch
            or cached.get("registry_revision") != registry_revision
            or not _is_current_authority_observation(cached.get("evidence", {}), intent)
        ):
            continue
        current_observations[project_id] = cached["evidence"]
    current_bindings = [selected_intents[project_id]["binding"] for project_id in current_observations]
    try:
        renewed = controller.advance_authority_renewals(
            current_bindings,
            registry_revision,
            current_observations,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        diagnostic_increment("scheduler.isolated.authority_renewal_failed")
        return
    for project_id, evidence in renewed.items():
        intent = selected_intents.get(project_id)
        if intent is None or not isinstance(evidence, Mapping):
            continue
        outcome = evidence.get("outcome")
        _publish_local_authority_state(runtime, intent["binding"], intent, evidence)
        if outcome in {"renewed", "not_required", "termination_requested", "observed_stale", "unavailable"}:
            if (
                intent["due_key"] not in runtime.authority_process_next_due
                and len(runtime.authority_process_next_due) >= _MAX_AUTHORITY_DUE_KEYS
            ):
                diagnostic_increment("scheduler.isolated.authority_due_capacity_exhausted")
                continue
            if outcome == "renewed":
                renew_after = evidence.get("renew_after_seconds")
                if type(renew_after) in {int, float} and math.isfinite(renew_after) and 0 < renew_after <= 86_400:
                    runtime.authority_process_next_due[intent["due_key"]] = time.monotonic() + renew_after
                else:
                    runtime.authority_process_next_due[intent["due_key"]] = time.monotonic() + 1.0
            elif outcome == "not_required":
                runtime.authority_process_next_due[intent["due_key"]] = time.monotonic() + 5.0
            else:
                runtime.authority_process_next_due[intent["due_key"]] = time.monotonic() + 1.0
            observations.pop(project_id, None)
            intents.pop(project_id, None)


_RUNNING_PUBLICATION_SOURCE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError)
_MAX_PENDING_RUNNING_PUBLICATIONS = 256
_MAX_RUNNING_PUBLICATION_BINDINGS = 64
_MAX_CACHED_TERMINAL_CANDIDATES = 256
_MAX_TERMINAL_BINDINGS = 64
_MAX_TERMINAL_STATES = 256
_MAX_TERMINATION_BINDINGS = 64
_MAX_TERMINATION_STATES = 256
_MAX_TERMINATION_CANDIDATES = 256
_MAX_INITIAL_RECONCILIATION_BINDINGS = 256
_INITIAL_RECONCILIATION_BINDINGS_PER_TURN = 64
_INITIAL_RECONCILIATION_ENTRIES_PER_LANE = 64
_TERMINATION_GRACE_SECONDS = 5.0
_TERMINATION_TRIGGERS = frozenset({"cancellation_requested", "holder_safe_deadline_elapsed"})
_TERMINATION_SOURCE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError)
_SUPERVISION_LANE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError)
_SUPERVISION_OPERATION_KINDS = frozenset(
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


@dataclass(frozen=True, slots=True)
class _RunningPublicationSource:
    """One validated local registration or launch intent."""

    record: dict[str, Any]
    parameters: dict[str, Any]
    identity: tuple[Any, ...]


@dataclass(slots=True)
class _RunningPublicationScans:
    """Persistent discovery state for one binding signature."""

    registrations: EvidenceScan
    launch_intents: EvidenceScan
    next_lane: str = "registration"
    registrations_complete: bool = False
    launch_intents_complete: bool = False

    def close(self) -> None:
        self.registrations.close()
        self.launch_intents.close()

    @property
    def complete(self) -> bool:
        """Whether both source lanes have reached EOF for this sweep."""
        return self.registrations_complete and self.launch_intents_complete


@dataclass(frozen=True, slots=True)
class _TerminalProcessRecord:
    """One validated local process manifest and its full canonical identity."""

    record: dict[str, Any]
    identity: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TerminalExitObservation:
    """One validated immutable exit observation."""

    record: dict[str, Any]
    identity: str
    exit_code: int


@dataclass(slots=True)
class _TerminalCandidate:
    """Cached exact local evidence retained across asynchronous Project I/O turns."""

    binding: ProjectBinding
    binding_signature: tuple[str, ...]
    process_path: Path
    observation_path: Path
    parameters: dict[str, Any]
    exit_code: int
    process_identity: str
    observation_identity: str
    pending_transition: dict[str, Any] | None = None
    post_update_process: dict[str, Any] | None = None
    post_update_process_identity: str | None = None


@dataclass(slots=True)
class _TerminalBindingState:
    """One bounded process scan paired with its local capacity reconciler."""

    process_scan: EvidenceScan
    reconciler: LocalExitReconciler
    process_scan_complete: bool = False

    def close(self) -> None:
        self.process_scan.close()
        self.reconciler.close()


@dataclass(slots=True)
class _TerminationBindingState:
    """Persistent bounded process discovery for live termination candidates."""

    process_scan: EvidenceScan
    reconciler: LocalExitReconciler
    process_scan_complete: bool = False

    def close(self) -> None:
        self.process_scan.close()
        self.reconciler.close()


@dataclass(slots=True)
class _TerminationCandidate:
    """One exact local process retained across bounded termination turns."""

    binding: ProjectBinding
    binding_signature: tuple[str, ...]
    process_path: Path
    parameters: dict[str, Any]
    process_identity: str
    trigger: str | None = None
    decision_id: str | None = None
    source_revisions: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    pending_commit: dict[str, Any] | None = None
    commit_evidence: dict[str, Any] | None = None
    pending_publication: dict[str, Any] | None = None
    sigterm_deadline: float | None = None
    post_update_process: dict[str, Any] | None = None
    post_update_process_identity: str | None = None
    convergence: TerminationConvergence | None = None
    is_retained_replay: bool = False


def _canonical_terminal_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _read_terminal_process_record(
    path: Path,
    project_root: Path,
    binding: ProjectBinding,
) -> _TerminalProcessRecord | None:
    """Read one exact local process manifest for natural-exit supervision."""
    if not validate_evidence_path(path, project_root):
        return None
    envelope = read_json_limited(path, max_bytes=65_536, record_type="process_manifest")
    if set(envelope) != {"process"}:
        return None
    record = envelope.get("process")
    if (
        not isinstance(record, dict)
        or type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != 1
    ):
        return None

    task_id = record.get("task_id")
    attempt_id = record.get("attempt_id")
    if not isinstance(task_id, str) or not isinstance(attempt_id, str) or path.stem != attempt_id:
        return None
    try:
        validate_identifier(task_id, "process task_id")
        validate_identifier(attempt_id, "process attempt_id")
    except ValueError:
        return None
    prefix = f"{task_id}-attempt-"
    if not attempt_id.startswith(prefix):
        return None
    suffix = attempt_id[len(prefix) :]
    if not suffix.isascii() or not suffix.isdigit():
        return None
    attempt_number = int(suffix)
    if attempt_number < 1 or attempt_id != f"{task_id}-attempt-{attempt_number}":
        return None

    fencing_token = record.get("fencing_token")
    reservation_id = record.get("reservation_id")
    if (
        record.get("machine_name") != binding.machine_name
        or type(fencing_token) is not int
        or fencing_token < 1
        or record.get("observed_state") not in {"running", "exited"}
    ):
        return None
    if reservation_id is not None:
        try:
            validate_identifier(reservation_id, "process reservation_id")
        except ValueError:
            return None

    process_identity: dict[str, int | None] = {}
    for field in PROCESS_IDENTITY_FIELDS:
        if field not in record:
            return None
        value = record[field]
        positive = field in {"wrapper_pid", "process_group_id"}
        if value is not None and (type(value) is not int or value < (1 if positive else 0)):
            return None
        process_identity[field] = value

    parameters = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "reservation_id": reservation_id,
        "process_identity": process_identity,
    }
    return _TerminalProcessRecord(
        record=record,
        identity=_canonical_terminal_record(record),
        parameters=parameters,
    )


def _read_terminal_exit_observation(
    path: Path,
    project_root: Path,
    task_id: str,
    attempt_id: str,
) -> _TerminalExitObservation | None:
    """Read one exact immutable exit observation paired with a local manifest."""
    if not validate_evidence_path(path, project_root):
        return None
    envelope = read_json_limited(path, max_bytes=65_536, record_type="exit_observation")
    if set(envelope) != {"exit_observation"}:
        return None
    record = envelope.get("exit_observation")
    if (
        not isinstance(record, dict)
        or type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != 1
        or record.get("attempt_id") != attempt_id
        or path.stem != attempt_id
        or record.get("task_id") not in {None, task_id}
        or type(record.get("observed_exit_code")) is not int
        or not _is_valid_utc_timestamp(record.get("observed_at"))
    ):
        return None
    return _TerminalExitObservation(
        record=record,
        identity=_canonical_terminal_record(record),
        exit_code=record["observed_exit_code"],
    )


def _termination_exit_observation_state(
    path: Path,
    project_root: Path,
    task_id: str,
    attempt_id: str,
    manifest_exit_code: object,
    *,
    manifest_complete: bool,
) -> str:
    """Return a bounded observation state without treating malformed reads as absence."""
    if not validate_evidence_path(path, project_root):
        return "absent"
    try:
        envelope = read_json_limited(path, max_bytes=65_536, record_type="exit_observation")
    except FileNotFoundError:
        return "absent"
    except OSError:
        raise
    except (TypeError, ValueError):
        return "invalid"
    if set(envelope) != {"exit_observation"}:
        return "invalid"
    record = envelope.get("exit_observation")
    if (
        not isinstance(record, Mapping)
        or type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != 1
        or record.get("attempt_id") != attempt_id
        or path.stem != attempt_id
        or record.get("task_id") not in {None, task_id}
        or type(record.get("observed_exit_code")) is not int
        or not _is_valid_utc_timestamp(record.get("observed_at"))
    ):
        return "invalid"
    if manifest_exit_code is None and not manifest_complete:
        return "available"
    if type(manifest_exit_code) is not int or record.get("observed_exit_code") != manifest_exit_code:
        return "mismatch"
    return "matching"


def _termination_reservation_state(
    runtime_root: Path,
    binding: ProjectBinding,
    process: _TerminalProcessRecord,
) -> str:
    """Read one exact GPU/CPU reservation identity across all state lanes."""
    reservation_id = process.parameters.get("reservation_id")
    if reservation_id is None:
        return "unassigned"
    if not isinstance(reservation_id, str):
        return "conflict"
    capacity_paths = local_paths(runtime_root)
    matches: list[tuple[str, ReservationIdentity]] = []
    for expected_state, name in (
        ("active", "active"),
        ("provisional", "provisional"),
        ("released", "released"),
        ("active", "cpu_active"),
        ("provisional", "cpu_provisional"),
        ("released", "cpu_released"),
    ):
        path = capacity_paths[name] / f"{reservation_id}.json"
        try:
            envelope = read_json_limited(path, max_bytes=65_536, record_type="reservation")
        except FileNotFoundError:
            continue
        except OSError:
            raise
        except (TypeError, ValueError):
            return "conflict"
        record = envelope.get("reservation") if isinstance(envelope, Mapping) else None
        if not isinstance(record, dict) or record.get("state") != expected_state:
            return "conflict"
        try:
            identity = ReservationIdentity.from_record(record)
        except (TypeError, ValueError):
            return "conflict"
        if (
            identity.reservation_id != reservation_id
            or identity.project_id not in {None, binding.project_id}
            or identity.task_id != process.parameters["task_id"]
            or identity.attempt_id != process.parameters["attempt_id"]
            or identity.fencing_token != process.parameters["fencing_token"]
        ):
            return "conflict"
        matches.append((expected_state, identity))
    if not matches:
        return "conflict"
    states = {state for state, _identity in matches}
    if (
        states in ({"active", "released"}, {"provisional", "released"})
        and len(matches) == 2
        and matches[0][1] == matches[1][1]
    ):
        return "release_pair"
    if len(matches) == 1:
        return matches[0][0]
    return "conflict"


def _record_terminal_observation_problem(
    state: _TerminalBindingState,
    process: _TerminalProcessRecord,
    observation_path: Path,
) -> None:
    """Preserve legacy local diagnostics when strict terminal evidence is unusable."""
    parameters = process.parameters
    try:
        present = is_path_present(observation_path)
    except OSError:
        state.reconciler.record_diagnostic(process.record, "exit_observation_unreadable")
        return
    if present:
        valid, _exit_code = state.reconciler.read_exit_observation(
            observation_path,
            parameters["task_id"],
            parameters["attempt_id"],
            process.record,
        )
        if valid:
            # The local reconciler accepts the older minimum envelope. The
            # isolated protocol also requires immutable time and exact shape.
            state.reconciler.record_diagnostic(process.record, "exit_observation_unreadable")
        return
    try:
        wrapper = inspect_wrapper_identity(process.record)
        group = inspect_local_group_identity(process.record)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return
    wrapper_was_registered = (
        process.record.get("wrapper_pid") is not None or process.record.get("wrapper_start_time_ticks") is not None
    )
    if group.state == "absent" and (not wrapper_was_registered or wrapper.state == "absent"):
        state.reconciler.record_diagnostic(process.record, "exit_observation_missing")


def _read_termination_process_record(
    path: Path,
    project_root: Path,
    binding: ProjectBinding,
) -> _TerminalProcessRecord | None:
    """Read a live manifest that has not acquired a natural exit observation."""
    process = _read_terminal_process_record(path, project_root, binding)
    if process is None or process.record.get("observed_state") != "running":
        return None
    observation_path = project_root / "process-observations" / f"{process.parameters['attempt_id']}.json"
    try:
        if is_path_present(observation_path):
            return None
        wrapper = inspect_wrapper_identity(process.record)
        group = inspect_local_group_identity(process.record)
    except _TERMINATION_SOURCE_ERRORS:
        return None
    if wrapper.state != "alive" and group.state != "alive":
        return None
    return process


def _termination_holder_safe_deadline(process: Mapping[str, Any]) -> float | None:
    """Return the wall-clock holder-safe deadline from a local lease record."""
    lease_expires_at = process.get("lease_expires_at")
    clock_error = process.get("clock_error_bound_seconds")
    if (
        not _is_valid_utc_timestamp(lease_expires_at)
        or type(clock_error) not in {int, float}
        or not math.isfinite(float(clock_error))
        or float(clock_error) < 0.0
    ):
        return None
    try:
        parsed = datetime.fromisoformat(
            lease_expires_at[:-1] + "+00:00" if lease_expires_at.endswith("Z") else lease_expires_at
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.timestamp() - float(clock_error)


def _termination_trigger(process: Mapping[str, Any], evidence: Mapping[str, Any]) -> str | None:
    """Choose the only local trigger allowed by an exact current observation."""
    if evidence.get("cancel_requested") is True:
        return "cancellation_requested"
    deadline = _termination_holder_safe_deadline(process)
    if deadline is not None and time.time() >= deadline:
        return "holder_safe_deadline_elapsed"
    return None


def _termination_decision_id(
    candidate: _TerminationCandidate,
    trigger: str,
) -> str:
    """Build a bounded replay-stable decision identifier from exact identity."""
    canonical = json.dumps(
        {
            "project_id": candidate.binding.project_id,
            "canonical_shared_root": str(candidate.binding.shared_root),
            "machine_name": candidate.binding.machine_name,
            "registration_generation": candidate.binding.registration_generation,
            "runtime_instance_id": candidate.binding.runtime_instance_id,
            "runtime_root": candidate.binding.runtime_root,
            "task_id": candidate.parameters["task_id"],
            "attempt_id": candidate.parameters["attempt_id"],
            "attempt_number": candidate.parameters["attempt_number"],
            "fencing_token": candidate.parameters["fencing_token"],
            "reservation_id": candidate.parameters["reservation_id"],
            "process_identity": candidate.parameters["process_identity"],
            "trigger": trigger,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"termination-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:48]}"


def _termination_local_config(
    runtime: MachineRuntime,
    binding: ProjectBinding,
) -> RootConfig:
    """Build local-only paths without consulting the binding's shared config."""
    paths = machine_project_paths(runtime.root, binding.project_id)
    return RootConfig.from_canonical_paths(
        binding.shared_root,
        binding.shared_root.parent,
        binding.machine_name,
        paths["root"],
    )


def _read_termination_decision(
    cfg: RootConfig,
    attempt_id: str,
    decision_id: str,
) -> dict[str, Any] | None:
    """Read one exact local termination decision envelope."""
    path = decision_path(cfg, attempt_id, decision_id)
    try:
        envelope = read_json_limited(path, max_bytes=65_536, record_type="termination_decision")
    except _TERMINATION_SOURCE_ERRORS:
        return None
    if set(envelope) != {"termination_decision"} or not isinstance(envelope["termination_decision"], dict):
        return None
    return envelope["termination_decision"]


def _termination_decision_matches_candidate(
    decision: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
    *,
    trigger: str | None = None,
) -> bool:
    """Validate a local decision without accepting a changed process identity."""
    if not isinstance(decision, Mapping):
        return False
    expected_trigger = trigger or candidate.trigger
    if expected_trigger not in _TERMINATION_TRIGGERS:
        return False
    parameters = candidate.parameters
    process_identity = parameters.get("process_identity")
    if not isinstance(process_identity, Mapping):
        return False
    if not all(decision.get(field) == parameters.get(field) for field in ("task_id", "attempt_id")):
        return False
    if (
        decision.get("decision_id") != candidate.decision_id
        or decision.get("authority_outcome") != expected_trigger
        or decision.get("reason") != expected_trigger
        or decision.get("process_group_id") != process_identity.get("process_group_id")
        or decision.get("process_group_start_time_ticks") != process_identity.get("process_group_start_time_ticks")
        or decision.get("state") not in {"pending", "signal_committed", "sigterm_sent", "sigkill_sent", "confirmed"}
        or decision.get("shared_commitment") not in {"pending", "committed"}
        or type(decision.get("decision_token")) is not int
        or decision["decision_token"] != parameters["fencing_token"]
    ):
        return False
    attempts = decision.get("signal_attempts")
    return isinstance(attempts, list) and all(isinstance(item, Mapping) for item in attempts)


def _termination_commit_parameters(candidate: _TerminationCandidate) -> dict[str, Any] | None:
    """Return the exact controller contract for a durable local decision."""
    decision = candidate.decision
    revisions = candidate.source_revisions
    if (
        not isinstance(decision, Mapping)
        or not _termination_decision_matches_candidate(decision, candidate)
        or not _has_exact_terminal_revisions(revisions)
    ):
        return None
    return {
        "task_id": candidate.parameters["task_id"],
        "attempt_id": candidate.parameters["attempt_id"],
        "attempt_number": candidate.parameters["attempt_number"],
        "fencing_token": candidate.parameters["fencing_token"],
        "reservation_id": candidate.parameters["reservation_id"],
        "process_identity": candidate.parameters["process_identity"],
        "decision_id": candidate.decision_id,
        "decision_token": decision["decision_token"],
        "authority_outcome": candidate.trigger,
        "reason": candidate.trigger,
        "source_revisions": dict(revisions),
    }


def _termination_commit_proof(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
) -> bool:
    """Accept only the exact shared commitment for this local decision."""
    parameters = _termination_commit_parameters(candidate)
    if (
        parameters is None
        or not isinstance(evidence, Mapping)
        or evidence.get("outcome") not in {"committed", "already_committed"}
        or evidence.get("shared_commitment") != "committed"
        or evidence.get("authority_granted") is not False
        or evidence.get("local_effects") not in ([], ())
    ):
        return False
    identity_fields = (
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "reservation_id",
        "process_identity",
        "decision_id",
        "decision_token",
        "authority_outcome",
        "source_revisions",
    )
    return (
        evidence.get("machine_name") == candidate.binding.machine_name
        and evidence.get("reason") is None
        and evidence.get("decision_reason") == parameters["reason"]
        and all(evidence.get(field) == parameters[field] for field in identity_fields)
        and _has_exact_terminal_revisions(evidence.get("committed_revisions"))
    )


def _termination_observation_is_current(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
) -> bool:
    """Validate the typed active observation used to authorize cancellation."""
    if not isinstance(evidence, Mapping):
        return False
    parameters = candidate.parameters
    return (
        evidence.get("outcome") == "current"
        and evidence.get("task_phase") == "running"
        and evidence.get("attempt_phase") == "running"
        and evidence.get("execution_machine_name") == candidate.binding.machine_name
        and evidence.get("reservation_machine_name") == candidate.binding.machine_name
        and evidence.get("termination_result") in {None, "termination_requested"}
        and _terminal_observation_matches_candidate(evidence, candidate)
        and all(evidence.get(field) == parameters[field] for field in ("task_id", "attempt_id", "mode"))
    )


def _termination_observation_has_shared_marker(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
) -> bool:
    """Validate a fresh exact observation after the shared termination commit."""
    return (
        _termination_observation_is_current(evidence, candidate)
        and candidate.decision is not None
        and candidate.decision.get("shared_commitment") == "committed"
        and (
            candidate.trigger == "holder_safe_deadline_elapsed"
            or (isinstance(evidence, Mapping) and evidence.get("cancel_requested") is True)
        )
    )


def _termination_terminal_target(
    candidate: _TerminationCandidate,
) -> tuple[str, str, str | None]:
    """Classify a committed termination after exact process absence."""
    attempts = None if candidate.decision is None else candidate.decision.get("signal_attempts")
    if isinstance(attempts, list) and attempts:
        return "cancelled", "terminated_by_agent", "terminated"
    if candidate.trigger == "cancellation_requested":
        return "cancelled", "termination_process_already_exited", "already_exited"
    return "failed", "process_exited_without_status", None


def _build_termination_transition(
    candidate: _TerminationCandidate,
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the exact cancelled terminal publication request."""
    if not _termination_observation_has_shared_marker(evidence, candidate):
        return None
    source_revisions = evidence.get("source_revisions")
    if not _has_exact_terminal_revisions(source_revisions):
        return None
    parameters = candidate.parameters
    phase, reason, termination_result = _termination_terminal_target(candidate)
    digest = authority_terminal_transition_digest(
        mode="active",
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=candidate.binding.machine_name,
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=phase,
        reason=reason,
        exit_code=None,
        termination_result=termination_result,
    )
    return {
        "task_id": parameters["task_id"],
        "attempt_id": parameters["attempt_id"],
        "attempt_number": parameters["attempt_number"],
        "fencing_token": parameters["fencing_token"],
        "reservation_id": parameters["reservation_id"],
        "process_identity": parameters["process_identity"],
        "mode": "active",
        "phase": phase,
        "reason": reason,
        "exit_code": None,
        "termination_result": termination_result,
        "source_revisions": dict(source_revisions),
        "transition_digest": digest,
    }


def _termination_publication_proof(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
    transition: Mapping[str, Any] | None,
) -> bool:
    """Accept only a committed cancelled terminal transition for this process."""
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(transition, Mapping)
        or evidence.get("outcome") not in {"committed", "already_committed"}
        or evidence.get("machine_name") != candidate.binding.machine_name
        or evidence.get("reservation_machine_name") not in {None, candidate.binding.machine_name}
        or evidence.get("authority_granted") is not False
        or evidence.get("local_effects") not in ([], ())
    ):
        return False
    parameters = candidate.parameters
    identity_fields = (
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "reservation_id",
        "process_identity",
        "mode",
    )
    if not all(evidence.get(field) == parameters.get(field) == transition.get(field) for field in identity_fields):
        return False
    phase, reason, termination_result = _termination_terminal_target(candidate)
    expected = {
        "phase": phase,
        "transition_reason": reason,
        "exit_code": None,
        "termination_result": termination_result,
    }
    if not all(
        evidence.get(field) == value and transition.get(field if field != "transition_reason" else "reason") == value
        for field, value in expected.items()
    ):
        return False
    source_revisions = transition.get("source_revisions")
    if (
        evidence.get("source_revisions") != source_revisions
        or not _has_exact_terminal_revisions(source_revisions)
        or evidence.get("transition_digest") != transition.get("transition_digest")
    ):
        return False
    digest = authority_terminal_transition_digest(
        mode="active",
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=candidate.binding.machine_name,
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=phase,
        reason=reason,
        exit_code=None,
        termination_result=termination_result,
    )
    if transition.get("transition_digest") != digest:
        return False
    return matches_terminal_lifecycle_event(
        evidence.get("lifecycle_event"),
        evidence.get("committed_revisions"),
        parameters,
        phase=phase,
        reason=reason,
        exit_code=None,
    )


def _termination_terminal_observation_proof(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminationCandidate,
) -> bool:
    """Accept an exact already-terminal observation as publication replay proof."""
    phase, reason, termination_result = _termination_terminal_target(candidate)
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("outcome") not in {"already_terminal", "settled_terminal"}
        or evidence.get("attempt_phase") != phase
        or evidence.get("task_phase") != phase
        or evidence.get("execution_machine_name") != candidate.binding.machine_name
        or evidence.get("reservation_machine_name") != candidate.binding.machine_name
        or evidence.get("attempt_result_reason") != reason
        or evidence.get("attempt_exit_code") is not None
        or evidence.get("termination_result") != termination_result
    ):
        return False
    return _terminal_observation_matches_candidate(evidence, candidate)


def _terminal_observation_matches_candidate(
    evidence: Mapping[str, Any],
    candidate: _TerminalCandidate,
) -> bool:
    return matches_terminal_observation_identity(
        evidence,
        candidate.parameters,
        candidate.binding.machine_name,
    )


def _terminal_transition_target(
    candidate: _TerminalCandidate,
    cancel_requested: bool,
) -> tuple[str, str, str | None] | None:
    """Return the exact terminal result allowed by this supervision mode."""
    return _select_terminal_transition_target(
        mode=candidate.parameters.get("mode"),
        exit_code=candidate.exit_code,
        cancel_requested=cancel_requested,
    )


def _build_terminal_transition(
    candidate: _TerminalCandidate,
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    mode = candidate.parameters.get("mode")
    task_phase = evidence.get("task_phase")
    attempt_phase = evidence.get("attempt_phase")
    if mode == "active":
        source_phases_match = task_phase == "running" and attempt_phase == "running"
    elif mode == "detached_orphan":
        # A retained local process identity is eligible for detached completion
        # only while the shared Attempt still records the expired claim as an
        # orphan. A recovered running Attempt has a rotated fencing identity.
        source_phases_match = task_phase == "blocked" and attempt_phase == "orphaned"
    else:
        return None
    if (
        evidence.get("outcome") != "current"
        or not _terminal_observation_matches_candidate(evidence, candidate)
        or not source_phases_match
        or evidence.get("termination_result") is not None
        or evidence.get("execution_machine_name") != candidate.binding.machine_name
        or evidence.get("reservation_machine_name") != candidate.binding.machine_name
    ):
        return None
    return _construct_terminal_transition(
        parameters=candidate.parameters,
        machine_name=candidate.binding.machine_name,
        exit_code=candidate.exit_code,
        source_revisions=evidence["source_revisions"],
        cancel_requested=evidence["cancel_requested"],
    )


def _recover_terminal_transition(
    request: object,
    candidate: _TerminalCandidate,
    registry_revision: int,
) -> dict[str, Any] | None:
    """Recover one exact persisted publication after coordinator restart."""
    binding = candidate.binding
    if (
        getattr(request, "operation_kind", None) != "authority_terminal_publish"
        or getattr(request, "project_id", None) != binding.project_id
        or getattr(request, "canonical_shared_root", None) != str(binding.shared_root)
        or getattr(request, "registration_generation", None) != binding.registration_generation
        or getattr(request, "registry_revision", None) != registry_revision
    ):
        return None
    request_parameters = getattr(request, "parameters", None)
    source_revisions = getattr(request, "source_revisions", None)
    if not isinstance(request_parameters, Mapping) or not _has_exact_terminal_revisions(source_revisions):
        return None
    parameters = candidate.parameters
    if request_parameters.get("machine_name") != binding.machine_name or not all(
        request_parameters.get(field) == parameters[field]
        for field in (
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "mode",
        )
    ):
        return None
    termination_result = request_parameters.get("termination_result")
    if termination_result not in {None, "already_exited"}:
        return None
    target = _terminal_transition_target(candidate, termination_result == "already_exited")
    if target is None:
        return None
    phase, reason, expected_termination = target
    if (
        request_parameters.get("phase") != phase
        or request_parameters.get("reason") != reason
        or request_parameters.get("exit_code") != candidate.exit_code
        or termination_result != expected_termination
    ):
        return None
    digest = authority_terminal_transition_digest(
        mode=parameters["mode"],
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=binding.machine_name,
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=phase,
        reason=reason,
        exit_code=candidate.exit_code,
        termination_result=termination_result,
    )
    if request_parameters.get("transition_digest") != digest:
        return None
    return {
        **parameters,
        "phase": phase,
        "reason": reason,
        "exit_code": candidate.exit_code,
        "termination_result": termination_result,
        "source_revisions": dict(source_revisions),
        "transition_digest": digest,
    }


def _is_terminal_observation_proof(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminalCandidate,
) -> TerminalObservationProof | None:
    return build_terminal_observation_proof(
        evidence=evidence,
        parameters=candidate.parameters,
        machine_name=candidate.binding.machine_name,
        exit_code=candidate.exit_code,
    )


def _is_terminal_publication_proof(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminalCandidate,
    transition: Mapping[str, Any] | None,
) -> TerminalPublicationProof | None:
    return build_terminal_publication_proof(
        evidence=evidence,
        parameters=candidate.parameters,
        machine_name=candidate.binding.machine_name,
        exit_code=candidate.exit_code,
        transition=transition,
    )


def _is_active_task_or_claim_stale(evidence: Mapping[str, Any] | None) -> bool:
    if not isinstance(evidence, Mapping) or evidence.get("outcome") != "stale":
        return False
    reason = evidence.get("reason")
    if reason in {"task_identity_mismatch", "claim_not_current"}:
        return True
    return reason == "task_phase_mismatch" and evidence.get("task_phase") in {None, "blocked"}


def _is_active_terminal_observation_fallback(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminalCandidate,
) -> bool:
    return (
        candidate.parameters.get("mode") == "active"
        and _is_active_task_or_claim_stale(evidence)
        and isinstance(evidence, Mapping)
        and _terminal_observation_matches_candidate(evidence, candidate)
    )


def _is_active_terminal_publication_fallback(
    evidence: Mapping[str, Any] | None,
    candidate: _TerminalCandidate,
    transition: Mapping[str, Any] | None,
) -> bool:
    if (
        candidate.parameters.get("mode") != "active"
        or not _is_active_task_or_claim_stale(evidence)
        or not isinstance(evidence, Mapping)
        or not isinstance(transition, Mapping)
        or evidence.get("authority_granted") is not False
        or evidence.get("local_effects") not in ([], ())
    ):
        return False
    parameters = candidate.parameters
    if not all(
        evidence.get(field) == parameters[field] and transition.get(field) == parameters[field]
        for field in (
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "reservation_id",
            "process_identity",
            "mode",
        )
    ):
        return False
    termination_result = transition.get("termination_result")
    if termination_result not in {None, "already_exited"}:
        return False
    target = _terminal_transition_target(candidate, termination_result == "already_exited")
    if target is None:
        return False
    phase, reason, expected_termination = target
    if (
        transition.get("phase") != phase
        or transition.get("reason") != reason
        or transition.get("exit_code") != candidate.exit_code
        or termination_result != expected_termination
        or evidence.get("phase") != phase
        or evidence.get("transition_reason") != reason
        or evidence.get("exit_code") != candidate.exit_code
        or evidence.get("termination_result") != termination_result
        or evidence.get("source_revisions") != transition.get("source_revisions")
        or not _has_exact_terminal_revisions(evidence.get("source_revisions"))
    ):
        return False
    digest = authority_terminal_transition_digest(
        mode="active",
        task_id=parameters["task_id"],
        attempt_id=parameters["attempt_id"],
        attempt_number=parameters["attempt_number"],
        fencing_token=parameters["fencing_token"],
        machine_name=candidate.binding.machine_name,
        reservation_id=parameters["reservation_id"],
        process_identity=parameters["process_identity"],
        phase=phase,
        reason=reason,
        exit_code=candidate.exit_code,
        termination_result=termination_result,
    )
    return (
        transition.get("transition_digest") == digest
        and evidence.get("transition_digest") == digest
        and evidence.get("reservation_machine_name") == candidate.binding.machine_name
        and evidence.get("reservation_id") == parameters["reservation_id"]
    )


def _apply_terminal_local_effects(
    candidate: _TerminalCandidate,
    proof: TerminalObservationProof | TerminalPublicationProof,
    paths: Mapping[str, Path],
    reconciler: LocalExitReconciler,
) -> str:
    """Apply replay-safe local completion only after exact shared terminal proof."""
    if not terminal_proof_matches_candidate(
        proof,
        parameters=candidate.parameters,
        machine_name=candidate.binding.machine_name,
        exit_code=candidate.exit_code,
    ):
        return "invalid"
    attempt_id = candidate.parameters["attempt_id"]
    acquired_guard = False
    exact_evidence = False
    with evidence_write_guard(paths["root"], attempt_id) as acquired:
        if not acquired:
            return "deferred"
        acquired_guard = True
        process = _read_terminal_process_record(candidate.process_path, paths["root"], candidate.binding)
        observation = _read_terminal_exit_observation(
            candidate.observation_path,
            paths["root"],
            candidate.parameters["task_id"],
            attempt_id,
        )
        if (
            process is None
            or observation is None
            or process.parameters != {key: value for key, value in candidate.parameters.items() if key != "mode"}
            or process.identity not in {candidate.process_identity, candidate.post_update_process_identity}
            or observation.identity != candidate.observation_identity
            or observation.exit_code != candidate.exit_code
        ):
            return "invalid"
        exact_evidence = True
        if candidate.post_update_process_identity == process.identity:
            pass
        elif (
            process.record.get("observed_state") == "exited"
            and process.record.get("observed_exit_code") == candidate.exit_code
            and _is_valid_utc_timestamp(process.record.get("observed_exited_at"))
        ):
            candidate.post_update_process = dict(process.record)
            candidate.post_update_process_identity = process.identity
        else:
            if candidate.post_update_process is None:
                updated = dict(process.record)
                updated.update(
                    {
                        "observed_state": "exited",
                        "observed_exit_code": candidate.exit_code,
                        "observed_exited_at": utc_now(),
                    }
                )
                candidate.post_update_process = updated
                candidate.post_update_process_identity = _canonical_terminal_record(updated)
            try:
                atomic_replace(candidate.process_path, {"process": candidate.post_update_process})
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                # Preserve the exact desired record so an ambiguous local replace
                # can be replayed without changing its timestamp or other fields.
                return "deferred"

    if acquired_guard and exact_evidence:
        reconciler.reconcile_observation(candidate.observation_path, bounded=True)
    return "applied"


def _read_running_publication_source(
    path: Path,
    project_root: Path,
    binding: ProjectBinding,
    envelope_key: str,
    *,
    allow_unassigned_reservation: bool = False,
) -> _RunningPublicationSource | None:
    """Read one bounded runner registration and derive its canonical Attempt number."""
    if not validate_evidence_path(path, project_root):
        return None
    envelope = read_json_limited(path, max_bytes=65_536, record_type=envelope_key)
    if set(envelope) != {envelope_key}:
        return None
    record = envelope.get(envelope_key)
    if (
        envelope_key != "process_registration"
        or not isinstance(record, dict)
        or type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != 1
    ):
        return None

    task_id = record.get("task_id")
    attempt_id = record.get("attempt_id")
    if not isinstance(task_id, str) or not isinstance(attempt_id, str) or path.stem != attempt_id:
        return None
    try:
        validate_identifier(task_id, "process task_id")
        validate_identifier(attempt_id, "process attempt_id")
    except ValueError:
        return None
    prefix = f"{task_id}-attempt-"
    suffix = attempt_id.removeprefix(prefix)
    if not attempt_id.startswith(prefix) or not suffix.isascii() or not suffix.isdecimal():
        return None
    attempt_number = int(suffix)
    if attempt_number < 1 or attempt_id != f"{task_id}-attempt-{attempt_number}":
        return None

    fencing_token = record.get("fencing_token")
    reservation_id = registration_reservation(record, project_root)
    process_created_at = record.get("process_created_at")
    if record.get("machine_name") != binding.machine_name or type(fencing_token) is not int or fencing_token < 1:
        return None
    if not isinstance(process_created_at, str) or not process_created_at or len(process_created_at) > 40:
        return None
    if reservation_id is None:
        if not allow_unassigned_reservation:
            return None
    elif isinstance(reservation_id, str):
        try:
            validate_identifier(reservation_id, "process reservation_id")
        except ValueError:
            return None
    else:
        return None
    try:
        parsed_created_at = datetime.fromisoformat(
            process_created_at[:-1] + "+00:00" if process_created_at.endswith("Z") else process_created_at
        )
    except ValueError:
        return None
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() != timezone.utc.utcoffset(parsed_created_at):
        return None

    process_identity: dict[str, int | None] = {}
    for field in PROCESS_IDENTITY_FIELDS:
        if field not in record:
            return None
        value = record.get(field)
        positive = field in {"wrapper_pid", "process_group_id"}
        if value is not None and (type(value) is not int or value < (1 if positive else 0)):
            return None
        process_identity[field] = value

    parameters = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "reservation_id": reservation_id,
        "process_identity": process_identity,
        "process_created_at": process_created_at,
    }
    identity = (
        task_id,
        attempt_id,
        attempt_number,
        fencing_token,
        binding.machine_name,
        process_created_at,
        *(process_identity[field] for field in PROCESS_IDENTITY_FIELDS),
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return _RunningPublicationSource(record=record, parameters=parameters, identity=identity)


def _read_launch_intent_source(
    path: Path,
    project_root: Path,
) -> _RunningPublicationSource | None:
    """Read the smaller pre-child Runner envelope without inventing registration fields."""
    if not validate_evidence_path(path, project_root):
        return None
    envelope = read_json_limited(path, max_bytes=65_536, record_type="launch_intent")
    if set(envelope) != {"launch_intent"}:
        return None
    record = envelope.get("launch_intent")
    if (
        not isinstance(record, dict)
        or type(record.get("protocol_version")) is not int
        or record.get("protocol_version") != 1
    ):
        return None
    task_id = record.get("task_id")
    attempt_id = record.get("attempt_id")
    fencing_token = record.get("fencing_token")
    if (
        not isinstance(task_id, str)
        or not isinstance(attempt_id, str)
        or path.stem != attempt_id
        or type(fencing_token) is not int
        or fencing_token < 1
        or "wrapper_pid" not in record
        or "wrapper_start_time_ticks" not in record
    ):
        return None
    try:
        validate_identifier(task_id, "launch intent task_id")
        validate_identifier(attempt_id, "launch intent attempt_id")
    except ValueError:
        return None
    prefix = f"{task_id}-attempt-"
    suffix = attempt_id.removeprefix(prefix)
    if (
        not attempt_id.startswith(prefix)
        or not suffix.isascii()
        or not suffix.isdecimal()
        or int(suffix) < 1
        or attempt_id != f"{task_id}-attempt-{int(suffix)}"
    ):
        return None
    wrapper_pid = record.get("wrapper_pid")
    wrapper_ticks = record.get("wrapper_start_time_ticks")
    if wrapper_pid is not None and (type(wrapper_pid) is not int or wrapper_pid < 1):
        return None
    if wrapper_ticks is not None and (type(wrapper_ticks) is not int or wrapper_ticks < 0):
        return None
    identity = (
        task_id,
        attempt_id,
        fencing_token,
        wrapper_pid,
        wrapper_ticks,
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return _RunningPublicationSource(record=record, parameters={}, identity=identity)


def _registration_matches_intent(
    registration: _RunningPublicationSource,
    intent: _RunningPublicationSource,
) -> bool:
    """Match only identity fields present before the child process exists."""
    return all(
        registration.record.get(field) == intent.record.get(field)
        for field in (
            "task_id",
            "attempt_id",
            "fencing_token",
            "wrapper_pid",
            "wrapper_start_time_ticks",
        )
    )


def _ensure_running_manifest(
    paths: Mapping[str, Path],
    source: _RunningPublicationSource,
) -> dict[str, Any]:
    """Materialize a registration once and return its typed publication fields."""
    attempt_id = source.parameters["attempt_id"]
    manifest = paths["processes"] / f"{attempt_id}.json"
    if not is_path_present(manifest):
        process = dict(source.record)
        process.update(
            {
                "observed_state": "running",
                "supervisor": "agent",
                "authority_state": "healthy",
                "created_by": "agent",
            }
        )
        atomic_replace(manifest, {"process": process})
    return dict(source.parameters)


def _terminal_matches_running_request(
    request: ProjectIORequest,
    binding: ProjectBinding,
    terminal: _TerminalProcessRecord,
) -> bool:
    """Correlate an exited manifest with only its exact running publication."""
    return (
        request.operation_kind == "authority_running_publish"
        and request.project_id == binding.project_id
        and dict(request.parameters)
        == {
            "machine_name": binding.machine_name,
            **terminal.parameters,
            "process_created_at": terminal.record.get("process_created_at"),
        }
    )


@dataclass(slots=True)
class _AuthorityQuiescenceCensus:
    turn: BindingTurn
    executor_epoch: str
    directory_versions: dict[Path, tuple[int, int, int, int] | None] = field(default_factory=dict)
    is_quiescent: bool = False
    mutation_watch: _DirectoryMutationWatch = field(default_factory=_DirectoryMutationWatch)

    @staticmethod
    def directory_version(path: Path) -> tuple[int, int, int, int] | None:
        try:
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(str(path))
        return metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_ctime_ns

    def has_unchanged_directories(self) -> bool:
        return self.mutation_watch.unchanged() and all(
            self.directory_version(path) == version for path, version in self.directory_versions.items()
        )

    def track_directory(self, path: Path) -> None:
        self.mutation_watch.watch(path)
        self.directory_versions[path] = self.directory_version(path)

    def close(self) -> None:
        self.mutation_watch.close()


class AttemptSupervisionCoordinator:
    """Materialize local running evidence before publishing exact typed intent."""

    def __init__(self, runtime: MachineRuntime, controller: ProjectIOController) -> None:
        self.runtime = runtime
        self.controller = controller
        self._reservation_recovery = ProcessReservationRecovery(runtime.root)
        self._scans: dict[tuple[str, ...], _RunningPublicationScans] = {}
        self._running_publication_intents: dict[tuple[str, ...], Mapping[str, Any]] = {}
        self._binding_offset = 0
        self._terminal_states: dict[tuple[str, ...], _TerminalBindingState] = {}
        self._terminal_candidates: dict[str, _TerminalCandidate] = {}
        self._terminal_binding_offset = 0
        self._termination_states: dict[tuple[str, ...], _TerminationBindingState] = {}
        self._termination_candidates: dict[tuple[str, str], _TerminationCandidate] = {}
        self._termination_binding_offset = 0
        self._orphan_recovery_scans: dict[tuple[str, ...], EvidenceScan] = {}
        self._orphan_recovery_scan_complete: set[tuple[str, ...]] = set()
        self._orphan_recovery_entries: dict[str, Mapping[str, Any]] = {}
        self._orphan_recovery_binding_offset = 0
        self._initial_reconciliation_scans: dict[tuple[str, ...], dict[str, EvidenceScan]] = {}
        self._initial_termination_scans: dict[tuple[str, ...], dict[str, EvidenceScan]] = {}
        self._initial_termination_top_complete: set[tuple[str, ...]] = set()
        self._initial_invalid_lanes: dict[tuple[str, ...], set[str]] = {}
        self._initial_deferred_lanes: set[tuple[str, ...]] = set()
        self._initial_pending_termination_lanes: set[tuple[str, ...]] = set()
        self._termination_convergence_findings: dict[tuple[str, ...], dict[str, TerminationConvergence]] = {}
        self._initial_observed: set[tuple[str, ...]] = set()
        self._initial_reconciled: set[tuple[str, ...]] = set()
        self._initial_reconciliation_offset = 0
        self._initial_complete_lanes: dict[tuple[str, ...], set[str]] = {}
        self._authority_quiescence: dict[tuple[str, ...], _AuthorityQuiescenceCensus] = {}

    def close(self) -> None:
        """Release all owned advisory scans; repeated calls are harmless."""
        self._reservation_recovery.close()
        for scans in self._scans.values():
            scans.close()
        self._scans.clear()
        self._running_publication_intents.clear()
        self._binding_offset = 0
        for state in self._terminal_states.values():
            state.close()
        self._terminal_states.clear()
        self._terminal_candidates.clear()
        self._terminal_binding_offset = 0
        for state in self._termination_states.values():
            state.close()
        self._termination_states.clear()
        self._termination_candidates.clear()
        self._termination_binding_offset = 0
        for scan in self._orphan_recovery_scans.values():
            scan.close()
        self._orphan_recovery_scans.clear()
        self._orphan_recovery_scan_complete.clear()
        self._orphan_recovery_entries.clear()
        self._orphan_recovery_binding_offset = 0
        for scans in self._initial_reconciliation_scans.values():
            for scan in scans.values():
                scan.close()
        self._initial_reconciliation_scans.clear()
        for scans in self._initial_termination_scans.values():
            for scan in scans.values():
                scan.close()
        self._initial_termination_scans.clear()
        self._initial_termination_top_complete.clear()
        self._initial_invalid_lanes.clear()
        self._initial_deferred_lanes.clear()
        self._initial_pending_termination_lanes.clear()
        self._termination_convergence_findings.clear()
        self._initial_observed.clear()
        self._initial_reconciled.clear()
        self._initial_reconciliation_offset = 0
        self._initial_complete_lanes.clear()
        for census in self._authority_quiescence.values():
            census.close()
        self._authority_quiescence.clear()

    def _discard_initial_scan(self, signature: tuple[str, ...]) -> None:
        for scan in self._initial_reconciliation_scans.pop(signature, {}).values():
            scan.close()
        for scan in self._initial_termination_scans.pop(signature, {}).values():
            scan.close()
        self._initial_termination_top_complete.discard(signature)
        self._initial_invalid_lanes.pop(signature, None)
        self._initial_deferred_lanes.discard(signature)
        self._initial_pending_termination_lanes.discard(signature)
        self._initial_complete_lanes.pop(signature, None)

    def _discard_authority_census(self, signature: tuple[str, ...]) -> None:
        census = self._authority_quiescence.pop(signature, None)
        if census is not None:
            census.close()

    @staticmethod
    def _initial_reconciliation_signature(
        binding: ProjectBinding,
        registry_revision: int,
    ) -> tuple[str, ...]:
        return _authority_binding_signature(binding, registry_revision)

    def _initial_reconciliation_pending(
        self,
        binding: ProjectBinding,
        registry_revision: int,
        signature: tuple[str, ...],
    ) -> bool:
        """Return whether this exact binding still has local or typed work pending."""
        if any(candidate.binding_signature == signature for candidate in self._terminal_candidates.values()):
            return True
        if any(
            candidate.binding_signature == signature
            and (candidate.convergence is None or candidate.convergence.state != "converged")
            for candidate in self._termination_candidates.values()
        ):
            return True
        if any(entry.get("binding_signature") == signature for entry in self._orphan_recovery_entries.values()):
            return True
        if signature in self._initial_deferred_lanes:
            return True
        if any(
            result.state != "converged" for result in self._termination_convergence_findings.get(signature, {}).values()
        ):
            return True
        try:
            unresolved = self.controller.executor.unresolved_requests()
        except _SUPERVISION_LANE_ERRORS:
            return True
        return any(
            request.operation_kind in _SUPERVISION_OPERATION_KINDS
            and request.project_id == binding.project_id
            and request.canonical_shared_root == str(binding.shared_root)
            and request.registration_generation == binding.registration_generation
            and request.registry_revision == registry_revision
            for request in unresolved
        )

    def _initial_reservation_settled(
        self,
        binding: ProjectBinding,
        process: _TerminalProcessRecord,
    ) -> bool:
        """Prove that a terminated process no longer owns a local reservation."""
        return _termination_reservation_state(self.runtime.root, binding, process) in {"unassigned", "released"}

    def _record_termination_convergence(
        self,
        signature: tuple[str, ...],
        attempt_id: str,
        result: TerminationConvergence,
    ) -> None:
        """Retain one bounded current convergence result for diagnostics and readiness."""
        findings = self._termination_convergence_findings.setdefault(signature, {})
        findings[attempt_id] = result
        if len(findings) > _MAX_TERMINATION_CANDIDATES:
            del findings[next(iter(findings))]

    def _termination_convergence_for_decision(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        signature: tuple[str, ...],
        decision_path_value: Path,
        *,
        process: _TerminalProcessRecord | None = None,
        decision: Mapping[str, Any] | None = None,
    ) -> TerminationConvergence | None:
        """Read complete bounded local evidence and classify one termination decision."""
        try:
            relative = decision_path_value.relative_to(paths["root"])
            if (
                not relative.parts
                or not validate_evidence_path(decision_path_value, paths["root"])
                or not decision_path_value.parent.name
            ):
                result = TerminationConvergence("invalid", "termination_decision_invalid", None, False, False)
                self._record_termination_convergence(signature, decision_path_value.parent.name, result)
                return result
            attempt_id = decision_path_value.parent.name
            try:
                validate_identifier(attempt_id, "termination decision attempt_id")
            except ValueError:
                result = TerminationConvergence(
                    "invalid",
                    "termination_decision_invalid",
                    None,
                    False,
                    False,
                )
                self._record_termination_convergence(signature, "invalid-attempt", result)
                return result
            if process is None:
                process = _read_terminal_process_record(
                    paths["processes"] / f"{attempt_id}.json",
                    paths["root"],
                    binding,
                )
            if process is None or process.parameters["attempt_id"] != attempt_id:
                result = TerminationConvergence("invalid", "termination_decision_invalid", None, False, False)
                self._record_termination_convergence(signature, attempt_id, result)
                return result
            if decision is None:
                try:
                    envelope = read_json_limited(
                        decision_path_value,
                        max_bytes=65_536,
                        record_type="termination_decision",
                    )
                except FileNotFoundError:
                    raise
                except OSError:
                    raise
                except (TypeError, ValueError):
                    result = TerminationConvergence("invalid", "termination_decision_invalid", None, False, False)
                    self._record_termination_convergence(signature, attempt_id, result)
                    return result
                decision = envelope.get("termination_decision") if set(envelope) == {"termination_decision"} else None
            if not isinstance(decision, Mapping):
                result = TerminationConvergence("invalid", "termination_decision_invalid", None, False, False)
                self._record_termination_convergence(signature, attempt_id, result)
                return result

            candidate = self._new_termination_candidate(
                binding,
                signature,
                paths["processes"] / f"{attempt_id}.json",
                process,
            )
            candidate.trigger = decision.get("authority_outcome")
            candidate.decision_id = decision_path_value.stem
            candidate.decision = dict(decision)
            decision_matches = (
                candidate.trigger in _TERMINATION_TRIGGERS
                and decision_path_value.stem == _termination_decision_id(candidate, candidate.trigger)
                and _termination_decision_matches_candidate(decision, candidate)
            )
            if not decision_matches:
                result = TerminationConvergence("invalid", "termination_decision_invalid", None, False, False)
                self._record_termination_convergence(signature, attempt_id, result)
                return result

            group = inspect_local_group_identity(process.record)
            wrapper_registered = (
                process.record.get("wrapper_pid") is not None
                or process.record.get("wrapper_start_time_ticks") is not None
            )
            wrapper = inspect_wrapper_identity(process.record) if wrapper_registered else None
            if group.state == "unknown" or (wrapper is not None and wrapper.state == "unknown"):
                return None
            process_absent = group.state == "absent" and (wrapper is None or wrapper.state == "absent")

            decision_state = decision.get("state")
            if decision_state != "confirmed":
                result = classify_termination_convergence(
                    TerminationConvergenceEvidence(
                        decision_matches=True,
                        decision_state=decision_state,
                        shared_commitment=decision.get("shared_commitment"),
                        confirmation=decision.get("confirmation"),
                        manifest_state=process.record.get("observed_state"),
                        manifest_exit_code=process.record.get("observed_exit_code"),
                        manifest_exited_at_valid=_is_valid_utc_timestamp(process.record.get("observed_exited_at")),
                        process_absent=process_absent,
                        registration_matches=True,
                        exit_observation_state="absent",
                        reservation_state="unassigned",
                    )
                )
                self._record_termination_convergence(signature, attempt_id, result)
                return result

            registration = _read_running_publication_source(
                paths["registrations"] / f"{attempt_id}.json",
                paths["root"],
                binding,
                "process_registration",
                allow_unassigned_reservation=True,
            )
            registration_matches = (
                registration is not None
                and all(
                    registration.parameters.get(field) == process.parameters.get(field)
                    for field in (
                        "task_id",
                        "attempt_id",
                        "attempt_number",
                        "fencing_token",
                        "reservation_id",
                        "process_identity",
                    )
                )
                and registration.record.get("process_created_at") == process.record.get("process_created_at")
            )
            observation_state = _termination_exit_observation_state(
                paths["observations"] / f"{attempt_id}.json",
                paths["root"],
                process.parameters["task_id"],
                attempt_id,
                process.record.get("observed_exit_code"),
                manifest_complete=(
                    process.record.get("observed_state") == "exited"
                    and _is_valid_utc_timestamp(process.record.get("observed_exited_at"))
                ),
            )
            reservation_state = _termination_reservation_state(self.runtime.root, binding, process)
            result = classify_termination_convergence(
                TerminationConvergenceEvidence(
                    decision_matches=True,
                    decision_state=decision.get("state"),
                    shared_commitment=decision.get("shared_commitment"),
                    confirmation=decision.get("confirmation"),
                    manifest_state=process.record.get("observed_state"),
                    manifest_exit_code=process.record.get("observed_exit_code"),
                    manifest_exited_at_valid=_is_valid_utc_timestamp(process.record.get("observed_exited_at")),
                    process_absent=process_absent,
                    registration_matches=registration_matches,
                    exit_observation_state=observation_state,
                    reservation_state=reservation_state,
                )
            )
            self._record_termination_convergence(signature, attempt_id, result)
            return result
        except _TERMINATION_SOURCE_ERRORS:
            raise

    def _initial_termination_decision_converged(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        signature: tuple[str, ...],
        decision_path_value: Path,
    ) -> bool:
        """Validate one durable termination decision through the shared classifier."""
        try:
            result = self._termination_convergence_for_decision(
                binding,
                paths,
                signature,
                decision_path_value,
            )
        except OSError:
            self._initial_deferred_lanes.add(signature)
            self._initial_pending_termination_lanes.add(signature)
            return False
        if result is None:
            self._initial_deferred_lanes.add(signature)
            self._initial_pending_termination_lanes.add(signature)
            return False
        if result.state == "repairable":
            self._initial_pending_termination_lanes.add(signature)
        return result.state == "converged"

    def _advance_initial_termination_lane(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        signature: tuple[str, ...],
        top_scan: EvidenceScan,
    ) -> tuple[bool, bool]:
        """Advance one bounded top-level and nested decision cursor.

        The top-level cursor yields one attempt directory at a time.  Its
        nested cursor is retained until EOF, so malformed first records cannot
        reset discovery and starve later decisions.
        """
        nested_scans = self._initial_termination_scans.setdefault(signature, {})
        invalid = False
        if nested_scans:
            nested_path, nested_scan = next(iter(nested_scans.items()))
            directory = Path(nested_path)
            page = nested_scan.take(_INITIAL_RECONCILIATION_ENTRIES_PER_LANE)
            for decision_path_value in page.paths:
                converged = self._initial_termination_decision_converged(
                    binding,
                    paths,
                    signature,
                    decision_path_value,
                )
                result = self._termination_convergence_findings.get(signature, {}).get(decision_path_value.parent.name)
                if not converged and result is not None and result.state == "invalid":
                    invalid = True
            if not page.paths and page.is_complete:
                invalid = True
            if page.is_complete:
                invalid = not self._is_termination_directory_unchanged(signature, directory) or invalid
                nested_scans.pop(nested_path).close()
        elif signature not in self._initial_termination_top_complete:
            page = top_scan.take(1)
            if page.is_complete:
                self._initial_termination_top_complete.add(signature)
            if page.paths:
                directory = page.paths[0]
                census = self._authority_quiescence.get(signature)
                if census is not None:
                    census.track_directory(directory)
                nested_scans[str(directory)] = EvidenceScan(directory)
                nested_scan = nested_scans[str(directory)]
                nested_page = nested_scan.take(_INITIAL_RECONCILIATION_ENTRIES_PER_LANE)
                for decision_path_value in nested_page.paths:
                    converged = self._initial_termination_decision_converged(
                        binding,
                        paths,
                        signature,
                        decision_path_value,
                    )
                    result = self._termination_convergence_findings.get(signature, {}).get(
                        decision_path_value.parent.name
                    )
                    if not converged and result is not None and result.state == "invalid":
                        invalid = True
                if not nested_page.paths and nested_page.is_complete:
                    invalid = True
                if nested_page.is_complete:
                    invalid = not self._is_termination_directory_unchanged(signature, directory) or invalid
                    nested_scans.pop(str(directory)).close()
        complete = signature in self._initial_termination_top_complete and not nested_scans
        return complete, invalid

    def _is_termination_directory_unchanged(self, signature: tuple[str, ...], directory: Path) -> bool:
        census = self._authority_quiescence.get(signature)
        if census is None:
            return True
        version = census.directory_versions.pop(directory)
        return census.directory_version(directory) == version

    def _initial_page_converged(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        signature: tuple[str, ...],
        lane: str,
        page: object,
        *,
        should_require_quiescence: bool = False,
    ) -> bool:
        """Validate that a local discovery page reflects an applied lane outcome."""
        page_paths = getattr(page, "paths", ())
        for source_path in page_paths:
            if lane == "registrations":
                source = _read_running_publication_source(
                    source_path,
                    paths["root"],
                    binding,
                    "process_registration",
                    allow_unassigned_reservation=True,
                )
                if source is None:
                    return False
                manifest = paths["processes"] / f"{source.record['attempt_id']}.json"
                process = _read_terminal_process_record(manifest, paths["root"], binding)
                if process is None or process.parameters["attempt_id"] != source.record["attempt_id"]:
                    return False
            elif lane == "launch_intents":
                source = _read_launch_intent_source(source_path, paths["root"])
                if source is None:
                    return False
                manifest = paths["processes"] / f"{source.record['attempt_id']}.json"
                process = _read_terminal_process_record(manifest, paths["root"], binding)
                if process is None or process.parameters["attempt_id"] != source.record["attempt_id"]:
                    return False
            elif lane == "processes":
                process = _read_terminal_process_record(source_path, paths["root"], binding)
                if process is None:
                    return False
                record = process.record
                if record.get("observed_state") == "exited":
                    if not self._initial_reservation_settled(binding, process):
                        return False
                elif record.get("authority_state") not in {"healthy", "local_safe"}:
                    return False
                if should_require_quiescence and record.get("observed_state") == "exited":
                    decision_match = self._find_termination_decision_for_process(
                        binding,
                        signature,
                        paths,
                        source_path,
                        process,
                    )
                    if decision_match is not None:
                        decision_path_value, decision = decision_match
                        result = self._termination_convergence_for_decision(
                            binding,
                            paths,
                            signature,
                            decision_path_value,
                            process=process,
                            decision=decision,
                        )
                        if result is None or result.state != "converged":
                            return False
            elif lane == "observations":
                attempt_id = source_path.stem
                process = _read_terminal_process_record(
                    paths["processes"] / f"{attempt_id}.json",
                    paths["root"],
                    binding,
                )
                if process is None:
                    return False
                observation = _read_terminal_exit_observation(
                    source_path,
                    paths["root"],
                    process.parameters["task_id"],
                    attempt_id,
                )
                if (
                    observation is None
                    or process.record.get("observed_state") != "exited"
                    or process.record.get("observed_exit_code") != observation.exit_code
                    or not _is_valid_utc_timestamp(process.record.get("observed_exited_at"))
                    or not self._initial_reservation_settled(binding, process)
                ):
                    return False
            if should_require_quiescence:
                if lane == "registrations" and (
                    any(
                        source.parameters.get(name) != process.parameters.get(name)
                        for name in (
                            "task_id",
                            "attempt_id",
                            "attempt_number",
                            "fencing_token",
                            "reservation_id",
                            "process_identity",
                        )
                    )
                    or source.record.get("process_created_at") != process.record.get("process_created_at")
                ):
                    return False
                if lane == "launch_intents" and any(
                    source.record.get(name) != process.record.get(name)
                    for name in ("task_id", "attempt_id", "fencing_token", "wrapper_pid", "wrapper_start_time_ticks")
                    if source.record.get(name) is not None
                ):
                    return False
                if (
                    process.record.get("observed_state") != "exited"
                    or not _is_valid_utc_timestamp(process.record.get("observed_exited_at"))
                    or not self._initial_reservation_settled(binding, process)
                    or inspect_local_group_identity(process.record).state != "absent"
                ):
                    return False
                wrapper_was_registered = (
                    process.record.get("wrapper_pid") is not None
                    or process.record.get("wrapper_start_time_ticks") is not None
                )
                if wrapper_was_registered and inspect_wrapper_identity(process.record).state != "absent":
                    return False
        return True

    def _advance_initial_reconciliation(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> None:
        """Advance startup readiness and fresh activation-bound authority censuses."""
        eligible = list(bindings)
        signatures = {
            self._initial_reconciliation_signature(binding, registry_revision): binding for binding in eligible
        }
        for signature in self._initial_reconciliation_scans.keys() | self._authority_quiescence.keys():
            if signature not in signatures:
                self._discard_initial_scan(signature)
                self._discard_authority_census(signature)
                self._initial_observed.discard(signature)
                self._initial_reconciled.discard(signature)
        self._initial_observed.intersection_update(signatures)
        self._initial_reconciled.intersection_update(signatures)
        try:
            executor_epoch = self.controller.executor.status_view().get("executor_epoch")
        except _SUPERVISION_LANE_ERRORS:
            executor_epoch = None
        if not eligible:
            self._initial_reconciliation_offset = 0
            return
        start = self._initial_reconciliation_offset % len(eligible)
        rotated = eligible[start:] + eligible[:start]
        selected = rotated[: min(_INITIAL_RECONCILIATION_BINDINGS_PER_TURN, len(rotated))]
        self._initial_reconciliation_offset = (start + len(selected)) % len(eligible)
        for binding in selected:
            signature = self._initial_reconciliation_signature(binding, registry_revision)
            should_require_quiescence = signature in self._initial_reconciled
            census = self._authority_quiescence.get(signature)
            if census is not None:
                if census.executor_epoch != executor_epoch or not (
                    self.runtime.working_set.is_current_turn(census.turn)
                    or self.runtime.working_set.is_turn_observation_pending(census.turn)
                ):
                    self._discard_initial_scan(signature)
                    self._discard_authority_census(signature)
                    census = None
                elif census.is_quiescent:
                    try:
                        if census.has_unchanged_directories() and not self._initial_reconciliation_pending(
                            binding, registry_revision, signature
                        ):
                            continue
                    except _SUPERVISION_LANE_ERRORS:
                        pass
                    self._discard_initial_scan(signature)
                    self._discard_authority_census(signature)
                    census = None
            if should_require_quiescence and not isinstance(executor_epoch, str):
                continue
            scans = self._initial_reconciliation_scans.get(signature)
            if scans is None:
                if len(self._initial_reconciliation_scans) >= _MAX_INITIAL_RECONCILIATION_BINDINGS:
                    diagnostic_increment("scheduler.isolated.initial_reconciliation_capacity_exhausted")
                    continue
                paths = machine_project_paths(self.runtime.root, binding.project_id)
                scans = {
                    name: EvidenceScan(paths[name], directories=name == "termination_decisions")
                    for name in (
                        "registrations",
                        "launch_intents",
                        "processes",
                        "observations",
                        "termination_decisions",
                    )
                }
                self._initial_reconciliation_scans[signature] = scans
                self._initial_invalid_lanes[signature] = set()
                self._initial_complete_lanes[signature] = set()
                self._initial_deferred_lanes.discard(signature)
                self._initial_pending_termination_lanes.discard(signature)
                self._termination_convergence_findings.pop(signature, None)
                if should_require_quiescence:
                    census = _AuthorityQuiescenceCensus(
                        self.runtime.working_set.begin_turn(binding, "authority"), executor_epoch
                    )
                    self._authority_quiescence[signature] = census
                    try:
                        for scan in scans.values():
                            census.track_directory(scan.directory)
                    except _SUPERVISION_LANE_ERRORS:
                        self._discard_initial_scan(signature)
                        self._discard_authority_census(signature)
                        continue
            complete = True
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            for lane, scan in scans.items():
                if lane in self._initial_complete_lanes.get(signature, set()):
                    continue
                try:
                    if lane == "termination_decisions":
                        lane_complete, lane_invalid = self._advance_initial_termination_lane(
                            binding,
                            paths,
                            signature,
                            scan,
                        )
                        lane_converged = not lane_invalid
                        page_complete = lane_complete
                        if lane_complete and signature in self._initial_pending_termination_lanes:
                            scan.close()
                            scans[lane] = EvidenceScan(paths[lane], directories=True)
                            self._initial_termination_top_complete.discard(signature)
                            self._initial_pending_termination_lanes.discard(signature)
                            self._initial_deferred_lanes.discard(signature)
                            self._termination_convergence_findings.pop(signature, None)
                            page_complete = False
                    else:
                        page = scan.take(_INITIAL_RECONCILIATION_ENTRIES_PER_LANE)
                        lane_converged = self._initial_page_converged(
                            binding,
                            paths,
                            signature,
                            lane,
                            page,
                            should_require_quiescence=should_require_quiescence,
                        )
                        page_complete = page.is_complete
                except _SUPERVISION_LANE_ERRORS:
                    complete = False
                    diagnostic_increment("scheduler.isolated.initial_reconciliation_scan_failed")
                    continue
                termination_repair_pending = lane == "processes" and any(
                    result.state == "repairable"
                    for result in self._termination_convergence_findings.get(signature, {}).values()
                )
                if not lane_converged and not termination_repair_pending:
                    self._initial_invalid_lanes.setdefault(signature, set()).add(lane)
                if page_complete:
                    self._initial_complete_lanes.setdefault(signature, set()).add(lane)
                complete = complete and page_complete
            pending = self._initial_reconciliation_pending(binding, registry_revision, signature)
            invalid_lanes = self._initial_invalid_lanes.get(signature, set())
            if should_require_quiescence:
                try:
                    can_retire = bool(
                        complete and not invalid_lanes and not pending and census and census.has_unchanged_directories()
                    )
                except _SUPERVISION_LANE_ERRORS:
                    can_retire = False
                if can_retire:
                    census.is_quiescent = True
                    if self.runtime.working_set.is_current_turn(census.turn):
                        self.runtime.working_set.acknowledge(census.turn, quiescent=True)
                elif complete:
                    if census is not None:
                        self.runtime.working_set.acknowledge(census.turn, quiescent=False)
                    self._discard_authority_census(signature)
            elif complete and not invalid_lanes and not pending:
                if signature in self._initial_observed:
                    self._initial_reconciled.add(signature)
                else:
                    self._initial_observed.add(signature)
                    self._initial_reconciled.discard(signature)
            else:
                self._initial_reconciled.discard(signature)
            if complete:
                self._discard_initial_scan(signature)

    def initially_reconciled(
        self,
        binding: ProjectBinding,
        registry_revision: int,
    ) -> bool:
        """Report the monotonic startup barrier for one exact registry generation."""
        if type(registry_revision) is not int or registry_revision < 0:
            return False
        signature = self._initial_reconciliation_signature(binding, registry_revision)
        return signature in self._initial_reconciled

    def read_termination_convergence_diagnostics(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        limit: int = 64,
    ) -> tuple[dict[str, object], ...]:
        """Return bounded current termination findings for scheduler diagnostics."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if type(limit) is not int or not 1 <= limit <= _MAX_TERMINATION_CANDIDATES:
            raise ValueError("limit must be a positive bounded integer.")
        binding_by_signature = {
            self._initial_reconciliation_signature(binding, registry_revision): binding for binding in bindings
        }
        findings: list[dict[str, object]] = []
        for signature in sorted(binding_by_signature):
            binding = binding_by_signature[signature]
            for attempt_id, result in sorted(self._termination_convergence_findings.get(signature, {}).items()):
                if result.state == "converged":
                    continue
                findings.append(
                    {
                        "project_id": binding.project_id,
                        "registration_generation": binding.registration_generation,
                        "attempt_id": attempt_id,
                        "state": result.state,
                        "reason": result.reason,
                        "manifest_exit_code": result.manifest_exit_code,
                        "process_absent": result.process_absent,
                        "reservation_settled": result.reservation_settled,
                        "registry_revision": registry_revision,
                    }
                )
                if len(findings) >= limit:
                    return tuple(findings)
        return tuple(findings)

    def termination_convergence_diagnostics(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        limit: int = 64,
    ) -> tuple[dict[str, object], ...]:
        """Compatibility alias for the bounded convergence finding reader."""
        return self.read_termination_convergence_diagnostics(bindings, registry_revision, limit=limit)

    def advance_all(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance every attempt-supervision lane in one admission turn.

        Each lane is independently bounded. A transient failure in one lane is
        recorded locally and does not prevent the other lanes or readiness sweep
        from progressing.
        """
        results: dict[str, Mapping[str, Any]] = {}
        lanes: tuple[tuple[str, Callable[[], Mapping[str, Any]]], ...] = (
            ("reservation_recovery", lambda: self._reservation_recovery.advance(bindings)),
            (
                "authority_renewals",
                lambda: advance_authority_renewals(self.runtime, self.controller, bindings, registry_revision) or {},
            ),
            (
                "terminal_completions",
                lambda: self.advance_terminal_completions(bindings, registry_revision),
            ),
            (
                "running_publications",
                lambda: self.advance_running_publications(bindings, registry_revision),
            ),
            (
                "terminations",
                lambda: self.advance_terminations(bindings, registry_revision),
            ),
            (
                "orphan_recoveries",
                lambda: self.advance_orphan_recoveries(bindings, registry_revision),
            ),
        )
        for name, advance in lanes:
            try:
                value = advance()
            except _SUPERVISION_LANE_ERRORS:
                diagnostic_increment(f"scheduler.isolated.{name}_dispatch_failed")
                continue
            if isinstance(value, Mapping):
                results[name] = value
        self._advance_initial_reconciliation(bindings, registry_revision)
        return results

    def advance_running_publications(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        pending_attempt_ids: Collection[str] = (),
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one bounded local evidence opportunity per selected binding."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if isinstance(pending_attempt_ids, (str, bytes)) or not isinstance(pending_attempt_ids, Collection):
            raise ValueError("pending_attempt_ids must be a finite collection of identifiers.")
        if len(pending_attempt_ids) > _MAX_PENDING_RUNNING_PUBLICATIONS:
            raise ValueError("pending_attempt_ids cannot contain more than 256 identifiers.")
        pending: set[str] = set()
        for attempt_id in pending_attempt_ids:
            pending.add(validate_identifier(attempt_id, "pending attempt_id"))

        eligible = list(bindings)
        signatures = [_authority_binding_signature(binding, registry_revision) for binding in eligible]
        current_signatures = set(signatures)
        for signature in tuple(self._scans):
            scans = self._scans[signature]
            if signature not in current_signatures or scans.complete:
                self._scans.pop(signature).close()
        self._running_publication_intents = {
            signature: parameters
            for signature, parameters in self._running_publication_intents.items()
            if signature in current_signatures
        }

        if eligible:
            start = self._binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected_count = min(_MAX_RUNNING_PUBLICATION_BINDINGS, len(rotated))
            selected = rotated[:selected_count]
            self._binding_offset = (start + selected_count) % len(eligible)
        else:
            selected = []
            self._binding_offset = 0

        publications: dict[str, Mapping[str, Any]] = {}
        publication_bindings: list[ProjectBinding] = []
        reconciliation_bindings: list[ProjectBinding] = []
        unresolved_running_requests = tuple(
            request
            for request in self.controller.executor.unresolved_requests()
            if request.operation_kind == "authority_running_publish"
        )
        for binding in selected:
            signature = _authority_binding_signature(binding, registry_revision)
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            scans = self._scans.get(signature)
            retained = self._running_publication_intents.get(signature)
            if retained is not None:
                try:
                    source = _read_running_publication_source(
                        paths["registrations"] / f"{retained['attempt_id']}.json",
                        paths["root"],
                        binding,
                        "process_registration",
                    )
                    process = _read_terminal_process_record(
                        paths["processes"] / f"{retained['attempt_id']}.json", paths["root"], binding
                    )
                except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                    continue
                if (
                    source is None
                    or source.parameters != retained
                    or process is None
                    or any(
                        process.parameters[field] != retained[field]
                        for field in (
                            "task_id",
                            "attempt_id",
                            "attempt_number",
                            "fencing_token",
                            "reservation_id",
                            "process_identity",
                        )
                    )
                    or process.record.get("observed_state") == "exited"
                ):
                    self._running_publication_intents.pop(signature, None)
                else:
                    if retained["attempt_id"] not in pending:
                        publications[binding.project_id] = retained
                        publication_bindings.append(binding)
                    continue
            if scans is None:
                if len(self._scans) >= _MAX_TERMINAL_STATES:
                    diagnostic_increment("scheduler.isolated.running_scan_capacity_exhausted")
                    continue
                scans = _RunningPublicationScans(
                    registrations=EvidenceScan(paths["registrations"]),
                    launch_intents=EvidenceScan(paths["launch_intents"]),
                )
                self._scans[signature] = scans

            if scans.registrations_complete:
                lane = "launch_intent"
            elif scans.launch_intents_complete:
                lane = "registration"
            else:
                lane = scans.next_lane
                scans.next_lane = "launch_intent" if lane == "registration" else "registration"
            scan = scans.registrations if lane == "registration" else scans.launch_intents
            try:
                page = scan.take(1)
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
            if lane == "registration":
                scans.registrations_complete = page.is_complete
            else:
                scans.launch_intents_complete = page.is_complete
            if not page.paths:
                if scans.complete:
                    self._scans.pop(signature, scans).close()
                continue

            source_path = page.paths[0]
            try:
                source = (
                    _read_running_publication_source(
                        source_path,
                        paths["root"],
                        binding,
                        "process_registration",
                    )
                    if lane == "registration"
                    else _read_launch_intent_source(source_path, paths["root"])
                )
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
            if source is None or source.record["attempt_id"] in pending:
                continue

            try:
                if lane == "registration":
                    with evidence_write_guard(paths["root"], source.record["attempt_id"]) as acquired:
                        if not acquired:
                            continue
                        current_source = _read_running_publication_source(
                            source_path,
                            paths["root"],
                            binding,
                            "process_registration",
                        )
                        if current_source is None or current_source.identity != source.identity:
                            continue
                        terminal_process = _read_terminal_process_record(
                            paths["processes"] / f"{source.record['attempt_id']}.json",
                            paths["root"],
                            binding,
                        )
                        if terminal_process is not None and terminal_process.record.get("observed_state") == "exited":
                            if any(
                                _terminal_matches_running_request(request, binding, terminal_process)
                                for request in unresolved_running_requests
                            ):
                                reconciliation_bindings.append(binding)
                            continue
                        parameters = _ensure_running_manifest(paths, current_source)
                else:
                    parameters, terminal = self._advance_launch_intent(
                        binding,
                        paths,
                        source_path,
                        source,
                        pending,
                    )
                    if terminal is not None and any(
                        _terminal_matches_running_request(request, binding, terminal)
                        for request in unresolved_running_requests
                    ):
                        reconciliation_bindings.append(binding)
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue

            if parameters is not None:
                if len(self._running_publication_intents) >= _MAX_RUNNING_PUBLICATION_BINDINGS:
                    continue
                self._running_publication_intents[signature] = parameters
                publications[binding.project_id] = parameters
                publication_bindings.append(binding)
            if scans.complete:
                self._scans.pop(signature, scans).close()

        if not publications and not reconciliation_bindings:
            return {}
        publication_project_ids = {binding.project_id for binding in publication_bindings}
        reconciliation_bindings = [
            binding for binding in reconciliation_bindings if binding.project_id not in publication_project_ids
        ]
        results = self.controller.advance_authority_running_publications(
            [*publication_bindings, *reconciliation_bindings],
            registry_revision,
            publications,
        )
        for binding in publication_bindings:
            if results.get(binding.project_id, {}).get("outcome") == "processed":
                self._running_publication_intents.pop(_authority_binding_signature(binding, registry_revision), None)
        return results

    def _advance_launch_intent(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        source_path: Path,
        source: _RunningPublicationSource,
        pending: set[str],
    ) -> tuple[dict[str, Any] | None, _TerminalProcessRecord | None]:
        attempt_id = source.record["attempt_id"]
        if attempt_id in pending:
            return None, None
        with evidence_write_guard(paths["root"], attempt_id) as acquired:
            if not acquired:
                return None, None
            current_intent = _read_launch_intent_source(source_path, paths["root"])
            if current_intent is None or current_intent.identity != source.identity:
                return None, None

            manifest = paths["processes"] / f"{attempt_id}.json"
            terminal_process = _read_terminal_process_record(manifest, paths["root"], binding)
            if terminal_process is not None and terminal_process.record.get("observed_state") == "exited":
                return None, terminal_process

            registration_path = paths["registrations"] / f"{attempt_id}.json"
            if is_path_present(registration_path):
                registration = _read_running_publication_source(
                    registration_path,
                    paths["root"],
                    binding,
                    "process_registration",
                )
                if registration is None or not _registration_matches_intent(registration, current_intent):
                    return None, None
                return _ensure_running_manifest(paths, registration), None

            if is_path_present(manifest):
                return None, None
            try:
                wrapper_state = inspect_wrapper_identity(current_intent.record).state
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                wrapper_state = "unknown"
            if wrapper_state == "alive":
                return None, None

            process = dict(current_intent.record)
            process.update(
                {
                    "observed_state": "launch_unverifiable",
                    "supervisor": "agent",
                    "authority_state": "isolated",
                    "created_by": "agent",
                }
            )
            atomic_replace(manifest, {"process": process})
            return None, None

    def advance_terminal_completions(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Publish exact exits and reconcile only proven local completion."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")

        eligible = list(bindings)
        signatures = [_authority_binding_signature(binding, registry_revision) for binding in eligible]
        current_signatures = set(signatures)
        binding_by_project = {binding.project_id: binding for binding in eligible}

        if eligible:
            start = self._terminal_binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected_count = min(_MAX_TERMINAL_BINDINGS, len(rotated))
            selected = rotated[:selected_count]
            self._terminal_binding_offset = (start + selected_count) % len(eligible)
        else:
            selected = []
            self._terminal_binding_offset = 0
        selected_projects = {binding.project_id for binding in selected}

        for signature in tuple(self._terminal_states):
            state = self._terminal_states[signature]
            has_candidate = any(
                candidate.binding_signature == signature for candidate in self._terminal_candidates.values()
            )
            if signature not in current_signatures or (state.process_scan_complete and not has_candidate):
                self._terminal_states.pop(signature).close()
        for project_id, candidate in tuple(self._terminal_candidates.items()):
            binding = binding_by_project.get(project_id)
            if (
                binding is None
                or _authority_binding_signature(binding, registry_revision) != candidate.binding_signature
            ):
                self._terminal_candidates.pop(project_id, None)

        examined_projects: set[str] = set()
        for project_id, candidate in tuple(self._terminal_candidates.items()):
            if project_id not in selected_projects:
                continue
            binding = binding_by_project[project_id]
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            examined_projects.add(project_id)
            try:
                process = _read_terminal_process_record(candidate.process_path, paths["root"], binding)
                observation = _read_terminal_exit_observation(
                    candidate.observation_path,
                    paths["root"],
                    candidate.parameters["task_id"],
                    candidate.parameters["attempt_id"],
                )
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                self._terminal_candidates.pop(project_id, None)
                continue
            state = self._terminal_states.get(candidate.binding_signature)
            if process is not None and observation is None and state is not None:
                _record_terminal_observation_problem(state, process, candidate.observation_path)
            if (
                process is None
                or observation is None
                or process.parameters != {key: value for key, value in candidate.parameters.items() if key != "mode"}
                or process.identity not in {candidate.process_identity, candidate.post_update_process_identity}
                or observation.identity != candidate.observation_identity
                or observation.exit_code != candidate.exit_code
            ):
                self._terminal_candidates.pop(project_id, None)
                continue
            candidate.binding = binding

        for binding in selected:
            project_id = binding.project_id
            if project_id in examined_projects or project_id in self._terminal_candidates:
                continue
            signature = _authority_binding_signature(binding, registry_revision)
            paths = machine_project_paths(self.runtime.root, project_id)
            state = self._terminal_states.get(signature)
            if state is None:
                if len(self._terminal_states) >= _MAX_TERMINAL_STATES:
                    diagnostic_increment("scheduler.isolated.terminal_scan_capacity_exhausted")
                    continue
                state = _TerminalBindingState(
                    process_scan=EvidenceScan(paths["processes"]),
                    reconciler=LocalExitReconciler(
                        paths["root"],
                        reservation_runtime_root=self.runtime.root,
                        project_id=project_id,
                    ),
                )
                self._terminal_states[signature] = state

            examined_projects.add(project_id)
            try:
                page = state.process_scan.take(1)
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
            state.process_scan_complete = page.is_complete
            if not page.paths:
                continue
            process_path = page.paths[0]
            try:
                process = _read_terminal_process_record(process_path, paths["root"], binding)
                if process is None:
                    continue
                observation_path = paths["observations"] / f"{process.parameters['attempt_id']}.json"
                observation = _read_terminal_exit_observation(
                    observation_path,
                    paths["root"],
                    process.parameters["task_id"],
                    process.parameters["attempt_id"],
                )
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
            if observation is None:
                _record_terminal_observation_problem(state, process, observation_path)
                continue
            if len(self._terminal_candidates) >= _MAX_CACHED_TERMINAL_CANDIDATES:
                continue
            parameters = dict(process.parameters)
            parameters["mode"] = "active"
            self._terminal_candidates[project_id] = _TerminalCandidate(
                binding=binding,
                binding_signature=signature,
                process_path=process_path,
                observation_path=observation_path,
                parameters=parameters,
                exit_code=observation.exit_code,
                process_identity=process.identity,
                observation_identity=observation.identity,
            )
            state.process_scan_complete = page.is_complete

        for signature, state in tuple(self._terminal_states.items()):
            if state.process_scan_complete and not any(
                candidate.binding_signature == signature for candidate in self._terminal_candidates.values()
            ):
                self._terminal_states.pop(signature).close()

        candidates = tuple(
            (project_id, candidate)
            for project_id, candidate in self._terminal_candidates.items()
            if project_id in selected_projects
        )
        # Capacity is machine-local truth. Once exact local process and exit
        # evidence agree, release it without waiting for shared terminal
        # publication; the retained evidence keeps publication replayable.
        for _project_id, candidate in candidates:
            state = self._terminal_states.get(candidate.binding_signature)
            if state is None:
                continue
            try:
                state.reconciler.reconcile_observation(candidate.observation_path, bounded=True)
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
        if candidates:
            try:
                unresolved = self.controller.executor.unresolved_requests()
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                unresolved = ()
            candidates_by_project = dict(candidates)
            for request in unresolved:
                candidate = candidates_by_project.get(request.project_id)
                if candidate is None or candidate.pending_transition is not None:
                    continue
                recovered = _recover_terminal_transition(request, candidate, registry_revision)
                if recovered is not None:
                    candidate.pending_transition = recovered

        observation_results: Mapping[str, Mapping[str, Any]] = {}
        if candidates:
            observation_bindings = [candidate.binding for _, candidate in candidates]
            attempts = {project_id: candidate.parameters for project_id, candidate in candidates}
            observation_results = self.controller.advance_authority_terminal_observations(
                observation_bindings,
                registry_revision,
                attempts,
            )

        for project_id, candidate in candidates:
            evidence = observation_results.get(project_id)
            if (
                candidate.pending_transition is None
                and isinstance(evidence, Mapping)
                and evidence.get("outcome") == "current"
            ):
                candidate.pending_transition = _build_terminal_transition(candidate, evidence)

        transition_requests = {
            project_id: candidate.pending_transition
            for project_id, candidate in candidates
            if candidate.pending_transition is not None
        }
        publication_results: Mapping[str, Mapping[str, Any]] = {}
        if transition_requests:
            publication_candidates = [
                candidate for project_id, candidate in candidates if project_id in transition_requests
            ]
            publication_bindings = [candidate.binding for candidate in publication_candidates]
            publication_results = self.controller.advance_authority_terminal_publications(
                publication_bindings,
                registry_revision,
                transition_requests,
            )

        for project_id, evidence in publication_results.items():
            candidate = self._terminal_candidates.get(project_id)
            if (
                candidate is not None
                and project_id in transition_requests
                and evidence.get("outcome") in {"committed", "already_committed", "stale", "unavailable"}
            ):
                candidate.pending_transition = None

        proven_projects: set[str] = set()
        for project_id, candidate in candidates:
            observation_evidence = observation_results.get(project_id)
            publication_evidence = publication_results.get(project_id)
            transition = transition_requests.get(project_id)
            publication_proof = _is_terminal_publication_proof(
                publication_evidence,
                candidate,
                transition,
            )
            observation_proof = _is_terminal_observation_proof(observation_evidence, candidate)
            proof = publication_proof if publication_proof is not None else observation_proof
            if proof is None:
                continue
            proven_projects.add(project_id)
            state = self._terminal_states.get(candidate.binding_signature)
            if state is None:
                continue
            paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
            try:
                local_outcome = _apply_terminal_local_effects(
                    candidate,
                    proof,
                    paths,
                    state.reconciler,
                )
            except _RUNNING_PUBLICATION_SOURCE_ERRORS:
                continue
            if local_outcome in {"applied", "invalid"}:
                self._terminal_candidates.pop(project_id, None)

        for project_id, candidate in candidates:
            if project_id in proven_projects:
                continue
            observation_evidence = observation_results.get(project_id)
            publication_evidence = publication_results.get(project_id)
            if not any(
                isinstance(evidence, Mapping) and evidence.get("outcome") == "stale"
                for evidence in (observation_evidence, publication_evidence)
            ):
                continue
            transition = transition_requests.get(project_id)
            can_fallback_to_detached = _is_active_terminal_observation_fallback(
                observation_evidence,
                candidate,
            ) or _is_active_terminal_publication_fallback(
                publication_evidence,
                candidate,
                transition,
            )
            if can_fallback_to_detached:
                candidate.parameters["mode"] = "detached_orphan"
                candidate.pending_transition = None
            else:
                state = self._terminal_states.get(candidate.binding_signature)
                if state is not None:
                    state.reconciler.record_diagnostic(
                        {"attempt_id": candidate.parameters["attempt_id"]},
                        "attempt_authority_superseded",
                    )
                self._terminal_candidates.pop(project_id, None)

        results: dict[str, Mapping[str, Any]] = {}
        for project_id, _candidate in candidates:
            if project_id in publication_results:
                results[project_id] = publication_results[project_id]
            elif project_id in observation_results:
                results[project_id] = observation_results[project_id]
        return results

    @staticmethod
    def _termination_key(candidate: _TerminationCandidate) -> tuple[str, str]:
        return candidate.binding.project_id, candidate.parameters["attempt_id"]

    def _new_termination_candidate(
        self,
        binding: ProjectBinding,
        signature: tuple[str, ...],
        process_path: Path,
        process: _TerminalProcessRecord,
    ) -> _TerminationCandidate:
        parameters = dict(process.parameters)
        parameters["mode"] = "active"
        return _TerminationCandidate(
            binding=binding,
            binding_signature=signature,
            process_path=process_path,
            parameters=parameters,
            process_identity=process.identity,
        )

    def _find_termination_decision_for_process(
        self,
        binding: ProjectBinding,
        signature: tuple[str, ...],
        paths: Mapping[str, Path],
        process_path: Path,
        process: _TerminalProcessRecord,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Find one exact durable termination decision for a retained manifest."""
        attempt_id = process.parameters["attempt_id"]
        directory = paths["termination_decisions"] / attempt_id
        scan = EvidenceScan(directory)
        try:
            page = scan.take(_INITIAL_RECONCILIATION_ENTRIES_PER_LANE)
        finally:
            scan.close()
        for decision_path_value in page.paths:
            try:
                envelope = read_json_limited(
                    decision_path_value,
                    max_bytes=65_536,
                    record_type="termination_decision",
                )
            except FileNotFoundError:
                continue
            except OSError:
                raise
            except (TypeError, ValueError):
                continue
            if set(envelope) != {"termination_decision"} or not isinstance(envelope.get("termination_decision"), dict):
                continue
            decision = envelope["termination_decision"]
            candidate = self._new_termination_candidate(binding, signature, process_path, process)
            candidate.trigger = decision.get("authority_outcome")
            candidate.decision_id = decision_path_value.stem
            candidate.decision = decision
            if (
                candidate.trigger in _TERMINATION_TRIGGERS
                and decision_path_value.stem == _termination_decision_id(candidate, candidate.trigger)
                and _termination_decision_matches_candidate(decision, candidate)
            ):
                return decision_path_value, decision
        return None

    @staticmethod
    def _termination_process_matches(
        process: _TerminalProcessRecord | None,
        candidate: _TerminationCandidate,
    ) -> bool:
        return (
            process is not None
            and process.parameters == {key: value for key, value in candidate.parameters.items() if key != "mode"}
            and process.identity
            in {
                candidate.process_identity,
                candidate.post_update_process_identity,
            }
        )

    @staticmethod
    def _termination_request_matches_candidate(
        request: object,
        candidate: _TerminationCandidate,
        registry_revision: int,
        operation_kind: str,
    ) -> Mapping[str, Any] | None:
        if (
            getattr(request, "operation_kind", None) != operation_kind
            or getattr(request, "project_id", None) != candidate.binding.project_id
            or getattr(request, "canonical_shared_root", None) != str(candidate.binding.shared_root)
            or getattr(request, "registration_generation", None) != candidate.binding.registration_generation
            or getattr(request, "registry_revision", None) != registry_revision
        ):
            return None
        parameters = getattr(request, "parameters", None)
        source_revisions = getattr(request, "source_revisions", None)
        if not isinstance(parameters, Mapping) or not _has_exact_terminal_revisions(source_revisions):
            return None
        candidate_parameters = candidate.parameters
        if not all(
            parameters.get(field) == candidate_parameters.get(field)
            for field in (
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "reservation_id",
                "process_identity",
            )
        ):
            return None
        if operation_kind == "authority_termination_commit":
            trigger = parameters.get("authority_outcome")
            if (
                trigger not in _TERMINATION_TRIGGERS
                or parameters.get("reason") != trigger
                or not isinstance(parameters.get("decision_id"), str)
                or type(parameters.get("decision_token")) is not int
                or parameters.get("decision_token") != candidate_parameters.get("fencing_token")
            ):
                return None
        else:
            terminal_target = (
                parameters.get("phase"),
                parameters.get("reason"),
                parameters.get("termination_result"),
            )
            if (
                parameters.get("mode") != "active"
                or terminal_target
                not in {
                    ("cancelled", "terminated_by_agent", "terminated"),
                    ("cancelled", "termination_process_already_exited", "already_exited"),
                    ("failed", "process_exited_without_status", None),
                }
                or parameters.get("exit_code") is not None
                or not isinstance(parameters.get("transition_digest"), str)
            ):
                return None
        recovered = {key: value for key, value in parameters.items() if key != "machine_name"}
        return {**recovered, "source_revisions": dict(source_revisions)}

    @staticmethod
    def _termination_transition_matches_candidate(
        transition: Mapping[str, Any] | None,
        candidate: _TerminationCandidate,
    ) -> bool:
        if not isinstance(transition, Mapping):
            return False
        expected = _build_termination_transition(
            candidate,
            {
                "outcome": "current",
                "machine_name": candidate.binding.machine_name,
                "task_phase": "running",
                "attempt_phase": "running",
                "execution_machine_name": candidate.binding.machine_name,
                "reservation_machine_name": candidate.binding.machine_name,
                "termination_result": None,
                "cancel_requested": True,
                "source_revisions": transition.get("source_revisions"),
                **candidate.parameters,
                "authority_granted": False,
                "local_effects": [],
            },
        )
        return expected is not None and dict(transition) == expected

    def _recover_termination_requests(
        self,
        candidates: Mapping[tuple[str, str], _TerminationCandidate],
        bindings: Mapping[str, ProjectBinding],
        registry_revision: int,
    ) -> tuple[object, ...]:
        """Pin and recover bounded executor requests after coordinator restart."""
        try:
            unresolved = self.controller.executor.unresolved_requests()
        except _TERMINATION_SOURCE_ERRORS:
            return ()
        recovered_requests: list[object] = []
        for request in unresolved:
            if len(recovered_requests) >= _MAX_TERMINATION_CANDIDATES:
                break
            operation_kind = getattr(request, "operation_kind", None)
            if operation_kind not in {"authority_termination_commit", "authority_terminal_publish"}:
                continue
            binding = bindings.get(getattr(request, "project_id", None))
            if binding is None:
                continue
            request_parameters = getattr(request, "parameters", None)
            if not isinstance(request_parameters, Mapping):
                continue
            attempt_id = request_parameters.get("attempt_id")
            if not isinstance(attempt_id, str):
                continue
            key = (binding.project_id, attempt_id)
            candidate = candidates.get(key)
            if candidate is None:
                paths = machine_project_paths(self.runtime.root, binding.project_id)
                process_path = paths["processes"] / f"{attempt_id}.json"
                try:
                    process = _read_terminal_process_record(process_path, paths["root"], binding)
                except _TERMINATION_SOURCE_ERRORS:
                    continue
                if process is None:
                    continue
                signature = _authority_binding_signature(binding, registry_revision)
                candidate = self._new_termination_candidate(binding, signature, process_path, process)
                if len(self._termination_candidates) >= _MAX_TERMINATION_CANDIDATES:
                    continue
                self._termination_candidates[key] = candidate
                candidates[key] = candidate
            recovered = self._termination_request_matches_candidate(
                request,
                candidate,
                registry_revision,
                operation_kind,
            )
            if recovered is None:
                continue
            recovered_requests.append(request)
            if operation_kind == "authority_termination_commit":
                candidate.trigger = recovered["authority_outcome"]
                candidate.decision_id = recovered["decision_id"]
                candidate.source_revisions = dict(recovered["source_revisions"])
                candidate.pending_commit = dict(recovered)
            else:
                candidate.pending_publication = dict(recovered)
        return tuple(recovered_requests)

    def _prepare_termination_decision(
        self,
        candidate: _TerminationCandidate,
        paths: Mapping[str, Path],
    ) -> str:
        """Create or validate the deterministic local decision under both local locks."""
        if candidate.trigger not in _TERMINATION_TRIGGERS:
            return "invalid"
        candidate.decision_id = _termination_decision_id(candidate, candidate.trigger)
        cfg = _termination_local_config(self.runtime, candidate.binding)
        attempt_id = candidate.parameters["attempt_id"]
        try:
            with evidence_write_guard(paths["root"], attempt_id) as evidence_acquired:
                if not evidence_acquired:
                    return "deferred"
                with attempt_control_lock(cfg, attempt_id, blocking=False) as lock_acquired:
                    if not lock_acquired:
                        return "deferred"
                    process = _read_terminal_process_record(candidate.process_path, paths["root"], candidate.binding)
                    if not self._termination_process_matches(process, candidate):
                        return "invalid"
                    decision = create_decision(
                        cfg,
                        task_id=candidate.parameters["task_id"],
                        attempt_id=attempt_id,
                        fencing_token=candidate.parameters["fencing_token"],
                        process=dict(process.record),
                        authority_outcome=candidate.trigger,
                        reason=candidate.trigger,
                        decision_id=candidate.decision_id,
                    )
                    if not _termination_decision_matches_candidate(decision, candidate):
                        return "invalid"
                    candidate.decision = dict(decision)
                    if candidate.source_revisions is None:
                        return "invalid"
                    return "applied"
        except _TERMINATION_SOURCE_ERRORS:
            return "deferred"

    def _advance_termination_signals(
        self,
        candidate: _TerminationCandidate,
        paths: Mapping[str, Path],
    ) -> str:
        """Apply one local shared-commit update and at most one signal step."""
        if candidate.decision_id is None or candidate.trigger not in _TERMINATION_TRIGGERS:
            return "invalid"
        cfg = _termination_local_config(self.runtime, candidate.binding)
        attempt_id = candidate.parameters["attempt_id"]
        try:
            with evidence_write_guard(paths["root"], attempt_id) as evidence_acquired:
                if not evidence_acquired:
                    return "deferred"
                with attempt_control_lock(cfg, attempt_id, blocking=False) as lock_acquired:
                    if not lock_acquired:
                        return "deferred"
                    process = _read_terminal_process_record(candidate.process_path, paths["root"], candidate.binding)
                    if not self._termination_process_matches(process, candidate):
                        return "invalid"
                    decision = _read_termination_decision(cfg, attempt_id, candidate.decision_id)
                    if not _termination_decision_matches_candidate(decision, candidate):
                        return "invalid"
                    candidate.decision = dict(decision)
                    if decision.get("shared_commitment") != "committed":
                        if not _termination_commit_proof(candidate.commit_evidence, candidate):
                            return "deferred"
                        decision = update_decision(
                            cfg,
                            attempt_id,
                            candidate.decision_id,
                            shared_commitment="committed",
                        )
                        if not _termination_decision_matches_candidate(decision, candidate):
                            return "invalid"
                        candidate.decision = dict(decision)
                    if decision.get("state") == "pending":
                        decision = commit_signal(cfg, attempt_id, candidate.decision_id)
                        if not _termination_decision_matches_candidate(decision, candidate):
                            return "invalid"
                        candidate.decision = dict(decision)
                    if decision.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent"}:
                        decision, deadline = advance_signals(
                            cfg,
                            attempt_id,
                            candidate.decision_id,
                            sigterm_deadline=candidate.sigterm_deadline,
                            grace_seconds=_TERMINATION_GRACE_SECONDS,
                        )
                        if not _termination_decision_matches_candidate(decision, candidate):
                            return "invalid"
                        candidate.decision = dict(decision)
                        candidate.sigterm_deadline = deadline
                    return "applied"
        except _TERMINATION_SOURCE_ERRORS:
            return "deferred"

    def _apply_termination_local_effects(
        self,
        candidate: _TerminationCandidate,
        paths: Mapping[str, Path],
        state: _TerminationBindingState,
    ) -> str:
        """Converge the exact manifest and owned reservation after a signal proof."""
        if candidate.decision_id is None:
            return "invalid"
        cfg = _termination_local_config(self.runtime, candidate.binding)
        attempt_id = candidate.parameters["attempt_id"]
        try:
            with evidence_write_guard(paths["root"], attempt_id) as evidence_acquired:
                if not evidence_acquired:
                    return "deferred"
                with attempt_control_lock(cfg, attempt_id, blocking=False) as lock_acquired:
                    if not lock_acquired:
                        return "deferred"
                    process = _read_terminal_process_record(candidate.process_path, paths["root"], candidate.binding)
                    decision = _read_termination_decision(cfg, attempt_id, candidate.decision_id)
                    if (
                        not self._termination_process_matches(process, candidate)
                        or not _termination_decision_matches_candidate(decision, candidate)
                        or decision.get("state") != "confirmed"
                        or decision.get("shared_commitment") != "committed"
                        or not isinstance(decision.get("signal_attempts"), list)
                        or decision.get("confirmation") not in {"identity_absent", "process_absent"}
                    ):
                        return "invalid"
                    convergence = self._termination_convergence_for_decision(
                        candidate.binding,
                        paths,
                        candidate.binding_signature,
                        paths["termination_decisions"] / attempt_id / f"{candidate.decision_id}.json",
                        process=process,
                        decision=decision,
                    )
                    if convergence is None:
                        return "deferred"
                    candidate.convergence = convergence
                    if convergence.state == "invalid":
                        return "invalid"
                    if convergence.state == "converged":
                        return "applied"
                    wrapper = inspect_wrapper_identity(process.record)
                    group = inspect_local_group_identity(process.record)
                    wrapper_was_registered = (
                        process.record.get("wrapper_pid") is not None
                        or process.record.get("wrapper_start_time_ticks") is not None
                    )
                    if group.state != "absent" or (wrapper_was_registered and wrapper.state != "absent"):
                        return "deferred"
                    updated = dict(process.record)
                    exit_code = process.record.get("observed_exit_code")
                    if type(exit_code) is not int:
                        observation = _read_terminal_exit_observation(
                            paths["observations"] / f"{attempt_id}.json",
                            paths["root"],
                            candidate.parameters["task_id"],
                            attempt_id,
                        )
                        exit_code = None if observation is None else observation.exit_code
                    if not (
                        process.record.get("observed_state") == "exited"
                        and (exit_code is None or type(exit_code) is int)
                        and _is_valid_utc_timestamp(process.record.get("observed_exited_at"))
                    ):
                        updated.update(
                            {
                                "observed_state": "exited",
                                "observed_exit_code": exit_code,
                                "observed_exited_at": utc_now(),
                            }
                        )
                        atomic_replace(candidate.process_path, {"process": updated})
                    candidate.post_update_process = updated
                    candidate.post_update_process_identity = _canonical_terminal_record(updated)
                    if state.reconciler.release_confirmed_termination(updated, dict(decision)):
                        return "applied"
                    if updated.get("reservation_id") is None:
                        return "applied"
                    return "deferred"
        except _TERMINATION_SOURCE_ERRORS:
            return "deferred"

    @staticmethod
    def _orphan_binding_signature(binding: ProjectBinding) -> tuple[str, ...]:
        """Return the stable binding identity persisted by shared recovery."""
        return (
            binding.project_id,
            str(binding.shared_root),
            binding.machine_name,
            binding.registration_generation,
            binding.runtime_instance_id,
            binding.runtime_root,
        )

    def _recovery_reservation(
        self,
        reservation_id: str,
        *,
        allowed_states: frozenset[str] = frozenset({"active"}),
    ) -> tuple[str, ReservationIdentity] | None:
        paths = local_paths(self.runtime.root)
        found: list[tuple[str, ReservationIdentity]] = []
        lanes = (
            ("active", "active"),
            ("provisional", "provisional"),
            ("released", "released"),
            ("cpu_active", "active"),
            ("cpu_provisional", "provisional"),
            ("cpu_released", "released"),
        )
        for name, expected_state in lanes:
            try:
                envelope = read_json_limited(
                    paths[name] / f"{reservation_id}.json",
                    max_bytes=65_536,
                    record_type="reservation",
                )
            except FileNotFoundError:
                continue
            reservation = envelope.get("reservation") if isinstance(envelope, dict) else None
            if not isinstance(reservation, dict) or reservation.get("state") != expected_state:
                return None
            found.append((expected_state, ReservationIdentity.from_record(reservation)))
        return found[0] if len(found) == 1 and found[0][0] in allowed_states else None

    @staticmethod
    def _orphan_recovery_proof(
        evidence: Mapping[str, Any] | None,
        entry: Mapping[str, Any],
        binding_signature: tuple[str, ...],
    ) -> bool:
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("outcome") not in {"recovered", "already_recovered"}
            or evidence.get("reason") is not None
            or not isinstance(evidence.get("binding_signature"), (list, tuple))
            or tuple(evidence["binding_signature"]) != binding_signature
            or evidence.get("authority_granted") is not False
            or evidence.get("local_effects") not in ([], ())
            or not _has_exact_terminal_revisions(evidence.get("source_revisions"))
            or not _has_exact_terminal_revisions(evidence.get("committed_revisions"))
            or not _authority_evidence_matches_intent(evidence, entry)
        ):
            return False
        token = evidence.get("recovered_fencing_token")
        return type(token) is int and token > entry["parameters"]["fencing_token"]

    def _apply_orphan_recovery_local_effects(
        self,
        binding: ProjectBinding,
        entry: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> bool:
        parameters = entry["parameters"]
        attempt_id = parameters["attempt_id"]
        recovered_token = evidence["recovered_fencing_token"]
        paths = machine_project_paths(self.runtime.root, binding.project_id)
        cfg = _termination_local_config(self.runtime, binding)
        try:
            with evidence_write_guard(paths["root"], attempt_id) as evidence_acquired:
                if not evidence_acquired:
                    return False
                with attempt_control_lock(cfg, attempt_id, blocking=False) as lock_acquired:
                    if not lock_acquired:
                        return False
                    current = _read_process_manifest(
                        self.runtime,
                        binding,
                        entry["path"],
                        entry["binding_signature"],
                    )
                    if current is None or current["intent_signature"] != entry["intent_signature"]:
                        return False
                    reservation = self._recovery_reservation(
                        parameters["reservation_id"],
                        allowed_states=frozenset({"active", "released"}),
                    )
                    if reservation is None:
                        return False
                    reservation_state, identity = reservation
                    if (
                        identity.project_id != binding.project_id
                        or identity.task_id != parameters["task_id"]
                        or identity.attempt_id != attempt_id
                        or identity.fencing_token not in {parameters["fencing_token"], recovered_token}
                    ):
                        return False
                    if reservation_state == "active" and identity.fencing_token == parameters["fencing_token"]:
                        retag = retag_cpu_if_matches if identity.cpu_slots is not None else retag_if_matches
                        if not retag(self.runtime.root, identity, attempt_id, recovered_token):
                            return False
                    process = dict(current["process"])
                    process.update(
                        fencing_token=recovered_token,
                        recovered_at=utc_now(),
                        observed_state="running",
                        supervisor="agent",
                        authority_state="healthy",
                        lease_expires_at=evidence["lease_expires_at"],
                    )
                    atomic_replace(entry["path"], {"process": process})
                    return True
        except _TERMINATION_SOURCE_ERRORS:
            return False

    def advance_orphan_recoveries(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Recover bounded live orphan manifests through one shared-only CAS."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        eligible = list(bindings)
        current_signatures = {_authority_binding_signature(binding, registry_revision) for binding in eligible}
        for signature in tuple(self._orphan_recovery_scans):
            has_entry = any(
                entry.get("binding_signature") == signature for entry in self._orphan_recovery_entries.values()
            )
            if signature not in current_signatures or (
                signature in self._orphan_recovery_scan_complete and not has_entry
            ):
                self._orphan_recovery_scans.pop(signature).close()
                self._orphan_recovery_scan_complete.discard(signature)
        binding_by_project = {binding.project_id: binding for binding in eligible}
        for project_id, entry in tuple(self._orphan_recovery_entries.items()):
            binding = binding_by_project.get(project_id)
            if binding is None or entry.get("binding_signature") != _authority_binding_signature(
                binding, registry_revision
            ):
                self._orphan_recovery_entries.pop(project_id, None)
        if eligible:
            start = self._orphan_recovery_binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected = rotated[: min(_MAX_TERMINATION_BINDINGS, len(rotated))]
            self._orphan_recovery_binding_offset = (start + len(selected)) % len(eligible)
        else:
            selected = []
            self._orphan_recovery_binding_offset = 0

        attempts: dict[str, Mapping[str, Any]] = {}
        entries: dict[str, Mapping[str, Any]] = {}
        request_bindings: list[ProjectBinding] = []
        for binding in selected:
            signature = _authority_binding_signature(binding, registry_revision)
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            entry = self._orphan_recovery_entries.get(binding.project_id)
            if entry is not None:
                try:
                    current = _read_process_manifest(
                        self.runtime,
                        binding,
                        entry["path"],
                        signature,
                    )
                except _TERMINATION_SOURCE_ERRORS:
                    current = None
                if current is None or current["intent_signature"] != entry["intent_signature"]:
                    self._orphan_recovery_entries.pop(binding.project_id, None)
                    entry = None
            scan = self._orphan_recovery_scans.get(signature)
            if scan is None:
                if len(self._orphan_recovery_scans) >= _MAX_TERMINATION_STATES:
                    diagnostic_increment("scheduler.isolated.orphan_scan_capacity_exhausted")
                    continue
                scan = EvidenceScan(paths["processes"])
                self._orphan_recovery_scans[signature] = scan
                self._orphan_recovery_scan_complete.discard(signature)
            try:
                if entry is None:
                    page = scan.take(1)
                    if page.is_complete:
                        self._orphan_recovery_scan_complete.add(signature)
                    if not page.paths:
                        continue
                    entry = _read_process_manifest(
                        self.runtime,
                        binding,
                        page.paths[0],
                        signature,
                    )
                    if entry is None or entry["process"].get("authority_state") == "healthy":
                        continue
                    if len(self._orphan_recovery_entries) >= _MAX_TERMINATION_CANDIDATES:
                        continue
                reservation_id = entry["parameters"].get("reservation_id")
                if not isinstance(reservation_id, str):
                    continue
                reservation_record = self._recovery_reservation(
                    reservation_id,
                    allowed_states=frozenset({"active", "released"}),
                )
                if (
                    reservation_record is None
                    or reservation_record[1].project_id != binding.project_id
                    or reservation_record[1].task_id != entry["parameters"]["task_id"]
                    or reservation_record[1].attempt_id != entry["parameters"]["attempt_id"]
                    or type(reservation_record[1].fencing_token) is not int
                    or reservation_record[1].fencing_token < entry["parameters"]["fencing_token"]
                ):
                    self._orphan_recovery_entries.pop(binding.project_id, None)
                    continue
                if binding.project_id not in self._orphan_recovery_entries:
                    self._orphan_recovery_entries[binding.project_id] = entry
            except _TERMINATION_SOURCE_ERRORS:
                continue
            entries[binding.project_id] = entry
            attempts[binding.project_id] = {
                **entry["parameters"],
                "binding_signature": self._orphan_binding_signature(binding),
                "source_revisions": {"task": None, "attempt_digest": None},
                # A process may exit after shared recovery commits but before
                # local application. Replay that exact marker, never create
                # fresh live authority from a dead manifest.
                "replay_only": reservation_record[0] == "released"
                or _read_isolated_authority_manifest(self.runtime, binding, entry["path"], signature) is None,
            }
            request_bindings.append(binding)
        if not attempts:
            return {}
        try:
            results = self.controller.advance_authority_orphan_recoveries(
                request_bindings,
                registry_revision,
                attempts,
            )
        except _TERMINATION_SOURCE_ERRORS:
            return {}
        for binding in request_bindings:
            entry = entries[binding.project_id]
            evidence = results.get(binding.project_id)
            stable_signature = self._orphan_binding_signature(binding)
            if self._orphan_recovery_proof(evidence, entry, stable_signature):
                if self._apply_orphan_recovery_local_effects(binding, entry, evidence):
                    self._orphan_recovery_entries.pop(binding.project_id, None)
            elif isinstance(evidence, Mapping) and evidence.get("outcome") == "stale":
                self._orphan_recovery_entries.pop(binding.project_id, None)
        for signature, scan in tuple(self._orphan_recovery_scans.items()):
            if signature in self._orphan_recovery_scan_complete and not any(
                entry.get("binding_signature") == signature for entry in self._orphan_recovery_entries.values()
            ):
                self._orphan_recovery_scans.pop(signature, scan).close()
                self._orphan_recovery_scan_complete.discard(signature)
        return results

    def advance_terminations(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Advance bounded local cancellation and holder-safe-deadline termination."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")

        eligible = list(bindings)
        signatures = [_authority_binding_signature(binding, registry_revision) for binding in eligible]
        current_signatures = set(signatures)
        binding_by_project = {binding.project_id: binding for binding in eligible}
        for signature in tuple(self._termination_states):
            state = self._termination_states[signature]
            has_candidate = any(
                candidate.binding_signature == signature for candidate in self._termination_candidates.values()
            )
            if signature not in current_signatures or (state.process_scan_complete and not has_candidate):
                self._termination_states.pop(signature).close()
        for key, candidate in tuple(self._termination_candidates.items()):
            binding = binding_by_project.get(candidate.binding.project_id)
            if (
                binding is None
                or _authority_binding_signature(binding, registry_revision) != candidate.binding_signature
            ):
                self._termination_candidates.pop(key, None)
        if eligible:
            start = self._termination_binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected = rotated[: min(_MAX_TERMINATION_BINDINGS, len(rotated))]
            self._termination_binding_offset = (start + len(selected)) % len(eligible)
        else:
            selected = []
            self._termination_binding_offset = 0
        selected_projects = {binding.project_id for binding in selected}

        # Existing candidates are re-read before any new scan page is consumed.
        for key, candidate in tuple(self._termination_candidates.items()):
            if candidate.binding.project_id not in selected_projects:
                continue
            binding = binding_by_project[candidate.binding.project_id]
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            try:
                process = _read_terminal_process_record(candidate.process_path, paths["root"], binding)
                natural_observation = is_path_present(
                    paths["observations"] / f"{candidate.parameters['attempt_id']}.json"
                )
            except _TERMINATION_SOURCE_ERRORS:
                continue
            if not self._termination_process_matches(process, candidate):
                self._termination_candidates.pop(key, None)
                continue
            if natural_observation and candidate.decision is None:
                self._termination_candidates.pop(key, None)
                continue
            candidate.binding = binding

        # Recover executor requests first so a restart can pin an exact process.
        selected_candidates: dict[tuple[str, str], _TerminationCandidate] = {
            key: candidate
            for key, candidate in self._termination_candidates.items()
            if candidate.binding.project_id in selected_projects
        }
        self._recover_termination_requests(self._termination_candidates, binding_by_project, registry_revision)
        selected_candidates = {
            key: candidate
            for key, candidate in self._termination_candidates.items()
            if candidate.binding.project_id in selected_projects
        }

        # A durable decision can outlive the in-memory candidate and lets an
        # unresolved request replay even when the authority observation is
        # temporarily unavailable.
        for candidate in selected_candidates.values():
            if candidate.decision_id is not None and candidate.trigger in _TERMINATION_TRIGGERS:
                cfg = _termination_local_config(self.runtime, candidate.binding)
                decision = _read_termination_decision(
                    cfg,
                    candidate.parameters["attempt_id"],
                    candidate.decision_id,
                )
                if _termination_decision_matches_candidate(decision, candidate):
                    candidate.decision = dict(decision)
            if candidate.decision is not None:
                continue
            for trigger in _TERMINATION_TRIGGERS:
                decision_id = _termination_decision_id(candidate, trigger)
                cfg = _termination_local_config(self.runtime, candidate.binding)
                decision = _read_termination_decision(
                    cfg,
                    candidate.parameters["attempt_id"],
                    decision_id,
                )
                candidate.trigger = trigger
                candidate.decision_id = decision_id
                if _termination_decision_matches_candidate(decision, candidate, trigger=trigger):
                    candidate.decision = dict(decision)
                    break
            if (
                candidate.decision is None
                and candidate.pending_commit is None
                and candidate.pending_publication is None
            ):
                candidate.trigger = None
                candidate.decision_id = None

        for binding in selected:
            signature = _authority_binding_signature(binding, registry_revision)
            state = self._termination_states.get(signature)
            if state is None:
                if len(self._termination_states) >= _MAX_TERMINATION_STATES:
                    diagnostic_increment("scheduler.isolated.termination_scan_capacity_exhausted")
                    continue
                paths = machine_project_paths(self.runtime.root, binding.project_id)
                state = _TerminationBindingState(
                    process_scan=EvidenceScan(paths["processes"]),
                    reconciler=LocalExitReconciler(
                        paths["root"],
                        reservation_runtime_root=self.runtime.root,
                        project_id=binding.project_id,
                    ),
                )
                self._termination_states[signature] = state
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            try:
                page = state.process_scan.take(1)
            except _TERMINATION_SOURCE_ERRORS:
                continue
            state.process_scan_complete = page.is_complete
            if not page.paths:
                continue
            process_path = page.paths[0]
            try:
                process = _read_termination_process_record(process_path, paths["root"], binding)
                if process is None:
                    retained = _read_terminal_process_record(process_path, paths["root"], binding)
                    if retained is None or retained.record.get("observed_state") != "exited":
                        continue
                    decision_match = self._find_termination_decision_for_process(
                        binding,
                        signature,
                        paths,
                        process_path,
                        retained,
                    )
                    if decision_match is None:
                        continue
                    process = retained
            except _TERMINATION_SOURCE_ERRORS:
                continue
            if process is None:
                continue
            key = (binding.project_id, process.parameters["attempt_id"])
            if key in self._termination_candidates or any(
                candidate.binding.project_id == binding.project_id
                for candidate in self._termination_candidates.values()
            ):
                continue
            if len(self._termination_candidates) >= _MAX_TERMINATION_CANDIDATES:
                continue
            candidate = self._new_termination_candidate(binding, signature, process_path, process)
            decision_match = self._find_termination_decision_for_process(
                binding,
                signature,
                paths,
                process_path,
                process,
            )
            if decision_match is not None:
                decision_path_value, decision = decision_match
                candidate.trigger = decision.get("authority_outcome")
                candidate.decision_id = decision_path_value.stem
                candidate.decision = dict(decision)
                candidate.is_retained_replay = True
            self._termination_candidates[key] = candidate
            selected_candidates[key] = candidate

        for signature, state in tuple(self._termination_states.items()):
            if state.process_scan_complete and not any(
                candidate.binding_signature == signature for candidate in self._termination_candidates.values()
            ):
                self._termination_states.pop(signature).close()

        # Reclassify retained durable decisions before requesting shared work.
        # Confirmed repairable records are replayed by the existing guarded local
        # effect path; process-inspection ambiguity remains deferred.
        for key, candidate in tuple(selected_candidates.items()):
            if (
                candidate.decision is None
                or not candidate.is_retained_replay
                or key not in self._termination_candidates
            ):
                continue
            state = self._termination_states.get(candidate.binding_signature)
            if state is None:
                continue
            paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
            try:
                result = self._termination_convergence_for_decision(
                    candidate.binding,
                    paths,
                    candidate.binding_signature,
                    paths["termination_decisions"]
                    / candidate.parameters["attempt_id"]
                    / f"{candidate.decision_id}.json",
                    decision=candidate.decision,
                )
            except OSError:
                continue
            if result is None:
                continue
            candidate.convergence = result
            if result.state == "repairable" and result.reason != "termination_decision_progress_pending":
                local_outcome = self._apply_termination_local_effects(candidate, paths, state)
                if local_outcome == "invalid":
                    candidate.convergence = TerminationConvergence(
                        "invalid",
                        "termination_decision_invalid",
                        result.manifest_exit_code,
                        result.process_absent,
                        False,
                    )
                elif local_outcome == "applied":
                    try:
                        refreshed = self._termination_convergence_for_decision(
                            candidate.binding,
                            paths,
                            candidate.binding_signature,
                            paths["termination_decisions"]
                            / candidate.parameters["attempt_id"]
                            / f"{candidate.decision_id}.json",
                            decision=candidate.decision,
                        )
                    except OSError:
                        refreshed = None
                    if refreshed is not None:
                        candidate.convergence = refreshed

        # Obtain one exact active observation for the selected candidates.
        observation_results: Mapping[str, Mapping[str, Any]] = {}
        candidates_by_project: dict[str, tuple[tuple[str, str], _TerminationCandidate]] = {}
        for key, candidate in selected_candidates.items():
            if candidate.decision is not None and candidate.decision.get("state") == "confirmed":
                continue
            candidates_by_project.setdefault(candidate.binding.project_id, (key, candidate))
        if candidates_by_project:
            observation_bindings = [candidate.binding for _, candidate in candidates_by_project.values()]
            attempts = {
                project_id: candidate.parameters for project_id, (_key, candidate) in candidates_by_project.items()
            }
            try:
                observation_results = self.controller.advance_authority_terminal_observations(
                    observation_bindings,
                    registry_revision,
                    attempts,
                )
            except _TERMINATION_SOURCE_ERRORS:
                observation_results = {}

        current_observations: dict[tuple[str, str], Mapping[str, Any]] = {}
        for key, candidate in tuple(selected_candidates.items()):
            evidence = observation_results.get(candidate.binding.project_id)
            if _termination_observation_is_current(evidence, candidate):
                paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
                try:
                    process = _read_terminal_process_record(candidate.process_path, paths["root"], candidate.binding)
                except _TERMINATION_SOURCE_ERRORS:
                    continue
                if not self._termination_process_matches(process, candidate):
                    self._termination_candidates.pop(key, None)
                    continue
                trigger = _termination_trigger(process.record, evidence)
                if trigger is None:
                    # A completed healthy observation is not pending termination
                    # work. Keeping it pinned blocks restart readiness forever.
                    if (
                        candidate.decision is None
                        and candidate.pending_commit is None
                        and candidate.pending_publication is None
                    ):
                        self._termination_candidates.pop(key, None)
                    continue
                if candidate.trigger is not None and candidate.trigger != trigger:
                    self._termination_candidates.pop(key, None)
                    continue
                candidate.trigger = trigger
                expected_id = _termination_decision_id(candidate, trigger)
                if candidate.decision_id is not None and candidate.decision_id != expected_id:
                    self._termination_candidates.pop(key, None)
                    continue
                candidate.decision_id = expected_id
                revisions = evidence.get("source_revisions")
                if not _has_exact_terminal_revisions(revisions):
                    self._termination_candidates.pop(key, None)
                    continue
                if (
                    candidate.source_revisions is not None
                    and candidate.source_revisions != revisions
                    and candidate.decision is None
                    and candidate.pending_commit is None
                    and candidate.pending_publication is None
                ):
                    self._termination_candidates.pop(key, None)
                    continue
                if candidate.source_revisions is None:
                    candidate.source_revisions = dict(revisions)
                current_observations[key] = evidence
            elif _termination_terminal_observation_proof(evidence, candidate):
                # A lost publication result can be replaced by a durable exact
                # terminal observation on the next bounded turn.
                continue
            elif isinstance(evidence, Mapping) and evidence.get("outcome") in {"unavailable", "deferred"}:
                continue
            elif evidence is not None:
                if candidate.decision is not None and candidate.decision.get("state") != "confirmed":
                    # A retained non-confirmed decision may only use its
                    # monotonic local decision state machine; it does not need
                    # a fresh running observation to advance that state.
                    continue
                self._termination_candidates.pop(key, None)

        # Create/reuse deterministic local decisions before calling shared I/O.
        for key, candidate in tuple(selected_candidates.items()):
            if key not in current_observations or candidate.trigger is None:
                continue
            paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
            result = self._prepare_termination_decision(candidate, paths)
            if result == "invalid":
                self._termination_candidates.pop(key, None)
            elif result == "deferred":
                continue

        commit_requests: dict[str, Mapping[str, Any]] = {}
        commit_candidates: dict[str, _TerminationCandidate] = {}
        for key, candidate in selected_candidates.items():
            if candidate.binding.project_id not in selected_projects or candidate.decision is None:
                continue
            has_durable_commit = candidate.decision.get("shared_commitment") == "committed"
            if candidate.pending_commit is None and (candidate.commit_evidence is not None or has_durable_commit):
                continue
            request = _termination_commit_parameters(candidate)
            if request is None:
                continue
            if candidate.pending_commit is not None and dict(candidate.pending_commit) != dict(request):
                self._termination_candidates.pop(key, None)
                continue
            commit_requests[candidate.binding.project_id] = request
            commit_candidates[candidate.binding.project_id] = candidate
        commit_results: Mapping[str, Mapping[str, Any]] = {}
        if commit_requests:
            commit_bindings = [commit_candidates[project_id].binding for project_id in commit_requests]
            try:
                commit_results = self.controller.advance_authority_termination_commits(
                    commit_bindings,
                    registry_revision,
                    commit_requests,
                )
            except _TERMINATION_SOURCE_ERRORS:
                commit_results = {}
        for project_id, evidence in commit_results.items():
            candidate = commit_candidates.get(project_id)
            if candidate is None:
                continue
            if _termination_commit_proof(evidence, candidate):
                candidate.commit_evidence = dict(evidence)
                candidate.pending_commit = None
            elif isinstance(evidence, Mapping) and evidence.get("outcome") in {"stale", "invalid"}:
                self._termination_candidates.pop(self._termination_key(candidate), None)

        # Advance one nonblocking local signal step only after exact commit proof.
        for key, candidate in tuple(selected_candidates.items()):
            if key not in self._termination_candidates or candidate.decision is None:
                continue
            if candidate.commit_evidence is None and candidate.decision.get("shared_commitment") != "committed":
                continue
            paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
            result = self._advance_termination_signals(candidate, paths)
            if result == "invalid":
                self._termination_candidates.pop(key, None)

        # A signal-confirmed process gets a fresh typed observation before publish.
        confirmed: dict[str, _TerminationCandidate] = {}
        for key, candidate in selected_candidates.items():
            if key not in self._termination_candidates or candidate.decision is None:
                continue
            attempts = candidate.decision.get("signal_attempts")
            if candidate.decision.get("state") == "confirmed" and isinstance(attempts, list):
                confirmed[candidate.binding.project_id] = candidate
        post_signal_observations: Mapping[str, Mapping[str, Any]] = {}
        needs_post_signal_observation = {
            project_id: candidate
            for project_id, candidate in confirmed.items()
            if candidate.pending_publication is None
        }
        if needs_post_signal_observation:
            try:
                post_signal_observations = self.controller.advance_authority_terminal_observations(
                    [candidate.binding for candidate in needs_post_signal_observation.values()],
                    registry_revision,
                    {
                        project_id: candidate.parameters
                        for project_id, candidate in needs_post_signal_observation.items()
                    },
                )
            except _TERMINATION_SOURCE_ERRORS:
                post_signal_observations = {}

        publication_requests: dict[str, Mapping[str, Any]] = {}
        publication_candidates: dict[str, _TerminationCandidate] = {}
        terminal_proofs: set[str] = set()
        for project_id, candidate in confirmed.items():
            evidence = post_signal_observations.get(project_id)
            if _termination_terminal_observation_proof(evidence, candidate):
                terminal_proofs.add(project_id)
                continue
            transition = candidate.pending_publication
            if transition is None and _termination_observation_has_shared_marker(evidence, candidate):
                transition = _build_termination_transition(candidate, evidence)
                candidate.pending_publication = transition
            if transition is None or not self._termination_transition_matches_candidate(transition, candidate):
                if isinstance(evidence, Mapping) and evidence.get("outcome") in {"stale", "invalid"}:
                    self._termination_candidates.pop(self._termination_key(candidate), None)
                continue
            publication_requests[project_id] = transition
            publication_candidates[project_id] = candidate

        publication_results: Mapping[str, Mapping[str, Any]] = {}
        if publication_requests:
            try:
                publication_results = self.controller.advance_authority_terminal_publications(
                    [candidate.binding for candidate in publication_candidates.values()],
                    registry_revision,
                    publication_requests,
                )
            except _TERMINATION_SOURCE_ERRORS:
                publication_results = {}
        for project_id, candidate in publication_candidates.items():
            evidence = publication_results.get(project_id)
            transition = publication_requests[project_id]
            if _termination_publication_proof(evidence, candidate, transition):
                terminal_proofs.add(project_id)
                candidate.pending_publication = None
            elif isinstance(evidence, Mapping) and evidence.get("outcome") in {"stale", "invalid"}:
                self._termination_candidates.pop(self._termination_key(candidate), None)

        results: dict[str, Mapping[str, Any]] = {}
        for project_id in terminal_proofs:
            candidate = confirmed.get(project_id)
            if candidate is None or self._termination_key(candidate) not in self._termination_candidates:
                continue
            state = self._termination_states.get(candidate.binding_signature)
            if state is None:
                continue
            paths = machine_project_paths(self.runtime.root, candidate.binding.project_id)
            local_outcome = self._apply_termination_local_effects(candidate, paths, state)
            if local_outcome in {"applied", "invalid"}:
                self._termination_candidates.pop(self._termination_key(candidate), None)
            if project_id in publication_results:
                results[project_id] = publication_results[project_id]
            elif project_id in post_signal_observations:
                results[project_id] = post_signal_observations[project_id]
        return results
