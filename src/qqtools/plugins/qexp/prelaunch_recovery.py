"""Conservative expiry and owner cleanup for bounded pre-launch claims."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent.bindings import decode_registry
from .agent.context import _effective_runtime_id
from .agent.identity import decode_identity_record
from .config_types import RootConfig
from .lease import ClockObservation, LeasePolicy, _boot_id, clock_capability, decode_lease_policy, reclaim_allowed_at
from .runtime.authority_lock import authority_locks
from .runtime.claims import archive_claim
from .runtime.durable_ownership import OwnershipTransitionConflict, _checked_receipt
from .runtime.group_discovery.changes import record_task_change
from .runtime.locks import exclusive, machine_lock, schema_writer_lock
from .runtime.paths import attempt_path, local_paths, machine_registration_path, shared_paths, task_path
from .runtime.ready import commit_ready_publication, prepare_ready_transition, retire_previous_ready_generation
from .runtime.records import AttemptRecord, TaskRecord, utc_now, validate_identifier
from .runtime.registration_authority import registration_state
from .runtime.resources.reservations import ReservationIdentity, release_if_matches
from .runtime.store import atomic_replace, fenced_mutations, read_json_limited
from .runtime.tasks import load_task, save_task

_EXPIRY = "lease_expired_before_launch"


class BindingEligibilityChanged(RuntimeError):
    """The owning registration lost write eligibility during recovery."""


_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError)
_RecordReader = Callable[[Path], dict[str, Any]]


def expected_target(identity: ReservationIdentity, runtime_id: str) -> str:
    """Digest the complete reservation identity and the owning runtime."""
    payload = {"version": 1, "runtime_id": runtime_id, "reservation": asdict(identity)}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_record(path: Path) -> dict[str, Any]:
    return read_json_limited(path, max_bytes=1024 * 1024)


def _optional_record(read: _RecordReader, path: Path) -> dict[str, Any] | None:
    try:
        return read(path)
    except FileNotFoundError:
        return None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        time_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if time_value.tzinfo is None or time_value.utcoffset().total_seconds() != 0:
        return None
    return time_value


def _valid_bound(value: object) -> bool:
    return type(value) in {int, float} and 0 <= value < math.inf


def _clock_result(
    cfg: RootConfig,
    claim: dict[str, Any],
    attempt: AttemptRecord,
    read: _RecordReader,
    acquire_clock: Callable | None,
    result: dict[str, Any],
) -> str:
    evidence = attempt.lease.get("clock_evidence")
    expires = claim.get("lease_expires_at")
    expiry_time = _timestamp(expires)
    holder_bound = claim.get("clock_error_bound_seconds")
    observation_id = claim.get("clock_observation_id")
    if (
        expiry_time is None
        or attempt.lease.get("expires_at") != expires
        or not _valid_bound(holder_bound)
        or not isinstance(evidence, dict)
        or evidence.get("clock_error_bound_seconds") != holder_bound
        or evidence.get("observation_id") != observation_id
        or evidence.get("provider") != claim.get("clock_provider")
    ):
        return "clock_evidence_invalid"
    try:
        validate_identifier(observation_id, "clock_observation_id")
        holder_record = read(
            shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{observation_id}.json"
        )
        holder = ClockObservation.from_dict(holder_record["clock_observation"])
        if (
            holder.observation_id != observation_id
            or holder.provider != evidence["provider"]
            or not _valid_bound(holder.max_drift_rate)
            or not _valid_bound(holder.provider_margin_seconds)
            or not _valid_bound(holder.monotonic_observed_at)
            or not _valid_bound(holder.observed_bound_seconds)
            or holder.lower_error_seconds > holder.upper_error_seconds
            or holder.bound_at(holder.monotonic_observed_at) > holder_bound
        ):
            return "clock_evidence_invalid"
        policy_path = shared_paths(cfg.shared_root)["lease_policy"]
        policy_record = _optional_record(read, policy_path)
        policy = LeasePolicy() if policy_record is None else decode_lease_policy(policy_record)
        if acquire_clock is not None:
            capability = acquire_clock(cfg, policy)
            if not capability.is_healthy or capability.observation is None:
                return "clock_evidence_unavailable"
            reclaimer = capability.observation
        else:
            cached = read(local_paths(cfg.runtime_root)["clock_health"])["clock_capability"]
            if cached["status"] != "healthy":
                return "clock_evidence_unavailable"
            reclaimer = ClockObservation.from_dict(cached["observation"])
            age = time.monotonic() - reclaimer.monotonic_observed_at
            if reclaimer.boot_id != _boot_id() or not 0 <= age <= policy.clock_observation_max_age_seconds:
                return "clock_evidence_unavailable"
        reclaimer_bound = reclaimer.bound_at(time.monotonic())
        if not _valid_bound(reclaimer_bound) or reclaimer_bound > policy.max_clock_skew_seconds:
            return "clock_evidence_unavailable"
        safe_at = reclaim_allowed_at(expires, holder_bound, reclaimer_bound)
    except (*_READ_ERRORS, RuntimeError):
        return "clock_evidence_unavailable"
    now = datetime.now(timezone.utc)
    result["safe_at"] = safe_at.isoformat().replace("+00:00", "Z")
    result["overdue_seconds"] = max(0.0, (now - safe_at).total_seconds())
    if now < expiry_time:
        return "waiting_for_launch"
    if now < safe_at:
        return "clock_safety_window"
    return "eligible_for_recovery"


def _local_records(cfg: RootConfig, attempt_id: str) -> list[str]:
    paths = local_paths(cfg.runtime_root)
    present = []
    for name in ("launch_intents", "registrations", "processes", "observations"):
        try:
            (paths[name] / f"{attempt_id}.json").lstat()
        except FileNotFoundError:
            continue
        except OSError:
            present.append(f"{name}:unreadable")
        else:
            present.append(name)
    return present


def _launch_result(
    cfg: RootConfig, task_id: str, claim: dict[str, Any], attempt: AttemptRecord, result: dict[str, Any]
) -> str | None:
    receipt = claim.get("ownership_transition")
    evidence = {
        "claim_launch_state": claim.get("launch_state"),
        "claim_authority_mode": claim.get("authority_mode"),
        "attempt_phase": attempt.phase,
        "launch_id": claim.get("launch_id") or attempt.authorization.get("launch_id"),
        "launch_authorized_at": claim.get("launch_authorized_at") or attempt.timestamps.get("launch_authorized_at"),
        "ownership_transition": receipt is not None,
        "local_records": _local_records(cfg, attempt.attempt_id),
    }
    result["launch_evidence"] = evidence
    if receipt is not None:
        try:
            checked, _ = _checked_receipt(receipt)
        except (OwnershipTransitionConflict, TypeError, ValueError):
            return "launch_evidence_conflict"
        if any(
            checked.get(key) != value
            for key, value in {
                "task_id": task_id,
                "attempt_id": attempt.attempt_id,
                "attempt_number": attempt.attempt_number,
                "fencing_token": attempt.current_fencing_token,
                "machine_name": attempt.machine_name,
                "reservation_id": attempt.reservation_id,
                "launch_id": claim.get("launch_id"),
                "authorization_timestamp": claim.get("launch_authorized_at"),
            }.items()
        ) or claim.get("launch_state") not in {"starting", "running"}:
            return "launch_evidence_conflict"
        return "launch_authorized"
    if attempt.phase in {"starting", "running"} or claim.get("launch_state") in {"starting", "running"}:
        return "launch_authorized"
    if (
        any(
            value is not None
            for value in (
                claim.get("launch_id"),
                claim.get("launch_authorized_at"),
                attempt.authorization.get("launch_id"),
                attempt.timestamps.get("launch_authorized_at"),
            )
        )
        or evidence["local_records"]
        or any(attempt.timestamps.get(key) is not None for key in ("process_created_at", "running_at", "orphaned_at"))
        or any(
            attempt.process.get(key) is not None
            for key in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
                "tmux_reference",
            )
        )
        or attempt.process.get("local_process_manifest")
        or claim.get("termination_decision_id") is not None
    ):
        return "launch_evidence_conflict"
    return None


def _binding_current(
    cfg: RootConfig, identity: ReservationIdentity, runtime_root: Path, runtime_id: str, read: _RecordReader
) -> bool:
    if not runtime_id:
        return True  # Standalone scheduler compatibility.
    if (
        identity.shared_root != str(cfg.shared_root)
        or not identity.project_id
        or cfg.runtime_root != runtime_root / "projects" / identity.project_id
    ):
        return False
    record = decode_identity_record(read(runtime_root / "identity.json"))
    if record is None:
        return False
    observed_runtime_id = _effective_runtime_id(record["instance_id"], record.get("runtime_id"))
    if observed_runtime_id != runtime_id:
        return False
    project = read(shared_paths(cfg.shared_root)["project"] / "identity.json")["project"]
    if project.get("project_id") != identity.project_id:
        return False
    _, bindings = decode_registry(read(runtime_root / "registry.json"))
    matches = [binding for binding in bindings if binding.project_id == identity.project_id]
    if len(matches) != 1:
        return False
    binding = matches[0]
    if (
        binding.shared_root != cfg.shared_root
        or binding.machine_name != cfg.machine_name
        or binding.runtime_instance_id != runtime_id
        or binding.runtime_root != str(runtime_root)
        or (identity.registration_generation is not None and identity.registration_generation != binding.generation)
    ):
        return False
    registration = read(machine_registration_path(cfg.shared_root, cfg.machine_name))["registration"]
    return (
        all(
            registration.get(key) == value
            for key, value in {
                "project_id": identity.project_id,
                "shared_root": identity.shared_root,
                "machine_name": cfg.machine_name,
                "generation": binding.generation,
                "runtime_instance_id": runtime_id,
                "runtime_root": str(runtime_root),
            }.items()
        )
        and registration.get("state") == "eligible"
        and registration_state(registration) == "eligible"
    )


def _reservation_state(
    cfg: RootConfig, identity: ReservationIdentity, runtime_root: Path, read: _RecordReader
) -> tuple[str, str | None]:
    paths = local_paths(runtime_root)
    prefix = "cpu_" if identity.cpu_slots is not None else ""
    filename = f"{identity.reservation_id}.json"
    states = {}
    for state in ("active", "provisional", "released"):
        value = _optional_record(read, paths[f"{prefix}{state}"] / filename)
        if value is None:
            continue
        record = value["reservation"]
        if (
            record.get("state") != state
            or record.get("machine_name") != cfg.machine_name
            or not identity.exactly_matches(record)
        ):
            return "conflict", None
        states[state] = record
    if "provisional" in states:
        return "conflict", None
    release_reason = states.get("released", {}).get("release_reason")
    if "active" in states:
        return ("release_pair" if "released" in states else "active"), release_reason
    return ("released" if "released" in states else "absent"), release_reason


def _claim_matches(claim: object, identity: ReservationIdentity) -> bool:
    return isinstance(claim, dict) and (
        claim.get("reservation_id") == identity.reservation_id
        and claim.get("attempt_id") == identity.attempt_id
        and type(claim.get("fencing_token")) is int
        and claim["fencing_token"] == identity.fencing_token
    )


def _expiry_archive(
    cfg: RootConfig, identity: ReservationIdentity, read: _RecordReader
) -> tuple[dict[str, Any] | None, Path]:
    path = shared_paths(cfg.shared_root)["claim_archive"] / identity.task_id / f"{identity.fencing_token}.json"
    record = _optional_record(read, path)
    if record is None:
        return None, path
    archive = record["claim_archive"]
    if archive.get("reason") != "lease_expired":
        return None, path
    if (
        archive.get("task_id") != identity.task_id
        or archive.get("attempt_id") != identity.attempt_id
        or archive.get("fencing_token") != identity.fencing_token
        or not _claim_matches(archive.get("claim"), identity)
    ):
        raise ValueError("Pre-launch expiry archive has a conflicting identity.")
    return archive["claim"], path


def _attempt_matches(
    cfg: RootConfig, attempt: AttemptRecord, claim: dict[str, Any], identity: ReservationIdentity
) -> bool:
    return (
        attempt.task_id == identity.task_id
        and attempt.attempt_id == identity.attempt_id
        and attempt.attempt_number == claim["attempt_number"]
        and type(attempt.current_fencing_token) is int
        and attempt.current_fencing_token == identity.fencing_token
        and attempt.reservation_id == identity.reservation_id
        and attempt.machine_name == claim.get("machine_name") == cfg.machine_name
        and type(attempt.assigned_gpus) is list
        and all(type(gpu) is int for gpu in attempt.assigned_gpus)
        and tuple(attempt.assigned_gpus) == identity.gpu_ids
    )


def _commit_expiry(
    cfg: RootConfig,
    task: TaskRecord,
    claim: dict[str, Any],
    attempt: AttemptRecord,
    target: str,
    attempt_file: Path,
    archive_file: Path,
    *,
    before_mutation: Callable[[], None] | None,
    task_writer: Callable,
) -> bool:
    archived_claim = {**claim, "prelaunch_expiry_target": target}
    with fenced_mutations(cfg.shared_root, before_mutation):
        if not archive_claim(cfg, task.task_id, archived_claim, "lease_expired"):
            return False
        if _read_record(archive_file)["claim_archive"]["claim"] != archived_claim:
            return False
    with (
        fenced_mutations(cfg.shared_root, before_mutation),
        record_task_change(
            cfg,
            task,
            "claim_loss",
            details={
                "expected_attempt_id": attempt.attempt_id,
                "expected_fencing_token": attempt.current_fencing_token,
                "reserved_generation": task.ready_generation + 1,
            },
        ),
    ):
        if attempt.phase == "claimed":
            attempt.phase = "cancelled"
            attempt.result["reason"] = _EXPIRY
            attempt.timestamps["finished_at"] = utc_now()
            atomic_replace(attempt_file, attempt.to_dict())
        task.state.update(projection="queued", reason=_EXPIRY)
        old_generation, _ = prepare_ready_transition(cfg, task, "prelaunch_expiry")
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        task_writer(cfg, task)
        commit_ready_publication(cfg, task)
        retire_previous_ready_generation(cfg, old_generation, task)
        return True


def _release_expired(
    cfg: RootConfig,
    identity: ReservationIdentity,
    runtime_root: Path,
    runtime_id: str,
    result: dict[str, Any],
    before_mutation: Callable[[], None] | None,
) -> None:
    if before_mutation is not None:
        before_mutation()
    try:
        if not _binding_current(cfg, identity, runtime_root, runtime_id, _read_record):
            result.update(reason="cleanup_pending", machine_outcome="cleanup_pending")
            return
        state, release_reason = _reservation_state(cfg, identity, runtime_root, _read_record)
        if state == "released" and release_reason == _EXPIRY:
            result.update(reason=_EXPIRY, machine_outcome="already_released")
            return
        can_release = state == "active" or (state == "release_pair" and release_reason == _EXPIRY)
        if can_release and release_if_matches(runtime_root, identity, _EXPIRY):
            result.update(reason=_EXPIRY, machine_outcome="released")
            return
    except _READ_ERRORS:
        pass
    result.update(reason="cleanup_pending", machine_outcome="cleanup_pending")


def _evaluate(
    cfg: RootConfig,
    task: TaskRecord | None,
    identity: ReservationIdentity,
    runtime_root: Path,
    runtime_id: str,
    result: dict[str, Any],
    read: _RecordReader,
    *,
    acquire_clock: Callable | None,
    before_mutation: Callable[[], None] | None,
    task_writer: Callable,
    apply: bool,
) -> bool:
    """Evaluate the fixed identity, returning whether a committed expiry may release."""
    target = result["expected_target"]
    active_claim = None if task is None else task.claim_control.get("active_claim")
    active_matches = _claim_matches(active_claim, identity)
    try:
        archive, archive_file = _expiry_archive(cfg, identity, read)
    except _READ_ERRORS:
        result["reason"] = "expiry_archive_unavailable"
        return False
    if not active_matches and archive is None:
        if isinstance(active_claim, dict) and active_claim.get("attempt_id") == identity.attempt_id:
            if not (
                active_claim.get("reservation_id") == identity.reservation_id
                and type(active_claim.get("fencing_token")) is int
                and active_claim["fencing_token"] > identity.fencing_token
            ):
                result["reason"] = "attempt_identity_changed"
                return False
        result.update(reason="not_prelaunch", project_outcome="missing")
        return False
    if (
        not active_matches
        and isinstance(active_claim, dict)
        and active_claim.get("reservation_id") == identity.reservation_id
    ):
        result["reason"] = "reservation_identity_changed"
        return False
    claim = active_claim if active_matches else archive
    number = claim.get("attempt_number")
    if type(number) is not int or number < 1:
        result["reason"] = "attempt_identity_changed"
        return False
    if active_matches and (
        task.task_id != identity.task_id
        or task.state.get("projection") != "running"
        or task.attempt_control.get("current_attempt_id") != identity.attempt_id
        or task.attempt_control.get("current_attempt_number") != number
    ):
        result["reason"] = "attempt_identity_changed"
        return False
    attempt_file = attempt_path(cfg.shared_root, identity.task_id, number)
    try:
        attempt = AttemptRecord.from_dict(read(attempt_file))
    except _READ_ERRORS:
        result["reason"] = "attempt_unavailable"
        return False
    if not _attempt_matches(cfg, attempt, claim, identity):
        result["reason"] = "attempt_identity_changed"
        return False
    result["attempt_phase"] = attempt.phase
    claimed_at = _timestamp(claim.get("claimed_at"))
    if claimed_at is not None:
        result["age_seconds"] = max(0.0, (datetime.now(timezone.utc) - claimed_at).total_seconds())
    if blocker := _launch_result(cfg, identity.task_id, claim, attempt, result):
        result["reason"] = blocker
        return False
    is_cancelled = attempt.phase == "cancelled" and attempt.result.get("reason") == _EXPIRY
    if archive is not None and archive.get("prelaunch_expiry_target") != target:
        result["reason"] = "expiry_archive_conflict"
        return False
    if not active_matches:
        if not is_cancelled or claim.get("launch_state") != "claimed":
            result["reason"] = "expiry_archive_conflict"
            return False
        result.update(reason="cleanup_pending", project_outcome="already_requeued", machine_outcome="cleanup_pending")
        finished_at = _timestamp(attempt.timestamps.get("finished_at"))
        if finished_at is not None:
            result["overdue_seconds"] = max(0.0, (datetime.now(timezone.utc) - finished_at).total_seconds())
    try:
        if not _binding_current(cfg, identity, runtime_root, runtime_id, read):
            result["reason"] = "cleanup_pending" if not active_matches else "runtime_binding_changed"
            return False
        state, release_reason = _reservation_state(cfg, identity, runtime_root, read)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        result["reason"] = "cleanup_pending" if not active_matches else "reservation_unavailable"
        return False
    if not active_matches:
        if state == "released" and release_reason == _EXPIRY:
            result.update(reason=_EXPIRY, machine_outcome="already_released")
            return False
        if state not in {"active", "release_pair"} or (state == "release_pair" and release_reason != _EXPIRY):
            return False
        if (
            apply
            and task is not None
            and (
                active_claim is None
                and task.state.get("projection") == "queued"
                and task.state.get("reason") == _EXPIRY
                and task.attempt_control.get("current_attempt_number") == number
            )
        ):
            # Finish a crash between the Task truth write and ready publication.
            with fenced_mutations(cfg.shared_root, before_mutation):
                commit_ready_publication(cfg, task)
        result["actionable"] = not apply
        return apply
    if state != "active":
        result["reason"] = "reservation_identity_changed" if state != "absent" else "reservation_missing"
        return False
    if (
        claim.get("launch_state") != "claimed"
        or (attempt.phase != "claimed" and not (is_cancelled and archive is not None))
        or task.spec.is_cpu_only != (identity.cpu_slots is not None)
        or task.placement_runtime.get("queue_scope") != claim.get("queue_origin")
        or claim.get("queue_origin") not in {"home", "shared"}
    ):
        result["reason"] = "launch_evidence_conflict"
        return False
    if claim.get("authority_mode") != "bounded_lease":
        result["reason"] = "waiting_for_launch"
        return False
    if attempt.authority_mode != "bounded_lease":
        result["reason"] = "launch_evidence_conflict"
        return False
    reason = _clock_result(cfg, claim, attempt, read, acquire_clock, result)
    result["reason"] = reason
    if reason != "eligible_for_recovery":
        return False
    if not apply:
        result["actionable"] = True
        return False
    if not _commit_expiry(
        cfg,
        task,
        claim,
        attempt,
        target,
        attempt_file,
        archive_file,
        before_mutation=before_mutation,
        task_writer=task_writer,
    ):
        result["reason"] = "expiry_archive_unavailable"
        return False
    result.update(reason=_EXPIRY, project_outcome="requeued", attempt_phase="cancelled")
    return True


def reconcile_prelaunch(
    cfg: RootConfig,
    identity: ReservationIdentity,
    *,
    runtime_root: Path,
    runtime_id: str,
    apply: bool,
    before_shared_mutation: Callable[[], None] | None = None,
    read_record: _RecordReader | None = None,
    defer_machine_release: bool = False,
    _acquire_clock: Callable = clock_capability,
    _task_writer: Callable = save_task,
) -> dict[str, Any]:
    """Observe or advance expiry for one exact reservation identity.

    Preview uses a fixed set of immutable record observations and does not
    acquire locks or refresh clock evidence. Applying re-reads the same claim
    evidence under the Project authority locks and releases capacity afterward.
    """
    result: dict[str, Any] = {
        "reason": "waiting_for_launch",
        "project_outcome": "retained",
        "machine_outcome": "retained",
        "gpu_ids": list(identity.gpu_ids),
        "task_id": identity.task_id,
        "attempt_id": identity.attempt_id,
        "attempt_phase": None,
        "runtime_id": runtime_id,
        "reservation_id": identity.reservation_id,
        "age_seconds": None,
        "safe_at": None,
        "overdue_seconds": 0.0,
        "launch_evidence": {},
        "expected_target": expected_target(identity, runtime_id),
        "actionable": False,
    }
    for field in ("task_id", "attempt_id", "reservation_id"):
        if getattr(identity, field) in {".", ".."}:
            raise ValueError(f"{field} is invalid.")
        validate_identifier(getattr(identity, field), field)
    if type(identity.fencing_token) is not int or identity.fencing_token < 1:
        raise ValueError("pre-launch recovery requires a positive fencing token.")
    runtime_root = Path(runtime_root)
    read = _read_record if apply or read_record is None else read_record
    filename = task_path(cfg.shared_root, identity.task_id)
    try:
        document = _optional_record(read, filename)
        task = None if document is None else TaskRecord.from_dict(document)
    except _READ_ERRORS:
        result["reason"] = "task_unavailable"
        return result
    group_name = task.group_name if task is not None else None
    locks = authority_locks(cfg, task) if apply and task is not None else nullcontext()
    with ExitStack() as ownership:
        if apply and runtime_id:
            # Takeover uses schema, registration, then shared machine locks.
            ownership.enter_context(schema_writer_lock(cfg, require_narrow=True))
            ownership.enter_context(exclusive(shared_paths(cfg.shared_root)["locks"] / "registrations.lock"))
            ownership.enter_context(machine_lock(cfg.shared_root, cfg.machine_name))
        with locks:
            if apply and task is not None:
                try:
                    task = load_task(cfg, identity.task_id)
                    if task.group_name != group_name:
                        result["reason"] = "task_identity_changed"
                        return result
                except _READ_ERRORS:
                    result["reason"] = "task_unavailable"
                    return result
            try:
                should_release = _evaluate(
                    cfg,
                    task,
                    identity,
                    runtime_root,
                    runtime_id,
                    result,
                    read,
                    acquire_clock=_acquire_clock if apply else None,
                    before_mutation=before_shared_mutation,
                    task_writer=_task_writer,
                    apply=apply,
                )
            except _READ_ERRORS:
                if apply:
                    raise
                result["reason"] = "prelaunch_evidence_unavailable"
                return result
        if should_release and defer_machine_release:
            result.update(reason=_EXPIRY, machine_outcome="release_pending")
        elif should_release:
            _release_expired(cfg, identity, runtime_root, runtime_id, result, before_shared_mutation)
    return result


def expire_standalone_claim(
    cfg: RootConfig,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    runtime_root: Path,
    *,
    acquire_clock: Callable,
    task_writer: Callable,
) -> bool:
    """Resolve the scheduler's historical standalone caller to an exact candidate."""
    task = load_task(cfg, task_id)
    claim = task.claim_control.get("active_claim") or {}
    reservation_id = claim.get("reservation_id")
    if (
        claim.get("attempt_id") != attempt_id
        or claim.get("fencing_token") != fencing_token
        or claim.get("machine_name") != cfg.machine_name
        or not isinstance(reservation_id, str)
        or Path(reservation_id).name != reservation_id
    ):
        return False
    directory = "cpu_active" if task.spec.is_cpu_only else "active"
    try:
        record = _read_record(local_paths(runtime_root)[directory] / f"{reservation_id}.json")["reservation"]
        identity = ReservationIdentity.from_record(record)
    except _READ_ERRORS:
        return False
    if (identity.task_id, identity.attempt_id, identity.fencing_token) != (task_id, attempt_id, fencing_token):
        return False
    outcome = reconcile_prelaunch(
        cfg,
        identity,
        runtime_root=runtime_root,
        runtime_id="",
        apply=True,
        _acquire_clock=acquire_clock,
        _task_writer=task_writer,
    )
    return outcome["project_outcome"] == "requeued"
