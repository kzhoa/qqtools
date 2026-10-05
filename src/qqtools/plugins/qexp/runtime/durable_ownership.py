"""Task-first durable ownership transitions for active qexp Attempts.

An authorization or exact running-process adoption changes the shared Task
before its companion Attempt.  The receipt stored in the Task makes that
ordering replayable: a later writer can either finish the matching Attempt or
fail closed when any part of the transition identity changed.

QQTOOLS-COMPAT-0021: durable ownership receipts bridge the Task-first
authorization write to the existing Attempt projection during this rollout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..config_types import RootConfig
from .paths import attempt_path, task_path
from .records import AttemptRecord, TaskRecord, utc_now
from .store import atomic_replace

OwnershipTargetPhase = Literal["starting", "running"]
_LEASE_CLAIM_FIELDS = (
    "lease_expires_at",
    "clock_error_bound_seconds",
    "clock_provider",
    "clock_observation_id",
)
_PROCESS_IDENTITY_FIELDS = (
    "wrapper_pid",
    "wrapper_start_time_ticks",
    "process_group_id",
    "process_group_start_time_ticks",
)
_RECEIPT_VERSION = 1


class OwnershipTransitionConflict(RuntimeError):
    """Raised when a replay no longer identifies the same Task/Attempt pair."""


@dataclass(frozen=True, slots=True)
class OwnershipTransitionResult:
    """Immutable outcome of one durable ownership transition."""

    attempt: AttemptRecord
    operation_id: str
    task_revision: int
    attempt_digest: str
    outcome: Literal["committed", "repaired", "idempotent"]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _persisted_json_bytes(value: dict[str, Any]) -> bytes:
    """Encode the same JSON representation used by ``atomic_replace``."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _clone_task(task: TaskRecord) -> TaskRecord:
    return TaskRecord.from_dict(copy.deepcopy(task.to_dict()))


def _clone_attempt(attempt: AttemptRecord) -> AttemptRecord:
    return AttemptRecord.from_dict(copy.deepcopy(attempt.to_dict()))


def _sync_task(target: TaskRecord, source: TaskRecord) -> None:
    for name in (
        "task_id",
        "group_name",
        "group_membership_sequence",
        "submission_operation_id",
        "name",
        "depends_on_task_ids",
        "ready_generation",
        "spec",
        "placement_policy",
        "placement_runtime",
        "state",
        "control",
        "attempt_control",
        "claim_control",
        "meta",
    ):
        setattr(target, name, copy.deepcopy(getattr(source, name)))


def _sync_attempt(target: AttemptRecord, source: AttemptRecord) -> None:
    for name in (
        "attempt_id",
        "task_id",
        "attempt_number",
        "phase",
        "machine_name",
        "assigned_gpus",
        "reservation_id",
        "current_fencing_token",
        "token_history",
        "lease",
        "authority_mode",
        "authorization",
        "process",
        "termination",
        "timestamps",
        "result",
        "meta",
    ):
        setattr(target, name, copy.deepcopy(getattr(source, name)))


def process_identity_matches(attempt: AttemptRecord, process_identity: Mapping[str, Any]) -> bool:
    """Return whether a process identity is complete and exactly current."""
    if not isinstance(process_identity, Mapping) or set(process_identity) != set(_PROCESS_IDENTITY_FIELDS):
        return False
    if any(attempt.process.get(field) != process_identity.get(field) for field in _PROCESS_IDENTITY_FIELDS):
        return False
    wrapper_pid = process_identity.get("wrapper_pid")
    wrapper_ticks = process_identity.get("wrapper_start_time_ticks")
    wrapper_matches = (wrapper_pid is None and wrapper_ticks is None) or (
        type(wrapper_pid) is int and wrapper_pid > 0 and type(wrapper_ticks) is int and wrapper_ticks >= 0
    )
    return (
        wrapper_matches
        and type(process_identity.get("process_group_id")) is int
        and process_identity["process_group_id"] > 0
        and type(process_identity.get("process_group_start_time_ticks")) is int
        and process_identity["process_group_start_time_ticks"] >= 0
    )


def _read_attempt_source(
    cfg: RootConfig,
    task: TaskRecord,
    attempt: AttemptRecord,
) -> tuple[AttemptRecord, bytes, str]:
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        persisted = AttemptRecord.from_dict(value)
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise OwnershipTransitionConflict("durable ownership source Attempt is unavailable") from exc
    if persisted.to_dict() != attempt.to_dict():
        raise OwnershipTransitionConflict("durable ownership source Attempt changed")
    return persisted, raw, _digest(raw)


def _lease_history(claim: Mapping[str, Any], attempt: AttemptRecord) -> dict[str, Any]:
    return {
        "claim": {field: copy.deepcopy(claim.get(field)) for field in _LEASE_CLAIM_FIELDS},
        "attempt": {
            "claimed_at": copy.deepcopy(attempt.lease.get("claimed_at")),
            "renewed_at": copy.deepcopy(attempt.lease.get("renewed_at")),
            "expires_at": copy.deepcopy(attempt.lease.get("expires_at")),
            "clock_evidence": copy.deepcopy(attempt.lease.get("clock_evidence")),
        },
    }


def _receipt_payload(
    *,
    task: TaskRecord,
    attempt: AttemptRecord,
    source_attempt_digest: str,
    target_phase: OwnershipTargetPhase,
    launch_id: str,
    authorized_at: str,
    launch_handoff_timeout_seconds: int | float | None,
    source_lease_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "receipt_version": _RECEIPT_VERSION,
        "source_task_revision": task.meta["revision"],
        "source_attempt_digest": source_attempt_digest,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "machine_name": attempt.machine_name,
        "reservation_id": attempt.reservation_id,
        "launch_id": launch_id,
        "source_authority_mode": attempt.authority_mode,
        "source_phase": attempt.phase,
        "target_authority_mode": "holder_bound",
        "target_phase": target_phase,
        "authorization_timestamp": authorized_at,
        "launch_handoff_timeout_seconds": launch_handoff_timeout_seconds,
        "source_active_lease_evidence": copy.deepcopy(dict(source_lease_evidence)),
    }


def _make_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(copy.deepcopy(dict(payload)))
    receipt["operation_id"] = _digest(_canonical_bytes(payload))
    # These aliases make the permanent receipt self-describing to older local
    # diagnostics without changing the canonical operation identity.
    receipt["authorized_at"] = receipt["authorization_timestamp"]
    receipt["launch_authorized_at"] = receipt["authorization_timestamp"]
    receipt["source_lease_evidence"] = copy.deepcopy(receipt["source_active_lease_evidence"])
    return receipt


def _checked_receipt(receipt: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise OwnershipTransitionConflict("ownership transition receipt is malformed")
    required = {
        "receipt_version",
        "source_task_revision",
        "source_attempt_digest",
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "machine_name",
        "reservation_id",
        "launch_id",
        "source_authority_mode",
        "source_phase",
        "target_authority_mode",
        "target_phase",
        "authorization_timestamp",
        "launch_handoff_timeout_seconds",
        "source_active_lease_evidence",
        "operation_id",
    }
    if not required.issubset(receipt):
        raise OwnershipTransitionConflict("ownership transition receipt is incomplete")
    payload = {key: copy.deepcopy(receipt[key]) for key in required if key != "operation_id"}
    operation_id = receipt.get("operation_id")
    if type(operation_id) is not str or _digest(_canonical_bytes(payload)) != operation_id:
        raise OwnershipTransitionConflict("ownership transition receipt identity is invalid")
    if (
        receipt.get("receipt_version") != _RECEIPT_VERSION
        or receipt.get("target_authority_mode") != "holder_bound"
        or receipt.get("target_phase") not in {"starting", "running"}
        or receipt.get("source_authority_mode") not in {"bounded_lease", "holder_bound"}
        or receipt.get("source_phase") not in {"claimed", "starting", "running"}
        or type(receipt.get("source_task_revision")) is not int
        or receipt["source_task_revision"] < 0
        or type(receipt.get("source_attempt_digest")) is not str
        or len(receipt["source_attempt_digest"]) != 64
        or any(character not in "0123456789abcdef" for character in receipt["source_attempt_digest"])
        or not _valid_utc_timestamp(receipt.get("authorization_timestamp"))
        or (receipt.get("target_phase") == "starting" and receipt.get("source_phase") not in {"claimed", "starting"})
        or (receipt.get("target_phase") == "running" and receipt.get("source_authority_mode") != "bounded_lease")
        or (receipt.get("target_phase") == "running" and receipt.get("source_phase") != "running")
    ):
        raise OwnershipTransitionConflict("ownership transition receipt semantics are invalid")
    if receipt.get("authorized_at", receipt["authorization_timestamp"]) != receipt["authorization_timestamp"]:
        raise OwnershipTransitionConflict("ownership transition receipt timestamp is inconsistent")
    if receipt.get("launch_authorized_at", receipt["authorization_timestamp"]) != receipt["authorization_timestamp"]:
        raise OwnershipTransitionConflict("ownership transition receipt launch timestamp is inconsistent")
    if (
        receipt.get("source_lease_evidence", receipt["source_active_lease_evidence"])
        != receipt["source_active_lease_evidence"]
    ):
        raise OwnershipTransitionConflict("ownership transition receipt lease history is inconsistent")
    return dict(receipt), payload


def _identity_matches(task: TaskRecord, attempt: AttemptRecord, claim: Mapping[str, Any]) -> bool:
    return (
        task.task_id == attempt.task_id
        and task.attempt_control.get("current_attempt_id") == attempt.attempt_id
        and task.attempt_control.get("current_attempt_number") == attempt.attempt_number
        and claim.get("attempt_id") == attempt.attempt_id
        and claim.get("attempt_number") == attempt.attempt_number
        and claim.get("fencing_token") == attempt.current_fencing_token
        and claim.get("machine_name") == attempt.machine_name
        and claim.get("reservation_id") == attempt.reservation_id
        and task.state.get("projection") == "running"
    )


def _receipt_identity_matches(receipt: Mapping[str, Any], task: TaskRecord, attempt: AttemptRecord) -> bool:
    return all(
        receipt.get(field) == expected
        for field, expected in (
            ("task_id", task.task_id),
            ("attempt_id", attempt.attempt_id),
            ("attempt_number", attempt.attempt_number),
            ("fencing_token", attempt.current_fencing_token),
            ("machine_name", attempt.machine_name),
            ("reservation_id", attempt.reservation_id),
        )
    )


def _target_matches(
    task: TaskRecord,
    attempt: AttemptRecord,
    receipt: Mapping[str, Any],
    target_phase: OwnershipTargetPhase,
) -> bool:
    claim = task.claim_control.get("active_claim") or {}
    timeout = receipt.get("launch_handoff_timeout_seconds")
    claim_timeout_matches = timeout is None or claim.get("launch_handoff_timeout_seconds") == timeout
    attempt_timeout_matches = timeout is None or attempt.authorization.get("launch_handoff_timeout_seconds") == timeout
    return (
        _identity_matches(task, attempt, claim)
        and claim.get("authority_mode") == "holder_bound"
        and claim.get("launch_state") == target_phase
        and claim.get("launch_id") == receipt.get("launch_id")
        and claim.get("launch_authorized_at") == receipt.get("authorization_timestamp")
        and all(claim.get(field) is None for field in _LEASE_CLAIM_FIELDS)
        and claim.get("ownership_transition") == dict(receipt)
        and attempt.authority_mode == "holder_bound"
        and attempt.phase == target_phase
        and attempt.authorization.get("launch_id") == receipt.get("launch_id")
        and attempt.timestamps.get("launch_authorized_at") == receipt.get("authorization_timestamp")
        and attempt.lease.get("expires_at") is None
        and attempt.lease.get("clock_evidence") is None
        and claim_timeout_matches
        and attempt_timeout_matches
    )


def _source_matches_receipt(
    attempt: AttemptRecord,
    source_attempt_digest: str,
    receipt: Mapping[str, Any],
) -> bool:
    return (
        source_attempt_digest == receipt.get("source_attempt_digest")
        and attempt.authority_mode == receipt.get("source_authority_mode")
        and attempt.phase == receipt.get("source_phase")
        and attempt.attempt_id == receipt.get("attempt_id")
        and attempt.attempt_number == receipt.get("attempt_number")
        and attempt.current_fencing_token == receipt.get("fencing_token")
        and attempt.machine_name == receipt.get("machine_name")
        and attempt.reservation_id == receipt.get("reservation_id")
    )


def _apply_projection(
    task: TaskRecord,
    attempt: AttemptRecord,
    receipt: Mapping[str, Any],
    target_phase: OwnershipTargetPhase,
) -> None:
    claim = task.claim_control.get("active_claim")
    if not isinstance(claim, dict):
        raise OwnershipTransitionConflict("active claim is missing")
    claim["launch_state"] = target_phase
    claim["authority_mode"] = "holder_bound"
    claim["launch_id"] = receipt["launch_id"]
    claim["launch_authorized_at"] = receipt["authorization_timestamp"]
    for field in _LEASE_CLAIM_FIELDS:
        claim[field] = None
    timeout = receipt.get("launch_handoff_timeout_seconds")
    if timeout is not None:
        claim["launch_handoff_timeout_seconds"] = timeout
    claim["ownership_transition"] = copy.deepcopy(dict(receipt))

    attempt.phase = target_phase
    attempt.authority_mode = "holder_bound"
    attempt.authorization["launch_id"] = receipt["launch_id"]
    timeout = receipt.get("launch_handoff_timeout_seconds")
    if timeout is not None:
        attempt.authorization["launch_handoff_timeout_seconds"] = timeout
    attempt.timestamps["launch_authorized_at"] = receipt["authorization_timestamp"]
    attempt.lease["expires_at"] = None
    attempt.lease["clock_evidence"] = None


def transition_attempt_ownership(
    cfg: RootConfig,
    task: TaskRecord,
    attempt: AttemptRecord,
    *,
    target_phase: OwnershipTargetPhase,
    launch_id: str | None = None,
    authorized_at: str | None = None,
    launch_handoff_timeout_seconds: int | float | None = None,
    mutation_fence: Callable[[], None] | None = None,
    task_writer: Callable[[TaskRecord], None] | None = None,
    attempt_writer: Callable[[AttemptRecord], None] | None = None,
    replay_only: bool = False,
    expected_task_revision: int | None = None,
    expected_attempt_digest: str | None = None,
) -> OwnershipTransitionResult:
    """Commit or replay a Task-first bounded-to-holder ownership transition.

    Callers own the Group/Task authority locks.  ``task_writer`` and
    ``attempt_writer`` are intentionally injected so direct scheduling can use
    observation-aware Task persistence while Project-I/O uses its isolated
    atomic replacement path.
    """
    if target_phase not in {"starting", "running"}:
        raise ValueError("ownership transition target phase must be 'starting' or 'running'")
    if type(task.meta.get("revision")) is not int or task.meta["revision"] < 0:
        raise OwnershipTransitionConflict("Task revision is malformed")
    if not _identity_matches(task, attempt, task.claim_control.get("active_claim") or {}):
        raise OwnershipTransitionConflict("Task and Attempt ownership identity changed")
    persisted_attempt, source_bytes, source_attempt_digest = _read_attempt_source(cfg, task, attempt)
    if expected_attempt_digest is not None and expected_attempt_digest != source_attempt_digest:
        raise OwnershipTransitionConflict("ownership transition source Attempt digest changed")
    if expected_task_revision is not None and type(expected_task_revision) is not int:
        raise ValueError("expected Task revision must be an integer")

    task_work = _clone_task(task)
    attempt_work = _clone_attempt(persisted_attempt)
    claim = task_work.claim_control.get("active_claim") or {}
    raw_receipt = claim.get("ownership_transition")
    if raw_receipt is not None:
        receipt, _payload = _checked_receipt(raw_receipt)
        if not _receipt_identity_matches(receipt, task_work, attempt_work):
            raise OwnershipTransitionConflict("ownership transition receipt identity changed")
        if expected_task_revision is not None and receipt.get("source_task_revision") != expected_task_revision:
            raise OwnershipTransitionConflict("ownership transition Task revision changed")
        if expected_attempt_digest is not None and receipt.get("source_attempt_digest") != expected_attempt_digest:
            raise OwnershipTransitionConflict("ownership transition source digest changed")
        if receipt.get("target_phase") != target_phase:
            raise OwnershipTransitionConflict("ownership transition target phase changed")
        if launch_id is not None and launch_id != receipt.get("launch_id"):
            raise OwnershipTransitionConflict("ownership transition launch identity changed")
        if authorized_at is not None and authorized_at != receipt.get("authorization_timestamp"):
            raise OwnershipTransitionConflict("ownership transition timestamp changed")
        timeout = receipt.get("launch_handoff_timeout_seconds")
        if launch_handoff_timeout_seconds is not None and launch_handoff_timeout_seconds != timeout:
            raise OwnershipTransitionConflict("ownership transition handoff timeout changed")
        if _target_matches(task_work, attempt_work, receipt, target_phase):
            return OwnershipTransitionResult(
                attempt_work,
                receipt["operation_id"],
                task_work.meta["revision"],
                _digest(source_bytes),
                "idempotent",
            )
        if not _source_matches_receipt(attempt_work, source_attempt_digest, receipt):
            raise OwnershipTransitionConflict("ownership transition source Attempt changed")
        _apply_projection(task_work, attempt_work, receipt, target_phase)
        if attempt_writer is None:
            attempt_writer = lambda value: atomic_replace(
                attempt_path(cfg.shared_root, value.task_id, value.attempt_number), value.to_dict()
            )
        if mutation_fence is not None:
            mutation_fence()
        attempt_writer(attempt_work)
        _sync_attempt(attempt, attempt_work)
        return OwnershipTransitionResult(
            attempt_work,
            receipt["operation_id"],
            task_work.meta["revision"],
            _digest(_persisted_json_bytes(attempt_work.to_dict())),
            "repaired",
        )

    if replay_only:
        raise OwnershipTransitionConflict("replay-only ownership renewal has no matching receipt")
    if expected_task_revision is not None and task.meta["revision"] != expected_task_revision:
        raise OwnershipTransitionConflict("ownership transition Task revision changed")
    if expected_attempt_digest is not None and source_attempt_digest != expected_attempt_digest:
        raise OwnershipTransitionConflict("ownership transition source digest changed")
    claim = task_work.claim_control.get("active_claim") or {}
    source_mode = attempt_work.authority_mode
    if source_mode not in {"bounded_lease", "holder_bound"} or claim.get("authority_mode") != source_mode:
        raise OwnershipTransitionConflict("ownership transition source authority mode changed")
    legacy_starting_import = (
        target_phase == "starting"
        and claim.get("launch_state") == "starting"
        and source_mode == "bounded_lease"
        and attempt_work.phase == "starting"
        and isinstance(claim.get("launch_id"), str)
        and bool(claim.get("launch_id"))
        and claim.get("launch_id") == attempt_work.authorization.get("launch_id")
        and _valid_utc_timestamp(claim.get("launch_authorized_at"))
        and claim.get("launch_authorized_at") == attempt_work.timestamps.get("launch_authorized_at")
        and claim.get("launch_handoff_timeout_seconds")
        == attempt_work.authorization.get("launch_handoff_timeout_seconds")
    )
    if target_phase == "starting":
        if not legacy_starting_import and (claim.get("launch_state") != "claimed" or attempt_work.phase != "claimed"):
            raise OwnershipTransitionConflict("starting ownership transition is not an uncommitted claim")
        if legacy_starting_import:
            if launch_id is not None and launch_id != claim.get("launch_id"):
                raise OwnershipTransitionConflict("legacy starting launch identity changed")
            if authorized_at is not None and authorized_at != claim.get("launch_authorized_at"):
                raise OwnershipTransitionConflict("legacy starting timestamp changed")
            if launch_handoff_timeout_seconds is not None and launch_handoff_timeout_seconds != claim.get(
                "launch_handoff_timeout_seconds"
            ):
                raise OwnershipTransitionConflict("legacy starting handoff timeout changed")
    if target_phase == "running" and attempt_work.phase != "running":
        raise OwnershipTransitionConflict("ownership transition source phase changed")
    launch_id = launch_id or claim.get("launch_id") or attempt_work.authorization.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        if replay_only:
            raise OwnershipTransitionConflict("ownership transition launch identity is missing")
        launch_id = uuid.uuid4().hex
    authorized_at = (
        authorized_at or claim.get("launch_authorized_at") or attempt_work.timestamps.get("launch_authorized_at")
    )
    if not isinstance(authorized_at, str) or not _valid_utc_timestamp(authorized_at):
        authorized_at = utc_now()
    timeout = launch_handoff_timeout_seconds
    if timeout is None:
        timeout = claim.get(
            "launch_handoff_timeout_seconds", attempt_work.authorization.get("launch_handoff_timeout_seconds")
        )
    if timeout is not None and (type(timeout) not in {int, float} or timeout < 0):
        raise OwnershipTransitionConflict("ownership transition handoff timeout is invalid")
    payload = _receipt_payload(
        task=task_work,
        attempt=attempt_work,
        source_attempt_digest=source_attempt_digest,
        target_phase=target_phase,
        launch_id=launch_id,
        authorized_at=authorized_at,
        launch_handoff_timeout_seconds=timeout,
        source_lease_evidence=_lease_history(claim, attempt_work),
    )
    receipt = _make_receipt(payload)
    _apply_projection(task_work, attempt_work, receipt, target_phase)
    task_work.meta["revision"] += 1
    task_work.meta["updated_at"] = authorized_at
    if task_writer is None:
        task_writer = lambda value: atomic_replace(task_path(cfg.shared_root, value.task_id), value.to_dict())
    if attempt_writer is None:
        attempt_writer = lambda value: atomic_replace(
            attempt_path(cfg.shared_root, value.task_id, value.attempt_number), value.to_dict()
        )
    if mutation_fence is not None:
        mutation_fence()
    task_writer(task_work)
    if mutation_fence is not None:
        mutation_fence()
    attempt_writer(attempt_work)
    _sync_task(task, task_work)
    _sync_attempt(attempt, attempt_work)
    return OwnershipTransitionResult(
        attempt,
        receipt["operation_id"],
        task_work.meta["revision"],
        _digest(_persisted_json_bytes(attempt_work.to_dict())),
        "committed",
    )


__all__ = [
    "OwnershipTransitionConflict",
    "OwnershipTransitionResult",
    "process_identity_matches",
    "transition_attempt_ownership",
]
