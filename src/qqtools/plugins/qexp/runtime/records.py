"""Validated records for the schema-6 qexp runtime."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

SCHEMA_VERSION = 6
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
MACHINE_PATTERN = TASK_ID_PATTERN
GROUP_PATTERN = TASK_ID_PATTERN

TASK_PHASES = ("queued", "running", "succeeded", "failed", "cancelled", "blocked")
ATTEMPT_PHASES = ("claimed", "starting", "running", "succeeded", "failed", "cancelled", "orphaned")
QUEUE_SCOPES = ("home", "shared")
SHARING_MODES = ("private", "spillover")
GROUP_ADMISSION_STATES = ("open", "sealed")
GROUP_DISPATCH_STATES = ("active", "paused")
WORKER_STATES = ("active", "borrow", "draining", "removing")
WORKER_ROLES = ("primary", "borrow")
SUBMISSION_STATES = ("preparing", "committing", "committed", "aborted", "blocked")
AUTHORITY_MODES = ("bounded_lease", "holder_bound", "legacy_migrated")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or not TASK_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' and '-'.")
    return value


def validate_group_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 64 or not GROUP_PATTERN.fullmatch(value):
        raise ValueError("group must be a non-empty identifier of at most 64 characters.")
    if value[0] in ".-" or value in {"experiments", "qqtools_internal"}:
        raise ValueError(f"group name {value!r} is reserved or invalid.")
    return value


def _meta(machine: str, revision: int = 1, created_at: str | None = None) -> dict[str, Any]:
    now = created_at or utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "created_at": now,
        "updated_at": now,
        "updated_by": {"actor_type": "cli", "machine_name": machine, "process_id": str(uuid.uuid4())},
    }


def _check_meta(data: dict[str, Any]) -> None:
    required = {"schema_version", "revision", "created_at", "updated_at", "updated_by"}
    if set(data) != required:
        raise ValueError("record meta has missing or unknown fields.")
    if data["schema_version"] != SCHEMA_VERSION or not isinstance(data["revision"], int):
        raise ValueError("record schema version or revision is invalid.")


@dataclass(slots=True)
class TaskSpec:
    command: list[str]
    working_directory: str
    requested_gpus: int

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(item, str) for item in self.command):
            raise ValueError("command must be a non-empty list of strings.")
        if not self.working_directory.startswith("/"):
            raise ValueError("working_directory must be absolute.")
        if not isinstance(self.requested_gpus, int) or self.requested_gpus <= 0:
            raise ValueError("requested_gpus must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return {"command": list(self.command), "working_directory": self.working_directory, "requested_gpus": self.requested_gpus}


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    group_name: str | None
    group_membership_sequence: int | None
    submission_operation_id: str | None
    name: str | None
    ready_generation: int
    spec: TaskSpec
    placement_policy: dict[str, Any]
    placement_runtime: dict[str, Any]
    state: dict[str, Any]
    control: dict[str, Any]
    attempt_control: dict[str, Any]
    claim_control: dict[str, Any]
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        validate_identifier(self.task_id, "task_id")
        validate_group_name(self.group_name)
        _check_meta(self.meta)
        if type(self.ready_generation) is not int or self.ready_generation < 0:
            raise ValueError("task ready_generation must be a non-negative integer.")
        if self.state.get("projection") not in TASK_PHASES:
            raise ValueError("task state projection is invalid.")
        if self.placement_policy.get("sharing_mode") not in SHARING_MODES:
            raise ValueError("task sharing_mode is invalid.")
        if self.placement_policy["sharing_mode"] == "private" and self.placement_runtime["queue_scope"] == "shared":
            raise ValueError("private tasks cannot enter the shared queue.")
        proof = self.placement_runtime.get("offer_clock_evidence")
        if proof is not None:
            required = {"creator_observation", "deadline_monotonic_at"}
            observation_fields = {
                "observation_id", "provider", "observed_at", "monotonic_observed_at", "boot_id",
                "lower_error_seconds", "upper_error_seconds", "max_drift_rate", "provider_margin_seconds",
            }
            observation = proof.get("creator_observation") if isinstance(proof, dict) else None
            if (not isinstance(proof, dict) or set(proof) != required or not isinstance(observation, dict)
                    or set(observation) != observation_fields
                    or not isinstance(proof["deadline_monotonic_at"], (int, float))):
                raise ValueError("timed offer clock proof is invalid.")
        claim = self.claim_control.get("active_claim")
        if claim is not None:
            if not isinstance(claim, dict) or claim.get("authority_mode") not in {
                    "bounded_lease", "holder_bound"}:
                raise ValueError("active claim authority mode is invalid.")
            if claim["authority_mode"] == "bounded_lease":
                required = {"clock_error_bound_seconds", "clock_provider", "clock_observation_id", "lease_expires_at"}
                if not required.issubset(claim):
                    raise ValueError("bounded lease claim lacks clock evidence.")
            elif any(claim.get(key) is not None for key in (
                    "clock_error_bound_seconds", "clock_provider", "clock_observation_id", "lease_expires_at")):
                raise ValueError("holder-bound claim cannot contain lease evidence.")

    @classmethod
    def new(cls, *, task_id: str, machine: str, spec: TaskSpec, group_name: str | None = None,
            name: str | None = None, sharing_mode: str = "private", fallback_machines: str | list[str] = "group",
            offer_after_seconds: int | None = None, offer_eligible_at: str | None = None,
            offer_clock_evidence: dict[str, Any] | None = None,
            operation_id: str | None = None) -> "TaskRecord":
        validate_identifier(task_id, "task_id")
        validate_identifier(machine, "machine_name")
        validate_group_name(group_name)
        if group_name is None and sharing_mode != "private":
            raise ValueError("ungrouped tasks must use private placement.")
        if sharing_mode not in SHARING_MODES:
            raise ValueError("sharing_mode must be 'private' or 'spillover'.")
        if offer_after_seconds is not None and sharing_mode != "spillover":
            raise ValueError("offer_after_seconds requires spillover placement.")
        now = utc_now()
        if offer_after_seconds is not None:
            if not isinstance(offer_after_seconds, int) or offer_after_seconds < 0:
                raise ValueError("offer_after_seconds must be a non-negative integer or null.")
            if not isinstance(offer_eligible_at, str) or offer_clock_evidence is None:
                raise ValueError("timed offer requires a deadline and qualified clock proof.")
        elif offer_eligible_at is not None or offer_clock_evidence is not None:
            raise ValueError("only timed offers may contain clock proof data.")
        return cls(
            task_id=task_id, group_name=group_name, group_membership_sequence=None,
            submission_operation_id=operation_id, name=name, ready_generation=0, spec=spec,
            placement_policy={"home_machine": machine, "sharing_mode": sharing_mode,
                              "fallback_constraint": fallback_machines, "offer_after_seconds": offer_after_seconds},
            placement_runtime={"queue_scope": "home", "queued_home_at": now,
                               "offer_eligible_at": offer_eligible_at, "offer_clock_evidence": offer_clock_evidence,
                               "offered_at": None, "offer_reason": None, "offered_by": None},
            state={"projection": "queued", "reason": None},
            control={"cancellation_requested_at": None, "cancellation_operation_id": None,
                     "terminate_running": False, "requested_by": None, "termination_acknowledged_at": None,
                     "termination_result": None, "cleanup_operation_id": None,
                     "cleanup_state": None},
            attempt_control={"next_attempt_number": 1, "current_attempt_id": None, "current_attempt_number": None},
            claim_control={"fencing_epoch": 0, "active_claim": None}, meta=_meta(machine),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "task": {"task_id": self.task_id, "group_name": self.group_name,
                "group_membership_sequence": self.group_membership_sequence,
                "submission_operation_id": self.submission_operation_id, "name": self.name,
                "ready_generation": self.ready_generation,
                "spec": self.spec.to_dict(), "placement_policy": self.placement_policy,
                "placement_runtime": self.placement_runtime, "state": self.state, "control": self.control,
                "attempt_control": self.attempt_control, "claim_control": self.claim_control}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        task = data["task"]
        return cls(meta=data["meta"], task_id=task["task_id"], group_name=task.get("group_name"),
                   group_membership_sequence=task.get("group_membership_sequence"),
                   submission_operation_id=task.get("submission_operation_id"), name=task.get("name"),
                   ready_generation=task.get("ready_generation", 0),
                   spec=TaskSpec(**task["spec"]), placement_policy=task["placement_policy"],
                   placement_runtime=task["placement_runtime"], state=task["state"], control=task["control"],
                   attempt_control=task["attempt_control"], claim_control=task["claim_control"])


@dataclass(slots=True)
class AttemptRecord:
    attempt_id: str
    task_id: str
    attempt_number: int
    phase: str
    machine_name: str
    assigned_gpus: list[int]
    reservation_id: str
    current_fencing_token: int
    token_history: list[int]
    lease: dict[str, Any]
    authority_mode: str
    authorization: dict[str, Any]
    process: dict[str, Any]
    termination: dict[str, Any]
    timestamps: dict[str, Any]
    result: dict[str, Any]
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        validate_identifier(self.attempt_id, "attempt_id")
        validate_identifier(self.task_id, "task_id")
        validate_identifier(self.machine_name, "machine_name")
        if self.phase not in ATTEMPT_PHASES:
            raise ValueError("attempt phase is invalid.")
        if self.authority_mode not in AUTHORITY_MODES:
            raise ValueError("attempt authority mode is invalid.")
        if self.authority_mode == "legacy_migrated" and self.phase in {
                "claimed", "starting", "running", "orphaned"}:
            raise ValueError("legacy-migrated attempts cannot retain execution authority.")
        _check_meta(self.meta)

    @classmethod
    def claimed(cls, task: TaskRecord, machine: str, gpus: list[int], reservation_id: str, token: int,
                *, authority_mode: str, clock_evidence: dict[str, Any] | None,
                lease_seconds: int = 60, attempt_id: str | None = None) -> "AttemptRecord":
        attempt_id = attempt_id or new_id()
        now = utc_now()
        expires = None if authority_mode == "holder_bound" else (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return cls(attempt_id, task.task_id, task.attempt_control["next_attempt_number"], "claimed", machine,
                   list(gpus), reservation_id, token, [token], {"claimed_at": now, "renewed_at": now,
                   "expires_at": expires, "clock_evidence": clock_evidence}, authority_mode,
                   {"group_name": task.group_name, "group_dispatch_epoch": None,
                   "group_worker_set_epoch": None}, {"wrapper_pid": None, "wrapper_start_time_ticks": None,
                   "process_group_id": None, "process_group_start_time_ticks": None,
                   "tmux_reference": None, "local_process_manifest": "", "log_references": []},
                   {"requested_by_operation_id": None, "requested_at": None, "acknowledged_at": None,
                    "result": None}, {"launch_authorized_at": None, "process_created_at": None,
                    "running_at": None, "orphaned_at": None, "recovered_at": None,
                    "finished_at": None}, {"exit_code": None, "signal": None,
                    "category": None, "reason": None}, _meta(machine))

    def to_dict(self) -> dict[str, Any]:
        values = {key: getattr(self, key) for key in ("attempt_id", "task_id", "attempt_number", "phase",
                 "machine_name", "assigned_gpus", "reservation_id", "current_fencing_token", "token_history",
                 "lease", "authority_mode", "authorization", "process", "termination", "timestamps", "result")}
        return {"meta": self.meta, "attempt": values}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptRecord":
        value = dict(data["attempt"])
        return cls(meta=data["meta"], **value)


def validate_gpu_limit(value: Any, label: str = "gpu_limit_gpus") -> int | None:
    """Validate and return a Group Worker GPU limit."""
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer or null.")
    return value


def normalize_worker_member(worker: dict[str, Any], *, machine: str = "worker") -> dict[str, Any]:
    """Validate one canonical persisted Worker member."""
    if not isinstance(worker, dict):
        raise ValueError(f"{machine!r} Worker record must be a mapping.")
    required = {"state", "scheduling_role", "gpu_limit_gpus"}
    missing = sorted(required - set(worker))
    if missing:
        raise ValueError(f"{machine!r} Worker is missing required fields: {', '.join(missing)}.")
    state = worker["state"]
    role = worker["scheduling_role"]
    if role not in WORKER_ROLES:
        raise ValueError(f"{machine!r} Worker scheduling_role is invalid.")
    if state not in {"active", "draining", "removing"}:
        raise ValueError(f"{machine!r} Worker state is invalid.")
    if "borrow_limit_gpus" in worker:
        raise ValueError(f"{machine!r} Worker has obsolete borrow_limit_gpus.")
    worker["gpu_limit_gpus"] = validate_gpu_limit(
        worker["gpu_limit_gpus"], f"{machine}.gpu_limit_gpus"
    )
    return worker


def normalize_group_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate canonical Worker fields in a Group record without writing it."""
    group = data.get("group")
    if not isinstance(group, dict) or not isinstance(group.get("worker_set"), dict):
        raise ValueError("Group Worker Set is malformed.")
    for machine, worker in group["worker_set"].items():
        normalize_worker_member(worker, machine=str(machine))
    return data


def new_group(name: str, machine: str) -> dict[str, Any]:
    validate_group_name(name)
    return {"meta": _meta(machine), "group": {"name": name, "admission_state": "open",
        "dispatch_state": "active", "dispatch_epoch": 0, "worker_set_epoch": 0,
        "next_membership_sequence": 1, "pending_submission_commit": None, "worker_set": {},
        "cancellation_barriers": []}}


def new_worker_member(*, scheduling_role: str = "primary", gpu_limit_gpus: int | None = None,
                      added_by_operation: str | None = None) -> dict[str, Any]:
    if scheduling_role not in WORKER_ROLES:
        raise ValueError("scheduling_role must be 'primary' or 'borrow'.")
    gpu_limit_gpus = validate_gpu_limit(gpu_limit_gpus)
    return {"state": "active",
            "scheduling_role": scheduling_role, "gpu_limit_gpus": gpu_limit_gpus,
            "state_epoch": 0, "added_at": utc_now(),
            "added_by_operation": added_by_operation, "drain_requested_at": None,
            "remove_requested_at": None, "terminate_running": False}


def new_submission(*, operation_id: str, kind: str, key: str, raw_digest: str, machine: str,
                   target_group: str | None, resolved_context: dict[str, Any]) -> dict[str, Any]:
    return {"meta": _meta(machine), "submission": {"operation_id": operation_id, "kind": kind,
        "idempotency_key": key, "raw_request_digest": raw_digest,
        "resolved_context_digest": hashlib.sha256(json.dumps(resolved_context, sort_keys=True).encode()).hexdigest(),
        "original_submitting_machine": machine, "target_group": target_group, "state": "preparing",
        "resolved_context": resolved_context, "commit_plan": {"group_membership_sequences": None,
        "pending_group_revision": None}, "staged_task_count": len(resolved_context["task_ids"]),
        "committed_at": None, "failure_reason": None}}
