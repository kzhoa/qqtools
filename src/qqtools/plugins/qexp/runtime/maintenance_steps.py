"""Bounded shared maintenance step execution."""

from __future__ import annotations

import errno
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from .directory_capture import read_directory_entry
from .observation.maintenance import ObservationMaintenance, request_rebuild
from .observation.projection import inspect_observation
from .paths import shared_paths
from .ready.diagnostics import parse_ready_reason
from .ready.group_members import group_ready_members_state
from .ready.group_members_rebuild import repair_group_ready_members
from .ready.rebuild import ready_task_projection_issue, repair_ready_index
from .ready.state import mark_ready_index_degraded, read_ready_index_state
from .records import AttemptRecord, TaskRecord, utc_now
from .store import atomic_replace, read_json_limited
from .submission_control import inspect_submission_control, request_control_rebuild
from .submission_control_maintenance import SubmissionControlMaintenance

SOURCE_RECORD_MAX_BYTES = 1_048_576


@dataclass(frozen=True)
class ReadyIndexStepResult:
    cursor: dict[str, Any]
    completed: bool
    repaired: tuple[str, ...]
    blocked: tuple[str, ...]
    failure: dict[str, str] | None
    prior_degraded_reasons: tuple[str, ...]
    build_identity: dict[str, str] | None


@dataclass(frozen=True)
class GroupReadyMembersStepResult:
    cursor: dict[str, Any]
    completed: bool
    repaired: tuple[str, ...]
    blocked: tuple[str, ...]
    failure: dict[str, str] | None
    build_identity: dict[str, str] | None


@dataclass(frozen=True)
class TaskObservationStepResult:
    cursor: dict[str, Any]
    completed: bool
    detail: dict[str, Any]
    failure: dict[str, str] | None
    meaningful_progress: bool


@dataclass(frozen=True)
class SubmissionControlStepResult:
    cursor: dict[str, Any]
    completed: bool
    detail: dict[str, Any]
    failure: dict[str, str] | None
    meaningful_progress: bool


@dataclass(frozen=True)
class OrphanRecoveryStepResult:
    cursor: dict[str, Any]
    completed: bool
    repaired: tuple[str, ...]
    blocked: tuple[str, ...]
    failure: dict[str, str] | None
    meaningful_progress: bool


class ReadyIndexStepError(RuntimeError):
    """Carry detected degradation evidence across a failed side effect."""

    def __init__(self, prior_degraded_reasons: tuple[str, ...], cause: BaseException) -> None:
        super().__init__(str(cause))
        self.prior_degraded_reasons = prior_degraded_reasons
        self.cause = cause


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else dict(value)


def is_transient_failure(exception: BaseException) -> bool:
    """Return whether a maintenance failure is safe to retry."""
    if isinstance(exception, (BlockingIOError, TimeoutError)):
        return True
    return isinstance(exception, OSError) and exception.errno in {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EBUSY,
        errno.ETIMEDOUT,
        errno.ESTALE,
    }


def _ready_result(
    cursor: dict[str, Any],
    *,
    completed: bool,
    repaired: list[str],
    blocked: list[str],
    failure: dict[str, str] | None,
    prior_degraded_reasons: list[str],
    build_identity: dict[str, str] | None,
) -> ReadyIndexStepResult:
    return ReadyIndexStepResult(
        cursor=dict(cursor),
        completed=completed,
        repaired=tuple(repaired),
        blocked=tuple(blocked),
        failure=_copy_mapping(failure),
        prior_degraded_reasons=tuple(prior_degraded_reasons),
        build_identity=_copy_mapping(build_identity),
    )


def advance_ready_index_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any],
    prior_degraded_reasons: list[str] | tuple[str, ...],
) -> ReadyIndexStepResult:
    """Advance one bounded ready-index audit or rebuild step."""
    current_cursor = dict(cursor)
    reasons = list(prior_degraded_reasons)
    mode = current_cursor.get("mode", "audit")
    build_identity: dict[str, str] | None = None
    if mode == "audit":
        offset = current_cursor.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("ready-index audit cursor is invalid.")
        directory = shared_paths(cfg.shared_root)["tasks"]
        name, next_offset = read_directory_entry(directory, offset)
        if name is None:
            if read_ready_index_state(cfg) == "active":
                return _ready_result(
                    {},
                    completed=True,
                    repaired=[],
                    blocked=[],
                    failure=None,
                    prior_degraded_reasons=reasons,
                    build_identity=None,
                )
            return _ready_result(
                {"mode": "build"},
                completed=False,
                repaired=[],
                blocked=[],
                failure=None,
                prior_degraded_reasons=reasons,
                build_identity=None,
            )
        next_cursor = {"mode": "audit", "offset": next_offset}
        if not name.endswith(".json"):
            return _ready_result(
                next_cursor,
                completed=False,
                repaired=[],
                blocked=[],
                failure=None,
                prior_degraded_reasons=reasons,
                build_identity=None,
            )
        task = TaskRecord.from_dict(
            read_json_limited(directory / name, max_bytes=SOURCE_RECORD_MAX_BYTES, record_type="ready_audit_task")
        )
        if read_ready_index_state(cfg) != "active":
            return _ready_result(
                {"mode": "build"},
                completed=False,
                repaired=[],
                blocked=[],
                failure=None,
                prior_degraded_reasons=reasons,
                build_identity=None,
            )
        issue = ready_task_projection_issue(cfg, task.task_id)
        if issue is None:
            return _ready_result(
                next_cursor,
                completed=False,
                repaired=[],
                blocked=[],
                failure=None,
                prior_degraded_reasons=reasons,
                build_identity=None,
            )
        try:
            diagnostic = parse_ready_reason(issue).diagnostic
        except ValueError:
            diagnostic = None
        if diagnostic is None:
            return _ready_result(
                next_cursor,
                completed=False,
                repaired=[],
                blocked=["ready_index"],
                failure={
                    "code": "ready_index_audit_unverifiable",
                    "phase": "ready_index",
                    "type": "ProjectionEvidence",
                },
                prior_degraded_reasons=reasons,
                build_identity=None,
            )
        if issue not in reasons:
            reasons.append(issue)
        try:
            mark_ready_index_degraded(cfg, diagnostic)
        except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ReadyIndexStepError(tuple(reasons), exc) from exc
        return _ready_result(
            {"mode": "build"},
            completed=False,
            repaired=[],
            blocked=[],
            failure=None,
            prior_degraded_reasons=reasons,
            build_identity=None,
        )

    if mode != "build":
        raise ValueError("ready-index phase cursor is invalid.")
    prior_state = read_ready_index_state(cfg)
    try:
        ready_record = repair_ready_index(cfg, max_tasks=1, bounded_initialization=True)
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return _ready_result(
            current_cursor,
            completed=False,
            repaired=[],
            blocked=["ready_index"],
            failure={
                "code": "ready_index_rebuild_failed",
                "phase": "ready_index",
                "type": "ProjectionFailure",
            },
            prior_degraded_reasons=reasons,
            build_identity=None,
        )
    if ready_record.get("state") == "degraded":
        return _ready_result(
            current_cursor,
            completed=False,
            repaired=[],
            blocked=["ready_index"],
            failure={
                "code": "ready_index_rebuild_failed",
                "phase": "ready_index",
                "type": "ProjectionFailure",
            },
            prior_degraded_reasons=reasons,
            build_identity=None,
        )
    build = ready_record.get("build") or {}
    if isinstance(build, dict) and isinstance(build.get("build_id"), str):
        build_identity = {"kind": "ready_index", "build_id": build["build_id"]}
    if ready_record.get("state") == "active":
        repaired = []
        if prior_state != "active":
            repaired.append(f"ready_index:{build.get('repaired', 0)}:{build.get('stale_removed', 0)}")
        return _ready_result(
            {},
            completed=True,
            repaired=repaired,
            blocked=[],
            failure=None,
            prior_degraded_reasons=reasons,
            build_identity=build_identity,
        )
    return _ready_result(
        current_cursor,
        completed=False,
        repaired=[],
        blocked=[],
        failure=None,
        prior_degraded_reasons=reasons,
        build_identity=build_identity,
    )


def _orphan_result(
    cursor: dict[str, Any],
    *,
    completed: bool,
    repaired: list[str],
    blocked: list[str],
    failure: dict[str, str] | None,
    meaningful_progress: bool,
) -> OrphanRecoveryStepResult:
    return OrphanRecoveryStepResult(
        cursor=deepcopy(cursor),
        completed=completed,
        repaired=tuple(repaired),
        blocked=tuple(blocked),
        failure=_copy_mapping(failure),
        meaningful_progress=meaningful_progress,
    )


def advance_orphan_recovery_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any],
    reservation_runtime_root: Path | None,
) -> OrphanRecoveryStepResult:
    """Advance one orphan-recovery child or scan step."""
    current_cursor = deepcopy(cursor)
    pending = current_cursor.get("orphan")
    if isinstance(pending, dict):
        stage = pending.get("stage")
        task_id = pending.get("task_id")
        number = pending.get("attempt_number")
        if not isinstance(task_id, str) or type(number) is not int or number < 0:
            raise ValueError("orphan child cursor is invalid.")
        attempt_file = shared_paths(cfg.shared_root)["attempts"] / task_id / f"{number}.json"
        if stage == "attempt":
            if not attempt_file.exists():
                return _orphan_result(
                    current_cursor,
                    completed=False,
                    repaired=[],
                    blocked=[task_id],
                    failure={
                        "code": "attempt_truth_missing",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                    meaningful_progress=False,
                )
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    attempt_file,
                    max_bytes=SOURCE_RECORD_MAX_BYTES,
                    record_type="maintenance_attempt",
                )
            )
            if attempt.task_id != task_id or attempt.attempt_number != number:
                raise ValueError("orphan Attempt identity does not match its Task.")
            if attempt.machine_name != cfg.machine_name:
                return _orphan_result(
                    current_cursor,
                    completed=False,
                    repaired=[],
                    blocked=[task_id],
                    failure={
                        "code": "orphan_attempt_owned_remotely",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                    meaningful_progress=False,
                )
            pending.update(
                {
                    "stage": "process",
                    "attempt_id": attempt.attempt_id,
                    "fencing_token": attempt.current_fencing_token,
                    "attempt_process_identity": {
                        key: attempt.process.get(key) for key in ("process_group_id", "process_group_start_time_ticks")
                    },
                }
            )
            return _orphan_result(
                current_cursor,
                completed=False,
                repaired=[],
                blocked=[],
                failure=None,
                meaningful_progress=False,
            )
        if stage == "process":
            attempt_id = pending.get("attempt_id")
            fencing_token = pending.get("fencing_token")
            if not isinstance(attempt_id, str) or type(fencing_token) is not int:
                raise ValueError("orphan Attempt continuation is malformed.")
            manifest_path = cfg.runtime_root / "processes" / f"{attempt_id}.json"
            if not manifest_path.exists():
                return _orphan_result(
                    current_cursor,
                    completed=False,
                    repaired=[],
                    blocked=[task_id],
                    failure={
                        "code": "orphan_process_evidence_missing",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                    meaningful_progress=False,
                )
            process = read_json_limited(
                manifest_path,
                max_bytes=SOURCE_RECORD_MAX_BYTES,
                record_type="maintenance_process",
            )
            process_record = process.get("process")
            if (
                not isinstance(process_record, dict)
                or process_record.get("task_id") != task_id
                or process_record.get("attempt_id") != attempt_id
            ):
                return _orphan_result(
                    current_cursor,
                    completed=False,
                    repaired=[],
                    blocked=[task_id],
                    failure={
                        "code": "orphan_process_identity_mismatch",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                    meaningful_progress=False,
                )
            from ..scheduler import finalize_orphaned_attempt
            from .attempt_recovery import recover_running_attempt
            from .process_evidence import inspect_group_identity

            evidence = inspect_group_identity(pending["attempt_process_identity"], process_record)
            if evidence.state == "alive":
                token = recover_running_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    fencing_token,
                    manifest=process_record,
                    reservation_runtime_root=reservation_runtime_root,
                )
                if token is None:
                    return _orphan_result(
                        current_cursor,
                        completed=False,
                        repaired=[],
                        blocked=[task_id],
                        failure={
                            "code": "orphan_recovery_cas_rejected",
                            "phase": "orphan_recovery",
                            "type": "Intervention",
                        },
                        meaningful_progress=False,
                    )
            elif evidence.state == "absent":
                if not finalize_orphaned_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    fencing_token,
                    exit_code=process_record.get("exit_code"),
                    was_terminated=bool(pending.get("was_terminated")),
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    return _orphan_result(
                        current_cursor,
                        completed=False,
                        repaired=[],
                        blocked=[task_id],
                        failure={
                            "code": "orphan_finalize_cas_rejected",
                            "phase": "orphan_recovery",
                            "type": "Intervention",
                        },
                        meaningful_progress=False,
                    )
                process_record["observed_state"] = "exited"
                process_record["reconciled_at"] = utc_now()
                atomic_replace(manifest_path, {"process": process_record})
            else:
                return _orphan_result(
                    current_cursor,
                    completed=False,
                    repaired=[],
                    blocked=[task_id],
                    failure={
                        "code": "orphan_process_identity_unverifiable",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                    meaningful_progress=False,
                )
            del current_cursor["orphan"]
            return _orphan_result(
                current_cursor,
                completed=False,
                repaired=[task_id],
                blocked=[],
                failure=None,
                meaningful_progress=True,
            )
        raise ValueError("orphan child stage is invalid.")

    offset = current_cursor.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("orphan-recovery cursor is invalid.")
    directory = shared_paths(cfg.shared_root)["tasks"]
    name, next_offset = read_directory_entry(directory, offset)
    if name is None:
        return _orphan_result(
            {"offset": next_offset},
            completed=True,
            repaired=[],
            blocked=[],
            failure=None,
            meaningful_progress=True,
        )
    next_cursor: dict[str, Any] = {"offset": next_offset}
    if not name.endswith(".json"):
        return _orphan_result(
            next_cursor,
            completed=False,
            repaired=[],
            blocked=[],
            failure=None,
            meaningful_progress=next_cursor != current_cursor,
        )
    task = TaskRecord.from_dict(
        read_json_limited(
            directory / name,
            max_bytes=SOURCE_RECORD_MAX_BYTES,
            record_type="maintenance_orphan_task",
        )
    )
    if task.state["projection"] != "blocked":
        return _orphan_result(
            next_cursor,
            completed=False,
            repaired=[],
            blocked=[],
            failure=None,
            meaningful_progress=next_cursor != current_cursor,
        )
    number = task.attempt_control.get("current_attempt_number")
    if type(number) is not int or number < 0:
        return _orphan_result(
            next_cursor,
            completed=False,
            repaired=[],
            blocked=[task.task_id],
            failure={
                "code": "orphan_current_attempt_missing",
                "phase": "orphan_recovery",
                "type": "Intervention",
            },
            meaningful_progress=next_cursor != current_cursor,
        )
    next_cursor["orphan"] = {
        "stage": "attempt",
        "task_id": task.task_id,
        "attempt_number": number,
        "was_terminated": bool(task.control.get("terminate_running")),
    }
    return _orphan_result(
        next_cursor,
        completed=False,
        repaired=[],
        blocked=[],
        failure=None,
        meaningful_progress=True,
    )


def advance_group_ready_members_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any],
) -> GroupReadyMembersStepResult:
    """Advance one bounded Group ready-members repair step."""
    initial_state = group_ready_members_state(cfg)
    projection_record = repair_group_ready_members(cfg, max_work_items=1, bounded_initialization=True)
    state = projection_record.get("state")
    audit = projection_record.get("audit") if isinstance(projection_record.get("audit"), dict) else {}
    build = projection_record.get("build") if isinstance(projection_record.get("build"), dict) else {}
    identity: dict[str, str] = {"kind": "group_ready_members"}
    for key in ("build_id", "projection_id"):
        value = build.get(key) if key == "build_id" else projection_record.get(key)
        if isinstance(value, str):
            identity[key] = value
    if isinstance(audit.get("audit_id"), str):
        identity["audit_id"] = audit["audit_id"]
    build_identity = identity if len(identity) > 1 else None
    if state == "degraded":
        return GroupReadyMembersStepResult(
            cursor={},
            completed=False,
            repaired=(),
            blocked=("group_ready_members",),
            failure={
                "code": "group_ready_members_rebuild_failed",
                "phase": "group_ready_members",
                "type": "ProjectionFailure",
            },
            build_identity=_copy_mapping(build_identity),
        )
    if state == "legacy" or (state == "active" and audit.get("state") == "completed"):
        repaired = ("group_ready_members",) if initial_state != "active" and state == "active" else ()
        return GroupReadyMembersStepResult(
            cursor={},
            completed=True,
            repaired=repaired,
            blocked=(),
            failure=None,
            build_identity=_copy_mapping(build_identity),
        )
    return GroupReadyMembersStepResult(
        cursor={},
        completed=False,
        repaired=(),
        blocked=(),
        failure=None,
        build_identity=_copy_mapping(build_identity),
    )


def advance_task_observation_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any],
) -> TaskObservationStepResult:
    """Advance one bounded Task-observation maintenance step."""
    initial_cursor = dict(cursor)
    current_cursor = dict(initial_cursor)
    observation = inspect_observation(cfg)
    if observation.get("state") in {"degraded", "unavailable"} and not current_cursor.get("requested"):
        requested = request_rebuild(cfg)
        current_cursor["requested"] = requested.get("state") != "waiting"
        return TaskObservationStepResult(
            cursor=current_cursor,
            completed=False,
            detail=dict(requested),
            failure=None,
            meaningful_progress=current_cursor != initial_cursor or requested.get("processed", 0) > 0,
        )
    if observation.get("state") == "active" and observation.get("dirty") is False:
        return TaskObservationStepResult(
            cursor={},
            completed=True,
            detail=dict(observation),
            failure=None,
            meaningful_progress=True,
        )
    maintenance = ObservationMaintenance(cfg)
    try:
        updated = maintenance.advance()
    finally:
        maintenance.close()
    failure = (
        {
            "code": "task_observation_rebuild_failed",
            "phase": "task_observation",
            "type": "ProjectionFailure",
        }
        if updated.get("state") == "degraded"
        else None
    )
    return TaskObservationStepResult(
        cursor=current_cursor,
        completed=False,
        detail=dict(updated),
        failure=failure,
        meaningful_progress=current_cursor != initial_cursor or updated.get("processed", 0) > 0,
    )


def advance_submission_control_step(
    cfg: RootConfig,
    *,
    cursor: dict[str, Any],
) -> SubmissionControlStepResult:
    """Advance one bounded Submission-control maintenance step."""
    initial_cursor = dict(cursor)
    current_cursor = dict(initial_cursor)
    control = inspect_submission_control(cfg)
    if control.get("state") == "active" and not current_cursor.get("requested"):
        return SubmissionControlStepResult(
            cursor={},
            completed=True,
            detail=dict(control),
            failure=None,
            meaningful_progress=True,
        )
    if control.get("state") == "unavailable" and not current_cursor.get("requested"):
        updated = request_control_rebuild(cfg)
        current_cursor["requested"] = updated.get("state") != "waiting"
        return SubmissionControlStepResult(
            cursor=current_cursor,
            completed=False,
            detail=dict(updated),
            failure=None,
            meaningful_progress=current_cursor != initial_cursor,
        )
    maintenance = SubmissionControlMaintenance(cfg)
    try:
        updated = maintenance.advance()
    finally:
        maintenance.close()
    return SubmissionControlStepResult(
        cursor=current_cursor if updated.get("state") != "active" else {},
        completed=updated.get("state") == "active",
        detail=dict(updated),
        failure=None,
        meaningful_progress=(current_cursor if updated.get("state") != "active" else {}) != initial_cursor
        or updated.get("state") == "active",
    )
