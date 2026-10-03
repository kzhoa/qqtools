"""Outbox-selected bounded maintenance advancement."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.maintenance_outbox import activate_work, read_work, retire_work, select_due_work, update_work
from ..runtime.observation.projection import inspect_observation
from ..runtime.operation_store import locate_operation_path
from ..runtime.paths import shared_paths, submission_path, task_path
from ..runtime.records import AttemptRecord, TaskRecord, utc_now
from ..runtime.store import read_json_limited
from ..runtime.submission_control import inspect_submission_control
from . import maintenance_full_audit, maintenance_steps


def _work_operation_path(cfg: RootConfig, descriptor: dict[str, Any]) -> Path | None:
    identity = descriptor["identity"]
    kind = identity["kind"]
    operation_key = descriptor["cursor"].get("operation_key")
    if kind == "submission":
        return submission_path(cfg.shared_root, identity["target_id"])
    if kind not in {"cleanup", "availability", "group_cancel"}:
        return None
    if not isinstance(operation_key, str):
        return None
    operation_kind = "group_control" if kind == "group_cancel" else kind
    return locate_operation_path(cfg, operation_kind, operation_key)


def _operation_section(kind: str) -> str:
    return "availability_operation" if kind == "availability" else ("group_control" if kind == "group_cancel" else kind)


def _advance_operation_work(
    cfg: RootConfig,
    descriptor: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> dict[str, Any]:
    identity = descriptor["identity"]
    kind = identity["kind"]
    operation_id = identity["target_id"]
    path = _work_operation_path(cfg, descriptor)
    if path is None:
        return {"state": "intervention", "reason": "maintenance_operation_key_missing"}
    try:
        operation = read_json_limited(
            path,
            max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES,
            record_type=f"maintenance_{kind}_operation",
        )
    except FileNotFoundError:
        return {"state": "intervention", "reason": "maintenance_operation_missing_after_prepare"}
    section = operation.get(_operation_section(kind)) if type(operation) is dict else None
    if type(section) is not dict or section.get("operation_id") != operation_id:
        return {"state": "intervention", "reason": "maintenance_operation_identity_mismatch"}
    state = section.get("state")
    if state in {"completed", "superseded"}:
        return {"state": "completed", "proof": {"operation_state": state, "operation_id": operation_id}}
    if state == "blocked":
        return {
            "state": "intervention",
            "reason": section.get("blocked_reason") or "maintenance_operation_blocked",
        }

    cursor = descriptor["cursor"]
    if kind == "cleanup":
        from . import cleanup_maintenance

        task_id = section.get("task_id")
        if not isinstance(task_id, str):
            return {"state": "intervention", "reason": "cleanup_operation_task_identity_missing"}
        outcome = cleanup_maintenance.advance_cleanup_maintenance_step(
            cfg,
            task_id,
            operation_id,
            cursor.get("child") if isinstance(cursor.get("child"), dict) else {},
            reservation_runtime_root=reservation_runtime_root,
        )
        if outcome.get("state") == "completed":
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "cleanup_step"}}
        if outcome.get("state") == "intervention":
            return {"state": "intervention", "reason": outcome.get("reason") or "cleanup_intervention"}
        next_cursor = dict(cursor)
        next_cursor["child"] = outcome.get("cursor", next_cursor.get("child", {}))
        return {
            "state": "waiting" if outcome.get("state") == "waiting" else "running",
            "cursor": next_cursor,
            "reason": outcome.get("reason"),
            "meaningful_progress": next_cursor != cursor,
        }

    if kind == "availability":
        from ..runtime.availability.transitions import AvailabilityTransitionRequest, apply_availability_transition

        outcome = apply_availability_transition(
            cfg,
            AvailabilityTransitionRequest(
                action=section.get("operation_type"),
                task_id=section.get("task_id"),
                helper_machines=section.get("helper_machines"),
                after_seconds=section.get("after_seconds"),
                reason=section.get("reason") or "manual",
                operation_id=operation_id,
            ),
        )
        return {"state": "completed", "proof": {"operation_id": operation_id, "source": "availability_transition"}}

    if kind == "submission":
        from ..runtime.submission import advance_submission_cleanup_step

        outcome = advance_submission_cleanup_step(cfg, operation_id)
        if outcome.get("state") == "completed":
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "submission_finalizer"}}
        if outcome.get("state") == "intervention":
            return {"state": "intervention", "reason": outcome.get("reason") or "submission_intervention"}
        return {
            "state": "running",
            "reason": outcome.get("stage"),
            "meaningful_progress": True,
        }

    operation_type = section.get("operation_type")
    if operation_type == "worker_remove_v2":
        from ..commands.worker_removal import reconcile_worker_removal

        outcome = reconcile_worker_removal(cfg, path, None, reservation_runtime_root=reservation_runtime_root)
        if outcome is None:
            return {"state": "waiting", "reason": "worker_removal_lock_or_state"}
        if outcome.get("state") in {"completed", "superseded"}:
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "worker_removal"}}
        if outcome.get("state") == "blocked":
            return {"state": "intervention", "reason": outcome.get("blocked_reason") or "worker_removal_blocked"}
        return {"state": "running", "reason": "worker_removal_progress", "meaningful_progress": True}
    if operation_type != "cancel":
        return {"state": "intervention", "reason": "legacy_worker_removal_requires_scoped_reconstruction"}
    if state not in {"preparing", "converging", "waiting_ack"}:
        return {"state": "intervention", "reason": "group_cancel_operation_state_unknown"}
    from ..runtime.group_namespace import is_group_authority_isolated

    if is_group_authority_isolated(cfg.shared_root):
        from ..commands.group_cancel import advance_indexed_cancel

        outcome = advance_indexed_cancel(cfg, path, reservation_runtime_root=reservation_runtime_root)
        if outcome is None:
            current = locate_operation_path(cfg, "group_control", operation_id)
            if not current.exists():
                return {"state": "intervention", "reason": "group_cancel_operation_missing_after_prepare"}
            outcome = {"state": "waiting", "reason": "group_cancel_no_progress"}
        if outcome.get("state") in {"completed", "superseded"}:
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "indexed_cancel"}}
        if outcome.get("state") == "blocked":
            return {"state": "intervention", "reason": outcome.get("blocked_reason") or "group_cancel_blocked"}
        return {"state": "running", "reason": "group_cancel_progress", "meaningful_progress": True}
    from ..commands.group import advance_legacy_cancel_step

    outcome = advance_legacy_cancel_step(
        cfg,
        operation_id,
        cursor.get("child", {}),
        reservation_runtime_root=reservation_runtime_root,
    )
    if outcome.get("state") == "completed":
        return {"state": "completed", "proof": {"operation_id": operation_id, "source": "legacy_cancel"}}
    if outcome.get("state") == "intervention":
        return {"state": "intervention", "reason": outcome.get("reason") or "group_cancel_intervention"}
    next_cursor = dict(cursor)
    next_cursor["child"] = outcome.get("cursor", {})
    return {
        "state": "running",
        "cursor": next_cursor,
        "meaningful_progress": next_cursor != cursor,
    }


def _advance_projection_work(
    cfg: RootConfig,
    descriptor: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> dict[str, Any]:
    kind = descriptor["identity"]["kind"]
    if kind == "ready_index":
        try:
            result = maintenance_steps.advance_ready_index_step(
                cfg,
                cursor=descriptor["cursor"],
                prior_degraded_reasons=[],
            )
        except maintenance_steps.ReadyIndexStepError as exc:
            raise exc.cause
        return {
            "state": "intervention" if result.failure is not None else "completed" if result.completed else "running",
            "cursor": result.cursor,
            "proof": {"source": "ready_index_fence", "repaired": list(result.repaired)},
            "reason": None if result.failure is None else result.failure["code"],
            "meaningful_progress": result.cursor != descriptor["cursor"] or result.completed,
        }
    if kind == "group_ready_members":
        result = maintenance_steps.advance_group_ready_members_step(
            cfg,
            cursor=descriptor["cursor"],
        )
        return {
            "state": "intervention" if result.failure is not None else "completed" if result.completed else "running",
            "cursor": result.cursor,
            "proof": {"source": "group_ready_members_fence", "repaired": list(result.repaired)},
            "reason": None if result.failure is None else result.failure["code"],
            "meaningful_progress": result.cursor != descriptor["cursor"] or result.completed,
        }
    if kind == "task_observation":
        result = maintenance_steps.advance_task_observation_step(cfg, cursor=descriptor["cursor"])
        return {
            "state": "intervention"
            if result.failure is not None
            else "completed"
            if result.completed
            else ("waiting" if result.detail.get("state") == "waiting" else "running"),
            "cursor": result.cursor,
            "reason": result.failure["code"] if result.failure else result.detail.get("reason"),
            "meaningful_progress": result.meaningful_progress,
        }
    if kind == "submission_control":
        result = maintenance_steps.advance_submission_control_step(cfg, cursor=descriptor["cursor"])
        return {
            "state": "intervention"
            if result.failure is not None
            else "completed"
            if result.completed
            else ("waiting" if result.detail.get("state") == "waiting" else "running"),
            "cursor": result.cursor,
            "reason": result.failure["code"] if result.failure else result.detail.get("reason"),
            "meaningful_progress": result.meaningful_progress,
        }
    if kind == "deadline_index":
        task_id = descriptor["identity"]["target_id"]
        from ..runtime.availability.offer_deadlines import reconcile_deadline_index

        acquired, _ = reconcile_deadline_index(cfg, task_id, blocking=False)
        if not acquired:
            return {"state": "waiting", "reason": "deadline_index_task_lock_busy"}
        return {"state": "completed", "proof": {"source": "deadline_index_task", "task_id": task_id}}
    if kind == "orphan_recovery":
        cursor = descriptor["cursor"]
        task_id = cursor.get("task_id")
        attempt_id = descriptor["identity"]["target_id"]
        attempt_number = cursor.get("attempt_number")
        fencing_token = cursor.get("fencing_token")
        if (
            not isinstance(task_id, str)
            or cursor.get("attempt_id") != attempt_id
            or type(attempt_number) is not int
            or attempt_number < 0
            or type(fencing_token) is not int
            or fencing_token < 0
        ):
            return {"state": "intervention", "reason": "orphan_attempt_descriptor_invalid"}
        attempt_path_value = shared_paths(cfg.shared_root)["attempts"] / task_id / f"{attempt_number}.json"
        try:
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    attempt_path_value,
                    max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES,
                    record_type="maintenance_known_attempt",
                )
            )
            task = TaskRecord.from_dict(
                read_json_limited(
                    task_path(cfg.shared_root, task_id),
                    max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES,
                    record_type="maintenance_known_attempt_task",
                )
            )
        except FileNotFoundError:
            return {"state": "intervention", "reason": "orphan_attempt_truth_missing"}
        if attempt.task_id != task_id or attempt.attempt_number != attempt_number or attempt.attempt_id != attempt_id:
            return {"state": "intervention", "reason": "orphan_attempt_identity_mismatch"}
        active_claim = task.claim_control.get("active_claim") or {}
        if attempt.phase != "orphaned":
            if (
                task.state.get("projection") == "running"
                and active_claim.get("attempt_id") != attempt_id
                and type(active_claim.get("fencing_token")) is int
                and active_claim["fencing_token"] > fencing_token
            ):
                return {
                    "state": "completed",
                    "proof": {
                        "source": "attempt_recovery_superseded",
                        "attempt_id": attempt_id,
                        "fencing_token": active_claim["fencing_token"],
                    },
                }
            return {"state": "intervention", "reason": "orphan_attempt_truth_not_committed"}
        if (
            task.attempt_control.get("current_attempt_number") != attempt_number
            or task.state.get("projection") != "blocked"
            or active_claim
        ):
            return {"state": "intervention", "reason": "orphan_task_authority_mismatch"}
        child = {
            key: cursor[key]
            for key in (
                "stage",
                "task_id",
                "attempt_number",
                "attempt_id",
                "fencing_token",
                "attempt_process_identity",
                "was_terminated",
            )
            if key in cursor
        }
        result = maintenance_steps.advance_orphan_recovery_step(
            cfg,
            cursor={"orphan": child},
            reservation_runtime_root=reservation_runtime_root,
        )
        child_after = result.cursor.get("orphan")
        if result.failure is not None:
            return {"state": "intervention", "reason": result.failure["code"]}
        if result.completed or child_after is None:
            return {
                "state": "completed",
                "proof": {
                    "source": "known_attempt_fence",
                    "attempt_id": attempt_id,
                    "repaired": list(result.repaired),
                },
                "meaningful_progress": True,
            }
        return {
            "state": "running",
            "cursor": {**cursor, **child_after},
            "reason": "known_attempt_recovery_pending",
            "meaningful_progress": result.meaningful_progress,
        }
    return {"state": "intervention", "reason": "maintenance_descriptor_kind_unknown"}


def _next_descriptor_retry(record: dict[str, Any], reason: str | None) -> tuple[str, int, dict[str, str]]:
    retry_count = record["retry_count"] + 1
    base_delay = min(60.0, float(2 ** min(retry_count - 1, 6)))
    delay = random.SystemRandom().uniform(0.8 * base_delay, base_delay)
    due_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
    failure = {"code": reason or "maintenance_waiting", "phase": record["phase"], "type": "TransientWait"}
    return due_at, retry_count, failure


def _prepared_truth_committed(cfg: RootConfig, descriptor: dict[str, Any]) -> bool:
    """Recognize producer handoff from authoritative truth, never from elapsed time."""
    identity = descriptor["identity"]
    kind = identity["kind"]
    if kind in {"cleanup", "availability", "group_cancel", "submission"}:
        path = _work_operation_path(cfg, descriptor)
        if path is None:
            return False
        try:
            operation = read_json_limited(path, max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES)
        except FileNotFoundError:
            return False
        section = operation.get(_operation_section(kind)) if type(operation) is dict else None
        return type(section) is dict and section.get("operation_id") == identity["target_id"]
    if kind == "ready_index":
        from ..runtime.ready import read_ready_index_status

        status = read_ready_index_status(cfg)
        build = status.get("build") if isinstance(status.get("build"), dict) else {}
        expected = descriptor["cursor"].get("build_id")
        if isinstance(expected, str):
            return build.get("build_id") == expected
        return status.get("state") == "degraded" and descriptor["cursor"].get("mode") == "build"
    if kind == "group_ready_members":
        from ..runtime.ready.group_members import read_group_ready_members_state

        status = read_group_ready_members_state(cfg)
        build = status.get("build") if isinstance(status.get("build"), dict) else {}
        park = status.get("park") if isinstance(status.get("park"), dict) else {}
        expected = descriptor["cursor"].get("build_id")
        return isinstance(expected, str) and expected in {build.get("build_id"), park.get("build_id")}
    if kind == "task_observation":
        status = inspect_observation(cfg)
        return status.get("state") in {"building", "degraded"} or bool(status.get("dirty"))
    if kind == "submission_control":
        return inspect_submission_control(cfg).get("state") in {"building", "degraded"}
    if kind == "orphan_recovery":
        cursor = descriptor["cursor"]
        task_id = cursor.get("task_id")
        attempt_number = cursor.get("attempt_number")
        if not isinstance(task_id, str) or type(attempt_number) is not int:
            return False
        try:
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    shared_paths(cfg.shared_root)["attempts"] / task_id / f"{attempt_number}.json",
                    max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES,
                )
            )
            task = TaskRecord.from_dict(
                read_json_limited(
                    task_path(cfg.shared_root, task_id),
                    max_bytes=maintenance_steps.SOURCE_RECORD_MAX_BYTES,
                )
            )
        except FileNotFoundError:
            return False
        return (
            attempt.attempt_id == identity["target_id"]
            and attempt.attempt_number == attempt_number
            and attempt.phase == "orphaned"
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("current_attempt_id") is None
            and task.state.get("projection") == "blocked"
            and task.state.get("reason") == "orphaned_attempt_requires_recovery"
            and not task.claim_control.get("active_claim")
        )
    # Deadline repair is idempotently derived from current Task truth. The
    # full-audit adapter has its own durable descriptor as the handoff proof.
    return kind in {"deadline_index", "full_audit"}


def advance_maintenance_work(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None,
    max_scan: int = 1,
) -> dict[str, Any]:
    """Advance one durable descriptor selected from the Project-local outbox."""
    ledger = maintenance_full_audit.create_invocation_ledger(1)
    ledger.charge_operations(operations=32)
    maintenance_full_audit.ensure_full_audit_outbox(cfg)
    selection = select_due_work(cfg, max_scan=max_scan)
    descriptor = selection["descriptor"]
    if descriptor is None:
        state = "pending" if selection.get("more") else "waiting" if selection.get("next_due_at") else "idle"
        return {
            "maintenance_state": state,
            "next_due_at": selection.get("next_due_at"),
            "more": selection.get("more", False),
            "idle_blocking": state == "waiting",
            "budget": ledger.report(),
        }

    identity = descriptor["identity"]
    kind = identity["kind"]
    if descriptor["state"] == "prepared":
        if _prepared_truth_committed(cfg, descriptor):
            cursor = descriptor.get("cursor")
            has_business_wake = isinstance(cursor, dict) and cursor.get("activation_mode") == "business"
            descriptor = activate_work(cfg, descriptor, publish_activation=not has_business_wake)
        else:
            due_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
            update_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                state="prepared",
                due_at=due_at,
                failure={"code": "producer_handoff_pending", "type": "TransientWait"},
                publish_activation=False,
            )
            return {
                "maintenance_state": "waiting",
                "next_due_at": due_at,
                "idle_blocking": False,
                "budget": ledger.report(),
            }

    if kind == "full_audit":
        progress = maintenance_full_audit.advance_full_audit(
            cfg,
            reservation_runtime_root=reservation_runtime_root,
            max_work_items=1,
            ledger=ledger,
            create_if_missing=False,
            create_successor=False,
            create_successor_on_change=True,
        )
        target_state = progress.get("maintenance_state")
        if target_state == "completed":
            retire_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                proof={"source": "full_audit", "capture_id": progress.get("scope", {}).get("capture_id")},
            )
        elif target_state == "intervention":
            retire_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                state="intervention",
                proof={"failure": progress.get("failure") or progress.get("intervention")},
            )
        else:
            phase = progress.get("phase") or descriptor["phase"]
            cursor = {"phase": phase, "cursor": progress.get("cursor", {})}
            if target_state == "waiting":
                due_at = progress.get("next_due_at") or descriptor["due_at"]
                update_work(
                    cfg,
                    kind=kind,
                    target_id=identity["target_id"],
                    work_generation=identity["work_generation"],
                    state="waiting",
                    phase=phase,
                    cursor=cursor,
                    due_at=due_at,
                )
            else:
                update_work(
                    cfg,
                    kind=kind,
                    target_id=identity["target_id"],
                    work_generation=identity["work_generation"],
                    state="running",
                    phase=phase,
                    cursor=cursor,
                    meaningful_progress=True,
                )
        return progress

    reservation_cost = maintenance_full_audit.PHASE_OPERATION_RESERVATIONS.get(kind, 128)
    reservation = ledger.reserve_operations(maximum_operations=reservation_cost)
    if reservation is None:
        due_at, retry_count, failure = _next_descriptor_retry(descriptor, "maintenance_budget_unavailable")
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="waiting",
            due_at=due_at,
            retry_count=retry_count,
            failure=failure,
        )
        return {"maintenance_state": "waiting", "next_due_at": due_at, "budget": ledger.report()}
    ledger.consume_semantic_item()
    try:
        if kind in {"cleanup", "availability", "group_cancel", "submission"}:
            outcome = _advance_operation_work(
                cfg,
                descriptor,
                reservation_runtime_root=reservation_runtime_root,
            )
        else:
            outcome = _advance_projection_work(
                cfg,
                descriptor,
                reservation_runtime_root=reservation_runtime_root,
            )
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if maintenance_steps.is_transient_failure(exc):
            outcome = {"state": "waiting", "reason": type(exc).__name__}
        else:
            outcome = {"state": "intervention", "reason": type(exc).__name__}
    finally:
        if reservation._active:
            reservation.commit(actual_operations=reservation_cost)

    latest = read_work(
        cfg,
        kind=kind,
        target_id=identity["target_id"],
        work_generation=identity["work_generation"],
    )
    if latest is not None and latest["state"] in {"completed", "intervention", "superseded"}:
        return {"maintenance_state": latest["state"], "descriptor": latest}
    state = outcome.get("state")
    if state == "completed":
        retire_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            proof=outcome.get("proof") or {"source": "maintenance_completion_fence"},
        )
    elif state == "intervention":
        retire_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="intervention",
            proof={"code": outcome.get("reason") or "maintenance_intervention"},
        )
    elif state == "waiting":
        due_at, retry_count, failure = _next_descriptor_retry(descriptor, outcome.get("reason"))
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="waiting",
            phase=str(outcome.get("phase") or descriptor["phase"]),
            cursor=outcome.get("cursor", descriptor["cursor"]),
            due_at=due_at,
            retry_count=retry_count,
            failure=failure,
            meaningful_progress=bool(outcome.get("meaningful_progress")),
        )
    else:
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="running",
            phase=str(outcome.get("phase") or descriptor["phase"]),
            cursor=outcome.get("cursor", descriptor["cursor"]),
            due_at=utc_now(),
            retry_count=0 if outcome.get("meaningful_progress") else descriptor["retry_count"],
            failure=None if outcome.get("meaningful_progress") else descriptor.get("failure"),
            meaningful_progress=bool(outcome.get("meaningful_progress")),
        )
    return {
        "maintenance_state": state or "running",
        "descriptor_identity": identity,
        "phase": outcome.get("phase") or descriptor["phase"],
        "cursor": outcome.get("cursor", descriptor["cursor"]),
        "next_due_at": due_at if state == "waiting" else None,
        "budget": ledger.report(),
    }
