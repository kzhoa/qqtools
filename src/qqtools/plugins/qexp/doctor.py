"""Non-mutating integrity checks and explicit safe repairs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from .commands.cleanup import reconcile_cleanup_operations
from .commands.group import reconcile_group_cancel_operations
from .config_types import RootConfig
from .lifecycle import (TerminalTransition, commit_terminal_transition_locked,
                        dispatch_task_lifecycle_hooks_noexcept)
from .runtime.availability import rebuild_deadline_indexes, reconcile_availability_operations
from .layout import validate_root_contract
from .runtime.locks import group_lock, task_lock
from .runtime.paths import attempt_path, group_path, local_paths, shared_paths, task_path
from .runtime.records import AttemptRecord, TaskRecord, utc_now
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.termination import list_decisions
from .runtime.tasks import load_task
from .runtime.ready import (
    READY_BUILD_PAGE_SIZE,
    mark_ready_index_degraded,
    read_ready_index_state,
    read_ready_index_status,
    ready_task_projection_issue,
    repair_ready_index,
    retire_current_ready_generation,
)
from .lease import clock_capability


def _cleaned_task_ids(cfg: RootConfig) -> set[str]:
    cleaned: set[str] = set()
    for path in iter_json(shared_paths(cfg.shared_root)["cleanup"]):
        try:
            operation = read_json(path).get("cleanup", {})
        except (OSError, ValueError):
            continue
        task_id = operation.get("task_id")
        if task_id and operation.get("state") in {"preparing", "waiting_ack", "completed"}:
            cleaned.add(task_id)
    return cleaned


def _issue(issues: list[dict[str, Any]], code: str, path: Any, severity: str,
           message: str | None = None) -> None:
    issue = {"code": code, "path": str(path), "severity": severity}
    if message:
        issue["message"] = message
    issues.append(issue)


def _records_by_stem(directory: Any, key: str, issues: list[dict[str, Any]],
                     invalid_code: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in iter_json(directory):
        try:
            record = read_json(path).get(key)
            if not isinstance(record, dict):
                raise ValueError(f"missing {key!r} object")
            records[path.stem] = record
        except (OSError, ValueError) as exc:
            _issue(issues, invalid_code, path, "high", str(exc))
    return records


def verify_integrity(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Verify shared truth and the selected project's local execution evidence.

    Args:
        cfg: Project configuration using the authoritative local runtime root.
        reservation_runtime_root: Reservation store to verify.
        project_id: Project filter for a machine-wide reservation store.

    Returns:
        Integrity summary and discovered issues.
    """
    if reservation_runtime_root is None:
        from .machine_runtime import resolve_execution_context

        context = resolve_execution_context(cfg)
        cfg = context.local_cfg
        reservation_runtime_root = context.reservation_root
        project_id = context.project_id
    validate_root_contract(cfg)
    issues: list[dict[str, Any]] = []
    ready_state = read_ready_index_state(cfg)
    if ready_state == "degraded":
        _issue(
            issues,
            "ready_index_degraded",
            shared_paths(cfg.shared_root)["ready"] / "state.json",
            "high",
        )
    paths = shared_paths(cfg.shared_root)
    cleaned = _cleaned_task_ids(cfg)
    submissions = _records_by_stem(
        paths["submissions"], "submission", issues, "submission_invalid"
    )
    group_records = _records_by_stem(paths["groups"], "group", issues, "group_invalid")
    groups = {name: {"group": group} for name, group in group_records.items()}
    deadline_task_ids: set[str] = set()
    for deadline_path in iter_json(paths["offer_deadlines"]):
        if deadline_path == paths["offer_deadlines_migration"]:
            continue
        try:
            deadline = read_json(deadline_path).get("offer_deadline", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "offer_deadline_index_invalid", deadline_path, "high", str(exc))
            continue
        task_id = deadline.get("task_id")
        if isinstance(task_id, str):
            deadline_task_ids.add(task_id)
        if task_id != deadline_path.stem or not task_path(cfg.shared_root, str(task_id)).exists():
            _issue(issues, "offer_deadline_index_stale", deadline_path, "high")
    for cleanup_path in iter_json(paths["cleanup"]):
        try:
            cleanup = read_json(cleanup_path).get("cleanup", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "cleanup_operation_invalid", cleanup_path, "high", str(exc))
            continue
        task_id = cleanup.get("task_id")
        if not task_id:
            _issue(issues, "cleanup_operation_invalid", cleanup_path, "high")
            continue
        task_exists = task_path(cfg.shared_root, task_id).exists()
        if cleanup.get("state") == "preparing" and not task_exists:
            _issue(issues, "cleanup_operation_incomplete", cleanup_path, "high")
        elif cleanup.get("state") == "completed" and task_exists:
            _issue(issues, "cleanup_completed_task_present", cleanup_path, "critical")
    checked = 0
    capability = clock_capability(cfg)
    for path in iter_json(paths["tasks"]):
        checked += 1
        try:
            task = TaskRecord.from_dict(read_json(path))
            if ready_state in {"active", "degraded"}:
                ready_issue = ready_task_projection_issue(cfg, task.task_id)
                if ready_issue is not None:
                    _issue(
                        issues,
                        "ready_projection_inconsistent",
                        path,
                        "high",
                        ready_issue,
                    )
            if path.stem != task.task_id:
                _issue(issues, "task_filename_id_mismatch", path, "high")
            submission = submissions.get(task.submission_operation_id or "")
            if not submission:
                _issue(issues, "task_submission_missing", path, "high")
            elif (task.state["projection"] in {"queued", "running"}
                  and submission.get("state") != "committed"):
                _issue(issues, "dispatch_visible_submission_uncommitted", path, "critical")
            if task.group_name:
                group_data = groups.get(task.group_name)
                if not group_data:
                    _issue(issues, "task_group_missing", path, "high")
                elif task.group_membership_sequence is None:
                    _issue(issues, "task_membership_sequence_missing", path, "high")
                elif task.placement_policy["home_machine"] not in group_data["group"].get(
                        "worker_set", {}):
                    _issue(issues, "task_home_outside_worker_set", path, "high")
            has_timed_offer = bool(task.placement_runtime.get("offer_eligible_at")
                                   and task.placement_runtime.get("offer_clock_evidence")
                                   and task.placement_runtime.get("queue_scope") == "home"
                                   and task.placement_policy.get("sharing_mode") == "spillover")
            if has_timed_offer and task.task_id not in deadline_task_ids:
                _issue(issues, "offer_deadline_index_missing", path, "high")
            claim = task.claim_control.get("active_claim") or {}
            number = task.attempt_control.get("current_attempt_number")
            if claim:
                mode = claim.get("authority_mode")
                if mode not in {"bounded_lease", "holder_bound"}:
                    _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if mode == "bounded_lease":
                    required_evidence = {"clock_error_bound_seconds", "clock_provider", "clock_observation_id", "lease_expires_at"}
                    if not required_evidence.issubset(claim) or not isinstance(
                            claim.get("clock_error_bound_seconds"), (int, float)):
                        _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if mode == "holder_bound" and any(claim.get(key) is not None for key in (
                        "clock_error_bound_seconds", "clock_provider", "clock_observation_id", "lease_expires_at")):
                    _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if number is None:
                    _issue(issues, "claim_attempt_number_missing", path, "critical")
                else:
                    attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
                    if not attempt_file.exists():
                        _issue(issues, "claim_attempt_missing", attempt_file, "critical")
                    else:
                        attempt = AttemptRecord.from_dict(read_json(attempt_file))
                        if (attempt.attempt_id != claim.get("attempt_id")
                                or attempt.current_fencing_token != claim.get("fencing_token")
                                or attempt.authority_mode != mode):
                            _issue(issues, "claim_attempt_token_mismatch", attempt_file, "critical")
                        if mode == "holder_bound" and attempt.machine_name != claim.get("machine_name"):
                            _issue(issues, "holder_bound_machine_mismatch", attempt_file, "critical")
                if mode == "bounded_lease":
                    try:
                        expires_at = datetime.fromisoformat(
                            claim["lease_expires_at"].replace("Z", "+00:00")
                        )
                        if expires_at <= datetime.now(timezone.utc):
                            _issue(issues, "active_claim_lease_expired", path, "high")
                    except (KeyError, TypeError, ValueError):
                        _issue(issues, "claim_lease_invalid", path, "high")
                if task.group_name and task.group_name in groups:
                    workers = groups[task.group_name]["group"].get("worker_set", {})
                    if claim.get("machine_name") not in workers:
                        _issue(issues, "claim_machine_outside_worker_set", path, "critical")
        except Exception as exc:
            _issue(issues, "task_invalid", path, "high", str(exc))
    for operation_id, submission in submissions.items():
        if submission.get("state") != "committed":
            continue
        for task_id in submission.get("resolved_context", {}).get("task_ids", []):
            if not task_path(cfg.shared_root, task_id).exists() and task_id not in cleaned:
                _issue(issues, "committed_submission_task_missing",
                       paths["submissions"] / f"{operation_id}.json", "critical", task_id)
    for mapping_path in iter_json(paths["idempotency"]):
        try:
            mapping = read_json(mapping_path)
        except (OSError, ValueError) as exc:
            _issue(issues, "idempotency_mapping_invalid", mapping_path, "high", str(exc))
            continue
        operation_id = mapping.get("operation_id")
        if not operation_id or operation_id not in submissions:
            _issue(issues, "idempotency_operation_missing", mapping_path, "high")
    for name, group_data in groups.items():
        for barrier in group_data["group"].get("cancellation_barriers", []):
            operation_file = paths["group_control"] / f"{barrier['operation_id']}.json"
            if not operation_file.exists():
                _issue(issues, "group_barrier_operation_missing", operation_file, "critical", name)
    for operation_path in iter_json(paths["group_control"]):
        try:
            control = read_json(operation_path).get("group_control", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "group_control_invalid", operation_path, "high", str(exc))
            continue
        if control.get("operation_type") != "cancel":
            continue
        group_data = groups.get(control.get("group_name"))
        barriers = group_data["group"].get("cancellation_barriers", []) if group_data else []
        if not any(item.get("operation_id") == control.get("operation_id") for item in barriers):
            _issue(issues, "cancellation_operation_barrier_missing", operation_path, "critical")
    for operation_path in iter_json(paths["availability"]):
        try:
            operation = read_json(operation_path).get("availability_operation", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "availability_operation_invalid", operation_path, "high", str(exc))
            continue
        task_id = operation.get("task_id")
        if operation.get("operation_id") != operation_path.stem or not isinstance(task_id, str):
            _issue(issues, "availability_operation_invalid", operation_path, "high")
            continue
        if (operation.get("state") == "prepared"
                or (operation.get("state") == "blocked" and not operation.get("blocked_reason"))):
            _issue(issues, "availability_operation_incomplete", operation_path, "high")
        if not task_path(cfg.shared_root, task_id).exists() and task_id not in cleaned:
            _issue(issues, "availability_operation_task_missing", operation_path, "high")
    for reservation_path in iter_json(local_paths(reservation_runtime_root)["active"]):
        try:
            reservation = read_json(reservation_path)["reservation"]
        except (KeyError, OSError, ValueError) as exc:
            _issue(issues, "reservation_invalid", reservation_path, "high", str(exc))
            continue
        if project_id is not None and reservation.get("project_id") != project_id:
            continue
        task_file = task_path(cfg.shared_root, reservation["task_id"])
        if not task_file.exists():
            code = ("cleaned_task_reservation_residual"
                    if reservation["task_id"] in cleaned else "reservation_task_missing")
            _issue(issues, code, task_file, "high")
            continue
        task = TaskRecord.from_dict(read_json(task_file))
        claim = task.claim_control.get("active_claim") or {}
        if task.state["projection"] == "blocked":
            number = task.attempt_control.get("current_attempt_number")
            attempt_file = attempt_path(cfg.shared_root, task.task_id, number) if number else None
            if not attempt_file or not attempt_file.exists():
                _issue(issues, "orphan_reservation_attempt_missing", task_file, "critical")
            else:
                attempt = AttemptRecord.from_dict(read_json(attempt_file))
                if (attempt.reservation_id != reservation["reservation_id"]
                        or attempt.current_fencing_token != reservation.get("fencing_token")):
                    _issue(issues, "orphan_reservation_attempt_mismatch", task_file, "critical")
        elif (claim.get("reservation_id") != reservation["reservation_id"]
              or claim.get("fencing_token") != reservation.get("fencing_token")):
            _issue(issues, "reservation_claim_mismatch", task_file, "critical")
    for task_attempts_dir in sorted(paths["attempts"].iterdir() if paths["attempts"].exists() else []):
        if not task_attempts_dir.is_dir():
            continue
        for attempt_file in iter_json(task_attempts_dir):
            try:
                attempt = AttemptRecord.from_dict(read_json(attempt_file))
            except Exception as exc:
                _issue(issues, "attempt_invalid", attempt_file, "high", str(exc))
                continue
            if attempt.task_id != task_attempts_dir.name:
                _issue(issues, "attempt_task_id_mismatch", attempt_file, "high")
            task_file = task_path(cfg.shared_root, attempt.task_id)
            if not task_file.exists():
                code = ("cleaned_task_attempt_residual"
                        if attempt.task_id in cleaned else "attempt_task_missing")
                _issue(issues, code, attempt_file, "high")
    for manifest_path in iter_json(cfg.runtime_root / "processes"):
        try:
            process = read_json(manifest_path).get("process", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "process_manifest_invalid", manifest_path, "high", str(exc))
            continue
        task_file = task_path(cfg.shared_root, process.get("task_id", ""))
        if not task_file.exists():
            code = ("cleaned_task_process_residual"
                    if process.get("task_id") in cleaned else "process_manifest_task_missing")
            _issue(issues, code, manifest_path, "high")
            continue
        task = TaskRecord.from_dict(read_json(task_file))
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            _issue(issues, "process_manifest_attempt_number_missing", manifest_path, "high")
            continue
        attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
        if not attempt_file.exists():
            _issue(issues, "process_manifest_attempt_missing", manifest_path, "critical")
            continue
        attempt = AttemptRecord.from_dict(read_json(attempt_file))
        if process.get("attempt_id") != attempt.attempt_id:
            _issue(issues, "process_manifest_attempt_mismatch", manifest_path, "critical")
        if process.get("fencing_token") not in attempt.token_history:
            _issue(issues, "process_manifest_token_unknown", manifest_path, "critical")
        claim = task.claim_control.get("active_claim") or {}
        if (claim.get("authority_mode") == "holder_bound"
                and (process.get("machine_name") != claim.get("machine_name")
                     or attempt.machine_name != claim.get("machine_name"))):
            _issue(issues, "holder_bound_machine_mismatch", manifest_path, "critical")
        from .scheduler import _process_evidence_state
        evidence_state = _process_evidence_state(attempt, process)
        if evidence_state in {"mismatch", "unverifiable"}:
            _issue(issues, f"process_identity_{evidence_state}", manifest_path, "high")
    for decision_path in list_decisions(cfg):
        try:
            decision = read_json(decision_path).get("termination_decision", {})
            if decision.get("shared_reconciliation") == "blocked":
                _issue(issues, "termination_reconciliation_blocked", decision_path, "high")
            if decision.get("state") not in {"confirmed", "superseded"}:
                _issue(issues, "termination_decision_incomplete", decision_path, "high")
        except (OSError, ValueError):
            _issue(issues, "termination_decision_invalid", decision_path, "high")
    return {"schema_version": 6, "tasks_checked": checked, "issues": issues, "healthy": not issues,
            "ready_index": read_ready_index_status(cfg),
            "clock_capability": {"status": capability.status, "reason": capability.reason,
                                 "provider": capability.observation.provider if capability.observation else None,
                                 "observation_id": capability.observation.observation_id if capability.observation else None,
                                 "scheduling_capability": "full" if capability.is_healthy else "local-safe"}}


def repair_metadata(
    cfg: RootConfig, *, reservation_runtime_root: Path | None = None
) -> dict[str, Any]:
    if reservation_runtime_root is None:
        from .machine_runtime import resolve_execution_context

        context = resolve_execution_context(cfg)
        cfg = context.local_cfg
        reservation_runtime_root = context.reservation_root
    repaired: list[str] = []
    blocked: list[str] = []
    initial_ready_state = read_ready_index_state(cfg)
    operations = shared_paths(cfg.shared_root)["submissions"]
    for path in iter_json(operations):
        operation = read_json(path)["submission"]
        group_name = operation.get("target_group")
        if not group_name:
            continue
        group_file = shared_paths(cfg.shared_root)["groups"] / f"{group_name}.json"
        if not group_file.exists():
            continue
        group_data = read_json(group_file)
        pending = group_data["group"].get("pending_submission_commit") or {}
        if pending.get("operation_id") != operation["operation_id"]:
            continue
        if operation["state"] in {"committed", "aborted", "blocked"}:
            group_data["group"]["pending_submission_commit"] = None
            group_data["meta"]["revision"] += 1
            group_data["meta"]["updated_at"] = operation.get("committed_at") or group_data["meta"]["updated_at"]
            atomic_replace(group_file, group_data)
            repaired.append(operation["operation_id"])
        else:
            blocked.append(operation["operation_id"])
    for result in reconcile_cleanup_operations(
            cfg, reservation_runtime_root=reservation_runtime_root
    ):
        if result["state"] == "completed":
            repaired.append(result["operation_id"])
        else:
            blocked.append(result["operation_id"])
    for result in reconcile_availability_operations(cfg):
        repaired.append(result["operation_id"])
    rebuilt_deadline_indexes = rebuild_deadline_indexes(cfg)
    if rebuilt_deadline_indexes:
        repaired.append(f"offer_deadline_indexes:{rebuilt_deadline_indexes}")
    group_controls = shared_paths(cfg.shared_root)["group_control"]
    for path in iter_json(group_controls):
        data = read_json(path)
        control = data.get("group_control", {})
        if control.get("operation_type") != "cancel" or control.get("state") not in {"converging", "waiting_ack"}:
            continue
        high_watermark = control["membership_high_watermark"]
        group_name = control["group_name"]
        post_commit_results = []
        with group_lock(cfg.shared_root, group_name):
            group_file = group_path(cfg.shared_root, group_name)
            if not group_file.exists():
                blocked.append(control["operation_id"])
                continue
            barriers = read_json(group_file)["group"].get("cancellation_barriers", [])
            if not any(item["operation_id"] == control["operation_id"] for item in barriers):
                control.update({"state": "blocked", "blocked_reason": "cancellation_barrier_missing"})
                data["meta"]["revision"] += 1
                atomic_replace(path, data)
                blocked.append(control["operation_id"])
                continue
            for task_path_value in iter_json(shared_paths(cfg.shared_root)["tasks"]):
                task = TaskRecord.from_dict(read_json(task_path_value))
                if task.group_name != group_name or (task.group_membership_sequence or 0) > high_watermark:
                    continue
                with task_lock(cfg.shared_root, task.task_id):
                    task = load_task(cfg, task.task_id)
                    has_saved_task = False
                    claim = task.claim_control.get("active_claim") or {}
                    if task.state["projection"] == "queued" and not claim:
                        task.state.update({"projection": "cancelled", "reason": "group_cancelled"})
                    elif claim.get("launch_state") == "claimed":
                        result = commit_terminal_transition_locked(
                            cfg, task, TerminalTransition(task.task_id, claim["attempt_id"],
                                task.attempt_control["current_attempt_number"], claim["fencing_token"],
                                "cancelled", "group_cancelled_before_launch", None,
                                frozenset({"running"}), frozenset({"claimed"}), "active",
                                allow_missing_attempt=True))
                        post_commit_results.append(result)
                        has_saved_task = result.outcome == "committed"
                    elif task.state["projection"] == "running" and control["terminate_running"]:
                        task.control.update({"cancellation_requested_at": utc_now(),
                                             "cancellation_operation_id": control["operation_id"],
                                             "terminate_running": True,
                                             "requested_by": cfg.machine_name})
                    if not has_saved_task:
                        task.meta["revision"] += 1
                        task.meta["updated_at"] = utc_now()
                        from .runtime.tasks import save_task

                        save_task(cfg, task)
                    if task.state["projection"] == "cancelled":
                        retire_current_ready_generation(cfg, task)
        for result in post_commit_results:
            if result.reservation_id and result.reservation_machine_name == cfg.machine_name:
                from .runtime.reservations import release
                release(reservation_runtime_root or cfg.runtime_root, result.reservation_id, "group_cancelled_before_launch")
            if result.event:
                dispatch_task_lifecycle_hooks_noexcept(cfg, result.event)
        pending: dict[str, list[str]] = {}
        acknowledged = 0
        blocked_tasks = 0
        for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(task_file))
            if task.group_name != control["group_name"] or (task.group_membership_sequence or 0) > high_watermark:
                continue
            if task.control.get("termination_acknowledged_at"):
                acknowledged += 1
                continue
            claim = task.claim_control.get("active_claim") or {}
            if task.state["projection"] == "running" and control["terminate_running"]:
                pending.setdefault(claim.get("machine_name") or task.placement_policy["home_machine"], []).append(task.task_id)
            elif task.state["projection"] == "blocked" and control["terminate_running"]:
                blocked_tasks += 1
        control["progress"]["termination_acknowledged"] = acknowledged
        control["progress"]["blocked"] = blocked_tasks
        control["pending_machine_acknowledgements"] = pending
        if blocked_tasks:
            control["state"] = "blocked"
            control["blocked_reason"] = "orphaned_tasks_require_resolution"
        elif not pending:
            control["state"] = "completed"
            control["completed_at"] = utc_now()
        else:
            control["state"] = "waiting_ack"
        control["updated_at"] = utc_now()
        data["meta"]["revision"] += 1
        atomic_replace(path, data)
        repaired.append(control["operation_id"])
    for control in reconcile_group_cancel_operations(cfg):
        if control["operation_id"] not in repaired:
            repaired.append(control["operation_id"])
    orphan_result = repair_orphans(cfg, reservation_runtime_root=reservation_runtime_root)
    repaired.extend(task_id for task_id in orphan_result["repaired"] if task_id not in repaired)
    blocked.extend(item["task_id"] for item in orphan_result["blocked"]
                   if item["task_id"] not in blocked)
    if read_ready_index_state(cfg) == "active":
        for ready_task_path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            issue = ready_task_projection_issue(cfg, ready_task_path.stem)
            if issue is not None:
                mark_ready_index_degraded(cfg, f"doctor_repair:{issue}")
                break
    ready_record = repair_ready_index(cfg, max_tasks=READY_BUILD_PAGE_SIZE)
    while ready_record.get("state") == "building":
        time.sleep(0)
        ready_record = repair_ready_index(cfg, max_tasks=READY_BUILD_PAGE_SIZE)
    ready_build = ready_record.get("build") or {}
    if ready_record.get("state") == "active" and initial_ready_state != "active":
        repaired.append(
            f"ready_index:{ready_build.get('repaired', 0)}:"
            f"{ready_build.get('stale_removed', 0)}"
        )
    elif ready_record.get("state") == "degraded":
        blocked.append("ready_index")
    return {"repaired": repaired, "blocked": blocked,
            "ready_index": {"state": ready_record.get("state"),
                            "build": ready_build,
                            "degraded_reasons": ready_record.get("degraded_reasons", [])},
            "message": "Submission, Group control, and ready-index operations reconciled."}


def repair_orphans(
    cfg: RootConfig, *, reservation_runtime_root: Path | None = None
) -> dict[str, Any]:
    from .runtime.recovery import recover_running_attempt
    from .scheduler import (_process_evidence_state, finalize_orphaned_attempt)
    repaired: list[str] = []
    blocked: list[dict[str, str]] = []
    for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(task_file))
        if task.state["projection"] != "blocked":
            continue
        number = task.attempt_control.get("current_attempt_number")
        if number is None:
            blocked.append({"task_id": task.task_id, "reason": "current_attempt_missing"})
            continue
        attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
        if not attempt_file.exists():
            blocked.append({"task_id": task.task_id, "reason": "attempt_truth_missing"})
            continue
        attempt = AttemptRecord.from_dict(read_json(attempt_file))
        if attempt.machine_name != cfg.machine_name:
            blocked.append({"task_id": task.task_id, "reason": "owning_machine_remote"})
            continue
        manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
        if not manifest_path.exists():
            blocked.append({"task_id": task.task_id, "reason": "process_evidence_missing"})
            continue
        process = read_json(manifest_path).get("process", {})
        if (process.get("task_id") != task.task_id
                or process.get("attempt_id") != attempt.attempt_id):
            blocked.append({"task_id": task.task_id, "reason": "process_identity_mismatch"})
            continue
        evidence_state = _process_evidence_state(attempt, process)
        if evidence_state == "alive":
            token = recover_running_attempt(
                cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                manifest=process,
                reservation_runtime_root=reservation_runtime_root,
            )
            if token is not None:
                repaired.append(task.task_id)
            else:
                blocked.append({"task_id": task.task_id, "reason": "recovery_cas_rejected"})
        elif evidence_state == "absent":
            is_terminated = bool(task.control.get("terminate_running"))
            if finalize_orphaned_attempt(
                    cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token,
                    exit_code=process.get("exit_code"),
                    was_terminated=is_terminated,
                    reservation_runtime_root=reservation_runtime_root,
            ):
                process["observed_state"] = "exited"
                process["reconciled_at"] = utc_now()
                atomic_replace(manifest_path, {"process": process})
                repaired.append(task.task_id)
            else:
                blocked.append({"task_id": task.task_id, "reason": "finalize_cas_rejected"})
        else:
            blocked.append({"task_id": task.task_id,
                            "reason": f"process_identity_{evidence_state}"})
    return {"repaired": repaired, "blocked": blocked,
            "message": "Local orphan evidence reconciled where authority was provable."}


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
    rebuilt = rebuild_deadline_indexes(cfg)
    return {"rebuilt": bool(rebuilt), "rebuilt_records": rebuilt,
            "message": "Derived deadline indexes rebuilt from Task truth."}


def cleanup_stale_locks(cfg: RootConfig) -> dict[str, Any]:
    return {"removed": [], "message": "Ownership is never inferred from lock age."}


def build_verify_jsonl_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"record_type": "summary", **result}, *result.get("issues", [])]


def normalize_verify_severity(value: str) -> str:
    return value


def resolve_verify_exit_code(result: dict[str, Any], *, strict: bool = False, fail_on: str | None = None) -> int:
    return 1 if result.get("issues") and (strict or fail_on) else 0
