"""Non-mutating integrity checks and explicit safe repairs."""

from __future__ import annotations

import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import validate_root_contract
from .lease import clock_capability
from .runtime.availability import rebuild_deadline_indexes
from .runtime.group_namespace import group_directory
from .runtime.locks import schema_lock, schema_reader_lock
from .runtime.observation.projection import inspect_observation, observation_path
from .runtime.paths import attempt_path, local_paths, shared_paths, task_path
from .runtime.process_evidence import ProcessEvidence, inspect_group_identity
from .runtime.ready import (
    READY_BUILD_PAGE_SIZE,
    mark_ready_index_degraded,
    parse_ready_reason,
    read_ready_index_state,
    read_ready_index_status,
    ready_task_projection_issue,
)
from .runtime.ready.group_members import (
    GROUP_MEMBER_PAGE_SIZE,
    group_ready_members_state,
    mark_group_ready_members_degraded,
    read_group_ready_members_state,
)
from .runtime.ready.group_members_rebuild import audit_group_ready_members
from .runtime.records import AttemptRecord, TaskRecord, normalize_group_record, utc_now
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.submission_control import (
    control_paths,
    inspect_submission_control,
    read_submission_state,
    request_control_rebuild,
)
from .runtime.termination import list_decisions

_TASK_OBSERVATION_INSTRUCTIONS = (
    "Paginated list unavailable until the global agent completes the background build; "
    "qexp admin repair --project PATH requests a rebuild on a damaged index."
)


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


def _issue(issues: list[dict[str, Any]], code: str, path: Any, severity: str, message: str | None = None) -> None:
    issue = {"code": code, "path": str(path), "severity": severity}
    if message:
        issue["message"] = message
    issues.append(issue)


def _process_identity_issue(evidence: ProcessEvidence) -> str | None:
    if evidence.state != "unknown":
        return None
    return "process_identity_mismatch" if evidence.reason == "identity_mismatch" else "process_identity_unverifiable"


def _records_by_stem(
    directory: Any, key: str, issues: list[dict[str, Any]], invalid_code: str
) -> dict[str, dict[str, Any]]:
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
    max_work_items: int = GROUP_MEMBER_PAGE_SIZE,
) -> dict[str, Any]:
    """Verify shared truth and the selected project's local execution evidence.

    Args:
        cfg: Project configuration using the authoritative local runtime root.
        reservation_runtime_root: Reservation store to verify.
        project_id: Project filter for a machine-wide reservation store.

    Returns:
        Integrity summary and discovered issues.
    """
    if type(max_work_items) is not int or not 1 <= max_work_items <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_work_items must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    if reservation_runtime_root is None:
        from .agent.context import resolve_execution_context

        context = resolve_execution_context(cfg)
        cfg = context.local_cfg
        reservation_runtime_root = context.reservation_root
        project_id = context.project_id
    validate_root_contract(cfg)
    issues: list[dict[str, Any]] = []
    ready_state = read_ready_index_state(cfg)
    if ready_state == "degraded":
        ready_status = read_ready_index_status(cfg)
        _issue(
            issues,
            "ready_index_degraded",
            shared_paths(cfg.shared_root)["ready"] / "state.json",
            "high",
            ";".join(ready_status.get("degraded_reasons", [])),
        )
    member_state = group_ready_members_state(cfg)
    member_verification: dict[str, Any] = {"state": "degraded" if member_state == "degraded" else "building"}
    if member_state == "degraded":
        _issue(
            issues,
            "group_ready_members_degraded",
            shared_paths(cfg.shared_root)["ready_group_members"] / "state.json",
            "high",
        )
    paths = shared_paths(cfg.shared_root)
    cleaned = _cleaned_task_ids(cfg)
    submissions = _records_by_stem(paths["submissions"], "submission", issues, "submission_invalid")
    with schema_reader_lock(cfg.shared_root):
        groups_root = group_directory(cfg.shared_root)
        group_records = _records_by_stem(groups_root, "group", issues, "group_invalid")
    for name, group in group_records.items():
        try:
            normalize_group_record({"group": group})
        except (TypeError, ValueError) as exc:
            _issue(
                issues,
                "group_worker_invalid",
                groups_root / f"{name}.json",
                "high",
                str(exc),
            )
    groups = {name: {"group": group} for name, group in group_records.items()}
    provisional_groups: list[dict[str, Any]] = []
    for name, group in group_records.items():
        operation_id = group.get("creation_operation_id")
        if operation_id is None:
            continue
        try:
            state = read_submission_state(cfg.shared_root, operation_id)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            state = "unavailable"
            _issue(
                issues,
                "submission_group_publication_unavailable",
                groups_root / f"{name}.json",
                "high",
                str(exc),
            )
        if state != "committed":
            provisional_groups.append({"name": name, "operation_id": operation_id, "state": state})
            _issue(
                issues,
                "submission_group_provisional",
                groups_root / f"{name}.json",
                "high",
                f"operation_id={operation_id};state={state}",
            )
    if member_state == "active":
        try:
            with schema_lock(cfg.shared_root):
                member_record = audit_group_ready_members(cfg, max_work_items=max_work_items)
            audit = member_record.get("audit") or {}
            member_verification = {
                "state": audit.get("state", "building"),
                "audit_id": audit.get("audit_id"),
                "projection_id": member_record.get("projection_id"),
                "phase": audit.get("phase"),
                "processed": audit.get("processed", 0),
            }
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # A verify finding is itself an admission-safety event.  Do not
            # leave an incomplete derived projection eligible for borrow.
            with schema_lock(cfg.shared_root):
                mark_group_ready_members_degraded(cfg, f"doctor_verify:{type(exc).__name__}")
            _issue(
                issues,
                "group_ready_members_inconsistent",
                shared_paths(cfg.shared_root)["ready_group_members"] / "state.json",
                "high",
                str(exc),
            )
            member_verification = {"state": "degraded", "reason": "audit_failed"}
    elif member_state == "building":
        member_verification = {"state": "building", "reason": "projection_rebuild"}
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
            elif task.state["projection"] in {"queued", "running"} and submission.get("state") != "committed":
                _issue(issues, "dispatch_visible_submission_uncommitted", path, "critical")
            if task.group_name:
                group_data = groups.get(task.group_name)
                if not group_data:
                    _issue(issues, "task_group_missing", path, "high")
                elif task.group_membership_sequence is None:
                    _issue(issues, "task_membership_sequence_missing", path, "high")
                elif task.placement_policy["home_machine"] not in group_data["group"].get("worker_set", {}):
                    _issue(issues, "task_home_outside_worker_set", path, "high")
            has_timed_offer = bool(
                task.placement_runtime.get("offer_eligible_at")
                and task.placement_runtime.get("offer_clock_evidence")
                and task.placement_runtime.get("queue_scope") == "home"
                and task.placement_policy.get("sharing_mode") == "spillover"
            )
            if has_timed_offer and task.task_id not in deadline_task_ids:
                _issue(issues, "offer_deadline_index_missing", path, "high")
            claim = task.claim_control.get("active_claim") or {}
            number = task.attempt_control.get("current_attempt_number")
            if claim:
                mode = claim.get("authority_mode")
                if mode not in {"bounded_lease", "holder_bound"}:
                    _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if mode == "bounded_lease":
                    required_evidence = {
                        "clock_error_bound_seconds",
                        "clock_provider",
                        "clock_observation_id",
                        "lease_expires_at",
                    }
                    if not required_evidence.issubset(claim) or not isinstance(
                        claim.get("clock_error_bound_seconds"), (int, float)
                    ):
                        _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if mode == "holder_bound" and any(
                    claim.get(key) is not None
                    for key in (
                        "clock_error_bound_seconds",
                        "clock_provider",
                        "clock_observation_id",
                        "lease_expires_at",
                    )
                ):
                    _issue(issues, "authority_mode_evidence_invalid", path, "critical")
                if number is None:
                    _issue(issues, "claim_attempt_number_missing", path, "critical")
                else:
                    attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
                    if not attempt_file.exists():
                        _issue(issues, "claim_attempt_missing", attempt_file, "critical")
                    else:
                        attempt = AttemptRecord.from_dict(read_json(attempt_file))
                        if (
                            attempt.attempt_id != claim.get("attempt_id")
                            or attempt.current_fencing_token != claim.get("fencing_token")
                            or attempt.authority_mode != mode
                        ):
                            _issue(issues, "claim_attempt_token_mismatch", attempt_file, "critical")
                        if mode == "holder_bound" and attempt.machine_name != claim.get("machine_name"):
                            _issue(issues, "holder_bound_machine_mismatch", attempt_file, "critical")
                if mode == "bounded_lease":
                    try:
                        expires_at = datetime.fromisoformat(claim["lease_expires_at"].replace("Z", "+00:00"))
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
                _issue(
                    issues,
                    "committed_submission_task_missing",
                    paths["submissions"] / f"{operation_id}.json",
                    "critical",
                    task_id,
                )
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
        if operation.get("state") == "prepared" or (
            operation.get("state") == "blocked" and not operation.get("blocked_reason")
        ):
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
            code = (
                "cleaned_task_reservation_residual" if reservation["task_id"] in cleaned else "reservation_task_missing"
            )
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
                if attempt.reservation_id != reservation[
                    "reservation_id"
                ] or attempt.current_fencing_token != reservation.get("fencing_token"):
                    _issue(issues, "orphan_reservation_attempt_mismatch", task_file, "critical")
        elif claim.get("reservation_id") != reservation["reservation_id"] or claim.get(
            "fencing_token"
        ) != reservation.get("fencing_token"):
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
                code = "cleaned_task_attempt_residual" if attempt.task_id in cleaned else "attempt_task_missing"
                _issue(issues, code, attempt_file, "high")
    for manifest_path in iter_json(cfg.runtime_root / "processes"):
        try:
            process = read_json(manifest_path).get("process", {})
        except (OSError, ValueError) as exc:
            _issue(issues, "process_manifest_invalid", manifest_path, "high", str(exc))
            continue
        task_file = task_path(cfg.shared_root, process.get("task_id", ""))
        if not task_file.exists():
            code = (
                "cleaned_task_process_residual"
                if process.get("task_id") in cleaned
                else "process_manifest_task_missing"
            )
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
        matches_attempt = process.get("attempt_id") == attempt.attempt_id
        if not matches_attempt:
            _issue(issues, "process_manifest_attempt_mismatch", manifest_path, "critical")
        if process.get("fencing_token") not in attempt.token_history:
            _issue(issues, "process_manifest_token_unknown", manifest_path, "critical")
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("authority_mode") == "holder_bound" and (
            process.get("machine_name") != claim.get("machine_name")
            or attempt.machine_name != claim.get("machine_name")
        ):
            _issue(issues, "holder_bound_machine_mismatch", manifest_path, "critical")
        if matches_attempt:
            evidence_issue = _process_identity_issue(inspect_group_identity(attempt.process, process))
            if evidence_issue is not None:
                _issue(issues, evidence_issue, manifest_path, "high")
    for decision_path in list_decisions(cfg):
        try:
            decision = read_json(decision_path).get("termination_decision", {})
            if decision.get("shared_reconciliation") == "blocked":
                _issue(issues, "termination_reconciliation_blocked", decision_path, "high")
            if decision.get("state") not in {"confirmed", "superseded"}:
                _issue(issues, "termination_decision_incomplete", decision_path, "high")
        except (OSError, ValueError):
            _issue(issues, "termination_decision_invalid", decision_path, "high")
    try:
        member_status = read_group_ready_members_state(cfg)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        member_status = {}
    task_observation = inspect_observation(cfg)
    observation_state = task_observation.get("state")
    if observation_state == "degraded" or (
        observation_state not in {"absent", "building"} and task_observation.get("dirty")
    ):
        _issue(
            issues,
            "task_observation_unavailable",
            observation_path(cfg) / "state.json",
            "low",
            _TASK_OBSERVATION_INSTRUCTIONS,
        )
    submission_control = inspect_submission_control(cfg)
    if submission_control["state"] == "unavailable":
        _issue(
            issues,
            "submission_control_unavailable",
            control_paths(cfg)["state"],
            "low",
            "Run 'qexp admin repair --project PATH' to restart background Submission visibility certification.",
        )
    incomplete_reasons: list[str] = []
    if member_verification["state"] == "building":
        incomplete_reasons.append("group_ready_members_building")
    if task_observation.get("state") == "building":
        incomplete_reasons.append("task_observation_building")
    if submission_control.get("state") == "waiting":
        incomplete_reasons.append("submission_control_waiting")
    is_complete = not incomplete_reasons
    is_healthy = is_complete and not issues
    issue_counts = dict(sorted(Counter(item.get("severity", "unknown") for item in issues).items()))
    if is_healthy:
        outcome = "healthy"
    elif any(item.get("severity") in {"critical", "high"} for item in issues):
        outcome = "unhealthy"
    else:
        outcome = "partial"
    repair_command = "qexp admin repair --project " + shlex.quote(str(cfg.project_root))
    return {
        "schema_version": 6,
        "tasks_checked": checked,
        "issues": issues,
        "issue_counts": issue_counts,
        "incomplete_reasons": incomplete_reasons,
        "complete": is_complete,
        "healthy": is_healthy,
        "outcome": outcome,
        "repair_command": None if is_healthy else repair_command,
        "next_action": None if is_healthy else repair_command,
        "ready_index": read_ready_index_status(cfg),
        "group_ready_members": {
            "state": group_ready_members_state(cfg),
            "verification": member_verification,
            "projection_id": member_status.get("projection_id"),
            "archive_count": member_status.get("archive_count", 0),
            "archive_cleanup": member_status.get("archive_cleanup"),
        },
        "clock_capability": {
            "status": capability.status,
            "reason": capability.reason,
            "provider": capability.observation.provider if capability.observation else None,
            "observation_id": capability.observation.observation_id if capability.observation else None,
            "scheduling_capability": "full" if capability.is_healthy else "local-safe",
        },
        "task_observation": task_observation,
        "submission_control": submission_control,
    }


def repair_metadata(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    max_work_items: int = GROUP_MEMBER_PAGE_SIZE,
    retry_intervention: bool = False,
) -> dict[str, Any]:
    if type(max_work_items) is not int or not 1 <= max_work_items <= GROUP_MEMBER_PAGE_SIZE:
        raise ValueError(f"max_work_items must be between 1 and {GROUP_MEMBER_PAGE_SIZE}.")
    from .runtime.maintenance import CONTEXT_RESOLUTION_OPERATIONS, advance_full_audit, create_invocation_ledger

    ledger = create_invocation_ledger(max_work_items)
    if reservation_runtime_root is None:
        from .agent.context import resolve_execution_context

        context_reservation = ledger.reserve_operations(maximum_operations=CONTEXT_RESOLUTION_OPERATIONS)
        if context_reservation is not None:
            try:
                context = resolve_execution_context(cfg)
                cfg = context.local_cfg
                reservation_runtime_root = context.reservation_root
            finally:
                context_reservation.commit(actual_operations=CONTEXT_RESOLUTION_OPERATIONS)

    # A damaged Submission-control fence blocks all source-backed reads. Fail
    # closed into a replayable build state before spending this explicit audit
    # slice on unrelated project records; the outbox descriptor owns the build.
    if inspect_submission_control(cfg).get("state") == "unavailable":
        request_control_rebuild(cfg)

    progress = advance_full_audit(
        cfg,
        reservation_runtime_root=reservation_runtime_root,
        max_work_items=max_work_items,
        ledger=ledger,
        retry_intervention=retry_intervention,
    )
    repaired = list(progress["repaired"])
    blocked = list(progress["blocked"])
    complete = progress["complete"] is True
    rerun_required = not complete or bool(blocked)
    if blocked:
        outcome = "blocked"
    elif not complete:
        outcome = "partial"
    elif repaired:
        outcome = "repaired"
    else:
        outcome = "no_change"
    next_action = "qexp admin repair --project " + shlex.quote(str(cfg.project_root)) if rerun_required else None
    if blocked and next_action is not None:
        next_action += " --retry-intervention"
    if outcome == "partial":
        message = f"Full repair slice is incomplete; rerun {next_action!r} to continue."
    elif outcome == "blocked":
        message = "Full-project repair is blocked; inspect the intervention evidence before rerunning."
    else:
        message = "Full-project repair completed."
    member_record = progress["group_ready_members"]
    audit = member_record.get("audit") if isinstance(member_record.get("audit"), dict) else {}
    member_record = {
        **member_record,
        "verification": {
            "state": "building" if member_record.get("state") == "building" else audit.get("state", "degraded"),
            "audit_id": audit.get("audit_id"),
            "projection_id": member_record.get("projection_id"),
            "phase": audit.get("phase"),
            "processed": audit.get("processed", 0),
        },
    }
    return {
        "repaired": repaired,
        "blocked": blocked,
        "repaired_count": len(repaired),
        "blocked_count": len(blocked),
        "outcome": outcome,
        "complete": complete,
        "scope": progress["scope"],
        "budget": progress["budget"],
        "phase": progress["phase"],
        "cursor": progress["cursor"],
        "deferred_phases": progress["deferred_phases"],
        "deferred_phase_count": progress["deferred_phase_count"],
        "next_due_at": progress["next_due_at"],
        "remaining_work": progress["remaining_work"],
        "source_capture": progress["source_capture"],
        "source_revision": progress["source_revision"],
        "build_identity": progress["build_identity"],
        "due_at": progress["due_at"],
        "retry_count": progress["retry_count"],
        "meaningful_progress_at": progress["meaningful_progress_at"],
        "failure": progress["failure"],
        "rerun_required": rerun_required,
        "next_action": next_action,
        "ready_index": progress["ready_index"],
        "group_ready_members": member_record,
        "prior_degraded_reasons": progress["prior_degraded_reasons"],
        "task_observation": progress["task_observation"],
        "submission_control": progress["submission_control"],
        "message": message,
        "intervention": progress["intervention"],
    }


def repair_orphans(cfg: RootConfig, *, reservation_runtime_root: Path | None = None) -> dict[str, Any]:
    from .runtime.attempt_recovery import recover_running_attempt
    from .scheduler import finalize_orphaned_attempt

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
        if process.get("task_id") != task.task_id or process.get("attempt_id") != attempt.attempt_id:
            blocked.append({"task_id": task.task_id, "reason": "process_identity_mismatch"})
            continue
        evidence = inspect_group_identity(attempt.process, process)
        if evidence.state == "alive":
            token = recover_running_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                manifest=process,
                reservation_runtime_root=reservation_runtime_root,
            )
            if token is not None:
                repaired.append(task.task_id)
            else:
                blocked.append({"task_id": task.task_id, "reason": "recovery_cas_rejected"})
        elif evidence.state == "absent":
            is_terminated = bool(task.control.get("terminate_running"))
            if finalize_orphaned_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
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
            blocked.append(
                {
                    "task_id": task.task_id,
                    "reason": _process_identity_issue(evidence) or "process_identity_unverifiable",
                }
            )
    submission_control = inspect_submission_control(cfg)
    if submission_control["state"] == "unavailable":
        try:
            submission_control = request_control_rebuild(cfg)
        except (OSError, ValueError, RuntimeError):
            blocked.append("submission_control")
        else:
            if submission_control["state"] == "waiting":
                blocked.append("submission_control")
    return {
        "repaired": repaired,
        "blocked": blocked,
        "message": "Local orphan evidence reconciled where authority was provable.",
    }


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
    rebuilt = rebuild_deadline_indexes(cfg)
    return {
        "rebuilt": bool(rebuilt),
        "rebuilt_records": rebuilt,
        "message": "Derived deadline indexes rebuilt from Task truth.",
    }


def cleanup_stale_locks(cfg: RootConfig) -> dict[str, Any]:
    return {"removed": [], "message": "Ownership is never inferred from lock age."}


def build_verify_jsonl_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"record_type": "summary", **result}, *result.get("issues", [])]


def normalize_verify_severity(value: str) -> str:
    return value


def resolve_verify_exit_code(result: dict[str, Any], *, strict: bool = False, fail_on: str | None = None) -> int:
    # Repair is a mutating convergence command: a blocked item is a failed
    # repair even when other items were repaired successfully.  Keep the
    # historical verify/check policy for non-strict checks below.
    if result.get("blocked"):
        return 1
    if strict and (not result.get("complete", True) or not result.get("healthy", False)):
        return 1
    return 1 if result.get("issues") and (strict or fail_on) else 0
