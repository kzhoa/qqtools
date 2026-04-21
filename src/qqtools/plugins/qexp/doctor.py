from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch_state import (
    build_batch_summary_from_counts,
    collect_batch_task_counts,
    resolve_declared_batch_total,
)
from .indexes import (
    collect_index_drift_report,
    rebuild_all_indexes,
    sync_task_state_index,
    update_index_on_phase_change,
)
from .locking import clean_lock
from .locking import submit_lock, task_operation_lock
from .layout import (
    RootConfig,
    global_batches_dir,
    global_events_dir,
    global_locks_dir,
    global_tasks_dir,
    list_forbidden_truth_layout_dirs,
    validate_root_contract,
    write_root_manifest,
)
from .lifecycle import read_agent_snapshot
from .models import (
    ACTIVE_PHASES,
    AGENT_STATE_FAILED,
    AGENT_STATE_STALE,
    AGENT_STATE_STOPPED,
    BATCH_COMMIT_ABORTED,
    BATCH_COMMIT_COMMITTED,
    BATCH_COMMIT_PREPARING,
    BatchSummary,
    PHASE_ORPHANED,
    RESUBMIT_STATE_ABORTED,
    RESUBMIT_STATE_COMMITTED,
    RESUBMIT_STATE_CREATING_NEW,
    RESUBMIT_STATE_DELETING_OLD,
    RESUBMIT_STATE_PREPARING,
    ROOT_SCOPE_PROJECT,
    TERMINAL_PHASES,
    RootGovernanceSnapshot,
    Task,
    utc_now_iso,
    validate_phase_transition,
)
from .storage import (
    CASConflict,
    cas_update_task,
    iter_all_tasks,
    iter_machines,
    iter_resubmit_operations,
    load_batch,
    load_resubmit_operation,
    load_task,
    save_resubmit_operation,
    save_batch,
    read_json,
    load_claim,
    release_claim,
)

log = logging.getLogger(__name__)

VERIFY_SEVERITIES = ("ok", "low", "medium", "high")
VERIFY_POLICY_EXIT_CODE = 2
VERIFY_CATEGORIES = (
    "root_contract",
    "truth_layout",
    "task_truth",
    "batch_truth",
    "resubmit_operation",
    "governance",
    "derived_index",
)
VERIFY_ACTION_POLICIES = (
    {
        "action_code": "run_doctor_repair_resubmit",
        "command": "qexp doctor repair",
        "reason": "unfinished resubmit or metadata repair operation needs truth convergence",
        "blocking": True,
        "issue_codes": (
            "resubmit_terminal_operation_left_behind",
            "resubmit_missing_new_submission_command",
            "resubmit_invalid_prepared_snapshot",
            "resubmit_prepared_snapshot_task_id_mismatch",
            "resubmit_illegal_batch_target",
            "resubmit_visible_truth_snapshot_mismatch",
        ),
    },
    {
        "action_code": "run_doctor_repair_batch_truth",
        "command": "qexp doctor repair",
        "reason": "batch truth needs metadata convergence",
        "blocking": True,
        "issue_codes": (
            "batch_missing_task_reference",
            "batch_summary_drift",
            "batch_committed_missing_tasks",
            "batch_preparing_complete_task_set",
        ),
    },
    {
        "action_code": "run_doctor_rebuild_index_state",
        "command": "qexp doctor rebuild-index",
        "reason": "runtime-critical state index drift must be rebuilt from truth",
        "blocking": True,
        "issue_codes": ("derived_index_state_drift",),
    },
    {
        "action_code": "manual_fix_truth_layout",
        "command": "manual_fix_required",
        "reason": "forbidden truth-layout directories must be removed or migrated before repair",
        "blocking": True,
        "issue_codes": ("forbidden_truth_layout_directory",),
    },
    {
        "action_code": "manual_fix_truth_corruption",
        "command": "manual_fix_required",
        "reason": "truth files are unreadable or root contract is broken; repair commands cannot safely infer intent",
        "blocking": True,
        "issue_codes": (
            "invalid_root_manifest",
            "task_truth_unreadable",
            "batch_truth_unreadable",
        ),
    },
)


def _build_verify_issue(
    *,
    code: str,
    category: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    normalized_severity = normalize_verify_severity(severity)
    if category not in VERIFY_CATEGORIES:
        raise ValueError(
            f"Unsupported verify category {category!r}; "
            f"expected one of {', '.join(VERIFY_CATEGORIES)}."
        )
    issue = {
        "code": code,
        "category": category,
        "severity": normalized_severity,
        "message": message,
    }
    if details:
        issue["details"] = details
    return issue


def _append_verify_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    category: str,
    severity: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(
        _build_verify_issue(
            code=code,
            category=category,
            severity=severity,
            message=message,
            **details,
        )
    )


def _issue_codes(issues: list[dict[str, Any]]) -> set[str]:
    return {issue["code"] for issue in issues}


def _issue_categories(issues: list[dict[str, Any]]) -> set[str]:
    return {issue["category"] for issue in issues}


def _issue_count_by_category(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        category = issue["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def _issue_count_by_code(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        code = issue["code"]
        counts[code] = counts.get(code, 0) + 1
    return counts


def normalize_verify_severity(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VERIFY_SEVERITIES:
        raise ValueError(
            f"Unsupported verify severity {value!r}; "
            f"expected one of {', '.join(VERIFY_SEVERITIES)}."
        )
    return normalized


def verify_severity_rank(value: str) -> int:
    normalized = normalize_verify_severity(value)
    return VERIFY_SEVERITIES.index(normalized)


def verify_matches_fail_policy(result: dict[str, Any], fail_on: str) -> bool:
    threshold = verify_severity_rank(fail_on)
    actual = verify_severity_rank(result["severity"])
    return actual >= threshold


def resolve_verify_exit_code(
    result: dict[str, Any],
    *,
    strict: bool = False,
    fail_on: str | None = None,
) -> int:
    effective_fail_on = fail_on
    if strict and effective_fail_on is None:
        effective_fail_on = "low"
    if effective_fail_on is None:
        return 0
    if verify_matches_fail_policy(result, effective_fail_on):
        return VERIFY_POLICY_EXIT_CODE
    return 0


def build_verify_jsonl_records(
    result: dict[str, Any],
    *,
    strict: bool = False,
    fail_on: str | None = None,
) -> list[dict[str, Any]]:
    effective_fail_on = fail_on
    if strict and effective_fail_on is None:
        effective_fail_on = "low"
    exit_code = resolve_verify_exit_code(
        result,
        strict=strict,
        fail_on=effective_fail_on,
    )
    records: list[dict[str, Any]] = [
        {
            "type": "verify_summary",
            "ok": result["ok"],
            "severity": result["severity"],
            "tasks_checked": result["tasks_checked"],
            "batches_checked": result["batches_checked"],
            "resubmit_ops_checked": result["resubmit_ops_checked"],
            "issue_count": len(result["issues"]),
            "issue_count_by_category": result["issue_count_by_category"],
            "issue_count_by_code": result["issue_count_by_code"],
            "strict": strict,
            "fail_on": effective_fail_on,
            "exit_code": exit_code,
        }
    ]
    for index, issue in enumerate(result["issues"], start=1):
        records.append(
            {
                "type": "verify_issue",
                "index": index,
                "issue_code": issue["code"],
                "category": issue["category"],
                "severity": issue["severity"],
                "message": issue["message"],
                "details": issue.get("details", {}),
            }
        )
    for index, action in enumerate(result["recommended_actions"], start=1):
        records.append(
            {
                "type": "verify_recommendation",
                "index": index,
                "action_code": action["action_code"],
                "command": action["command"],
                "blocking": action["blocking"],
                "reason": action["reason"],
            }
        )
    records.append(
        {
            "type": "verify_result",
            "ok": result["ok"],
            "severity": result["severity"],
            "diagnosis": result["diagnosis"],
            "index_drift": result["index_drift"],
            "governance": result["governance"],
            "root_manifest": result["root_manifest"],
            "forbidden_truth_dirs": result["forbidden_truth_dirs"],
            "issue_count_by_category": result["issue_count_by_category"],
            "issue_count_by_code": result["issue_count_by_code"],
            "recommended_actions": result["recommended_actions"],
            "strict": strict,
            "fail_on": effective_fail_on,
            "exit_code": exit_code,
        }
    )
    return records


def _append_recommendation(
    recommendations: list[dict[str, Any]],
    *,
    action_code: str,
    command: str,
    reason: str,
    blocking: bool,
) -> None:
    for item in recommendations:
        if (
            item["action_code"] == action_code
            and item["command"] == command
            and item["reason"] == reason
            and item["blocking"] is blocking
        ):
            return
    recommendations.append(
        {
            "action_code": action_code,
            "command": command,
            "reason": reason,
            "blocking": blocking,
        }
    )


def _policy_matches_issue_codes(
    issue_codes: set[str],
    policy: dict[str, Any],
) -> bool:
    return bool(issue_codes & set(policy["issue_codes"]))


def _verify_severity_from_issues(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "ok"
    max_rank = max(verify_severity_rank(issue["severity"]) for issue in issues)
    return VERIFY_SEVERITIES[max_rank]


def _build_verify_recommendations(
    issues: list[dict[str, Any]],
    index_drift: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    issue_codes = _issue_codes(issues)
    for policy in VERIFY_ACTION_POLICIES:
        if not _policy_matches_issue_codes(issue_codes, policy):
            continue
        _append_recommendation(
            recommendations,
            action_code=policy["action_code"],
            command=policy["command"],
            reason=policy["reason"],
            blocking=policy["blocking"],
        )
    return recommendations


def _count_event_files(base: Path) -> int:
    if not base.is_dir():
        return 0
    total = 0
    for path in base.rglob("*.json"):
        if not path.name.endswith(".tmp"):
            total += 1
    return total


def collect_governance_snapshot(cfg: RootConfig) -> RootGovernanceSnapshot:
    total_tasks = 0
    terminal_tasks = 0
    tasks_dir = global_tasks_dir(cfg)
    if tasks_dir.is_dir():
        for path in tasks_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                task = Task.from_dict(read_json(path))
            except Exception:
                continue
            total_tasks += 1
            if task.status.phase in TERMINAL_PHASES:
                terminal_tasks += 1
    return RootGovernanceSnapshot(
        total_tasks=total_tasks,
        terminal_tasks=terminal_tasks,
        total_batches=len(list(global_batches_dir(cfg).glob("*.json")))
        if global_batches_dir(cfg).is_dir()
        else 0,
        total_events=_count_event_files(global_events_dir(cfg)),
        total_machines=len(iter_machines(cfg)),
        updated_at=utc_now_iso(),
    )


def refresh_root_manifest_governance(cfg: RootConfig) -> RootGovernanceSnapshot:
    manifest = validate_root_contract(cfg)
    manifest.governance = collect_governance_snapshot(cfg)
    write_root_manifest(cfg, manifest)
    return manifest.governance


def _advance_resubmit_operation_state(cfg: RootConfig, operation, state: str):
    operation.state = state
    operation.meta.revision += 1
    operation.meta.updated_at = utc_now_iso()
    operation.meta.updated_by_machine = cfg.machine_name
    save_resubmit_operation(cfg, operation)
    return operation


def _resubmit_snapshot_matches_current_task(operation, task: Task) -> bool:
    expected = Task.from_dict(operation.new_task_snapshot)
    return (
        task.task_id == expected.task_id
        and task.name == expected.name
        and task.group == expected.group
        and task.batch_id == expected.batch_id
        and task.machine_name == expected.machine_name
        and task.attempt == expected.attempt
        and task.spec.command == expected.spec.command
        and task.spec.requested_gpus == expected.spec.requested_gpus
        and task.lineage.retry_of == expected.lineage.retry_of
        and task.timestamps.created_at == expected.timestamps.created_at
        and task.timestamps.queued_at == expected.timestamps.queued_at
    )


def _repair_resubmit_operation(cfg: RootConfig, operation) -> str:
    from .api import _delete_task_truth, _materialize_resubmit_task, _persist_submitted_task_truth
    from .events import write_event
    from .storage import delete_resubmit_operation

    task_id = operation.task_id
    with task_operation_lock(cfg, task_id):
        with submit_lock(cfg):
            try:
                current = load_resubmit_operation(cfg, task_id)
            except FileNotFoundError:
                return "missing"
            operation = current

            old_task = None
            try:
                old_task = load_task(cfg, task_id)
            except FileNotFoundError:
                pass

            if operation.state == RESUBMIT_STATE_PREPARING and old_task is not None:
                if _resubmit_snapshot_matches_current_task(operation, old_task):
                    operation = _advance_resubmit_operation_state(
                        cfg, operation, RESUBMIT_STATE_COMMITTED
                    )
                else:
                    operation = _advance_resubmit_operation_state(
                        cfg, operation, RESUBMIT_STATE_DELETING_OLD
                    )
            elif operation.state == RESUBMIT_STATE_PREPARING and old_task is None:
                operation = _advance_resubmit_operation_state(
                    cfg, operation, RESUBMIT_STATE_CREATING_NEW
                )

            if operation.state == RESUBMIT_STATE_DELETING_OLD:
                if old_task is not None:
                    if _resubmit_snapshot_matches_current_task(operation, old_task):
                        raise RuntimeError(
                            f"Resubmit operation for task {task_id} is inconsistent: "
                            "operation is deleting_old but visible task truth already matches "
                            "the prepared replacement snapshot."
                        )
                    _delete_task_truth(cfg, old_task)
                operation = _advance_resubmit_operation_state(
                    cfg, operation, RESUBMIT_STATE_CREATING_NEW
                )

            if operation.state == RESUBMIT_STATE_CREATING_NEW:
                current_task = None
                try:
                    current_task = load_task(cfg, task_id)
                except FileNotFoundError:
                    pass

                if current_task is None:
                    new_task = _materialize_resubmit_task(operation)
                    _persist_submitted_task_truth(cfg, new_task)
                elif not _resubmit_snapshot_matches_current_task(operation, current_task):
                    raise RuntimeError(
                        f"Resubmit operation for task {task_id} cannot be auto-repaired: "
                        "visible task truth does not match the prepared replacement snapshot."
                    )

                operation = _advance_resubmit_operation_state(
                    cfg, operation, RESUBMIT_STATE_COMMITTED
                )

            if operation.state == RESUBMIT_STATE_COMMITTED:
                delete_resubmit_operation(cfg, task_id)
                write_event(
                    cfg,
                    "resubmit_repaired",
                    task_id=task_id,
                    details={"state": RESUBMIT_STATE_COMMITTED},
                )
                return "committed"

            if operation.state == RESUBMIT_STATE_ABORTED:
                delete_resubmit_operation(cfg, task_id)
                return "aborted"

    return operation.state


# ---------------------------------------------------------------------------
# rebuild-index
# ---------------------------------------------------------------------------


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
    with clean_lock(cfg):
        stats = rebuild_all_indexes(cfg)
        governance = refresh_root_manifest_governance(cfg)
        return {
            "index_stats": stats,
            "governance": governance.to_dict(),
        }


# ---------------------------------------------------------------------------
# repair-orphans
# ---------------------------------------------------------------------------


def repair_orphans(
    cfg: RootConfig,
    heartbeat_stale_seconds: float = 120.0,
) -> list[str]:
    from .agent import get_agent_status
    from .events import write_event

    now = datetime.now(timezone.utc)

    def _machine_cfg(machine_name: str) -> RootConfig:
        return RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=machine_name,
            runtime_root=cfg.runtime_root,
        )

    def _evaluate_agent(machine_name: str) -> tuple[bool, str]:
        snapshot = read_agent_snapshot(_machine_cfg(machine_name))
        if snapshot is None:
            return False, "agent_snapshot_missing"

        if snapshot.agent_state in {
            AGENT_STATE_STOPPED,
            AGENT_STATE_FAILED,
            AGENT_STATE_STALE,
        }:
            return False, f"agent_{snapshot.agent_state}"

        status = get_agent_status(_machine_cfg(machine_name), probe_local_pid=False)
        if status["is_running"]:
            return True, "agent_running"

        hb = snapshot.last_heartbeat
        if hb is None:
            return False, "agent_missing_heartbeat"
        try:
            dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
            elapsed = (now - dt).total_seconds()
            if elapsed > heartbeat_stale_seconds:
                return False, "agent_heartbeat_stale"
        except Exception:
            return False, "agent_heartbeat_invalid"
        return True, "agent_recent_heartbeat"

    orphaned: list[str] = []
    for task in iter_all_tasks(cfg):
        if task.status.phase not in ACTIVE_PHASES:
            continue

        agent_healthy, agent_reason = _evaluate_agent(task.machine_name)
        claim_reason = "claim_present"
        try:
            claim = load_claim(cfg, task.task_id, machine_name=task.machine_name)
        except FileNotFoundError:
            claim = None
            claim_reason = "active_claim_missing"
        else:
            if claim.get("machine_name") != task.machine_name:
                claim_reason = "active_claim_machine_mismatch"
                claim = None

        if agent_healthy and claim is not None:
            continue

        old_phase = task.status.phase
        try:
            validate_phase_transition(old_phase, PHASE_ORPHANED)
        except ValueError:
            continue

        task.status.phase = PHASE_ORPHANED
        task.status.reason = (
            claim_reason if claim_reason != "claim_present" else agent_reason
        )
        try:
            cas_update_task(cfg, task, task.meta.revision)
            update_index_on_phase_change(cfg, task.task_id, old_phase, PHASE_ORPHANED)
            if claim is not None:
                release_claim(
                    cfg,
                    task.task_id,
                    reason="task_orphaned",
                    machine_name=task.machine_name,
                )
            write_event(
                cfg,
                "task_orphaned",
                task_id=task.task_id,
                details={
                    "previous_phase": old_phase,
                    "agent_reason": agent_reason,
                    "claim_reason": claim_reason,
                    "machine_name": task.machine_name,
                },
            )
            orphaned.append(task.task_id)
            log.info("Marked task %s as orphaned (was %s).", task.task_id, old_phase)
        except CASConflict:
            log.debug("CAS conflict repairing %s, skipping.", task.task_id)

    return orphaned


# ---------------------------------------------------------------------------
# cleanup-locks
# ---------------------------------------------------------------------------


def cleanup_stale_locks(
    cfg: RootConfig,
    max_age_seconds: float = 300.0,
) -> list[str]:
    locks_dir = global_locks_dir(cfg)
    if not locks_dir.is_dir():
        return []

    cleaned: list[str] = []
    for lock_file in locks_dir.iterdir():
        if not lock_file.is_file():
            continue
        try:
            mtime = lock_file.stat().st_mtime
            if (time.time() - mtime) > max_age_seconds:
                lock_file.unlink()
                cleaned.append(str(lock_file))
                log.info("Removed stale lock: %s", lock_file)
        except OSError:
            continue

    return cleaned


def repair_metadata(cfg: RootConfig) -> dict[str, Any]:
    """Repair unfinished metadata operations into a converged truth state.

    This is the recovery entrypoint for interrupted metadata mutations:
    - continue unfinished resubmit operations
    - prune missing task_ids from batch truth
    - recompute batch summaries from surviving tasks
    - leave any remaining state-index cleanup to rebuild-index when needed
    """
    repaired_batches: list[str] = []
    committed_batches: list[str] = []
    aborted_batches: list[str] = []
    pruned_task_refs = 0
    repaired_resubmits: list[str] = []

    with clean_lock(cfg):
        for operation in iter_resubmit_operations(cfg):
            if operation.state in {RESUBMIT_STATE_COMMITTED, RESUBMIT_STATE_ABORTED}:
                continue
            result = _repair_resubmit_operation(cfg, operation)
            if result == "committed":
                repaired_resubmits.append(operation.task_id)

        batches_dir = global_batches_dir(cfg)
        if batches_dir.is_dir():
            for path in sorted(batches_dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                batch_id = path.stem
                batch = load_batch(cfg, batch_id)

                removed = 0
                missing_task_ids: list[str] = []

                for task_id in batch.task_ids:
                    try:
                        load_task(cfg, task_id)
                    except FileNotFoundError:
                        removed += 1
                        missing_task_ids.append(task_id)
                surviving_ids, counts = collect_batch_task_counts(
                    cfg,
                    batch.task_ids,
                    ignore_missing=True,
                )
                declared_count = resolve_declared_batch_total(
                    commit_state=batch.commit_state,
                    expected_task_count=batch.expected_task_count,
                    declared_task_ids=batch.task_ids,
                    surviving_ids=surviving_ids,
                )
                new_summary = build_batch_summary_from_counts(
                    total=declared_count,
                    counts=counts,
                )
                next_commit_state = batch.commit_state
                if batch.commit_state == BATCH_COMMIT_PREPARING:
                    if len(surviving_ids) == declared_count:
                        next_commit_state = BATCH_COMMIT_COMMITTED
                        committed_batches.append(batch.batch_id)
                    else:
                        next_commit_state = BATCH_COMMIT_ABORTED
                        aborted_batches.append(batch.batch_id)

                if (
                    surviving_ids != batch.task_ids
                    or new_summary != batch.summary
                    or next_commit_state != batch.commit_state
                    or batch.expected_task_count != declared_count
                ):
                    batch.task_ids = surviving_ids
                    batch.summary = new_summary
                    batch.commit_state = next_commit_state
                    batch.expected_task_count = declared_count
                    batch.meta.revision += 1
                    batch.meta.updated_at = utc_now_iso()
                    batch.meta.updated_by_machine = cfg.machine_name
                    save_batch(cfg, batch)
                    repaired_batches.append(batch.batch_id)
                    pruned_task_refs += removed

                for missing_task_id in missing_task_ids:
                    sync_task_state_index(cfg, missing_task_id, None)

        index_drift = collect_index_drift_report(cfg)
        governance = refresh_root_manifest_governance(cfg)

    return {
        "repaired_resubmit_count": len(repaired_resubmits),
        "repaired_resubmits": repaired_resubmits,
        "repaired_batch_count": len(repaired_batches),
        "repaired_batches": repaired_batches,
        "committed_batches": committed_batches,
        "aborted_batches": aborted_batches,
        "pruned_task_ref_count": pruned_task_refs,
        "index_drift": index_drift,
        "governance": governance.to_dict(),
    }


# ---------------------------------------------------------------------------
# verify-integrity
# ---------------------------------------------------------------------------


def verify_integrity(cfg: RootConfig) -> dict[str, Any]:
    """Non-destructive integrity check.

    Returns dict with 'ok' bool and 'issues' list.
    """
    issues: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    tasks_dir = global_tasks_dir(cfg)
    batches_dir = global_batches_dir(cfg)

    try:
        root_manifest = validate_root_contract(cfg)
        manifest = root_manifest.to_dict()["root_manifest"]
        if root_manifest.root_scope != ROOT_SCOPE_PROJECT:
            _append_verify_issue(
                issues,
                code="invalid_root_scope",
                category="root_contract",
                severity="high",
                message=(
                    f"Invalid root scope {root_manifest.root_scope!r}; "
                    f"expected {ROOT_SCOPE_PROJECT!r}."
                ),
                actual_root_scope=root_manifest.root_scope,
                expected_root_scope=ROOT_SCOPE_PROJECT,
            )
    except Exception as exc:
        _append_verify_issue(
            issues,
            code="invalid_root_manifest",
            category="root_contract",
            severity="high",
            message=f"Invalid root manifest: {exc}",
            error=str(exc),
        )

    forbidden_truth_dirs = [
        str(path) for path in list_forbidden_truth_layout_dirs(cfg)
    ]
    for path in forbidden_truth_dirs:
        _append_verify_issue(
            issues,
            code="forbidden_truth_layout_directory",
            category="truth_layout",
            severity="high",
            message=(
                f"Forbidden truth-layout directory detected: {path}. "
                "Only global object truth and machine-private subtrees are allowed."
            ),
            path=path,
        )

    checked = 0
    if tasks_dir.is_dir():
        for path in tasks_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            checked += 1
            expected_id = path.stem
            try:
                data = read_json(path)
                task_id = data.get("task", {}).get("task_id")
                if task_id != expected_id:
                    _append_verify_issue(
                        issues,
                        code="task_filename_id_mismatch",
                        category="task_truth",
                        severity="high",
                        message=f"Filename/ID mismatch: file={path.name}, task_id={task_id}",
                        file=path.name,
                        expected_task_id=expected_id,
                        actual_task_id=task_id,
                    )
                revision = data.get("meta", {}).get("revision")
                if not isinstance(revision, int) or revision < 1:
                    _append_verify_issue(
                        issues,
                        code="task_invalid_revision",
                        category="task_truth",
                        severity="low",
                        message=f"Invalid revision for {expected_id}: {revision}",
                        task_id=expected_id,
                        revision=revision,
                    )
            except Exception as e:
                _append_verify_issue(
                    issues,
                    code="task_truth_unreadable",
                    category="task_truth",
                    severity="high",
                    message=f"Cannot read {path.name}: {e}",
                    file=path.name,
                    error=str(e),
                )

    batches_checked = 0
    if batches_dir.is_dir():
        for path in batches_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            batches_checked += 1
            expected_id = path.stem
            try:
                data = read_json(path)
                batch_data = data.get("batch", {})
                batch_id = batch_data.get("batch_id")
                if batch_id != expected_id:
                    _append_verify_issue(
                        issues,
                        code="batch_filename_id_mismatch",
                        category="batch_truth",
                        severity="high",
                        message=(
                            f"Batch filename/ID mismatch: file={path.name}, batch_id={batch_id}"
                        ),
                        file=path.name,
                        expected_batch_id=expected_id,
                        actual_batch_id=batch_id,
                    )
                    continue

                declared_task_ids = list(batch_data.get("task_ids", []))
                stored_expected_task_count = batch_data.get(
                    "expected_task_count",
                    len(declared_task_ids),
                )
                commit_state = batch_data.get("commit_state")
                if commit_state not in {
                    BATCH_COMMIT_PREPARING,
                    BATCH_COMMIT_COMMITTED,
                    BATCH_COMMIT_ABORTED,
                }:
                    _append_verify_issue(
                        issues,
                        code="batch_invalid_commit_state",
                        category="batch_truth",
                        severity="low",
                        message=f"Batch {expected_id} has invalid commit_state {commit_state!r}",
                        batch_id=expected_id,
                        commit_state=commit_state,
                    )
                if (
                    not isinstance(stored_expected_task_count, int)
                    or stored_expected_task_count < 0
                ):
                    _append_verify_issue(
                        issues,
                        code="batch_invalid_expected_task_count",
                        category="batch_truth",
                        severity="low",
                        message=(
                            f"Batch {expected_id} has invalid expected_task_count "
                            f"{stored_expected_task_count!r}"
                        ),
                        batch_id=expected_id,
                        expected_task_count=stored_expected_task_count,
                    )
                    stored_expected_task_count = len(declared_task_ids)
                if len(declared_task_ids) > stored_expected_task_count:
                    _append_verify_issue(
                        issues,
                        code="batch_declared_tasks_exceed_expected_count",
                        category="batch_truth",
                        severity="low",
                        message=(
                            f"Batch {expected_id} has more persisted tasks than expected_task_count."
                        ),
                        batch_id=expected_id,
                        declared_task_count=len(declared_task_ids),
                        expected_task_count=stored_expected_task_count,
                    )
                counts: dict[str, int] = {}
                surviving_ids: list[str] = []
                for task_id in declared_task_ids:
                    try:
                        task = load_task(cfg, task_id)
                    except FileNotFoundError:
                        _append_verify_issue(
                            issues,
                            code="batch_missing_task_reference",
                            category="batch_truth",
                            severity="medium",
                            message=f"Batch {expected_id} references missing task {task_id}",
                            batch_id=expected_id,
                            task_id=task_id,
                        )
                        continue
                    surviving_ids.append(task_id)
                    phase = task.status.phase
                    counts[phase] = counts.get(phase, 0) + 1
                    if task.batch_id != expected_id:
                        _append_verify_issue(
                            issues,
                            code="batch_task_batch_id_mismatch",
                            category="batch_truth",
                            severity="low",
                            message=(
                                f"Batch {expected_id} contains task {task_id} "
                                f"with mismatched batch_id {task.batch_id!r}"
                            ),
                            batch_id=expected_id,
                            task_id=task_id,
                            actual_task_batch_id=task.batch_id,
                        )

                expected_task_count = resolve_declared_batch_total(
                    commit_state=commit_state,
                    expected_task_count=stored_expected_task_count,
                    declared_task_ids=declared_task_ids,
                    surviving_ids=surviving_ids,
                )

                expected_summary = build_batch_summary_from_counts(
                    total=expected_task_count,
                    counts=counts,
                ).to_dict()
                actual_summary = batch_data.get("summary", {})
                if actual_summary != expected_summary:
                    _append_verify_issue(
                        issues,
                        code="batch_summary_drift",
                        category="batch_truth",
                        severity="medium",
                        message=(
                            f"Batch {expected_id} summary drift: "
                            f"expected={expected_summary}, actual={actual_summary}"
                        ),
                        batch_id=expected_id,
                        expected_summary=expected_summary,
                        actual_summary=actual_summary,
                    )
                if (
                    commit_state == BATCH_COMMIT_COMMITTED
                    and len(surviving_ids) != expected_task_count
                ):
                    _append_verify_issue(
                        issues,
                        code="batch_committed_missing_tasks",
                        category="batch_truth",
                        severity="medium",
                        message=(
                            f"Batch {expected_id} is committed but only has "
                            f"{len(surviving_ids)}/{expected_task_count} tasks."
                        ),
                        batch_id=expected_id,
                        actual_task_count=len(surviving_ids),
                        expected_task_count=expected_task_count,
                    )
                if (
                    commit_state == BATCH_COMMIT_PREPARING
                    and len(surviving_ids) == expected_task_count
                ):
                    _append_verify_issue(
                        issues,
                        code="batch_preparing_complete_task_set",
                        category="batch_truth",
                        severity="medium",
                        message=(
                            f"Batch {expected_id} is still preparing despite complete task set."
                        ),
                        batch_id=expected_id,
                        task_count=len(surviving_ids),
                    )
            except Exception as e:
                _append_verify_issue(
                    issues,
                    code="batch_truth_unreadable",
                    category="batch_truth",
                    severity="high",
                    message=f"Cannot read batch {path.name}: {e}",
                    file=path.name,
                    error=str(e),
                )

    resubmit_ops_checked = 0
    for operation in iter_resubmit_operations(cfg):
        resubmit_ops_checked += 1
        if operation.state in {RESUBMIT_STATE_COMMITTED, RESUBMIT_STATE_ABORTED}:
            _append_verify_issue(
                issues,
                code="resubmit_terminal_operation_left_behind",
                category="resubmit_operation",
                severity="high",
                message=(
                    f"Resubmit operation {operation.task_id} is left behind "
                    f"in terminal state {operation.state!r}."
                ),
                task_id=operation.task_id,
                state=operation.state,
            )
        if not operation.new_submission.command:
            _append_verify_issue(
                issues,
                code="resubmit_missing_new_submission_command",
                category="resubmit_operation",
                severity="high",
                message=(
                    f"Resubmit operation {operation.task_id} is missing new submission command."
                ),
                task_id=operation.task_id,
            )
        try:
            expected_task = Task.from_dict(operation.new_task_snapshot)
        except Exception as exc:
            _append_verify_issue(
                issues,
                code="resubmit_invalid_prepared_snapshot",
                category="resubmit_operation",
                severity="high",
                message=(
                    f"Resubmit operation {operation.task_id} "
                    f"has invalid prepared task snapshot: {exc}"
                ),
                task_id=operation.task_id,
                error=str(exc),
            )
            expected_task = None
        else:
            if expected_task.task_id != operation.task_id:
                _append_verify_issue(
                    issues,
                    code="resubmit_prepared_snapshot_task_id_mismatch",
                    category="resubmit_operation",
                    severity="high",
                    message=(
                        f"Resubmit operation {operation.task_id} prepared snapshot "
                        f"task_id mismatches: {expected_task.task_id!r}."
                    ),
                    task_id=operation.task_id,
                    snapshot_task_id=expected_task.task_id,
                )
        if operation.old_task_summary.batch_id is not None:
            _append_verify_issue(
                issues,
                code="resubmit_illegal_batch_target",
                category="resubmit_operation",
                severity="high",
                message=(
                    f"Resubmit operation {operation.task_id} illegally targets "
                    f"batch task {operation.old_task_summary.batch_id}."
                ),
                task_id=operation.task_id,
                batch_id=operation.old_task_summary.batch_id,
            )
        try:
            current_task = load_task(cfg, operation.task_id)
        except FileNotFoundError:
            current_task = None
        if (
            operation.state == RESUBMIT_STATE_CREATING_NEW
            and expected_task is not None
            and current_task is not None
            and not _resubmit_snapshot_matches_current_task(operation, current_task)
        ):
            _append_verify_issue(
                issues,
                code="resubmit_visible_truth_snapshot_mismatch",
                category="resubmit_operation",
                severity="high",
                message=(
                    f"Resubmit operation {operation.task_id} is creating_new but "
                    "visible task truth does not match the prepared replacement snapshot."
                ),
                task_id=operation.task_id,
                state=operation.state,
            )

    governance = collect_governance_snapshot(cfg).to_dict()
    if governance["terminal_tasks"] > governance["total_tasks"] * 0.8 and governance["total_tasks"] > 20:
        _append_verify_issue(
            issues,
            code="governance_terminal_task_ratio_high",
            category="governance",
            severity="low",
            message=(
                "Terminal task ratio exceeds 80%; lifecycle cleanup or archive governance is overdue."
            ),
            terminal_tasks=governance["terminal_tasks"],
            total_tasks=governance["total_tasks"],
        )

    index_drift = collect_index_drift_report(cfg)
    if not index_drift["ok"]:
        for family_name, family in index_drift["families"].items():
            if family["ok"]:
                continue
            _append_verify_issue(
                issues,
                code="derived_index_state_drift",
                category="derived_index",
                severity="high",
                message=(
                    f"Derived index drift detected in {family_name}: "
                    f"missing={family['missing_count']} unexpected={family['unexpected_count']}"
                ),
                family=family_name,
                missing_count=family["missing_count"],
                unexpected_count=family["unexpected_count"],
            )
    severity = _verify_severity_from_issues(issues)
    issue_codes = _issue_codes(issues)
    issue_categories = _issue_categories(issues)
    recommended_actions = _build_verify_recommendations(issues, index_drift)

    return {
        "ok": len(issues) == 0,
        "severity": severity,
        "issues": issues,
        "messages": [issue["message"] for issue in issues],
        "issue_count_by_category": _issue_count_by_category(issues),
        "issue_count_by_code": _issue_count_by_code(issues),
        "diagnosis": {
            "truth_ok": not (issue_categories & {"root_contract", "truth_layout", "task_truth"}),
            "index_drift_only": len(issues) > 0 and issue_categories == {"derived_index"},
            "has_resubmit_gap": "resubmit_operation" in issue_categories,
            "has_batch_truth_drift": bool(
                issue_codes
                & {
                    "batch_missing_task_reference",
                    "batch_summary_drift",
                    "batch_committed_missing_tasks",
                    "batch_preparing_complete_task_set",
                }
            ),
        },
        "recommended_actions": recommended_actions,
        "root_manifest": manifest,
        "forbidden_truth_dirs": forbidden_truth_dirs,
        "governance": governance,
        "index_drift": index_drift,
        "tasks_checked": checked,
        "batches_checked": batches_checked,
        "resubmit_ops_checked": resubmit_ops_checked,
    }
