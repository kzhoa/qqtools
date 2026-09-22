from __future__ import annotations

import json

from qqtools.plugins.qexp.cli.local_handlers import _upgrade_payload
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.doctor import resolve_verify_exit_code


def test_doctor_healthy_and_blocked_repair_have_distinct_outcomes() -> None:
    healthy = {
        "schema_version": 6,
        "tasks_checked": 3,
        "issues": [],
        "issue_counts": {},
        "incomplete_reasons": [],
        "complete": True,
        "healthy": True,
        "outcome": "healthy",
        "repair_command": None,
    }
    blocked = {
        "repaired": ["operation-1"],
        "blocked": ["operation-2"],
        "repaired_count": 1,
        "blocked_count": 1,
        "outcome": "blocked",
        "rerun_required": True,
        "next_action": "qexp admin repair --project /project",
        "message": "Repair remains blocked.",
    }

    healthy_human = render(CliOutput(OutputKind.DOCTOR_VERIFY, healthy), "human")
    blocked_human = render(CliOutput(OutputKind.DOCTOR_REPAIR, blocked), "human")

    assert "Outcome: healthy" in healthy_human
    assert "Repair command" not in healthy_human
    assert "Outcome: blocked" in blocked_human
    assert "Blocked: 1" in blocked_human
    assert "qexp admin repair --project /project" in blocked_human
    assert resolve_verify_exit_code(blocked) == 1
    assert json.loads(render(CliOutput(OutputKind.DOCTOR_REPAIR, blocked), "json")) == blocked


def test_clean_dry_run_and_pending_operation_are_not_called_removed() -> None:
    preview = {
        "dry_run": True,
        "deletion": "none",
        "deletion_performed": False,
        "candidates": ["task-1"],
        "removed": [],
        "skipped": {},
        "candidate_count": 1,
        "removed_count": 0,
        "pending_count": 0,
        "skipped_count": 0,
        "blocked_count": 0,
        "removed_task_ids": [],
        "pending_task_ids": [],
        "skipped_task_ids": [],
        "blocked_task_ids": [],
        "outcome": "preview",
    }
    pending = {
        **preview,
        "dry_run": False,
        "deletion": "requested",
        "outcome": "waiting",
        "pending_count": 1,
        "pending_task_ids": ["task-1"],
        "operations": {
            "task-1": {
                "state": "waiting_ack",
                "operation_reference": "opaque-ref",
                "follow_up_command": "qexp admin operation show opaque-ref --project /project",
            }
        },
    }

    preview_human = render(CliOutput(OutputKind.CLEAN, preview), "human")
    pending_human = render(CliOutput(OutputKind.CLEAN, pending), "human")

    assert "Deletion: none (dry run)" in preview_human
    assert "Deletion performed: no" in preview_human
    assert "Outcome: waiting" in pending_human
    assert "Pending: 1" in pending_human
    assert "Removed: 0" in pending_human
    assert "opaque-ref" in pending_human
    assert "qexp admin operation show opaque-ref --project /project" in pending_human


def test_operation_renderer_uses_lifecycle_outcome_and_timestamps() -> None:
    result = {
        "reference": "opaque-ref",
        "kind": "cleanup",
        "operation_id": "operation-1",
        "target": {"task_id": "task-1"},
        "state": "blocked",
        "outcome": "ok",
        "lifecycle_outcome": "blocked",
        "progress": None,
        "blockers": ["waiting_for_machine"],
        "pending_machines": ["gpu-2"],
        "created_at": "2026-09-22T10:00:00Z",
        "updated_at": "2026-09-22T10:01:00Z",
        "completed_at": None,
        "next_action": "qexp admin operation show opaque-ref --project /project",
        "error": None,
    }

    human = render(CliOutput(OutputKind.OPERATION, result), "human")

    assert "Outcome: blocked" in human
    assert "Created: 2026-09-22T10:00:00Z" in human
    assert "Updated: 2026-09-22T10:01:00Z" in human
    assert "Completed:" not in human
    assert "Next action: qexp admin operation show opaque-ref --project /project" in human


def test_agent_migration_partial_success_keeps_durable_state_and_recovery() -> None:
    result = {
        "action": "project_migrated",
        "outcome": "partial",
        "project_id": "project-1",
        "shared_root": "/project/.qexp",
        "machine_name": "gpu-1",
        "enabled": True,
        "agent_state": "unavailable",
        "pid": None,
        "is_running": False,
        "migration_state": "active",
        "migration": {"state": "active", "updated_at": "2026-09-22T10:00:00Z"},
        "reason": "project migration committed but machine agent start failed",
        "error": {"code": "agent_start_failed", "message": "unable to start"},
        "follow_up_command": "qexp agent start",
        "next_action": "qexp agent start",
        "migration_candidates": [],
    }

    human = render(CliOutput(OutputKind.AGENT_OPERATION, result), "human")

    assert "Status: partial" in human
    assert "Migration state: active" in human
    assert "project migration committed" in human
    assert "unable to start" in human
    assert "qexp agent start" in human
    assert json.loads(render(CliOutput(OutputKind.AGENT_OPERATION, result), "json")) == result


def test_upgrade_advance_inaccessible_only_never_reports_no_results() -> None:
    result = {
        "action": "advance",
        "outcome": "blocked",
        "reason": "registered root is unavailable",
        "projects": [],
        "inaccessible_projects": [{"project_id": "project-1", "state": "inaccessible", "error": "storage offline"}],
        "aggregate_state": "inaccessible",
        "all_roots_complete": False,
        "pending_project_ids": ["project-1"],
        "slices": 0,
        "worker_state": "idle",
        "discovery_source": "machine_registry",
        "discovery_boundary": "locally_registered_bindings",
        "next_action": "qexp admin upgrade status",
    }

    human = render(CliOutput(OutputKind.UPGRADE_ADVANCE, result), "human")

    assert "Upgrade state: inaccessible" in human
    assert "project-1" in human
    assert "storage offline" in human
    assert "No results" not in human
    assert "qexp admin upgrade status" in human


def test_schema_migration_renders_truthful_committed_and_already_current_phases() -> None:
    committed = {
        "action": "migrate_schema",
        "outcome": "completed",
        "phase": "committed",
        "project": "/project",
        "source_schema": 5,
        "target_schema": 6,
        "blockers": [],
        "destructive_boundary": {"name": "schema_root_replacement", "reached": True, "recoverable": False},
        "next_action": None,
    }
    current = {**committed, "outcome": "no_change", "phase": "already_current", "source_schema": 6}

    committed_human = render(CliOutput(OutputKind.SCHEMA6_UPGRADE, committed), "human")
    current_human = render(CliOutput(OutputKind.SCHEMA6_UPGRADE, current), "human")

    assert "Phase: committed" in committed_human
    assert "Source schema: 5" in committed_human
    assert "Target schema: 6" in committed_human
    assert "Phase: already_current" in current_human
    assert "Outcome: no_change" in current_human


def test_failed_upgrade_validation_is_not_canonicalized_as_completed() -> None:
    result = _upgrade_payload(
        {
            "repair_id": "repair-1",
            "target": "upgrade-journal-v1",
            "state": "validation_failed",
            "error": "repair validation did not complete",
        },
        "validate",
        project_root="/project with spaces",
    )

    assert result["outcome"] == "failed"
    assert result["reason"] == "repair validation did not complete"
    assert result["next_action"] == (
        "qexp admin upgrade plan --project '/project with spaces' --target upgrade-journal-v1"
    )
    human = render(CliOutput(OutputKind.UPGRADE_REPAIR, result), "human")
    assert "Status: failed" in human
    assert "repair validation did not complete" in human
