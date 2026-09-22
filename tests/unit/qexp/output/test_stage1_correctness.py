from __future__ import annotations

import json

from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render


def test_committed_submission_reports_activation_failure_and_recovery() -> None:
    payload = {
        "schema_version": 1,
        "mode": "command",
        "outcome": "committed",
        "project": {"path": "/work/project", "source": "cli"},
        "group": {"name": None, "source": "none", "disposition": "none"},
        "operation": {"id": "op-1", "state": "committed"},
        "idempotency_key": "key-1",
        "task_ids": ["task-1"],
        "preview": None,
        "error": {"code": "activation_failed", "message": "agent unavailable"},
        "activation": {
            "outcome": "failed",
            "follow_up_command": "qexp agent start",
        },
    }

    output = CliOutput(OutputKind.SUBMISSION, payload)
    human = render(output, "human")

    assert json.loads(render(output, "json")) == payload
    assert "Committed" in human
    assert "activation failed" in human.lower()
    assert "agent unavailable" in human
    assert "qexp agent start" in human
    assert "task-1" in human


def test_task_cancel_renders_owner_acknowledgement_and_follow_up() -> None:
    payload = {
        "action": "cancel",
        "outcome": "waiting",
        "task_id": "task-1",
        "task_state": "running",
        "owning_machine": "gpu one",
        "operation_state": "waiting_ack",
        "pending_acknowledgement": True,
        "termination_acknowledged_at": None,
        "follow_up_command": "qexp --project /work task show task-1",
    }

    human = render(CliOutput(OutputKind.TASK_CANCEL, payload), "human")

    assert "gpu one" in human
    assert "waiting" in human.lower()
    assert "qexp --project /work task show task-1" in human


def test_group_retry_reports_counts_skips_and_zero_eligible() -> None:
    retried = {
        "action": "retry",
        "outcome": "completed",
        "group": "training",
        "retried_task_ids": ["task-1"],
        "retried_count": 1,
        "skipped": {"blocked": ["task-2"], "orphaned": [], "other": []},
        "follow_up_command": "qexp group show training --project /work",
    }
    empty = {
        **retried,
        "outcome": "no_change",
        "retried_task_ids": [],
        "retried_count": 0,
        "skipped": {"blocked": [], "orphaned": [], "other": []},
    }

    rendered = render(CliOutput(OutputKind.GROUP_RETRY, retried), "human")
    empty_rendered = render(CliOutput(OutputKind.GROUP_RETRY, empty), "human")

    assert "Retried: 1" in rendered
    assert "Blocked: 1" in rendered
    assert "task-1" in rendered
    assert "No eligible failed Tasks" in empty_rendered


def test_group_async_operation_renders_reference_pending_and_follow_up() -> None:
    payload = {
        "action": "cancel",
        "outcome": "waiting",
        "group": "training",
        "pending_machines": ["gpu-1"],
        "reason": None,
        "operation_reference": "group_cancel:op-1",
        "follow_up_command": "qexp admin operation show group_cancel:op-1 --project /work",
    }

    human = render(CliOutput(OutputKind.GROUP_CANCEL, payload), "human")

    assert "gpu-1" in human
    assert "group_cancel:op-1" in human
    assert payload["follow_up_command"] in human


def test_availability_renders_queue_state_once_and_no_change_outcome() -> None:
    payload = {
        "action": "share_now",
        "outcome": "no_change",
        "task_id": "task-1",
        "group": None,
        "home_machine": "gpu-1",
        "eligible_helper_machines": ["gpu-2"],
        "effective_at": None,
        "resulting_state": "shared",
        "idempotent": True,
        "operation_id": "op-1",
        "message": "already shared",
    }

    human = render(CliOutput(OutputKind.AVAILABILITY, payload), "human")

    assert human.count("Queue scope: shared") == 1
    assert "Status: shared" not in human
    assert "no change" in human.lower()


def test_upgrade_registry_distinguishes_inaccessible_only_from_empty() -> None:
    inaccessible = {
        "projects": [],
        "inaccessible_projects": [{"project_id": "p1", "reason": "permission denied"}],
        "aggregate_state": "inaccessible",
        "pending_project_ids": ["p1"],
        "all_roots_complete": False,
        "discovery_source": "machine_registry",
        "discovery_boundary": "locally_registered_bindings",
    }

    human = render(CliOutput(OutputKind.UPGRADE_REGISTRY_STATUS, inaccessible), "human")

    assert "inaccessible" in human.lower()
    assert "p1" in human
    assert "permission denied" in human
    assert "No results." not in human
