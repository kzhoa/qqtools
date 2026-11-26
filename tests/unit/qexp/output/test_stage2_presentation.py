from __future__ import annotations

import json

from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render


def _human(kind: OutputKind, payload: object) -> str:
    return render(CliOutput(kind, payload), "human")


def test_task_list_prioritizes_daily_fields_and_stays_terminal_width() -> None:
    output = _human(
        OutputKind.TASK_LIST,
        [
            {
                "task_id": "task-1",
                "name": "train",
                "phase": "queued",
                "reason": None,
                "gpus": 1,
                "group": "experiment",
                "home_machine": "gpu-1",
                "queue_scope": "home",
                "current_attempt_id": None,
                "claim_machine": None,
                "depends_on_task_ids": [],
                "dependency_state": "ready",
            }
        ],
    )

    header = output.splitlines()[0]
    assert all(label in header for label in ("Task ID", "Name", "State", "GPUs", "Group", "Home"))
    assert "Attempt" not in header
    assert "Dependencies" not in header
    assert max(map(len, output.splitlines())) <= 120


def test_empty_inventory_messages_are_resource_specific() -> None:
    assert "No Tasks" in _human(OutputKind.TASK_LIST, [])
    assert "No Groups" in _human(
        OutputKind.GROUP_LIST,
        {"items": [], "next_cursor": None, "consistency": "live", "stop_reason": "exhausted"},
    )
    projects = _human(
        OutputKind.PROJECT_LIST,
        {"revision": 0, "inventory_revision": 0, "registry_revision": 0, "projects": []},
    )
    assert "No enrolled Projects" in projects
    assert "qexp project register PATH" in projects


def test_group_page_renders_stop_reason_and_continuation() -> None:
    payload = {
        "items": [
            {
                "group": {
                    "name": "experiment",
                    "dispatch_state": "active",
                    "worker_set": {"gpu-1": {"state": "active"}},
                }
            }
        ],
        "next_cursor": "opaque-token",
        "consistency": "live",
        "stop_reason": "page_full",
    }
    output = render(
        CliOutput(OutputKind.GROUP_LIST, payload, {"continuation_command": "qexp group list --cursor opaque-token"}),
        "human",
    )

    assert "experiment" in output
    assert "Stop reason: page_full" in output
    assert "Continue with: qexp group list --cursor opaque-token" in output
    assert json.loads(render(CliOutput(OutputKind.GROUP_LIST, payload), "json")) == payload


def test_wait_timeout_omits_irrelevant_terminal_rows_and_states_no_mutation() -> None:
    output = _human(
        OutputKind.TASK_WAIT,
        {
            "schema_version": 1,
            "task_id": "task-1",
            "project": "/work/project",
            "selected_attempt_number": None,
            "selected_attempt_id": None,
            "outcome": "timeout",
            "reason": "deadline_reached",
            "task_exit_code": None,
            "error": None,
        },
    )

    assert "timeout" in output.lower()
    assert "no mutation" in output.lower() or "no changes" in output.lower()
    assert "Selected Attempt" not in output
    assert "Task exit code" not in output
    assert "Error" not in output


def test_group_operation_uses_only_canonical_action_and_follow_up() -> None:
    output = _human(
        OutputKind.GROUP_CANCEL,
        {
            "action": "cancel",
            "outcome": "waiting_ack",
            "group": "experiment",
            "status": "cancelling",
            "pending_machines": ["gpu-2"],
            "operation_reference": "group-cancel:experiment",
            "follow_up_command": "qexp group show experiment",
        },
    )

    assert "cancel" in output.lower()
    assert "waiting ack" in output.lower()
    assert "gpu-2" in output
    assert "group-cancel:experiment" in output
    assert "qexp group show experiment" in output


def test_context_mutations_confirm_selected_and_cleared_state() -> None:
    selected = _human(
        OutputKind.CONTEXT,
        {"action": "selected", "shared_root": "/work/project/.qexp", "changed": True},
    )
    cleared = _human(OutputKind.CONTEXT, {"action": "cleared", "shared_root": None, "changed": False})

    assert "selected" in selected.lower()
    assert "/work/project/.qexp" in selected
    assert "already clear" in cleared.lower()


def test_machine_list_and_show_are_bounded_sections_not_mapping_dumps() -> None:
    machine = {
        "machine": {"machine_name": "gpu-1"},
        "state": {
            "freshness": "fresh",
            "agent": {"agent": {"observed_state": "active"}},
            "gpu": {
                "gpu": {
                    "visible_gpu_ids": [0, 1],
                    "reserved_gpu_ids": [0],
                    "free_gpu_ids": [1],
                    "gpu_policy": {"source": "persisted", "mode": "explicit", "draining_gpu_ids": []},
                }
            },
        },
        "task_summary": {"running": 1},
        "reason": None,
    }
    listing = _human(OutputKind.MACHINES, [machine])
    assert max(map(len, listing.splitlines())) <= 120
    assert "GPU source" not in listing.splitlines()[0]

    detail = _human(
        OutputKind.MACHINE_SHOW,
        {
            "machine_name": "gpu-1",
            "project": "/work/project",
            "complete": False,
            "declaration": {"agent_mode": "daemon"},
            "agent": {"observed_state": "active"},
            "gpu": {"visible_gpu_ids": [0, 1], "free_gpu_ids": [1]},
            "summary": {"running": 1},
            "warnings": ["summary is stale"],
        },
    )
    assert all(section in detail for section in ("Declaration", "Agent", "GPU", "Summary", "Warnings"))
    assert "{'" not in detail


def test_machine_list_does_not_present_stale_gpu_observations_as_free_capacity() -> None:
    stale = {
        "machine": {"machine_name": "gpu-stale"},
        "state": {
            "freshness": "stale",
            "agent": {"agent": {"observed_state": "idle"}},
            "gpu": {"gpu": {"free_gpu_ids": [7], "reserved_gpu_ids": []}},
        },
    }

    output = _human(OutputKind.MACHINES, [stale])
    row = output.splitlines()[-1]

    assert "gpu-stale" in row
    assert "  7  " not in row
    assert "resource observation is not current" in row


def test_restart_reports_process_replacement_and_pending_readiness() -> None:
    output = _human(
        OutputKind.AGENT_OPERATION,
        {
            "action": "restarted",
            "machine_runtime_root": "/runtime",
            "agent_state": "starting",
            "pid": 22,
            "previous_pid": 11,
            "is_running": True,
            "configured_agent_mode": "daemon",
            "ready": None,
            "projects": [],
        },
    )

    assert output.startswith("Agent restarted.\n")
    assert "Project readiness has not been verified." in output
    assert "qexp --machine-runtime-root /runtime agent status" in output
    assert "0/0" not in output
    assert "continue without another restart" not in output


def test_task_show_selects_the_authoritative_attempt_not_lexicographic_tail() -> None:
    output = _human(
        OutputKind.TASK_SHOW,
        {
            "task": {
                "task_id": "task-1",
                "name": "train",
                "group_name": None,
                "depends_on_task_ids": [],
                "spec": {"command": ["python", "train.py"], "requested_gpus": 1},
                "placement_policy": {"home_machine": "gpu-1"},
                "placement_runtime": {"queue_scope": "home"},
                "state": {"projection": "failed", "reason": "exit_nonzero"},
                "control": {"cancellation_operation_id": None},
                "attempt_control": {"current_attempt_id": "attempt-10"},
            },
            "attempts": [
                {"attempt": {"attempt_id": "attempt-1", "attempt_number": 1, "machine_name": "gpu-1"}},
                {
                    "attempt": {
                        "attempt_id": "attempt-10",
                        "attempt_number": 10,
                        "machine_name": "gpu-2",
                        "phase": "failed",
                        "exit_code": 10,
                    }
                },
                {
                    "attempt": {
                        "attempt_id": "attempt-2",
                        "attempt_number": 2,
                        "machine_name": "gpu-1",
                        "phase": "failed",
                        "exit_code": 2,
                    }
                },
            ],
            "dependency_gate": {},
            "progress": {},
            "observation": {"tmux_override": "inherit"},
        },
    )

    assert "Current Attempt: 10" in output
    assert "Exit code: 10" in output
    assert all(section in output for section in ("Task\n", "Execution\n", "Configuration\n", "Progress\n", "Command\n"))
    assert output.index("State: failed") < output.index("Command\n")
    assert output.endswith("python train.py")


def test_group_show_highlights_abnormal_state_and_existing_operations() -> None:
    output = _human(
        OutputKind.GROUP_SHOW,
        {
            "group": {
                "name": "experiment",
                "dispatch_state": "paused",
                "worker_set": {
                    "gpu-1": {"state": "active"},
                    "gpu-2": {"state": "removing"},
                },
            },
            "cancellation_operation": {"operation_id": "cancel-1", "state": "completed"},
            "worker_control": {
                "operation_id": "remove-1",
                "state": "completed",
                "machine_name": "gpu-2",
            },
        },
    )

    assert "Attention required\n  Dispatch: paused\n  Workers: gpu-2=removing" in output
    assert "Recent operations\n  Cancellation: cancel-1 (completed)" in output
    assert "Worker control: remove-1 (completed; gpu-2)" in output


def test_project_status_groups_state_actions_and_scheduler_diagnostics() -> None:
    payload = {
        "status": "complete",
        "project": {"path": "/work/project", "project_id": "project-1", "selection_source": "explicit"},
        "local_participation": {"state": "registered"},
        "local_agent": {"state": "active", "configured_mode": "on_demand"},
        "task_observation": {"state": "absent", "reason": "projection_missing"},
        "next_actions": ["qexp task list --project /work/project", "qexp agent diagnostics active"],
        "scheduler_diagnostics": {
            "status": "available",
            "coverage": "incomplete",
            "observed_at": "2026-10-05T00:00:00Z",
            "active_count": 0,
            "reason": "probe_data_unavailable",
        },
    }

    output = _human(OutputKind.STATUS, payload)

    assert output.startswith("Project: /work/project")
    assert "Local enrollment: registered" in output
    assert "Task observation index: absent (projection missing)" in output
    assert "Run: qexp task list --project /work/project" in output
    assert "Run: qexp agent diagnostics active" in output
    assert "Scheduling diagnostics\n  State: available\n  Coverage: incomplete" in output
    assert "Reason: probe data unavailable" in output


def test_project_status_clean_active_observation_suppresses_actions() -> None:
    payload = {
        "status": "complete",
        "project": {"path": "/work/project", "project_id": "project-1", "selection_source": "explicit"},
        "local_participation": {"state": "registered"},
        "local_agent": {"state": "active", "configured_mode": "daemon", "observed_mode": "daemon"},
        "task_observation": {"state": "active", "reason": None},
        "next_actions": [],
        "warnings": [],
        "scheduler_diagnostics": {
            "status": "available",
            "coverage": "complete",
            "observed_at": "2026-10-05T00:00:00Z",
            "active_count": 0,
            "active_findings": [],
        },
    }

    output = _human(OutputKind.STATUS, payload)

    assert output.startswith("Project: /work/project")
    assert "Local enrollment: registered" in output
    assert output.splitlines()[1:3] == ["Local enrollment: registered", "Local agent: active"]
    assert len(output.splitlines()) <= 8
    assert max(map(len, output.splitlines())) <= 80
    assert "Next actions" not in output
    assert "running Tasks" not in output


def test_project_status_dirty_active_observation_keeps_task_action() -> None:
    payload = {
        "status": "partial",
        "project": {"path": "/work/project", "project_id": "project-1", "selection_source": "explicit"},
        "local_participation": {"state": "registered"},
        "local_agent": {"state": "active", "configured_mode": "daemon", "observed_mode": "daemon"},
        "task_observation": {"state": "active", "reason": "projection_dirty"},
        "next_actions": ["qexp task list --project /work/project"],
        "warnings": [],
        "scheduler_diagnostics": {"status": "available", "coverage": "complete", "active_count": 0},
    }

    output = _human(OutputKind.STATUS, payload)

    assert "Run: qexp task list --project /work/project" in output


def test_project_status_unregistered_action_is_qualified_and_path_is_preserved() -> None:
    payload = {
        "status": "partial",
        "project": {
            "path": "/work/project with spaces",
            "project_id": "project-1",
            "selection_source": "environment",
        },
        "local_participation": {"state": "unregistered", "reason": "no_local_binding_or_inventory"},
        "local_agent": {"state": "unavailable"},
        "task_observation": {"state": "active"},
        "next_actions": ["qexp project register '/work/project with spaces'"],
        "warnings": ["local agent status unavailable (record_missing)."],
    }

    output = _human(OutputKind.STATUS, payload)

    assert "Project: /work/project with spaces" in output
    assert "Local enrollment: not registered" in output
    assert "To run Tasks on this machine" in output
    assert "qexp project register '/work/project with spaces'" in output


def test_project_status_renders_bounded_finding_scope_counts_and_command() -> None:
    payload = {
        "status": "partial",
        "project": {"path": "/work/project", "project_id": "project-1", "selection_source": "explicit"},
        "local_participation": {"state": "registered"},
        "local_agent": {"state": "active", "configured_mode": "daemon", "observed_mode": "daemon"},
        "task_observation": {"state": "active"},
        "next_actions": [
            "qexp agent diagnostics active --project-id project-1 --reason termination_process_identity_present"
        ],
        "warnings": [],
        "scheduler_diagnostics": {
            "status": "available",
            "coverage": "complete",
            "observed_at": "2026-10-05T00:00:00Z",
            "active_count": 3,
            "active_findings": [
                {
                    "identity": {
                        "reason_code": "termination_process_identity_present",
                        "project_id": "project-1",
                        "attempt_id": "attempt-7",
                    },
                    "severity": "fault",
                    "last_observed_at": "2026-10-05T00:00:00Z",
                }
            ],
            "truncated": True,
        },
    }

    output = _human(OutputKind.STATUS, payload)

    assert "Active findings: 3" in output
    assert "Displayed findings: 1" in output
    assert "termination process identity present" in output
    assert "project-1" in output
    assert "attempt-7" in output
    assert "2026-10-05T00:00:00Z" in output
    assert "qexp agent diagnostics active" in output
    assert "Next actions" in output
    assert json.loads(render(CliOutput(OutputKind.STATUS, payload), "json")) == payload


def test_agent_status_counts_write_eligible_projects_without_false_blockers() -> None:
    output = _human(
        OutputKind.AGENT_OPERATION,
        {
            "action": "already_running",
            "outcome": "already_running",
            "machine_runtime_root": "/runtime",
            "agent_state": "active",
            "pid": 42,
            "is_running": True,
            "ready": True,
            "projects": [
                {
                    "project_id": "project-1",
                    "state": "enabled",
                    "eligibility": {"state": "eligible"},
                    "write_eligible": True,
                }
            ],
        },
    )

    assert "Projects ready: 1/1" in output
    assert "Project | State | Reason" not in output


def test_worker_change_renders_effective_policy_and_blockers() -> None:
    output = _human(
        OutputKind.GROUP_WORKER_CHANGE,
        {
            "action": "set",
            "outcome": "waiting_ack",
            "worker_machine": "gpu-2",
            "worker_state": "draining",
            "scheduling_role": "borrow",
            "gpu_limit_gpus": 2,
            "blockers": ["task-1"],
            "group": {
                "name": "experiment",
                "dispatch_state": "active",
                "worker_set": {},
            },
        },
    )

    assert "waiting ack" in output.lower()
    assert "Worker state: draining" in output
    assert "Scheduling role: borrow" in output
    assert "GPU limit: 2" in output
    assert "task-1" in output


def test_machine_show_marks_absent_observations_unavailable() -> None:
    output = _human(
        OutputKind.MACHINE_SHOW,
        {
            "machine_name": "gpu-1",
            "project": "/work/project",
            "complete": False,
            "declaration": {"machine_name": "gpu-1"},
            "agent": None,
            "gpu": None,
            "summary": None,
            "warnings": ["records missing"],
        },
    )

    assert "Agent: unavailable" in output
    assert "GPU: unavailable" in output
    assert "Summary: unavailable" in output
    assert "Agent: present" not in output
