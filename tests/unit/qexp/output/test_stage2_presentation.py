from __future__ import annotations

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
    assert "No Groups" in _human(OutputKind.GROUP_LIST, [])
    projects = _human(OutputKind.PROJECT_LIST, {"revision": 0, "projects": []})
    assert "No enrolled Projects" in projects
    assert "qexp project register PATH" in projects


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

    assert "Previous PID: 11" in output
    assert "PID: 22" in output
    assert "Configured mode: daemon" in output
    assert "Ready: pending" in output


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
                "admission_state": "open",
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
