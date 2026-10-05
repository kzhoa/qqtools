from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render


def _gpu_policy_view() -> dict[str, object]:
    return {
        "mode": "auto",
        "source": "discovery",
        "revision": 0,
        "configured_gpu_ids": None,
        "discovered_gpu_ids": [0],
        "visible_gpu_ids": [0],
        "undiscovered_configured_gpu_ids": [],
        "reserved_gpu_ids": [],
        "unreserved_gpu_ids": [0],
        "draining_gpu_ids": [],
        "discovery_status": "available",
        "visible_status": "available",
        "warnings": [],
        "agent_running": False,
    }


def test_submission_result_fixtures_satisfy_json_and_human_contracts() -> None:
    fixtures = json.loads(
        (Path(__file__).parents[3] / "fixtures" / "qexp" / "submission_results.json").read_text(encoding="utf-8")
    )

    for payload in fixtures.values():
        output = CliOutput(OutputKind.SUBMISSION, payload)
        assert json.loads(render(output, "json")) == payload
        assert render(output, "human")

    diagnostic = render(CliOutput(OutputKind.SUBMISSION, fixtures["publication_failure"]), "human")
    assert "stage=entry_validate" in diagnostic
    assert "check=write.member_page.size" in diagnostic
    assert "reason=member_page_too_large" in diagnostic
    assert "actual_bytes=65537" in diagnostic


def test_cpu_lane_human_output_requires_and_renders_policy_fields() -> None:
    rendered = render(
        CliOutput(
            OutputKind.CPU_LANE,
            {
                "action": "shown",
                "machine_runtime_root": "/machine-runtime",
                "cpu_lane": {"capacity": 3, "revision": 7},
            },
        ),
        "human",
    )

    assert rendered.splitlines() == [
        "Outcome: shown",
        "MachineRuntime root: /machine-runtime",
        "Capacity: 3",
        "Revision: 7",
    ]

    with pytest.raises((TypeError, ValueError), match="capacity"):
        render(
            CliOutput(
                OutputKind.CPU_LANE,
                {"action": "shown", "machine_runtime_root": "/machine-runtime", "cpu_lane": {"revision": 7}},
            ),
            "human",
        )


def test_upgrade_registry_and_advance_render_their_distinct_project_shapes() -> None:
    registry = {
        "projects": [
            {
                "project_id": "nested-project",
                "upgrade": {
                    "phase": "backfill",
                    "state": "pending",
                    "pending": True,
                    "admission_blocked": False,
                    "blockers": [],
                },
            }
        ],
        "inaccessible_projects": [],
        "aggregate_state": "pending",
        "pending_project_ids": ["nested-project"],
        "all_roots_complete": False,
        "discovery_source": "machine_registry",
        "discovery_boundary": "locally_registered_bindings",
    }
    advance = {
        "projects": [
            {
                "project_id": "flat-project",
                "phase": "audit",
                "state": "runnable",
                "pending": True,
                "admission_blocked": True,
                "blockers": ["waiting"],
            }
        ],
        "slices": 1,
        "pending_project_ids": ["flat-project"],
        "worker_state": "runnable",
        "discovery_source": "machine_registry",
    }

    nested = render(CliOutput(OutputKind.UPGRADE_REGISTRY_STATUS, registry), "human")
    flat = render(CliOutput(OutputKind.UPGRADE_ADVANCE, advance), "human")

    assert "nested-project" in nested
    assert "backfill" in nested
    assert "pending" in nested
    assert "flat-project" in flat
    assert "audit" in flat
    assert "runnable" in flat
    assert "waiting" in flat


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_agent_operation_rejects_an_action_without_its_required_payload(output_format: str) -> None:
    output = CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_added"})

    with pytest.raises((TypeError, ValueError), match="project_id"):
        render(output, output_format)


def test_agent_status_accepts_nested_gpu_policy_view_without_runtime_root() -> None:
    payload = {
        "action": "status",
        "machine_runtime_root": "/machine-runtime",
        "agent_state": "stopped",
        "pid": None,
        "registry_revision": 0,
        "projects": [],
        "upgrade": {"projects": []},
        "gpu_policy": _gpu_policy_view(),
    }

    assert json.loads(render(CliOutput(OutputKind.AGENT_STATUS, payload), "json")) == payload


def test_agent_status_renders_project_io_isolation_only_when_degraded() -> None:
    base = {
        "action": "status",
        "machine_runtime_root": "/machine-runtime",
        "agent_state": "active",
        "pid": 17,
        "registry_revision": 0,
        "projects": [],
        "upgrade": {"projects": []},
    }
    isolation = {
        "protocol_version": 1,
        "executor_epoch": "a" * 32,
        "capacity": 4,
        "active_worker_count": 1,
        "overdue_worker_count": 1,
        "exit_unverified_worker_count": 0,
        "unreaped_worker_count": 0,
        "free_slot_count": 3,
        "supported_hang_limit": 2,
        "envelope": "degraded",
        "oldest_overdue_at": "2026-09-28T00:00:00Z",
        "blocking_project_ids": ["project-a"],
    }

    degraded = render(CliOutput(OutputKind.AGENT_STATUS, {**base, "project_io_isolation": isolation}), "human")
    healthy = render(
        CliOutput(
            OutputKind.AGENT_STATUS,
            {
                **base,
                "project_io_isolation": {
                    **isolation,
                    "active_worker_count": 0,
                    "overdue_worker_count": 0,
                    "free_slot_count": 4,
                    "envelope": "healthy",
                    "oldest_overdue_at": None,
                    "blocking_project_ids": [],
                },
            },
        ),
        "human",
    )

    assert "Attention required\n  Project I/O isolation status: degraded" in degraded
    assert "  Blocking projects: project-a" in degraded
    assert "  Workers (active / overdue / exit unverified): 1 / 1 / 0" in degraded
    assert "Attention required" not in healthy


def test_agent_status_groups_diagnostics_for_human_scanning() -> None:
    payload = {
        "action": "status",
        "machine_runtime_root": "/machine-runtime",
        "agent_state": "active",
        "pid": 17,
        "registry_revision": 0,
        "projects": [],
        "upgrade": {"projects": []},
        "diagnostics": {
            "instance_id": "current-instance",
            "state": "available",
            "log_path": "/tmp/agent.log",
            "log_available": True,
            "capture_mode": "detached",
            "capture_health": "healthy",
            "summary": {"startup_outcome": "started", "signal_attempts": "none"},
            "last_exit": {
                "reason": "stopped_by_signal",
                "handled_signal": 15,
                "cleanup_outcome": "succeeded",
                "instance_id": "previous-instance",
                "cleanup_steps": {"control_plane_stop": "outcome=succeeded"},
            },
            "last_exit_source": "last_exit",
            "last_exit_stale": True,
            "coverage": {"evicted_count": 0, "unresolved_evicted_through": 0},
        },
        "scheduler_diagnostics": {
            "schema_version": 1,
            "status": "available",
            "coverage": "incomplete",
            "truncated": False,
            "reason": "probe_data_unavailable",
            "observed_at": "2026-10-05T01:42:07Z",
            "active_count": 0,
        },
    }

    rendered = render(CliOutput(OutputKind.AGENT_STATUS, payload), "human")

    assert "Diagnostics\n  State: available\n  Capture: detached\n  Capture health: healthy" in rendered
    assert "Current startup\n  Startup outcome: started\n  Signal attempts: none" in rendered
    assert "Previous exit (stale)\n  Source: last_exit\n  Reason: stopped_by_signal" in rendered
    assert "  Handled signal: 15\n  Cleanup: succeeded\n  Instance: previous-instance" in rendered
    assert "  Full details: qexp agent status --format json" in rendered
    assert "Cleanup steps" not in rendered
    assert "Diagnostic coverage\n  Evicted count: 0\n  Unresolved evicted through: 0" in rendered
    assert "Scheduling diagnostics\n  State: available\n  Coverage: incomplete" in rendered
    assert "  Reason: probe_data_unavailable" in rendered
    assert json.loads(render(CliOutput(OutputKind.AGENT_STATUS, payload), "json")) == payload


def test_agent_status_omits_previous_exit_section_without_exit_evidence() -> None:
    payload = {
        "action": "status",
        "machine_runtime_root": "/machine-runtime",
        "agent_state": "active",
        "pid": 17,
        "registry_revision": 0,
        "projects": [],
        "upgrade": {"projects": []},
        "diagnostics": {"state": "available", "last_exit": None, "last_exit_source": None},
    }

    rendered = render(CliOutput(OutputKind.AGENT_STATUS, payload), "human")

    assert "Diagnostics\n  State: available" in rendered
    assert "Previous exit" not in rendered
    assert "Full details" not in rendered


def test_standalone_gpu_policy_still_requires_runtime_root() -> None:
    with pytest.raises((TypeError, ValueError), match="machine_runtime_root"):
        render(CliOutput(OutputKind.GPU_POLICY, {"action": "shown", **_gpu_policy_view()}), "json")

    payload = {"action": "shown", "machine_runtime_root": "/machine-runtime", **_gpu_policy_view()}
    assert json.loads(render(CliOutput(OutputKind.GPU_POLICY, payload), "json")) == payload
