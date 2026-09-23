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


def test_standalone_gpu_policy_still_requires_runtime_root() -> None:
    with pytest.raises((TypeError, ValueError), match="machine_runtime_root"):
        render(CliOutput(OutputKind.GPU_POLICY, {"action": "shown", **_gpu_policy_view()}), "json")

    payload = {"action": "shown", "machine_runtime_root": "/machine-runtime", **_gpu_policy_view()}
    assert json.loads(render(CliOutput(OutputKind.GPU_POLICY, payload), "json")) == payload
