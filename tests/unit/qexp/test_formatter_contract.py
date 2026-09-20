import json

import pytest

from qqtools.plugins.qexp.formatter import CliOutput, OutputKind, render


def test_json_serializes_only_the_canonical_payload() -> None:
    payload = {"cpu_lane": {"capacity": 3, "revision": 7}}
    output = CliOutput(OutputKind.CPU_LANE, payload, {"action": "ignored"})

    assert json.loads(render(output, "json")) == payload


def test_cpu_lane_human_output_requires_and_renders_policy_fields() -> None:
    rendered = render(
        CliOutput(OutputKind.CPU_LANE, {"cpu_lane": {"capacity": 3, "revision": 7}}),
        "human",
    )

    assert rendered.splitlines() == ["Capacity: 3", "Revision: 7"]

    with pytest.raises((TypeError, ValueError), match="capacity"):
        render(CliOutput(OutputKind.CPU_LANE, {"cpu_lane": {"revision": 7}}), "human")


def test_project_list_human_output_has_a_distinct_table_and_empty_state() -> None:
    project = {
        "project_id": "project-1",
        "shared_root": "/shared/project",
        "machine_name": "gpu-1",
        "enabled": True,
        "state": "enabled",
        "eligibility": {"state": "eligible", "write_eligible": True},
        "write_eligible": True,
    }

    rendered = render(
        CliOutput(OutputKind.AGENT_PROJECT_LIST, {"action": "project_list", "projects": [project]}),
        "human",
    )

    assert rendered.splitlines()[0].startswith("Project ID")
    assert "project-1" in rendered
    assert "/shared/project" in rendered
    assert "gpu-1" in rendered
    assert "eligible" in rendered
    assert (
        render(
            CliOutput(OutputKind.AGENT_PROJECT_LIST, {"action": "project_list", "projects": []}),
            "human",
        )
        == "No results."
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


def test_unknown_output_kind_has_no_generic_human_fallback() -> None:
    output = CliOutput("not-registered", {"value": 1})  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match="not-registered"):
        render(output, "human")


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_agent_operation_rejects_an_action_without_its_required_payload(output_format: str) -> None:
    output = CliOutput(OutputKind.AGENT_OPERATION, {"action": "project_added"})

    with pytest.raises((TypeError, ValueError), match="project_id"):
        render(output, output_format)
