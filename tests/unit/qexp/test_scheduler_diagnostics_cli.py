from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.errors import CliUsageError
from qqtools.plugins.qexp.cli.local_handlers import dispatch_local
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.commands.status import MAX_READ_ATTEMPTS, project_status
from qqtools.plugins.qexp.layout import project_id as stable_project_id


def _finding_identity(project_id: str) -> dict[str, object]:
    return {
        "producer": "primary_probe",
        "reason_code": "ready_index_unreadable",
        "component": "scheduler",
        "stage": "admission",
        "check": "primary_demand",
        "scope_type": "project_route",
        "runtime_id": "a" * 64,
        "project_id": project_id,
        "registration_generation": "registration-1",
        "resource_lane": "gpu",
        "route_scope": "shared",
    }


def _publish_finding(runtime: MachineRuntime, project_id: str) -> None:
    SchedulerDiagnosticStore(runtime).observe_finding(
        identity=_finding_identity(project_id),
        severity="fault",
        source_revision={"registry_revision": 1},
        observed_at="2026-09-28T00:00:00Z",
    )


def test_parser_exposes_finite_machine_local_diagnostic_commands() -> None:
    active = build_parser().parse_args(
        [
            "agent",
            "diagnostics",
            "active",
            "--project-id",
            "project-a",
            "--reason",
            "ready_index_unreadable",
            "--producer",
            "primary_probe",
            "--scope",
            "project_route",
            "--limit",
            "7",
            "--format",
            "json",
        ]
    )
    history = build_parser().parse_args(
        ["agent", "diagnostics", "history", "--project-id", "project-a", "--cursor", "token", "--limit", "3"]
    )

    assert active.command_spec.handler == "agent_diagnostics_active"
    assert active.command_spec.context.value == "machine"
    assert active.limit == 7
    assert active.format == "json"
    assert history.command_spec.handler == "agent_diagnostics_history"
    assert history.cursor == "token"


def test_local_active_command_returns_unknown_for_missing_store_without_creating_it(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    scheduler_root = runtime.root / "diagnostics" / "scheduler-v1"
    args = build_parser().parse_args(
        ["agent", "diagnostics", "active", "--machine-runtime-root", str(runtime.root), "--format", "json"]
    )

    outcome = dispatch_local("agent_diagnostics_active", args)

    assert outcome.exit_code == 0
    assert outcome.output is not None
    assert outcome.output.payload["coverage"] == "unknown"
    assert outcome.output.payload["reason"] == "store_not_initialized"
    assert not scheduler_root.exists()


def test_local_command_maps_invalid_limit_to_usage_error(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    args = build_parser().parse_args(
        ["agent", "diagnostics", "active", "--machine-runtime-root", str(runtime.root), "--limit", "33"]
    )

    with pytest.raises(CliUsageError, match="limit"):
        dispatch_local("agent_diagnostics_active", args)


def test_agent_status_keeps_process_diagnostics_and_adds_scheduler_sibling(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    _publish_finding(runtime, "project-a")

    status = get_machine_agent_status(runtime)

    assert "scheduler_diagnostics" in status
    assert status["scheduler_diagnostics"]["status"] == "available"
    assert status["scheduler_diagnostics"]["coverage"] == "complete"
    assert "diagnostics" in status


def test_project_status_reads_only_precomputed_selected_binding_summary(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    project_id = stable_project_id(cfg.shared_root)
    _publish_finding(runtime, project_id)

    result = project_status(cfg, selection_source="explicit", machine_runtime=runtime)

    summary = result["scheduler_diagnostics"]
    assert summary["status"] == "available"
    assert all(item["identity"].get("project_id") == project_id for item in summary.get("items", []))
    assert result["budget"]["read_attempts"] <= MAX_READ_ATTEMPTS


def test_diagnostics_output_contract_preserves_json_and_labels_human_history() -> None:
    payload = {
        "schema_version": 1,
        "action": "diagnostics_history",
        "machine_runtime_root": "/machine-runtime",
        "project_id": None,
        "coverage": "complete",
        "observed_at": "2026-09-28T00:00:00Z",
        "snapshot_revision": 3,
        "items": [],
        "truncated": False,
        "reason": None,
        "captured_at": "2026-09-28T00:01:00Z",
        "captured_max_sequence": 4,
        "end_of_capture": False,
        "next_cursor": "opaque-token",
        "retention": {"first_sequence": 1, "last_sequence": 4, "evicted_entries": 0},
    }
    output = CliOutput(OutputKind.SCHEDULER_DIAGNOSTICS, payload)

    assert json.loads(render(output, "json")) == payload
    human = render(output, "human")
    assert "diagnostics history" in human.lower()
    assert "opaque-token" in human
