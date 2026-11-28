from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import scheduler_diagnostics
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status
from qqtools.plugins.qexp.agent.machine_snapshot_publication import MachineSnapshotPublication
from qqtools.plugins.qexp.agent.project_admin import register_project
from qqtools.plugins.qexp.agent.scheduler_diagnostics import ProducerCoverageCheckpoint, SchedulerDiagnosticStore
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.agent.snapshot_publication_health import SNAPSHOT_COVERAGE_FILENAME
from qqtools.plugins.qexp.cli.errors import CliUsageError
from qqtools.plugins.qexp.cli.local_handlers import dispatch_local
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.commands.status import MAX_READ_ATTEMPTS, project_status
from qqtools.plugins.qexp.layout import project_id as stable_project_id
from qqtools.plugins.qexp.runtime.store import atomic_replace
from tests.helpers.qexp.scheduler_recovery import overdue_identity


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
    with patch("qqtools.plugins.qexp.agent.scheduler_diagnostics.time", SimpleNamespace(monotonic_ns=lambda: 0)):
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
    assert [item["identity"]["project_id"] for item in summary["active_findings"]] == [project_id]
    assert (
        shlex.join(
            [
                "qexp",
                "--machine-runtime-root",
                str(runtime.root),
                "agent",
                "diagnostics",
                "active",
                "--project-id",
                project_id,
                "--reason",
                "ready_index_unreadable",
            ]
        )
        in result["next_actions"]
    )
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


def test_project_status_distinguishes_heartbeat_snapshot_and_post_repair_verification(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    cfg = init_shared_root(tmp_path / "selected project" / ".qexp", "gpu-1", runtime_root=tmp_path / "partition")
    binding = register_project(runtime, cfg.shared_root, "gpu-1").binding
    _publish_finding(runtime, binding.project_id)
    now = datetime.now(timezone.utc)
    heartbeat_at = (now - timedelta(seconds=1)).isoformat()
    stale_at = (now - timedelta(seconds=60)).isoformat()

    atomic_replace(
        runtime.paths["agent"] / "status.json",
        {"machine_agent": {"instance_id": "current-agent", "runtime_id": runtime.instance_id, "state": "active"}},
    )
    atomic_replace(
        runtime.root / "control_plane_health.json",
        {
            "schema_version": 1,
            "instance_id": "current-agent",
            "heartbeat": {"last_success_at": heartbeat_at, "effective_stale_after_seconds": 15},
        },
    )
    report_path = runtime.paths["agent"] / "scheduler-recovery.json"
    report = {
        "schema_version": 1,
        "runtime_id": runtime.instance_id,
        "agent_instance_id": "current-agent",
        "state": "awaiting_observation",
        "last_progress_at": (now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
        "retired": 256,
        "pending_transition": False,
        "reason": None,
        "eligible": 256,
        "deferred": 0,
        "retained_current": 0,
        "retained_uncertain": 0,
        "next_check_at": None,
    }
    atomic_replace(report_path, report)
    receipt_path = runtime.project_paths(binding.project_id)["root"] / "snapshot-observation.json"
    receipt = {
        "runtime_id": runtime.instance_id,
        "agent_instance_id": "current-agent",
        "registration_generation": binding.registration_generation,
        "snapshot_at": stale_at,
        "heartbeat_interval_seconds": 5,
    }
    atomic_replace(receipt_path, {"snapshot_observation": receipt})

    def status():
        return project_status(cfg, selection_source="explicit", machine_runtime=runtime)

    pending = status()
    assert pending["local_agent"]["heartbeat"] == {"observed_at": heartbeat_at, "freshness": "fresh"}
    assert pending["project_snapshot"] == {"observed_at": stale_at, "freshness": "stale"}
    assert pending["scheduling_readiness"]["state"] == "unknown"
    assert "snapshot_stale" in pending["scheduling_readiness"]["reasons"]
    assert pending["diagnostic_recovery"]["state"] == "awaiting_observation"
    assert pending["scheduler_diagnostics"]["coverage"] == "incomplete"
    follow_up = pending["primary_action"]
    assert shlex.split(follow_up["command"]) == [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "agent",
        "diagnostics",
        "active",
        "--project-id",
        binding.project_id,
    ]
    assert follow_up["explanation"]
    human = render(CliOutput(OutputKind.STATUS, pending), "human")
    for evidence in ("fresh", "stale", "incomplete", "awaiting observation"):
        assert evidence in human.lower()
    assert follow_up["command"] in human
    assert "running normally" not in human.lower()

    receipt["snapshot_at"] = heartbeat_at
    atomic_replace(receipt_path, {"snapshot_observation": {**receipt, "registration_generation": "other"}})
    assert status()["project_snapshot"]["freshness"] == "unknown"

    atomic_replace(receipt_path, {"snapshot_observation": receipt})
    assert status()["diagnostic_recovery"]["state"] == "awaiting_observation"
    store = SchedulerDiagnosticStore(runtime)

    publication = MachineSnapshotPublication(
        runtime, SimpleNamespace(snapshot_evidence=lambda **_identity: (heartbeat_at, heartbeat_at)), monotonic
    )

    def publish(project, observed_at, *, cached=()):
        checkpoint = store.publish_cycle(
            None,
            None,
            (
                *cached,
                {
                    "coverage": "complete",
                    "covered_bindings": [
                        {"project_id": project, "registration_generation": binding.registration_generation}
                    ],
                },
            ),
            None,
            observed_at=observed_at,
        )
        assert checkpoint is not None
        publication.record_scheduler_coverage(checkpoint, instance_id="current-agent")

    cached = {
        "coverage": "complete",
        "coverage_observed_at": stale_at,
        "covered_bindings": [
            {"project_id": binding.project_id, "registration_generation": binding.registration_generation}
        ],
    }
    publish("other-project", None, cached=(cached,))
    assert status()["diagnostic_recovery"]["state"] == "awaiting_observation"
    publish(binding.project_id, (now - timedelta(seconds=2)).isoformat())
    assert status()["diagnostic_recovery"]["state"] == "awaiting_observation"
    publish(binding.project_id, None, cached=(cached,))
    before = report_path.read_bytes()
    verified = status()
    assert verified["scheduling_readiness"]["state"] == "observed"
    assert verified["scheduling_readiness"]["diagnostic_coverage"] == {"scope": "project_binding", "state": "complete"}
    assert verified["diagnostic_recovery"]["state"] == "verified"
    assert "scheduler diagnostic recovery" not in render(CliOutput(OutputKind.STATUS, verified), "human").lower()
    assert report_path.read_bytes() == before

    with store._writer_lock():
        unavailable = status()
    assert unavailable["scheduler_diagnostics"]["status"] == "unavailable"
    assert unavailable["diagnostic_recovery"]["state"] == "awaiting_observation"

    store.publish_cycle(None, None, ({"coverage": "incomplete"},), None)
    assert status()["diagnostic_recovery"]["state"] == "awaiting_observation"
    publish(binding.project_id, None)

    publish("other-project", None, cached=(cached,))
    assert status()["diagnostic_recovery"]["state"] == "verified"

    receipt["snapshot_at"] = datetime.now(timezone.utc).isoformat()
    atomic_replace(receipt_path, {"snapshot_observation": receipt})
    assert status()["diagnostic_recovery"]["state"] == "verified"

    monkeypatch.setattr(scheduler_diagnostics, "MAX_ACTIVE_RECORDS", 1)
    with pytest.raises(OSError, match="active_overflow"):
        store.observe_finding(
            identity={
                **overdue_identity(runtime, binding.project_id, binding.registration_generation, 1),
                "target_id": "current-agent",
            },
            severity="fault",
            source_revision={"registry_revision": 1},
        )
    after_omission = status()
    assert after_omission["scheduler_diagnostics"]["coverage"] == "incomplete"
    assert after_omission["diagnostic_recovery"]["state"] == "awaiting_observation"


def test_complete_coverage_services_all_bindings_through_a_bounded_rotating_window(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    timestamp = datetime.now(timezone.utc).isoformat()
    publication = MachineSnapshotPublication(
        runtime, SimpleNamespace(snapshot_evidence=lambda **_identity: (timestamp, timestamp)), monotonic
    )
    covered = tuple((f"project-{index:02d}", "generation") for index in range(65))

    def count():
        return sum(
            (runtime.project_paths(project)["root"] / SNAPSHOT_COVERAGE_FILENAME).is_file()
            for project, _generation in covered
        )

    checkpoint = ProducerCoverageCheckpoint(dict.fromkeys(covered, datetime.now(timezone.utc).isoformat()))
    publication.record_scheduler_coverage(checkpoint, instance_id="current")
    assert count() == 16
    for _turn in range(4):
        publication.record_scheduler_coverage(checkpoint, instance_id="current")
    assert count() == len(covered)
