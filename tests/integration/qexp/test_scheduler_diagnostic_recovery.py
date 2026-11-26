"""Ordinary commands expose background diagnostic recovery from the original Project."""

from __future__ import annotations

import json
import os
import shlex
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import lifecycle
from qqtools.plugins.qexp.agent import scheduler_diagnostics as diagnostics
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.cli import local_handlers
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import read_json
from tests.helpers.qexp.lifecycle import LifecycleLab, wait_until
from tests.helpers.qexp.scheduler_recovery import first_project_history, overdue_identity, seed_over_budget_store

pytestmark = pytest.mark.integration


@pytest.fixture
def project_machine(tmp_path, monkeypatch, qexp_resource_scope):
    runtime = MachineRuntime(qexp_resource_scope.runtime_root)
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    project = tmp_path / "original project"
    initialize_project(project)
    assert register_projects(runtime, [project])["projects"][0]["status"] == "registered"
    binding = runtime.load_registry()[1][0]
    monkeypatch.chdir(project)
    lab = LifecycleLab(runtime=runtime, available_gpus=[])
    real_restart = lifecycle.restart_machine_agent

    def restart(target):
        process = lab.adopt_process(real_restart(target))
        qexp_resource_scope.record_resource(
            "participant",
            {"pid": process.pid, "process_group_id": os.getpgid(process.pid)},
        )
        return process

    monkeypatch.setattr(local_handlers, "restart_machine_agent", restart)
    try:
        yield runtime, binding, lab
    finally:
        lab.close()


def _cli_json(capsys, *args):
    assert main([*args, "--format=json"]) == 0, capsys.readouterr().out
    return json.loads(capsys.readouterr().out)


def _recovery_record(runtime):
    try:
        return read_json(runtime.paths["agent"] / "scheduler-recovery.json")
    except (OSError, ValueError):
        return {}


def test_restart_from_project_recovers_full_store_without_foreground_repair(project_machine, capsys, monkeypatch):
    runtime, binding, lab = project_machine
    with monkeypatch.context() as setup:
        setup.setattr(diagnostics, "time", SimpleNamespace(monotonic_ns=lambda: 0))
        seed_over_budget_store(runtime, binding.project_id, binding.registration_generation)
        store = SchedulerDiagnosticStore(runtime)
        with pytest.raises(OSError, match="active_overflow"):
            store.observe_finding(
                identity=overdue_identity(runtime, binding.project_id, binding.registration_generation, 999),
                severity="fault",
                source_revision={"registry_revision": 1},
            )
        assert store.summary_view()["reason"] == "active_overflow"
        item = store.active_view(project_id=binding.project_id, limit=1)["items"][0]
        with pytest.raises(OSError, match="publication_byte_limit"):
            store.resolve_finding(
                identity=item["identity"],
                episode=item["episode"],
                observation_revision=item["observation_revision"],
                source_revision=item["source_revision"],
            )

    restarted = _cli_json(capsys, "agent", "restart")
    assert restarted["pid"] == lab.process.pid
    assert restarted["diagnostic_recovery"]["state"] in {"pending", "progressing", "awaiting_observation"}
    now = _cli_json(capsys, "status")
    assert now["diagnostic_recovery"]["state"] in {"pending", "progressing", "awaiting_observation", "idle", "verified"}

    def background_finished():
        report = _recovery_record(runtime)
        if "scheduler_diagnostic_recovery" in report:
            report = report["scheduler_diagnostic_recovery"]
        return report.get("retired", 0) == diagnostics.MAX_ACTIVE_RECORDS

    # Only observe the durable checkpoint. No status, repair, or watch request
    # drives the backlogged agent while it performs the bounded work.
    wait_until("all former findings retired", background_finished, timeout=45, stage="diagnostic recovery")
    assert lab.process.poll() is None
    assert store.active_view(project_id=binding.project_id, reason="project_snapshot_overdue")["total"] == 0
    history = first_project_history(store, binding.project_id)
    assert history["disposition"] == "superseded"

    def verified_status():
        result = _cli_json(capsys, "status")
        if result["scheduling_readiness"]["state"] == "observed":
            return result
        return None

    wait_until("selected binding verified", verified_status, timeout=35, stage="diagnostic recovery")
    status = verified_status()
    assert status["scheduler_diagnostics"]["coverage"] == "complete"
    assert status["scheduler_diagnostics"]["machine_coverage"] == "incomplete"
    assert status["project_snapshot"]["freshness"] == "fresh"
    assert status["local_agent"]["heartbeat"]["freshness"] == "fresh"
    assert status["diagnostic_recovery"]["state"] in {"idle", "verified"}
    assert main(["status"]) == 0
    human = capsys.readouterr().out
    assert "awaiting observation" not in human.lower()
    assert "superseded" not in human.lower()
    assert main(["agent", "status"]) == 0
    machine_human = capsys.readouterr().out.lower()
    assert "recovery is awaiting" not in machine_human
    assert "recovery is progressing" not in machine_human


def test_restart_blocker_gives_direct_preview_and_committed_queue_outcome(project_machine, capsys, monkeypatch):
    runtime, binding, _lab = project_machine
    with monkeypatch.context() as setup:
        setup.setattr(diagnostics, "time", SimpleNamespace(monotonic_ns=lambda: 0))
        store = SchedulerDiagnosticStore(runtime)
        store.observe_finding(
            identity=overdue_identity(runtime, binding.project_id, binding.registration_generation, 1),
            severity="fault",
            source_revision={"registry_revision": 1},
        )
    index = store.paths["scheduler_diagnostics_history_index"]
    original = index.read_bytes()
    index.rename(index.with_suffix(".saved"))
    index.mkdir()

    _cli_json(capsys, "agent", "restart")

    def blocked_status():
        result = _cli_json(capsys, "status")
        return result["diagnostic_recovery"]["state"] == "blocked"

    wait_until("storage blocker published", blocked_status, timeout=10, stage="diagnostic recovery")
    status = _cli_json(capsys, "status")
    blocker = status["diagnostic_recovery"]
    assert blocker["state"] == "blocked"
    assert blocker["reason"]
    assert status["scheduling_readiness"]["state"] == "unknown"
    preview_args = shlex.split(blocker["next_action"])
    assert preview_args == [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "agent",
        "diagnostics",
        "repair",
        "--dry-run",
    ]
    preview = _cli_json(capsys, *preview_args[1:])
    assert preview["state"] == "blocked"
    assert preview["scope"] == "machine_runtime"
    assert preview["backup_path"] is None
    assert index.is_dir() and index.with_suffix(".saved").read_bytes() == original

    assert main(["submit", "--format=json", "--no-activate", "--", "echo", "queued"]) == 0
    submitted = json.loads(capsys.readouterr().out)
    assert submitted["outcome"] == "committed"
    assert submitted["activation"]["outcome"] == "not_requested"
    assert submitted["diagnostic_recovery"]["next_action"] == blocker["next_action"]
