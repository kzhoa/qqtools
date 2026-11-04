"""Machine-only daemon startup retains ownership and generation fences."""

import json
import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent import readiness
from qqtools.plugins.qexp.agent.config import load_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _pid_start_time_ticks
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, stop_machine_agent
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_empty_daemon_cli_start_is_ready_and_idempotent(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    args = ["--machine-runtime-root", str(runtime.root), "agent", "start", "--timeout", "15", "--format=json"]
    try:
        assert main(args) == 0, capsys.readouterr().out
        first = json.loads(capsys.readouterr().out)
        assert first["ready"] is True
        assert first["projects"] == []
        assert main(args) == 0
        second = json.loads(capsys.readouterr().out)
        assert second["pid"] == first["pid"]
        assert runtime.load_registry()[1] == []
        status = get_machine_agent_status(runtime)
        assert status["is_running"] is True
        assert status["waiting_for_first_registration"] is False
    finally:
        stop_machine_agent(runtime)


def test_empty_on_demand_cli_start_does_not_spawn(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="on_demand")
    assert main(["--machine-runtime-root", str(runtime.root), "agent", "start", "--format=json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert "no_enabled_projects" in result["reasons"]
    assert not runtime.paths["pid"].exists()


def test_unknown_roster_cannot_enable_machine_only_start(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")

    def unavailable(*args, **kwargs):
        raise OSError("registry unavailable")

    monkeypatch.setattr(readiness, "reconcile_live_bindings_locked", unavailable)
    snapshot = readiness.capture_readiness_snapshot(runtime)
    assert not readiness.allows_machine_only_start(runtime, snapshot)
    result = readiness.evaluate_readiness(runtime, snapshot)
    assert result["ready"] is False
    assert result["reason"] == "registry_enablement_unknown"


@pytest.mark.parametrize("mismatch", ["instance_id", "pid", "pid_start_time_ticks", "runtime_id"])
def test_machine_only_readiness_rejects_stale_generation(tmp_path, monkeypatch, mismatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    snapshot = readiness.capture_readiness_snapshot(runtime)
    identity = (os.getpid(), "current-instance", _pid_start_time_ticks(os.getpid()))
    monkeypatch.setattr(readiness, "_active_machine_identity", lambda _: identity)
    revision = load_agent_config(runtime).revision
    status = {
        "instance_id": identity[1],
        "pid": identity[0],
        "pid_start_time_ticks": identity[2],
        "runtime_id": runtime.instance_id,
        "ready": True,
        "policy_revision_requested": revision,
        "policy_revision_acknowledged": revision,
        "reconciled_project_ids": [],
    }
    atomic_replace(runtime.paths["agent"] / "status.json", {"machine_agent": status})
    assert readiness.evaluate_readiness(runtime, snapshot)["ready"] is True
    status[mismatch] = "old" if isinstance(status[mismatch], str) else -1
    atomic_replace(runtime.paths["agent"] / "status.json", {"machine_agent": status})
    assert readiness.evaluate_readiness(runtime, snapshot)["ready"] is False


def test_machine_only_start_does_not_ignore_uncertain_reservations(tmp_path, capsys):
    from qqtools.plugins.qexp.runtime.paths import local_paths

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    path = local_paths(runtime.root)["active"] / "uncertain.json"
    atomic_replace(path, {"reservation": {"state": "active", "gpu_ids": [0]}})
    try:
        assert (
            main(["--machine-runtime-root", str(runtime.root), "agent", "start", "--timeout", "1", "--format=json"])
            == 2
        )
        result = json.loads(capsys.readouterr().out)
        assert not result["ready"]
        assert path.exists()
    finally:
        stop_machine_agent(runtime)


def test_daemon_with_only_disabled_registered_project_reaches_readiness(tmp_path, capsys):
    from qqtools.plugins.qexp.agent.setup import initialize_project, register_projects, set_project_enablement

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    project = tmp_path / "project"
    initialize_project(project)
    registered = register_projects(runtime, [project])
    assert registered["projects"][0]["status"] == "registered"
    set_project_enablement(runtime, project, False)
    try:
        assert (
            main(["--machine-runtime-root", str(runtime.root), "agent", "start", "--timeout", "15", "--format=json"])
            == 0
        ), capsys.readouterr().out
        result = json.loads(capsys.readouterr().out)
        assert result["ready"]
        assert result["projects"] == []
    finally:
        stop_machine_agent(runtime)


def test_orphan_project_ownership_blocks_machine_only_readiness(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    evidence = runtime.project_paths("orphan")["launch_intents"] / "unresolved.json"
    atomic_replace(evidence, {"launch_intent": {"state": "unknown"}})
    try:
        assert (
            main(["--machine-runtime-root", str(runtime.root), "agent", "start", "--timeout", "1", "--format=json"])
            == 2
        )
        assert not json.loads(capsys.readouterr().out)["ready"]
        assert evidence.exists()
        assert runtime.load_registry()[1] == []
    finally:
        stop_machine_agent(runtime)


def test_orphan_partition_discovery_is_bounded_and_eventually_completes(tmp_path):
    from qqtools.plugins.qexp.agent.helpers import _unregistered_partitions_settled

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    for index in range(150):
        (runtime.paths["projects"] / str(index)).mkdir()
    assert not _unregistered_partitions_settled(runtime, [])
    assert not _unregistered_partitions_settled(runtime, [])
    assert _unregistered_partitions_settled(runtime, [])


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_orphan_partition_scan_rejects_root_replacement(tmp_path, replacement):
    from qqtools.plugins.qexp.agent.helpers import _unregistered_partitions_settled

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    root = runtime.paths["projects"]
    for index in range(150):
        (root / str(index)).mkdir()
    assert not _unregistered_partitions_settled(runtime, [])
    root.rename(tmp_path / "old-projects")
    if replacement == "symlink":
        root.symlink_to(tmp_path / "old-projects", target_is_directory=True)
    else:
        atomic_replace(root / "orphan" / "launch-intents" / "unknown.json", {"launch_intent": {}})
    for _ in range(4):
        assert not _unregistered_partitions_settled(runtime, [])
