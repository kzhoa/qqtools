"""Machine-only daemon startup retains ownership and generation fences."""

import json
import os
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import lifecycle, readiness
from qqtools.plugins.qexp.agent.config import load_agent_config, set_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _pid_start_time_ticks
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, stop_machine_agent
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.cli import local_handlers
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


def test_cli_reports_verified_machine_only_start_and_status(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    prefix = ["--machine-runtime-root", str(runtime.root), "agent"]
    try:
        assert main([*prefix, "start", "--timeout", "15"]) == 0
        started = capsys.readouterr().out
        assert "started" in started.splitlines()[0].lower()
        assert "no enabled projects" in started.lower() and str(runtime.root) in started
        assert "0/0" not in started
        assert main([*prefix, "status", "--format=json"]) == 0
        status = json.loads(capsys.readouterr().out)
        assert status["ready"] and status["projects"] == []
        assert status["project_readiness"]["state"] == "machine_only"
        assert status["project_readiness"]["enabled_count"] == 0
        assert main([*prefix, "status"]) == 0
        human = capsys.readouterr().out
        assert "no enabled projects" in human.lower() and "0/0" not in human
        assert "Last cycle started" not in human
        assert "Log rotation" not in human

        previous_pid = status["pid"]
        assert main([*prefix, "restart", "--format=json"]) == 0
        restarted = json.loads(capsys.readouterr().out)
        assert restarted["previous_pid"] == previous_pid and restarted["pid"] != previous_pid
        assert restarted["project_readiness"]["enabled_count"] == 0
        assert restarted["project_readiness"]["state"] in {"pending", "machine_only"}
    finally:
        stop_machine_agent(runtime)


def test_restart_cli_reports_only_fresh_observations_for_the_replaced_agent(tmp_path, monkeypatch, capsys):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "test-machine")
    project_names = ("fresh project", "stale project", "old agent project")
    bindings = []
    for index, name in enumerate(project_names):
        project = tmp_path / name
        initialize_project(project)
        assert register_projects(runtime, [project], machine_name=f"g{index}")["projects"][0]["status"] == "registered"
        bindings.append(runtime.resolve_project_binding(project / ".qexp"))

    pid = os.getpid()
    status_path = runtime.paths["agent"] / "status.json"
    runtime.paths["pid"].write_text(str(pid), encoding="utf-8")
    atomic_replace(
        status_path,
        {
            "machine_agent": {
                "state": "active",
                "pid": pid,
                "pid_start_time_ticks": _pid_start_time_ticks(pid),
                "instance_id": "replacement-agent",
                "runtime_id": runtime.instance_id,
                "inventory_revision": runtime.load_inventory_snapshot()[0],
                "ready": True,
                "reconciled_project_ids": [binding.project_id for binding in bindings],
            },
        },
    )
    now = datetime.now(timezone.utc)
    for index, binding in enumerate(bindings):
        atomic_replace(
            runtime.project_paths(binding.project_id)["root"] / "snapshot-observation.json",
            {
                "snapshot_observation": {
                    "runtime_id": runtime.instance_id,
                    "registration_generation": binding.registration_generation,
                    "agent_instance_id": "previous-agent" if index == 2 else "replacement-agent",
                    "snapshot_at": (now - timedelta(seconds=60 if index == 1 else 1)).isoformat(),
                    "heartbeat_interval_seconds": 5,
                },
            },
        )

    monkeypatch.setattr(
        local_handlers, "restart_machine_agent", lambda runtime: SimpleNamespace(pid=pid, previous_pid=1)
    )

    def forbid_wait(*args, **kwargs):
        pytest.fail("agent restart must not enter the start readiness wait")

    monkeypatch.setattr(local_handlers, "wait_for_readiness", forbid_wait)
    prefix = ["--machine-runtime-root", str(runtime.root), "agent", "restart"]
    assert main([*prefix, "--format=json"]) == 0
    result = json.loads(capsys.readouterr().out)
    evidence = result["project_readiness"]
    assert evidence["state"] == "pending"
    assert (evidence["enabled_count"], evidence["checked_count"], evidence["fresh_count"]) == (3, 3, 1)
    assert {Path(item["path"]).name for item in evidence["affected_projects"]} == set(project_names[1:])
    assert result["ready"] is True
    assert main(prefix) == 0
    human = capsys.readouterr().out
    assert "restarted" in human.splitlines()[0].lower()
    assert "stale project" in human and "old agent project" in human
    assert "0/0" not in human
    assert "do not repeat" not in human.lower()


@pytest.mark.parametrize("missing", ["inventory", "registry"])
def test_agent_status_leaves_missing_roster_unknown_and_does_not_repair(tmp_path, capsys, missing):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    runtime.paths[missing].unlink()
    prefix = ["--machine-runtime-root", str(runtime.root), "agent", "status"]

    assert main([*prefix, "--format=json"]) == 0
    status = json.loads(capsys.readouterr().out)
    evidence = status["project_readiness"]
    assert evidence["state"] == "inventory_unknown"
    assert evidence["enabled_count"] is evidence["fresh_count"] is None
    assert not runtime.paths[missing].exists()
    assert not runtime.paths["pid"].exists()
    assert main(prefix) == 0
    human = capsys.readouterr().out
    assert "inventory" in human.lower() and "0/0" not in human
    assert "no enabled projects" not in human.lower()
    assert str(runtime.root) in human
    assert not runtime.paths[missing].exists()


def test_on_demand_start_explains_missing_enrollment_without_spawning(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "runtime with spaces")
    initialize_machine(runtime, "test-machine", agent_mode="on_demand")
    assert main(["--machine-runtime-root", str(runtime.root), "agent", "start"]) == 2
    output = capsys.readouterr().out
    assert "ready" in output.splitlines()[0].lower()
    assert "enabled Project" in output
    assert str(runtime.root) in output
    assert not runtime.paths["pid"].exists()


def _publish_test_agent_status(runtime, *, reconciled_project_ids):
    config = load_agent_config(runtime)
    pid = os.getpid()
    runtime.paths["pid"].write_text(str(pid), encoding="utf-8")
    atomic_replace(
        runtime.paths["agent"] / "status.json",
        {
            "machine_agent": {
                "state": "active",
                "pid": pid,
                "pid_start_time_ticks": _pid_start_time_ticks(pid),
                "instance_id": "test-agent",
                "runtime_id": runtime.instance_id,
                "inventory_revision": runtime.load_inventory_snapshot()[0],
                "observed_agent_mode": config.agent_mode,
                "policy_revision_requested": config.revision,
                "policy_revision_acknowledged": config.revision,
                "ready": True,
                "reconciled_project_ids": reconciled_project_ids,
            },
        },
    )


def test_machine_only_status_waits_for_current_policy_acknowledgement(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "original", agent_mode="daemon")
    _publish_test_agent_status(runtime, reconciled_project_ids=[])
    prefix = ["--machine-runtime-root", str(runtime.root), "agent", "status"]
    assert main([*prefix, "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["project_readiness"]["state"] == "machine_only"

    set_agent_config(runtime, name="new-name")

    assert main([*prefix, "--format=json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ready"] is True
    assert status["project_readiness"]["state"] == "pending"
    assert status["project_readiness"]["enabled_count"] == 0
    assert main(prefix) == 0
    human = capsys.readouterr().out
    assert "Machine-only daemon ready" not in human
    assert "readiness has not been verified" in human


def test_agent_start_waits_for_each_captured_project_to_be_reconciled(tmp_path, monkeypatch, capsys):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "machine", agent_mode="daemon")
    project = tmp_path / "training project"
    initialize_project(project)
    assert register_projects(runtime, [project])["projects"][0]["status"] == "registered"
    binding = runtime.resolve_project_binding(project / ".qexp")
    _publish_test_agent_status(runtime, reconciled_project_ids=[])
    calls = []

    def running(runtime):
        calls.append(runtime.root)
        return None, {}

    monkeypatch.setattr(local_handlers, "ensure_machine_agent_started", running)
    prefix = ["--machine-runtime-root", str(runtime.root), "agent", "start", "--timeout", "0.05"]
    assert main([*prefix, "--format=json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["timed_out"] and not result["ready"]
    assert result["projects"][0]["reason"] == "initial_reconciliation_pending"
    assert calls == [runtime.root]

    _publish_test_agent_status(runtime, reconciled_project_ids=[binding.project_id])
    assert main(prefix) == 0
    assert "Startup readiness verified" in capsys.readouterr().out
    assert calls == [runtime.root]


@pytest.mark.parametrize("lose_identity_during_observation", [False, True])
def test_agent_status_never_recreates_missing_layout_or_identity(
    tmp_path, monkeypatch, capsys, lose_identity_during_observation
):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "test-machine")
    for name in ("archives", "diagnostics"):
        runtime.paths[name].rename(tmp_path / f"retained-{name}")
    before = set(runtime.root.rglob("*"))
    if lose_identity_during_observation:
        observe = lifecycle.observe_live_bindings

        def lose_identity(runtime, *, runtime_id):
            runtime.paths["identity"].unlink()
            return observe(runtime, runtime_id=runtime_id)

        monkeypatch.setattr(lifecycle, "observe_live_bindings", lose_identity)
        before.remove(runtime.paths["identity"])

    assert main(["--machine-runtime-root", str(runtime.root), "agent", "status", "--format=json"]) == 0
    status = json.loads(capsys.readouterr().out)
    observed = status["project_readiness"]
    assert observed["state"] == ("inventory_unknown" if lose_identity_during_observation else "inactive")
    assert set(runtime.root.rglob("*")) == before
