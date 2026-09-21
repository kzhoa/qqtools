from __future__ import annotations

import json
import time


def test_gpu_policy_cli_is_machine_scoped_and_revisioned(tmp_path, capsys) -> None:
    from qqtools.plugins.qexp.cli import main

    runtime_root = tmp_path / "machine"
    common = ["--machine-runtime-root", str(runtime_root), "agent", "gpus"]

    assert main([*common, "set", "--visible", "2,0", "--expected-revision", "0", "--format", "json"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["previous_revision"] == 0
    assert current["current_revision"] == 1
    assert current["mode"] == "explicit"
    assert current["configured_gpu_ids"] == [0, 2]

    assert main([*common, "set", "--none", "--expected-revision", "1", "--format", "json"]) == 0
    none = json.loads(capsys.readouterr().out)
    assert none["current_revision"] == 2
    assert none["configured_gpu_ids"] == []

    assert main([*common, "reset", "--expected-revision", "2", "--format", "json"]) == 0
    reset = json.loads(capsys.readouterr().out)
    assert reset["current_revision"] == 3
    assert reset["mode"] == "auto"
    assert reset["source"] == "pending"

    assert main([*common, "show", "--format", "json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["revision"] == 3
    assert shown["source"] == "pending"
    assert shown["agent_running"] is False
    assert shown["visible_gpu_ids"] is None


def test_gpu_policy_cli_rejects_invalid_lists_without_replacing_policy(tmp_path, capsys) -> None:
    from qqtools.plugins.qexp.cli import main

    runtime_root = tmp_path / "machine"
    common = ["--machine-runtime-root", str(runtime_root), "agent", "gpus"]
    assert main([*common, "set", "--visible", "0,1"]) == 0
    capsys.readouterr()

    for invalid in ("", "0,", "0,0", "+1", "1-2"):
        assert main([*common, "set", "--visible", invalid]) == 2
        capsys.readouterr()

    assert main([*common, "show", "--format", "json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["revision"] == 1
    assert shown["configured_gpu_ids"] == [0, 1]


def test_explicit_none_does_not_change_cpu_lane(tmp_path, capsys) -> None:
    from qqtools.plugins.qexp.cli import main

    runtime_root = tmp_path / "machine"
    root = ["--machine-runtime-root", str(runtime_root), "agent"]
    assert main([*root, "cpu-lane", "set", "--capacity", "2", "--format", "json"]) == 0
    capsys.readouterr()
    assert main([*root, "gpus", "set", "--none", "--format", "json"]) == 0
    capsys.readouterr()
    assert main([*root, "cpu-lane", "show", "--format", "json"]) == 0
    lane = json.loads(capsys.readouterr().out)
    assert lane["cpu_lane"] == {"capacity": 2, "revision": 1}


def test_removed_reserved_gpu_drains_without_releasing_reservation(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp import gpu_policy
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy
    from qqtools.plugins.qexp.runtime.resources.reservations import attach, reserve
    from qqtools.plugins.qexp.runtime.store import read_json

    runtime = MachineRuntime(tmp_path / "machine")
    discovery = GpuDiscovery((0, 1), "available", None)
    monkeypatch.setattr(gpu_policy, "discover_gpu_inventory", lambda: discovery)
    gpu_policy.set_gpu_policy(runtime, (0, 1))
    context = GpuReservationPolicy(discovery, None, "absent")
    reservation = reserve(runtime.root, "task-running", [1], gpu_policy=context)
    reservation_id = reservation["reservation"]["reservation_id"]
    attach(runtime.root, reservation_id, "attempt-running", 7)

    changed = gpu_policy.set_gpu_policy(runtime, (0,))
    assert changed["entered_draining_gpu_ids"] == [1]
    shown = gpu_policy.show_gpu_policy(runtime)
    assert shown["visible_gpu_ids"] == [0]
    assert shown["reserved_gpu_ids"] == [1]
    assert shown["unreserved_gpu_ids"] == [0]
    assert shown["draining_gpu_ids"] == [1]
    active = read_json(runtime.paths["active"] / f"{reservation_id}.json")["reservation"]
    assert active["attempt_id"] == "attempt-running"
    assert active["fencing_token"] == 7


def test_running_agent_applies_policy_without_pid_change(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.gpu_policy import set_gpu_policy, show_gpu_policy

    monkeypatch.delenv("QEXP_VISIBLE_GPUS", raising=False)
    runtime = MachineRuntime(tmp_path / "machine")
    process = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.05)
    try:
        original_pid = process.pid
        set_gpu_policy(runtime, (0,))
        deadline = time.monotonic() + 5
        shown = show_gpu_policy(runtime)
        while shown["visible_gpu_ids"] != [0] and time.monotonic() < deadline:
            time.sleep(0.05)
            shown = show_gpu_policy(runtime)

        assert shown["visible_gpu_ids"] == [0]
        assert shown["source"] == "persisted"
        assert get_machine_agent_status(runtime)["pid"] == original_pid
        assert process.poll() is None
    finally:
        stop_machine_agent(runtime)


def test_startup_retains_actionable_missing_gpu_warning(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.gpu_policy import set_gpu_policy, show_gpu_policy

    monkeypatch.delenv("QEXP_VISIBLE_GPUS", raising=False)
    runtime = MachineRuntime(tmp_path / "machine")
    set_gpu_policy(runtime, (12, 1))
    process = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.05)
    try:
        deadline = time.monotonic() + 5
        status = get_machine_agent_status(runtime)
        while not status.get("warnings") and time.monotonic() < deadline:
            time.sleep(0.05)
            status = get_machine_agent_status(runtime)

        warning = next(item for item in status["warnings"] if item["reason"] == "configured_gpu_ids_not_discovered")
        assert warning["undiscovered_configured_gpu_ids"] == [12]
        assert "--visible 1" in "\n".join(warning["repair_commands"])
        shown = show_gpu_policy(runtime)
        assert shown["warnings"] == status["warnings"]
        assert shown["agent_running"] is True
        assert process.poll() is None
    finally:
        stop_machine_agent(runtime)


def test_crashed_agent_observation_is_not_treated_as_running(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp import gpu_policy
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy, resolve_gpu_policy
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()
    dead_pid = 99_999_999
    runtime.paths["pid"].write_text(str(dead_pid), encoding="utf-8")
    atomic_replace(
        runtime.paths["agent"] / "status.json",
        {
            "machine_agent": {
                "instance_id": "dead-instance",
                "pid": dead_pid,
                "pid_start_time_ticks": 1,
                "state": "active",
            }
        },
    )
    discovery = GpuDiscovery((9,), "available", None)
    view = resolve_gpu_policy(
        runtime.root,
        discovery=discovery,
        environment_gpu_ids=(9,),
        environment_status="valid",
    )
    gpu_policy.persist_gpu_policy_observation(
        runtime,
        instance_id="dead-instance",
        pid=dead_pid,
        view=view,
        policy=GpuReservationPolicy(discovery, (9,), "valid"),
    )
    monkeypatch.setattr(gpu_policy, "discover_gpu_inventory", lambda: GpuDiscovery((0,), "available", None))
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "7")

    shown = gpu_policy.show_gpu_policy(runtime)
    assert shown["agent_running"] is False
    assert shown["source"] == "pending"
    assert shown["visible_gpu_ids"] is None
