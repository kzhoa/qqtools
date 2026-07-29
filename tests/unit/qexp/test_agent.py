from pathlib import Path

from qqtools.plugins.qexp.agent import _runtime_pid_value, run_agent_loop
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.machine_state import (
    publish_machine_snapshots,
    publish_machine_stop_snapshot,
)
from qqtools.plugins.qexp.observer import list_machines
from qqtools.plugins.qexp.runtime.store import read_json


def test_runtime_pid_value_preserves_existing_live_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "agent.pid"
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.agent._pid_alive", lambda pid: pid == 4321)

    assert _runtime_pid_value(pid_path) == 4321


def test_runtime_pid_value_falls_back_when_existing_pid_is_dead(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "agent.pid"
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.agent._pid_alive", lambda pid: False)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.os.getpid", lambda: 9876)

    assert _runtime_pid_value(pid_path) == 9876


def test_on_demand_agent_stays_active_while_reservation_exists(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    dispatch_calls = []
    reserved_values = iter([{0}, {0}, set()])
    monotonic_values = iter([0.0, 10.0, 20.0, 31.0])

    monkeypatch.setattr("qqtools.plugins.qexp.agent.release_expired_provisionals", lambda _: [])
    monkeypatch.setattr("qqtools.plugins.qexp.agent._reconcile_reservations", lambda _: None)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.reconcile_running_tasks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.reconcile_group_cancel_operations", lambda _: None)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.reconcile_cleanup_operations", lambda _: None)
    monkeypatch.setattr("qqtools.plugins.qexp.agent._offer_due_tasks", lambda _: None)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.reserved_gpu_ids", lambda _: next(reserved_values))
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.run_dispatch_cycle",
        lambda *_args, **_kwargs: dispatch_calls.append(True) or [],
    )
    monkeypatch.setattr("qqtools.plugins.qexp.agent.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("qqtools.plugins.qexp.agent.time.sleep", lambda _: None)

    run_agent_loop(cfg, idle_timeout=10, available_gpus=[0])

    assert len(dispatch_calls) == 3
    state_dir = cfg.shared_root / "machines" / "gpu-1" / "state"
    agent = read_json(state_dir / "agent.json")["agent"]
    gpu = read_json(state_dir / "gpu.json")["gpu"]
    assert agent["observed_state"] == "stopped"
    assert agent["stop_reason"] == "idle_timeout"
    assert gpu["visible_gpu_ids"] == [0]
    machine = list_machines(cfg)[0]
    assert machine["state"]["summary"]["summary"]["machine_name"] == "gpu-1"


def test_stopping_old_agent_does_not_replace_newer_machine_snapshot(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    publish_machine_snapshots(
        cfg, instance_id="new-agent", pid=123, agent_mode="daemon", observed_state="active",
        active_attempt_ids=[], visible_gpu_ids=[0], reserved_gpu_ids=[],
        heartbeat_interval_seconds=5, started_at="2026-07-29T00:00:00Z", idle_since_at=None,
    )

    assert not publish_machine_stop_snapshot(
        cfg, instance_id="old-agent", pid=None, agent_mode="daemon", visible_gpu_ids=[0],
        reserved_gpu_ids=[], heartbeat_interval_seconds=5, started_at="2026-07-28T00:00:00Z",
        idle_since_at=None, stop_reason="stopped",
    )
    agent = read_json(cfg.shared_root / "machines" / "gpu-1" / "state" / "agent.json")["agent"]
    assert agent["instance_id"] == "new-agent"
    assert agent["observed_state"] == "active"
