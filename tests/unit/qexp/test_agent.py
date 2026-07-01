from __future__ import annotations

from collections import deque

import pytest

from qqtools.plugins.qexp.api import submit
from qqtools.plugins.qexp.agent import (
    _build_live_agent_snapshot,
    get_agent_status,
    is_agent_running,
    read_agent_state,
    run_agent_loop,
    start_agent_record,
    stop_agent_record,
    write_gpu_probe_failure_state,
    write_gpu_state,
    write_heartbeat,
    IDLE_TIMEOUT_DEFAULT,
)
from qqtools.plugins.qexp.indexes import update_index_on_phase_change
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.lifecycle import build_machine_workset, read_agent_snapshot
from qqtools.plugins.qexp.models import (
    AGENT_STATE_DRAINING,
    AGENT_STATE_IDLE,
    AGENT_STATE_STARTING,
    AGENT_STATE_STOPPED,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
)
from qqtools.plugins.qexp.storage import cas_update_task, load_task


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


class TestAgentState:
    def test_no_state_file(self, cfg):
        assert read_agent_state(cfg) is None
        assert not is_agent_running(cfg)

    def test_start_writes_state(self, cfg):
        start_agent_record(cfg, persistent=False)
        state = read_agent_state(cfg)
        assert state is not None
        assert state["agent_state"] == AGENT_STATE_STARTING
        assert state["pid"] is not None
        assert state["idle_timeout_seconds"] == IDLE_TIMEOUT_DEFAULT
        assert state["workset"]["has_active_responsibility"] is False

    def test_persistent_flag(self, cfg):
        start_agent_record(cfg, persistent=True)
        state = read_agent_state(cfg)
        assert state["idle_timeout_seconds"] == 0
        assert state["agent_mode"] == "persistent"

    def test_stop_clears_pid(self, cfg):
        start_agent_record(cfg)
        stop_agent_record(cfg, reason="test")
        state = read_agent_state(cfg)
        assert state["agent_state"] == AGENT_STATE_STOPPED
        assert state["pid"] is None
        assert state["last_exit_reason"] == "test"

    def test_heartbeat(self, cfg):
        start_agent_record(cfg)
        old_hb = read_agent_state(cfg)["last_heartbeat"]
        write_heartbeat(cfg)
        new_hb = read_agent_state(cfg)["last_heartbeat"]
        assert new_hb >= old_hb

    def test_write_gpu_state_updates_runtime_snapshot_and_machine_snapshot(self, cfg):
        from types import SimpleNamespace

        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.storage import load_machine, read_json

        tracker = SimpleNamespace(
            visible_gpu_ids=[0, 1, 2],
            reserved_gpu_ids={1},
            task_id_to_gpu_ids={"task-a": [1]},
            backend_name="stub",
        )

        write_gpu_state(cfg, tracker)

        gpu_state = read_json(gpu_state_path(cfg))
        assert gpu_state["gpu_count"] == 3
        assert gpu_state["visible_gpu_ids"] == [0, 1, 2]
        assert gpu_state["reserved_gpu_ids"] == [1]
        assert gpu_state["backend"] == "stub"
        assert gpu_state["probe_succeeded"] is True

        machine = load_machine(cfg)
        assert machine.gpu_inventory.count == 3
        assert machine.gpu_inventory.visible_gpu_ids == [0, 1, 2]

    def test_write_gpu_probe_failure_state_persists_error_without_overwriting_machine_snapshot(self, cfg):
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.models import GpuInventory
        from qqtools.plugins.qexp.storage import load_machine, read_json, save_machine

        machine = load_machine(cfg)
        machine.gpu_inventory = GpuInventory(count=4, visible_gpu_ids=[0, 1, 2, 3])
        save_machine(cfg, machine)

        write_gpu_probe_failure_state(cfg, RuntimeError("nvml exploded"), backend="pynvml")

        gpu_state = read_json(gpu_state_path(cfg))
        assert gpu_state["probe_succeeded"] is False
        assert gpu_state["probe_error"] == "nvml exploded"
        assert gpu_state["backend"] == "pynvml"
        assert gpu_state["gpu_count"] is None
        assert gpu_state["visible_gpu_ids"] == []

        machine = load_machine(cfg)
        assert machine.gpu_inventory.count == 4
        assert machine.gpu_inventory.visible_gpu_ids == [0, 1, 2, 3]


class TestGetAgentStatus:
    def test_no_agent(self, cfg):
        status = get_agent_status(cfg)
        assert status["agent_state"] == AGENT_STATE_STOPPED
        assert not status["is_running"]

    def test_running_agent(self, cfg):
        start_agent_record(cfg)
        status = get_agent_status(cfg)
        assert status["is_running"]

    def test_stale_agent(self, cfg):
        start_agent_record(cfg)
        # Fake a dead PID
        from qqtools.plugins.qexp.storage import write_atomic_json
        from qqtools.plugins.qexp.layout import agent_state_path
        snapshot = read_agent_snapshot(cfg)
        snapshot.pid = 99999999
        write_atomic_json(agent_state_path(cfg), snapshot.to_dict())
        status = get_agent_status(cfg)
        assert not status["is_running"]
        assert status["agent_state"] == "stale"


class TestAgentLifecycle:
    def test_running_task_keeps_agent_draining_until_idle(self, cfg, monkeypatch):
        task = submit(cfg, command=["echo"], task_id="t1")
        self._set_phase(cfg, task.task_id, PHASE_QUEUED, PHASE_RUNNING)

        timestamps = deque([
            "2026-04-14T00:00:00Z",
            "2026-04-14T00:00:01Z",
            "2026-04-14T00:00:02Z",
            "2026-04-14T00:00:03Z",
            "2026-04-14T00:00:04Z",
        ])
        snapshots = []
        original_write_snapshot = None

        import qqtools.plugins.qexp.agent as agent_mod

        original_write_snapshot = agent_mod.write_agent_snapshot
        monkeypatch.setattr(
            agent_mod,
            "utc_now_iso",
            lambda: timestamps.popleft(),
        )

        def capture_snapshot(local_cfg, snapshot):
            snapshots.append(snapshot.agent_state)
            original_write_snapshot(local_cfg, snapshot)

        monkeypatch.setattr(agent_mod, "write_agent_snapshot", capture_snapshot)
        monkeypatch.setattr(agent_mod.time, "sleep", lambda _: None)

        call_count = {"reconcile": 0}

        def dispatch_fn(local_cfg, tracker_factory):
            return []

        def reconcile_fn(local_cfg):
            call_count["reconcile"] += 1
            if call_count["reconcile"] == 2:
                self._set_phase(local_cfg, "t1", PHASE_RUNNING, PHASE_FAILED)

        ret = run_agent_loop(
            cfg,
            loop_interval=0.0,
            idle_timeout=1,
            dispatch_fn=dispatch_fn,
            reconcile_fn=reconcile_fn,
            tracker_factory=lambda: object(),
        )

        assert ret == 0
        assert AGENT_STATE_DRAINING in snapshots
        assert AGENT_STATE_IDLE in snapshots
        final_state = read_agent_state(cfg)
        assert final_state["agent_state"] == AGENT_STATE_STOPPED
        assert final_state["last_exit_reason"] == "idle_timeout"

    def test_starting_phase_is_active_not_idle(self, cfg):
        task = submit(cfg, command=["echo"], task_id="t1")
        self._set_phase(cfg, task.task_id, PHASE_QUEUED, PHASE_STARTING)
        workset = build_machine_workset(cfg)
        snapshot = _build_live_agent_snapshot(
            cfg=cfg,
            previous=None,
            workset=workset,
            persistent=False,
            idle_timeout=600,
        )
        assert snapshot.agent_state != AGENT_STATE_IDLE
        assert snapshot.agent_state != AGENT_STATE_DRAINING
        assert snapshot.agent_state == "active"

    def test_other_machine_running_task_does_not_block_idle_exit(self, cfg, monkeypatch):
        other_cfg = init_shared_root(
            cfg.shared_root,
            "gpu2",
            runtime_root=cfg.runtime_root.parent / "runtime2",
        )
        remote_task = submit(other_cfg, command=["echo"], task_id="remote")
        self._set_phase(other_cfg, remote_task.task_id, PHASE_QUEUED, PHASE_RUNNING)

        timestamps = deque([
            "2026-04-14T00:00:00Z",
            "2026-04-14T00:00:01Z",
            "2026-04-14T00:00:02Z",
            "2026-04-14T00:00:03Z",
        ])
        import qqtools.plugins.qexp.agent as agent_mod

        monkeypatch.setattr(agent_mod, "utc_now_iso", lambda: timestamps.popleft())
        monkeypatch.setattr(agent_mod.time, "sleep", lambda _: None)

        ret = run_agent_loop(
            cfg,
            loop_interval=0.0,
            idle_timeout=1,
            dispatch_fn=lambda local_cfg, tracker_factory: [],
            reconcile_fn=lambda local_cfg: None,
            tracker_factory=lambda: object(),
        )

        assert ret == 0
        final_state = read_agent_state(cfg)
        assert final_state["agent_state"] == AGENT_STATE_STOPPED
        assert final_state["last_exit_reason"] == "idle_timeout"

    @staticmethod
    def _set_phase(cfg, task_id: str, old_phase: str, new_phase: str) -> None:
        task = load_task(cfg, task_id)
        task.status.phase = new_phase
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, task_id, old_phase, new_phase)
