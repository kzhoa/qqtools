"""Advisory write suppression preserves freshness, state changes, and repair."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp import machine_state
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _publish_process_status
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime import store

pytestmark = pytest.mark.integration


def test_machine_snapshots_skip_only_identical_persisted_content(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    now = "2026-09-18T10:00:00Z"
    monkeypatch.setattr(machine_state, "utc_now", lambda: now)
    writes = []
    original = store.atomic_replace

    def record(path, value):
        writes.append(path.name)
        original(path, value)

    monkeypatch.setattr(store, "atomic_replace", record)
    kwargs = dict(
        instance_id="agent-1",
        pid=123,
        agent_mode="daemon",
        observed_state="active",
        active_attempt_ids=[],
        visible_gpu_ids=[0],
        reserved_gpu_ids=[],
        heartbeat_interval_seconds=0.1,
        started_at=now,
        idle_since_at=None,
    )
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == ["agent.json", "gpu.json", "summary.json"]
    writes.clear()
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == []

    # A real change within the same timestamp must still become visible.
    kwargs.update(active_attempt_ids=["attempt-1"], reserved_gpu_ids=[0])
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == ["agent.json", "gpu.json", "summary.json"]
    writes.clear()
    now = "2026-09-18T10:00:01Z"
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == ["agent.json", "gpu.json", "summary.json"]
    writes.clear()

    root = cfg.shared_root / "machines" / "gpu-1" / "state"
    (root / "gpu.json").unlink()
    (root / "summary.json").write_text("broken")
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == ["gpu.json", "summary.json"]
    assert store.read_json(root / "gpu.json")["gpu"]["reserved_gpu_ids"] == [0]
    writes.clear()

    kwargs.update(instance_id="agent-2", pid=456)
    machine_state.publish_machine_snapshots(cfg, **kwargs)
    assert writes == ["agent.json"]
    assert store.read_json(root / "agent.json")["agent"]["instance_id"] == "agent-2"


def test_process_status_changes_and_missing_file_are_published(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    kwargs = dict(instance_id="agent-1", pid=123, start_ticks=456, waiting_for_first_registration=True)
    writes = []
    original = store.atomic_replace

    def record(path, value):
        writes.append(value)
        original(path, value)

    monkeypatch.setattr(store, "atomic_replace", record)
    _publish_process_status(runtime, **kwargs)
    _publish_process_status(runtime, **kwargs)
    assert len(writes) == 1
    kwargs["waiting_for_first_registration"] = False
    _publish_process_status(runtime, **kwargs)
    assert len(writes) == 2
    path = runtime.paths["agent"] / "status.json"
    path.unlink()
    _publish_process_status(runtime, **kwargs)
    assert len(writes) == 3
    _publish_process_status(runtime, **kwargs, state="stopped")
    assert len(writes) == 4
    assert store.read_json(path)["machine_agent"]["pid"] is None


def test_failed_snapshot_write_is_retried(tmp_path: Path, monkeypatch):
    path = tmp_path / "snapshot.json"
    original = store.atomic_replace

    def fail(*_args):
        raise OSError("write failed")

    monkeypatch.setattr(store, "atomic_replace", fail)
    with pytest.raises(OSError, match="write failed"):
        store.replace_snapshot_if_changed(path, {"state": "active"})
    monkeypatch.setattr(store, "atomic_replace", original)
    assert store.replace_snapshot_if_changed(path, {"state": "active"})
    assert store.read_json(path) == {"state": "active"}
