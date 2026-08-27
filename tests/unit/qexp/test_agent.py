from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent import run_agent_loop
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.machine_state import (
    publish_machine_snapshots,
    publish_machine_stop_snapshot,
)
from qqtools.plugins.qexp.runtime.store import read_json


def test_standalone_agent_runtime_is_removed(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    with pytest.raises(RuntimeError, match="standalone agent runtime was removed"):
        run_agent_loop(cfg)


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
