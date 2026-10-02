from __future__ import annotations

import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_stop import publish_project_stop_snapshots
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.layout import machine_state_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _bindings(tmp_path: Path, count: int):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    pairs = []
    for index in range(count):
        cfg = init_shared_root(tmp_path / f"project-{index}" / ".qexp", "gpu-1")
        pairs.append((cfg, runtime.add_binding(cfg.shared_root, cfg.machine_name)))
    revision, _registered = runtime.load_registry()
    return runtime, pairs, revision


def test_stop_snapshots_publish_through_fresh_executor_epoch(tmp_path: Path) -> None:
    runtime, pairs, _revision = _bindings(tmp_path, 2)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    executor.shutdown()

    publish_project_stop_snapshots(
        runtime,
        executor,
        instance_id="agent-a",
        available_gpus=[0],
        heartbeat_interval_seconds=0.1,
        started_at="2026-10-01T00:00:00Z",
        stop_reason="idle",
    )

    for cfg, _binding in pairs:
        agent = read_json(machine_state_path(cfg, "agent.json"))["agent"]
        assert agent["instance_id"] == "agent-a"
        assert agent["observed_state"] == "stopped"
        assert agent["stop_reason"] == "idle"
    assert executor.status_view()["active_worker_count"] == 0


def test_two_retained_blockers_do_not_hide_healthy_stop_publication(tmp_path: Path) -> None:
    runtime, pairs, revision = _bindings(tmp_path, 3)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    restored = []
    try:
        for cfg, binding in pairs[:2]:
            identity = cfg.shared_root / "project" / "identity.json"
            saved = identity.with_suffix(".saved")
            identity.rename(saved)
            os.mkfifo(identity)
            restored.append((identity, saved))
            request = executor.prepare_validate_binding(binding, revision)
            assert executor.start(request.request_id) is not None
        executor.shutdown()

        with pytest.raises(RuntimeError, match="2 Project stop publication"):
            publish_project_stop_snapshots(
                runtime,
                executor,
                instance_id="agent-a",
                available_gpus=[0],
                heartbeat_interval_seconds=0.1,
                started_at="2026-10-01T00:00:00Z",
                stop_reason="stopped_by_signal",
            )

        healthy = read_json(machine_state_path(pairs[2][0], "agent.json"))["agent"]
        assert healthy["observed_state"] == "stopped"
        assert healthy["stop_reason"] == "stopped_by_signal"
        assert len(executor.status_view()["blocking_project_ids"]) == 2
    finally:
        for identity, saved in restored:
            identity.unlink(missing_ok=True)
            saved.rename(identity)
        executor.shutdown()
