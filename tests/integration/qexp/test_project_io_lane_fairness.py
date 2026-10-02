from __future__ import annotations

import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.layout import machine_state_path
from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_cpu_claim_progresses_while_same_binding_has_free_gpu_capacity(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    set_cpu_lane_capacity(runtime.root, capacity=1)
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    task = submit(cfg, ["echo", "cpu"], requested_gpus=0, requested_cpus=1, working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    runtime.authority_ready_generations = {binding.project_id: binding.registration_generation}
    try:
        deadline = time.monotonic() + 5.0
        while binding.project_id not in controller.advance_binding_validation(bindings, revision):
            assert time.monotonic() < deadline, "binding validation did not finish"
            time.sleep(0.02)
        deadline = time.monotonic() + 10.0
        active = ()
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(runtime, available_gpus=[0], supervise=False, publish_snapshots=False)
            active = reservation_snapshot(runtime.root).active
            if active:
                break
            time.sleep(0.02)
        assert len(active) == 1, "GPU observation/empty-cursor work starved the CPU claim"
        assert active[0]["task_id"] == task.task_id
        assert active[0]["cpu_slots"] == 1
        assert active[0]["project_id"] == binding.project_id
        assert (
            load_task(cfg, task.task_id).claim_control["active_claim"]["reservation_id"] == active[0]["reservation_id"]
        )
    finally:
        executor.shutdown()


def test_snapshot_publication_progresses_despite_continuous_activation_observation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        deadline = time.monotonic() + 5.0
        while binding.project_id not in controller.advance_binding_validation(bindings, revision):
            assert time.monotonic() < deadline, "binding validation did not finish"
            time.sleep(0.02)
        runtime.working_set.reconcile(bindings, revision=revision)
        runtime.working_set.apply_activation_registrations(
            bindings,
            {binding.project_id: {"outcome": "registered", "acknowledgement": None}},
        )
        completed = set()
        deadline = time.monotonic() + 10.0
        while completed != {"observed", "published"} and time.monotonic() < deadline:
            with controller.admission_turn():
                observed = controller.advance_activation_observations(bindings, revision)
                published = controller.advance_machine_snapshot_publications(
                    bindings,
                    revision,
                    instance_id="agent-1",
                    pid=123,
                    visible_gpu_ids=[0],
                    reservations=[],
                    heartbeat_interval_seconds=5.0,
                    started_at="2026-09-28T00:00:00+00:00",
                    gpu_policy={"mode": "all", "warnings": []},
                )
            for results in (observed, published):
                if binding.project_id in results:
                    completed.add(results[binding.project_id]["outcome"])
            time.sleep(0.02)
        assert completed == {"observed", "published"}
        assert read_json(machine_state_path(cfg, "agent.json"))["agent"]["instance_id"] == "agent-1"
    finally:
        executor.shutdown()
