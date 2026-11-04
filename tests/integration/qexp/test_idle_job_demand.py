"""Idle demand scans consider only work eligible for this machine's capacity."""

import time

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_probe import PrimaryProbeSession
from qqtools.plugins.qexp.agent.primary_probe_transport import encode_probe_session
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.ready import repair_ready_index

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _scan(executor, binding, revision, *, gpu_capacity=1, cpu_capacity=1):
    state = encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        request = executor.prepare_idle_job_probe(
            binding, revision, probe_state=state, gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity
        )
        assert executor.start(request.request_id) is not None
        result = None
        while time.monotonic() < deadline:
            executor.poll()
            result = executor.load_result(request.request_id)
            if result is not None:
                break
            time.sleep(0.02)
        assert result is not None, "Idle demand worker did not complete"
        assert result.status == "completed", result
        while time.monotonic() < deadline:
            executor.poll()
            if executor.consume(request.request_id, request) is not None:
                break
            time.sleep(0.02)
        if result.evidence["state"] != "pending":
            return result.evidence["state"]
        state = result.evidence["probe_state"]
    pytest.fail("Idle demand census did not reach a decision")


@pytest.mark.parametrize("lane", ["gpu", "cpu"])
@pytest.mark.parametrize("eligible_peer", [False, True])
def test_ineligible_task_does_not_hide_eligible_peer_or_block_idle(tmp_path, lane, eligible_peer):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "test-machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _ = runtime.load_registry()
    submit(
        cfg,
        ["echo", "too-large"],
        requested_gpus=2 if lane == "gpu" else 0,
        requested_cpus=2 if lane == "cpu" else None,
        working_dir=tmp_path,
    )
    if eligible_peer:
        submit(
            cfg,
            ["echo", "fits"],
            requested_gpus=1 if lane == "gpu" else 0,
            requested_cpus=1 if lane == "cpu" else None,
            working_dir=tmp_path,
        )
    repair_ready_index(cfg)
    before = {str(p.relative_to(cfg.shared_root)): p.read_bytes() for p in cfg.shared_root.rglob("*.json")}
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        assert _scan(executor, binding, revision) == ("active" if eligible_peer else "quiescent")
        assert {str(p.relative_to(cfg.shared_root)): p.read_bytes() for p in cfg.shared_root.rglob("*.json")} == before
        if not eligible_peer:
            assert _scan(executor, binding, revision, gpu_capacity=2, cpu_capacity=2) == "active"
    finally:
        executor.shutdown()


@pytest.mark.parametrize("reason", ["paused", "dependency_waiting", "dependency_blocked"])
def test_known_ineligible_work_does_not_suppress_idle_job(tmp_path, reason):
    from qqtools.plugins.qexp.commands.group import create_group, group_control
    from qqtools.plugins.qexp.commands.task import cancel

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "test-machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _ = runtime.load_registry()
    create_group(cfg, "experiment", [cfg.machine_name])
    if reason == "paused":
        submit(cfg, ["echo", "paused"], group="experiment", working_dir=tmp_path)
        group_control(cfg, "experiment", "pause", reservation_runtime_root=runtime.root)
    else:
        parent = submit(cfg, ["echo", "parent"], requested_gpus=2, group="experiment", working_dir=tmp_path)
        submit(cfg, ["echo", "child"], group="experiment", working_dir=tmp_path, depends_on_task_ids=[parent.task_id])
        if reason == "dependency_blocked":
            cancel(cfg, parent.task_id)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        assert _scan(executor, binding, revision) == "quiescent"
        if reason == "paused":
            group_control(cfg, "experiment", "resume", reservation_runtime_root=runtime.root)
            assert _scan(executor, binding, revision) == "active"
    finally:
        executor.shutdown()
