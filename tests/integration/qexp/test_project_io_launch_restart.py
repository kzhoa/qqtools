"""Restarted agents do not spend admission on already registered processes."""

import time
from dataclasses import replace

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import ReservationIdentity, reservation_snapshot
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("released_registration", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_registered_incumbent_leaves_launch_admission_for_new_work(tmp_path, released_registration, stale):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["true"])
    attempt = claim_task(cfg, task.task_id, [0], reservation_runtime_root=runtime.root, project_id=binding.project_id)
    assert attempt is not None
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    reservation = ReservationIdentity.from_record(reservation_snapshot(runtime.root).active[0])
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    registration = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token + int(stale),
        "machine_name": binding.machine_name,
        "reservation_id": attempt.reservation_id,
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
        "process_created_at": "2026-10-03T00:00:00Z",
    }
    atomic_replace(paths["processes"] / f"{attempt.attempt_id}.json", {"process": registration})
    if released_registration:
        registration.pop("reservation_id")
    atomic_replace(paths["registrations"] / f"{attempt.attempt_id}.json", {"process_registration": registration})
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            controller.advance_binding_validation(bindings, revision)
            if controller.validated_config(binding, revision) is not None:
                break
            time.sleep(0.02)
        assert controller.validated_config(binding, revision) is not None
        assert controller._reservation_has_registered_process(binding, reservation) is not stale
        assert not controller._reservation_has_registered_process(binding, replace(reservation, reservation_id="other"))
        controller.advance_scheduler_launch_authorizations(bindings, revision, reservations=[reservation])
        requests = [r for r in executor.unresolved_requests() if r.operation_kind == "scheduler_launch_authorize"]
        assert bool(requests) is stale
    finally:
        executor.shutdown()
