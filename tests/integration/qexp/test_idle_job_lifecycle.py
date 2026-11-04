"""Machine Idle Job lifecycle without Project or experiment identities."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job import IdleJobSupervisor
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.idle_job_process import idle_job_processes_exited
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy, set_gpu_policy
from qqtools.plugins.qexp.runtime.resources.cpu_lane import reserve_cpu, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.idle_job import (
    IdleJobAdmissionBlocked,
    idle_job_instance_path,
    read_idle_job_ownership,
)
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot, reserve

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

POLICY = GpuReservationPolicy(GpuDiscovery((0, 1, 3), "available"), None, "unset")


def _runtime(tmp_path, **changes):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    set_idle_job_config(
        runtime,
        enabled=True,
        command=f'{sys.executable} -c "import time; time.sleep(60)"',
        idle_delay_seconds=0,
        stop_timeout_seconds=0.2,
        **changes,
    )
    return runtime


def _until(supervisor, predicate, *, demand="idle", policy=POLICY, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = supervisor.tick(demand=demand, gpu_policy=policy)
        if predicate(status):
            return status
        time.sleep(0.03)
    pytest.fail(f"Idle Job did not converge: {status}")


def test_empty_machine_launch_disable_and_no_experiment_reservations(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        status = _until(supervisor, lambda value: value["state"] == "running")
        assert status["resolved_gpu_ids"] == [3]
        assert runtime.load_registry()[1] == []
        assert not reservation_snapshot(runtime.root).reservations
        set_idle_job_config(runtime, enabled=False)
        _until(supervisor, lambda value: value["state"] == "disabled")
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("lane", ["gpu", "different_gpu", "cpu"])
def test_eligible_admission_waits_for_process_cleanup(tmp_path, lane):
    runtime = _runtime(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    supervisor = IdleJobSupervisor(runtime)

    def admit():
        if lane == "cpu":
            return reserve_cpu(runtime.root, "training", 1)
        return reserve(runtime.root, "training", [3 if lane == "gpu" else 0])

    try:
        _until(supervisor, lambda value: value["state"] == "running")
        with pytest.raises(IdleJobAdmissionBlocked):
            admit()
        assert not reservation_snapshot(runtime.root).reservations
        _until(supervisor, lambda value: read_idle_job_ownership(runtime.root) is None, demand="busy")
        assert admit()["reservation"]["task_id"] == "training"
    finally:
        assert supervisor.shutdown()


def test_gpu_policy_removal_stops_and_reselects_allowed_device(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda value: value["state"] == "running")
        old = read_idle_job_ownership(runtime.root)["instance_id"]
        set_gpu_policy(runtime, (0, 1))
        status = _until(
            supervisor,
            lambda value: value["state"] == "running" and read_idle_job_ownership(runtime.root)["instance_id"] != old,
        )
        assert status["resolved_gpu_ids"] == [1]
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("demand", ["busy", "unknown"])
def test_no_launch_without_proven_idle(tmp_path, demand):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        for _ in range(3):
            status = supervisor.tick(demand=demand, gpu_policy=POLICY)
            assert status["state"] != "running"
            assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("lane", ["other_gpu", "cpu"])
def test_concurrent_launch_and_admission_share_exclusion(tmp_path, lane):
    runtime = _runtime(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    supervisor = IdleJobSupervisor(runtime)
    gate = threading.Barrier(2)

    def idle_launch():
        gate.wait(timeout=5)
        return supervisor.tick(demand="idle", gpu_policy=POLICY)

    def training_admission():
        gate.wait(timeout=5)
        try:
            if lane == "cpu":
                return reserve_cpu(runtime.root, "training", 1)
            return reserve(runtime.root, "training", [0])
        except IdleJobAdmissionBlocked:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            idle_future = pool.submit(idle_launch)
            training_future = pool.submit(training_admission)
            idle_future.result(timeout=10)
            admitted = training_future.result(timeout=10)
        if admitted is not None:
            assert read_idle_job_ownership(runtime.root) is None
            for _ in range(3):
                supervisor.tick(demand="idle", gpu_policy=POLICY)
                assert read_idle_job_ownership(runtime.root) is None
        else:
            assert not reservation_snapshot(runtime.root).reservations
    finally:
        assert supervisor.shutdown()


def test_command_exit_uses_backoff_and_training_supersedes_retry(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, command=f'{sys.executable} -c "raise SystemExit(7)"')
    supervisor = IdleJobSupervisor(runtime)
    try:
        status = _until(supervisor, lambda value: value["state"] == "backoff")
        assert status["next_retry_at"] is not None
        assert status["last_exit_reason"] is not None
        assert status["failure_count"] >= 1
        for _ in range(3):
            status = supervisor.tick(demand="busy", gpu_policy=POLICY)
            assert status["state"] == "waiting"
            assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


def test_unknown_evidence_requires_a_fresh_idle_interval(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=0.3)
    supervisor = IdleJobSupervisor(runtime)
    try:
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        time.sleep(0.2)
        supervisor.tick(demand="unknown", gpu_policy=POLICY)
        time.sleep(0.15)
        status = supervisor.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "waiting"
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda value: value["state"] == "running")
    finally:
        assert supervisor.shutdown()


def test_recreated_supervisor_adopts_only_one_instance(tmp_path):
    runtime = _runtime(tmp_path)
    first = IdleJobSupervisor(runtime)
    second = None
    try:
        _until(first, lambda value: value["state"] == "running")
        owner = read_idle_job_ownership(runtime.root)
        second = IdleJobSupervisor(runtime)
        _until(second, lambda value: value["state"] == "running")
        assert read_idle_job_ownership(runtime.root)["instance_id"] == owner["instance_id"]
        assert len(list((runtime.root / "idle-job" / "instances").iterdir())) == 1
    finally:
        if second is not None:
            assert second.shutdown()
        assert first.shutdown()


def test_configuration_replacement_waits_for_old_group_exit(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda value: value["state"] == "running")
        old = read_idle_job_ownership(runtime.root)
        old_directory = idle_job_instance_path(runtime.root, old["instance_id"])
        config = set_idle_job_config(runtime, gpu_ids=(0, 1))
        _until(
            supervisor,
            lambda value: value["state"] == "running" and value["config_revision"] == config.revision,
        )
        assert idle_job_processes_exited(old_directory)
        assert read_idle_job_ownership(runtime.root)["instance_id"] != old["instance_id"]
    finally:
        assert supervisor.shutdown()


def test_reservation_observation_requires_fresh_idle_interval_after_release(tmp_path):
    from qqtools.plugins.qexp.runtime.resources.reservations import release

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=0.3)
    supervisor = IdleJobSupervisor(runtime)
    try:
        reservation = reserve(runtime.root, "training", [0])["reservation"]
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        time.sleep(0.35)
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        release(runtime.root, reservation["reservation_id"])
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda value: value["state"] == "running")
    finally:
        assert supervisor.shutdown()


def test_spawn_failure_retention_and_large_retry_count(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    now = [time.time()]

    def fail_spawn(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module, "spawn_idle_job", fail_spawn)
    supervisor = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
    try:
        for _ in range(8):
            status = supervisor.tick(demand="idle", gpu_policy=POLICY)
            assert status["state"] == "backoff"
            now[0] += 61
        instances = runtime.root / "idle-job" / "instances"
        assert len(list(instances.iterdir())) <= 5
        state_path = runtime.root / "idle-job" / "state.json"
        state = read_json(state_path)
        state["idle_job_state"]["failure_count"] = 1025
        atomic_replace(state_path, state)
        restarted = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
        status = restarted.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "backoff"
        assert status["failure_count"] == 1026
        assert status["next_retry_at"] == now[0] + 60
        assert restarted.shutdown()
    finally:
        assert supervisor.shutdown()


def test_crash_after_ownership_release_preserves_retry_accounting(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import read_json

    class InjectedCrash(BaseException):
        pass

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, command=f'{sys.executable} -c "raise SystemExit(7)"')
    supervisor = IdleJobSupervisor(runtime)
    original_clear = module.clear_idle_job_ownership

    def crash_after_release(*args, **kwargs):
        original_clear(*args, **kwargs)
        raise InjectedCrash

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "clear_idle_job_ownership", crash_after_release)
            with pytest.raises(InjectedCrash):
                _until(supervisor, lambda value: value["state"] == "backoff")
        state = read_json(runtime.root / "idle-job" / "state.json")["idle_job_state"]
        assert state["failure_count"] == 1
        assert state["retry_at"] > time.time()
        assert module.idle_job_status(runtime)["state"] != "running"
        restarted = IdleJobSupervisor(runtime)
        status = restarted.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "backoff"
        assert status["failure_count"] == 1
        assert read_idle_job_ownership(runtime.root) is None
        assert restarted.shutdown()
    finally:
        assert supervisor.shutdown()


def test_terminal_instances_from_previous_boot_are_pruned(tmp_path, monkeypatch):
    import uuid

    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    now = [time.time()]

    def fail_spawn(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module, "spawn_idle_job", fail_spawn)
    supervisor = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
    try:
        for _ in range(5):
            assert supervisor.tick(demand="idle", gpu_policy=POLICY)["state"] == "backoff"
            now[0] += 61
        instances = runtime.root / "idle-job" / "instances"
        previous = list(instances.iterdir())
        previous_boot = str(uuid.uuid4())
        for instance in previous:
            for filename, envelope in (("intent.json", "idle_job_intent"), ("exit.json", "idle_job_exit")):
                value = read_json(instance / filename)
                value[envelope]["boot_id"] = previous_boot
                atomic_replace(instance / filename, value)
        for _ in range(5):
            assert supervisor.tick(demand="idle", gpu_policy=POLICY)["state"] == "backoff"
            now[0] += 61
        assert len(list(instances.iterdir())) <= 5
        assert sum(path.exists() for path in previous) < len(previous)
    finally:
        assert supervisor.shutdown()


def test_bounded_partition_progress_preserves_only_established_idle_interval(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0.0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
    try:
        supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)
        clock[0] = 100.0
        assert supervisor.tick(demand="idle", gpu_policy=POLICY)["reason"] == "idle_delay"
        clock[0] = 130.0
        assert (
            supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)["reason"]
            == "partition_scan_pending"
        )
        clock[0] = 161.0
        assert (
            supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)["reason"]
            == "partition_scan_pending"
        )
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda status: status["state"] == "running")
    finally:
        # Shutdown must use a moving clock for its bounded process wait.
        supervisor._monotonic = time.monotonic
        assert supervisor.shutdown()


def test_agent_stop_reclaims_idle_ownership_without_live_agent(tmp_path):
    from qqtools.plugins.qexp.agent.lifecycle import stop_machine_agent

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda status: status["state"] == "running")
        owner = read_idle_job_ownership(runtime.root)
        directory = idle_job_instance_path(runtime.root, owner["instance_id"])
        assert stop_machine_agent(runtime) is False
        assert idle_job_processes_exited(directory)
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()
